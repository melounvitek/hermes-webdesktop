"""Config/env loaders for runtime knobs (busy modes, reasoning, service tier, timeouts, fallback) for GatewayRunner.

Split out of ``gateway/run.py``; bound onto ``GatewayRunner`` via the MRO.
``gateway.run`` internals are imported lazily inside method bodies (import cycle),
so ``patch("gateway.run.X")`` keeps intercepting them at call time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import json
import os
import time
from gateway.config import Platform
from gateway.restart import (
    DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT,
    DEFAULT_GATEWAY_POST_INTERRUPT_GRACE_TIMEOUT,
    DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT,
    DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT,
    DEFAULT_GATEWAY_SIGNAL_INTERRUPT_GRACE_TIMEOUT,
    parse_cron_drain_timeout,
    parse_restart_after_turn_timeout,
    parse_restart_drain_timeout,
    parse_signal_interrupt_grace_timeout,
)
from gateway.session import SessionSource
from gateway.session_state import SERVICE_TIER_UNSET as _SERVICE_TIER_UNSET
from hermes_cli.config import cfg_get
from hermes_cli.fallback_config import get_fallback_chain
from pathlib import Path
from typing import Any, Dict, List, Optional
from utils import is_truthy_value

if TYPE_CHECKING:  # string annotations only; never imported at runtime (cycle)
    from gateway.run import GatewayRunner, TurnRunner  # noqa: F401

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


class GatewayConfigLoadersMixin:
    """Config/env loaders for runtime knobs (busy modes, reasoning, service tier, timeouts, fallback) for GatewayRunner."""

    @staticmethod
    def _load_prefill_messages() -> List[Dict[str, Any]]:
        """Load ephemeral prefill messages from config or env var.

        HERMES_PREFILL_MESSAGES_FILE env wins, then top-level prefill_messages_file in config.yaml,
        then legacy agent.prefill_messages_file. Relative paths resolve from ~/.hermes/.
        """
        from gateway.run import _hermes_home, _load_gateway_runtime_config
        file_path = os.getenv("HERMES_PREFILL_MESSAGES_FILE", "")
        if not file_path:
            cfg = _load_gateway_runtime_config()
            file_path = str(cfg.get("prefill_messages_file", "") or "")
            if not file_path:
                file_path = str(cfg_get(cfg, "agent", "prefill_messages_file", default="") or "")
        if not file_path:
            return []
        path = Path(file_path).expanduser()
        if not path.is_absolute():
            path = _hermes_home / path
        if not path.exists():
            logger.warning("Prefill messages file not found: %s", path)
            return []
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, list):
                logger.warning("Prefill messages file must contain a JSON array: %s", path)
                return []
            return data
        except Exception as e:
            logger.warning("Failed to load prefill messages from %s: %s", path, e)
            return []

    @staticmethod
    def _load_ephemeral_system_prompt() -> str:
        """Load ephemeral system prompt: HERMES_EPHEMERAL_SYSTEM_PROMPT env var first, then
        ``display.personality`` / ``agent.system_prompt`` in config.yaml.
        """
        from gateway.run import _load_gateway_runtime_config
        from hermes_cli.config import resolve_ephemeral_system_prompt_from_config

        prompt = os.getenv("HERMES_EPHEMERAL_SYSTEM_PROMPT", "")
        if prompt:
            return prompt
        cfg = _load_gateway_runtime_config()
        return resolve_ephemeral_system_prompt_from_config(cfg)

    def _resolve_model_for_channel(
        self,
        platform: Platform,
        chat_id: str,
        *,
        user_config: Optional[dict] = None,
        thread_id: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> str:
        """Resolve model for this channel: channel_overrides else global default.

        Precedence lives in :func:`hermes_cli.model_switch.resolve_effective_model` (shared with the
        API server so the surfaces cannot diverge). No session tier here: session /model overrides
        are applied later by ``_apply_session_model_override``.
        """
        from gateway.run import _get_channel_override, _resolve_gateway_model
        from hermes_cli.model_switch import resolve_effective_model

        override = None
        config = getattr(self, "config", None)
        if config:
            override = _get_channel_override(
                config,
                platform,
                chat_id,
                thread_id=thread_id,
                parent_id=parent_id,
            )
        return resolve_effective_model(
            None,  # session tier applied downstream (_apply_session_model_override)
            override,
            _resolve_gateway_model(user_config),
        )

    def _get_system_prompt_for_channel(
        self,
        platform: Platform,
        chat_id: str,
        *,
        thread_id: Optional[str] = None,
        parent_id: Optional[str] = None,
    ) -> str:
        """Ephemeral system prompt for this channel/thread.

        ``channel_overrides`` when set, else the gateway prompt resolved from the CURRENT profile's
        config on every call (callers run inside ``_profile_runtime_scope``, so routed multiplex
        profiles get their own personality/system_prompt and ``/personality`` edits apply next turn).
        Legacy ``channel_prompts`` are applied separately via ``event.channel_prompt`` in ``run_sync``.
        """
        from gateway.run import _get_channel_override
        config = getattr(self, "config", None)
        if config:
            override = _get_channel_override(
                config,
                platform,
                chat_id,
                thread_id=thread_id,
                parent_id=parent_id,
            )
            if override and override.system_prompt:
                return (override.system_prompt or "").strip()
        return self._load_ephemeral_system_prompt()

    @staticmethod
    def _load_reasoning_config(model: str = "") -> dict | None:
        """Load reasoning effort from config.yaml, respecting per-model overrides.

        Thin wrapper over :func:`hermes_constants.resolve_reasoning_config` (per-model override >
        global ``agent.reasoning_effort``; YAML False = disabled). Empty ``model`` uses ``model.default``.
        """
        from gateway.run import _load_gateway_runtime_config
        from hermes_constants import resolve_reasoning_config
        cfg = _load_gateway_runtime_config()
        return resolve_reasoning_config(cfg, model)

    @staticmethod
    def _parse_reasoning_command_args(raw_args: str) -> tuple[str, bool]:
        """Parse `/reasoning` args into `(value, persist_global)`.

        Session-scoped by default; `--global` in any position persists the change to config.yaml.
        """
        import shlex

        text = str(raw_args or "").strip().replace("—", "--")
        if not text:
            return "", False
        try:
            tokens = shlex.split(text)
        except ValueError:
            tokens = text.split()

        persist_global = False
        value_tokens = []
        for token in tokens:
            if token == "--global":
                persist_global = True
            else:
                value_tokens.append(token)
        return " ".join(value_tokens).strip().lower(), persist_global

    def _resolve_session_reasoning_config(
        self,
        *,
        source: Optional[SessionSource] = None,
        session_key: Optional[str] = None,
        model: str = "",
    ) -> dict | None:
        """Resolve reasoning effort for a session, honoring session overrides.

        Priority: session ``/reasoning --session`` > per-model ``agent.reasoning_overrides`` > global
        ``agent.reasoning_effort``. ``model`` must be the session's *effective* model (session
        ``/model`` override included); empty uses ``model.default``.
        """
        resolved_session_key = self._resolve_session_key_or_none(source, session_key)

        if resolved_session_key:
            _r_state = self._peek_session_state(resolved_session_key)
            if _r_state is not None and _r_state.conversation.reasoning_override is not None:
                return _r_state.conversation.reasoning_override
        return self._load_reasoning_config(model)

    def _set_session_reasoning_override(
        self,
        session_key: str,
        reasoning_config: Optional[dict],
    ) -> None:
        """Set or clear the session-scoped reasoning override."""
        if not session_key:
            return
        # Per-session field write: a lazy ``_session_reasoning_overrides = {}`` init replaced the
        # WHOLE dict, racing concurrent sessions; a SessionState field reset cannot cross sessions.
        self._session_state(session_key).conversation.reasoning_override = (
            None if reasoning_config is None else dict(reasoning_config)
        )

    def _resolve_session_service_tier(
        self,
        source=None,
        session_key: Optional[str] = None,
    ) -> Optional[str]:
        """Resolve the effective service tier for a session.

        A session-scoped /fast override beats the config default; the override dict stores
        "priority" or None (explicit normal), so key presence — not truthiness — decides.
        """
        resolved_session_key = self._resolve_session_key_or_none(source, session_key)

        if resolved_session_key:
            _t_state = self._peek_session_state(resolved_session_key)
            if (
                _t_state is not None
                and _t_state.conversation.service_tier_override
                is not _SERVICE_TIER_UNSET
            ):
                return _t_state.conversation.service_tier_override
        return self._load_service_tier()

    def _set_session_service_tier_override(
        self,
        session_key: str,
        service_tier,
        clear: bool = False,
    ) -> None:
        """Set or clear the session-scoped /fast override.

        ``service_tier`` is "priority" or None (explicit normal). Pass
        ``clear=True`` to remove the override entirely (fall back to config).
        """
        if not session_key:
            return
        # Presence-sensitive: "priority" or None (explicit normal) both count as an override; the
        # sentinel means "no override". Per-session field write: a lazy dict replace races sessions.
        self._session_state(session_key).conversation.service_tier_override = (
            _SERVICE_TIER_UNSET if clear else service_tier
        )

    @staticmethod
    def _load_service_tier() -> str | None:
        """Load Priority Processing (agent.service_tier) from config.yaml: "fast"/"priority"/"on" =>
        "priority"; "normal"/"off" disable; None when unset/unsupported.
        """
        from gateway.run import _load_gateway_runtime_config
        cfg = _load_gateway_runtime_config()
        raw = str(cfg_get(cfg, "agent", "service_tier", default="") or "").strip()

        value = raw.lower()
        if not value or value in {"normal", "default", "standard", "off", "none"}:
            return None
        if value in {"fast", "priority", "on"}:
            return "priority"
        if value in {"auto", "cold"}:
            return value
        logger.warning("Unknown service_tier '%s', ignoring", raw)
        return None

    @staticmethod
    def _load_show_reasoning() -> bool:
        """Load show_reasoning toggle from config.yaml display section."""
        from gateway.run import _load_gateway_runtime_config
        cfg = _load_gateway_runtime_config()
        return is_truthy_value(
            cfg_get(cfg, "display", "show_reasoning"),
            default=False,
        )

    @staticmethod
    def _load_busy_input_mode() -> str:
        """Load gateway drain-time busy-input behavior from config/env."""
        from gateway.run import _load_gateway_runtime_config
        mode = os.getenv("HERMES_GATEWAY_BUSY_INPUT_MODE", "").strip().lower()
        if not mode:
            cfg = _load_gateway_runtime_config()
            mode = str(cfg_get(cfg, "display", "busy_input_mode", default="") or "").strip().lower()
        if mode == "queue":
            return "queue"
        if mode == "steer":
            return "steer"
        return "interrupt"

    @staticmethod
    def _load_busy_text_mode() -> str:
        """Resolve normal busy TEXT follow-up behavior.

        ``busy_input_mode`` is the source of truth (default ``interrupt``); legacy ``busy_text_mode``
        is honored only when explicitly set so existing queue setups keep working.
        """
        from gateway.run import GatewayRunner, _load_gateway_runtime_config
        # Legacy explicit override wins for backward compat.
        legacy = os.getenv("HERMES_GATEWAY_BUSY_TEXT_MODE", "").strip().lower()
        if not legacy:
            cfg = _load_gateway_runtime_config()
            legacy = str(cfg_get(cfg, "display", "busy_text_mode", default="") or "").strip().lower()
        if legacy == "interrupt":
            return "interrupt"
        if legacy == "queue":
            return "queue"
        # No explicit legacy knob → follow busy_input_mode.
        input_mode = GatewayRunner._load_busy_input_mode()
        return "queue" if input_mode == "queue" else "interrupt"

    @staticmethod
    def _busy_modes_from_config(
        config: dict,
        *,
        fallback_input: str,
        fallback_text: str,
    ) -> tuple[str, str]:
        """Resolve one profile's busy modes without consulting process env."""
        raw_input = str(
            cfg_get(config, "display", "busy_input_mode", default="") or ""
        ).strip().lower()
        input_mode = (
            raw_input
            if raw_input in {"interrupt", "queue", "steer"}
            else fallback_input
        )

        raw_text = str(
            cfg_get(config, "display", "busy_text_mode", default="") or ""
        ).strip().lower()
        if raw_text in {"interrupt", "queue"}:
            text_mode = raw_text
        elif raw_input in {"interrupt", "queue", "steer"}:
            text_mode = "queue" if input_mode == "queue" else "interrupt"
        else:
            text_mode = fallback_text
        return input_mode, text_mode

    def _snapshot_profile_busy_modes(self, profile_name: str, config: dict) -> None:
        """Cache a routed profile's busy policy for this gateway lifetime."""
        input_mode, text_mode = self._busy_modes_from_config(
            config,
            fallback_input=getattr(self, "_busy_input_mode", "interrupt"),
            fallback_text=getattr(self, "_busy_text_mode", "interrupt"),
        )
        input_modes = self.__dict__.setdefault("_busy_input_modes_by_profile", {})
        text_modes = self.__dict__.setdefault("_busy_text_modes_by_profile", {})
        input_modes[profile_name] = input_mode
        text_modes[profile_name] = text_mode

    def _busy_profile_name_for_source(self, source: SessionSource) -> Optional[str]:
        """Return the routed profile whose busy policy applies, if any."""
        if not getattr(getattr(self, "config", None), "multiplex_profiles", False):
            return None
        name = str(getattr(source, "profile", "") or "").strip()
        if not name:
            try:
                name = str(self._profile_name_for_source(source) or "").strip()
            except Exception:
                name = ""
        return name or None

    def _effective_busy_input_mode(self, source: SessionSource) -> str:
        """Resolve busy input mode from the routed profile startup snapshot."""
        fallback = getattr(self, "_busy_input_mode", "interrupt")
        profile_name = self._busy_profile_name_for_source(source)
        if not profile_name:
            return fallback
        modes = getattr(self, "_busy_input_modes_by_profile", None)
        return modes.get(profile_name, fallback) if isinstance(modes, dict) else fallback

    def _effective_busy_text_mode(self, source: SessionSource) -> str:
        """Resolve legacy busy text mode from the routed profile snapshot."""
        fallback = getattr(self, "_busy_text_mode", "interrupt")
        profile_name = self._busy_profile_name_for_source(source)
        if not profile_name:
            return fallback
        modes = getattr(self, "_busy_text_modes_by_profile", None)
        return modes.get(profile_name, fallback) if isinstance(modes, dict) else fallback

    @staticmethod
    def _load_restart_drain_timeout() -> float:
        """Load graceful gateway restart/stop drain timeout in seconds."""
        from gateway.run import _load_gateway_runtime_config
        raw = os.getenv("HERMES_RESTART_DRAIN_TIMEOUT", "").strip()
        if not raw:
            cfg = _load_gateway_runtime_config()
            raw = str(cfg_get(cfg, "agent", "restart_drain_timeout", default="") or "").strip()
        value = parse_restart_drain_timeout(raw)
        if raw and value == DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT:
            try:
                float(raw)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid restart_drain_timeout '%s', using default %.0fs",
                    raw,
                    DEFAULT_GATEWAY_RESTART_DRAIN_TIMEOUT,
                )
        return value

    @staticmethod
    def _load_env_or_agent_cfg_timeout(env_var: str, cfg_key: str, parse, default: float) -> float:
        """Env var (non-empty) else ``agent.<cfg_key>``; warn once when a supplied value fails to parse.

        ``0`` is a valid value; the parser falls back to ``default`` on garbage."""
        from gateway.run import _load_gateway_runtime_config
        env_raw = os.getenv(env_var)
        if env_raw is not None and str(env_raw).strip() != "":
            raw: object = env_raw
        else:
            cfg = _load_gateway_runtime_config()
            raw = cfg_get(cfg, "agent", cfg_key, default=None)
        value = parse(raw)
        if raw is not None and str(raw).strip() != "":
            try:
                float(raw)
            except (TypeError, ValueError):
                logger.warning("Invalid %s '%s', using default %.0fs", cfg_key, raw, default)
        return value

    @classmethod
    def _load_restart_after_turn_timeout(cls) -> float:
        """Load in-band restart wait-for-idle timeout in seconds."""
        return cls._load_env_or_agent_cfg_timeout(
            "HERMES_RESTART_AFTER_TURN_TIMEOUT", "restart_after_turn_timeout",
            parse_restart_after_turn_timeout, DEFAULT_GATEWAY_RESTART_AFTER_TURN_TIMEOUT,
        )

    @classmethod
    def _load_cron_drain_timeout(cls) -> float:
        """Load the cron-only floor under the stop()/drain wait."""
        return cls._load_env_or_agent_cfg_timeout(
            "HERMES_CRON_DRAIN_TIMEOUT", "cron_drain_timeout",
            parse_cron_drain_timeout, DEFAULT_GATEWAY_CRON_DRAIN_TIMEOUT,
        )

    @staticmethod
    def _load_signal_interrupt_grace_timeout() -> float:
        """Load the unexpected-signal post-interrupt grace in seconds."""
        from gateway.run import _load_gateway_runtime_config
        cfg = _load_gateway_runtime_config()
        raw = cfg_get(
            cfg,
            "gateway",
            "signal_interrupt_grace_timeout",
            default=None,
        )
        value = parse_signal_interrupt_grace_timeout(raw)
        if raw is not None and raw != "":
            try:
                float(raw)
            except (TypeError, ValueError):
                logger.warning(
                    "Invalid signal_interrupt_grace_timeout '%s', using default %.0fs",
                    raw,
                    DEFAULT_GATEWAY_SIGNAL_INTERRUPT_GRACE_TIMEOUT,
                )
        return value

    def _post_interrupt_grace_timeout(self) -> float:
        """Return the grace before teardown after forcibly interrupting agents."""
        if (
            getattr(self, "_signal_initiated_shutdown", False)
            and not getattr(self, "_restart_requested", False)
        ):
            return max(
                0.0,
                float(
                    getattr(
                        self,
                        "_signal_interrupt_grace_timeout",
                        DEFAULT_GATEWAY_SIGNAL_INTERRUPT_GRACE_TIMEOUT,
                    )
                ),
            )
        return DEFAULT_GATEWAY_POST_INTERRUPT_GRACE_TIMEOUT

    @staticmethod
    def _load_background_notifications_mode() -> str:
        """Load background process notification mode from config or env var."""
        from gateway.run import _load_gateway_runtime_config
        mode = os.getenv("HERMES_BACKGROUND_NOTIFICATIONS", "")
        if not mode:
            cfg = _load_gateway_runtime_config()
            raw = cfg_get(cfg, "display", "background_process_notifications")
            if raw is False:
                mode = "off"
            elif raw not in {None, ""}:
                mode = str(raw)
        mode = (mode or "concise").strip().lower()
        valid = {"concise", "all", "result", "error", "off"}
        if mode not in valid:
            logger.warning(
                "Unknown background_process_notifications '%s', defaulting to 'concise'",
                mode,
            )
            return "concise"
        return mode

    @staticmethod
    def _load_provider_routing() -> dict:
        """Load OpenRouter provider routing preferences from config.yaml."""
        from gateway.run import _load_gateway_runtime_config
        try:
            # Canonical gateway loader (fail-open): managed overlay + ${VAR}
            # expansion now apply to provider_routing too.
            cfg = _load_gateway_runtime_config()
            return cfg.get("provider_routing", {}) or {}
        except Exception:
            pass
        return {}

    @staticmethod
    def _load_fallback_model() -> list | None:
        """Load fallback provider chain from config.yaml.

        Merges ``fallback_providers`` (kept first) with legacy ``fallback_model`` entries.
        """
        from gateway.run import _load_gateway_runtime_config
        try:
            # Canonical gateway loader (fail-open): managed overlay + ${VAR}
            # expansion now apply to the fallback chain too.
            cfg = _load_gateway_runtime_config()
            fb = get_fallback_chain(cfg)
            if fb:
                return fb
        except Exception:
            pass
        return None

    def _refresh_fallback_model(self) -> list | None:
        """Re-read fallback_providers from disk for the next agent create/reuse.

        Lets a chain edited after startup reach messaging sessions (cron already re-reads per job).
        A TRANSIENT read/parse failure (user mid-edit, non-atomic write) keeps the last known-good
        chain; only a successful read that genuinely lacks the key clears it.
        """
        from gateway.run import _hermes_home
        try:
            from hermes_cli.config import read_user_config_raw
            cfg_path = _hermes_home / "config.yaml"
            if not cfg_path.exists():
                self._fallback_model = None
                return self._fallback_model
            # Raw primitive (raises on parse failure) is required here: the canonical fail-open
            # loader would return {} on a torn mid-edit write and WIPE the last known-good chain.
            # The overlay/expansion below fixes the managed-scope/${VAR} drift without losing that.
            cfg = read_user_config_raw(cfg_path)
            try:
                from hermes_cli import managed_scope
                cfg = managed_scope.apply_managed_overlay(cfg)
            except Exception:
                pass
            try:
                from hermes_cli.config import _expand_env_vars
                expanded = _expand_env_vars(cfg)
                if isinstance(expanded, dict):
                    cfg = expanded
            except Exception:
                pass
        except Exception:
            # Transient failure — keep last known-good chain.
            logger.debug(
                "fallback_providers refresh: config.yaml read failed; "
                "keeping last known-good chain", exc_info=True,
            )
            return self._fallback_model
        self._fallback_model = get_fallback_chain(cfg) or None
        return self._fallback_model

    @staticmethod
    def _apply_fallback_chain_to_agent(agent: Any, chain: list | None) -> None:
        """Keep a cached agent's fallback chain aligned with current config.

        Skips the rewrite while a cooldown holds the agent on an activated fallback provider
        (``restore_primary_runtime`` owns that lifecycle); otherwise replaces the chain so
        mid-uptime ``fallback_providers`` edits apply without a restart.
        """
        if agent is None:
            return
        new_chain = list(chain or [])
        rate_limited_until = getattr(agent, "_rate_limited_until", 0) or 0
        if (
            getattr(agent, "_fallback_activated", False)
            and rate_limited_until > time.monotonic()
        ):
            return
        old_chain = list(getattr(agent, "_fallback_chain", []) or [])
        agent._fallback_chain = new_chain
        agent._fallback_model = new_chain[0] if new_chain else None
        if not getattr(agent, "_fallback_activated", False):
            agent._fallback_index = 0
        # A config edit means the user changed something — drop the session-scoped unavailability
        # memo so re-configured entries (e.g. credentials added mid-uptime) get retried. Only on real
        # content change, so the per-message no-op refresh keeps the memo's rate-limiting benefit.
        if new_chain != old_chain:
            unavailable = getattr(agent, "_unavailable_fallback_keys", None)
            if unavailable:
                unavailable.clear()
