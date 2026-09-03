"""Dashboard config schema and model-assignment logic: CONFIG_SCHEMA construction, dynamic provider options, web<->config normalisation, main/aux model assignment.

Split out of ``hermes_cli.web_server``; every externally used name is re-imported
there, so ``web_server.<name>`` keeps resolving (and monkeypatching) as before.
Helpers that tests patch on ``web_server`` are reached lazily through it.
"""

import logging
import os
from fastapi import HTTPException
from typing import Any, Dict, List, Optional, Tuple
from hermes_cli.config import (
    DEFAULT_CONFIG,
    build_cron_model_impact,
    cfg_get,
    clear_model_endpoint_credentials,
    find_provider_entry,
    read_raw_config,
    resolve_cron_model_drift_defaults,
)
from hermes_cli.web_server_memory import _normalize_memory_provider_name

# Same logger the code used before extraction (record parity).
_log = logging.getLogger("hermes_cli.web_server")


# ---------------------------------------------------------------------------
# Config schema — auto-generated from DEFAULT_CONFIG
# ---------------------------------------------------------------------------

# Manual overrides for fields that need select options or custom types
def _memory_provider_options() -> List[str]:
    """Discovered memory providers for the ``memory.provider`` select.

    Directory-scan only (no provider imports), so it's safe at module import
    time. ``""`` (built-in only) is always first; discovery failures degrade to
    the bundled defaults rather than dropping the field. The literal
    ``builtin`` alias is deliberately NOT offered — built-in memory is not a
    provider plugin, and ``_normalize_memory_provider_name`` already maps any
    legacy ``builtin``/``built-in``/``none`` value back to ``""`` (#49513).
    """
    options = [""]
    try:
        from plugins.memory import list_memory_provider_names

        options.extend(list_memory_provider_names())
    except Exception:
        options.extend(["honcho"])
    # Dedupe, preserve order
    return list(dict.fromkeys(options))


def _timezone_options() -> List[str]:
    """Return sorted IANA timezone identifiers, cached at import time."""
    try:
        import zoneinfo
        return sorted(zoneinfo.available_timezones()) or ["UTC"]
    except Exception:  # pragma: no cover
        return ["UTC"]


_SCHEMA_OVERRIDES: Dict[str, Dict[str, Any]] = {
    "timezone": {
        "type": "select",
        "description": "IANA timezone (e.g. America/New_York). Blank uses the system timezone.",
        "options": _timezone_options(),
        "searchable": True,
        "clearable": True,
    },
    "memory.provider": {
        "type": "select",
        "description": "Memory provider plugin",
        "options": _memory_provider_options(),
    },
    "model": {
        "type": "string",
        "description": "Default model (e.g. anthropic/claude-sonnet-4.6)",
        "category": "general",
    },
    "model_context_length": {
        "type": "number",
        "description": "Context window override (0 = auto-detect from model metadata)",
        "category": "general",
    },
    "terminal.backend": {
        "type": "select",
        "description": "Terminal execution backend",
        "options": ["local", "docker", "ssh", "modal", "daytona", "vercel_sandbox", "singularity"],
    },
    "terminal.vercel_runtime": {
        "type": "select",
        "description": "Vercel Sandbox runtime",
        "options": ["node24", "node22", "python3.13"],  # sync with _SUPPORTED_VERCEL_RUNTIMES in terminal_tool.py
    },
    "terminal.modal_mode": {
        "type": "select",
        "description": "Modal sandbox mode",
        "options": ["sandbox", "function"],
    },
    "proxy.enabled": {
        "type": "boolean",
        "description": (
            "Docker-only egress credential firewall. Requires `hermes egress setup` "
            "and `hermes egress start`; Modal/SSH/Daytona are not wired yet."
        ),
        "category": "security",
    },
    "proxy.credential_source": {
        "type": "select",
        "description": "Where iron-proxy loads real upstream secrets at start time",
        "options": ["env", "bitwarden"],
        "category": "security",
    },
    "proxy.enforce_on_docker": {
        "type": "boolean",
        "description": "Refuse Docker sandboxes when egress is enabled but not configured/running",
        "category": "security",
    },
    "tts.provider": {
        "type": "select",
        "description": "Text-to-speech provider",
        "options": ["edge", "elevenlabs", "openai", "xai", "minimax", "mistral", "gemini", "neutts", "kittentts", "piper"],
    },
    "stt.provider": {
        "type": "select",
        "description": "Speech-to-text provider",
        # "mistral" temporarily removed — mistralai PyPI package quarantined
        # (malicious 2.4.6 release on 2026-05-12). Restore once available.
        "options": ["local", "groq", "openai", "xai", "elevenlabs"],
    },
    "stt.local.model": {
        "type": "select",
        "description": "Local faster-whisper model size",
        "options": ["tiny", "base", "small", "medium", "large-v3"],
    },
    "stt.groq.model": {
        "type": "select",
        "description": "Groq Whisper model",
        "options": ["whisper-large-v3-turbo", "whisper-large-v3", "distil-whisper-large-v3-en"],
    },
    "stt.openai.model": {
        "type": "select",
        "description": "OpenAI transcription model",
        "options": ["whisper-1", "gpt-4o-mini-transcribe", "gpt-4o-transcribe", "gpt-transcribe"],
    },
    "stt.elevenlabs.model_id": {
        "type": "select",
        "description": "ElevenLabs Scribe model",
        "options": ["scribe_v2", "scribe_v1"],
    },
    "display.skin": {
        "type": "select",
        "description": "CLI visual theme",
        "options": ["default", "ares", "mono", "slate"],
    },
    "dashboard.theme": {
        "type": "select",
        "description": "Web dashboard visual theme",
        "options": ["default", "midnight", "ember", "mono", "cyberpunk", "rose"],
    },
    "display.resume_display": {
        "type": "select",
        "description": "How resumed sessions display history",
        "options": ["minimal", "full", "off"],
    },
    "display.busy_input_mode": {
        "type": "select",
        "description": "Input behavior while agent is running",
        "options": ["interrupt", "queue", "steer"],
    },
    "approvals.mode": {
        "type": "select",
        "description": "Dangerous command approval mode",
        "options": ["manual", "smart", "off"],
    },
    "context.engine": {
        "type": "select",
        "description": "Context management engine",
        "options": ["default", "custom"],
    },
    "human_delay.mode": {
        "type": "select",
        "description": "Simulated typing delay mode",
        "options": ["off", "typing", "fixed"],
    },
    "logging.level": {
        "type": "select",
        "description": "Log level for agent.log",
        "options": ["DEBUG", "INFO", "WARNING", "ERROR"],
    },
    "agent.service_tier": {
        "type": "select",
        "description": "Fast mode: fast = always, auto = first N seconds of each turn, cold = first turn only",
        "options": ["", "normal", "fast", "auto", "cold"],
    },
    "delegation.reasoning_effort": {
        "type": "select",
        "description": "Reasoning effort for delegated subagents",
        "options": ["", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"],
    },
    "updates.non_interactive_local_changes": {
        "type": "select",
        "description": (
            "When the chat app / gateway updates Hermes (no terminal prompt), "
            "what to do with uncommitted local source edits. 'stash' keeps them "
            "and re-applies them after the update; 'discard' throws them away. "
            "Terminal updates always ask, regardless of this setting."
        ),
        "options": ["stash", "discard"],
    },
    "updates.refresh_cua_driver": {
        "type": "boolean",
        "description": (
            "Refresh an already-installed cua-driver during hermes update. "
            "Disable this on non-admin macOS accounts where /Applications is "
            "not writable."
        ),
    },
    "browser.headed": {
        "type": "boolean",
        "description": "Run the local browser in headed mode (visible window). Also keeps the window open between turns; idle sessions are still reaped after browser.inactivity_timeout.",
    },
    "plugins.hook_callback_timeout": {
        "type": "number",
        "description": (
            "Wall-clock cap (seconds) for timeout-bounded in-process Python "
            "plugin hook callbacks (hot-path observers + pre_tool_call). "
            "Timed-out pre_tool_call fails closed. 0 disables the cap; "
            "values above 600 are clamped. Caller-thread hooks such as "
            "subagent_stop are never moved onto a timeout worker."
        ),
    },
}

# Categories with fewer fields get merged into "general" to avoid tab sprawl.
_CATEGORY_MERGE: Dict[str, str] = {
    "privacy": "security",
    "context": "agent",
    "skills": "agent",
    "cron": "agent",
    "network": "agent",
    # `models_dev.url` (mirror override) is the only schema-surfaced
    # models_dev field — fold it in with the other network/agent plumbing
    # rather than spawning a one-field orphan tab.
    "models_dev": "agent",
    "checkpoints": "agent",
    "approvals": "security",
    "human_delay": "display",
    "dashboard": "display",
    "code_execution": "agent",
    "prompt_caching": "agent",
    # bot_mode holds a couple of relay tuning knobs — keep it folded into the
    # agent tab rather than spawning a tiny standalone category.
    "bot_mode": "agent",
    "goals": "agent",
    "updates": "general",
    # `onboarding.profile_build` is the only schema-surfaced onboarding field
    # (`onboarding.seen` is an internal latch dict, not a user setting), so fold
    # it into the agent tab rather than spawning a one-field orphan category.
    "onboarding": "agent",
    # Only `telegram.reactions` currently lives under telegram — fold it in
    # with the other messaging-platform config (discord) so it isn't an
    # orphan tab of one field.
    "telegram": "discord",
    # `mcp.auto_reload_on_config_change` is the only schema-surfaced mcp
    # runtime field (server definitions live under mcp_servers, edited via
    # the MCP tab) — fold it into the agent tab rather than spawning a
    # one-field orphan category.
    "mcp": "agent",
    # `computer_use.cua_telemetry` is the only schema-surfaced computer_use
    # field — fold it into the agent tab rather than spawning a one-field
    # orphan category.
    "computer_use": "agent",
    # `telemetry.shared_metrics.enabled` is the only schema-surfaced telemetry
    # field — fold it into security alongside the other privacy-posture toggles.
    "telemetry": "security",
    # `plugins.hook_callback_timeout` is the only schema-surfaced plugins field
    # (`enabled`/`disabled` are list allow-lists omitted from DEFAULT_CONFIG) —
    # fold it into the agent tab rather than spawning a one-field orphan category.
    "plugins": "agent",
    # `doctor.live_probe_timeout` is the only schema-surfaced doctor field —
    # fold it into general rather than spawning a one-field orphan category.
    "doctor": "general",
    # `runtime.nofile_soft_limit` (#78873) is the only schema-surfaced runtime
    # field — fold it into the agent tab rather than spawning a one-field
    # orphan category.
    "runtime": "agent",
    # `session.terminal_continue` is the only schema-surfaced session field —
    # fold it into general rather than spawning a one-field orphan category.
    "session": "general",
    # `nous.keepalive_interval_seconds` is the only schema-surfaced nous field
    # (Portal tokens live in auth.json) — fold it into the agent tab.
    "nous": "agent",
}


def _infer_type(value: Any) -> str:
    """Infer a UI field type from a Python value."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "number"
    if isinstance(value, float):
        return "number"
    if isinstance(value, list):
        return "list"
    if isinstance(value, dict):
        return "object"
    return "string"


def _build_schema_from_config(
    config: Dict[str, Any],
    prefix: str = "",
) -> Dict[str, Dict[str, Any]]:
    """Walk DEFAULT_CONFIG and produce a flat dot-path → field schema dict."""
    schema: Dict[str, Dict[str, Any]] = {}
    for key, value in config.items():
        full_key = f"{prefix}.{key}" if prefix else key

        # Skip internal / version keys
        if full_key in {"_config_version"}:
            continue

        # Category is the first path component for nested keys, or "general"
        # for top-level scalar fields (model, toolsets, timezone, etc.).
        if prefix:
            category = prefix.split(".")[0]
        elif isinstance(value, dict):
            category = key
        else:
            category = "general"

        if isinstance(value, dict):
            # Recurse into nested dicts
            schema.update(_build_schema_from_config(value, full_key))
        else:
            entry: Dict[str, Any] = {
                "type": _infer_type(value),
                "description": full_key.replace(".", " → ").replace("_", " ").title(),
                "category": category,
            }
            # Apply manual overrides
            if full_key in _SCHEMA_OVERRIDES:
                entry.update(_SCHEMA_OVERRIDES[full_key])
            # Merge small categories
            entry["category"] = _CATEGORY_MERGE.get(entry["category"], entry["category"])
            schema[full_key] = entry
    return schema


def _config_schema_with_virtual_fields() -> Dict[str, Dict[str, Any]]:
    """DEFAULT_CONFIG schema plus the virtual fields the normalize/denormalize
    cycle surfaces: ``model_context_length`` is inserted right after ``model``
    so it renders adjacent in the frontend."""
    ordered: Dict[str, Dict[str, Any]] = {}
    for key, entry in _build_schema_from_config(DEFAULT_CONFIG).items():
        ordered[key] = entry
        if key == "model":
            ordered["model_context_length"] = _SCHEMA_OVERRIDES["model_context_length"]
    return ordered


CONFIG_SCHEMA = _config_schema_with_virtual_fields()


def _is_command_provider_block(value: Any) -> bool:
    """Return True when *value* declares a command-type voice provider.

    Mirrors the runtime discriminators
    (``tools.tts_tool._is_command_provider_config`` /
    ``tools.transcription_tools._is_command_stt_provider_config``) and the
    desktop's ``isCommandProvider`` in
    ``apps/desktop/src/app/settings/helpers.ts``: ``type`` is OPTIONAL and
    case/space-insensitive (absent or normalizing to ``"command"``), and
    ``command`` MUST be a non-empty string. Built-in blocks (which carry
    ``voice``/``model`` and no ``command``) and the ``providers`` container
    itself are rejected.
    """
    if not isinstance(value, dict):
        return False
    ptype = str(value.get("type") or "").strip().lower()
    if ptype and ptype != "command":
        return False
    command = value.get("command")
    return isinstance(command, str) and bool(command.strip())


def _custom_provider_options(
    kind: str,
    builtin_names: List[str],
    cfg: Dict[str, Any],
) -> List[str]:
    """Return a merged provider option list without hard-coding vendor names.

    *kind* is ``"tts"`` or ``"stt"``. The result keeps the built-in display
    names first (original order — NOT re-sorted), then appends:

    1. Command-type providers declared under the canonical
       ``<kind>.providers.<name>`` location, plus the legacy top-level
       ``<kind>.<name>`` fallback — exactly the dual resolution the runtime
       performs in ``_get_named_provider_config`` /
       ``_get_named_stt_provider_config``. Names colliding with a RUNTIME
       built-in are excluded case-insensitively (the runtime rejects a
       built-in name as a command provider before any config lookup), so a
       ``providers.EDGE`` command block is not offered.
    2. Plugin-registered provider names from ``agent.tts_registry`` /
       ``agent.transcription_registry`` — opportunistic only: plugins
       register at runtime via ``ctx.register_tts_provider()``, and this
       process does not necessarily call ``discover_plugins()``, so the
       registry may legitimately be empty here. (There is no static
       ``provides: [tts]`` manifest convention to scan — real manifests only
       carry ``provides_tools``/``provides_hooks``.)
    3. The current ``<kind>.provider`` value when not already present — a
       custom name that only appears as the active provider stays
       selectable (matches desktop ``enumOptionsFor``'s current-value
       preservation).

    Guard semantics deliberately mirror
    ``apps/desktop/src/app/settings/helpers.ts:commandProviderNames`` so the
    backend schema (web dashboard) and the desktop client agree on which
    names are offered.
    """
    names = [str(n) for n in builtin_names]
    seen = {n.strip().lower() for n in names}

    # Guard against the RUNTIME built-in sets, not the display shortlist
    # above: the display list drifts from the runtime sets (e.g. omits
    # ``deepinfra``), and filtering on it would offer names the runtime
    # would never honour as command providers.
    if kind == "tts":
        from tools.tts_tool import BUILTIN_TTS_PROVIDERS as _runtime_builtins
    else:
        from tools.transcription_tools import BUILTIN_STT_PROVIDERS as _runtime_builtins

    def _add(name: Any) -> None:
        if not isinstance(name, str):
            return
        stripped = name.strip()
        key = stripped.lower()
        if stripped and key not in seen:
            names.append(stripped)
            seen.add(key)

    section = cfg.get(kind)
    if not isinstance(section, dict):
        section = {}

    # Canonical nested location first, then the legacy top-level fallback —
    # the same order the runtime resolves them in.
    candidate_blocks: List[Any] = []
    providers_map = section.get("providers")
    if isinstance(providers_map, dict):
        candidate_blocks.append(providers_map)
    candidate_blocks.append(
        {k: v for k, v in section.items() if k != "providers"}
    )
    for block in candidate_blocks:
        for name, value in block.items():
            if (
                isinstance(name, str)
                and name.strip().lower() not in _runtime_builtins
                and _is_command_provider_block(value)
            ):
                _add(name)

    # Plugin-registered providers (only populated when plugins are loaded in
    # this process). Registry names can never collide with built-ins — the
    # registries reject such registrations.
    try:
        if kind == "tts":
            from agent.tts_registry import list_providers as _list_voice_providers
        else:
            from agent.transcription_registry import list_providers as _list_voice_providers
        for _p in _list_voice_providers():
            _add(getattr(_p, "name", None))
    except Exception:  # pragma: no cover - registry import should not break schema
        pass

    # Current-value preservation (``cfg_get`` takes *keys*, not dotted paths).
    _add(cfg_get(cfg, kind, "provider"))

    return names


def _memory_provider_schema_options(cfg: Dict[str, Any]) -> List[str]:
    """Discovered memory providers for a per-request schema merge.

    Reuses the cheap directory scan of :func:`_memory_provider_options` and
    additionally preserves the currently-configured provider, so a value
    selected in config but not (yet) discoverable — e.g. a plugin removed from
    disk — never silently vanishes from the dropdown.
    """
    from hermes_cli.web_server import _memory_provider_options
    options = _memory_provider_options()

    memory = cfg.get("memory")
    configured = memory.get("provider") if isinstance(memory, dict) else None
    current = _normalize_memory_provider_name(configured)

    if current and current not in options:
        options = [*options, current]

    return options


def _schema_with_dynamic_provider_options() -> Dict[str, Dict[str, Any]]:
    """Return CONFIG_SCHEMA with per-request discovery-driven options merged.

    Some ``*.provider`` selects have options that are discovered at runtime
    (voice backends via the tts/stt registries + config.yaml command
    providers; memory providers via a plugin-dir scan). The module-level
    ``_SCHEMA_OVERRIDES`` freezes those lists at import time, so a provider
    installed after the server started never appears. This recomputes them at
    request time — reflecting the CURRENT config.yaml, the profile-scoped
    config when the request carries a ``profile`` param, and mid-session
    plugin installs — for every surface that reads the schema (desktop, CLI,
    dashboard), with no extra frontend round-trips.

    The module-level ``CONFIG_SCHEMA`` is never mutated; entries that change
    are shallow-copied onto a copied mapping.
    """
    from hermes_cli.web_server import _plugin_terminal_backend_rows, load_config
    try:
        cfg = load_config()
    except Exception:  # pragma: no cover - schema must survive config errors
        return CONFIG_SCHEMA

    overlay: Dict[str, Dict[str, Any]] = {}

    def merge(key: str, options: List[str]) -> None:
        entry = CONFIG_SCHEMA.get(key)

        if isinstance(entry, dict) and isinstance(entry.get("options"), list) and options != entry["options"]:
            overlay[key] = {**entry, "options": options}

    for kind in ("tts", "stt"):
        entry = CONFIG_SCHEMA.get(f"{kind}.provider")
        existing = entry.get("options") if isinstance(entry, dict) else None

        if isinstance(existing, list):
            merge(f"{kind}.provider", _custom_provider_options(kind, list(existing), cfg))

    merge("memory.provider", _memory_provider_schema_options(cfg))

    tb_entry = CONFIG_SCHEMA.get("terminal.backend")
    if isinstance(tb_entry, dict) and isinstance(tb_entry.get("options"), list):
        try:
            plugin_names = sorted(
                {row["name"] for row in _plugin_terminal_backend_rows()}
                - set(tb_entry["options"])
            )
        except Exception:
            plugin_names = []
        if plugin_names:
            merge("terminal.backend", [*tb_entry["options"], *plugin_names])

    if not overlay:
        return CONFIG_SCHEMA

    return {**CONFIG_SCHEMA, **overlay}


def _normalize_main_model_assignment(provider: str, model: str) -> tuple[str, str]:
    """Normalize a main-slot (provider, model) pair before persisting.

    The Models page has two assignment paths and only one of them was safe:

    - The "Change" picker sends a real Hermes provider slug — fine.
    - The per-card "Use as → Main model" menu sends ``entry.provider``
      from the analytics rows, falling back to the model's VENDOR prefix
      (``modelVendor("anthropic/claude-opus-4.6") == "anthropic"``) when
      the session row has no ``billing_provider`` (older sessions, NULL
      rows).  That wrote ``provider: anthropic`` +
      ``default: anthropic/claude-opus-4.6`` to config — a vendor-prefixed
      OpenRouter slug on the NATIVE Anthropic provider.  New sessions then
      400 against api.anthropic.com ("model: anthropic/claude-opus-4.6 not
      found") and the user reads it as "changing models does nothing".

    Two repairs, both at this single chokepoint so every caller inherits:

    1. Vendor-name → Hermes-provider mapping: when the provider string is
       not a known Hermes provider/alias (e.g. ``moonshotai``, ``x-ai`` is
       known but ``poolside`` isn't) but the model is a vendor-prefixed
       aggregator slug, keep the user's CURRENT aggregator if they're on
       one, else fall back to openrouter.

       Named custom providers (``custom:litellm``, etc.) are excluded from
       this fallback: ``_KNOWN_PROVIDER_NAMES`` only lists the bare
       ``"custom"`` bucket, never a specific ``custom:<name>`` slug, so
       without this exclusion every named custom provider paired with a
       slash-bearing model (e.g. ``ollama/glm-5.2`` behind a LiteLLM proxy)
       looked exactly like the stray-vendor-prefix case above and got
       silently reassigned to ``openrouter``.
    2. Model-format normalization for the resolved provider via
       ``normalize_model_for_provider`` (e.g. ``anthropic/claude-opus-4.6``
       on native anthropic → ``claude-opus-4-6``).
    """
    from hermes_cli.web_server import load_config
    from hermes_cli.config import get_compatible_custom_providers
    from hermes_cli.models import _KNOWN_PROVIDER_NAMES, normalize_provider
    from hermes_cli.model_normalize import normalize_model_for_provider
    from hermes_cli.providers import resolve_custom_provider, resolve_user_provider

    prov_in = (provider or "").strip()
    model_in = (model or "").strip()
    canonical = normalize_provider(prov_in)

    # User-declared providers are real routing targets, not analytics vendor
    # labels. Resolve them before the unknown-vendor fallback. ``providers:``
    # keeps its declared bare slug; ``custom_providers:`` canonicalizes both a
    # bare display name and ``custom:<name>`` to the durable custom slug.
    try:
        cfg = load_config()
    except Exception:
        cfg = {}
    user_providers = cfg.get("providers") if isinstance(cfg, dict) else None
    user_provider = resolve_user_provider(
        prov_in, user_providers if isinstance(user_providers, dict) else {}
    )
    custom_provider = resolve_custom_provider(
        prov_in,
        get_compatible_custom_providers(cfg) if isinstance(cfg, dict) else [],
    )
    if user_provider is not None:
        return user_provider.id, model_in
    if custom_provider is not None:
        return custom_provider.id, model_in

    # A named custom provider that didn't resolve above (typo, config
    # mismatch, entry missing from custom_providers/providers) must still
    # not be treated as a stray vendor prefix -- it isn't a known Hermes
    # provider/alias, but it also isn't the analytics-vendor case this
    # fallback exists for. Match only the durable named-custom syntax
    # (bare "custom" bucket, or "custom:<name>" per
    # ``providers.custom_provider_slug``) -- a bare ``startswith("custom")``
    # would also swallow unrelated unconfigured vendor names that merely
    # happen to start with "custom" (e.g. "customproxy").
    is_custom_provider_slug = canonical == "custom" or canonical.startswith("custom:")
    if (
        canonical not in _KNOWN_PROVIDER_NAMES
        and not is_custom_provider_slug
        and "/" in model_in
    ):
        # Vendor prefix posing as a provider (analytics fallback). Resolve
        # against the user's current provider when it's an aggregator that
        # serves vendor-prefixed slugs; otherwise default to openrouter.
        try:
            cur_cfg = cfg.get("model", {})
            cur_provider = (
                str(cur_cfg.get("provider", "") or "").strip().lower()
                if isinstance(cur_cfg, dict) else ""
            )
        except Exception:
            cur_provider = ""
        from hermes_cli.models import _AGGREGATOR_PROVIDERS
        if cur_provider and normalize_provider(cur_provider) in _AGGREGATOR_PROVIDERS:
            canonical = normalize_provider(cur_provider)
            prov_in = cur_provider
        else:
            canonical = "openrouter"
            prov_in = "openrouter"

    # Custom/user-config providers keep the model verbatim — the registry
    # normalizer doesn't know their namespaces.
    if canonical in _KNOWN_PROVIDER_NAMES and not canonical.startswith("custom"):
        try:
            normalized_model = normalize_model_for_provider(model_in, canonical)
            if normalized_model:
                model_in = normalized_model
        except Exception:
            _log.debug("model normalization failed for %s/%s", prov_in, model_in, exc_info=True)

    return prov_in, model_in


def _apply_main_model_assignment(
    model_cfg: "Any", provider: str, model: str, base_url: str = "", api_key: str = ""
) -> dict:
    """Apply a main-slot model assignment to a ``model`` config dict in place.

    Sets ``provider``/``default``, then reconciles ``base_url``:

    - An explicitly supplied ``base_url`` is always persisted (covers
      ``custom``/local endpoints and any provider whose key is bound to a
      non-default host).
    - Otherwise, a stale ``base_url`` is cleared ONLY when switching to a
      *different* provider — that URL belonged to the old provider. When the
      provider is unchanged and no new URL is supplied, the existing
      ``base_url`` is preserved. This keeps a user's custom endpoint (e.g. a
      Xiaomi MiMo Token Plan host, ``https://token-plan-*.xiaomimimo.com/v1``)
      alive when they merely re-pick a model under the same provider — picking
      a model previously wiped it, forcing the registry default and breaking
      Token Plan keys.

    The runtime resolver reads ``model.base_url`` from config (it ignores
    ``OPENAI_BASE_URL``) and only honors it when the configured provider matches
    and the pool entry is on the registry default, so preserving it here is what
    lets the override actually route. The hardcoded ``context_length`` override
    is always dropped since the new model may have a different context window.

    Returns the same dict (coerced to a fresh dict if the input wasn't one) so
    callers can assign it straight back onto the model config.
    """
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    prev_provider = str(model_cfg.get("provider") or "").strip().lower()
    new_provider = provider.strip().lower()
    model_cfg["provider"] = provider
    model_cfg["default"] = model
    if base_url.strip():
        model_cfg["base_url"] = base_url.strip()
    elif model_cfg.get("base_url") and new_provider != prev_provider:
        # Switching providers: the old URL belonged to the old provider, drop
        # it so the new provider's default endpoint is used. Same-provider
        # re-assignment keeps the user's configured base_url intact.
        model_cfg["base_url"] = ""
    # The endpoint key follows the same lifecycle as base_url: an explicit key
    # is always persisted; an existing key is dropped only when switching to a
    # different provider (it belonged to the old endpoint), and preserved on a
    # same-provider re-pick so re-selecting a model doesn't wipe the key.
    if api_key.strip():
        model_cfg["api_key"] = api_key.strip()
        model_cfg.pop("api", None)
    elif (model_cfg.get("api_key") or model_cfg.get("api")) and new_provider != prev_provider:
        # A stale endpoint secret can live under the legacy ``api`` alias with
        # no ``api_key`` (the resolver still reads ``model.api`` as a key), so
        # the switch-clears-the-key path must trigger on either field — else the
        # old endpoint's secret survives in config.yaml and contaminates a later
        # custom resolution. clear_model_endpoint_credentials scrubs both.
        clear_model_endpoint_credentials(model_cfg, clear_api_mode=False)
    if new_provider != prev_provider:
        clear_model_endpoint_credentials(model_cfg, clear_api_key=False)
    model_cfg.pop("context_length", None)
    return model_cfg


def _normalize_config_for_web(config: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize config for the web UI.

    Hermes supports ``model`` as either a bare string (``"anthropic/claude-sonnet-4"``)
    or a dict (``{default: ..., provider: ..., base_url: ...}``).  The schema is built
    from DEFAULT_CONFIG where ``model`` is a string, but user configs often have the
    dict form.  Normalize to the string form so the frontend schema matches.

    Also surfaces ``model_context_length`` as a top-level field so the web UI can
    display and edit it.  A value of 0 means "auto-detect".
    """
    config = dict(config)  # shallow copy
    model_val = config.get("model")
    if isinstance(model_val, dict):
        # Extract context_length before flattening the dict
        ctx_len = model_val.get("context_length", 0)
        config["model"] = model_val.get("default", model_val.get("name", ""))
        config["model_context_length"] = ctx_len if isinstance(ctx_len, int) else 0
    else:
        config["model_context_length"] = 0
    return config


# ---------------------------------------------------------------------------
# Model assignment — pick provider+model for main slot or auxiliary slots.
# Mirrors the model.options JSON-RPC from tui_gateway but uses REST so the
# Models page (which has no chat PTY open) can drive it.
# ---------------------------------------------------------------------------

# Canonical auxiliary task slots. Keep in sync with DEFAULT_CONFIG["auxiliary"]
# in hermes_cli/config.py — listed here for deterministic ordering in the UI.
_AUX_TASK_SLOTS: Tuple[str, ...] = (
    "vision",
    "compression",
    "skills_hub",
    "approval",
    "mcp",
    "title_generation",
    "review",
    "triage_specifier",
    "kanban_decomposer",
    "profile_describer",
    "curator",
)


def _dashboard_code_skew_guard() -> Optional[str]:
    """Return a clear \"restart required\" message when this process runs stale code.

    The dashboard and Desktop-owned ``hermes serve`` are long-lived; their
    ``sys.modules`` is frozen at boot.  When ``hermes update`` (or a manual
    ``git pull``) replaces the checkout underneath them, a first-time lazy
    import on a new code path can resolve a freshly-pulled consumer module
    against a stale cached dependency -> ImportError — e.g. ``/api/model/options``
    500 after the update added ``agent.model_metadata.is_grok_46_family`` while
    the running process kept serving the pre-update module (#86207).  Mirror
    the gateway's ``_model_switch_skew_guard``: refuse the risky call with an
    actionable, deployment-aware message instead of crashing with a cryptic
    import error (#97046).

    Returns None when no drift is detectable (fresh process, or a non-git
    install where the boot fingerprint could not be read — never a false
    positive).
    """
    from gateway.code_skew import detect_code_skew

    skew = detect_code_skew()
    if not skew:
        return None
    boot_rev, disk_rev = skew
    return (
        f"This process is running code from {boot_rev} but the checkout on "
        f"disk is now {disk_rev}. The model picker would risk a stale-module "
        f"crash — {_dashboard_skew_restart_hint()}"
    )


def _dashboard_skew_restart_hint() -> str:
    """Restart advice that matches how this process is actually owned.

    The same FastAPI app backs the browser dashboard *and* Desktop-owned
    ``hermes serve --isolated`` (local or SSH). Hardcoding a systemd unit
    misleads macOS/launchd hosts and Desktop SSH backends, which have no
    ``hermes-dashboard`` unit (#97046).
    """
    if os.environ.get("HERMES_SERVE_HEADLESS") == "1":
        return (
            "restart the Desktop-owned backend to load the new code "
            "(use Restart backend in Hermes Desktop, or quit and reopen the app)"
        )
    return (
        "restart this Hermes process to load the new code "
        "(hermes dashboard --port <port>, or the equivalent service restart for this install)"
    )


def _apply_model_assignment_sync(
    scope: str, provider: str, model: str, task: str, base_url: str, api_key: str = ""
):
    """Synchronous body of POST /api/model/set.

    Runs inside ``_profile_scope`` (in a worker thread) so every
    load_config/save_config lands in the requested profile.  Raises
    HTTPException for validation errors — the async wrapper re-raises them.
    """
    from hermes_cli.web_server import load_config, save_config
    cfg = load_config()

    if scope == "main":
        if not provider or not model:
            raise HTTPException(status_code=400, detail="provider and model required for main")
        provider, model = _normalize_main_model_assignment(provider, model)
        providers_cfg = cfg.get("providers")
        provider_entry = providers_cfg.get(provider) if isinstance(providers_cfg, dict) else None
        if not base_url and isinstance(provider_entry, dict) and provider_entry.get("base_url"):
            base_url = str(provider_entry.get("base_url") or "").strip()
        model_cfg = _apply_main_model_assignment(
            cfg.get("model", {}), provider, model, base_url, api_key
        )
        _raw_assign_entry = None
        try:
            _stored, _raw_assign_entry = find_provider_entry(
                read_raw_config().get("providers"), provider
            )
        except Exception:
            _raw_assign_entry = None
        _assign_key_env = (
            str(_raw_assign_entry.get("key_env") or "").strip()
            if isinstance(_raw_assign_entry, dict)
            else ""
        )
        if _assign_key_env:
            # #88990: carry the credential POINTER, never a resolved secret.
            model_cfg["key_env"] = _assign_key_env
            model_cfg.pop("api_key", None)
        elif isinstance(provider_entry, dict) and provider_entry.get("api_key"):
            # #88990: provider_entry comes from load_config(), which expands
            # ${VAR} env refs to plaintext. Copying that resolved value into
            # model.api_key writes the SECRET into config.yaml (and recreates
            # it on every re-apply, even after the user deletes it by hand).
            # Prefer the raw ${VAR} template; only fall back to the expanded
            # value when the raw yaml itself stores the key as a literal (no
            # new exposure in that case).
            _raw_key = (
                str(_raw_assign_entry.get("api_key") or "").strip()
                if isinstance(_raw_assign_entry, dict)
                else ""
            )
            if _raw_key.startswith("${") and _raw_key.endswith("}"):
                model_cfg["api_key"] = _raw_key
            else:
                model_cfg["api_key"] = provider_entry["api_key"]
        cfg["model"] = model_cfg

        # When switching the main provider to Nous, mirror the CLI's
        # post-model-selection behaviour (hermes_cli/main.py
        # prompt_enable_tool_gateway / tools_config apply_nous_managed_defaults):
        # auto-route any *unconfigured* tools through the Nous Tool Gateway.
        # This is purely additive — apply_nous_managed_defaults skips every
        # tool where the user already has a direct key (FIRECRAWL_API_KEY,
        # FAL_KEY, etc.) or an explicit backend/provider in config, so it
        # never overwrites a user's own setup. GUI users thus land on the
        # gateway the same way CLI users do, without a separate prompt.
        gateway_tools: list[str] = []
        if provider.strip().lower() == "nous":
            try:
                from hermes_cli.nous_subscription import apply_nous_managed_defaults
                from hermes_cli.tools_config import _get_platform_tools

                enabled = _get_platform_tools(
                    cfg, "cli", include_default_mcp_servers=False
                )
                changed = apply_nous_managed_defaults(
                    cfg,
                    enabled_toolsets=enabled,
                    force_fresh=True,
                )
                gateway_tools = sorted(changed)
            except Exception:
                # Portal lookup hiccups / non-subscriber / non-nous gating
                # must never block saving the model assignment.
                _log.debug("apply_nous_managed_defaults skipped", exc_info=True)

        save_config(cfg)

        # Register a named ``custom_providers`` entry for a custom/local
        # endpoint, mirroring the ``hermes model`` custom flow
        # (_save_custom_provider). Without this the endpoint only lives in
        # ``model.*`` and the picker has no proper ready row for it — the
        # GUI then surfaces a "needs setup" dead-end on the bare ``custom``
        # provider. Dedups by base_url, so re-saving is idempotent.
        if provider.strip().lower() in {"custom", "local"} and base_url:
            try:
                from hermes_cli.main import _auto_provider_name, _save_custom_provider

                _save_custom_provider(
                    base_url,
                    api_key,
                    model,
                    name=_auto_provider_name(base_url),
                )
            except Exception:
                # Never block the assignment on the bookkeeping write —
                # model.* is already persisted and routable.
                _log.debug("custom_providers registration skipped", exc_info=True)

        # Surface auxiliary slots still pinned to a *different* provider than
        # the new main one. Switching the main model does NOT touch aux pins
        # (they're independent, sticky per-task overrides — see
        # auxiliary_client._resolve_auto). A user who switches main away from
        # a now-unpaid provider (e.g. nous with $0 balance) keeps paying 402s
        # on every background aux call until they reset those pins. We never
        # auto-clear them — pinning aux to a cheaper/different model is a
        # legitimate config — but we tell the caller so the UI can offer a
        # "reset to main" nudge instead of silently burning credits.
        new_provider = provider.strip().lower()
        stale_aux: list[dict] = []
        aux_cfg = cfg.get("auxiliary", {})
        if isinstance(aux_cfg, dict):
            for slot in _AUX_TASK_SLOTS:
                slot_cfg = aux_cfg.get(slot)
                if not isinstance(slot_cfg, dict):
                    continue
                slot_provider = str(slot_cfg.get("provider", "") or "").strip()
                if (
                    slot_provider
                    and slot_provider.lower() not in {"auto", ""}
                    and slot_provider.lower() != new_provider
                ):
                    stale_aux.append({
                        "task": slot,
                        "provider": slot_provider,
                        "model": str(slot_cfg.get("model", "") or ""),
                    })

        try:
            effective_config = load_config()
            effective_provider, effective_model = resolve_cron_model_drift_defaults(
                effective_config
            )
            cron_model_impact = build_cron_model_impact(
                current_provider=effective_provider or provider,
                current_model=effective_model or model,
                config=effective_config,
            )
        except Exception:
            _log.debug("cron model impact inspection failed", exc_info=True)
            cron_model_impact = build_cron_model_impact(config=cfg, jobs={})

        return {
            "ok": True,
            "scope": "main",
            "provider": provider,
            "model": model,
            "base_url": model_cfg.get("base_url", ""),
            "gateway_tools": gateway_tools,
            "stale_aux": stale_aux,
            "cron_model_impact": cron_model_impact,
        }

    # scope == "auxiliary"
    aux = cfg.get("auxiliary")
    if not isinstance(aux, dict):
        aux = {}

    if task == "__reset__":
        # Reset every slot to provider="auto", model="" — keeps other fields intact.
        for slot in _AUX_TASK_SLOTS:
            slot_cfg = aux.get(slot)
            if not isinstance(slot_cfg, dict):
                slot_cfg = {}
            slot_cfg["provider"] = "auto"
            slot_cfg["model"] = ""
            slot_cfg.pop("base_url", None)
            clear_model_endpoint_credentials(slot_cfg)
            aux[slot] = slot_cfg
        cfg["auxiliary"] = aux
        save_config(cfg)
        return {"ok": True, "scope": "auxiliary", "reset": True}

    if not provider:
        raise HTTPException(status_code=400, detail="provider required for auxiliary")

    targets = [task] if task else list(_AUX_TASK_SLOTS)
    for slot in targets:
        if slot not in _AUX_TASK_SLOTS:
            raise HTTPException(status_code=400, detail=f"unknown auxiliary task: {slot}")
        slot_cfg = aux.get(slot)
        if not isinstance(slot_cfg, dict):
            slot_cfg = {}
        prev_provider = str(slot_cfg.get("provider") or "").strip().lower()
        new_provider = provider.strip().lower()
        slot_cfg["provider"] = provider
        slot_cfg["model"] = model
        if base_url:
            # Sibling of the main-slot endpoint handling (#65254): an aux
            # assignment for a custom/local endpoint must carry its own
            # base_url, or the slot silently rebinds to whatever
            # model.base_url happens to hold — and breaks entirely once the
            # main slot switches away and clears it. The auxiliary resolver
            # already reads auxiliary.<task>.base_url/api_key
            # (_resolve_task_provider_model), so persisting them here is
            # what actually wires the endpoint in.
            slot_cfg["base_url"] = base_url
            if api_key:
                slot_cfg["api_key"] = api_key
        elif new_provider != prev_provider and new_provider != "custom":
            slot_cfg.pop("base_url", None)
            clear_model_endpoint_credentials(slot_cfg)
        aux[slot] = slot_cfg

    cfg["auxiliary"] = aux
    save_config(cfg)
    return {
        "ok": True,
        "scope": "auxiliary",
        "tasks": targets,
        "provider": provider,
        "model": model,
    }


def _infer_provider_on_model_change(model_val: str, prev_provider: str) -> tuple[str, str]:
    """Infer which provider serves ``model_val`` when the flat Config-page Model
    field changes, given the previously-saved ``prev_provider``.

    Returns ``(provider, model)``; ``provider`` is empty when no switch is
    warranted (leave the existing provider untouched). Two signals, in order:

    1. Curated-catalog detection (``detect_provider_for_model``) — handles the
       ~28 OpenRouter-curated models and direct provider-static catalogs.
    2. Vendor-slug heuristic — a ``vendor/model`` slug cannot belong to a
       single-model / non-aggregator provider (e.g. ``ollama-local``). When the
       current provider is not an aggregator that serves vendor-prefixed slugs,
       route to an aggregator. ``_normalize_main_model_assignment`` (called by
       the caller) keeps the user's current aggregator when they're already on
       one, else falls back to openrouter — the same chokepoint logic as
       ``POST /api/model/set``.
    """
    name = (model_val or "").strip()
    if not name:
        return "", name
    try:
        from hermes_cli.models import (
            _AGGREGATOR_PROVIDERS,
            detect_provider_for_model,
            normalize_provider,
        )
    except Exception:
        return "", name

    try:
        detected = detect_provider_for_model(name, prev_provider)
    except Exception:
        detected = None
    if detected:
        return detected[0], detected[1]

    # Vendor-prefixed slug under a non-aggregator provider → reassign. Use a
    # sentinel "openrouter" here; _normalize_main_model_assignment resolves the
    # real aggregator (keeps a current aggregator, else openrouter).
    if "/" in name:
        try:
            cur_is_aggregator = normalize_provider(prev_provider) in _AGGREGATOR_PROVIDERS
        except Exception:
            cur_is_aggregator = False
        if not cur_is_aggregator:
            return "openrouter", name

    return "", name


def _denormalize_config_from_web(config: Dict[str, Any]) -> Dict[str, Any]:
    """Reverse _normalize_config_for_web before saving.

    Reconstructs ``model`` as a dict by reading the current on-disk config
    to recover model subkeys (provider, base_url, api_mode, etc.) that were
    stripped from the GET response.  The frontend only sees model as a flat
    string; the rest is preserved transparently.

    Also handles ``model_context_length`` — writes it back into the model dict
    as ``context_length``.  A value of 0 means "auto-detect" (omitted from the
    dict so get_model_context_length() uses its normal resolution). ``config``
    may be a partial update (e.g. the Settings autosave diff) that omits
    ``model_context_length`` entirely when the user didn't touch it — that
    must leave the on-disk override untouched, not get treated the same as an
    explicit 0 and cleared.
    """
    from hermes_cli.web_server import load_config
    config = dict(config)
    # Remove any _model_meta that might have leaked in (shouldn't happen
    # with the stripped GET response, but be defensive)
    config.pop("_model_meta", None)

    # Extract and remove model_context_length before processing model, but
    # remember whether it was actually present: a partial update omitting the
    # key means "unchanged", which is different from an explicit 0.
    ctx_sent = "model_context_length" in config
    ctx_override = config.pop("model_context_length", 0)
    if not isinstance(ctx_override, int):
        try:
            ctx_override = int(ctx_override)
        except (TypeError, ValueError):
            ctx_override = 0

    model_val = config.get("model")
    if (isinstance(model_val, str) and model_val) or ctx_sent:
        # Read the current disk config to recover model subkeys
        try:
            disk_config = load_config()
            disk_model = disk_config.get("model")
            if isinstance(disk_model, dict):
                if isinstance(model_val, str) and model_val:
                    prev_default = str(disk_model.get("default") or "").strip()
                    prev_provider = str(disk_model.get("provider") or "").strip()
                    # When the model name actually changed, re-detect which
                    # provider serves it. The Config-page Model field is a flat
                    # string with no provider info, so without this a user who
                    # picks an OpenRouter model while their default provider is
                    # ollama-local keeps the stale provider and 404s. Only fires
                    # on a real model change so saving unrelated config fields
                    # never overwrites an explicit provider.
                    if model_val != prev_default and prev_provider:
                        new_provider, resolved_model = _infer_provider_on_model_change(
                            model_val, prev_provider
                        )
                        if new_provider and new_provider.strip().lower() != prev_provider.lower():
                            # Route through the canonical assignment chokepoints so
                            # the model is normalized for the new provider and stale
                            # base_url/api_mode/api_key are cleared on the switch
                            # (and preserved on a same-provider re-pick).
                            norm_provider, norm_model = _normalize_main_model_assignment(
                                new_provider, resolved_model
                            )
                            disk_model = _apply_main_model_assignment(
                                disk_model, norm_provider, norm_model
                            )
                            model_val = norm_model
                    # Preserve all subkeys, update default with the new value
                    disk_model["default"] = model_val
                # Write context_length into the model dict (0 = remove/auto),
                # but only when the payload actually carried the key.
                if ctx_sent:
                    if ctx_override > 0:
                        disk_model["context_length"] = ctx_override
                    else:
                        disk_model.pop("context_length", None)
                config["model"] = disk_model
            # Model was previously a bare string (or absent) — upgrade to a
            # dict if the user is setting a context_length override.
            elif ctx_sent and ctx_override > 0:
                if isinstance(model_val, str) and model_val:
                    default = model_val
                elif isinstance(disk_model, str) and disk_model:
                    default = disk_model
                else:
                    default = ""
                config["model"] = {
                    "default": default,
                    "context_length": ctx_override,
                }
        except Exception:
            pass  # can't read disk config — just use the string form
    return config
