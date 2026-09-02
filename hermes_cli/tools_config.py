"""Unified tool configuration for Hermes Agent."""

import json as _json
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set


from hermes_cli.config import (
    cfg_get,
    load_config, save_config, get_env_value, save_env_value,
)
from hermes_cli.colors import Colors, color
from hermes_cli.nous_subscription import (
    MANAGED_FEATURE_COVERAGE_CATEGORY,
    NousSubscriptionFeatures,
    apply_nous_managed_defaults,
    get_nous_subscription_features,
)
from hermes_cli.nous_account import format_nous_portal_entitlement_message
from hermes_cli.toolset_scope import (
    _TOOLSET_PLATFORM_RESTRICTIONS,
    toolset_allowed_for_platform as _toolset_allowed_for_platform,
)
from tools.tool_backend_helpers import NOUS_MANAGED_PROVIDER, fal_key_is_configured
from utils import base_url_hostname, is_truthy_value

logger = logging.getLogger(__name__)


# Platforms already warned about an all-invalid platform_toolsets list, so the
# runtime check in _get_platform_tools warns once per platform instead of on
# every tool resolution for a persistently-corrupt config (#38798).
_warned_invalid_platform_toolsets: Set[str] = set()

PROJECT_ROOT = Path(__file__).parent.parent.resolve()


# ─── UI Helpers (shared with setup.py) ────────────────────────────────────────

from hermes_cli.cli_output import (  # noqa: E402 — late import block
    print_error as _print_error,
    print_info as _print_info,
    print_success as _print_success,
    print_warning as _print_warning,
    prompt as _prompt,
)
from hermes_cli.tools_config_cua import (  # noqa: F401 — re-exported for hermes_cli.tools_config.X callers and test patches
    _post_setup_no_window_flags,
    _cua_driver_cmd,
    _cua_version_summary,
    _resolved_cua_driver_cmd,
    _cua_driver_env,
    _CUA_DRIVER_CONTRACT_CACHE,
    _cua_driver_contract_status,
    _cua_driver_install_ready,
    _pip_install,
    _cua_install_target_writable,
    _cua_driver_version,
    install_cua_driver,
    _CUA_INSTALLER_TIMEOUT,
    _CUA_INSTALLER_DRAIN_GRACE,
    _CUA_BACKGROUND_UPDATE_TIMEOUT,
    _CUA_LOCK_STALE_AFTER,
    _cua_install_home,
    _cua_install_lock_dir,
    _cua_windows_install_lock_file,
    _clear_stale_windows_cua_install_lock,
    _clear_stale_cua_install_lock,
    _cua_install_lock_held,
    _cua_release_endpoint_reachable,
    _ps_single_quote,
    _cua_driver_autostart_registered_windows,
    _repair_cua_driver_autostart_windows,
    _remove_quietly,
    _print_cua_platform_notes,
    _run_cua_driver_installer,
)
from hermes_cli.tools_config_providers import (  # noqa: F401 — re-exported for hermes_cli.tools_config.X callers and test patches
    _plugin_provider_rows,
    _plugin_image_gen_providers,
    _plugin_video_gen_providers,
    _plugin_web_search_providers,
    _plugin_browser_providers,
    _plugin_tts_providers,
    web_provider_capabilities,
    _PLUGIN_ROW_BUILDERS,
    _visible_providers,
    provider_readiness_status,
    _toolset_needs_configuration_prompt,
    _any_plugin_provider_available,
    _configure_tool_category,
    _web_tier_matches,
    _is_provider_active,
    _detect_active_provider_index,
    _fal_model_catalog,
    IMAGEGEN_BACKENDS,
    _plugin_model_catalog,
    _plugin_image_gen_catalog,
    _plugin_video_gen_catalog,
    _pick_model_from_catalog,
    _configure_imagegen_model,
    _configure_imagegen_model_for_plugin,
    _configure_videogen_model_for_plugin,
    _configure_xai_imagine_storage,
    _select_plugin_gen_provider,
    _select_plugin_image_gen_provider,
    _select_plugin_video_gen_provider,
    STT_MODEL_CATALOG,
    _STT_MODEL_CONFIG_KEY,
    _configure_stt_model,
    _PROVIDER_MARKER_SECTIONS,
    _write_provider_config,
    apply_provider_selection,
    _nous_provider_gate,
    _finish_provider_selection,
    _print_provider_selection,
    _configure_provider,
    _reconfigure_provider,
    _configure_vision_backend,
    _configure_vision_provider_model,
    _configure_simple_requirements,
)

# ─── Toolset Registry ─────────────────────────────────────────────────────────

# Toolsets shown in the configurator, grouped for display.
# Each entry: (toolset_name, label, description)
# These map to keys in toolsets.py TOOLSETS dict.
CONFIGURABLE_TOOLSETS = [
    ("web",             "🔍 Web Search & Scraping",    "web_search, web_extract"),
    ("browser",         "🌐 Browser Automation",       "navigate, click, type, scroll"),
    ("terminal",        "💻 Terminal & Processes",      "terminal, process"),
    ("file",            "📁 File Operations",           "read, write, patch, search"),
    ("code_execution",  "⚡ Code Execution",            "execute_code"),
    ("vision",          "👁️  Vision / Image Analysis",  "vision_analyze"),
    ("video",           "🎬 Video Analysis",            "video_analyze (requires video-capable model)"),
    ("image_gen",       "🎨 Image Generation",          "image_generate"),
    ("video_gen",       "🎬 Video Generation",          "video_generate (text/image/reference)"),
    ("x_search",        "🐦 X (Twitter) Search",        "x_search (requires xAI OAuth or XAI_API_KEY)"),
    ("tts",             "🔊 Text-to-Speech",            "text_to_speech"),
    ("stt",             "🎙️ Speech-to-Text",           "voice transcription (gateway voice messages + voice mode)"),
    ("skills",          "📚 Skills",                    "list, view, manage"),
    ("todo",            "📋 Task Planning",             "todo_list"),
    ("memory",          "💾 Memory",                    "persistent memory across sessions"),
    ("context_engine",  "🧩 Context Engine",            "runtime tools from the active context engine"),
    ("session_search",  "🔎 Session Search",            "search past conversations"),
    ("clarify",         "❓ Clarifying Questions",      "clarify"),
    ("delegation",      "👥 Task Delegation",           "delegate_task"),
    ("cronjob",         "⏰ Cron Jobs",                 "create/list/update/pause/resume/run, with optional attached skills"),
    ("homeassistant",    "🏠 Home Assistant",           "smart home device control"),
    ("spotify",          "🎵 Spotify",                  "playback, search, playlists, library"),
    ("discord",         "💬 Discord (read/participate)", "fetch messages, search members, create thread"),
    ("discord_admin",   "🛡️  Discord Server Admin",    "list channels/roles, pin, assign roles"),
    ("yuanbao",          "🤖 Yuanbao",                  "group info, member queries, DM"),
    ("computer_use",     "🖱️  Computer Use (macOS/Windows/Linux)", "background desktop control via cua-driver"),
]


def gui_toolset_label(label: str) -> str:
    """Strip leading emoji/icons from toolset titles for GUI surfaces.

    Registry labels use ``<emoji> <title>``; plugin toolsets prefix with ``🔌``. CLI/TUI keeps the
    raw ``label`` — only HTTP APIs call this helper.
    """
    text = (label or "").strip()
    if not text:
        return text
    parts = text.split(None, 1)
    if len(parts) == 2 and parts[0] and not any(ch.isascii() and ch.isalnum() for ch in parts[0]):
        return parts[1].strip()
    return text


# Toolsets that are OFF by default for new installs.
# They're still in _HERMES_CORE_TOOLS (available at runtime if enabled),
# but the setup checklist won't pre-select them for first-time users.
#
# Video gen is off by default — it's a niche, paid, slow feature. Users
# who want it opt in via `hermes tools` → Video Generation, which walks
# them through provider + model selection.
#
# X search is off by default for users without xAI credentials, but
# auto-enables when SuperGrok OAuth tokens are stored OR XAI_API_KEY is
# set — mirroring the HASS_TOKEN → homeassistant auto-enable below. The
# `hermes tools` → X (Twitter) Search setup walks users through credential
# setup. The tool's check_fn means the schema still won't appear to the
# model if the credential later goes missing or expires.
_DEFAULT_OFF_TOOLSETS = {"homeassistant", "spotify", "discord", "discord_admin", "video", "video_gen", "x_search", "a2a"}


# Config-only capabilities: they appear in `hermes tools` for provider/API-key
# configuration (TOOL_CATEGORIES) but are NOT model toolsets — they ship zero
# tool schemas and their on/off switch lives in their own config section
# (e.g. ``stt.enabled``), not ``platform_toolsets``. Excluded from the
# per-platform enable/disable checklist; configured via the "Reconfigure an
# existing tool" flow and the GUI provider matrix instead.
_CONFIG_ONLY_TOOLSETS = {"stt"}


def _xai_credentials_present() -> bool:
    """Cheap, side-effect-free check for usable xAI credentials.

    Does NOT hit the network — only inspects the local auth store and environment. The tool's
    runtime ``check_fn`` still gates schema registration if creds later expire or get revoked. Also
    reused by ``provider_readiness_status`` for ``post_setup: "xai_grok"`` picker rows (xAI TTS,
    Grok OAuth x_search).
    """
    try:
        from hermes_cli.auth import _read_xai_oauth_tokens

        _read_xai_oauth_tokens()
        return True
    except Exception:
        pass
    try:
        from tools.xai_http import get_env_value as _xai_get_env_value

        if str(_xai_get_env_value("XAI_API_KEY") or "").strip():
            return True
    except Exception:
        pass
    try:
        from agent.secret_scope import get_secret
    except ImportError:  # pragma: no cover — secret_scope is in-repo
        return bool(str(os.environ.get("XAI_API_KEY") or "").strip())
    return bool(str(get_secret("XAI_API_KEY") or "").strip())


def _homeassistant_credentials_present() -> bool:
    """Return whether the active profile has a Home Assistant token."""
    try:
        from agent.secret_scope import get_secret

        return bool((get_secret("HASS_TOKEN", "") or "").strip())
    except Exception:
        return False

def _toolset_configuration_platform(ts_key: str, default: str = "cli") -> str:
    """Return the platform a platform-less configuration UI should target.

    Most configurable toolsets retain the historical desktop/CLI target. A toolset restricted away
    from that platform must instead be configured on one of its supported platforms; otherwise the
    shared save helper correctly drops it and the UI reports a successful no-op.
    """
    allowed = _TOOLSET_PLATFORM_RESTRICTIONS.get(ts_key)
    if not allowed or default in allowed:
        return default
    return sorted(allowed)[0]


def _get_effective_configurable_toolsets():
    """Return CONFIGURABLE_TOOLSETS + any plugin-provided toolsets.

    Plugin toolsets are appended at the end so they appear after the built-in toolsets in the TUI
    checklist. A plugin whose toolset key already appears in ``CONFIGURABLE_TOOLSETS`` is skipped —
    bundled plugins (e.g.
    """
    result = list(CONFIGURABLE_TOOLSETS)
    seen = {ts_key for ts_key, _, _ in result}
    try:
        from hermes_cli.plugins import discover_plugins, get_plugin_toolsets
        discover_plugins()  # idempotent — ensures plugins are loaded
        for entry in get_plugin_toolsets():
            if entry[0] in seen:
                continue
            seen.add(entry[0])
            result.append(entry)
    except Exception:
        pass
    return result


def _get_plugin_toolset_keys() -> set:
    """Return the set of toolset keys provided by plugins."""
    try:
        from hermes_cli.plugins import get_plugin_toolset_keys_nowait
        # Non-blocking on the CLI startup path: while background plugin
        # discovery is still importing modules, this serves last launch's
        # persisted key set (used only to exclude plugin toolsets from
        # composite expansion) instead of joining the discovery thread.
        return get_plugin_toolset_keys_nowait()
    except Exception:
        return set()


def _checklist_toolset_keys(platform: str) -> Set[str]:
    """Return the toolset keys the ``hermes tools`` checklist actually offers for ``platform``.

    This mirrors exactly what ``_prompt_toolset_checklist`` renders:
    ``_get_effective_configurable_toolsets()`` (built-in + plugin toolsets), filtered by
    ``_toolset_allowed_for_platform``. The checklist's returned selection can therefore only ever be
    a subset of this universe.

    Non-configurable toolsets that ``_get_platform_tools`` resolves at read time — ``kanban`` and
    other check_fn-gated toolsets, recovered platform composites, MCP server names — are NOT in this
    set because the checklist never shows them.
    """
    return {
        ts_key
        for ts_key, _, _ in _get_effective_configurable_toolsets()
        if _toolset_allowed_for_platform(ts_key, platform)
        and ts_key not in _CONFIG_ONLY_TOOLSETS
    }

# Platform display config — derived from the canonical registry so every
# module shares the same data.  Kept as dict-of-dicts for backward
# compatibility with existing ``PLATFORMS[key]["label"]`` access patterns.
from hermes_cli.platforms import PLATFORMS as _PLATFORMS_REGISTRY

PLATFORMS = {
    k: {"label": info.label, "default_toolset": info.default_toolset}
    for k, info in _PLATFORMS_REGISTRY.items()
}


def _platform_default_toolset(platform: str) -> str:
    """Composite toolset a platform falls back to (plugin platforms derive ``hermes-<platform>``)."""
    plat_info = PLATFORMS.get(platform)
    return plat_info["default_toolset"] if plat_info else f"hermes-{platform}"


def _cfg_section(config: dict, key: str) -> dict:
    """Return ``config[key]`` as a dict, replacing a missing or non-dict value with ``{}``."""
    section = config.setdefault(key, {})
    if not isinstance(section, dict):
        section = {}
        config[key] = section
    return section


def _is_configurable(ts_key: str) -> bool:
    """True when the toolset has provider options or simple env-var requirements to prompt for."""
    return bool(TOOL_CATEGORIES.get(ts_key) or TOOLSET_ENV_REQUIREMENTS.get(ts_key))


def _toolset_label(ts_key: str) -> str:
    """Display label for a toolset key (built-in or plugin), falling back to the key itself."""
    return next((l for k, l, _ in _get_effective_configurable_toolsets() if k == ts_key), ts_key)


# ─── Tool Categories (provider-aware configuration) ──────────────────────────
# Maps toolset keys to their provider options. When a toolset is newly enabled,
# we use this to show provider selection and prompt for the right API keys.
# Toolsets not in this map either need no config or use the simple fallback.

TOOL_CATEGORIES = {
    "tts": {
        "name": "Text-to-Speech",
        "icon": "🔊",
        "providers": [
            {"name": "Microsoft Edge TTS", "badge": "★ recommended · free",
             "tag": "Good quality, no API key needed", "env_vars": [],
             "tts_provider": "edge"},
            {"name": "Nous Subscription", "badge": "subscription",
             "tag": "Managed OpenAI TTS billed to your subscription", "env_vars": [],
             "tts_provider": "openai", "requires_nous_auth": True, "managed_nous_feature": "tts",
             "override_env_vars": ["VOICE_TOOLS_OPENAI_KEY", "OPENAI_API_KEY"]},
            {"name": "OpenAI TTS", "badge": "paid", "tag": "High quality voices",
             "env_vars": [
                 {"key": "VOICE_TOOLS_OPENAI_KEY", "prompt": "OpenAI API key", "url": "https://platform.openai.com/api-keys"},
             ],
             "tts_provider": "openai"},
            {"name": "xAI TTS", "tag": "Grok voices — uses xAI Grok OAuth or XAI_API_KEY", "env_vars": [],
             "tts_provider": "xai", "post_setup": "xai_grok"},
            {"name": "ElevenLabs", "badge": "paid", "tag": "Most natural voices",
             "env_vars": [
                 {"key": "ELEVENLABS_API_KEY", "prompt": "ElevenLabs API key", "url": "https://elevenlabs.io/app/settings/api-keys"},
             ],
             "tts_provider": "elevenlabs"},
            # Mistral Voxtral TTS — `mistralai` SDK lazy-installs on first use.
            {"name": "Mistral (Voxtral TTS)", "badge": "paid", "tag": "Multilingual, native Opus",
             "env_vars": [{"key": "MISTRAL_API_KEY", "prompt": "Mistral API key", "url": "https://console.mistral.ai/"}],
             "tts_provider": "mistral"},
            {"name": "Google Gemini TTS", "badge": "preview",
             "tag": "30 prebuilt voices, controllable via prompts",
             "env_vars": [
                 {"key": "GEMINI_API_KEY", "prompt": "Gemini API key", "url": "https://aistudio.google.com/app/apikey"},
             ],
             "tts_provider": "gemini"},
            {"name": "KittenTTS", "badge": "local · free",
             "tag": "Lightweight local ONNX TTS (~25MB), no API key", "env_vars": [],
             "tts_provider": "kittentts", "post_setup": "kittentts"},
            {"name": "Piper", "badge": "local · free",
             "tag": "Local neural TTS, 44 languages (voices ~20-90MB)", "env_vars": [],
             "tts_provider": "piper", "post_setup": "piper"},
            {"name": "DeepInfra TTS", "badge": "paid",
             "tag": "Chatterbox, Qwen3-TTS, … — live catalog from api.deepinfra.com",
             "env_vars": [
                 {"key": "DEEPINFRA_API_KEY", "prompt": "DeepInfra API key", "url": "https://deepinfra.com/dash/api_keys"},
             ],
             "tts_provider": "deepinfra"},
        ],
    },
    "stt": {
        "name": "Speech-to-Text",
        "icon": "🎙️",
        "providers": [
            {"name": "Local Whisper", "badge": "★ recommended · free",
             "tag": "faster-whisper on-device, no API key", "env_vars": [],
             "stt_provider": "local", "post_setup": "faster_whisper"},
            {"name": "Nous Subscription", "badge": "subscription",
             "tag": "Managed OpenAI transcription billed to your subscription", "env_vars": [],
             "stt_provider": "openai", "requires_nous_auth": True, "managed_nous_feature": "stt",
             "override_env_vars": ["VOICE_TOOLS_OPENAI_KEY", "OPENAI_API_KEY"]},
            {"name": "OpenAI", "badge": "paid", "tag": "whisper-1, gpt-4o-transcribe, gpt-transcribe",
             "env_vars": [
                 {"key": "VOICE_TOOLS_OPENAI_KEY", "prompt": "OpenAI API key", "url": "https://platform.openai.com/api-keys"},
             ],
             "stt_provider": "openai"},
            {"name": "Groq", "badge": "free tier", "tag": "Whisper large-v3 family — very fast",
             "env_vars": [{"key": "GROQ_API_KEY", "prompt": "Groq API key", "url": "https://console.groq.com/keys"}],
             "stt_provider": "groq"},
            {"name": "xAI", "tag": "grok-stt — uses xAI Grok OAuth or XAI_API_KEY", "env_vars": [],
             "stt_provider": "xai", "post_setup": "xai_grok"},
            {"name": "ElevenLabs Scribe", "badge": "paid",
             "tag": "scribe_v2 — diarization + audio-event tagging",
             "env_vars": [
                 {"key": "ELEVENLABS_API_KEY", "prompt": "ElevenLabs API key", "url": "https://elevenlabs.io/app/settings/api-keys"},
             ],
             "stt_provider": "elevenlabs"},
            # Mistral Voxtral STT intentionally omitted — mistralai PyPI
            # package quarantined (malicious 2.4.6 release, 2026-05-12).
            # Restore alongside the dashboard stt.provider option.
            {"name": "DeepInfra", "badge": "paid", "tag": "Live STT catalog from api.deepinfra.com",
             "env_vars": [
                 {"key": "DEEPINFRA_API_KEY", "prompt": "DeepInfra API key", "url": "https://deepinfra.com/dash/api_keys"},
             ],
             "stt_provider": "deepinfra"},
        ],
    },
    "web": {
        "name": "Web Search & Extract",
        "setup_title": "Select Search Provider",
        "setup_note": "A free DuckDuckGo search skill is also included — skip this if you don't need a premium provider.",
        "icon": "🔍",
        # Provider rows come from plugins.web.<vendor> via _plugin_web_search_providers()
        # (PR #25182). Only the two non-provider firecrawl setup-flow rows live here:
        # managed Firecrawl via Nous subscription, and a self-hosted FIRECRAWL_API_URL.
        "providers": [
            {"name": "Nous Subscription", "badge": "subscription",
             "tag": "Managed Firecrawl billed to your subscription", "web_backend": "firecrawl",
             "env_vars": [],
             "requires_nous_auth": True, "managed_nous_feature": "web",
             "override_env_vars": ["FIRECRAWL_API_KEY", "FIRECRAWL_API_URL"]},
            {"name": "Firecrawl Self-Hosted", "badge": "free · self-hosted",
             "tag": "Run your own Firecrawl instance (Docker)", "web_backend": "firecrawl",
             "env_vars": [{"key": "FIRECRAWL_API_URL", "prompt": "Your Firecrawl instance URL (e.g., http://localhost:3002)"}]},
        ],
    },
    "image_gen": {
        "name": "Image Generation",
        "icon": "🎨",
        # Provider rows (FAL, OpenAI, OpenAI Codex, xAI) come from plugins.image_gen.<vendor>
        # via _plugin_image_gen_providers(). Only the managed "Nous Subscription" setup-flow row
        # lives here — it uses the fal plugin as backend but has a distinct UX.
        "providers": [
            {"name": "Nous Subscription", "badge": "subscription",
             "tag": "Managed FAL image generation billed to your subscription", "env_vars": [],
             "requires_nous_auth": True, "managed_nous_feature": "image_gen",
             "override_env_vars": ["FAL_KEY"],
             "imagegen_backend": "fal"},
        ],
    },
    "video_gen": {
        "name": "Video Generation",
        "icon": "🎬",
        # "Nous Subscription" row mirrors the image_gen pattern — managed
        # FAL video generation billed via the Nous Portal.  Plugin-backed
        # provider rows (FAL BYOK, xAI, …) are injected at runtime by
        # ``_plugin_video_gen_providers()`` in ``_visible_providers``.
        "providers": [
            {
                "name": "Nous Subscription",
                "badge": "subscription",
                "tag": "Managed FAL video generation billed to your subscription",
                "env_vars": [],
                "requires_nous_auth": True,
                "managed_nous_feature": "video_gen",
                "override_env_vars": ["FAL_KEY"],
                # The underlying plugin backend — when the user picks
                # "Nous Subscription" we set video_gen.provider = "fal"
                # and video_gen.use_gateway = True so the FAL plugin
                # routes through the managed queue gateway.
                "video_gen_plugin_name": "fal",
            },
        ],
    },
    "x_search": {
        "name": "X (Twitter) Search",
        "setup_title": "Select xAI Credential Source",
        "setup_note": (
            "Hermes routes X searches through xAI's built-in x_search "
            "Responses tool for read-only public X discovery. Use the xurl "
            "skill for authenticated X API reads and account actions. Both "
            "credential sources hit the same "
            "https://api.x.ai/v1/responses endpoint — pick whichever you "
            "already have. SuperGrok OAuth is preferred when both are set "
            "(uses your subscription quota instead of API spend)."
        ),
        "icon": "🐦",
        "providers": [
            {"name": "xAI Grok OAuth (SuperGrok / Premium+)", "badge": "subscription",
             "tag": "Browser login at accounts.x.ai — no API key required", "env_vars": [],
             "post_setup": "xai_grok"},
            {"name": "xAI API key", "badge": "paid", "tag": "Direct xAI API billing via XAI_API_KEY",
             "env_vars": [{"key": "XAI_API_KEY", "prompt": "xAI API key", "url": "https://console.x.ai/"}]},
        ],
    },
    "browser": {
        "name": "Browser Automation",
        "icon": "🌐",
        # Cloud provider rows (Browserbase, Browser Use, Firecrawl) come from
        # plugins.browser.<vendor> via _plugin_browser_providers() (PR #25214). Only
        # non-provider setup-flow rows live here. "Local Browser" MUST stay first so a fresh
        # install's Enter lands on the free local backend (index 0), never on the paid Nous row.
        # Lightpanda is local too (cloud_provider: local, browser.engine: lightpanda — Browser Use
        # mode spawns ``lightpanda serve``, built-in tools use ``agent-browser --engine
        # lightpanda``; no Chromium). Camofox short-circuits the cloud dispatch via
        # _is_camofox_mode().
        "providers": [
            {"name": "Local Browser", "badge": "★ recommended · free",
             "tag": "Headless Chromium, no API key needed", "env_vars": [],
             "browser_provider": "local", "browser_engine": "auto", "post_setup": "agent_browser"},
            {"name": "Lightpanda", "badge": "free · local · no Chromium",
             "tag": "Zig headless browser spawned by Hermes, text-only (no screenshots)", "env_vars": [],
             "browser_provider": "local", "browser_engine": "lightpanda", "post_setup": "lightpanda"},
            {
                "name": "Nous Subscription (Browser Use cloud)",
                "badge": "subscription",
                "tag": "Managed Browser Use billed to your subscription",
                "env_vars": [],
                "browser_provider": "browser-use",
                "requires_nous_auth": True,
                "managed_nous_feature": "browser",
                "override_env_vars": ["BROWSER_USE_API_KEY"],
                # Cloud hook: installs the agent-browser CLI only. Browser Use
                # hosts its own Chromium, so the local-Chromium install (and
                # the local-Chromium readiness gate) must not apply here —
                # with "agent_browser" this row read "needs setup" forever on
                # machines without a local Chromium build.
                "post_setup": "browserbase",
            },
            {"name": "Camofox", "badge": "free · local", "tag": "Anti-detection browser (Firefox/Camoufox)",
             "env_vars": [
                 {"key": "CAMOFOX_URL", "prompt": "Camofox server URL", "default": "http://localhost:9377", "url": "https://github.com/jo-inc/camofox-browser"},
             ],
             "browser_provider": "camofox", "post_setup": "camofox"},
            {"name": "Browser Use", "badge": "free · local · cloud", "tag": "New SOTA web harness (CLI 3.0)",
             "env_vars": [],
             "browser_backend": "browser-use", "post_setup": "browser_use_cli"},
        ],
    },
    "homeassistant": {
        "name": "Smart Home",
        "icon": "🏠",
        "providers": [
            {"name": "Home Assistant", "tag": "REST API integration",
             "env_vars": [
                 {"key": "HASS_TOKEN", "prompt": "Home Assistant Long-Lived Access Token"},
                 {"key": "HASS_URL", "prompt": "Home Assistant URL", "default": "http://homeassistant.local:8123"},
             ]},
        ],
    },
    "spotify": {
        "name": "Spotify",
        "icon": "🎵",
        "providers": [
            {"name": "Spotify Web API", "tag": "PKCE OAuth — opens the setup wizard", "env_vars": [],
             "post_setup": "spotify"},
        ],
    },
    "computer_use": {
        "name": "Computer Use (macOS/Windows/Linux)",
        "icon": "🖱️",
        # Runtime backends ship for macOS, Windows, and Linux (X11 today,
        # Wayland via XWayland). Per-host gaps surface via `computer-use doctor`.
        "platform_gate": ["darwin", "win32", "linux"],
        "providers": [
            {
                "name": "cua-driver (background)",
                "badge": "★ recommended · free · local",
                "tag": (
                    "Background computer-use via cua-driver — does NOT steal "
                    "your cursor or focus. Works with any model."
                ),
                "env_vars": [
                    # cua-driver reads HOME/TMPDIR from the process env, no
                    # extra keys required. Set HERMES_CUA_DRIVER_CMD to use a
                    # specific binary (e.g. a local build); there is no
                    # version-pin env var.
                ],
                "computer_use_backend": "cua",
                "post_setup": "cua_driver",
            },
        ],
    },
    "langfuse": {
        "name": "Langfuse Observability",
        "icon": "📊",
        "providers": [
            {"name": "Langfuse Cloud", "tag": "Hosted Langfuse (cloud.langfuse.com)",
             "env_vars": [
                 {"key": "HERMES_LANGFUSE_PUBLIC_KEY", "prompt": "Langfuse public key (pk-lf-...)", "url": "https://cloud.langfuse.com"},
                 {"key": "HERMES_LANGFUSE_SECRET_KEY", "prompt": "Langfuse secret key (sk-lf-...)", "url": "https://cloud.langfuse.com"},
             ],
             "post_setup": "langfuse"},
            {"name": "Langfuse Self-Hosted", "tag": "Self-hosted Langfuse instance",
             "env_vars": [
                 {"key": "HERMES_LANGFUSE_PUBLIC_KEY", "prompt": "Langfuse public key (pk-lf-...)"},
                 {"key": "HERMES_LANGFUSE_SECRET_KEY", "prompt": "Langfuse secret key (sk-lf-...)"},
                 {"key": "HERMES_LANGFUSE_BASE_URL", "prompt": "Langfuse server URL (e.g. http://localhost:3000)", "default": "http://localhost:3000"},
             ],
             "post_setup": "langfuse"},
        ],
    },
}

# Simple env-var requirements for toolsets NOT in TOOL_CATEGORIES.
# Used as a fallback for toolsets that just need an API key.
#
# `vision` is listed here only so it registers as a *configurable* toolset
# (the value gates the reconfigure menu + the "[no API key]" suffix). Its
# actual setup runs through `_configure_vision_backend()` — a full
# provider+model picker like `hermes model` — NOT this single-key prompt, so
# users are never forced onto OpenRouter. `_toolset_has_keys("vision")`
# resolves via `resolve_vision_provider_client()`, so the tuple below is never
# prompted or read for vision; it's purely a presence marker.
TOOLSET_ENV_REQUIREMENTS = {
    "vision":     [("OPENROUTER_API_KEY",   "https://openrouter.ai/keys")],
}


# ─── Post-Setup Hooks ─────────────────────────────────────────────────────────


def _ensure_browser_use_cli(*, verbose_hints: bool = False) -> None:
    """Install the Browser Use CLI if it isn't already runnable.

    The Browser Use CLI 3.0 is the primary driver engine for EVERY browser backend except Camofox
    (which is Firefox-based with no CDP surface, so the CDP-only browser-use harness cannot drive
    it).

    MANAGED-FIRST: a browser-use on the user's PATH does NOT satisfy this check — only the Hermes-
    managed ``$HERMES_HOME/bin`` copy does.
    """
    _print_info("    Ensuring browser-use CLI (managed install)...")
    try:
        from tools.browser_use_cli import install_cli

        ok, message = install_cli()
    except Exception as exc:  # pragma: no cover — defensive
        ok, message = False, f"install failed: {exc}"
    if ok:
        _print_success(f"    {message}")
    else:
        for line in str(message).splitlines():
            _print_warning(f"    {line[:200]}")
        if shutil.which("uvx"):
            _print_info("    Falling back to zero-install runs via `uvx browser-use`")
        else:
            _print_info("    Install manually: uv tool install browser-use  (https://docs.astral.sh/uv/)")
    if verbose_hints:
        _print_info("    Local Chrome needs remote debugging: chrome://inspect/#remote-debugging")
        _print_info("    Cloud browsers: browser-use auth login  (or set BROWSER_USE_API_KEY)")


def _post_setup_lightpanda() -> None:
    # Browser Use mode drives Lightpanda directly (Hermes spawns
    # ``lightpanda serve``); the built-in tools go through agent-browser.
    # Neither needs a Chromium build.
    _ensure_browser_use_cli()
    from tools.browser_lightpanda import (
        LIGHTPANDA_INSTALL_HINT,
        find_lightpanda_binary,
    )

    lightpanda_bin = find_lightpanda_binary()
    if lightpanda_bin:
        _print_success(f"    Lightpanda found: {lightpanda_bin}")
    else:
        _print_warning(
            "    lightpanda binary not found on PATH, ~/.lightpanda or ~/.local/bin"
        )
        _print_info(f"    {LIGHTPANDA_INSTALL_HINT}")
        if os.name == "nt":
            _print_info("    Lightpanda has no native Windows build; run Hermes under WSL2.")


def _post_setup_agent_browser(post_setup_key: str) -> None:
    """``agent_browser`` (local Chromium) and ``browserbase`` (cloud rows) hooks.

    agent-browser is no longer a root package.json dependency (#43564) — it resolves lazily via npx
    (or a global/Hermes-managed install), so there is no ``npm install`` step here.
    """
    # Every non-Camofox browser backend drives through the Browser Use
    # CLI when it's runnable — install it here too, not only on the
    # explicit "Browser Use" picker row.
    _ensure_browser_use_cli()
    try:
        # Import lazily so the tools_config UI doesn't pull in the full
        # browser_tool module at import time.
        from tools.browser_tool import (
            _chromium_installed,
            _running_in_docker,
            _find_agent_browser,
            _resolve_npx_bin,
            _is_npx_agent_browser_sentinel,
            AGENT_BROWSER_NPX_SPEC,
        )
    except Exception as exc:  # pragma: no cover — defensive
        _print_warning(f"    Could not check Chromium status: {exc}")
        return

    # Reuse the same resolution cascade browser tools use at runtime
    # (PATH -> Homebrew/Hermes-managed node -> npx) rather than a bare
    # shutil.which — Hermes-managed-Node-only setups resolve agent-browser
    # / npx only through the extended fallback path.
    try:
        browser_cmd = _find_agent_browser(validate=False)
    except FileNotFoundError:
        _print_warning(
            "    npx not found - browser tools require Node.js: https://nodejs.org"
        )
        return

    # Only the local browser provider actually needs Chromium on disk. Cloud
    # providers (Browserbase, Browser Use, Firecrawl) host their own Chromium.
    if post_setup_key != "agent_browser":
        return

    # Ensure the Chromium / headless-shell build agent-browser drives is
    # installed. Without it the CLI hangs on first use until the command
    # timeout fires. Skip inside Docker — the image bakes Chromium in at
    # build time, and runtime users usually can't write to
    # PLAYWRIGHT_BROWSERS_PATH anyway.
    if _chromium_installed():
        _print_success("    Chromium browser already installed, nothing to do")
        return

    if _running_in_docker():
        _print_warning(
            "    Chromium is missing but you're running in Docker."
        )
        _print_info(
            "    Pull the latest image to get the bundled Chromium:"
        )
        _print_info(
            "      docker pull ghcr.io/nousresearch/hermes-agent:latest"
        )
        return

    if _is_npx_agent_browser_sentinel(browser_cmd):
        # Re-resolve via the same PATH + extended-PATH cascade
        # _find_agent_browser used — a bare shutil.which("npx") would
        # silently diverge and hand subprocess.run a None argument.
        npx_bin = _resolve_npx_bin()
        if not npx_bin:
            _print_warning(
                "    npx not found - install Chromium manually: npx agent-browser install --with-deps"
            )
            return
        install_cmd = [npx_bin, "--ignore-scripts", "-y", AGENT_BROWSER_NPX_SPEC, "install", "--with-deps"]
    else:
        install_cmd = [browser_cmd, "install", "--with-deps"]

    _print_info("    Installing Chromium (~170MB one-time download)...")
    try:
        result = subprocess.run(
            install_cmd,
            capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(PROJECT_ROOT), timeout=600,
            creationflags=_post_setup_no_window_flags(),
        )
        if result.returncode == 0:
            _print_success("    Chromium installed")
            # Invalidate the cached "missing" result so subsequent
            # check_browser_requirements() calls see the new install.
            import tools.browser_tool as _bt
            _bt._cached_chromium_installed = None
        else:
            _print_warning("    Chromium install failed:")
            tail = (result.stderr or result.stdout or "").strip().splitlines()[-3:]
            for line in tail:
                _print_info(f"      {line[:200]}")
            _print_info("    Run manually: npx agent-browser install --with-deps")
    except subprocess.TimeoutExpired:
        _print_warning("    Chromium install timed out (>10min)")
        _print_info("    Run manually: npx agent-browser install --with-deps")
    except Exception as exc:
        _print_warning(f"    Chromium install failed: {exc}")
        _print_info("    Run manually: npx agent-browser install --with-deps")


def _post_setup_camofox() -> None:
    from hermes_constants import find_node_executable

    camofox_dir = PROJECT_ROOT / "node_modules" / "@askjo" / "camofox-browser"
    _npm_bin = find_node_executable("npm")
    if camofox_dir.exists():
        _print_success("    Camofox already installed, nothing to do")
    elif _npm_bin:
        _print_info("    Installing Camofox browser server...")
        # Absolute npm path so .cmd shim executes on Windows.
        result = subprocess.run(
            # --workspaces=false avoids resolving apps/desktop. See #38772.
            [_npm_bin, "install", "--silent", "--workspaces=false"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", cwd=str(PROJECT_ROOT),
            creationflags=_post_setup_no_window_flags(),
        )
        if result.returncode == 0:
            _print_success("    Camofox installed")
        else:
            _print_warning("    npm install failed - run manually: npm install --workspaces=false")
    if camofox_dir.exists():
        _print_info("    Start the Camofox server:")
        _print_info("      npx @askjo/camofox-browser")
        _print_info("    First run downloads the Camoufox engine (~300MB)")
        _print_info("    Or use Docker: docker run -p 9377:9377 -e CAMOFOX_PORT=9377 jo-inc/camofox-browser")
    elif not _npm_bin:
        _print_warning("    Node.js not found. Install Camofox via Docker:")
        _print_info("      docker run -p 9377:9377 -e CAMOFOX_PORT=9377 jo-inc/camofox-browser")


_KITTENTTS_WHEEL_URL = (
    "https://github.com/KittenML/KittenTTS/releases/download/"
    "0.8.1/kittentts-0.8.1-py3-none-any.whl"
)

# Post-setup hooks that only pip-install a Python package. Fields:
#   module        import probe (already installed → skip the install)
#   label         package name used in status lines
#   installing    progress line printed before the install
#   args          _pip_install arguments (keep in sync with
#                 _RESTORABLE_PYTHON_TOOL_DEPENDENCIES)
#   manual        the "Run manually:" command shown on failure/timeout
#   on_install    info lines printed only after a fresh successful install
#   always        info lines printed whenever the package ends up present
_PIP_POST_SETUP_HOOKS: dict = {
    "faster_whisper": {
        "module": "faster_whisper",
        "label": "faster-whisper",
        "installing": "Installing faster-whisper (model ~150MB downloads on first use)...",
        "args": ["-U", "faster-whisper", "--quiet"],
        "manual": "uv pip install -U faster-whisper",
        "on_install": (
            "Model sizes: tiny, base (default), small, medium, large-v3",
            "Change via stt.local.model in ~/.hermes/config.yaml",
        ),
        "always": (),
    },
    "kittentts": {
        "module": "kittentts",
        "label": "kittentts",
        "installing": "Installing kittentts (~25-80MB model, CPU-only)...",
        "args": ["-U", _KITTENTTS_WHEEL_URL, "soundfile", "--quiet"],
        "manual": f"uv pip install -U '{_KITTENTTS_WHEEL_URL}' soundfile",
        "on_install": (
            "Voices: Jasper, Bella, Luna, Bruno, Rosie, Hugo, Kiki, Leo",
            "Models: KittenML/kitten-tts-nano-0.8-int8 (25MB), micro (41MB), mini (80MB)",
        ),
        "always": (),
    },
    "piper": {
        "module": "piper",
        "label": "piper-tts",
        "installing": "Installing piper-tts (~14MB wheel, voices downloaded on first use)...",
        "args": ["-U", "piper-tts", "--quiet"],
        "manual": "uv pip install -U piper-tts",
        "on_install": (),
        "always": (
            "Default voice: en_US-lessac-medium (downloaded on first TTS call)",
            "Full voice list: https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/VOICES.md",
            "Switch voices by setting tts.piper.voice in ~/.hermes/config.yaml",
        ),
    },
    "ddgs": {
        "module": "ddgs",
        "label": "ddgs",
        "installing": "Installing ddgs (DuckDuckGo search package)...",
        "args": ["-U", "ddgs", "--quiet"],
        "manual": "uv pip install -U ddgs",
        "on_install": (),
        "always": (
            "No API key required. DuckDuckGo enforces server-side rate limits.",
            "Pair with an extract provider if you also need web_extract.",
        ),
    },
}


def _post_setup_pip(spec: dict) -> None:
    """Run one ``_PIP_POST_SETUP_HOOKS`` entry."""
    label = spec["label"]
    freshly_installed = False
    try:
        __import__(spec["module"])
        _print_success(f"    {label} is already installed")
    except ImportError:
        _print_info(f"    {spec['installing']}")
        try:
            result = _pip_install(spec["args"], timeout=300)
            if result.returncode == 0:
                _print_success(f"    {label} installed")
                freshly_installed = True
            else:
                _print_warning(f"    {label} install failed:")
                _print_info(f"      {(result.stderr or '').strip()[:300]}")
                _print_info(f"    Run manually: {spec['manual']}")
                return
        except subprocess.TimeoutExpired:
            _print_warning(f"    {label} install timed out (>5min)")
            _print_info(f"    Run manually: {spec['manual']}")
            return
    if freshly_installed:
        for line in spec["on_install"]:
            _print_info(f"    {line}")
    for line in spec["always"]:
        _print_info(f"    {line}")


def _post_setup_spotify() -> None:
    # Run the full `hermes auth spotify` flow — if the user has no
    # client_id yet, this drops them into the interactive wizard
    # (opens the Spotify dashboard, prompts for client_id, persists
    # to ~/.hermes/.env), then continues straight into PKCE. If they
    # already have an app, it skips the wizard and just does OAuth.
    from types import SimpleNamespace
    try:
        from hermes_cli.auth import login_spotify_command
    except Exception as exc:
        _print_warning(f"    Could not load Spotify auth: {exc}")
        _print_info("    Run manually: hermes auth spotify")
        return
    _print_info("    Starting Spotify login...")
    try:
        login_spotify_command(SimpleNamespace(
            client_id=None, redirect_uri=None, scope=None,
            no_browser=False, timeout=None,
        ))
        _print_success("    Spotify authenticated")
    except SystemExit as exc:
        # User aborted the wizard, or OAuth failed — don't fail the
        # toolset enable; they can retry with `hermes auth spotify`.
        _print_warning(f"    Spotify login did not complete: {exc}")
        _print_info("    Run later: hermes auth spotify")
    except Exception as exc:
        _print_warning(f"    Spotify login failed: {exc}")
        _print_info("    Run manually: hermes auth spotify")


def _post_setup_langfuse() -> None:
    # Install the langfuse SDK.
    try:
        __import__("langfuse")
        _print_success("    langfuse SDK already installed")
    except ImportError:
        _print_info("    Installing langfuse SDK...")
        result = _pip_install(["langfuse", "--quiet"], timeout=120)
        if result.returncode == 0:
            _print_success("    langfuse SDK installed")
        else:
            _print_warning("    langfuse SDK install failed — run manually: uv pip install langfuse")
    # Opt the bundled observability/langfuse plugin into plugins.enabled.
    # The plugin ships in the repo but doesn't load until the user enables
    # it (standalone plugins are opt-in).
    try:
        from hermes_cli.plugins_cmd import _get_enabled_set, _save_enabled_set
        enabled = _get_enabled_set()
        if "observability/langfuse" in enabled or "langfuse" in enabled:
            _print_success("    Plugin observability/langfuse already enabled")
        else:
            enabled.add("observability/langfuse")
            _save_enabled_set(enabled)
            _print_success("    Plugin observability/langfuse enabled")
    except Exception as exc:
        _print_warning(f"    Could not enable plugin automatically: {exc}")
        _print_info("    Run manually: hermes plugins enable observability/langfuse")
    _print_info("    Restart Hermes for tracing to take effect.")
    _print_info("    Verify: hermes plugins list")


def _post_setup_xai_grok() -> None:
    """Shared xAI credential bootstrap for any picker row that talks to xAI (TTS, STT, Video Gen,
    x_search …). Accepts a SuperGrok-tier OAuth token (preferred — billed to the existing
    subscription) or a raw XAI_API_KEY; the rows declare empty env_vars so the auth UX lives here.
    """
    try:
        from hermes_cli.auth import get_xai_oauth_auth_status
        oauth_logged_in = bool(get_xai_oauth_auth_status().get("logged_in"))
    except Exception:
        oauth_logged_in = False
    existing_api_key = get_env_value("XAI_API_KEY")

    if oauth_logged_in:
        _print_success(
            "    xAI will use your xAI Grok OAuth (SuperGrok / Premium+) credentials"
        )
        return
    if existing_api_key:
        _print_success("    xAI will use your existing XAI_API_KEY")
        return

    _print_info("    xAI needs credentials. Choose one:")
    try:
        from hermes_cli.setup import (
            _run_xai_oauth_login_from_setup,
            prompt_choice,
            prompt as _setup_prompt,
        )
        from hermes_cli.config import save_env_value
    except Exception as exc:
        _print_warning(f"    Could not load setup helpers: {exc}")
        _print_info("    Run later: hermes auth add xai-oauth   (or set XAI_API_KEY)")
        return

    idx = prompt_choice(
        "    How do you want xAI to authenticate?",
        choices=[
            "Sign in with xAI Grok OAuth (SuperGrok / Premium+) — browser login",
            "Paste an xAI API key (console.x.ai)",
            "Skip — configure later via `hermes auth add xai-oauth`",
        ],
        default=0,
    )
    if idx == 0:
        if _run_xai_oauth_login_from_setup():
            _print_success(
                "    Logged in — xAI will use these OAuth credentials"
            )
        else:
            _print_warning(
                "    xAI Grok OAuth login did not complete. "
                "Run later: hermes auth add xai-oauth"
            )
    elif idx == 1:
        api_key = _setup_prompt("    xAI API key", password=True)
        if api_key:
            save_env_value("XAI_API_KEY", api_key)
            _print_success("    XAI_API_KEY saved")
        else:
            _print_warning(
                "    No API key provided. Run later: hermes auth add xai-oauth"
            )
    else:
        _print_info("    xAI will remain inactive until credentials are configured.")


# post_setup key -> hook. Unknown keys are a silent no-op (callers validate
# against valid_post_setup_keys()).
_POST_SETUP_HOOKS: dict = {
    "lightpanda": _post_setup_lightpanda,
    "agent_browser": lambda: _post_setup_agent_browser("agent_browser"),
    "browserbase": lambda: _post_setup_agent_browser("browserbase"),
    "browser_use_cli": lambda: _ensure_browser_use_cli(verbose_hints=True),
    "camofox": _post_setup_camofox,
    "cua_driver": lambda: install_cua_driver(upgrade=False),
    "spotify": _post_setup_spotify,
    "langfuse": _post_setup_langfuse,
    "xai_grok": _post_setup_xai_grok,
    **{key: (lambda spec=spec: _post_setup_pip(spec)) for key, spec in _PIP_POST_SETUP_HOOKS.items()},
}


def _run_post_setup(post_setup_key: str):
    """Run post-setup hooks for tools that need extra installation steps."""
    hook = _POST_SETUP_HOOKS.get(post_setup_key)
    if hook is not None:
        hook()


def valid_post_setup_keys() -> Set[str]:
    """Return the set of post-setup keys declared by any visible provider.

    Collected from ``TOOL_CATEGORIES`` plus plugin-registered web/image/video/browser providers.
    This is the allowlist the ``post-setup`` command and dashboard endpoint validate against, so a
    caller cannot drive ``_run_post_setup`` with an arbitrary key.
    """
    keys: Set[str] = set()
    for cat in TOOL_CATEGORIES.values():
        for prov in cat.get("providers", []):
            ps = prov.get("post_setup")
            if ps:
                keys.add(ps)
    # Plugin-registered providers can declare their own post_setup hooks.
    for builder in (
        _plugin_web_search_providers,
        _plugin_image_gen_providers,
        _plugin_video_gen_providers,
        _plugin_browser_providers,
    ):
        try:
            for prov in builder():
                ps = prov.get("post_setup")
                if ps:
                    keys.add(ps)
        except Exception:  # pragma: no cover — defensive; plugins optional
            continue
    return keys


def run_post_setup_command(args) -> int:
    """``hermes tools post-setup <key>`` — non-interactive post-setup runner.

    Stable, scriptable target the dashboard spawns so the GUI can drive backend setup without
    re-implementing install logic. Returns a process exit code (0 ok, 2 unknown key).
    """
    key = getattr(args, "post_setup_key", None)
    if not key:
        _print_error("Usage: hermes tools post-setup <key>")
        return 2
    valid = valid_post_setup_keys()
    if key not in valid:
        _print_error(
            f"Unknown post-setup key: {key!r}. "
            f"Valid keys: {', '.join(sorted(valid)) or '(none)'}"
        )
        return 2
    _print_info(f"Running post-setup hook: {key}")
    try:
        _run_post_setup(key)
    except Exception as exc:  # pragma: no cover — defensive
        _print_error(f"Post-setup failed: {exc}")
        return 1
    _print_success(f"Post-setup '{key}' complete")
    return 0


# ─── Platform / Toolset Helpers ───────────────────────────────────────────────

_PLATFORM_ENABLE_ENV_VARS = (
    ("telegram", "TELEGRAM_BOT_TOKEN"),
    ("discord", "DISCORD_BOT_TOKEN"),
    ("slack", "SLACK_BOT_TOKEN"),
    ("whatsapp", "WHATSAPP_ENABLED"),
    ("qqbot", "QQ_APP_ID"),
)


def _get_enabled_platforms() -> List[str]:
    """Return platform keys that are configured (have tokens or are CLI)."""
    return ["cli"] + [
        platform
        for platform, env_var in _PLATFORM_ENABLE_ENV_VARS
        if get_env_value(env_var)
    ]


def _platform_toolset_summary(config: dict, platforms: Optional[List[str]] = None) -> Dict[str, Set[str]]:
    """Return a summary of enabled toolsets per platform.

    When ``platforms`` is None, this uses ``_get_enabled_platforms`` to auto-detect platforms. Tests
    can pass an explicit list to avoid relying on environment variables.
    """
    if platforms is None:
        platforms = _get_enabled_platforms()

    summary: Dict[str, Set[str]] = {}
    for pkey in platforms:
        summary[pkey] = _get_platform_tools(config, pkey)
    return summary


def _parse_enabled_flag(value, default: bool = True) -> bool:
    """Parse bool-like config values used by tool/platform settings."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value != 0
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "1", "yes", "on"}:
            return True
        if lowered in {"false", "0", "no", "off"}:
            return False
    return default


def enabled_mcp_server_names(config: dict) -> Set[str]:
    """Names of MCP servers globally enabled in config.yaml or by a plugin.

    Shared by the platform resolver and the cron toolset resolver so every path agrees on MCP
    membership. A server is enabled unless ``enabled`` is explicitly falsey; missing/unknown values
    count as enabled. Portable-plugin servers (in-memory, not in config.yaml) are included so their
    tools reach the model's schema — enabling the plugin is the user's opt-in.
    """
    mcp_servers = (config or {}).get("mcp_servers") or {}
    names = {
        str(name)
        for name, server_cfg in mcp_servers.items()
        if isinstance(server_cfg, dict)
        and _parse_enabled_flag(server_cfg.get("enabled", True), default=True)
    }
    try:
        from hermes_cli.plugins import (
            get_portable_mcp_server_names_nowait,
        )

        portable = get_portable_mcp_server_names_nowait()
        # Native config wins on a name collision (mirrors _load_mcp_config).
        names |= portable - set(mcp_servers)
    except Exception:
        logger.debug("Failed to include portable MCP servers", exc_info=True)
    return names


def _exempt_explicit_platform_native(
    default_off: Set[str], platform: str, *, explicitly_configured: bool
) -> None:
    """Let platform-native default-off toolsets through on explicit config.

    Default-off toolsets restricted to a platform (e.g. ``discord`` on discord) are that platform's
    native tools: kept off for unconfigured platforms as a security opt-in, but once the user
    explicitly saves a toolset list, stripping them silently would defeat that configuration.
    Mutates ``default_off`` in place.
    """
    if not explicitly_configured:
        return
    for ts in list(default_off):
        allowed = _TOOLSET_PLATFORM_RESTRICTIONS.get(ts)
        if allowed is not None and platform in allowed:
            default_off.discard(ts)


#: Toolsets young enough that absence from a saved ``platform_toolsets`` list
#: means "never offered" rather than "declined".
#:
#: Saving ``hermes tools`` (or one toggle in the desktop Toolsets UI) replaces
#: a platform's composite with a frozen explicit list, and nothing ever adds to
#: that list — so a toolset shipped afterwards stays off forever for anyone who
#: has touched the picker, while everyone still on ``[hermes-cli]`` inherits it
#: on upgrade. Listing it here restores that parity.
#:
#: MUST ship in the same release as the toolset it names, and be emptied in the
#: next one. The inference only holds while no released build has put the
#: toolset on a checklist: once one has, a user who unchecks it writes a config
#: byte-identical to one saved before the toolset existed (the record below is
#: only written from that point on), and this rule turns their opt-out back on.
#: Landing late — or leaving an entry here for a second release — converts a
#: back-fill into a stuck checkbox.
#:
#: A ``check_fn``-gated toolset costs nothing here for users who cannot call
#: it: an enabled toolset still ships zero schemas when its check fails — the
#: same split Home Assistant uses. Probing a remote service from this path
#: would put a network call on every CLI start, gateway session and cron tick.
_RECENTLY_SHIPPED_TOOLSETS: frozenset = frozenset()


def _enable_recently_shipped_toolsets(
    enabled_toolsets: Set[str], config: dict, platform: str
) -> None:
    """Turn on toolsets that shipped after this platform's saved list.

    Either way of saying no outlives this: unchecking in ``hermes tools`` records the toolset in
    ``known_builtin_toolsets`` so it reads as declined from then on, and ``agent.disabled_toolsets``
    is subtracted after every rule in :func:`_get_platform_tools`. Mutates ``enabled_toolsets`` in
    place.
    """
    from toolsets import resolve_toolset

    offered = (config.get("known_builtin_toolsets") or {}).get(platform)
    declined = {str(ts) for ts in offered} if isinstance(offered, list) else set()

    default_ts = _platform_default_toolset(platform)
    composite_tools = None

    for ts_key in sorted(_RECENTLY_SHIPPED_TOOLSETS):
        if ts_key in enabled_toolsets or ts_key in declined:
            continue
        if not _toolset_allowed_for_platform(ts_key, platform):
            continue
        # Parity is the whole justification, so only enable the toolset where
        # staying on the composite would have enabled it anyway. Deliberately
        # narrow composites (hermes-acp, hermes-webhook) stay narrow.
        ts_tools = set(resolve_toolset(ts_key, include_registry=False))
        if composite_tools is None:
            composite_tools = set(resolve_toolset(default_ts))
        if not ts_tools or not ts_tools.issubset(composite_tools):
            continue
        enabled_toolsets.add(ts_key)


def _configurable_subset_of(tool_names: Set[str], platform: str) -> Set[str]:
    """Configurable toolsets whose STATIC membership is contained in ``tool_names``.

    Compares ``resolve_toolset(ts, include_registry=False)``: a tool registered into a toolset at
    runtime (e.g. delegate_cli -> delegation, desktop-only read_terminal -> terminal) that the
    composite never listed must not drop the whole toolset (issue #49622).
    """
    from toolsets import resolve_toolset

    enabled = set()
    for ts_key, _, _ in CONFIGURABLE_TOOLSETS:
        if not _toolset_allowed_for_platform(ts_key, platform):
            continue
        ts_tools = set(resolve_toolset(ts_key, include_registry=False))
        if ts_tools and ts_tools.issubset(tool_names):
            enabled.add(ts_key)
    return enabled


def _default_off_toolsets(platform: str, explicitly_configured: bool) -> Set[str]:
    """Toolsets to strip from an implicit (composite-derived) enable set for ``platform``.

    Legacy safety: a platform whose own name matches a default-off toolset (``homeassistant``)
    keeps that toolset on first install — except platform-restricted toolsets, which stay opt-in
    even on their own platform (``discord`` + ``discord`` stays OFF). Home Assistant is runtime-
    gated by its check_fn (needs HASS_TOKEN), so a configured token is an explicit opt-in and must
    not be stripped here — otherwise HA silently vanished from platforms like cron that resolve
    without a saved toolset list (regression after #14798).
    """
    default_off = set(_DEFAULT_OFF_TOOLSETS)
    if platform in default_off and platform not in _TOOLSET_PLATFORM_RESTRICTIONS:
        default_off.remove(platform)
    if "homeassistant" in default_off and _homeassistant_credentials_present():
        default_off.remove("homeassistant")
    _exempt_explicit_platform_native(
        default_off, platform, explicitly_configured=explicitly_configured
    )
    return default_off


def _get_platform_tools(
    config: dict,
    platform: str,
    *,
    include_default_mcp_servers: bool = True,
) -> Set[str]:
    """Resolve which individual toolset names are enabled for a platform."""
    from toolsets import resolve_toolset, TOOLSETS

    platform_toolsets = config.get("platform_toolsets") or {}
    toolset_names = platform_toolsets.get(platform)
    # Track whether the user explicitly saved a toolset list for this platform
    # (vs. falling back to the platform default). An explicit composite (e.g.
    # ``hermes-discord``) is an opt-in to the platform's native default-off
    # toolsets — see _exempt_explicit_platform_native (#35527).
    explicitly_configured = isinstance(toolset_names, list)

    if not explicitly_configured:
        toolset_names = [_platform_default_toolset(platform)]

    # YAML may parse bare numeric names (e.g. ``12306:``) as int.
    # Normalise to str so downstream sorted() never mixes types.
    toolset_names = [str(ts) for ts in toolset_names]

    configurable_keys = {ts_key for ts_key, _, _ in CONFIGURABLE_TOOLSETS}
    plugin_ts_keys = _get_plugin_toolset_keys()
    platform_default_keys = {p["default_toolset"] for p in PLATFORMS.values()}
    # Plugin-provided toolsets are first-class on a platform-toolsets list —
    # explicit config like ``[hermes-cli, a2a]`` must survive filtering just
    # like a built-in configurable toolset would. See issue #81163.
    explicit_known_keys = configurable_keys | plugin_ts_keys

    # If the saved list contains any configurable keys directly, the user
    # has explicitly configured this platform — use direct membership.
    # This avoids the subset-inference bug where composite toolsets like
    # "hermes-cli" (which include all _HERMES_CORE_TOOLS) cause disabled
    # toolsets to re-appear as enabled.
    has_explicit_config = any(ts in explicit_known_keys for ts in toolset_names)

    if has_explicit_config:
        enabled_toolsets = {
            ts for ts in toolset_names
            if ts in explicit_known_keys and _toolset_allowed_for_platform(ts, platform)
        }
        # Mixed config: composite toolset alongside configurables (e.g.
        # ``[hermes-cli, spotify]`` after enabling Spotify via ``hermes
        # tools``). Without expansion the composite name is silently dropped,
        # leaving sessions with only the configurable opt-ins and no native
        # tools. Mirror the else-branch's subset inference, but apply
        # _DEFAULT_OFF_TOOLSETS only to the implicit expansion — anything the
        # user explicitly listed (e.g. ``spotify``) must survive.
        composite_tools = set()
        for ts_name in toolset_names:
            if ts_name not in explicit_known_keys and ts_name in TOOLSETS:
                composite_tools.update(resolve_toolset(ts_name))

        if composite_tools:
            enabled_toolsets |= _configurable_subset_of(composite_tools, platform) - _default_off_toolsets(
                platform, explicitly_configured
            )

        _enable_recently_shipped_toolsets(enabled_toolsets, config, platform)
    else:
        # No explicit config — fall back to resolving composite toolset names
        # (e.g. "hermes-cli") to individual tool names and reverse-mapping.
        all_tool_names = set()
        for ts_name in toolset_names:
            all_tool_names.update(resolve_toolset(ts_name))
        enabled_toolsets = _configurable_subset_of(all_tool_names, platform)

        # Auto-enable ``x_search`` when xAI credentials are configured.
        # Unlike ``homeassistant`` (whose ``ha_*`` tools live inside the
        # platform composite and thus pass the subset check above),
        # ``x_search`` is its own one-tool toolset that the composite does
        # NOT include, so the subset loop never picks it up. Inject it
        # directly here, mirroring the HASS_TOKEN → ``homeassistant`` rule
        # below: once you have working creds, you don't have to also click
        # through ``hermes tools`` to flip the toolset on. Only fires when
        # the user has not yet saved an explicit toolset list — once they
        # do, the saved list is authoritative.
        x_search_auto_enabled = (
            _toolset_allowed_for_platform("x_search", platform)
            and _xai_credentials_present()
        )
        if x_search_auto_enabled:
            enabled_toolsets.add("x_search")

        default_off = _default_off_toolsets(platform, explicitly_configured)
        # Symmetric carve-out for x_search auto-enable (see the inject
        # block above). Without this, the default_off subtraction would
        # strip the entry we just added.
        if x_search_auto_enabled:
            default_off.discard("x_search")
        enabled_toolsets -= default_off

    _recover_platform_native_toolsets(enabled_toolsets, platform, skip=configurable_keys | plugin_ts_keys | platform_default_keys)

    # Plugin toolsets: enabled by default unless explicitly disabled, or
    # unless the toolset is in _DEFAULT_OFF_TOOLSETS (e.g. spotify —
    # shipped as a bundled plugin but user must opt in via `hermes tools`
    # so we don't ship 7 Spotify tool schemas to users who don't use it).
    # A plugin toolset is "known" for a platform once `hermes tools`
    # has been saved for that platform (tracked via known_plugin_toolsets).
    # Unknown plugins default to enabled; known-but-absent = disabled.
    if plugin_ts_keys:
        known_map = config.get("known_plugin_toolsets", {}) or {}
        known_for_platform = set(known_map.get(platform, []) or [])
        for pts in plugin_ts_keys:
            if pts in toolset_names or (
                pts not in _DEFAULT_OFF_TOOLSETS and pts not in known_for_platform
            ):
                enabled_toolsets.add(pts)

    # Context-engine tools are runtime-provided by the active engine, so they
    # are not part of any static platform composite. When a non-default engine
    # is selected, keep its recovery/status tools available even after a user
    # saves an explicit platform toolset list. Preserve the explicit empty-list
    # contract: selecting no configurable tools means no context-engine tools
    # either unless the user adds ``context_engine`` manually later.
    context_cfg = config.get("context") or {}
    if not isinstance(context_cfg, dict):
        context_cfg = {}
    context_engine_name = str(context_cfg.get("engine") or "compressor").strip().lower()
    if context_engine_name and context_engine_name != "compressor" and not (
        explicitly_configured and not toolset_names
    ):
        enabled_toolsets.add("context_engine")

    # Preserve any explicit non-configurable toolset entries (for example,
    # custom toolsets or MCP server names saved in platform_toolsets).
    explicit_passthrough = {
        ts for ts in toolset_names
        if ts not in explicit_known_keys and ts not in platform_default_keys
    }
    enabled_toolsets |= _merge_mcp_servers(
        config, toolset_names, explicit_passthrough, include_default_mcp_servers
    )

    # Honor agent.disabled_toolsets from config.yaml — allows users to
    # globally suppress specific toolsets (e.g. "memory") across all
    # platforms without per-platform toolset configuration.  This runs
    # last so it overrides everything above.  The value may arrive as a
    # JSON-array string (e.g. "['memory']") from `hermes config set` or a
    # JSON-mode editor save; parse it so the list is not silently dead (#86661).
    agent_cfg = config.get("agent") or {}
    disabled_toolsets = agent_cfg.get("disabled_toolsets") or []
    if disabled_toolsets:
        from agent.skill_utils import parse_config_string_list

        enabled_toolsets -= {
            name.strip() for name in parse_config_string_list(disabled_toolsets) if name.strip()
        }

    if explicitly_configured and toolset_names:
        _warn_all_invalid_platform_toolsets(platform, platform_toolsets[platform])

    return enabled_toolsets


def _recover_platform_native_toolsets(enabled_toolsets: Set[str], platform: str, *, skip: Set[str]) -> None:
    """Add non-configurable platform toolsets (e.g. discord, feishu_doc, feishu_drive) in place.

    These are part of the platform's default composite but absent from CONFIGURABLE_TOOLSETS, so
    they can't appear in the TUI checklist or in a user-saved config. Must run for BOTH the explicit
    and the composite branch of ``_get_platform_tools`` — otherwise saving via ``hermes tools``
    silently drops them.
    """
    from toolsets import resolve_toolset, TOOLSETS

    platform_tool_universe = set(resolve_toolset(_platform_default_toolset(platform)))
    configurable_tool_universe = set()
    for ts_key, _, _ in CONFIGURABLE_TOOLSETS:
        configurable_tool_universe.update(resolve_toolset(ts_key))
    claimed = set()
    for ts_key in enabled_toolsets:
        claimed.update(resolve_toolset(ts_key))
    skip = skip | {k for k in TOOLSETS if k.startswith("hermes-")}
    skip |= set(_DEFAULT_OFF_TOOLSETS) - {platform}
    for ts_key, ts_def in TOOLSETS.items():
        # Posture toolsets (e.g. ``coding``) are session-level selections made
        # by agent/coding_context.py — not per-platform capabilities to recover.
        if ts_key in skip or ts_def.get("includes") or ts_def.get("posture"):
            continue
        # Static membership (see #49622): a registry-added tool absent from the
        # platform composite must not block recovery of a non-configurable
        # toolset whose authored tools the composite does list.
        ts_tools = set(resolve_toolset(ts_key, include_registry=False))
        if not ts_tools or not ts_tools.issubset(platform_tool_universe):
            continue
        if ts_tools.issubset(configurable_tool_universe):
            continue
        if not ts_tools.issubset(claimed):
            enabled_toolsets.add(ts_key)
            claimed.update(ts_tools)


def _merge_mcp_servers(
    config: dict, toolset_names: List[str], explicit_passthrough: Set[str], include_default_mcp_servers: bool
) -> Set[str]:
    """Explicit passthrough entries plus the MCP servers enabled for this platform.

    MCP servers are available on all platforms by default. If the platform explicitly lists one or
    more MCP server names, that is an allowlist; otherwise every globally enabled server is included
    (when ``include_default_mcp_servers``). The ``no_mcp`` sentinel disables all MCP servers.
    """
    enabled_mcp_servers = enabled_mcp_server_names(config)
    no_mcp = "no_mcp" in toolset_names
    if no_mcp:
        explicit_mcp_servers = set()
        result = explicit_passthrough - enabled_mcp_servers - {"no_mcp"}
    else:
        explicit_mcp_servers = explicit_passthrough & enabled_mcp_servers
        result = explicit_passthrough - enabled_mcp_servers
    if include_default_mcp_servers and not explicit_mcp_servers and not no_mcp:
        return result | enabled_mcp_servers
    return result | explicit_mcp_servers


def _warn_all_invalid_platform_toolsets(platform: str, explicit: list) -> None:
    """#38798: warn once when an explicitly configured platform has only invalid toolset names.

    A migration or hand-edit that left ``hermes`` instead of ``hermes-cli`` makes resolve_toolset()
    return [] for every entry and the platform silently ends up with no native tools. Surface it
    where tools are resolved for a session, not only during ``hermes update``/``hermes doctor``.
    """
    from toolsets import validate_toolset

    named = [str(t) for t in explicit if isinstance(t, str) and t]
    if (
        named
        and not any(validate_toolset(t) for t in named)
        and platform not in _warned_invalid_platform_toolsets
    ):
        _warned_invalid_platform_toolsets.add(platform)
        logger.warning(
            "platform '%s' has no valid toolsets configured (unknown "
            "name(s): %s) - tools will be unavailable. Run `hermes tools` "
            "to reconfigure. See issue #38798.",
            platform,
            ", ".join(named),
        )


def _save_platform_tools(config: dict, platform: str, enabled_toolset_keys: Set[str]):
    """Save the selected toolset keys for a platform to config."""
    config.setdefault("platform_toolsets", {})

    # Drop platform-scoped toolsets that don't apply here.  Prevents the
    # "Configure all platforms" checklist (or a hand-edited config.yaml)
    # from turning on, say, the `discord` toolset for Telegram.
    enabled_toolset_keys = {
        ts for ts in enabled_toolset_keys
        if _toolset_allowed_for_platform(ts, platform)
    }

    # Get the set of all configurable toolset keys (built-in + plugin)
    configurable_keys = {ts_key for ts_key, _, _ in CONFIGURABLE_TOOLSETS}
    plugin_keys = _get_plugin_toolset_keys()
    configurable_keys |= plugin_keys

    # Also exclude platform default toolsets (hermes-cli, hermes-telegram, etc.)
    # These are "super" toolsets that resolve to ALL tools, so preserving them
    # would silently override the user's unchecked selections on the next read.
    platform_default_keys = {p["default_toolset"] for p in PLATFORMS.values()}

    # Get existing toolsets for this platform
    existing_toolsets = cfg_get(config, "platform_toolsets", platform, default=[])
    if not isinstance(existing_toolsets, list):
        existing_toolsets = []
    existing_toolsets = [str(ts) for ts in existing_toolsets]

    # Preserve any entries that are NOT configurable toolsets and NOT platform
    # defaults (i.e. only MCP server names should be preserved)
    preserved_entries = {
        entry for entry in existing_toolsets
        if entry not in configurable_keys and entry not in platform_default_keys
    }
    # Opening `hermes tools` is the user's opt-in to reconfigure tools, so treat
    # saving from the picker as consent to clear the "no_mcp" sentinel. The
    # picker has no checkbox for no_mcp, so without this users who once set it
    # by hand could never re-enable MCP servers through the UI.
    preserved_entries.discard("no_mcp")

    # Merge preserved entries with new enabled toolsets
    config["platform_toolsets"][platform] = sorted(enabled_toolset_keys | preserved_entries)

    # Track which plugin toolsets are "known" for this platform so we can
    # distinguish "new plugin, default enabled" from "user disabled it".
    # _cfg_section normalizes a present-but-null key ("known_plugin_toolsets:"
    # in config.yaml parses to None) that setdefault alone would not replace.
    if plugin_keys:
        _cfg_section(config, "known_plugin_toolsets")[platform] = sorted(plugin_keys)

    # Same record for builtin toolsets: which ones this platform's checklist
    # has actually put in front of the user. Without it, a toolset the user
    # unchecks here is indistinguishable from one that shipped after they
    # saved, and _enable_recently_shipped_toolsets would turn it straight back
    # on. Recorded from the full catalog, since that is what the picker showed.
    _cfg_section(config, "known_builtin_toolsets")[platform] = sorted(
        ts_key for ts_key, _, _ in CONFIGURABLE_TOOLSETS
    )

    # Reconcile with agent.disabled_toolsets. _get_platform_tools() applies
    # that list as a final override AFTER reading platform_toolsets.<platform>,
    # so a toolset listed there stays permanently OFF no matter what this
    # function writes — the toggle "saves" but silently can't ever take
    # effect. Blank Slate installs pre-populate this list with ~27 toolsets,
    # making most of the desktop Toolsets UI unusable for re-enabling
    # anything (issue #49995).
    #
    # Only toolsets the user just explicitly enabled FOR THIS PLATFORM are
    # cleared from the global disabled list — toolsets the user did not
    # touch (still unchecked) or that remain disabled on other platforms
    # are left alone, so agent.disabled_toolsets keeps working as a
    # cross-platform suppression list for anything not actively re-enabled.
    agent_cfg = config.get("agent")
    newly_enabled = enabled_toolset_keys - preserved_entries
    if isinstance(agent_cfg, dict) and agent_cfg.get("disabled_toolsets") and newly_enabled:
        from agent.skill_utils import parse_config_string_list

        parsed_disabled = parse_config_string_list(agent_cfg["disabled_toolsets"])
        remaining = [ts for ts in parsed_disabled if ts not in newly_enabled]
        if remaining != parsed_disabled:
            agent_cfg["disabled_toolsets"] = remaining

    save_config(config)


def _provider_env_ready(provider: dict) -> bool:
    """True when every env var a provider row declares is set (trivially true for no-key rows)."""
    return all(get_env_value(e["key"]) for e in provider.get("env_vars", []))


def _toolset_has_keys(
    ts_key: str,
    config: dict = None,
    *,
    force_fresh: bool = False,
    features: Optional[NousSubscriptionFeatures] = None,
) -> bool:
    """Check if a toolset's required API keys are configured."""
    if config is None:
        config = load_config()

    if ts_key == "vision":
        try:
            from agent.auxiliary_client import resolve_vision_provider_client

            _provider, client, _model = resolve_vision_provider_client()
            return client is not None
        except Exception:
            return False

    if ts_key in {"web", "image_gen", "video_gen", "tts", "stt", "browser"}:
        if features is None:
            features = get_nous_subscription_features(
                config, force_fresh=force_fresh
            )
        feature = features.features.get(ts_key)
        if feature and (feature.available or feature.managed_by_nous):
            return True

    # Check TOOL_CATEGORIES first (provider-aware). A no-key provider
    # (Local Browser, Edge TTS) counts as configured.
    cat = TOOL_CATEGORIES.get(ts_key)
    if cat:
        return any(
            _provider_env_ready(provider)
            for provider in _visible_providers(cat, config, force_fresh=force_fresh, features=features)
        )

    # Fallback to simple requirements
    requirements = TOOLSET_ENV_REQUIREMENTS.get(ts_key, [])
    if not requirements:
        return True
    return all(get_env_value(var) for var, _ in requirements)


# ─── Menu Helpers ─────────────────────────────────────────────────────────────

def _prompt_choice(question: str, choices: list, default: int = 0) -> int:
    """Single-select menu (arrow keys). Delegates to curses_radiolist."""
    from hermes_cli.curses_ui import curses_radiolist
    return curses_radiolist(question, choices, selected=default, cancel_returns=default)


# ─── Token Estimation ────────────────────────────────────────────────────────

# Profile-keyed cache so one process can serve distinct plugin tool catalogs.
_tool_token_cache: Optional[Dict[tuple[str, int], Dict[str, int]]] = None


def _estimate_tool_tokens() -> Dict[str, int]:
    """Return estimated token counts per individual tool name.

    Counts tiktoken (cl100k_base) tokens of the JSON-serialised OpenAI tool schema. Triggers tool
    discovery on first call and caches for the process. Empty dict if tiktoken/registry unavailable.
    """
    global _tool_token_cache
    from hermes_constants import hermes_home_key

    scope = hermes_home_key()
    _tool_token_cache = _tool_token_cache or {}

    try:
        # Trigger full tool discovery (imports all tool modules).
        import model_tools  # noqa: F401
        from tools.registry import registry
        cache_key = (scope, registry._generation)
    except Exception:
        logger.debug("Tool registry unavailable; skipping token estimation")
        return _tool_token_cache.setdefault((scope, -1), {})

    if cache_key in _tool_token_cache:
        return _tool_token_cache[cache_key]

    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
    except Exception:
        logger.debug("tiktoken unavailable; skipping tool token estimation")
        return _tool_token_cache.setdefault(cache_key, {})

    counts: Dict[str, int] = {}
    for name in registry.get_all_tool_names():
        schema = registry.get_schema(name)
        if schema:
            # Mirror what gets sent to the API:
            # {"type": "function", "function": <schema>}
            text = _json.dumps({"type": "function", "function": schema})
            counts[name] = len(enc.encode(text))
    _tool_token_cache[cache_key] = counts
    return counts


def _prompt_toolset_checklist(
    platform_label: str,
    enabled: Set[str],
    platform: str = "cli",
    *,
    force_fresh: bool = True,
) -> Set[str]:
    """Multi-select checklist of toolsets. Returns set of selected toolset keys."""
    from hermes_cli.curses_ui import curses_checklist
    from toolsets import resolve_toolset

    # Pre-compute per-tool token counts (cached after first call).
    tool_tokens = _estimate_tool_tokens()

    effective_all = _get_effective_configurable_toolsets()
    # Drop platform-scoped toolsets that don't apply to this platform, and
    # config-only capabilities (stt) that have no per-platform toggle.
    effective = [
        (k, l, d) for (k, l, d) in effective_all
        if _toolset_allowed_for_platform(k, platform)
        and k not in _CONFIG_ONLY_TOOLSETS
    ]

    labels = []
    for ts_key, ts_label, ts_desc in effective:
        suffix = ""
        if (
            not _toolset_has_keys(ts_key, force_fresh=force_fresh)
            and _is_configurable(ts_key)
        ):
            suffix = "  [no API key]"
        labels.append(f"{ts_label}  ({ts_desc}){suffix}")

    pre_selected = {
        i for i, (ts_key, _, _) in enumerate(effective)
        if ts_key in enabled
    }

    # Build a live status function that shows deduplicated total token cost.
    status_fn = None
    if tool_tokens:
        ts_keys = [ts_key for ts_key, _, _ in effective]

        def status_fn(chosen: set) -> str:
            # Collect unique tool names across all selected toolsets
            all_tools: set = set()
            for idx in chosen:
                all_tools.update(resolve_toolset(ts_keys[idx]))
            total = sum(tool_tokens.get(name, 0) for name in all_tools)
            if total >= 1000:
                return f"Est. tool context: ~{total / 1000:.1f}k tokens"
            return f"Est. tool context: ~{total} tokens"

    chosen = curses_checklist(
        f"Tools for {platform_label}",
        labels,
        pre_selected,
        cancel_returns=pre_selected,
        status_fn=status_fn,
    )
    return {effective[i][0] for i in chosen}


# ─── Provider-Aware Configuration ────────────────────────────────────────────

def _configure_toolset(
    ts_key: str,
    config: dict,
    *,
    force_fresh: bool = True,
):
    """Configure a toolset - provider selection + API keys.

    Uses TOOL_CATEGORIES for provider-aware config, falls back to simple env var prompts for
    toolsets not in TOOL_CATEGORIES.
    """
    cat = TOOL_CATEGORIES.get(ts_key)

    if cat:
        _configure_tool_category(ts_key, cat, config, force_fresh=force_fresh)
    else:
        # Simple fallback for vision, moa, etc.
        _configure_simple_requirements(ts_key)




_POST_SETUP_INSTALLED: dict = {
    # post_setup_key -> predicate(): True when the install side-effect
    # is already satisfied. Used by `_toolset_needs_configuration_prompt`
    # to force the provider-setup flow when a no-key provider still needs
    # a binary/dependency install (otherwise an already-configured user
    # who toggles the toolset on via `hermes tools` gets a silent no-op
    # because the gate sees "no env vars to ask about" and skips the
    # provider-setup flow that would have run the post_setup hook).
    #
    # Only entries here are gated; other post_setup hooks (kittentts,
    # piper, agent_browser, etc.) keep their existing behaviour. Add an
    # entry when (a) the post_setup is the ONLY install side-effect for
    # a no-key provider, and (b) an installed-state check is local, bounded,
    # and doesn't trigger a heavy import.
    "cua_driver": lambda: _cua_driver_install_ready(),
}


def _post_setup_already_installed(post_setup_key: str) -> bool:
    """Return True when the post_setup install side-effect is satisfied."""
    predicate = _POST_SETUP_INSTALLED.get(post_setup_key)
    if predicate is None:
        # No install-state check registered → assume satisfied (don't
        # change behaviour for hooks we haven't explicitly opted in).
        return True
    try:
        return bool(predicate())
    except Exception:
        return True


def _module_installed(module_name: str) -> bool:
    """Cheap importable-without-importing check (no heavy side effects)."""
    import importlib.util

    try:
        return importlib.util.find_spec(module_name) is not None
    except Exception:
        return False


# Python dependencies installed explicitly through ``hermes tools`` are not
# part of the managed runtime's locked ``all`` sync. A runtime replacement
# therefore needs a small, static allowlist that can be snapshotted before the
# old site-packages disappears and restored afterward. Keep these install
# arguments in sync with the corresponding ``_run_post_setup`` branches.
_RESTORABLE_PYTHON_TOOL_DEPENDENCIES: dict[str, tuple[str, tuple[str, ...]]] = {
    "faster_whisper": ("faster_whisper", ("-U", "faster-whisper")),
    "kittentts": ("kittentts", ("-U", _KITTENTTS_WHEEL_URL, "soundfile")),
    "piper": ("piper", ("-U", "piper-tts")),
    "ddgs": ("ddgs", ("-U", "ddgs")),
    "langfuse": ("langfuse", ("langfuse",)),
}


def active_restorable_python_tool_dependencies() -> list[str]:
    """Return ``hermes tools`` Python dependencies present in this runtime."""
    return [
        name
        for name, (module_name, _install_args) in (
            _RESTORABLE_PYTHON_TOOL_DEPENDENCIES.items()
        )
        if _module_installed(module_name)
    ]


def restorable_python_tool_dependency(
    name: str,
) -> tuple[str, tuple[str, ...]] | None:
    """Return the import probe and pip arguments for an allowlisted tool."""
    return _RESTORABLE_PYTHON_TOOL_DEPENDENCIES.get(name)


def _agent_browser_installed() -> bool:
    """True when everything ``_run_post_setup("agent_browser")`` installs is present: the agent-browser
    CLI *and* the Chromium build it drives (or the Lightpanda engine, which needs no Chromium).
    Mirrors the hook so "Run setup" flips to an installed state only when re-running it would be a
    no-op.
    """
    from hermes_cli.nous_subscription import _local_browser_runnable

    # The install hook runs in a spawned ``hermes tools post-setup`` process,
    # but this probe runs in the long-lived web-server/CLI process, whose
    # browser_tool module may have cached a stale "Chromium missing" result
    # from before the install. Drop the cache (when the module is loaded) so
    # the readiness pill flips to Ready right after a successful setup run.
    bt = sys.modules.get("tools.browser_tool")
    if bt is not None:
        bt._cached_chromium_installed = None

    return _local_browser_runnable()


def _camofox_installed() -> bool:
    """True when the Camofox npm package ``_run_post_setup("camofox")``
    installs is already in node_modules."""
    return (PROJECT_ROOT / "node_modules" / "@askjo" / "camofox-browser").exists()


# post_setup_key -> predicate(): True when the install side-effect is already
# satisfied. Used by ``provider_readiness_status`` to decide whether a keyless
# post_setup row (KittenTTS, Piper, Local Browser, …) is honestly "ready" or
# still "needs_setup". Mirrors the installed-checks ``_run_post_setup`` itself
# performs before installing. ``xai_grok`` is intentionally absent — it is a
# credential bootstrap, not an install, and is handled as an auth check.
def _lightpanda_installed() -> bool:
    """True when a lightpanda binary is on PATH or in a known install dir."""
    try:
        from tools.browser_lightpanda import find_lightpanda_binary

        return find_lightpanda_binary() is not None
    except Exception:
        return False


def _cloud_agent_browser_installed() -> bool:
    """Installed-check for the ``browserbase`` hook (cloud provider rows).

    Cloud providers host their own Chromium, so their hook only installs the agent-browser npm
    package — presence of the CLI is the whole contract.
    """
    from hermes_cli.nous_subscription import _has_agent_browser

    return _has_agent_browser()


# Late-bound lambdas so tests can monkeypatch the underlying predicates.
_POST_SETUP_READY: dict = {
    **{key: (lambda m=module: _module_installed(m)) for key, (module, _args) in _RESTORABLE_PYTHON_TOOL_DEPENDENCIES.items()},
    "agent_browser": lambda: _agent_browser_installed(),
    "browserbase": lambda: _cloud_agent_browser_installed(),
    "camofox": lambda: _camofox_installed(),
    "lightpanda": lambda: _lightpanda_installed(),
    "cua_driver": lambda: _cua_driver_install_ready(),
}






def _reconfigure_tool(
    config: dict,
    *,
    force_fresh: bool = True,
):
    """Let user reconfigure an existing tool's provider or API key."""
    # Build list of configurable tools that are currently set up
    configurable = []
    for ts_key, ts_label, _ in _get_effective_configurable_toolsets():
        if _is_configurable(ts_key) and (
            _toolset_has_keys(ts_key, config, force_fresh=force_fresh)
            or _toolset_enabled_for_reconfigure(ts_key, config)
        ):
            configurable.append((ts_key, ts_label))

    if not configurable:
        _print_info("No configured tools to reconfigure.")
        return

    choices = [label for _, label in configurable]
    choices.append("Cancel")

    idx = _prompt_choice("  Which tool would you like to reconfigure?", choices, len(choices) - 1)

    if idx >= len(configurable):
        return  # Cancel

    ts_key, ts_label = configurable[idx]
    cat = TOOL_CATEGORIES.get(ts_key)

    if cat:
        _configure_tool_category(ts_key, cat, config, force_fresh=force_fresh, reconfigure=True)
    else:
        _configure_simple_requirements(ts_key, reconfigure=True)

    save_config(config)


def _toolset_enabled_for_reconfigure(ts_key: str, config: dict) -> bool:
    """Return True if a configurable toolset is enabled anywhere.

    Reconfigure must include enabled-but-unconfigured categories so users can finish provider/API-
    key setup without disabling and re-enabling the toolset.
    """
    for platform in PLATFORMS:
        if not _toolset_allowed_for_platform(ts_key, platform):
            continue
        try:
            enabled = _get_platform_tools(
                config,
                platform,
                include_default_mcp_servers=False,
            )
        except Exception:
            continue
        if ts_key in enabled:
            return True
    return False


# ─── Main Entry Point ─────────────────────────────────────────────────────────

def _shared_metrics_state(config: dict) -> tuple[bool, bool]:
    """Return (collection_enabled, send_enabled) from a config dict."""
    telemetry = config.get("telemetry")
    telemetry = telemetry if isinstance(telemetry, dict) else {}
    shared = telemetry.get("shared_metrics")
    shared = shared if isinstance(shared, dict) else {}
    return shared.get("enabled") is True, shared.get("send") is True


def _shared_metrics_menu_label(config: dict) -> str:
    """Menu row for shared metrics, showing both consent states."""
    enabled, send = _shared_metrics_state(config)
    if not enabled:
        state = "off"
    elif send:
        state = "collecting + sending to Nous"
    else:
        state = "collecting locally"
    return f"Configure shared metrics  ({state})"


def _configure_shared_metrics_interactive(config: dict) -> None:
    """Toggle shared-metrics collection and sending from `hermes tools`.

    Delegates to the setup wizard's prompt so the consent rules live in one place: sending requires
    collection, and disabling collection also disables sending.
    """
    from hermes_cli.setup import setup_telemetry

    before = _shared_metrics_state(config)
    setup_telemetry(config)
    after = _shared_metrics_state(config)
    if before != after:
        save_config(config)


def _print_toolset_diff(added: Set[str], removed: Set[str], *, indent: str = "  ") -> None:
    """Print ``+ label`` / ``- label`` lines for a checklist change."""
    for ts in sorted(added):
        print(color(f"{indent}+ {_toolset_label(ts)}", Colors.GREEN))
    for ts in sorted(removed):
        print(color(f"{indent}- {_toolset_label(ts)}", Colors.RED))


def _toolsets_needing_setup(new_enabled: Set[str], config: dict) -> List[str]:
    """Selected toolsets still missing provider/API-key setup, in sorted order.

    These must open configuration even when the checklist selection itself didn't change (e.g. Web
    Search already enabled but ``web.backend`` missing).
    """
    return [
        ts_key for ts_key in sorted(new_enabled)
        if _is_configurable(ts_key)
        and _toolset_needs_configuration_prompt(ts_key, config, force_fresh=True)
    ]


def _configure_newly_added(added: Set[str], already: Set[str], config: dict) -> None:
    """Configure newly enabled toolsets that need keys, skipping those already handled."""
    for ts_key in sorted(added - already):
        if _is_configurable(ts_key) and _toolset_needs_configuration_prompt(
            ts_key, config, force_fresh=True,
        ):
            _configure_toolset(ts_key, config)


def _platform_menu_label(config: dict, pkey: str) -> str:
    count = len(_get_platform_tools(config, pkey, include_default_mcp_servers=False))
    total = len(_get_effective_configurable_toolsets())
    return f"Configure {PLATFORMS[pkey]['label']}  ({count}/{total} enabled)"


def tools_command(args=None, first_install: bool = False, config: dict = None):
    """Entry point for `hermes tools` and `hermes setup tools`.

    ``first_install`` (fresh installs) skips the platform menu, goes straight to the CLI checklist
    and prompts for API keys on enabled tools. When the wizard passes its own ``config`` dict,
    platform_toolsets are written into it so they survive the wizard's final save_config().
    """
    if config is None:
        config = load_config()
    enabled_platforms = _get_enabled_platforms()

    print()

    # Non-interactive summary mode for CLI usage
    if getattr(args, "summary", False):
        total = len(_get_effective_configurable_toolsets())
        print(color("⚕ Tool Summary", Colors.CYAN, Colors.BOLD))
        print()
        summary = _platform_toolset_summary(config, enabled_platforms)
        for pkey in enabled_platforms:
            pinfo = PLATFORMS[pkey]
            enabled = summary.get(pkey, set())
            count = len(enabled)
            print(color(f"  {pinfo['label']}", Colors.BOLD) + color(f"  ({count}/{total})", Colors.DIM))
            if enabled:
                for ts_key in sorted(enabled):
                    print(color(f"    ✓ {_toolset_label(ts_key)}", Colors.GREEN))
            else:
                print(color("    (none enabled)", Colors.DIM))
        print()
        return
    print(color("⚕ Hermes Tool Configuration", Colors.CYAN, Colors.BOLD))
    print(color("  Enable or disable tools per platform.", Colors.DIM))
    print(color("  Tools that need API keys will be configured when enabled.", Colors.DIM))
    print(color("  Guide: https://hermes-agent.nousresearch.com/docs/user-guide/features/tools", Colors.DIM))
    print()

    def _configure_list(to_configure, *, selected=True):
        if not to_configure:
            return
        print()
        what = "selected tool(s)" if selected else "tool(s)"
        print(color(f"  Configuring {len(to_configure)} {what}:", Colors.YELLOW))
        for ts_key in to_configure:
            print(color(f"    • {_toolset_label(ts_key)}", Colors.DIM))
        print(color("  You can skip any tool you don't need right now.", Colors.DIM))
        print()
        for ts_key in to_configure:
            _configure_toolset(ts_key, config)

    # ── First-time install: linear flow, no platform menu ──
    if first_install:
        for pkey in enabled_platforms:
            pinfo = PLATFORMS[pkey]
            current_enabled = _get_platform_tools(config, pkey, include_default_mcp_servers=False)

            # Uncheck toolsets that should be off by default
            checklist_preselected = current_enabled - _DEFAULT_OFF_TOOLSETS

            # Show checklist
            new_enabled = _prompt_toolset_checklist(pinfo["label"], checklist_preselected, pkey)

            # Only diff against toolsets the checklist actually offered. The
            # resolved ``current_enabled`` can include non-configurable toolsets
            # (e.g. ``kanban``, recovered platform composites) the user was
            # never shown a checkbox for; without this scope the summary would
            # print spurious ``- kanban`` removals even though the config keeps
            # them. See _checklist_toolset_keys.
            _diff_universe = _checklist_toolset_keys(pkey)
            _print_toolset_diff(
                (new_enabled - current_enabled) & _diff_universe,
                (current_enabled - new_enabled) & _diff_universe,
            )

            auto_configured = apply_nous_managed_defaults(
                config,
                enabled_toolsets=new_enabled,
                force_fresh=True,
            )
            for ts_key in sorted(auto_configured):
                label = next((l for k, l, _ in CONFIGURABLE_TOOLSETS if k == ts_key), ts_key)
                print(color(f"  ✓ {label}: using your Nous subscription defaults", Colors.GREEN))

            # Walk through ALL selected tools that have provider options or
            # need API keys.  This ensures browser (Local vs Browserbase),
            # TTS (Edge vs OpenAI vs ElevenLabs), etc. are shown even when
            # a free provider exists.
            _configure_list(
                [
                    ts_key for ts_key in sorted(new_enabled)
                    if _is_configurable(ts_key) and ts_key not in auto_configured
                ],
                selected=False,
            )

            _save_platform_tools(config, pkey, new_enabled)
            save_config(config)
            print(color(f"  ✓ Saved {pinfo['label']} tool configuration", Colors.GREEN))
            print()

        return

    # ── Returning user: platform menu loop ──
    platform_keys = list(enabled_platforms)
    platform_choices = [_platform_menu_label(config, pkey) for pkey in platform_keys]

    if len(platform_keys) > 1:
        platform_choices.append("Configure all platforms (global)")
    platform_choices.append("Reconfigure an existing tool's provider or API key")
    platform_choices.append(_shared_metrics_menu_label(config))

    # Show MCP option if any MCP servers are configured
    _has_mcp = bool(config.get("mcp_servers"))
    if _has_mcp:
        platform_choices.append("Configure MCP server tools")

    platform_choices.append("Done")

    # Index offsets for the extra options after per-platform entries
    _global_idx = len(platform_keys) if len(platform_keys) > 1 else -1
    _reconfig_idx = len(platform_keys) + (1 if len(platform_keys) > 1 else 0)
    _metrics_idx = _reconfig_idx + 1
    _mcp_idx = (_metrics_idx + 1) if _has_mcp else -1
    _done_idx = _metrics_idx + (2 if _has_mcp else 1)

    while True:
        idx = _prompt_choice("Select an option:", platform_choices, default=0)

        if idx == _done_idx:
            break

        if idx == _reconfig_idx:
            _reconfigure_tool(config, force_fresh=True)
            print()
            continue

        if idx == _metrics_idx:
            _configure_shared_metrics_interactive(config)
            platform_choices[_metrics_idx] = _shared_metrics_menu_label(config)
            print()
            continue

        if idx == _mcp_idx:
            _configure_mcp_tools_interactive(config)
            print()
            continue

        if idx == _global_idx:
            # Use the union of all platforms' current tools as the starting state
            all_current = set()
            for pk in platform_keys:
                all_current |= _get_platform_tools(config, pk, include_default_mcp_servers=False)
            new_enabled = _prompt_toolset_checklist(
                "All platforms",
                all_current,
                force_fresh=True,
            )
            selected_to_configure = _toolsets_needing_setup(new_enabled, config)
            _configure_list(selected_to_configure)

            if new_enabled != all_current or selected_to_configure:
                for pk in platform_keys:
                    prev = _get_platform_tools(config, pk, include_default_mcp_servers=False)
                    # Scope the printed diff to the checklist's universe (see
                    # _checklist_toolset_keys) so non-configurable toolsets like
                    # ``kanban`` aren't reported as added/removed.
                    _diff_universe = _checklist_toolset_keys(pk)
                    added = (new_enabled - prev) & _diff_universe
                    removed = (prev - new_enabled) & _diff_universe
                    if added or removed:
                        print(color(f"  {PLATFORMS[pk]['label']}:", Colors.DIM))
                        _print_toolset_diff(added, removed, indent="    ")
                    # Configure API keys for newly enabled tools not already
                    # handled by the global selected-tool pass above, so a
                    # tool that was already enabled globally but lacked
                    # provider configuration doesn't drop the user back to the
                    # main menu.
                    _configure_newly_added(added, set(selected_to_configure), config)
                    _save_platform_tools(config, pk, new_enabled)
                save_config(config)
                print(color("  ✓ Saved configuration for all platforms", Colors.GREEN))
                for ci, pk in enumerate(platform_keys):
                    platform_choices[ci] = _platform_menu_label(config, pk)
            else:
                print(color("  No changes", Colors.DIM))
            print()
            continue

        pkey = platform_keys[idx]
        pinfo = PLATFORMS[pkey]

        current_enabled = _get_platform_tools(config, pkey, include_default_mcp_servers=False)
        new_enabled = _prompt_toolset_checklist(
            pinfo["label"],
            current_enabled,
            force_fresh=True,
        )

        selected_to_configure = _toolsets_needing_setup(new_enabled, config)
        _configure_list(selected_to_configure)

        if new_enabled != current_enabled or selected_to_configure:
            _diff_universe = _checklist_toolset_keys(pkey)
            added = (new_enabled - current_enabled) & _diff_universe
            removed = (current_enabled - new_enabled) & _diff_universe
            _print_toolset_diff(added, removed)
            _configure_newly_added(added, set(selected_to_configure), config)

            _save_platform_tools(config, pkey, new_enabled)
            save_config(config)
            print(color(f"  ✓ Saved {pinfo['label']} configuration", Colors.GREEN))
        else:
            print(color(f"  No changes to {pinfo['label']}", Colors.DIM))

        print()
        platform_choices[idx] = _platform_menu_label(config, pkey)

    print()
    from hermes_constants import display_hermes_home
    print(color(f"  Tool configuration saved to {display_hermes_home()}/config.yaml", Colors.DIM))
    print(color("  Changes take effect on next 'hermes' or gateway restart.", Colors.DIM))
    print()


# ─── MCP Tools Interactive Configuration ─────────────────────────────────────


def _mcp_match_filter():
    """Runtime name-filter matcher (exact names or fnmatch globs), with a literal fallback.

    Must use the SAME semantics as tools/mcp_tool.py registration — a literal ``in`` check renders
    glob excludes (e.g. ``*team_member*`` from catalog default_excluded manifests) as if nothing
    were excluded.
    """
    try:
        from tools.mcp_tool import matches_name_filter

        return matches_name_filter
    except ImportError:  # pragma: no cover — defensive fallback
        return lambda tool_name, patterns: tool_name in patterns


def _mcp_preselected(tool_names: List[str], include_set, exclude_set, match) -> Set[int]:
    """Indices of tools currently enabled: include mode, exclude mode, or all when unfiltered."""
    if include_set:
        return {i for i, tn in enumerate(tool_names) if match(tn, include_set)}
    if exclude_set:
        return {i for i, tn in enumerate(tool_names) if not match(tn, exclude_set)}
    return set(range(len(tool_names)))


def _apply_mcp_checklist(server_name: str, tools_cfg: dict, tool_names: List[str], chosen: Set[int],
                         include_set, exclude_set, match) -> None:
    """Write a checklist result back as ``tools.include`` / ``tools.exclude``."""
    exclude_mode = bool(exclude_set) and not include_set

    if len(chosen) == len(tool_names) and not exclude_mode:
        # All tools enabled — clear filters (cleanest config shape; the
        # server's native tool set is the active set, and any tools the
        # server adds later are auto-enabled).
        tools_cfg.pop("exclude", None)
        tools_cfg.pop("include", None)
    elif exclude_mode:
        # Exclude-mode server (catalog default_excluded / hand-written
        # tools.exclude): stay in exclude mode — do NOT demote the
        # dynamic filter to a frozen include list. Unchecked tools are
        # added as literal excludes; re-checked literals are dropped;
        # glob patterns are preserved (they intentionally keep matching
        # tools the vendor ships later).
        old_exclude = sorted(exclude_set or set())
        glob_entries = [p for p in old_exclude if "*" in p or "?" in p or "[" in p]
        literal_entries = {p for p in old_exclude if p not in glob_entries}
        unchecked = {tn for i, tn in enumerate(tool_names) if i not in chosen}
        checked = {tool_names[i] for i in chosen}
        new_literals = (literal_entries - checked) | {
            tn for tn in unchecked if not match(tn, set(old_exclude))
        }
        new_exclude = glob_entries + sorted(new_literals)
        glob_shadowed = sorted(
            tn for tn in checked if glob_entries and match(tn, set(glob_entries))
        )
        if glob_shadowed:
            _print_warning(
                f"  {server_name}: {len(glob_shadowed)} re-enabled "
                f"tool(s) still match glob exclude pattern(s) "
                f"{glob_entries} and stay excluded — edit "
                f"mcp_servers.{server_name}.tools.exclude in config.yaml "
                "to enable them."
            )
        if new_exclude:
            tools_cfg["exclude"] = new_exclude
        else:
            tools_cfg.pop("exclude", None)
        tools_cfg.pop("include", None)
    else:
        tools_cfg["include"] = [tool_names[i] for i in sorted(chosen)]
        # Drop any legacy exclude block — we're include-mode now.
        tools_cfg.pop("exclude", None)


def _configure_mcp_tools_interactive(config: dict):
    """Probe MCP servers for available tools and let user toggle them on/off.

    Connects to each server, discovers tools, shows a per-server curses checklist, and writes the
    result back as ``tools.exclude`` entries in config.yaml.
    """
    from hermes_cli.curses_ui import curses_checklist

    mcp_servers = config.get("mcp_servers") or {}
    if not mcp_servers:
        _print_info("No MCP servers configured.")
        return

    enabled_names = [
        k for k, v in mcp_servers.items()
        if v.get("enabled", True) not in {False, "false", "0", "no", "off"}
    ]
    if not enabled_names:
        _print_info("All MCP servers are disabled.")
        return

    print()
    print(color("  Discovering tools from MCP servers...", Colors.YELLOW))
    print(color(f"  Connecting to {len(enabled_names)} server(s): {', '.join(enabled_names)}", Colors.DIM))

    try:
        from tools.mcp_tool import probe_mcp_server_tools
        server_tools = probe_mcp_server_tools()
    except Exception as exc:
        _print_error(f"Failed to probe MCP servers: {exc}")
        return

    if not server_tools:
        _print_warning("Could not discover tools from any MCP server.")
        _print_info("Check that server commands/URLs are correct and dependencies are installed.")
        return

    for name in enabled_names:
        if name not in server_tools:
            _print_warning(f"  Could not connect to '{name}'")

    total_tools = sum(len(tools) for tools in server_tools.values())
    print(color(f"  Found {total_tools} tool(s) across {len(server_tools)} server(s)", Colors.GREEN))
    print()

    any_changes = False
    for server_name, tools in server_tools.items():
        if not tools:
            _print_info(f"  {server_name}: no tools found")
            continue

        tools_cfg = mcp_servers.get(server_name, {}).get("tools") or {}
        include_list = tools_cfg.get("include") or []
        exclude_list = tools_cfg.get("exclude") or []

        labels = []
        for tool_name, description in tools:
            desc_short = description[:70] + "..." if len(description) > 70 else description
            labels.append(f"{tool_name}  ({desc_short})" if desc_short else tool_name)

        match = _mcp_match_filter()
        tool_names = [t[0] for t in tools]
        include_set = {str(p) for p in include_list} if include_list else None
        exclude_set = {str(p) for p in exclude_list} if exclude_list else None
        pre_selected = _mcp_preselected(tool_names, include_set, exclude_set, match)

        chosen = curses_checklist(
            f"MCP Server: {server_name}  ({len(tools)} tools)",
            labels,
            pre_selected,
            cancel_returns=pre_selected,
        )

        if chosen == pre_selected:
            _print_info(f"  {server_name}: no changes")
            continue

        tools_cfg = mcp_servers.setdefault(server_name, {}).setdefault("tools", {})
        _apply_mcp_checklist(server_name, tools_cfg, tool_names, chosen, include_set, exclude_set, match)

        _print_success(
            f"  {server_name}: {len(chosen)} enabled, {len(tools) - len(chosen)} disabled"
        )
        any_changes = True

    if any_changes:
        save_config(config)
        print()
        print(color("  ✓ MCP tool configuration saved", Colors.GREEN))
    else:
        print(color("  No changes to MCP tools", Colors.DIM))


# ─── Non-interactive disable/enable ──────────────────────────────────────────


def _apply_toolset_change(config: dict, platform: str, toolset_names: List[str], action: str):
    """Add or remove built-in toolsets for a platform."""
    enabled = _get_platform_tools(config, platform, include_default_mcp_servers=False)
    if action == "disable":
        updated = enabled - set(toolset_names)
    else:
        updated = enabled | set(toolset_names)
    _save_platform_tools(config, platform, updated)


def _apply_mcp_change(config: dict, targets: List[str], action: str) -> Set[str]:
    """Add or remove specific MCP tools from a server's exclude list."""
    failed_servers: Set[str] = set()
    mcp_servers = config.get("mcp_servers") or {}

    for target in targets:
        server_name, tool_name = target.split(":", 1)
        if server_name not in mcp_servers:
            failed_servers.add(server_name)
            continue
        tools_cfg = mcp_servers[server_name].setdefault("tools", {})
        exclude = list(tools_cfg.get("exclude") or [])
        if action == "disable":
            if tool_name not in exclude:
                exclude.append(tool_name)
        else:
            exclude = [t for t in exclude if t != tool_name]
        tools_cfg["exclude"] = exclude

    return failed_servers


def _print_tools_list(enabled_toolsets: set, mcp_servers: dict, platform: str = "cli"):
    """Print a summary of enabled/disabled toolsets and MCP tool filters."""
    effective_all = _get_effective_configurable_toolsets()
    effective = [
        (k, l, d) for (k, l, d) in effective_all
        if _toolset_allowed_for_platform(k, platform)
    ]
    builtin_keys = {ts_key for ts_key, _, _ in CONFIGURABLE_TOOLSETS}

    def _print_rows(entries):
        for ts_key, label in entries:
            status = (color("✓ enabled", Colors.GREEN) if ts_key in enabled_toolsets
                      else color("✗ disabled", Colors.RED))
            print(f"  {status}  {ts_key}  {color(label, Colors.DIM)}")

    print(f"Built-in toolsets ({platform}):")
    _print_rows((k, l) for k, l, _ in effective if k in builtin_keys)

    plugin_entries = [(k, l) for k, l, _ in effective if k not in builtin_keys]
    if plugin_entries:
        print()
        print(f"Plugin toolsets ({platform}):")
        _print_rows(plugin_entries)

    if mcp_servers:
        print()
        print("MCP servers:")
        for srv_name, srv_cfg in mcp_servers.items():
            tools_cfg = srv_cfg.get("tools") or {}
            exclude = tools_cfg.get("exclude") or []
            include = tools_cfg.get("include") or []
            if include:
                _print_info(f"{srv_name}  [include only: {', '.join(include)}]")
            elif exclude:
                _print_info(f"{srv_name}  [excluded: {color(', '.join(exclude), Colors.YELLOW)}]")
            else:
                _print_info(f"{srv_name}  {color('all tools enabled', Colors.DIM)}")


def _known_tool_platforms() -> set[str]:
    """Return built-in plus discovered plugin platform names.

    Plugin platforms are registered at runtime rather than in the static CLI display registry. Tool
    introspection/configuration must recognize those names too, otherwise an active plugin platform
    cannot audit its authority.
    """
    known = set(PLATFORMS)
    try:
        from hermes_cli.plugins import discover_plugins
        from gateway.platform_registry import platform_registry

        discover_plugins()  # idempotent
        known.update(platform_registry.registered_names())
    except Exception:
        # Plugin discovery is optional. Preserve the built-in CLI path when a
        # third-party plugin is malformed or its dependencies are unavailable.
        pass
    return known


def tools_disable_enable_command(args):
    """Enable, disable, or list tools for a platform."""
    action = args.tools_action
    platform = getattr(args, "platform", "cli")
    config = load_config()

    valid_platforms = _known_tool_platforms()
    if platform not in valid_platforms:
        _print_error(f"Unknown platform '{platform}'. Valid: {', '.join(sorted(valid_platforms))}")
        return

    if action == "list":
        _print_tools_list(_get_platform_tools(config, platform, include_default_mcp_servers=False),
                          config.get("mcp_servers") or {}, platform)
        return

    targets: List[str] = args.names
    toolset_targets = [t for t in targets if ":" not in t]
    mcp_targets = [t for t in targets if ":" in t]

    valid_toolsets = {ts_key for ts_key, _, _ in CONFIGURABLE_TOOLSETS} | _get_plugin_toolset_keys()
    unknown_toolsets = [t for t in toolset_targets if t not in valid_toolsets]
    if unknown_toolsets:
        for name in unknown_toolsets:
            _print_error(f"Unknown toolset '{name}'")
        toolset_targets = [t for t in toolset_targets if t in valid_toolsets]

    # Reject platform-scoped toolsets on platforms that don't allow them.
    restricted_targets = [
        t for t in toolset_targets
        if not _toolset_allowed_for_platform(t, platform)
    ]
    if restricted_targets:
        for name in restricted_targets:
            allowed = sorted(_TOOLSET_PLATFORM_RESTRICTIONS.get(name) or set())
            _print_error(
                f"Toolset '{name}' is not available on platform '{platform}' "
                f"(only: {', '.join(allowed)})"
            )
        toolset_targets = [t for t in toolset_targets if t not in restricted_targets]

    if toolset_targets:
        _apply_toolset_change(config, platform, toolset_targets, action)

    failed_servers: Set[str] = set()
    if mcp_targets:
        failed_servers = _apply_mcp_change(config, mcp_targets, action)
        for srv in failed_servers:
            _print_error(f"MCP server '{srv}' not found in config")

    save_config(config)

    successful = [
        t for t in targets
        if t not in unknown_toolsets
        and t not in restricted_targets
        and (":" not in t or t.split(":")[0] not in failed_servers)
    ]
    if successful:
        verb = "Disabled" if action == "disable" else "Enabled"
        _print_success(f"{verb}: {', '.join(successful)}")
