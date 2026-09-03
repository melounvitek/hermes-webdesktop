"""Setup-completion summary (tool availability + "Setup Complete!" banner).

Names originating in setup.py are imported lazily inside function bodies so test patches on
``hermes_cli.setup.<name>`` take effect.
"""

import logging

logger = logging.getLogger("hermes_cli.setup")

# provider -> (label, env vars: any one set means available; empty = always).
# Local engines are (label, module, hint) and must be importable.
_TTS_SUMMARY_ROWS = {
    "elevenlabs": ("ElevenLabs", ("ELEVENLABS_API_KEY",)),
    "openai": ("OpenAI", ("VOICE_TOOLS_OPENAI_KEY", "OPENAI_API_KEY")),
    "minimax": ("MiniMax", ("MINIMAX_API_KEY",)),
    "mistral": ("Mistral Voxtral", ("MISTRAL_API_KEY",)),
    "gemini": ("Google Gemini", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
    "neutts": ("NeuTTS", "neutts", "run 'hermes setup tts'"),
    "kittentts": ("KittenTTS", "kittentts", "run 'hermes setup tts'"),
}
_TTS_SUMMARY_DEFAULT = ("Edge TTS", ())
_STT_SUMMARY_ROWS = {
    "openai": ("OpenAI", ("VOICE_TOOLS_OPENAI_KEY", "OPENAI_API_KEY")),
    "groq": ("Groq Whisper", ("GROQ_API_KEY",)),
    "elevenlabs": ("ElevenLabs Scribe", ("ELEVENLABS_API_KEY",)),
    "xai": ("xAI", ()),
    "deepinfra": ("DeepInfra", ("DEEPINFRA_API_KEY",)),
}
_STT_SUMMARY_DEFAULT = ("Local Whisper", "faster_whisper", "run 'hermes tools' → Speech-to-Text")

# Browser "missing" hint keyed by the configured provider; anything else gets the generic hint.
_BROWSER_MISSING_HINTS = {
    "Browserbase": "npm install -g agent-browser and set BROWSERBASE_API_KEY/BROWSERBASE_PROJECT_ID",
    "Browser Use": "npm install -g agent-browser and set BROWSER_USE_API_KEY",
    "Camofox": "CAMOFOX_URL",
    "Local browser": "npm install -g agent-browser && agent-browser install --with-deps",
}
_BROWSER_MISSING_DEFAULT = "npm install -g agent-browser, set CAMOFOX_URL, or configure Browser Use or Browserbase"
_WEB_MISSING = ("EXA_API_KEY, PARALLEL_API_KEY, FIRECRAWL_API_KEY/FIRECRAWL_API_URL, TAVILY_API_KEY, "
                "KEENABLE_API_KEY, or SEARXNG_URL")

_DONE_BANNER = (
    "┌─────────────────────────────────────────────────────────┐",
    "│              ✓ Setup Complete!                          │",
    "└─────────────────────────────────────────────────────────┘",
)
# (command, description) rows; the description carries its own alignment padding.
_EDIT_WIZARD_ROWS = (
    ("hermes setup", "          Re-run the full wizard"),
    ("hermes setup model", "    Change model/provider"),
    ("hermes setup terminal", " Change terminal backend"),
    ("hermes setup gateway", "  Configure messaging"),
    ("hermes setup tools", "    Configure tool providers"),
)
_EDIT_CONFIG_ROWS = (
    ("hermes config", "         View current settings"),
    ("hermes config edit", "    Open config in your editor"),
    ("hermes config set <key> <value>", ""),
)
_READY_ROWS = (
    ("hermes", "              Start chatting"),
    ("hermes gateway", "      Start messaging gateway"),
    ("hermes doctor", "       Check for issues"),
)


def _voice_provider_status(kind: str, provider: str, rows: dict, default: tuple) -> tuple:
    """Summary row for a TTS/STT provider. A keyed provider whose key is missing
    falls through to the default row, matching the runtime fallback."""
    from hermes_cli.setup import get_env_value, _module_installed
    row = rows.get(provider, default)
    if isinstance(row[1], tuple) and row[1] and not any(get_env_value(v) for v in row[1]):
        row = default
    if isinstance(row[1], tuple):
        return (f"{kind} ({row[0]})", True, None)
    label, module, hint = row
    if _module_installed(module):
        return (f"{kind} ({label}{' local' if kind == 'Text-to-Speech' else ''})", True, None)
    return (f"{kind} ({label} — not installed)", False, hint)


def _first_available_plugin_provider(registry: str, skip: str = None):
    """display_name of the first plugin-registered provider in ``agent.<registry>`` that reports
    available (fail-soft: any error means none), skipping ``skip``."""
    try:
        import importlib
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
        for provider in importlib.import_module(f"agent.{registry}").list_providers():
            if provider.name == skip:
                continue
            try:
                if provider.is_available():
                    return provider.display_name
            except Exception:
                continue
    except Exception:
        pass
    return None


# ---- tool_status row builders: each takes (config, subscription_features) and returns
# a (name, available, hint) row or None (row omitted). Evaluated in _TOOL_ROW_BUILDERS order.

def _vision_row(config, feats):
    # Use the same runtime resolver as the actual vision tools.
    try:
        from agent.auxiliary_client import get_available_vision_backends

        ok = bool(get_available_vision_backends())
    except Exception:
        ok = False
    return ("Vision (image analysis)", True, None) if ok else ("Vision (image analysis)", False, "run 'hermes setup' to configure")


def _managed_or_provider_row(feature, name: str, managed_label: str, missing_hint: str):
    """Row for a Nous-manageable feature: managed > available (with provider) > missing hint."""
    if feature.managed_by_nous:
        return (f"{name} ({managed_label})", True, None)
    if feature.available:
        return (f"{name} ({feature.current_provider})" if feature.current_provider else name, True, None)
    return (name, False, missing_hint)


def _web_row(config, feats):
    # Web tools (Exa, Parallel, Firecrawl, Tavily, or Keenable)
    return _managed_or_provider_row(feats.web, "Web Search & Extract", "Nous subscription", _WEB_MISSING)


def _browser_row(config, feats):
    # Browser tools (local Chromium, Camofox, Browserbase, Browser Use, or Firecrawl)
    hint = _BROWSER_MISSING_HINTS.get(feats.browser.current_provider, _BROWSER_MISSING_DEFAULT)
    return _managed_or_provider_row(feats.browser, "Browser Automation", "Nous Browser Use", hint)


def _image_gen_row(config, feats):
    # FAL (direct or via Nous), or any plugin-registered provider (OpenAI, etc.)
    if feats.image_gen.managed_by_nous:
        return ("Image Generation (Nous subscription)", True, None)
    if feats.image_gen.available:
        return ("Image Generation", True, None)
    # Probe plugin-registered providers so OpenAI-only setups don't show as "missing FAL_KEY".
    backend = _first_available_plugin_provider("image_gen_registry", skip="fal")
    if backend:
        return (f"Image Generation ({backend})", True, None)
    return ("Image Generation", False, "FAL_KEY or OPENAI_API_KEY")


def _video_gen_row(config, feats):
    # Opt-in via `hermes tools` → Video Generation. Only show the row when a plugin reports
    # available so we don't badger users who don't care about video gen with a "missing" line.
    if feats.video_gen.managed_by_nous:
        return ("Video Generation (FAL via Nous subscription)", True, None)
    backend = _first_available_plugin_provider("video_gen_registry")
    return (f"Video Generation ({backend})", True, None) if backend else None


def _tts_row(config, feats):
    # Configured provider, gated on its key (or local install)
    from hermes_cli.setup import cfg_get
    tts_provider = cfg_get(config, "tts", "provider", default="edge")
    if feats.tts.managed_by_nous:
        return ("Text-to-Speech (OpenAI via Nous subscription)", True, None)
    return _voice_provider_status("Text-to-Speech", tts_provider, _TTS_SUMMARY_ROWS, _TTS_SUMMARY_DEFAULT)


def _stt_row(config, feats):
    from hermes_cli.setup import cfg_get
    stt_provider = cfg_get(config, "stt", "provider", default="local") or "local"
    _stt_feature = feats.features.get("stt")
    if _stt_feature is not None and _stt_feature.managed_by_nous:
        return ("Speech-to-Text (OpenAI via Nous subscription)", True, None)
    return _voice_provider_status("Speech-to-Text", stt_provider, _STT_SUMMARY_ROWS, _STT_SUMMARY_DEFAULT)


def _modal_row(config, feats):
    from hermes_cli.setup import cfg_get, managed_nous_tools_enabled
    if feats.modal.managed_by_nous:
        return ("Modal Execution (Nous subscription)", True, None)
    if cfg_get(config, "terminal", "backend") == "modal":
        if feats.modal.direct_override:
            return ("Modal Execution (direct Modal)", True, None)
        return ("Modal Execution", False, "run 'hermes setup terminal'")
    if managed_nous_tools_enabled() and feats.nous_auth_present:
        return ("Modal Execution (optional via Nous subscription)", True, None)
    return None


def _home_assistant_row(config, feats):
    from hermes_cli.setup import get_env_value
    return ("Smart Home (Home Assistant)", True, None) if get_env_value("HASS_TOKEN") else None


def _spotify_row(config, feats):
    # OAuth via hermes auth spotify — check auth.json, not env vars
    try:
        from hermes_cli.auth import get_provider_auth_state
        _spotify_state = get_provider_auth_state("spotify") or {}
        if _spotify_state.get("access_token") or _spotify_state.get("refresh_token"):
            return ("Spotify (PKCE OAuth)", True, None)
    except Exception:
        pass
    return None


def _skills_hub_row(config, feats):
    from hermes_cli.setup import get_env_value
    ok = bool(get_env_value("GITHUB_TOKEN"))
    return ("Skills Hub (GitHub)", ok, None if ok else "GITHUB_TOKEN")


def _always_on_rows(config, feats):
    # Terminal (system deps met), task planning (in-memory), skills (bundled + user-created).
    return [("Terminal/Commands", True, None), ("Task Planning (todo)", True, None),
            ("Skills (view, create, edit)", True, None)]


_TOOL_ROW_BUILDERS = (
    _vision_row, _web_row, _browser_row, _image_gen_row, _video_gen_row, _tts_row, _stt_row,
    _modal_row, _home_assistant_row, _spotify_row, _skills_hub_row, _always_on_rows,
)


def _print_cmd_rows(rows):
    """Print (command, description) rows as '   <green cmd><desc>'."""
    from hermes_cli.setup import color, Colors
    for cmd, desc in rows:
        print(f"   {color(cmd, Colors.GREEN)}{desc}")


def _print_section_header(title):
    from hermes_cli.setup import color, Colors
    print(color("─" * 60, Colors.DIM))
    print()
    print(color(title, Colors.CYAN, Colors.BOLD))
    print()


def _print_setup_summary(config: dict, hermes_home):
    """Print the setup completion summary."""
    from hermes_cli.setup import (
        color, Colors, get_config_path, get_env_path, get_nous_subscription_features, _info, print_header,
        print_warning,
    )
    from hermes_constants import display_hermes_home as _dhh
    # Provider readiness — the one thing setup must produce. A user who cancelled the API-key
    # prompt mid-wizard used to exit "successfully" with NO working model; say so loudly.
    try:
        from hermes_cli.auth import resolve_provider

        resolve_provider()
    except Exception:
        print()
        print_warning("No inference provider is configured — Hermes cannot chat yet.")
        _info("  Finish this one step with either of:",
              "    hermes model            (pick any provider/model)",
              "    hermes setup --portal   (Nous Portal OAuth, no API key)")

    print()
    print_header("Tool Availability Summary")

    tool_status = []
    subscription_features = get_nous_subscription_features(config)
    for build in _TOOL_ROW_BUILDERS:
        row = build(config, subscription_features)
        tool_status.extend(row if isinstance(row, list) else [row] if row is not None else [])

    available_count = sum(1 for _, avail, _ in tool_status if avail)
    _info(f"{available_count}/{len(tool_status)} tool categories available:", None)
    for name, available, missing_var in tool_status:
        if available:
            print(f"   {color('✓', Colors.GREEN)} {name}")
        else:
            print(f"   {color('✗', Colors.RED)} {name} {color(f'(missing {missing_var})', Colors.DIM)}")
    print()

    if available_count < len(tool_status):
        print_warning("Some tools are disabled. Run 'hermes setup tools' to configure them,")
        print_warning(f"or edit {_dhh()}/.env directly to add the missing API keys.")
        print()

    print()
    for line in _DONE_BANNER:
        print(color(line, Colors.GREEN))
    print()
    print(color(f"📁 All your files are in {_dhh()}/:", Colors.CYAN, Colors.BOLD))
    print()
    print(f"   {color('Settings:', Colors.YELLOW)}  {get_config_path()}")
    print(f"   {color('API Keys:', Colors.YELLOW)}  {get_env_path()}")
    print(f"   {color('Data:', Colors.YELLOW)}      {hermes_home}/cron/, sessions/, logs/")
    print()

    _print_section_header("📝 To edit your configuration:")
    _print_cmd_rows(_EDIT_WIZARD_ROWS)
    print()
    _print_cmd_rows(_EDIT_CONFIG_ROWS)
    print("                          Set a specific value")
    print()
    print("   Or edit the files directly:")
    print(f"   {color(f'nano {get_config_path()}', Colors.DIM)}")
    print(f"   {color(f'nano {get_env_path()}', Colors.DIM)}")
    print()

    _print_section_header("🚀 Ready to go!")
    _print_cmd_rows(_READY_ROWS)
    print()
