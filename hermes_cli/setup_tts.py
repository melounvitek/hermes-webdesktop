"""Text-to-speech provider setup (provider picker, API-key prompts, local engine installs, xAI OAuth).

Extracted from hermes_cli/setup.py; setup.py re-exports the public entry points.
"""

import logging
import shutil
import subprocess
import sys

logger = logging.getLogger("hermes_cli.setup")


def _check_espeak_ng() -> bool:
    """Check if espeak-ng is installed."""
    return shutil.which("espeak-ng") is not None or shutil.which("espeak") is not None


def _pip_install_tts_package(name: str, pip_args: list, manual_cmd: str) -> bool:
    """Install a local TTS engine through the canonical uv → pip → ensurepip
    ladder so pip-less venvs (Ubuntu 25.10 ``python -m venv``, ``uv venv``) work."""
    from hermes_cli.setup import print_error, print_info, print_success
    from hermes_cli.tools_config import _pip_install
    try:
        result = _pip_install(pip_args, timeout=300)
    except Exception as e:
        print_error(f"Failed to install {name}: {e}")
        print_info(f"Try manually: {manual_cmd}")
        return False
    if result.returncode == 0:
        print_success(f"{name} installed successfully")
        return True
    err = (result.stderr or "").strip()
    print_error(f"Failed to install {name}: {err[:300] if err else 'install failed'}")
    print_info(f"Try manually: {manual_cmd}")
    return False


# sys.platform -> (manual install hint, install command); anything else uses "linux".
_ESPEAK_INSTALL = {
    "darwin": ("Install with: brew install espeak-ng", ["brew", "install", "espeak-ng"]),
    "win32": ("Install with: choco install espeak-ng", ["choco", "install", "espeak-ng", "-y"]),
    "linux": ("Install with: sudo apt install espeak-ng", ["sudo", "apt", "install", "-y", "espeak-ng"]),
}


def _install_neutts_deps() -> bool:
    """Install NeuTTS dependencies with user approval. Returns True on success."""
    from hermes_cli.setup import _info, print_info, print_success, print_warning, prompt_yes_no
    if not _check_espeak_ng():
        hint, install_cmd = _ESPEAK_INSTALL.get(sys.platform, _ESPEAK_INSTALL["linux"])
        print()
        print_warning("NeuTTS requires espeak-ng for phonemization.")
        print_info(hint)
        print()
        if prompt_yes_no("Install espeak-ng now?", True):
            try:
                subprocess.run(install_cmd, check=True)
                print_success("espeak-ng installed")
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                print_warning(f"Could not install espeak-ng automatically: {e}")
                print_info("Please install it manually and re-run setup.")
                return False
        else:
            print_warning("espeak-ng is required for NeuTTS. Install it manually before using NeuTTS.")

    _info(None, "Installing neutts Python package...",
          "This will also download the TTS model (~300MB) on first use.", None)
    return _pip_install_tts_package("neutts", ["-U", "neutts[all]", "--quiet"], "uv pip install -U 'neutts[all]'")


def _install_kittentts_deps() -> bool:
    """Install KittenTTS dependencies with user approval. Returns True on success."""
    from hermes_cli.setup import _info
    wheel_url = (
        "https://github.com/KittenML/KittenTTS/releases/download/"
        "0.8.1/kittentts-0.8.1-py3-none-any.whl"
    )
    _info(None, "Installing kittentts Python package (~25-80MB model downloaded on first use)...", None)
    return _pip_install_tts_package(
        "kittentts", ["-U", wheel_url, "soundfile", "--quiet"], f"uv pip install -U '{wheel_url}' soundfile",
    )


def _xai_oauth_logged_in_for_setup() -> bool:
    """True iff xAI Grok OAuth credentials are already stored locally.

    Lets TTS / STT setup skip the API-key prompt for users who logged in
    through ``hermes model`` -> xAI Grok OAuth (SuperGrok / Premium+).
    """
    try:
        from hermes_cli.auth import get_xai_oauth_auth_status

        return bool(get_xai_oauth_auth_status().get("logged_in"))
    except Exception:
        return False


def _run_xai_oauth_login_from_setup() -> bool:
    """Run the xAI Grok OAuth device-code login from inside the setup wizard.

    Saves OAuth tokens only. Does **not** switch the active inference
    provider or rewrite ``model.provider`` — callers (TTS setup, tools
    config) only need credentials for side tools.

    Returns True on success, False on any failure (the caller falls back
    to whatever the user picked next, e.g. Edge TTS).
    """
    from hermes_cli.setup import _info, print_warning
    try:
        from hermes_cli.auth import (
            _is_remote_session, _save_xai_oauth_tokens, _xai_oauth_device_code_login,
            unsuppress_credential_source,
        )
    except Exception as exc:
        print_warning(f"xAI Grok OAuth helpers unavailable: {exc}")
        return False

    open_browser = not _is_remote_session()
    _info(None, "Signing in to xAI Grok OAuth (SuperGrok / Premium+)...")
    try:
        creds = _xai_oauth_device_code_login(open_browser=open_browser)
        _save_xai_oauth_tokens(
            creds["tokens"], discovery=creds.get("discovery"), redirect_uri=creds.get("redirect_uri", ""),
            last_refresh=creds.get("last_refresh"), auth_mode="oauth_device_code", set_active=False,
        )
        # Mirror model/dashboard re-login: clear device_code suppression so
        # the pool can seed from the singleton after a prior `auth remove`.
        unsuppress_credential_source("xai-oauth", "device_code")
        return True
    except Exception as exc:
        print_warning(f"xAI Grok OAuth login failed: {exc}")
        return False


_TTS_PROVIDER_LABELS = {
    "edge": "Edge TTS", "elevenlabs": "ElevenLabs", "openai": "OpenAI TTS", "xai": "xAI TTS",
    "minimax": "MiniMax TTS", "mistral": "Mistral Voxtral TTS", "gemini": "Google Gemini TTS",
    "neutts": "NeuTTS", "kittentts": "KittenTTS",
}
_TTS_PROVIDER_CHOICES = [
    ("edge", "Edge TTS (free, cloud-based, no setup needed)"),
    ("elevenlabs", "ElevenLabs (premium quality, needs API key)"),
    ("openai", "OpenAI TTS (good quality, needs API key)"),
    ("xai", "xAI TTS (Grok voices — OAuth login or API key)"),
    ("minimax", "MiniMax TTS (high quality with voice cloning, needs API key)"),
    ("mistral", "Mistral Voxtral TTS (multilingual, native Opus, needs API key)"),
    ("gemini", "Google Gemini TTS (30 prebuilt voices, prompt-controllable, needs API key)"),
    ("neutts", "NeuTTS (local on-device, free, ~300MB model download)"),
    ("kittentts", "KittenTTS (local on-device, free, lightweight ~25-80MB ONNX)"),
]
# provider -> (env vars that satisfy it, env var to save, prompt, success line, pre-prompt hint)
_TTS_API_KEY_PROVIDERS = {
    "elevenlabs": (("ELEVENLABS_API_KEY",), "ELEVENLABS_API_KEY", "ElevenLabs API key", "ElevenLabs API key saved", ""),
    "openai": (
        ("VOICE_TOOLS_OPENAI_KEY", "OPENAI_API_KEY"), "VOICE_TOOLS_OPENAI_KEY",
        "OpenAI API key for TTS", "OpenAI TTS API key saved", "",
    ),
    "minimax": (("MINIMAX_API_KEY",), "MINIMAX_API_KEY", "MiniMax API key for TTS", "MiniMax TTS API key saved", ""),
    "mistral": (("MISTRAL_API_KEY",), "MISTRAL_API_KEY", "Mistral API key for TTS", "Mistral TTS API key saved", ""),
    "gemini": (
        ("GEMINI_API_KEY", "GOOGLE_API_KEY"), "GEMINI_API_KEY", "Gemini API key for TTS", "Gemini TTS API key saved",
        "Get a free API key at https://aistudio.google.com/app/apikey",
    ),
}
# provider -> (module, display name, requirement lines, install question, installer)
_TTS_LOCAL_PROVIDERS = {
    "neutts": (
        "neutts", "NeuTTS",
        ("NeuTTS requires:", "  • Python package: neutts (~50MB install + ~300MB model on first use)",
         "  • System package: espeak-ng (phonemizer)"),
        "Install NeuTTS dependencies now?", _install_neutts_deps,
    ),
    "kittentts": (
        "kittentts", "KittenTTS",
        ("KittenTTS is lightweight (~25-80MB, CPU-only, no API key required).",
         "Voices: Jasper, Bella, Luna, Bruno, Rosie, Hugo, Kiki, Leo"),
        "Install KittenTTS now?", _install_kittentts_deps,
    ),
}


def _tts_api_key_step(selected: str) -> str:
    """Ensure the key for an API-key TTS provider exists; fall back to edge otherwise."""
    from hermes_cli.setup import get_env_value, print_info, print_success, print_warning, prompt, save_env_value
    env_vars, save_var, prompt_label, saved_msg, hint = _TTS_API_KEY_PROVIDERS[selected]
    if any(get_env_value(v) for v in env_vars):
        return selected
    print()
    if hint:
        print_info(hint)
    api_key = prompt(prompt_label, password=True)
    if api_key:
        save_env_value(save_var, api_key)
        print_success(saved_msg)
        return selected
    print_warning("No API key provided. Falling back to Edge TTS.")
    return "edge"


def _tts_local_install_step(selected: str) -> str:
    """Offer to install a local TTS engine; fall back to edge if declined/failed."""
    from hermes_cli.setup import _module_installed, print_info, print_success, print_warning, prompt_yes_no
    module, name, lines, question, installer = _TTS_LOCAL_PROVIDERS[selected]
    if _module_installed(module):
        print_success(f"{name} is already installed")
        return selected
    print()
    for line in lines:
        print_info(line)
    print()
    if not prompt_yes_no(question, True):
        print_info(f"Skipping install. Set tts.provider to '{selected}' after installing manually.")
        return "edge"
    if not installer():
        print_warning(f"{name} installation incomplete. Falling back to Edge TTS.")
        return "edge"
    return selected


def _tts_xai_step(config: dict) -> str:
    """xAI TTS auth. Order: existing OAuth tokens (free for SuperGrok) > existing
    XAI_API_KEY > offer both paths — xAI TTS works with OAuth bearer tokens too."""
    from hermes_cli.setup import (
        _run_xai_oauth_login_from_setup, _xai_oauth_logged_in_for_setup, get_env_value, print_success,
        print_warning, prompt, prompt_choice, save_env_value,
    )
    selected = "xai"
    if _xai_oauth_logged_in_for_setup():
        print_success("xAI TTS will use your xAI Grok OAuth (SuperGrok / Premium+) credentials")
    elif get_env_value("XAI_API_KEY"):
        print_success("xAI TTS will use your existing XAI_API_KEY")
    else:
        print()
        choice_idx = prompt_choice(
            "How do you want xAI TTS to authenticate?",
            choices=[
                "Sign in with xAI Grok OAuth (SuperGrok / Premium+) — browser login",
                "Paste an xAI API key (console.x.ai)", "Skip → fallback to Edge TTS",
            ],
            default=0,
        )
        if choice_idx == 0:
            if _run_xai_oauth_login_from_setup():
                print_success("Logged in — xAI TTS will use these OAuth credentials")
            else:
                print_warning("xAI Grok OAuth login did not complete. Falling back to Edge TTS.")
                selected = "edge"
        elif choice_idx == 1:
            api_key = prompt("xAI API key for TTS", password=True)
            if api_key:
                save_env_value("XAI_API_KEY", api_key)
                print_success("xAI TTS API key saved")
            else:
                from hermes_constants import display_hermes_home as _dhh
                print_warning(
                    "No xAI API key provided for TTS. Configure XAI_API_KEY "
                    f"via hermes setup model or {_dhh()}/.env to use xAI TTS. "
                    "Falling back to Edge TTS."
                )
                selected = "edge"
        else:
            print_warning("xAI TTS skipped. Falling back to Edge TTS.")
            selected = "edge"

    if selected == "xai":
        print()
        voice_id = prompt("xAI voice_id (Enter for 'eve', or paste a custom voice ID)")
        if voice_id and voice_id.strip():
            config.setdefault("tts", {}).setdefault("xai", {})["voice_id"] = voice_id.strip()
            print_success(f"xAI voice_id set to: {voice_id.strip()}")
    return selected


def _setup_tts_provider(config: dict):
    """Interactive TTS provider selection with install flow for local engines."""
    from hermes_cli.setup import (
        get_env_value, get_nous_subscription_features, _info, managed_nous_tools_enabled, print_header,
        print_info, print_success, print_warning, prompt_choice, save_config,
    )
    current_provider = config.get("tts", {}).get("provider", "edge")
    current_label = _TTS_PROVIDER_LABELS.get(current_provider, current_provider)

    print()
    print_header("Text-to-Speech Provider (optional)")
    _info(f"Current: {current_label}", None)

    options = list(_TTS_PROVIDER_CHOICES)
    if managed_nous_tools_enabled() and get_nous_subscription_features(config).nous_auth_present:
        options.insert(0, ("nous-openai", "Nous Subscription (managed OpenAI TTS, billed to your subscription)"))
    choices = [label for _, label in options] + [f"Keep current ({current_label})"]
    keep_current_idx = len(choices) - 1
    idx = prompt_choice("Select TTS provider:", choices, keep_current_idx)
    if idx == keep_current_idx:
        return

    selected = options[idx][0]
    selected_via_nous = selected == "nous-openai"
    if selected_via_nous:
        selected = "openai"
        print_info("OpenAI TTS will use the managed Nous gateway and bill to your subscription.")
        if get_env_value("VOICE_TOOLS_OPENAI_KEY") or get_env_value("OPENAI_API_KEY"):
            print_warning(
                "Direct OpenAI credentials are still configured and may take precedence until removed from ~/.hermes/.env."
            )

    if selected in _TTS_LOCAL_PROVIDERS:
        selected = _tts_local_install_step(selected)
    elif selected in _TTS_API_KEY_PROVIDERS and not selected_via_nous:
        selected = _tts_api_key_step(selected)
    elif selected == "xai":
        selected = _tts_xai_step(config)

    config.setdefault("tts", {})["provider"] = selected
    save_config(config)
    print_success(f"TTS provider set to: {_TTS_PROVIDER_LABELS.get(selected, selected)}")


def setup_tts(config: dict):
    """Standalone TTS setup (for 'hermes setup tts')."""
    _setup_tts_provider(config)


