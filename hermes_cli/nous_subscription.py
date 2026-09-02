"""Helpers for Nous subscription managed-tool capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Set

from hermes_cli.config import get_env_value, load_config
from hermes_cli.nous_account import (
    NousPortalAccountInfo,
    format_nous_portal_entitlement_message,
    get_nous_portal_account_info,
)
from tools.managed_tool_gateway import is_managed_tool_gateway_ready
from utils import is_truthy_value
from tools.tool_backend_helpers import (
    fal_key_is_configured,
    has_direct_modal_credentials,
    managed_nous_tools_enabled,  # noqa: F401  (test-patchable re-export)
    normalize_browser_cloud_provider,
    normalize_modal_mode,
    resolve_modal_backend_state,
    resolve_openai_audio_api_key,
)


_DEFAULT_PLATFORM_TOOLSETS = {
    "cli": "hermes-cli",
}


@dataclass(frozen=True)
class _FeatureSpec:
    """Per-feature parameters shared by the status, defaults and Tool Gateway offer surfaces."""

    label: str
    included_by_default: bool
    # Tool-pool coverage category (hermes_cli.nous_account.TOOL_COVERAGE_CATEGORIES). Lets the
    # `hermes tools` picker scope its entitlement gate to the selected backend, so a free-tool-pool
    # user is allowed image gen but denied video gen at select time — consistent with the
    # per-category feature gates in get_nous_subscription_features. STT shares the TTS category:
    # both ride the managed "openai-audio" gateway endpoint (speech + transcriptions).
    coverage: str
    # Managed gateway probed for readiness. Video rides image's fal-queue gateway but is gated on
    # its own coverage category: the free tool pool funds image and NOT video (paid users get both).
    gateway: str
    # Config section + selection field written by apply_gateway_defaults and read by
    # get_nous_subscription_features (web uses "backend", browser "cloud_provider", else "provider").
    # None = not offered by the Tool Gateway prompt (modal).
    section_field: Optional[tuple[str, str]] = None
    offer_label: str = ""
    direct_label: str = ""
    # Direct-credential env vars that keep apply_nous_managed_defaults from switching the
    # category to the managed selection (tts/stt also honour resolve_openai_audio_api_key()).
    default_direct_env: tuple[str, ...] = ()


_FEATURES: Dict[str, _FeatureSpec] = {
    "web": _FeatureSpec(
        "Web tools", True, "firecrawl", "firecrawl", ("web", "backend"),
        "Web search & extract (Firecrawl)", "Firecrawl/Exa/Parallel/Keenable key or SearXNG",
        ("PARALLEL_API_KEY", "TAVILY_API_KEY", "FIRECRAWL_API_KEY", "FIRECRAWL_API_URL"),
    ),
    "image_gen": _FeatureSpec(
        "Image generation", True, "fal", "fal-queue", ("image_gen", "provider"),
        "Image generation (FAL)", "FAL key",
    ),
    "video_gen": _FeatureSpec(
        "Video generation", False, "fal-video", "fal-queue", ("video_gen", "provider"),
        "Video generation (FAL)", "FAL key",
    ),
    "tts": _FeatureSpec(
        "OpenAI TTS", True, "openai-audio", "openai-audio", ("tts", "provider"),
        "Text-to-speech (OpenAI TTS)", "OpenAI/ElevenLabs key", ("ELEVENLABS_API_KEY",),
    ),
    "stt": _FeatureSpec(
        "Speech-to-text", True, "openai-audio", "openai-audio", ("stt", "provider"),
        "Speech-to-text (OpenAI Whisper)", "OpenAI/Groq/Mistral key", ("GROQ_API_KEY", "MISTRAL_API_KEY"),
    ),
    "browser": _FeatureSpec(
        "Browser automation", True, "browser-use", "browser-use", ("browser", "cloud_provider"),
        "Browser automation (Browser Use)", "Browser Use/Browserbase key or Camofox",
        ("BROWSER_USE_API_KEY", "BROWSERBASE_API_KEY"),
    ),
    "modal": _FeatureSpec("Modal execution", False, "modal", "modal"),
}

_FEATURE_ORDER = tuple(_FEATURES)
# Public / test-referenced views over the table.
MANAGED_FEATURE_COVERAGE_CATEGORY: Dict[str, str] = {k: s.coverage for k, s in _FEATURES.items()}
_GATEWAY_SECTION_FIELDS = {k: s.section_field for k, s in _FEATURES.items() if s.section_field}
_ALL_GATEWAY_KEYS = tuple(_GATEWAY_SECTION_FIELDS)
_GATEWAY_TOOL_LABELS = {k: _FEATURES[k].offer_label for k in _ALL_GATEWAY_KEYS}


def _uses_gateway(section: object) -> bool:
    """Return True when a config section explicitly opts into the gateway."""
    if not isinstance(section, dict):
        return False
    return is_truthy_value(section.get("use_gateway"), default=False)


def _selected_provider(section: object, name_key: str = "provider") -> Optional[str]:
    """Return the stored provider string for a config section dict.

    Mirrors :func:`tools.tool_backend_helpers.read_selection`'s semantics on an in-memory section
    dict: ``"nous"`` for the managed selection (stored ``nous`` value or legacy ``use_gateway:
    true``), a vendor name for BYOK picks, or ``None`` when no selection is stored.
    """
    if not isinstance(section, dict):
        return None
    if is_truthy_value(section.get("use_gateway"), default=False):
        return "nous"
    value = section.get(name_key)
    if value is None:
        return None
    name = str(value).strip().lower()
    return name or None


@dataclass(frozen=True)
class NousFeatureState:
    key: str
    label: str
    included_by_default: bool
    available: bool
    active: bool
    managed_by_nous: bool
    direct_override: bool
    toolset_enabled: bool
    current_provider: str = ""
    explicit_configured: bool = False


@dataclass(frozen=True)
class NousSubscriptionFeatures:
    subscribed: bool
    nous_auth_present: bool
    provider_is_nous: bool
    features: Dict[str, NousFeatureState]
    account_info: Optional[NousPortalAccountInfo] = None

    def __getattr__(self, name: str) -> NousFeatureState:
        # ``features.web`` / ``features.tts`` … resolve to the per-key state.
        if name in _FEATURE_ORDER:
            return self.features[name]
        raise AttributeError(name)

    def items(self) -> Iterable[NousFeatureState]:
        for key in _FEATURE_ORDER:
            yield self.features[key]


def _section(config: Dict[str, object], key: str) -> Dict[str, object]:
    """Return ``config[key]`` when it is a dict, else ``{}`` (read-only view)."""
    value = config.get(key)
    return value if isinstance(value, dict) else {}


def _ensure_section(config: Dict[str, object], key: str) -> Dict[str, object]:
    """Return ``config[key]`` as a dict, creating/replacing it in ``config`` when missing."""
    value = config.get(key)
    if not isinstance(value, dict):
        value = {}
        config[key] = value
    return value


def _select_nous(config: Dict[str, object], key: str) -> None:
    """Store the managed ``nous`` selection in the ``key`` section (field per _GATEWAY_SECTION_FIELDS)."""
    section_key, field = _GATEWAY_SECTION_FIELDS[key]
    section = _ensure_section(config, section_key)
    section[field] = "nous"
    section.pop("use_gateway", None)


def _model_config_dict(config: Dict[str, object]) -> Dict[str, object]:
    model_cfg = config.get("model")
    if isinstance(model_cfg, dict):
        return dict(model_cfg)
    if isinstance(model_cfg, str) and model_cfg.strip():
        return {"default": model_cfg.strip()}
    return {}


def _toolset_enabled(config: Dict[str, object], toolset_key: str) -> bool:
    from toolsets import resolve_toolset

    platform_toolsets = config.get("platform_toolsets")
    if not isinstance(platform_toolsets, dict) or not platform_toolsets:
        platform_toolsets = {"cli": [_DEFAULT_PLATFORM_TOOLSETS["cli"]]}

    target_tools = set(resolve_toolset(toolset_key))
    if not target_tools:
        return False

    for platform, raw_toolsets in platform_toolsets.items():
        toolset_names = list(raw_toolsets) if isinstance(raw_toolsets, list) else []
        if not toolset_names:
            default_toolset = _DEFAULT_PLATFORM_TOOLSETS.get(platform)
            toolset_names = [default_toolset] if default_toolset else []

        available_tools: Set[str] = set()
        for toolset_name in toolset_names:
            if isinstance(toolset_name, str) and toolset_name:
                try:
                    available_tools.update(resolve_toolset(toolset_name))
                except Exception:
                    continue

        if target_tools.issubset(available_tools):
            return True

    return False


def _has_agent_browser() -> bool:
    import shutil

    from hermes_constants import agent_browser_runnable

    # agent-browser is no longer a root package.json dependency (#43564) — it
    # resolves lazily via npx for most installs, which a bare PATH +
    # node_modules probe can't see. Mirror the local-CLI tail of
    # :func:`tools.browser_tool.check_browser_requirements` (same cascade, same
    # Termux carve-out) so the setup/status surfaces can't diverge from what
    # browser tools actually find at runtime; validate=False keeps this a cheap
    # existence check with no subprocess spawn.
    try:
        from tools.browser_tool import (
            _find_agent_browser,
            _requires_real_termux_browser_install,
        )
    except Exception:
        # If the runtime probe can't be imported, fall back to binary presence
        # (prior behaviour) rather than crashing the setup/status surface.
        # Validate the resolved binary actually runs — a dangling global
        # symlink (issue #48521) is reported by ``which`` but fails at exec.
        from hermes_constants import with_hermes_node_path

        # Rungs: PATH; Hermes-managed Node dirs (Windows installer / POSIX
        # $HERMES_HOME/node — prepended to PATH at runtime but usually absent
        # from the *probe* process's PATH, without which a successful install
        # keeps reporting "needs setup" on Windows); local node_modules/.bin
        # (PATHEXT-aware ``shutil.which`` so Windows picks the executable
        # ``.cmd`` shim — probing the extensionless POSIX shim directly fails
        # exec (WinError 193) even right after a successful ``npm install``).
        local_bin_dir = Path(__file__).parent.parent / "node_modules" / ".bin"
        search_paths = [
            None,
            with_hermes_node_path().get("PATH", ""),
            str(local_bin_dir) if local_bin_dir.is_dir() else "",
        ]
        for path in search_paths:
            if path == "":
                continue
            hit = shutil.which("agent-browser") if path is None else shutil.which("agent-browser", path=path)
            if hit and agent_browser_runnable(hit):
                return True
        return False

    try:
        browser_cmd = _find_agent_browser(validate=False)
    except FileNotFoundError:
        return False
    # On Termux, the bare npx fallback is too fragile to advertise as ready —
    # require a real install, matching check_browser_requirements.
    return not _requires_real_termux_browser_install(browser_cmd)


def _local_browser_runnable() -> bool:
    """Return True when the *local* browser backend would actually start.

    The ``agent-browser`` CLI being present is necessary but not sufficient for local mode: agent-
    browser also needs a Chromium build on disk (without one it hangs on first use until the command
    timeout fires), unless the Lightpanda engine is selected — text-only navigation needs no
    Chromium.

    This mirrors the local-mode tail of :func:`tools.browser_tool.check_browser_requirements`, so
    the setup/status surfaces advertise local browser readiness only when the runtime would actually
    run it.
    """
    if not _has_agent_browser():
        return False
    try:
        from tools.browser_tool import _chromium_installed, _using_lightpanda_engine
    except Exception:
        # If the runtime probe can't be imported, fall back to binary presence
        # (prior behaviour) rather than crashing the setup/status surface.
        return True
    return _using_lightpanda_engine() or _chromium_installed()


_PROVIDER_LABELS = {
    "browser": ("local", {
        "browserbase": "Browserbase",
        "browser-use": "Browser Use",
        "firecrawl": "Firecrawl",
        "camofox": "Camofox",
        "local": "Local browser",
    }),
    "tts": ("edge", {
        "openai": "OpenAI TTS",
        "elevenlabs": "ElevenLabs",
        "edge": "Edge TTS",
        "xai": "xAI TTS",
        "mistral": "Mistral Voxtral TTS",
        "neutts": "NeuTTS",
    }),
    "stt": ("local", {
        "openai": "OpenAI Whisper",
        "groq": "Groq Whisper",
        "mistral": "Mistral Voxtral Transcribe",
        "local": "Local faster-whisper",
    }),
}


def _provider_label(kind: str, current_provider: str) -> str:
    default, mapping = _PROVIDER_LABELS[kind]
    return mapping.get(current_provider or default, current_provider or mapping[default])


def _local_stt_backend_available() -> bool:
    """Whether a local STT backend could serve transcription right now.

    True when faster-whisper imports or a custom local STT command is configured. Also stops
    ``apply_nous_managed_defaults`` from flipping a working local setup to the managed gateway.
    """
    if get_env_value("HERMES_LOCAL_STT_COMMAND"):
        return True
    try:
        from tools.transcription_tools import _HAS_FASTER_WHISPER

        return bool(_HAS_FASTER_WHISPER)
    except Exception:
        return False


def _resolve_browser_feature_state(
    *,
    browser_tool_enabled: bool,
    browser_provider: str,
    browser_provider_explicit: bool,
    browser_local_available: bool,
    browser_local_runnable: bool,
    direct_camofox: bool,
    direct_browserbase: bool,
    direct_browser_use: bool,
    direct_firecrawl: bool,
    managed_browser_available: bool,
) -> tuple[str, bool, bool, bool]:
    """Resolve browser availability using the same precedence as runtime.

    ``browser_local_available`` means "the agent-browser CLI is present" — the only local
    requirement for cloud providers, which host their own Chromium.
    """
    browser_use_managed = bool(
        browser_tool_enabled
        and browser_local_available
        and managed_browser_available
        and not direct_browser_use
    )
    if browser_provider_explicit:
        # Camofox is a stored selection (browser.cloud_provider: camofox);
        # CAMOFOX_URL is only the server address.
        cloud_available = {
            "camofox": direct_camofox,
            "browserbase": browser_local_available and direct_browserbase,
            "browser-use": browser_local_available and (managed_browser_available or direct_browser_use),
            "firecrawl": browser_local_available and direct_firecrawl,
        }
        current_provider = browser_provider or "local"
        if current_provider not in cloud_available:
            current_provider = "local"
        available = bool(cloud_available.get(current_provider, browser_local_runnable))
        managed = browser_use_managed if current_provider == "browser-use" else False
    # Never-configured autodetect: CAMOFOX_URL keeps activating Camofox
    # exactly as before when no cloud_provider selection was ever stored.
    elif direct_camofox:
        return "camofox", True, bool(browser_tool_enabled), False
    elif managed_browser_available or direct_browser_use:
        current_provider, available, managed = "browser-use", bool(browser_local_available), browser_use_managed
    elif direct_browserbase:
        current_provider, available, managed = "browserbase", bool(browser_local_available), False
    else:
        current_provider, available, managed = "local", bool(browser_local_runnable), False
    return current_provider, available, bool(browser_tool_enabled and available), managed


def _any_env(*names: str) -> bool:
    """True when any of the named env vars (via get_env_value) is set."""
    return any(get_env_value(name) for name in names)


def _fal_provider_label(selected: Optional[str], direct_fal: bool, managed: bool) -> str:
    if selected not in (None, "nous") or (selected is None and direct_fal):
        return "FAL"
    return "Nous Subscription" if (managed or selected == "nous") else ""


def _account_info_or_none(**kwargs) -> Optional[NousPortalAccountInfo]:
    """``get_nous_portal_account_info(**kwargs)``, failing closed to ``None`` on any error."""
    try:
        return get_nous_portal_account_info(**kwargs)
    except Exception:
        return None


def get_nous_subscription_features(
    config: Optional[Dict[str, object]] = None,
    *,
    force_fresh: bool = False,
) -> NousSubscriptionFeatures:
    if config is None:
        config = load_config() or {}
    config = dict(config)
    model_cfg = _model_config_dict(config)
    provider_is_nous = str(model_cfg.get("provider") or "").strip().lower() == "nous"

    account_info = _account_info_or_none(**({"force_fresh": True} if force_fresh else {}))

    # Coarse "entitled to any managed tool" gate: paid access OR a live free
    # tool pool. Per-backend availability is then narrowed by coverage below
    # (the pool funds image but not video, etc.).
    managed_tools_flag = bool(
        account_info
        and account_info.logged_in
        and account_info.tool_gateway_entitled
    )
    nous_auth_present = bool(account_info and account_info.logged_in)
    subscribed = provider_is_nous or nous_auth_present

    enabled = {key: _toolset_enabled(config, key) for key in ("web", "image_gen", "video_gen", "tts", "browser", "terminal")}
    web_tool_enabled, image_tool_enabled, video_tool_enabled = enabled["web"], enabled["image_gen"], enabled["video_gen"]
    tts_tool_enabled, browser_tool_enabled, modal_tool_enabled = enabled["tts"], enabled["browser"], enabled["terminal"]

    web_cfg, tts_cfg, stt_cfg, browser_cfg, terminal_cfg = (
        _section(config, key) for key in ("web", "tts", "stt", "browser", "terminal")
    )

    web_backend = str(web_cfg.get("backend") or "").strip().lower()
    # Per-capability overrides: if set, they determine which backend is active for
    # search/extract independently of web.backend.
    web_search_backend = str(web_cfg.get("search_backend") or "").strip().lower()
    web_extract_backend = str(web_cfg.get("extract_backend") or "").strip().lower()
    tts_provider = str(tts_cfg.get("provider") or "edge").strip().lower()
    # STT default is "local" (faster-whisper) per DEFAULT_CONFIG, which
    # requires `pip install faster-whisper`. For Nous subscribers we'd
    # rather route through the managed OpenAI audio gateway — see
    # apply_nous_managed_defaults below.
    stt_provider = str(stt_cfg.get("provider") or "local").strip().lower()
    browser_provider_explicit = "cloud_provider" in browser_cfg
    browser_provider = normalize_browser_cloud_provider(
        browser_cfg.get("cloud_provider") if browser_provider_explicit else None
    )
    terminal_backend = str(terminal_cfg.get("backend") or "local").strip().lower()
    modal_mode = normalize_modal_mode(terminal_cfg.get("modal_mode"))

    # Stored selections (strict model): one provider string per category.
    # "nous" (stored value or legacy use_gateway: true) = managed gateway;
    # vendor name = that vendor direct; None = never configured (autodetect).
    # Lockstep with tools.tool_backend_helpers.read_selection: these are
    # merged-config sections, so the legacy DEFAULT_CONFIG-seeded
    # ``stt.provider: local`` COULD appear here without a user pick on old
    # versions. Current DEFAULT_CONFIG no longer seeds it, so a merged
    # ``local`` implies the raw file holds it — a genuine selection.
    selected = {
        key: _selected_provider(_section(config, section_key), field)
        for key, (section_key, field) in _GATEWAY_SECTION_FIELDS.items()
    }
    # Managed selection flags (use_gateway is interpreted only inside _selected_provider).
    use_gateway = {key: value == "nous" for key, value in selected.items()}
    web_gw, image_gw, video_gw = use_gateway["web"], use_gateway["image_gen"], use_gateway["video_gen"]
    tts_gw, stt_gw, browser_gw = use_gateway["tts"], use_gateway["stt"], use_gateway["browser"]

    # The "nous" selection is serviced by a concrete vendor implementation —
    # normalize the current-provider labels so downstream vendor checks hold.
    if web_backend == "nous" or web_gw:
        web_backend = "firecrawl"
    if tts_provider == "nous" or tts_gw:
        tts_provider = "openai"
    if stt_provider == "nous" or stt_gw:
        stt_provider = "openai"
    if browser_provider == "nous" or browser_gw:
        browser_provider = "browser-use"

    # Direct credentials. When the managed selection is stored for a category,
    # its direct credentials are suppressed for managed detection.
    direct_exa = _any_env("EXA_API_KEY") and not web_gw
    direct_firecrawl = _any_env("FIRECRAWL_API_KEY", "FIRECRAWL_API_URL") and not web_gw
    direct_parallel = _any_env("PARALLEL_API_KEY") and not web_gw
    direct_tavily = _any_env("TAVILY_API_KEY") and not web_gw
    # Keyless Tavily is opt-in: selecting it in `hermes tools` / setup writes
    # web.backend (or a per-capability override) without requiring a key.
    tavily_selected = "tavily" in {web_backend, web_search_backend, web_extract_backend} and not web_gw
    direct_searxng = _any_env("SEARXNG_URL")
    fal_configured = fal_key_is_configured()
    direct_fal = fal_configured and not image_gw
    direct_fal_video = fal_configured and not video_gw  # same FAL_KEY; separate var so use_gateway is independent
    # OpenAI Whisper reuses the same audio key as OpenAI TTS —
    # resolve_openai_audio_api_key() reads VOICE_TOOLS_OPENAI_KEY and falls
    # back to OPENAI_API_KEY.
    audio_key = bool(resolve_openai_audio_api_key())
    direct_openai_tts = audio_key and not tts_gw
    direct_elevenlabs = _any_env("ELEVENLABS_API_KEY") and not tts_gw
    direct_camofox = _any_env("CAMOFOX_URL")
    direct_browserbase = (
        bool(get_env_value("BROWSERBASE_API_KEY") and get_env_value("BROWSERBASE_PROJECT_ID")) and not browser_gw
    )
    direct_browser_use = _any_env("BROWSER_USE_API_KEY") and not browser_gw
    direct_modal = has_direct_modal_credentials()

    # STT direct providers. The local provider's "direct" signal is whether
    # faster-whisper is importable (lazy-imported so this module stays cheap).
    direct_openai_stt = audio_key and not stt_gw
    direct_groq_stt = _any_env("GROQ_API_KEY") and not stt_gw
    direct_mistral_stt = _any_env("MISTRAL_API_KEY") and not stt_gw
    local_stt_available = _local_stt_backend_available() and not stt_gw

    # Managed availability per feature. Strict selection: a stored VENDOR
    # selection pins the category to direct credentials — managed availability
    # must not light the feature up (the runtime will error, not reroute).
    managed = {
        key: (
            managed_tools_flag
            and nous_auth_present
            and is_managed_tool_gateway_ready(_FEATURES[key].gateway)
            and bool(account_info and account_info.tool_gateway_entitled_for(_FEATURES[key].coverage))
            and (selected.get(key) is None or use_gateway[key])
        )
        for key in _FEATURE_ORDER
    }
    modal_state = resolve_modal_backend_state(
        modal_mode,
        has_direct=direct_modal,
        managed_ready=managed["modal"],
        managed_enabled=managed_tools_flag,
    )
    if selected["browser"] is not None and selected["browser"] != "camofox":
        # CAMOFOX_URL is the server address, not a selection: an explicit
        # different browser choice wins over the env var.
        direct_camofox = False

    # Direct web readiness per vendor. web.backend and the per-capability
    # overrides (search_backend / extract_backend, split config from #20061)
    # may each name a vendor; extract_backend only supports tavily.
    web_direct = {
        "exa": direct_exa,
        "firecrawl": direct_firecrawl,
        "parallel": direct_parallel,
        "tavily": direct_tavily or tavily_selected,
        "searxng": direct_searxng,
    }
    web_managed = web_backend == "firecrawl" and managed["web"] and not direct_firecrawl
    web_active = bool(
        web_tool_enabled
        and (
            web_managed
            or web_direct.get(web_backend)
            or web_direct.get(web_search_backend)
            or (web_extract_backend == "tavily" and web_direct["tavily"])
        )
    )
    web_available = bool(managed["web"] or any(web_direct.values()))

    tts_current_provider = tts_provider or "edge"
    tts_managed = (
        tts_tool_enabled
        and tts_current_provider == "openai"
        and managed["tts"]
        and not direct_openai_tts
    )
    tts_available = bool({
        "edge": True,
        "neutts": True,
        "openai": managed["tts"] or direct_openai_tts,
        "elevenlabs": direct_elevenlabs,
        "mistral": _any_env("MISTRAL_API_KEY"),
    }.get(tts_current_provider, False))
    tts_active = bool(tts_tool_enabled and tts_available)

    # STT availability per provider. Unlike TTS, STT isn't a model-callable
    # tool — the gateway voice middleware calls it on every inbound voice
    # message — so toolset_enabled is N/A and we treat stt as always
    # "enabled" if a usable provider is configured.
    stt_current_provider = stt_provider or "local"
    stt_managed = (
        stt_current_provider == "openai"
        and managed["stt"]
        and not direct_openai_stt
    )
    stt_available = bool({
        "local": local_stt_available,
        "openai": managed["stt"] or direct_openai_stt,
        "groq": direct_groq_stt,
        "mistral": direct_mistral_stt,
    }.get(stt_current_provider, False))

    browser_local_available = _has_agent_browser()
    (
        browser_current_provider,
        browser_available,
        browser_active,
        browser_managed,
    ) = _resolve_browser_feature_state(
        browser_tool_enabled=browser_tool_enabled,
        browser_provider=browser_provider,
        browser_provider_explicit=browser_provider_explicit,
        browser_local_available=browser_local_available,
        browser_local_runnable=_local_browser_runnable(),
        direct_camofox=direct_camofox,
        direct_browserbase=direct_browserbase,
        direct_browser_use=direct_browser_use,
        direct_firecrawl=direct_firecrawl,
        managed_browser_available=managed["browser"],
    )

    # Modal: a non-modal terminal backend, or a resolved managed/direct
    # selection, is always "available"; otherwise report what the mode could use.
    modal_selected = modal_state["selected_backend"] if terminal_backend == "modal" else None
    if terminal_backend != "modal" or modal_selected in ("managed", "direct"):
        modal_available, modal_active = True, bool(modal_tool_enabled)
        modal_managed = modal_selected == "managed" and bool(modal_tool_enabled)
        modal_direct_override = modal_selected == "direct" and bool(modal_tool_enabled)
    else:
        modal_managed = modal_direct_override = modal_active = False
        modal_available = bool(
            {"managed": managed["modal"], "direct": direct_modal}.get(
                modal_mode, managed["modal"] or direct_modal
            )
        )

    def _state(key: str, **fields) -> NousFeatureState:
        spec = _FEATURES[key]
        fields.setdefault("direct_override", fields["active"] and not fields["managed_by_nous"])
        return NousFeatureState(key=key, label=spec.label, included_by_default=spec.included_by_default, **fields)

    def _fal_state(key: str, tool_enabled: bool, direct: bool) -> NousFeatureState:
        # image_gen / video_gen: same FAL_KEY, independently gated managed availability.
        fal_managed = tool_enabled and managed[key] and not direct
        return _state(
            key,
            available=bool(managed[key] or direct),
            active=bool(tool_enabled and (fal_managed or direct)),
            managed_by_nous=fal_managed,
            toolset_enabled=tool_enabled,
            current_provider=_fal_provider_label(selected[key], direct, fal_managed),
            explicit_configured=selected[key] is not None or direct,
        )

    features = {
        "web": _state(
            "web",
            available=web_available,
            active=web_active,
            managed_by_nous=web_managed,
            toolset_enabled=web_tool_enabled,
            current_provider=web_backend or web_search_backend or web_extract_backend or "",
            explicit_configured=bool(web_backend or web_search_backend or web_extract_backend),
        ),
        "image_gen": _fal_state("image_gen", image_tool_enabled, direct_fal),
        "video_gen": _fal_state("video_gen", video_tool_enabled, direct_fal_video),
        "tts": _state(
            "tts",
            available=tts_available,
            active=tts_active,
            managed_by_nous=tts_managed,
            toolset_enabled=tts_tool_enabled,
            current_provider=_provider_label("tts", tts_current_provider),
            # Explicit-configured mirrors the stored selections so status/picker
            # markers stay in lockstep with runtime dispatch.
            explicit_configured=selected["tts"] is not None and selected["tts"] != "edge",
        ),
        "stt": _state(
            "stt",
            available=stt_available,
            active=stt_available,
            managed_by_nous=stt_managed,
            # STT isn't toolset-gated (gateway middleware calls it
            # unconditionally on inbound voice), so report True so the
            # status display doesn't flag it as "tool disabled".
            toolset_enabled=True,
            current_provider=_provider_label("stt", stt_current_provider),
            explicit_configured=selected["stt"] is not None,
        ),
        "browser": _state(
            "browser",
            available=browser_available,
            active=browser_active,
            managed_by_nous=browser_managed,
            toolset_enabled=browser_tool_enabled,
            current_provider=_provider_label("browser", browser_current_provider),
            explicit_configured=browser_provider_explicit,
        ),
        "modal": _state(
            "modal",
            available=modal_available,
            active=modal_active,
            managed_by_nous=modal_managed,
            direct_override=terminal_backend == "modal" and modal_direct_override,
            toolset_enabled=modal_tool_enabled,
            current_provider="Modal" if terminal_backend == "modal" else terminal_backend or "local",
            explicit_configured=terminal_backend == "modal",
        ),
    }

    return NousSubscriptionFeatures(
        subscribed=subscribed,
        nous_auth_present=nous_auth_present,
        provider_is_nous=provider_is_nous,
        features=features,
        account_info=account_info,
    )


def _has_managed_default_direct(key: str) -> bool:
    if key in ("tts", "stt") and resolve_openai_audio_api_key():
        return True
    return _any_env(*_FEATURES[key].default_direct_env)


def apply_nous_managed_defaults(
    config: Dict[str, object],
    *,
    enabled_toolsets: Optional[Iterable[str]] = None,
    force_fresh: bool = False,
) -> set[str]:
    features = get_nous_subscription_features(config, force_fresh=force_fresh)
    account_info = features.account_info
    if not (
        account_info
        and account_info.logged_in
        and account_info.tool_gateway_entitled
        and features.provider_is_nous
    ):
        return set()

    selected_toolsets = set(enabled_toolsets or ())
    changed: set[str] = set()

    for key in ("web", "tts", "stt", "browser"):
        _ensure_section(config, key)

    for key in ("web", "tts", "stt", "browser"):
        if features.features[key].explicit_configured or _has_managed_default_direct(key):
            continue
        if key == "stt":
            # STT: same pattern as TTS. The DEFAULT_CONFIG seed is "local"
            # (requires `pip install faster-whisper`); for Nous subscribers we
            # flip it to the managed selection so the managed audio gateway handles
            # transcription via the same auth as TTS. Not toolset-gated. Skipped when
            # the user has a working local backend (faster-whisper installed or a
            # custom local command — strong intent signal that "local" was a choice,
            # not just the DEFAULT_CONFIG seed), or isn't entitled to the managed
            # "openai-audio" category (flipping would point at a gateway that
            # refuses them, silently breaking voice transcription).
            if _local_stt_backend_available() or not (
                account_info is not None and account_info.tool_gateway_entitled_for("openai-audio")
            ):
                continue
        elif key not in selected_toolsets:
            continue
        _select_nous(config, key)
        changed.add(key)

    # Video gen is not funded by the free tool pool, so only wire managed video
    # defaults for users entitled to it (paid). Pool-only users keep video off.
    for key, category in (("image_gen", None), ("video_gen", "fal-video")):
        if (
            key in selected_toolsets
            and not fal_key_is_configured()
            and (category is None or account_info.tool_gateway_entitled_for(category))
        ):
            _select_nous(config, key)
            changed.add(key)

    return changed


# ---------------------------------------------------------------------------
# Tool Gateway offer — single Y/n prompt after model selection
# ---------------------------------------------------------------------------


def _get_gateway_direct_credentials() -> Dict[str, bool]:
    """Return a dict of tool_key -> has_direct_credentials.

    Env-configured keyless local backends count as configured: a reachable self-hosted SearXNG
    (autodetected by tools/web_tools.py) or CAMOFOX_URL (never-configured autodetect in
    _resolve_browser_feature_state) is a working setup even with no stored selection, so it must
    not be classified "unconfigured" and pre-checked (#92647). OpenAI Whisper shares the audio key
    with TTS via resolve_openai_audio_api_key(), so it counts for both tts and stt.
    """
    fal_direct = fal_key_is_configured()
    audio_direct = bool(resolve_openai_audio_api_key())
    return {
        "web": _any_env(
            "FIRECRAWL_API_KEY", "FIRECRAWL_API_URL", "PARALLEL_API_KEY",
            "TAVILY_API_KEY", "EXA_API_KEY", "SEARXNG_URL",
        ),
        "image_gen": fal_direct,
        "video_gen": fal_direct,
        "tts": audio_direct or _any_env("ELEVENLABS_API_KEY"),
        "stt": audio_direct or _any_env("GROQ_API_KEY", "MISTRAL_API_KEY"),
        "browser": (
            _any_env("BROWSER_USE_API_KEY", "CAMOFOX_URL")
            or bool(get_env_value("BROWSERBASE_API_KEY") and get_env_value("BROWSERBASE_PROJECT_ID"))
        ),
    }


def get_gateway_eligible_tools(
    config: Optional[Dict[str, object]] = None,
    *,
    force_fresh: bool = False,
) -> tuple[list[str], list[str], list[str], list[str]]:
    """Return (unconfigured, has_direct, explicit_configured, already_managed) tool key lists.

    - unconfigured: tools with no direct credentials and no explicit non-nous selection (easy
    switch, safe to pre-check) - has_direct: tools where the user has their own API keys -
    explicit_configured: tools with an explicit non-nous selection stored (e.g.
    """
    # Fetch entitlement once: it gates the offer (paid access OR a live free tool
    # pool) AND tells us which categories are covered (the pool funds image but
    # not video, etc.). Fails closed on any error.
    account_info = _account_info_or_none(force_fresh=force_fresh)
    if not (account_info and account_info.logged_in and account_info.tool_gateway_entitled):
        return [], [], [], []

    if config is None:
        config = load_config() or {}

    # Quick provider check without the heavy get_nous_subscription_features call
    model_cfg = config.get("model")
    if not isinstance(model_cfg, dict) or str(model_cfg.get("provider") or "").strip().lower() != "nous":
        return [], [], [], []

    direct = _get_gateway_direct_credentials()

    # Buckets: already_managed = use_gateway explicitly set (distinct from
    # managed_by_nous, which fires implicitly when no direct keys exist);
    # explicit_configured = an explicit non-nous selection (e.g. a keyless local
    # backend like SearXNG or Camofox) configured on purpose even though it has
    # no direct credentials to detect.
    buckets: Dict[str, list[str]] = {
        "unconfigured": [], "has_direct": [], "explicit_configured": [], "already_managed": [],
    }
    for key in _ALL_GATEWAY_KEYS:
        # Only offer tools the user's entitlement actually covers. For a free
        # tool pool that means image but not video; paid users are covered for
        # everything.
        if not account_info.tool_gateway_entitled_for(_FEATURES[key].coverage):
            continue
        section_key, field = _GATEWAY_SECTION_FIELDS[key]
        selected = _selected_provider(config.get(section_key), field)
        if _uses_gateway(config.get(key)):
            bucket = "already_managed"
        elif selected is not None and selected != "nous":
            bucket = "explicit_configured"
        elif direct.get(key):
            bucket = "has_direct"
        else:
            bucket = "unconfigured"
        buckets[bucket].append(key)
    return (
        buckets["unconfigured"], buckets["has_direct"],
        buckets["explicit_configured"], buckets["already_managed"],
    )


def apply_gateway_defaults(
    config: Dict[str, object],
    tool_keys: list[str],
) -> set[str]:
    """Apply Tool Gateway config for the given tool keys.

    Sets ``use_gateway: true`` in each tool's section so the runtime prefers the gateway even when
    direct API keys are present. Returns the set of tools actually changed.
    """
    changed: set[str] = set()

    for key in ("web", "tts", "stt", "browser"):
        _ensure_section(config, key)

    for key in _ALL_GATEWAY_KEYS:
        if key in tool_keys:
            _select_nous(config, key)
            changed.add(key)

    return changed


def prompt_enable_tool_gateway(
    config: Dict[str, object],
    *,
    force_fresh: bool = True,
) -> set[str]:
    """If eligible tools exist, prompt the user (per tool) to enable the Tool Gateway.

    "Pool enabled" is the trigger: a user with a live free tool pool (or paid access) is shown a
    per-tool checklist of the covered managed backends and picks which to route through the gateway.
    """
    # explicit_configured tools (e.g. an explicit `web.backend: searxng`) are
    # configured on purpose and are never offered here — same treatment as
    # already_managed, just for a non-nous vendor.
    unconfigured, has_direct, _explicit_configured, already_managed = get_gateway_eligible_tools(
        config,
        force_fresh=force_fresh,
    )
    if not unconfigured and not has_direct:
        return set()

    try:
        from hermes_cli.setup import prompt_checklist
    except Exception:
        return set()

    # Frame the offer by entitlement: a $0 free-tool-pool user is not on a paid
    # plan, so don't call it "your subscription".
    account_info = _account_info_or_none(force_fresh=False)
    pool_only = bool(
        account_info
        and account_info.paid_service_access is not True
        and account_info.tool_access is not None
        and account_info.tool_access.enabled
    )
    source_label = "free tool pool" if pool_only else "Nous subscription"

    # Per-tool checklist: unconfigured tools first (pre-checked for new users),
    # then tools where the user already has their own key (left unchecked so we
    # don't override their own setup unless they ask).
    #
    # Decline persistence (#92647): tools the user has previously seen offered
    # and left unchecked are recorded in ``tool_gateway_declined_tools`` and
    # are never pre-checked again — the offer downgrades to opt-in-only.
    # Acceptance used to be sticky while refusal was not, so the identical
    # pre-checked checklist re-fired on every Nous model swap.
    declined_raw = config.get("tool_gateway_declined_tools")
    declined: set[str] = (
        {str(k) for k in declined_raw} if isinstance(declined_raw, list) else set()
    )

    offer_keys: list[str] = list(unconfigured) + list(has_direct)
    labels: list[str] = [_GATEWAY_TOOL_LABELS[k] for k in unconfigured]
    labels += [
        f"{_GATEWAY_TOOL_LABELS[k]} — keep using your {_FEATURES[k].direct_label}"
        for k in has_direct
    ]
    pre_selected = [
        i for i, k in enumerate(unconfigured) if k not in declined
    ]

    title = (
        "Your free Nous tool pool — pick the tools to enable:"
        if pool_only
        else "Your Nous subscription includes the Tool Gateway — pick the tools to enable:"
    )

    try:
        chosen_idx = prompt_checklist(title, labels, pre_selected)
    except (KeyboardInterrupt, EOFError, OSError, SystemExit):
        return set()

    chosen_keys = [offer_keys[i] for i in chosen_idx if 0 <= i < len(offer_keys)]

    # Persist per-tool declines: every unconfigured tool that was offered and
    # NOT chosen was actively left (or unchecked) by the user — remember that
    # so the next Nous model swap doesn't pre-check it again. Cancel paths
    # (Ctrl-C/ESC above) return before this and record nothing. Choosing a
    # previously-declined tool clears its decline.
    newly_declined = [k for k in unconfigured if k not in chosen_keys and k not in declined]
    undeclined = declined & set(chosen_keys)
    if newly_declined or undeclined:
        config["tool_gateway_declined_tools"] = sorted(
            (declined | set(newly_declined)) - set(chosen_keys)
        )

    if not chosen_keys:
        changed: set[str] = set()
    else:
        changed = apply_gateway_defaults(config, chosen_keys)
    if changed or newly_declined:
        from hermes_cli.config import save_config

        save_config(config)
        for key in sorted(changed):
            label = _GATEWAY_TOOL_LABELS.get(key, key)
            print(f"  ✓ {label}: enabled via {source_label}")
    return changed


# ---------------------------------------------------------------------------
# Inline Nous Portal login for the Tool Gateway picker (`hermes tools`)
# ---------------------------------------------------------------------------


def ensure_nous_portal_access(
    *,
    capability: str = "the Nous Tool Gateway",
    coverage_category: Optional[str] = None,
) -> bool:
    """Make sure the user is entitled to the Nous Tool Gateway, logging in if needed.

    It only performs the Nous Portal device-code OAuth (when the user isn't already logged in) and
    refreshes entitlement, so the caller can enable the single tool the user picked.

    Entitlement is satisfied by paid service access OR a live free tool pool. When
    ``coverage_category`` is given (e.g. ``"fal"`` for image gen), the pool must cover that category
    specifically — so a pool user selecting video (``"fal-video"``, not pool-funded) is correctly
    denied.
    """

    def _entitled(account) -> bool:
        if account is None:
            return False
        if coverage_category is not None:
            return account.tool_gateway_entitled_for(coverage_category)
        return account.tool_gateway_entitled

    # Fast path: already entitled.
    info = _account_info_or_none(force_fresh=True)
    if _entitled(info):
        return True

    # If not logged in at all, run the device-code login (auth only).
    if info is None or not info.logged_in:
        if not _run_nous_portal_login_only(capability=capability):
            return False
        info = _account_info_or_none(force_fresh=True)

    if _entitled(info):
        return True

    # Logged in but not entitled for this capability — surface neutral billing
    # guidance, do not enable. coverage_category keeps a pool user who lacks this
    # one category from being told their credits are exhausted.
    message = format_nous_portal_entitlement_message(
        info, capability=capability, coverage_category=coverage_category
    )
    if message:
        for line in message.splitlines():
            print(f"  {line}")
    return False


def _run_nous_portal_login_only(*, capability: str) -> bool:
    """Run the Nous Portal device-code OAuth and persist credentials only.

    No model selection, no provider switch, no Tool Gateway bulk prompt. Returns ``True`` on a
    successful login, ``False`` if the user declined or the flow failed.
    """
    try:
        import hermes_cli.auth as auth
    except Exception as exc:  # pragma: no cover - defensive
        print(f"  Could not start Nous Portal login: {exc}")
        return False

    print()
    print(f"  {capability} requires a Nous Portal login.")
    try:
        proceed = input("  Log in to Nous Portal now? [Y/n]: ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print()
        return False
    if proceed not in {"", "y", "yes"}:
        print("  Skipped Nous Portal login.")
        return False

    try:
        # Snapshot the active_provider so a tool-config login never silently
        # switches the user's inference provider to Nous.
        with auth._auth_store_lock():
            prior_active_provider = auth._load_auth_store().get("active_provider")

        auth_state = None
        if auth._read_shared_nous_state():
            try:
                do_import = input(
                    "  Found existing Nous OAuth credentials. Import them? [Y/n]: "
                ).strip().lower()
            except (EOFError, KeyboardInterrupt):
                do_import = "y"
            if do_import in {"", "y", "yes"}:
                auth_state = auth._try_import_shared_nous_state(timeout_seconds=15.0)

        if auth_state is None:
            auth_state = auth._nous_device_code_login()

        with auth._auth_store_lock():
            auth_store = auth._load_auth_store()
            auth._save_provider_state(auth_store, "nous", auth_state)
            # Preserve the user's existing inference provider — this login is
            # for tool entitlement only, not a provider switch.
            if prior_active_provider:
                auth_store["active_provider"] = prior_active_provider
            else:
                auth_store.pop("active_provider", None)
            auth._save_auth_store(auth_store)

        auth._write_shared_nous_state(auth_state)
        auth._sync_nous_pool_from_auth_store()
        print("  Nous Portal login successful.")
        return True
    except KeyboardInterrupt:
        print("\n  Login cancelled.")
        return False
    except SystemExit:
        # _nous_device_code_login raises SystemExit on subscription_required;
        # it already printed billing guidance.
        return False
    except Exception as exc:
        print(f"  Nous Portal login failed: {exc}")
        return False
