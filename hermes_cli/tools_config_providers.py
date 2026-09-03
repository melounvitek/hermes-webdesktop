"""Provider rows, active-provider detection, model pickers and provider-selection persistence for `hermes tools`."""

from __future__ import annotations

import importlib
import logging
from typing import Callable, Optional

from hermes_cli.cli_output import (
    print_info as _print_info,
    print_success as _print_success,
    print_warning as _print_warning,
    prompt as _prompt,
)
from hermes_cli.colors import Colors, color
from hermes_cli.config import cfg_get, get_env_value, load_config, save_config, save_env_value
from hermes_cli.nous_account import format_nous_portal_entitlement_message
from hermes_cli.nous_subscription import MANAGED_FEATURE_COVERAGE_CATEGORY, NousSubscriptionFeatures
from tools.tool_backend_helpers import NOUS_MANAGED_PROVIDER, fal_key_is_configured
from utils import base_url_hostname, is_truthy_value

logger = logging.getLogger("hermes_cli.tools_config")

# tools_config-internal names (TOOL_CATEGORIES, _cfg_section, _prompt_choice, ...) are imported lazily
# inside the functions that need them: tools_config re-imports this module, and tests patch those names there.


def _plugin_registry(module: str):
    """Import a plugin registry module after plugin discovery; ``None`` on any failure."""
    try:
        registry = importlib.import_module(module)
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
        return registry
    except Exception:
        return None


def _plugin_provider_rows(
    registry_module: str,
    marker_keys: tuple[str, ...],
    *,
    require_name: bool = True,
    skip_builtin: bool = False,
    flatten_variants: bool = False,
) -> list[dict]:
    """Picker-row dicts (TOOL_CATEGORIES-shaped) for a plugin registry's providers.

    ``marker_keys`` are all set to the registry name so downstream config/model-picker code routes through
    the registry. ``skip_builtin`` drops names shadowing ``_BUILTIN_NAMES``; ``flatten_variants`` expands a
    schema's tier ``variants`` into separate rows sharing one backend name, distinguished by ``web_tier``.
    """
    registry = _plugin_registry(registry_module)
    if registry is None:
        return []
    try:
        providers = registry.list_providers()
        builtin = registry._BUILTIN_NAMES if skip_builtin else frozenset()
    except Exception:
        return []

    rows: list[dict] = []
    for provider in providers:
        if require_name:
            name = getattr(provider, "name", None)
            if not name or (skip_builtin and name.lower().strip() in builtin):
                continue
        try:
            schema = provider.get_setup_schema()
        except Exception:
            continue
        if not isinstance(schema, dict):
            continue
        if not require_name:
            name = provider.name
        entries = [schema]
        if flatten_variants:
            entries += [v for v in (schema.get("variants") or []) if isinstance(v, dict)]
        for entry in entries:
            row = {"name": entry.get("name", provider.display_name), "badge": entry.get("badge", ""),
                   "tag": entry.get("tag", ""), "env_vars": entry.get("env_vars", [])}
            row.update({key: name for key in marker_keys})
            if flatten_variants and entry.get("web_tier"):
                row["web_tier"] = entry["web_tier"]
            if entry.get("post_setup"):
                row["post_setup"] = entry["post_setup"]
            rows.append(row)
    return rows


# Category -> (registry module, marker keys, _plugin_provider_rows kwargs). image/video ``*_plugin_name`` route
# config writes + model pickers through the registry (video has no in-tree backend); all bundled web providers
# are plugins (only firecrawl setup-flow rows stay hardcoded); ``browser_provider`` is the legacy key written to
# ``browser.cloud_provider``; TTS rows render below the built-in rows and write ``tts.provider: <name>``.
_PLUGIN_PROVIDER_ROW_SPECS = {
    "image_gen": ("agent.image_gen_registry", ("image_gen_plugin_name",), {"require_name": False}),
    "video_gen": ("agent.video_gen_registry", ("video_gen_plugin_name",), {"require_name": False}),
    "web": ("agent.web_search_registry", ("web_backend", "web_search_plugin_name"), {"flatten_variants": True}),
    "browser": ("agent.browser_registry", ("browser_provider", "browser_plugin_name"), {}),
    "tts": ("agent.tts_registry", ("tts_provider", "tts_plugin_name"), {"skip_builtin": True}),
}


def _plugin_rows_for(category: str) -> list[dict]:
    module, markers, kwargs = _PLUGIN_PROVIDER_ROW_SPECS[category]
    return _plugin_provider_rows(module, markers, **kwargs)


def _plugin_image_gen_providers() -> list[dict]:
    """Picker rows for plugin-registered image gen providers."""
    return _plugin_rows_for("image_gen")


def _plugin_video_gen_providers() -> list[dict]:
    """Picker rows for plugin-registered video gen providers (the only Video Generation rows)."""
    return _plugin_rows_for("video_gen")


def _plugin_web_search_providers() -> list[dict]:
    """Picker rows for plugin-registered web search providers."""
    return _plugin_rows_for("web")


def _plugin_browser_providers() -> list[dict]:
    """Picker rows for plugin-registered cloud browser providers."""
    return _plugin_rows_for("browser")


def _plugin_tts_providers() -> list[dict]:
    """Picker rows for plugin-registered TTS providers."""
    return _plugin_rows_for("tts")


def web_provider_capabilities(backend: str) -> list:
    """Capabilities (``search`` / ``extract``) a web backend supports, per the registry provider instance.

    Lets the Capabilities GUI offer ``web.search_backend`` / ``web.extract_backend`` only where it makes sense
    (ddgs and brave-free are search-only). Unknown backend or registry failure -> both.
    """
    try:
        from agent.web_search_registry import get_provider

        provider = get_provider(backend)
        if provider is not None:
            caps = []
            if provider.supports_search():
                caps.append("search")
            if provider.supports_extract():
                caps.append("extract")
            return caps
    except Exception:
        pass
    return ["search", "extract"]


# TOOL_CATEGORIES[<key>]["name"] -> builder of plugin-registered picker rows.
_PLUGIN_ROW_BUILDERS = {
    "Image Generation": _plugin_image_gen_providers,
    "Video Generation": _plugin_video_gen_providers,
    "Web Search & Extract": _plugin_web_search_providers,
    "Browser Automation": _plugin_browser_providers,
    "Text-to-Speech": _plugin_tts_providers,
}


def _visible_providers(
    cat: dict,
    config: dict,
    *,
    force_fresh: bool = False,
    features: Optional[NousSubscriptionFeatures] = None,
) -> list[dict]:
    """Provider entries visible for the current auth/config state.

    Nous-managed rows (``managed_nous_feature``) are always shown, even logged-out/unentitled, to advertise
    the capability.
    """
    from hermes_cli.tools_config import get_nous_subscription_features

    if features is None:
        features = get_nous_subscription_features(config, force_fresh=force_fresh)
    acct = features.account_info
    # Pool-only users (free tool pool, no paid access) get image gen but NOT video gen — the pool doesn't
    # fund `fal-video`, so hide the managed video row rather than advertise a denial.
    pool_only = bool(acct and acct.logged_in and acct.paid_service_access is not True and acct.tool_gateway_entitled)
    visible = []
    for provider in cat.get("providers", []):
        # Managed rows stay visible regardless of auth (selecting one drives an inline Portal login); a
        # `requires_nous_auth` row without a managed feature hides until logged in.
        if provider.get("requires_nous_auth") and not provider.get("managed_nous_feature") and not features.nous_auth_present:
            continue
        if (
            pool_only
            and provider.get("managed_nous_feature") == "video_gen"
            and not (acct and acct.tool_gateway_entitled_for("fal-video"))
        ):
            continue
        visible.append(provider)

    # Plugin-registered rows render BELOW the hardcoded rows (for web/browser they are the only real provider rows).
    builder = _PLUGIN_ROW_BUILDERS.get(cat.get("name"))
    if builder is not None:
        visible.extend(builder())
    return visible


def provider_readiness_status(
    provider: dict,
    config: dict,
    *,
    features=None,
    is_active: Optional[bool] = None,
) -> str:
    """Honest readiness state for a provider picker row.

    ``features`` avoids re-fetching portal state per row. ``is_active`` is the completed-setup fallback for
    post_setup hooks with no registered installed-check (selecting a row runs its hook).
    """
    from hermes_cli.tools_config import (
        _POST_SETUP_READY,
        _provider_env_ready,
        _xai_credentials_present,
        get_nous_subscription_features,
    )

    if provider.get("env_vars", []):
        return "ready" if _provider_env_ready(provider) else "needs_keys"

    managed_feature = provider.get("managed_nous_feature")
    if provider.get("requires_nous_auth") or managed_feature:
        if features is None:
            features = get_nous_subscription_features(config)
        if not features.nous_auth_present:
            return "needs_auth"
        if managed_feature:
            # Same per-category entitlement gate the CLI applies at selection time.
            acct = features.account_info
            category = MANAGED_FEATURE_COVERAGE_CATEGORY.get(managed_feature)
            entitled = bool(acct and acct.logged_in and (
                acct.tool_gateway_entitled_for(category) if category else acct.tool_gateway_entitled))
            if not entitled:
                return "needs_auth"
        # Signed in and entitled — fall through: a managed row may still carry a local install hook.

    post_setup = provider.get("post_setup")
    if post_setup:
        if post_setup == "xai_grok":
            return "ready" if _xai_credentials_present() else "needs_auth"
        predicate = _POST_SETUP_READY.get(post_setup)
        if predicate is not None:
            try:
                return "ready" if predicate() else "needs_setup"
            except Exception:
                return "ready"  # flaky detection must not manufacture a warning state
        # No installed-check registered → the active-provider signal means "setup completed".
        if is_active is None:
            is_active = _is_provider_active(provider, config)
        return "ready" if is_active else "needs_setup"

    return "ready"


def _toolset_needs_configuration_prompt(
    ts_key: str,
    config: dict,
    *,
    force_fresh: bool = False,
) -> bool:
    """Return True when enabling this toolset should open provider setup."""
    from hermes_cli.tools_config import TOOL_CATEGORIES, _post_setup_already_installed, _toolset_has_keys

    cat = TOOL_CATEGORIES.get(ts_key)
    if not cat:
        return not _toolset_has_keys(ts_key, config, force_fresh=force_fresh)

    # An unsatisfied post_setup install-state check (e.g. cua-driver not on PATH yet) forces the
    # configuration flow so `_configure_provider` runs the hook and the install actually happens.
    for provider in _visible_providers(cat, config, force_fresh=force_fresh):
        post_setup = provider.get("post_setup")
        if post_setup and not _post_setup_already_installed(post_setup):
            return True

    # Categories whose "configured" signal is a selected provider key.
    selection_key = {"tts": "provider", "web": "backend", "browser": "cloud_provider"}.get(ts_key)
    if selection_key:
        section = config.get(ts_key, {})
        return not isinstance(section, dict) or selection_key not in section
    if ts_key == "image_gen":  # in-tree FAL backend OR any available plugin image gen provider satisfies
        return not fal_key_is_configured() and not _any_plugin_provider_available("agent.image_gen_registry")
    if ts_key == "video_gen":  # no in-tree fallback — every video backend is a plugin
        return not _any_plugin_provider_available("agent.video_gen_registry")

    return not _toolset_has_keys(ts_key, config, force_fresh=force_fresh)


def _any_plugin_provider_available(registry_module: str) -> bool:
    """True when any provider in the plugin registry reports ``is_available()``."""
    registry = _plugin_registry(registry_module)
    if registry is None:
        return False
    try:
        for provider in registry.list_providers():
            try:
                if provider.is_available():
                    return True
            except Exception:
                continue
    except Exception:
        pass
    return False


def _configure_tool_category(
    ts_key: str,
    cat: dict,
    config: dict,
    *,
    force_fresh: bool = True,
    reconfigure: bool = False,
):
    """Provider selection for a tool category, then API-key setup for the chosen row.

    ``reconfigure`` ("Reconfigure an existing tool"): no setup note / skip row / Nous marker, and the chosen
    provider goes through the key-update prompts instead of the new-enable prompts.
    """
    from hermes_cli.tools_config import _prompt_choice, _provider_env_ready, get_nous_subscription_features

    icon = cat.get("icon", "")
    name = cat["name"]
    providers = _visible_providers(cat, config, force_fresh=force_fresh)

    if len(providers) == 1:
        provider = providers[0]
        print()
        print(color(f"  --- {icon} {name} ({provider['name']}) ---", Colors.CYAN))
        if not reconfigure:
            if provider.get("tag"):
                _print_info(f"  {provider['tag']}")
            if cat.get("setup_note"):
                _print_info(f"  {cat['setup_note']}")
        _configure_provider(provider, config, force_fresh=force_fresh, reconfigure=reconfigure)
        return

    print()
    title = "Choose a provider" if reconfigure else cat.get("setup_title", "Choose a provider")
    print(color(f"  --- {icon} {name} - {title} ---", Colors.CYAN))
    if not reconfigure and cat.get("setup_note"):
        _print_info(f"  {cat['setup_note']}")
    print()

    # Logged-in Nous users get a marker on rows included in their subscription (cost-extra vs. included).
    _nous_logged_in = False
    if not reconfigure:
        try:
            _nous_logged_in = bool(get_nous_subscription_features(config, force_fresh=force_fresh).nous_auth_present)
        except Exception:
            _nous_logged_in = False

    provider_choices = []  # plain text labels only (no ANSI codes in menu items)
    for p in providers:
        badge = f" [{p['badge']}]" if p.get("badge") else ""
        tag = f" — {p['tag']}" if p.get("tag") else ""
        configured = ""
        if _provider_env_ready(p):
            if _is_provider_active(p, config, force_fresh=force_fresh):
                configured = " [active]"
            elif p.get("env_vars", []):
                configured = " [configured]"
        # Subscribers get the "included" star; everyone else a hint that selecting triggers a Portal login.
        sub_marker = ""
        if not reconfigure and p.get("managed_nous_feature"):
            sub_marker = "  ★ Included with your Nous subscription" if _nous_logged_in else "  ★ via Nous Portal (login on select)"
        provider_choices.append(f"{p['name']}{badge}{tag}{configured}{sub_marker}")

    if not reconfigure:
        provider_choices.append("Skip — keep defaults / configure later")

    default_idx = _detect_active_provider_index(providers, config, force_fresh=force_fresh)
    question = "  Select provider:" if reconfigure else f"  {title}:"
    provider_idx = _prompt_choice(question, provider_choices, default_idx)
    if provider_idx >= len(providers):
        _print_info(f"  Skipped {name}")
        return
    _configure_provider(providers[provider_idx], config, force_fresh=force_fresh, reconfigure=reconfigure)


def _web_tier_matches(provider: dict, config: dict) -> bool:
    """True when a web picker row's tier matches the configured tier (``web.provider_tier.<backend>``).

    Tiered rows (Exa/Parallel Free vs Paid) share one ``web_backend`` and differ only in ``web_tier``. No
    ``web_tier`` on the row → matches; configured tier set → must equal the row's tier; unset → "auto":
    paid when the row's key is present, free otherwise (highlight the row the runtime would actually use).
    """
    row_tier = provider.get("web_tier")
    if not row_tier:
        return True
    web_cfg = config.get("web") if isinstance(config.get("web"), dict) else {}
    tiers = web_cfg.get("provider_tier")
    if not isinstance(tiers, dict):
        tiers = {}
    configured = str(tiers.get(provider["web_backend"], "") or "").lower().strip()
    if configured in ("free", "paid"):
        return configured == row_tier
    # Auto: mirror plugins.web.keyless_mcp.use_keyless — key present → paid.
    try:
        from agent.web_search_provider import get_provider_env

        key_var = {"exa": "EXA_API_KEY", "parallel": "PARALLEL_API_KEY"}.get(provider["web_backend"])
        has_key = bool(get_provider_env(key_var)) if key_var else False
    except Exception:
        has_key = False
    return row_tier == ("paid" if has_key else "free")


def _managed_provider_active(provider: dict, config: dict, managed_feature: str, force_fresh: bool) -> bool:
    """Active check for a Nous-managed row: the feature must be managed AND the category's selected
    provider must be the row's vendor or ``nous``."""
    from hermes_cli.tools_config import get_nous_subscription_features

    features = get_nous_subscription_features(config, force_fresh=force_fresh)
    feature = features.features.get(managed_feature)
    if feature is None:
        return False
    if managed_feature in ("image_gen", "video_gen"):
        gen_cfg = config.get(managed_feature, {})
        if isinstance(gen_cfg, dict):
            configured_provider = gen_cfg.get("provider")
            if configured_provider not in {None, "", "fal", NOUS_MANAGED_PROVIDER}:
                return False
            if (
                configured_provider != NOUS_MANAGED_PROVIDER
                and gen_cfg.get("use_gateway") is not None
                and not is_truthy_value(gen_cfg.get("use_gateway"), default=False)
            ):
                return False
        return feature.managed_by_nous
    if provider.get("tts_provider"):
        current = cfg_get(config, "tts", "provider")
        return feature.managed_by_nous and current in {provider["tts_provider"], NOUS_MANAGED_PROVIDER}
    if provider.get("stt_provider"):
        current = cfg_get(config, "stt", "provider")
        return feature.managed_by_nous and current in {provider["stt_provider"], NOUS_MANAGED_PROVIDER}
    if "browser_provider" in provider:
        # Browser Use mode is a driver on top of the provider (attaches to its CDP endpoint), so the
        # provider row stays active alongside the Browser Use row.
        current = cfg_get(config, "browser", "cloud_provider")
        return feature.managed_by_nous and current in {provider["browser_provider"], NOUS_MANAGED_PROVIDER}
    if provider.get("web_backend"):
        current = cfg_get(config, "web", "backend")
        return (
            feature.managed_by_nous
            and current in {provider["web_backend"], NOUS_MANAGED_PROVIDER}
            and _web_tier_matches(provider, config)
        )
    return feature.managed_by_nous


def _browser_use_default_active(config: dict) -> bool:
    """``browser.backend`` unset: Browser Use mode is the default, so the row is active whenever the
    effective mode resolves on (legacy direct-API cloud config, or CLI runnable and no Camofox)."""
    browser_cfg = config.get("browser") if isinstance(config, dict) else None
    try:
        from tools.browser_use_cli import _find_cli, is_legacy_browser_use_cloud_config

        if is_legacy_browser_use_cloud_config(browser_cfg or {}):
            return True
        try:
            from tools.browser_camofox import is_camofox_mode

            if is_camofox_mode():
                return False
        except Exception:
            pass
        return _find_cli() is not None
    except Exception:
        return False


def _tts_active(provider: dict, config: dict) -> bool:
    return cfg_get(config, "tts", "provider") == provider["tts_provider"]


def _stt_active(provider: dict, config: dict) -> bool:
    # Default stt.provider is "local" — an unset key means Local Whisper.
    return (cfg_get(config, "stt", "provider") or "local") == provider["stt_provider"]


def _browser_provider_active(provider: dict, config: dict) -> bool:
    # Browser Use mode composes with the provider (driver over its CDP endpoint) — don't deactivate the row.
    if provider["browser_provider"] != cfg_get(config, "browser", "cloud_provider"):
        return False
    # Two local rows differ only by engine ("Local Browser" vs "Lightpanda"): config.yaml is the picker's
    # source of truth here, the AGENT_BROWSER_ENGINE env var is not consulted.
    if provider.get("browser_engine"):
        engine = str(cfg_get(config, "browser", "engine") or "auto").strip().lower()
        return engine == provider["browser_engine"]
    return True


def _browser_backend_active(provider: dict, config: dict) -> bool:
    backend = cfg_get(config, "browser", "backend")
    if backend is False:
        backend = "off"  # YAML 1.1: unquoted `off` parses as boolean False
    if backend == provider["browser_backend"]:
        return True
    if backend:
        return False  # explicit other choice ("off", …) wins
    if provider["browser_backend"] != "browser-use":
        return False
    return _browser_use_default_active(config)


def _web_backend_active(provider: dict, config: dict) -> bool:
    return cfg_get(config, "web", "backend") == provider["web_backend"] and _web_tier_matches(provider, config)


def _computer_use_active(provider: dict, config: dict) -> bool:
    return cfg_get(config, "computer_use", "backend") == provider["computer_use_backend"]


def _imagegen_backend_active(provider: dict, config: dict) -> bool:
    image_cfg = config.get("image_gen", {})
    return (
        isinstance(image_cfg, dict)
        and provider["imagegen_backend"] == "fal"
        and image_cfg.get("provider") in {None, "", "fal"}
        and not is_truthy_value(image_cfg.get("use_gateway"), default=False)
    )


# Non-managed active checks, evaluated in order; the first marker the row carries decides. ``browser_provider``
# is a membership test (a local row carries ``browser_provider: ""``), every other marker is a truthiness test.
_ACTIVE_CHECKS: tuple[tuple[str, Callable[[dict, dict], bool]], ...] = (
    ("tts_provider", _tts_active),
    ("stt_provider", _stt_active),
    ("browser_provider", _browser_provider_active),
    ("browser_backend", _browser_backend_active),
    ("web_backend", _web_backend_active),
    ("computer_use_backend", _computer_use_active),
    ("imagegen_backend", _imagegen_backend_active),
)
_MEMBERSHIP_MARKERS = frozenset({"browser_provider"})


def _is_provider_active(
    provider: dict,
    config: dict,
    *,
    force_fresh: bool = False,
) -> bool:
    """Check if a provider entry matches the currently active config."""
    managed_feature = provider.get("managed_nous_feature")
    plugin_name = provider.get("image_gen_plugin_name")
    if plugin_name and not managed_feature:
        # Managed entries fall through to the managed branch, which also checks use_gateway — otherwise a
        # managed FAL pick and a direct-key FAL pick would both report active.
        image_cfg = config.get("image_gen", {})
        if not (isinstance(image_cfg, dict) and image_cfg.get("provider") == plugin_name):
            return False
        # A direct-key entry is only active when the managed route is OFF.
        return not is_truthy_value(image_cfg.get("use_gateway"), default=False)

    video_plugin_name = provider.get("video_gen_plugin_name")
    if video_plugin_name and not managed_feature:
        video_cfg = config.get("video_gen", {})
        return isinstance(video_cfg, dict) and video_cfg.get("provider") == video_plugin_name

    if managed_feature:
        return _managed_provider_active(provider, config, managed_feature, force_fresh)

    for marker, check in _ACTIVE_CHECKS:
        present = marker in provider if marker in _MEMBERSHIP_MARKERS else provider.get(marker)
        if present:
            return check(provider, config)
    return False


def _detect_active_provider_index(
    providers: list,
    config: dict,
    *,
    force_fresh: bool = False,
) -> int:
    """Return the index of the currently active provider, or 0."""
    from hermes_cli.tools_config import _provider_env_ready

    for i, p in enumerate(providers):
        if _is_provider_active(p, config, force_fresh=force_fresh):
            return i
        if p.get("env_vars", []) and _provider_env_ready(p):  # fallback: env vars present → likely configured
            return i
    return 0


def _fal_model_catalog():
    """Lazy-load the FAL model catalog from the tool module."""
    from tools.image_generation_tool import FAL_MODELS, DEFAULT_MODEL
    return FAL_MODELS, DEFAULT_MODEL


# Per-backend model catalog (config_key = top-level config.yaml section, catalog_fn -> ({model_id: metadata},
# default_model)); a TOOL_CATEGORIES row tagged `imagegen_backend: "<name>"` selects the catalog at picker time.
IMAGEGEN_BACKENDS = {
    "fal": {"display": "FAL.ai", "config_key": "image_gen", "catalog_fn": _fal_model_catalog},
}


def _plugin_model_catalog(registry_module: str, plugin_name: str):
    """``(catalog_dict, default_model_id)`` for a plugin provider; ``catalog_dict`` is shaped like the legacy
    ``FAL_MODELS`` table so the picker path is shared. ``({}, None)`` if unregistered or no models."""
    registry = _plugin_registry(registry_module)
    if registry is None:
        return {}, None
    try:
        provider = registry.get_provider(plugin_name)
    except Exception:
        return {}, None
    if provider is None:
        return {}, None
    try:
        models = provider.list_models() or []
        default = provider.default_model()
    except Exception:
        return {}, None
    catalog = {m["id"]: m for m in models if isinstance(m, dict) and "id" in m}
    return catalog, default


def _plugin_image_gen_catalog(plugin_name: str):
    """``(catalog, default_model)`` for a plugin-registered image gen provider."""
    return _plugin_model_catalog("agent.image_gen_registry", plugin_name)


def _plugin_video_gen_catalog(plugin_name: str):
    """``(catalog, default_model)`` for a plugin-registered video gen provider."""
    return _plugin_model_catalog("agent.video_gen_registry", plugin_name)


def _pick_model_from_catalog(
    catalog: dict,
    default_model,
    cfg_key: str,
    display: str,
    config: dict,
    *,
    row_indent: str = "",
) -> None:
    """Column-aligned model picker shared by the FAL, plugin image gen and video gen flows.

    Writes the choice to ``config[cfg_key]["model"]``. The current model is listed first so the cursor lands
    on it; a saved model belonging to another provider (shared config key) or a drifted catalog default never
    indexes the catalog. Safe when stdin is not a TTY — curses_radiolist keeps the current selection.
    """
    from hermes_cli.tools_config import _cfg_section, _prompt_choice

    if not catalog:
        return
    cur_cfg = _cfg_section(config, cfg_key)
    current_model = cur_cfg.get("model") or default_model
    if current_model not in catalog:
        current_model = default_model if default_model in catalog else next(iter(catalog))

    model_ids = list(catalog.keys())
    ordered = [current_model] + [m for m in model_ids if m != current_model]
    widths = {
        "model": max(len(m) for m in model_ids),
        "speed": max((len(catalog[m].get("speed", "")) for m in model_ids), default=6),
        "strengths": max((len(catalog[m].get("strengths", "")) for m in model_ids), default=0),
    }

    print()
    header = (f"  {'Model':<{widths['model']}}  {'Speed':<{widths['speed']}}  "
              f"{'Strengths':<{widths['strengths']}}  Price")
    print(color(header, Colors.CYAN))

    rows = []
    for mid in ordered:
        meta = catalog[mid]
        row = (f"{row_indent}{mid:<{widths['model']}}  {meta.get('speed', ''):<{widths['speed']}}  "
               f"{meta.get('strengths', ''):<{widths['strengths']}}  {meta.get('price', '')}")
        if mid == current_model:
            row += "  ← currently in use"
        rows.append(row)

    idx = _prompt_choice(f"  Choose {display} model:", rows, default=0)
    chosen = ordered[idx]
    cur_cfg["model"] = chosen
    _print_success(f"  Model set to: {chosen}")


def _configure_imagegen_model(backend_name: str, config: dict) -> None:
    """Prompt for a model of an in-tree imagegen backend (``IMAGEGEN_BACKENDS``)."""
    backend = IMAGEGEN_BACKENDS.get(backend_name)
    if not backend:
        return
    catalog, default_model = backend["catalog_fn"]()
    _pick_model_from_catalog(catalog, default_model, backend["config_key"], backend["display"], config)


def _configure_imagegen_model_for_plugin(plugin_name: str, config: dict) -> None:
    """Prompt for a model from a plugin-registered image gen backend's catalog."""
    catalog, default_model = _plugin_image_gen_catalog(plugin_name)
    _pick_model_from_catalog(catalog, default_model, "image_gen", plugin_name, config)


def _configure_videogen_model_for_plugin(plugin_name: str, config: dict) -> None:
    """Prompt for a video gen model from a plugin's catalog."""
    catalog, default_model = _plugin_video_gen_catalog(plugin_name)
    # Video rows historically carry a two-space indent; kept for output parity.
    _pick_model_from_catalog(catalog, default_model, "video_gen", plugin_name, config, row_indent="  ")


def _configure_xai_imagine_storage(section_name: str, config: dict) -> None:
    """Prompt for xAI Imagine stored public URL behavior."""
    from hermes_cli.tools_config import _cfg_section, _prompt_choice

    section = _cfg_section(config, section_name)
    xai_cfg = _cfg_section(section, "xai")
    storage_cfg = _cfg_section(xai_cfg, "storage")

    _print_warning(
        "  xAI Imagine can store generated media and create reusable public URLs. "
        "xAI may bill for stored files and public URL hosting."
    )
    choices = ["Enable public URLs without automatic expiry (recommended)", "Disable stored public URLs",
               "Enable public URLs for 2 days"]
    idx = _prompt_choice("  Stored public URLs:", choices, default=0)
    if idx == 1:
        storage_cfg["enabled"] = False
        _print_success("  xAI stored public URLs disabled")
    elif idx == 2:
        storage_cfg["enabled"] = True
        storage_cfg["public_url"] = True
        storage_cfg["expires_after"] = 2 * 24 * 60 * 60
        _print_success("  xAI stored public URLs enabled for 2 days")
    else:
        storage_cfg["enabled"] = True
        storage_cfg["public_url"] = True
        storage_cfg["expires_after"] = None
        _print_success("  xAI stored public URLs enabled without automatic expiry")


def _select_plugin_gen_provider(section: str, plugin_name: str, config: dict, *, use_gateway: bool) -> None:
    """Persist a plugin-backed image/video gen provider selection and run its model picker.

    ``use_gateway=True`` (Nous-managed pick) stores ``<section>.provider: nous``; BYOK picks store the plugin
    name. Any legacy ``use_gateway`` key is removed so old read-time shims cannot override the selection.
    """
    from hermes_cli.tools_config import _cfg_section

    cfg = _cfg_section(config, section)
    cfg["provider"] = NOUS_MANAGED_PROVIDER if use_gateway else plugin_name
    cfg.pop("use_gateway", None)
    _print_success(f"  {section}.provider set to: {cfg['provider']}")
    if section == "image_gen":
        _configure_imagegen_model_for_plugin(plugin_name, config)
    else:
        _configure_videogen_model_for_plugin(plugin_name, config)
    if plugin_name == "xai":
        _configure_xai_imagine_storage(section, config)


def _select_plugin_image_gen_provider(plugin_name: str, config: dict, *, use_gateway: bool = False) -> None:
    _select_plugin_gen_provider("image_gen", plugin_name, config, use_gateway=use_gateway)


def _select_plugin_video_gen_provider(plugin_name: str, config: dict, *, use_gateway: bool = False) -> None:
    _select_plugin_gen_provider("video_gen", plugin_name, config, use_gateway=use_gateway)


# Per-provider STT model catalogs for the picker; keys are ``stt.<provider>`` sections, first entry is the
# default. Kept in sync with the dashboard selects (web_server _CONFIG_FIELD_META) and the desktop settings
# enums (apps/desktop/src/app/settings/constants.ts).
STT_MODEL_CATALOG = {
    "local": ["base", "tiny", "small", "medium", "large-v3"],
    "groq": ["whisper-large-v3-turbo", "whisper-large-v3", "distil-whisper-large-v3-en"],
    "openai": ["whisper-1", "gpt-4o-mini-transcribe", "gpt-4o-transcribe", "gpt-transcribe"],
    "elevenlabs": ["scribe_v2", "scribe_v1"],
}

# ElevenLabs historically uses ``model_id`` instead of ``model``.
_STT_MODEL_CONFIG_KEY = {"elevenlabs": "model_id"}


def _configure_stt_model(stt_provider: str, config: dict) -> None:
    """Prompt for the STT model after a provider pick (when a catalog exists)."""
    from hermes_cli.tools_config import _cfg_section, _prompt_choice

    catalog = STT_MODEL_CATALOG.get(stt_provider)
    if not catalog:
        return
    stt_cfg = _cfg_section(config, "stt")
    prov_cfg = _cfg_section(stt_cfg, stt_provider)
    model_key = _STT_MODEL_CONFIG_KEY.get(stt_provider, "model")
    current = str(prov_cfg.get(model_key) or "").strip()
    ordered = list(catalog)
    default_idx = ordered.index(current) if current in ordered else 0
    idx = _prompt_choice("  Select STT model:", ordered, default_idx)
    chosen = ordered[idx]
    prov_cfg[model_key] = chosen
    _print_success(f"  STT model set to: {chosen}")


# Provider-row marker key -> config section it selects into.
_PROVIDER_MARKER_SECTIONS = {
    "tts_provider": "tts",
    "stt_provider": "stt",
    "browser_provider": "browser",
    "web_backend": "web",
    "image_gen_plugin_name": "image_gen",
    "imagegen_backend": "image_gen",
    "video_gen_plugin_name": "video_gen",
}


def _write_provider_config(provider: dict, config: dict, *, managed_feature) -> None:
    """Persist the provider/backend config keys for a selected provider.

    Pure, non-interactive core of :func:`_configure_provider` (no env prompts, post-setup hooks, Nous auth
    gating or model pickers) shared by the CLI and the GUI ``PUT .../provider`` endpoint. Each pick writes
    exactly ONE provider string per category (``nous`` for managed rows) and removes any legacy
    ``use_gateway`` key so the read-time shim cannot override the new choice.
    """
    from hermes_cli.tools_config import TOOL_CATEGORIES, _cfg_section

    def _drop_use_gateway(section) -> None:
        if isinstance(section, dict):
            section.pop("use_gateway", None)

    def _set_selection(section_key: str, name_key: str, vendor_value) -> None:
        section = _cfg_section(config, section_key)
        section[name_key] = NOUS_MANAGED_PROVIDER if managed_feature else vendor_value
        _drop_use_gateway(section)

    if provider.get("tts_provider"):
        _set_selection("tts", "provider", provider["tts_provider"])
    if provider.get("stt_provider"):
        _set_selection("stt", "provider", provider["stt_provider"])

    if "browser_provider" in provider:
        bp = provider["browser_provider"]
        browser_cfg = config.setdefault("browser", {})
        if bp or managed_feature:
            # Browser Use mode (browser.backend) composes with the provider — keep the driver choice intact.
            _set_selection("browser", "cloud_provider", bp)
        else:
            browser_cfg.pop("use_gateway", None)
    if provider.get("browser_backend"):
        config.setdefault("browser", {})["backend"] = provider["browser_backend"]
    # Local engine rows ("Local Browser" resets to auto, "Lightpanda" sets lightpanda); composes with browser.backend.
    if provider.get("browser_engine"):
        config.setdefault("browser", {})["engine"] = provider["browser_engine"]

    if provider.get("web_backend"):
        _set_selection("web", "backend", provider["web_backend"])
        web_cfg = config.get("web")
        if isinstance(web_cfg, dict):
            tier = provider.get("web_tier")
            tiers = web_cfg.setdefault("provider_tier", {}) if tier else web_cfg.get("provider_tier")
            if isinstance(tiers, dict):
                if tier:
                    tiers[provider["web_backend"]] = tier
                else:
                    tiers.pop(provider["web_backend"], None)

    if provider.get("computer_use_backend"):
        config.setdefault("computer_use", {})["backend"] = provider["computer_use_backend"]

    if managed_feature and managed_feature not in {"web", "tts", "stt", "browser"}:
        # Managed rows without a marker above (image_gen/video_gen "Nous Subscription" rows carry only
        # managed_nous_feature) still persist the "nous" selection.
        section = config.setdefault(managed_feature, {})
        if isinstance(section, dict):
            section["provider"] = NOUS_MANAGED_PROVIDER
            _drop_use_gateway(section)
    elif not managed_feature:
        # Non-gateway pick — clear any stale legacy use_gateway key on the category. Resolve the category from
        # the row's own markers first (plugin-injected rows are NOT in TOOL_CATEGORIES' hardcoded lists), then
        # fall back to the category-membership walk.
        sections = [section_key for marker, section_key in _PROVIDER_MARKER_SECTIONS.items() if marker in provider]
        if not sections:
            sections = [cat_key for cat_key, cat in TOOL_CATEGORIES.items() if provider in cat.get("providers", [])][:1]
        for section_key in sections:
            _drop_use_gateway(config.get(section_key))


def apply_provider_selection(ts_key: str, provider_name: str, config: dict) -> None:
    """Non-interactively persist a provider selection for a toolset (config keys only — API keys, post-setup
    hooks, auth gating and model pickers are separate GUI endpoints). ``provider_name`` is resolved among
    :func:`_visible_providers` rows; raises ``KeyError`` for an unknown toolset or provider."""
    from hermes_cli.tools_config import TOOL_CATEGORIES, _cfg_section

    cat = TOOL_CATEGORIES.get(ts_key)
    if cat is None:
        raise KeyError(f"Toolset has no configurable category: {ts_key}")

    providers = _visible_providers(cat, config, force_fresh=True)
    provider = next((p for p in providers if p.get("name") == provider_name), None)
    if provider is None:
        raise KeyError(f"Unknown provider {provider_name!r} for toolset {ts_key!r}")

    managed_feature = provider.get("managed_nous_feature")
    _write_provider_config(provider, config, managed_feature=managed_feature)

    # Plugin image/video gen backends record the provider name in their own section (model choice is a separate
    # GUI flow); managed picks store "nous". The in-tree FAL BYOK row always persists an explicit
    # ``image_gen.provider: fal`` so a deliberate pick is distinguishable from a never-configured install.
    selections = [
        ("image_gen", provider.get("image_gen_plugin_name")),
        ("video_gen", provider.get("video_gen_plugin_name")),
        ("image_gen", "fal" if provider.get("imagegen_backend") and not managed_feature else None),
    ]
    for section_key, vendor in selections:
        if vendor:
            section = _cfg_section(config, section_key)
            section["provider"] = NOUS_MANAGED_PROVIDER if managed_feature else vendor
            section.pop("use_gateway", None)


def _nous_provider_gate(provider: dict, config: dict, managed_feature, *, force_fresh: bool) -> bool:
    """Return False (after printing why) when a Nous-gated row cannot be selected.

    Managed Tool Gateway rows are always listed but only *activate* with paid Nous Portal access — selecting
    one runs an inline Portal login (auth + entitlement only, no inference-provider switch). Pure pre-auth UX
    rows (``requires_nous_auth`` without a managed feature) keep the older logged-in + entitled gate.
    """
    from hermes_cli.tools_config import get_nous_subscription_features

    if managed_feature:
        from hermes_cli.nous_subscription import ensure_nous_portal_access

        if not ensure_nous_portal_access(
            capability=f"{provider.get('name', 'the Nous Tool Gateway')}",
            coverage_category=MANAGED_FEATURE_COVERAGE_CATEGORY.get(managed_feature),
        ):
            _print_warning("  Not enabled — Nous Portal access is required for this backend.")
            return False
        return True

    if provider.get("requires_nous_auth"):
        features = get_nous_subscription_features(config, force_fresh=force_fresh)
        entitled = bool(features.account_info and features.account_info.paid_service_access is True)
        if not features.nous_auth_present or not entitled:
            message = format_nous_portal_entitlement_message(
                features.account_info, capability=f"{provider.get('name', 'Nous Subscription')}"
            )
            _print_warning(f"  {message or 'Nous Subscription is only available after logging into Nous Portal.'}")
            return False
    return True


def _finish_provider_selection(provider: dict, config: dict, managed_feature) -> None:
    """Model pickers that follow a provider pick: plugin image/video gen, in-tree FAL, STT."""
    from hermes_cli.tools_config import _cfg_section

    plugin_name = provider.get("image_gen_plugin_name")
    if plugin_name:
        _select_plugin_image_gen_provider(plugin_name, config, use_gateway=bool(managed_feature))
        return
    video_plugin = provider.get("video_gen_plugin_name")
    if video_plugin:
        _select_plugin_video_gen_provider(video_plugin, config, use_gateway=bool(managed_feature))
        return
    backend = provider.get("imagegen_backend")
    if backend:
        _configure_imagegen_model(backend, config)
        # In-tree FAL is the only non-plugin backend: "nous" for a managed row, "fal" for BYOK, drop legacy
        # use_gateway — never clobber a managed pick back onto direct keys.
        img_cfg = _cfg_section(config, "image_gen")
        img_cfg["provider"] = NOUS_MANAGED_PROVIDER if managed_feature else "fal"
        img_cfg.pop("use_gateway", None)
    # STT rows prompt for a model after the pick (skipped for managed rows — the gateway pins it).
    if provider.get("stt_provider") and not managed_feature:
        _configure_stt_model(provider["stt_provider"], config)


def _print_provider_selection(provider: dict, managed_feature, *, reconfigure: bool) -> None:
    """Status lines announcing which backend/provider keys a pick writes."""
    if reconfigure and provider.get("tts_provider"):
        _print_success(f"  TTS provider set to: {provider['tts_provider']}")
    if provider.get("stt_provider"):
        _print_success(f"  STT provider set to: {provider['stt_provider']}")
    if "browser_provider" in provider:
        bp = provider["browser_provider"]
        if reconfigure and managed_feature:
            _print_success(f"  Browser cloud provider set to: {bp or 'nous'}")
        elif bp == "local":
            _print_success("  Browser set to local mode")
        elif bp:
            _print_success(f"  Browser cloud provider set to: {bp}")
    if provider.get("browser_backend"):
        _print_success("  Browser set to Browser Use (browser_exec via CLI 3.0)")
    if provider.get("browser_engine") and provider["browser_engine"] != "auto":
        _print_success(f"  Browser engine set to: {provider['browser_engine']}")
    if provider.get("web_backend"):
        if reconfigure and provider.get("web_tier"):
            _print_success(f"  Web backend set to: {provider['web_backend']} ({provider['web_tier']} tier)")
        else:
            _print_success(f"  Web backend set to: {provider['web_backend']}")
    if reconfigure and provider.get("computer_use_backend"):
        _print_success(f"  Computer Use backend set to: {provider['computer_use_backend']}")


def _show_portal_hint(provider: dict, config: dict, managed_feature, force_fresh: bool) -> bool:
    """True when a BYOK row shares its category with a Nous-managed sibling and the user is not authed to
    Nous — a single dim hint tells them the key is avoidable via a Portal subscription."""
    from hermes_cli.tools_config import TOOL_CATEGORIES, get_nous_subscription_features

    if managed_feature or provider.get("requires_nous_auth"):
        return False
    try:
        for _cat in TOOL_CATEGORIES.values():
            _providers = _cat.get("providers", [])
            if provider in _providers and any(sib.get("managed_nous_feature") for sib in _providers):
                return not get_nous_subscription_features(config, force_fresh=force_fresh).nous_auth_present
    except Exception:
        pass
    return False


def _prompt_env_vars(env_vars: list, *, reconfigure: bool) -> bool:
    """Prompt for a provider's env vars; True when every key ended up configured.

    Reconfigure mode re-prompts every key ("Enter to keep current") and always returns True; the new-enable
    flow keeps already-set keys without asking and reports False on any skipped key.
    """
    all_configured = True
    for var in env_vars:
        existing = get_env_value(var["key"])
        if reconfigure:
            if existing:
                _print_info(f"  {var['key']}: configured ({existing[:8]}...)")
            url = var.get("url", "")
            if url:
                _print_info(f"  Get yours at: {url}")
            default_val = var.get("default", "")
            value = _prompt(f"    {var.get('prompt', var['key'])} (Enter to keep current)", password=not default_val)
            if value and value.strip():
                save_env_value(var["key"], value.strip())
                _print_success("    Updated")
            else:
                _print_info("    Kept current")
            continue
        if existing:
            _print_success(f"  {var['key']}: already configured")
            continue
        url = var.get("url", "")
        if url:
            _print_info(f"  Get yours at: {url}")
        default_val = var.get("default", "")
        if default_val:
            value = _prompt(f"    {var.get('prompt', var['key'])}", default_val)
        else:
            value = _prompt(f"    {var.get('prompt', var['key'])}", password=True)
        if value:
            save_env_value(var["key"], value)
            _print_success("    Saved")
        else:
            _print_warning("    Skipped")
            all_configured = False
    return all_configured


def _configure_provider(
    provider: dict,
    config: dict,
    *,
    force_fresh: bool = True,
    reconfigure: bool = False,
):
    """Configure a single provider - prompt for API keys and set config.

    ``reconfigure=False`` (new-enable): already-set keys are kept without asking and the post-setup hook only
    runs when every key was provided. ``reconfigure=True`` re-prompts every key and always runs the hook.
    """
    from hermes_cli.tools_config import _run_post_setup

    env_vars = provider.get("env_vars", [])
    managed_feature = provider.get("managed_nous_feature")

    if not _nous_provider_gate(provider, config, managed_feature, force_fresh=force_fresh):
        return

    _print_provider_selection(provider, managed_feature, reconfigure=reconfigure)
    # Shared with the GUI provider-select endpoint (apply_provider_selection): one source of truth for config writes.
    _write_provider_config(provider, config, managed_feature=managed_feature)

    if not env_vars:
        if provider.get("post_setup"):
            _run_post_setup(provider["post_setup"])
        _print_success(f"  {provider['name']} - no configuration needed!")
        if managed_feature:
            _print_info("  Requests for this tool will be billed to your Nous subscription.")
        _finish_provider_selection(provider, config, managed_feature)
        return

    if not reconfigure and _show_portal_hint(provider, config, managed_feature, force_fresh):
        _print_info("  Available through Nous Portal subscription.")

    all_configured = _prompt_env_vars(env_vars, reconfigure=reconfigure)
    if provider.get("post_setup") and all_configured:
        _run_post_setup(provider["post_setup"])
    if all_configured:
        if not reconfigure:
            _print_success(f"  {provider['name']} configured!")
        _finish_provider_selection(provider, config, managed_feature)


def _reconfigure_provider(provider: dict, config: dict, *, force_fresh: bool = True):
    """Reconfigure a provider - update API keys."""
    _configure_provider(provider, config, force_fresh=force_fresh, reconfigure=True)


def _configure_vision_backend() -> None:
    """Interactive vision-backend configuration (``auxiliary.vision.{provider,model,base_url}``).

    Offers any authenticated provider + model (same surface as ``hermes model``) or a custom endpoint rather
    than forcing OpenRouter. "Auto" leaves the keys empty so the resolver uses the main-model fallback chain.
    """
    from hermes_cli.tools_config import _cfg_section, _prompt_choice

    print()
    print(color("  Vision / Image Analysis needs a multimodal model.", Colors.YELLOW))
    print(color("  Pick any provider + model (like /model), or let it auto-detect.", Colors.DIM))

    choices = [
        "Auto — use your main model / aggregator fallback (recommended)",
        "Pick a provider and model",
        "Custom OpenAI-compatible endpoint — base URL, API key, model",
        "Skip",
    ]
    idx = _prompt_choice("  Configure vision backend", choices, 0)

    config = load_config()
    vision_cfg = _cfg_section(_cfg_section(config, "auxiliary"), "vision")

    if idx == 0:
        # Auto: clear any pinned override so the resolver auto-detects.
        for key in ("provider", "model", "base_url", "api_key", "api_mode"):
            vision_cfg.pop(key, None)
        save_config(config)
        _print_success("  Vision set to auto (main model / aggregator fallback)")
        return

    if idx == 1:
        _configure_vision_provider_model(config, vision_cfg)
        return

    if idx == 2:
        base_url = _prompt("    Base URL (blank for OpenAI)").strip() or "https://api.openai.com/v1"
        is_native_openai = base_url_hostname(base_url) == "api.openai.com"
        key_label = "    OPENAI_API_KEY" if is_native_openai else "    API key"
        api_key = _prompt(key_label, password=True)
        if not (api_key and api_key.strip()):
            _print_warning("    Skipped")
            return
        default_model = "gpt-4o-mini" if is_native_openai else ""
        model = _prompt(f"    Vision model{f' (blank for {default_model})' if default_model else ''}").strip() or default_model
        save_env_value("OPENAI_API_KEY", api_key.strip())
        # Only base_url + model go to config.yaml; the key is the secret. Pin provider="custom" so the resolver
        # routes through this endpoint — at the "auto" default _resolve_task_provider_model ignores base_url
        # unless paired with a config api_key.
        vision_cfg["provider"] = "custom"
        vision_cfg["base_url"] = base_url
        if model:
            vision_cfg["model"] = model
        else:
            vision_cfg.pop("model", None)
        save_config(config)
        _print_success(f"  Vision set to custom endpoint{f' ({model})' if model else ''}")
        return

    _print_info("  Skipped vision configuration")


def _configure_vision_provider_model(config: dict, vision_cfg: dict) -> None:
    """Provider + model picker for vision, mirroring the ``/model`` surface.

    Rows come from ``build_aux_picker_rows()`` so this lists exactly what the ``hermes model`` aux-task picker
    lists, including user-defined ``providers:`` / ``custom_providers:`` endpoints. Persists
    ``auxiliary.vision.provider`` + ``.model``.
    """
    from hermes_cli.tools_config import _prompt_choice

    try:
        from hermes_cli.inventory import build_aux_picker_rows, format_aux_picker_entries
    except Exception as exc:  # pragma: no cover - import guard
        _print_warning(f"  Could not load provider list: {exc}")
        return

    current_provider = str(vision_cfg.get("provider") or "").strip()
    current_model = str(vision_cfg.get("model") or "").strip()
    current_base_url = str(vision_cfg.get("base_url") or "").strip()

    try:
        providers = build_aux_picker_rows(current_provider=current_provider, current_model=current_model,
                                          current_base_url=current_base_url, max_models=40)
    except Exception as exc:
        _print_warning(f"  Could not detect providers: {exc}")
        providers = []

    if not providers:
        _print_warning(
            "  No authenticated providers found. Configure a provider first "
            "with `hermes model`, then re-run this."
        )
        return

    provider_labels = [label for _slug, label, _models in format_aux_picker_entries(
        providers, current_provider=current_provider, current_base_url=current_base_url)]
    provider_labels.append("Cancel")

    pidx = _prompt_choice("  Choose vision provider:", provider_labels, 0)
    if pidx >= len(providers):
        _print_info("  Cancelled")
        return

    chosen = providers[pidx]
    slug = chosen.get("slug")
    models = list(chosen.get("models", []))
    model_choices = list(models) + ["Type a custom model id…"]
    midx = _prompt_choice(f"  Choose vision model for {chosen.get('name') or slug}:", model_choices, 0)
    if midx < len(models):
        model = models[midx]
    else:
        model = _prompt("    Model id").strip()
        if not model:
            _print_warning("  No model entered — cancelled")
            return

    vision_cfg["provider"] = slug
    vision_cfg["model"] = model
    # A provider selection supersedes any prior custom endpoint override.
    vision_cfg.pop("base_url", None)
    vision_cfg.pop("api_key", None)
    save_config(config)
    _print_success(f"  Vision set to {slug} / {model}")


def _configure_simple_requirements(ts_key: str, *, reconfigure: bool = False):
    """Fallback for toolsets that just need env vars (no provider selection).

    Vision has its own provider/model picker — run it directly so neither flow falls back to the generic
    single-key prompt (which would re-ask for OPENROUTER_API_KEY).
    """
    from hermes_cli.tools_config import TOOLSET_ENV_REQUIREMENTS, _toolset_has_keys, _toolset_label

    if ts_key == "vision":
        if reconfigure or not _toolset_has_keys("vision"):
            _configure_vision_backend()
        return

    requirements = TOOLSET_ENV_REQUIREMENTS.get(ts_key, [])
    if not reconfigure:
        requirements = [(var, url) for var, url in requirements if not get_env_value(var)]
    if not requirements:
        return

    ts_label = _toolset_label(ts_key)
    print()
    if reconfigure:
        print(color(f"  {ts_label}:", Colors.CYAN))
    else:
        print(color(f"  {ts_label} requires configuration:", Colors.YELLOW))

    for var, url in requirements:
        if reconfigure:
            existing = get_env_value(var)
            if existing:
                _print_info(f"  {var}: configured ({existing[:8]}...)")
        if url:
            _print_info(f"  Get key at: {url}")
        suffix = " (Enter to keep current)" if reconfigure else ""
        value = _prompt(f"    {var}{suffix}", password=True)
        if value and value.strip():
            save_env_value(var, value.strip())
            _print_success("    Updated" if reconfigure else "    Saved")
        elif reconfigure:
            _print_info("    Kept current")
        else:
            _print_warning("    Skipped")
