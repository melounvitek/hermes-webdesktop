"""Per-provider model-selection wizard flows for ``hermes setup`` / ``hermes model``.

Contract: ``select_provider_and_model`` in main.py re-imports every ``_model_flow_*``
here, so tests patching ``hermes_cli.main._model_flow_*`` keep working. main.py-internal
helpers (``_prompt_api_key``, ``_save_custom_provider``, ...) and config/auth/models
functions are imported lazily inside function bodies: that avoids the main.py import
cycle and lets tests patch ``hermes_cli.config.load_config`` etc. at call time.

The shared skeleton (credential prompt, persist step, OAuth gate, pickers) lives in
:mod:`hermes_cli.model_setup_flows_common`; it is re-exported here so existing imports
and ``patch("hermes_cli.model_setup_flows.<name>")`` keep resolving.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import urllib.parse

from hermes_cli.cli_output import line_input
from hermes_cli.config import clear_model_endpoint_credentials
from hermes_cli.providers import custom_provider_slug
from hermes_cli.model_setup_flows_common import (  # noqa: F401
    _activate_provider_model,
    _ask,
    _begin_model_config,
    _commit_model_config,
    _curses_choice,
    _ensure_dict_section,
    _ensure_flow_api_key,
    _existing_api_key_for_model_flow,
    _finish_model,
    _load_config_model_section,
    _models_dev_merged,
    _oauth_gate,
    _persist_model,
    _pick_model_or_prompt,
    _print_numbered,
    _prompt_auth_credentials_choice,
    _prune_replaced_custom_model_config_credentials,
    _run_login,
    _say,
    _show_curated,
)

_HTTP = ("http://", "https://")


# AWS cross-region inference profile prefixes. A geo-prefixed profile only routes
# from endpoints in its own geography (us.* from eu-central-2 is rejected by AWS
# regardless of credentials); global.* routes from everywhere.
BEDROCK_GEO_PREFIXES = ("us.", "eu.", "ap.", "apac.", "jp.", "ca.", "sa.", "me.", "af.")

# region-name prefixes -> inference-profile geo prefix
_REGION_GEO = (("us.", ("us-", "us_gov")), ("eu.", ("eu-",)), ("ap.", ("ap-",)), ("ca.", ("ca-",)),
               ("sa.", ("sa-",)), ("me.", ("me-",)), ("af.", ("af-",)))


def bedrock_region_geo_prefix(region_name: str) -> str:
    """Map an AWS region name to its inference-profile geo prefix ('' = unknown)."""
    r = (region_name or "").lower()
    return next((geo for geo, prefixes in _REGION_GEO if r.startswith(prefixes)), "")


def bedrock_model_routable_from_region(model_id: str, region_name: str) -> bool:
    """True when *model_id* can be invoked from *region_name*'s endpoint.

    Bare foundation-model ids and ``global.*`` profiles route from anywhere;
    geo-prefixed profiles only from their own geography. Unknown regions hide nothing.
    """
    mid = (model_id or "").lower()
    matched_geo = next((p for p in BEDROCK_GEO_PREFIXES if mid.startswith(p)), None)
    if matched_geo is None or mid.startswith("global."):
        return True
    geo = bedrock_region_geo_prefix(region_name)
    if not geo:
        return True
    if geo == "ap.":
        # Asia-Pacific regions can carry ap./apac./jp. profile spellings.
        return matched_geo in ("ap.", "apac.", "jp.")
    return matched_geo == geo


def _model_flow_openrouter(config, current_model=""):
    """OpenRouter provider: ensure API key, then pick model."""
    from hermes_constants import OPENROUTER_BASE_URL
    from hermes_cli.auth import ProviderConfig, _prompt_model_selection

    # OpenRouter isn't in PROVIDER_REGISTRY so we synthesize a minimal pconfig.
    pconfig = ProviderConfig(id="openrouter", name="OpenRouter", auth_type="api_key", api_key_env_vars=("OPENROUTER_API_KEY",))
    existing_key, _resolved, abort = _ensure_flow_api_key(
        "openrouter", pconfig, missing_hint=("Get one at: https://openrouter.ai/keys", "")
    )
    if abort:
        return

    from hermes_cli.models import model_ids, get_pricing_for_provider

    openrouter_models = model_ids(force_refresh=True)
    # Live pricing is non-blocking — empty dict on failure.
    pricing = get_pricing_for_provider("openrouter", force_refresh=True)
    selected = _prompt_model_selection(
        openrouter_models, current_model=current_model, pricing=pricing, confirm_provider="openrouter",
        confirm_base_url=OPENROUTER_BASE_URL, confirm_api_key=_resolved or existing_key,
    )
    _finish_model(selected, "openrouter", f"Default model set to: {selected} (via OpenRouter)",
                  base_url=OPENROUTER_BASE_URL, api_mode="chat_completions")


def _print_moa_preset(name: str, preset: dict) -> None:
    """Print the full reference-models + aggregator breakdown for a preset."""
    print(f"  Preset: {name}")
    print("  Reference models:")
    for idx, slot in enumerate(preset.get("reference_models") or [], start=1):
        print(f"    {idx}. {slot.get('provider')}:{slot.get('model')}")
    agg = preset.get("aggregator") or {}
    print(f"  Aggregator:  {agg.get('provider')}:{agg.get('model')}")


def _model_flow_ai_gateway(config, current_model=""):
    """Vercel AI Gateway provider: ensure API key, then pick model with pricing."""
    from hermes_constants import AI_GATEWAY_BASE_URL
    from hermes_cli.main import _prompt_api_key
    from hermes_cli.auth import PROVIDER_REGISTRY, _prompt_model_selection
    from hermes_cli.config import get_env_value

    pconfig = PROVIDER_REGISTRY["ai-gateway"]
    existing_key = get_env_value("AI_GATEWAY_API_KEY") or ""
    if not existing_key:
        _say("Create API key here: https://vercel.com/d?to=%2F%5Bteam%5D%2F%7E%2Fai-gateway&title=AI+Gateway",
             "Add a payment method to get $5 in free credits.", "")
    _resolved, abort = _prompt_api_key(pconfig, existing_key, provider_id="ai-gateway")
    if abort:
        return

    from hermes_cli.models import ai_gateway_model_ids, get_pricing_for_provider

    models_list = ai_gateway_model_ids(force_refresh=True)
    pricing = get_pricing_for_provider("ai-gateway", force_refresh=True)
    selected = _prompt_model_selection(models_list, current_model=current_model, pricing=pricing)
    # Inline credentials are deliberately left untouched here (historical behavior).
    _finish_model(selected, "ai-gateway", f"Default model set to: {selected} (via Vercel AI Gateway)",
                  base_url=AI_GATEWAY_BASE_URL, api_mode="chat_completions", clear_creds=False)


def _model_flow_moa(config, current_model=""):
    """Mixture of Agents virtual provider: pick a preset, then persist it.

    No credential step — presets reference already-configured providers. The preset
    list is always shown (even with one entry), then the full breakdown on selection.
    """
    from hermes_cli.auth import _save_model_choice
    from hermes_cli.moa_config import normalize_moa_config

    moa = normalize_moa_config(config.get("moa") if isinstance(config, dict) else {})
    presets = moa.get("presets") or {}
    if not presets:
        print("No MoA presets configured. Run `hermes moa configure <name>` first.")
        return

    names = list(presets.keys())
    default_name = moa.get("default_preset") or names[0]
    # Rows show the aggregator so the picker is informative before drilling in.
    rows = []
    for n in names:
        agg = presets[n].get("aggregator") or {}
        agg_label = f"{agg.get('provider')}:{agg.get('model')}" if agg else ""
        ref_count = len(presets[n].get("reference_models") or [])
        suffix = "  ← default" if n == default_name else ""
        rows.append(f"{n}  (agg {agg_label}, {ref_count} refs){suffix}")
    default_idx = names.index(default_name) if default_name in names else 0

    title = "Select a Mixture of Agents preset:"
    idx = _curses_choice(title, rows, default_idx)
    if idx is None:
        _print_numbered(title, rows, default_idx)
        raw = _ask(f"  Choice [1-{len(rows)}]: ", raw=True, cancel_msg="No change.")
        if raw is None:
            return
        try:
            idx = default_idx if not raw else max(0, min(len(rows) - 1, int(raw) - 1))
        except ValueError:
            print("No change.")
            return
    if idx < 0:
        print("No change.")
        return

    selected_name = names[idx]
    cfg, model = _load_config_model_section()
    model["default"] = selected_name
    model["provider"] = "moa"
    # Virtual local provider: drop stale endpoint credentials AND base_url (which
    # clear_model_endpoint_credentials intentionally leaves alone).
    clear_model_endpoint_credentials(model, clear_api_mode=True)
    model.pop("base_url", None)
    _commit_model_config(cfg)
    _save_model_choice(selected_name)

    _say("", f"Default model set to: {selected_name} (via Mixture of Agents)")
    _print_moa_preset(selected_name, presets[selected_name])


def _nous_login_args(args) -> argparse.Namespace:
    return argparse.Namespace(
        portal_url=getattr(args, "portal_url", None),
        inference_url=getattr(args, "inference_url", None),
        client_id=getattr(args, "client_id", None),
        scope=getattr(args, "scope", None),
        no_browser=bool(getattr(args, "no_browser", False)),
        timeout=getattr(args, "timeout", None) or 15.0,
        ca_bundle=getattr(args, "ca_bundle", None),
        insecure=bool(getattr(args, "insecure", False)),
    )


def _nous_model_catalog(free_tier: bool, portal_url: str, model_ids: list, pricing: dict):
    """Free/paid-tier catalog for the Nous picker.

    Returns ``(model_ids, pricing, unavailable_models, unavailable_message,
    policy_narrowed)`` or None (message already printed) when nothing is selectable.
    """
    from hermes_cli.models import (
        nous_policy_allowed_ids, partition_nous_models_by_tier, restrict_to_nous_policy,
        union_with_portal_free_recommendations, union_with_portal_paid_recommendations,
    )

    # Free users: augment with the Portal's freeRecommendedModels (so newly launched
    # free models appear before this build's curated list catches up), then partition
    # into selectable/unavailable by Portal pricing. Paid users: same idea with
    # paidRecommendedModels, no partition.
    unavailable_models: list[str] = []
    unavailable_message = ""
    # Org policy narrows BEFORE the tier split, so a rescued id still has to pass
    # the free/paid predicate instead of going around it.
    _policy_allowed = nous_policy_allowed_ids()
    if free_tier:
        try:
            from hermes_cli.nous_account import format_nous_portal_entitlement_message, get_nous_portal_account_info

            _account_info = get_nous_portal_account_info(force_fresh=True)
            unavailable_message = format_nous_portal_entitlement_message(_account_info, capability="paid Nous models") or ""
        except Exception:
            unavailable_message = ""
        model_ids, pricing = union_with_portal_free_recommendations(model_ids, pricing, portal_url)
    else:
        model_ids, pricing = union_with_portal_paid_recommendations(model_ids, pricing, portal_url)
    _before_policy = model_ids
    model_ids = restrict_to_nous_policy(model_ids, _policy_allowed, rescue_empty=True)
    _policy_narrowed = model_ids != _before_policy
    if free_tier:
        model_ids, unavailable_models = partition_nous_models_by_tier(model_ids, pricing, free_tier=True)

    if not model_ids and not unavailable_models:
        print("No models available for Nous Portal after filtering.")
        return None
    if free_tier and not model_ids:
        print("No free models currently available.")
        if unavailable_models:
            from hermes_cli.auth import DEFAULT_NOUS_PORTAL_URL

            _url = (portal_url or DEFAULT_NOUS_PORTAL_URL).rstrip("/")
            print(unavailable_message or f"Upgrade at {_url} to access paid models.")
        return None
    return model_ids, pricing, unavailable_models, unavailable_message, _policy_narrowed


def _model_flow_nous(config, current_model="", args=None):
    """Nous Portal provider: ensure logged in, then pick model."""
    from hermes_cli.auth import (
        get_provider_auth_state, _prompt_model_selection, _save_model_choice, _update_config_for_provider,
        resolve_nous_runtime_credentials, AuthError, format_auth_error, _login_nous, PROVIDER_REGISTRY,
    )
    from hermes_cli.config import get_env_value, load_config, save_config, save_env_value
    from hermes_cli.nous_subscription import prompt_enable_tool_gateway

    state = get_provider_auth_state("nous")
    if not state or not state.get("access_token"):
        _say("Not logged into Nous Portal. Starting login...", "")

        def _login_then_offer_gateway(login_args, pconfig):
            _login_nous(login_args, pconfig)
            # Offer Tool Gateway enablement for paid subscribers
            try:
                prompt_enable_tool_gateway(load_config() or {})
            except Exception:
                pass

        # login_nous already handles model selection + config update
        _run_login(_login_then_offer_gateway, _nous_login_args(args), PROVIDER_REGISTRY["nous"])
        return

    # Already logged in — the curated list (agentic models users know from
    # OpenRouter) instead of the hundreds returned by the live /models endpoint.
    from hermes_cli.models import check_nous_free_tier, get_curated_nous_model_ids, get_pricing_for_provider

    model_ids = get_curated_nous_model_ids()
    if not model_ids:
        print("No curated models available for Nous Portal.")
        return

    # Verify credentials are still valid (catches expired sessions early)
    try:
        creds = resolve_nous_runtime_credentials()
    except Exception as exc:
        relogin = isinstance(exc, AuthError) and exc.relogin_required
        msg = format_auth_error(exc) if isinstance(exc, AuthError) else str(exc)
        if relogin:
            _say(f"Session expired: {msg}", "Re-authenticating with Nous Portal...\n")
            try:
                _login_nous(_nous_login_args(None), PROVIDER_REGISTRY["nous"])
            except Exception as login_exc:
                print(f"Re-login failed: {login_exc}")
            return
        print(f"Could not verify credentials: {msg}")
        return

    pricing = get_pricing_for_provider("nous")
    # Force fresh account data so recent credit purchases are reflected immediately.
    free_tier = check_nous_free_tier(force_fresh=True)
    if not free_tier:
        try:
            creds = resolve_nous_runtime_credentials(force_refresh=True) or creds
        except Exception:
            # Runtime inference has its own paid-entitlement recovery; don't block.
            pass

    # Portal URL is needed for upgrade links and the recommendations endpoints.
    _nous_portal_url = ""
    try:
        _nous_state = get_provider_auth_state("nous")
        if _nous_state:
            _nous_portal_url = _nous_state.get("portal_base_url", "")
    except Exception:
        pass

    catalog = _nous_model_catalog(free_tier, _nous_portal_url, model_ids, pricing)
    if catalog is None:
        return
    model_ids, pricing, unavailable_models, unavailable_message, _policy_narrowed = catalog

    from hermes_cli.nous_account import nous_policy_notice

    _policy_notice = nous_policy_notice(removed=_policy_narrowed)
    if _policy_notice:
        print(_policy_notice)
    print(f'Showing {len(model_ids)} curated models — use "Enter custom model name" for others.')

    selected = _prompt_model_selection(
        model_ids, current_model=current_model, pricing=pricing, unavailable_models=unavailable_models,
        portal_url=_nous_portal_url, unavailable_message=unavailable_message, confirm_provider="nous",
        confirm_base_url=creds.get("base_url", ""), confirm_api_key=creds.get("api_key", ""),
    )
    if not selected:
        print("No change.")
        return
    _save_model_choice(selected)
    inference_url = creds.get("base_url", "")
    _update_config_for_provider("nous", inference_url)
    # Reload after the auth helper writes provider state; the incoming config
    # object may still contain stale custom-provider fields.
    config = load_config()
    current_model_cfg = config.get("model")
    if isinstance(current_model_cfg, dict):
        model_cfg = dict(current_model_cfg)
    elif isinstance(current_model_cfg, str) and current_model_cfg.strip():
        model_cfg = {"default": current_model_cfg.strip()}
    else:
        model_cfg = {}
    model_cfg["provider"] = "nous"
    model_cfg["default"] = selected
    if inference_url and inference_url.strip():
        model_cfg["base_url"] = inference_url.rstrip("/")
    else:
        model_cfg.pop("base_url", None)
    clear_model_endpoint_credentials(model_cfg)
    config["model"] = model_cfg
    # Clear any custom endpoint that might conflict
    if get_env_value("OPENAI_BASE_URL"):
        save_env_value("OPENAI_BASE_URL", "")
        save_env_value("OPENAI_API_KEY", "")
    save_config(config)
    print(f"Default model set to: {selected} (via Nous Portal)")
    # Offer Tool Gateway enablement for paid subscribers
    prompt_enable_tool_gateway(config)


def _model_flow_openai_codex(config, current_model=""):
    """OpenAI Codex provider: ensure logged in, then pick model."""
    from hermes_cli.auth import (
        get_codex_auth_status, _prompt_model_selection, _login_openai_codex, PROVIDER_REGISTRY, DEFAULT_CODEX_BASE_URL,
    )
    from hermes_cli.codex_models import get_codex_model_ids

    if not _oauth_gate(
        bool(get_codex_auth_status().get("logged_in")), "OpenAI Codex", _login_openai_codex, argparse.Namespace(),
        PROVIDER_REGISTRY["openai-codex"], recheck=lambda: get_codex_auth_status().get("logged_in"),
    ):
        return

    # Prefer the credential pool (where `hermes auth` stores device_code tokens),
    # fall back to legacy provider state.
    _codex_token = None
    try:
        _codex_status = get_codex_auth_status()
        if _codex_status.get("logged_in"):
            _codex_token = _codex_status.get("api_key")
    except Exception:
        pass
    if not _codex_token:
        try:
            from hermes_cli.auth import resolve_codex_runtime_credentials

            _codex_token = resolve_codex_runtime_credentials().get("api_key")
        except Exception:
            pass

    codex_models = get_codex_model_ids(access_token=_codex_token)
    selected = _prompt_model_selection(
        codex_models, current_model=current_model, confirm_provider="openai-codex",
        confirm_base_url=DEFAULT_CODEX_BASE_URL, confirm_api_key=_codex_token or "",
    )
    _activate_provider_model(selected, "openai-codex", DEFAULT_CODEX_BASE_URL,
                             f"Default model set to: {selected} (via OpenAI Codex)")


def _model_flow_xai_oauth(_config, current_model="", *, args=None):
    """xAI Grok OAuth (SuperGrok / Premium+) provider: ensure logged in, then pick model."""
    from hermes_cli.auth import (
        get_xai_oauth_auth_status, _prompt_model_selection, resolve_xai_oauth_runtime_credentials, _login_xai_oauth,
        DEFAULT_XAI_OAUTH_BASE_URL, PROVIDER_REGISTRY,
    )
    from hermes_cli.models import provider_model_ids

    login_args = argparse.Namespace(no_browser=bool(getattr(args, "no_browser", False)), timeout=getattr(args, "timeout", None))
    if not _oauth_gate(
        bool(get_xai_oauth_auth_status().get("logged_in")), "xAI Grok OAuth (SuperGrok / Premium+)", _login_xai_oauth,
        login_args, PROVIDER_REGISTRY["xai-oauth"], fresh_name="xAI OAuth",
    ):
        return

    # ``resolve_xai_oauth_runtime_credentials`` only reads the auth.json singleton,
    # but credentials may live only in the pool (``hermes auth add xai-oauth``) —
    # fall back to the default base URL so the picker still completes.
    base_url = DEFAULT_XAI_OAUTH_BASE_URL
    try:
        creds = resolve_xai_oauth_runtime_credentials()
        base_url = (creds.get("base_url") or "").strip().rstrip("/") or base_url
    except Exception:
        pass

    models = provider_model_ids("xai-oauth")
    selected = _prompt_model_selection(models, current_model=current_model or (models[0] if models else "grok-4.6"))
    _activate_provider_model(selected, "xai-oauth", base_url,
                             f"Default model set to: {selected} (via xAI Grok OAuth — SuperGrok / Premium+)")


def _model_flow_qwen_oauth(_config, current_model=""):
    """Qwen OAuth provider: reuse local Qwen CLI login, then pick model."""
    from hermes_cli.main import _DEFAULT_QWEN_PORTAL_MODELS
    from hermes_cli.auth import (
        get_qwen_auth_status, resolve_qwen_runtime_credentials, _prompt_model_selection, DEFAULT_QWEN_BASE_URL,
    )
    from hermes_cli.models import fetch_api_models

    status = get_qwen_auth_status()
    if not status.get("logged_in"):
        _say("Not logged into Qwen CLI OAuth.", "Run: qwen auth qwen-oauth")
        if status.get("auth_file"):
            print(f"Expected credentials file: {status.get('auth_file')}")
        if status.get("error"):
            print(f"Error: {status.get('error')}")
        return

    # Try live model discovery, fall back to curated list.
    models = None
    try:
        creds = resolve_qwen_runtime_credentials(refresh_if_expiring=True)
        models = fetch_api_models(creds["api_key"], creds["base_url"])
    except Exception:
        pass
    if not models:
        models = list(_DEFAULT_QWEN_PORTAL_MODELS)

    default = current_model or (models[0] if models else "qwen3-coder-plus")
    selected = _prompt_model_selection(models, current_model=default, confirm_provider="qwen-oauth", confirm_base_url=DEFAULT_QWEN_BASE_URL)
    _activate_provider_model(selected, "qwen-oauth", DEFAULT_QWEN_BASE_URL, f"Default model set to: {selected} (via Qwen OAuth)")


def _model_flow_minimax_oauth(config, current_model="", args=None):
    """MiniMax OAuth provider: ensure logged in, then pick model."""
    from hermes_cli.auth import (
        get_provider_auth_state, _prompt_model_selection, resolve_minimax_oauth_runtime_credentials, AuthError,
        format_auth_error, _login_minimax_oauth, PROVIDER_REGISTRY,
    )

    state = get_provider_auth_state("minimax-oauth")
    if not state or not state.get("access_token"):
        _say("Not logged into MiniMax. Starting OAuth login...", "")
        mock_args = argparse.Namespace(
            region=getattr(args, "region", None) or "global",
            no_browser=bool(getattr(args, "no_browser", False)),
            timeout=getattr(args, "timeout", None) or 15.0,
        )
        if not _run_login(_login_minimax_oauth, mock_args, PROVIDER_REGISTRY["minimax-oauth"]):
            return

    try:
        creds = resolve_minimax_oauth_runtime_credentials()
    except AuthError as exc:
        print(format_auth_error(exc))
        return

    from hermes_cli.models import _PROVIDER_MODELS

    model_ids = _PROVIDER_MODELS.get("minimax-oauth", [])
    selected = _prompt_model_selection(model_ids, current_model, confirm_provider="minimax-oauth", confirm_base_url=creds["base_url"])
    _activate_provider_model(selected, "minimax-oauth", creds["base_url"], f"\u2713 Using MiniMax model: {selected}", no_change=None)


def _parse_context_length(text: str):
    """``128k`` / ``128,000`` -> int; None when blank, non-positive, or unparsable (warns)."""
    if not text:
        return None
    try:
        value = int(text.replace(",", "").replace("k", "000").replace("K", "000"))
    except ValueError:
        print(f"Invalid context length: {text} — will auto-detect.")
        return None
    return value if value > 0 else None


def _probe_custom_endpoint(effective_key: str, effective_url: str) -> tuple[dict, str]:
    """Verify a custom endpoint via ``probe_api_models`` and report; returns
    ``(probe, effective_url)`` where the URL may be the working fallback base."""
    from hermes_cli.models import probe_api_models

    probe = probe_api_models(effective_key, effective_url)
    if probe.get("used_fallback") and probe.get("resolved_base_url"):
        print(f"Warning: endpoint verification worked at {probe['resolved_base_url']}/models, "
              f"not the exact URL you entered. Saving the working base URL instead.")
        effective_url = probe["resolved_base_url"]
    elif probe.get("models") is not None:
        print(f"Verified endpoint via {probe.get('probed_url')} ({len(probe.get('models') or [])} model(s) visible)")
    else:
        print(f"Warning: could not verify this endpoint via {probe.get('probed_url')}. Hermes will still save it.")
        suggested = probe.get("suggested_base_url")
        if suggested and suggested.endswith("/v1"):
            print(f"  If this server expects /v1 in the path, try base URL: {suggested}")
        elif suggested:
            print(f"  If /v1 should not be in the base URL, try: {suggested}")
    return probe, effective_url


def _pick_detected_model(detected_models: list) -> str:
    """Model-name step of the custom flow: confirm a single detection, number-pick from
    several, or type one. Raises KeyboardInterrupt/EOFError like the prompts it wraps."""
    manual = "Model name (e.g. gpt-4, llama-3-70b): "
    if len(detected_models) == 1:
        print(f"  Detected model: {detected_models[0]}")
        if input("  Use this model? [Y/n]: ").strip().lower() in {"", "y", "yes"}:
            return detected_models[0]
        return line_input(manual).strip()
    if len(detected_models) > 1:
        print("  Available models:")
        for i, m in enumerate(detected_models, 1):
            print(f"    {i}. {m}")
        pick = input(f"  Select model [1-{len(detected_models)}] or type name: ").strip()
        if pick.isdigit() and 1 <= int(pick) <= len(detected_models):
            return detected_models[int(pick) - 1]
        return pick
    return line_input(manual).strip()


def _model_flow_custom(config):
    """Custom endpoint: collect URL, API key, and model name.

    Also saves the endpoint to ``custom_providers`` in config.yaml so it appears
    in the provider menu on subsequent runs.
    """
    from hermes_cli.main import _auto_provider_name, _prompt_custom_api_mode_selection, _save_custom_provider
    from hermes_cli.auth import _save_model_choice, deactivate_provider
    from hermes_cli.config import custom_endpoint_key_env, get_env_value, save_env_value
    from hermes_cli.secret_prompt import masked_secret_prompt

    current_url = get_env_value("OPENAI_BASE_URL") or ""
    current_key = get_env_value("OPENAI_API_KEY") or ""

    print("Custom OpenAI-compatible endpoint configuration:")
    if current_url:
        print(f"  Current URL: {current_url}")
    if current_key:
        print(f"  Current key: {current_key[:8]}...")
    print()

    try:
        base_url = line_input(f"API base URL [{current_url or 'e.g. https://api.example.com/v1'}]: ").strip()
        api_key = masked_secret_prompt(f"API key [{current_key[:8] + '...' if current_key else 'optional'}]: ").strip()
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return

    if not base_url and not current_url:
        print("No URL provided. Cancelled.")
        return
    effective_url = base_url or current_url
    if not effective_url.startswith(_HTTP):
        print(f"Invalid URL: {effective_url} (must start with http:// or https://)")
        return
    effective_key = api_key or current_key

    # Most local servers (Ollama, vLLM, llama.cpp) need /v1 for OpenAI-compatible
    # chat completions — offer to append it when the URL looks local without it.
    _url_lower = effective_url.rstrip("/").lower()
    _looks_local = any(h in _url_lower for h in ("localhost", "127.0.0.1", "0.0.0.0", ":11434", ":8080", ":5000"))
    if _looks_local and not _url_lower.endswith("/v1"):
        _say("", "  Hint: Did you mean to add /v1 at the end?",
             "  Most local model servers (Ollama, vLLM, llama.cpp) require it.", f"  e.g. {effective_url.rstrip('/')}/v1")
        if _ask("  Add /v1? [Y/n]: ", raw=True, cancel_msg=None, on_cancel="n").lower() in {"", "y", "yes"}:
            effective_url = effective_url.rstrip("/") + "/v1"
            print(f"  Updated URL: {effective_url}")
        print()

    probe, effective_url = _probe_custom_endpoint(effective_key, effective_url)

    # Ask for the API mode explicitly so codex-compatible custom providers don't
    # silently fall back to chat_completions.
    current_model_cfg = config.get("model")
    current_api_mode = str(current_model_cfg.get("api_mode") or "").strip() if isinstance(current_model_cfg, dict) else ""
    api_mode = _prompt_custom_api_mode_selection(effective_url, current_api_mode=current_api_mode)
    print(f"  API mode: {api_mode}" if api_mode else "  API mode: auto-detect")

    # Select model — use probe results when available, fall back to manual input
    try:
        model_name = _pick_detected_model(probe.get("models") or [])
        context_length_str = line_input("Context length in tokens [leave blank for auto-detect]: ").strip()
        # Display name — shown in the provider menu on future runs
        default_name = _auto_provider_name(effective_url)
        display_name = line_input(f"Display name [{default_name}]: ").strip() or default_name
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled.")
        return
    context_length = _parse_context_length(context_length_str)

    # The key goes to .env and config.yaml only references it. Keyed on host:port
    # so two servers on one machine keep separate credentials.
    custom_key_env = ""
    if effective_key:
        _parsed = urllib.parse.urlparse(effective_url)
        _identity = _parsed.hostname or ""
        if _parsed.port:
            _identity = f"{_identity}_{_parsed.port}"
        custom_key_env = custom_endpoint_key_env(_identity)
        save_env_value(custom_key_env, effective_key)
        print(f"  API key saved to .env as {custom_key_env}")

    def _apply_endpoint(model: dict) -> None:
        model["provider"] = "custom"
        model["base_url"] = effective_url
        if custom_key_env:
            model["api_key"] = f"${{{custom_key_env}}}"
        if api_mode:
            model["api_mode"] = api_mode
        else:
            model.pop("api_mode", None)

    if model_name:
        _save_model_choice(model_name)
        cfg, model = _load_config_model_section()
        _apply_endpoint(model)
        _commit_model_config(cfg)
        # Sync the caller's config dict so the setup wizard's final save_config(config)
        # doesn't overwrite model.provider/base_url with its stale values.
        config["model"] = dict(model)
        print(f"Default model set to: {model_name} (via {effective_url})")
    else:
        if base_url or api_key:
            deactivate_provider()
        # Even without a model name, persist the endpoint on the caller's config dict.
        _caller_model = config.get("model")
        if not isinstance(_caller_model, dict):
            _caller_model = {"default": _caller_model} if _caller_model else {}
        _apply_endpoint(_caller_model)
        config["model"] = _caller_model
        print("Endpoint saved. Use `/model` in chat or `hermes model` to set a model.")

    # Auto-save to custom_providers so it appears in the menu next time
    _save_custom_provider(effective_url, effective_key, model_name or "", context_length=context_length,
                          name=display_name, api_mode=api_mode, key_env=custom_key_env)
    _prune_replaced_custom_model_config_credentials(effective_url, provider_name=display_name)


def _azure_mode_label(mode: str) -> str:
    return "OpenAI-style" if mode == "chat_completions" else "Anthropic-style"


def _azure_entra_preflight(current_entra: dict):
    """Entra ID credential preflight for the Azure flow. Returns
    ``(token_provider, entra_overrides)``; ``None`` when the user cancelled;
    ``False`` when the adapter is missing (caller falls back to API-key auth)."""
    try:
        from agent.azure_identity_adapter import (
            EntraIdentityConfig, SCOPE_AI_AZURE_DEFAULT, build_token_provider, describe_active_credential,
            has_azure_identity_installed,
        )
    except ImportError as exc:
        _say("", f"⚠ Could not import azure-identity adapter: {exc}", "  Falling back to API key auth.")
        return False

    print()
    if not has_azure_identity_installed():
        _say("◐ The 'azure-identity' package is not installed yet.",
             "  Hermes will install it now (the preflight below triggers the lazy-install). "
             "To skip lazy installs, run:  pip install azure-identity")

    # Only the optional scope override is persisted; identity selection (tenant,
    # user-assigned MI, workload identity, SP) stays in AZURE_* SDK env vars.
    entra_overrides: dict = {}
    _persisted_scope_override = str(current_entra.get("scope") or "").strip()
    entra_scope = _persisted_scope_override or SCOPE_AI_AZURE_DEFAULT
    if _persisted_scope_override:
        entra_overrides["scope"] = _persisted_scope_override

    _say("", "◐ Probing Microsoft Entra ID credential chain (up to 10s)...")
    _config = EntraIdentityConfig(scope=entra_scope)
    info = describe_active_credential(config=_config, timeout_seconds=10.0)
    if info.get("ok"):
        env_sources = info.get("env_sources") or []
        tag = ", ".join(env_sources) if env_sources else "default chain"
        print(f"✓ Entra ID token acquired ({tag}, scope={entra_scope})")
    else:
        err = info.get("error") or "credential chain exhausted"
        hint = info.get("hint") or (
            "Run `az login`, attach a managed identity to this VM, or set AZURE_TENANT_ID/AZURE_CLIENT_ID/AZURE_CLIENT_SECRET."
        )
        _say(f"⚠ {err}", f"  Hint: {hint}")
        ans = _ask("Save Entra config anyway and validate later? [Y/n]: ", raw=True)
        if ans is None:
            return None
        if ans.lower() not in ("", "y", "yes"):
            print("Cancelled.")
            return None

    # Best-effort token provider for the detection probe; on failure the probe
    # falls back to manual entry.
    try:
        token_provider = build_token_provider(config=_config)
    except Exception as exc:
        print(f"⚠ Could not build token provider for probing: {exc}")
        token_provider = None
    return token_provider, entra_overrides


def _azure_pick_model(discovered_models: list, current_model: str):
    """Model/deployment step of the Azure flow; None when cancelled."""
    if not discovered_models:
        model_name = _ask(f"Model / deployment name [{current_model or 'e.g. gpt-5.4, claude-sonnet-4-6'}]: ")
        return None if model_name is None else (model_name or current_model)
    print("Available models on this endpoint:")
    for i, mid in enumerate(discovered_models[:30], start=1):
        print(f"  {i:>2}. {mid}")
    if len(discovered_models) > 30:
        print(f"  ... and {len(discovered_models) - 30} more (type name manually if not shown)")
    print()
    pick = _ask(f"Pick by number, or type a deployment name [{current_model or discovered_models[0]}]: ", raw=True)
    if pick is None:
        return None
    if not pick:
        return current_model or discovered_models[0]
    if pick.isdigit() and 1 <= int(pick) <= min(len(discovered_models), 30):
        return discovered_models[int(pick) - 1]
    return pick


def _model_flow_azure_foundry(config, current_model=""):
    """Azure Foundry provider: configure endpoint, auth mode, API mode, and model.

    Two transports (OpenAI-style ``/v1/chat/completions``, Anthropic-style
    ``/v1/messages``) and two auth modes: **API key** (``AZURE_FOUNDRY_API_KEY``) or
    **Microsoft Entra ID** (keyless RBAC via ``azure-identity``; the same ``Azure AI
    User`` role covers both transports — OpenAI SDK takes a callable ``api_key``,
    Anthropic gets a bearer-injecting ``httpx.Client`` from
    :func:`agent.azure_identity_adapter.build_bearer_http_client`).

    Detection order: ``/anthropic`` URL suffix → Anthropic; ``GET <base>/models``
    success → OpenAI-style + model picker; Anthropic Messages probe; manual entry.
    Context length resolves via :func:`agent.model_metadata.get_model_context_length`.
    """
    from hermes_cli.config import get_env_value, save_env_value
    from hermes_cli import azure_detect

    # ── Load current Azure Foundry configuration ─────────────────────
    model_cfg = config.get("model", {})
    current_base_url = current_api_mode = ""
    current_auth_mode, current_entra = "api_key", {}
    if isinstance(model_cfg, dict) and model_cfg.get("provider") == "azure-foundry":
        current_base_url = str(model_cfg.get("base_url", "") or "")
        current_api_mode = str(model_cfg.get("api_mode", "") or "")
        current_auth_mode = str(model_cfg.get("auth_mode") or "api_key").strip().lower() or "api_key"
        _cur_entra = model_cfg.get("entra") or {}
        current_entra = _cur_entra if isinstance(_cur_entra, dict) else {}
    current_api_key = get_env_value("AZURE_FOUNDRY_API_KEY") or ""

    _say("", "Azure Foundry Configuration", "=" * 50, "",
         "Azure Foundry can host models with either OpenAI-style or",
         "Anthropic-style API endpoints.  Hermes will probe your",
         "endpoint to auto-detect the transport and the deployed",
         "models when possible.", "")
    if current_base_url:
        print(f"  Current endpoint:  {current_base_url}")
    if current_api_mode:
        print(f"  Current API mode:  {_azure_mode_label(current_api_mode)}")
    if current_auth_mode == "entra_id":
        print("  Current auth mode: Microsoft Entra ID (keyless)")
    elif current_api_key:
        print(f"  Current auth mode: API key ({current_api_key[:8]}...)")
    print()

    # ── Step 1: endpoint URL ─────────────────────────────────────────
    _placeholder = current_base_url or (
        "e.g. https://<resource>.openai.azure.com/openai/v1 or https://<resource>.services.ai.azure.com/anthropic"
    )
    base_url = _ask(f"API endpoint URL [{_placeholder}]: ")
    if base_url is None:
        return
    effective_url = (base_url or current_base_url).rstrip("/")
    if not effective_url:
        print("No endpoint URL provided. Cancelled.")
        return
    if not effective_url.startswith(_HTTP):
        print(f"Invalid URL: {effective_url} (must start with http:// or https://)")
        return

    # ── Step 2: authentication mode ──────────────────────────────────
    _say("", "Authentication:", "  1. API key                  (AZURE_FOUNDRY_API_KEY in .env)",
         "  2. Microsoft Entra ID       (managed identity / workload identity / az login)",
         "     Recommended by Microsoft. Works for both OpenAI-style and Anthropic-style endpoints.",
         "     Requires the 'Azure AI User' role on the Foundry resource.")
    _auth_default = "2" if current_auth_mode == "entra_id" else "1"
    auth_choice = _ask(f"Authentication mode [1/2] ({_auth_default}): ", raw=True)
    if auth_choice is None:
        return
    use_entra = (auth_choice or _auth_default) == "2"

    # ── Step 3: credentials (key OR Entra preflight) ─────────────────
    effective_key: str = ""
    entra_overrides: dict = {}
    token_provider = None  # callable when entra
    if use_entra:
        preflight = _azure_entra_preflight(current_entra)
        if preflight is None:
            return
        if preflight is False:
            use_entra = False
        else:
            token_provider, entra_overrides = preflight
    if not use_entra:
        print()
        api_key = _ask(f"API key [{current_api_key[:8] + '...' if current_api_key else 'required'}]: ", secret=True)
        if api_key is None:
            return
        effective_key = api_key or current_api_key
        if not effective_key:
            print("No API key provided. Cancelled.")
            return

    # ── Step 4: auto-detect transport + models ───────────────────────
    _say("", "◐ Probing endpoint to auto-detect transport and models...")
    detection = azure_detect.detect(effective_url, api_key=effective_key, token_provider=token_provider)
    discovered_models: list[str] = list(detection.models)
    api_mode: str = detection.api_mode or ""
    if api_mode:
        print(f"✓ Detected API transport: {_azure_mode_label(api_mode)}")
        if detection.reason:
            print(f"    ({detection.reason})")
        if discovered_models:
            print(f"✓ Found {len(discovered_models)} deployed model(s) on this endpoint")
    else:
        _say(f"⚠ Auto-detection incomplete: {detection.reason}", "",
             "Select the API format your Azure Foundry endpoint uses:",
             "  1. OpenAI-style  (POST /v1/chat/completions)",
             "     For: GPT models, Llama, Mistral, and most open models",
             "  2. Anthropic-style  (POST /v1/messages)",
             "     For: Claude models deployed via Anthropic API format")
        default_choice = "2" if current_api_mode == "anthropic_messages" else "1"
        mode_choice = _ask(f"API format [1/2] ({default_choice}): ", raw=True)
        if mode_choice is None:
            return
        api_mode = "anthropic_messages" if (mode_choice or default_choice) == "2" else "chat_completions"

    # ── Step 5: model name ───────────────────────────────────────────
    print()
    effective_model = _azure_pick_model(discovered_models, current_model)
    if effective_model is None:
        return
    if not effective_model:
        print("No model name provided. Cancelled.")
        return

    # ── Step 6: context-length lookup ────────────────────────────────
    ctx_len = azure_detect.lookup_context_length(effective_model, effective_url, api_key=effective_key, token_provider=token_provider)

    # ── Step 7: persist ──────────────────────────────────────────────
    if not use_entra:
        save_env_value("AZURE_FOUNDRY_API_KEY", effective_key)
    cfg, model = _load_config_model_section()
    model["provider"] = "azure-foundry"
    model["base_url"] = effective_url
    model["api_mode"] = api_mode
    model["default"] = effective_model
    model["auth_mode"] = "entra_id" if use_entra else "api_key"
    clear_model_endpoint_credentials(model, clear_api_mode=False)
    # Persist only a non-default Entra scope so config.yaml stays tidy.
    clean_entra = {k: v for k in ("scope",) if (v := entra_overrides.get(k))}
    if use_entra and clean_entra:
        model["entra"] = clean_entra
    else:
        model.pop("entra", None)
    if ctx_len:
        model["context_length"] = ctx_len
    _commit_model_config(cfg)
    config["model"] = dict(model)

    # Clear conflicting env vars so auxiliary clients don't pick up a stale
    # OpenAI base URL / key.
    for var in ("OPENAI_BASE_URL", "OPENAI_API_KEY"):
        if get_env_value(var):
            save_env_value(var, "")

    _say("", "✓ Azure Foundry configured:", f"    Endpoint:       {effective_url}",
         f"    API mode:       {_azure_mode_label(api_mode)}",
         f"    Auth:           {'Microsoft Entra ID (keyless)' if use_entra else 'API key'}",
         f"    Model:          {effective_model}",
         f"    Context length: {ctx_len:,} tokens" if ctx_len else "    Context length: not auto-detected (will fall back at runtime)",
         "")


def _configured_model_ids(cfg_models) -> list[str]:
    """Model ids from a ``custom_providers[].models`` mapping or list (marker keys skipped)."""
    if isinstance(cfg_models, dict):
        markers = {"__explicit_model_allowlist__", "__discovered_model_catalog__"}
        return [str(m) for m in cfg_models if m not in markers and str(m).strip()]
    out: list[str] = []
    if isinstance(cfg_models, list):
        for entry in cfg_models:
            if isinstance(entry, dict):
                model_id = str(entry.get("id") or entry.get("model") or "").strip()
            else:
                model_id = str(entry).strip() if isinstance(entry, str) else ""
            if model_id:
                out.append(model_id)
    return out


def _discover_named_custom_models(provider_info: dict, api_key: str, configured_models: list, explicit_catalog: bool):
    """Live catalog probe for a named custom endpoint (native ``/api/tags`` for Ollama).
    Returns ``(models, native_catalog_empty)``; persists the live catalog as a side effect."""
    from hermes_cli.config import normalize_extra_headers
    from hermes_cli.models import (
        fetch_api_models, fetch_ollama_local_models, _get_ollama_native_headers, _normalize_openai_base_url,
        should_use_ollama_native_catalog,
    )

    name, base_url = provider_info["name"], provider_info["base_url"]
    api_mode = provider_info.get("api_mode", "")
    provider_key = (provider_info.get("provider_key") or "").strip()
    print("Fetching available models...")
    fetch_kwargs = {"timeout": 8.0}
    if api_mode:
        fetch_kwargs["api_mode"] = api_mode
    native_catalog_provider = "ollama" if provider_key.lower() == "ollama" or name.strip().lower() == "ollama" else "custom"
    extra_headers = normalize_extra_headers(provider_info.get("extra_headers")) or {}
    candidate_headers = _get_ollama_native_headers(base_url, api_key=api_key)
    for key in tuple(candidate_headers):
        if any(key.lower() == existing.lower() for existing in extra_headers):
            del candidate_headers[key]
    candidate_headers.update(extra_headers)
    caller_has_authorization = any(key.lower() == "authorization" for key in extra_headers)
    if api_key and not caller_has_authorization:
        for key in tuple(candidate_headers):
            if key.lower() == "authorization":
                del candidate_headers[key]
        candidate_headers["Authorization"] = f"Bearer {api_key}"
    use_native = should_use_ollama_native_catalog(native_catalog_provider, base_url, headers=candidate_headers or None)
    native_headers_arg = candidate_headers or None if use_native else (extra_headers or None)
    native_catalog_empty = False
    if use_native:
        if explicit_catalog and configured_models:
            live_models = configured_models
        else:
            live_models = fetch_ollama_local_models(base_url, timeout=8.0, headers=native_headers_arg)
            native_catalog_empty = live_models == []
            if live_models is None:
                live_models = fetch_api_models(api_key, _normalize_openai_base_url(base_url), headers=native_headers_arg, **fetch_kwargs)
                native_catalog_empty = False
    else:
        live_models = fetch_api_models(api_key, base_url, headers=native_headers_arg, **fetch_kwargs)
    models = configured_models if explicit_catalog else [] if native_catalog_empty else (live_models or configured_models)
    # Persist the live catalog to the custom_providers entry so no-probe surfaces
    # (dashboard, desktop, ACP) show the full list; mirrors model_switch.py's
    # _save_discovered_models_to_config. A failed save is non-fatal.
    if live_models:
        try:
            from hermes_cli.model_switch import _save_discovered_models_to_config

            _save_discovered_models_to_config(base_url, live_models, api_mode=api_mode, headers=extra_headers or None)
        except Exception:
            pass
    return models, native_catalog_empty


def _pick_named_custom_model(name: str, models: list, saved_model: str):
    """Searchable radiolist over *models* (numbered prompt without curses); None = cancelled."""
    default_idx = models.index(saved_model) if saved_model and saved_model in models else 0
    print(f"Found {len(models)} model(s):\n")
    try:
        from hermes_cli.curses_ui import curses_radiolist

        menu_items = [f"{m} (current)" if m == saved_model else m for m in models] + ["Cancel"]
        idx = curses_radiolist(f"Select model from {name}:", menu_items, selected=default_idx, cancel_returns=-1, searchable=True)
        print()
    except (ImportError, NotImplementedError, OSError, subprocess.SubprocessError):
        for i, m in enumerate(models, 1):
            print(f"  {i}. {m}{' (current)' if m == saved_model else ''}")
        _say(f"  {len(models) + 1}. Cancel", "")
        try:
            val = input(f"Choice [1-{len(models) + 1}]: ").strip()
            if not val:
                print("Cancelled.")
                return None
            idx = int(val) - 1
        except (ValueError, KeyboardInterrupt, EOFError):
            print("\nCancelled.")
            return None
    if idx < 0 or idx >= len(models):
        print("Cancelled.")
        return None
    return models[idx]


def _model_flow_named_custom(config, provider_info):
    """Handle a named custom provider from config.yaml custom_providers list.

    Probes the endpoint's model catalog (native ``/api/tags`` for endpoints
    conservatively identified as Ollama); a previously saved model is pre-selected
    and used as the fallback when probing fails.
    """
    from hermes_cli.main import _custom_provider_api_key_config_value, _custom_provider_base_url_config_value, _save_custom_provider
    from hermes_cli.auth import _save_model_choice
    from hermes_cli.config import load_config, save_config
    from hermes_cli.model_switch import _entry_models_discovered, _models_config_is_allowlist

    name = provider_info["name"]
    base_url = provider_info["base_url"]
    api_mode = provider_info.get("api_mode", "")
    api_key = provider_info.get("api_key", "")
    key_env = provider_info.get("key_env", "")
    saved_model = provider_info.get("model", "")
    provider_key = (provider_info.get("provider_key") or "").strip()

    # Resolve key from env var if api_key not set directly
    if not api_key and key_env:
        api_key = os.environ.get(key_env, "")
    config_api_key = _custom_provider_api_key_config_value(provider_info, api_key)

    # ``discover_models: false`` (default True) uses the configured ``models:`` list
    # verbatim and skips the live probe, so operators can restrict the picker to the
    # subset their plan serves. Same semantics as the slash-command picker.
    discover = provider_info.get("discover_models", True)
    if isinstance(discover, str):
        discover = discover.lower() not in {"false", "no", "0"}
    cfg_models = provider_info.get("models", {})
    explicit_catalog = _models_config_is_allowlist(cfg_models, _entry_models_discovered(provider_info))
    configured_models = _configured_model_ids(cfg_models)

    print(f"  Provider: {name}")
    print(f"  URL:      {base_url}")
    if saved_model:
        print(f"  Current:  {saved_model}")
    print()

    native_catalog_empty = False
    if not discover:
        # Never probe. The active model is a usable sole choice, not a catalog.
        models = configured_models or ([saved_model] if saved_model else [])
        print(f"Using configured models (discover_models: false): {len(models)}")
    else:
        models, native_catalog_empty = _discover_named_custom_models(provider_info, api_key, configured_models, explicit_catalog)

    if models:
        model_name = _pick_named_custom_model(name, models, saved_model)
        if model_name is None:
            return
    elif saved_model and not native_catalog_empty:
        print("Could not fetch models from endpoint.")
        model_name = _ask(f"Model name [{saved_model}]: ")
        if model_name is None:
            return
        model_name = model_name or saved_model
    else:
        print("Could not fetch models from endpoint. Enter model name manually.")
        model_name = _ask("Model name: ")
        if model_name is None:
            return
        if not model_name:
            print("No model specified. Cancelled.")
            return

    # Activate and save the model to the custom_providers entry
    _save_model_choice(model_name)
    cfg, model = _load_config_model_section()
    if provider_key:
        model["provider"] = custom_provider_slug(name, provider_key)
        model.pop("base_url", None)
        model.pop("api_key", None)
    else:
        model["provider"] = "custom"
        model["base_url"] = _custom_provider_base_url_config_value(provider_info, base_url)
        if config_api_key:
            model["api_key"] = config_api_key
    # Apply api_mode from custom_providers entry, or clear stale value
    if api_mode:
        model["api_mode"] = api_mode
    else:
        model.pop("api_mode", None)  # let runtime auto-detect from URL
    _commit_model_config(cfg)

    # Persist the selected model back to whichever schema owns this endpoint.
    if provider_key:
        cfg = load_config()
        providers_cfg = cfg.get("providers")
        provider_entry = providers_cfg.get(provider_key) if isinstance(providers_cfg, dict) else None
        if isinstance(provider_entry, dict):
            provider_entry["default_model"] = model_name
            # Only persist an inline api_key when the user originally had one
            # (literal or ``${VAR}``). Entries relying on ``key_env`` must not get
            # a synthesized api_key — the runtime resolves key_env directly and
            # writing it would downgrade credential hygiene.
            had_inline_api_key = bool(
                str(provider_info.get("api_key_ref", "") or "").strip() or str(provider_info.get("api_key", "") or "").strip()
            )
            if had_inline_api_key and config_api_key and not str(provider_entry.get("api_key", "") or "").strip():
                provider_entry["api_key"] = config_api_key
            if key_env and not str(provider_entry.get("key_env", "") or "").strip():
                provider_entry["key_env"] = key_env
            cfg["providers"] = providers_cfg
            save_config(cfg)
    else:
        # Save model name to the custom_providers entry for next time
        _save_custom_provider(base_url, config_api_key, model_name, api_mode=api_mode)

    print(f"\n✅ Model set to: {model_name}")
    print(f"   Provider: {name} ({base_url})")


def _copilot_model_list(live_ids) -> list:
    """Live GitHub Copilot ids, or the curated fallback with a warning."""
    from hermes_cli.models import _PROVIDER_MODELS

    if live_ids:
        model_list = [model_id for model_id in live_ids if model_id]
        print(f"  Found {len(model_list)} model(s) from GitHub Copilot")
        return model_list
    model_list = _PROVIDER_MODELS.get("copilot", [])
    if model_list:
        _say("  ⚠ Could not auto-detect models from GitHub Copilot — showing defaults.",
             '    Use "Enter custom model name" if you do not see your model.')
    return model_list


def _copilot_obtain_token() -> bool:
    """No Copilot token yet: offer device-code login or manual entry. False = stop."""
    from hermes_cli.config import save_env_value

    _say("No GitHub token configured for GitHub Copilot.", "", "  Supported token types:",
         "    → OAuth token (gho_*)          via `copilot login` or device code flow",
         "    → Fine-grained PAT (github_pat_*)  with Copilot Requests permission",
         "    → GitHub App token (ghu_*)     via environment variable",
         "    ✗ Classic PAT (ghp_*)          NOT supported by Copilot API", "", "  Options:",
         "    1. Login with GitHub (OAuth device code flow)", "    2. Enter a token manually", "    3. Cancel", "")
    choice = _ask("  Choice [1-3]: ", raw=True, cancel_msg="")
    if choice is None:
        return False
    if choice == "1":
        try:
            from hermes_cli.copilot_auth import copilot_device_code_login

            token = copilot_device_code_login()
            if not token:
                print("  Login cancelled or failed.")
                return False
            save_env_value("COPILOT_GITHUB_TOKEN", token)
            _say("  Copilot token saved.", "")
        except Exception as exc:
            print(f"  Login failed: {exc}")
            return False
        return True
    if choice == "2":
        new_key = _ask("  Token (COPILOT_GITHUB_TOKEN): ", secret=True, cancel_msg="")
        if new_key is None:
            return False
        if not new_key:
            print("  Cancelled.")
            return False
        # Validate token type
        try:
            from hermes_cli.copilot_auth import validate_copilot_token

            valid, msg = validate_copilot_token(new_key)
            if not valid:
                print(f"  ✗ {msg}")
                return False
        except ImportError:
            pass
        save_env_value("COPILOT_GITHUB_TOKEN", new_key)
        _say("  Token saved.", "")
        return True
    print("  Cancelled.")
    return False


def _model_flow_copilot(config, current_model=""):
    """GitHub Copilot flow using env vars, gh CLI, or OAuth device code."""
    from hermes_cli.main import _prompt_reasoning_effort_selection
    from hermes_cli.setup import _current_reasoning_effort, _set_reasoning_effort
    from hermes_cli.auth import PROVIDER_REGISTRY, resolve_api_key_provider_credentials
    from hermes_cli.config import load_config
    from hermes_cli.models import (
        fetch_api_models, fetch_github_model_catalog, github_model_reasoning_efforts, copilot_model_api_mode,
        normalize_copilot_model_id,
    )

    provider_id = "copilot"
    pconfig = PROVIDER_REGISTRY[provider_id]
    creds = resolve_api_key_provider_credentials(provider_id)
    api_key = creds.get("api_key", "")
    source = creds.get("source", "")
    if not api_key:
        if not _copilot_obtain_token():
            return
        creds = resolve_api_key_provider_credentials(provider_id)
        api_key = creds.get("api_key", "")
    else:
        if source in {"GITHUB_TOKEN", "GH_TOKEN"}:
            from hermes_cli.env_loader import format_secret_source_suffix

            print(f"  GitHub token: {api_key[:8]}... ✓ ({source}{format_secret_source_suffix(source)})")
        elif source == "gh auth token":
            print("  GitHub token: ✓ (from `gh auth token`)")
        else:
            print("  GitHub token: ✓")
        print()

    effective_base = pconfig.inference_base_url
    catalog = fetch_github_model_catalog(api_key)
    live_models = [item.get("id", "") for item in catalog if item.get("id")] if catalog else fetch_api_models(api_key, effective_base)

    def _normalize(mid):
        return normalize_copilot_model_id(mid, catalog=catalog, api_key=api_key) or mid

    selected = _pick_model_or_prompt(
        _copilot_model_list(live_models), "Model name: ", current_model=_normalize(current_model),
        confirm_provider=provider_id, confirm_base_url=effective_base, confirm_api_key=api_key,
    )
    if not selected:
        print("No change.")
        return
    selected = _normalize(selected)
    current_effort = _current_reasoning_effort(load_config())
    reasoning_efforts = github_model_reasoning_efforts(selected, catalog=catalog, api_key=api_key)
    selected_effort = None
    if reasoning_efforts:
        print(f"  {selected} supports reasoning controls.")
        selected_effort = _prompt_reasoning_effort_selection(reasoning_efforts, current_effort=current_effort)

    def _finish(cfg, _model):
        if selected_effort is not None:
            _set_reasoning_effort(cfg, selected_effort)

    _persist_model(selected, provider_id, base_url=effective_base,
                   api_mode=copilot_model_api_mode(selected, catalog=catalog, api_key=api_key), finish=_finish)
    print(f"Default model set to: {selected} (via {pconfig.name})")
    if reasoning_efforts:
        if selected_effort == "none":
            print("Reasoning disabled for this model.")
        elif selected_effort:
            print(f"Reasoning effort set to: {selected_effort}")


def _model_flow_copilot_acp(config, current_model=""):
    """GitHub Copilot ACP flow using the local Copilot CLI."""
    from hermes_cli.auth import (
        PROVIDER_REGISTRY, get_external_process_provider_status, resolve_api_key_provider_credentials,
        resolve_external_process_provider_credentials,
    )
    from hermes_cli.models import fetch_github_model_catalog, normalize_copilot_model_id

    del config
    provider_id = "copilot-acp"
    pconfig = PROVIDER_REGISTRY[provider_id]
    status = get_external_process_provider_status(provider_id)
    resolved_command = status.get("resolved_command") or status.get("command") or "copilot"
    effective_base = status.get("base_url") or pconfig.inference_base_url

    _say("  GitHub Copilot ACP delegates Hermes turns to `copilot --acp`.",
         "  Hermes currently starts its own ACP subprocess for each request.",
         "  Hermes uses your selected model as a hint for the Copilot ACP session.",
         f"  Command: {resolved_command}", f"  Backend marker: {effective_base}", "")
    try:
        creds = resolve_external_process_provider_credentials(provider_id)
    except Exception as exc:
        _say(f"  ⚠ {exc}", "  Set HERMES_COPILOT_ACP_COMMAND or COPILOT_CLI_PATH if Copilot CLI is installed elsewhere.")
        return
    effective_base = creds.get("base_url") or effective_base

    catalog_api_key = ""
    try:
        catalog_api_key = resolve_api_key_provider_credentials("copilot").get("api_key", "")
    except Exception:
        pass
    catalog = fetch_github_model_catalog(catalog_api_key)

    def _normalize(mid):
        return normalize_copilot_model_id(mid, catalog=catalog, api_key=catalog_api_key) or mid

    model_list = _copilot_model_list([item.get("id", "") for item in catalog if item.get("id")] if catalog else [])
    selected = _pick_model_or_prompt(
        model_list, "Model name: ", current_model=_normalize(current_model), confirm_provider=provider_id,
        confirm_base_url=effective_base, confirm_api_key=catalog_api_key,
    )
    if selected:
        selected = _normalize(selected)
    _finish_model(selected, provider_id, f"Default model set to: {selected} (via {pconfig.name})",
                  base_url=effective_base, api_mode="chat_completions")


def _model_flow_kimi(config, current_model=""):
    """Kimi / Moonshot model selection with automatic endpoint routing.

    - sk-kimi-* keys   → api.kimi.com/coding/v1  (Kimi Coding Plan)
    - Other keys        → api.moonshot.ai/v1      (legacy Moonshot)

    No manual base URL prompt — endpoint is determined by key prefix.
    """
    from hermes_cli.auth import PROVIDER_REGISTRY, KIMI_CODE_BASE_URL
    from hermes_cli.config import get_env_value, save_env_value
    from hermes_cli.models import _PROVIDER_MODELS

    provider_id = "kimi-coding"
    pconfig = PROVIDER_REGISTRY[provider_id]
    base_url_env = pconfig.base_url_env_var or ""

    # Step 1: Check / prompt for API key
    _, existing_key, abort = _ensure_flow_api_key(provider_id, pconfig)
    if abort:
        return

    # Step 2: Auto-detect endpoint from key prefix
    is_coding_plan = existing_key.startswith("sk-kimi-")
    if is_coding_plan:
        effective_base = KIMI_CODE_BASE_URL
        print(f"  Detected Kimi Coding Plan key → {effective_base}")
    else:
        effective_base = pconfig.inference_base_url
        print(f"  Using Moonshot endpoint → {effective_base}")
    # Clear any manual base URL override so auto-detection works at runtime
    if base_url_env and get_env_value(base_url_env):
        save_env_value(base_url_env, "")
    print()

    # Step 3: Model selection — show appropriate models for the endpoint
    model_list = _PROVIDER_MODELS.get("kimi-coding" if is_coding_plan else "moonshot", [])
    selected = _pick_model_or_prompt(
        model_list, "Enter model name: ", current_model=current_model, confirm_provider=provider_id,
        confirm_base_url=effective_base, confirm_api_key=existing_key,
    )
    # api_mode is dropped so the runtime auto-detects it from the URL.
    _finish_model(selected, provider_id, f"Default model set to: {selected} (via {'Kimi Coding' if is_coding_plan else 'Moonshot'})",
                  base_url=effective_base, drop_api_mode=True)


def _model_flow_stepfun(config, current_model=""):
    """StepFun Step Plan flow with region-specific endpoints."""
    from hermes_cli.main import _infer_stepfun_region, _prompt_provider_choice, _stepfun_base_url_for_region
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.config import get_env_value, save_env_value
    from hermes_cli.models import _PROVIDER_MODELS, fetch_api_models

    provider_id = "stepfun"
    pconfig = PROVIDER_REGISTRY[provider_id]
    base_url_env = pconfig.base_url_env_var or ""

    _, existing_key, abort = _ensure_flow_api_key(provider_id, pconfig)
    if abort:
        return

    current_base = ""
    if base_url_env:
        current_base = get_env_value(base_url_env) or os.getenv(base_url_env, "")
    if not current_base:
        model_cfg = config.get("model")
        if isinstance(model_cfg, dict):
            current_base = str(model_cfg.get("base_url") or "").strip()
    current_region = _infer_stepfun_region(current_base or pconfig.inference_base_url)

    region_choices = [
        ("international", f"International ({_stepfun_base_url_for_region('international')})"),
        ("china", f"China ({_stepfun_base_url_for_region('china')})"),
    ]
    ordered_regions = []
    for region_key, label in region_choices:
        if region_key == current_region:
            ordered_regions.insert(0, (region_key, f"{label}  ← currently active"))
        else:
            ordered_regions.append((region_key, label))
    ordered_regions.append(("cancel", "Cancel"))

    region_idx = _prompt_provider_choice([label for _, label in ordered_regions])
    if region_idx is None or ordered_regions[region_idx][0] == "cancel":
        print("No change.")
        return
    effective_base = _stepfun_base_url_for_region(ordered_regions[region_idx][0])
    if base_url_env:
        save_env_value(base_url_env, effective_base)

    model_list = fetch_api_models(existing_key, effective_base)
    if model_list:
        print(f"  Found {len(model_list)} model(s) from {pconfig.name} API")
    else:
        model_list = _PROVIDER_MODELS.get(provider_id, [])
        if model_list:
            print(f"  Could not auto-detect models from {pconfig.name} API — showing Step Plan fallback catalog.")

    selected = _pick_model_or_prompt(
        model_list, "Model name: ", current_model=current_model, confirm_provider=provider_id,
        confirm_base_url=effective_base, confirm_api_key=existing_key,
    )
    model = _finish_model(selected, provider_id, f"Default model set to: {selected} (via {pconfig.name})",
                          base_url=effective_base, drop_api_mode=True)
    if model is not None:
        config["model"] = dict(model)


def _model_flow_bedrock_api_key(config, region, current_model=""):
    """Bedrock API Key mode — uses the OpenAI-compatible bedrock-mantle endpoint.

    For developers without an AWS account who received a Bedrock API Key from
    their AWS admin. Works like any OpenAI-compatible endpoint.
    """
    from hermes_cli.auth import _resolve_api_key_provider_secret, ProviderConfig
    from hermes_cli.config import save_env_value
    from hermes_cli.models import _PROVIDER_MODELS

    mantle_base_url = f"https://bedrock-mantle.{region}.api.aws/v1"

    # Check env var and credential pool (keys added via `hermes auth`)
    bedrock_pconfig = ProviderConfig(id="bedrock", name="Bedrock", auth_type="api_key", api_key_env_vars=("AWS_BEARER_TOKEN_BEDROCK",))
    existing_key, existing_source = _resolve_api_key_provider_secret("bedrock", bedrock_pconfig)
    if existing_key:
        from hermes_cli.env_loader import format_secret_source_suffix

        source_suffix = format_secret_source_suffix(existing_source or "AWS_BEARER_TOKEN_BEDROCK")
        print(f"  Bedrock API Key: {existing_key[:12]}... ✓{source_suffix}")
    else:
        _say(f"  Endpoint: {mantle_base_url}", "")
        api_key = _ask("  Bedrock API Key: ", secret=True, cancel_msg="")
        if api_key is None:
            return
        if not api_key:
            print("  Cancelled.")
            return
        save_env_value("AWS_BEARER_TOKEN_BEDROCK", api_key)
        existing_key = api_key
        print("  ✓ API key saved.")
    print()

    # Static list — mantle doesn't need boto3 for discovery
    model_list = _PROVIDER_MODELS.get("bedrock", [])
    print(f"  Showing {len(model_list)} curated models")
    selected = _pick_model_or_prompt(
        model_list, "  Model ID: ", current_model=current_model, confirm_provider="custom",
        confirm_base_url=mantle_base_url, confirm_api_key=existing_key,
    )

    def _finish(cfg, _model):
        # The bearer token rides on a named provider entry: a bare ``provider: custom``
        # cannot carry a credential for this host because OPENAI_API_KEY is gated to
        # openai.com, so requests would go out as "no-key-required".
        providers = _ensure_dict_section(cfg, "providers")
        mantle_entry = providers.get("bedrock-mantle")
        if not isinstance(mantle_entry, dict):
            mantle_entry = {}
        mantle_entry["base_url"] = mantle_base_url
        mantle_entry["key_env"] = "AWS_BEARER_TOKEN_BEDROCK"
        providers["bedrock-mantle"] = mantle_entry
        # Also save region in bedrock config for reference
        _ensure_dict_section(cfg, "bedrock")["region"] = region

    # Saved as a custom provider pointing to bedrock-mantle (no inline endpoint fields).
    if _finish_model(selected, "custom:bedrock-mantle", f"  Default model set to: {selected} (via Bedrock API Key, {region})",
                     no_change="  No change.", drop_base_url=True, drop_api_mode=True, finish=_finish) is not None:
        print(f"  Endpoint: {mantle_base_url}")


_BEDROCK_EXCLUDE_PREFIXES = ("stability.", "cohere.embed", "twelvelabs.", "us.stability.", "us.cohere.embed",
                             "us.twelvelabs.", "global.cohere.embed", "global.twelvelabs.")
_BEDROCK_EXCLUDE_SUBSTRINGS = ("safeguard", "voxtral", "palmyra-vision")
_BEDROCK_PROFILE_PREFIXES = BEDROCK_GEO_PREFIXES + ("global.",)
# Recommended models, matched geo-agnostically so an EU (eu.*) or APAC (apac.*)
# picker pins its own region's profile rather than a us.* one.
_BEDROCK_RECOMMENDED_BASES = (
    "anthropic.claude-sonnet-4-6", "anthropic.claude-opus-4-6", "anthropic.claude-haiku-4-5", "amazon.nova-pro",
    "amazon.nova-lite", "amazon.nova-micro", "deepseek.v3", "meta.llama4-maverick", "meta.llama4-scout",
)


def _bedrock_text_model_ids(live_models: list, region: str) -> list[str]:
    """Filter live Bedrock models to routable text models, dedupe bare ids against their
    inference profiles, and order: recommended (in-region profile before global.*),
    then other global.* profiles, then the rest."""
    def _base_id(mid: str) -> str:
        _pp = next((p for p in _BEDROCK_PROFILE_PREFIXES if mid.startswith(p)), None)
        return mid[len(_pp):] if _pp else mid

    filtered = [
        m for m in live_models
        if not any(m["id"].startswith(p) for p in _BEDROCK_EXCLUDE_PREFIXES)
        and not any(s in m["id"].lower() for s in _BEDROCK_EXCLUDE_SUBSTRINGS)
        and bedrock_model_routable_from_region(m["id"], region)
    ]
    # Deduplicate: prefer inference profiles (geo-prefixed or global.*) over bare foundation model IDs.
    profile_base_ids = {_base_id(m["id"]) for m in filtered if m["id"].startswith(_BEDROCK_PROFILE_PREFIXES)}
    deduped = [m for m in filtered if m["id"].startswith(_BEDROCK_PROFILE_PREFIXES) or m["id"] not in profile_base_ids]

    def _sort_key(m):
        mid = m["id"]
        base = _base_id(mid)
        for i, rec in enumerate(_BEDROCK_RECOMMENDED_BASES):
            if base.startswith(rec):
                # In-region geo profile beats global.* for the same model
                return (0, i, 0 if not mid.startswith("global.") else 1, mid)
        if mid.startswith("global."):
            return (1, 0, 0, mid)
        return (2, 0, 0, mid)

    deduped.sort(key=_sort_key)
    return [m["id"] for m in deduped]


def _model_flow_bedrock(config, current_model=""):
    """AWS Bedrock provider: verify credentials, pick region, discover models.

    Uses the native Converse API via boto3 — not the OpenAI-compatible endpoint.
    Auth is the AWS SDK default credential chain (env vars, profile, instance
    role), so no API key prompt is needed.
    """
    from hermes_cli.models import _PROVIDER_MODELS

    # 1. Check for AWS credentials
    try:
        from agent.bedrock_adapter import has_aws_credentials, resolve_aws_auth_env_var, resolve_bedrock_region, discover_bedrock_models
    except ImportError:
        _say("  ✗ boto3 is not installed. Install it with:", "    pip install boto3", "")
        return

    if not has_aws_credentials():
        _say("  ⚠ No AWS credentials detected via environment variables.",
             "  Bedrock will use boto3's default credential chain (IMDS, SSO, etc.)", "")
    auth_var = resolve_aws_auth_env_var()
    print(f"  AWS credentials: {auth_var} ✓" if auth_var else "  AWS credentials: boto3 default chain (instance role / SSO)")
    print()

    # 2. Region selection
    current_region = resolve_bedrock_region()
    region_input = _ask(f"  AWS Region [{current_region}]: ", cancel_msg="")
    if region_input is None:
        return
    region = region_input or current_region

    # 2b. Authentication mode
    _say("  Choose authentication method:", "", "    1. IAM credential chain (recommended)",
         "       Works with EC2 instance roles, SSO, env vars, aws configure", "    2. Bedrock API Key",
         "       Enter your Bedrock API Key directly — also supports",
         "       team scenarios where an admin distributes keys", "")
    auth_choice = _ask("  Choice [1]: ", raw=True, cancel_msg="")
    if auth_choice is None:
        return
    if auth_choice == "2":
        _model_flow_bedrock_api_key(config, region, current_model)
        return

    # 3. Model discovery — try live API first, fall back to static list
    print(f"  Discovering models in {region}...")
    live_models = discover_bedrock_models(region)
    if live_models:
        model_list = _bedrock_text_model_ids(live_models, region)
        print(f"  Found {len(model_list)} text model(s) (filtered from {len(live_models)} total)")
    else:
        model_list = _PROVIDER_MODELS.get("bedrock", [])
        if not model_list:
            print("  No models found. Check IAM permissions for bedrock:ListFoundationModels.")
            return
        print(f"  Using {len(model_list)} curated models (live discovery unavailable)")

    # 4. Model selection
    runtime_url = f"https://bedrock-runtime.{region}.amazonaws.com"
    selected = _pick_model_or_prompt(model_list, "  Model ID: ", current_model=current_model, confirm_provider="bedrock", confirm_base_url=runtime_url)
    # api_mode is dropped: bedrock_converse is auto-detected.
    _finish_model(selected, "bedrock", f"  Default model set to: {selected} (via AWS Bedrock, {region})", no_change="  No change.",
                  base_url=runtime_url, drop_api_mode=True,
                  finish=lambda cfg, _m: _ensure_dict_section(cfg, "bedrock").__setitem__("region", region))


def _model_flow_vertex(config, current_model=""):
    """Google Vertex AI provider: Gemini via the OpenAI-compatible endpoint.

    Auth is OAuth2 — short-lived tokens minted from a service-account JSON or
    Application Default Credentials (ADC). No static API key. The credential
    *path* lives in .env (VERTEX_CREDENTIALS_PATH / GOOGLE_APPLICATION_CREDENTIALS);
    project ID and region are non-secret and saved to config.yaml under vertex:.
    """
    from hermes_cli.auth import _prompt_model_selection
    from hermes_cli.config import load_config, get_env_value
    from hermes_cli.models import _PROVIDER_MODELS

    # 1. Credential source detection (fast, no network / no google-auth import).
    sa_path = (get_env_value("VERTEX_CREDENTIALS_PATH") or get_env_value("GOOGLE_APPLICATION_CREDENTIALS") or "").strip()
    if sa_path:
        print(f"  Vertex credentials: service account JSON ({sa_path}) ✓")
    else:
        _say("  Vertex credentials: Application Default Credentials (ADC)",
             "    Vertex uses OAuth2, not a static API key. Either:",
             "      • run 'gcloud auth application-default login', or",
             "      • set VERTEX_CREDENTIALS_PATH in ~/.hermes/.env to a service account JSON")
    print()

    vertex_cfg = load_config().get("vertex")
    if not isinstance(vertex_cfg, dict):
        vertex_cfg = {}

    # 2. Project ID (optional — falls back to the project embedded in creds).
    current_project = str(vertex_cfg.get("project_id") or "").strip()
    project_input = _ask(f"  GCP project ID [{current_project or 'from credentials'}]: ", cancel_msg="")
    if project_input is None:
        return
    project_id = project_input or current_project

    # 3. Region (default global — required for the Gemini 3.x previews).
    current_region = str(vertex_cfg.get("region") or "global").strip() or "global"
    region_input = _ask(f"  Vertex region [{current_region}]: ", cancel_msg="")
    if region_input is None:
        return
    region = region_input or current_region

    # 4. Model selection (curated list — Vertex has no /models listing route).
    model_list = _PROVIDER_MODELS.get("vertex", []) or ["google/gemini-3-pro-preview", "google/gemini-3-flash-preview"]
    host = "aiplatform.googleapis.com" if region == "global" else f"{region}-aiplatform.googleapis.com"
    base_url_preview = f"https://{host}/v1beta1/projects/<project>/locations/{region}/endpoints/openapi"
    selected = _prompt_model_selection(model_list, current_model=current_model, confirm_provider="vertex", confirm_base_url=base_url_preview)

    def _finish(cfg, _model):
        vcfg = _ensure_dict_section(cfg, "vertex")
        vcfg["project_id"] = project_id
        vcfg["region"] = region

    # base_url is computed at runtime from project+region; do not pin it.
    # api_mode is dropped: chat_completions is the profile default.
    _finish_model(selected, "vertex", f"  Default model set to: {selected} (via Google Vertex AI, {region})", no_change="  No change.",
                  drop_base_url=True, drop_api_mode=True, finish=_finish)


def _select_zai_endpoint(current_base: str) -> str:
    """Picker for the four official Z.AI endpoints (sourced from ``ZAI_ENDPOINTS``
    in ``hermes_cli.auth`` so it stays in sync with the probe list) plus a
    custom-proxy option. Returns the selected base URL; *current_base* on cancel/error.
    """
    from hermes_cli.main import _prompt_provider_choice
    from hermes_cli.auth import ZAI_ENDPOINTS

    options = [(label, url) for _, url, _, label in ZAI_ENDPOINTS]
    normalized_current = (current_base or "").strip().rstrip("/")

    # Default to the active endpoint when known; a custom URL defaults to "Custom proxy".
    default_idx = next((idx for idx, (_, url) in enumerate(options) if normalized_current == url.rstrip("/")),
                       len(options) if normalized_current else 0)
    choices = [f"{label} ({url})" for label, url in options] + ["Custom proxy URL"]
    selected = _prompt_provider_choice(choices, default=default_idx, title="Select Z.AI / GLM endpoint:")
    if selected is None:
        return current_base
    if selected != len(options):
        return options[selected][1].rstrip("/")
    # Custom proxy URL
    override = _ask(f"Custom base URL [{current_base}]: ", cancel_msg="")
    if not override:
        return current_base
    if not override.startswith(_HTTP):
        print("  Invalid URL — must start with http:// or https://. Keeping current value.")
        return current_base
    return override.rstrip("/")


_GEMINI_FREE_TIER_NOTICE = (
    "", "❌ This Google API key is on the free tier (<= 250 requests/day for gemini-2.5-flash).",
    "   Hermes typically makes 3-10 API calls per user turn (tool iterations + auxiliary tasks),",
    "   so the free tier is exhausted after a handful of messages and cannot sustain",
    "   an agent session.", "",
    "   To use Gemini with Hermes, enable billing on your Google Cloud project and regenerate",
    "   the key in a billing-enabled project: https://aistudio.google.com/apikey", "",
    "   Alternatives with workable free usage: DeepSeek, OpenRouter (free models), Groq, Nous.", "",
    "Not saving Gemini as the default provider.",
)


def _gemini_tier_ok(existing_key: str, pconfig, base_url_env: str) -> bool:
    """Gemini free-tier gate: free-tier daily quotas (<= 250 RPD for Flash) are
    exhausted in a handful of agent turns, so refuse a free-tier key. The probe
    is best-effort; network or auth errors fall through without blocking."""
    from hermes_cli.config import get_env_value

    try:
        from agent.gemini_native_adapter import probe_gemini_tier
    except Exception:
        return True
    print("  Checking Gemini API tier...")
    probe_base = (get_env_value(base_url_env) if base_url_env else "") or os.getenv(base_url_env or "", "") or pconfig.inference_base_url
    tier = probe_gemini_tier(existing_key, probe_base)
    if tier == "free":
        _say(*_GEMINI_FREE_TIER_NOTICE)
        return False
    # "unknown" (network/auth/unexpected response): don't block; the
    # runtime 429 handler surfaces free-tier guidance if needed.
    _say("  Tier check: paid ✓" if tier == "paid" else "  Tier check: could not verify (proceeding anyway).", "")
    return True


def _api_key_provider_model_list(provider_id: str, pconfig, existing_key: str, key_env: str, effective_base: str) -> list:
    """Model list for an API-key provider. Resolution order:
      1. models.dev registry (cached, filtered for agentic/tool-capable models)
      2. Curated static fallback list (offline insurance)
      3. Live /models endpoint probe (small providers without models.dev data)
    LM Studio: live /api/v1/models probe only. Ollama Cloud: merged discovery.
    """
    from hermes_cli.config import get_env_value
    from hermes_cli.models import _PROVIDER_MODELS, fetch_api_models

    curated = _PROVIDER_MODELS.get(provider_id, [])
    api_key_for_probe = existing_key or (get_env_value(key_env) if key_env else "")

    def _mdev():
        model_list = _models_dev_merged(provider_id, curated)
        if model_list:
            print(f"  Found {len(model_list)} model(s) from models.dev registry")
        return model_list

    if provider_id == "lmstudio":
        from hermes_cli.auth import AuthError
        from hermes_cli.models import fetch_lmstudio_models

        try:
            model_list = fetch_lmstudio_models(api_key=api_key_for_probe, base_url=effective_base)
        except AuthError as exc:
            _say(f"  LM Studio rejected the request: {exc}", "  Set LM_API_KEY (or update it) to match the server's bearer token.")
            model_list = []
        if model_list:
            print(f"  Found {len(model_list)} model(s) from LM Studio")
        return model_list
    if provider_id == "ollama-cloud":
        from hermes_cli.models import fetch_ollama_cloud_models

        # Force a live refresh so newly released models appear the moment the user
        # enters their key, not when the disk cache TTL expires.
        model_list = fetch_ollama_cloud_models(api_key=api_key_for_probe, base_url=effective_base, force_refresh=True)
        if model_list:
            print(f"  Found {len(model_list)} model(s) from Ollama Cloud")
        return model_list
    if provider_id == "opencode-free":
        # Keyless tier: the curated list is synced against anonymous live probes
        # (models.dev's cost.input==0 filter lags reality).
        if curated:
            print(f'  Showing {len(curated)} keyless free models — use "Enter custom model name" for others.')
        return curated
    if provider_id == "novita":
        # Live first, then models.dev, then curated.
        live_models = fetch_api_models(api_key_for_probe, effective_base)
        if live_models:
            print(f"  Found {len(live_models)} model(s) from {pconfig.name} API")
            return live_models
        model_list = _mdev()
        if model_list:
            return model_list
        _show_curated(curated)
        return curated
    # models.dev first (tool-capable, noise-filtered), merged with curated so
    # newly added models still appear.
    model_list = _mdev()
    if model_list:
        return model_list
    if curated and len(curated) >= 8:
        # Substantial curated list — use it directly, skip live probe
        _show_curated(curated)
        return curated
    live_models = fetch_api_models(api_key_for_probe, effective_base)
    if live_models and len(live_models) >= len(curated):
        print(f"  Found {len(live_models)} model(s) from {pconfig.name} API")
        return live_models
    _show_curated(curated)  # may be empty: falls through to raw input
    return curated


def _model_flow_api_key_provider(config, provider_id, current_model=""):
    """Generic flow for API-key providers (z.ai, MiniMax, OpenCode, etc.)."""
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.config import get_env_value, save_env_value, load_config
    from hermes_cli.models import opencode_model_api_mode, normalize_opencode_model_id

    pconfig = PROVIDER_REGISTRY[provider_id]
    key_env = pconfig.api_key_env_vars[0] if pconfig.api_key_env_vars else ""
    base_url_env = pconfig.base_url_env_var or ""
    is_opencode = provider_id in {"opencode-zen", "opencode-go", "opencode-free"}

    # OpenCode Free is keyless — the tier is served anonymously and any
    # unrecognized bearer 401s, so there is no key to prompt for.
    if provider_id == "opencode-free":
        print("  OpenCode Free is keyless — no API key or account needed.")
        existing_key = ""
    else:
        _, existing_key, abort = _ensure_flow_api_key(provider_id, pconfig)
        if abort:
            return
    if provider_id == "gemini" and existing_key and not _gemini_tier_ok(existing_key, pconfig, base_url_env):
        return

    # Optional base URL override. Precedence: env var → config.yaml model.base_url →
    # registry default; reading config.yaml keeps a saved remote URL from being
    # overwritten with localhost when the user just presses Enter.
    current_base = ""
    if base_url_env:
        current_base = get_env_value(base_url_env) or os.getenv(base_url_env, "")
    if not current_base:
        try:
            _m = load_config().get("model") or {}
            if str(_m.get("provider") or "").strip().lower() == provider_id:
                current_base = str(_m.get("base_url") or "").strip()
        except Exception:
            pass
    effective_base = current_base or pconfig.inference_base_url

    if provider_id == "zai":
        # Four official endpoints with separate billing paths — a picker lets users
        # match the endpoint to their key type.
        chosen_base = _select_zai_endpoint(effective_base)
        if chosen_base and chosen_base != effective_base and base_url_env:
            save_env_value(base_url_env, chosen_base)
        effective_base = chosen_base
    else:
        override = _ask(f"Base URL [{effective_base}]: ", cancel_msg="", on_cancel="")
        if override and base_url_env:
            if not override.startswith(_HTTP):
                print("  Invalid URL — must start with http:// or https://. Keeping current value.")
            else:
                save_env_value(base_url_env, override)
                effective_base = override

    model_list = _api_key_provider_model_list(provider_id, pconfig, existing_key, key_env, effective_base)
    if is_opencode:
        model_list = [normalize_opencode_model_id(provider_id, mid) for mid in model_list]
        current_model = normalize_opencode_model_id(provider_id, current_model)
        model_list = list(dict.fromkeys(mid for mid in model_list if mid))

    # Per-model pricing when the provider supports it; get_pricing_for_provider() is
    # memoized and returns {} otherwise — never a blocking fetch beyond the catalog
    # lookup that already happened above.
    pricing: dict = {}
    if model_list:
        try:
            from hermes_cli.models import get_pricing_for_provider

            pricing = get_pricing_for_provider(provider_id) or {}
        except Exception:
            pricing = {}
    selected = _pick_model_or_prompt(
        model_list, "Model name: ", current_model=current_model, pricing=pricing, confirm_provider=provider_id,
        confirm_base_url=effective_base, confirm_api_key=existing_key,
    )
    if selected and is_opencode:
        selected = normalize_opencode_model_id(provider_id, selected)
    # OpenCode pins its api_mode; everyone else drops it so the runtime auto-detects.
    _finish_model(
        selected, provider_id, f"Default model set to: {selected} (via {pconfig.name})", base_url=effective_base,
        api_mode=opencode_model_api_mode(provider_id, selected) if selected and is_opencode else None,
        drop_api_mode=not is_opencode,
    )


def _model_flow_anthropic(config, current_model=""):
    """Flow for Anthropic provider — OAuth subscription, API key, or Claude Code creds."""
    from hermes_cli.main import _run_anthropic_oauth_flow
    from hermes_cli.auth import get_anthropic_key
    from hermes_cli.config import save_env_value, save_anthropic_api_key
    from hermes_cli.models import _PROVIDER_MODELS

    # Check ALL credential sources
    existing_key = get_anthropic_key()
    cc_available = False
    try:
        from agent.anthropic_adapter import read_claude_code_credentials, is_claude_code_token_valid, _is_oauth_token

        cc_creds = read_claude_code_credentials()
        if cc_creds and is_claude_code_token_valid(cc_creds):
            cc_available = True
    except Exception:
        pass

    # Stale-OAuth guard: an expired OAuth token with no valid cc_creds fallback is
    # treated as missing so the re-auth path is offered.
    existing_is_stale_oauth = bool(existing_key and _is_oauth_token(existing_key) and not cc_available)
    has_creds = (bool(existing_key) and not existing_is_stale_oauth) or cc_available
    needs_auth = not has_creds

    if has_creds:
        if existing_key:
            from hermes_cli.env_loader import format_secret_source_suffix
            from hermes_cli.auth import PROVIDER_REGISTRY

            # Surface which env var supplied the key so Bitwarden users see
            # "(from Bitwarden)" instead of a key indistinguishable from .env.
            source_suffix = ""
            for var in PROVIDER_REGISTRY["anthropic"].api_key_env_vars:
                if os.getenv(var, "").strip() == existing_key:
                    source_suffix = format_secret_source_suffix(var)
                    if source_suffix:
                        break
            print(f"  Anthropic credentials: {existing_key[:12]}... ✓{source_suffix}")
        elif cc_available:
            print("  Claude Code credentials: ✓ (auto-detected)")
        print()
        choice = _prompt_auth_credentials_choice("Anthropic credentials:")
        if choice == "reauth":
            needs_auth = True
        elif choice == "cancel":
            return
        # "use" (default): proceed to model selection with existing creds

    if needs_auth:
        _say("", "  Choose authentication method:", "", "    1. Claude Pro/Max subscription (OAuth login)",
             "    2. Anthropic API key (pay-per-token)", "    3. Cancel", "")
        choice = _ask("  Choice [1/2/3]: ", raw=True, cancel_msg="")
        if choice is None:
            return
        if choice == "1":
            if not _run_anthropic_oauth_flow(save_env_value):
                return
        elif choice == "2":
            _say("", "  Get an API key at: https://platform.claude.com/settings/keys", "")
            api_key = _ask("  API key (sk-ant-...): ", secret=True, cancel_msg="")
            if api_key is None:
                return
            if not api_key:
                print("  Cancelled.")
                return
            save_anthropic_api_key(api_key, save_fn=save_env_value)
            print("  ✓ API key saved.")
        else:
            print("  No change.")
            return
    print()

    selected = _pick_model_or_prompt(
        _PROVIDER_MODELS.get("anthropic", []), "Model name (e.g., claude-sonnet-4-20250514): ",
        current_model=current_model, confirm_provider="anthropic",
    )
    # Clear base_url: resolve_runtime_provider() always hardcodes Anthropic's URL,
    # and a stale value can contaminate other providers on a later switch.
    _finish_model(selected, "anthropic", f"Default model set to: {selected} (via Anthropic)", drop_base_url=True, drop_api_mode=True)
