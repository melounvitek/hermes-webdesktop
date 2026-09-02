"""Custom-provider resolution: ``providers:`` / ``custom_providers:`` lookup, identity
recovery, custom credential pools, and the named-custom runtime builder.

Extracted from :mod:`hermes_cli.runtime_provider`; every public/private name here is
re-exported there. Origin-internal collaborators (``load_config``, ``_get_model_config``,
``load_pool``, ``has_usable_secret``, ``custom_provider_pool_key_candidates``, …) are looked up
on the origin module AT CALL TIME via :func:`_rp` so ``monkeypatch.setattr(runtime_provider,
name, …)`` in tests keeps working for moved bodies.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Callable, Dict, Optional

from hermes_cli.providers import custom_provider_aliases, custom_provider_slug
from utils import base_url_hostname

logger = logging.getLogger("hermes_cli.runtime_provider")


def _rp():
    """Origin module, late-bound so test patches on ``hermes_cli.runtime_provider.*`` apply."""
    import hermes_cli.runtime_provider as origin

    return origin


def _normalize_custom_provider_name(value: str) -> str:
    return value.strip().lower().replace(" ", "-")


def _normalize_base_url_for_match(value) -> str:
    return str(value or "").strip().rstrip("/").lower()


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _entry_url(entry: Dict[str, Any]) -> str:
    return entry.get("api") or entry.get("url") or entry.get("base_url") or ""


# ── field lifting shared by ``providers:`` and legacy ``custom_providers:`` entries ────────


def _filter_capabilities(value: Any) -> Dict[str, bool]:
    """Return the string-keyed boolean capabilities accepted at runtime."""
    if not isinstance(value, dict):
        return {}
    return {k: v for k, v in value.items() if isinstance(k, str) and isinstance(v, bool)}


def _lift_model_capabilities(entry: Dict[str, Any], model: Optional[str], result: Dict[str, Any]) -> None:
    """Copy explicit boolean per-model capabilities into the runtime."""
    capabilities = _filter_capabilities(entry.get("capabilities"))
    models = entry.get("models")
    model_config = models.get(model) if isinstance(models, dict) and model else None
    if isinstance(model_config, dict):
        capabilities.update(_filter_capabilities(model_config))
    if capabilities:
        result["capabilities"] = capabilities


def _lift_max_output_tokens(entry: Dict[str, Any], result: Dict[str, Any]) -> None:
    """``max_output_tokens`` or ``max_tokens`` on a provider entry pins its own output limit.

    Gateway/CLI map it onto ``AIAgent.max_tokens`` only when top-level ``model.max_tokens`` is
    unset, so the documented global key still wins.
    """
    for key in ("max_output_tokens", "max_tokens"):
        value = entry.get(key)
        if isinstance(value, int) and value > 0:
            result["max_output_tokens"] = value
            return


def _lift_extra_headers(entry: Dict[str, Any], result: Dict[str, Any]) -> None:
    """Copy a validated ``extra_headers`` dict. SECURITY: values carry credentials — never log."""
    extra_headers = _rp().normalize_extra_headers(entry.get("extra_headers"))
    if extra_headers:
        result["extra_headers"] = extra_headers


def _lift_common_custom_fields(
    entry: Dict[str, Any],
    result: Dict[str, Any],
    *,
    provider_key: str,
    key_env: str,
    api_mode: Optional[str],
) -> None:
    """Copy the optional fields shared by ``providers:`` and legacy ``custom_providers:`` entries."""
    if key_env:
        result["key_env"] = key_env
    if provider_key:
        result["provider_key"] = provider_key
    extra_body = entry.get("extra_body")
    if isinstance(extra_body, dict):
        result["extra_body"] = dict(extra_body)
    _lift_extra_headers(entry, result)
    if api_mode:
        result["api_mode"] = api_mode
    _lift_max_output_tokens(entry, result)
    capabilities = _filter_capabilities(entry.get("capabilities"))
    if capabilities:
        result["capabilities"] = capabilities


# ── config lookup ──────────────────────────────────────────────────────────────────────────


def _shadowed_by_builtin(requested_norm: str) -> bool:
    """Raw names map to custom providers only when they are not canonical built-ins.

    Explicit ``custom:<name>`` keys always target the saved entry, and bare ``custom`` is
    exempt: a user may literally name a ``providers:`` entry "custom" (returning None before
    the config scan made such cron jobs fail with ``auth_unavailable``). Defer to the built-in
    only when the raw name IS the canonical provider (``nous``); an entry matching merely an
    alias (``kimi`` → ``kimi-coding``) is the user's target.
    """
    if requested_norm == "custom" or requested_norm.startswith("custom:"):
        return False
    rp = _rp()
    try:
        canonical = rp.auth_mod.resolve_provider(requested_norm)
    except rp.AuthError:
        return False
    return (canonical or "").strip().lower() == requested_norm


def _match_new_style_provider(requested_norm: str, providers: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Scan ``providers:`` (new-style, keyed) for ``requested_norm``."""
    from hermes_cli.config import is_provider_enabled

    rp = _rp()
    for ep_name, entry in providers.items():
        # ``providers.<name>.enabled: false`` entries stay in config but are invisible here.
        if not isinstance(entry, dict) or not is_provider_enabled(entry):
            continue
        # API key from the env var named by key_env, else the inline api_key. Read BEFORE the
        # alias match (scope-aware ``_getenv`` fails closed identically for every entry).
        key_env = _clean(entry.get("key_env") or entry.get("api_key_env"))
        api_key = rp._getenv(key_env, "").strip() if key_env else ""
        if requested_norm not in custom_provider_aliases(str(entry.get("name", "") or ep_name), str(ep_name)):
            continue
        base_url = _entry_url(entry)
        if not base_url:
            continue
        result: Dict[str, Any] = {
            "name": entry.get("name", ep_name),
            "base_url": base_url.strip(),
            "api_key": api_key or _clean(entry.get("api_key", "")),
            "model": entry.get("default_model", ""),
        }
        # Command that PRINTS a short-lived credential; wrapped in a per-request token provider.
        key_cmd = _clean(entry.get("key_cmd", ""))
        if key_cmd:
            result["key_cmd"] = key_cmd
        # v12 migration writes ``transport``; hand-edited configs may still use ``api_mode``.
        # Accept both or migrated configs silently downgrade to chat_completions.
        _lift_common_custom_fields(
            entry, result,
            provider_key=_clean(ep_name),
            key_env=key_env,
            api_mode=rp._parse_api_mode(entry.get("api_mode") or entry.get("transport")),
        )
        return result
    return None


def _match_legacy_custom_provider(requested_norm: str, custom_providers) -> Optional[Dict[str, Any]]:
    """Scan the legacy ``custom_providers:`` list for ``requested_norm``."""
    for entry in custom_providers:
        if not isinstance(entry, dict):
            continue
        name, base_url = entry.get("name"), entry.get("base_url")
        if not isinstance(name, str) or not isinstance(base_url, str):
            continue
        provider_key = _clean(entry.get("provider_key", ""))
        if requested_norm not in custom_provider_aliases(name, provider_key):
            continue
        result = {"name": name.strip(), "base_url": base_url.strip(), "api_key": _clean(entry.get("api_key", ""))}
        model_name = _clean(entry.get("model", ""))
        if model_name:
            result["model"] = model_name
        _lift_common_custom_fields(
            entry, result,
            provider_key=provider_key,
            key_env=_clean(entry.get("key_env", "")),
            api_mode=_rp()._parse_api_mode(entry.get("api_mode")),
        )
        return result
    return None


def _get_named_custom_provider(requested_provider: str) -> Optional[Dict[str, Any]]:
    requested_norm = _normalize_custom_provider_name(requested_provider or "")
    if not requested_norm or requested_norm == "auto" or _shadowed_by_builtin(requested_norm):
        return None

    rp = _rp()
    config = rp.load_config()
    providers = config.get("providers")
    if isinstance(providers, dict):
        found = _match_new_style_provider(requested_norm, providers)
        if found:
            return found

    if isinstance(config.get("custom_providers"), dict):
        logger.warning(
            "custom_providers in config.yaml is a dict, not a list. "
            "Each entry must be prefixed with '-' in YAML. "
            "Run 'hermes doctor' for details."
        )
        return None
    custom_providers = rp.get_compatible_custom_providers(config)
    if not custom_providers:
        return None
    return _match_legacy_custom_provider(requested_norm, custom_providers)


def has_named_custom_provider(requested_provider: str) -> bool:
    """True when config defines a ``providers:`` / ``custom_providers:`` entry matching the request.

    Public wrapper so other modules (e.g. the cronjob tool) need not reach into a private helper.
    """
    try:
        return _rp()._get_named_custom_provider(requested_provider) is not None
    except Exception:
        return False


# ── identity recovery (bare "custom" -> durable ``custom:<name>``) ─────────────────────────


def _find_custom_identity(matches: Callable[[Dict[str, Any]], bool]) -> Optional[str]:
    """First entry in ``providers:`` then legacy ``custom_providers:`` where ``matches(entry)``
    holds, as its canonical ``custom:<name>`` slug."""
    rp = _rp()
    try:
        config = rp.load_config()
    except Exception:
        return None

    providers = config.get("providers")
    if isinstance(providers, dict):
        for ep_name, entry in providers.items():
            if isinstance(entry, dict) and matches(entry):
                return custom_provider_slug(str(ep_name), str(ep_name))

    try:
        custom_providers = rp.get_compatible_custom_providers(config)
    except Exception:
        custom_providers = None
    for entry in custom_providers or []:
        if not isinstance(entry, dict):
            continue
        name = entry.get("name")
        if isinstance(name, str) and name.strip() and matches(entry):
            return custom_provider_slug(name, str(entry.get("provider_key", "") or ""))
    return None


def find_custom_provider_identity(base_url: str) -> Optional[str]:
    """Map an endpoint URL back to its canonical ``custom:<name>`` menu key.

    Session persistence stores the agent's *resolved* provider, and for every named custom
    endpoint that is the literal string ``"custom"`` — the entry name is lost, and the api_key is
    deliberately never persisted.
    """
    target = _normalize_base_url_for_match(base_url)
    if not target:
        return None
    return _find_custom_identity(lambda entry: _normalize_base_url_for_match(_entry_url(entry)) == target)


def find_custom_provider_identity_by_model(model: str) -> Optional[str]:
    """Map a model id back to the ``custom:<name>`` entry that serves it.

    Companion to :func:`find_custom_provider_identity` for persistence paths where no base_url
    survived the round-trip: the session row always stores the model name.
    """
    target = str(model or "").strip().lower()
    if not target:
        return None

    def _entry_serves_model(entry: Dict[str, Any]) -> bool:
        for key in ("model", "default_model"):
            value = entry.get(key)
            if isinstance(value, str) and value.strip().lower() == target:
                return True
        models = entry.get("models")
        if isinstance(models, dict):
            return any(str(mid).strip().lower() == target for mid in models)
        if isinstance(models, list):
            for item in models:
                if isinstance(item, str) and item.strip().lower() == target:
                    return True
                if isinstance(item, dict):
                    mid = item.get("id") or item.get("name")
                    if isinstance(mid, str) and mid.strip().lower() == target:
                        return True
        return False

    return _find_custom_identity(_entry_serves_model)


def canonical_custom_identity(
    *,
    base_url: Optional[str] = None,
    config_provider: Optional[str] = None,
    model: Optional[str] = None,
) -> Optional[str]:
    """Recover a routable ``custom:<name>`` identity for a bare custom provider.

    Every path that persists or restores a session's provider override must run the resolved
    provider through this so a bare ``"custom"`` is upgraded back to its durable
    ``custom:<name>`` menu key. Recovery sources, in priority order: (1) ``base_url`` reverse
    lookup — the one fact that always survives the round-trip when a URL was recorded; (2)
    ``model`` reverse lookup (``model``/``default_model``/``models`` catalog); (3) the configured
    provider (arg, then ``model.provider``, then ``HERMES_INFERENCE_PROVIDER``) when it names a
    real entry.
    """
    rp = _rp()
    if base_url:
        identity = find_custom_provider_identity(base_url)
        if identity:
            return identity
    if model:
        identity = find_custom_provider_identity_by_model(model)
        if identity:
            return identity

    candidate = str(config_provider or "").strip()
    if not candidate:
        try:
            candidate = str(rp._get_model_config().get("provider") or "").strip()
        except Exception:
            candidate = ""
    if not candidate:
        candidate = os.environ.get("HERMES_INFERENCE_PROVIDER", "").strip()

    candidate_norm = _normalize_custom_provider_name(candidate)
    # A bare/non-routable candidate cannot heal a bare custom override.
    if not candidate_norm or candidate_norm in {"custom", "auto", "openrouter"}:
        return None
    # Only when it resolves to a configured entry — never invent a ``custom:<x>`` resolution
    # can't honor.
    try:
        entry = rp._get_named_custom_provider(candidate)
        if entry is not None:
            # ``candidate`` may be the entry's DISPLAY NAME, not the durable identity of a keyed
            # ``providers:`` entry — re-resolve via its endpoint so every path returns the same
            # config-key slug.
            identity = find_custom_provider_identity(str(entry.get("base_url") or ""))
            if identity:
                return identity
            return candidate_norm if candidate_norm.startswith("custom:") else f"custom:{candidate_norm}"
    except Exception:
        pass
    return None


def is_routable_provider(provider: Optional[str]) -> bool:
    """Whether a provider name currently resolves to a routable route.

    Empty/None/``auto`` is vacuously routable (agent build falls back to the configured
    default). Bare ``custom`` is the resolved billing class shared by every named entry — not a
    routable identity; restore paths must heal it (:func:`canonical_custom_identity`) or fall
    back. Anything else is routable iff the full chain (built-in -> ``providers:`` ->
    ``custom_providers:`` -> models.dev) resolves it.
    """
    name = str(provider or "").strip()
    if not name or name.lower() == "auto":
        return True
    if name.lower() == "custom":
        return False
    try:
        from hermes_cli.providers import resolve_provider_full

        rp = _rp()
        config = rp.load_config()
        return resolve_provider_full(
            name, config.get("providers"), rp.get_compatible_custom_providers(config)
        ) is not None
    except Exception:
        return False


# ── runtime builders ───────────────────────────────────────────────────────────────────────


def _try_resolve_from_custom_pool(
    base_url: str,
    provider_label: str,
    api_mode_override: Optional[str] = None,
    provider_name: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Runtime dict from the first credential pool that owns this custom endpoint, else None."""
    rp = _rp()
    try:
        raw_keys = list(rp.custom_provider_pool_key_candidates(base_url, provider_name))
    except Exception:
        raw_keys = []
    # Order-preserving dedupe of normalized keys.
    candidates = list(dict.fromkeys(k for k in (str(key or "").strip().lower() for key in raw_keys) if k))
    for pool_key in candidates:
        try:
            pool = rp.load_pool(pool_key)
            if not pool.has_credentials():
                continue
            entry = pool.select()
            if entry is None:
                continue
            pool_api_key = rp._pool_entry_api_key(entry)
            if not pool_api_key:
                continue
            if not rp.has_usable_secret(pool_api_key) and rp._loopback_hostname(base_url_hostname(base_url)):
                # Legacy configs used short placeholder keys ('123', 'm') for local no-auth
                # services; has_usable_secret's 4-char floor rejects them. Every other path
                # substitutes "no-key-required" for a loopback endpoint — this was the one gap.
                pool_api_key = "no-key-required"
            return rp._runtime(
                provider_label,
                api_mode_override or rp._detect_api_mode_for_url(base_url) or "chat_completions",
                base_url,
                pool_api_key,
                source=f"pool:{pool_key}",
                credential_pool=pool,
            )
        except Exception:
            continue
    return None


def _custom_provider_request_overrides(custom_provider: Dict[str, Any]) -> Dict[str, Any]:
    extra_body = custom_provider.get("extra_body")
    if not isinstance(extra_body, dict) or not extra_body:
        return {}
    return {"extra_body": dict(extra_body)}


def _apply_custom_provider_extras(
    custom_provider: Dict[str, Any], target_model: Optional[str], result: Dict[str, Any]
) -> None:
    """Copy model / capabilities / max_output_tokens / extra_headers / request_overrides onto a
    resolved custom runtime.

    An explicit ``target_model`` wins over the provider's configured default (auxiliary slots /
    background-review resolve a concrete model and must not fall back to ``default_model``).
    ``extra_headers`` may carry credentials — NEVER log them.
    """
    model_name = target_model or custom_provider.get("model")
    if model_name:
        result["model"] = model_name
    _lift_model_capabilities(custom_provider, model_name, result)
    if isinstance(custom_provider.get("max_output_tokens"), int):
        result["max_output_tokens"] = custom_provider["max_output_tokens"]
    if custom_provider.get("extra_headers"):
        result["extra_headers"] = dict(custom_provider["extra_headers"])
    request_overrides = _custom_provider_request_overrides(custom_provider)
    if request_overrides:
        result["request_overrides"] = {**dict(result.get("request_overrides") or {}), **request_overrides}


def _resolve_llamacpp_runtime(requested_provider: str, explicit_api_key: Optional[str]) -> Dict[str, Any]:
    """Managed llama.cpp runtime: the supervised (or detected external) server, or a typed error.

    No server => say so and stop; falling through to the generic custom path would surface "local
    server is off" as OpenRouter's baffling "401 Invalid API key". The switch's state picks the
    message (server off → point at the switch; else the setup pane).
    """
    rp = _rp()
    try:
        from hermes_cli.local_runtime.endpoint import resolve_llamacpp_endpoint

        endpoint = resolve_llamacpp_endpoint()
    except Exception:  # noqa: BLE001 — resolution is best-effort
        endpoint = None
    if endpoint:
        return rp._runtime(
            "custom",
            "chat_completions",
            endpoint["base_url"],
            (explicit_api_key or "").strip() or endpoint["api_key"] or "no-key-required",
            source="local-runtime",
            requested_provider=requested_provider,
        )
    try:
        enabled = bool((rp.load_config().get("local_runtime") or {}).get("enabled"))
    except Exception:  # noqa: BLE001
        enabled = False
    if enabled:
        raise ValueError(
            "The local model server isn't running. It may still be "
            "starting — try again in a moment, or check Settings → "
            "Providers → Local models."
        )
    raise ValueError(
        "The local model server is turned off. Turn it back on in "
        "Settings → Providers → Local models, or switch to another "
        "model."
    )


def _resolve_direct_alias_runtime(
    requested_provider: str, explicit_api_key: Optional[str], explicit_base_url: str
) -> Dict[str, Any]:
    """Bare ``custom`` + explicit base_url (e.g. a ``model_aliases:`` direct alias)."""
    rp = _rp()
    base_url = explicit_base_url.strip().rstrip("/")
    # Pool first — mirrors the named-custom path so bare `provider: custom` with a configured
    # custom_providers entry gets its api_key from the pool instead of env fallbacks.
    pool_result = rp._try_resolve_from_custom_pool(base_url, "custom", None)
    if pool_result:
        pool_result["source"] = "direct-alias"
        return pool_result
    # OLLAMA_API_KEY gets its own gate here: without it a `model_aliases:` entry pointing at
    # Ollama Cloud resolved no key at all.
    candidates = [(explicit_api_key or "").strip(), *rp._host_gated_env_key_candidates(base_url, ollama=True)]
    api_key = next((c for c in candidates if rp.has_usable_secret(c)), "") or "no-key-required"
    return rp._runtime(
        "custom",
        rp._detect_api_mode_for_url(base_url) or "chat_completions",
        base_url,
        api_key,
        source="direct-alias",
        requested_provider=requested_provider,
    )


def _opencode_family_for_custom(requested_provider: str, base_url: str) -> Optional[str]:
    """OpenCode family by provider name, else by opencode.ai host (``/zen/go`` => opencode-go)."""
    from hermes_cli.models import opencode_provider_family

    family = opencode_provider_family(requested_provider)
    if family is not None:
        return family
    try:
        if base_url_hostname(base_url).lower() == "opencode.ai":
            return "opencode-go" if "/zen/go" in base_url.lower() else "opencode-zen"
    except Exception:
        pass
    return None


def _resolve_named_custom_runtime(
    *,
    requested_provider: str,
    explicit_api_key: Optional[str] = None,
    explicit_base_url: Optional[str] = None,
    target_model: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Runtime for a llamacpp alias, a bare-custom direct alias, or a configured custom entry.

    Aliases resolving to "custom" (ollama, vllm, llamacpp, …) are treated like bare ``custom``.
    A llamacpp alias with no explicit base_url resolves to the managed server first; an explicit
    base_url always wins.
    """
    rp = _rp()
    requested_norm = (requested_provider or "").strip().lower()
    if requested_norm in ("llamacpp", "llama.cpp", "llama-cpp") and not explicit_base_url:
        return _resolve_llamacpp_runtime(requested_provider, explicit_api_key)
    if requested_norm and requested_norm != "custom" and rp._resolves_to_custom(requested_norm):
        requested_norm = "custom"
    if requested_norm == "custom" and explicit_base_url:
        return _resolve_direct_alias_runtime(requested_provider, explicit_api_key, explicit_base_url)

    custom_provider = rp._get_named_custom_provider(requested_provider)
    if not custom_provider:
        return None
    base_url = ((explicit_base_url or "").strip() or custom_provider.get("base_url", "")).rstrip("/")
    if not base_url:
        return None

    pool_result = rp._try_resolve_from_custom_pool(
        base_url,
        "custom",
        custom_provider.get("api_mode"),
        provider_name=custom_provider.get("provider_key") or custom_provider.get("name"),
    )
    if pool_result:
        # The pool doesn't know the custom_providers fields — propagate them here too.
        _apply_custom_provider_extras(custom_provider, target_model, pool_result)
        return pool_result

    candidates = [
        (explicit_api_key or "").strip(),
        _clean(custom_provider.get("api_key", "")),
        rp._getenv(_clean(custom_provider.get("key_env", "")), "").strip(),
        *rp._host_gated_env_key_candidates(base_url, ollama=False),
    ]
    api_key: Any = next((c for c in candidates if rp.has_usable_secret(c)), "")

    # ``key_cmd`` credentials are minted per request (short-lived bearers would go stale
    # mid-session); both wire clients accept a callable api_key (the Entra ID contract). An
    # explicit --api-key still wins as the one-off recovery escape hatch.
    key_cmd = _clean(custom_provider.get("key_cmd", ""))
    if key_cmd and not rp.has_usable_secret((explicit_api_key or "").strip()):
        from agent.command_token_source import build_command_token_provider

        token_provider = build_command_token_provider(
            key_cmd, str(custom_provider.get("name", requested_provider) or "custom")
        )
        if token_provider is not None:
            api_key = token_provider

    result = rp._runtime(
        "custom",
        custom_provider.get("api_mode") or rp._detect_api_mode_for_url(base_url) or "chat_completions",
        base_url,
        api_key or "no-key-required",
        source=f"custom_provider:{custom_provider.get('name', requested_provider)}",
        requested_provider=requested_provider,
    )
    _apply_custom_provider_extras(custom_provider, target_model, result)

    # OpenCode-family custom providers (opencode-go/zen names, or opencode.ai hosts) serve models
    # on different API surfaces — a static api_mode 503s for /v1/responses-only models. Re-derive
    # api_mode from the model and normalize /v1 like the built-in paths.
    family = _opencode_family_for_custom(requested_provider, base_url)
    if family is not None and not custom_provider.get("api_mode"):
        from hermes_cli.models import normalize_opencode_base_url, opencode_model_api_mode

        effective_model = str(
            target_model or custom_provider.get("model") or rp._get_model_config().get("default") or ""
        ).strip()
        if effective_model:
            result["api_mode"] = opencode_model_api_mode(family, effective_model)
        result["base_url"] = normalize_opencode_base_url(family, result["api_mode"], result["base_url"])
    return result
