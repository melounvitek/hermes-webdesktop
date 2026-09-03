"""Custom-provider entry normalization and per-route lookups (TLS, headers, context length, capabilities).

Split out of ``hermes_cli/config.py``; every name is re-imported there, so
``hermes_cli.config.<name>`` keeps resolving (and monkeypatching) as before.
"""

import logging
import re
from typing import Any, Dict, List, Optional, Tuple
from hermes_cli.route_identity import normalize_route_base_url

# Log-record parity with the origin module.
logger = logging.getLogger("hermes_cli.config")


# ``_normalize_custom_provider_entry`` runs on every ``load_picker_context()``
# call (i.e. per interactive picker/inventory request), so any warning it emits
# fires repeatedly for the same static config. Deduplicate per (provider,
# signature): on Windows a repeated-warning storm contends on
# ``concurrent-log-handler``'s cross-process rotation lock and can peg a core /
# stall the gateway/serve event loop. The cache lives for the process lifetime.
_PROVIDER_NORMALIZE_WARNED: set = set()


def _warn_once_per_provider(
    provider_key: str, signature: str, msg: str, *args: Any
) -> None:
    """Emit ``logger.warning(msg, *args)`` at most once per (provider, signature)."""
    dedup_key = (provider_key or "?", signature)
    if dedup_key in _PROVIDER_NORMALIZE_WARNED:
        return
    _PROVIDER_NORMALIZE_WARNED.add(dedup_key)
    logger.warning(msg, *args)


_API_MODE_ALIASES = {
    # Values accepted by earlier releases (and natural spellings) mapped to
    # the canonical transport names consumed by agent_init. Before this map
    # existed, an unrecognized api_mode was silently ignored and the
    # transport fell through to hostname-based guessing, so a config that
    # said ``api_mode: openai`` (valid on older releases) could flip to
    # ``codex_responses`` after an update and break the provider (#66543
    # discussion; observed live against api.actual.inc).
    "openai": "chat_completions",
    "openai_chat": "chat_completions",
    "openai-chat": "chat_completions",
    "chat-completions": "chat_completions",
    "chatcompletions": "chat_completions",
    "responses": "codex_responses",
    "openai_responses": "codex_responses",
    "openai-responses": "codex_responses",
    "anthropic": "anthropic_messages",
    "anthropic-messages": "anthropic_messages",
    "messages": "anthropic_messages",
    "bedrock": "bedrock_converse",
    "bedrock-converse": "bedrock_converse",
}


def _canonical_api_mode(api_mode: str) -> str:
    """Map legacy/alias ``api_mode`` spellings to canonical transport names.

    Unknown values pass through unchanged (callers keep their existing fall-through behavior); known
    aliases are rewritten so downstream consumers (``agent_init``'s accepted-set check, runtime
    resolution) see a canonical name instead of silently discarding the user's intent.
    """
    cleaned = api_mode.strip()
    return _API_MODE_ALIASES.get(cleaned.lower(), cleaned)


def coerce_provider_id(value: Any) -> str:
    """Provider identity fields are strings."""
    if value is None:
        return ""
    return str(value).strip()


def stringify_provider_map(providers: Any) -> dict:
    """Copy a ``providers:`` mapping so keys are strings.

    Desktop Custom Endpoints store the name as the dict key; an unquoted YAML key ``2070:`` loads
    as int, so picker code calling ``ep_name.lower()`` crashes and CRUD lookups of ``"2070"`` miss.
    """
    if not isinstance(providers, dict):
        return {}
    out: Dict[str, Any] = {}
    for stored, value in providers.items():
        key = coerce_provider_id(stored)
        if key:
            out[key] = value
    return out


def find_provider_entry(providers: Any, key: Any) -> Tuple[Any, Optional[Dict[str, Any]]]:
    """Return ``(stored_key, entry)`` matching *key* by string identity.

    Prefer an exact string hit, then scan.
    """
    if not isinstance(providers, dict):
        return None, None
    want = coerce_provider_id(key)
    if not want:
        return None, None
    exact = providers.get(want)
    if isinstance(exact, dict):
        return want, exact
    for stored, entry in providers.items():
        if coerce_provider_id(stored) == want and isinstance(entry, dict):
            return stored, entry
    return None, None


# camelCase aliases commonly used in hand-written provider configs.
_CAMEL_ALIASES: Dict[str, str] = {
    "apiKey": "api_key",
    "baseUrl": "base_url",
    "apiMode": "api_mode",
    "keyEnv": "key_env",
    "apiKeyEnv": "key_env",  # alias — OpenClaw-compatible + docs variant
    "defaultModel": "default_model",
    "contextLength": "context_length",
    "rateLimitDelay": "rate_limit_delay",
}


_KNOWN_PROVIDER_KEYS = {
    # ``provider`` duplicates the ``providers.<name>`` mapping key and is unused
    # here, but Hermes' own config writer has historically emitted it into
    # provider entries. Accept it silently so self-written configs don't warn.
    "provider",
    "name", "api", "url", "base_url", "api_key", "key_env", "api_key_env", "key_cmd",
    "api_mode", "transport", "model", "default_model", "models", "models_discovered",
    "context_length", "rate_limit_delay", "request_timeout_seconds", "stale_timeout_seconds",
    "discover_models", "extra_body", "extra_headers", "capabilities", "ssl_ca_cert", "ssl_verify",
}


def _pick_provider_base_url(entry: Dict[str, Any], provider_key: str) -> str:
    """First usable URL among ``base_url``/``url``/``api``, or "".

    URLs containing unresolved placeholder tokens — ``${ENV_VAR}`` env-refs and bare ``{region}``
    templates — are accepted without validation: they are expanded at runtime, and a caller reaching
    this normalizer with raw config would otherwise see the provider silently dropped.
    """
    from urllib.parse import urlparse

    for url_key in ("base_url", "url", "api"):
        raw_url = entry.get(url_key)
        if not (isinstance(raw_url, str) and raw_url.strip()):
            continue
        candidate = raw_url.strip()
        if re.search(r"\{[^}]+\}", candidate):
            return candidate
        parsed = urlparse(candidate)
        if parsed.scheme and parsed.netloc:
            return candidate
        logger.warning(
            "providers.%s: '%s' value '%s' is not a valid URL "
            "(no scheme or host) — skipped",
            provider_key or "?", url_key, candidate,
        )
    return ""


def _normalize_provider_models(models: Any) -> Tuple[Dict[str, Any], bool]:
    """Normalize an entry's ``models`` to the dict shape downstream expects.

    Returns ``(models_dict, discovered_flag)``. Older Hermes versions wrote an in-mapping
    ``__discovered_model_catalog__`` sentinel (accepted on read, stripped so sentinel keys never
    surface as model IDs). Hand-edited/older configs may write a plain list of ids or ``[{id: ...}]``
    rows; both are converted so /model doesn't show the provider with (0) models.
    """
    discovered = False
    if isinstance(models, dict) and models:
        # Shallow-copy: `models` may alias a cached config sub-dict, and the
        # normalized entry escapes into long-lived runtime state.
        models_copy = dict(models)
        if models_copy.pop("__discovered_model_catalog__", None) is True:
            discovered = True
        models_copy.pop("__explicit_model_allowlist__", None)
        return models_copy, discovered
    if isinstance(models, list) and models:
        normalized_models: Dict[str, Any] = {}
        for item in models:
            if isinstance(item, str) and item.strip():
                normalized_models[item.strip()] = {}
                continue
            if not isinstance(item, dict):
                continue
            model_id = item.get("id")
            if not isinstance(model_id, str) or not model_id.strip():
                model_id = item.get("name")
            if not isinstance(model_id, str) or not model_id.strip():
                continue
            normalized_models[model_id.strip()] = {
                k: v for k, v in item.items() if k not in {"id", "name"}
            }
        return normalized_models, discovered
    return {}, discovered


def _normalize_custom_provider_entry(
    entry: Any,
    *,
    provider_key: str = "",
) -> Optional[Dict[str, Any]]:
    """Return a runtime-compatible custom provider entry or ``None``."""
    from hermes_cli.config import normalize_extra_headers
    if not isinstance(entry, dict):
        return None

    # Shallow-copy before alias normalization writes into the entry: callers
    # pass live sub-dicts from load_config_readonly()'s shared cache, and
    # mutating those violates the cache's no-mutation contract and leaks alias
    # keys back into config.yaml on a later save_config(load_config()).
    entry = dict(entry)
    provider_key = coerce_provider_id(provider_key)

    # api_key_env is a documented snake_case alias for key_env (see
    # website/docs/guides/azure-foundry.md).  Normalize it up front so the
    # rest of the normalizer treats it as the canonical field.
    if "api_key_env" in entry and "key_env" not in entry:
        entry["key_env"] = entry["api_key_env"]
    for camel, snake in _CAMEL_ALIASES.items():
        if camel in entry and snake not in entry:
            _warn_once_per_provider(
                provider_key, f"camel:{camel}",
                "providers.%s: camelCase key '%s' auto-mapped to '%s' "
                "(use snake_case to avoid this warning)",
                provider_key or "?", camel, snake,
            )
            entry[snake] = entry[camel]
    unknown = set(entry.keys()) - _KNOWN_PROVIDER_KEYS - set(_CAMEL_ALIASES.keys())
    if unknown:
        _warn_once_per_provider(
            provider_key, "unknown:" + ",".join(sorted(unknown)),
            "providers.%s: unknown config keys ignored: %s",
            provider_key or "?", ", ".join(sorted(unknown)),
        )

    base_url = _pick_provider_base_url(entry, provider_key)
    if not base_url:
        return None

    name = coerce_provider_id(entry.get("name")) or provider_key
    if not name:
        return None

    normalized: Dict[str, Any] = {
        "name": name,
        "base_url": base_url,
    }

    provider_key = provider_key.strip()
    if provider_key:
        normalized["provider_key"] = provider_key

    def _stripped(*keys: str) -> str:
        val = None
        for k in keys:
            val = entry.get(k)
            if val:
                break
        return val.strip() if isinstance(val, str) else ""

    if _stripped("api_key"):
        normalized["api_key"] = _stripped("api_key")

    key_env = _stripped("key_env", "api_key_env")
    if key_env:
        normalized["key_env"] = key_env
        if entry.get("api_key_env") and not entry.get("key_env"):
            normalized["api_key_env"] = key_env

    api_mode = _stripped("api_mode", "transport")
    if api_mode:
        normalized["api_mode"] = _canonical_api_mode(api_mode)

    model_name = _stripped("model", "default_model")
    if model_name:
        normalized["model"] = model_name

    # Entry-level marker: the ``models`` mapping was auto-discovered by Hermes
    # (``_save_discovered_models_to_config``), not hand-curated.
    models_dict, discovered = _normalize_provider_models(entry.get("models"))
    if models_dict:
        normalized["models"] = models_dict
    if entry.get("models_discovered") is True or discovered:
        normalized["models_discovered"] = True

    capabilities = entry.get("capabilities")
    if isinstance(capabilities, dict):
        normalized_capabilities = {
            key: value
            for key, value in capabilities.items()
            if isinstance(key, str) and isinstance(value, bool)
        }
        if normalized_capabilities:
            normalized["capabilities"] = normalized_capabilities

    context_length = entry.get("context_length")
    if isinstance(context_length, int) and context_length > 0:
        normalized["context_length"] = context_length

    rate_limit_delay = entry.get("rate_limit_delay")
    if isinstance(rate_limit_delay, (int, float)) and rate_limit_delay >= 0:
        normalized["rate_limit_delay"] = rate_limit_delay

    if isinstance(entry.get("discover_models"), bool):
        normalized["discover_models"] = entry["discover_models"]

    if isinstance(entry.get("extra_body"), dict):
        normalized["extra_body"] = dict(entry["extra_body"])

    # Per-provider extra HTTP headers (proxies, gateways, custom auth).
    # Values may carry credentials (e.g. CF-Access-Client-Secret) — never
    # log them anywhere downstream.
    normalized_headers = normalize_extra_headers(entry.get("extra_headers"))
    if normalized_headers:
        normalized["extra_headers"] = normalized_headers

    ssl_ca_cert = entry.get("ssl_ca_cert")
    if isinstance(ssl_ca_cert, str) and ssl_ca_cert.strip():
        normalized["ssl_ca_cert"] = ssl_ca_cert.strip()

    ssl_verify = entry.get("ssl_verify")
    if isinstance(ssl_verify, bool):
        normalized["ssl_verify"] = ssl_verify
    elif isinstance(ssl_verify, str) and ssl_verify.strip():
        normalized["ssl_verify"] = ssl_verify.strip()

    return normalized


def _custom_provider_entry_to_provider_config(
    entry: Any,
    *,
    provider_key: str = "",
) -> Optional[Dict[str, Any]]:
    """Translate a legacy custom provider entry to the v12 providers shape."""
    from hermes_cli.config import _normalize_custom_provider_entry
    normalized = _normalize_custom_provider_entry(
        dict(entry) if isinstance(entry, dict) else entry,
        provider_key=provider_key,
    )
    if normalized is None:
        return None

    provider_entry: Dict[str, Any] = {"api": normalized["base_url"]}

    for field in (
        "name", "api_key", "key_env", "models", "models_discovered", "context_length",
        "rate_limit_delay", "discover_models", "extra_body", "extra_headers",
        "ssl_ca_cert", "ssl_verify",
    ):
        if field in normalized:
            provider_entry[field] = normalized[field]

    if "model" in normalized:
        provider_entry["default_model"] = normalized["model"]
    if "api_mode" in normalized:
        provider_entry["transport"] = normalized["api_mode"]

    return provider_entry


def providers_dict_to_custom_providers(providers_dict: Any) -> List[Dict[str, Any]]:
    """Normalize ``providers`` config entries into the legacy custom-provider shape."""
    from hermes_cli.config import _normalize_custom_provider_entry
    if not isinstance(providers_dict, dict):
        return []

    custom_providers: List[Dict[str, Any]] = []
    for key, entry in providers_dict.items():
        if isinstance(entry, dict) and not is_provider_enabled(entry):
            continue
        normalized = _normalize_custom_provider_entry(
            entry, provider_key=coerce_provider_id(key)
        )
        if normalized is not None:
            custom_providers.append(normalized)

    return custom_providers


def get_compatible_custom_providers(
    config: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """Return a deduplicated custom-provider view across legacy and v12+ config.

    ``custom_providers`` remains the on-disk legacy format, while ``providers`` is the newer keyed
    schema. Runtime and picker flows still need a single list-shaped view, but we should not
    materialise that compatibility layer back into config.yaml because it duplicates entries in UIs.
    """
    from hermes_cli.config import _normalize_custom_provider_entry, load_config
    if config is None:
        config = load_config()

    compatible: List[Dict[str, Any]] = []
    seen_provider_keys: set = set()
    seen_name_url_pairs: set = set()

    def _append_if_new(entry: Optional[Dict[str, Any]]) -> None:
        if entry is None:
            return
        provider_key = str(entry.get("provider_key", "") or "").strip().lower()
        name = str(entry.get("name", "") or "").strip().lower()
        base_url = str(entry.get("base_url", "") or "").strip().rstrip("/").lower()
        model = str(entry.get("model", "") or "").strip().lower()
        pair = (name, base_url, model)

        if provider_key and provider_key in seen_provider_keys:
            return
        if name and base_url and pair in seen_name_url_pairs:
            return

        compatible.append(entry)
        if provider_key:
            seen_provider_keys.add(provider_key)
        if name and base_url:
            seen_name_url_pairs.add(pair)

    custom_providers = config.get("custom_providers")
    if custom_providers is not None:
        if not isinstance(custom_providers, list):
            return []
        for entry in custom_providers:
            _append_if_new(_normalize_custom_provider_entry(entry))

    for entry in providers_dict_to_custom_providers(config.get("providers")):
        _append_if_new(entry)

    return compatible


def _entries_for_route(
    base_url: str,
    custom_providers: Optional[List[Dict[str, Any]]],
    config: Optional[Dict[str, Any]],
):
    """Yield custom-provider entries whose normalized route identity equals *base_url*.

    Loads ``get_compatible_custom_providers(config)`` when *custom_providers* is None (failures →
    no entries). Yields nothing for an empty *base_url* or non-list input.
    """
    from hermes_cli.config import get_compatible_custom_providers
    if custom_providers is None:
        try:
            custom_providers = get_compatible_custom_providers(config)
        except Exception:
            custom_providers = []
    if not base_url or not isinstance(custom_providers, list):
        return
    target_url = normalize_route_base_url(base_url)
    if not target_url:
        return
    for entry in custom_providers:
        if not isinstance(entry, dict):
            continue
        entry_url = normalize_route_base_url(entry.get("base_url"))
        if entry_url and entry_url == target_url:
            yield entry


def _route_model_cfg(entry: Dict[str, Any], model: str) -> Optional[Dict[str, Any]]:
    """Return ``entry.models[model]`` when both are mappings, else None."""
    models = entry.get("models")
    if not isinstance(models, dict):
        return None
    model_cfg = models.get(model)
    return model_cfg if isinstance(model_cfg, dict) else None


def _coerce_ssl_verify(value: Any) -> Optional[bool]:
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"false", "0", "no", "off"}:
            return False
        if lowered in {"true", "1", "yes", "on"}:
            return True
    return None


def get_custom_provider_tls_settings(
    base_url: str,
    custom_providers: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return TLS settings from a matching ``custom_providers`` / ``providers`` entry."""
    for entry in _entries_for_route(base_url, custom_providers, config):
        out: Dict[str, Any] = {}
        ca = entry.get("ssl_ca_cert")
        if isinstance(ca, str) and ca.strip():
            out["ssl_ca_cert"] = ca.strip()
        verify = _coerce_ssl_verify(entry.get("ssl_verify"))
        if verify is not None:
            out["ssl_verify"] = verify
        return out
    return {}


def apply_custom_provider_tls_to_client_kwargs(
    client_kwargs: Dict[str, Any],
    base_url: str,
    custom_providers: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Attach per-provider TLS knobs to OpenAI client kwargs when matched."""
    tls = get_custom_provider_tls_settings(base_url, custom_providers, config)
    if tls.get("ssl_ca_cert"):
        client_kwargs["ssl_ca_cert"] = tls["ssl_ca_cert"]
    if "ssl_verify" in tls:
        client_kwargs["ssl_verify"] = tls["ssl_verify"]


def normalize_extra_headers(extra_headers: Any) -> Dict[str, str]:
    """Normalize a raw ``extra_headers`` value into a ``dict[str, str]``.

    SECURITY: header values routinely carry credentials (Cloudflare Access service tokens, proxy
    auth, custom bearer schemes). Callers must never log the returned values.
    """
    if not isinstance(extra_headers, dict) or not extra_headers:
        return {}
    return {str(k): str(v) for k, v in extra_headers.items() if v is not None}


def get_custom_provider_extra_headers(
    base_url: str,
    custom_providers: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, str]:
    """Return ``extra_headers`` from a matching ``providers`` / ``custom_providers`` entry.

    Matches the entry whose normalized route identity equals *base_url*, mirroring
    :func:`get_custom_provider_tls_settings`, and returns its ``extra_headers`` dict, or ``{}`` when
    no entry matches or declares none.

    SECURITY: header values routinely carry credentials (Cloudflare Access service tokens, proxy
    auth, custom bearer schemes). Callers must never log the returned values.
    """
    from hermes_cli.config import normalize_extra_headers
    for entry in _entries_for_route(base_url, custom_providers, config):
        headers = normalize_extra_headers(entry.get("extra_headers"))
        if headers:
            return headers
    return {}


def apply_custom_provider_extra_headers_to_client_kwargs(
    client_kwargs: Dict[str, Any],
    base_url: str,
    custom_providers: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """Merge per-provider ``extra_headers`` onto OpenAI client ``default_headers``.

    Provider-specific headers win over SDK/provider defaults already in ``client_kwargs`` (they
    are the most specific level). No-op when base_url matches no provider entry or none declares
    headers. SECURITY: values may carry credentials -- never log them.
    """
    extra_headers = get_custom_provider_extra_headers(base_url, custom_providers, config)
    if not extra_headers:
        return
    merged = dict(client_kwargs.get("default_headers") or {})
    merged.update(extra_headers)
    client_kwargs["default_headers"] = merged


def get_custom_provider_context_length(
    model: str,
    base_url: str,
    custom_providers: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[int]:
    """Look up a per-model ``context_length`` override from ``custom_providers``.

    Matches any entry whose normalized route identity equals ``base_url`` and returns
    ``custom_providers[i].models.<model>.context_length`` if present and valid. Returns ``None``
    when no override applies.
    """
    from hermes_cli.config import get_compatible_custom_providers
    if not model or not base_url:
        return None
    if custom_providers is None:
        try:
            custom_providers = get_compatible_custom_providers(config)
        except Exception:
            if config is None:
                return None
            raw = config.get("custom_providers")
            custom_providers = raw if isinstance(raw, list) else []

    for entry in _entries_for_route(base_url, custom_providers, config):
        model_cfg = _route_model_cfg(entry, model)
        if model_cfg is None:
            continue
        raw_ctx = model_cfg.get("context_length")
        if raw_ctx is None:
            continue
        try:
            ctx = int(raw_ctx)
        except (TypeError, ValueError):
            continue
        if ctx > 0:
            return ctx
    return None


def get_custom_provider_model_capability(
    model: str,
    base_url: str,
    capability: str,
    custom_providers: Optional[List[Dict[str, Any]]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> Optional[bool]:
    """Return an explicit boolean capability for one custom-provider model.

    Matching is scoped to the normalized route and exact runtime model id so aliases can declare
    capabilities without changing the id sent upstream. Missing or non-boolean declarations return
    ``None``.
    """
    from hermes_cli.config import get_compatible_custom_providers, load_config_readonly
    if not model or not base_url or not capability:
        return None
    if custom_providers is None:
        try:
            if config is None:
                # Read-only path: this helper never mutates the entries it
                # scans, and get_compatible_custom_providers shallow-copies
                # each entry before normalizing, so the no-deepcopy cache is
                # safe here (~135us saved per call on the blank-stub paths).
                config = load_config_readonly()
            custom_providers = get_compatible_custom_providers(config)
        except Exception:
            return None

    for entry in _entries_for_route(base_url, custom_providers, config):
        model_cfg = _route_model_cfg(entry, model)
        if model_cfg is None:
            continue
        value = model_cfg.get(capability)
        if isinstance(value, bool):
            return value
    return None


def is_provider_enabled(provider_cfg: Optional[Dict[str, Any]]) -> bool:
    """Return whether a ``providers.<name>`` config block is enabled.

    A provider is enabled by default. Only an explicit ``enabled: false`` in the block hides it from
    the model picker, ``/models`` listings, the runtime resolver and the doctor / status output.

    Backward-compat: configs without the ``enabled`` key keep working as before — the default is
    ``True``.
    """
    if not isinstance(provider_cfg, dict):
        return True
    flag = provider_cfg.get("enabled", True)
    if isinstance(flag, bool):
        return flag
    # YAML can produce strings for "true"/"false" depending on quoting.
    if isinstance(flag, str):
        return flag.strip().lower() not in {"false", "0", "no", "off"}
    return bool(flag)
