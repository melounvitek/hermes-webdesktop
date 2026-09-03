"""Picker provider listing: credential discovery, curated/live model lists, row builders for list_authenticated_providers / list_picker_providers, and the parallel cache prefetch.

Split out of ``hermes_cli/model_switch.py``; every moved name is re-imported there so
``hermes_cli.model_switch.<name>`` keeps resolving (and monkeypatching) as before.
"""

from __future__ import annotations

import logging
import http.client
import os
import time
import threading as _threading
from dataclasses import dataclass, field
from typing import Any, List, Optional
from hermes_cli.providers import (
    custom_provider_aliases,
    custom_provider_slug,
    get_label,
)
from utils import base_url_host_matches

# Log-record parity with the origin module.
logger = logging.getLogger("hermes_cli.model_switch")


# Providers whose picker model list should NOT be capped by max_models.
# OpenCode Zen / Go are aggregators whose full catalogs (70+ models each) must
# be visible so users can pick any model they have access to.
_UNCAPPED_PICKER_PROVIDERS: frozenset[str] = frozenset({"opencode-zen", "opencode-go"})


def _save_discovered_models_to_config(
    api_url: str,
    model_ids: list[str],
    *,
    api_mode: Optional[str] = None,
    headers: Optional[dict[str, str]] = None,
) -> None:
    """Persist discovered models into ``custom_providers`` in config.yaml.

    Called after a successful ``/v1/models`` probe so that the next read
    with ``discover_models: false`` uses the cached list instead of a stale
    or minimal manually-configured subset.

    Matches entries by ``base_url`` (trailing-slash-normalised).  A failed
    config write is swallowed — the picker still shows the live models for
    this session.
    """
    from hermes_cli.model_switch import _extra_headers_from_config
    if not api_url or not model_ids:
        return
    try:
        from hermes_cli.config import load_config, save_config

        cfg = load_config()
        providers = cfg.get("custom_providers") or []
        if not isinstance(providers, list):
            return

        norm_url = api_url.strip().rstrip("/").lower()
        changed = False
        for entry in providers:
            if not isinstance(entry, dict):
                continue
            entry_url = (entry.get("base_url", "") or entry.get("url", "")).strip()
            if entry_url.rstrip("/").lower() != norm_url:
                continue
            entry_mode = str(
                entry.get("api_mode") or entry.get("transport") or ""
            ).strip().lower() or None
            if entry_mode != api_mode:
                continue
            if headers is not None:
                entry_headers = _extra_headers_from_config(entry)
                if entry_headers != headers:
                    continue
            existing = entry.get("models")
            legacy_discovered = (
                isinstance(existing, dict)
                and existing.get("__discovered_model_catalog__") is True
            )
            entry_discovered = (
                entry.get("models_discovered") is True or legacy_discovered
            )
            # Preserve per-model metadata: when ``models`` is a mapping
            # (e.g. ``{"model-a": {"context_length": 8192}}``) or a list of
            # dicts (e.g. ``[{"id": "model-a", "context_length": 8192}]``),
            # the user has curated metadata per model — do not replace it.
            # A mapping Hermes itself discovered (``models_discovered: true``
            # or the legacy in-mapping sentinel) is ours to refresh.
            if isinstance(existing, dict) and not entry_discovered:
                continue
            if isinstance(existing, list) and any(
                isinstance(m, dict) for m in existing
            ):
                continue
            # Only update when models are stale — avoids unnecessary
            # config writes on every picker open.  A legacy-shape entry
            # (sentinel inside ``models``) is always rewritten so the next
            # save migrates it to the clean entry-level flag.
            if isinstance(existing, list) and existing == model_ids:
                continue
            if (
                isinstance(existing, dict)
                and entry_discovered
                and not legacy_discovered
                and list(existing) == model_ids
            ):
                continue
            entry["models"] = {model_id: {} for model_id in model_ids}
            entry["models_discovered"] = True
            changed = True

        if changed:
            cfg["custom_providers"] = providers
            save_config(cfg)
    except Exception:
        pass


_MODEL_DISCOVERY_ERRORS = (
    ImportError,
    OSError,
    RuntimeError,
    TimeoutError,
    TypeError,
    ValueError,
    http.client.HTTPException,
)


class _NativePickerModelList(list[str]):
    """A successful native catalog, including an authoritative empty one."""


def _fetch_picker_live_models(
    api_key: str,
    api_url: str,
    native_catalog_provider: str,
    preserve_native_models: bool,
    headers: dict[str, str] | None = None,
    timeout: float = 5.0,
    api_mode: str | None = None,
) -> list[str] | None:
    """Fetch picker models with native Ollama and cached generic discovery."""
    from hermes_cli.models import (
        _get_ollama_native_headers,
        _normalize_openai_base_url,
        cached_fetch_api_models,
        fetch_ollama_local_models,
        should_use_ollama_native_catalog,
    )

    candidate_headers = _get_ollama_native_headers(api_url, api_key=api_key)
    caller_has_authorization = any(
        key.lower() == "authorization" for key in (headers or {})
    )
    if caller_has_authorization:
        for key in tuple(candidate_headers):
            if key.lower() == "authorization":
                del candidate_headers[key]
    if headers:
        for key in tuple(candidate_headers):
            if any(key.lower() == existing.lower() for existing in headers):
                del candidate_headers[key]
        candidate_headers.update(headers)
    if api_key and not caller_has_authorization:
        for key in tuple(candidate_headers):
            if key.lower() == "authorization":
                del candidate_headers[key]
        candidate_headers["Authorization"] = f"Bearer {api_key}"
    use_native = should_use_ollama_native_catalog(
        native_catalog_provider, api_url, headers=candidate_headers or None
    )
    resolved_headers = candidate_headers or None if use_native else headers

    if use_native:
        if preserve_native_models:
            return None
        native_models = fetch_ollama_local_models(
            api_url, timeout=timeout, headers=resolved_headers
        )
        if native_models is not None:
            return _NativePickerModelList(native_models)
        # A failed native probe is not authoritative: retry the cached generic
        # OpenAI-compatible catalog before reporting no models.
        return cached_fetch_api_models(
            api_key,
            _normalize_openai_base_url(api_url),
            timeout=timeout,
            headers=resolved_headers,
            api_mode=api_mode,
        )
    generic_models = cached_fetch_api_models(
        api_key,
        api_url,
        timeout=timeout,
        headers=resolved_headers,
        api_mode=api_mode,
    )
    return generic_models if generic_models else None


# Process-level guard so the picker prewarm thread is spawned at most once per
# process — mirrors run_agent's _openrouter_prewarm_done. Without a guard a
# long-lived process (or repeated triggers) would leak one OS thread per call.
_picker_prewarm_done = _threading.Event()


def _credential_pool_is_usable(provider: str, *, raw_pool_present: bool = False) -> bool:
    """Return whether *provider* has a credential that can be selected now.

    ``auth.json`` historically allowed opaque token-style pool values that do
    not deserialize into ``PooledCredential`` entries. Preserve visibility for
    those legacy values, but when a real pool exists its availability state is
    authoritative: an all-exhausted/dead pool is not authenticated.
    """
    try:
        from agent.credential_pool import load_pool

        pool = load_pool(provider)
        if pool.has_credentials():
            return pool.has_available()
    except Exception:
        pass
    return raw_pool_present


def prewarm_picker_cache_async() -> Optional["_threading.Thread"]:
    """Warm the provider-models disk cache in a background daemon thread.

    The no-args ``/model`` picker calls ``list_authenticated_providers()``,
    which fetches each authenticated provider's live ``/v1/models`` list on a
    cold/stale cache. Those fetches are independent HTTP round-trips but run
    serially, so the first ``/model`` open in a session (or any open after the
    1h cache TTL expires) blocks ~1-2s on the user's critical path.

    This pre-warms that exact path off-thread during idle session time: it
    runs ``list_authenticated_providers()`` once, which populates
    ``provider_models_cache.json`` for every authed provider. By the time the
    user types ``/model``, the picker hits the warm disk cache and renders in
    ~100ms.

    Fire-and-forget. Process-level Event guard ensures it runs at most once.
    Fully exception-isolated — a slow or offline provider can never affect the
    session. Returns the spawned thread (for tests) or None if already warmed.
    """
    from hermes_cli.model_switch import list_authenticated_providers
    if _picker_prewarm_done.is_set():
        return None
    _picker_prewarm_done.set()

    def _warm() -> None:
        try:
            from hermes_cli.inventory import load_picker_context

            ctx = load_picker_context()
            # Calling this is what populates cached_provider_model_ids() ->
            # provider_models_cache.json for each authed provider. We discard
            # the result; the side effect (warm disk cache) is the point.
            list_authenticated_providers(
                current_provider=ctx.current_provider,
                current_base_url=ctx.current_base_url,
                current_model=ctx.current_model,
                user_providers=ctx.user_providers,
                custom_providers=ctx.custom_providers,
                excluded_providers=ctx.excluded_providers or [],
            )
        except Exception:
            # Best-effort warmup — never surface errors into the session.
            logger.debug("picker cache prewarm failed", exc_info=True)

    t = _threading.Thread(target=_warm, daemon=True, name="picker-cache-prewarm")
    t.start()
    return t


_PARALLEL_PREFETCH_WORKERS = 8


def _prefetch_provider_models_parallel(provider_slugs: list[str]) -> None:
    """Fetch model catalogs for multiple providers in parallel.

    Run before the picker build loop: when the 1h disk cache lapses (or on a
    cold first open) ``list_authenticated_providers`` would otherwise call
    ``cached_provider_model_ids`` serially, blocking 1-8s per provider on a live
    /v1/models round-trip (15-30s+ with 10+ providers); after the prefetch the
    loop hits warm entries and total wait is the slowest single provider.

    Only providers whose cache entry is stale or missing are fetched; fresh
    entries are skipped to avoid unnecessary network calls.  Each worker uses
    :func:`update_provider_cache_entry` (thread-safe) to persist its result,
    so concurrent writes to ``provider_models_cache.json`` don't clobber each
    other.

    :param provider_slugs: Hermes provider IDs to prefetch (e.g. ``["openrouter",
        "anthropic", "deepseek"]``).  Unknown providers are silently skipped.
    """
    from hermes_cli.models import cached_provider_model_ids

    # Quick-stale-check: skip providers whose cache is already fresh so we
    # don't waste network calls on a warm cache.  We check staleness the same
    # way cached_provider_model_ids does internally: load the cache, compare
    # age to TTL.  This is a read-only check — if the cache file changes
    # between this check and the actual fetch, cached_provider_model_ids will
    # still do the right thing (it re-reads the cache internally).
    from hermes_cli.models import (
        _load_provider_models_cache,
        _credential_fingerprint,
        _PROVIDER_MODELS_CACHE_TTL,
        normalize_provider,
    )

    now = time.time()
    stale_slugs: list[str] = []
    cache = _load_provider_models_cache()
    for slug in provider_slugs:
        normalized = normalize_provider(slug) or (slug or "")
        if not normalized:
            continue
        entry = cache.get(normalized)
        fp = _credential_fingerprint(normalized)
        if (
            isinstance(entry, dict)
            and entry.get("fp") == fp
            and isinstance(entry.get("models"), list)
            and entry["models"]
        ):
            age = now - float(entry.get("at", 0))
            if age < _PROVIDER_MODELS_CACHE_TTL:
                continue  # fresh, skip
        stale_slugs.append(normalized)

    if not stale_slugs:
        return

    import concurrent.futures

    def _fetch_one(slug: str) -> None:
        try:
            models = cached_provider_model_ids(slug, force_refresh=True)
            # cached_provider_model_ids already persists the result, but in a
            # non-locked read-modify-write.  Re-persist via the thread-safe
            # path to guarantee no lost writes under concurrency.
            if models:
                from hermes_cli.models import update_provider_cache_entry
                update_provider_cache_entry(slug, models)
        except Exception:
            pass  # best-effort; picker falls back to curated list

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=min(_PARALLEL_PREFETCH_WORKERS, len(stale_slugs)),
        thread_name_prefix="model-cache-prefetch",
    ) as executor:
        list(executor.map(_fetch_one, stale_slugs))


def _iter_builtin_candidates(models_dev_data: dict, excluded: set, seen: set):
    """Yield ``(hermes_id, mdev_id, pconfig, env_vars)`` for section-1 rows.

    Skips vendor names that are aliases routing through an aggregator (bare
    "openai" -> "openrouter": emitting them would silently switch a user onto an
    endpoint they may have no key for), hermes_ids that are aliases of another
    canonical profile ("kimi" -> "kimi-coding"), non-api_key auth types (section
    2 handles them with auth-store checks), and providers Hermes cannot route.
    PROVIDER_REGISTRY env var names win over models.dev's (which can be wrong).
    """
    from agent.models_dev import PROVIDER_TO_MODELS_DEV
    from hermes_cli.auth import PROVIDER_REGISTRY, is_runtime_provider_routable
    from hermes_cli.models import _AGGREGATOR_PROVIDERS
    from hermes_cli.providers import ALIASES

    for hermes_id, mdev_id in PROVIDER_TO_MODELS_DEV.items():
        alias_target = ALIASES.get(hermes_id)
        if alias_target and alias_target != hermes_id and alias_target in _AGGREGATOR_PROVIDERS:
            continue
        canonical = hermes_id
        try:
            from providers import get_provider_profile
            prof = get_provider_profile(hermes_id)
            if prof is not None:
                canonical = prof.name
        except Exception:
            pass
        if canonical != hermes_id or hermes_id.lower() in seen:
            continue
        if hermes_id.lower() in excluded or mdev_id.lower() in excluded:
            continue
        pdata = models_dev_data.get(mdev_id)
        if not isinstance(pdata, dict):
            continue
        pconfig = PROVIDER_REGISTRY.get(hermes_id)
        if pconfig and pconfig.auth_type != "api_key":
            continue
        if not is_runtime_provider_routable(hermes_id):
            continue
        if pconfig and pconfig.api_key_env_vars:
            env_vars = list(pconfig.api_key_env_vars)
        else:
            env_vars = pdata.get("env", [])
            if not isinstance(env_vars, list):
                continue
        yield hermes_id, mdev_id, pconfig, env_vars


def _auth_store_has_provider(*keys: str) -> bool:
    """True when ``auth.json`` has a ``providers`` entry under any of *keys*."""
    try:
        from hermes_cli.auth import _load_auth_store
        store = _load_auth_store()
        providers_store = store.get("providers", {})
        return bool(store and any(k in providers_store for k in keys))
    except Exception as exc:
        logger.debug("Auth store check failed for %s: %s", keys[0] if keys else "", exc)
        return False


def _raw_pool_usable(hermes_id: str) -> bool:
    """Section-1 pool check: only consult the pool when auth.json lists a raw entry."""
    from hermes_cli.model_switch import _credential_pool_is_usable
    try:
        from hermes_cli.auth import _load_auth_store
        store = _load_auth_store()
        if store and store.get("credential_pool", {}).get(hermes_id):
            return _credential_pool_is_usable(hermes_id, raw_pool_present=True)
    except Exception:
        pass
    return False


def _pool_usable(slug: str) -> bool:
    from hermes_cli.model_switch import _credential_pool_is_usable
    try:
        return _credential_pool_is_usable(slug)
    except Exception as exc:
        logger.debug("Credential pool check failed for %s: %s", slug, exc)
        return False


def _overlay_has_env_creds(pid: str, hermes_slug: str, overlay, read_env) -> bool:
    """Section-2 env/SDK credential check shared by the picker and the prefetch scan.

    Vertex authenticates via OAuth2 (service-account JSON / ADC), not an API
    key, so it gets its own probe; otherwise the provider is hidden from the
    picker even when fully configured.
    """
    from hermes_cli.auth import PROVIDER_REGISTRY

    has_creds = False
    if overlay.auth_type == "vertex":
        try:
            from agent.vertex_adapter import has_vertex_credentials
            has_creds = has_vertex_credentials()
        except Exception as exc:
            logger.debug("Vertex credential check failed: %s", exc)
    elif overlay.extra_env_vars:
        has_creds = any(read_env(ev) for ev in overlay.extra_env_vars)
    if not has_creds and overlay.auth_type == "api_key":
        for key in (pid, hermes_slug):
            pcfg = PROVIDER_REGISTRY.get(key)
            if pcfg and pcfg.api_key_env_vars and any(read_env(ev) for ev in pcfg.api_key_env_vars):
                return True
    return has_creds


def _has_fast_aws_sdk_signal() -> bool:
    """True when explicit AWS auth config is present in the environment.

    Deliberately avoids botocore's full credential chain: picker discovery runs
    for non-Bedrock providers too, and botocore may probe EC2 IMDS
    (169.254.169.254) on local machines before returning no credentials.
    """
    env = os.environ
    if env.get("AWS_BEARER_TOKEN_BEDROCK", "").strip():
        return True
    if env.get("AWS_ACCESS_KEY_ID", "").strip() and env.get("AWS_SECRET_ACCESS_KEY", "").strip():
        return True
    return any(
        env.get(name, "").strip()
        for name in (
            "AWS_PROFILE",
            "AWS_CONTAINER_CREDENTIALS_RELATIVE_URI",
            "AWS_CONTAINER_CREDENTIALS_FULL_URI",
            "AWS_WEB_IDENTITY_TOKEN_FILE",
        )
    )


def _has_aws_sdk_creds_for_listing(slug: str, current_provider: str) -> bool:
    """Credential check for AWS SDK providers in non-runtime discovery.

    The full boto3 chain is only consulted for the *current* provider.
    """
    if _has_fast_aws_sdk_signal():
        return True
    if str(slug or "").strip().lower() != str(current_provider or "").strip().lower():
        return False
    try:
        from agent.bedrock_adapter import has_aws_credentials
        return bool(has_aws_credentials())
    except Exception:
        return False


def _is_aws_sdk(pconfig) -> bool:
    return bool(pconfig) and getattr(pconfig, "auth_type", "") == "aws_sdk"


def _live_or_curated_ids(slug: str, curated: dict, *fallback_keys: str, merge_models_dev: bool = True) -> list:
    """Unified pathway: ``cached_provider_model_ids`` so the /model picker sees the
    SAME list ``hermes model`` builds (disk-cached), falling back to the curated
    static list (merged with models.dev for preferred providers) when live is empty.
    """
    from hermes_cli.models import _MODELS_DEV_PREFERRED, _merge_with_models_dev, cached_provider_model_ids

    model_ids = cached_provider_model_ids(slug)
    if not model_ids:
        for key in fallback_keys or (slug,):
            model_ids = curated.get(key, [])
            if model_ids:
                break
        if merge_models_dev and slug in _MODELS_DEV_PREFERRED:
            model_ids = _merge_with_models_dev(slug, model_ids)
    return model_ids


def _aws_live_or_curated_ids(slug: str, curated: dict, *fallback_keys: str) -> list:
    """Bedrock: live discovery reflects the active region (eu.*, ap.*) rather than
    the static us.* list; any failure falls back to the curated list."""
    from hermes_cli.models import cached_provider_model_ids

    fallback_keys = fallback_keys or (slug,)
    try:
        ids = cached_provider_model_ids(slug)
        if ids:
            return ids
    except Exception:
        pass
    for key in fallback_keys:
        ids = curated.get(key, [])
        if ids:
            return ids
    return []


def _nous_picker_model_ids(curated: dict, force_fresh_nous_tier: bool) -> list:
    """Nous serves a huge alphabetical live catalog; the picker shows ONLY the
    curated agentic list, augmented with the Portal's free/paid recommendations
    (so newly launched models surface without a CLI release) and narrowed by org
    policy. Mirrors ``_model_flow_nous`` so GUI pickers match the CLI. A failed
    recommendation fetch still yields a policy-filtered curated list.
    """
    model_ids = curated.get("nous", [])
    try:
        from hermes_cli.models import (
            get_pricing_for_provider,
            check_nous_free_tier,
            union_with_portal_free_recommendations,
            union_with_portal_paid_recommendations,
        )
        from hermes_cli.auth import get_provider_auth_state

        pricing = get_pricing_for_provider("nous") or {}
        try:
            portal = (get_provider_auth_state("nous") or {}).get("portal_base_url", "") or ""
        except Exception:
            portal = ""
        if check_nous_free_tier(force_fresh=force_fresh_nous_tier):
            model_ids, _ = union_with_portal_free_recommendations(model_ids, pricing, portal)
        else:
            model_ids, _ = union_with_portal_paid_recommendations(model_ids, pricing, portal)
    except Exception:
        pass
    try:
        from hermes_cli.models import nous_policy_allowed_ids, restrict_to_nous_policy

        model_ids = restrict_to_nous_policy(model_ids, nous_policy_allowed_ids(), rescue_empty=True)
    except Exception:
        pass
    return model_ids


def _cap_models(model_ids: list, max_models: int | None, slug: str = "") -> list:
    """Apply ``max_models``; aggregators in ``_UNCAPPED_PICKER_PROVIDERS`` show everything."""
    if slug in _UNCAPPED_PICKER_PROVIDERS or max_models is None:
        return model_ids
    return model_ids[:max_models]


def _norm_url(url: Any) -> str:
    return str(url or "").strip().rstrip("/").lower()


def _entry_base_url(entry: dict, keys: tuple = ("base_url", "url", "api")) -> str:
    for key in keys:
        value = entry.get(key, "")
        if value:
            return value
    return ""


def _entry_api_mode(entry: dict) -> str | None:
    return str(entry.get("api_mode") or entry.get("transport") or "").strip().lower() or None


def _credential_identity(inline_api_key: str, key_env: str) -> str:
    return inline_api_key if inline_api_key else (f"env:{key_env}" if key_env else "")


def _discover_flag(entry: dict):
    """``discover_models`` (default True); ``"false"/"no"/"0"`` strings mean False."""
    discover = entry.get("discover_models", True)
    if isinstance(discover, str):
        discover = discover.lower() not in {"false", "no", "0"}
    return discover


def _display_prefix(name: str) -> str:
    """Text before the per-model separator Hermes's own writer uses ("—" / " - ")."""
    for sep in ("—", " - "):
        if sep in name:
            return name.split(sep)[0].strip()
    return name


def _discover_endpoint_models(
    api_key: str,
    api_url: str,
    native_catalog_provider: str,
    has_explicit_models: bool,
    *,
    headers: dict | None,
    api_mode: str | None,
    probe_live: bool,
    discovery_allowed: bool,
    for_picker: bool,
) -> tuple[list | None, bool]:
    """Return ``(models, native_catalog_empty)`` for a custom endpoint row.

    ``probe_live`` runs the native-aware picker fetch; otherwise, when discovery
    is allowed, a warm same-fingerprint cache entry still serves the full catalog
    with no round-trip. ``has_explicit_models`` gates the *probe* (a network-cost
    guard for keyless endpoints that declare a catalog), never the cache read —
    applying it to the read re-pins the endpoint to its declared subset. Returns
    ``(None, False)`` when nothing usable was found.
    """
    from hermes_cli.model_switch import _fetch_picker_live_models
    timeout = 1.5 if for_picker else 5.0
    if probe_live:
        try:
            live_models = _fetch_picker_live_models(
                api_key, api_url, native_catalog_provider, has_explicit_models,
                headers=headers, timeout=timeout, api_mode=api_mode,
            )
            is_native = isinstance(live_models, _NativePickerModelList)
            if live_models is not None and (live_models or not has_explicit_models or is_native):
                return live_models, (is_native and not live_models)
        except Exception:
            pass
    elif discovery_allowed:
        try:
            from hermes_cli.models import cached_fetch_api_models

            cached_models = cached_fetch_api_models(
                api_key, api_url, cache_only=True, timeout=timeout, headers=headers, api_mode=api_mode,
            )
            if cached_models:
                return cached_models, False
        except _MODEL_DISCOVERY_ERRORS:
            pass
    return None, False


def _collect_authed_provider_slugs(
    models_dev_data: dict,
    curated: dict[str, list[str]],
    excluded: list[str],
) -> list[str]:
    """Quick-scan which providers have credentials, without fetching model lists.

    Mirrors the credential checks of sections 1, 2 and 2b of
    :func:`list_authenticated_providers` but never calls
    ``cached_provider_model_ids``; the result feeds
    :func:`_prefetch_provider_models_parallel`. Env vars are read through the
    per-profile secret scope. AWS SDK providers are skipped (heavier detection).
    """
    from hermes_cli.model_switch import _scoped_key_env
    from agent.models_dev import PROVIDER_TO_MODELS_DEV
    from hermes_cli.auth import PROVIDER_REGISTRY
    from hermes_cli.providers import HERMES_OVERLAYS
    from hermes_cli.models import CANONICAL_PROVIDERS

    excluded_set = {str(p).strip().lower() for p in excluded if p}
    slugs: list[str] = []
    seen: set[str] = set()

    for hermes_id, _mdev_id, _pconfig, env_vars in _iter_builtin_candidates(models_dev_data, excluded_set, seen):
        if any(_scoped_key_env(ev) for ev in env_vars) or _raw_pool_usable(hermes_id):
            slugs.append(hermes_id)
            seen.add(hermes_id.lower())

    mdev_to_hermes = {v: k for k, v in PROVIDER_TO_MODELS_DEV.items()}
    for pid, overlay in HERMES_OVERLAYS.items():
        hermes_slug = mdev_to_hermes.get(pid, pid)
        if pid.lower() in seen or hermes_slug.lower() in seen:
            continue
        if pid.lower() in excluded_set or hermes_slug.lower() in excluded_set:
            continue
        if overlay.auth_type == "aws_sdk":
            continue
        if (
            _overlay_has_env_creds(pid, hermes_slug, overlay, _scoped_key_env)
            or _auth_store_has_provider(pid, hermes_slug)
            or _pool_usable(hermes_slug)
        ):
            slugs.append(hermes_slug)
            seen.add(pid.lower())
            seen.add(hermes_slug.lower())

    for cp in CANONICAL_PROVIDERS:
        if cp.slug.lower() in seen or cp.slug.lower() in excluded_set:
            continue
        cp_config = PROVIDER_REGISTRY.get(cp.slug)
        has_creds = bool(
            cp_config and cp_config.api_key_env_vars and any(_scoped_key_env(ev) for ev in cp_config.api_key_env_vars)
        )
        if has_creds or _auth_store_has_provider(cp.slug) or _pool_usable(cp.slug):
            slugs.append(cp.slug)
            seen.add(cp.slug.lower())

    # Nous excluded: its picker branch builds from the curated list and never
    # reads the api_key-only cache entry a prefetch would write.
    return [s for s in slugs if s != "nous"]


@dataclass
class _PickerBuild:
    """Mutable state threaded through the ``list_authenticated_providers`` sections:
    1 built-ins mapped to models.dev, 2 Hermes-only overlays (nous, openai-codex,
    copilot, opencode-*), 2b canonical providers missed by 1/2 (keeps /model in sync
    with `hermes model`), 3 ``providers:`` entries + 3b the bare active custom
    endpoint, 4 ``custom_providers:`` entries. Every ``hermes_cli.auth/models`` import
    in the row builders stays lazy so tests can patch those modules."""

    current_provider: str
    current_base_url: str
    current_model: str
    max_models: int | None
    for_picker: bool
    force_fresh_nous_tier: bool
    probe_custom_providers: bool
    probe_current_custom_provider: bool
    refresh: bool
    excluded: set
    curated: dict
    results: list = field(default_factory=list)
    seen_slugs: set = field(default_factory=set)  # lowercase-normalized to catch case variants
    # Effective base URLs of every built-in row, so section 4 hides
    # ``custom_providers`` entries that duplicate a built-in endpoint.
    builtin_endpoints: set = field(default_factory=set)
    # (display_name, base_url) pairs emitted by section 3 so section 4 skips
    # overlapping ``custom_providers`` rows (callers often pass both).
    section3_pairs: set = field(default_factory=set)
    current_provider_norm: str = field(init=False)
    current_base_url_norm: str = field(init=False)

    def __post_init__(self):
        self.current_provider_norm = self.current_provider.lower()
        self.current_base_url_norm = self.current_base_url.rstrip("/").lower()

    def can_probe_custom(self, *, row_is_current: bool) -> bool:
        return bool(self.probe_custom_providers or (self.probe_current_custom_provider and row_is_current))

    def record_builtin_endpoint(self, slug: str) -> None:
        """Prefer the live env override (e.g. DASHSCOPE_BASE_URL) over the static
        inference_base_url so dedup matches what a user typing that URL into
        custom_providers would actually hit."""
        try:
            from hermes_cli.auth import PROVIDER_REGISTRY
        except Exception:
            return
        pcfg = PROVIDER_REGISTRY.get(slug)
        if not pcfg:
            return
        url = os.environ.get(pcfg.base_url_env_var, "") if getattr(pcfg, "base_url_env_var", "") else ""
        normed = _norm_url(url or getattr(pcfg, "inference_base_url", "") or "")
        if normed:
            self.builtin_endpoints.add(normed)

    def add_builtin_row(self, slug: str, name: str, is_current: bool, model_ids: list, source: str, *, uncapped_ok: bool = True) -> None:
        self.results.append({
            "slug": slug,
            "name": name,
            "is_current": is_current,
            "is_user_defined": False,
            "models": _cap_models(model_ids, self.max_models, slug if uncapped_ok else ""),
            "total_models": len(model_ids),
            "source": source,
        })
        self.seen_slugs.add(slug.lower())
        self.record_builtin_endpoint(slug)


def _lap_builtin_rows(b: _PickerBuild, data: dict, user_providers: dict) -> None:
    """Section 1: models.dev-mapped providers with api_key auth."""
    from hermes_cli.model_switch import _declared_model_ids
    from agent.models_dev import get_provider_info

    for hermes_id, mdev_id, pconfig, env_vars in _iter_builtin_candidates(data, b.excluded, b.seen_slugs):
        if not (any(os.environ.get(ev) for ev in env_vars) or _raw_pool_usable(hermes_id)):
            continue
        model_ids = _live_or_curated_ids(hermes_id, b.curated)
        # A providers.<built-in>.models block extends the discovered catalog;
        # section 3 cannot emit it later because this row owns the slug.
        configured = user_providers.get(hermes_id) if isinstance(user_providers, dict) else None
        configured_models = _declared_model_ids(configured.get("models")) if isinstance(configured, dict) else []
        model_ids = list(dict.fromkeys([*configured_models, *model_ids]))
        pinfo = get_provider_info(mdev_id)
        display_name = pconfig.name if pconfig and pconfig.name else (pinfo.name if pinfo else mdev_id)
        b.add_builtin_row(
            hermes_id, display_name, b.current_provider in (hermes_id, mdev_id), model_ids, "built-in",
        )


def _lap_overlay_rows(b: _PickerBuild, data: dict) -> None:
    """Section 2: Hermes-only providers (nous, openai-codex, copilot, opencode-go, ...)."""
    from hermes_cli.model_switch import _credential_pool_is_usable
    from agent.models_dev import PROVIDER_TO_MODELS_DEV
    from hermes_cli.providers import HERMES_OVERLAYS

    # HERMES_OVERLAYS keys may be models.dev IDs ("github-copilot") while
    # config.yaml uses Hermes IDs ("copilot").
    mdev_to_hermes = {v: k for k, v in PROVIDER_TO_MODELS_DEV.items()}
    for pid, overlay in HERMES_OVERLAYS.items():
        hermes_slug = mdev_to_hermes.get(pid, pid)
        if pid.lower() in b.seen_slugs or hermes_slug.lower() in b.seen_slugs:
            continue
        if pid.lower() in b.excluded or hermes_slug.lower() in b.excluded:
            continue

        if getattr(overlay, "keyless", False):
            has_creds = True  # served anonymously (opencode-free)
        elif overlay.auth_type == "aws_sdk":
            has_creds = _has_aws_sdk_creds_for_listing(hermes_slug, b.current_provider)
        else:
            has_creds = _overlay_has_env_creds(pid, hermes_slug, overlay, os.environ.get)
        # External-process providers (copilot-acp) hold no key/token/pool entry by
        # design — the spawned ACP subprocess brings its own auth. "Configured"
        # means the executable resolves, which is what get_auth_status() reports;
        # without this the has_creds filter hides the provider from every picker.
        if not has_creds and overlay.auth_type == "external_process":
            try:
                from hermes_cli.auth import get_auth_status
                _ext_status = get_auth_status(hermes_slug) or {}
                has_creds = bool(_ext_status.get("logged_in") or _ext_status.get("configured"))
            except Exception as exc:
                logger.debug("External-process check failed for %s: %s", pid, exc)
        # Auth store / credential pool cover OAuth providers AND api_key providers
        # that also support OAuth (anthropic via Claude Code credential files).
        if not has_creds:
            has_creds = _auth_store_has_provider(pid, hermes_slug)
        if not has_creds:
            # Full auto-seeding pool check catches external stores (Codex CLI
            # ~/.codex/auth.json) not yet in auth.json.
            try:
                if _credential_pool_is_usable(hermes_slug):
                    has_creds = True
                elif b.for_picker:
                    # Show providers whose pool is entirely in cooldown: limits are
                    # per-model for many providers, so another model may work.
                    try:
                        from agent.credential_pool import load_pool
                        has_creds = load_pool(hermes_slug).has_credentials()
                    except Exception:
                        pass
            except Exception as exc:
                logger.debug("Credential pool check failed for %s: %s", hermes_slug, exc)
        if not has_creds and hermes_slug == "anthropic":
            # The pool gates anthropic behind is_provider_explicitly_configured()
            # (aux tasks must not consume Claude Code tokens); the picker is
            # discovery-oriented, so read the external credential files directly.
            try:
                from agent.anthropic_adapter import read_claude_code_credentials, read_hermes_oauth_credentials
                hermes_creds = read_hermes_oauth_credentials()
                cc_creds = read_claude_code_credentials()
                if (hermes_creds and hermes_creds.get("accessToken")) or (cc_creds and cc_creds.get("accessToken")):
                    has_creds = True
            except Exception as exc:
                logger.debug("Anthropic external creds check failed: %s", exc)
        if not has_creds:
            continue

        if hermes_slug in {"openai-codex", "copilot", "copilot-acp"}:
            # Live OAuth-backed discovery so Pro-only Codex slugs not in the static
            # catalog appear; falls back to curated when unreachable.
            from hermes_cli.models import cached_provider_model_ids
            model_ids = cached_provider_model_ids(hermes_slug)
        elif overlay.auth_type == "aws_sdk":
            model_ids = _aws_live_or_curated_ids(hermes_slug, b.curated, hermes_slug, pid)
        elif hermes_slug == "nous":
            model_ids = _nous_picker_model_ids(b.curated, b.force_fresh_nous_tier)
        else:
            model_ids = _live_or_curated_ids(hermes_slug, b.curated, hermes_slug, pid)
        b.add_builtin_row(
            hermes_slug, get_label(hermes_slug), b.current_provider in (hermes_slug, pid), model_ids, "hermes",
        )
        b.seen_slugs.add(pid.lower())


def _lap_canonical_rows(b: _PickerBuild) -> None:
    """Section 2b: CANONICAL_PROVIDERS missed by sections 1/2."""
    from hermes_cli.auth import PROVIDER_REGISTRY
    try:
        from hermes_cli.models import CANONICAL_PROVIDERS
    except ImportError:
        CANONICAL_PROVIDERS = []

    for cp in CANONICAL_PROVIDERS:
        if cp.slug.lower() in b.seen_slugs or cp.slug.lower() in b.excluded:
            continue
        cp_config = PROVIDER_REGISTRY.get(cp.slug)
        has_creds = False
        if cp_config and cp_config.api_key_env_vars:
            lit = {ev for ev in cp_config.api_key_env_vars if os.environ.get(ev)}
            has_creds = bool(lit)
            # A regional "-cn" twin lit only by key vars shared with its non-CN
            # sibling is a phantom row: hide it unless it is the current provider,
            # and only when it has a dedicated var of its own the user could set.
            sib = PROVIDER_REGISTRY.get(cp.slug[:-3]) if cp.slug.endswith("-cn") else None
            sib_vars = set(sib.api_key_env_vars) if sib else set()
            if lit and lit <= sib_vars < set(cp_config.api_key_env_vars) and cp.slug != b.current_provider:
                continue
        if not has_creds:
            has_creds = _auth_store_has_provider(cp.slug) or _pool_usable(cp.slug)
        if not has_creds and _is_aws_sdk(cp_config):
            has_creds = _has_aws_sdk_creds_for_listing(cp.slug, b.current_provider)
        if not has_creds:
            continue
        if _is_aws_sdk(cp_config):
            model_ids = _aws_live_or_curated_ids(cp.slug, b.curated)
        else:
            model_ids = _live_or_curated_ids(cp.slug, b.curated, merge_models_dev=False)
        b.add_builtin_row(
            cp.slug, cp.label, cp.slug == b.current_provider, model_ids, "canonical", uncapped_ok=False,
        )


def _lap_user_provider_rows(b: _PickerBuild, user_providers: dict) -> None:
    """Section 3: ``providers:`` dict entries, grouped by (api_url, credential,
    api_mode, extra_headers) so keyed providers on one endpoint with the same
    wire protocol collapse into one row (e.g. two Palantir Claude entries ->
    one "Palantir Claude" row); a different key_env/api_mode/headers keeps
    distinct rows since the wire protocol or tenant differs."""
    from hermes_cli.model_switch import _declared_model_ids, _entry_models_discovered, _extra_headers_from_config, _models_config_is_allowlist, _scoped_key_env
    from collections import OrderedDict
    from hermes_cli.config import coerce_provider_id, is_provider_enabled

    ep_groups: "OrderedDict[tuple, dict]" = OrderedDict()
    for ep_name, ep_cfg in user_providers.items():
        if not isinstance(ep_cfg, dict) or not is_provider_enabled(ep_cfg):
            continue
        if ep_name.lower() in b.seen_slugs:
            continue
        display_name = coerce_provider_id(ep_cfg.get("name")) or ep_name
        api_url = _entry_base_url(ep_cfg, ("base_url", "api", "url"))
        key_env = str(ep_cfg.get("key_env") or ep_cfg.get("api_key_env") or "").strip()
        inline_api_key = str(ep_cfg.get("api_key", "") or "").strip()
        api_mode = _entry_api_mode(ep_cfg)
        headers_identity = tuple(sorted(_extra_headers_from_config(ep_cfg).items()))
        group_key = (_norm_url(api_url), _credential_identity(inline_api_key, key_env), api_mode, headers_identity)

        # ``default_model`` is the legacy key; ``model`` matches custom_providers.
        default_model = ep_cfg.get("default_model", "") or ep_cfg.get("model", "")
        entry_models = [default_model] if default_model else []
        for model_id in _declared_model_ids(ep_cfg.get("models", [])):
            if model_id not in entry_models:
                entry_models.append(model_id)

        if group_key not in ep_groups:
            # Strip the per-model suffix and trailing version tokens ("Palantir
            # Claude 4.7 Opus" -> "Palantir Claude"): cut at the first token with
            # a digit, only when >=2 words remain (avoids over-trimming).
            grp_display = _display_prefix(display_name)
            toks = grp_display.split()
            cut_at = next((i for i, t in enumerate(toks) if any(c.isdigit() for c in t.strip(".,()"))), None)
            if cut_at is not None and cut_at >= 2:
                grp_display = " ".join(toks[:cut_at]).strip()
            ep_groups[group_key] = {
                "slug": ep_name,  # first ep_name encountered
                "name": grp_display or display_name,
                "api_url": api_url,
                "models": [],
                "has_explicit_models": False,
                "ep_cfg": ep_cfg,
                "raw_names": [],
                "aliases": set(),
            }
        grp = ep_groups[group_key]
        for m in entry_models:
            if m and m not in grp["models"]:
                grp["models"].append(m)
        # A singular default_model/model is only the active selection and must
        # not suppress discovery; dict-shaped ``models:`` is context_length
        # metadata, not an allowlist — see ``_models_config_is_allowlist``.
        if _models_config_is_allowlist(ep_cfg.get("models"), _entry_models_discovered(ep_cfg)):
            grp["has_explicit_models"] = True
        grp["raw_names"].append(display_name)
        grp["aliases"].update(custom_provider_aliases(display_name, str(ep_name)))

    for grp in ep_groups.values():
        ep_cfg, ep_name, display_name, api_url = grp["ep_cfg"], grp["slug"], grp["name"], grp["api_url"]
        models_list = list(grp["models"])
        # Official OpenAI rows often have base_url but no models: dict — avoid a
        # misleading zero count.
        if not models_list and base_url_host_matches(str(api_url).strip().lower(), "api.openai.com"):
            models_list = list(b.curated.get("openai") or [])

        # Probe policy (mirrors section 4): with an api_key always probe; without
        # one, skip only when an allowlist-shaped ``models:`` narrows the endpoint.
        api_key = str(ep_cfg.get("api_key", "") or "").strip()
        if not api_key:
            key_env = str(ep_cfg.get("key_env") or ep_cfg.get("api_key_env") or "").strip()
            api_key = _scoped_key_env(key_env) if key_env else ""
        has_explicit_models = bool(grp.get("has_explicit_models"))
        ep_url_norm = _norm_url(api_url)
        ep_aliases = {str(alias).lower() for alias in grp.get("aliases", set())}
        is_current = (
            str(ep_name).strip().lower() == b.current_provider_norm
            or b.current_provider_norm in ep_aliases
            or (
                b.current_provider_norm == "custom"
                and bool(b.current_base_url_norm)
                and ep_url_norm == b.current_base_url_norm
            )
        )
        discovery_allowed = bool(api_url) and _discover_flag(ep_cfg)
        discovered, native_catalog_empty = _discover_endpoint_models(
            api_key,
            api_url,
            ep_name if str(ep_name).strip().lower() in {"ollama", "custom:ollama"} else "custom",
            has_explicit_models,
            headers=_extra_headers_from_config(ep_cfg) or None,
            api_mode=ep_cfg.get("api_mode"),
            probe_live=(
                discovery_allowed
                and (bool(api_key) or not has_explicit_models)
                and b.can_probe_custom(row_is_current=is_current)
            ),
            discovery_allowed=discovery_allowed,
            for_picker=b.for_picker,
        )
        if discovered is not None:
            models_list = discovered

        b.results.append({
            "slug": ep_name,
            "name": display_name,
            "is_current": is_current,
            "is_user_defined": True,
            "models": models_list,
            "total_models": len(models_list) if models_list else 0,
            "source": "user-config",
            "api_url": api_url,
            "native_catalog_empty": native_catalog_empty,
        })
        b.seen_slugs.add(ep_name.lower())
        b.seen_slugs.update(ep_aliases)
        # Record every raw member name so section 4 can match per-model
        # custom_providers rows even though the group label was collapsed.
        for raw_name in grp.get("raw_names") or [display_name]:
            pair = (str(raw_name).strip().lower(), ep_url_norm)
            if pair[0] and pair[1]:
                b.section3_pairs.add(pair)
                b.seen_slugs.add(custom_provider_slug(raw_name).lower())
        pair = (str(display_name).strip().lower(), ep_url_norm)
        if pair[0] and pair[1]:
            b.section3_pairs.add(pair)


def _lap_bare_custom_row(b: _PickerBuild, custom_providers: list | None) -> None:
    """Section 3b: ``model.provider: custom`` + ``model.base_url`` with no named
    providers:/custom_providers row — surface it so /model does not look like it
    ignored config.yaml."""
    if not (b.current_provider_norm == "custom" and b.current_base_url and "custom" not in b.seen_slugs):
        return
    if any(
        isinstance(cp, dict) and _norm_url(_entry_base_url(cp)) == _norm_url(b.current_base_url)
        for cp in (custom_providers or [])
    ):
        return
    api_url = str(b.current_base_url).strip().rstrip("/")
    models = [b.current_model] if b.current_model else []
    native_catalog_empty = False
    try:
        discovered, native_catalog_empty = _discover_endpoint_models(
            "", api_url, "custom", False,
            headers=None, api_mode=None,
            probe_live=bool(b.refresh or b.probe_current_custom_provider),
            discovery_allowed=True,
            for_picker=b.for_picker,
        )
        if discovered is not None:
            models = discovered
    except Exception:
        pass
    b.results.append({
        "slug": "custom",
        "name": "Custom endpoint",
        "is_current": True,
        "is_user_defined": True,
        "models": _cap_models(models, b.max_models),
        "total_models": len(models),
        "source": "model-config",
        "api_url": api_url,
        "native_catalog_empty": native_catalog_empty,
    })
    b.seen_slugs.add("custom")


def _lap_custom_provider_rows(b: _PickerBuild, custom_providers: list) -> None:
    """Section 4: ``custom_providers:`` entries (one model each) grouped into one
    row per (endpoint, credential identity, api_mode, extra_headers, display
    prefix). Four "Ollama — X" entries on one host become one "Ollama" row;
    distinct prefixes sharing a proxy URL keep their own rows."""
    from hermes_cli.model_switch import _declared_model_ids, _entry_models_discovered, _extra_headers_from_config, _models_config_is_allowlist, _save_discovered_models_to_config, _scoped_key_env
    from collections import OrderedDict
    from hermes_cli.config import coerce_provider_id

    groups: "OrderedDict[tuple, dict]" = OrderedDict()
    for entry in custom_providers:
        if not isinstance(entry, dict):
            continue
        raw_name = coerce_provider_id(entry.get("name"))
        api_url = str(_entry_base_url(entry) or "").strip().rstrip("/")
        if not raw_name or not api_url:
            continue
        inline_api_key = str(entry.get("api_key") or "").strip()
        key_env = str(entry.get("key_env") or "").strip()
        api_key = inline_api_key or _scoped_key_env(key_env)
        api_mode = _entry_api_mode(entry)
        discover = _discover_flag(entry)
        entry_extra_headers = _extra_headers_from_config(entry)
        prefix = _display_prefix(raw_name)
        group_key = (
            api_url, _credential_identity(inline_api_key, key_env), api_mode,
            tuple(sorted(entry_extra_headers.items())), prefix.lower(),
        )
        if group_key not in groups:
            display_name = prefix or raw_name
            groups[group_key] = {
                "slug": custom_provider_slug(display_name, str(entry.get("provider_key") or "").strip()),
                "name": display_name,
                "api_url": api_url,
                "api_key": api_key,
                "models": [],
                "has_explicit_models": False,
                "discover_models": discover,
                "api_mode": api_mode,
                "extra_headers": entry_extra_headers,
                "aliases": set(),
            }
        else:
            if api_key and not groups[group_key].get("api_key"):
                groups[group_key]["api_key"] = api_key
            if not discover:  # one opt-out pins the whole grouped row
                groups[group_key]["discover_models"] = False
        grp = groups[group_key]
        grp["aliases"].update(custom_provider_aliases(raw_name, str(entry.get("provider_key") or "")))
        # ``model:`` is only the active selection; every configured model lives
        # under ``models:`` (dict written by _save_custom_provider).
        default_model = (entry.get("model") or "").strip()
        if default_model and default_model not in grp["models"]:
            grp["models"].append(default_model)
        models_field = entry.get("models", {})
        if _models_config_is_allowlist(models_field, _entry_models_discovered(entry)):
            grp["has_explicit_models"] = True
        for model_id in _declared_model_ids(models_field):
            if model_id not in grp["models"]:
                grp["models"].append(model_id)

    section4_slugs: set = set()
    current_url_group_count = sum(
        1 for grp in groups.values()
        if b.current_base_url_norm and _norm_url(grp["api_url"]) == b.current_base_url_norm
    )
    for grp in groups.values():
        api_url, api_key, slug = grp["api_url"], grp.get("api_key", ""), grp["slug"]
        # Slug claimed by a built-in/overlay/providers: row -> skip (don't shadow).
        if slug.lower() in b.seen_slugs and slug.lower() not in section4_slugs:
            continue
        # Two custom endpoints with the same cleaned name: suffix a counter so
        # both stay visible.
        if slug.lower() in section4_slugs:
            base_slug, n = slug, 2
            while f"{base_slug}-{n}".lower() in b.seen_slugs:
                n += 1
            slug = f"{base_slug}-{n}"
            grp["slug"] = slug
        grp_url_norm = _norm_url(api_url)
        pair_key = (str(grp["name"]).strip().lower(), grp_url_norm)
        if pair_key[0] and pair_key[1] and pair_key in b.section3_pairs:
            continue
        # A built-in row already represents this endpoint (e.g. "my-dashscope"
        # vs the alibaba-coding-plan row): keep the built-in, hide the shadow.
        if grp_url_norm and grp_url_norm in b.builtin_endpoints:
            continue
        is_current = (
            slug.lower() == b.current_provider_norm
            or b.current_provider_norm in {str(alias).lower() for alias in grp.get("aliases", set())}
        ) or (
            b.current_provider_norm == "custom"
            and bool(b.current_base_url_norm)
            and grp_url_norm == b.current_base_url_norm
            and current_url_group_count == 1
        )
        # Probe policy: with an api_key live /models is the source of truth (replace
        # the partial ``models:`` subset); without one, an allowlist-shaped
        # ``models:`` narrows a public endpoint and skips the probe. A dict-shaped
        # ``models:`` is metadata, so still probe; pin with discover_models: false.
        has_explicit_models = bool(grp.get("has_explicit_models"))
        discovery_allowed = bool(api_url) and grp.get("discover_models", True)
        probe_live = (
            discovery_allowed
            and (bool(api_key) or not has_explicit_models)
            and b.can_probe_custom(row_is_current=is_current)
        )
        discovered, native_catalog_empty = _discover_endpoint_models(
            api_key,
            api_url,
            "ollama" if "ollama" in {str(slug).strip().lower(), str(grp.get("name") or "").strip().lower()} else "custom",
            has_explicit_models,
            headers=grp.get("extra_headers") or None,
            api_mode=grp.get("api_mode"),
            probe_live=probe_live,
            discovery_allowed=discovery_allowed,
            for_picker=b.for_picker,
        )
        if discovered is not None:
            grp["models"] = discovered
            if probe_live:
                # A successful live probe persists the catalog for no-probe surfaces.
                try:
                    _save_discovered_models_to_config(
                        api_url, discovered, api_mode=grp.get("api_mode"), headers=grp.get("extra_headers") or None,
                    )
                except Exception:
                    pass
        b.results.append({
            "slug": slug,
            "name": grp["name"],
            "is_current": is_current,
            "is_user_defined": True,
            "models": grp["models"],
            "total_models": len(grp["models"]),
            "source": "user-config",
            "api_url": grp["api_url"],
            "native_catalog_empty": native_catalog_empty,
        })
        b.seen_slugs.add(slug.lower())
        section4_slugs.add(slug.lower())


def _build_curated_lists(current_provider: str, current_base_url: str, current_model: str) -> dict[str, list[str]]:
    """Curated model lists keyed by hermes provider id, plus the dynamic ones
    (nous manifest, Ollama Cloud, LM Studio live probe)."""
    from hermes_cli.models import OPENROUTER_MODELS, _PROVIDER_MODELS, get_curated_nous_model_ids

    curated: dict[str, list[str]] = dict(_PROVIDER_MODELS)
    curated["openrouter"] = [mid for mid, _ in OPENROUTER_MODELS]
    # Remote model-catalog manifest so new Portal models surface without a
    # release; falls back to the in-repo snapshot when unreachable.
    curated["nous"] = get_curated_nous_model_ids()
    if "ollama-cloud" not in curated:
        from hermes_cli.models import fetch_ollama_cloud_models
        curated["ollama-cloud"] = fetch_ollama_cloud_models()
    # LM Studio has no static catalog: probe its native endpoint live. Base URL
    # precedence: LM_BASE_URL > active config base_url (when current) > default.
    # On auth rejection / unreachable, fall back to the current model so the
    # picker still shows something offline.
    is_current_lmstudio = current_provider.strip().lower() == "lmstudio"
    if "lmstudio" not in curated and (os.environ.get("LM_API_KEY") or os.environ.get("LM_BASE_URL") or is_current_lmstudio):
        from hermes_cli.models import fetch_lmstudio_models
        from hermes_cli.auth import AuthError
        lm_base = (
            os.environ.get("LM_BASE_URL")
            or (current_base_url if is_current_lmstudio and current_base_url else None)
            or "http://127.0.0.1:1234/v1"
        )
        try:
            live = fetch_lmstudio_models(api_key=os.environ.get("LM_API_KEY", ""), base_url=lm_base, timeout=1.5)
        except AuthError:
            live = []
        if not live and is_current_lmstudio and current_model:
            live = [current_model]
        curated["lmstudio"] = live
    return curated


def list_authenticated_providers(
    current_provider: str = "",
    current_base_url: str = "",
    user_providers: dict = None,
    custom_providers: list | None = None,
    *,
    force_fresh_nous_tier: bool = False,
    max_models: int | None = None,
    current_model: str = "",
    refresh: bool = False,
    probe_custom_providers: bool = True,
    probe_current_custom_provider: bool = False,
    for_picker: bool = False,
    excluded_providers: list | None = None,
) -> List[dict]:
    """Detect which providers have credentials and list their curated models.

    Uses the curated lists from hermes_cli/models.py (OPENROUTER_MODELS,
    _PROVIDER_MODELS) — hand-picked agentic models, NOT the full models.dev
    catalog. Only providers with API keys set or user-defined endpoints appear.

    Returns a list of dicts: ``slug`` (the --provider value), ``name``,
    ``is_current``, ``is_user_defined``, ``models`` (up to max_models),
    ``total_models``, ``source`` ("built-in", "hermes", "canonical",
    "user-config", "model-config").

    ``force_fresh_nous_tier`` bypasses the short Nous tier cache for explicit
    account-sensitive flows; picker opens should leave it false.
    ``refresh`` busts the per-provider model-id disk cache up front so every row
    re-fetches live — for an explicit user "refresh models" action only.
    ``probe_custom_providers`` controls live ``/models`` discovery for saved
    custom endpoints (default true for CLI parity; GUI opens pass false).
    ``probe_current_custom_provider`` probes only the currently-selected custom
    endpoint so its list matches without blocking on offline ones.
    """
    from hermes_cli.model_switch import _collect_authed_provider_slugs, _prefetch_provider_models_parallel
    from agent.models_dev import fetch_models_dev
    from hermes_cli.config import coerce_provider_id, stringify_provider_map

    # Explicit refresh: drop every cached model-id list so the calls below all
    # re-fetch live. A stale cache can fall back to the curated static list when
    # its live fetch fails, silently dropping live-only models the user had seen.
    if refresh:
        try:
            from hermes_cli.models import clear_provider_models_cache
            clear_provider_models_cache()
        except Exception:
            pass

    # PyYAML parses unquoted numeric names (`provider: 2070`) as int.
    current_provider = coerce_provider_id(current_provider)
    current_base_url = str(current_base_url or "").strip()
    current_model = str(current_model or "").strip()
    user_providers = stringify_provider_map(user_providers)
    data = fetch_models_dev()

    b = _PickerBuild(
        current_provider=current_provider,
        current_base_url=current_base_url,
        current_model=current_model,
        max_models=max_models,
        for_picker=for_picker,
        force_fresh_nous_tier=force_fresh_nous_tier,
        probe_custom_providers=probe_custom_providers,
        probe_current_custom_provider=probe_current_custom_provider,
        refresh=refresh,
        # A single entry like ``copilot`` hides the provider under every key it
        # surfaces as (hermes_id / mdev_id / canonical slug).
        excluded={str(p).strip().lower() for p in (excluded_providers or []) if p},
        curated=_build_curated_lists(current_provider, current_base_url, current_model),
    )

    # Warm the disk cache in parallel before the serial section loops, which
    # otherwise stack 15-30s of live /v1/models round-trips on a cold cache.
    # Skipped when refresh=True (serial path force-refreshes) and for <=3
    # providers (serial is fast enough; avoids thread-pool overhead).
    prefetch_slugs = [] if refresh else _collect_authed_provider_slugs(data, b.curated, excluded_providers or [])
    if len(prefetch_slugs) > 3:
        try:
            _prefetch_provider_models_parallel(prefetch_slugs)
        except Exception:
            pass  # best-effort; serial path still works

    _lap_builtin_rows(b, data, user_providers)
    _lap_overlay_rows(b, data)
    _lap_canonical_rows(b)
    if user_providers and isinstance(user_providers, dict):
        _lap_user_provider_rows(b, user_providers)
    _lap_bare_custom_row(b, custom_providers)
    if custom_providers and isinstance(custom_providers, list):
        _lap_custom_provider_rows(b, custom_providers)
    results = b.results

    # ``providers.<name>.enabled: false`` post-filter covers built-in rows
    # (sections 1-2) that bypass the per-section gate; matched by slug and
    # ``provider_id``.
    try:
        from hermes_cli.config import is_provider_enabled
        if isinstance(user_providers, dict):
            disabled = {
                str(name).strip().lower()
                for name, cfg in user_providers.items()
                if isinstance(cfg, dict) and not is_provider_enabled(cfg)
            }
            if disabled:
                results = [
                    r for r in results
                    if str(r.get("provider_id", "")).strip().lower() not in disabled
                    and str(r.get("slug", "")).strip().lower() not in disabled
                ]
    except Exception:
        pass

    # A custom/uncurated model set via `/model <provider>/<name>` would be
    # invisible in every picker (main and MoA slot pickers read these rows);
    # inject it at the front of the current provider's row as a uniform post-pass.
    if current_model:
        for row in results:
            if not row.get("is_current") or row.get("native_catalog_empty"):
                continue
            models = row.get("models") or []
            if current_model not in models:
                row["models"] = [current_model, *models]
                row["total_models"] = row.get("total_models", len(models)) + 1
            break

    # Current provider first, then by model count descending
    results.sort(key=lambda r: (not r["is_current"], -r["total_models"]))
    return results


def _prepend_moa_picker_provider(providers: List[dict], current_provider: str = "") -> List[dict]:
    """Add the virtual MoA provider row used by interactive model pickers.

    ``list_authenticated_providers()`` only returns real/auth-backed providers.
    The CLI model inventory adds MoA separately so named presets appear next to
    normal providers; gateway pickers call ``list_picker_providers()`` directly,
    so they need the same virtual row here. Reuse the inventory's single row
    builder so the row shape stays defined in one place.
    """
    try:
        from hermes_cli.inventory import _moa_provider_row

        moa_row = _moa_provider_row(current_provider)
        if moa_row is None:
            return providers
        return [moa_row] + [p for p in providers if str(p.get("slug", "")).lower() != "moa"]
    except Exception:
        return providers


def list_picker_providers(
    current_provider: str = "",
    current_base_url: str = "",
    user_providers: dict = None,
    custom_providers: list | None = None,
    max_models: int | None = None,
    current_model: str = "",
    include_moa: bool = False,
    excluded_providers: list | None = None,
) -> List[dict]:
    """Interactive-picker variant of :func:`list_authenticated_providers`.

    Post-processes the base list so the ``/model`` picker (Telegram/Discord
    inline keyboards) only surfaces models that are actually callable in the
    current install:

    - OpenRouter's model list is replaced with the output of
      :func:`hermes_cli.models.fetch_openrouter_models`, which filters the
      curated ``OPENROUTER_MODELS`` snapshot against the live OpenRouter
      catalog.  IDs the live catalog no longer carries drop out, so the
      picker never offers a model the user can't call.
    - Provider rows whose model list ends up empty are dropped, except
      custom endpoints (``is_user_defined=True`` with an ``api_url``) where
      the user may supply their own model set through config.

    All other providers and metadata fields are passed through unchanged.
    The typed ``/model <name>`` path is unaffected -- only the interactive
    picker payload is narrowed.
    """
    from hermes_cli.model_switch import list_authenticated_providers
    from hermes_cli.models import fetch_openrouter_models

    providers = list_authenticated_providers(
        current_provider=current_provider,
        current_base_url=current_base_url,
        user_providers=user_providers,
        custom_providers=custom_providers,
        max_models=max_models,
        current_model=current_model,
        for_picker=True,
        excluded_providers=excluded_providers,
    )
    if include_moa:
        providers = _prepend_moa_picker_provider(providers, current_provider=current_provider)

    filtered: List[dict] = []
    for p in providers:
        slug = str(p.get("slug", "")).lower()
        if slug == "openrouter":
            try:
                live = fetch_openrouter_models()
                live_ids = [mid for mid, _ in live]
            except Exception:
                live_ids = list(p.get("models", []))
            p = dict(p)
            p["models"] = live_ids[:max_models] if max_models is not None else live_ids
            p["total_models"] = len(live_ids)

        has_models = bool(p.get("models"))
        is_custom_endpoint = bool(p.get("is_user_defined")) and bool(p.get("api_url"))
        if not has_models and not is_custom_endpoint:
            continue
        filtered.append(p)

    return filtered
