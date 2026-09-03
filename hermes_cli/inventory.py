"""Provider/model inventory context — shared substrate for the dashboard ``/api/model/options``,
the TUI ``model.options``/``model.save_key`` JSON-RPC handlers, and the interactive picker.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Optional


@dataclass(frozen=True)
class ConfigContext:
    """Snapshot of the model + provider config every inventory caller needs.

    Built once via ``load_picker_context()``; the TUI overlays live agent state via
    ``with_overrides()`` before passing through.
    """

    current_provider: str
    current_model: str
    current_base_url: str
    user_providers: dict
    custom_providers: list
    excluded_providers: list = None

    def with_overrides(
        self, *, current_provider: Optional[str] = None, current_model: Optional[str] = None,
        current_base_url: Optional[str] = None,
    ) -> "ConfigContext":
        """Return a copy with truthy overrides applied.

        Truthy-only: the TUI reads agent attributes that may be empty strings before an agent is
        spawned — empties must NOT clobber the disk-config values.
        """
        overrides = (("current_provider", current_provider), ("current_model", current_model),
                     ("current_base_url", current_base_url))
        kw = {k: v for k, v in overrides if v}
        return replace(self, **kw) if kw else self


def load_picker_context() -> ConfigContext:
    """Load the disk-config snapshot every consumer needs."""
    from hermes_cli.config import (
        coerce_provider_id, get_compatible_custom_providers, load_config, stringify_provider_map,
    )
    cfg = load_config()
    model_cfg = cfg.get("model", {})
    if isinstance(model_cfg, dict):
        # PyYAML parses unquoted scalars as int (`provider: 2070`); keep strings so
        # picker/options paths never call `.strip()` on an int.
        current_model = str(model_cfg.get("default", model_cfg.get("name", "")) or "")
        current_provider = coerce_provider_id(model_cfg.get("provider", ""))
        current_base_url = str(model_cfg.get("base_url", "") or "")
    else:
        # config.model can be a bare string in older configs.
        current_model = str(model_cfg) if model_cfg else ""
        current_provider = ""
        current_base_url = ""
    excluded = cfg.get("model_catalog", {}).get("excluded_providers") or []
    return ConfigContext(
        current_provider=current_provider, current_model=current_model, current_base_url=current_base_url,
        user_providers=stringify_provider_map(cfg.get("providers")),
        custom_providers=get_compatible_custom_providers(cfg),
        excluded_providers=excluded if isinstance(excluded, list) else [],
    )


def _slug(row: dict) -> str:
    return str(row.get("slug") or "").strip().lower()


def _without_slug(rows: list[dict], slug: str) -> list[dict]:
    return [r for r in rows if _slug(r) != slug]


# ─── Public: payload builder ────────────────────────────────────────────


def build_models_payload(
    ctx: ConfigContext, *, explicit_only: bool = False, include_unconfigured: bool = False,
    picker_hints: bool = False, canonical_order: bool = False, pricing: bool = False,
    capabilities: bool = False, featured: bool = False, force_fresh_nous_tier: bool = False,
    refresh: bool = False, probe_custom_providers: bool = True, probe_current_custom_provider: bool = False,
    for_picker: bool = False, max_models: int | None = None,
) -> dict:
    """Build the ``{providers, model, provider}`` shape every consumer needs.

    ``explicit_only`` keeps only providers the user explicitly configured (current provider,
    config providers, provider-specific env vars) — hides ambient/auto-seeded credentials from
    desktop chat pickers.
    """
    from hermes_cli.model_switch import list_authenticated_providers

    rows = list_authenticated_providers(
        current_provider=ctx.current_provider, current_base_url=ctx.current_base_url,
        current_model=ctx.current_model, user_providers=ctx.user_providers,
        custom_providers=ctx.custom_providers, force_fresh_nous_tier=force_fresh_nous_tier,
        max_models=max_models, refresh=refresh, probe_custom_providers=probe_custom_providers,
        probe_current_custom_provider=probe_current_custom_provider, for_picker=for_picker,
        excluded_providers=ctx.excluded_providers or [],
    )

    # Managed local runtime: staged GGUFs are selectable like any provider's models.
    # list_authenticated_providers can't know about them (no credential, no custom_providers
    # entry — the credential is reachability), so inject the row here where every picker
    # surface inherits it. Picking one routes through the llamacpp alias -> managed server.
    local_row = _local_runtime_row(ctx)
    if local_row is not None:
        rows = _without_slug(rows, "llamacpp") + [local_row]
        # A live session on the managed server reports provider "custom" (the resolution seam's
        # generic label for a raw base_url), which would materialize a duplicate "Custom endpoint"
        # row carrying the same staged models and stealing the checkmark. The Local row owns the
        # managed server's identity — drop custom rows that point at the managed endpoint.
        if local_row.get("is_current"):
            staged = set(local_row["models"])

            def _is_managed_custom(row: dict) -> bool:
                models = {str(m) for m in (row.get("models") or [])}
                return _slug(row) == "custom" and bool(models) and models <= staged

            rows = [r for r in rows if not _is_managed_custom(r)]

    moa_row = _moa_provider_row(ctx.current_provider)
    if moa_row is not None:
        rows = [moa_row] + _without_slug(rows, "moa")

    if explicit_only:
        rows = _filter_explicit_provider_rows(rows, ctx)
        # Desktop chat pickers request the explicit subset without the full unconfigured
        # universe. If the current provider lost its credential, list_authenticated_providers()
        # omits it; keep that one row so the UI shows the saved selection and a re-auth
        # affordance instead of appearing to jump to another provider. Exception: a "custom"
        # current whose endpoint is the managed local server is already represented (with the
        # checkmark) by the Local row — the skeleton would resurrect the duplicate removed above.
        _local_owns_current = bool(local_row and local_row.get("is_current")
                                   and (ctx.current_provider or "").lower() == "custom")
        if not _local_owns_current:
            rows = list(rows) + _append_unconfigured_rows(rows, ctx, current_only=True)

    # Dedup aggregator models against user-defined providers: a local proxy serving a model that
    # also appears in an aggregator's catalog would show under both, and selecting the aggregator
    # row sets model.provider to the aggregator — silently breaking the call. Aggregator rows
    # only show models the user can't get from a more-specific provider.
    _strip_aggregator_overlaps(rows)

    if include_unconfigured:
        rows = list(rows) + _without_slug(_append_unconfigured_rows(rows, ctx), "moa")
    if picker_hints:
        _apply_picker_hints(rows)
    if canonical_order:
        rows = _reorder_canonical(rows)
    if pricing:
        _apply_pricing(rows, force_fresh_nous_tier=force_fresh_nous_tier)
    if capabilities:
        _apply_capabilities(rows)
    if featured:
        _apply_featured(rows)
    _apply_custom_aliases(rows)

    return {"providers": rows, "model": ctx.current_model, "provider": ctx.current_provider}


def _strip_aggregator_overlaps(rows: list[dict]) -> None:
    """Drop models from TRUE routing aggregators (OpenRouter, custom:* proxies) that a
    user-defined provider also serves, so the picker never lists them under both.

    A user's own configured provider is never an "aggregator duplicate" of itself: user_models is
    built from these very rows and is_routing_aggregator() is True for every custom:* slug, so
    without the is_user_defined guard the dedup would empty a user-defined custom row.
    Flat-namespace resellers (opencode-go / opencode-zen) serve every model first-party and keep
    models a user's proxy happens to share a name with.
    """
    try:
        from hermes_cli.providers import is_routing_aggregator
    except Exception:
        return

    user_models: set[str] = set()
    for row in rows:
        if row.get("is_user_defined"):
            user_models.update(m.lower() for m in (row.get("models") or []))
    if not user_models:
        return
    for row in rows:
        if row.get("is_user_defined") or not is_routing_aggregator(row.get("slug", "")):
            continue
        original = row.get("models") or []
        filtered = [m for m in original if m.lower() not in user_models]
        if len(filtered) < len(original):
            row["models"] = filtered
            row["total_models"] = len(filtered)


def build_model_options_payload(
    ctx: ConfigContext, *, explicit_only: bool = False, include_unconfigured: bool = False,
    refresh: bool = False,
) -> dict:
    """Shared API-server/dashboard/TUI model-options payload with the safe probe policy.

    Normal open: probe only the current custom provider so offline saved endpoints don't block
    the picker. Explicit refresh: probe every custom provider and bust the model cache.
    """
    refresh = bool(refresh)
    return build_models_payload(
        ctx, explicit_only=bool(explicit_only), include_unconfigured=bool(include_unconfigured),
        picker_hints=True, canonical_order=True, pricing=True, capabilities=True, featured=True,
        refresh=refresh, probe_custom_providers=refresh, probe_current_custom_provider=not refresh,
    )


# ─── Public: auxiliary-task pickers ─────────────────────────────────────


def build_aux_picker_rows(
    *, current_provider: str = "", current_model: str = "", current_base_url: str = "",
    max_models: int | None = None,
) -> list[dict]:
    """Provider rows for any auxiliary-task picker (vision, compression, …).

    Honours ``model_catalog.excluded_providers`` like ``/model``; exhausted-credential-pool
    providers stay visible (``for_picker``); only the active custom endpoint is probed so the
    picker never blocks on a dead local server. The virtual ``moa`` row is excluded: auxiliary
    tasks must not run the MoA fan-out, and ``auxiliary_client`` unwraps ``moa`` to its aggregator
    anyway (``_resolve_auto``), so offering it would be a choice silently rewritten.
    """
    ctx = load_picker_context().with_overrides(
        current_provider=current_provider, current_model=current_model, current_base_url=current_base_url,
    )
    rows = build_models_payload(
        ctx, for_picker=True, probe_custom_providers=False, probe_current_custom_provider=True,
        max_models=max_models,
    )["providers"]
    return _without_slug(rows, "moa")


def format_aux_picker_entries(
    rows: list[dict], *, current_provider: str = "", current_base_url: str = "",
) -> list[tuple[str, str, list[str]]]:
    """Render aux-picker rows as ``(slug, label, models)`` menu entries.

    A custom endpoint set via a raw ``base_url`` is "current" only through that URL, never a
    slug — so when ``current_base_url`` is set no provider row is marked.
    """
    entries: list[tuple[str, str, list[str]]] = []
    current_slug = str(current_provider or "").strip().lower()
    has_base_url = bool(str(current_base_url or "").strip())
    for row in rows:
        slug = str(row.get("slug") or "")
        name = row.get("name") or slug
        total = row.get("total_models") or len(row.get("models") or [])
        model_hint = f" — {total} models" if total else ""
        marker = "  ← current" if slug.lower() == current_slug and current_slug and not has_base_url else ""
        entries.append((slug, f"{name}{model_hint}{marker}", list(row.get("models") or [])))
    return entries


def _reasoning_catalog_reader(slug: str):
    """Per-model reasoning-capability reader for aggregators that publish one.

    Cache-only — building the picker must never block on HTTP. A cold cache warms in the
    background; until then the model reports no restriction and the UI offers the full scale.
    """
    try:
        from hermes_cli.models import (
            nous_model_reasoning_capabilities, openrouter_model_reasoning_capabilities,
            warm_nous_reasoning_caps_async, warm_openrouter_reasoning_caps_async,
        )
    except Exception:
        return None

    readers = {
        "nous": (warm_nous_reasoning_caps_async, nous_model_reasoning_capabilities),
        "openrouter": (warm_openrouter_reasoning_caps_async, openrouter_model_reasoning_capabilities),
    }
    if slug not in readers:
        return None
    warm, read = readers[slug]
    warm()
    return read


def _apply_capabilities(rows: list[dict]) -> None:
    """Attach a ``{model: {fast, reasoning, ...}}`` map to each provider row.

    ``fast`` mirrors the runtime gate. ``reasoning`` defaults True when the catalog is silent: the
    dial is a no-op on models that ignore it, and hiding it from a capable model is worse. A serving
    aggregator's per-model detail overrides models.dev (adds ``can_disable_reasoning``).
    ``supported_efforts`` is deliberately NOT forwarded — it under-reports levels that work.
    """
    from hermes_cli.models import model_supports_fast_mode

    try:
        from agent.models_dev import get_model_capabilities
    except Exception:
        get_model_capabilities = None  # type: ignore[assignment]

    for row in rows:
        slug = row.get("slug") or ""
        caps: dict[str, dict[str, Any]] = {}
        read_reasoning_catalog = _reasoning_catalog_reader(slug.lower())

        for model in row.get("models") or []:
            reasoning = True
            if get_model_capabilities is not None and slug:
                try:
                    meta = get_model_capabilities(slug, model)
                    if meta is not None:
                        reasoning = bool(meta.supports_reasoning)
                except Exception:
                    reasoning = True

            entry: dict[str, Any] = {"fast": bool(model_supports_fast_mode(model)), "reasoning": reasoning}

            if reasoning and read_reasoning_catalog is not None:
                try:
                    detail = read_reasoning_catalog(model)
                except Exception:
                    detail = None
                if detail and not detail.get("supports_reasoning"):
                    # For a route it serves, the aggregator's catalog beats models.dev: no reasoning
                    # parameter means no reasoning controls, so no disable to describe either.
                    entry["reasoning"] = False
                elif detail:
                    entry["can_disable_reasoning"] = not detail.get("mandatory")

            caps[model] = entry

        row["capabilities"] = caps


# Models per lab the picker features by default. Aggregator rows keep the newest N of each lab
# (by models.dev release_date) and hide the older tail behind search / show-all. 5 keeps a lab's
# current headliners without letting a prolific vendor flood the default view.
_FEATURED_PER_LAB = 5


def _apply_featured(rows: list[dict]) -> None:
    """Attach a ``featured_models`` shortlist to each aggregator provider row.

    Aggregators serve many labs, so a flat top-N would drop whole labs; instead surface the newest
    ``_FEATURED_PER_LAB`` per vendor, ranked by models.dev ``release_date`` within the row's own
    models (never vs. today, so the choice is stable). Ties fall back to the curated flagship-first
    order. Non-aggregators get an empty list and keep top-N behaviour.
    """
    try:
        from agent.models_dev import get_model_info
    except Exception:
        get_model_info = None  # type: ignore[assignment]

    for row in rows:
        slug = str(row.get("slug") or "").strip().lower()
        models = row.get("models") or []

        # Group models by lab; only multi-lab aggregators get a shortlist.
        by_lab: dict[str, list[tuple[int, str, str]]] = {}
        for pos, model in enumerate(models):
            lab = model.split("/", 1)[0] if "/" in model else ""
            if not lab:
                # No vendor prefix → single-namespace provider, not an aggregator. Bail on the row.
                by_lab = {}
                break
            date = ""
            if get_model_info is not None:
                info = get_model_info(slug, model) or get_model_info("openrouter", model)
                date = getattr(info, "release_date", "") if info else ""
            by_lab.setdefault(lab, []).append((pos, date, model))

        if len(by_lab) < 2:
            row["featured_models"] = []
            continue

        featured: list[str] = []
        for entries in by_lab.values():
            # Newest release_date first; earlier list position breaks ties and is the sole key
            # when a lab has no dated models (all "").
            ranked = sorted(entries, key=lambda e: (e[1], -e[0]), reverse=True)
            featured.extend(model for _pos, _date, model in ranked[:_FEATURED_PER_LAB])
        # Preserve the row's model order for stable rendering.
        order = {m: i for i, m in enumerate(models)}
        row["featured_models"] = sorted(featured, key=lambda m: order[m])


def _apply_custom_aliases(rows: list[dict]) -> None:
    """Attach the accepted identity set to each user-defined provider row.

    A session's ``model.options`` reports the canonical ``custom:<key>`` identity while catalog rows
    carry the bare config key as ``slug``; GUI pickers compare the two to find the active row, and
    exact equality never matches for custom providers.
    """
    from hermes_cli.providers import custom_provider_aliases

    for row in rows:
        if not row.get("is_user_defined"):
            continue
        try:
            row["aliases"] = sorted(
                custom_provider_aliases(str(row.get("name", "")), str(row.get("slug", "")))
            )
        except Exception:
            continue


# ─── Internal: row post-processing ──────────────────────────────────────


def _provider_auth_hint(slug: str) -> tuple[str, str]:
    """``(auth_type, key_env)`` for a canonical provider (``("api_key", "")`` when unregistered)."""
    from hermes_cli.auth import PROVIDER_REGISTRY

    cfg = PROVIDER_REGISTRY.get(slug)
    auth_type = cfg.auth_type if cfg else "api_key"
    key_env = cfg.api_key_env_vars[0] if (cfg and cfg.api_key_env_vars) else ""
    return auth_type, key_env


def _canonical_row(entry, cur: str, **extra: Any) -> dict:
    from hermes_cli.models import _PROVIDER_LABELS

    return {
        "slug": entry.slug, "name": _PROVIDER_LABELS.get(entry.slug, entry.label),
        "is_current": entry.slug.lower() == cur, "is_user_defined": False, **extra,
    }


def _append_unconfigured_rows(
    rows: list[dict], ctx: ConfigContext, *, current_only: bool = False,
) -> list[dict]:
    """Build fallback rows for canonical providers missing from ``rows``.

    Missing providers become empty setup skeletons, except the *current* configured provider: if
    config.yaml still points at it but credentials are unavailable, keep a visible row carrying the
    saved model so GUI pickers don't silently snap to another provider.
    """
    from hermes_cli.models import CANONICAL_PROVIDERS

    seen = {r["slug"].lower() for r in rows}
    cur = (ctx.current_provider or "").lower()
    cur_model = str(ctx.current_model or "").strip()
    extras: list[dict] = []
    for entry in CANONICAL_PROVIDERS:
        if entry.slug.lower() in seen:
            continue
        if current_only and entry.slug.lower() != cur:
            continue
        if entry.slug.lower() == cur:
            auth_type, key_env = _provider_auth_hint(entry.slug)
            warning = (
                f"Configured provider missing usable credentials; paste {key_env} to reactivate. "
                "Showing the saved model only."
                if auth_type == "api_key" and key_env
                else "Configured provider is not authenticated; run `hermes model` to reactivate. "
                "Showing the saved model only."
            )
            extras.append(_canonical_row(
                entry, cur, models=[cur_model] if cur_model else [], total_models=1 if cur_model else 0,
                source="configured-current", authenticated=False, auth_type=auth_type, key_env=key_env,
                warning=warning,
            ))
            continue
        extras.append(_canonical_row(entry, cur, models=[], total_models=0, source="canonical"))
    return extras


def _anthropic_oauth_credentials_present() -> bool:
    """True when the user explicitly authenticated Anthropic via OAuth.

    Hermes' own device flow (auth.json token) and a Claude Code login (~/.claude/.credentials.json)
    leave no trace in active_provider / model.provider / API-key env vars.
    """
    try:
        from agent.anthropic_adapter import read_claude_code_credentials, read_hermes_oauth_credentials

        readers = (read_hermes_oauth_credentials, read_claude_code_credentials)
        if any((read() or {}).get("accessToken") for read in readers):
            return True
    except Exception:
        return False
    # Pool-only OAuth entries (auth.json credential_pool.anthropic) are equally deliberate — the
    # discovery side accepts them via pool.has_credentials(), so the filter must too or those rows
    # are built and then silently dropped. Read-only access (no load_pool) so a picker open never
    # mutates auth.json.
    try:
        from agent.credential_pool import AUTH_TYPE_OAUTH
        from hermes_cli.auth import read_credential_pool

        for entry in read_credential_pool("anthropic"):
            if (isinstance(entry, dict) and entry.get("auth_type") == AUTH_TYPE_OAUTH
                    and str(entry.get("access_token") or "").strip()):
                return True
    except Exception:
        pass
    return False


def _filter_explicit_provider_rows(rows: list[dict], ctx: ConfigContext) -> list[dict]:
    """Keep only rows backed by explicit user configuration.

    ``list_authenticated_providers`` also discovers ambient credentials (e.g. GitHub CLI ->
    Copilot); Desktop chat pickers want only what the user configured for Hermes.
    """
    from hermes_cli.auth import is_provider_explicitly_configured

    current_slug = str(ctx.current_provider or "").strip().lower()

    def _is_explicit(row: dict, slug: str) -> bool:
        if (
            row.get("is_user_defined")
            or (current_slug and slug == current_slug)
            # Managed local models are explicit configuration by existence (gigabytes downloaded
            # into the machine-scoped models dir). There is deliberately no config credential
            # (credential is reachability), so without this clause the row only survives on the
            # profile where Use was last clicked — every other profile loses local models.
            or row.get("source") == "local-runtime"
        ):
            return True
        if slug == "moa":
            # MoA is a virtual routing mode, not a configured provider. Hide it unless current
            # (handled above) or the user wrote an enabled preset into config.yaml. Raw config,
            # so the DEFAULT_CONFIG preset does not make every desktop picker show MoA.
            return _raw_config_has_enabled_moa_preset()
        return (
            # Keyless providers need no configuration at all; hiding them would defeat their
            # zero-setup discoverability.
            _provider_is_keyless(slug)
            # Anthropic OAuth logins (Hermes device flow / Claude Code) are deliberate sign-ins that
            # leave no trace in active_provider, model.provider, or env; the strict gate below would
            # drop the row list_authenticated_providers just accepted.
            or (slug == "anthropic" and _anthropic_oauth_credentials_present())
            # External-process providers (copilot-acp) authenticate through their own CLI — same
            # class as Anthropic OAuth: keep the row picker-discovery just accepted.
            or _external_process_signed_in(slug)
            or is_provider_explicitly_configured(slug)
        )

    kept: list[dict] = []
    for row in rows:
        slug = str(row.get("slug", "")).strip().lower()
        if slug and _is_explicit(row, slug):
            kept.append(row)
    return kept


def _external_process_signed_in(slug: str) -> bool:
    """True when an external-process provider has verified CLI credentials."""
    try:
        from hermes_cli.auth import PROVIDER_REGISTRY, get_external_process_provider_status
        pconfig = PROVIDER_REGISTRY.get(slug)
        if not pconfig or pconfig.auth_type != "external_process":
            return False
        return bool(get_external_process_provider_status(slug).get("auth_verified"))
    except Exception:
        return False


def _provider_is_keyless(slug: str) -> bool:
    """True when the provider's Hermes overlay declares it keyless."""
    try:
        from hermes_cli.providers import HERMES_OVERLAYS
        overlay = HERMES_OVERLAYS.get(slug)
        return bool(overlay is not None and getattr(overlay, "keyless", False))
    except Exception:
        return False


def _raw_config_has_enabled_moa_preset() -> bool:
    """True when the user's raw config explicitly enables MoA.

    ``load_config()`` merges ``DEFAULT_CONFIG["moa"].presets.default`` for everyone; that default is
    not a user choice. MoA stays visible once the user saved at least one enabled preset (or an
    older flat MoA config) in their own config.yaml.
    """
    try:
        from hermes_cli.config import read_raw_config

        raw = read_raw_config()
    except Exception:
        return False

    moa = raw.get("moa") if isinstance(raw, dict) else None
    if not isinstance(moa, dict):
        return False

    presets = moa.get("presets")
    if isinstance(presets, dict):
        return any(
            not isinstance(preset, dict) or preset.get("enabled", True)
            for name, preset in presets.items() if str(name or "").strip()
        )

    legacy_keys = {
        "reference_models", "aggregator", "reference_temperature", "aggregator_temperature",
        "max_tokens", "reference_max_tokens", "fanout",
    }
    return any(key in moa for key in legacy_keys) and bool(moa.get("enabled", True))


def _apply_picker_hints(rows: list[dict]) -> None:
    """Add ``authenticated``/``auth_type``/``key_env``/``warning`` per row."""
    for row in rows:
        if "authenticated" in row:
            continue
        # Skeleton rows (from _append_unconfigured_rows) have empty `models` AND source="canonical";
        # authenticated rows have populated `models` OR a non-canonical source.
        is_skeleton = row.get("source") == "canonical" and not row.get("models")
        row["authenticated"] = not is_skeleton
        if not is_skeleton or row.get("is_user_defined"):
            continue
        auth_type, key_env = _provider_auth_hint(row["slug"])
        row["auth_type"] = auth_type
        row["key_env"] = key_env
        row["warning"] = (
            f"paste {key_env} to activate"
            if auth_type == "api_key" and key_env
            else f"run `hermes model` to configure ({auth_type})"
        )


def _reorder_canonical(rows: list[dict]) -> list[dict]:
    """Canonical slugs in ``CANONICAL_PROVIDERS`` declaration order; truly-custom rows last.

    Keys on slug membership, NOT ``is_user_defined``: rows from the ``providers:`` config dict carry
    that flag even for canonical slugs, so keying on it would demote canonical providers.
    """
    from hermes_cli.models import CANONICAL_PROVIDERS

    order = {e.slug: i for i, e in enumerate(CANONICAL_PROVIDERS)}
    canon = sorted((r for r in rows if r["slug"] in order), key=lambda r: order[r["slug"]])
    extras = [r for r in rows if r["slug"] not in order]
    return canon + extras


def _apply_pricing(rows: list[dict], *, force_fresh_nous_tier: bool = False) -> None:
    """Enrich each provider row with per-model pricing + Nous tier gating.

    Sets ``row["pricing"] = {model_id: {input, output, cache | None, free}}``; for Nous also
    ``row["free_tier"]`` (account is free-tier) and ``row["unavailable_models"]`` (paid models a
    free user can't pick).
    """
    from hermes_cli.models import (
        _format_price_per_mtok, check_nous_free_tier, compute_sale_discount, get_pricing_for_provider,
        partition_nous_models_by_tier,
    )

    # Resolve Nous free-tier once (cached in models.py for the TTL window).
    nous_free_tier: Optional[bool] = None

    for row in rows:
        slug = str(row.get("slug", "")).lower()
        models = row.get("models") or []
        if not models:
            continue
        try:
            raw_pricing = get_pricing_for_provider(slug) or {}
        except Exception:
            raw_pricing = {}
        if not raw_pricing:
            continue

        formatted: dict[str, dict] = {}
        for mid in models:
            p = raw_pricing.get(mid)
            if not p:
                continue
            inp_raw = p.get("prompt", "")
            out_raw = p.get("completion", "")
            cache_raw = p.get("input_cache_read", "")
            inp = _format_price_per_mtok(inp_raw) if inp_raw != "" else ""
            out = _format_price_per_mtok(out_raw) if out_raw != "" else ""
            entry: dict = {
                "input": inp, "output": out,
                "cache": _format_price_per_mtok(cache_raw) if cache_raw else None,
                # "free" when both input and output cost nothing.
                "free": inp == "free" and out in ("free", ""),
            }
            # Sale chrome is Nous Portal-only: other providers never get discount_percent / was_*
            # even if a nested pricing.original appears in their catalog. Free / $0 models get flat
            # -100% chrome (was_* only when the gateway served an original).
            if slug == "nous":
                sale = compute_sale_discount(inp_raw, out_raw, p.get("original"))
                if sale is not None:
                    discount_percent, was_prompt_raw, was_out_raw = sale
                    entry["discount_percent"] = discount_percent
                    for key, was_raw in (("was_input", was_prompt_raw), ("was_output", was_out_raw)):
                        if was_raw != "":
                            entry[key] = _format_price_per_mtok(was_raw)
            formatted[mid] = entry

        if formatted:
            row["pricing"] = formatted

        if slug == "nous":
            try:
                if nous_free_tier is None:
                    nous_free_tier = check_nous_free_tier(force_fresh=force_fresh_nous_tier)
                row["free_tier"] = bool(nous_free_tier)
                row["unavailable_models"] = (
                    partition_nous_models_by_tier(list(models), raw_pricing, free_tier=True)[1]
                    if nous_free_tier else []
                )
            except Exception:
                # Tier detection failed — fail open (no gating) so the user can always pick a model.
                row["free_tier"] = False
                row["unavailable_models"] = []


def _local_runtime_row(ctx: "ConfigContext") -> dict | None:
    """Build the ``llamacpp`` provider row from staged local models, or ``None`` when none staged.

    Present whenever GGUFs are staged in the managed models directory — downloaded models must be
    selectable before the server runs (selection starts it via the runtime_provider seam).
    """
    try:
        from hermes_cli.local_runtime.bootstrap import staged_model_ids

        staged = staged_model_ids()
        if not staged:
            return None
        current = (ctx.current_provider or "").strip().lower() in ("llamacpp", "llama.cpp", "llama-cpp")
        if not current:
            # A LIVE session on the managed server reports provider "custom" (the resolution seam's
            # label) with the managed base_url. Match on the endpoint so the picker still marks
            # this row current — otherwise the session being chatted in shows no selection.
            try:
                from hermes_cli.local_runtime.endpoint import _state_endpoint

                managed = _state_endpoint()
                current = bool(
                    managed
                    and (ctx.current_base_url or "").strip().rstrip("/") == managed["base_url"].rstrip("/"))
            except Exception:
                current = False
        return {
            "slug": "llamacpp",
            # Bare "Local" everywhere user-facing: the engine name is an implementation detail.
            "name": "Local",
            "is_current": current, "is_user_defined": False, "models": staged, "total_models": len(staged),
            "source": "local-runtime",
            "authenticated": True,  # the credential is reachability
            "auth_type": "local", "warning": None,
        }
    except Exception:
        return None


def _moa_provider_row(current_provider: str = "") -> dict | None:
    """Build the virtual ``moa`` provider row shared by the CLI inventory and gateway picker.

    Returns ``None`` when no MoA presets exist.
    """
    try:
        from hermes_cli.config import load_config
        from hermes_cli.moa_config import normalize_moa_config

        cfg = normalize_moa_config(load_config().get("moa") or {})
        models = list(cfg.get("presets", {}).keys())
        if not models:
            return None
        return {
            "slug": "moa", "name": "Mixture of Agents",
            "is_current": (current_provider or "").lower() == "moa", "is_user_defined": False,
            "models": models, "total_models": len(models), "source": "virtual",
            "authenticated": True, "auth_type": "virtual",
            "warning": "Aggregator acts as the selected model; references provide analysis before each call.",
        }
    except Exception:
        return None
