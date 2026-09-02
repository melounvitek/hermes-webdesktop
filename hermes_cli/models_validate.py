"""Validate a requested ``/model`` value against the active provider's catalog.

Split out of ``hermes_cli.models``; :func:`validate_requested_model` is re-imported there, so
``hermes_cli.models.validate_requested_model`` keeps resolving. Every catalog fetcher this module
calls is looked up on ``hermes_cli.models`` at call time (``_m.<name>``), so existing
``patch("hermes_cli.models.<name>")`` mocks keep intercepting.

Every provider branch returns one of four verdict shapes (see :func:`_verdict`) or ``None`` to
mean "not decided here — keep walking the ladder". The ladder ORDER is behavior: moa → whitespace
→ OpenRouter preset parse → LM Studio → Ollama native → custom → codex/xai static → MiniMax →
Anthropic native → Anthropic Messages → live listing → Bedrock → curated-catalog fallback.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from difflib import get_close_matches
from typing import Any, Optional

from utils import base_url_host_matches


# ── Verdicts ─────────────────────────────────────────────────────────────

def _verdict(accepted: bool, persist: bool, recognized: bool, message: Optional[str],
             corrected_model: Optional[str] = None) -> dict[str, Any]:
    """Build the verdict dict. ``corrected_model`` is only present when set (key order matters
    to nobody, but keep it identical to the historical literals anyway)."""
    out: dict[str, Any] = {"accepted": accepted, "persist": persist, "recognized": recognized}
    if corrected_model is not None:
        out["corrected_model"] = corrected_model
    out["message"] = message
    return out


def _accept() -> dict[str, Any]:
    return _verdict(True, True, True, None)


def _reject(message: str) -> dict[str, Any]:
    return _verdict(False, False, False, message)


def _soft_accept(message: Optional[str]) -> dict[str, Any]:
    """Accept + persist an unrecognized name, with a warning."""
    return _verdict(True, True, False, message)


def _corrected(requested: str, corrected: str) -> dict[str, Any]:
    return _verdict(True, True, True, f"Auto-corrected `{requested}` → `{corrected}`",
                    corrected_model=corrected)


# ── Catalog matching ─────────────────────────────────────────────────────

@dataclass
class _Match:
    exact: bool = False
    corrected: Optional[str] = None
    suggestion_text: str = ""


def _match_in_catalog(
    query: str,
    candidates,
    *,
    case_insensitive: bool = False,
    auto_correct: bool = True,
    suggest_query: Optional[str] = None,
    suggest_cutoff: float = 0.5,
    suggest_label: str = "Similar models",
) -> _Match:
    """The shared ladder: exact membership → typo auto-correct (cutoff .9) → suggestion text.

    ``case_insensitive`` matches on lower-cased ids and maps results back to the catalog's
    spelling (MiniMax ships mixed-case ids). ``suggest_query`` overrides the string the
    suggestion search uses (some branches search on the raw request, not the lookup form).
    """
    candidates = list(candidates)
    if case_insensitive:
        display = {c.lower(): c for c in candidates}
        pool = list(display)
        query = query.lower()
        suggest_query = query if suggest_query is None else suggest_query.lower()
    else:
        display = None
        pool = candidates
        suggest_query = query if suggest_query is None else suggest_query

    def _show(cid: str) -> str:
        return display[cid] if display is not None else cid

    if query in set(pool):
        return _Match(exact=True)
    if auto_correct:
        auto = get_close_matches(query, pool, n=1, cutoff=0.9)
        if auto:
            return _Match(corrected=_show(auto[0]))
    suggestions = get_close_matches(suggest_query, pool, n=3, cutoff=suggest_cutoff)
    text = ""
    if suggestions:
        text = f"\n  {suggest_label}: " + ", ".join(f"`{_show(s)}`" for s in suggestions)
    return _Match(suggestion_text=text)


# ── Request context ──────────────────────────────────────────────────────

@dataclass
class _Request:
    requested: str
    lookup: str                 # id used for catalog membership (copilot-normalized / preset base)
    provider: Optional[str]     # raw caller value (Ollama checks look at this, not ``normalized``)
    normalized: str
    api_key: Optional[str]
    base_url: Optional[str]
    api_mode: Optional[str]
    headers: Optional[dict[str, str]]
    preset_suffix: str = ""

    def with_preset_suffix(self, model_id: str) -> str:
        """Re-attach a preserved ``@preset/<slug>`` suffix after auto-correction."""
        return f"{model_id}{self.preset_suffix}"


# ── Provider branches (None = not decided here) ─────────────────────────

def _validate_moa(requested: str) -> dict[str, Any]:
    try:
        from hermes_cli.config import load_config
        from hermes_cli.moa_config import normalize_moa_config

        cfg = normalize_moa_config(load_config().get("moa") or {})
        if requested in cfg["presets"]:
            return _accept()
        return _reject(f"MoA preset `{requested}` was not found. Run `hermes moa list`.")
    except Exception as exc:
        return _reject(f"Could not read MoA presets: {exc}")


def _parse_openrouter_preset(req: _Request) -> Optional[dict[str, Any]]:
    """OpenRouter presets are account-scoped, so ``@preset/<slug>`` never appears in the public
    /v1/models listing. A bare preset is accepted unverified; ``<model>@preset/<slug>`` validates
    the base model and preserves the suffix through auto-correction. OpenRouter validates the slug
    at request time."""
    marker = "@preset/"
    if marker not in req.requested:
        return None
    if req.requested.count(marker) != 1:
        preset_slug, preset_base = "", req.requested
    else:
        preset_base, preset_slug = req.requested.split(marker, 1)
    if re.fullmatch(r"[A-Za-z0-9._~-]+", preset_slug) is None:
        return _reject(
            "OpenRouter preset slugs must be non-empty URL-safe "
            "identifiers using only letters, digits, '.', '_', "
            "'~', or '-'."
        )
    req.preset_suffix = f"{marker}{preset_slug}"
    if not preset_base:
        return _soft_accept(None)
    req.lookup = preset_base
    return None


def _validate_lmstudio(req: _Request) -> dict[str, Any]:
    from hermes_cli import models as _m
    from hermes_cli.auth import AuthError

    # probe_lmstudio_models distinguishes None (unreachable / malformed) from [] (reachable,
    # nothing chat-capable loaded); fetch_lmstudio_models collapses both to [].
    try:
        models = _m.probe_lmstudio_models(api_key=req.api_key, base_url=req.base_url)
    except AuthError as exc:
        return _reject(f"{exc} Set `LM_API_KEY` (or update it) to match the server's bearer token.")
    if models is None:
        return _reject(f"Could not reach LM Studio's `/api/v1/models` to validate `{req.requested}`.")
    if not models:
        return _reject(
            f"LM Studio is reachable but no chat-capable models are loaded. "
            f"Load `{req.requested}` in LM Studio (Developer tab → Load Model) and try again."
        )
    if req.lookup in set(models):
        return _accept()
    return _reject(f"Model `{req.requested}` was not found in LM Studio's model listing.")


def _ollama_probe_headers(req: _Request) -> dict[str, str]:
    """Headers for the Ollama native probe.

    Configured ``providers.ollama.extra_headers`` are only applied when the probed endpoint is
    the configured one (never leak them to a different host). Caller headers win; a caller
    ``api_key`` becomes the Authorization header unless the caller already sent one.
    """
    from hermes_cli import models as _m
    from hermes_cli.models_local import _configured_ollama_base_url

    configured_base = _configured_ollama_base_url()
    configured_allowed = not (
        configured_base and not _m._same_ollama_native_root(req.base_url or "", configured_base)
    )
    if req.headers is None:
        return _m._get_ollama_native_headers(req.base_url, api_key=req.api_key) if configured_allowed else {}
    out: dict[str, str] = {}
    if configured_allowed:
        out.update(_m._get_ollama_native_headers(req.base_url, api_key=req.api_key))
    for key in tuple(out):
        if key.lower() == "authorization":
            del out[key]
    out.update(req.headers)
    caller_has_authorization = any(key.lower() == "authorization" for key in req.headers)
    if req.api_key and not caller_has_authorization:
        for key in tuple(out):
            if key.lower() == "authorization":
                del out[key]
        out["Authorization"] = f"Bearer {req.api_key}"
    return out


def _validate_ollama_native(req: _Request) -> Optional[dict[str, Any]]:
    """Runs for EVERY provider: the native ``/api/tags`` catalog is used whenever the endpoint
    looks like a local Ollama server. Also resolves ``base_url`` for the raw ``ollama`` provider,
    which later branches (custom) rely on."""
    from hermes_cli import models as _m

    if str(req.provider or "").strip().lower() == "ollama" and not req.base_url:
        req.base_url = _m._get_ollama_base_url()
    headers = _ollama_probe_headers(req)
    if not _m.should_use_ollama_native_catalog(req.provider, req.base_url, headers=headers):
        return None
    models = _m.probe_ollama_local_models(req.base_url, headers=headers)
    if models is None:
        # A failed native probe is not authoritative; fall back to the OpenAI-compatible
        # catalog before accepting blindly.
        models = _m.probe_api_models(
            req.api_key,
            _m._normalize_openai_base_url(req.base_url),
            request_headers=headers,
        ).get("models")
    if models is None:
        return _soft_accept(
            f"Note: could not reach this Ollama endpoint's `/api/tags` model listing to validate `{req.requested}`. "
            "Hermes will save the model name, but local Ollama model discovery could not verify it."
        )
    match = _match_in_catalog(req.lookup, models, auto_correct=False,
                              suggest_label="Similar local Ollama models")
    if match.exact:
        return _accept()
    empty_hint = " No models are currently listed by `/api/tags`." if not models else ""
    return _soft_accept(
        f"Note: `{req.requested}` was not found in this Ollama endpoint's `/api/tags` model listing."
        f"{empty_hint} It may still work if the server supports hidden or aliased models."
        f"{match.suggestion_text}"
    )


def _validate_custom(req: _Request) -> dict[str, Any]:
    from hermes_cli import models as _m

    # Probe with the auth shape the api_mode expects.
    if req.api_mode == "anthropic_messages":
        probe = _m.probe_api_models(req.api_key, req.base_url, api_mode=req.api_mode,
                                    request_headers=req.headers)
    else:
        probe = _m.probe_api_models(req.api_key, req.base_url, request_headers=req.headers)
    api_models = probe.get("models")
    if api_models is not None:
        match = _match_in_catalog(req.lookup, api_models, suggest_query=req.requested)
        if match.exact:
            return _accept()
        if match.corrected:
            return _corrected(req.requested, match.corrected)
        message = (
            f"Note: `{req.requested}` was not found in this custom endpoint's model listing "
            f"({probe.get('probed_url')}). It may still work if the server supports hidden or aliased models."
            f"{match.suggestion_text}"
        )
        if probe.get("used_fallback"):
            message += (
                f"\n  Endpoint verification succeeded after trying `{probe.get('resolved_base_url')}`. "
                f"Consider saving that as your base URL."
            )
        return _soft_accept(message)

    message = (
        f"Note: could not reach this custom endpoint's model listing at `{probe.get('probed_url')}`. "
        f"Hermes will still save `{req.requested}`, but the endpoint should expose `/models` for verification."
    )
    if req.api_mode == "anthropic_messages":
        message += (
            "\n  Many Anthropic-compatible proxies do not implement the Models API "
            "(GET /v1/models).  The model name has been accepted without verification."
        )
    if probe.get("suggested_base_url"):
        message += f"\n  If this server expects `/v1`, try base URL: `{probe.get('suggested_base_url')}`"
    # Anthropic-style proxies routinely lack /v1/models, so only they are accepted unverified.
    return _verdict(req.api_mode == "anthropic_messages", True, False, message)


def _static_catalog(normalized: str) -> list[str]:
    from hermes_cli import models as _m

    try:
        return _m.provider_model_ids(normalized)
    except Exception:
        return []


_STATIC_FAMILY_PREFIXES = {
    "openai-codex": ("gpt-", "codex-", "o1", "o3", "o4"),
    "xai-oauth": ("grok-",),
}
_STATIC_LABELS = {"openai-codex": "OpenAI Codex", "xai-oauth": "xAI Grok OAuth (SuperGrok / Premium+)"}


def _validate_static_catalog(req: _Request) -> Optional[dict[str, Any]]:
    """openai-codex / xai-oauth: no /v1/models probing — validate against the curated catalog.
    Returns None (fall through) when the catalog is empty."""
    catalog = _static_catalog(req.normalized)
    if req.normalized == "openai-codex":
        from agent.model_metadata import CODEX_CONTEXT_VARIANT_SUFFIX, is_codex_context_variant

        # Ineligible ``-900k`` aliases must be rejected BEFORE the hidden-slug soft-accept:
        # the suffix is a Hermes picker convention, so an unknown `*-900k` can never be a real
        # hidden provider slug — soft-accepting one silently runs at 272K on a different model.
        if req.lookup.strip().lower().endswith(CODEX_CONTEXT_VARIANT_SUFFIX) and req.lookup not in set(catalog):
            if is_codex_context_variant(req.lookup):
                # Valid variant a stale catalog hasn't synthesized yet. Accept directly — the typo
                # auto-corrector would otherwise "fix" it to the base slug and drop the opt-in.
                return _accept()
            base_guess = req.lookup[: -len(CODEX_CONTEXT_VARIANT_SUFFIX)]
            return _reject(
                f"`{req.requested}` is not a valid large-context variant — "
                f"`{base_guess}` enforces the standard 272K window on "
                f"Codex, so no `-900k` option exists for it. Pick the "
                f"base model, or a verified variant from the `/model` "
                f"picker (e.g. `gpt-5.6-sol-900k`)."
            )
    if not catalog:
        return None
    match = _match_in_catalog(req.lookup, catalog)
    if match.exact:
        return _accept()
    if match.corrected:
        return _corrected(req.requested, match.corrected)
    label = _STATIC_LABELS[req.normalized]
    # Plausibility gate: the soft-accept exists for entitlement-gated *hidden* slugs the curated
    # listing hasn't caught up with — always the provider's own family (gpt-* / grok-*). An
    # unrelated name (`qwen3.5-4b`) would turn an actionable "did you mean --provider <x>?" into
    # a confusing success that 400s on the next turn, so reject it with guidance instead.
    prefixes = _STATIC_FAMILY_PREFIXES.get(req.normalized, ())
    lower = req.lookup.strip().lower()
    if prefixes and not any(lower.startswith(p) for p in prefixes):
        return _reject(
            f"`{req.requested}` doesn't look like a {label} model "
            f"and isn't in its listing, so it was not accepted. If it "
            f"belongs to another configured provider, switch with "
            f"`--provider <slug>` (or select it from the `/model` "
            f"picker)."
            f"{match.suggestion_text}"
        )
    return _soft_accept(
        f"Note: `{req.requested}` was not found in the {label} model listing. "
        "It may still work if your account has access to a newer or hidden model ID."
        f"{match.suggestion_text}"
    )


def _validate_minimax(req: _Request) -> Optional[dict[str, Any]]:
    """MiniMax has no /models endpoint — static catalog, case-insensitive (ids like MiniMax-M2.7).
    Returns None when the catalog is empty."""
    catalog = _static_catalog(req.normalized)
    if not catalog:
        return None
    match = _match_in_catalog(req.lookup, catalog, case_insensitive=True)
    if match.exact:
        return _accept()
    if match.corrected:
        return _corrected(req.requested, match.corrected)
    return _soft_accept(
        f"Note: `{req.requested}` was not found in the MiniMax catalog."
        f"{match.suggestion_text}"
        "\n  MiniMax does not expose a /models endpoint, so Hermes cannot verify the model name."
        "\n  The model may still work if it exists on the server."
    )


def _validate_anthropic(req: _Request) -> Optional[dict[str, Any]]:
    """Native Anthropic: /v1/models needs x-api-key (or OAuth Bearer) + anthropic-version, so the
    generic Bearer probe 401s — use the native fetcher. Returns None (fall through to the generic
    ladder) when no token is resolvable or the network failed."""
    from hermes_cli import models as _m

    models = _m._fetch_anthropic_models(base_url=req.base_url or None, api_key=req.api_key or None)
    if models is None:
        return None
    match = _match_in_catalog(req.lookup, models, suggest_query=req.requested)
    if match.exact:
        return _accept()
    if match.corrected:
        return _corrected(req.requested, match.corrected)
    # Accept anyway — Anthropic gates newer/preview models (snapshot IDs, early access) behind
    # accounts even though they aren't listed on /v1/models.
    return _soft_accept(
        f"Note: `{req.requested}` was not found in Anthropic's /v1/models listing. "
        f"It may still work if you have early-access or snapshot IDs."
        f"{match.suggestion_text}"
    )


def _validate_anthropic_messages(req: _Request) -> dict[str, Any]:
    """Anthropic Messages transport: many proxies don't implement /v1/models — probe, and accept
    with a warning when the probe fails or the model isn't listed."""
    from hermes_cli import models as _m

    models = _m.fetch_api_models(req.api_key, req.base_url, api_mode=req.api_mode)
    if models is not None:
        match = _match_in_catalog(req.lookup, models)
        if match.exact:
            return _accept()
        if match.corrected:
            return _corrected(req.requested, match.corrected)
    return _soft_accept(
        f"Note: could not verify `{req.requested}` against this endpoint's "
        f"model listing.  Many Anthropic-compatible proxies do not "
        f"implement GET /v1/models.  The model name has been accepted "
        f"without verification."
    )


def _nous_portal_recommended_names() -> set[str]:
    """Lower-cased ids from the Portal's live recommended-models feed (empty on any failure)."""
    from hermes_cli import models as _m

    try:
        payload = _m.fetch_nous_recommended_models(_m._resolve_nous_portal_url())
        return {
            name.lower()
            for tier in ("freeRecommendedModels", "paidRecommendedModels")
            for entry in (payload.get(tier) or [])
            if (name := _m._extract_model_name(entry))
        }
    except Exception:
        return set()


def _validate_live_listing(req: _Request) -> Optional[dict[str, Any]]:
    """Generic live /v1/models probe. Returns None when the API was unreachable (the caller then
    tries Bedrock discovery / the curated catalog)."""
    from hermes_cli import models as _m

    api_models = _m.fetch_api_models(req.api_key, req.base_url)
    if api_models is None:
        return None
    if req.normalized == "gemini":
        # Gemini's OpenAI-compat listing prefixes ids with "models/"; curated list and user
        # input use the bare id, so strip before comparing.
        api_models = [
            m[len("models/"):] if isinstance(m, str) and m.startswith("models/") else m
            for m in api_models
        ]
    match = _match_in_catalog(req.lookup, api_models)
    if match.exact:
        return _accept()
    # OpenRouter routing variants (":nitro", ":floor", ...) are request-time modifiers, not
    # catalog entries — validate the BASE but keep the suffixed id. Must run BEFORE fuzzy
    # auto-correction, which would otherwise "correct" `model:nitro` → `model` and silently
    # strip the routing opt-in.
    variant_base = _m._openrouter_variant_base(req.lookup) if req.normalized == "openrouter" else None
    if variant_base is not None and variant_base in set(api_models):
        return _accept()
    # Listed but not found: the account may reach models absent from the public listing
    # (e.g. Z.AI Pro/Max plans use glm-5 on coding endpoints) — warn but allow where plausible.
    if match.corrected:
        corrected = req.with_preset_suffix(match.corrected)
        return _corrected(req.requested, corrected)
    # Curated-catalog soft-accept: providers omit valid models from live listings (stale cache,
    # partial rollout, gated previews). EXCEPTION: official OpenAI hosts (canonical + data-
    # residency regional) — their listing is access-scoped and authoritative, so an absent model
    # is one this key CANNOT serve; a soft-accept would 400 at first use. Custom OpenAI-compatible
    # proxies keep the fallback.
    listing_authoritative = False
    if req.normalized in ("openai", "openai-api"):
        from hermes_cli.providers import is_official_openai_host

        listing_authoritative = is_official_openai_host(req.base_url)
    if not listing_authoritative and _m._model_in_provider_catalog(
        (variant_base or req.lookup).lower(), _m._provider_keys(req.normalized)
    ):
        return _verdict(True, True, True,
                        f"Note: `{req.requested}` was not found in the live /v1/models listing "
                        f"but exists in the curated catalog — accepted.")
    # Nous: the Portal's recommended-models feed can list a model before the curated list or the
    # docs-hosted manifest catches up; `hermes chat` already accepts those at model-list build
    # time, so mirror that source of truth for per-message /model validation.
    if req.normalized == "nous" and req.lookup.lower() in _nous_portal_recommended_names():
        return _verdict(True, True, True,
                        f"Note: `{req.requested}` was not found in the live /v1/models "
                        f"listing but is a current Nous Portal recommendation — accepted.")
    return _reject(
        f"Model `{req.requested}` was not found in this provider's model listing."
        f"{match.suggestion_text}"
    )


def _validate_bedrock(req: _Request) -> Optional[dict[str, Any]]:
    """Bedrock's runtime URL has no /models; discovery goes through the AWS control plane
    (ListFoundationModels + ListInferenceProfiles). Any failure falls through (None)."""
    try:
        from agent.bedrock_adapter import discover_bedrock_models, resolve_bedrock_runtime_region

        region = resolve_bedrock_runtime_region()
        discovered_ids = {m["id"] for m in discover_bedrock_models(region)}
        match = _match_in_catalog(req.requested, list(discovered_ids), auto_correct=False,
                                  suggest_cutoff=0.4)
        if match.exact:
            return _accept()
        # Still accept (custom inference profiles / cross-account access), but warn.
        return _soft_accept(
            f"Note: `{req.requested}` was not found in Bedrock model discovery for {region}. "
            f"It may still work with custom inference profiles or cross-account access."
            f"{match.suggestion_text}"
        )
    except Exception:
        return None


def _validate_catalog_fallback(req: _Request) -> dict[str, Any]:
    """The /models probe was unreachable: validate against the curated ``provider_model_ids()``
    list so gateway /model switches keep working while a provider's endpoint is down (otherwise
    switch_model() would fail and the gateway never writes the session override). No catalog at
    all → accept with a warning."""
    from hermes_cli import models as _m

    label = _m._PROVIDER_LABELS.get(req.normalized, req.normalized)
    catalog = _static_catalog(req.normalized)
    if not catalog:
        return _soft_accept(
            f"Note: could not reach the {label} API to validate `{req.requested}`. "
            f"If the service isn't down, this model may not be valid."
        )
    match = _match_in_catalog(req.lookup, catalog, case_insensitive=True)
    if match.exact:
        return _accept()
    # Same OpenRouter routing-variant rule as the live-listing path.
    if req.normalized == "openrouter":
        variant_base = _m._openrouter_variant_base(req.lookup)
        if variant_base is not None and variant_base.lower() in {m.lower() for m in catalog}:
            return _accept()
    if match.corrected:
        corrected = req.with_preset_suffix(match.corrected)
        return _corrected(req.requested, corrected)
    return _soft_accept(
        f"Note: `{req.requested}` was not found in the {label} curated catalog "
        f"and the /models endpoint was unreachable.{match.suggestion_text}"
        f"\n  The model may still work if it exists on the provider."
    )


# ── Orchestrator ─────────────────────────────────────────────────────────

def validate_requested_model(
    model_name: str,
    provider: Optional[str],
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    api_mode: Optional[str] = None,
    headers: Optional[dict[str, str]] = None,
) -> dict[str, Any]:
    """Validate a ``/model`` value for the active provider.

    Returns a dict with: - accepted: whether the CLI should switch to the requested model now -
    persist: whether it is safe to save to config - recognized: whether it matched a known provider
    catalog - message: optional warning / guidance for the user (- corrected_model: when a typo
    was auto-corrected).
    """
    from hermes_cli import models as _m

    requested = (model_name or "").strip()
    normalized = _m.normalize_provider(provider)
    if normalized == "openrouter" and base_url and not base_url_host_matches(base_url, "openrouter.ai"):
        normalized = "custom"
    lookup = requested
    if normalized == "copilot":
        lookup = _m.normalize_copilot_model_id(requested, api_key=api_key) or requested

    if not requested:
        return _reject("Model name cannot be empty.")
    if normalized == "moa":
        return _validate_moa(requested)
    if any(ch.isspace() for ch in requested):
        return _reject("Model names cannot contain spaces.")

    req = _Request(requested, lookup, provider, normalized, api_key, base_url, api_mode, headers)
    if normalized == "openrouter":
        verdict = _parse_openrouter_preset(req)
        if verdict is not None:
            return verdict
    if normalized == "lmstudio":
        return _validate_lmstudio(req)
    verdict = _validate_ollama_native(req)
    if verdict is not None:
        return verdict
    if normalized == "custom" or normalized.startswith("custom:"):
        return _validate_custom(req)
    if normalized in {"openai-codex", "xai-oauth"}:
        verdict = _validate_static_catalog(req)
        if verdict is not None:
            return verdict
    if normalized in {"minimax", "minimax-cn"}:
        verdict = _validate_minimax(req)
        if verdict is not None:
            return verdict
    if normalized == "anthropic":
        verdict = _validate_anthropic(req)
        if verdict is not None:
            return verdict
    if api_mode == "anthropic_messages":
        return _validate_anthropic_messages(req)
    verdict = _validate_live_listing(req)
    if verdict is not None:
        return verdict
    # API unreachable — accept and persist, but warn so typos don't silently break things.
    if normalized == "bedrock":
        verdict = _validate_bedrock(req)
        if verdict is not None:
            return verdict
    return _validate_catalog_fallback(req)
