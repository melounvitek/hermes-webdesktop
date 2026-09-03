"""Models.dev registry integration — primary database for providers and models.

Fetches https://models.dev/api.json. Resolution order: in-memory cache (fresh, or
stale served immediately while one background daemon thread refreshes) → disk
cache (~/.hermes/models_dev_cache.json, any age) → network, only when no cache
exists at all. Failed refreshes back off 5 min process-wide. Refreshes use ETag
conditional GET whenever a servable registry is held (a 304 re-confirms without
re-downloading ~2 MB). Hot paths pass ``allow_network=False`` and never do I/O.
A corrupt/empty disk cache is quarantined, never served as ``{}``. The URL can
be overridden via ``models_dev.url`` in config.yaml (mirrors)."""

import json
import logging
import re
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils import atomic_json_write, atomic_write_text

import requests

logger = logging.getLogger(__name__)

MODELS_DEV_URL = "https://models.dev/api.json"
_MODELS_DEV_CACHE_TTL = 4 * 3600  # 4 hours — ETag conditional GET makes refresh cheap
_MODELS_DEV_RETRY_DELAY = 300  # 5 minutes after a failed refresh

# In-memory cache
_models_dev_cache: Dict[str, Any] = {}
_models_dev_cache_time: float = 0
_models_dev_retry_after: float = 0
_models_dev_fetch_lock = threading.Lock()
_models_dev_refresh_lock = threading.Lock()
_models_dev_refresh_in_flight = False


# --- Dataclasses -----------------------------------------------------------

@dataclass
class ModelInfo:
    """Full metadata for a single model from models.dev."""

    id: str
    name: str
    family: str
    provider_id: str        # models.dev provider ID (e.g. "anthropic")
    # Capabilities
    reasoning: bool = False
    tool_call: bool = False
    attachment: bool = False       # supports image/file attachments (vision)
    temperature: bool = False
    structured_output: bool = False
    open_weights: bool = False
    # Modalities
    input_modalities: Tuple[str, ...] = ()    # ("text", "image", "pdf", ...)
    output_modalities: Tuple[str, ...] = ()
    # Limits
    context_window: int = 0
    max_output: int = 0
    max_input: Optional[int] = None
    # Cost (per million tokens, USD)
    cost_input: float = 0.0
    cost_output: float = 0.0
    cost_cache_read: Optional[float] = None
    cost_cache_write: Optional[float] = None
    # Metadata
    knowledge_cutoff: str = ""
    release_date: str = ""
    status: str = ""          # "alpha", "beta", "deprecated", or ""
    interleaved: Any = False  # True or {"field": "reasoning_content"}

    def has_cost_data(self) -> bool:
        return self.cost_input > 0 or self.cost_output > 0

    def supports_vision(self) -> bool:
        return self.attachment or "image" in self.input_modalities

    def supports_pdf(self) -> bool:
        return "pdf" in self.input_modalities

    def supports_audio_input(self) -> bool:
        return "audio" in self.input_modalities

    def format_capabilities(self) -> str:
        """Human-readable capabilities, e.g. 'reasoning, tools, vision, PDF'."""
        flags = (
            (self.reasoning, "reasoning"), (self.tool_call, "tools"),
            (self.supports_vision(), "vision"), (self.supports_pdf(), "PDF"),
            (self.supports_audio_input(), "audio"),
            (self.structured_output, "structured output"), (self.open_weights, "open weights"),
        )
        caps = [label for on, label in flags if on]
        return ", ".join(caps) if caps else "basic"


@dataclass
class ProviderInfo:
    """Full metadata for a provider from models.dev."""

    id: str                         # models.dev provider ID
    name: str                       # display name
    env: Tuple[str, ...]            # env var names for API key
    api: str                        # base URL
    doc: str = ""                   # documentation URL
    model_count: int = 0


@dataclass
class ModelCapabilities:
    """Structured capability metadata for a model from models.dev."""

    supports_tools: bool = True
    supports_vision: bool = False
    supports_reasoning: bool = False
    context_window: int = 200000
    max_output_tokens: int = 8192
    model_family: str = ""


# --- Provider ID mapping: Hermes ↔ models.dev ------------------------------

# Hermes provider names → models.dev provider IDs
PROVIDER_TO_MODELS_DEV: Dict[str, str] = {
    "openrouter": "openrouter",
    "novita": "novita-ai",
    "anthropic": "anthropic",
    "openai": "openai",
    "openai-codex": "openai",
    "zai": "zai",
    "kimi": "kimi-for-coding",
    "kimi-coding": "kimi-for-coding",
    "moonshot": "kimi-for-coding",
    "stepfun": "stepfun",
    "kimi-coding-cn": "kimi-for-coding",
    "minimax": "minimax",
    "minimax-oauth": "minimax",
    "minimax-cn": "minimax-cn",
    "deepseek": "deepseek",
    "alibaba": "alibaba",
    "qwen-oauth": "alibaba",
    "copilot": "github-copilot",
    "ai-gateway": "vercel",
    "opencode-zen": "opencode",
    "opencode-go": "opencode-go",
    "kilocode": "kilo",
    "fireworks": "fireworks-ai",
    "huggingface": "huggingface",
    "gemini": "google",
    "google": "google",
    "xai": "xai",
    "xai-oauth": "xai",  # OAuth is a transport path for the same xAI catalog
    "xiaomi": "xiaomi",
    "nvidia": "nvidia",
    # Meta Model API (Muse Spark, api.meta.ai): models.dev keys it "meta", the
    # Hermes provider is "meta-ai"; both aliases are needed or muse-spark-*
    # falls back to the generic 256K default instead of its true 1M window.
    "meta-ai": "meta",
    "meta": "meta",
    "groq": "groq",
    "mistral": "mistral",
    "togetherai": "togetherai",
    "perplexity": "perplexity",
    "cohere": "cohere",
    "ollama-cloud": "ollama-cloud",
}

# Reverse mapping: models.dev id → Hermes ids (built lazily; many-to-one).
_MODELS_DEV_TO_PROVIDER: Optional[Dict[str, List[str]]] = None


def _models_dev_to_hermes_ids(mdev_id: str) -> List[str]:
    """Return the Hermes provider ids that map to *mdev_id* (may be [])."""
    global _MODELS_DEV_TO_PROVIDER
    if _MODELS_DEV_TO_PROVIDER is None:
        reverse: Dict[str, List[str]] = {}
        for hermes_id, mapped in PROVIDER_TO_MODELS_DEV.items():
            reverse.setdefault(mapped, []).append(hermes_id)
        _MODELS_DEV_TO_PROVIDER = reverse
    return _MODELS_DEV_TO_PROVIDER.get(mdev_id, [])


def _dict_or_empty(value: Any) -> Dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _cfg_get(*keys: str, default: Any) -> Any:
    """``cfg_get`` over the read-only config; *default* on any failure."""
    try:
        from hermes_cli.config import cfg_get, load_config_readonly
        return cfg_get(load_config_readonly(), *keys, default=default)
    except Exception:
        return default


# --- Disk cache + ETag sidecar ---------------------------------------------

def _hermes_path(name: str) -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / name


def _get_cache_path() -> Path:
    return _hermes_path("models_dev_cache.json")


def _get_etag_path() -> Path:
    return _hermes_path("models_dev_cache.etag")


def _load_etag() -> str:
    """Last-known ETag from disk, or "" if missing."""
    try:
        etag_path = _get_etag_path()
        if etag_path.exists():
            return etag_path.read_text(encoding="utf-8").strip()
    except Exception as e:
        logger.debug("Failed to load models.dev ETag: %s", e)
    return ""


def _save_etag(etag: str) -> None:
    try:
        etag_path = _get_etag_path()
        etag_path.parent.mkdir(parents=True, exist_ok=True)
        atomic_write_text(etag_path, etag)
    except Exception as e:
        logger.debug("Failed to save models.dev ETag: %s", e)


def _clear_etag() -> None:
    """Delete the ETag sidecar so the next fetch is unconditional: an
    If-None-Match without a servable cache invites a 304 that leaves the
    process with no data at all."""
    try:
        _get_etag_path().unlink(missing_ok=True)
    except Exception as e:
        logger.debug("Failed to clear models.dev ETag: %s", e)


def _get_models_dev_url() -> str:
    """The models.dev API URL, honoring the ``models_dev.url`` config override."""
    url = _cfg_get("models_dev", "url", default="")
    if isinstance(url, str) and url.strip():
        return url.strip()
    # Module global (not a captured constant) so patching MODELS_DEV_URL works.
    return MODELS_DEV_URL


def _validate_registry(data: Any) -> bool:
    """True if *data* is a non-empty dict suitable for serving."""
    return isinstance(data, dict) and len(data) > 0


def _load_disk_cache() -> Dict[str, Any]:
    """Load the disk cache; a corrupt/empty one is quarantined with a warning
    so it never masquerades as ``{}`` and breaks provider/model resolution."""
    try:
        cache_path = _get_cache_path()
        if cache_path.exists():
            with open(cache_path, encoding="utf-8") as f:
                data = json.load(f)
            if _validate_registry(data):
                return data
            logger.warning(
                "models.dev disk cache is corrupt or empty; "
                "quarantining (will refetch from network)"
            )
            _quarantine_corrupt_cache(cache_path)
    except Exception as e:
        logger.warning("Failed to load models.dev disk cache; quarantining: %s", e)
        try:
            _quarantine_corrupt_cache(_get_cache_path())
        except Exception:
            pass
    return {}


def _quarantine_corrupt_cache(cache_path: Path) -> None:
    """Rename a rejected cache aside and drop its ETag sidecar.

    Renaming makes the rejection a one-time event — otherwise every hot-path
    call that finds the in-memory cache empty re-parses and re-warns until a
    network fetch succeeds. The sidecar vouches for a registry we no longer
    hold, so it goes too.
    """
    try:
        cache_path.rename(cache_path.with_suffix(".json.corrupt"))
    except Exception as e:
        logger.debug("Could not quarantine corrupt models.dev cache: %s", e)
    _clear_etag()


def _disk_cache_age_seconds() -> Optional[float]:
    """Age of the disk cache file in seconds, or None if missing/unreadable.

    An mtime in the future (clock skew) is also None — unknown freshness — so
    callers fall through to the network rather than trusting it forever.
    """
    try:
        cache_path = _get_cache_path()
        if not cache_path.exists():
            return None
        age = time.time() - cache_path.stat().st_mtime
        return age if age >= 0 else None
    except Exception as e:
        logger.debug("Failed to stat models.dev disk cache: %s", e)
        return None


def _save_disk_cache(data: Dict[str, Any], etag: str = "") -> None:
    """Save the registry atomically, plus the ETag sidecar when non-empty."""
    try:
        atomic_json_write(_get_cache_path(), data, indent=None, separators=(",", ":"))
    except Exception as e:
        logger.debug("Failed to save models.dev disk cache: %s", e)
    if etag:
        _save_etag(etag)


# --- Network refresh (all state mutation happens under _models_dev_fetch_lock)

class _NotModified(Exception):
    """Server returned 304 Not Modified — existing cache is still valid."""


def _fetch_models_dev_from_network(*, conditional: bool = False) -> Tuple[Dict[str, Any], str]:
    """Fetch the live registry; returns ``(registry, etag)`` (etag "" if none).

    ``conditional`` sends ``If-None-Match`` with the sidecar's ETag and raises
    ``_NotModified`` on 304. Pass True ONLY while holding
    ``_models_dev_fetch_lock`` AND a servable registry — a conditional request
    without one invites a 304 that leaves the process with no data.
    Raises on network errors and on an empty/invalid payload.
    """
    headers: Dict[str, str] = {}
    if conditional and (etag := _load_etag()):
        headers["If-None-Match"] = etag
    # (connect, read): 5 s connect fails fast on blackholed hosts instead of
    # stalling the first turn; 10 s read tolerates a slow registry.
    response = requests.get(_get_models_dev_url(), headers=headers, timeout=(5, 10))
    if response.status_code == 304:
        raise _NotModified()
    response.raise_for_status()
    data = response.json()
    if not _validate_registry(data):
        raise ValueError("models.dev returned an empty or invalid registry")
    return data, response.headers.get("ETag", "")


def _mark_stale_cache_grace() -> None:
    """Give stale cache data a 5-minute in-memory grace before retrying refresh.

    Only ever moves the timestamp forward, so a background refresh that
    completed between the caller's staleness check and this call keeps its
    fresh timestamp.
    """
    global _models_dev_cache_time
    grace_time = time.time() - _MODELS_DEV_CACHE_TTL + _MODELS_DEV_RETRY_DELAY
    if grace_time > _models_dev_cache_time:
        _models_dev_cache_time = grace_time


def _serve_stale(msg: str, *args: Any) -> Dict[str, Any]:
    """Arm the grace window, kick off a background refresh, return the held cache."""
    _mark_stale_cache_grace()
    _start_background_refresh_models_dev()
    logger.debug(msg, *args)
    return _models_dev_cache


def _commit_registry(data: Dict[str, Any], *, etag: str = "", where: str) -> None:
    """Persist a fetched registry: disk + in-mem + clear backoff.

    Callers hold ``_models_dev_fetch_lock`` so a failing refresh on one path
    can never stomp state a succeeding refresh on the other just committed.
    """
    global _models_dev_cache, _models_dev_cache_time, _models_dev_retry_after
    _save_disk_cache(data, etag)
    _models_dev_cache = data
    _models_dev_cache_time = time.time()
    _models_dev_retry_after = 0
    logger.debug(
        "Refreshed models.dev registry (%s): %d providers, %d total models", where, len(data),
        sum(len(p.get("models", {})) for p in data.values() if isinstance(p, dict)),
    )


def _confirm_cache_not_modified(*, where: str) -> None:
    """After a 304: clear backoff and re-mark the held cache fresh (disk is
    untouched — only the freshness marker advances). Caller holds the lock."""
    global _models_dev_cache_time, _models_dev_retry_after
    if not _models_dev_cache:
        # Should be unreachable (conditional GETs require a servable cache) but
        # previously caused a permanent empty-registry loop: drop the sidecar
        # and arm the normal backoff rather than marking {} "fresh".
        _clear_etag()
        _models_dev_retry_after = time.time() + _MODELS_DEV_RETRY_DELAY
        logger.warning(
            "models.dev returned 304 but no cached registry is held (%s); "
            "cleared ETag sidecar, will refetch unconditionally", where,
        )
        return
    _models_dev_cache_time = time.time()
    _models_dev_retry_after = 0
    logger.debug("models.dev registry unchanged (304 Not Modified, %s); cache re-confirmed fresh", where)


def _note_refresh_failure(exc: Exception, *, where: str) -> None:
    """Arm the process-wide 5-minute backoff. Caller holds the lock."""
    global _models_dev_retry_after
    _models_dev_retry_after = time.time() + _MODELS_DEV_RETRY_DELAY
    logger.debug(
        "models.dev refresh failed (%s); retry suppressed for %ds: %s", where, _MODELS_DEV_RETRY_DELAY, exc,
    )


def _background_refresh_models_dev() -> None:
    """Best-effort refresh after serving stale cache data."""
    global _models_dev_refresh_in_flight
    try:
        # Fetch INSIDE the lock, symmetric with the foreground path: the
        # conditional-GET inputs (memory cache + sidecar) can't be mutated
        # mid-fetch by a concurrent force_refresh and the two paths can't
        # double-download. Hot-path callers never touch this lock.
        with _models_dev_fetch_lock:
            data, etag = _fetch_models_dev_from_network(conditional=bool(_models_dev_cache))
            _commit_registry(data, etag=etag, where="background")
    except _NotModified:
        with _models_dev_fetch_lock:
            _confirm_cache_not_modified(where="background")
    except Exception as e:
        with _models_dev_fetch_lock:
            _note_refresh_failure(e, where="background")
    finally:
        with _models_dev_refresh_lock:
            _models_dev_refresh_in_flight = False


def _start_background_refresh_models_dev() -> None:
    """Start one daemon refresh worker if none is running and the failure
    backoff has elapsed."""
    global _models_dev_refresh_in_flight
    if time.time() < _models_dev_retry_after:
        return
    with _models_dev_refresh_lock:
        if _models_dev_refresh_in_flight:
            return
        _models_dev_refresh_in_flight = True
    thread = threading.Thread(target=_background_refresh_models_dev, name="models-dev-refresh", daemon=True)
    try:
        thread.start()
    except Exception as e:
        # Thread/fd exhaustion: clear the flag so refresh isn't disabled for
        # the rest of the process. Callers still get stale data.
        with _models_dev_refresh_lock:
            _models_dev_refresh_in_flight = False
        logger.debug("Failed to start models.dev refresh thread: %s", e)


def fetch_models_dev(force_refresh: bool = False, *, allow_network: bool = True) -> Dict[str, Any]:
    """Fetch the models.dev registry (dict keyed by provider ID; {} on failure).

    Cache hierarchy when ``force_refresh=False``: fresh in-memory → stale
    in-memory (returned immediately, refreshed in one background daemon thread
    — stale data beats a foreground timeout) → disk cache of any age (a stale
    one triggers the same background refresh; corrupt/empty is rejected) →
    singleflight foreground network fetch. Any failed refresh suppresses
    automatic refreshes for 5 minutes.

    ``force_refresh=True`` (``hermes config refresh``) bypasses the cache fast
    paths and the backoff, falling back to cached data only if the call fails.
    ``allow_network=False`` returns any memory/disk cache regardless of age and
    never makes a request — for latency-sensitive paths.
    """
    global _models_dev_cache, _models_dev_cache_time, _models_dev_retry_after

    if not allow_network:
        if not _models_dev_cache and (disk_data := _load_disk_cache()):
            _models_dev_cache = disk_data
            disk_age = _disk_cache_age_seconds()
            _models_dev_cache_time = time.time() - disk_age if disk_age is not None else 0
        return _models_dev_cache

    if not force_refresh:
        # Stage 1: fresh in-memory cache — the hot path, no I/O.
        if _models_dev_cache and (time.time() - _models_dev_cache_time) < _MODELS_DEV_CACHE_TTL:
            return _models_dev_cache
        # Stage 2: stale in-memory cache beats blocking on the network.
        if _models_dev_cache:
            return _serve_stale("Using stale in-memory models.dev cache; refreshing in background")
        # Stage 3: disk cache (cold-start only). A stale disk cache is
        # deliberately usable so resolution doesn't hang when models.dev is
        # unreachable.
        disk_age = _disk_cache_age_seconds()
        if disk_age is not None and (disk_data := _load_disk_cache()):
            _models_dev_cache = disk_data
            if disk_age >= _MODELS_DEV_CACHE_TTL:
                return _serve_stale(
                    "Using stale models.dev disk cache (age=%.0fs); refreshing in background", disk_age,
                )
            # Anchor the in-mem TTL to the file's age so an aging cache isn't
            # extended by another full TTL.
            _models_dev_cache_time = time.time() - disk_age
            logger.debug(
                "Loaded models.dev from fresh disk cache (%d providers, age=%.0fs)", len(disk_data), disk_age,
            )
            return _models_dev_cache
        # Process-wide backoff: don't make every caller retry an unreachable
        # endpoint while no usable cache exists.
        if time.time() < _models_dev_retry_after:
            return _models_dev_cache

    # Stage 4: singleflight foreground fetch. Recheck state under the lock —
    # another caller may have refreshed or armed the backoff while we waited.
    with _models_dev_fetch_lock:
        if not force_refresh and (_models_dev_cache or time.time() < _models_dev_retry_after):
            return _models_dev_cache
        # Cold force_refresh: stages 1-3 were skipped, so hydrate memory from
        # disk first so the conditional GET fires and a 304 can re-confirm it.
        if force_refresh and not _models_dev_cache and (disk := _load_disk_cache()):
            _models_dev_cache = disk
            _models_dev_cache_time = 0  # servable but not fresh
        try:
            data, etag = _fetch_models_dev_from_network(conditional=bool(_models_dev_cache))
            _commit_registry(data, etag=etag, where="foreground")
            return data
        except _NotModified:
            _confirm_cache_not_modified(where="foreground")
            return _models_dev_cache
        except Exception as e:
            _note_refresh_failure(e, where="foreground")
        # Stage 5: network failed — serve any stale memory/disk cache. Freshness
        # stays expired; the retry-after timestamp gates the next attempt.
        if not _models_dev_cache:
            _models_dev_cache = _load_disk_cache()
            _models_dev_cache_time = 0
            if _models_dev_cache:
                logger.debug("Loaded stale models.dev disk cache (%d providers)", len(_models_dev_cache))
        return _models_dev_cache


# --- Catalog access helpers ------------------------------------------------

def _fetch_registry(allow_network: bool) -> Dict[str, Any]:
    # Keep the zero-argument call on the allow_network path: dozens of test
    # sites monkeypatch fetch_models_dev with zero-arg lambdas.
    return fetch_models_dev() if allow_network else fetch_models_dev(allow_network=False)


def _registry_provider(mdev_id: str, allow_network: bool) -> Optional[Dict[str, Any]]:
    """The raw models.dev provider entry, or None."""
    provider_data = _fetch_registry(allow_network).get(mdev_id)
    return provider_data if isinstance(provider_data, dict) else None


def _registry_models(mdev_id: str, *, allow_network: bool) -> Optional[Dict[str, Any]]:
    """The ``models`` dict of a models.dev provider entry, or None."""
    provider_data = _registry_provider(mdev_id, allow_network)
    if provider_data is None:
        return None
    models = provider_data.get("models", {})
    return models if isinstance(models, dict) else None


def _get_provider_models(provider: str, *, allow_network: bool = False) -> Optional[Dict[str, Any]]:
    """Resolve a Hermes provider ID to its models dict, or None if unknown.
    ``allow_network`` defaults to False — hot-path callers must never block."""
    mdev_provider_id = PROVIDER_TO_MODELS_DEV.get(provider)
    if not mdev_provider_id:
        return None
    return _registry_models(mdev_provider_id, allow_network=allow_network)


def _iter_model_entries(models: Dict[str, Any], model: str, *, suffix_fallback: bool = True):
    """Yield ``(model_id, entry)`` candidates: exact, case-insensitive, then
    (optionally) ``:cloud``/``-cloud`` suffixed forms.

    The suffix fallback exists because some providers (e.g. ollama-cloud)
    store ``kimi-k2.6:cloud`` while the live API returns the bare name;
    without it context lookup falls through to stale OpenRouter metadata and
    trips the 64k minimum-context guard. Every consumer shares this order so a
    suffix-keyed catalog model counts as KNOWN for ``model_overrides``
    fill-gap ``_default`` semantics.
    """
    candidates = [model]
    if suffix_fallback:
        candidates += [model + suffix for suffix in (":cloud", "-cloud")]
    for name in candidates:
        entry = models.get(name)
        if isinstance(entry, dict):
            yield name, entry
        name_lower = name.lower()
        for mid, mdata in models.items():
            if mid.lower() == name_lower and isinstance(mdata, dict):
                yield mid, mdata


def _find_model_entry(models: Dict[str, Any], model: str) -> Optional[Dict[str, Any]]:
    """First catalog entry for *model* (exact, case-insensitive, suffix), or None."""
    return next((entry for _mid, entry in _iter_model_entries(models, model)), None)


def _extract_limit(entry: Any, key: str) -> Optional[int]:
    """Positive int ``entry.limit[key]`` or None (audio/image models have context=0)."""
    value = _dict_or_empty(_dict_or_empty(entry).get("limit")).get(key)
    return int(value) if isinstance(value, (int, float)) and value > 0 else None


def _extract_context(entry: Dict[str, Any]) -> Optional[int]:
    """Context length from a models.dev model entry, or None if invalid/zero."""
    return _extract_limit(entry, "context")


def lookup_models_dev_context(provider: str, model: str, *, allow_network: bool = False) -> Optional[int]:
    """Context window in tokens for provider+model, or None if not found.

    An EXPLICIT ``model_overrides`` entry wins over the catalog; ``_default``
    entries fill the gap only when the catalog has no answer (the supported
    self-unblock path for models with wrong/missing context in models.dev).
    Catalog entries with context=0 are skipped in favour of later candidates.
    ``allow_network`` defaults to False — this runs every turn and must never
    block.
    """
    override_ctx = _override_context_window(provider, model)
    if override_ctx is not None:
        return override_ctx
    models = _get_provider_models(provider, allow_network=allow_network)
    if models is not None:
        for _mid, entry in _iter_model_entries(models, model):
            ctx = _extract_context(entry)
            if ctx:
                return ctx
    return _default_override_context(provider)


# --- Per-model metadata overrides (config.yaml → model_overrides) ----------
#
# Canonical override schema (the ONLY key space consumers accept):
#   context_window, max_output_tokens, supports_tools, supports_vision,
#   supports_reasoning, model_family
#
# ``model_overrides.<provider>.<model_id>`` is an explicit override that always
# wins over the catalog for the fields it sets (partial patch).
# ``model_overrides.<provider>._default`` / ``model_overrides._default`` are
# FILL-GAP defaults: they apply ONLY to models the catalog does not know and
# never displace catalog data — a ``_default: {context_window: 128000}`` cannot
# clamp every catalog-known model of a provider.
#
# Provider keys accept the Hermes provider id or the models.dev provider id.
# Model ids match exactly, then case-insensitively (mirroring catalog lookup).

_OVERRIDE_WARNED_KEYS: set = set()

# Safe defaults for models absent from the catalog (tools on, vision/reasoning
# off, 200K context); shared by get_model_capabilities and get_model_info so
# the two unknown-model paths agree.
_UNKNOWN_MODEL_BASE: Dict[str, Any] = {
    "limit": {"context": 200000, "output": 8192},
    "tool_call": True,
}


def _load_model_overrides() -> Dict[str, Any]:
    """The ``model_overrides`` config section ({} on any failure).

    Deliberately not memoized: ``load_config_readonly()`` is already
    (mtime, size)-cached upstream, and an ``id(cfg)``-keyed layer here can
    serve stale overrides after a reload when CPython reuses the dict address.
    """
    return _dict_or_empty(_cfg_get("model_overrides", default={}))


def _provider_override_section(provider: str) -> Optional[Dict[str, Any]]:
    """Override section for *provider* (keyed by Hermes OR models.dev id), or None."""
    overrides = _load_model_overrides()
    provider_key = (provider or "").strip()
    if not overrides or not provider_key:
        return None
    # Forward (Hermes → models.dev id) and reverse (caller passed a models.dev
    # id, config keyed by Hermes id) aliases.
    candidates = [provider_key, PROVIDER_TO_MODELS_DEV.get(provider_key)]
    candidates += _models_dev_to_hermes_ids(provider_key)
    for key in candidates:
        section = overrides.get(key) if key else None
        if isinstance(section, dict):
            return section
    return None


def _explicit_model_override(provider: str, model: str) -> Optional[Dict[str, Any]]:
    """Explicit per-provider+model override dict (exact, then case-insensitive
    skipping the ``_default`` sentinel), or None."""
    model_key = (model or "").strip()
    section = _provider_override_section(provider) if model_key else None
    if section is None:
        return None
    entry = section.get(model_key)
    if isinstance(entry, dict):
        return entry
    model_lower = model_key.lower()
    for mid, mdata in section.items():
        if mid != "_default" and mid.lower() == model_lower and isinstance(mdata, dict):
            return mdata
    return None


def _default_model_override(provider: str) -> Optional[Dict[str, Any]]:
    """Fill-gap ``_default`` override: per-provider first, then global; or None."""
    section = _provider_override_section(provider)
    if section is not None and isinstance(section.get("_default"), dict):
        return section["_default"]
    global_default = _load_model_overrides().get("_default")
    return global_default if isinstance(global_default, dict) else None


def _override_for(provider: str, model: str, *, catalog_hit: bool) -> Optional[Dict[str, Any]]:
    """Explicit override if any; else the ``_default`` only on a catalog miss."""
    explicit = _explicit_model_override(provider, model)
    if explicit is not None or catalog_hit:
        return explicit
    return _default_model_override(provider)


def _override_int(override: Dict[str, Any], key: str) -> Optional[int]:
    """Coerce an override field to a positive int, warning once on garbage."""
    raw = override.get(key)
    if raw is None:
        return None
    try:
        value = int(raw)
        if value > 0:
            return value
    except (TypeError, ValueError):
        pass
    warn_key = (key, repr(raw))
    if warn_key not in _OVERRIDE_WARNED_KEYS:
        _OVERRIDE_WARNED_KEYS.add(warn_key)
        logger.warning(
            "model_overrides: ignoring invalid %s value %r (expected a positive integer)", key, raw,
        )
    return None


def _override_context_window(provider: str, model: str) -> Optional[int]:
    """EXPLICITLY overridden context_window, or None.

    Explicit-only on purpose: this runs early in the resolution chain
    (agent/model_metadata.py, before custom_providers and live probes) where
    a ``_default`` must not preempt more specific sources; fill-gap defaults
    apply later in ``lookup_models_dev_context`` once the catalog has missed.
    """
    ov = _explicit_model_override(provider, model)
    return _override_int(ov, "context_window") if ov is not None else None


def _default_override_context(provider: str) -> Optional[int]:
    """Fill-gap context from a ``_default`` override, for catalog misses."""
    default = _default_model_override(provider)
    return _override_int(default, "context_window") if default is not None else None


def _override_to_catalog_shape(override: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[bool]]:
    """Translate canonical override keys into a models.dev-shaped patch.
    Returns ``(patch, vision)`` — vision is out-of-band because it maps onto the
    ``modalities.input`` list rather than a scalar field."""
    patch: Dict[str, Any] = {}
    limit = {
        catalog_key: value
        for catalog_key, override_key in (("context", "context_window"), ("output", "max_output_tokens"))
        if (value := _override_int(override, override_key)) is not None
    }
    if limit:
        patch["limit"] = limit
    for override_key, catalog_key in (("supports_tools", "tool_call"), ("supports_reasoning", "reasoning")):
        if override_key in override:
            patch[catalog_key] = bool(override[override_key])
    vision: Optional[bool] = None
    if "supports_vision" in override:
        vision = patch["attachment"] = bool(override["supports_vision"])
    if "model_family" in override:
        patch["family"] = str(override["model_family"] or "")
    return patch, vision


def _merge_catalog_entry_with_override(raw: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Patch a catalog entry with a canonical-schema override. Sub-dicts
    (``limit``, ``modalities``) are merged, not clobbered — setting only
    ``context_window`` must not wipe the catalog's ``limit.output``."""
    shaped, vision_override = _override_to_catalog_shape(override)
    merged = dict(raw)
    limit_patch = shaped.pop("limit", None)
    if limit_patch:
        merged["limit"] = {**_dict_or_empty(raw.get("limit")), **limit_patch}
    if vision_override is not None:
        base_mods = dict(_dict_or_empty(raw.get("modalities")))
        input_mods = base_mods.get("input")
        input_mods = list(input_mods) if isinstance(input_mods, list) else []
        if vision_override and "image" not in input_mods:
            input_mods.append("image")
        elif not vision_override and "image" in input_mods:
            input_mods.remove("image")
        base_mods["input"] = input_mods
        merged["modalities"] = base_mods
    merged.update(shaped)
    return merged


def _apply_overrides(provider: str, model: str, entry: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Catalog *entry* patched by its override; ``_UNKNOWN_MODEL_BASE`` patched
    by a fill-gap override on a catalog miss; None when neither exists.
    The override is selected AFTER the catalog lookup: _default only fills misses."""
    override = _override_for(provider, model, catalog_hit=entry is not None)
    if override is None:
        return entry
    return _merge_catalog_entry_with_override(entry if entry is not None else _UNKNOWN_MODEL_BASE, override)


# --- Model capability metadata ---------------------------------------------

def _entry_supports_vision(entry: Dict[str, Any]) -> bool:
    """Prefer explicit ``modalities.input`` (the older ``attachment`` flag can be
    stale or too broad for image routing); fall back to it only when the input
    modalities are absent/invalid."""
    input_mods = _dict_or_empty(entry.get("modalities", {})).get("input")
    if isinstance(input_mods, list):
        return "image" in input_mods
    return bool(entry.get("attachment", False))


def get_model_capabilities(provider: str, model: str, *, allow_network: bool = False) -> Optional[ModelCapabilities]:
    """Capability metadata from the models.dev cache, or None if unresolvable.

    EXPLICIT ``model_overrides`` entries patch catalog values for the fields
    they set; ``_default`` entries fill the gap only for models the catalog
    does not know. Unspecified fields fall through to the catalog value, or to
    safe defaults (tools on, vision/reasoning off, 200K/8K) when absent.
    ``allow_network`` defaults to False — vision/image routing is a hot path.
    """
    models = _get_provider_models(provider, allow_network=allow_network)
    entry = _find_model_entry(models, model) if models is not None else None
    raw = _apply_overrides(provider, model, entry)
    if raw is None:
        return None
    return ModelCapabilities(
        supports_tools=bool(raw.get("tool_call", False)),
        supports_vision=_entry_supports_vision(raw),
        supports_reasoning=bool(raw.get("reasoning", False)),
        context_window=_extract_limit(raw, "context") or 200000,
        max_output_tokens=_extract_limit(raw, "output") or 8192,
        model_family=raw.get("family", "") or "",
    )


def list_provider_models(provider: str, *, allow_network: bool = True) -> List[str]:
    """All model IDs for a provider ([] if unknown). ``allow_network`` defaults
    to True: the model picker is interactive and a fresh catalog is worth a
    short wait."""
    from hermes_cli.models import normalize_provider
    provider = normalize_provider(provider) or provider
    models = _get_provider_models(provider, allow_network=allow_network)
    if models is None:
        return []
    return [mid for mid in models if not _should_hide_from_provider_catalog(provider, mid)]


# Non-agentic or noise models (TTS, embedding, dated preview snapshots,
# live/streaming-only, image-only).
_NOISE_PATTERNS: re.Pattern = re.compile(
    r"-tts\b|embedding|live-|-(preview|exp)-\d{2,4}[-_]|"
    r"-image\b|-image-preview\b|-customtools\b",
    re.IGNORECASE,
)

# Hidden from the Gemini catalogs surfaced in setup/model selection (capability
# metadata stays available for direct/manual use).
_GOOGLE_HIDDEN_MODELS = frozenset({
    # Low-TPM Gemma models that trip Google input-token quota walls under
    # agent-style traffic despite advertising large context windows.
    "gemma-4-31b-it", "gemma-4-26b-it", "gemma-4-26b-a4b-it",
    "gemma-3-1b", "gemma-3-1b-it", "gemma-3-2b", "gemma-3-2b-it",
    "gemma-3-4b", "gemma-3-4b-it", "gemma-3-12b", "gemma-3-12b-it",
    "gemma-3-27b", "gemma-3-27b-it",
    # Stale/retired Google slugs that 404 on the current endpoints.
    "gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.5-flash-8b",
    "gemini-2.0-flash", "gemini-2.0-flash-lite",
})


def _should_hide_from_provider_catalog(provider: str, model_id: str) -> bool:
    provider_lower = (provider or "").strip().lower()
    model_lower = (model_id or "").strip().lower()
    return provider_lower in {"gemini", "google"} and model_lower in _GOOGLE_HIDDEN_MODELS


def list_agentic_models(provider: str, *, allow_network: bool = True) -> List[str]:
    """Model IDs suitable for agentic use: tool_call=True, minus hidden and
    noise models. [] on any failure. ``allow_network`` defaults to True (called
    from interactive model selection)."""
    models = _get_provider_models(provider, allow_network=allow_network)
    if models is None:
        return []
    return [
        mid for mid, entry in models.items()
        if isinstance(entry, dict)
        and not _should_hide_from_provider_catalog(provider, mid)
        and entry.get("tool_call", False)
        and not _NOISE_PATTERNS.search(mid)
    ]


# --- Rich dataclass constructors + queries ---------------------------------

def _parse_model_info(model_id: str, raw: Dict[str, Any], provider_id: str) -> ModelInfo:
    """Convert a raw models.dev model entry dict into a ModelInfo dataclass."""
    cost = _dict_or_empty(raw.get("cost"))
    modalities = _dict_or_empty(raw.get("modalities"))
    input_mods = modalities.get("input") or []
    output_mods = modalities.get("output") or []

    def _cost(key: str) -> Optional[float]:
        return float(cost[key]) if key in cost and cost[key] is not None else None

    return ModelInfo(
        id=model_id,
        name=raw.get("name", "") or model_id,
        family=raw.get("family", "") or "",
        provider_id=provider_id,
        reasoning=bool(raw.get("reasoning", False)),
        tool_call=bool(raw.get("tool_call", False)),
        attachment=bool(raw.get("attachment", False)),
        temperature=bool(raw.get("temperature", False)),
        structured_output=bool(raw.get("structured_output", False)),
        open_weights=bool(raw.get("open_weights", False)),
        input_modalities=tuple(input_mods) if isinstance(input_mods, list) else (),
        output_modalities=tuple(output_mods) if isinstance(output_mods, list) else (),
        context_window=_extract_limit(raw, "context") or 0,
        max_output=_extract_limit(raw, "output") or 0,
        max_input=_extract_limit(raw, "input"),
        cost_input=float(cost.get("input", 0) or 0),
        cost_output=float(cost.get("output", 0) or 0),
        cost_cache_read=_cost("cache_read"),
        cost_cache_write=_cost("cache_write"),
        knowledge_cutoff=raw.get("knowledge", "") or "",
        release_date=raw.get("release_date", "") or "",
        status=raw.get("status", "") or "",
        interleaved=raw.get("interleaved", False),
    )


def _parse_provider_info(provider_id: str, raw: Dict[str, Any]) -> ProviderInfo:
    """Convert a raw models.dev provider entry dict into a ProviderInfo."""
    env = raw.get("env") or []
    models = raw.get("models") or {}
    return ProviderInfo(
        id=provider_id,
        name=raw.get("name", "") or provider_id,
        env=tuple(env) if isinstance(env, list) else (),
        api=raw.get("api", "") or "",
        doc=raw.get("doc", "") or "",
        model_count=len(models) if isinstance(models, dict) else 0,
    )


def get_provider_info(provider_id: str, *, allow_network: bool = True) -> Optional[ProviderInfo]:
    """Provider metadata by Hermes or models.dev ID, or None if not cataloged.
    ``allow_network`` defaults to True — the primary caller is interactive
    setup (``resolve_provider_full``). Hot-path callers pass False."""
    mdev_id = PROVIDER_TO_MODELS_DEV.get(provider_id, provider_id)
    raw = _registry_provider(mdev_id, allow_network)
    return _parse_provider_info(mdev_id, raw) if raw is not None else None


def get_model_info(provider_id: str, model_id: str, *, allow_network: bool = False) -> Optional[ModelInfo]:
    """Full model metadata by Hermes or models.dev provider ID (exact match,
    then case-insensitive), or None if not found.

    ``model_overrides`` use the same canonical schema as every other consumer:
    EXPLICIT entries patch known catalog models; ``_default`` entries fill the
    gap only for models the catalog does not know.
    ``allow_network`` defaults to False — cost guard and inventory are hot paths.
    """
    mdev_id = PROVIDER_TO_MODELS_DEV.get(provider_id, provider_id)
    models = _registry_models(mdev_id, allow_network=allow_network)
    mid, entry = model_id, None
    if models is not None:
        mid, entry = next(_iter_model_entries(models, model_id, suffix_fallback=False), (model_id, None))
    # Not in catalog — an override (explicit or _default) may still provide it.
    raw = _apply_overrides(provider_id, model_id, entry)
    return _parse_model_info(mid, raw, mdev_id) if raw is not None else None
