"""Per-model reasoning capabilities from OpenRouter-schema ``/v1/models`` catalogs.

Split out of ``hermes_cli.models``; every public/patched name is re-imported there. The
OpenRouter and Nous Portal catalogs share one implementation parametrized by
:class:`_CapsSource`; the per-source module globals (``_openrouter_reasoning_caps_cache``,
``_nous_caps_disk_checked``, ...) stay defined on ``hermes_cli.models`` — tests reset them there —
and are read/written by attribute name through the origin module.

Tri-state contract for callers deciding whether to emit reasoning controls:
- dict with ``supports_reasoning: True`` (+ ``supported_efforts``, ``mandatory``) — the route
  advertises reasoning controls;
- dict with ``supports_reasoning: False`` — the catalog knows the model and it does NOT accept
  reasoning controls (definitive negative);
- ``None`` — unknown: catalog not loaded, model not listed (private/custom route), malformed.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

from utils import atomic_json_write

logger = logging.getLogger("hermes_cli.models")

Caps = dict[str, Optional[dict[str, Any]]]


def _origin():
    from hermes_cli import models

    return models


def parse_openrouter_reasoning_capabilities(item: Any) -> Optional[dict[str, Any]]:
    """Normalize one OpenRouter catalog entry's reasoning metadata.

    ``supported_parameters`` contains ``"reasoning"`` when the route accepts reasoning controls at
    all; a top-level ``reasoning`` object may add detail (``mandatory``, ``supported_efforts``).
    A missing/malformed ``supported_parameters`` is "unknown" (None), mirroring the permissive
    stance of ``_openrouter_model_supports_tools``.
    """
    if not isinstance(item, dict):
        return None
    params = item.get("supported_parameters")
    if not isinstance(params, list):
        return None
    if "reasoning" not in params:
        return {"supports_reasoning": False}
    reasoning = item.get("reasoning")
    mandatory = isinstance(reasoning, dict) and reasoning.get("mandatory") is True
    efforts: Optional[list[str]] = None
    if isinstance(reasoning, dict):
        raw_efforts = reasoning.get("supported_efforts")
        if isinstance(raw_efforts, list):
            efforts = list(dict.fromkeys(
                str(effort).strip().lower()
                for effort in raw_efforts
                if str(effort).strip()
            ))
    return {
        "supports_reasoning": True,
        "supported_efforts": efforts,
        "mandatory": mandatory,
    }


# ── Disk mirror ────────────────────────────────────────────────────────
#
# The in-process caches are always cold in a short-lived process, and every consumer is on a hot
# path that must never block on HTTP — so without a disk copy, `hermes -p`, a cron job, or a
# freshly booted gateway answers "capability unknown" for its whole first turn and falls back to
# the conservative wire shape. One file holds every catalog, keyed by the URL it came from:
# OpenRouter and the Nous Portal list different models, and a staging Portal must not answer for
# production.
_REASONING_CAPS_DISK_TTL_SECONDS = 24 * 3600


def _reasoning_caps_disk_path() -> Path:
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "cache" / "reasoning_caps.json"


def _read_reasoning_caps_disk() -> dict[str, Any]:
    try:
        with _reasoning_caps_disk_path().open(encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def _load_reasoning_caps_disk(url: str) -> tuple[Optional[Caps], float]:
    """Return ``(caps, age_seconds)`` for *url*, or ``(None, 0.0)``."""
    entry = _origin()._read_reasoning_caps_disk().get(url)
    if not isinstance(entry, dict):
        return None, 0.0
    caps = entry.get("caps")
    if not isinstance(caps, dict) or not caps:
        return None, 0.0
    try:
        age = max(0.0, time.time() - float(entry.get("ts") or 0))
    except (TypeError, ValueError):
        age = float(_REASONING_CAPS_DISK_TTL_SECONDS)
    return {str(mid): model_caps for mid, model_caps in caps.items()}, age


def _save_reasoning_caps_disk(url: str, caps: Caps) -> None:
    """Merge *url*'s catalog into the shared disk mirror, atomically."""
    try:
        data = _origin()._read_reasoning_caps_disk()
        data[url] = {"ts": time.time(), "caps": caps}
        path = _reasoning_caps_disk_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_json_write(path, data, indent=0, separators=(",", ":"))
    except Exception as exc:
        logger.debug("Failed to save reasoning-caps disk cache: %s", exc)


def _warm_reasoning_caps_async(refresh) -> None:
    """Run *refresh* in a background thread. Fire-and-forget.

    Called from hot paths that found the cache cold or the disk copy stale, so the next call — or,
    via the disk mirror, the next process — benefits without this turn ever blocking on HTTP.
    Callers own the once-per-process guard; the fetch keeps its own failure TTL.
    """
    if os.environ.get("PYTEST_CURRENT_TEST"):
        return
    threading.Thread(target=refresh, name="reasoning-caps-warm", daemon=True).start()


def _hydrate_reasoning_caps_from_disk(url: str, refresh) -> Optional[Caps]:
    """The disk copy of *url*'s catalog, queueing *refresh* when it's stale.

    A copy past its TTL is still returned — a stale verdict beats no verdict, and reasoning
    capabilities change rarely — with a background refresh so the next run is current.
    """
    caps, age = _load_reasoning_caps_disk(url)
    if caps is None:
        return None
    if age >= _REASONING_CAPS_DISK_TTL_SECONDS:
        _warm_reasoning_caps_async(refresh)
    return caps


def _seed_reasoning_caps(url: str, items: Any) -> Optional[Caps]:
    """Parse a ``/v1/models`` ``data`` array and mirror it for *url*.

    Takes the payload rather than fetching it, so picker and pricing fetches (which pull the same
    document) leave the mirror warm at no network cost. Returns None when the array has no usable
    entries, which callers remember as a failure rather than caching as empty.
    """
    if not isinstance(items, list):
        return None
    caps_by_id: Caps = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        mid = str(item.get("id") or "").strip()
        if not mid:
            continue
        caps_by_id[mid] = parse_openrouter_reasoning_capabilities(item)
    if not caps_by_id:
        return None
    _save_reasoning_caps_disk(url, caps_by_id)
    return caps_by_id


def _fetch_reasoning_caps_catalog(url: str, timeout: float) -> Optional[Caps]:
    """Fetch one OpenRouter-shaped ``/v1/models`` catalog → per-model caps.

    Returns None when the catalog is unreachable or has no usable entries, so callers remember the
    failure and fall back rather than caching an empty result. Sends a User-Agent because the
    Portal 403s anonymous catalog reads.
    """
    m = _origin()
    headers = {"Accept": "application/json", "User-Agent": m._HERMES_USER_AGENT}
    try:
        req = urllib.request.Request(url, headers=headers)
        with m._urlopen_model_catalog_request(req, timeout=timeout) as resp:
            payload = json.loads(resp.read().decode())
    except Exception:
        return None
    return _seed_reasoning_caps(url, payload.get("data"))


# ── Per-source cache (OpenRouter, Nous Portal) ─────────────────────────

@dataclass(frozen=True)
class _CapsSource:
    """One catalog's cache slots on ``hermes_cli.models`` plus how to name its URL.

    ``cache``: model id → parsed caps, populated by one full-catalog fetch and kept for the process
    lifetime (capabilities don't change). ``failed_at``: monotonic timestamp of the last FAILED
    fetch; suppresses re-fetch storms from per-turn callers while the catalog is unreachable (60s,
    mirrors the LM Studio/Ollama capability-probe caching). ``disk_checked`` / ``warm_started``:
    once-per-process guards for the disk hydrate and the background warm.
    """
    cache: str
    failed_at: str
    disk_checked: str
    warm_started: str
    url: Callable[[], str]


def _fetch_caps(src: _CapsSource, timeout: float = 6.0, *, force: bool = False) -> Optional[Caps]:
    """Fetch + cache the source's per-model caps. None (without poisoning the cache) when
    unreachable, so callers retry later and fall back meanwhile."""
    m = _origin()
    cached = getattr(m, src.cache)
    if cached is not None and not force:
        return cached
    failed_at = getattr(m, src.failed_at)
    if failed_at is not None and (time.monotonic() - failed_at) < 60:
        return None
    caps_by_id = _fetch_reasoning_caps_catalog(src.url(), timeout)
    if caps_by_id is None:
        setattr(m, src.failed_at, time.monotonic())
        return None
    setattr(m, src.cache, caps_by_id)
    return caps_by_id


def _caps_cached(src: _CapsSource) -> Optional[Caps]:
    """Cache-only caps: memory, else the disk mirror. Never HTTP.

    Guarded to one disk attempt per process: for the Portal, naming the catalog means resolving
    credentials, which can itself reach the network to refresh a token — far too expensive for a
    caller that runs every turn.
    """
    m = _origin()
    if getattr(m, src.cache) is None and not getattr(m, src.disk_checked):
        setattr(m, src.disk_checked, True)
        setattr(m, src.cache, _hydrate_reasoning_caps_from_disk(src.url(), lambda: _fetch_caps(src, force=True)))
    return getattr(m, src.cache)


def _model_caps(src: _CapsSource, model_id: Optional[str], *, timeout: float, allow_fetch: bool) -> Optional[dict[str, Any]]:
    model = str(model_id or "").strip()
    if not model:
        return None
    caps_by_id = _caps_cached(src)
    if caps_by_id is None and allow_fetch:
        caps_by_id = _fetch_caps(src, timeout=timeout)
    if caps_by_id is None:
        return None
    return caps_by_id.get(model)


def _warm_caps_async(src: _CapsSource) -> None:
    m = _origin()
    if getattr(m, src.warm_started) or _caps_cached(src) is not None:
        return
    setattr(m, src.warm_started, True)
    _warm_reasoning_caps_async(lambda: _fetch_caps(src, force=True))


_OPENROUTER_CATALOG_URL = "https://openrouter.ai/api/v1/models"

_OPENROUTER_CAPS = _CapsSource(
    "_openrouter_reasoning_caps_cache", "_openrouter_reasoning_caps_failed_at",
    "_openrouter_caps_disk_checked", "_openrouter_caps_warm_started",
    lambda: _OPENROUTER_CATALOG_URL,
)
# Nous Portal serves OpenRouter's catalog schema, so the same parser and contract apply. Its own
# cache because the two catalogs list different models (and different capabilities for shared ids).
_NOUS_CAPS = _CapsSource(
    "_nous_reasoning_caps_cache", "_nous_reasoning_caps_failed_at",
    "_nous_caps_disk_checked", "_nous_caps_warm_started",
    lambda: _origin().nous_catalog_url(),
)


def nous_catalog_url() -> str:
    """The Portal ``/v1/models`` URL for the endpoint we actually talk to.

    Resolved through the ladder ``NOUS_INFERENCE_BASE_URL`` → resolved credential base → prod
    rather than pinned to production, so a staging profile reads staging's capabilities.
    """
    return f"{_origin()._resolve_nous_pricing_credentials()[1]}/v1/models"


def openrouter_model_reasoning_capabilities(
    model_id: Optional[str], *, timeout: float = 6.0, allow_fetch: bool = False,
) -> Optional[dict[str, Any]]:
    """Live-catalog reasoning capabilities for an OpenRouter model (tri-state, see module doc).

    CACHE-ONLY by default — safe on per-request hot paths (never blocks on HTTP)."""
    return _model_caps(_OPENROUTER_CAPS, model_id, timeout=timeout, allow_fetch=allow_fetch)


def nous_model_reasoning_capabilities(
    model_id: Optional[str], *, timeout: float = 6.0, allow_fetch: bool = False,
) -> Optional[dict[str, Any]]:
    """Nous Portal counterpart of :func:`openrouter_model_reasoning_capabilities`; warm the cache
    with :func:`warm_nous_reasoning_caps_async` from hot paths."""
    return _model_caps(_NOUS_CAPS, model_id, timeout=timeout, allow_fetch=allow_fetch)


def warm_openrouter_reasoning_caps_async() -> None:
    """Warm the OpenRouter reasoning-capability cache in the background."""
    _warm_caps_async(_OPENROUTER_CAPS)


def warm_nous_reasoning_caps_async() -> None:
    """Nous Portal counterpart of :func:`warm_openrouter_reasoning_caps_async`."""
    _warm_caps_async(_NOUS_CAPS)
