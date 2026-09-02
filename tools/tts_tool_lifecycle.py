"""Local-engine lifecycle for ``tools.tts_tool``: warm-up / release leases.

Local engines load their model lazily on first synthesis, so the first spoken
reply after a user turns speech output on pays the whole load as dead air,
and the model then stays resident forever. The toggles ARE the intent
signal: every surface that flips speech output on holds a *lease* here
(warming the configured engine); when the last lease is released the local
model caches are dropped. Lease-counting keeps one surface's "off" from
unloading a model another surface in this process still needs. Cloud
providers have nothing resident; warming them only ensures the lazily
installed SDK is importable.

Seams tests monkeypatch on the origin (``_load_tts_config``, ``_get_provider``,
``warm_tts_provider``, ``_run_command_tts``) are resolved through
:func:`_origin` at call time.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Callable, Dict, List, Optional

from tools.tts_command_provider import (
    BUILTIN_TTS_PROVIDERS,
    _get_command_tts_timeout,
    _get_named_provider_config,
    _is_command_provider_config,
    command_env_passthrough as _command_provider_env_passthrough,
    render_command_template as _render_command_tts_template,
)
from tools.tts_tool_local import _LOCAL_TTS_MODEL_CACHES
from tools.tts_tool_plugins import _lookup_plugin_provider

logger = logging.getLogger("tools.tts_tool")


def _origin():
    """``tools.tts_tool``, resolved per call so monkeypatched seams there still apply."""
    from tools import tts_tool

    return tts_tool


_tts_lease_lock = threading.Lock()
_tts_leases: set = set()


def _local_tts_warmers() -> Dict[str, Callable[[Dict[str, Any]], Any]]:
    """Provider name → loader populating that engine's cache slot (same key synthesis uses)."""
    return {
        "piper": lambda cfg: _origin()._load_piper_voice_for_config(cfg)[0],
        "kittentts": lambda cfg: _origin()._load_kittentts_model_for_config(cfg)[0],
    }


def _lazy_sdk_feature_for_provider(provider: str) -> Optional[str]:
    """tools.lazy_deps feature key for providers whose SDK installs on first use."""
    return {
        "edge": "tts.edge",
        "elevenlabs": "tts.elevenlabs",
        "mistral": "tts.mistral",
    }.get(provider)


def _signal_user_tts_provider(name: str, tts_config: Dict[str, Any], hook: str) -> Optional[str]:
    """Forward a lease ``hook`` (``"warm"`` / ``"release"``) to a user-declared provider.

    Command providers run their optional ``warm_command`` / ``release_command``
    (same template/env/timeout rules as ``command``; output discarded) on a
    background thread so a toggle never waits on a model server. Plugin
    providers get :meth:`TTSProvider.warm` / :meth:`TTSProvider.release`.
    Best-effort: failures are logged at debug. Returns the action taken.
    """
    if not name or name in BUILTIN_TTS_PROVIDERS:
        return None
    cfg = _get_named_provider_config(tts_config, name)
    try:
        if _is_command_provider_config(cfg):
            template = str(cfg.get(f"{hook}_command") or "").strip()
            if not template:
                return None
            command = _render_command_tts_template(template, {
                "voice": str(cfg.get("voice", "")),
                "model": str(cfg.get("model", "")),
                "speed": str(cfg.get("speed", tts_config.get("speed", ""))),
            })

            def _run() -> None:
                try:
                    _origin()._run_command_tts(command, _get_command_tts_timeout(cfg),
                                     env_passthrough=_command_provider_env_passthrough(cfg))
                except Exception as exc:  # noqa: BLE001 — best-effort hook
                    logger.debug("[TTS] %s_command for %s failed: %s", hook, name, exc)

            threading.Thread(target=_run, name=f"tts-{hook}-{name}", daemon=True).start()
            return hook
        plugin_provider = _lookup_plugin_provider(name)
        if plugin_provider is None:
            return None
        getattr(plugin_provider, hook)()
        return hook
    except Exception as exc:  # noqa: BLE001 — best-effort hook
        logger.debug("[TTS] %s hook for %s failed: %s", hook, name, exc)
        return "error"


def warm_tts_provider(
    tts_config: Optional[Dict[str, Any]] = None,
    provider: Optional[str] = None,
) -> Dict[str, Any]:
    """Pre-load the configured TTS provider so the next synthesis starts hot.

    Local engines load their voice/model into the same LRU slot synthesis
    reads (including first-use download); lazily-installed cloud SDKs are
    made importable; user-declared providers get ``warm_command`` /
    :meth:`TTSProvider.warm`; everything else is ``action: "noop"``.
    Never raises — the result dict carries ``warmed`` / ``action`` /
    ``error``. Blocking; UI threads should run it in the background.
    """
    if tts_config is None:
        tts_config = _origin()._load_tts_config()
    name = (provider or _origin()._get_provider(tts_config) or "").lower().strip()
    result: Dict[str, Any] = {"provider": name, "warmed": False, "action": "noop"}

    warmer = _local_tts_warmers().get(name)
    if warmer is not None:
        cache = _LOCAL_TTS_MODEL_CACHES.get(name)
        before = len(cache) if cache is not None else 0
        started = time.monotonic()
        try:
            warmer(tts_config)
        except Exception as exc:  # engine missing, download failed, bad voice…
            logger.warning("[TTS] warm-up for %s failed: %s", name, exc)
            result.update(action="error", error=str(exc))
            return result
        after = len(cache) if cache is not None else 0
        result.update(
            warmed=True,
            action="loaded" if after > before else "cached",
            elapsed_ms=int((time.monotonic() - started) * 1000),
        )
        logger.info("[TTS] warm-up %s: %s in %dms", name, result["action"], result["elapsed_ms"])
        return result

    signalled = _signal_user_tts_provider(name, tts_config, "warm")
    if signalled is not None:
        result.update(warmed=signalled != "error", action="warmed" if signalled != "error" else "error")
        return result

    feature = _lazy_sdk_feature_for_provider(name)
    if feature is not None:
        try:
            from tools.lazy_deps import ensure, is_available

            if is_available(feature):
                result.update(warmed=True, action="cached")
            else:
                ensure(feature, prompt=False)
                result.update(warmed=True, action="installed")
        except Exception as exc:
            logger.debug("[TTS] SDK warm-up for %s skipped: %s", name, exc)
            result.update(action="error", error=str(exc))
    return result


def release_tts_provider(provider: Optional[str] = None) -> Dict[str, Any]:
    """Drop resident local TTS models so their memory is returned.

    With ``provider`` given only that engine's cache is cleared; otherwise
    every local cache is, and the configured user-declared provider is
    signalled (plugin ``release()`` / command ``release_command``). Returns
    ``{"released": <model instances dropped>}``.
    """
    name = (provider or "").lower().strip()
    if not name:
        tts_config = _origin()._load_tts_config()
        _signal_user_tts_provider(_origin()._get_provider(tts_config), tts_config, "release")
    released = 0
    for cache_name, cache in _LOCAL_TTS_MODEL_CACHES.items():
        if name and cache_name != name:
            continue
        released += len(cache)
        cache.clear()
    if released:
        logger.info("[TTS] released %d resident local model(s)", released)
    return {"released": released}


def acquire_tts_lease(lease: str, tts_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Register ``lease`` (e.g. ``"desktop:read-aloud"``) as a live consumer and warm the provider.

    Re-acquiring is idempotent but still re-warms (cheap on a cache hit, and
    heals a cache cleared elsewhere).
    """
    with _tts_lease_lock:
        _tts_leases.add(lease)
        holders = len(_tts_leases)
    result = _origin().warm_tts_provider(tts_config)
    result["leases"] = holders
    return result


def release_tts_lease(lease: str) -> Dict[str, Any]:
    """Drop ``lease``; when it was the last one, unload resident local models.

    Releasing a never-acquired lease is a no-op (still reports the holder
    count) so surfaces can call it unconditionally on their "off" path.
    """
    with _tts_lease_lock:
        _tts_leases.discard(lease)
        holders = len(_tts_leases)
        result: Dict[str, Any] = {"leases": holders, "released": 0}
        if holders == 0:
            result["released"] = release_tts_provider()["released"]
    return result


def tts_lease_holders() -> List[str]:
    """Snapshot of live lease names (diagnostics / tests)."""
    with _tts_lease_lock:
        return sorted(_tts_leases)


def _reset_tts_leases_for_tests() -> None:
    with _tts_lease_lock:
        _tts_leases.clear()

