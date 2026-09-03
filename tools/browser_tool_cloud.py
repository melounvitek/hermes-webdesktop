"""Cloud browser provider resolution (explicit browser.cloud_provider, auto-detect, per-profile cache), backend/engine selection and headed-mode flags.

Split out of ``tools/browser_tool.py``; every name is re-imported there so
``tools.browser_tool.<name>`` keeps resolving (and monkeypatching). Origin
symbols and module state are read/written through ``_bt`` (the origin module,
resolved per call by :func:`tools.browser_tool_origin.origin_module`) so
``patch("tools.browser_tool.X")`` is honoured and no import cycle exists.
"""

from __future__ import annotations

import os
from typing import Optional

from agent.browser_provider import BrowserProvider as CloudBrowserProvider
from tools.browser_tool_origin import origin_module as _origin


def _is_legacy_provider_registry_overridden() -> bool:
    """True when a test has patched ``_PROVIDER_REGISTRY`` to a custom value.

    Each registered value is compared by identity against the canonical class
    in ``_DEFAULT_PROVIDER_REGISTRY`` (extra keys count too); adding a built-in
    provider only requires extending that default dict.
    """
    _bt = _origin()
    try:
        for key, default_cls in _bt._DEFAULT_PROVIDER_REGISTRY.items():
            if _bt._PROVIDER_REGISTRY.get(key) is not default_cls:
                return True
        # Extra keys not in the default registry → also an override.
        return len(_bt._PROVIDER_REGISTRY) != len(_bt._DEFAULT_PROVIDER_REGISTRY)
    except Exception:
        return False


def _ensure_browser_plugins_loaded() -> None:
    """Idempotently trigger plugin discovery so the browser registry is populated.

    ``model_tools`` normally does this as an import side effect, but
    ``_get_cloud_provider`` is also reached from standalone scripts and test
    harnesses that never import it; cheap on repeat calls.
    """
    _bt = _origin()
    try:
        from hermes_cli.plugins import _ensure_plugins_discovered

        _ensure_plugins_discovered()
    except Exception as exc:
        _bt.logger.debug("Browser plugin discovery failed (non-fatal): %s", exc)


def _get_cloud_provider() -> Optional[CloudBrowserProvider]:
    """Return the provider cached for the active Hermes profile."""
    _bt = _origin()

    scope = _bt.hermes_home_key()
    with _bt._cloud_provider_cache_lock:
        # Tests and legacy reset paths clear the boolean. Treat that as a full
        # reset even if a previous scoped resolution remains mirrored here.
        if not _bt._cloud_provider_resolved:
            _bt._cached_cloud_provider_scope = None
            _bt._cached_cloud_providers.clear()
        while True:
            before_generation = _bt._browser_registry_generation(scope=scope)
            cache_key = (scope, before_generation)
            if cache_key in _bt._cached_cloud_providers:
                _bt._cached_cloud_provider = _bt._cached_cloud_providers[cache_key]
                _bt._cloud_provider_resolved = True
                _bt._cached_cloud_provider_scope = scope
                return _bt._cached_cloud_provider

            _bt._cached_cloud_provider = None
            _bt._cloud_provider_resolved = False
            resolved = _bt._resolve_cloud_provider_uncached()
            after_generation = _bt._browser_registry_generation(scope=scope)
            if before_generation != after_generation:
                # A force reload replaced/unloaded this profile's provider
                # while resolution was in progress. Discard the stale result
                # and resolve against the new registry generation.
                continue
            if _bt._cloud_provider_resolved:
                _bt._cached_cloud_provider_scope = scope
                for stale_key in [
                    key for key in _bt._cached_cloud_providers if key[0] == scope
                ]:
                    _bt._cached_cloud_providers.pop(stale_key, None)
                _bt._cached_cloud_providers[cache_key] = resolved
            return resolved


def _instantiate_explicit_cloud_provider(provider_key: str) -> Optional[CloudBrowserProvider]:
    """Build the provider named by ``browser.cloud_provider``.

    Test fixtures that patch ``_PROVIDER_REGISTRY`` drive the legacy dict;
    otherwise the plugin registry is consulted (after idempotent discovery).
    Strict selection: a stored-but-unregistered name raises ``ValueError``
    (never a silent reroute to auto-detect). Any other instantiation error is
    logged and yields None so the next call retries.
    """
    _bt = _origin()
    try:
        if _bt._is_legacy_provider_registry_overridden():
            factory = _bt._PROVIDER_REGISTRY.get(provider_key)
            resolved = factory() if factory is not None else None
        else:
            _bt._ensure_browser_plugins_loaded()
            resolved = _bt._registry_get_browser_provider(provider_key)
        if resolved is None:
            from tools.tool_backend_helpers import selection_error

            raise ValueError(selection_error(
                "browser",
                f"'{provider_key}'",
                "no registered browser plugin has that name (install "
                "the corresponding plugin or fix the config key "
                "spelling)",
            ))
        return resolved
    except ValueError:
        raise
    except Exception:
        _bt.logger.warning(
            "Failed to instantiate explicit cloud_provider %r; will retry on next call",
            provider_key,
            exc_info=True,
        )
        return None


def _autodetect_cloud_provider() -> Optional[CloudBrowserProvider]:
    """Auto-detect: Browser Use (managed Nous gateway or API key), then Browserbase.

    Uses the legacy class names bound on this module so tests that
    ``monkeypatch.setattr(browser_tool, "BrowserUseProvider", ...)`` keep
    driving this branch. Third-party plugins are intentionally NOT reachable
    from auto-detect — only via explicit ``browser.cloud_provider: <name>``.
    Never raises (a failure must not poison the cache).
    """
    _bt = _origin()
    try:
        for cls in (_bt.BrowserUseProvider, _bt.BrowserbaseProvider):
            fallback_provider = cls()
            if fallback_provider.is_configured():
                return fallback_provider
    except Exception:  # pragma: no cover - defensive: never poison cache
        _bt.logger.debug("Cloud provider auto-detect failed", exc_info=True)
    return None


def _resolve_cloud_provider_uncached() -> Optional[CloudBrowserProvider]:
    """Return the configured cloud browser provider, or None for local mode.

    Reads ``browser.cloud_provider`` and pins the result in the cache only when
    it is definitive (explicit ``local``/``camofox``, or a resolved provider).
    Explicit selection routes through :mod:`agent.browser_registry` so
    third-party plugins participate; auto-detect (only when no selection was
    ever written) walks Browser Use then Browserbase. A transient None
    (unreadable config, missing credentials) is NOT cached so it can self-heal.
    """
    _bt = _origin()

    resolved: Optional[CloudBrowserProvider] = None
    provider_key = None
    try:
        from hermes_cli.config import read_raw_config
        browser_cfg = read_raw_config().get("browser", {})
        if isinstance(browser_cfg, dict) and "cloud_provider" in browser_cfg:
            provider_key = _bt.normalize_browser_cloud_provider(browser_cfg.get("cloud_provider"))
            if provider_key in ("local", "camofox"):
                # Camofox runs through the built-in browser tools, not a cloud provider.
                _bt._cached_cloud_provider = None
                _bt._cloud_provider_resolved = True
                return None
            if provider_key == "nous":
                # Managed "Nous Subscription" is serviced by the Browser Use provider.
                provider_key = "browser-use"
        if provider_key:
            resolved = _bt._instantiate_explicit_cloud_provider(provider_key)
            if resolved is None:
                return None
    except ValueError:
        raise
    except Exception as e:
        # Config may be temporarily unreadable; still try auto-detect so
        # env-based / managed-gateway credentials can resolve. Don't pin cache.
        _bt.logger.debug("Could not read cloud_provider from config: %s", e)

    if resolved is None and provider_key is None:
        resolved = _bt._autodetect_cloud_provider()
    if resolved is None:
        return None

    _bt._cached_cloud_provider = resolved
    _bt._cloud_provider_resolved = True
    return _bt._cached_cloud_provider


def _is_local_mode() -> bool:
    """Return True when the browser tool will use a local browser backend."""
    _bt = _origin()
    if _bt._get_cdp_override_raw():
        return False
    return _bt._get_cloud_provider() is None


def _is_local_backend() -> bool:
    """Return True when the browser runs locally AND the terminal is also local.

    SSRF protection only matters when the browser can reach networks the user's
    terminal cannot: cloud backends, and a local browser paired with a
    containerized terminal (docker/modal/daytona/ssh/singularity). A CDP
    override is never trusted as local (that Chrome may live off-host) and MUST
    be checked before the Camofox short-circuit so Camofox + override still
    fails the local check; ``_is_local_mode`` treats overrides the same way —
    keep the two in agreement.
    """
    _bt = _origin()
    if _bt._get_cdp_override_raw():
        return False
    if _bt._is_camofox_mode():
        return True
    if _bt._get_cloud_provider() is not None:
        return False
    # Scope-aware: under gateway multiplexing the routed profile's terminal
    # backend lives in the per-turn terminal scope, not the process env.
    from tools.terminal_scope import terminal_env

    terminal_backend = terminal_env("TERMINAL_ENV", "local").strip().lower()
    return terminal_backend in ("local", "")


def _get_browser_engine() -> str:
    """Return the browser engine: ``auto`` (no ``--engine`` flag), ``lightpanda`` or ``chrome``.

    ``browser.engine`` first, then ``AGENT_BROWSER_ENGINE``, then ``auto``;
    cached. Lightpanda is much faster on navigation but has no graphical
    renderer (no screenshots).
    """
    _bt = _origin()
    if _bt._browser_engine_resolved:
        return _bt._cached_browser_engine

    _bt._browser_engine_resolved = True
    # Config file takes priority; env var only if config didn't set a value.
    _bt._cached_browser_engine = _bt._browser_cfg(
        "engine", "auto",
        lambda v: str(v).strip().lower() if v and str(v).strip() else "auto",
        "browser.engine from config",
    )
    if _bt._cached_browser_engine == "auto":
        env_val = os.environ.get("AGENT_BROWSER_ENGINE", "").strip().lower()
        if env_val:
            _bt._cached_browser_engine = env_val

    # Validate: agent-browser only accepts "chrome" and "lightpanda".
    _VALID_ENGINES = {"auto", "lightpanda", "chrome"}
    if _bt._cached_browser_engine not in _VALID_ENGINES:
        _bt.logger.warning(
            "Unknown browser engine %r (valid: %s), falling back to 'auto'",
            _bt._cached_browser_engine, ", ".join(sorted(_VALID_ENGINES)),
        )
        _bt._cached_browser_engine = "auto"

    return _bt._cached_browser_engine


def _is_headed_mode() -> bool:
    """Return True when the browser should launch in headed (visible) mode.

    Reads ``config["browser"]["headed"]`` with ``AGENT_BROWSER_HEADED`` env
    var as fallback.  Result is cached after the first call.
    """
    _bt = _origin()
    if _bt._headed_mode_resolved:
        return _bt._cached_headed_mode  # type: ignore[return-value]

    _bt._headed_mode_resolved = True
    _bt._cached_headed_mode = _bt._browser_cfg(
        "headed", False,
        lambda v: False if v is None else str(v).strip().lower() in ("true", "1", "yes"),
        "browser.headed from config",
    )
    if not _bt._cached_headed_mode:
        env_val = os.environ.get("AGENT_BROWSER_HEADED", "").strip()
        if env_val and env_val.lower() in ("true", "1", "yes"):
            _bt._cached_headed_mode = True

    return _bt._cached_headed_mode


def _should_inject_engine(engine: str) -> bool:
    """Return True when the engine flag should be added to agent-browser commands.

    Only inject ``--engine`` for non-cloud, non-camofox local sessions where
    the engine is explicitly set (not ``auto``).
    """
    _bt = _origin()
    if engine == "auto":
        return False
    if _bt._is_camofox_mode():
        return False
    return _bt._is_local_mode()


def _auto_local_for_private_urls() -> bool:
    """``browser.auto_local_for_private_urls`` (default True), cached for the process.

    When on, ``browser_navigate`` routes private/loopback/LAN URLs to a local
    Chromium sidecar even with a cloud provider configured; public URLs keep
    using the cloud provider in the same conversation.
    """
    _bt = _origin()
    if _bt._auto_local_for_private_urls_resolved:
        return _bt._cached_auto_local_for_private_urls

    _bt._auto_local_for_private_urls_resolved = True
    _bt._cached_auto_local_for_private_urls = _bt._browser_cfg(
        "auto_local_for_private_urls", _bt._cached_auto_local_for_private_urls, bool,
        "auto_local_for_private_urls from config",
    )
    return _bt._cached_auto_local_for_private_urls


def _use_real_profile() -> bool:
    """Return whether the user consented to real-profile local browsing.

    Reads ``browser.use_real_profile`` (default False) on EVERY call — it is a
    consent switch, so flipping it off must take effect without a restart, and
    in a multiplexed gateway each profile's config must decide for itself.
    The read is one YAML load per local session creation (not per command),
    so there is no hot-path cost to keeping it uncached.
    """
    _bt = _origin()
    return _bt._browser_cfg("use_real_profile", False, bool, "use_real_profile from config")


def _allow_private_urls() -> bool:
    """Return whether the browser is allowed to navigate to private/internal addresses.

    Reads ``config["browser"]["allow_private_urls"]``. Single-profile calls
    cache the result for the process lifetime; multiplexed profile turns resolve
    their context-local config on each call. Defaults to ``False`` (SSRF
    protection active).
    """
    _bt = _origin()

    # The profile multiplexer scopes config with a ContextVar while sharing
    # this module. Never reuse another profile's private-network opt-out.
    if _bt.get_hermes_home_override() is not None:
        return _bt._resolve_allow_private_urls()

    if _bt._allow_private_urls_resolved:
        return _bt._cached_allow_private_urls

    _bt._allow_private_urls_resolved = True
    _bt._cached_allow_private_urls = _bt._resolve_allow_private_urls()
    return _bt._cached_allow_private_urls


def _resolve_allow_private_urls() -> bool:
    """Read the browser private-URL toggle from the active config scope."""
    _bt = _origin()
    return _bt._browser_cfg(
        "allow_private_urls", False,
        lambda v: _bt.is_truthy_value(v, default=False),
        "allow_private_urls from config",
    )
