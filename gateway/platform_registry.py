"""
Platform Adapter Registry

Platform adapters (built-in and plugin) self-register here so the gateway can
discover and instantiate them without hardcoded if/elif chains.  Plugins
register via ``PluginContext.register_platform()``; ``GatewayRunner
._create_adapter()`` consults the registry first and falls back to the legacy
built-in path.

Usage (plugin side)::

    platform_registry.register(PlatformEntry(
        name="irc", label="IRC",
        adapter_factory=lambda cfg: IRCAdapter(cfg),
        check_fn=check_requirements,
        validate_config=lambda cfg: bool(cfg.extra.get("server")),
        required_env=["IRC_SERVER"], install_hint="pip install irc",
    ))

Usage (gateway side)::

    adapter = platform_registry.create_adapter("irc", platform_config)
"""

import logging
import sys
import threading
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Optional

from hermes_constants import hermes_home_key

logger = logging.getLogger(__name__)

_LoadKey = tuple[Optional[str], str]
_Loader = Callable[[], None]


def _plugin_scope_from_callable(callback: Callable) -> Optional[str]:
    """Infer a plugin profile from code registered outside PluginContext."""
    try:
        from tools.registry import registry as tool_registry

        return tool_registry.plugin_scope_for_callable(callback)
    except (ImportError, AttributeError):
        return None


def _caller_plugin_scope() -> Optional[str]:
    try:
        module_name = sys._getframe(2).f_globals.get("__name__", "") or ""
    except Exception:
        return None
    return _plugin_scope_from_callable(
        type("_Caller", (), {"__module__": module_name})
    )


@dataclass
class PlatformEntry:
    """Metadata and factory for a single platform adapter."""

    # Identifier used in config.yaml (e.g. "irc") and human-readable label.
    name: str
    label: str

    # Receives a PlatformConfig, returns an adapter instance.  A factory (not a
    # bare class) lets plugins do custom init / wrapping.
    adapter_factory: Callable[[Any], Any]

    # PASSIVE dependency probe: True when deps are importable RIGHT NOW.  Must
    # be side-effect free -- it runs from status displays (`hermes setup`,
    # `hermes status`, dashboard readiness) and the config enablement pass,
    # none of which may pip-install.  Install logic belongs in ensure_deps_fn.
    check_fn: Callable[[], bool]

    # Optional config check.  None = skip and let connect() fail descriptively.
    validate_config: Optional[Callable[[Any], bool]] = None

    # ACTIVE dependency installer: make deps importable (pip / lazy_deps),
    # returning True on success.  Called by ``create_adapter()`` only when
    # ``check_fn`` is False -- the one moment the user has the platform
    # enabled+configured and the gateway is about to connect it.  None = a
    # False ``check_fn`` is a hard block (correct for platforms without
    # optional deps).  The passive/active split exists because a single field
    # either pip-installed from every status display or never installed at all.
    ensure_deps_fn: Optional[Callable[[], bool]] = None

    # Is the platform connected/enabled for this PlatformConfig?  Used by
    # ``GatewayConfig.get_connected_platforms()`` and setup UI status; None
    # falls back to ``validate_config`` or ``check_fn``.
    is_connected: Optional[Callable[[Any], bool]] = None

    # Env vars this platform needs (``hermes setup`` display) and the hint
    # shown when check_fn returns False.
    required_env: list = field(default_factory=list)
    install_hint: str = ""

    # Interactive setup ``() -> None``.  None falls back to
    # _setup_standard_platform (needs token_var + vars) or a generic env display.
    setup_fn: Optional[Callable[[], None]] = None

    # "builtin" or "plugin"; plugin_name is the owning manifest (empty for
    # built-ins) so ``hermes gateway setup`` can auto-enable the plugin.
    source: str = "plugin"
    plugin_name: str = ""

    # Auth env var names for _is_user_authorized: comma-separated allowed user
    # IDs, and a truthy "allow everyone" switch.
    allowed_users_env: str = ""
    allow_all_env: str = ""

    # Max message length for smart-chunking; 0 = no limit.
    max_message_length: int = 0

    # If True, session descriptions redact PII (phone numbers, etc.).
    pii_safe: bool = False

    # Emoji for CLI/gateway display.
    emoji: str = "🔌"

    # Whether /update may be issued from this platform (_UPDATE_ALLOWED_PLATFORMS).
    allow_update_command: bool = True

    # Platform hint injected into the system prompt; empty = none.
    platform_hint: str = ""

    # Env-driven auto-configuration ``() -> Optional[dict]``: returns
    # ``PlatformConfig.extra`` fields to seed when the platform is auto-enabled.
    # Runs during ``_apply_env_overrides`` BEFORE the adapter is constructed so
    # ``gateway status`` reflects env-only config.  None/empty dict = skip.
    env_enablement_fn: Optional[Callable[[], Optional[dict]]] = None

    # YAML->env bridge ``(yaml_cfg, platform_cfg) -> Optional[dict]``: lets a
    # plugin own its config.yaml translation instead of core gateway/config.py.
    # Called from ``load_gateway_config()`` after the generic shared-key loop and
    # before ``_apply_env_overrides``.  May mutate ``os.environ`` (guard with
    # ``not os.getenv(...)`` to keep env > YAML precedence); the returned dict is
    # merged into ``PlatformConfig.extra``.  Exceptions are logged at debug.
    # Full contract: website/docs/developer-guide/adding-platform-adapters.md.
    apply_yaml_config_fn: Optional[Callable[[dict, dict], Optional[dict]]] = None

    # Home-channel env var (e.g. "IRC_HOME_CHANNEL").  When set, cron.scheduler
    # accepts ``deliver=<name>`` and reads it for the default chat/room ID.
    cron_deliver_env_var: str = ""

    # Target parsing ``(target_ref) -> Optional[(chat_id, thread_id)]``, run by
    # ``tools/send_message_tool._parse_target_ref`` before channel-directory
    # fallback so plugins can declare native target syntax
    # (e.g. ``fmsg:@alice@example.com``).  None result = continue to directory
    # resolution; no opaque fallback is applied.
    parse_target_ref_fn: Optional[Callable[[str], Optional[tuple[str, Optional[str]]]]] = None

    # Validation after parsing/normalization or directory resolution: True to
    # accept, False to reject, or a non-empty string to reject with diagnostic.
    validate_target_ref_fn: Optional[Callable[[str], bool | str]] = None

    # Whole-request delivery handler ``(args, normalized_chat_id, platform_name,
    # pconfig)``, sync or async.  Prefer standalone_sender_fn when the standard
    # send contract suffices.
    send_message_handler: Optional[Callable[[dict, str, str, Any], Any]] = None

    # Out-of-process sender used by ``_send_via_adapter`` when cron runs apart
    # from the gateway and the in-process adapter weakref is None::
    #     async (pconfig, chat_id, message, *, thread_id=None,
    #            media_files=None, force_document=False) -> dict
    # Returns ``{"success": True, "message_id": ...}`` or ``{"error": str}``.
    # Without it, plugin platforms cannot be cron ``deliver=`` targets when the
    # gateway is not co-resident.
    standalone_sender_fn: Optional[Callable[..., Awaitable[dict]]] = None


class PlatformRegistry:
    """Central registry of platform adapters.

    Registrations are serialized, and concurrent lazy lookups share an
    in-flight event while the loader runs outside the registry lock.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        # Process-global registrations (e.g. the built-in relay).
        self._entries: dict[str, PlatformEntry] = {}
        # Plugin adapters are isolated per resolved HERMES_HOME and overlay the
        # process-global entries for lookups in that profile's runtime scope.
        self._scoped_entries: dict[str, dict[str, PlatformEntry]] = {}
        # Deferred loaders: name -> zero-arg callable that imports the owning
        # plugin module (which calls register()).  Adapter modules import heavy
        # SDKs at module level; eagerly loading ~20 bundled platforms added
        # seconds to every `hermes` invocation, so the real import happens only
        # when a lookup actually asks for that platform.
        self._deferred: dict[str, _Loader] = {}
        self._scoped_deferred: dict[str, dict[str, _Loader]] = {}
        self._inflight: dict[_LoadKey, threading.Event] = {}
        self._inflight_loaders: dict[_LoadKey, _Loader] = {}
        self._inflight_owners: dict[_LoadKey, int] = {}
        self._cancelled_inflight: set[_LoadKey] = set()
        # A failed loader is no longer discoverable, but its identity remains
        # until ownership teardown can CAS-restore the displaced predecessor.
        self._consumed_loaders: dict[_LoadKey, _Loader] = {}

    @staticmethod
    def current_scope_key() -> str:
        return hermes_home_key()

    def _scope_maps(
        self, scope: Optional[str], *, create: bool = False
    ) -> tuple[dict[str, PlatformEntry], dict[str, _Loader]]:
        if scope is None:
            return self._entries, self._deferred
        if create:
            return self._scoped_entries.setdefault(scope, {}), self._scoped_deferred.setdefault(scope, {})
        return self._scoped_entries.get(scope, {}), self._scoped_deferred.get(scope, {})

    def _registration_state(
        self, scope: Optional[str], name: str, *, create: bool = False
    ) -> tuple[Optional[PlatformEntry], Optional[_Loader]]:
        """(entry, loader) for *name*; the loader falls back to in-flight, then consumed."""
        entries, deferred = self._scope_maps(scope, create=create)
        entry = entries.get(name)
        loader = deferred.get(name)
        if entry is None and loader is None:
            loader = self._inflight_loaders.get((scope, name))
        if entry is None and loader is None:
            loader = self._consumed_loaders.get((scope, name))
        return entry, loader

    def _prune_scope(self, scope: Optional[str]) -> None:
        if scope is None:
            return
        if not self._scoped_entries.get(scope):
            self._scoped_entries.pop(scope, None)
        if not self._scoped_deferred.get(scope):
            self._scoped_deferred.pop(scope, None)

    # -- deferred loading ----------------------------------------------------

    def register_deferred(self, name: str, loader: _Loader, *, scope: Optional[str] = None) -> None:
        """Register a lazy loader for a platform that hasn't been imported yet.

        *loader* imports the owning plugin module, which must call
        :meth:`register` for *name*.  It runs at most once, on first lookup
        (or full materialization).  A concrete registration takes precedence
        and drops the loader.
        """
        with self._lock:
            entries, deferred = self._scope_maps(scope, create=True)
            self._consumed_loaders.pop((scope, name), None)
            if name not in entries:
                deferred[name] = loader

    def snapshot_registration(
        self, name: str, *, scope: Optional[str] = None
    ) -> tuple[Optional[PlatformEntry], Optional[_Loader]]:
        """Concrete and deferred state for *name* without resolving it.

        Lets the plugin ledger restore a deferred loader displaced by a concrete
        registration without importing the displaced adapter as a side effect.
        """
        with self._lock:
            return self._registration_state(scope, name)

    def restore_registration(
        self,
        name: str,
        current: tuple[Optional[PlatformEntry], Optional[_Loader]],
        previous: tuple[Optional[PlatformEntry], Optional[_Loader]],
        *,
        scope: Optional[str] = None,
    ) -> bool:
        """Restore a registration if its full state is still *current* (CAS).

        Identity checks protect a later registration from removal while letting
        an unloaded override reveal what it displaced.  Deferred loaders are
        part of the state because bundled platform plugins load lazily.
        """
        with self._lock:
            entry, loader = self._registration_state(scope, name, create=True)
            if entry is not current[0] or loader is not current[1]:
                return False
            entries, deferred = self._scope_maps(scope)
            load_key = (scope, name)
            previous_entry, previous_loader = previous
            if previous_entry is None:
                entries.pop(name, None)
            else:
                entries[name] = previous_entry
            if previous_loader is None:
                deferred.pop(name, None)
            else:
                deferred[name] = previous_loader
            if load_key in self._inflight:
                self._cancelled_inflight.add(load_key)
            self._consumed_loaders.pop(load_key, None)
            self._prune_scope(scope)
            return True

    def _resolve(self, name: str, scope: Optional[str] = None) -> None:
        """Run the deferred loader for *name* if one is pending."""
        loader: Optional[_Loader] = None
        event: Optional[threading.Event] = None
        load_key: _LoadKey
        is_loader = False
        with self._lock:
            active_scope = scope or self.current_scope_key()
            entries, deferred = self._scope_maps(active_scope)
            scoped_key = (active_scope, name)
            global_key = (None, name)
            event = self._inflight.get(scoped_key)
            load_key = scoped_key
            if event is None and name not in entries:
                loader = deferred.pop(name, None)
            if event is None and loader is None and name not in entries:
                event = self._inflight.get(global_key)
                load_key = global_key
            if event is None and loader is None and name not in entries:
                loader = self._deferred.pop(name, None)
                load_key = global_key
            if event is None and loader is not None:
                event = threading.Event()
                self._inflight[load_key] = event
                self._inflight_loaders[load_key] = loader
                self._inflight_owners[load_key] = threading.get_ident()
                is_loader = True
            if event is None:
                return
            if not is_loader and self._inflight_owners.get(load_key) == threading.get_ident():
                logger.warning("Deferred platform '%s' recursively requested while loading", name)
                return

        if not is_loader:
            event.wait()
            # Teardown may have restored an older deferred generation while
            # cancelling the one we waited for; resolve that predecessor now
            # instead of returning a one-shot false negative.
            self._resolve(name, active_scope)
            return

        try:
            loader()
        except Exception as e:
            logger.warning("Deferred load of platform '%s' failed: %s", name, e, exc_info=True)
        finally:
            with self._lock:
                was_cancelled = load_key in self._cancelled_inflight
                entries, deferred = self._scope_maps(load_key[0])
                if not was_cancelled and name not in entries and name not in deferred:
                    self._consumed_loaders[load_key] = loader
                self._inflight.pop(load_key, None)
                self._inflight_loaders.pop(load_key, None)
                self._inflight_owners.pop(load_key, None)
                self._cancelled_inflight.discard(load_key)
                event.set()
        if was_cancelled:
            self._resolve(name, active_scope)

    def is_deferred_load_cancelled(self, name: str, *, scope: Optional[str] = None) -> bool:
        """Return whether ownership teardown cancelled an in-flight loader."""
        with self._lock:
            return (scope, name) in self._cancelled_inflight

    def _resolve_all(self) -> None:
        """Run every pending deferred loader.

        Only the iterate-all accessors (``all_entries``/``plugin_entries``) call
        this, from paths that genuinely need every adapter (gateway startup,
        ``hermes setup``/``gateway status``, channel directory).  CLI chat never
        iterates the full set.
        """
        active_scope = self.current_scope_key()
        with self._lock:
            _entries, scoped_deferred = self._scope_maps(active_scope)
            scoped_names = set(scoped_deferred)
            global_names = set(self._deferred)
            for inflight_scope, name in self._inflight:
                if inflight_scope == active_scope:
                    scoped_names.add(name)
                elif inflight_scope is None:
                    global_names.add(name)
        # Load outside the registry lock; each name has an in-flight event so
        # concurrent readers wait for the same materialization.
        for name in (*sorted(scoped_names), *sorted(global_names)):
            self._resolve(name, active_scope)

    def register(self, entry: PlatformEntry, *, scope: Optional[str] = None) -> None:
        """Register a platform adapter entry (last writer wins on name clash)."""
        with self._lock:
            if scope is None and entry.source == "plugin":
                scope = _caller_plugin_scope()
                if scope is None:
                    scope = _plugin_scope_from_callable(entry.adapter_factory)
                if scope is None:
                    scope = _plugin_scope_from_callable(entry.check_fn)
            # A concrete registration supersedes any pending deferred loader.
            entries, deferred = self._scope_maps(scope, create=True)
            self._consumed_loaders.pop((scope, entry.name), None)
            deferred.pop(entry.name, None)
            prev = entries.get(entry.name)
            if prev is not None:
                logger.info(
                    "Platform '%s' re-registered (was %s, now %s)", entry.name, prev.source, entry.source
                )
            entries[entry.name] = entry
            logger.debug("Registered platform adapter: %s (%s)", entry.name, entry.source)

    def unregister(self, name: str, *, scope: Optional[str] = None) -> bool:
        """Remove a platform entry.  Returns True if it existed."""
        with self._lock:
            inferred_scope = scope if scope is not None else _caller_plugin_scope()
            active_scope = inferred_scope or self.current_scope_key()
            entries, deferred = self._scope_maps(active_scope)
            if inferred_scope is not None or name in entries or name in deferred:
                deferred.pop(name, None)
                removed = entries.pop(name, None) is not None
                self._prune_scope(active_scope)
                return removed
            self._deferred.pop(name, None)
            return self._entries.pop(name, None) is not None

    def _load_pending(self, scope: str, name: str) -> bool:
        """True when a lookup of *name* must run/await a deferred loader (lock held)."""
        _entries, deferred = self._scope_maps(scope)
        return (
            name in deferred
            or (name not in self._entries and name in self._deferred)
            or (scope, name) in self._inflight
            or (None, name) in self._inflight
        )

    def get(self, name: str) -> Optional[PlatformEntry]:
        """Look up a platform entry by name."""
        scope = self.current_scope_key()
        with self._lock:
            entries, _deferred = self._scope_maps(scope)
            needs_resolve = name not in entries and self._load_pending(scope, name)
        if needs_resolve:
            self._resolve(name, scope)
        with self._lock:
            entries, _deferred = self._scope_maps(scope)
            return entries.get(name) or self._entries.get(name)

    def all_entries(self) -> list[PlatformEntry]:
        """Return all registered platform entries."""
        self._resolve_all()
        with self._lock:
            entries = dict(self._entries)
            entries.update(self._scoped_entries.get(self.current_scope_key(), {}))
            return list(entries.values())

    def plugin_entries(self) -> list[PlatformEntry]:
        """Return only plugin-registered platform entries."""
        return [e for e in self.all_entries() if e.source == "plugin"]

    def registered_names(self) -> set[str]:
        """Concrete and deferred platform names without loading adapters.

        Same scope semantics as ``is_registered()``: current profile scope AND
        process-global names.  Plugin platforms register deferred loaders under
        a profile scope, so the global maps alone would miss every plugin.
        """
        with self._lock:
            entries, deferred = self._scope_maps(self.current_scope_key())
            return entries.keys() | deferred.keys() | self._entries.keys() | self._deferred.keys()

    def is_registered(self, name: str) -> bool:
        # A deferred (not-yet-imported) platform still counts as registered so
        # cheap membership checks (toolset resolution, webhook deliver-target
        # checks) never trigger a heavy import.
        with self._lock:
            scope = self.current_scope_key()
            entries, _deferred = self._scope_maps(scope)
            return name in entries or name in self._entries or self._load_pending(scope, name)

    def create_adapter(self, name: str, config: Any) -> Optional[Any]:
        """Create an adapter instance for *name*.

        Returns None when no entry exists, deps are missing and cannot be
        installed, ``validate_config`` fails, or the factory raises.
        """
        entry = self.get(name)
        if entry is None:
            return None

        deps_ok = False
        try:
            deps_ok = bool(entry.check_fn())
        except Exception as e:
            logger.warning("Platform '%s' check_fn raised: %s", entry.label, e)
        if not deps_ok and entry.ensure_deps_fn is not None:
            # The ONE place the active installer runs in the adapter path: the
            # platform is enabled+configured and about to connect, so an install
            # is what the user wants. (An installer inside connect() is never
            # reached when check_fn is False -- create_adapter() would have
            # already returned None.)
            logger.info("Platform '%s' dependencies missing — attempting install...", entry.label)
            try:
                deps_ok = bool(entry.ensure_deps_fn())
            except Exception as e:
                logger.warning("Platform '%s' dependency install raised: %s", entry.label, e)
                deps_ok = False
        if not deps_ok:
            hint = f" ({entry.install_hint})" if entry.install_hint else ""
            logger.warning("Platform '%s' requirements not met%s", entry.label, hint)
            return None

        if entry.validate_config is not None:
            try:
                if not entry.validate_config(config):
                    logger.warning("Platform '%s' config validation failed", entry.label)
                    return None
            except Exception as e:
                logger.warning("Platform '%s' config validation error: %s", entry.label, e)
                return None

        try:
            return entry.adapter_factory(config)
        except Exception as e:
            logger.error(
                "Failed to create adapter for platform '%s': %s", entry.label, e, exc_info=True
            )
            return None


# Module-level singleton
platform_registry = PlatformRegistry()
