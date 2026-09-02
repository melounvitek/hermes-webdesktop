"""Memory provider plugin discovery.

Scans four sources for memory provider plugins:

1. Bundled providers: ``plugins/memory/<name>/`` (shipped with hermes-agent)
2. User-installed providers: ``$HERMES_HOME/plugins/<name>/``
3. Project-local providers: ``./.hermes/plugins/<name>/``, opt-in via
   ``HERMES_ENABLE_PROJECT_PLUGINS``
4. Pip-installed providers: ``hermes_agent.memory_providers`` entry points

Directory providers must contain ``__init__.py`` with a class implementing
the MemoryProvider ABC. Pip packages expose a provider or ``register(ctx)``
callback through the entry-point group.

These are the same four sources the general ``PluginManager`` scans, but the
precedence is deliberately the reverse of its later-source-wins order: here
**bundled wins**, then user, then project, then entry point. A memory provider
is activated by name, so letting a directory dropped into the working tree
shadow a shipped provider would silently redirect the agent's memory. Changing
this order is a breaking change, not a cleanup.

Only ONE provider can be active at a time, selected via
``memory.provider`` in config.yaml.

Usage:
    from plugins.memory import discover_memory_providers, load_memory_provider

    available = discover_memory_providers()   # [(name, desc, available), ...]
    provider = load_memory_provider("mnemosyne")  # MemoryProvider instance
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import logging
import sys
from pathlib import Path
from typing import List, Optional, Tuple, TYPE_CHECKING

from hermes_cli.config import cfg_get
from plugins import plugin_loader as _loader

if TYPE_CHECKING:
    from agent.memory_provider import MemoryProvider

logger = logging.getLogger(__name__)

_MEMORY_PLUGINS_DIR = Path(__file__).parent
ENTRY_POINTS_GROUP = "hermes_agent.memory_providers"
_REGISTERED_MEMORY_PROVIDER_SKILLS: dict[str, Path] = {}

# Synthetic parent package for user-installed providers, so they don't
# collide with bundled providers in sys.modules.
_USER_NAMESPACE = "_hermes_user_memory"

_register_synthetic_package = _loader.register_synthetic_package
_get_user_plugins_dir = _loader.user_plugins_dir


# ---------------------------------------------------------------------------
# Directory helpers
# ---------------------------------------------------------------------------

def _get_project_plugins_dir() -> Optional[Path]:
    """Return ``./.hermes/plugins/`` or None if unavailable or not opted in.

    Gated on ``HERMES_ENABLE_PROJECT_PLUGINS`` exactly as the general
    ``PluginManager`` gates its own project scan — a repository you merely
    ``cd`` into must not be able to offer the agent a memory backend.
    """
    try:
        from hermes_cli.plugins import _env_enabled

        if not _env_enabled("HERMES_ENABLE_PROJECT_PLUGINS"):
            return None
        d = Path.cwd() / ".hermes" / "plugins"
        return d if d.is_dir() else None
    except Exception:
        return None


def _is_memory_provider_dir(path: Path) -> bool:
    """Cheap text heuristic (no import): ``__init__.py`` mentions the memory provider contract."""
    init_file = path / "__init__.py"
    if not init_file.exists():
        return False
    try:
        source = init_file.read_text(errors="replace", encoding="utf-8")[:8192]
        return "register_memory_provider" in source or "MemoryProvider" in source
    except Exception:
        return False


def _is_bundled(provider_dir: Path) -> bool:
    return _MEMORY_PLUGINS_DIR in provider_dir.parents or provider_dir.parent == _MEMORY_PLUGINS_DIR


def _module_name(provider_dir: Path, name: str) -> str:
    """``plugins.memory.<name>`` for bundled providers, else under the synthetic user namespace."""
    return f"plugins.memory.{name}" if _is_bundled(provider_dir) else f"{_USER_NAMESPACE}.{name}"


def _iter_provider_dirs() -> List[Tuple[str, Path]]:
    """``(name, path)`` for bundled, then user-installed, then project-local; first-seen wins."""
    dirs = [(child.name, child) for child in _loader.iter_plugin_dirs(_MEMORY_PLUGINS_DIR)]
    seen = {name for name, _ in dirs}
    for source_dir in (_get_user_plugins_dir(), _get_project_plugins_dir()):
        if not source_dir:
            continue
        for child in sorted(source_dir.iterdir()):
            if (
                child.is_dir()
                and not child.name.startswith(("_", "."))
                and child.name not in seen
                and _is_memory_provider_dir(child)
            ):
                seen.add(child.name)
                dirs.append((child.name, child))
    return dirs


def _iter_entry_points():
    """Yield pip-installed memory provider entry points."""
    try:
        eps = importlib.metadata.entry_points()
        if hasattr(eps, "select"):
            return list(eps.select(group=ENTRY_POINTS_GROUP))
        if isinstance(eps, dict):
            return list(eps.get(ENTRY_POINTS_GROUP, []))
        return [ep for ep in eps if ep.group == ENTRY_POINTS_GROUP]
    except Exception as exc:
        logger.debug("Memory provider entry-point scan failed: %s", exc)
        return []


def find_provider_dir(name: str) -> Optional[Path]:
    """Resolve a provider name to its directory: bundled, user, project, then the
    package directory of a pip entry-point provider.

    The entry-point case matters because two of a provider's files are read from
    disk rather than imported: ``config_schema.py`` (loaded by path so the web
    server never pulls in the agent runtime) and ``cli.py`` (loaded by
    ``discover_plugin_cli_commands`` at argparse time). Without a directory a
    pip-installed provider silently loses its dashboard config panel and its
    ``hermes <provider>`` subcommands.
    """
    bundled = _MEMORY_PLUGINS_DIR / name
    if bundled.is_dir() and (bundled / "__init__.py").exists():
        return bundled
    for source_dir in (_get_user_plugins_dir(), _get_project_plugins_dir()):
        if not source_dir:
            continue
        candidate = source_dir / name
        if candidate.is_dir() and _is_memory_provider_dir(candidate):
            return candidate
    return _entry_point_package_dir(find_provider_entry_point(name))


def _entry_point_package_dir(entry_point) -> Optional[Path]:
    """The directory of an entry point's module, resolved WITHOUT importing it.

    Discovery must stay free of third-party imports: ``find_provider_dir`` runs
    from the dashboard and argparse setup, long before the operator has selected
    a provider, so importing every installed candidate would run arbitrary code
    on the strength of a package merely being present.

    Only package entry points (``pkg/__init__.py``) yield a directory — a bare
    ``module.py`` has nowhere to put a sibling ``config_schema.py``.
    """
    if entry_point is None:
        return None
    try:
        from hermes_cli.plugins import resolve_module_origin

        module_name = (entry_point.value or "").split(":")[0].strip()
        origin = resolve_module_origin(module_name)
        if not origin:
            return None
        path = Path(origin)
        return path.parent if path.name == "__init__.py" else None
    except Exception as exc:
        logger.debug("Could not resolve directory for entry point '%s': %s",
                     getattr(entry_point, "name", "?"), exc)
        return None


def find_provider_entry_point(name: str):
    """Resolve a provider name to a pip entry point, if installed."""
    return next((ep for ep in _iter_entry_points() if ep.name == name), None)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def list_memory_provider_names() -> List[str]:
    """Cheap name-only listing: directory scan plus entry-point *enumeration*
    (distribution metadata only — no provider import, no availability check),
    so it is safe at module-import time (dashboard ``memory.provider`` dropdown).
    """
    names = {name for name, _ in _iter_provider_dirs()}
    names.update(ep.name for ep in _iter_entry_points())
    return sorted(names)


def discover_memory_providers() -> List[Tuple[str, str, bool]]:
    """``[(name, description, is_available), ...]`` for directory then entry-point
    providers; bundled wins on name collisions, then user directories, then pip."""
    results = []
    for name, child in _iter_provider_dirs():
        results.append((
            name,
            _loader.read_plugin_description(child),
            _loader.probe_availability(lambda c=child: _load_provider_from_dir(c, register_skills=False)),
        ))
    seen = {name for name, _, _ in results}
    for entry_point in _iter_entry_points():
        if entry_point.name not in seen:
            seen.add(entry_point.name)
            results.append((
                entry_point.name, "",
                _loader.probe_availability(
                    lambda ep=entry_point: _load_provider_from_entry_point(ep, register_skills=False)
                ),
            ))
    return results


def load_memory_provider(
    name: str,
    *,
    register_skills: Optional[bool] = None,
) -> Optional["MemoryProvider"]:
    """Load a MemoryProvider by name (bundled, user, project, then pip entry point).

    Skills register only when *name* is the configured active provider unless
    ``register_skills`` is passed explicitly, so status/setup inspection of
    inactive providers leaves no registry side effects.

    Returns None if the provider is not found or fails to load.
    """
    if register_skills is None:
        register_skills = name == _get_active_memory_provider()

    provider_dir = find_provider_dir(name)
    entry_point = None if provider_dir else find_provider_entry_point(name)
    if not provider_dir and entry_point is None:
        logger.debug(
            "Memory provider '%s' not found in bundled, user plugins, or entry points",
            name,
        )
        return None

    def _load(_dir):
        if provider_dir:
            return _load_provider_from_dir(provider_dir, register_skills=register_skills)
        return _load_provider_from_entry_point(entry_point, register_skills=register_skills)

    return _loader.load_named(
        name, provider_dir, _load, kind="Memory provider", noun="provider", logger=logger,
    )


def _instantiate_subclass(namespace) -> Optional["MemoryProvider"]:
    """First instantiable ``MemoryProvider`` subclass found among *namespace*'s attributes."""
    from agent.memory_provider import MemoryProvider

    for attr_name in dir(namespace):
        attr = getattr(namespace, attr_name, None)
        if (isinstance(attr, type) and issubclass(attr, MemoryProvider)
                and attr is not MemoryProvider):
            try:
                return attr()
            except Exception:
                pass
    return None


def _load_provider_from_entry_point(
    entry_point,
    *,
    register_skills: bool = True,
) -> Optional["MemoryProvider"]:
    """Import a provider entry point and extract the MemoryProvider instance."""
    from agent.memory_provider import MemoryProvider

    loaded = entry_point.load()

    if isinstance(loaded, MemoryProvider):
        return loaded

    if isinstance(loaded, type) and issubclass(loaded, MemoryProvider):
        try:
            return loaded()
        except Exception:
            pass

    if hasattr(loaded, "register"):
        collector = _ProviderCollector(entry_point.name, register_skills=register_skills)
        loaded.register(collector)
        if collector.provider:
            return collector.provider

    if callable(loaded):
        try:
            provider = loaded()
            if isinstance(provider, MemoryProvider):
                return provider
        except TypeError:
            pass

        collector = _ProviderCollector(entry_point.name, register_skills=register_skills)
        loaded(collector)
        return collector.provider

    provider = _instantiate_subclass(loaded)
    if provider is None:
        logger.debug("Memory provider entry point '%s' loaded no provider", entry_point.name)
    return provider


def _load_provider_from_dir(
    provider_dir: Path,
    *,
    register_skills: bool = True,
) -> Optional["MemoryProvider"]:
    """Import a provider module and extract its MemoryProvider: ``register(ctx)``
    first (how our plugins are written), else instantiate a top-level subclass."""
    name = provider_dir.name
    mod = _loader.load_plugin_module(
        _module_name(provider_dir, name), provider_dir,
        parents=("plugins", "plugins.memory"),
        logger=logger,
        synthetic_namespace=None if _is_bundled(provider_dir) else _USER_NAMESPACE,
    )
    if mod is None:
        return None

    if hasattr(mod, "register"):
        collector = _ProviderCollector(name, register_skills=register_skills)
        try:
            mod.register(collector)
        except Exception as e:
            # A raise AFTER register_memory_provider() must not cost us the
            # provider. Falling through to the subclass scan below would
            # discard the instance the plugin configured and hand back a bare
            # second one — a silent downgrade that looks like success.
            if collector.provider is None:
                logger.debug("register() failed for %s: %s", name, e)
            else:
                logger.warning(
                    "Memory provider '%s' raised after registering (%s) — "
                    "using the registered provider; later registrations were skipped",
                    name, e,
                )
        if collector.provider:
            return collector.provider

    return _instantiate_subclass(mod)


class _ProviderCollector:
    """Plugin context for memory providers.

    Captures ``register_memory_provider`` directly — that is the one call the
    exclusive activation path owns — and delegates everything else to a real
    ``PluginContext`` (see ``__getattr__``), so a memory provider has the same
    registration surface as any other plugin.
    """

    def __init__(self, name: str, *, register_skills: bool = True):
        self.name = name
        self.provider = None
        self._register_skills = register_skills
        self._context = None

    def register_memory_provider(self, provider):
        self.provider = provider

    def register_skill(self, *args, **kwargs):
        """Forward plugin-provided skills to the general plugin registry.

        Handled explicitly rather than through ``__getattr__`` because skills
        are tracked for pruning: switching the active provider has to retract
        the skills the previous one registered, which needs the qualified name
        and resolved path recorded here.

        Gated on ``register_skills`` so merely *inspecting* an inactive
        provider — ``hermes memory status``, the setup picker — leaves no
        registry side effects behind.
        """
        if not self._register_skills:
            return
        try:
            manager_context = self._plugin_context()
            manager_context.register_skill(*args, **kwargs)
            skill_name = args[0] if args else kwargs.get("name")
            qualified_name = f"{self.name}:{skill_name}"

            from hermes_cli.plugins import get_plugin_manager

            registered_path = get_plugin_manager().find_plugin_skill(qualified_name)
            if registered_path is not None:
                _REGISTERED_MEMORY_PROVIDER_SKILLS[qualified_name] = registered_path
        except Exception as exc:
            logger.debug("Memory provider '%s' failed to register skill: %s", self.name, exc)

    def register_cli_command(self, *args, **kwargs):
        pass  # CLI registration happens via discover_plugin_cli_commands()

    def __getattr__(self, attr: str):
        """Delegate any other ``register_*`` call to a real ``PluginContext``.

        A hand-maintained stub used to silently drop calls it knew
        (``register_tool``, ``register_hook``) and raise ``AttributeError`` on
        ones it didn't (``register_auxiliary_task``), which surfaced as
        "register() failed" and cost the provider. Delegating means this can
        never drift behind ``PluginContext`` again.

        Only ``register_*`` is forwarded. Everything else raises normally, so a
        typo still fails loudly rather than being absorbed.
        """
        if not attr.startswith("register_"):
            raise AttributeError(attr)

        def _forward(*args, **kwargs):
            try:
                return self._plugin_context().__getattribute__(attr)(*args, **kwargs)
            except Exception as exc:
                # A secondary registration must not cost the provider itself —
                # by the time these run, register_memory_provider has usually
                # already handed us the instance the agent needs.
                logger.warning(
                    "Memory provider '%s' failed to %s: %s", self.name, attr, exc
                )
                return None

        return _forward

    def _plugin_context(self):
        """A real ``PluginContext`` for this provider, built once on demand.

        Lazy because the common case — a provider that only calls
        ``register_memory_provider`` — must not pay for importing the general
        plugin manager, which discovery touches on every hermes startup.
        """
        if self._context is None:
            from hermes_cli.plugins import PluginContext, PluginManifest, get_plugin_manager

            manifest = PluginManifest(name=self.name, key=self.name)
            self._context = PluginContext(manifest, get_plugin_manager())
        return self._context


def _get_active_memory_provider() -> Optional[str]:
    """Active provider name from config.yaml (``memory.provider``), or None. Reads config only."""
    try:
        from hermes_cli.config import load_config
        config = load_config()
        return cfg_get(config, "memory", "provider") or None
    except Exception:
        return None


def _prune_inactive_memory_provider_skills(
    active_provider: Optional[str] = None,
) -> None:
    """Remove tracked skills that no longer belong to the active provider."""
    if active_provider is None:
        active_provider = _get_active_memory_provider()

    from hermes_cli.plugins import get_plugin_manager

    manager = get_plugin_manager()
    for qualified_name, registered_path in list(
        _REGISTERED_MEMORY_PROVIDER_SKILLS.items()
    ):
        namespace, _, _ = qualified_name.partition(":")
        if namespace == active_provider:
            continue
        if manager.find_plugin_skill(qualified_name) == registered_path:
            manager.remove_plugin_skill(qualified_name)
        _REGISTERED_MEMORY_PROVIDER_SKILLS.pop(qualified_name, None)


def discover_plugin_cli_commands() -> List[dict]:
    """CLI commands for the **active** memory plugin only (``memory.provider``).

    Lightweight: imports only the active plugin's ``cli.py`` (looking for
    ``register_cli(subparser)``), never the provider module, so it is safe during
    argparse setup. Returns at most one dict with keys ``name``, ``help``,
    ``description``, ``setup_fn``, ``handler_fn``, ``plugin``.
    """
    results: List[dict] = []
    if not _MEMORY_PLUGINS_DIR.is_dir():
        return results

    active_provider = _get_active_memory_provider()
    if not active_provider:
        return results

    plugin_dir = find_provider_dir(active_provider)
    if not plugin_dir:
        return results

    cli_file = plugin_dir / "cli.py"
    if not cli_file.exists():
        return results

    module_name = _module_name(plugin_dir, active_provider) + ".cli"
    try:
        if module_name in sys.modules:
            cli_mod = sys.modules[module_name]
        else:
            if not _is_bundled(plugin_dir):
                # cli.py imports as _hermes_user_memory.<name>.cli, usually before
                # the provider itself is loaded. Register its parent packages so
                # relative imports inside cli.py resolve without executing the
                # plugin's __init__.py. The shell has no __file__, so
                # _load_provider_from_dir() still loads the real module later.
                _register_synthetic_package(_USER_NAMESPACE, [])
                _register_synthetic_package(
                    f"{_USER_NAMESPACE}.{active_provider}", [str(plugin_dir)]
                )
            spec = importlib.util.spec_from_file_location(module_name, str(cli_file))
            if not spec or not spec.loader:
                return results
            cli_mod = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = cli_mod
            spec.loader.exec_module(cli_mod)

        register_cli = getattr(cli_mod, "register_cli", None)
        if not callable(register_cli):
            return results

        desc = _loader.read_plugin_description(plugin_dir)
        handler_fn = getattr(cli_mod, f"{active_provider}_command", None) or \
                     getattr(cli_mod, "honcho_command", None)

        results.append({
            "name": active_provider,
            "help": desc or f"Manage {active_provider} memory plugin",
            "description": desc or "",
            "setup_fn": register_cli,
            "handler_fn": handler_fn,
            "plugin": active_provider,
        })
    except Exception as e:
        logger.debug("Failed to scan CLI for memory plugin '%s': %s", active_provider, e)

    return results
