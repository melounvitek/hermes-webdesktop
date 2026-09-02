"""Hermes Plugin System — discovers, loads, and manages plugins.

Sources, later overriding earlier on key collision: bundled ``<repo>/plugins/<name>/`` (``memory/``
and ``context_engine/`` have their own discovery), user ``~/.hermes/plugins/<name>/``, project
``./.hermes/plugins/<name>/`` (opt-in via ``HERMES_ENABLE_PROJECT_PLUGINS``), and pip packages in
the ``hermes_agent.plugins`` entry-point group. A directory plugin needs a ``plugin.yaml`` manifest
and an ``__init__.py`` exposing ``register(ctx)``. Plugins register callbacks for ``VALID_HOOKS``
(core fires ``invoke_hook(name, **kwargs)``) and tools via ``PluginContext.register_tool()``.
"""

from __future__ import annotations

import asyncio
import builtins
import importlib.metadata
import inspect
import json
import logging
import os
import queue
import re
import sys
import threading
import types
from contextlib import suppress
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Set, Tuple, Union

from hermes_constants import get_hermes_home, hermes_home_key
from registration_lifecycle import replacement_coordinator
from utils import env_var_enabled
from hermes_cli.config import cfg_get, load_config_readonly
from hermes_cli.middleware import VALID_MIDDLEWARE
from hermes_cli.plugin_capabilities import (  # noqa: F401 — re-exported
    CAPABILITY_REGISTRY,
    VALID_CAPABILITY_IDS,
    plugin_capability_granted,
)
from hermes_cli.relay_plugin_cutover import RELAY_PLUGINS_CONFIG_ENV, legacy_relay_plugin_keys
from hermes_cli.plugins_manifest import (  # noqa: F401 — re-exported
    manifest_key,
    parse_manifest_file,
    portable_plugin_manifest,
    _CONFIG_SCHEMA_TYPES,
    SUPPORTED_MANIFEST_VERSION,
    PluginManifest,
    _portable_skill_namespace,
    resolve_module_origin,
    resolve_plugin_load_order,
    validate_config_schema,
)
from hermes_cli.plugins_discovery import (  # noqa: F401 — re-exported
    collect_directory_manifests,
    gate_manifest,
    scan_directory,
    ENTRY_POINT_CAPABILITIES_GROUP,
    ENTRY_POINTS_GROUP,
    _get_disabled_plugins,
    _get_enabled_plugins,
    discover_entrypoint_manifests,
)
from hermes_cli.plugins_loader import (  # noqa: F401 — re-exported
    PluginLoaderMixin,
    _NS_PARENT,
    _MODULE_NAMESPACE_LOCK,
    _BARE_MODULE_SCOPE,
    _evict_modules,
    _serialized_replacement,
    _plugin_home_scope,
)
from hermes_cli.plugins_dispatch import (  # noqa: F401 — re-exported
    PluginDispatchMixin,
    _HOOK_TIMEOUT_SUPPRESSION_SECONDS,
    _PRE_TOOL_CALL_TIMEOUT_BLOCK_MESSAGE,
    SYSTEM_PROMPT_SECTION_POSITIONS,
    DEFAULT_SYSTEM_PROMPT_SECTION_MAX_CHARS,
    MAX_SYSTEM_PROMPT_SECTION_CHARS,
    MAX_SYSTEM_PROMPT_SECTIONS,
    MAX_SYSTEM_PROMPT_SECTIONS_TOTAL_CHARS,
    PLUGIN_SECTIONS_START,
    PLUGIN_SECTIONS_END,
    is_valid_system_prompt_section_id,
    format_system_prompt_section,
    format_system_prompt_sections,
    HERMES_EVENT_NAMESPACE,
    _EVENT_EMIT_DEPTH_CAP,
    _EVENT_PENDING_CAP,
    PluginSystemPromptSection,
    RenderedPluginSystemPromptSection,
    _EventSubscription,
    _HOOK_CALLBACK_TIMEOUT_SECS,
    _MAX_HOOK_CALLBACK_TIMEOUT_SECS,
)
from hermes_cli.plugins_ledger import (  # noqa: F401 — re-exported
    PluginLedgerMixin,
    PluginRegistration,
)
from hermes_cli.plugins_state import (  # noqa: F401 — re-exported
    PluginState,
    _locked_plugin_state,
    _nested_plugin_mapping,
    _nested_plugin_value,
    _plugin_relative_segments,
)


def get_bundled_plugins_dir() -> Path:
    """Bundled ``plugins/`` dir: ``HERMES_BUNDLED_PLUGINS`` (Nix wrapper / packaged installs, read-only
    store paths) first, else the in-repo path."""
    env_override = os.getenv("HERMES_BUNDLED_PLUGINS")
    if env_override:
        return Path(env_override)
    return Path(__file__).resolve().parent.parent / "plugins"


class PluginToolOverrideError(PermissionError):
    """Plugin tried to override a built-in tool without ``plugins.entries.<id>.allow_tool_override``."""


logger = logging.getLogger(__name__)


# ``HERMES_PLUGINS_DEBUG=1`` tees verbose discovery logs to stderr in addition to agent.log. Read
# once at import; tests flip it mid-process via ``_install_plugin_debug_handler(force=True)``.
_PLUGINS_DEBUG = os.getenv("HERMES_PLUGINS_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
_DEBUG_HANDLER_INSTALLED = False


def _install_plugin_debug_handler(force: bool = False) -> None:
    """When HERMES_PLUGINS_DEBUG is on, tee plugin logs to stderr at DEBUG (once per process)."""
    global _DEBUG_HANDLER_INSTALLED, _PLUGINS_DEBUG
    if force:
        _PLUGINS_DEBUG = os.getenv("HERMES_PLUGINS_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
    if not _PLUGINS_DEBUG or _DEBUG_HANDLER_INSTALLED:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(logging.Formatter("[plugins] %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    logger.propagate = True
    _DEBUG_HANDLER_INSTALLED = True
    logger.debug("HERMES_PLUGINS_DEBUG=1 — verbose plugin discovery logging enabled")


_install_plugin_debug_handler()

VALID_HOOKS: Set[str] = {
    "pre_tool_call", "post_tool_call", "transform_terminal_output", "transform_tool_result",
    # Return a string to replace the response text (first non-None wins) or None to leave it.
    "transform_llm_output", "pre_llm_call", "post_llm_call",
    # Streaming observers fired off the token path by agent.plugin_stream_hooks; payloads are
    # immutable normalized text/lifecycle and cannot transform the stream.
    "on_stream_start", "on_stream_delta", "on_stream_end", "on_interim_message",
    # Fired once per turn when the agent edited code and is about to verify/finish. Return
    # {"action": "continue", "message"} (or Claude-Code Stop shape {"decision": "block", "reason"})
    # to keep going; anything else finishes. Bounded by agent.max_verify_nudges.
    "pre_verify", "pre_api_request", "post_api_request", "api_request_error",
    # Fired once per failed API call BEFORE agent/error_classifier.classify_api_error(). Kwargs:
    # provider, model, status_code, error_type, error_code, error_message, error_body, error,
    # approx_tokens, context_length, num_messages. Return None, or {"reason": <FailoverReason name>
    # (required), "retryable"/"should_compress"/"should_rotate_credential"/"should_fallback": bool,
    # "message": str, "error_context": dict}. Run-all-then-pick-first (see
    # get_plugin_error_classification). Privacy: error_message/error_body may be unredacted.
    "transform_api_error_classification", "on_session_start", "on_session_end",
    "on_session_finalize", "on_session_reset",
    # Successful skill lifecycle facts (local skill name visible to plugins).
    "on_skill_lifecycle", "subagent_start", "subagent_stop",
    # Once per incoming MessageEvent, after the internal-event guard, BEFORE auth/pairing and
    # dispatch. Kwargs: event, gateway, session_store. Return {"action": "skip", "reason"} -> drop;
    # {"action": "rewrite", "text"} -> replace event.text; {"action": "allow"} / None -> normal.
    "pre_gateway_dispatch",
    # Approval observers (tools/approval.py). Return values ignored — plugins cannot veto or
    # pre-answer (use pre_tool_call). Kwargs: command, description, pattern_key, pattern_keys,
    # session_key, surface: "cli" | "gateway" | "smart"; post_approval_response adds choice
    # ("once"|"session"|"always"|"deny"|"timeout"|"smart_approve"|"smart_deny") and decided_by.
    "pre_approval_request", "post_approval_response",
    # Fired by transcribe_audio after provider resolution, BEFORE any backend runs. Kwargs:
    # file_path, provider, model, language, prompt, source. Return None or a dict mutating
    # prompt/language/model (registration order, last-writer-wins; file_path is read-only).
    "pre_transcription",
    # Kanban task observers (hermes_cli.kanban_db), fired AFTER the DB commit so a slow plugin never
    # holds the SQLite write lock. Return values ignored. Process matters: claimed fires in the
    # DISPATCHER right before spawn; completed/blocked fire in the WORKER (or whichever process
    # drove it). Kwargs: task_id, board, assignee, run_id, profile_name; completed adds summary,
    # blocked adds reason.
    "kanban_task_claimed", "kanban_task_completed", "kanban_task_blocked",
    # Kanban worker/mutation/tick observers; return values ignored; fire sites short-circuit on
    # has_hook(). Kwargs: task_id, profile_name, board, assignee, run_id plus:
    # worker_spawned (DISPATCHER, after PID persisted, inside the dispatch lock — stay fast):
    #   worker_pid, workspace_path (privacy: project layout/usernames).
    "on_kanban_worker_spawned",
    # worker_exited (tick-derived on dead-PID reclaim): worker_pid, exit_kind ("clean_exit" |
    #   "rate_limited" | "nonzero_exit" | "signaled" | "unknown"), exit_code, outcome, retry_status.
    "on_kanban_worker_exited",
    # worker_stale_claim (TTL-expired claim reclaimed; live-PID extensions do NOT fire):
    #   worker_pid, heartbeat_stale, retry_status.
    "on_kanban_worker_stale_claim",
    # task_updated (committed task-row write outside claim/complete/block, in whichever process
    #   committed it): changed_fields — field NAMES only, never values.
    "on_kanban_task_updated",
    # dispatch_tick: once per dispatch_once, strictly AFTER the dispatch lock is released. Kwargs:
    #   board, profile_name, dry_run, outcome ("ok"|"skipped_locked"|"idle"), result: DispatchResult
    #   (privacy: task ids, assignees, workspace paths).
    "on_kanban_dispatch_tick",
    # Gateway platform-boundary observer: normalized envelopes only, never raw SDK objects or
    # adapter handles. Kwargs: platform, event_type, payload (event_type-local; see hooks.md).
    # New event types land only together with real fire-sites.
    "gateway_platform_event",
    # Fired BEFORE a recognized slash command's handler on CLI and gateway canonical dispatch.
    # Return values IGNORED in v1. Deliberately NOT fired for the gateway's running-agent intercept
    # path (/stop, /approve, busy_policy) — a slow/hostile plugin must not touch the operator's
    # escape hatches. Kwargs: surface, command (canonical), alias_used, args_raw, session_key,
    # platform.
    "pre_command",
}

# Hooks whose directive the shell-hook response parser has no channel for. VALID_HOOKS doubles as
# the shell-hook allow-list, so these are refused loudly instead of having output silently ignored.
SHELL_UNSUPPORTED_HOOKS: Set[str] = {"transform_api_error_classification"}

_env_enabled = env_var_enabled  # imported by plugins/memory


@dataclass
class LoadedPlugin:
    """Runtime state for a single loaded plugin."""

    manifest: PluginManifest
    module: Optional[types.ModuleType] = None
    tools_registered: List[str] = field(default_factory=list)
    hooks_registered: List[str] = field(default_factory=list)
    middleware_registered: List[str] = field(default_factory=list)
    commands_registered: List[str] = field(default_factory=list)
    enabled: bool = False
    error: Optional[str] = None
    # Bundled platform recorded as a not-yet-imported loader (see _register_deferred_platform).
    deferred: bool = False


class PluginContext:
    """Facade given to plugins so they can register tools and hooks."""

    def __init__(self, manifest: PluginManifest, manager: "PluginManager"):
        self.manifest = manifest
        self._manager = manager
        # Lazy-built facades (see the matching properties).
        self._llm: Any = None
        self._subagent_lifecycle: Any = None
        self._state: PluginState | None = None
        self._platform_actions: Any = None

    @property
    def plugin_id(self) -> str:
        """Return the effective registry id used for this plugin's namespaces."""
        return manifest_key(self.manifest)

    def has_plugin(self, plugin_id: str) -> bool:
        """Return True when another plugin is loaded and enabled (runtime probe for advisory
        ``requires_plugins``). Matches on registry key or manifest name."""
        return any(
            loaded.enabled and (key == plugin_id or loaded.manifest.name == plugin_id)
            for key, loaded in self._manager._plugins.items()
        )

    def get_config(self, key: str, default: Any = None) -> Any:
        """Read plugin-relative ``plugins.entries.<plugin_id>.settings.<key>`` (falls back to the
        legacy ``config`` subtree for migration compatibility)."""
        try:
            segments = _plugin_relative_segments(key)
        except ValueError:
            logger.warning("Rejected config path %r from plugin %s", key, self.plugin_id)
            raise
        from hermes_cli.config import load_config_readonly
        config = load_config_readonly() or {}
        plugins = config.get("plugins") if isinstance(config, Mapping) else None
        entries = plugins.get("entries") if isinstance(plugins, Mapping) else None
        entry = entries.get(self.plugin_id) if isinstance(entries, Mapping) else None
        if not isinstance(entry, Mapping):
            return default
        missing = object()
        value = _nested_plugin_value(entry.get("settings"), segments, missing)
        if value is not missing:
            return value
        return _nested_plugin_value(entry.get("config"), segments, default)

    def set_config(self, key: str, value: Any) -> None:
        """Atomically write one value in this plugin's ``settings`` subtree."""
        try:
            segments = _plugin_relative_segments(key)
        except ValueError:
            logger.warning("Rejected config path %r from plugin %s", key, self.plugin_id)
            raise
        from hermes_cli import config as config_mod
        if config_mod.is_managed():
            raise PermissionError("Plugin settings cannot be changed in a managed install")
        from hermes_cli import managed_scope
        dotted_path = ".".join(("plugins", "entries", self.plugin_id, "settings", *segments))
        if managed_scope.is_key_managed(dotted_path):
            raise PermissionError(f"Plugin setting {dotted_path!r} is administrator-managed")
        full_path = ("plugins", "entries", self.plugin_id, "settings", *segments)
        partial = _nested_plugin_mapping(full_path[:4], _nested_plugin_mapping(segments, value))
        # The lock covers merge-read plus atomic save so sibling plugin writes (threads or
        # processes) cannot race between the two steps.
        with _locked_plugin_state(config_mod.get_config_path()):
            with config_mod._CONFIG_LOCK:
                # Fail closed on malformed YAML: save_config degrades parse failures to {} — safe
                # for reads, destructive for read-modify-write.
                config_mod.read_user_config_raw()
                config_mod.save_config(partial, preserve_keys={full_path}, merge_existing=True)

    @property
    def state(self) -> PluginState:
        """Return this plugin's profile-scoped durable JSON state facade."""
        if self._state is None:
            self._state = PluginState(self.plugin_id, self.manifest.skill_namespace)
        return self._state

    @property
    def platform_actions(self):
        """Capability-gated platform action facade (``add_reaction``, ``set_thread_title``). Every call
        re-checks ``gateway.platform_actions`` (legacy ``plugins.entries.<id>.allow_platform_actions``,
        default OFF) and returns ``{"ok": bool, ...}`` — verbs never raise into hook dispatch; no adapter
        handles or raw SDK objects."""
        if self._platform_actions is None:
            from hermes_cli.platform_actions import PlatformActions

            self._platform_actions = PlatformActions(self.plugin_id)
        return self._platform_actions

    def _wrong_type(self, obj: Any, base_class: type, label: str, article: str = "a") -> bool:
        """Warn-and-ignore gate shared by every registrar that requires a base class."""
        if isinstance(obj, base_class):
            return False
        logger.warning(
            "Plugin '%s' tried to register %s %s that does not inherit from %s. Ignoring.",
            self.manifest.name, article, label, base_class.__name__,
        )
        return True

    def _refuse(self, what: str) -> ValueError:
        """``ValueError`` for a malformed registration (``what`` completes "tried to register ...")."""
        return ValueError(f"Plugin '{self.manifest.name}' tried to register {what}.")

    def _track(
        self, kind: str, key: str, release: Callable[[], None], *, persistent: bool = False,
    ) -> PluginRegistration:
        """Record host-owned cleanup for a successful registration (see
        :meth:`PluginManager._track_registration` for ``persistent``)."""
        return self._manager._track_registration(
            self.manifest, kind, key, release, persistent=persistent
        )

    def _track_replacement(
        self, kind: str, key: str, *, slot: tuple, current: Any, previous: Any,
        restore: Callable[[Any], bool],
    ) -> PluginRegistration:
        """Track one generation in a replaceable manager-local registration slot."""
        lease = replacement_coordinator.acquire(
            slot, current=current, previous=previous, restore=restore
        )
        return self._track(kind, key, lease.dispose)

    def _track_mapping_entry(
        self, kind: str, key: str, mapping: Dict[str, Any], entry: Any, previous: Any
    ) -> PluginRegistration:
        """Store ``entry`` under ``key`` in a manager-local mapping and lease the slot; unload restores
        ``previous`` (or removes the key) only while ``entry`` is still current."""
        mapping[key] = entry
        return self._track_replacement(
            kind, key, slot=("manager_mapping", builtins.id(mapping), key),
            current=entry, previous=previous,
            restore=lambda replacement: self._manager._restore_mapping(
                mapping, key, entry, replacement
            ),
        )

    def _register_scoped_provider(
        self, provider: Any, *, kind: str, base_class: type, registry: Any, label: str,
        article: str = "a", normalize: Optional[Callable[[str], str]] = lambda n: n.strip(),
        register: Optional[Callable[..., Any]] = None, reject_message: Optional[str] = None,
    ) -> Optional[PluginRegistration]:
        """Shared body of the ``register_<category>_provider`` methods: type-check (warn + ignore),
        register in the scope-keyed ``registry``, lease the slot so unload restores the displaced entry.
        ``None`` when the registry refused/replaced the provider (``ValueError`` with ``reject_message``
        set, or a falsy ``register``)."""
        if self._wrong_type(provider, base_class, label, article):
            return None
        registry_name = provider.name if normalize is None else normalize(provider.name)
        scope = self._manager.scope_key
        previous = registry.snapshot_registration(registry_name, scope=scope)
        register_fn = register or registry.register_provider
        try:
            accepted = register_fn(provider, scope=scope)
        except ValueError as exc:
            if reject_message is None:
                raise
            logger.warning(reject_message, self.manifest.name, exc)
            return None
        if register is not None and not accepted:
            return None
        if registry.snapshot_registration(registry_name, scope=scope) is not provider:
            return None
        handle = self._manager._track_scoped_registration(
            self.manifest, kind, registry_name, registry, provider, previous
        )
        logger.info("Plugin '%s' registered %s: %s", self.manifest.name, label, registry_name)
        return handle

    @property
    def llm(self) -> Any:
        """Host-owned :class:`agent.plugin_llm.PluginLlm` facade: completions on the user's active
        model/auth. Overrides (model, agent id, auth profile) are fail-closed, gated by
        ``plugins.entries.<plugin_id>.llm.*``."""
        if self._llm is None:
            from agent.plugin_llm import PluginLlm
            self._llm = PluginLlm(plugin_id=self.plugin_id)
        return self._llm

    @property
    def subagent_lifecycle(self) -> Any:
        """Plugin-safe subagent lifecycle service: serializable handles and immutable snapshots,
        never a live agent or private registry."""
        if self._subagent_lifecycle is None:
            from agent.subagent_lifecycle import (
                SubagentLifecycleService, get_active_subagent_parent,
            )
            self._subagent_lifecycle = SubagentLifecycleService(get_active_subagent_parent)
        return self._subagent_lifecycle

    @property
    def profile_name(self) -> str:
        """Active profile name (``"default"``, the ``~/.hermes/profiles/<name>`` id, or ``"custom"``),
        derived from ``HERMES_HOME`` — not ``_cli_ref``, which is None outside the interactive CLI —
        so gateway and kanban workers get it too."""
        try:
            from hermes_cli.profiles import get_active_profile_name
            return get_active_profile_name()
        except Exception:
            return "default"

    def on_unload(self, callback: Callable[[], None]) -> PluginRegistration:
        """Register a cleanup callback for unload: runs in reverse acquisition order interleaved
        with registration teardown; exceptions are logged, never propagated."""
        if not callable(callback):
            raise TypeError("on_unload callback must be callable")
        handle = self._track("on_unload", getattr(callback, "__name__", "callback"), callback)
        logger.debug("Plugin %s registered on_unload callback", self.manifest.name)
        return handle

    def spawn_task(self, coro, *, name: Optional[str] = None) -> "asyncio.Task":
        """Spawn a supervised asyncio task; unload/force reload cancels it. Needs a running loop."""
        if not asyncio.iscoroutine(coro):
            raise TypeError("spawn_task expects a coroutine")
        loop = asyncio.get_running_loop()
        task_name = name or f"plugin:{self.plugin_id}:task"
        task = loop.create_task(coro, name=task_name)

        def _cancel_task() -> None:
            if not task.done():
                task.cancel()

        handle = self._track("background_task", task_name, _cancel_task)
        task.add_done_callback(lambda _t: handle.dispose())
        logger.debug("Plugin %s spawned supervised task: %s", self.manifest.name, task_name)
        return task

    def register_approval_transport(self, name: str, present_fn: Callable) -> None:
        """Register a human approval transport, inactive until ``security.approval.transport:
        <name>`` selects it. It receives a redacted ``ApprovalRequest`` and returns only a
        correlated decision; policy and persistence stay host-owned. ``present_fn`` may be async."""
        self._manager.register_approval_transport(name, present_fn, plugin_id=self.plugin_id)
        # Record ownership so unload/force-reload removes this transport. Duplicate names are
        # rejected above (raise), so there is never a displaced previous entry to restore.
        clean = str(name).strip().lower()
        entry = self._manager._approval_transports.get(clean)
        if entry is not None:
            self._track_mapping_entry(
                "approval_transport", clean, self._manager._approval_transports, entry, None
            )

    @_serialized_replacement
    def register_tool(
        self, name: str, toolset: str, schema: dict, handler: Callable,
        check_fn: Callable | None = None, requires_env: list | None = None, is_async: bool = False,
        description: str = "", emoji: str = "", override: bool = False,
    ) -> Optional[PluginRegistration]:
        """Register a tool in the global registry and track it as plugin-provided. ``override=True``
        replaces a same-named built-in (without it a name claimed by another toolset is rejected) and
        needs operator opt-in via ``plugins.entries.<plugin_id>.allow_tool_override: true`` — otherwise
        any enabled plugin could silently replace a privileged built-in like ``write_file``."""
        if override and not self._tool_override_allowed(name):
            raise PluginToolOverrideError(
                f"Plugin {self.manifest.name!r} cannot override built-in tool " f"{name!r}. Set "
                f"plugins.entries.{self.plugin_id}.allow_tool_override: true "
                f"in config.yaml to allow this plugin to replace built-in tools."
            )

        from tools.registry import registry
        scope = self._manager.scope_key
        previous = registry.snapshot_registration(name, scope=scope)
        effective = registry.get_entry(name, scope=scope)
        if previous is None and effective is not None and not override:
            logger.warning(
                "Plugin %s tried to shadow global tool %s without override=True",
                self.manifest.name, name,
            )
            return None
        registry.register(
            name=name, toolset=toolset, schema=schema, handler=handler, check_fn=check_fn,
            requires_env=requires_env, is_async=is_async, description=description, emoji=emoji,
            override=override, scope=scope,
        )
        registered = registry.snapshot_registration(name, scope=scope)
        if (
            registered is not None and registered is not previous and registered.handler is handler
        ):
            self._manager._plugin_tool_names.add(name)
            handle = self._manager._track_scoped_registration(
                self.manifest, "tool", name, registry, registered, previous,
                finalize=lambda: self._manager._remove_tool_name_if_unowned(name),
            )
        else:
            handle = None
        logger.debug(
            "Plugin %s registered tool: %s%s",
            self.manifest.name, name, " (override)" if override else "",
        )
        return handle

    def has_capability(self, capability: str) -> bool:
        """True when *capability* is live for this plugin (probe, then degrade gracefully). Bundled
        plugins are trusted for ``tools.override``; otherwise granted_capabilities or the legacy
        ``allow_*`` key decides. Unknown ids / unreadable consent -> False (fail closed)."""
        source = getattr(self.manifest, "source", "") or ""
        if source == "bundled" and capability == "tools.override":
            return True
        return plugin_capability_granted(self.plugin_id, capability)

    def call_mcp(
        self, server: str, tool: str, arguments: Optional[Dict[str, Any]] = None,
        timeout: float = 30,
    ) -> Dict[str, Any]:
        """Call ``tool`` on MCP ``server`` synchronously through :mod:`tools.mcp_tool`'s native client
        (same trust gates, breaker, reconnect — never a parallel connection). Default-off per-server
        grant: servers not in ``plugins.entries.<plugin_id>.mcp_allowlist`` raise ``PermissionError``.
        ``timeout`` clamps to 1–600s. Returns ``{"ok": True, "result"}`` / ``{"ok": False, "error"}``;
        results over ~64KB are truncated with a marker."""
        allowlist = self._mcp_allowlist(self.plugin_id)
        if server not in allowlist:
            raise PermissionError(
                f"Plugin {self.manifest.name!r} is not allowed to call MCP "
                f"server {server!r}. Add it to "
                f"plugins.entries.{self.plugin_id}.mcp_allowlist in config.yaml "
                f"to grant access (default is no MCP access)."
            )

        try:
            timeout = float(timeout)
        except (TypeError, ValueError):
            timeout = 30.0
        timeout = max(1.0, min(timeout, 600.0))

        from tools.mcp_tool import _make_tool_handler
        handler = _make_tool_handler(server, tool, timeout)
        raw = handler(dict(arguments or {}))

        logger.debug(
            "Plugin %s called MCP %s/%s (timeout=%ss, %d chars returned)",
            self.manifest.name, server, tool, timeout, len(raw or ""),
        )
        return self._mcp_envelope(raw)

    _MCP_RESULT_CHAR_CAP = 65536

    @classmethod
    def _mcp_envelope(cls, raw: Any) -> Dict[str, Any]:
        """Normalize an MCP handler result string into a stable envelope."""
        if not isinstance(raw, str):
            raw = "" if raw is None else str(raw)
        if len(raw) > cls._MCP_RESULT_CHAR_CAP:
            raw = raw[: cls._MCP_RESULT_CHAR_CAP] + "… [truncated]"
            truncated = True
        else:
            truncated = False
        parsed: Any = None
        try:
            parsed = json.loads(raw)
        except (ValueError, TypeError):
            parsed = None
        if isinstance(parsed, dict) and "error" in parsed:
            envelope: Dict[str, Any] = {"ok": False, "error": parsed["error"]}
        elif isinstance(parsed, dict) and "result" in parsed:
            envelope = {"ok": True, "result": parsed["result"]}
            if "structuredContent" in parsed:
                envelope["structuredContent"] = parsed["structuredContent"]
        else:
            envelope = {"ok": True, "result": parsed if parsed is not None else raw}
        if truncated:
            envelope["truncated"] = True
        return envelope

    @staticmethod
    def _mcp_allowlist(plugin_id: str) -> List[str]:
        """Operator-granted MCP server allowlist; missing/unreadable -> [] (default-deny)."""
        try:
            from hermes_cli.config import load_config
            cfg = load_config() or {}
        except Exception:
            return []
        entries = (cfg.get("plugins") or {}).get("entries") or {}
        entry = entries.get(plugin_id) or {}
        allowlist = entry.get("mcp_allowlist")
        if not isinstance(allowlist, list):
            return []
        return [str(item) for item in allowlist]

    def _tool_override_allowed(self, tool_name: str) -> bool:
        """Whether this plugin may override built-in tools: bundled plugins are trusted (a maintainer
        choice, not privilege escalation); others need ``tools.override`` via
        :func:`plugin_capability_granted` (granted_capabilities OR legacy ``allow_tool_override: true``)."""
        source = getattr(self.manifest, "source", "") or ""
        if source == "bundled":
            return True
        try:
            from hermes_cli.config import load_config

            with _plugin_home_scope(self._manager.home_path):
                cfg = load_config() or {}
        except Exception:
            return False  # fail closed: better to break the override than silently grant it
        # Pass THIS manager's profile-scoped config so a multi-profile process never consults the
        # active profile's consent state instead.
        return plugin_capability_granted(self.plugin_id, "tools.override", config=cfg)

    def inject_message(
        self, content: str, role: str = "user", *, session_key: str | None = None,
    ) -> bool:
        """Inject a message into a CLI or gateway conversation (new turn if idle, interrupt if running).
        Gateway injection needs an existing ``session_key`` plus
        ``plugins.entries.<plugin_id>.allow_gateway_injection``; ``True`` means the gateway accepted the
        request for async dispatch, not that delivery completed."""
        cli = self._manager._cli_ref
        msg = content if role == "user" else f"[{role}] {content}"

        if cli is not None:
            if getattr(cli, "_agent_running", False):
                cli._interrupt_queue.put(msg)
            else:
                cli._pending_input.put(msg)
            return True

        if not session_key:
            logger.warning("inject_message: gateway mode requires an existing session_key")
            return False
        if not self._gateway_injection_allowed():
            logger.warning(
                "inject_message: gateway injection denied for plugin %s; set "
                "plugins.entries.%s.allow_gateway_injection: true to allow it", self.plugin_id,
                self.plugin_id,
            )
            return False

        if not self._manager.has_gateway_message_injector:
            logger.warning("inject_message: no live gateway is available")
            return False

        try:
            return bool(
                self._manager.inject_gateway_message(
                    session_key=session_key, content=msg, plugin_id=self.plugin_id,
                )
            )
        except Exception:
            logger.warning(
                "inject_message: gateway scheduling failed for plugin %s", self.plugin_id,
                exc_info=True,
            )
            return False

    def _gateway_injection_allowed(self) -> bool:
        """Return whether this plugin may trigger gateway session turns."""
        try:
            cfg = load_config_readonly() or {}
        except Exception:
            return False

        return (
            cfg_get(
                cfg, "plugins", "entries", self.plugin_id, "allow_gateway_injection", default=False,
            ) is True
        )

    @_serialized_replacement
    def register_cli_command(
        self, name: str, help: str, setup_fn: Callable, handler_fn: Callable | None = None,
        description: str = "",
    ) -> PluginRegistration:
        """Register a CLI subcommand (``hermes <name> ...``). *setup_fn* receives the argparse
        subparser; *handler_fn* becomes ``set_defaults(func=...)``."""
        previous = self._manager._cli_commands.get(name)
        entry = {
            "name": name, "help": help, "description": description, "setup_fn": setup_fn,
            "handler_fn": handler_fn, "plugin": self.manifest.name, "plugin_key": self.plugin_id,
        }
        handle = self._track_mapping_entry(
            "cli_command", name, self._manager._cli_commands, entry, previous
        )
        logger.debug("Plugin %s registered CLI command: %s", self.manifest.name, name)
        return handle

    @_serialized_replacement
    def register_command(
        self, name: str, handler: Callable, description: str = "", args_hint: str = "",
        argument_mode: str | None = None,
    ) -> Optional[PluginRegistration]:
        """Register an in-session slash command (``/name``) for CLI and gateway sessions. Handler:
        ``fn(raw_args: str) -> str | None`` (sync or async). ``args_hint`` (e.g. ``"<file>"``) lets
        adapters like Discord surface an argument field; without it the command registers parameterless
        there but still accepts trailing text as free-form chat."""
        clean = name.lower().strip().lstrip("/").replace(" ", "-")
        if not clean:
            logger.warning(
                "Plugin '%s' tried to register a command with an empty name.", self.manifest.name,
            )
            return

        # Reject if it conflicts with a built-in command
        try:
            from hermes_cli.commands import resolve_command
            if resolve_command(clean) is not None:
                logger.warning(
                    "Plugin '%s' tried to register command '/%s' which conflicts "
                    "with a built-in command. Skipping.", self.manifest.name, clean,
                )
                return
        except Exception:
            pass

        previous = self._manager._plugin_commands.get(clean)
        hint = (args_hint or "").strip()
        mode = argument_mode if argument_mode in {"options", "text", "mixed"} else (
            "text" if hint else None
        )
        entry = {
            "handler": handler, "description": description or "Plugin command",
            "plugin": self.manifest.name, "plugin_key": self.plugin_id, "args_hint": hint,
            "argument_mode": mode,
        }
        handle = self._track_mapping_entry(
            "command", clean, self._manager._plugin_commands, entry, previous
        )
        logger.debug("Plugin %s registered command: /%s", self.manifest.name, clean)
        return handle

    def dispatch_tool(self, tool_name: str, args: dict, **kwargs) -> str:
        """Dispatch a tool call through the registry with the parent agent (when available)
        resolved automatically; returns the handler's JSON string. ``kwargs`` forward to dispatch."""
        from tools.registry import registry
        # In gateway mode _cli_ref is None — tools degrade gracefully (no spinner, TERMINAL_CWD).
        if "parent_agent" not in kwargs:
            cli = self._manager._cli_ref
            agent = getattr(cli, "agent", None) if cli else None
            if agent is not None:
                kwargs["parent_agent"] = agent

        return registry.dispatch(tool_name, args, scope=self._manager.scope_key, **kwargs)

    @_serialized_replacement
    def register_context_engine(self, engine) -> Optional[PluginRegistration]:
        """Register the (single) ``agent.context_engine.ContextEngine`` replacing the built-in
        ContextCompressor; a second registration is rejected with a warning."""
        if self._manager._context_engine is not None:
            logger.warning(
                "Plugin '%s' tried to register a context engine, but one is "
                "already registered. Only one context engine plugin is allowed.",
                self.manifest.name,
            )
            return
        from agent.context_engine import ContextEngine
        if self._wrong_type(engine, ContextEngine, "context engine"):
            return
        previous = self._manager._context_engine
        self._manager._context_engine = engine
        handle = self._track_replacement(
            "context_engine", engine.name,
            slot=("manager_value", id(self._manager), "_context_engine"),
            current=engine, previous=previous,
            restore=lambda replacement: self._manager._restore_value(
                "_context_engine", engine, replacement
            ),
        )
        logger.info("Plugin '%s' registered context engine: %s", self.manifest.name, engine.name)
        return handle

    def register_context_reference(self, provider) -> None:
        """Register a :class:`agent.context_references.ContextReferenceProvider`; ``provider.prefix``
        defines ``@<prefix>:``. Built-in prefixes (diff, staged, file, folder, git, url) are
        rejected."""
        from agent.context_references import (
            ContextReferenceProvider as _CRP, register_context_reference_provider as _register,
        )
        if self._wrong_type(provider, _CRP, "context reference provider"):
            return
        try:
            _register(provider)
        except ValueError as exc:
            logger.warning(
                "Plugin '%s' context reference registration failed: %s", self.manifest.name, exc,
            )
            return
        logger.info(
            "Plugin '%s' registered context reference: @%s:", self.manifest.name, provider.prefix,
        )

    def register_memory_provider(self, provider) -> None:
        """Record a memory provider (inert). Activation is owned by ``plugins/memory`` via
        ``memory.provider``; a provider reaching here was loaded by the general manager, and
        without this method its ``register()`` would fail on a missing attribute."""
        from agent.memory_provider import MemoryProvider
        if self._wrong_type(provider, MemoryProvider, "memory provider"):
            return
        self._memory_provider = provider
        logger.debug(
            "Plugin '%s' registered memory provider: %s",
            self.manifest.name, getattr(provider, "name", "?"),
        )

    @_serialized_replacement
    def register_dashboard_auth_provider(self, provider) -> Optional[PluginRegistration]:
        """Register a :class:`hermes_cli.dashboard_auth.DashboardAuthProvider` for the dashboard
        auth gate (non-loopback bind without ``--insecure``). Wrong type / duplicate name warn and
        are ignored, never raised."""
        from hermes_cli.dashboard_auth import DashboardAuthProvider
        from hermes_cli.dashboard_auth.registry import (
            register_global_provider, unregister_global_provider,
        )
        if self._wrong_type(provider, DashboardAuthProvider, "dashboard-auth provider"):
            return
        registry_name = provider.name
        # The auth registry is process-global (lifetime = web server). Disposing it on a routine
        # per-home manager teardown emptied it for the WHOLE process and disabled sign-in until
        # restart — so upsert and keep it out of reverse-order teardown (``persistent=True``).
        try:
            register_global_provider(provider)
        except (TypeError, ValueError) as e:
            logger.warning(
                "Plugin '%s' failed to register dashboard-auth provider %r: %s",
                self.manifest.name, getattr(provider, "name", "?"), e,
            )
            return
        handle = self._track(
            "dashboard_auth_provider", registry_name,
            lambda: unregister_global_provider(registry_name, provider), persistent=True,
        )
        logger.info(
            "Plugin '%s' registered dashboard-auth provider: %s (%s)",
            self.manifest.name, registry_name, provider.display_name,
        )
        return handle

    @_serialized_replacement
    def register_platform(
        self, name: str, label: str, adapter_factory: Callable, check_fn: Callable,
        validate_config: Callable | None = None, required_env: list | None = None,
        install_hint: str = "", **entry_kwargs: Any,
    ) -> Optional[PluginRegistration]:
        """Register a gateway platform adapter (``adapter_factory(PlatformConfig) -> BasePlatformAdapter``).
        ``check_fn`` is a PASSIVE "deps importable?" probe that must never install (status displays call
        it freely); pass an ACTIVE installer as ``ensure_deps_fn`` (the gateway calls it from
        ``create_adapter()`` when ``check_fn`` is False). Extra kwargs (``setup_fn``, ``emoji``,
        ``allowed_users_env``, ``platform_hint``, ``ensure_deps_fn``) forward to ``PlatformEntry``;
        unknown keys raise TypeError."""
        from gateway.platform_registry import platform_registry, PlatformEntry
        entry_kwargs.setdefault("plugin_name", self.manifest.name)
        entry = PlatformEntry(
            name=name, label=label, adapter_factory=adapter_factory, check_fn=check_fn,
            validate_config=validate_config, required_env=required_env or [],
            install_hint=install_hint, source="plugin", **entry_kwargs,
        )
        scope = self._manager.scope_key
        previous = platform_registry.snapshot_registration(name, scope=scope)
        platform_registry.register(entry, scope=scope)
        current = platform_registry.snapshot_registration(name, scope=scope)
        if current[0] is not entry or current[1] is not None:
            return None
        self._manager._plugin_platform_names.add(name)
        handle = self._manager._track_scoped_registration(
            self.manifest, "platform", name, platform_registry, current, previous,
            finalize=lambda: self._manager._remove_platform_name_if_unowned(name),
        )
        logger.debug("Plugin %s registered platform: %s", self.manifest.name, name)
        return handle

    def register_slack_action_handler(
        self, action_id: Any, callback: Callable,
    ) -> PluginRegistration:
        """Register a Slack Block Kit action handler, wired into ``slack_bolt.AsyncApp`` at connect.
        ``action_id`` is anything ``slack_bolt.App.action()`` accepts; ``callback`` is
        ``async def handler(ack, body, action)`` (``await ack()`` within 3s). Raises ``ValueError`` for
        a non-callable callback or empty ``action_id``."""
        if not callable(callback):
            raise self._refuse("a Slack action handler with a non-callable callback")
        if action_id is None or (isinstance(action_id, str) and not action_id.strip()):
            raise self._refuse("a Slack action handler with an empty action_id")
        entry = (action_id, callback, self.manifest.name)
        self._manager._slack_action_handlers.append(entry)
        handle = self._track(
            "slack_action_handler", repr(action_id),
            lambda: self._manager._remove_identity(
                self._manager._slack_action_handlers, entry
            ),
        )
        logger.debug("Plugin %s registered Slack action handler: %s", self.manifest.name, action_id)
        return handle

    def register_platform_handler(self, platform: str, factory: Callable) -> None:
        """Register a native-client handler factory for a gateway platform, invoked at ``connect()`` as
        ``factory(native, adapter)`` before/as the core handlers register (``adapter`` read-only).
        ``native``: telegram PTB ``Application``, discord ``commands.Bot``, slack ``AsyncApp``, matrix
        client, teams ``App``, dingtalk ``DingTalkStreamClient``, line aiohttp ``web.Application``,
        others ``None``. Keep SDK imports inside the factory; exceptions are logged and the platform
        still connects. Always scope handlers hooked into first-match dispatch tables so core flows
        keep working. Raises ``ValueError`` for a non-callable factory or empty platform."""
        if not callable(factory):
            raise self._refuse("a platform handler factory with a non-callable factory")
        key = (platform or "").strip().lower()
        if not key:
            raise self._refuse("a platform handler factory with an empty platform name")
        self._manager._platform_handler_factories.setdefault(key, []).append(
            (factory, self.manifest.name)
        )
        logger.debug(
            "Plugin %s registered %s handler factory: %s", self.manifest.name, key,
            getattr(factory, "__name__", repr(factory)),
        )

    def register_telegram_handler(self, factory: Callable) -> None:
        """Alias of ``register_platform_handler("telegram", factory)``: ``factory(application,
        adapter)`` runs before the core handlers. PTB dispatches only the FIRST matching handler per
        group and core registers a catch-all ``CallbackQueryHandler`` — always scope with
        ``pattern=`` or you swallow the core button flows. Raises ``ValueError`` if not callable."""
        self.register_platform_handler("telegram", factory)

    @_serialized_replacement
    def register_auxiliary_task(
        self, key: str, *, display_name: str, description: str,
        defaults: Optional[Dict[str, Any]] = None,
    ) -> PluginRegistration:
        """Register an auxiliary LLM task with its own ``auxiliary.<key>`` config block (picker entry,
        ``AUXILIARY_<KEY>_*`` env bridge, defaults merged into loaded configs). ``key`` is snake_case
        and must not shadow a built-in task; ``defaults`` may override
        provider/model/base_url/api_key/timeout/extra_body (unknown keys preserved verbatim). Raises
        ``ValueError`` for an empty/invalid key, a built-in key, or another plugin's key."""
        if not key or not isinstance(key, str):
            raise ValueError(
                f"Plugin '{self.manifest.name}' tried to register auxiliary task with invalid key {key!r}"
            )
        if not all(c.isalnum() or c == "_" for c in key):
            raise ValueError(
                f"Plugin '{self.manifest.name}' auxiliary task key {key!r} "
                f"must contain only alphanumeric characters and underscores"
            )

        from hermes_cli.main import _AUX_TASKS as _BUILTIN_AUX_TASKS
        builtin_keys = {k for k, _name, _desc in _BUILTIN_AUX_TASKS}
        if key in builtin_keys:
            raise ValueError(
                f"Plugin '{self.manifest.name}' cannot register auxiliary task "
                f"{key!r} — that key is reserved for a built-in task. "
                f"Pick a plugin-namespaced key (e.g. '{self.manifest.name}_{key}')."
            )

        # Owner is the canonical id ``ctx.llm`` is bound to, so agent/plugin_llm.py can match it.
        owner_id = self.plugin_id

        existing = self._manager._aux_tasks.get(key)
        if existing is not None and existing.get("plugin") != owner_id:
            raise ValueError(
                f"Plugin '{self.manifest.name}' cannot register auxiliary task "
                f"{key!r} — already registered by plugin " f"'{existing.get('plugin')}'"
            )

        # Plugin owns the schema; routing fields are guaranteed present so consumers don't crash.
        merged_defaults: Dict[str, Any] = {
            "provider": "auto", "model": "", "base_url": "", "api_key": "",
            "timeout": 60, "extra_body": {},
        }
        merged_defaults.update(defaults or {})

        entry = {
            "key": key, "display_name": display_name, "description": description,
            "defaults": merged_defaults, "plugin": owner_id, "plugin_key": owner_id,
        }
        handle = self._track_mapping_entry(
            "auxiliary_task", key, self._manager._aux_tasks, entry, existing
        )
        logger.debug(
            "Plugin %s registered auxiliary task: %s (%s)", self.manifest.name, key, display_name,
        )
        return handle

    def register_redaction_patterns(self, patterns) -> int:
        """Additively register secret-token regexes with :mod:`agent.redact`; returns the count accepted.
        Plugins can over-redact, never weaken built-ins; ``security.redact_secrets: false`` applies
        equally. Each pattern must compile and start with >= 2 literal characters; invalid entries warn
        and are skipped."""
        from agent.redact import register_redaction_patterns as _register
        try:
            count = _register(patterns, source=f"plugin:{self.manifest.name}")
        except Exception as exc:
            logger.warning(
                "Plugin '%s' redaction pattern registration failed: %s", self.manifest.name, exc,
            )
            return 0
        logger.debug("Plugin %s registered %d redaction pattern(s)", self.manifest.name, count)
        return count

    def register_hook(self, hook_name: str, callback: Callable) -> PluginRegistration:
        """Register a lifecycle hook callback (unknown names warn but are still stored)."""
        return self._track_callback("hook", hook_name, callback, self._manager._hooks, VALID_HOOKS)

    def _track_callback(
        self, kind: str, key: str, callback: Callable, mapping: Dict[str, List[Callable]],
        valid: Set[str],
    ) -> PluginRegistration:
        """Append ``callback`` under ``key`` (warning on unknown ``key``) and lease its removal."""
        if key not in valid:
            logger.warning(
                "Plugin '%s' registered unknown %s '%s' (valid: %s)",
                self.manifest.name, kind, key, ", ".join(sorted(valid)),
            )
        mapping.setdefault(key, []).append(callback)
        handle = self._track(
            kind, key, lambda: self._manager._remove_callback(mapping, key, callback),
        )
        logger.debug("Plugin %s registered %s: %s", self.manifest.name, kind, key)
        return handle

    def register_system_prompt_section(
        self, id: str, content: Union[str, Callable[[Mapping[str, Any]], str]], *,
        position: str = "after_memory", max_chars: int = DEFAULT_SYSTEM_PROMPT_SECTION_MAX_CHARS,
    ) -> PluginRegistration:
        """Register bounded context frozen into each new session prompt. Callables receive a
        read-only session-info mapping; the rendered prompt is persisted by core verbatim."""
        if not is_valid_system_prompt_section_id(id):
            raise ValueError(
                "system prompt section id must be 1-128 lowercase characters "
                "using letters, numbers, '.', '_', or '-'"
            )
        if not isinstance(content, str) and not callable(content):
            raise TypeError("system prompt section content must be a string or callable")
        if position not in SYSTEM_PROMPT_SECTION_POSITIONS:
            raise ValueError(
                "system prompt section position must be one of: "
                + ", ".join(sorted(SYSTEM_PROMPT_SECTION_POSITIONS))
            )
        if (
            isinstance(max_chars, bool) or not isinstance(max_chars, int)
            or not 0 < max_chars <= MAX_SYSTEM_PROMPT_SECTION_CHARS
        ):
            raise ValueError(
                "system prompt section max_chars must be between 1 and "
                f"{MAX_SYSTEM_PROMPT_SECTION_CHARS}"
            )
        existing = self._manager._system_prompt_sections.get(id)
        if existing is not None:
            raise ValueError(
                f"system prompt section {id!r} is already registered by "
                f"plugin {existing.plugin!r}"
            )
        section = PluginSystemPromptSection(
            id=id, content=content, position=position, max_chars=max_chars, plugin=self.plugin_id,
        )
        handle = self._track_mapping_entry(
            "system_prompt_section", id, self._manager._system_prompt_sections, section, existing
        )
        logger.debug("Plugin %s registered system prompt section: %s", self.manifest.name, id)
        return handle

    def emit(self, event: str, payload: Optional[dict] = None) -> int:
        """Publish bare *event* as ``<plugin_key>:<event>`` (namespace FORCED to this plugin); return
        the subscriber count scheduled. Any ``':'`` in the name (``hermes:x`` is reserved for core,
        foreign namespaces forbidden) raises ``ValueError``. Delivery is fire-and-forget via a
        single-worker queue: order preserved, a blocking subscriber cannot stall the emitter."""
        plugin_key = self.plugin_id
        if not event or not isinstance(event, str):
            logger.warning("Plugin '%s' tried to emit an invalid event name %r", plugin_key, event)
            raise ValueError(f"Plugin '{plugin_key}' emit() requires a non-empty event name")
        if ":" in event:
            logger.warning(
                "Plugin '%s' tried to emit namespaced/reserved event '%s' — "
                "a plugin may only emit bare event names under its own '%s:' "
                "namespace (the '%s:' prefix is reserved for core, and foreign "
                "namespaces are forbidden)", plugin_key, event, plugin_key, HERMES_EVENT_NAMESPACE,
            )
            raise ValueError(
                f"Plugin '{plugin_key}' may not emit '{event}': emit only the "
                f"bare event name; the namespace is forced to '{plugin_key}:' "
                f"and the '{HERMES_EVENT_NAMESPACE}:' prefix is reserved for core"
            )
        if payload is not None and not isinstance(payload, dict):
            raise TypeError(f"Plugin '{plugin_key}' emit() payload must be a dict or None")
        full_event = f"{plugin_key}:{event}"
        return self._manager._dispatch_event(full_event, payload or {})

    def subscribe(self, event: str, callback: Callable) -> None:
        """Subscribe to a fully-qualified ``<plugin_key>:<event>`` name (unrestricted — only
        emitting is namespace-gated). Owner-tagged so unload removes zombie callbacks."""
        if not event or not isinstance(event, str):
            raise ValueError(
                f"Plugin '{self.manifest.name}' subscribe() requires a " f"non-empty event name"
            )
        plugin_key = self.plugin_id
        self._manager._subscribe_event(plugin_key, event, callback)
        logger.debug("Plugin %s subscribed to event: %s", self.manifest.name, event)

    def register_middleware(self, kind: str, callback: Callable) -> PluginRegistration:
        """Register behavior-changing middleware (request kinds rewrite the payload, execution kinds
        wrap the callback). Unknown kinds warn but are stored."""
        return self._track_callback(
            "middleware", kind, callback, self._manager._middleware, VALID_MIDDLEWARE
        )

    @_serialized_replacement
    def register_skill(
        self, name: str, path: Path, description: str = "",
        frontmatter: Optional[Mapping[str, Any]] = None,
    ) -> PluginRegistration:
        """Register a read-only skill resolvable as ``'<plugin_name>:<name>'`` via ``skill_view()``.
        Not in ``~/.hermes/skills/`` nor ``<available_skills>`` — explicit loads only. Raises
        ``ValueError`` (``':'``/invalid chars) or ``FileNotFoundError``."""
        from agent.skill_utils import _NAMESPACE_RE
        if ":" in name:
            raise ValueError(
                f"Skill name '{name}' must not contain ':' "
                f"(the namespace is derived from the plugin name "
                f"'{self.manifest.name}' automatically)."
            )
        if not name or not _NAMESPACE_RE.match(name):
            raise ValueError(f"Invalid skill name '{name}'. Must match [a-zA-Z0-9_-]+.")
        if not path.exists():
            raise FileNotFoundError(f"SKILL.md not found at {path}")

        namespace = self.manifest.skill_namespace or self.manifest.name
        qualified = f"{namespace}:{name}"
        if self.manifest.portable and qualified in self._manager._plugin_skills:
            raise ValueError(f"Plugin skill '{qualified}' is already registered")
        previous = self._manager._plugin_skills.get(qualified)
        entry = {
            "path": path, "plugin": namespace, "plugin_key": self.plugin_id, "bare_name": name,
            "description": description, "frontmatter": dict(frontmatter or {}),
        }
        handle = self._track_mapping_entry(
            "skill", qualified, self._manager._plugin_skills, entry, previous
        )
        logger.debug("Plugin %s registered skill: %s", self.manifest.name, qualified)
        return handle


# -- scoped provider registrars ------------------------------------------------------------------
# Every ``register_<category>_provider`` shares one body (:meth:`PluginContext._register_scoped_provider`):
# type-check, register in the scope-keyed process-global registry, lease the slot so unload restores
# the displaced entry. Rows: (method, kind, registry module, base-class module:attr, label, options).
# ``normalize`` defaults to ``str.strip``; ``None`` keeps the raw name; ``lower`` also lowercases.
_SCOPED_PROVIDER_REGISTRARS: Tuple[Tuple[str, str, str, str, str, Dict[str, Any]], ...] = (
    ("register_image_gen_provider", "image_gen_provider", "agent.image_gen_registry",
     "agent.image_gen_provider:ImageGenProvider", "image_gen provider", {"article": "an"}),
    ("register_video_gen_provider", "video_gen_provider", "agent.video_gen_registry",
     "agent.video_gen_provider:VideoGenProvider", "video_gen provider", {}),
    ("register_web_search_provider", "web_search_provider", "agent.web_search_registry",
     "agent.web_search_provider:WebSearchProvider", "web provider", {}),
    ("register_browser_provider", "browser_provider", "agent.browser_registry",
     "agent.browser_provider:BrowserProvider", "browser provider", {}),
    ("register_terminal_environment_provider", "terminal_environment_provider",
     "agent.terminal_env_registry", "agent.terminal_env_provider:TerminalEnvironmentProvider",
     "terminal environment provider",
     {"normalize": "lower", "reject_message": "Plugin '%s' terminal environment provider rejected: %s"}),
    ("register_secret_source", "secret_source", "agent.secret_sources.registry",
     "agent.secret_sources.base:SecretSource", "secret source",
     {"normalize": None, "register": "register_source"}),
    ("register_tts_provider", "tts_provider", "agent.tts_registry",
     "agent.tts_provider:TTSProvider", "TTS provider", {"normalize": "lower"}),
    ("register_transcription_provider", "transcription_provider", "agent.transcription_registry",
     "agent.transcription_provider:TranscriptionProvider", "transcription provider",
     {"normalize": "lower"}),
)

_SCOPED_PROVIDER_DOCS: Dict[str, str] = {
    "register_image_gen_provider": "Register an :class:`agent.image_gen_provider.ImageGenProvider`; "
        "``provider.name`` is matched by ``image_gen.provider``.",
    "register_video_gen_provider": "Register an :class:`agent.video_gen_provider.VideoGenProvider`; "
        "``provider.name`` is matched by ``video_gen.provider``.",
    "register_web_search_provider": "Register an :class:`agent.web_search_provider.WebSearchProvider`; "
        "``provider.name`` is matched by ``web.search_backend`` / ``web.extract_backend`` / ``web.backend``.",
    "register_browser_provider": "Register an :class:`agent.browser_provider.BrowserProvider`; "
        "``provider.name`` is matched by ``browser.cloud_provider`` (consulted by "
        "``tools.browser_tool._get_cloud_provider``).",
    "register_terminal_environment_provider": "Register a "
        ":class:`agent.terminal_env_provider.TerminalEnvironmentProvider`; ``provider.name`` is matched "
        "by ``terminal.backend`` when no built-in backend has that name. Built-in names (local, docker, "
        "singularity, modal, daytona, vercel_sandbox, ssh) are rejected — plugins never shadow in-tree "
        "backends.",
    "register_secret_source": "Register a :class:`agent.secret_sources.base.SecretSource`, run by "
        "``load_hermes_dotenv()`` (after ``~/.hermes/.env``, before credentials are read) when "
        "``secrets.<name>`` is enabled. The orchestrator owns ordering/precedence/provenance; the source "
        "only fetches. Since dotenv usually loads before discovery, the manager re-pulls enabled plugin "
        "sources afterwards.",
    "register_tts_provider": "Register an :class:`agent.tts_provider.TTSProvider`; ``provider.name`` is "
        "matched by ``tts.provider`` unless it is a built-in name (rejected with a warning) or a "
        "``tts.providers.<name>: type: command`` entry shares it (command-providers win).",
    "register_transcription_provider": "Register an "
        ":class:`agent.transcription_provider.TranscriptionProvider`; ``provider.name`` is matched by "
        "``stt.provider`` unless it is a built-in name (rejected) or a ``stt.providers.<name>: type: "
        "command`` entry shares it (command-providers win).",
}


def _make_scoped_provider_registrar(method_name, kind, registry_mod, base_ref, label, options):
    """Build one ``register_<category>_provider`` method from a ``_SCOPED_PROVIDER_REGISTRARS`` row."""
    base_mod, base_attr = base_ref.split(":")
    normalize = options.get("normalize", "strip")
    normalize_fn = (
        None if normalize is None else (lambda n: n.strip().lower()) if normalize == "lower"
        else (lambda n: n.strip())
    )

    def register(self, provider) -> Optional[PluginRegistration]:
        registry = importlib.import_module(registry_mod)
        base_class = getattr(importlib.import_module(base_mod), base_attr)
        register_fn = options.get("register")
        return self._register_scoped_provider(
            provider, kind=kind, base_class=base_class, registry=registry, label=label,
            article=options.get("article", "a"), normalize=normalize_fn,
            register=getattr(registry, register_fn) if register_fn else None,
            reject_message=options.get("reject_message"),
        )

    register.__name__ = method_name
    register.__qualname__ = f"PluginContext.{method_name}"
    register.__doc__ = _SCOPED_PROVIDER_DOCS[method_name]
    return _serialized_replacement(register)


for _row in _SCOPED_PROVIDER_REGISTRARS:
    setattr(PluginContext, _row[0], _make_scoped_provider_registrar(*_row))
del _row


def _resolve_hook_callback_timeout() -> float:
    """Effective hook-callback timeout from ``plugins.hook_callback_timeout`` (default 30s; ``<= 0``
    disables the threaded path; clamped to ``_MAX_HOOK_CALLBACK_TIMEOUT_SECS``)."""
    timeout = _HOOK_CALLBACK_TIMEOUT_SECS
    try:
        from hermes_cli.config import load_config_readonly

        plugins_cfg = (load_config_readonly() or {}).get("plugins")
        if isinstance(plugins_cfg, dict) and "hook_callback_timeout" in plugins_cfg:
            raw = plugins_cfg.get("hook_callback_timeout")
            if raw is not None:
                timeout = float(raw)
    except (TypeError, ValueError):
        logger.warning(
            "plugins.hook_callback_timeout is not a number; using default %gs",
            _HOOK_CALLBACK_TIMEOUT_SECS,
        )
        timeout = _HOOK_CALLBACK_TIMEOUT_SECS
    except Exception:
        timeout = _HOOK_CALLBACK_TIMEOUT_SECS

    if timeout < 0:
        logger.warning(
            "plugins.hook_callback_timeout=%g is negative; using default %gs", timeout,
            _HOOK_CALLBACK_TIMEOUT_SECS,
        )
        return _HOOK_CALLBACK_TIMEOUT_SECS
    if timeout > _MAX_HOOK_CALLBACK_TIMEOUT_SECS:
        logger.warning(
            "plugins.hook_callback_timeout=%g exceeds max %gs; clamping", timeout,
            _MAX_HOOK_CALLBACK_TIMEOUT_SECS,
        )
        return _MAX_HOOK_CALLBACK_TIMEOUT_SECS
    return timeout


class PluginManager(PluginLoaderMixin, PluginDispatchMixin, PluginLedgerMixin):
    """Central manager that discovers, loads, and invokes plugins."""

    def __init__(self, scope_key: Optional[str] = None) -> None:
        # Home is captured immutably: unload may run from another profile context, but every
        # inverse must target the registration's original scope.
        self.scope_key = scope_key or hermes_home_key()
        self.home_path = Path(self.scope_key)
        self._discovery_lock = threading.RLock()
        self._plugins: Dict[str, LoadedPlugin] = {}
        self._hooks: Dict[str, List[Callable]] = {}
        self._middleware: Dict[str, List[Callable]] = {}
        self._plugin_tool_names: Set[str] = set()
        self._plugin_platform_names: Set[str] = set()
        self._cli_commands: Dict[str, dict] = {}
        self._context_engine = None  # Set by a plugin via register_context_engine()
        self._plugin_commands: Dict[str, dict] = {}  # Slash commands registered by plugins
        self._system_prompt_sections: Dict[str, PluginSystemPromptSection] = {}
        self._discovered: bool = False
        self._cli_ref = None  # Set by CLI after plugin discovery
        self._gateway_message_injector: tuple[object, Callable] | None = None
        self._plugin_skills: Dict[str, Dict[str, Any]] = {}  # qualified name -> metadata
        self._portable_mcp_servers: Dict[str, Dict[str, Any]] = {}
        self._aux_tasks: Dict[str, Dict[str, Any]] = {}  # see register_auxiliary_task
        self._approval_transports: Dict[str, Any] = {}
        # Event bus: owner-tagged subscriptions (unload removes zombies); one daemon worker keeps
        # registration order while emitters never block.
        self._subscriptions: Dict[str, List[_EventSubscription]] = {}
        self._event_lock = threading.RLock()
        self._event_idle = threading.Condition(self._event_lock)
        self._event_generation = 0
        self._event_pending_by_generation: Dict[int, int] = {0: 0}
        self._event_queue: queue.Queue[Any] = queue.Queue(maxsize=_EVENT_PENDING_CAP)
        self._event_worker: Optional[threading.Thread] = None
        self._emit_depth = threading.local()  # per-worker chain depth caps mutual emitters
        self._slack_action_handlers: List[tuple] = []  # (matcher, callback, plugin_name)
        # In-flight / recently-timed-out hook callbacks keyed by (hook_name, id(cb)) so a stuck
        # policy hook cannot spawn a new abandoned thread on every fire.
        self._hook_running_callbacks: Dict[tuple, object] = {}
        self._hook_timeout_suppressed_until: Dict[tuple, float] = {}
        self._hook_timeout_lock = threading.Lock()
        self._hook_timeout_suppression_seconds = _HOOK_TIMEOUT_SUPPRESSION_SECONDS
        # Ledger per plugin (ownership) plus global order (reverse teardown across plugins).
        # Process-global registries are shared across profiles while several managers coexist, so
        # the ledger is keyed per (hermes_home, plugin_id) and every inverse is identity-conditional
        # — one profile's unload can never clear another's registrations.
        self._ownership_ledger: Dict[str, List[PluginRegistration]] = {}
        self._registration_order: List[PluginRegistration] = []
        # Persistent registrations that survived an unload-all; force re-discovery drains this via
        # _evict_stale_persistent_registrations().
        self._persistent_carryover: List[PluginRegistration] = []
        # Deferred platforms whose client tools registered at discovery (see
        # _register_deferred_platform_tools): imported package (don't re-execute on materialize)
        # and contributed tool names (so `hermes plugins list` still attributes them).
        self._predeclared_modules: Dict[str, types.ModuleType] = {}
        self._predeclared_tools: Dict[str, List[str]] = {}
        # Native platform handler factories keyed by lowercase platform name.
        self._platform_handler_factories: Dict[str, List[tuple]] = {}

    @property
    def has_gateway_message_injector(self) -> bool:
        """Return whether a live gateway can accept plugin-triggered turns."""
        return self._gateway_message_injector is not None

    def set_gateway_message_injector(self, owner: object, injector: Callable[..., bool]) -> None:
        """Publish a live gateway injector and its lifecycle owner."""
        self._gateway_message_injector = (owner, injector)

    def clear_gateway_message_injector(self, owner: object) -> None:
        """Clear the injector only when it still belongs to ``owner``."""
        registered = self._gateway_message_injector
        if registered is not None and registered[0] is owner:
            self._gateway_message_injector = None

    def inject_gateway_message(self, **kwargs: Any) -> bool:
        """Submit a plugin-triggered turn to the live gateway."""
        registered = self._gateway_message_injector
        if registered is None:
            return False
        return bool(registered[1](**kwargs))

    def discover_and_load(self, force: bool = False) -> None:
        """Scan all plugin sources and load each plugin found; ``force`` unloads first so config
        changes / new bundled backends become visible in long-lived sessions."""
        with self._discovery_lock, _plugin_home_scope(self.home_path):
            if self._discovered and not force:
                return
            if force:
                self.unload()  # the ledger owns teardown of process-global registries
            if env_var_enabled("HERMES_SAFE_MODE"):
                logger.info("HERMES_SAFE_MODE=1 — plugin discovery skipped")
                self._discovered = True
                return
            # Flag set up front as a re-entrancy guard (register() can trigger discovery again) but
            # reset on failure so a failed scan is NOT cached as "discovered with an empty registry"
            # — callers swallow the exception and would be stranded on the early return above.
            self._discovered = True
            try:
                self._discover_and_load_inner()
                # Persistent registrations survived the unload-all; now that plugins re-registered,
                # dispose the ones whose plugin did not come back.
                self._evict_stale_persistent_registrations()
                # load_hermes_dotenv() ran at import, before plugin secret sources existed: re-pull.
                self._refresh_secret_sources_after_discovery()
                if force:
                    # config.yaml shell hooks / outbound webhooks live in ``_hooks`` but are
                    # config-owned; unload() wiped them and cannot restore them.
                    self._re_register_config_hooks_after_force()
            except BaseException:
                self._discovered = False
                raise

    def _re_register_config_hooks_after_force(self) -> None:
        """Restore config-owned shell hooks/outbound webhooks after a force clear; each guarded
        independently so one failing does not skip the other."""
        try:
            from agent.shell_hooks import re_register_config_hooks

            re_register_config_hooks()
        except Exception as exc:
            logger.debug("force-reload shell-hook re-register skipped: %s", exc)
        try:
            from agent.outbound_webhooks import (
                re_register_config_hooks as re_register_outbound_webhooks,
            )

            re_register_outbound_webhooks()
        except Exception as exc:
            logger.debug("force-reload outbound-webhook re-register skipped: %s", exc)

    def _refresh_secret_sources_after_discovery(self) -> None:
        """If any plugin secret source is enabled (per its own ``is_enabled(cfg)``, honoring custom
        activation), reset the cache and re-apply. Fail-open: never raises into discover_and_load."""
        try:
            from agent.secret_sources.registry import list_plugin_sources
            from hermes_cli.env_loader import load_hermes_dotenv, reset_secret_source_cache
        except Exception:
            return
        try:
            plugin_sources = list_plugin_sources()
        except Exception:
            return
        if not plugin_sources:
            return
        try:
            from hermes_cli.config import load_config

            cfg = load_config() or {}
            secrets = cfg.get("secrets") or {}
        except Exception:
            secrets = {}
        enabled_names = []
        for source in plugin_sources:
            name = getattr(source, "name", "")
            section = secrets.get(name)
            section = section if isinstance(section, dict) else {}
            try:
                if source.is_enabled(section):
                    enabled_names.append(name)
            except Exception:
                continue  # mirrors the orchestrator: a raising is_enabled() is skipped
        if not enabled_names:
            return
        try:
            reset_secret_source_cache()
            load_hermes_dotenv()
            logger.debug(
                "Re-applied secret sources after plugin discovery for: %s",
                ", ".join(sorted(enabled_names)),
            )
        except Exception as exc:
            logger.debug("secret source re-apply after discovery failed: %s", exc)

    def _discover_and_load_inner(self) -> None:
        """The actual discovery sweep — see :meth:`discover_and_load`."""
        manifests: List[PluginManifest] = self._collect_directory_manifests()
        # Entry points are separate from the directory scan: the startup MCP probe must not import
        # or register them.
        ep_manifests = self._scan_entry_points()
        logger.debug("  entrypoints: %d manifest(s)", len(ep_manifests))
        manifests.extend(ep_manifests)

        disabled = _get_disabled_plugins()
        enabled = _get_enabled_plugins()  # None = opt-in default (nothing enabled)
        stale_relay_keys = legacy_relay_plugin_keys(enabled)
        if stale_relay_keys:
            logger.warning(
                "Removed Hermes plugin %s is still listed in plugins.enabled; "
                "remove it and configure native Relay plugins with %s", ", ".join(stale_relay_keys),
                RELAY_PLUGINS_CONFIG_ENV,
            )
        # Later sources win on key collision (project > user > bundled); gate the winners, then
        # load survivors in requires_plugins order (see resolve_plugin_load_order).
        winners = {manifest_key(m): m for m in manifests}
        to_load = {
            key: manifest for key, manifest in winners.items()
            if self._gate_manifest(manifest, disabled, enabled)
        }
        for lookup_key in resolve_plugin_load_order(to_load):
            manifest = to_load[lookup_key]
            self._warn_python_dependencies(manifest)
            self._validate_plugin_config_schema(manifest)
            self._load_plugin(manifest)

        if manifests:
            logger.info(
                "Plugin discovery complete: %d found, %d enabled", len(self._plugins),
                sum(1 for p in self._plugins.values() if p.enabled),
            )

    def _record_placeholder(
        self, manifest: PluginManifest, *, enabled: bool, error: Optional[str] = None
    ) -> None:
        """Record a manifest that discovery will not import (introspection-only entry)."""
        loaded = LoadedPlugin(manifest=manifest, enabled=enabled)
        if error is not None:
            loaded.error = error
        self._plugins[manifest_key(manifest)] = loaded

    def _gate_manifest(
        self, manifest: PluginManifest, disabled: Set[str], enabled: Optional[Set[str]],
    ) -> bool:
        """Route one winning manifest per :func:`gate_manifest`: load now, defer, or record as
        skipped. Returns True only for plugins that go through the dependency-ordered load pass."""
        verdict = gate_manifest(manifest, disabled, enabled)
        if verdict.action == "load":
            return True
        if verdict.action == "load_now":
            self._load_plugin(manifest)
        elif verdict.action == "defer":
            self._register_deferred_platform(manifest)
        else:
            self._record_placeholder(manifest, enabled=verdict.enabled, error=verdict.error)
        if verdict.log:
            logger.log(*verdict.log)
        return False

    def register_approval_transport(
        self, name: str, present_fn: Callable, *, plugin_id: str,
    ) -> None:
        """Register one plugin-owned approval transport for this profile."""
        from hermes_cli.approval_transport import RegisteredApprovalTransport
        clean = str(name).strip().lower()
        if clean == "builtin":
            raise ValueError("approval transport name 'builtin' is reserved")
        if not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", clean):
            raise ValueError("approval transport name must match [a-z0-9][a-z0-9_-]{0,63}")
        if not callable(present_fn):
            raise TypeError("approval transport present_fn must be callable")
        if clean in self._approval_transports:
            owner = self._approval_transports[clean].plugin_id
            raise ValueError(f"approval transport {clean!r} is already registered by {owner!r}")
        self._approval_transports[clean] = RegisteredApprovalTransport(
            name=clean, present=present_fn, plugin_id=plugin_id,
            profile_home=str(get_hermes_home().resolve()),
        )
        logger.info("Plugin %s registered approval transport: %s", plugin_id, clean)

    def get_approval_transport(self, name: str):
        """Return a transport only inside the profile that registered it."""
        registered = self._approval_transports.get(str(name).strip().lower())
        if registered is None:
            return None
        if registered.profile_home != str(get_hermes_home().resolve()):
            return None
        return registered

    def _collect_directory_manifests(self) -> List[PluginManifest]:
        """Directory manifests in full-discovery order (see :func:`collect_directory_manifests`)."""
        return collect_directory_manifests()

    def has_enabled_portable_mcp(self, raw_config: Mapping[str, Any]) -> bool:
        """Probe enabled portable MCP packages without loading plugins (shares the full-discovery
        manifest collection so precedence/gating cannot diverge)."""
        if _env_enabled("HERMES_SAFE_MODE"):
            return False

        plugins_config = raw_config.get("plugins")
        if not isinstance(plugins_config, dict):
            return False

        def _names(value: Any) -> Set[str]:
            return {v for v in value if isinstance(v, str)} if isinstance(value, list) else set()

        enabled = _names(plugins_config.get("enabled"))
        disabled = _names(plugins_config.get("disabled", []))
        if not enabled:
            return False

        winners = {manifest_key(m): m for m in self._collect_directory_manifests()}
        for manifest in winners.values():
            if not manifest.portable:
                continue
            lookup_key = manifest_key(manifest)
            if lookup_key in disabled or manifest.name in disabled:
                continue
            if lookup_key not in enabled and manifest.name not in enabled:
                continue
            try:
                from hermes_cli.agent_plugins import _discover_mcp

                if _discover_mcp(
                    Path(manifest.path), get_hermes_home() / "plugin-data"
                    / (manifest.skill_namespace or lookup_key), [], create_data=False,
                ):
                    return True
            except (OSError, RuntimeError, ValueError):
                continue  # fail closed on an unreadable package; full discovery reports it
        return False

    def _scan_directory(
        self, path: Path, source: str, skip_names: Optional[Set[str]] = None,
    ) -> List[PluginManifest]:
        """Read manifests under *path* (see :func:`scan_directory`)."""
        return scan_directory(path, source, skip_names=skip_names)

    def _scan_entry_points(self) -> List[PluginManifest]:
        """Read installed plugin entry points (see :func:`discover_entrypoint_manifests`)."""
        return discover_entrypoint_manifests()

    def get_slack_action_handlers(self) -> List[tuple]:
        """``(action_id, callback, plugin_name)`` tuples for the Slack adapter to wire at connect."""
        return list(self._slack_action_handlers)

    def get_platform_handler_factories(self, platform: str) -> List[tuple]:
        """``(factory, plugin_name)`` tuples for one platform; adapters call ``factory(native,
        adapter)`` at connect (see :meth:`PluginContext.register_platform_handler`)."""
        key = (platform or "").strip().lower()
        return list(self._platform_handler_factories.get(key, []))

    def list_plugins(self) -> List[Dict[str, Any]]:
        """Return a list of info dicts for all discovered plugins."""
        result: List[Dict[str, Any]] = []
        for key, loaded in sorted(self._plugins.items()):
            result.append(
                {
                    "name": loaded.manifest.name, "key": manifest_key(loaded.manifest),
                    "kind": loaded.manifest.kind, "version": loaded.manifest.version,
                    "description": loaded.manifest.description, "source": loaded.manifest.source,
                    "enabled": loaded.enabled, "tools": len(loaded.tools_registered),
                    "hooks": len(loaded.hooks_registered),
                    "middleware": len(loaded.middleware_registered),
                    "commands": len(loaded.commands_registered), "error": loaded.error,
                }
            )
        return result

    def find_plugin_skill(self, qualified_name: str) -> Optional[Path]:
        """Return the ``Path`` to a plugin skill's SKILL.md, or ``None``."""
        entry = self._plugin_skills.get(qualified_name)
        return entry["path"] if entry else None

    def list_plugin_skills(self, plugin_name: str) -> List[str]:
        """Return sorted bare names of all skills registered by *plugin_name*."""
        prefix = f"{plugin_name}:"
        return sorted(
            e["bare_name"] for qn, e in self._plugin_skills.items() if qn.startswith(prefix)
        )

    def list_plugin_skill_metadata(self) -> List[Dict[str, Any]]:
        """Return progressive-disclosure metadata for registered plugin skills."""
        return [
            {
                "name": qualified, "description": str(entry.get("description", "")),
                "category": "plugin", "frontmatter": dict(entry.get("frontmatter", {})),
            } for qualified, entry in sorted(self._plugin_skills.items())
        ]

    def get_portable_mcp_servers(self) -> Dict[str, Dict[str, Any]]:
        """Return a defensive copy of enabled portable MCP server configs."""
        return {name: dict(config) for name, config in self._portable_mcp_servers.items()}

    def remove_plugin_skill(self, qualified_name: str) -> None:
        """Remove a stale registry entry (silently ignores missing keys)."""
        self._plugin_skills.pop(qualified_name, None)


# Module-level singleton & convenience functions.

# Legacy single-slot "current" manager, kept so tests that monkeypatch ``_plugin_manager`` keep
# working — ``get_plugin_manager()`` still reads/writes this name.
_plugin_manager: Optional[PluginManager] = None

# Resolved Hermes home -> PluginManager. A process can switch profiles via
# ``set_hermes_home_override()``; a single slot would leak one profile's plugin/context-engine state
# into another, and keying by resolved home lets a re-entered profile reuse its imported modules.
_plugin_managers_by_home: Dict[Path, PluginManager] = {}
_plugin_managers_lock = threading.RLock()


def _plugin_home_key() -> Path:
    """Resolved active Hermes home — the key for per-profile plugin managers (plugins capture the
    home at registration, so a process serving several profiles cannot share one manager)."""
    try:
        return get_hermes_home().expanduser().resolve()
    except Exception:
        return get_hermes_home().expanduser()


def _clear_plugin_submodules(manager: Optional[PluginManager]) -> None:
    """Purge ``sys.modules`` entries for this manager's directory plugins (package AND submodules —
    otherwise a same-slug plugin in another profile reuses the previous profile's submodule state).
    """
    if manager is None:
        return
    for loaded in getattr(manager, "_plugins", {}).values():
        module = getattr(loaded, "module", None)
        module_name = getattr(module, "__name__", None)
        if not module_name or not module_name.startswith(f"{_NS_PARENT}."):
            continue
        _evict_modules(module_name)
        with _MODULE_NAMESPACE_LOCK:
            if _BARE_MODULE_SCOPE.get(module_name) == manager.scope_key:
                _BARE_MODULE_SCOPE.pop(module_name, None)


def get_plugin_manager() -> PluginManager:
    """Return the plugin manager for the active Hermes profile/home (cached per resolved home; a
    profile switch gets its own manager and plugin submodules)."""
    global _plugin_manager
    current_home = _plugin_home_key()

    with _plugin_managers_lock:
        # Tests/embedders monkeypatch ``_plugin_manager`` directly: adopt a single-slot manager the
        # keyed cache doesn't know about at all.
        if (
            _plugin_manager is not None and _plugin_manager not in _plugin_managers_by_home.values()
        ):
            _plugin_managers_by_home[current_home] = _plugin_manager
            return _plugin_manager

        manager = _plugin_managers_by_home.get(current_home)
        if manager is None:
            manager = PluginManager(scope_key=hermes_home_key(current_home))
            _plugin_managers_by_home[current_home] = manager

        _plugin_manager = manager
        return manager


def _reset_plugin_managers_for_tests() -> None:
    """Test-only: drop every cached manager and its submodules for a fully clean slate."""
    global _plugin_manager
    with _plugin_managers_lock:
        managers = list(dict.fromkeys(_plugin_managers_by_home.values()))
        if _plugin_manager is not None and _plugin_manager not in managers:
            managers.append(_plugin_manager)
        for manager in managers:
            _clear_plugin_submodules(manager)
            try:
                manager.unload()
            except Exception:
                logger.debug("test plugin-manager unload failed", exc_info=True)
        _plugin_managers_by_home.clear()
        _plugin_manager = None
    # Dashboard-auth providers are persistent and survive a routine unload, so the clean-slate
    # reset must clear that process-global registry explicitly or a test's provider leaks.
    try:
        from hermes_cli.dashboard_auth.registry import (
            clear_providers as _clear_dashboard_auth_providers,
        )

        _clear_dashboard_auth_providers()
    except Exception:
        logger.debug("dashboard-auth registry clear failed", exc_info=True)


def has_enabled_agent_plugin_mcp(raw_config: Mapping[str, Any]) -> bool:
    """Whether config enables a portable package with MCP servers (manifest-only scan on a fresh
    manager; imports nothing, mutates no registry)."""
    return PluginManager().has_enabled_portable_mcp(raw_config)


def discover_plugins(force: bool = False) -> None:
    """Discover and load all plugins (idempotent; ``force=True`` rescans). Joins an in-flight
    background discovery instead of racing a second scan."""
    _join_background_discovery()
    get_plugin_manager().discover_and_load(force=force)


_background_discovery_thread: Optional[threading.Thread] = None
_background_discovery_lock = threading.Lock()


def start_background_plugin_discovery() -> None:
    """Run discovery in a daemon thread to overlap the rest of CLI startup (~150ms). Every
    synchronous consumer joins it via :func:`discover_plugins`, so no one sees a half-loaded
    registry. No-op when already done or in flight."""
    global _background_discovery_thread
    manager = get_plugin_manager()
    if manager._discovered:
        return
    with _background_discovery_lock:
        if _background_discovery_thread is not None and _background_discovery_thread.is_alive():
            return

        def _run() -> None:
            try:
                manager.discover_and_load()
                _persist_plugin_toolset_keys()
            except Exception:
                logger.warning("background plugin discovery failed", exc_info=True)

        _background_discovery_thread = threading.Thread(
            target=_run, name="plugin-discovery", daemon=True
        )
        _background_discovery_thread.start()


def _join_background_discovery(timeout: float = 30.0) -> None:
    """Wait for an in-flight background discovery (no-op from its own thread)."""
    t = _background_discovery_thread
    if t is None or not t.is_alive() or t is threading.current_thread():
        return
    t.join(timeout=timeout)


def _plugin_toolset_keys_cache_path():
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "cache" / "plugin_toolset_keys.json"


def _persist_plugin_toolset_keys() -> None:
    """Persist discovered plugin toolset keys + portable MCP names (best-effort)."""
    try:
        import tempfile
        keys = sorted({ts_key for ts_key, _, _ in get_plugin_toolsets()})
        try:
            portable = sorted(get_plugin_manager().get_portable_mcp_servers())
        except Exception:
            portable = []
        path = _plugin_toolset_keys_cache_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".pt_keys.")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump({"toolset_keys": keys, "portable_mcp": portable}, fh)
        os.replace(tmp, path)
    except Exception:
        logger.debug("plugin toolset key persist failed", exc_info=True)


def _read_plugin_keys_cache() -> Optional[dict]:
    try:
        blob = json.loads(_plugin_toolset_keys_cache_path().read_text(encoding="utf-8"))
        if isinstance(blob, dict):
            return blob
    except Exception:
        pass
    return None


def _nowait_plugin_set(cache_field: str, live: Callable[[PluginManager], "set[str]"]) -> "set[str]":
    """Shared body of the ``*_nowait`` probes: live registry, else last launch's cache, else block."""
    manager = get_plugin_manager()
    t = _background_discovery_thread
    if manager._discovered and (t is None or not t.is_alive()):
        return live(manager)
    if t is not None and t.is_alive():
        blob = _read_plugin_keys_cache()
        if blob is not None:
            values = blob.get(cache_field)
            if isinstance(values, list) and all(isinstance(v, str) for v in values):
                return set(values)
    discover_plugins()
    return live(manager)


def get_plugin_toolset_keys_nowait() -> "set[str]":
    """Plugin toolset keys without blocking on in-flight discovery: live registry when done, last
    launch's persisted set while a background scan runs (callers only EXCLUDE these keys, so a stale
    set is harmless and self-heals), else block via discover_plugins()."""
    return _nowait_plugin_set(
        "toolset_keys", lambda _m: {ts_key for ts_key, _, _ in get_plugin_toolsets()}
    )


def get_portable_mcp_server_names_nowait() -> "set[str]":
    """Portable MCP server names; same contract as :func:`get_plugin_toolset_keys_nowait`."""
    return _nowait_plugin_set("portable_mcp", lambda m: set(m.get_portable_mcp_servers()))


def _delivery_manager() -> PluginManager:
    """Active manager, lazily discovering if it never ran — delivery must not depend on WHICH
    surface imported us (dashboards/TUI/cron never import model_tools). ``getattr`` default
    ``True`` leaves test doubles untouched."""
    manager = get_plugin_manager()
    if not getattr(manager, "_discovered", True):
        _join_background_discovery()
        manager.discover_and_load()
    return manager


def invoke_hook(hook_name: str, **kwargs: Any) -> List[Any]:
    """Invoke a lifecycle hook (lazy-discovers first); return non-``None`` callback results."""
    return _delivery_manager().invoke_hook(hook_name, **kwargs)


def render_system_prompt_sections(
    session_info: Mapping[str, Any],
) -> List[RenderedPluginSystemPromptSection]:
    """Render plugin prompt sections after idempotent plugin discovery."""
    return _ensure_plugins_discovered().render_system_prompt_sections(session_info)


def invoke_middleware(kind: str, **kwargs: Any) -> List[Any]:
    """Invoke registered middleware callbacks (lazy-discovers like :func:`invoke_hook`)."""
    return _delivery_manager().invoke_middleware(kind, **kwargs)


def has_middleware(kind: str) -> bool:
    """True when middleware is registered for ``kind``; lazy-discovers first since callers gate
    :func:`invoke_middleware` on it."""
    manager = get_plugin_manager()
    if not getattr(manager, "_discovered", True):
        manager = _delivery_manager()
    method = getattr(manager, "has_middleware", None)
    if callable(method):
        return bool(method(kind))
    return bool(getattr(manager, "_middleware", {}).get(kind))


def has_hook(hook_name: str) -> bool:
    """True when a loaded plugin handles a hook (lazy-discovers first, like :func:`has_middleware`)."""
    return _delivery_manager().has_hook(hook_name)


def iter_hook_callbacks(hook_name: str) -> tuple[Callable, ...]:
    """Return a stable snapshot of callbacks registered for a hook."""
    return get_plugin_manager().iter_hook_callbacks(hook_name)


def fire_pre_command_hook(
    *, surface: str, command: str, alias_used: str, args_raw: str,
    session_key: Optional[str] = None, platform: Optional[str] = None,
) -> None:
    """Fire the observer-only ``pre_command`` hook; never raises. Directive-shaped returns are
    logged at debug so future block/rewrite adopters are discoverable."""
    try:
        manager = get_plugin_manager()
        if not manager.has_hook("pre_command"):
            return
        results = manager.invoke_hook(
            "pre_command", surface=surface, command=command, alias_used=alias_used,
            args_raw=args_raw, session_key=session_key, platform=platform,
        )
        for result in results:
            if isinstance(result, dict) and ("action" in result or "decision" in result):
                logger.debug(
                    "pre_command is observer-only in v1: ignoring directive "
                    "%r for /%s (surface=%s). Block/rewrite will arrive with "
                    "the command middleware variant (#64204/#64231).", result, command, surface,
                )
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("pre_command hook dispatch failed (non-fatal): %s", exc)


_thread_tool_whitelist = threading.local()


@dataclass(frozen=True)
class _PreToolCallDirective:
    action: Optional[str] = None
    message: Optional[str] = None
    rule_key: Optional[str] = None
    modified_args: Optional[Dict[str, Any]] = None


def set_thread_tool_whitelist(
    allowed: Optional[Set[str]],
    deny_msg_fmt: str = "Tool '{tool_name}' denied: not in this thread's tool whitelist",
) -> None:
    _thread_tool_whitelist.allowed = allowed
    _thread_tool_whitelist.fmt = deny_msg_fmt


def clear_thread_tool_whitelist() -> None:
    _thread_tool_whitelist.allowed = None


def _get_pre_tool_call_directive_details(
    tool_name: str, args: Optional[Dict[str, Any]], task_id: str = "", session_id: str = "",
    tool_call_id: str = "", turn_id: str = "", api_request_id: str = "",
    middleware_trace: Optional[List[Dict[str, Any]]] = None,
) -> _PreToolCallDirective:
    """Check ``pre_tool_call`` hooks for ``{"action": "block", "message"}`` (veto; message becomes
    the tool result) or ``{"action": "approve", "message", "rule_key"?}`` (escalate ANY tool to the
    human-approval gate; ``rule_key`` picks the ``[a]lways`` allowlist grain). First valid directive
    wins; irrelevant returns are ignored."""
    allowed = getattr(_thread_tool_whitelist, "allowed", None)
    if allowed is not None and tool_name not in allowed:
        fmt = getattr(_thread_tool_whitelist, "fmt", "Tool '{tool_name}' denied")
        return _PreToolCallDirective(action="block", message=fmt.format(tool_name=tool_name))

    from hermes_cli.lifecycle import invoke_hook as invoke_lifecycle_hook
    hook_results = invoke_lifecycle_hook(
        "pre_tool_call", tool_name=tool_name, args=args if isinstance(args, dict) else {},
        task_id=task_id, session_id=session_id, tool_call_id=tool_call_id, turn_id=turn_id,
        api_request_id=api_request_id, middleware_trace=list(middleware_trace or []),
    )

    modified_args: Optional[Dict[str, Any]] = None

    for result in hook_results:
        if not isinstance(result, dict):
            continue
        action = result.get("action")
        # "modify" — transform tool_input before dispatch. Processed before the block/approve gate
        # so modify directives are visible even when a later hook blocks. Each modify directive
        # shallow-merges its keys into one accumulated dict built from the original args.
        if action == "modify":
            partial = result.get("args")
            if isinstance(partial, dict) and partial:
                if modified_args is None:
                    modified_args = dict(args) if isinstance(args, dict) else {}
                modified_args.update(partial)
            continue
        if action not in ("block", "approve"):
            continue
        message = result.get("message")
        message = message if isinstance(message, str) and message else None
        # A block directive requires a message (it becomes the tool result); approve's is optional.
        if action == "block" and not message:
            continue
        rule_key = result.get("rule_key") if action == "approve" else None
        rule_key = (rule_key.strip() or None) if isinstance(rule_key, str) else None
        return _PreToolCallDirective(
            action=action, message=message, rule_key=rule_key, modified_args=modified_args,
        )

    return _PreToolCallDirective(modified_args=modified_args)


def get_pre_tool_call_directive(
    tool_name: str, args: Optional[Dict[str, Any]], **hook_kwargs: Any
) -> tuple[Optional[str], Optional[str]]:
    """Back-compat: ``(directive, message)`` with directive ``"block"`` / ``"approve"`` / ``None``.
    ``hook_kwargs`` are the observability ids of :func:`_get_pre_tool_call_directive_details`."""
    details = _get_pre_tool_call_directive_details(tool_name, args, **hook_kwargs)
    return (details.action, details.message)


def get_pre_tool_call_block_message(
    tool_name: str, args: Optional[Dict[str, Any]], **hook_kwargs: Any
) -> Optional[str]:
    """Deprecated shim: only the ``block`` message (or ``None``); ``approve`` is invisible here."""
    directive, message = get_pre_tool_call_directive(tool_name, args, **hook_kwargs)
    return message if directive == "block" else None


def resolve_pre_tool_block(
    tool_name: str, args: Optional[Dict[str, Any]], **hook_kwargs: Any
) -> Optional[str]:
    """Resolve the pre_tool_call directive to a final block message (or ``None`` to proceed),
    running the human-approval gate for ``approve``. See :func:`_resolve_block_from_details`."""
    return _dispatch_pre_tool_call_hooks(tool_name, args, **hook_kwargs)[0]


def _resolve_block_from_details(
    details: "_PreToolCallDirective", tool_name: str, *, turn_id: str = "", tool_call_id: str = "",
    session_id: str = "",
) -> Optional[str]:
    """The ONE place for the fail-closed approval logic: ``block`` blocks with its message; an
    ``approve`` whose gate errors, denies, or times out is blocked; anything else proceeds."""
    if details.action == "block":
        return details.message
    if details.action != "approve":
        return None
    try:
        from tools.approval import (
            request_tool_approval, reset_current_observability_context,
            set_current_observability_context,
        )

        approval_tokens = None
        with suppress(Exception):
            approval_tokens = set_current_observability_context(
                turn_id=turn_id, tool_call_id=tool_call_id, session_id=session_id,
            )
        try:
            result = request_tool_approval(
                tool_name, details.message or "", rule_key=details.rule_key or tool_name,
            )
        finally:
            if approval_tokens is not None:
                with suppress(Exception):
                    reset_current_observability_context(approval_tokens)
    except Exception:
        # Fail-closed: if the gate itself errors, block rather than silently execute an action a
        # plugin flagged for approval.
        return f"BLOCKED: plugin approval gate failed for {tool_name}"
    if not result.get("approved"):
        return str(result.get("message") or f"BLOCKED: plugin approval required for {tool_name}")
    return None


def _dispatch_pre_tool_call_hooks(
    tool_name: str, args: Optional[Dict[str, Any]], **hook_kwargs: Any
) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """Invoke ``pre_tool_call`` hooks once; return ``(block_message, modified_args)`` — the resolved
    block/approve message (``None`` to proceed) and merged ``modify`` args (``None`` if none)."""
    details = _get_pre_tool_call_directive_details(tool_name, args, **hook_kwargs)
    block_msg = _resolve_block_from_details(
        details, tool_name,
        turn_id=hook_kwargs.get("turn_id", ""), tool_call_id=hook_kwargs.get("tool_call_id", ""),
        session_id=hook_kwargs.get("session_id", ""),
    )
    return (block_msg, details.modified_args)


def get_pre_verify_continue_message(
    *, session_id: str = "", platform: str = "", model: str = "", coding: bool = False,
    attempt: int = 0, final_response: str = "", changed_paths: Optional[List[str]] = None,
) -> Optional[str]:
    """Check ``pre_verify`` hooks for ``{"action": "continue", "message"}`` (or Claude-Code Stop
    ``{"decision": "block", "reason"}``) to keep the turn going; first non-empty message wins, any
    other return lets the turn finish. ``coding``/``attempt`` let hooks scope and self-throttle."""
    hook_results = invoke_hook(
        "pre_verify", session_id=session_id, platform=platform, model=model, coding=coding,
        attempt=attempt, final_response=final_response, changed_paths=list(changed_paths or []),
    )

    for result in hook_results:
        if not isinstance(result, dict):
            continue
        action = str(result.get("action") or result.get("decision") or "").strip().lower()
        if action not in ("continue", "block"):
            continue
        message = result.get("message") or result.get("reason")
        if isinstance(message, str) and message.strip():
            return message.strip()

    return None


def get_plugin_error_classification(
    *, provider: str = "", model: str = "", status_code: Optional[int] = None, error_type: str = "",
    error_code: str = "", error_message: str = "", error_body: Optional[Dict[str, Any]] = None,
    error: Optional[BaseException] = None, approx_tokens: int = 0, context_length: int = 0,
    num_messages: int = 0,
) -> Optional[Dict[str, Any]]:
    """Consult ``transform_api_error_classification`` hooks BEFORE the built-in classifier.
    Run-all-then-pick-first: every callback runs isolated, the first valid result in registration
    order wins, losing valid results warn (conflicts visible, not shadowed). Returns a sanitized dict
    (``reason`` -> ``FailoverReason``, hint flags -> bool, ``message`` capped at 500) or ``None``.
    Privacy: ``error_message``/``error_body`` may be unredacted."""
    from agent.error_classifier import FailoverReason
    hook_results = invoke_hook(
        "transform_api_error_classification", provider=provider, model=model,
        status_code=status_code, error_type=error_type, error_code=error_code,
        error_message=error_message, error_body=error_body if isinstance(error_body, dict) else {},
        error=error, approx_tokens=approx_tokens, context_length=context_length,
        num_messages=num_messages,
    )

    winner: Optional[Dict[str, Any]] = None
    skipped_valid = 0
    for result in hook_results:
        if not isinstance(result, dict):
            continue
        reason = result.get("reason")
        if isinstance(reason, str):
            try:
                reason = FailoverReason(reason.strip().lower())
            except ValueError:
                continue
        if not isinstance(reason, FailoverReason):
            continue

        if winner is not None:
            skipped_valid += 1
            continue

        out: Dict[str, Any] = {"reason": reason}
        for key in ("retryable", "should_compress", "should_rotate_credential", "should_fallback"):
            if key in result:
                out[key] = bool(result[key])
        message = result.get("message")
        if isinstance(message, str) and message.strip():
            out["message"] = message.strip()[:500]
        error_context = result.get("error_context")
        if isinstance(error_context, dict):
            out["error_context"] = error_context
        winner = out

    if winner is not None and skipped_valid:
        logger.warning(
            "transform_api_error_classification: skipped %d valid "
            "classification(s) after the first result in registration order "
            "won (run-all-then-pick-first)", skipped_valid,
        )
    return winner


def _ensure_plugins_discovered(force: bool = False) -> PluginManager:
    """Return the global manager after idempotent (or ``force``d) discovery."""
    manager = get_plugin_manager()
    manager.discover_and_load(force=force)
    return manager


def get_plugin_context_engine():
    """Return the plugin-registered context engine, or None."""
    return _ensure_plugins_discovered()._context_engine


def get_plugin_command_handler(name: str) -> Optional[Callable]:
    """Return the handler for a plugin-registered slash command, or ``None``."""
    entry = _ensure_plugins_discovered()._plugin_commands.get(name)
    return entry["handler"] if entry else None


_PLUGIN_COMMAND_AWAIT_TIMEOUT_SECS = 30.0


def resolve_plugin_command_result(result: Any) -> Any:
    """Resolve a plugin command result, awaiting async handlers: ``asyncio.run`` when no loop is
    running, else a helper thread with its own loop (30s bound so a hung handler cannot wedge the
    terminal)."""
    if not inspect.isawaitable(result):
        return result

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(result)

    outcome: Dict[str, Any] = {}
    failure: Dict[str, BaseException] = {}
    done = threading.Event()

    def _runner() -> None:
        try:
            outcome["value"] = asyncio.run(result)
        except BaseException as exc:  # pragma: no cover - re-raised below
            failure["exc"] = exc
        finally:
            done.set()

    thread = threading.Thread(target=_runner, name="hermes-plugin-command-await", daemon=True)
    thread.start()
    if not done.wait(timeout=_PLUGIN_COMMAND_AWAIT_TIMEOUT_SECS):
        raise TimeoutError(
            "Plugin command async handler did not complete within "
            f"{_PLUGIN_COMMAND_AWAIT_TIMEOUT_SECS:.0f}s"
        )
    if "exc" in failure:
        raise failure["exc"]
    return outcome.get("value")


def get_plugin_commands() -> Dict[str, dict]:
    """Plugin commands dict (name -> {handler, description, plugin}) after idempotent discovery."""
    return _ensure_plugins_discovered()._plugin_commands


def get_plugin_auxiliary_tasks() -> List[Dict[str, Any]]:
    """Plugin auxiliary-task registration dicts sorted by ``key`` (after idempotent discovery)."""
    manager = _ensure_plugins_discovered()
    return [manager._aux_tasks[k] for k in sorted(manager._aux_tasks)]


def get_plugin_toolsets() -> List[tuple]:
    """Plugin toolsets as ``(key, label, description)`` tuples for the ``hermes tools`` TUI."""
    manager = get_plugin_manager()
    if not manager._plugin_tool_names:
        return []

    try:
        from tools.registry import registry
    except Exception:
        return []

    # Group plugin tool names by their toolset
    toolset_tools: Dict[str, List[str]] = {}
    for tool_name in manager._plugin_tool_names:
        entry = registry.get_entry(tool_name)
        if entry:
            toolset_tools.setdefault(entry.toolset, []).append(entry.name)

    # Map toolsets back to the plugin that registered them
    toolset_plugin: Dict[str, LoadedPlugin] = {}
    for loaded in manager._plugins.values():
        for tool_name in loaded.tools_registered:
            entry = registry.get_entry(tool_name)
            if entry and entry.toolset in toolset_tools:
                toolset_plugin.setdefault(entry.toolset, loaded)

    result = []
    for ts_key in sorted(toolset_tools):
        plugin = toolset_plugin.get(ts_key)
        desc = (plugin.manifest.description if plugin else "") or ", ".join(
            sorted(toolset_tools[ts_key])
        )
        result.append((ts_key, f"🔌 {ts_key.replace('_', ' ').title()}", desc))
    return result
