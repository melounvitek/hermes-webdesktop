"""
Event Hook System

Fires handlers at gateway lifecycle points. Hooks are discovered from
~/.hermes/hooks/<name>/ directories, each containing HOOK.yaml (name,
description, events) and handler.py (``def handle(event_type, context)``,
sync or async). Handler errors are logged and never block the pipeline.

Events:
  gateway:startup, session:start, session:end (user ran /new or /reset),
  session:reset, agent:start, agent:step (each tool-loop turn), agent:end,
  command:* (any slash command; wildcard match).

Context passed to ``agent:start`` / ``agent:end``:
  platform, user_id, chat_id, thread_id (Telegram forum-topic / thread root
  id as string; empty when not in a thread), chat_type ("dm" | "group" |
  "forum" | ""), session_id, message (truncated to 500 chars).
``agent:end`` adds: response (truncated to 500 chars), model, provider.

Handlers posting a follow-up into the same Telegram forum-topic should pass
``message_thread_id=int(thread_id)`` when ``chat_type == "forum"`` and
``thread_id`` is non-empty.
"""

import asyncio
import importlib.util
import sys
from typing import Any, Callable, Dict, List, Optional

import yaml

from hermes_cli.config import get_hermes_home


HOOKS_DIR = get_hermes_home() / "hooks"


class HookRegistry:
    """Discovers, loads, and fires event hooks."""

    def __init__(self):
        self._handlers: Dict[str, List[Callable]] = {}  # event_type -> handlers
        self._loaded_hooks: List[dict] = []  # metadata for listing

    @property
    def loaded_hooks(self) -> List[dict]:
        return list(self._loaded_hooks)

    def _register_builtin_hooks(self) -> None:
        """Extension point for always-on built-in hooks; currently none shipped."""
        return

    def discover_and_load(self) -> None:
        """Register built-in hooks, then load every valid hook dir under HOOKS_DIR."""
        self._register_builtin_hooks()

        if not HOOKS_DIR.exists():
            return

        for hook_dir in sorted(HOOKS_DIR.iterdir()):
            if not hook_dir.is_dir():
                continue

            manifest_path = hook_dir / "HOOK.yaml"
            handler_path = hook_dir / "handler.py"

            if not manifest_path.exists() or not handler_path.exists():
                continue

            try:
                manifest = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
                if not manifest or not isinstance(manifest, dict):
                    print(f"[hooks] Skipping {hook_dir.name}: invalid HOOK.yaml", flush=True)
                    continue

                hook_name = manifest.get("name", hook_dir.name)
                events = manifest.get("events", [])
                if not events:
                    print(f"[hooks] Skipping {hook_name}: no events declared", flush=True)
                    continue

                # Register in sys.modules BEFORE exec_module so Pydantic/dataclass
                # forward references (from ``from __future__ import annotations``)
                # resolve; otherwise a handler declaring a BaseModel fails at first
                # dispatch with "TypeAdapter ... is not fully defined".
                module_name = f"hermes_hook_{hook_name}"
                spec = importlib.util.spec_from_file_location(module_name, handler_path)
                if spec is None or spec.loader is None:
                    print(f"[hooks] Skipping {hook_name}: could not load handler.py", flush=True)
                    continue

                module = importlib.util.module_from_spec(spec)
                sys.modules[module_name] = module
                try:
                    spec.loader.exec_module(module)
                except Exception:
                    sys.modules.pop(module_name, None)
                    raise

                handle_fn = getattr(module, "handle", None)
                if handle_fn is None:
                    print(f"[hooks] Skipping {hook_name}: no 'handle' function found", flush=True)
                    continue

                for event in events:
                    self._handlers.setdefault(event, []).append(handle_fn)

                self._loaded_hooks.append({
                    "name": hook_name,
                    "description": manifest.get("description", ""),
                    "events": events,
                    "path": str(hook_dir),
                })

                print(f"[hooks] Loaded hook '{hook_name}' for events: {events}", flush=True)

            except Exception as e:
                print(f"[hooks] Error loading hook {hook_dir.name}: {e}", flush=True)

    def _resolve_handlers(self, event_type: str) -> List[Callable]:
        """Exact-match handlers first, then ``<base>:*`` wildcard handlers.

        A handler registered for a bare base type ("agent") does NOT fire for
        "agent:start" — only exact matches and explicit wildcards.
        """
        handlers = list(self._handlers.get(event_type, []))
        if ":" in event_type:
            handlers.extend(self._handlers.get(f"{event_type.split(':')[0]}:*", []))
        return handlers

    async def emit(self, event_type: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Fire all handlers for an event, discarding return values."""
        await self.emit_collect(event_type, context)

    async def emit_collect(
        self,
        event_type: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Any]:
        """Fire handlers and return their non-None return values in order.

        Used for decision-style hooks (e.g. ``command:<name>`` policies that
        allow/deny/rewrite a command). A failing handler is logged and does not
        abort the remaining handlers.
        """
        if context is None:
            context = {}

        results: List[Any] = []
        for fn in self._resolve_handlers(event_type):
            try:
                result = fn(event_type, context)
                if asyncio.iscoroutine(result):  # sync and async handlers both supported
                    result = await result
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"[hooks] Error in handler for '{event_type}': {e}", flush=True)
        return results
