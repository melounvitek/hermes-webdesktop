"""Entry point for the `computer_use` tool.

Any-model desktop control (macOS/Windows/Linux) via cua-driver; standard OpenAI
function-calling schema. Return contract: text-only results are a JSON string;
captures / `capture_after=True` return ``{"_multimodal": True, "content":
[text part, image_url part], "text_summary": <fallback>}`` which run_agent.py /
the Anthropic adapter turn into provider-specific image tool content.
"""

from __future__ import annotations

import atexit
import base64
import contextlib
import json
import logging
import os
import re
import sys
import threading
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

from tools.computer_use.backend import (
    ActionResult,
    CaptureResult,
    ComputerUseBackend,
    UIElement,
    image_dimensions_from_bytes,
)

logger = logging.getLogger(__name__)


# ── Approval & safety ───────────────────────────────────────────────────────

_approval_callback = None


def set_approval_callback(cb) -> None:
    """Register the CLI approval prompt (terminal_tool._approval_callback pattern).
    ``cb(action, args, summary)`` returns "approve_once" | "approve_session" |
    "always_approve" | "deny"."""
    global _approval_callback
    _approval_callback = cb


# Actions that mutate user-visible state go through approval; the rest read.
_DESTRUCTIVE_ACTIONS = frozenset({"click", "double_click", "right_click", "middle_click",
                                  "drag", "scroll", "type", "key", "set_value", "focus_app"})

# Hard-blocked regardless of approval level (e.g. logout kills the session
# Hermes runs in). Alt is canonicalized to option, so the Windows variants are
# blocked before any backend sees them.
_BLOCKED_KEY_COMBOS = {
    frozenset({"cmd", "shift", "backspace"}),   # empty trash
    frozenset({"cmd", "option", "backspace"}),   # force delete
    frozenset({"cmd", "ctrl", "q"}),             # lock screen
    frozenset({"cmd", "shift", "q"}),            # log out
    frozenset({"cmd", "option", "shift", "q"}),  # force log out
    frozenset({"win", "l"}),
    frozenset({"ctrl", "option", "delete"}),
    frozenset({"ctrl", "option", "del"}),
    frozenset({"option", "f4"}),
}

_KEY_ALIASES = {
    "command": "cmd", "control": "ctrl", "alt": "option", "⌘": "cmd", "⌥": "option",
    "windows": "win", "super": "win", "meta": "win",
}

# Dangerous shell patterns for the `type` action.
_BLOCKED_TYPE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in (
    r"curl\s+[^|]*\|\s*bash", r"curl\s+[^|]*\|\s*sh", r"wget\s+[^|]*\|\s*bash",
    r"\bsudo\s+rm\s+-[rf]", r"\brm\s+-rf\s+/\s*$",
    r":\s*\(\)\s*\{\s*:\|:\s*&\s*\}",  # fork bomb
)]


def _canon_key_combo(keys: str) -> frozenset:
    # Split on "+" AND "-": cua-driver accepts hyphenated combos, so
    # "ctrl-alt-delete" would bypass the gate otherwise.
    parts = [p.strip().lower() for p in re.split(r"\s*[+\-]\s*", keys) if p.strip()]
    return frozenset(_KEY_ALIASES.get(p, p) for p in parts)


def _reject_unsafe(action: str, args: Dict[str, Any]) -> Optional[str]:
    """JSON error for hard-blocked input, else None. Runs BEFORE the approval prompt."""
    if action == "type":
        text = args.get("text", "")
        pat = next((p.pattern for p in _BLOCKED_TYPE_PATTERNS if p.search(text)), None)
        if pat:
            return json.dumps({"error": f"blocked pattern in type text: {pat!r}",
                               "hint": "Dangerous shell patterns cannot be typed via computer_use."})
    if action == "key":
        combo = _canon_key_combo(args.get("keys", ""))
        for blocked in _BLOCKED_KEY_COMBOS:
            if blocked.issubset(combo) and len(blocked) <= len(combo):
                return json.dumps({"error": f"blocked key combo: {sorted(blocked)}",
                                   "hint": "Destructive system shortcuts are hard-blocked."})
    if args.get("bring_to_front") and args.get("delivery_mode") != "foreground":
        return json.dumps({"error": "bring_to_front requires delivery_mode='foreground'",
                           "code": "bring_to_front_requires_foreground"})
    return None


def _input_target_mismatch(backend, requested_app: str) -> Optional[str]:
    """Current sticky-target app when it provably differs from *requested_app*:
    both known and neither a substring of the other (names are localized/variant —
    'Google-chrome' vs 'chrome'). Unknown current target -> None (fail open; the
    verify ladder catches wrong-window delivery)."""
    current = (getattr(backend, "_last_app", None) or "").strip().lower()
    wanted = requested_app.strip().lower()
    if not current or not wanted or wanted in current or current in wanted:
        return None
    return getattr(backend, "_last_app", None)


# ── Backend selection — env-swappable for tests ─────────────────────────────

# Per-Hermes-session cached backends; each owns its own cua-driver session,
# native target, refs, and grant namespace.
_backend_lock = threading.Lock()
# Process-scoped aux-vision routing cache: (provider, model) → bool.
_AUX_VISION_ROUTE_CACHE: Dict[Tuple[str, str], bool] = {}
# `_backend` is the backward-compatible empty-session injection hook (older tests).
_backend: Optional[ComputerUseBackend] = None
_backends: Dict[str, ComputerUseBackend] = {}
_backend_call_locks: Dict[str, threading.RLock] = {}
_backend_permission_modes: Dict[str, str] = {}
# Approval state keyed by session_id so a gateway serving concurrent sessions
# can't leak one run's "always approve" into another; no session_id -> "".
#   _session_auto_approve[sid] -> bool   ("always_approve everything")
#   _always_allow[sid]         -> set of (action, delivery_mode) scope keys
_approval_lock = threading.Lock()
_session_auto_approve: Dict[str, bool] = {}
_always_allow: Dict[str, set] = {}
# Sessions already warned that a bypass widened the driver mode (resolver runs per dispatch).
_escalation_warned: set = set()


def _warn_bypass_escalation(session_id: str) -> None:
    """Warn once per session that ``-z``/``--yolo`` swapped the driver onto a private
    ``unrestricted`` daemon, dropping the configured ceiling. Deliberate (``unrestricted``
    is intentionally not a config value), but easy to trigger by accident."""
    key = str(session_id or "")
    with _approval_lock:
        if key in _escalation_warned:
            return
        _escalation_warned.add(key)
    configured = _configured_permission_mode()
    logger.warning(
        "computer_use: approval bypass (--yolo / -z) escalated the cua-driver "
        "permission mode from the configured '%s' to 'unrestricted' for this "
        "session. Runtime approval prompts are disabled and the driver's "
        "residual ceilings no longer apply. Drop the bypass flag to keep '%s', "
        "or declare a version-3 computer_use.capability_manifest to keep a "
        "ceiling on bypassed runs.", configured, configured)


def _configured_permission_mode() -> str:
    """Configured cua mode (standard | bounded); "standard" if unresolvable. bounded
    needs computer_use.capability_manifest; the backend fails loudly without it."""
    try:
        from tools.computer_use.cua_backend import _cua_configured_permission_mode

        return _cua_configured_permission_mode()
    except Exception:
        return "standard"


def _cua_permission_mode(session_id: str) -> str:
    """Map Hermes's approval bypass onto Cua's immutable mode. Both identity
    namespaces are consulted — DB ``session_id`` and gateway ``session_key``
    contextvar — or a gateway ``/yolo`` would be invisible here. Fails closed."""
    try:
        from tools.approval import get_current_session_key, is_approval_bypass_active_for_session

        if is_approval_bypass_active_for_session(session_id):
            _warn_bypass_escalation(session_id)
            return "unrestricted"
        current_key = get_current_session_key(default="")
        if current_key and is_approval_bypass_active_for_session(current_key):
            _warn_bypass_escalation(session_id)
            return "unrestricted"
    except Exception:
        pass
    return _configured_permission_mode()


def _new_backend(permission_mode: str) -> ComputerUseBackend:
    backend_name = os.environ.get("HERMES_COMPUTER_USE_BACKEND", "cua").lower()
    if backend_name in {"cua", "cua-driver", ""}:
        from tools.computer_use.cua_backend import CuaDriverBackend

        return CuaDriverBackend(permission_mode=permission_mode)
    if backend_name == "noop":  # pragma: no cover
        return _NoopBackend()
    raise RuntimeError(f"Unknown HERMES_COMPUTER_USE_BACKEND={backend_name!r}")


def _install_backend(sid: str, backend: ComputerUseBackend, permission_mode: str) -> None:
    """Record a backend in the session caches. Caller holds ``_backend_lock``."""
    _backends[sid] = backend
    _backend_call_locks[sid] = threading.RLock()
    _backend_permission_modes[sid] = permission_mode


def _pop_session_locked(sid: str) -> Tuple[Optional[ComputerUseBackend], Optional[threading.RLock]]:
    """Remove one session's cache entries; caller holds ``_backend_lock``."""
    _backend_permission_modes.pop(sid, None)
    return _backends.pop(sid, None), _backend_call_locks.pop(sid, None)


def _stop_backend(backend: ComputerUseBackend, call_lock: Optional[threading.RLock]) -> None:
    """Stop under the session call lock (if any) so an in-flight action finishes first.
    Never called under ``_backend_lock``: unrelated sessions stay free meanwhile. Raises."""
    if call_lock is not None:
        with call_lock:
            backend.stop()
    else:
        backend.stop()


def _get_backend(session_id: str = "") -> ComputerUseBackend:
    global _backend
    sid = str(session_id or "")
    while True:
        with _backend_lock:
            # Resolve the mode under the cache lock; YOLO mutation never holds
            # the approval lock while releasing this cache, so no lock cycle.
            permission_mode = _cua_permission_mode(sid)
            if sid == "" and _backend is not None and sid not in _backends:
                # Fold the empty-session injection hook into the session cache.
                _install_backend(sid, _backend, permission_mode)
            cached = _backends.get(sid)
            if cached is None:
                backend = _new_backend(permission_mode)
                # Starting under the cache lock preserves one-backend-per-session.
                # A concurrent mode toggle releases this backend before returning.
                backend.start()
                _install_backend(sid, backend, permission_mode)
                if sid == "":
                    _backend = backend
                return backend
            if _backend_permission_modes.get(sid, "standard") == permission_mode:
                return cached
            # Cua's permission mode cannot change after daemon startup: a /yolo
            # toggle replaces only this session's backend.
            _, stale_lock = _pop_session_locked(sid)
            if sid == "":
                _backend = None

        # Stop outside the cache lock; the loop re-reads the authoritative mode
        # before installing a replacement.
        with contextlib.suppress(Exception):
            _stop_backend(cached, stale_lock)


def release_computer_use_session(session_id: str) -> bool:
    """Release one session-owned backend (lifecycle seam for hosts/plugins).

    Cache entries are removed BEFORE stopping so new lookups cannot retain the
    stale target/ref namespace; approval state is cleared even without a backend.
    Returns True when a backend was released, False if already absent; idempotent.
    """
    global _backend
    sid = str(session_id or "")
    with _backend_lock:
        backend, call_lock = _pop_session_locked(sid)
        # Older callers/tests may populate only the `_backend` injection hook.
        if sid == "" and backend is None:
            backend = _backend
        if sid == "" and _backend is backend:
            _backend = None

    with _approval_lock:
        _session_auto_approve.pop(sid, None)
        _always_allow.pop(sid, None)
    if backend is None:
        return False
    try:
        _stop_backend(backend, call_lock)
    except Exception:
        logger.debug("computer_use backend release failed for session %s", sid, exc_info=True)
    return True


def _shutdown_backend_atexit() -> None:
    """Stop all cached backends so cua-driver subprocesses don't outlive us.
    atexit only, no signal handlers: a ``SystemExit`` from a prompt_toolkit key
    binding corrupts its coroutine state and makes the process unkillable. Never raises."""
    global _backend
    # Drop the global lock before stop() — teardown budgets 5s and shouldn't
    # block an unrelated caller waiting to spawn.
    with _backend_lock:
        unique = {id(b): (b, _backend_call_locks.get(sid)) for sid, b in _backends.items()}
        if _backend is not None:
            unique.setdefault(id(_backend), (_backend, _backend_call_locks.get("")))
        _backend = None
        for cache in (_backends, _backend_call_locks, _backend_permission_modes):
            cache.clear()
    with _approval_lock:
        for cache in (_session_auto_approve, _always_allow, _escalation_warned):
            cache.clear()

    for backend, call_lock in unique.values():
        try:
            _stop_backend(backend, call_lock)
        except Exception as e:
            logger.debug("cua-driver atexit teardown failed: %s", e)


atexit.register(_shutdown_backend_atexit)


def reset_backend_for_tests() -> None:  # pragma: no cover
    """Test helper — tear down the cached backend and per-session state."""
    _shutdown_backend_atexit()
    _AUX_VISION_ROUTE_CACHE.clear()


class _NoopBackend(ComputerUseBackend):  # pragma: no cover
    """Test/CI stub. Records calls; returns trivial results."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []
        self._started = False

    def start(self) -> None: self._started = True
    def stop(self) -> None: self._started = False
    def is_available(self) -> bool: return True

    def _record(self, name: str, kw: Dict[str, Any]) -> ActionResult:
        self.calls.append((name, kw))
        return ActionResult(ok=True, action=name)

    def _record_list(self, name: str) -> List[Dict[str, Any]]:
        self.calls.append((name, {}))
        return []

    def capture(self, mode: str = "som", app: Optional[str] = None,
                pid: Optional[int] = None, window_id: Optional[int] = None) -> CaptureResult:
        self.calls.append(("capture", {"mode": mode, "app": app, "pid": pid, "window_id": window_id}))
        return CaptureResult(mode=mode, width=1024, height=768, png_b64=None,
                             elements=[], app=app or "", window_title="")

    def click(self, **kw) -> ActionResult: return self._record("click", kw)
    def drag(self, **kw) -> ActionResult: return self._record("drag", kw)
    def scroll(self, **kw) -> ActionResult: return self._record("scroll", kw)
    def type_text(self, text: str, **kw) -> ActionResult: return self._record("type", {"text": text, **kw})
    def key(self, keys: str, **kw) -> ActionResult: return self._record("key", {"keys": keys, **kw})

    def list_apps(self) -> List[Dict[str, Any]]: return self._record_list("list_apps")
    def list_windows(self) -> List[Dict[str, Any]]: return self._record_list("list_windows")

    def focus_app(self, app: str, raise_window: bool = False) -> ActionResult:
        return self._record("focus_app", {"app": app, "raise": raise_window})

    def set_value(self, value: str, element: Optional[int] = None) -> ActionResult:
        return self._record("set_value", {"value": value, "element": element})


# ── Dispatch ────────────────────────────────────────────────────────────────

def handle_computer_use(args: Dict[str, Any], **kwargs) -> Any:
    """Main entry point — dispatched by tools.registry. Returns a JSON string
    (text-only) or a dict marked `_multimodal` (image + summary)."""
    action = (args.get("action") or "").strip().lower()
    if not action:
        return json.dumps({"error": "missing `action`"})
    # Per-run key for approval-state and daemon-mode isolation across sessions.
    session_id = str(kwargs.get("session_id") or "")

    err = _reject_unsafe(action, args)
    if err is not None:
        return err

    # Approval gate (destructive actions only). Persistent focus is a separate,
    # visible side effect with its own scope even when the input rung is approved.
    scopes = [action] if action in _DESTRUCTIVE_ACTIONS else []
    if args.get("bring_to_front") or (action == "focus_app" and args.get("raise_window")):
        scopes.append("bring_to_front")
    for scope in scopes:
        err = _request_approval(scope, args, session_id)
        if err is not None:
            return err

    try:
        backend = _get_backend(session_id=session_id)
    except Exception as e:
        return json.dumps({
            "error": f"computer_use backend unavailable: {e}",
            "hint": "If the cua-driver binary is missing, run `hermes computer-use install`. "
                    "If a Python dependency is missing, the error above shows the exact install command."})

    try:
        with _backend_lock:
            call_lock = _backend_call_locks.setdefault(session_id, threading.RLock())
        with call_lock:
            return _dispatch(backend, action, args)
    except Exception as e:
        logger.exception("computer_use %s failed", action)
        return json.dumps({"error": f"{action} failed: {e}"})


def _request_approval(action: str, args: Dict[str, Any],
                      session_id: str = "") -> Optional[str]:
    """None if approved, else a JSON error string. Scoped by (action, delivery_mode)
    AND session_id: foreground delivery is a visible focus change, so a background
    ``approve_session`` must NOT cover it; the blanket ``always_approve`` does."""
    scope_key = (action, "foreground" if args.get("delivery_mode") == "foreground" else "background")
    with _approval_lock:
        if _session_auto_approve.get(session_id) or scope_key in _always_allow.get(session_id, set()):
            return None
    cb = _approval_callback
    if cb is None:
        # No CLI approval wired — default allow. Gateway approval is handled
        # one layer out via the normal tool-approval infra.
        return None
    try:
        verdict = cb(action, args, _summarize_action(action, args))
    except Exception as e:
        logger.warning("approval callback failed: %s", e)
        verdict = "deny"
    if verdict == "approve_once":
        return None
    if verdict in ("approve_session", "always_approve"):
        with _approval_lock:
            _always_allow.setdefault(session_id, set()).add(scope_key)
            if verdict == "always_approve":
                _session_auto_approve[session_id] = True
        return None
    if verdict == "timeout":
        return json.dumps({"error": ("approval prompt timed out — the user did not respond. "
                                     "Silence is not consent; do not retry without the user."),
                           "action": action})
    return json.dumps({"error": "denied by user", "action": action})


# action -> (forced button or None, click_count)
_CLICK_VARIANTS = {"click": (None, 1), "double_click": (None, 2),
                   "right_click": ("right", 1), "middle_click": ("middle", 1)}


def _summarize_click(action: str, args: Dict[str, Any], fg: str) -> str:
    if args.get("element") is not None:
        return f"{action} element #{args['element']}{fg}"
    coord = args.get("coordinate")
    return f"{action} at {tuple(coord)}{fg}" if coord else action + fg


def _summarize_type(action: str, args: Dict[str, Any], fg: str) -> str:
    text = args.get("text", "")
    return f"type {text[:60]!r}" + ("..." if len(text) > 60 else "") + fg


# action -> (action, args, fg_suffix) -> one-line approval summary
_ACTION_SUMMARIES: Dict[str, Callable[[str, Dict[str, Any], str], str]] = {
    **dict.fromkeys(_CLICK_VARIANTS, _summarize_click),
    "drag": lambda a, args, fg: (f"drag {args.get('from_element') or args.get('from_coordinate')} → "
                                 f"{args.get('to_element') or args.get('to_coordinate')}{fg}"),
    "scroll": lambda a, args, fg: f"scroll {args.get('direction', '?')} x{args.get('amount', 3)}{fg}",
    "type": _summarize_type,
    "key": lambda a, args, fg: f"key {args.get('keys', '')!r}{fg}",
    "focus_app": lambda a, args, fg: (f"focus {args.get('app', '')!r}"
                                      + (" (raise)" if args.get("raise_window") else "")),
}


def _summarize_action(action: str, args: Dict[str, Any]) -> str:
    fg = (" [FOREGROUND — briefly raises the window / changes focus]"
          if args.get("delivery_mode") == "foreground" else "")
    summarize = _ACTION_SUMMARIES.get(action)
    return summarize(action, args, fg) if summarize else action + fg


# --- read-only / focus actions: (backend, args) -> final tool result ---------

def _do_capture(backend: ComputerUseBackend, args: Dict[str, Any]) -> Any:
    mode = str(args.get("mode", "som"))
    if mode not in {"som", "vision", "ax"}:
        return json.dumps({"error": f"bad mode {mode!r}; use som|vision|ax"})
    capture_kwargs: Dict[str, Any] = {"mode": mode, "app": args.get("app")}
    # pid/window_id forwarded only when given so older backends keep their defaults.
    if args.get("pid") is not None or args.get("window_id") is not None:
        capture_kwargs.update({"pid": args.get("pid"), "window_id": args.get("window_id")})
    return _capture_response(backend.capture(**capture_kwargs))


def _do_focus_app(backend: ComputerUseBackend, args: Dict[str, Any]) -> Any:
    app = args.get("app")
    if not app:
        return json.dumps({"error": "focus_app requires `app`"})
    return _maybe_follow_capture(backend, backend.focus_app(app, raise_window=bool(args.get("raise_window"))),
                                 bool(args.get("capture_after")))


def _listing(key: str, items: List[Dict[str, Any]]) -> str:
    return json.dumps({key: items, "count": len(items)})


_SIMPLE_ACTIONS: Dict[str, Callable[[ComputerUseBackend, Dict[str, Any]], Any]] = {
    "capture": _do_capture,
    "wait": lambda backend, args: _text_response(backend.wait(float(args.get("seconds", 1.0)))),
    "list_apps": lambda backend, args: _listing("apps", backend.list_apps()),
    "list_windows": lambda backend, args: _listing("windows", backend.list_windows()),
    "focus_app": _do_focus_app,
}

# --- input actions: (backend, action, args, delivery_mode, bring_to_front)
#     -> ActionResult, or a JSON error string for a rejected call -------------

def _xy(args: Dict[str, Any]) -> Tuple[Any, Any]:
    coord = args.get("coordinate") or (None, None)
    return (coord[0], coord[1]) if coord and coord[0] is not None else (None, None)


def _do_click(backend, action, args, delivery_mode, bring_to_front):
    forced_button, click_count = _CLICK_VARIANTS[action]
    x, y = _xy(args)
    return backend.click(
        element=args.get("element"), x=x, y=y,
        button=forced_button or args.get("button") or "left", click_count=click_count,
        modifiers=args.get("modifiers"), delivery_mode=delivery_mode, bring_to_front=bring_to_front,
    )


def _do_drag(backend, action, args, delivery_mode, bring_to_front):
    has_elements = args.get("from_element") is not None and args.get("to_element") is not None
    has_coords = args.get("from_coordinate") and args.get("to_coordinate")
    if not has_elements and not has_coords:
        return json.dumps({"error": "drag requires from_coordinate/to_coordinate or from_element/to_element"})
    return backend.drag(
        from_element=args.get("from_element"), to_element=args.get("to_element"),
        from_xy=tuple(args["from_coordinate"]) if args.get("from_coordinate") else None,
        to_xy=tuple(args["to_coordinate"]) if args.get("to_coordinate") else None,
        button=args.get("button", "left"), modifiers=args.get("modifiers"),
        delivery_mode=delivery_mode, bring_to_front=bring_to_front,
    )


def _do_scroll(backend, action, args, delivery_mode, bring_to_front):
    x, y = _xy(args)
    return backend.scroll(
        direction=args.get("direction", "down"), amount=int(args.get("amount", 3)),
        element=args.get("element"), x=x, y=y,
        modifiers=args.get("modifiers"), delivery_mode=delivery_mode, bring_to_front=bring_to_front,
    )


def _do_set_value(backend, action, args, delivery_mode, bring_to_front):
    value = args.get("value")
    if value is None:
        return json.dumps({"error": "set_value requires `value`"})
    return backend.set_value(value=str(value), element=args.get("element"))


_INPUT_HANDLERS = {
    **dict.fromkeys(_CLICK_VARIANTS, _do_click),
    "drag": _do_drag, "scroll": _do_scroll, "set_value": _do_set_value,
    "type": lambda backend, action, args, dm, btf: backend.type_text(
        args.get("text", ""), delivery_mode=dm, bring_to_front=btf),
    "key": lambda backend, action, args, dm, btf: backend.key(
        args.get("keys", ""), delivery_mode=dm, bring_to_front=btf),
}
# Native input actions deliver to the backend's sticky target; `app=` on these
# calls is NOT a targeting parameter — see the mismatch guard in _dispatch.
_INPUT_ACTIONS = frozenset(_INPUT_HANDLERS)

# Unknown actions are never aliased (no repairing bad model output), but the
# nearest real action is named so a bare error isn't the only guidance.
_ACTION_SUGGESTIONS = {
    "hotkey": "key", "press_key": "key", "keypress": "key",
    "key_combo": "key", "shortcut": "key",
    "type_text": "type", "input_text": "type",
    "screenshot": "capture", "get_window_state": "capture",
    "left_click": "click", "mouse_click": "click",
}


def _dispatch(backend: ComputerUseBackend, action: str, args: Dict[str, Any]) -> Any:
    simple = _SIMPLE_ACTIONS.get(action)
    if simple is not None:
        return simple(backend, args)

    handler = _INPUT_HANDLERS.get(action)
    if handler is None:
        hint = _ACTION_SUGGESTIONS.get(str(action))
        if hint:
            return json.dumps({"error": (f"unknown action {action!r} — did you mean {hint!r}? "
                                         "See the action enum in the tool schema.")})
        return json.dumps({"error": f"unknown action {action!r}"})

    # app= guard: input goes to the sticky target from the last capture/focus_app
    # and the backend drops app= silently — refuse a clear mismatch rather than
    # type into the wrong window while reporting ok:true.
    requested_app = args.get("app")
    if isinstance(requested_app, str) and requested_app.strip():
        mismatch = _input_target_mismatch(backend, requested_app)
        if mismatch is not None:
            return json.dumps({
                "ok": False, "action": action, "code": "input_target_mismatch",
                "error": (f"{action} would go to the current target "
                          f"{mismatch!r}, not {requested_app.strip()!r} — input "
                          "actions always hit the sticky target from the last "
                          f"capture/focus_app. Call capture(app={requested_app.strip()!r}) "
                          "or focus_app first, then retry."),
            })

    # delivery_mode / bring_to_front thread through every input action so the
    # model can escalate background → foreground per cua-driver's ladder.
    res = handler(backend, action, args, args.get("delivery_mode"), bool(args.get("bring_to_front")))
    if isinstance(res, str):
        return res
    return _maybe_follow_capture(backend, res, bool(args.get("capture_after")))


# ── Response shaping ────────────────────────────────────────────────────────

def _classify_action_result(res: ActionResult) -> Dict[str, Any]:
    """Next ladder step from semantic evidence, in precedence order. Escalation is
    advisory: it never overrides a confirmed effect nor licenses repeating input."""
    if res.effect == "confirmed" or res.verified is True:
        return {"decision": "done"}
    if res.effect == "unverifiable":
        return {"decision": "verify_fresh_state",
                "hint": ("Input was delivered but not confirmed. Re-capture and check "
                         "the result BEFORE any retry — do not repeat the input on an "
                         "escalation recommendation alone.")}
    if res.effect == "suspected_noop" or not res.ok or res.code is not None:
        decision: Dict[str, Any] = {"decision": "escalate"}
        if isinstance(res.escalation, dict):
            decision["recommended"] = res.escalation.get("recommended")
        decision["hint"] = ("The input likely did not land. Climb one rung following "
                            "`recommended`: 'px' → re-issue by coordinate; 'foreground' (or a "
                            "failed pixel click) → re-issue with delivery_mode='foreground' "
                            "(separate approval). Do not predict the rung from the app being "
                            "Electron/Chromium — react to this signal.")
        return decision
    # Transport success without semantic proof is not proof of effect.
    return {"decision": "verify_fresh_state",
            "hint": ("Transport succeeded but the effect is unproven. Re-capture and "
                     "confirm before continuing.")}


def _action_payload(res: ActionResult) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"ok": res.ok, "action": res.action}
    if res.message:
        payload["message"] = res.message
    # cua-driver's structured verdict, only for fields it returned (None = old
    # driver). ok is transport success; effect/escalation are the semantic verdict.
    for key in ("verified", "effect", "escalation", "path", "degraded", "delivery_mode", "code"):
        value = getattr(res, key)
        if value is not None:
            payload[key] = value
    if res.meta:
        payload["meta"] = res.meta
    payload["verdict"] = _classify_action_result(res)
    return payload


def _text_response(res: ActionResult) -> str:
    return json.dumps(_action_payload(res))


# Cap for the AX `elements` array: dense UIs publish 500+ AX nodes, which would
# exhaust context after one capture. The full tree spills to `elements_file`.
_DEFAULT_MAX_ELEMENTS = 100
# Some providers reject images below 8x8 before the model sees the tool result;
# such captures fall back to the AX/SOM text payload.
_MIN_PROVIDER_IMAGE_DIMENSION = 8
# Some AX trees (Discord/Slack via UIA, Electron chat clients) expose ENTIRE
# message bodies as labels; uncapped they blew the tool-result budget and leaked
# private chat text. Labels identify a control; captures aren't text extraction.
_MAX_ELEMENT_LABEL_CHARS = 120
# Bounded cache trails: every dense capture can spill, and CLI-only sessions
# never run the gateway's periodic media-cache cleanup.
_MAX_SPILL_FILES = 20
_MAX_CAPTURE_FILES = 20


def _image_dimensions_from_b64(image_b64: str) -> Optional[Tuple[int, int]]:
    """(width, height) of an inline PNG/JPEG screenshot, or None."""
    if not image_b64:
        return None
    try:
        raw = base64.b64decode(image_b64, validate=False)
    except Exception:
        return None
    return image_dimensions_from_bytes(raw)


def _capture_mime(cap: CaptureResult) -> str:
    """Prefer cua-driver's explicit MIME type; sniff the base64 prefix for older
    builds (JPEG base64 starts with /9j/, PNG with iVBOR)."""
    if cap.image_mime_type:
        return cap.image_mime_type
    return "image/jpeg" if (cap.png_b64 or "")[:8].startswith("/9j/") else "image/png"


def _capture_image_ext(cap: CaptureResult) -> str:
    """File extension matching the on-disk bytes so MIME sniffing agrees."""
    return ".jpg" if _capture_mime(cap).lower() == "image/jpeg" else ".png"


def _present(**fields: Any) -> Dict[str, Any]:
    """Only the truthy optional fields, in the given order."""
    return {k: v for k, v in fields.items() if v}


def _text_capture_payload(
    cap: CaptureResult, elements: List[UIElement], total_elements: int,
    width: int, height: int, summary: str, *,
    extra: Optional[Dict[str, Any]] = None, truncated_elements: int = 0,
    elements_file: Optional[str] = None, screenshot_path: Optional[str] = None,
    bounds_scale: Optional[float] = None,
) -> str:
    """JSON text payload shared by the AX, vision-unavailable and aux-vision branches.
    Key order is contract: fixed fields, ``extra`` branch markers, then set optionals."""
    payload: Dict[str, Any] = {
        "mode": cap.mode, "width": width, "height": height,
        "app": cap.app, "window_title": cap.window_title,
        "elements": [_element_to_dict(e) for e in elements],
        "total_elements": total_elements, "summary": summary,
        **(extra or {}),
    }
    payload.update(_present(truncated_elements=truncated_elements, elements_file=elements_file,
                            screenshot_path=screenshot_path, bounds_scale=bounds_scale))
    return json.dumps(payload)


def _capture_summary_lines(
    cap: CaptureResult, visible: List[UIElement], total: int, width: int, height: int,
    bounds_scale: Optional[float], elements_file: Optional[str], screenshot_path: Optional[str],
    omitted_dims: Optional[Tuple[int, int]],
) -> List[str]:
    """Human-readable capture summary. Line ORDER is contract. Indexes only what is
    surfaced in `elements`, otherwise the summary names indices the model can't find."""
    bounds_note = _bounds_space_note(visible, width, height)
    if bounds_note and bounds_scale:
        bounds_note += (f"; estimated scale ~{bounds_scale}x (screenshot position x "
                        f"{bounds_scale} ≈ native coordinate)")
    lines = [
        f"capture mode={cap.mode} {width}x{height}"
        + (f" app={cap.app}" if cap.app else "") + (f" window={cap.window_title!r}" if cap.window_title else ""),
        f"{total} interactable element(s):",
    ]
    if bounds_note:
        lines.append(f"  ({bounds_note})")
    if screenshot_path:
        lines.append(f"  (shareable screenshot saved to {screenshot_path})")
    if cap.note:
        lines.append(f"  ({cap.note})")
    if elements_file:
        lines.append(f"  (full element tree with untruncated labels saved to "
                     f"{elements_file} — read_file/search_files it if you need "
                     "dropped label text or elements beyond the cap)")
    lines.extend(_format_elements(visible))
    if omitted_dims:
        lines.append(f"  (screenshot omitted: {omitted_dims[0]}x{omitted_dims[1]} "
                     f"is below the {_MIN_PROVIDER_IMAGE_DIMENSION}x{_MIN_PROVIDER_IMAGE_DIMENSION} "
                     "provider minimum)")
    return lines


def _multimodal_capture(cap: CaptureResult, summary: str, width: int, height: int, total: int,
                        screenshot_path: Optional[str], elements_file: Optional[str],
                        bounds_scale: Optional[float]) -> Dict[str, Any]:
    """Envelope carrying the screenshot (not the elements array, so no truncation note)."""
    return {
        "_multimodal": True,
        "content": [{"type": "text", "text": summary},
                    {"type": "image_url",
                     "image_url": {"url": f"data:{_capture_mime(cap)};base64,{cap.png_b64}"}}],
        "text_summary": summary,
        "meta": {"mode": cap.mode, "width": width, "height": height,
                 "elements": total, "png_bytes": cap.png_bytes_len,
                 **_present(screenshot_path=screenshot_path, elements_file=elements_file,
                            bounds_scale=bounds_scale)},
    }


def _capture_response(cap: CaptureResult, max_elements: int = _DEFAULT_MAX_ELEMENTS) -> Any:
    total = len(cap.elements)
    visible = cap.elements[:max_elements]
    truncated = max(0, total - len(visible))
    dims = _image_dimensions_from_b64(cap.png_b64 or "")
    width, height = dims or (cap.width, cap.height)
    bounds_scale = _bounds_scale(visible, width, height)
    # Capped labels / capped element array: spill the complete tree for on-demand reads.
    lost_detail = bool(truncated) or any(len(e.label) > _MAX_ELEMENT_LABEL_CHARS for e in visible)
    elements_file = _spill_elements_to_file(cap) if lost_detail else None
    image_too_small = bool(dims) and min(dims) < _MIN_PROVIDER_IMAGE_DIMENSION
    has_image = bool(cap.png_b64) and cap.mode != "ax" and not image_too_small
    screenshot_path = _persist_capture_image(cap) if has_image else None
    lines = _capture_summary_lines(cap, visible, total, width, height, bounds_scale,
                                   elements_file, screenshot_path, dims if image_too_small else None)
    # Multimodal/aux paths use this summary; text paths append notes and rebuild.
    summary = "\n".join(lines)

    extra = None
    if has_image:
        # Hand the screenshot to auxiliary.vision (text-only result) when the main
        # model may not consume images natively; returning the multimodal envelope
        # unconditionally tripped HTTP 404/400 at the provider boundary.
        if not _should_route_through_aux_vision():
            return _multimodal_capture(cap, summary, width, height, total,
                                       screenshot_path, elements_file, bounds_scale)
        routed = _route_capture_through_aux_vision(
            cap, summary, visible_elements=visible, truncated_elements=truncated,
            elements_file=elements_file, screenshot_path=screenshot_path,
        )
        if routed is not None:
            return routed
        # Aux routing requested but failed (vision node down, empty analysis...).
        # The multimodal envelope could now break with a provider error, so
        # degrade to the AX/SOM text payload.
        lines.append("  (vision unavailable: the auxiliary vision model could not "
                     "be reached; screenshot omitted. Element-index actions still "
                     "work — drive via the element list above.)")
        extra = {"vision_unavailable": True}
    # Text paths carry the `elements` array, so the truncation note applies.
    if truncated:
        lines.append(
            f"  (response truncated to {len(visible)} of {total} elements; "
            "the full tree is in elements_file — read_file/search_files it, or pass app= to narrow scope)")
    return _text_capture_payload(
        cap, visible, total, width, height, "\n".join(lines),
        extra=extra, truncated_elements=truncated, elements_file=elements_file,
        screenshot_path=screenshot_path, bounds_scale=bounds_scale,
    )


# ── auxiliary.vision routing for captured screenshots ───────────────────────

# Longest image side handed to the aux vision model. Full-resolution desktop
# captures tokenize heavily and can overflow small local-model context windows;
# ~1456px keeps SOM badges legible while cutting per-capture vision latency.
_MAX_VISION_DIM = 1456


def _shrink_capture_for_vision(raw: bytes, ext: str,
                               max_dim: int = _MAX_VISION_DIM,
                               ) -> tuple[bytes, Optional[str]]:
    """Downscale encoded image bytes so the longest side is <= max_dim.
    Returns ``(bytes, scale_note)``; note is None when unchanged (fits, or Pillow
    unavailable/failed), else it tells the vision model the factor so reported
    coordinates map back to the real screen instead of being silently wrong."""
    try:
        from io import BytesIO
        from PIL import Image
        img = Image.open(BytesIO(raw))
        if max(img.size) <= max_dim:
            return raw, None
        orig_w, orig_h = img.size
        img.thumbnail((max_dim, max_dim))
        new_w, new_h = img.size
        out = BytesIO()
        img.save(out, format="JPEG" if ext == ".jpg" else "PNG")
        fx = orig_w / new_w if new_w else 1.0
        fy = orig_h / new_h if new_h else 1.0
        if f"{fx:.2f}" == f"{fy:.2f}":
            factor_clause = f"multiply any coordinates you report by {fx:.2f} to map back to the real screen."
        else:
            factor_clause = (f"multiply any x coordinates you report by {fx:.2f} and "
                             f"any y coordinates by {fy:.2f} to map back to the real screen.")
        return out.getvalue(), (f"Screenshot downscaled from {orig_w}x{orig_h} to "
                                f"{new_w}x{new_h} for vision; {factor_clause}")
    except Exception as exc:
        logger.debug("computer_use: vision downscale skipped: %s", exc)
        return raw, None


def _should_route_through_aux_vision() -> bool:
    """True when ``_capture_response`` should hand the PNG to aux vision. Any failure
    returns False (fail open) so a broken config never silently drops the screenshot
    for vision-capable main models."""
    try:
        from agent.auxiliary_client import _read_main_model, _read_main_provider
        from hermes_cli.config import load_config
        from tools.computer_use.vision_routing import should_route_capture_to_aux_vision
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("computer_use: aux-vision routing import failed: %s", exc)
        return False
    try:
        provider, model = _read_main_provider() or "", _read_main_model() or ""
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("computer_use: aux-vision routing config read failed: %s", exc)
        return False
    cache_key = (str(provider), str(model))
    cached = _AUX_VISION_ROUTE_CACHE.get(cache_key)
    if cached is not None:
        return cached
    try:
        decision = bool(should_route_capture_to_aux_vision(provider, model, load_config()))
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("computer_use: aux-vision routing decision failed: %s", exc)
        return False
    _AUX_VISION_ROUTE_CACHE[cache_key] = decision
    return decision


def _capture_after_mode() -> str:
    """Mode for ``capture_after`` follow-ups. Default ``som`` (screenshot)."""
    try:
        from hermes_cli.config import load_config

        raw = ((load_config() or {}).get("computer_use") or {}).get("capture_after_mode", "som")
    except Exception:
        return "som"
    mode = str(raw or "som").strip().lower()
    return mode if mode in {"som", "vision", "ax"} else "som"


def _route_capture_through_aux_vision(
    cap: CaptureResult,
    summary: str,
    *,
    visible_elements: Optional[List[UIElement]] = None,
    truncated_elements: int = 0,
    elements_file: Optional[str] = None,
    screenshot_path: Optional[str] = None,
) -> Optional[str]:
    """Pre-analyse the capture via ``vision_analyze_tool`` (temp file under
    ``$HERMES_HOME/cache/vision/``) and merge the description with the AX/SOM
    summary into one text payload. Returns JSON, or None on any failure."""
    if not cap.png_b64:
        return None
    try:
        from model_tools import _run_async
        from tools.vision_tools import vision_analyze_tool
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("computer_use: aux-vision import failed: %s", exc)
        return None

    try:
        raw = base64.b64decode(cap.png_b64, validate=False)
    except Exception as exc:
        logger.debug("computer_use: failed to decode capture base64: %s", exc)
        return None

    temp_image_path = None
    try:
        ext = _capture_image_ext(cap)
        temp_image_path = _cache_file("cache/vision", "temp_vision_images", f"computer_use_{uuid.uuid4().hex}{ext}")
        raw, scale_note = _shrink_capture_for_vision(raw, ext)
        temp_image_path.write_bytes(raw)

        prompt = ("Describe what is visible in this desktop application screenshot in "
                  "concise but specific terms. Mention the app name and window "
                  "title if visible, the overall layout, any labelled buttons, "
                  "menus or text fields, and any prominent text content the user "
                  "would need to know about. Do not invent details that are not "
                  f"actually visible.\n\nAX/SOM index for cross-reference:\n{summary}")
        if scale_note:
            prompt += f"\n\nNote: {scale_note}"

        result_json = _run_async(vision_analyze_tool(str(temp_image_path), prompt))
    except Exception as exc:
        logger.warning("computer_use: auxiliary.vision pre-analysis failed (%s); "
                       "returning to caller without aux analysis", exc)
        return None
    finally:
        if temp_image_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(str(temp_image_path))

    analysis_text = ""
    if isinstance(result_json, str):
        try:
            parsed = json.loads(result_json)
            if isinstance(parsed, dict):
                analysis_text = str(parsed.get("analysis") or "").strip()
        except (TypeError, json.JSONDecodeError):
            analysis_text = result_json.strip()
    if not analysis_text:
        return None
    # Same element cap as every other capture branch; dumping cap.elements in
    # full would bypass max_elements exactly for non-vision main models.
    elements_out = cap.elements if visible_elements is None else visible_elements
    return _text_capture_payload(
        cap, elements_out, len(cap.elements), cap.width, cap.height, summary,
        extra={"vision_analysis": analysis_text, "vision_analysis_routed_via": "auxiliary.vision"},
        truncated_elements=truncated_elements, elements_file=elements_file,
        screenshot_path=screenshot_path,
    )


def _maybe_follow_capture(backend: ComputerUseBackend, res: ActionResult, do_capture: bool) -> Any:
    # No follow-up capture after a failed action: a normal-looking screenshot
    # would suggest it succeeded.
    if not do_capture or not res.ok:
        return _text_response(res)
    try:
        # Recapture the exact window when known: on Linux several unrelated
        # windows may share an app name, so app-only recapture can switch targets.
        target = getattr(backend, "_last_target", None) or {}
        pid, window_id = target.get("pid"), target.get("window_id")
        mode = _capture_after_mode()
        if pid is not None and window_id is not None:
            cap = backend.capture(mode=mode, pid=pid, window_id=window_id)
        else:
            cap = backend.capture(mode=mode, app=getattr(backend, "_last_app", None))
    except Exception as e:
        logger.warning("follow-up capture failed: %s", e)
        return _text_response(res)
    resp = _capture_response(cap)
    if isinstance(resp, dict) and resp.get("_multimodal"):
        # Keep the evidence/verdict contract visible alongside the image — it
        # governs whether repeating input is allowed.
        prefix = json.dumps(_action_payload(res))
        resp["content"][0]["text"] = prefix + "\n\n" + resp["content"][0]["text"]
        resp["text_summary"] = prefix + "\n\n" + resp["text_summary"]
        resp["action_result"] = _action_payload(res)
        return resp
    try:  # text capture: merge the action payload in
        data = json.loads(resp)
    except (TypeError, json.JSONDecodeError):
        data = {"capture": resp}
    data.update(_action_payload(res))
    return json.dumps(data)


def _bounds_unknown(bounds) -> bool:
    """True when the AX tree reported no real geometry. KDE/Qt apps report
    ``[0, 0, 0, 0]`` for elements clickable by index; serializing that as a rect
    invites ``coordinate=[0, 0]`` clicks on the screen corner."""
    try:
        return all(int(v) == 0 for v in bounds)
    except (TypeError, ValueError):
        return False


def _format_elements(elements: List[UIElement], max_lines: int = 40) -> List[str]:
    out: List[str] = []
    for e in elements[:max_lines]:
        label = e.label.replace("\n", " ")[:60]
        where = "@ bounds-unknown (click by element index)" if _bounds_unknown(e.bounds) else f"@ {e.bounds}"
        out.append(f"  #{e.index} {e.role} {label!r} {where}" + (f" [{e.app}]" if e.app else ""))
    if len(elements) > max_lines:
        out.append(f"  ... +{len(elements) - max_lines} more (call capture with app= to narrow)")
    return out


def _cache_file(subdir: str, legacy: str, name: str, pattern: str = "", cap: int = 0):
    """Path for a new file under ``$HERMES_HOME/<subdir>`` (dir created). With
    ``pattern``/``cap``, first unlinks the oldest matching files so at most ``cap - 1``
    remain (best-effort). Imports lazily so tests can patch ``get_hermes_dir``."""
    from hermes_constants import get_hermes_dir

    cache_dir = get_hermes_dir(subdir, legacy)
    cache_dir.mkdir(parents=True, exist_ok=True)
    if pattern:
        with contextlib.suppress(Exception):
            files = sorted(cache_dir.glob(pattern), key=lambda p: p.stat().st_mtime)
            for stale in files[: max(0, len(files) - (cap - 1))]:
                stale.unlink(missing_ok=True)
    return cache_dir / name


def _persist_capture_image(cap: CaptureResult) -> Optional[str]:
    """Save a bounded copy of the capture in Hermes' media cache so attachment
    surfaces can deliver it; returns the path. Best-effort: an unwritable cache
    must never break computer control."""
    if not cap.png_b64:
        return None
    try:
        raw = base64.b64decode(cap.png_b64, validate=False)
        path = _cache_file("cache/images", "image_cache", f"computer_use_{uuid.uuid4().hex}{_capture_image_ext(cap)}",
                           "computer_use_*.*", _MAX_CAPTURE_FILES)
        path.write_bytes(raw)
        return str(path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("computer_use: screenshot persistence failed: %s", exc)
        return None


def _spill_elements_to_file(cap: CaptureResult) -> Optional[str]:
    """Write the FULL element tree (untruncated labels) to a cache file — the
    read_file/search_files escape hatch for capped text. Returns the path, or None
    on any failure (a capture must never fail on an unwritable cache)."""
    try:
        path = _cache_file("cache/computer_use", "computer_use_cache", f"elements_{uuid.uuid4().hex}.json",
                           "elements_*.json", _MAX_SPILL_FILES)
        payload = {
            "app": cap.app,
            "window_title": cap.window_title,
            "total_elements": len(cap.elements),
            "elements": [
                {"index": e.index, "role": e.role, "label": e.label,
                 "bounds": list(e.bounds), "app": e.app}
                for e in cap.elements
            ],
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
        return str(path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("computer_use: element spill failed: %s", exc)
        return None


def _bounds_divergence(elements: List[UIElement], image_width: int, image_height: int) -> Optional[Tuple[int, int]]:
    """(max right edge, max bottom edge) of element bounds when they exceed the
    screenshot, else None. 5% slack: window chrome can hang a few px past the
    captured frame without implying a different coordinate space."""
    if not elements or image_width <= 0 or image_height <= 0:
        return None
    max_x = max_y = 0
    for e in elements:
        try:
            x, y, w, h = e.bounds
        except (TypeError, ValueError):
            continue
        max_x = max(max_x, int(x) + int(w))
        max_y = max(max_y, int(y) + int(h))
    if max_x <= image_width * 1.05 and max_y <= image_height * 1.05:
        return None
    return max_x, max_y


def _bounds_scale(elements: List[UIElement], image_width: int, image_height: int) -> Optional[float]:
    """Estimated native-bounds → screenshot-pixel scale factor, or None when the
    spaces don't diverge (same condition as ``_bounds_space_note``). Larger axis
    ratio wins so real extent data drives it; rounded to 2 decimals (heuristic)."""
    extent = _bounds_divergence(elements, image_width, image_height)
    if extent is None:
        return None
    return round(max(extent[0] / image_width, extent[1] / image_height), 2)


def _bounds_space_note(elements: List[UIElement], image_width: int, image_height: int) -> Optional[str]:
    """Warn when element bounds live in a different coordinate space: on HiDPI
    displays AX bounds are native while the screenshot is downscaled, so coordinate=
    clicks read off the screenshot missed by the scale factor."""
    extent = _bounds_divergence(elements, image_width, image_height)
    if extent is None:
        return None
    return (f"element bounds are in native desktop coordinates (extend to "
            f"~{extent[0]}x{extent[1]}), NOT screenshot pixels ({image_width}x"
            f"{image_height}). coordinate= clicks expect the native space — "
            "derive click points from element bounds, or scale screenshot "
            "positions up accordingly")


def _element_to_dict(e: UIElement) -> Dict[str, Any]:
    # A zero rect is "geometry unknown", not a position — null it so no
    # coordinate= is ever derived from it. The element index still works.
    out: Dict[str, Any] = {
        "index": e.index, "role": e.role, "label": e.label[:_MAX_ELEMENT_LABEL_CHARS],
        "bounds": None if _bounds_unknown(e.bounds) else list(e.bounds), "app": e.app,
    }
    if len(e.label) > _MAX_ELEMENT_LABEL_CHARS:
        out["label_truncated"] = True
    return out


# ── Availability check (used by the tool registry check_fn) ─────────────────

def check_computer_use_requirements() -> bool:
    """True iff computer_use can run here: macOS/Windows/Linux + cua-driver binary
    (or env override). `hermes computer-use doctor` names blocked Linux checks."""
    if sys.platform not in ("darwin", "win32", "linux"):
        return False
    from tools.computer_use.cua_backend import cua_driver_binary_available
    return cua_driver_binary_available()


def get_computer_use_schema() -> Dict[str, Any]:
    from tools.computer_use.schema import COMPUTER_USE_SCHEMA
    return COMPUTER_USE_SCHEMA
