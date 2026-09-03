"""`computer_use` tool entry point: any-model desktop control (macOS/Windows/Linux) via cua-driver.
Return contract: text-only results are a JSON string; captures / `capture_after=True` return
``{"_multimodal": True, "content": [text, image_url], "text_summary": <fallback>}`` (run_agent.py /
the Anthropic adapter turn it into provider-specific image tool content)."""

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
from collections import namedtuple
from functools import partial
from types import SimpleNamespace
from typing import Any, Callable, Dict, List, Optional, Tuple

from tools.computer_use.backend import ActionResult, CaptureResult, ComputerUseBackend, UIElement, image_dimensions_from_bytes

logger = logging.getLogger(__name__)


# ── Approval & safety ───────────────────────────────────────────────────────

_approval_callback = None

def set_approval_callback(cb) -> None:
    """Register the CLI approval prompt (terminal_tool pattern); ``cb(action, args, summary)`` ->
    "approve_once" | "approve_session" | "always_approve" | "deny"."""
    global _approval_callback
    _approval_callback = cb

# Hard-blocked regardless of approval level (e.g. logout kills the session Hermes runs in). Alt is
# canonicalized to option, so the Windows variants are blocked before any backend sees them.
_BLOCKED_KEY_COMBOS = {
    frozenset({"cmd", "shift", "backspace"}), frozenset({"cmd", "option", "backspace"}),  # empty trash / force delete
    frozenset({"cmd", "ctrl", "q"}), frozenset({"cmd", "shift", "q"}),                    # lock screen / log out
    frozenset({"cmd", "option", "shift", "q"}), frozenset({"win", "l"}),                  # force log out / lock
    frozenset({"ctrl", "option", "delete"}), frozenset({"ctrl", "option", "del"}), frozenset({"option", "f4"}),
}
_KEY_ALIASES = {"command": "cmd", "control": "ctrl", "alt": "option", "⌘": "cmd", "⌥": "option",
                "windows": "win", "super": "win", "meta": "win"}
# Dangerous shell patterns for the `type` action (last one: fork bomb).
_BLOCKED_TYPE_PATTERNS = [re.compile(p, re.IGNORECASE) for p in (
    r"curl\s+[^|]*\|\s*bash", r"curl\s+[^|]*\|\s*sh", r"wget\s+[^|]*\|\s*bash",
    r"\bsudo\s+rm\s+-[rf]", r"\brm\s+-rf\s+/\s*$", r":\s*\(\)\s*\{\s*:\|:\s*&\s*\}")]

def _canon_key_combo(keys: str) -> frozenset:
    # Split on "+" AND "-": cua-driver accepts hyphenated combos, so "ctrl-alt-delete" would bypass otherwise.
    return frozenset(_KEY_ALIASES.get(p, p) for p in (q.strip().lower() for q in re.split(r"\s*[+\-]\s*", keys)) if p)

def _reject_unsafe(action: str, args: Dict[str, Any]) -> Optional[str]:
    """JSON error for hard-blocked input, else None. Runs BEFORE the approval prompt."""
    if action == "type" and (pat := next((p.pattern for p in _BLOCKED_TYPE_PATTERNS if p.search(args.get("text", ""))), None)):
        return json.dumps({"error": f"blocked pattern in type text: {pat!r}",
                           "hint": "Dangerous shell patterns cannot be typed via computer_use."})
    if action == "key":
        combo = _canon_key_combo(args.get("keys", ""))
        if (blocked := next((b for b in _BLOCKED_KEY_COMBOS if b.issubset(combo)), None)) is not None:
            return json.dumps({"error": f"blocked key combo: {sorted(blocked)}",
                               "hint": "Destructive system shortcuts are hard-blocked."})
    if args.get("bring_to_front") and args.get("delivery_mode") != "foreground":
        return json.dumps({"error": "bring_to_front requires delivery_mode='foreground'",
                           "code": "bring_to_front_requires_foreground"})
    return None

def _input_target_mismatch(backend, requested_app: str) -> Optional[str]:
    """Current sticky-target app when it provably differs from *requested_app*: both known and neither a substring
    of the other ('Google-chrome' vs 'chrome'). Unknown target -> None (fail open; the verify ladder catches it)."""
    last_app = getattr(backend, "_last_app", None)
    current, wanted = (last_app or "").strip().lower(), requested_app.strip().lower()
    return None if not current or not wanted or wanted in current or current in wanted else last_app


# ── Backend selection — env-swappable for tests ─────────────────────────────

# Per-Hermes-session cached backends (own cua-driver session, native target, refs, grant namespace).
# `_backend` is the backward-compatible empty-session injection hook (older tests).
_backend_lock = threading.Lock()
_backend: Optional[ComputerUseBackend] = None
_backends: Dict[str, ComputerUseBackend] = {}
_backend_call_locks: Dict[str, threading.RLock] = {}
_backend_permission_modes: Dict[str, str] = {}
_AUX_VISION_ROUTE_CACHE: Dict[Tuple[str, str], bool] = {}  # process-scoped: (provider, model) → bool
# Approval state keyed by session_id so a gateway serving concurrent sessions can't leak one run's
# "always approve" into another; callers without a session_id share "".
_approval_lock = threading.Lock()
_session_auto_approve: Dict[str, bool] = {}   # sid -> "always_approve everything"
_always_allow: Dict[str, set] = {}            # sid -> set of (action, delivery_mode) scope keys
_escalation_warned: set = set()               # sids already warned that a bypass widened the driver mode

def _configured_permission_mode() -> str:
    """Configured cua mode (standard | bounded; "standard" if unresolvable). bounded needs a
    computer_use.capability_manifest — the backend fails loudly without it."""
    with contextlib.suppress(Exception):
        from tools.computer_use.cua_backend import _cua_configured_permission_mode
        return _cua_configured_permission_mode()
    return "standard"

def _cua_permission_mode(session_id: str) -> str:
    """Map Hermes's approval bypass onto Cua's immutable mode; fails closed. Both identity namespaces are consulted
    (DB ``session_id`` and gateway ``session_key`` contextvar) or a gateway ``/yolo`` would be invisible here.
    Warns once per session that ``-z``/``--yolo`` swapped the driver onto a private ``unrestricted`` daemon, dropping
    the configured ceiling: deliberate (``unrestricted`` is not a config value) but easy to trigger by accident."""
    with contextlib.suppress(Exception):
        from tools.approval import get_current_session_key, is_approval_bypass_active_for_session
        if is_approval_bypass_active_for_session(session_id) or (
                bool(key := get_current_session_key(default="")) and is_approval_bypass_active_for_session(key)):
            with _approval_lock:
                warn = (key := str(session_id or "")) not in _escalation_warned
                _escalation_warned.add(key)
            if warn:
                configured = _configured_permission_mode()
                logger.warning(
                    "computer_use: approval bypass (--yolo / -z) escalated the cua-driver permission mode from the "
                    "configured '%s' to 'unrestricted' for this session. Runtime approval prompts are disabled and the "
                    "driver's residual ceilings no longer apply. Drop the bypass flag to keep '%s', or declare a "
                    "version-3 computer_use.capability_manifest to keep a ceiling on bypassed runs.", configured, configured)
            return "unrestricted"
    return _configured_permission_mode()

def _new_backend(permission_mode: str) -> ComputerUseBackend:
    backend_name = os.environ.get("HERMES_COMPUTER_USE_BACKEND", "cua").lower()
    if backend_name in {"cua", "cua-driver", ""}:
        from tools.computer_use.cua_backend import CuaDriverBackend
        return CuaDriverBackend(permission_mode=permission_mode)
    if backend_name != "noop":
        raise RuntimeError(f"Unknown HERMES_COMPUTER_USE_BACKEND={backend_name!r}")
    return _NoopBackend()  # pragma: no cover

def _install_backend(sid: str, backend: ComputerUseBackend, permission_mode: str) -> ComputerUseBackend:
    """Record a backend in the session caches (the empty session also mirrors it onto the ``_backend`` hook).
    Caller holds ``_backend_lock``."""
    global _backend
    _backends[sid], _backend_permission_modes[sid] = backend, permission_mode
    _backend_call_locks[sid] = threading.RLock()
    if sid == "":
        _backend = backend
    return backend

def _detach_locked(sid: str) -> Tuple[Optional[ComputerUseBackend], Optional[threading.RLock]]:
    """Remove one session's cache entries, plus the ``_backend`` injection hook when it aliases the empty session
    (older callers/tests may populate only the hook). Caller holds ``_backend_lock``."""
    global _backend
    _backend_permission_modes.pop(sid, None)
    backend, call_lock = _backends.pop(sid, None), _backend_call_locks.pop(sid, None)
    if sid == "":
        backend = _backend if backend is None else backend
        _backend = None if _backend is backend else _backend
    return backend, call_lock

def _stop_backend(backend: ComputerUseBackend, call_lock: Optional[threading.RLock],
                  on_error: Optional[Callable[[Exception], None]] = None) -> None:
    """Stop under the session call lock (if any) so an in-flight action finishes first. Never called under
    ``_backend_lock`` (unrelated sessions stay free). Raises unless ``on_error`` absorbs the failure."""
    try:
        with call_lock if call_lock is not None else contextlib.nullcontext():
            backend.stop()
    except Exception as e:
        if on_error is None:
            raise
        on_error(e)

def _get_backend(session_id: str = "") -> ComputerUseBackend:
    sid = str(session_id or "")
    while True:
        with _backend_lock:
            # Mode resolved under the cache lock; YOLO mutation never holds the approval lock while releasing it.
            permission_mode = _cua_permission_mode(sid)
            if sid == "" and _backend is not None and sid not in _backends:
                _install_backend(sid, _backend, permission_mode)  # fold the injection hook into the cache
            if (cached := _backends.get(sid)) is None:
                backend = _new_backend(permission_mode)
                backend.start()  # under the cache lock: one backend per session; a concurrent toggle releases it
                return _install_backend(sid, backend, permission_mode)
            if _backend_permission_modes.get(sid, "standard") == permission_mode:
                return cached
            # Cua's mode is immutable after daemon startup: a /yolo toggle replaces only this session's backend.
            _, stale_lock = _detach_locked(sid)  # stopped outside the cache lock; the loop re-reads the mode first
        _stop_backend(cached, stale_lock, lambda e: None)

def release_computer_use_session(session_id: str) -> bool:
    """Release one session-owned backend (lifecycle seam for hosts/plugins); idempotent, True iff one was released.
    Cache entries are removed BEFORE stopping so new lookups cannot retain the stale target/ref namespace; approval
    state is cleared even without a backend."""
    sid = str(session_id or "")
    with _backend_lock:
        backend, call_lock = _detach_locked(sid)
    with _approval_lock:
        _session_auto_approve.pop(sid, None), _always_allow.pop(sid, None)
    if backend is None:
        return False
    _stop_backend(backend, call_lock,
                  lambda e: logger.debug("computer_use backend release failed for session %s", sid, exc_info=True))
    return True

def _shutdown_backend_atexit() -> None:
    """Stop all cached backends so cua-driver subprocesses don't outlive us. atexit only, no signal handlers: a
    ``SystemExit`` from a prompt_toolkit key binding corrupts its coroutine state and makes the process unkillable.
    Never raises. Drops the global lock before stop(): teardown budgets 5s and must not block spawns."""
    global _backend
    with _backend_lock:
        unique = {id(b): (b, _backend_call_locks.get(sid)) for sid, b in _backends.items()}
        if _backend is not None:
            unique.setdefault(id(_backend), (_backend, _backend_call_locks.get("")))
        _backend = None
        _backends.clear(), _backend_call_locks.clear(), _backend_permission_modes.clear()
    with _approval_lock:
        _session_auto_approve.clear(), _always_allow.clear(), _escalation_warned.clear()
    for backend, call_lock in unique.values():
        _stop_backend(backend, call_lock, lambda e: logger.debug("cua-driver atexit teardown failed: %s", e))

atexit.register(_shutdown_backend_atexit)

def reset_backend_for_tests() -> None:  # pragma: no cover — tear down the cached backend and per-session state
    _shutdown_backend_atexit()
    _AUX_VISION_ROUTE_CACHE.clear()

def _noop_stub(name: str, *params: str, result: Any = None):
    # Recording stub: positional args are folded in under *params* (declared params default to None).
    def method(self, *pos, **kw):
        return self._record(name, {**dict.fromkeys(params), **dict(zip(params, pos)), **kw}, result)
    return method

class _NoopBackend(ComputerUseBackend):  # pragma: no cover
    """Test/CI stub (HERMES_COMPUTER_USE_BACKEND=noop). Records ``(name, kwargs)`` calls; returns trivial results."""

    def __init__(self) -> None:
        self.calls: List[Tuple[str, Dict[str, Any]]] = []

    def start(self) -> None: pass
    def stop(self) -> None: pass
    def is_available(self) -> bool: return True

    def _record(self, name: str, kw: Dict[str, Any], result: Any = None) -> Any:
        self.calls.append((name, kw))
        return ActionResult(ok=True, action=name) if result is None else result

    def capture(self, mode="som", app=None, pid=None, window_id=None) -> CaptureResult:
        return self._record("capture", {"mode": mode, "app": app, "pid": pid, "window_id": window_id},
                            CaptureResult(mode=mode, width=1024, height=768, png_b64=None, elements=[],
                                          app=app or "", window_title=""))

    click, drag, scroll = _noop_stub("click"), _noop_stub("drag"), _noop_stub("scroll")
    type_text, key, set_value = _noop_stub("type", "text"), _noop_stub("key", "keys"), _noop_stub("set_value", "value", "element")
    list_apps, list_windows = _noop_stub("list_apps", result=[]), _noop_stub("list_windows", result=[])
    def focus_app(self, app: str, raise_window: bool = False) -> ActionResult:
        return self._record("focus_app", {"app": app, "raise": raise_window})


# ── Dispatch ────────────────────────────────────────────────────────────────

def handle_computer_use(args: Dict[str, Any], **kwargs) -> Any:
    """Main entry point (tools.registry): a JSON string (text-only) or a dict marked `_multimodal`. Order: hard
    blocks (_reject_unsafe) -> approval scopes (destructive action, then 'bring_to_front' — persistent focus is a
    separate visible side effect with its own scope) -> backend -> dispatch under the session call lock."""
    action = (args.get("action") or "").strip().lower()
    if not action:
        return json.dumps({"error": "missing `action`"})
    session_id = str(kwargs.get("session_id") or "")  # approval-state / daemon-mode isolation key
    if (err := _reject_unsafe(action, args)) is not None:
        return err
    scopes = [action] if action in _ACTIONS and _ACTIONS[action].destructive else []
    if args.get("bring_to_front") or (action == "focus_app" and args.get("raise_window")):
        scopes.append("bring_to_front")
    for scope in scopes:
        if (err := _request_approval(scope, args, session_id)) is not None:
            return err
    try:
        backend = _get_backend(session_id=session_id)
    except Exception as e:
        return json.dumps({"error": f"computer_use backend unavailable: {e}",
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

def _request_approval(action: str, args: Dict[str, Any], session_id: str = "") -> Optional[str]:
    """None if approved, else a JSON error string. Scoped by (action, delivery_mode) AND session_id: foreground
    delivery is a visible focus change, so a background ``approve_session`` must NOT cover it; the blanket
    ``always_approve`` does. No CLI approval wired -> default allow (gateway approval runs one layer out)."""
    scope_key = (action, "foreground" if args.get("delivery_mode") == "foreground" else "background")
    with _approval_lock:
        if _session_auto_approve.get(session_id) or scope_key in _always_allow.get(session_id, set()):
            return None
    if (cb := _approval_callback) is None:
        return None
    try:
        verdict = cb(action, args, _summarize_action(action, args))
    except Exception as e:
        logger.warning("approval callback failed: %s", e)
        verdict = "deny"
    if verdict in ("approve_session", "always_approve"):
        with _approval_lock:
            _always_allow.setdefault(session_id, set()).add(scope_key)
            if verdict == "always_approve":
                _session_auto_approve[session_id] = True
    if verdict in ("approve_once", "approve_session", "always_approve"):
        return None
    return json.dumps({"error": ("approval prompt timed out — the user did not respond. Silence is not consent; "
                                 "do not retry without the user.") if verdict == "timeout" else "denied by user",
                       "action": action})

def _summarize_action(action: str, args: Dict[str, Any]) -> str:
    fg = " [FOREGROUND — briefly raises the window / changes focus]" if args.get("delivery_mode") == "foreground" else ""
    return _ACTIONS.get(action, _ActionSpec(None)).summarize(action, args, fg)

# --- handlers: (backend, action, args, **delivery) -> ActionResult (_dispatch applies the follow-up capture) or a
#     final str/dict result. `delivery` = delivery_mode + bring_to_front; only input actions use it.

def _xy(args: Dict[str, Any]) -> Tuple[Any, Any]:
    coord = args.get("coordinate")
    return (coord[0], coord[1]) if coord and coord[0] is not None else (None, None)

def _do_click(backend, action, args, button=None, count=1, **delivery):
    x, y = _xy(args)
    return backend.click(element=args.get("element"), x=x, y=y, button=button or args.get("button") or "left",
                         click_count=count, modifiers=args.get("modifiers"), **delivery)

def _do_drag(backend, action, args, **delivery):
    src, dst = args.get("from_coordinate"), args.get("to_coordinate")
    if (args.get("from_element") is None or args.get("to_element") is None) and not (src and dst):
        return json.dumps({"error": "drag requires from_coordinate/to_coordinate or from_element/to_element"})
    return backend.drag(from_element=args.get("from_element"), to_element=args.get("to_element"),
                        from_xy=tuple(src) if src else None, to_xy=tuple(dst) if dst else None,
                        button=args.get("button", "left"), modifiers=args.get("modifiers"), **delivery)

def _do_scroll(backend, action, args, **delivery):
    x, y = _xy(args)
    return backend.scroll(direction=args.get("direction", "down"), amount=int(args.get("amount", 3)),
                          element=args.get("element"), x=x, y=y, modifiers=args.get("modifiers"), **delivery)

def _do_set_value(backend, action, args, **delivery):
    if args.get("value") is None:
        return json.dumps({"error": "set_value requires `value`"})
    return backend.set_value(value=str(args["value"]), element=args.get("element"))

def _do_capture(backend, action, args, **_):
    mode = str(args.get("mode", "som"))
    if mode not in {"som", "vision", "ax"}:
        return json.dumps({"error": f"bad mode {mode!r}; use som|vision|ax"})
    # pid/window_id forwarded only when given so older backends keep their defaults.
    target = {k: args.get(k) for k in ("pid", "window_id")}
    return _capture_response(backend.capture(mode=mode, app=args.get("app"),
                                             **(target if any(v is not None for v in target.values()) else {})))

def _do_focus_app(backend, action, args, **_):
    if not args.get("app"):
        return json.dumps({"error": "focus_app requires `app`"})
    return backend.focus_app(args["app"], raise_window=bool(args.get("raise_window")))

def _listing(key: str, method: str):
    return lambda backend, action, args, **_: json.dumps({key: (items := getattr(backend, method)()), "count": len(items)})

def _summarize_click(action: str, args: Dict[str, Any], fg: str) -> str:
    where = (f" element #{args['element']}" if args.get("element") is not None
             else f" at {tuple(args['coordinate'])}" if args.get("coordinate") else "")
    return f"{action}{where}{fg}"

# One `action`. ``input``: native input to the backend's sticky target (gets delivery kwargs + the `app=` mismatch
# guard). ``destructive``: mutates user-visible state -> approval prompt (the rest only read).
# ``summarize(action, args, fg_suffix)`` renders the one-line approval prompt.
_ActionSpec = namedtuple("_ActionSpec", "handler input destructive summarize",
                         defaults=(False, False, lambda a, args, fg: a + fg))

def _input(handler, summarize=None) -> _ActionSpec:
    return _ActionSpec(handler, True, True, **({"summarize": summarize} if summarize else {}))

_ACTIONS: Dict[str, _ActionSpec] = {
    "click": _input(_do_click, _summarize_click),
    "double_click": _input(partial(_do_click, count=2), _summarize_click),
    "right_click": _input(partial(_do_click, button="right"), _summarize_click),
    "middle_click": _input(partial(_do_click, button="middle"), _summarize_click),
    "drag": _input(_do_drag, lambda a, args, fg: (f"drag {args.get('from_element') or args.get('from_coordinate')} → "
                                                  f"{args.get('to_element') or args.get('to_coordinate')}{fg}")),
    "scroll": _input(_do_scroll, lambda a, args, fg: f"scroll {args.get('direction', '?')} x{args.get('amount', 3)}{fg}"),
    "type": _input(lambda backend, action, args, **delivery: backend.type_text(args.get("text", ""), **delivery),
                   lambda a, args, fg: (f"type {args.get('text', '')[:60]!r}"
                                        + ("..." if len(args.get("text", "")) > 60 else "") + fg)),
    "key": _input(lambda backend, action, args, **delivery: backend.key(args.get("keys", ""), **delivery),
                  lambda a, args, fg: f"key {args.get('keys', '')!r}{fg}"),
    "set_value": _input(_do_set_value),
    "focus_app": _ActionSpec(_do_focus_app, destructive=True, summarize=lambda a, args, fg: (
        f"focus {args.get('app', '')!r}" + (" (raise)" if args.get("raise_window") else ""))),
    "capture": _ActionSpec(_do_capture),
    "wait": _ActionSpec(lambda backend, action, args, **_: _text_response(backend.wait(float(args.get("seconds", 1.0))))),
    "list_apps": _ActionSpec(_listing("apps", "list_apps")),
    "list_windows": _ActionSpec(_listing("windows", "list_windows")),
}
# Native input actions deliver to the backend's sticky target; `app=` on these calls is NOT a
# targeting parameter — see the mismatch guard in _dispatch.
_INPUT_ACTIONS = frozenset(a for a, s in _ACTIONS.items() if s.input)

# Unknown actions are never aliased (no repairing bad model output), but the nearest real action is
# named so a bare error isn't the only guidance.
_ACTION_SUGGESTIONS = {
    "hotkey": "key", "press_key": "key", "keypress": "key", "key_combo": "key", "shortcut": "key",
    "type_text": "type", "input_text": "type", "screenshot": "capture", "get_window_state": "capture",
    "left_click": "click", "mouse_click": "click",
}

def _dispatch(backend: ComputerUseBackend, action: str, args: Dict[str, Any]) -> Any:
    spec = _ACTIONS.get(action)
    if spec is None:
        hint = _ACTION_SUGGESTIONS.get(str(action))
        return json.dumps({"error": f"unknown action {action!r}"
                           + (f" — did you mean {hint!r}? See the action enum in the tool schema." if hint else "")})
    # app= guard: input goes to the sticky target from the last capture/focus_app and the backend drops app=
    # silently — refuse a clear mismatch rather than type into the wrong window while reporting ok:true.
    if (spec.input and isinstance(requested_app := args.get("app"), str) and requested_app.strip()
            and (mismatch := _input_target_mismatch(backend, requested_app)) is not None):
        return json.dumps({"ok": False, "action": action, "code": "input_target_mismatch", "error": (
            f"{action} would go to the current target {mismatch!r}, not {requested_app.strip()!r} "
            "— input actions always hit the sticky target from the last capture/focus_app. "
            f"Call capture(app={requested_app.strip()!r}) or focus_app first, then retry.")})
    # delivery_mode / bring_to_front thread through every input action (background → foreground ladder).
    res = spec.handler(backend, action, args, delivery_mode=args.get("delivery_mode"),
                       bring_to_front=bool(args.get("bring_to_front")))
    return res if isinstance(res, (str, dict)) else _maybe_follow_capture(backend, res, bool(args.get("capture_after")))


# ── Response shaping ────────────────────────────────────────────────────────

def _classify_action_result(res: ActionResult) -> Dict[str, Any]:
    """Next ladder step from semantic evidence, in precedence order. Escalation is advisory: it never overrides
    a confirmed effect nor licenses repeating input."""
    if res.effect == "confirmed" or res.verified is True:
        return {"decision": "done"}
    if res.effect == "unverifiable":
        return {"decision": "verify_fresh_state", "hint": (
            "Input was delivered but not confirmed. Re-capture and check the result BEFORE any "
            "retry — do not repeat the input on an escalation recommendation alone.")}
    if res.effect == "suspected_noop" or not res.ok or res.code is not None:
        return {"decision": "escalate", **({"recommended": res.escalation.get("recommended")}
                                           if isinstance(res.escalation, dict) else {}), "hint": (
            "The input likely did not land. Climb one rung following `recommended`: 'px' → "
            "re-issue by coordinate; 'foreground' (or a failed pixel click) → re-issue with "
            "delivery_mode='foreground' (separate approval). Do not predict the rung from the "
            "app being Electron/Chromium — react to this signal.")}
    # Transport success without semantic proof is not proof of effect.
    return {"decision": "verify_fresh_state",
            "hint": "Transport succeeded but the effect is unproven. Re-capture and confirm before continuing."}

_VERDICT_FIELDS = ("verified", "effect", "escalation", "path", "degraded", "delivery_mode", "code")

def _present(**fields: Any) -> Dict[str, Any]:
    return {k: v for k, v in fields.items() if v}  # only the truthy optional fields, in the given order

def _action_payload(res: ActionResult) -> Dict[str, Any]:
    # cua-driver's structured verdict fields only when returned (None = old driver). ok is transport success;
    # effect/escalation are the semantic verdict.
    return {"ok": res.ok, "action": res.action, **_present(message=res.message),
            **{k: v for k in _VERDICT_FIELDS if (v := getattr(res, k)) is not None}, **_present(meta=res.meta),
            "verdict": _classify_action_result(res)}

def _text_response(res: ActionResult) -> str:
    return json.dumps(_action_payload(res))

# AX `elements` cap: dense UIs publish 500+ nodes (one capture would exhaust context); the full tree spills to a file.
_DEFAULT_MAX_ELEMENTS = 100
# Some providers reject images below 8x8 before the model sees the result; such captures fall back to text.
_MIN_PROVIDER_IMAGE_DIMENSION = 8
# Some AX trees (Discord/Slack via UIA, Electron chat clients) expose ENTIRE message bodies as labels; uncapped
# they blew the tool-result budget and leaked private chat text. Labels identify a control, not text extraction.
_MAX_ELEMENT_LABEL_CHARS = 120
# Bounded cache trails: every dense capture can spill, and CLI-only sessions never run the gateway's media cleanup.
_MAX_SPILL_FILES = _MAX_CAPTURE_FILES = 20

def _capture_mime(cap: CaptureResult) -> str:
    # cua-driver's explicit MIME type, else sniff the base64 prefix (JPEG starts with /9j/, PNG with iVBOR).
    return cap.image_mime_type or ("image/jpeg" if (cap.png_b64 or "").startswith("/9j/") else "image/png")

def _capture_image_ext(cap: CaptureResult) -> str:
    return ".jpg" if _capture_mime(cap).lower() == "image/jpeg" else ".png"  # matches on-disk bytes for MIME sniffing

def _bounds_unknown(bounds) -> bool:
    # No real geometry: KDE/Qt apps report [0, 0, 0, 0] for elements clickable by index; serializing that as a
    # rect invites coordinate=[0, 0] clicks.
    with contextlib.suppress(TypeError, ValueError):
        return all(int(v) == 0 for v in bounds)
    return False

def _element_to_dict(e: UIElement) -> Dict[str, Any]:
    # A zero rect is "geometry unknown", not a position — null it so no coordinate= is ever derived from it.
    # The element index still works.
    return {"index": e.index, "role": e.role, "label": e.label[:_MAX_ELEMENT_LABEL_CHARS],
            "bounds": None if _bounds_unknown(e.bounds) else list(e.bounds), "app": e.app,
            **({"label_truncated": True} if len(e.label) > _MAX_ELEMENT_LABEL_CHARS else {})}

def _format_elements(elements: List[UIElement], max_lines: int = 40) -> List[str]:
    out: List[str] = []
    for e in elements[:max_lines]:
        where = "@ bounds-unknown (click by element index)" if _bounds_unknown(e.bounds) else f"@ {e.bounds}"
        out.append(f"  #{e.index} {e.role} {e.label.replace(chr(10), ' ')[:60]!r} {where}" + (f" [{e.app}]" if e.app else ""))
    if len(elements) > max_lines:
        out.append(f"  ... +{len(elements) - max_lines} more (call capture with app= to narrow)")
    return out

def _bounds_hints(elements: List[UIElement], image_width: int, image_height: int
                  ) -> Tuple[Optional[float], Optional[str]]:
    """(scale, note) when element bounds live in a different coordinate space than the screenshot, else (None, None).
    On HiDPI displays AX bounds are native while the screenshot is downscaled, so coordinate= clicks read off the
    screenshot miss by the scale factor. 5% slack: window chrome can hang a few px past the captured frame without
    implying a different space. Scale heuristic: larger axis ratio wins."""
    if not elements or image_width <= 0 or image_height <= 0:
        return None, None
    max_x = max_y = 0
    for e in elements:
        with contextlib.suppress(TypeError, ValueError):
            x, y, w, h = e.bounds
            max_x, max_y = max(max_x, int(x) + int(w)), max(max_y, int(y) + int(h))
    if max_x <= image_width * 1.05 and max_y <= image_height * 1.05:
        return None, None
    note = (f"element bounds are in native desktop coordinates (extend to ~{max_x}x{max_y}), "
            f"NOT screenshot pixels ({image_width}x{image_height}). coordinate= clicks expect the native "
            "space — derive click points from element bounds, or scale screenshot positions up accordingly")
    return round(max(max_x / image_width, max_y / image_height), 2), note

def _bounds_scale(elements: List[UIElement], image_width: int, image_height: int) -> Optional[float]:
    return _bounds_hints(elements, image_width, image_height)[0]

def _bounds_space_note(elements: List[UIElement], image_width: int, image_height: int) -> Optional[str]:
    return _bounds_hints(elements, image_width, image_height)[1]

def _view(cap: CaptureResult, visible: List[UIElement], width: int, height: int, **facts: Any) -> SimpleNamespace:
    """One capture's derived facts, computed once for every response branch: ``visible`` is the capped element list,
    ``dims_omitted`` an image below the provider minimum; ``facts`` override the None/False defaults."""
    return SimpleNamespace(**{**dict(cap=cap, visible=visible, total=len(cap.elements), width=width, height=height,
                                     truncated=len(cap.elements) - len(visible), bounds_scale=None, bounds_note=None,
                                     elements_file=None, screenshot_path=None, dims_omitted=None, has_image=False), **facts})

def _capture_view(cap: CaptureResult, max_elements: int) -> SimpleNamespace:
    visible, dims = cap.elements[:max_elements], None
    with contextlib.suppress(Exception):  # (width, height) of the inline PNG/JPEG screenshot, else the backend's
        dims = image_dimensions_from_bytes(base64.b64decode(cap.png_b64, validate=False)) if cap.png_b64 else None
    width, height = dims or (cap.width, cap.height)
    scale, note = _bounds_hints(visible, width, height)
    # Capped labels / capped element array: spill the complete tree for on-demand reads.
    lost_detail = len(cap.elements) > len(visible) or any(len(e.label) > _MAX_ELEMENT_LABEL_CHARS for e in visible)
    too_small = bool(dims) and min(dims) < _MIN_PROVIDER_IMAGE_DIMENSION
    has_image = bool(cap.png_b64) and cap.mode != "ax" and not too_small
    return _view(cap, visible, width, height, bounds_scale=scale, bounds_note=note,
                 elements_file=_spill_elements_to_file(cap) if lost_detail else None,
                 screenshot_path=_persist_capture_image(cap) if has_image else None,
                 dims_omitted=dims if too_small else None, has_image=has_image)

def _capture_summary_lines(v: SimpleNamespace) -> List[str]:
    """Human-readable capture summary; line ORDER is contract. Lists only what `elements` surfaces, otherwise the
    summary names indices the model can't find."""
    cap = v.cap
    notes = (
        v.bounds_note and v.bounds_note + (f"; estimated scale ~{v.bounds_scale}x (screenshot position x "
                                           f"{v.bounds_scale} ≈ native coordinate)" if v.bounds_scale else ""),
        v.screenshot_path and f"shareable screenshot saved to {v.screenshot_path}",
        cap.note,
        v.elements_file and (f"full element tree with untruncated labels saved to {v.elements_file} — "
                             "read_file/search_files it if you need dropped label text or elements beyond the cap"),
    )
    return [
        f"capture mode={cap.mode} {v.width}x{v.height}"
        + (f" app={cap.app}" if cap.app else "") + (f" window={cap.window_title!r}" if cap.window_title else ""),
        f"{v.total} interactable element(s):",
        *(f"  ({note})" for note in notes if note),
        *_format_elements(v.visible),
        *([f"  (screenshot omitted: {v.dims_omitted[0]}x{v.dims_omitted[1]} is below the "
           f"{_MIN_PROVIDER_IMAGE_DIMENSION}x{_MIN_PROVIDER_IMAGE_DIMENSION} provider minimum)"] if v.dims_omitted else []),
    ]

def _multimodal_capture(v: SimpleNamespace, summary: str) -> Dict[str, Any]:
    cap = v.cap  # envelope carrying the screenshot (not the elements array, so no truncation note)
    return {
        "_multimodal": True,
        "content": [{"type": "text", "text": summary},
                    {"type": "image_url", "image_url": {"url": f"data:{_capture_mime(cap)};base64,{cap.png_b64}"}}],
        "text_summary": summary,
        "meta": {"mode": cap.mode, "width": v.width, "height": v.height, "elements": v.total,
                 "png_bytes": cap.png_bytes_len, **_present(screenshot_path=v.screenshot_path,
                                                          elements_file=v.elements_file, bounds_scale=v.bounds_scale)},
    }

def _text_capture_payload(v: SimpleNamespace, summary: str, extra: Optional[Dict[str, Any]] = None) -> str:
    """JSON text payload shared by the AX, vision-unavailable and aux-vision branches. Key order is contract:
    fixed fields, ``extra`` branch markers, then set optionals."""
    cap = v.cap
    return json.dumps({
        "mode": cap.mode, "width": v.width, "height": v.height, "app": cap.app, "window_title": cap.window_title,
        "elements": [_element_to_dict(e) for e in v.visible], "total_elements": v.total, "summary": summary,
        **(extra or {}),
        **_present(truncated_elements=v.truncated, elements_file=v.elements_file,
                   screenshot_path=v.screenshot_path, bounds_scale=v.bounds_scale),
    })

def _capture_response(cap: CaptureResult, max_elements: int = _DEFAULT_MAX_ELEMENTS) -> Any:
    v = _capture_view(cap, max_elements)
    lines = _capture_summary_lines(v)
    summary = "\n".join(lines)  # multimodal/aux paths use this; text paths append notes and rebuild
    extra = None
    if v.has_image:
        # Hand the screenshot to auxiliary.vision (text-only result) when the main model may not consume images
        # natively; returning the multimodal envelope unconditionally tripped HTTP 404/400 at the provider.
        if not _should_route_through_aux_vision():
            return _multimodal_capture(v, summary)
        routed = _route_capture_through_aux_vision(
            cap, summary, visible_elements=v.visible, truncated_elements=v.truncated,
            elements_file=v.elements_file, screenshot_path=v.screenshot_path)
        if routed is not None:
            return routed
        # Aux routing requested but failed (vision node down, empty analysis...): the multimodal envelope could
        # now break with a provider error, so degrade to text.
        lines.append("  (vision unavailable: the auxiliary vision model could not be reached; screenshot "
                     "omitted. Element-index actions still work — drive via the element list above.)")
        extra = {"vision_unavailable": True}
    if v.truncated:  # text paths carry the `elements` array, so the truncation note applies
        lines.append(f"  (response truncated to {len(v.visible)} of {v.total} elements; the full tree is in "
                     "elements_file — read_file/search_files it, or pass app= to narrow scope)")
    return _text_capture_payload(v, "\n".join(lines), extra)

def _maybe_follow_capture(backend: ComputerUseBackend, res: ActionResult, do_capture: bool) -> Any:
    # No follow-up capture after a failed action: a normal-looking screenshot would suggest success.
    if not do_capture or not res.ok:
        return _text_response(res)
    try:
        # Recapture the exact window when known: on Linux several unrelated windows may share an app name, so
        # app-only recapture can switch targets.
        target = getattr(backend, "_last_target", None) or {}
        exact = {k: target.get(k) for k in ("pid", "window_id")}
        cap = backend.capture(mode=_capture_after_mode(), **(exact if None not in exact.values()
                                                            else {"app": getattr(backend, "_last_app", None)}))
    except Exception as e:
        logger.warning("follow-up capture failed: %s", e)
        return _text_response(res)
    resp, payload = _capture_response(cap), _action_payload(res)
    if isinstance(resp, dict) and resp.get("_multimodal"):
        # Keep the evidence/verdict contract visible alongside the image — it governs whether input may repeat.
        prefix = json.dumps(payload) + "\n\n"
        resp["content"][0]["text"] = prefix + resp["content"][0]["text"]
        resp["text_summary"] = prefix + resp["text_summary"]
        resp["action_result"] = payload
        return resp
    data = {"capture": resp}  # text capture: merge the action payload in
    with contextlib.suppress(TypeError, json.JSONDecodeError):
        data = json.loads(resp)
    return json.dumps({**data, **payload})


# ── Cache files (screenshots, element spills, vision temps) ─────────────────

def _cache_file(subdir: str, legacy: str, name: str, pattern: str = "", cap: int = 0):
    """Path for a new file under ``$HERMES_HOME/<subdir>`` (dir created). With ``pattern``/``cap``, first unlinks the
    oldest matching files so at most ``cap - 1`` remain (best-effort)."""
    from hermes_constants import get_hermes_dir  # lazy so tests can patch get_hermes_dir
    cache_dir = get_hermes_dir(subdir, legacy)
    cache_dir.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(Exception):
        files = sorted(cache_dir.glob(pattern), key=lambda p: p.stat().st_mtime) if pattern else []
        for stale in files[: max(0, len(files) - (cap - 1))]:
            stale.unlink(missing_ok=True)
    return cache_dir / name

def _write_cache_file(what: str, subdir: str, legacy: str, name: str, pattern: str, cap: int,
                      write: Callable[[Any], None]) -> Optional[str]:
    """Bounded cache write via ``write(path)``; the path, or None on any failure — an unwritable cache must never
    break control (a capture keeps working without its spill/screenshot copy)."""
    try:
        write(path := _cache_file(subdir, legacy, name, pattern, cap))
        return str(path)
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("computer_use: %s failed: %s", what, exc)
        return None

def _persist_capture_image(cap: CaptureResult) -> Optional[str]:
    """Copy of the capture in Hermes' media cache so attachment surfaces can deliver it (None without an image)."""
    return _write_cache_file(
        "screenshot persistence", "cache/images", "image_cache", f"computer_use_{uuid.uuid4().hex}{_capture_image_ext(cap)}",
        "computer_use_*.*", _MAX_CAPTURE_FILES, lambda p: p.write_bytes(base64.b64decode(cap.png_b64, validate=False)),
    ) if cap.png_b64 else None

def _spill_elements_to_file(cap: CaptureResult) -> Optional[str]:
    """FULL element tree (untruncated labels) in a cache file — the read_file/search_files escape hatch for capped text."""
    def write(path) -> None:
        payload = {"app": cap.app, "window_title": cap.window_title, "total_elements": len(cap.elements),
                   "elements": [{"index": e.index, "role": e.role, "label": e.label,
                                 "bounds": list(e.bounds), "app": e.app} for e in cap.elements]}
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    return _write_cache_file("element spill", "cache/computer_use", "computer_use_cache",
                             f"elements_{uuid.uuid4().hex}.json", "elements_*.json", _MAX_SPILL_FILES, write)


# ── auxiliary.vision routing for captured screenshots ───────────────────────

# Longest image side handed to the aux vision model. Full-resolution desktop captures tokenize heavily and can
# overflow small local-model context windows; ~1456px keeps SOM badges legible while cutting vision latency.
_MAX_VISION_DIM = 1456

def _shrink_capture_for_vision(raw: bytes, ext: str, max_dim: int = _MAX_VISION_DIM) -> tuple[bytes, Optional[str]]:
    """Downscale encoded image bytes so the longest side is <= max_dim -> ``(bytes, scale_note)``. note is None when
    unchanged (fits, or Pillow unavailable/failed), else it tells the vision model the factor so reported
    coordinates map back to the real screen instead of being silently wrong."""
    try:
        from io import BytesIO
        from PIL import Image
        img = Image.open(BytesIO(raw))
        if max(img.size) <= max_dim:
            return raw, None
        (orig_w, orig_h), out = img.size, BytesIO()
        img.thumbnail((max_dim, max_dim))
        new_w, new_h = img.size
        img.save(out, format="JPEG" if ext == ".jpg" else "PNG")
        fx, fy = (orig_w / new_w if new_w else 1.0), (orig_h / new_h if new_h else 1.0)
        factor_clause = (f"multiply any coordinates you report by {fx:.2f} to map back to the real screen."
                         if f"{fx:.2f}" == f"{fy:.2f}" else
                         f"multiply any x coordinates you report by {fx:.2f} and "
                         f"any y coordinates by {fy:.2f} to map back to the real screen.")
        return out.getvalue(), (f"Screenshot downscaled from {orig_w}x{orig_h} to "
                                f"{new_w}x{new_h} for vision; {factor_clause}")
    except Exception as exc:
        logger.debug("computer_use: vision downscale skipped: %s", exc)
        return raw, None

def _should_route_through_aux_vision() -> bool:
    """True when ``_capture_response`` should hand the PNG to aux vision. Any failure returns False (fail open) so a
    broken config never silently drops the screenshot for vision-capable main models."""
    stage = "import"
    try:
        from agent.auxiliary_client import _read_main_model, _read_main_provider
        from hermes_cli.config import load_config
        from tools.computer_use.vision_routing import should_route_capture_to_aux_vision
        stage = "config read"
        provider, model = _read_main_provider() or "", _read_main_model() or ""
        if (cached := _AUX_VISION_ROUTE_CACHE.get(key := (str(provider), str(model)))) is not None:
            return cached
        stage = "decision"
        _AUX_VISION_ROUTE_CACHE[key] = decision = bool(should_route_capture_to_aux_vision(provider, model, load_config()))
        return decision
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("computer_use: aux-vision routing %s failed: %s", stage, exc)
        return False

def _capture_after_mode() -> str:
    """Mode for ``capture_after`` follow-ups. Default ``som`` (screenshot)."""
    with contextlib.suppress(Exception):
        from hermes_cli.config import load_config
        mode = str(((load_config() or {}).get("computer_use") or {}).get("capture_after_mode", "som") or "som")
        return mode if (mode := mode.strip().lower()) in {"som", "vision", "ax"} else "som"
    return "som"

_VISION_PROMPT = ("Describe what is visible in this desktop application screenshot in concise but specific "
                  "terms. Mention the app name and window title if visible, the overall layout, any labelled "
                  "buttons, menus or text fields, and any prominent text content the user would need to know "
                  "about. Do not invent details that are not actually visible.\n\nAX/SOM index for "
                  "cross-reference:\n")

def _route_capture_through_aux_vision(
    cap: CaptureResult, summary: str, *, visible_elements: Optional[List[UIElement]] = None,
    truncated_elements: int = 0, elements_file: Optional[str] = None, screenshot_path: Optional[str] = None,
) -> Optional[str]:
    """Pre-analyse the capture via ``vision_analyze_tool`` (temp file under ``$HERMES_HOME/cache/vision/``) and merge
    the description with the AX/SOM summary into one text payload. JSON, or None on any failure."""
    if not cap.png_b64:
        return None
    problem, temp_image_path = "aux-vision import failed", None
    try:
        from model_tools import _run_async
        from tools.vision_tools import vision_analyze_tool
        problem = "failed to decode capture base64"
        raw = base64.b64decode(cap.png_b64, validate=False)
        problem = None  # from here on failures are loud (warning)
        ext = _capture_image_ext(cap)
        temp_image_path = _cache_file("cache/vision", "temp_vision_images", f"computer_use_{uuid.uuid4().hex}{ext}")
        raw, scale_note = _shrink_capture_for_vision(raw, ext)
        temp_image_path.write_bytes(raw)
        prompt = _VISION_PROMPT + summary + (f"\n\nNote: {scale_note}" if scale_note else "")
        result_json = _run_async(vision_analyze_tool(str(temp_image_path), prompt))
    except Exception as exc:
        if problem:
            logger.debug("computer_use: %s: %s", problem, exc)
        else:
            logger.warning("computer_use: auxiliary.vision pre-analysis failed (%s); "
                           "returning to caller without aux analysis", exc)
        return None
    finally:
        if temp_image_path is not None:
            with contextlib.suppress(Exception):
                os.unlink(str(temp_image_path))
    # The ``analysis`` field of vision_analyze_tool's JSON result; raw text when it isn't JSON; empty -> no merge.
    analysis_text = result_json.strip() if isinstance(result_json, str) else ""
    with contextlib.suppress(TypeError, json.JSONDecodeError):
        parsed = json.loads(analysis_text)
        analysis_text = str(parsed.get("analysis") or "").strip() if isinstance(parsed, dict) else ""
    if not analysis_text:
        return None
    # Same element cap as every other capture branch; dumping cap.elements in full would bypass max_elements
    # exactly for non-vision main models. Dimensions are the backend's on this branch.
    view = _view(cap, cap.elements if visible_elements is None else visible_elements, cap.width, cap.height,
                 truncated=truncated_elements, elements_file=elements_file, screenshot_path=screenshot_path)
    return _text_capture_payload(view, summary, {"vision_analysis": analysis_text,
                                                 "vision_analysis_routed_via": "auxiliary.vision"})


# ── Availability check (used by the tool registry check_fn) ─────────────────

def check_computer_use_requirements() -> bool:
    """macOS/Windows/Linux + cua-driver binary (or env override). `hermes computer-use doctor` names blocked checks."""
    if sys.platform not in ("darwin", "win32", "linux"):
        return False
    from tools.computer_use.cua_backend import cua_driver_binary_available
    return cua_driver_binary_available()

def get_computer_use_schema() -> Dict[str, Any]:
    from tools.computer_use.schema import COMPUTER_USE_SCHEMA
    return COMPUTER_USE_SCHEMA
