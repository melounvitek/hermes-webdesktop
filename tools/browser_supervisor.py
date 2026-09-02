"""Persistent CDP supervisor for browser dialog + frame detection.

One ``CDPSupervisor`` runs per Hermes ``task_id`` with a reachable CDP endpoint.
It holds one persistent WebSocket, subscribes to ``Page`` / ``Runtime`` /
``Target`` events on every attached session (top page + auto-attached OOPIF /
worker targets), and exposes pending dialogs + frame tree through a
thread-safe snapshot that tool handlers read synchronously.

Not in the agent's tool schema. Output reaches the agent via
``browser_snapshot`` (merges supervisor state, see ``tools/browser_tool.py``)
and ``browser_dialog`` (calls ``respond_to_dialog()``).
Design spec: ``website/docs/developer-guide/browser-supervisor.md``.

Dialog capture lives in ``tools.browser_supervisor_dialogs``, frame tracking in
``tools.browser_supervisor_frames``; both are mixed into ``CDPSupervisor`` and
their public names are re-exported here.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING

from tools.browser_supervisor_dialogs import (  # noqa: F401 — re-exported
    DEFAULT_DIALOG_POLICY,
    DEFAULT_DIALOG_TIMEOUT_S,
    DIALOG_BRIDGE_HOST,
    DIALOG_BRIDGE_URL_PATTERN,
    DIALOG_POLICY_AUTO_ACCEPT,
    DIALOG_POLICY_AUTO_DISMISS,
    DIALOG_POLICY_MUST_RESPOND,
    RECENT_DIALOGS_MAX,
    _DIALOG_BRIDGE_SCRIPT,
    _VALID_POLICIES,
    DialogRecord,
    DialogSupervisionMixin,
    PendingDialog,
    _redact_supervisor_text,
    _trim_ring,
)
from tools.browser_supervisor_frames import (  # noqa: F401 — re-exported
    FRAME_TREE_MAX_ENTRIES,
    FRAME_TREE_MAX_OOPIF_DEPTH,
    FrameInfo,
    FrameTrackingMixin,
)

# ``websockets`` costs ~22 ms at import and is only needed once a supervisor
# connects; with postponed annotations the type import stays under TYPE_CHECKING.
if TYPE_CHECKING:
    from websockets.asyncio.client import ClientConnection

logger = logging.getLogger(__name__)

# Ring buffer of recent console-level events.
CONSOLE_HISTORY_MAX = 50


def _redact_cdp_error_text(exc: object) -> str:
    """Redact CDP endpoint credentials from an exception's string form.

    ``websockets`` bakes the raw target URL (``?token=`` / ``user:pass@``) into
    its exception messages. Every egress point that turns such an exception into
    log text or a re-raised message MUST route through here; falls back to a
    fixed sentinel if redaction itself raises, erring toward masking.
    """
    try:
        from agent.redact import redact_cdp_url

        return redact_cdp_url(str(exc))
    except Exception:
        return "<error redacted>"


class _LoopUnavailable(RuntimeError):
    """The supervisor loop refused new work (closed / shutting down)."""


def _schedule(coro, loop, *, timeout: float):
    """Run ``coro`` on the supervisor loop from a sync caller and wait for its result."""
    from agent.async_utils import safe_schedule_threadsafe

    fut = safe_schedule_threadsafe(coro, loop)
    if fut is None:
        raise _LoopUnavailable("Browser supervisor loop unavailable")
    return fut.result(timeout=timeout)


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class ConsoleEvent:
    """Ring buffer entry for console + exception traffic."""

    ts: float
    level: str  # "log" | "error" | "warning" | "exception"
    text: str
    url: Optional[str] = None


@dataclass(frozen=True)
class SupervisorSnapshot:
    """Read-only (frozen) snapshot of supervisor state for tool handlers."""

    pending_dialogs: Tuple[PendingDialog, ...]
    recent_dialogs: Tuple[DialogRecord, ...]
    frame_tree: Dict[str, Any]
    console_errors: Tuple[ConsoleEvent, ...]
    active: bool  # False if supervisor is detached/stopped
    cdp_url: str
    task_id: str

    def to_dict(self) -> Dict[str, Any]:
        """Serialize for inclusion in ``browser_snapshot`` output."""
        out: Dict[str, Any] = {
            "pending_dialogs": [d.to_dict() for d in self.pending_dialogs],
            "frame_tree": self.frame_tree,
        }
        if self.recent_dialogs:
            out["recent_dialogs"] = [d.to_dict() for d in self.recent_dialogs]
        return out


# ── Supervisor core ───────────────────────────────────────────────────────────


class CDPSupervisor(DialogSupervisionMixin, FrameTrackingMixin):
    """One supervisor per (task_id, cdp_url) pair.

    ``start()`` spawns a daemon thread running its own asyncio loop, connects,
    attaches to the first page target, enables domains and auto-attach.
    ``snapshot()`` / ``respond_to_dialog()`` / ``evaluate_runtime()`` are sync,
    thread-safe bridges onto that loop; ``stop()`` tears it down. All CDP I/O
    lives on the supervisor's own loop.
    """

    def __init__(
        self,
        task_id: str,
        cdp_url: str,
        *,
        dialog_policy: str = DEFAULT_DIALOG_POLICY,
        dialog_timeout_s: float = DEFAULT_DIALOG_TIMEOUT_S,
    ) -> None:
        if dialog_policy not in _VALID_POLICIES:
            raise ValueError(
                f"Invalid dialog_policy {dialog_policy!r}; "
                f"must be one of {sorted(_VALID_POLICIES)}"
            )
        self.task_id = task_id
        self.cdp_url = cdp_url
        self.dialog_policy = dialog_policy
        self.dialog_timeout_s = float(dialog_timeout_s)

        # State protected by ``_state_lock`` for cross-thread reads.
        self._state_lock = threading.Lock()
        self._pending_dialogs: Dict[str, PendingDialog] = {}
        self._recent_dialogs: List[DialogRecord] = []
        self._frames: Dict[str, FrameInfo] = {}
        self._console_events: List[ConsoleEvent] = []
        self._active = False

        # Supervisor loop machinery — populated in start().
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._ready_event = threading.Event()
        self._start_error: Optional[BaseException] = None
        self._stop_requested = False

        # CDP call tracking (runs on supervisor loop only).
        self._next_call_id = 1
        self._pending_calls: Dict[int, asyncio.Future] = {}
        self._ws: Optional[ClientConnection] = None
        self._page_session_id: Optional[str] = None

        # Dialog auto-dismiss watchdog handles (per dialog id) + id generator.
        self._dialog_watchdogs: Dict[str, asyncio.TimerHandle] = {}
        self._dialog_seq = 0

    # ── Public sync API ──────────────────────────────────────────────────────

    def start(self, timeout: float = 15.0) -> None:
        """Launch the background loop and block until attachment completes.

        Raises whatever attach failed with (redacted). On return, dialog events
        are already being captured.
        """
        if self._thread and self._thread.is_alive():
            return
        self._ready_event.clear()
        self._start_error = None
        self._stop_requested = False
        self._thread = threading.Thread(
            target=self._thread_main,
            name=f"cdp-supervisor-{self.task_id}",
            daemon=True,
        )
        self._thread.start()
        if not self._ready_event.wait(timeout=timeout):
            self.stop()
            try:
                from agent.redact import redact_cdp_url
                _safe_url = redact_cdp_url(self.cdp_url)
            except Exception:
                _safe_url = "<cdp_url redacted>"
            raise TimeoutError(
                f"CDP supervisor did not attach within {timeout}s "
                f"(cdp_url={_safe_url[:80]}...)"
            )
        if self._start_error is not None:
            err = self._start_error
            self.stop()
            # ``err`` is a raw ``websockets`` exception embedding the full cdp_url
            # (token / userinfo). Re-raise redacted and suppress the cause
            # (``from None``) so nothing leaks via message OR traceback chain.
            raise RuntimeError(
                f"CDP supervisor failed to start: {_redact_cdp_error_text(err)}"
            ) from None

    def stop(self, timeout: float = 5.0) -> None:
        """Cancel the supervisor task and join the thread."""
        self._stop_requested = True
        loop = self._loop
        if loop is not None and loop.is_running():
            # Close the WebSocket from inside the loop so ``async for raw in
            # self._ws`` returns cleanly, ``_run`` hits its ``finally``, pending
            # tasks cancel in order, THEN the thread exits.
            try:
                _schedule(self._close_ws(), loop, timeout=2.0)
            except Exception:
                pass  # loop already shutting down / close timed out
        if self._thread is not None:
            self._thread.join(timeout=timeout)
        with self._state_lock:
            self._active = False

    def snapshot(self) -> SupervisorSnapshot:
        """Return an immutable snapshot of current state."""
        with self._state_lock:
            return SupervisorSnapshot(
                pending_dialogs=tuple(self._pending_dialogs.values()),
                recent_dialogs=tuple(self._recent_dialogs[-RECENT_DIALOGS_MAX:]),
                frame_tree=self._build_frame_tree_locked(),
                console_errors=tuple(self._console_events[-CONSOLE_HISTORY_MAX:]),
                active=self._active,
                cdp_url=self.cdp_url,
                task_id=self.task_id,
            )

    def respond_to_dialog(
        self,
        action: str,
        *,
        prompt_text: Optional[str] = None,
        dialog_id: Optional[str] = None,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Accept/dismiss a pending dialog (sync bridge onto the supervisor loop).

        Returns ``{"ok": True, "dialog": {...}}`` or ``{"ok": False, "error": ...}``
        for recoverable errors (no dialog, ambiguous dialog_id, inactive).
        """
        if action not in {"accept", "dismiss"}:
            return {"ok": False, "error": f"action must be 'accept' or 'dismiss', got {action!r}"}

        with self._state_lock:
            if not self._active:
                return {"ok": False, "error": "supervisor is not active"}
            pending = list(self._pending_dialogs.values())
            if not pending:
                return {"ok": False, "error": "no dialog is currently open"}
            if dialog_id:
                dialog = self._pending_dialogs.get(dialog_id)
                if dialog is None:
                    return {
                        "ok": False,
                        "error": f"dialog_id {dialog_id!r} not found "
                        f"(known: {sorted(self._pending_dialogs)})",
                    }
            elif len(pending) > 1:
                return {
                    "ok": False,
                    "error": (
                        f"{len(pending)} pending dialogs; specify dialog_id. "
                        f"Candidates: {[d.id for d in pending]}"
                    ),
                }
            else:
                dialog = pending[0]

        loop = self._loop
        if loop is None:
            return {"ok": False, "error": "supervisor loop is not running"}

        try:
            _schedule(
                self._handle_dialog_cdp(
                    dialog, accept=(action == "accept"), prompt_text=prompt_text or ""
                ),
                loop,
                timeout=timeout,
            )
        except _LoopUnavailable as e:
            return {"ok": False, "error": str(e)}
        except Exception as e:
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}
        return {"ok": True, "dialog": dialog.to_dict()}

    def evaluate_runtime(
        self,
        expression: str,
        *,
        return_by_value: bool = True,
        await_promise: bool = True,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Evaluate ``expression`` in the page's Runtime context over the live WS.

        Zero subprocess cost vs the agent-browser CLI ``eval``. Returns
        ``{"ok": True, "result": <value>, "result_type": ...}`` or
        ``{"ok": False, "error": ...}``. ``return_by_value=True`` JSON-serializes
        the result (DevTools-console semantics); non-serializable objects come
        back as a description string.
        """
        loop = self._loop
        if loop is None or not loop.is_running():
            return {"ok": False, "error": "supervisor loop is not running"}

        with self._state_lock:
            if not self._active:
                return {"ok": False, "error": "supervisor is not active"}
            session_id = self._page_session_id

        if not session_id:
            return {"ok": False, "error": "supervisor has no attached page session"}

        def _run_eval(by_value: bool) -> Dict[str, Any]:
            coro = self._cdp(
                "Runtime.evaluate",
                {
                    "expression": expression,
                    "returnByValue": by_value,
                    "awaitPromise": await_promise,
                    # userGesture: clipboard / fullscreen APIs need user activation.
                    "userGesture": True,
                },
                session_id=session_id,
                timeout=timeout,
            )
            return _schedule(coro, loop, timeout=timeout + 1)

        try:
            response = _run_eval(return_by_value)
        except Exception as exc:
            # Deep-serializing live DOM nodes / NodeLists / Window can blow past
            # CDP's recursion guard with the protocol-level error ``Object
            # reference chain is too long``. Retry once with returnByValue=False
            # so Chrome returns the description string instead of failing.
            if return_by_value and "reference chain is too long" in str(exc).lower():
                try:
                    response = _run_eval(False)
                except Exception as exc2:
                    return {"ok": False, "error": f"{type(exc2).__name__}: {exc2}"}
            else:
                return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

        # Response: {"result": {"result": {"type", "value", ...}, "exceptionDetails"?}}
        result_payload = response.get("result", {}) if isinstance(response, dict) else {}
        exception_details = result_payload.get("exceptionDetails")
        if exception_details:
            exc_text = exception_details.get("text") or "JavaScript exception"
            description = (exception_details.get("exception") or {}).get("description")
            if description:
                exc_text = f"{exc_text}: {description}"
            return {"ok": False, "error": exc_text}

        result_obj = result_payload.get("result", {})
        result_type = result_obj.get("type", "undefined")

        if "value" in result_obj:
            value = result_obj["value"]
        elif result_type == "undefined":
            value = None
        else:
            # Non-serializable (functions, DOM nodes…) — give the model the
            # browser's description so it gets *something*.
            value = result_obj.get("description") or result_obj.get("unserializableValue")

        return {"ok": True, "result": value, "result_type": result_type}

    # ── Supervisor loop internals ────────────────────────────────────────────

    def _thread_main(self) -> None:
        """Entry point for the supervisor's dedicated thread."""
        loop = asyncio.new_event_loop()
        self._loop = loop
        try:
            asyncio.set_event_loop(loop)
            loop.run_until_complete(self._run())
        except BaseException as e:  # noqa: BLE001 — propagate via _start_error
            if not self._ready_event.is_set():
                self._start_error = e
                self._ready_event.set()
            else:
                logger.warning("CDP supervisor %s crashed: %s", self.task_id, e)
        finally:
            # Flush remaining tasks before closing the loop to avoid
            # "Task was destroyed but it is pending" warnings.
            try:
                pending = [t for t in asyncio.all_tasks(loop) if not t.done()]
                for t in pending:
                    t.cancel()
                if pending:
                    loop.run_until_complete(asyncio.gather(*pending, return_exceptions=True))
            except Exception:
                pass
            try:
                loop.close()
            except Exception:
                pass
            with self._state_lock:
                self._active = False

    async def _close_ws(self) -> None:
        """Detach and close the current WebSocket, swallowing close errors."""
        ws = self._ws
        self._ws = None
        if ws is not None:
            try:
                await ws.close()
            except Exception:
                pass

    async def _run(self) -> None:
        """Top-level reconnecting supervisor coroutine.

        Browserbase tears down the CDP socket every time a short-lived client
        (e.g. agent-browser's per-command CDP client) disconnects, so on drop we
        reset per-session ids, re-attach, and keep going.
        """
        attempt = 0
        last_success_at = 0.0
        backoff = 0.5
        import websockets  # deferred: only supervisors that connect pay the import
        while not self._stop_requested:
            try:
                self._ws = await asyncio.wait_for(
                    websockets.connect(self.cdp_url, max_size=50 * 1024 * 1024),
                    timeout=10.0,
                )
            except Exception as e:
                attempt += 1
                if not self._ready_event.is_set():
                    # Never connected once — fatal for start().
                    self._start_error = e
                    self._ready_event.set()
                    return
                logger.warning(
                    "CDP supervisor %s: connect failed (attempt %s): %s",
                    self.task_id, attempt, _redact_cdp_error_text(e),
                )
                await asyncio.sleep(min(backoff, 10.0))
                backoff = min(backoff * 2, 10.0)
                continue

            reader_task = asyncio.create_task(self._read_loop(), name="cdp-reader")
            try:
                # Reset the per-connection page session id. ``_pending_dialogs``
                # and ``_frames`` are deliberately kept — they reconcile as fresh
                # events arrive; worst case a stale dialog entry is rejected
                # with "no dialog is showing" (logged, not surfaced).
                self._page_session_id = None
                await self._attach_initial_page()
                with self._state_lock:
                    self._active = True
                last_success_at = time.time()
                backoff = 0.5  # reset after a successful attach
                if not self._ready_event.is_set():
                    self._ready_event.set()
                await reader_task
            except BaseException as e:
                if not self._ready_event.is_set():
                    # Never got to ready — propagate to start().
                    self._start_error = e
                    self._ready_event.set()
                    raise
                logger.warning(
                    "CDP supervisor %s: session dropped after %.1fs: %s",
                    self.task_id,
                    time.time() - last_success_at,
                    _redact_cdp_error_text(e),
                )
            finally:
                with self._state_lock:
                    self._active = False
                if not reader_task.done():
                    reader_task.cancel()
                    try:
                        await reader_task
                    except (asyncio.CancelledError, Exception):
                        pass
                for handle in list(self._dialog_watchdogs.values()):
                    handle.cancel()
                self._dialog_watchdogs.clear()
                await self._close_ws()

            if self._stop_requested:
                return

            logger.debug(
                "CDP supervisor %s: reconnecting in %.1fs...", self.task_id, backoff,
            )
            await asyncio.sleep(backoff)
            backoff = min(backoff * 2, 10.0)

    async def _attach_initial_page(self) -> None:
        """Find a page target, attach flattened session, enable domains, install dialog bridge."""
        resp = await self._cdp("Target.getTargets")
        targets = resp.get("result", {}).get("targetInfos", [])
        page_target = next((t for t in targets if t.get("type") == "page"), None)
        if page_target is None:
            created = await self._cdp("Target.createTarget", {"url": "about:blank"})
            target_id = created["result"]["targetId"]
        else:
            target_id = page_target["targetId"]

        attach = await self._cdp(
            "Target.attachToTarget",
            {"targetId": target_id, "flatten": True},
        )
        self._page_session_id = attach["result"]["sessionId"]
        await self._enable_page_domains(self._page_session_id, timeout=10.0)
        await self._install_dialog_bridge(self._page_session_id)

    async def _cdp(
        self,
        method: str,
        params: Optional[Dict[str, Any]] = None,
        *,
        session_id: Optional[str] = None,
        timeout: float = 10.0,
    ) -> Dict[str, Any]:
        """Send a CDP command and await its response."""
        if self._ws is None:
            raise RuntimeError("supervisor WebSocket is not connected")
        call_id = self._next_call_id
        self._next_call_id += 1
        payload: Dict[str, Any] = {"id": call_id, "method": method}
        if params:
            payload["params"] = params
        if session_id:
            payload["sessionId"] = session_id
        fut: asyncio.Future = asyncio.get_running_loop().create_future()
        self._pending_calls[call_id] = fut
        await self._ws.send(json.dumps(payload))
        try:
            return await asyncio.wait_for(fut, timeout=timeout)
        finally:
            self._pending_calls.pop(call_id, None)

    async def _read_loop(self) -> None:
        """Continuously dispatch incoming CDP frames."""
        assert self._ws is not None
        try:
            async for raw in self._ws:
                if self._stop_requested:
                    break
                try:
                    msg = json.loads(raw)
                except Exception:
                    logger.debug("CDP supervisor: non-JSON frame dropped")
                    continue
                if "id" in msg:
                    fut = self._pending_calls.pop(msg["id"], None)
                    if fut is not None and not fut.done():
                        if "error" in msg:
                            fut.set_exception(
                                RuntimeError(f"CDP error on id={msg['id']}: {msg['error']}")
                            )
                        else:
                            fut.set_result(msg)
                elif "method" in msg:
                    await self._on_event(msg["method"], msg.get("params", {}), msg.get("sessionId"))
        except Exception as e:
            logger.debug("CDP read loop exited: %s", e)

    # ── Event dispatch ──────────────────────────────────────────────────────

    async def _on_event(
        self, method: str, params: Dict[str, Any], session_id: Optional[str]
    ) -> None:
        handler = self._EVENT_HANDLERS.get(method)
        if handler is None:
            return
        result = handler(self, params, session_id)
        if result is not None:
            await result

    # ── Console / exception ring buffer ─────────────────────────────────────

    def _on_console(self, params: Dict[str, Any], *, level_from: str) -> None:
        if level_from == "exception":
            details = params.get("exceptionDetails") or {}
            text = str(details.get("text") or "")
            url = details.get("url")
            event = ConsoleEvent(ts=time.time(), level="exception", text=text, url=url)
        else:
            raw_level = str(params.get("type") or "log")
            level = "error" if raw_level in {"error", "assert"} else (
                "warning" if raw_level == "warning" else "log"
            )
            args = params.get("args") or []
            parts: List[str] = []
            for a in args[:4]:
                if isinstance(a, dict):
                    parts.append(str(a.get("value") or a.get("description") or ""))
            event = ConsoleEvent(ts=time.time(), level=level, text=" ".join(parts))
        with self._state_lock:
            self._console_events.append(event)
            self._console_events = _trim_ring(self._console_events, CONSOLE_HISTORY_MAX)

    # CDP event → handler(self, params, session_id). Async handlers return an
    # awaitable that ``_on_event`` awaits; sync handlers return None.
    _EVENT_HANDLERS: Dict[str, Callable[..., Any]] = {
        **DialogSupervisionMixin.EVENT_HANDLERS,
        **FrameTrackingMixin.EVENT_HANDLERS,
        "Runtime.consoleAPICalled": lambda self, p, _sid: self._on_console(p, level_from="api"),
        "Runtime.exceptionThrown": lambda self, p, _sid: self._on_console(p, level_from="exception"),
    }


# ── Registry ─────────────────────────────────────────────────────────────────


class _SupervisorRegistry:
    """Process-global (task_id → supervisor) map with idempotent start/stop.

    One instance, exposed as ``SUPERVISOR_REGISTRY``; mutations go through ``_lock``.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._by_task: Dict[str, CDPSupervisor] = {}

    def get(self, task_id: str) -> Optional[CDPSupervisor]:
        """Return the supervisor for ``task_id`` if running, else ``None``."""
        with self._lock:
            return self._by_task.get(task_id)

    def get_or_start(
        self,
        task_id: str,
        cdp_url: str,
        *,
        dialog_policy: str = DEFAULT_DIALOG_POLICY,
        dialog_timeout_s: float = DEFAULT_DIALOG_TIMEOUT_S,
        start_timeout: float = 15.0,
    ) -> CDPSupervisor:
        """Idempotently ensure a supervisor is running for ``(task_id, cdp_url)``.

        An existing supervisor bound to a different ``cdp_url`` (or unhealthy)
        is stopped and replaced.
        """
        with self._lock:
            existing = self._by_task.get(task_id)
            if existing is not None:
                if existing.cdp_url == cdp_url:
                    thread_ok = existing._thread is not None and existing._thread.is_alive()
                    loop_ok = existing._loop is not None and existing._loop.is_running()
                    if thread_ok and loop_ok:
                        return existing
                # URL changed or unhealthy — tear down, fall through to re-create.
                self._by_task.pop(task_id, None)
        if existing is not None:
            existing.stop()

        supervisor = CDPSupervisor(
            task_id=task_id,
            cdp_url=cdp_url,
            dialog_policy=dialog_policy,
            dialog_timeout_s=dialog_timeout_s,
        )
        supervisor.start(timeout=start_timeout)
        with self._lock:
            # Guard against a concurrent get_or_start from another thread.
            already = self._by_task.get(task_id)
            if already is not None and already.cdp_url == cdp_url:
                supervisor.stop()
                return already
            self._by_task[task_id] = supervisor
        return supervisor

    def stop(self, task_id: str) -> None:
        """Stop and discard the supervisor for ``task_id`` if it exists."""
        with self._lock:
            supervisor = self._by_task.pop(task_id, None)
        if supervisor is not None:
            supervisor.stop()

    def stop_all(self) -> None:
        """Stop every running supervisor. For shutdown / test teardown."""
        with self._lock:
            items = list(self._by_task.items())
            self._by_task.clear()
        for _, supervisor in items:
            supervisor.stop()


SUPERVISOR_REGISTRY = _SupervisorRegistry()


__all__ = [
    "CDPSupervisor",
    "ConsoleEvent",
    "DEFAULT_DIALOG_POLICY",
    "DEFAULT_DIALOG_TIMEOUT_S",
    "DIALOG_POLICY_AUTO_ACCEPT",
    "DIALOG_POLICY_AUTO_DISMISS",
    "DIALOG_POLICY_MUST_RESPOND",
    "DialogRecord",
    "FrameInfo",
    "PendingDialog",
    "SUPERVISOR_REGISTRY",
    "SupervisorSnapshot",
    "_SupervisorRegistry",
]
