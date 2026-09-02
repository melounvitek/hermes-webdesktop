"""Dialog capture + response half of the CDP supervisor.

Two capture paths feed the same ``PendingDialog`` queue:

* native ``Page.javascriptDialogOpening`` events (answered with
  ``Page.handleJavaScriptDialog``), and
* the injected *dialog bridge*: a page script that rewrites alert/confirm/prompt
  into a sync XHR to a magic host we intercept via the CDP ``Fetch`` domain and
  answer with ``Fetch.fulfillRequest``. Works on Browserbase, whose CDP proxy
  auto-dismisses real native dialogs, because the native dialog never fires.

``DialogSupervisionMixin`` is mixed into ``tools.browser_supervisor.CDPSupervisor``
and relies on the state that class initialises (``_state_lock``,
``_pending_dialogs``, ``_recent_dialogs``, ``_dialog_watchdogs``, ``_cdp`` ...).
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Coroutine, Dict, Optional
from urllib.parse import parse_qs, urlparse

# Logger-name parity with the origin module (records must look unchanged).
logger = logging.getLogger("tools.browser_supervisor")


def _redact_supervisor_text(value: str) -> str:
    """Redact page-originated text before exposing supervisor snapshots."""
    from agent.redact import redact_sensitive_text

    return redact_sensitive_text(value, force=True)


def _trim_ring(events: list, keep: int) -> list:
    """Cap a ring buffer at ``keep`` entries once it overflows to 2x (slack reduces churn)."""
    return events[-keep:] if len(events) > keep * 2 else events


# ── Policy / config defaults ─────────────────────────────────────────────────

DIALOG_POLICY_MUST_RESPOND = "must_respond"
DIALOG_POLICY_AUTO_DISMISS = "auto_dismiss"
DIALOG_POLICY_AUTO_ACCEPT = "auto_accept"

_VALID_POLICIES = frozenset(
    {DIALOG_POLICY_MUST_RESPOND, DIALOG_POLICY_AUTO_DISMISS, DIALOG_POLICY_AUTO_ACCEPT}
)

DEFAULT_DIALOG_POLICY = DIALOG_POLICY_MUST_RESPOND
DEFAULT_DIALOG_TIMEOUT_S = 300.0

# Last N closed dialogs kept in ``recent_dialogs`` so agents on backends that
# auto-dismiss server-side (Browserbase) can still observe that a dialog fired.
RECENT_DIALOGS_MAX = 20

# Magic host the injected dialog bridge XHRs to. Intercepted via the CDP Fetch
# domain before any network resolution, so it never has to exist. Keep ASCII +
# URL-safe; Fetch patterns are gated on it.
DIALOG_BRIDGE_HOST = "hermes-dialog-bridge.invalid"
DIALOG_BRIDGE_URL_PATTERN = f"http://{DIALOG_BRIDGE_HOST}/*"

# Injected into every frame via Page.addScriptToEvaluateOnNewDocument.
_DIALOG_BRIDGE_SCRIPT = r"""
(() => {
  if (window.__hermesDialogBridgeInstalled) return;
  window.__hermesDialogBridgeInstalled = true;
  const ENDPOINT = "http://hermes-dialog-bridge.invalid/";
  function ask(kind, message, defaultPrompt) {
    try {
      const xhr = new XMLHttpRequest();
      // Use GET with query params so we don't need to worry about request
      // body encoding in the Fetch interceptor.
      const params = new URLSearchParams({
        kind: String(kind || ""),
        message: String(message == null ? "" : message),
        default_prompt: String(defaultPrompt == null ? "" : defaultPrompt),
      });
      xhr.open("GET", ENDPOINT + "?" + params.toString(), false);  // sync
      xhr.send(null);
      if (xhr.status !== 200) return null;
      const body = xhr.responseText || "";
      let parsed;
      try { parsed = JSON.parse(body); } catch (e) { return null; }
      if (kind === "alert") return undefined;
      if (kind === "confirm") return Boolean(parsed && parsed.accept);
      if (kind === "prompt") {
        if (!parsed || !parsed.accept) return null;
        return parsed.prompt_text == null ? "" : String(parsed.prompt_text);
      }
      return null;
    } catch (e) {
      // If the bridge is unreachable, fall back to the native call so the
      // page still sees *some* behavior (the backend will auto-dismiss).
      return null;
    }
  }
  const realAlert   = window.alert;
  const realConfirm = window.confirm;
  const realPrompt  = window.prompt;
  window.alert   = function(message) { ask("alert",   message, ""); };
  window.confirm = function(message) {
    const r = ask("confirm", message, "");
    return r === null ? false : Boolean(r);
  };
  window.prompt  = function(message, def) {
    const r = ask("prompt", message, def == null ? "" : def);
    return r === null ? null : String(r);
  };
  // onbeforeunload — we can't really synchronously prompt the user from this
  // event without racing navigation.  Leave native behavior for now; the
  // supervisor's native-dialog fallback path still surfaces them in
  // recent_dialogs.
})();
"""


# ── Data model ────────────────────────────────────────────────────────────────


@dataclass
class PendingDialog:
    """A JS dialog currently open on some frame's session."""

    id: str
    type: str  # "alert" | "confirm" | "prompt" | "beforeunload"
    message: str
    default_prompt: str
    opened_at: float
    cdp_session_id: str  # which attached CDP session the dialog fired in
    frame_id: Optional[str] = None
    # Set when captured via the bridge XHR path: respond via Fetch.fulfillRequest,
    # NOT Page.handleJavaScriptDialog — the native dialog never fired.
    bridge_request_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "message": _redact_supervisor_text(self.message),
            "default_prompt": _redact_supervisor_text(self.default_prompt),
            "opened_at": self.opened_at,
            "frame_id": self.frame_id,
        }


@dataclass
class DialogRecord:
    """A dialog that was opened and then handled (kept briefly in ``recent_dialogs``)."""

    id: str
    type: str
    message: str
    opened_at: float
    closed_at: float
    closed_by: str  # "agent" | "auto_policy" | "remote" | "watchdog"
    frame_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "type": self.type,
            "message": _redact_supervisor_text(self.message),
            "opened_at": self.opened_at,
            "closed_at": self.closed_at,
            "closed_by": self.closed_by,
            "frame_id": self.frame_id,
        }


# ── Mixin ─────────────────────────────────────────────────────────────────────


class DialogSupervisionMixin:
    """Dialog event handling for ``CDPSupervisor`` (all methods run on its loop)."""

    async def _install_dialog_bridge(self, session_id: str) -> None:
        """Install the dialog-bridge init script + Fetch interceptor on a session.

        The JS override runs in every frame before page scripts; Fetch.enable
        scoped to the bridge URL catches the XHRs, which surface as pending
        dialogs and are fulfilled when the agent responds. Idempotent at the CDP
        level (Chromium de-dupes identical add-script calls; Fetch.enable
        replaces prior patterns).
        """
        try:
            await self._cdp(
                "Page.addScriptToEvaluateOnNewDocument",
                {"source": _DIALOG_BRIDGE_SCRIPT, "runImmediately": True},
                session_id=session_id,
                timeout=5.0,
            )
        except Exception as e:
            logger.debug(
                "dialog bridge: addScriptToEvaluateOnNewDocument failed on sid=%s: %s",
                (session_id or "")[:16], e,
            )
        try:
            await self._cdp(
                "Fetch.enable",
                {
                    "patterns": [
                        {
                            "urlPattern": DIALOG_BRIDGE_URL_PATTERN,
                            "requestStage": "Request",
                        }
                    ],
                    "handleAuthRequests": False,
                },
                session_id=session_id,
                timeout=5.0,
            )
        except Exception as e:
            logger.debug(
                "dialog bridge: Fetch.enable failed on sid=%s: %s",
                (session_id or "")[:16], e,
            )
        # Best-effort inject into the already-loaded document so existing pages
        # pick up the override on reconnect.
        try:
            await self._cdp(
                "Runtime.evaluate",
                {"expression": _DIALOG_BRIDGE_SCRIPT, "returnByValue": True},
                session_id=session_id,
                timeout=3.0,
            )
        except Exception:
            pass

    # ── Capture ──────────────────────────────────────────────────────────────

    async def _on_dialog_opening(
        self, params: Dict[str, Any], session_id: Optional[str]
    ) -> None:
        dialog = self._new_dialog(
            type=str(params.get("type") or ""),
            message=str(params.get("message") or ""),
            default_prompt=str(params.get("defaultPrompt") or ""),
            session_id=session_id,
            frame_id=params.get("frameId"),
        )
        self._admit_dialog(dialog, self._auto_handle_dialog)

    async def _on_fetch_paused(
        self, params: Dict[str, Any], session_id: Optional[str]
    ) -> None:
        """Bridge XHR captured mid-flight — materialize as a pending dialog.

        The page's JS thread is blocked on the XHR until we Fetch.fulfillRequest
        (from ``respond_to_dialog`` or the watchdog).
        """
        url = str(params.get("request", {}).get("url") or "")
        request_id = params.get("requestId")
        if not request_id:
            return
        if DIALOG_BRIDGE_HOST not in url:
            # Not ours — forward unchanged so the page sees its own request.
            try:
                await self._cdp(
                    "Fetch.continueRequest", {"requestId": request_id},
                    session_id=session_id, timeout=3.0,
                )
            except Exception:
                pass
            return

        q = parse_qs(urlparse(url).query)
        dialog = self._new_dialog(
            type=q.get("kind", [""])[0] or "alert",
            message=q.get("message", [""])[0],
            default_prompt=q.get("default_prompt", [""])[0],
            session_id=session_id,
            frame_id=params.get("frameId"),
            bridge_request_id=str(request_id),
        )
        self._admit_dialog(dialog, self._fulfill_bridge_request)

    def _new_dialog(
        self,
        *,
        type: str,
        message: str,
        default_prompt: str,
        session_id: Optional[str],
        frame_id: Optional[str],
        bridge_request_id: Optional[str] = None,
    ) -> PendingDialog:
        self._dialog_seq += 1
        return PendingDialog(
            id=f"d-{self._dialog_seq}",
            type=type,
            message=message,
            default_prompt=default_prompt,
            opened_at=time.time(),
            cdp_session_id=session_id or self._page_session_id or "",
            frame_id=frame_id,
            bridge_request_id=bridge_request_id,
        )

    def _admit_dialog(
        self,
        dialog: PendingDialog,
        responder: Callable[..., Coroutine[Any, Any, None]],
    ) -> None:
        """Apply the dialog policy: auto-respond via ``responder`` or queue + arm watchdog.

        Auto policies archive FIRST (tagged ``auto_policy``) so the ``closed``
        event that follows our own response isn't re-archived as ``remote``.
        """
        auto = {
            DIALOG_POLICY_AUTO_DISMISS: (False, ""),
            DIALOG_POLICY_AUTO_ACCEPT: (True, dialog.default_prompt),
        }.get(self.dialog_policy)
        if auto is not None:
            accept, prompt_text = auto
            with self._state_lock:
                self._archive_dialog_locked(dialog, "auto_policy")
            asyncio.create_task(responder(dialog, accept=accept, prompt_text=prompt_text))
            return
        # must_respond → add to pending and arm watchdog.
        with self._state_lock:
            self._pending_dialogs[dialog.id] = dialog
        loop = asyncio.get_running_loop()
        handle = loop.call_later(
            self.dialog_timeout_s,
            lambda: asyncio.create_task(self._dialog_timeout_expired(dialog.id)),
        )
        self._dialog_watchdogs[dialog.id] = handle

    # ── Responding ───────────────────────────────────────────────────────────

    async def _respond(
        self, dialog: PendingDialog, *, accept: bool, prompt_text: Optional[str]
    ) -> None:
        """Bridge-fulfill for XHR-captured dialogs, else native CDP. Raises on native CDP failure."""
        if dialog.bridge_request_id:
            await self._fulfill_bridge_request(dialog, accept=accept, prompt_text=prompt_text or "")
        else:
            await self._native_handle_dialog(dialog, accept=accept, prompt_text=prompt_text)

    async def _native_handle_dialog(
        self, dialog: PendingDialog, *, accept: bool, prompt_text: Optional[str]
    ) -> None:
        """Page.handleJavaScriptDialog; ``promptText`` sent only for prompt dialogs
        when ``prompt_text`` is given. Raises on CDP failure."""
        params: Dict[str, Any] = {"accept": accept}
        if prompt_text is not None and dialog.type == "prompt":
            params["promptText"] = prompt_text
        await self._cdp(
            "Page.handleJavaScriptDialog",
            params,
            session_id=dialog.cdp_session_id or None,
            timeout=5.0,
        )

    async def _fulfill_bridge_request(
        self, dialog: PendingDialog, *, accept: bool, prompt_text: str
    ) -> None:
        """Resolve a bridge XHR via Fetch.fulfillRequest so the page unblocks."""
        if not dialog.bridge_request_id:
            return
        payload = {
            "accept": bool(accept),
            "prompt_text": prompt_text if dialog.type == "prompt" else "",
            "dialog_id": dialog.id,
        }
        body = json.dumps(payload).encode()
        try:
            await self._cdp(
                "Fetch.fulfillRequest",
                {
                    "requestId": dialog.bridge_request_id,
                    "responseCode": 200,
                    "responseHeaders": [
                        {"name": "Content-Type", "value": "application/json"},
                        {"name": "Access-Control-Allow-Origin", "value": "*"},
                    ],
                    "body": base64.b64encode(body).decode(),
                },
                session_id=dialog.cdp_session_id or None,
                timeout=5.0,
            )
        except Exception as e:
            logger.debug("bridge fulfill failed for %s: %s", dialog.id, e)

    async def _auto_handle_dialog(
        self, dialog: PendingDialog, *, accept: bool, prompt_text: str
    ) -> None:
        """Auto-policy response for a native dialog (already archived by the caller)."""
        try:
            await self._native_handle_dialog(dialog, accept=accept, prompt_text=prompt_text)
        except Exception as e:
            logger.debug("auto-handle CDP call failed for %s: %s", dialog.id, e)

    async def _handle_dialog_cdp(
        self, dialog: PendingDialog, *, accept: bool, prompt_text: str
    ) -> None:
        """Agent response path.

        The dialog is retired regardless of outcome — a CDP error usually means
        it already closed (browser auto-dismissed after navigation, etc.).
        """
        try:
            await self._respond(dialog, accept=accept, prompt_text=prompt_text)
        finally:
            self._retire_dialog(dialog.id, "agent")

    async def _dialog_timeout_expired(self, dialog_id: str) -> None:
        with self._state_lock:
            dialog = self._pending_dialogs.get(dialog_id)
        if dialog is None:
            return
        logger.warning(
            "CDP supervisor %s: dialog %s (%s) auto-dismissed after %ss timeout",
            self.task_id,
            dialog_id,
            dialog.type,
            self.dialog_timeout_s,
        )
        try:
            # Archive with watchdog tag BEFORE unblocking the page.
            with self._state_lock:
                if self._pending_dialogs.pop(dialog_id, None) is not None:
                    self._archive_dialog_locked(dialog, "watchdog")
            await self._respond(dialog, accept=False, prompt_text=None)
        except Exception as e:
            logger.debug("auto-dismiss failed for %s: %s", dialog_id, e)

    # ── Bookkeeping ──────────────────────────────────────────────────────────

    def _retire_dialog(self, dialog_id: str, closed_by: str) -> None:
        """Remove a pending dialog (archiving it with ``closed_by``) and cancel its watchdog."""
        with self._state_lock:
            dialog = self._pending_dialogs.pop(dialog_id, None)
            if dialog is not None:
                self._archive_dialog_locked(dialog, closed_by)
        handle = self._dialog_watchdogs.pop(dialog_id, None)
        if handle is not None:
            handle.cancel()

    def _archive_dialog_locked(self, dialog: PendingDialog, closed_by: str) -> None:
        """Move a pending dialog to the recent_dialogs ring buffer. Must hold state_lock."""
        record = DialogRecord(
            id=dialog.id,
            type=dialog.type,
            message=dialog.message,
            opened_at=dialog.opened_at,
            closed_at=time.time(),
            closed_by=closed_by,
            frame_id=dialog.frame_id,
        )
        self._recent_dialogs.append(record)
        self._recent_dialogs = _trim_ring(self._recent_dialogs, RECENT_DIALOGS_MAX)

    async def _on_dialog_closed(
        self, params: Dict[str, Any], session_id: Optional[str]
    ) -> None:
        # ``Page.javascriptDialogClosed`` carries only ``result``/``userInput``, not
        # the message. Match by session id and clear the oldest native dialog on
        # it — the JS thread blocks while a dialog is up, so at most one is in
        # flight per session. Bridge dialogs resolve via Fetch.fulfillRequest.
        with self._state_lock:
            candidate_ids = [
                d.id
                for d in self._pending_dialogs.values()
                if d.cdp_session_id == session_id and d.bridge_request_id is None
            ]
        if candidate_ids:
            self._retire_dialog(candidate_ids[0], "remote")

    # CDP event → handler(self, params, session_id); merged into CDPSupervisor._EVENT_HANDLERS.
    EVENT_HANDLERS: Dict[str, Callable[..., Any]] = {
        "Page.javascriptDialogOpening": _on_dialog_opening,
        "Page.javascriptDialogClosed": _on_dialog_closed,
        "Fetch.requestPaused": _on_fetch_paused,
    }
