"""WebSocket transport for the tui_gateway JSON-RPC server.

Reuses :func:`tui_gateway.server.dispatch` verbatim so every RPC method, slash
command, approval/clarify/sudo flow and agent event flows through the same
handlers whether the client is Ink over stdio or an iOS/web client over WS.

Wire protocol is identical to stdio: newline-delimited JSON-RPC both ways. The
server emits ``gateway.ready`` right after accept, then echoes responses/events.

Mounting::

    @app.websocket("/api/ws")
    async def ws(ws: WebSocket):
        await handle_ws(ws)
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import json
import logging
import socket
import threading
import time
from typing import Any

from tui_gateway import server
from tui_gateway.event_replay import replay_epoch

_log = logging.getLogger(__name__)

# Scale-to-zero: tell the (separate) gateway process a dashboard/desktop/TUI
# client is attached via the mtime of a marker file it reads in its idle
# predicate. Clients ping every 15s; one write per 5s per process is plenty.
# See gateway/scale_to_zero.py.
_DASHBOARD_CLIENT_TOUCH_MIN_INTERVAL_S = 5.0
_dashboard_client_touched_at = 0.0
_dashboard_client_touch_lock = threading.Lock()


def _note_dashboard_client_activity(*, force: bool = False) -> None:
    """Refresh the dashboard-client liveness marker (throttled, best-effort)."""
    global _dashboard_client_touched_at
    now = time.monotonic()
    with _dashboard_client_touch_lock:
        if not force and now - _dashboard_client_touched_at < _DASHBOARD_CLIENT_TOUCH_MIN_INTERVAL_S:
            return
        _dashboard_client_touched_at = now
    try:
        from gateway.scale_to_zero import touch_dashboard_client_heartbeat

        touch_dashboard_client_heartbeat()
    except Exception:  # noqa: BLE001 - liveness garnish must never break the WS
        _log.debug("dashboard client heartbeat touch failed", exc_info=True)


# Max seconds a pool-dispatched handler blocks waiting for the loop to flush a
# WS frame before we give up waiting (the transport is NOT marked dead).
_WS_WRITE_TIMEOUT_S = 10.0
_WS_LOG_PAYLOAD_PREVIEW = 240

# Per-token streaming frames are coalesced: buffered and flushed as a batch on a
# short timer instead of waking the loop once per token (each wakeup competes
# with the agent turn for the GIL). Keep this set to genuinely high-frequency,
# display-only events — anything a client must see promptly (tool/approval/
# status/completion) is non-streaming and flushes the buffer ahead of itself,
# so ordering is preserved.
_STREAMING_EVENT_TYPES = frozenset({"message.delta", "reasoning.delta", "thinking.delta"})
# Max time a streamed token waits in the buffer (~30 fps; imperceptible).
_TOKEN_COALESCE_S = 0.033

# starlette stays optional at import time; fall back to a generic sentinel.
try:
    from starlette.websockets import WebSocketDisconnect as _WebSocketDisconnect
except ImportError:  # pragma: no cover - starlette is a required install path
    _WebSocketDisconnect = Exception  # type: ignore[assignment]


class WSTransport:
    """Per-connection WS transport.

    ``write`` is safe from any thread *other than* the loop thread owning the
    socket (pool workers marshal onto the loop and block on the future). Called
    from the loop thread itself that would deadlock, so we detect it and
    fire-and-forget; loop-thread callers that need completion use ``write_async``.
    """

    def __init__(
        self,
        ws: Any,
        loop: asyncio.AbstractEventLoop,
        *,
        peer: str = "unknown",
        auth_identity: dict | None = None,
    ) -> None:
        self._ws = ws
        self._loop = loop
        self._peer = peer
        #: Server-verified identity from the WS-upgrade credential (dashboard
        #: ticket / internal credential), stamped by ``web_server._ws_auth_reason``.
        #: None for legacy-token/stdio transports. RPC params can never populate
        #: this: it is the only identity authority for browser-controller registration.
        self.auth_identity = auth_identity
        self._closed = False
        # Token-coalescing buffer. The lock guards the buffer + "armed" flag
        # against worker threads calling write(); the timer handle is only ever
        # touched on the loop thread.
        self._token_lock = threading.Lock()
        self._pending_tokens: list[str] = []
        self._token_flush_handle: asyncio.TimerHandle | None = None
        self._token_flush_armed = False
        # Socket writes need an async boundary: several batches can be queued on
        # the owning loop while it recovers from a stall.
        self._send_lock = asyncio.Lock()

    @staticmethod
    def _is_streaming_frame(obj: dict) -> bool:
        params = obj.get("params") if isinstance(obj, dict) else None
        return isinstance(params, dict) and params.get("type") in _STREAMING_EVENT_TYPES

    def write(self, obj: dict) -> bool:
        if self._closed:
            return False
        line = json.dumps(obj, ensure_ascii=False)
        try:
            on_loop = asyncio.get_running_loop() is self._loop
        except RuntimeError:
            on_loop = False

        # Streamed token: buffer it and arm the flush timer; the worker returns
        # immediately. call_soon_threadsafe is safe from a worker or the loop.
        if self._is_streaming_frame(obj):
            with self._token_lock:
                self._pending_tokens.append(line)
                if not self._token_flush_armed:
                    self._token_flush_armed = True
                    self._loop.call_soon_threadsafe(self._arm_token_flush)
            return not self._closed

        # Non-streaming frame: append behind any buffered tokens and flush the
        # whole batch NOW so it can never overtake them. The send is scheduled
        # INSIDE the lock so wire order matches buffer order even if the
        # coalesce timer fires on the loop at the same moment.
        from agent.async_utils import safe_schedule_threadsafe

        with self._token_lock:
            self._pending_tokens.append(line)
            batch, self._pending_tokens = self._pending_tokens, []
            if on_loop:
                self._loop.create_task(self._safe_send_many(batch))
                return True
            fut = safe_schedule_threadsafe(self._safe_send_many(batch), self._loop)
            if fut is None:
                self._closed = True
                return False

        try:
            fut.result(timeout=_WS_WRITE_TIMEOUT_S)
            return not self._closed
        except concurrent.futures.TimeoutError:  # builtin TimeoutError on 3.11+
            # The loop is stalled (GIL-heavy turn, delegation), NOT the socket
            # dead: the send is already scheduled and flushes once the loop
            # breathes. Latching _closed here permanently silenced live windows
            # after one slow write; _safe_send_many latches on a real error.
            _log.warning(
                "ws write slow (loop stalled >%ss) peer=%s — frame left in flight",
                _WS_WRITE_TIMEOUT_S, self._peer,
            )
            return not self._closed
        except Exception as exc:
            self._closed = True
            _log.warning(
                "ws write failed peer=%s error_type=%s error=%s",
                self._peer, type(exc).__name__, exc,
            )
            return False

    def _arm_token_flush(self) -> None:
        """Arm the coalesce timer. Runs on the loop thread."""
        if self._closed:
            return
        self._token_flush_handle = self._loop.call_later(_TOKEN_COALESCE_S, self._flush_tokens)

    def _flush_tokens(self) -> None:
        """Timer callback (loop thread): send buffered tokens as one batch. Scheduled
        under the lock so wire order is fixed relative to a concurrent ``write``."""
        with self._token_lock:
            self._token_flush_handle = None
            self._token_flush_armed = False
            batch, self._pending_tokens = self._pending_tokens, []
            if batch and not self._closed:
                self._loop.create_task(self._safe_send_many(batch))

    async def write_async(self, obj: dict) -> bool:
        """Send from the owning loop; awaits until the frame is on the wire. Buffered
        tokens are flushed ahead of it in the SAME batch so nothing slips between."""
        if self._closed:
            return False
        with self._token_lock:
            batch, self._pending_tokens = self._pending_tokens, []
            batch.append(json.dumps(obj, ensure_ascii=False))
        await self._safe_send_many(batch)
        return not self._closed

    async def _safe_send_many(self, lines: list[str]) -> None:
        """Send one indivisible batch of pre-serialized frames in wire order."""
        async with self._send_lock:
            if self._closed:
                return
            try:
                for line in lines:
                    if self._closed:
                        return
                    await self._ws.send_text(line)
            except Exception as exc:
                # Latch while holding the writer lock so queued batches observe
                # the failure before touching the socket.
                self._closed = True
                _log.warning(
                    "ws send failed peer=%s error_type=%s error=%s",
                    self._peer, type(exc).__name__, exc,
                )

    def close(self) -> None:
        self._closed = True
        # Runs on the loop thread (handle_ws finally), so the TimerHandle is safe.
        handle = self._token_flush_handle
        if handle is not None:
            handle.cancel()
            self._token_flush_handle = None


def _ws_peer_label(ws: Any) -> str:
    """``host:port`` when available, else a stable placeholder."""
    client = getattr(ws, "client", None)
    if client is None:
        return "unknown"
    host = getattr(client, "host", None) or "unknown"
    port = getattr(client, "port", None)
    return f"{host}:{port}" if port is not None else host


def _disable_nagle(ws: Any) -> None:
    """Disable Nagle + enable TCP keepalive on the raw socket (best-effort).

    Without TCP_NODELAY the kernel coalesces small per-token frames, so a burst
    after the model's think-pause lands in one tick and no client-side smoothing
    can recover the cadence. Without keepalive a silently-dropped client (SSH
    tunnel reset, sleep) leaves the leg half-open forever: receive_text() blocks
    and the disconnect teardown (detach + orphan reap + resume replay) never runs.
    """
    try:
        scope = getattr(ws, "scope", None) or {}
        transport = (scope.get("extensions") or {}).get("transport") or getattr(ws, "transport", None)
        sock = transport.get_extra_info("socket") if transport is not None else None
        if sock is not None:
            sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
            if hasattr(socket, "TCP_KEEPIDLE"):  # Linux
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPIDLE, 30)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPINTVL, 10)
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPCNT, 3)
            elif hasattr(socket, "TCP_KEEPALIVE"):  # macOS idle seconds
                sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_KEEPALIVE, 30)
    except Exception as exc:  # pragma: no cover - best-effort tuning
        _log.debug("ws TCP_NODELAY skip: %s", exc)


def _error_frame(code: int, message: str, req_id: Any) -> dict:
    return {"jsonrpc": "2.0", "error": {"code": code, "message": message}, "id": req_id}


async def handle_ws(
    ws: Any,
    *,
    auth_identity: dict | None = None,
    subprotocol: str | None = None,
) -> None:
    """Run one WebSocket session. Wire-compatible with ``tui_gateway.entry``.

    *auth_identity* is the server-minted ``{user_id, provider}`` recorded at
    WS-upgrade auth; stored as ``WSTransport.auth_identity``, the only identity
    authority for browser-controller registration. Callers that omit it
    (harnesses, the embedded TUI child) get a ``None`` transport identity.
    """
    peer = _ws_peer_label(ws)
    transport: WSTransport | None = None
    messages = parse_errors = dispatch_crashes = send_failures = 0
    disconnect_reason = "not_connected"

    async def _reply(frame: dict, reason: str, msg: str, *args: Any) -> bool:
        """write_async; on failure record *reason* and log *msg*. False => break."""
        nonlocal disconnect_reason, send_failures
        if await transport.write_async(frame):
            return True
        disconnect_reason = reason
        send_failures += 1
        _log.warning(msg, *args)
        return False

    try:
        if subprotocol:
            await ws.accept(subprotocol=subprotocol)
        else:
            await ws.accept()
        disconnect_reason = "connected"
        # A client is attached from the moment the upgrade is accepted — mark it
        # before the (possibly slow) ready/skin setup so scale-to-zero sees it.
        _note_dashboard_client_activity(force=True)
        _disable_nagle(ws)
        _log.info("ws accepted peer=%s", peer)

        transport = WSTransport(ws, asyncio.get_running_loop(), peer=peer, auth_identity=auth_identity)

        # resolve_skin() is synchronous I/O + CPU work; run it in the pool so the
        # WS read loop stays free to drain the frontend's initial RPC burst.
        skin_payload = await asyncio.to_thread(server.resolve_skin)
        ready_ok = await transport.write_async(
            {
                "jsonrpc": "2.0",
                "method": "event",
                "params": {
                    "type": "gateway.ready",
                    # change_events: this backend broadcasts pet/cron/sessions
                    # .changed, so clients can demote legacy polls to backstops.
                    "payload": {
                        "skin": skin_payload,
                        "change_events": True,
                        "heartbeat": True,
                        # Lets reconnecting clients detect a backend restart and
                        # reset their per-session seq watermarks (event_replay).
                        "replay_epoch": replay_epoch(),
                    },
                },
            }
        )
        if ready_ok:
            # Live-apply skins Hermes activates mid-conversation, and track this
            # peer for session-less global broadcasts write_json can't route.
            server._ensure_skin_watcher()
            server.register_live_transport(transport)
        # Cross-backend liveness: a heartbeat row lets the startup orphan sweep
        # tell "live but idle backend" from "truly orphaned". Idempotent and
        # once-per-process, like the orphan sweep below (the desktop app and web
        # dashboard reach the agent via this sidecar, not entry.main()).
        try:
            server._start_backend_heartbeat_refresher()
        except Exception:
            _log.warning("backend heartbeat refresher start failed", exc_info=True)
        try:
            server._schedule_startup_orphan_sweep()
        except Exception:
            _log.warning("startup orphan sweep scheduling failed", exc_info=True)
        if not ready_ok:
            disconnect_reason = "ready_send_failed"
            send_failures += 1
            _log.error("ws ready frame send failed peer=%s", peer)
            return

        while True:
            try:
                raw = await ws.receive_text()
                _note_dashboard_client_activity()
            except _WebSocketDisconnect as exc:
                disconnect_reason = (
                    "client_disconnect("
                    f"code={getattr(exc, 'code', None)},"
                    f"reason={getattr(exc, 'reason', None)})"
                )
                break
            except Exception:
                disconnect_reason = "receive_failed"
                _log.exception("ws receive failed peer=%s", peer)
                break

            line = raw.strip()
            if not line:
                continue
            messages += 1

            try:
                req = json.loads(line)
            except json.JSONDecodeError as exc:
                parse_errors += 1
                _log.warning(
                    "ws parse error peer=%s index=%d error=%s payload=%r",
                    peer, messages, exc, line[:_WS_LOG_PAYLOAD_PREVIEW],
                )
                if not await _reply(
                    _error_frame(-32700, "parse error", None),
                    "send_failed_after_parse_error",
                    "ws parse-error reply send failed peer=%s", peer,
                ):
                    break
                continue

            req_id = req.get("id") if isinstance(req, dict) else None
            req_method = req.get("method") if isinstance(req, dict) else None

            if req_method == "gateway.ping":
                if not await _reply(
                    {"jsonrpc": "2.0", "result": {"ok": True}, "id": req_id},
                    "send_failed_after_heartbeat",
                    "ws heartbeat reply send failed peer=%s id=%s", peer, req_id,
                ):
                    break
                continue

            # dispatch() may schedule long handlers on the pool; it returns None
            # then and the worker writes the response itself via transport.write
            # (a separate thread, so that is the safe path). Inline handlers
            # return the response dict, written here from the loop.
            try:
                resp = await asyncio.to_thread(server.dispatch, req, transport)
            except Exception:
                dispatch_crashes += 1
                _log.exception("ws dispatch crash peer=%s id=%s method=%s", peer, req_id, req_method)
                if not await _reply(
                    _error_frame(-32603, "internal error", req_id),
                    "send_failed_after_dispatch_crash",
                    "ws dispatch-crash reply send failed peer=%s id=%s method=%s", peer, req_id, req_method,
                ):
                    break
                continue
            if resp is not None and not await _reply(
                resp,
                "send_failed_after_response",
                "ws response send failed peer=%s id=%s method=%s", peer, req_id, req_method,
            ):
                break
    finally:
        reaped_sessions = detached_sessions = 0
        if transport is not None:
            server.unregister_live_transport(transport)

            # Owner-safely park browser controllers this transport registered (a
            # reconnect with the same identity may deliver a terminal result for
            # in-flight work; no new dispatch is admitted while offline).
            # Offloaded: disconnect takes the controller's send_lock, which a
            # worker-thread dispatch may hold while blocking on THIS loop to
            # transmit (result(timeout=10)); inline would park the loop behind it.
            try:
                from gateway.browser_control_broker import get_browser_control_broker

                await asyncio.to_thread(get_browser_control_broker().disconnect_owner, transport)
            except Exception:
                _log.exception("ws browser-controller disconnect failed peer=%s", peer)

            transport.close()

            try:
                await asyncio.to_thread(server._release_wake_for_transport, transport)
            except Exception:
                _log.exception("ws wake-word teardown failed peer=%s", peer)

            # The single WS-disconnect teardown path: reap sessions this transport
            # owned (close_on_disconnect sidecars) or detach the rest to the drop
            # sentinel so later emits don't hit a closed socket; detached ones go
            # to the grace-windowed orphan reaper (a quick resume cancels it).
            # Offloaded: worker.close() blocks (terminate + waits) plus a sync DB
            # write, which inline would freeze the loop for every other peer.
            try:
                reaped_sessions, detached_sessions = await asyncio.to_thread(
                    server._close_sessions_for_transport, transport, end_reason="ws_disconnect"
                )
            except Exception:
                _log.exception("ws transport teardown failed peer=%s", peer)
        try:
            await ws.close()
        except Exception as exc:
            _log.debug("ws close failed peer=%s error=%s", peer, exc)
        _log.info(
            "ws closed peer=%s reason=%s messages=%d parse_errors=%d "
            "dispatch_crashes=%d send_failures=%d reaped_sessions=%d detached_sessions=%d",
            peer, disconnect_reason, messages, parse_errors,
            dispatch_crashes, send_failures, reaped_sessions, detached_sessions,
        )
