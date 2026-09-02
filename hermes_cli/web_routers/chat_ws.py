"""Chat-tab WebSocket routes: /api/console, /api/pty, the /api/ws gateway sidecar and /api/pub + /api/events broadcast.

Extracted from ``hermes_cli.web_server``; helpers/state that tests monkeypatch on
``web_server`` stay there and are imported lazily at call time (cycle-safe).
"""

import asyncio
import functools
import logging
from fastapi import APIRouter
from fastapi import HTTPException, WebSocket, WebSocketDisconnect
from hermes_cli.pty_session import RegistryFull
from pathlib import Path
from typing import Any, Dict, Optional

_log = logging.getLogger("hermes_cli.web_server")
router = APIRouter()


# ---------------------------------------------------------------------------
# /api/console — safe Hermes Console command WebSocket.
#
# Unlike /api/pty, this endpoint never spawns a PTY, shell, or full Hermes CLI
# subprocess. It runs the curated console engine in-process and exchanges
# structured JSON frames with the dashboard xterm overlay.
# ---------------------------------------------------------------------------

_CONSOLE_PROMPT = "hermes> "


_CONSOLE_COMMAND_TIMEOUT_SECONDS = 60.0


_CONSOLE_OUTPUT_LIMIT = 50000


def _console_profile_from_ws(ws: WebSocket) -> Optional[str]:
    profile = (ws.query_params.get("profile") or "").strip()
    return profile or None


def _execute_console_line(
    engine: Any,
    line: str,
    *,
    confirmed: bool,
    profile: Optional[str],
) -> Any:
    # _profile_scope swaps process-global skill module paths; keep it inside
    # the worker thread and never hold it across awaits.
    from hermes_cli.web_server import _profile_scope
    with _profile_scope(profile):
        return engine.execute(line, confirmed=confirmed)


async def _console_send(
    ws: WebSocket,
    send_lock: asyncio.Lock,
    payload: Dict[str, Any],
) -> None:
    async with send_lock:
        await ws.send_json(payload)


async def _console_send_result(
    ws: WebSocket,
    send_lock: asyncio.Lock,
    result: Any,
    *,
    command_id: int,
) -> None:
    command = result.command or ""
    status = result.status
    if status == "ok":
        if result.output:
            await _console_send(
                ws,
                send_lock,
                {
                    "type": "output",
                    "id": command_id,
                    "stream": "stdout",
                    "data": result.output,
                    "command": command,
                },
            )
        await _console_send(
            ws,
            send_lock,
            {
                "type": "complete",
                "id": command_id,
                "status": "ok",
                "command": command,
                "prompt": _CONSOLE_PROMPT,
            },
        )
        return

    if status == "error":
        await _console_send(
            ws,
            send_lock,
            {
                "type": "error",
                "id": command_id,
                "message": result.output or "Command failed.",
                "command": command,
            },
        )
        await _console_send(
            ws,
            send_lock,
            {
                "type": "complete",
                "id": command_id,
                "status": "error",
                "command": command,
                "prompt": _CONSOLE_PROMPT,
            },
        )
        return

    if status == "confirm_required":
        await _console_send(
            ws,
            send_lock,
            {
                "type": "confirm_required",
                "id": command_id,
                "command": command,
                "message": result.confirmation_message or f"Run `{command}`?",
                "prompt": _CONSOLE_PROMPT,
            },
        )
        await _console_send(
            ws,
            send_lock,
            {
                "type": "complete",
                "id": command_id,
                "status": "confirm_required",
                "command": command,
                "prompt": _CONSOLE_PROMPT,
            },
        )
        return

    if status == "clear":
        await _console_send(ws, send_lock, {"type": "clear", "id": command_id})
        await _console_send(
            ws,
            send_lock,
            {
                "type": "complete",
                "id": command_id,
                "status": "clear",
                "command": command,
                "prompt": _CONSOLE_PROMPT,
            },
        )
        return

    if status == "exit":
        await _console_send(
            ws,
            send_lock,
            {
                "type": "complete",
                "id": command_id,
                "status": "exit",
                "command": command,
                "prompt": "",
            },
        )
        return

    await _console_send(
        ws,
        send_lock,
        {
            "type": "error",
            "id": command_id,
            "message": f"Unknown console result status: {status}",
            "command": command,
        },
    )


def _console_json_payload(msg: Any) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    from hermes_cli.web_server import json
    raw: str | bytes | None = msg.get("text")
    if raw is None:
        raw = msg.get("bytes")
    if raw is None:
        return None, None
    if isinstance(raw, bytes):
        try:
            raw = raw.decode("utf-8")
        except UnicodeDecodeError:
            return None, "Console frames must be UTF-8 JSON."
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError:
        return None, "Console frames must be JSON objects."
    if not isinstance(payload, dict):
        return None, "Console frames must be JSON objects."
    return payload, None


@router.websocket("/api/console")
async def console_ws(ws: WebSocket) -> None:
    from hermes_cli.web_server import (
        _DASHBOARD_EMBEDDED_CHAT_ENABLED,
        _get_console_executor,
        _resolve_profile_dir,
        _ws_auth_mode,
        _ws_auth_reason,
        _ws_client_reason,
        _ws_close_reason,
        _ws_host_origin_reason,
        asyncio,
    )
    peer = ws.client.host if ws.client else "?"

    if not _DASHBOARD_EMBEDDED_CHAT_ENABLED:
        _log.info("console refused: embedded chat disabled peer=%s", peer)
        await ws.close(code=4404, reason="embedded chat disabled")
        return

    auth_reason, cred = _ws_auth_reason(ws)
    mode = _ws_auth_mode()
    if auth_reason is not None:
        _log.warning(
            "console auth rejected reason=%s mode=%s cred=%s peer=%s",
            auth_reason, mode, cred, peer,
        )
        await ws.close(code=4401, reason=_ws_close_reason(f"auth: {auth_reason}"))
        return

    host_origin_reason = _ws_host_origin_reason(ws)
    if host_origin_reason is not None:
        _log.warning("console refused: %s peer=%s", host_origin_reason, peer)
        await ws.close(code=4403, reason=_ws_close_reason(host_origin_reason))
        return

    client_reason = _ws_client_reason(ws)
    if client_reason is not None:
        _log.warning("console refused: %s", client_reason)
        await ws.close(code=4408, reason=_ws_close_reason(client_reason))
        return

    await ws.accept()

    profile = _console_profile_from_ws(ws)
    send_lock = asyncio.Lock()

    try:
        from hermes_cli.console_engine import HermesConsoleEngine

        engine = HermesConsoleEngine(output_limit=_CONSOLE_OUTPUT_LIMIT)
        if profile and profile.lower() != "current":
            _resolve_profile_dir(profile)
    except HTTPException as exc:
        await _console_send(
            ws,
            send_lock,
            {
                "type": "error",
                "message": str(exc.detail),
                "prompt": "",
            },
        )
        await ws.close(code=4400, reason=_ws_close_reason(str(exc.detail)))
        return
    except Exception as exc:
        _log.exception("console failed to initialize")
        await _console_send(
            ws,
            send_lock,
            {
                "type": "error",
                "message": f"Console unavailable: {exc}",
                "prompt": "",
            },
        )
        await ws.close(code=1011)
        return

    _log.info(
        "console accepted peer=%s mode=%s cred=%s profile=%s",
        peer,
        mode,
        cred,
        profile or "current",
    )
    await _console_send(
        ws,
        send_lock,
        {
            "type": "ready",
            "profile": profile or "current",
            "prompt": _CONSOLE_PROMPT,
        },
    )

    active_task: asyncio.Task | None = None
    pending_confirmation: Optional[str] = None
    command_generation = 0

    async def run_command(line: str, *, confirmed: bool, command_id: int) -> None:
        nonlocal active_task, pending_confirmation, command_generation
        try:
            loop = asyncio.get_running_loop()
            result = await asyncio.wait_for(
                loop.run_in_executor(
                    _get_console_executor(),
                    functools.partial(
                        _execute_console_line,
                        engine,
                        line,
                        confirmed=confirmed,
                        profile=profile,
                    ),
                ),
                timeout=_CONSOLE_COMMAND_TIMEOUT_SECONDS,
            )
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            if command_id == command_generation:
                pending_confirmation = None
                await _console_send(
                    ws,
                    send_lock,
                    {
                        "type": "error",
                        "id": command_id,
                        "message": (
                            "Command timed out. Hermes Console returned to the prompt."
                        ),
                        "command": line,
                    },
                )
                await _console_send(
                    ws,
                    send_lock,
                    {
                        "type": "complete",
                        "id": command_id,
                        "status": "timeout",
                        "command": line,
                        "prompt": _CONSOLE_PROMPT,
                    },
                )
        except Exception as exc:
            if command_id == command_generation:
                pending_confirmation = None
                _log.exception("console command failed")
                await _console_send(
                    ws,
                    send_lock,
                    {
                        "type": "error",
                        "id": command_id,
                        "message": str(exc) or exc.__class__.__name__,
                        "command": line,
                    },
                )
                await _console_send(
                    ws,
                    send_lock,
                    {
                        "type": "complete",
                        "id": command_id,
                        "status": "error",
                        "command": line,
                        "prompt": _CONSOLE_PROMPT,
                    },
                )
        else:
            if command_id != command_generation:
                return
            pending_confirmation = (
                result.command if result.status == "confirm_required" else None
            )
            await _console_send_result(
                ws,
                send_lock,
                result,
                command_id=command_id,
            )
            if result.status == "exit":
                await ws.close(code=1000)
        finally:
            if command_id == command_generation:
                active_task = None

    async def start_command(line: str, *, confirmed: bool = False) -> None:
        nonlocal active_task, command_generation
        command_generation += 1
        command_id = command_generation
        active_task = asyncio.create_task(
            run_command(line, confirmed=confirmed, command_id=command_id)
        )

    try:
        while True:
            try:
                msg = await ws.receive()
            except RuntimeError:
                break
            msg_type = msg.get("type")
            if msg_type == "websocket.disconnect":
                break

            payload, error = _console_json_payload(msg)
            if error:
                await _console_send(
                    ws,
                    send_lock,
                    {
                        "type": "error",
                        "message": error,
                        "prompt": _CONSOLE_PROMPT,
                    },
                )
                continue
            if payload is None:
                continue

            frame_type = str(payload.get("type") or "").strip().lower()
            if frame_type == "ping":
                await _console_send(
                    ws,
                    send_lock,
                    {
                        "type": "pong",
                        "prompt": _CONSOLE_PROMPT,
                    },
                )
                continue

            if frame_type == "cancel":
                if active_task and not active_task.done():
                    command_generation += 1
                    active_task.cancel()
                    active_task = None
                    pending_confirmation = None
                    await _console_send(
                        ws,
                        send_lock,
                        {
                            "type": "complete",
                            "status": "cancelled",
                            "prompt": _CONSOLE_PROMPT,
                        },
                    )
                elif pending_confirmation:
                    pending_confirmation = None
                    await _console_send(
                        ws,
                        send_lock,
                        {
                            "type": "complete",
                            "status": "cancelled",
                            "prompt": _CONSOLE_PROMPT,
                        },
                    )
                else:
                    await _console_send(
                        ws,
                        send_lock,
                        {
                            "type": "complete",
                            "status": "idle",
                            "prompt": _CONSOLE_PROMPT,
                        },
                    )
                continue

            if active_task and not active_task.done():
                await _console_send(
                    ws,
                    send_lock,
                    {
                        "type": "error",
                        "message": "A console command is already running.",
                        "prompt": _CONSOLE_PROMPT,
                    },
                )
                continue

            if frame_type == "confirm":
                command = str(payload.get("command") or pending_confirmation or "").strip()
                if not pending_confirmation:
                    await _console_send(
                        ws,
                        send_lock,
                        {
                            "type": "error",
                            "message": "No command is waiting for confirmation.",
                            "prompt": _CONSOLE_PROMPT,
                        },
                    )
                    continue
                if command != pending_confirmation:
                    await _console_send(
                        ws,
                        send_lock,
                        {
                            "type": "error",
                            "message": "Confirmation does not match the pending command.",
                            "prompt": _CONSOLE_PROMPT,
                        },
                    )
                    continue
                pending_confirmation = None
                await start_command(command, confirmed=True)
                continue

            if frame_type in {"input", "command"}:
                line = str(payload.get("line") or payload.get("command") or "").strip()
                if not line:
                    await _console_send(
                        ws,
                        send_lock,
                        {
                            "type": "complete",
                            "status": "ok",
                            "prompt": _CONSOLE_PROMPT,
                        },
                    )
                    continue
                if pending_confirmation:
                    await _console_send(
                        ws,
                        send_lock,
                        {
                            "type": "error",
                            "message": (
                                "Confirm or cancel the pending command before "
                                "running another one."
                            ),
                            "prompt": _CONSOLE_PROMPT,
                        },
                    )
                    continue
                await start_command(line)
                continue

            await _console_send(
                ws,
                send_lock,
                {
                    "type": "error",
                    "message": f"Unsupported console frame: {frame_type or '?'}",
                    "prompt": _CONSOLE_PROMPT,
                },
            )
    except WebSocketDisconnect:
        pass
    finally:
        if active_task and not active_task.done():
            active_task.cancel()
            try:
                await active_task
            except (asyncio.CancelledError, Exception):
                pass


@router.websocket("/api/pty")
async def pty_ws(ws: WebSocket) -> None:
    from hermes_cli.web_server import (
        PTY_REGISTRY,
        PtyBridge,
        PtyUnavailableError,
        _DASHBOARD_EMBEDDED_CHAT_ENABLED,
        _PTY_BRIDGE_AVAILABLE,
        _RESIZE_RE,
        _active_session_file_for_channel,
        _build_sidecar_url,
        _channel_or_close_code,
        _forget_active_session_file,
        _legacy_pump,
        _read_active_session_file,
        _resolve_chat_argv_async,
        _ws_auth_mode,
        _ws_auth_reason,
        _ws_client_reason,
        _ws_close_reason,
        _ws_host_origin_reason,
    )
    peer = ws.client.host if ws.client else "?"

    if not _DASHBOARD_EMBEDDED_CHAT_ENABLED:
        _log.info("pty refused: embedded chat disabled peer=%s", peer)
        await ws.close(code=4404, reason="embedded chat disabled")
        return

    # --- auth + host/origin/peer check (before accept so we can close
    #     cleanly AND tell the client WHY via the close code + reason).
    #     Each gate maps to a distinct close code so the log and the
    #     browser banner agree on the cause:
    #       4401 bad credential   4403 host/origin mismatch
    #       4408 peer not allowed  4404 chat disabled
    auth_reason, cred = _ws_auth_reason(ws)
    mode = _ws_auth_mode()
    if auth_reason is not None:
        _log.warning(
            "pty auth rejected reason=%s mode=%s cred=%s peer=%s",
            auth_reason, mode, cred, peer,
        )
        await ws.close(code=4401, reason=_ws_close_reason(f"auth: {auth_reason}"))
        return

    host_origin_reason = _ws_host_origin_reason(ws)
    if host_origin_reason is not None:
        _log.warning("pty refused: %s peer=%s", host_origin_reason, peer)
        await ws.close(code=4403, reason=_ws_close_reason(host_origin_reason))
        return

    client_reason = _ws_client_reason(ws)
    if client_reason is not None:
        _log.warning("pty refused: %s", client_reason)
        await ws.close(code=4408, reason=_ws_close_reason(client_reason))
        return

    await ws.accept()
    _log.info("pty accepted peer=%s mode=%s cred=%s", peer, mode, cred)

    # On native Windows, the POSIX PTY bridge can't be imported.  Tell the
    # client and close cleanly rather than pretending the feature works.
    if not _PTY_BRIDGE_AVAILABLE:
        await ws.send_text(
            "\r\n\x1b[31mChat unavailable: the embedded terminal requires a "
            "POSIX PTY, which native Windows Python doesn't provide.\x1b[0m\r\n"
            "\x1b[33mInstall Hermes inside WSL2 to use the dashboard's /chat "
            "tab — the rest of the dashboard works here.\x1b[0m\r\n"
        )
        await ws.close(code=1011)
        return

    # --- spawn PTY ------------------------------------------------------
    raw_resume = ws.query_params.get("resume") or None
    resume = raw_resume
    profile = ws.query_params.get("profile") or None
    channel = _channel_or_close_code(ws)
    sidecar_url = _build_sidecar_url(channel) if channel else None
    force_fresh = (ws.query_params.get("fresh") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    active_session_file: Optional[Path] = None

    if channel:
        active_session_file = _active_session_file_for_channel(ws.app, channel)
        if force_fresh:
            resume = None
            _forget_active_session_file(active_session_file)
        elif not resume:
            resume = _read_active_session_file(active_session_file)
            if resume:
                # The client only knows to pin the viewport to the bottom
                # when it requested `?resume=`. Tell it a replay is coming
                # anyway so the implicit active-session fallback gets the
                # same follow-scroll treatment as an explicit resume (#93518).
                await ws.send_json({"type": "resume", "id": resume})

    resolve_kwargs = {
        "resume": resume,
        "sidecar_url": sidecar_url,
        "profile": profile,
    }
    if active_session_file is not None:
        resolve_kwargs["active_session_file"] = str(active_session_file)

    try:
        argv, cwd, env = await _resolve_chat_argv_async(**resolve_kwargs)
    except HTTPException as exc:
        # Unknown/invalid profile from _resolve_profile_dir.
        await ws.send_text(f"\r\n\x1b[31mChat unavailable: {exc.detail}\x1b[0m\r\n")
        await ws.close(code=1011)
        return
    except SystemExit as exc:
        # _make_tui_argv calls sys.exit(1) when node/npm is missing.
        await ws.send_text(f"\r\n\x1b[31mChat unavailable: {exc}\x1b[0m\r\n")
        await ws.close(code=1011)
        return


    attach_token = ws.query_params.get("attach") or None
    registry_resume = raw_resume
    if raw_resume and env:
        registry_resume = env.get("HERMES_TUI_RESUME") or raw_resume
    if attach_token is not None and (registry_resume or profile):
        # Key explicit resumes on their canonical target, never the active-session fallback.
        attach_token = f"{attach_token}\0{profile or ''}\0{registry_resume or ''}"

    def _spawn():
        return PtyBridge.spawn(argv, cwd=cwd, env=env)

    if attach_token is None:
        # Legacy path: 1:1 socket<->PTY, killed on disconnect (unchanged).
        try:
            bridge = _spawn()
        except PtyUnavailableError as exc:
            await ws.send_text(f"\r\n\x1b[31mChat unavailable: {exc}\x1b[0m\r\n")
            await ws.close(code=1011)
            return
        except (FileNotFoundError, OSError) as exc:
            await ws.send_text(f"\r\n\x1b[31mChat failed to start: {exc}\x1b[0m\r\n")
            await ws.close(code=1011)
            return
        await _legacy_pump(ws, bridge)
        return

    # Keep-alive path: the PTY outlives this socket; reattach by token.
    try:
        session, _created = await PTY_REGISTRY.attach_or_spawn(
            attach_token, spawn=_spawn
        )
    except PtyUnavailableError as exc:
        await ws.send_text(f"\r\n\x1b[31mChat unavailable: {exc}\x1b[0m\r\n")
        await ws.close(code=1011)
        return
    except (FileNotFoundError, OSError, RegistryFull) as exc:
        await ws.send_text(f"\r\n\x1b[31mChat unavailable: {exc}\x1b[0m\r\n")
        await ws.close(code=1011)
        return

    # A fresh xterm cannot reliably reconstruct the TUI from an arbitrary
    # bounded tail of alternate-screen, differential ANSI output. Reused PTYs
    # emit a complete frame after replay so reconnects never reopen blank.
    await session.attach(ws, force_redraw=not _created)

    # --- writer loop: WebSocket → PTY master ----------------------------
    # No reader task here: the session's drain task (spawned once per PTY,
    # inside the registry) forwards PTY output to whichever socket is
    # attached and rings-buffers it while detached.  On child EOF the drain
    # closes the attached socket with 4410, which unparks ``ws.receive()``
    # below — same half-open-socket protection the legacy pump has (#54028).
    try:
        while True:
            try:
                msg = await ws.receive()
            except RuntimeError:
                # ws.receive() after the socket is already disconnected
                # (e.g. closed by the drain task on process exit).
                break
            if msg.get("type") == "websocket.disconnect":
                break
            raw = msg.get("bytes")
            if raw is None:
                text = msg.get("text")
                raw = text.encode("utf-8") if isinstance(text, str) else b""
            if not raw:
                continue

            # Resize escape is consumed locally, never written to the PTY.
            match = _RESIZE_RE.match(raw)
            if match and match.end() == len(raw):
                session.bridge.resize(cols=int(match.group(1)), rows=int(match.group(2)))
                continue

            session.bridge.write(raw)
    except WebSocketDisconnect:
        pass
    finally:
        # Detach only — the PTY keeps running for a reattach; the registry
        # reaper closes it after the TTL (or immediately on process exit).
        PTY_REGISTRY.detach(attach_token, ws)


# ---------------------------------------------------------------------------
# /api/ws — JSON-RPC WebSocket sidecar for the dashboard "Chat" tab.
#
# Drives the same `tui_gateway.dispatch` surface Ink uses over stdio, so the
# dashboard can render structured metadata (model badge, tool-call sidebar,
# slash launcher, session info) alongside the xterm.js terminal that PTY
# already paints. Both transports bind to the same session id when one is
# active, so a tool.start emitted by the agent fans out to both sinks.
# ---------------------------------------------------------------------------


@router.websocket("/api/ws")
async def gateway_ws(ws: WebSocket) -> None:
    from hermes_cli.web_server import (
        _DASHBOARD_EMBEDDED_CHAT_ENABLED,
        _ws_auth_ok,
        _ws_request_is_allowed,
    )
    if not _DASHBOARD_EMBEDDED_CHAT_ENABLED:
        await ws.close(code=4403)
        return

    if not _ws_auth_ok(ws):
        await ws.close(code=4401)
        return

    if not _ws_request_is_allowed(ws):
        await ws.close(code=4403)
        return

    from tui_gateway.ws import handle_ws

    # The authenticated identity (ticket / internal credential) was stamped
    # onto the WS object by _ws_auth_reason; carry it into the gateway
    # transport where it becomes the identity authority for privileged RPCs
    # (browser.controller.register). None on the legacy token path.
    await handle_ws(
        ws,
        auth_identity=getattr(ws, "_hermes_auth_identity", None),
        subprotocol=getattr(ws, "_hermes_ws_subprotocol", None),
    )


# ---------------------------------------------------------------------------
# /api/pub + /api/events — chat-tab event broadcast.
#
# The PTY-side ``tui_gateway.entry`` opens /api/pub at startup (driven by
# HERMES_TUI_SIDECAR_URL set in /api/pty's PTY env) and writes every
# dispatcher emit through it.  The dashboard fans those frames out to any
# subscriber that opened /api/events on the same channel id.  This is what
# gives the React sidebar its tool-call feed without breaking the PTY
# child's stdio handshake with Ink.
# ---------------------------------------------------------------------------


@router.websocket("/api/pub")
async def pub_ws(ws: WebSocket) -> None:
    from hermes_cli.web_server import (
        _DASHBOARD_EMBEDDED_CHAT_ENABLED,
        _broadcast_event,
        _channel_or_close_code,
        _ws_auth_ok,
        _ws_request_is_allowed,
    )
    if not _DASHBOARD_EMBEDDED_CHAT_ENABLED:
        await ws.close(code=4403)
        return

    if not _ws_auth_ok(ws):
        await ws.close(code=4401)
        return

    if not _ws_request_is_allowed(ws):
        await ws.close(code=4403)
        return

    channel = _channel_or_close_code(ws)
    if not channel:
        await ws.close(code=4400)
        return

    await ws.accept()

    try:
        while True:
            await _broadcast_event(ws.app, channel, await ws.receive_text())
    except WebSocketDisconnect:
        pass


@router.websocket("/api/events")
async def events_ws(ws: WebSocket) -> None:
    from hermes_cli.web_server import (
        _DASHBOARD_EMBEDDED_CHAT_ENABLED,
        _channel_or_close_code,
        _get_event_state,
        _ws_auth_ok,
        _ws_request_is_allowed,
    )
    if not _DASHBOARD_EMBEDDED_CHAT_ENABLED:
        await ws.close(code=4403)
        return

    if not _ws_auth_ok(ws):
        await ws.close(code=4401)
        return

    if not _ws_request_is_allowed(ws):
        await ws.close(code=4403)
        return

    channel = _channel_or_close_code(ws)
    if not channel:
        await ws.close(code=4400)
        return

    await ws.accept()

    event_channels, event_lock = _get_event_state(ws.app)
    async with event_lock:
        event_channels.setdefault(channel, set()).add(ws)

    try:
        while True:
            # Subscribers don't speak — the receive() just blocks until
            # disconnect so the connection stays open as long as the
            # browser holds it.
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        async with event_lock:
            subs = event_channels.get(channel)

            if subs is not None:
                subs.discard(ws)

                if not subs:
                    event_channels.pop(channel, None)
