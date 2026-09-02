"""Persistent dashboard compute-host process.

Phase 0 used this module as a deterministic line-JSON spike.  Phase 1 keeps the
same transport and turns it into the long-lived child that owns live AIAgent
objects when ``dashboard.turn_isolation`` is enabled.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import json
import os
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Collection

from agent.interrupt_compat import request_hard_interrupt
from tui_gateway.host_supervisor import MUTATOR_ROUTE_TABLE, _build_sha


def now_ns() -> int:
    return time.perf_counter_ns()


@dataclass
class SpikeAgent:
    """A deterministic AIAgent-shaped object for pipe/interrupt measurements."""

    session_id: str
    history: list[dict[str, str]] = field(default_factory=list)
    _interrupt: threading.Event = field(default_factory=threading.Event)

    def clear_interrupt(self) -> None:
        self._interrupt.clear()

    def interrupt(self, *, hard_cancel: bool = False) -> None:
        self._interrupt.set()

    def run_conversation(
        self,
        prompt: str,
        *,
        conversation_history: list[dict[str, str]] | None = None,
        stream_callback: Callable[[str], None] | None = None,
        delta_count: int = 24,
        delay_s: float = 0.001,
    ) -> dict[str, Any]:
        base_history = list(conversation_history if conversation_history is not None else self.history)
        chunks: list[str] = []
        interrupted = False
        for index in range(max(0, int(delta_count))):
            if self._interrupt.is_set():
                interrupted = True
                break
            chunk = f"{self.session_id}:{prompt}:{index:04d} "
            chunks.append(chunk)
            if stream_callback is not None:
                stream_callback(chunk)
            if delay_s > 0:
                time.sleep(delay_s)
        if self._interrupt.is_set():
            interrupted = True
        final = "".join(chunks)
        if interrupted:
            final += "[interrupted]"
        messages = [*base_history, {"role": "user", "content": prompt}, {"role": "assistant", "content": final}]
        self.history = messages
        return {"final_response": final, "messages": messages, "interrupted": interrupted}


@dataclass
class HostSession:
    sid: str
    agent: SpikeAgent
    history_version: int = 0
    running: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


class _HostTransport:
    def __init__(self, emit: Callable[[dict[str, Any]], None]) -> None:
        self._emit = emit

    def write(self, obj: dict) -> bool:
        sid = ""
        with contextlib.suppress(Exception):
            if obj.get("method") == "event":
                sid = str(((obj.get("params") or {}).get("session_id")) or "")
        self._emit({"type": "rpc", "sid": sid, "message": obj})
        return True

    def close(self) -> None:
        return None


# Slice of ``ComputeHost.shutdown``'s budget held back for the post-drain
# finalize.  ``HostSupervisor._terminate_pid`` SIGKILLs the host
# ``_SHUTDOWN_TIMEOUT_SECS`` (10s, same as ``shutdown``'s default ``wait``)
# after SIGTERM, so a drain allowed to consume the whole budget would leave the
# flush racing that kill and persist nothing at all.
_FLUSH_RESERVE_SECS = 1.0


class ComputeHost:
    # frame ``type`` -> handler method name (resolved per call so instance
    # monkeypatches of a handler still take effect).
    _FRAME_HANDLERS: dict[str, str] = {
        "session.seed": "_handle_seed",
        "turn.start": "_handle_turn_start",
        "interrupt": "_handle_interrupt",
        "respond": "_handle_respond",
        "reload_mcp": "_handle_reload_mcp",
        "control": "_handle_control",
        "shutdown": "_handle_shutdown",
    }

    def __init__(
        self,
        *,
        stdout: Any = None,
        max_workers: int | None = None,
        heartbeat_secs: int | float | None = None,
    ) -> None:
        self._stdout = stdout or sys.stdout
        self._write_lock = threading.Lock()
        self._sessions: dict[str, HostSession] = {}
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers or _default_workers(),
            thread_name_prefix="compute-host-turn",
        )
        self._closed = threading.Event()
        self._parent_pid = os.getppid()
        self._boot_id = uuid.uuid4().hex
        self._progress_counter = 0
        self._progress_lock = threading.Lock()
        # Future -> the ``sid`` whose turn it is running.  ``shutdown`` needs to
        # know *whose* turn is still live so it can leave those sessions
        # unfinalized; a bare set cannot answer that.
        self._turn_futures: dict[concurrent.futures.Future, str] = {}
        self._turn_futures_lock = threading.Lock()
        self._transport = _HostTransport(self.emit)
        self._heartbeat_secs = (
            float(heartbeat_secs)
            if heartbeat_secs is not None
            else float(os.environ.get("HERMES_COMPUTE_HOST_HEARTBEAT_SECS") or "15")
        )
        if self._heartbeat_secs > 0:
            threading.Thread(target=self._heartbeat_loop, name="compute-host-heartbeat", daemon=True).start()
            threading.Thread(target=self._parent_guard_loop, name="compute-host-ppid-guard", daemon=True).start()

    def emit(self, frame: dict[str, Any]) -> None:
        frame.setdefault("host_ns", now_ns())
        data = json.dumps(frame, separators=(",", ":"), ensure_ascii=False)
        with self._write_lock:
            print(data, file=self._stdout, flush=True)

    def close(self) -> None:
        self._closed.set()
        self._executor.shutdown(wait=False, cancel_futures=True)

    def shutdown(self, *, reason: str = "shutdown", wait: float = 10.0) -> None:
        """Drain in-flight turns, then finalize every session.

        Order matters: ``_finalize_session`` is a one-shot latch (sets
        ``session["_finalized"]``), so finalizing before the drain would spend
        the flush's single chance while turns were still producing output,
        fire ``on_session_end(interrupted=True)`` against a running session and
        release the active-session lease under a live turn.

        ``_FLUSH_RESERVE_SECS`` of the budget (never more than half, so a short
        explicit ``wait`` still gets a real drain) is withheld from the drain so
        the flush still runs when in-flight turns outlast the window; ``wait``
        itself is unchanged, so no added latency or kill-escalation exposure.

        Sessions whose turn is *still running* at the drain deadline are
        excluded from the flush: finalizing one mid-turn (``_executor.shutdown``
        below does not join it) would leave it permanently un-finalizable with
        its lease released — the very race the drain closes. Leaving them
        unfinalized keeps them recoverable.

        NOTE: ``server._shutdown_sessions`` (atexit) runs after ``shutdown()``
        returns on the SIGTERM / stdin_closed paths and may re-finalize skipped
        sessions still in ``server._sessions``; the orphan path (``os._exit``)
        bypasses atexit. Pre-existing gap, not worsened by this ordering; a
        follow-up could gate ``_shutdown_sessions`` on
        ``not _finalized and not running``.
        """
        self._closed.set()
        budget = max(0.0, wait)
        deadline = time.monotonic() + budget - min(_FLUSH_RESERVE_SECS, budget / 2.0)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            with self._turn_futures_lock:
                pending = [f for f in self._turn_futures if not f.done()]
            if not pending:
                break
            # Bounded by ``remaining``: a flat 0.05s sleep would overshoot the
            # deadline and eat the reserve it protects (all of it for small ``wait``).
            time.sleep(min(0.05, remaining))
        with self._turn_futures_lock:
            live_sids = {sid for future, sid in self._turn_futures.items() if sid and not future.done()}
        self.flush_all_sessions(reason=reason, skip_sids=live_sids)
        self._executor.shutdown(wait=False, cancel_futures=True)

    def flush_all_sessions(
        self,
        *,
        reason: str = "shutdown",
        skip_sids: Collection[str] | None = None,
    ) -> None:
        """Finalize every server session except ``skip_sids`` (sessions whose
        turn is still live and must not spend their one-shot finalize)."""
        try:
            from tui_gateway import server
        except Exception:
            return
        skip = set(skip_sids or ())
        for sid, session in list(getattr(server, "_sessions", {}).items()):
            if sid in skip:
                continue
            with contextlib.suppress(Exception):
                server._finalize_session(session, end_reason=f"compute_host_{reason}")

    def handle_frame(self, frame: dict[str, Any]) -> None:
        kind = str(frame.get("type") or "")
        handler = self._FRAME_HANDLERS.get(kind)
        if handler is None:
            self.emit({"type": "error", "request_id": frame.get("request_id"), "message": f"unknown frame type: {kind}"})
            return
        getattr(self, handler)(frame)

    def _handle_shutdown(self, frame: dict[str, Any]) -> None:
        self.emit({"type": "shutdown.ack", "request_id": frame.get("request_id")})
        # Explicit supervisor/test shutdown is a clean child-process close;
        # SIGTERM and orphan paths are the durability flush paths.
        self._closed.set()
        self._executor.shutdown(wait=False, cancel_futures=True)

    # ── Phase-0 deterministic spike frames ─────────────────────────────

    def _handle_seed(self, frame: dict[str, Any]) -> None:
        sid = str(frame.get("sid") or "")
        if not sid:
            self.emit({"type": "error", "request_id": frame.get("request_id"), "message": "sid required"})
            return
        history = frame.get("history")
        if not isinstance(history, list):
            history = []
        self._sessions[sid] = HostSession(sid=sid, agent=SpikeAgent(sid, list(history)))
        self.emit({"type": "session.seeded", "sid": sid, "request_id": frame.get("request_id")})

    def _track_turn_future(self, future: concurrent.futures.Future, sid: str) -> None:
        """Register an in-flight turn against the session running it; the done
        callback must pop under the lock or the mapping grows for the host's life."""
        with self._turn_futures_lock:
            self._turn_futures[future] = sid
        future.add_done_callback(self._untrack_turn_future)

    def _untrack_turn_future(self, future: concurrent.futures.Future) -> None:
        with self._turn_futures_lock:
            self._turn_futures.pop(future, None)

    def _handle_turn_start(self, frame: dict[str, Any]) -> None:
        sid = str(frame.get("sid") or "")
        if sid in self._sessions:
            self._handle_spike_turn_start(frame)
            return
        future = self._executor.submit(self._run_real_turn, dict(frame))
        self._track_turn_future(future, sid)

    def _handle_spike_turn_start(self, frame: dict[str, Any]) -> None:
        sid = str(frame.get("sid") or "")
        session = self._sessions.get(sid)
        if session is None:
            self.emit({"type": "turn.error", "sid": sid, "request_id": frame.get("request_id"), "message": "unknown session"})
            return
        with session.lock:
            if session.running:
                self.emit({"type": "turn.error", "sid": sid, "request_id": frame.get("request_id"), "message": "session busy"})
                return
            session.running = True
        future = self._executor.submit(self._run_spike_turn, session, dict(frame))
        self._track_turn_future(future, sid)

    def _handle_interrupt(self, frame: dict[str, Any]) -> None:
        sid = str(frame.get("sid") or "")
        request_id = frame.get("request_id")
        spike = self._sessions.get(sid)
        if spike is not None:
            request_hard_interrupt(spike.agent)
            self.emit({"type": "interrupt.ack", "sid": sid, "request_id": request_id, "applied": True, "applied_ns": now_ns()})
            return
        try:
            from tui_gateway import server

            session = server._sessions.get(sid)
            if session is None:
                self.emit({"type": "interrupt.ack", "sid": sid, "request_id": request_id, "applied": False})
                return
            # In the child, `_session_uses_compute_host()` is false, so the shared
            # helper interrupts the local agent and releases this process's pending
            # clarify Event; the parent only has a metadata mirror and cannot.
            server._interrupt_session_turn(sid, session)
            self.emit({"type": "interrupt.ack", "sid": sid, "request_id": request_id, "applied": True, "applied_ns": now_ns()})
        except Exception as exc:
            self.emit({"type": "interrupt.ack", "sid": sid, "request_id": request_id, "applied": False, "message": str(exc)})

    def _handle_respond(self, frame: dict[str, Any]) -> None:
        """Resolve an interactive request in the host-owned pending registry."""
        sid = str(frame.get("sid") or "")
        request_id = frame.get("request_id")
        try:
            from tui_gateway import server

            if sid not in server._sessions:
                self.emit({"type": "respond.error", "sid": sid, "request_id": request_id, "message": "session not found"})
                return
            params = frame.get("params")
            if not isinstance(params, dict):
                self.emit({"type": "respond.error", "sid": sid, "request_id": request_id, "message": "response params must be an object"})
                return
            response = server._methods["clarify.respond"](request_id, params)
            self.emit({"type": "respond.ack", "sid": sid, "request_id": request_id, "response": response})
        except Exception as exc:
            self.emit({"type": "respond.error", "sid": sid, "request_id": request_id, "message": str(exc)})

    def _run_spike_turn(self, session: HostSession, frame: dict[str, Any]) -> None:
        request_id = frame.get("request_id") or uuid.uuid4().hex
        prompt = str(frame.get("prompt") or frame.get("text") or "")
        delta_count = _coerce(frame.get("delta_count", 24), int, 24)
        delay_s = _coerce(frame.get("delay_s", 0.001), float, 0.001)
        with session.lock:
            history = list(session.agent.history)
        session.agent.clear_interrupt()
        self.emit({"type": "turn.started", "sid": session.sid, "request_id": request_id, "started_ns": now_ns()})

        def stream(delta: str) -> None:
            self._bump_progress()
            self.emit({"type": "delta", "sid": session.sid, "request_id": request_id, "text": delta, "emitted_ns": now_ns()})

        try:
            result = session.agent.run_conversation(
                prompt,
                conversation_history=history,
                stream_callback=stream,
                delta_count=delta_count,
                delay_s=delay_s,
            )
            with session.lock:
                session.history_version += 1
                session.running = False
                history_version = session.history_version
            self._bump_progress()
            self.emit(
                {"type": "turn.end", "sid": session.sid, "request_id": request_id, "history_version": history_version,
                 "message_count": len(result.get("messages") or []), "interrupted": bool(result.get("interrupted")), "ended_ns": now_ns()}
            )
        except Exception as exc:  # pragma: no cover - defensive host boundary
            with session.lock:
                session.running = False
            self.emit({"type": "turn.error", "sid": session.sid, "request_id": request_id, "message": str(exc)})

    # ── Real dashboard turn path ───────────────────────────────────────

    def _run_real_turn(self, frame: dict[str, Any]) -> None:
        sid = str(frame.get("sid") or "")
        request_id = str(frame.get("request_id") or uuid.uuid4().hex)
        if not sid:
            self.emit({"type": "turn.error", "sid": sid, "request_id": request_id, "message": "sid required"})
            return
        try:
            from tui_gateway import server

            session = self._ensure_server_session(server, frame)
            text = frame.get("text") if "text" in frame else frame.get("prompt", "")
            with session["history_lock"]:
                queued_gen = frame.get("queued_prompt_generation")
                if queued_gen is not None and int(session.get("_queued_prompt_generation", 0)) != int(queued_gen):
                    self.emit({"type": "turn.end", "sid": sid, "request_id": request_id, "interrupted": True, "ended_ns": now_ns()})
                    return
                if session.get("running"):
                    self.emit({"type": "turn.error", "sid": sid, "request_id": request_id, "message": "session busy"})
                    return
                session["running"] = True
                session["_turn_cancel_requested"] = False
                session["last_active"] = time.time()
                server._start_inflight_turn(session, frame.get("text") if "text" in frame else frame.get("prompt"))
            self.emit({"type": "turn.started", "sid": sid, "request_id": request_id, "started_ns": now_ns()})
            with contextlib.suppress(Exception):
                server._ensure_session_db_row(session)
            with contextlib.suppress(Exception):
                import hermes_undo

                hermes_undo.on_user_message_appended(session["session_key"])
            with contextlib.suppress(Exception):
                server._persist_branch_seed(session)
            server._run_prompt_submit(request_id, sid, session, text, display_kind=frame.get("display_kind") or None)
            run_thread = session.get("_run_thread")
            if run_thread is not None and hasattr(run_thread, "join"):
                run_thread.join()
            with session["history_lock"]:
                history_version = int(session.get("history_version", 0))
                message_count = len(session.get("history") or [])
                interrupted = bool(session.get("_turn_cancel_requested"))
                session_key = str(session.get("session_key") or "")
            session_info = server._session_info(session.get("agent"), session)
            self._bump_progress()
            self.emit(
                {"type": "turn.end", "sid": sid, "request_id": request_id, "history_version": history_version, "session_key": session_key,
                 "message_count": message_count, "interrupted": interrupted, "ended_ns": now_ns(), "session_info": session_info, "session_info_emitted": True}
            )
        except Exception as exc:
            with contextlib.suppress(Exception):
                from tui_gateway import server

                session = server._sessions.get(sid)
                if session is not None:
                    with session.get("history_lock", threading.Lock()):
                        session["running"] = False
                        server._clear_inflight_turn(session)
            self.emit({"type": "turn.error", "sid": sid, "request_id": request_id, "reason": "exception", "message": str(exc)})

    def _ensure_server_session(self, server: Any, frame: dict[str, Any]) -> dict:
        sid = str(frame.get("sid") or "")
        key = str(frame.get("session_key") or sid)
        session = server._sessions.get(sid)
        if session is not None:
            session["transport"] = self._transport
            if frame.get("cols") is not None:
                session["cols"] = int(frame.get("cols") or 80)
            if frame.get("cwd"):
                session["cwd"] = str(frame.get("cwd"))
            if frame.get("profile_home"):
                session["profile_home"] = str(frame.get("profile_home"))
            if isinstance(frame.get("attached_images"), list):
                session["attached_images"] = list(frame.get("attached_images") or [])
            return session

        history = frame.get("history") if isinstance(frame.get("history"), list) else []
        profile_home = str(frame.get("profile_home") or "")
        session_db = None
        owns_db = False
        home_token = None
        secret_token = None
        try:
            if profile_home:
                from hermes_constants import set_hermes_home_override
                from agent.secret_scope import build_profile_secret_scope, set_secret_scope
                from hermes_state import get_shared_session_db

                home_token = set_hermes_home_override(profile_home)
                secret_token = set_secret_scope(build_profile_secret_scope(Path(profile_home)))
                # DEDICATED handle — ours only until _make_agent succeeds; after that
                # the agent (registered in server._sessions[sid] via _init_session or
                # the fallback dict below) owns it. A RAISING _make_agent is the one
                # path where nothing takes it, hence ``owns_db``.
                session_db = get_shared_session_db(Path(profile_home) / "state.db")
                owns_db = True
            agent = server._make_agent(
                sid,
                key,
                session_id=key,
                model_override=frame.get("model_override"),
                reasoning_config_override=frame.get("reasoning_config_override"),
                service_tier_override=frame.get("service_tier_override"),
                platform_override=frame.get("source"),
                context_cwd_is_launch_artifact=bool(frame.get("context_cwd_is_launch_artifact", False)),
                session_db=session_db,
            )
            if server._transfer_db_to_agent(agent, session_db):
                owns_db = False
        finally:
            if owns_db and session_db is not None:
                with contextlib.suppress(Exception):
                    from hermes_state import release_or_close

                    release_or_close(session_db)
            if home_token is not None:
                with contextlib.suppress(Exception):
                    from hermes_constants import reset_hermes_home_override
                    from agent.secret_scope import reset_secret_scope

                    reset_hermes_home_override(home_token)
                    reset_secret_scope(secret_token)
        try:
            from tui_gateway.transport import bind_transport, reset_transport

            token = bind_transport(self._transport)
            try:
                server._init_session(
                    sid,
                    key,
                    agent,
                    list(history),
                    cols=int(frame.get("cols") or 80),
                    cwd=str(frame.get("cwd") or "") or None,
                    session_db=session_db,
                    source=frame.get("source"),
                )
            finally:
                reset_transport(token)
        except Exception:
            # If _init_session's side machinery (slash worker, approval notify) is
            # unavailable, keep a minimal host-owned session rather than failing
            # the turn after the expensive agent build succeeded.
            server._sessions[sid] = {
                "agent": agent,
                "session_key": key,
                "history": list(history),
                "history_lock": threading.Lock(),
                "history_version": int(frame.get("history_version") or 0),
                "inflight_turn": None,
                "created_at": time.time(),
                "last_active": time.time(),
                "running": False,
                "attached_images": [],
                "image_counter": 0,
                "cwd": str(frame.get("cwd") or os.getcwd()),
                "cols": int(frame.get("cols") or 80),
                "slash_worker": None,
                "show_reasoning": server._load_show_reasoning(),
                "tool_progress_mode": server._load_tool_progress_mode(),
                "edit_snapshots": {},
                "tool_started_at": {},
                "model_override": frame.get("model_override"),
                "source": server._sanitize_client_source(frame.get("source")),
                "transport": self._transport,
            }
        session = server._sessions[sid]
        session["transport"] = self._transport
        session["profile_home"] = profile_home or session.get("profile_home")
        if isinstance(frame.get("attached_images"), list):
            session["attached_images"] = list(frame.get("attached_images") or [])
        if frame.get("model_override") is not None:
            session["model_override"] = frame.get("model_override")
        return session

    def _handle_reload_mcp(self, frame: dict[str, Any]) -> None:
        sid = str(frame.get("sid") or "")
        request_id = frame.get("request_id")
        try:
            from tui_gateway import server

            resp = server.handle_request({"id": request_id, "method": "reload.mcp", "params": {"session_id": sid, "confirm": True}})
            self.emit({"type": "reload_mcp.ack", "sid": sid, "request_id": request_id, "response": resp})
        except Exception as exc:
            self.emit({"type": "control.error", "sid": sid, "request_id": request_id, "message": str(exc)})

    def _handle_control(self, frame: dict[str, Any]) -> None:
        sid = str(frame.get("sid") or "")
        request_id = frame.get("request_id")
        route_name = str(frame.get("route_name") or "")

        def _error(message: str) -> None:
            self.emit({"type": "control.error", "sid": sid, "request_id": request_id, "message": message})

        def _ack(**extra: Any) -> None:
            self.emit({"type": "control.ack", "sid": sid, "request_id": request_id, "route_name": route_name, **extra})

        def _call_method(name: str, params: dict[str, Any], failure: str) -> dict | None:
            """Run a server method; emit control.error and return None on error."""
            response = server._methods[name](request_id, params)
            if "error" in response:
                _error(str(response["error"].get("message") or failure))
                return None
            return response

        try:
            from tui_gateway import server

            route = MUTATOR_ROUTE_TABLE.get(route_name)
            if route is None:
                _error(f"unclassified route: {route_name}")
                return
            session = server._sessions.get(sid)
            if session is None:
                _error("session not found")
                return
            if route == "idle-gated" and session.get("running"):
                _error("session busy")
                return
            if route_name == "reload.mcp":
                self._handle_reload_mcp({**frame, "type": "reload_mcp"})
                return
            if route_name == "session.save":
                response = _call_method("session.save", {"session_id": sid}, "session save failed")
                if response is not None:
                    _ack(result=response.get("result") or {})
                return
            if route_name == "session.compress":
                focus_topic = str(frame.get("command") or "").removeprefix("/compress").strip()
                params = {"session_id": sid, **({"focus_topic": focus_topic} if focus_topic else {})}
                response = _call_method("session.compress", params, "session compression failed")
                if response is None:
                    return
                with session["history_lock"]:
                    session_key = str(session.get("session_key") or "")
                    history_version = int(session.get("history_version", 0))
                    message_count = len(session.get("history") or [])
                _ack(
                    result=response.get("result") or {},
                    session_key=session_key,
                    history_version=history_version,
                    message_count=message_count,
                    session_info=server._session_info(session.get("agent"), session),
                )
                return
            command = str(frame.get("command") or "")
            output = server._mirror_slash_side_effects(sid, session, command) if command else ""
            with session["history_lock"]:
                messages = server._history_to_messages(list(session.get("history") or []))
                history_version = int(session.get("history_version", 0))
                message_count = len(session.get("history") or [])
                session_key = str(session.get("session_key") or "")
            _ack(
                output=output,
                session_key=session_key,
                history_version=history_version,
                message_count=message_count,
                messages=messages,
                session_info=server._session_info(session.get("agent"), session),
            )
        except Exception as exc:
            if route_name in {"session.compress", "slash.compress"}:
                # The compress mirror defers the context-engine boundary notification
                # until the host commits. If anything raises between queueing and
                # finalize (e.g. building the ack's session_info), discard the pending
                # notification so it can't fire against a rejected boundary on a later
                # compress. finalize is exactly-once, so this is a no-op if the mirror
                # already emitted or discarded it.
                with contextlib.suppress(Exception):
                    from tui_gateway import server as _server
                    from agent.conversation_compression import finalize_context_engine_compression_notification

                    _agent = (_server._sessions.get(sid) or {}).get("agent")
                    if _agent is not None:
                        finalize_context_engine_compression_notification(_agent, committed=False)
            _error(str(exc))

    def _bump_progress(self) -> None:
        with self._progress_lock:
            self._progress_counter += 1

    def _heartbeat_loop(self) -> None:
        while not self._closed.wait(self._heartbeat_secs):
            with self._turn_futures_lock:
                active_turns = sum(1 for f in self._turn_futures if not f.done())
            with self._progress_lock:
                counter = self._progress_counter
            self.emit({"type": "hb", "active_turns": active_turns, "progress_counter": counter, "rss_mb": _rss_mb(os.getpid())})

    def _parent_guard_loop(self) -> None:
        while not self._closed.wait(1.0):
            ppid = os.getppid()
            if ppid in {0, 1} or (self._parent_pid and ppid != self._parent_pid):
                self.emit({"type": "orphan", "old_ppid": self._parent_pid, "ppid": ppid})
                self.shutdown(reason="orphan")
                os._exit(0)


def _coerce(value: Any, cast: Callable[[Any], Any], default: Any) -> Any:
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


def _rss_mb(pid: int) -> float:
    try:
        out = subprocess.check_output(["ps", "-o", "rss=", "-p", str(pid)], text=True, encoding="utf-8", errors="replace", stdin=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=2).strip()
        return int(out.splitlines()[-1].strip()) / 1024.0 if out else 0.0
    except Exception:
        return 0.0


def _default_workers() -> int:
    return _coerce(os.environ.get("HERMES_TUI_RPC_POOL_WORKERS") or "8", lambda v: max(2, int(v)), 8)


def run_host(stdin: Any = None, stdout: Any = None) -> None:
    os.environ["HERMES_COMPUTE_HOST_CHILD"] = "1"
    stdin = stdin or sys.stdin
    host = ComputeHost(stdout=stdout or sys.stdout)
    shutting_down = threading.Event()

    def _signal_handler(_signum, _frame) -> None:
        if shutting_down.is_set():
            return
        shutting_down.set()
        host.shutdown(reason="sigterm")
        raise SystemExit(0)

    with contextlib.suppress(Exception):
        signal.signal(signal.SIGTERM, _signal_handler)
        signal.signal(signal.SIGINT, _signal_handler)

    host.emit(
        {"type": "hello", "host_pid": os.getpid(), "boot_id": host._boot_id, "build_sha": _build_sha(),
         "cwd": os.getcwd(), "hermes_home": os.environ.get("HERMES_HOME", "")}
    )

    def _reader() -> None:
        for raw in stdin:
            if host._closed.is_set():
                break
            try:
                frame = json.loads(raw)
            except json.JSONDecodeError as exc:
                host.emit({"type": "error", "message": f"invalid json: {exc}"})
                continue
            if not isinstance(frame, dict):
                host.emit({"type": "error", "message": "frame must be an object"})
                continue
            host.handle_frame(frame)
            if frame.get("type") == "shutdown":
                os._exit(0)
            if host._closed.is_set():
                break

    reader = threading.Thread(target=_reader, name="compute-host-control-reader", daemon=True)
    reader.start()
    try:
        while not host._closed.wait(0.2):
            if not reader.is_alive():
                break
    finally:
        host.shutdown(reason="stdin_closed", wait=2.0)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Dashboard compute-host process")
    parser.parse_args(argv)
    run_host()
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
