"""
A2A protocol helpers — Agent Card, JSON-RPC framing, task store, conversation persistence.

Wire shape is A2A v1.0 (JSON-RPC 2.0 over HTTP): SCREAMING_SNAKE_CASE states/roles;
Parts and StreamResponse events (``statusUpdate`` / ``artifactUpdate``) are
discriminated by member presence (no ``kind`` / ``final`` fields); SSE stream
closure signals the terminal state; push configs carry ``configId`` + ``createdAt``.
Stdlib only (no a2a-sdk). ``extract_text`` stays tolerant of v0.3 peers.
"""

from __future__ import annotations

import json
import copy
import os
import threading
import time
import uuid
from collections import OrderedDict, defaultdict, deque
from concurrent.futures import Future
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

PROTOCOL_VERSION = "1.0"

# A2A v1.0 task lifecycle states.
STATE_SUBMITTED = "TASK_STATE_SUBMITTED"
STATE_WORKING = "TASK_STATE_WORKING"
STATE_INPUT_REQUIRED = "TASK_STATE_INPUT_REQUIRED"
STATE_AUTH_REQUIRED = "TASK_STATE_AUTH_REQUIRED"
STATE_COMPLETED = "TASK_STATE_COMPLETED"
STATE_FAILED = "TASK_STATE_FAILED"
STATE_CANCELED = "TASK_STATE_CANCELED"
STATE_REJECTED = "TASK_STATE_REJECTED"

TERMINAL_STATES = frozenset({STATE_COMPLETED, STATE_FAILED, STATE_CANCELED, STATE_REJECTED})

# A2A v1.0 message roles.
ROLE_USER = "ROLE_USER"
ROLE_AGENT = "ROLE_AGENT"

# The agent starts its reply with this marker when it needs clarification; the
# adapter maps such replies to TASK_STATE_INPUT_REQUIRED (marker stripped).
INPUT_REQUIRED_MARKER = "[INPUT_REQUIRED]"

# JSON-RPC / A2A error codes. -32001..-32003 are A2A spec-defined; custom errors
# live at -32050..-32059 (implementation-defined space, clear of the A2A block).
ERR_PARSE = -32700
ERR_INVALID_PARAMS = -32602
ERR_METHOD_NOT_FOUND = -32601
ERR_TASK_NOT_FOUND = -32001        # A2A spec: TaskNotFoundError
ERR_TASK_NOT_CANCELABLE = -32002   # A2A spec: TaskNotCancelableError
ERR_UNAUTHORIZED = -32050
ERR_RATE_LIMITED = -32051
ERR_UNTRUSTED_PEER = -32052

# Anti-loop: max inbound turns per context. A2A_MAX_PINGPONG_TURNS env, capped at 20.
_DEFAULT_MAX_PINGPONG = 5
_HARD_MAX_PINGPONG = 20

_RATE_LIMIT_DEFAULT = 60  # requests per minute
_RATE_WINDOW = 60.0  # seconds


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except (ValueError, TypeError):
        return default


def max_pingpong_turns() -> int:
    v = _env_int("A2A_MAX_PINGPONG_TURNS", _DEFAULT_MAX_PINGPONG)
    return max(1, min(v, _HARD_MAX_PINGPONG))


def _rate_limit_per_minute() -> int:
    return max(1, _env_int("A2A_RATE_LIMIT", _RATE_LIMIT_DEFAULT))


def now_iso() -> str:
    """ISO 8601 UTC timestamp with millisecond precision (A2A v1.0)."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _hermes_home() -> Path:
    try:
        from hermes_constants import get_hermes_home
        return Path(get_hermes_home())
    except Exception:
        return Path(os.path.expanduser("~/.hermes"))


# ── Agent Card (v1.0) ─────────────────────────────────────────────────────────

def build_agent_card(*, name: str, url: str, description: str, skills: Optional[list[dict]] = None,
                     streaming: bool = False, push_notifications: bool = False, auth_required: bool = False,
                     tenant: str = "") -> dict:
    """Construct an A2A v1.0 Agent Card.

    ``tenant`` is the optional v1.0 multi-tenancy routing key on AgentInterface;
    when present, clients MUST echo it in request params.
    """
    iface: dict[str, Any] = {"url": url, "protocolBinding": "JSONRPC", "protocolVersion": PROTOCOL_VERSION}
    if tenant:
        iface["tenant"] = tenant
    card: dict[str, Any] = {
        "name": name,
        "description": description,
        "url": url,  # convenience for pre-1.0 clients; canonical is supportedInterfaces
        "version": "1.0.0",
        "provider": {"organization": os.getenv("A2A_PROVIDER_ORG", "Hermes Agent"), "url": os.getenv("A2A_PROVIDER_URL", "") or url},
        "supportedInterfaces": [iface],
        "capabilities": {"streaming": streaming, "pushNotifications": push_notifications,
                         "stateTransitionHistory": False, "extendedAgentCard": False},
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["text/plain"],
        "skills": skills or [],
    }
    if auth_required:
        card["securitySchemes"] = {"bearer": {"type": "http", "scheme": "bearer"}}
        card["security"] = [{"bearer": []}]
    return card


def skills_from_toolsets(toolsets: "list[str] | dict[str, list[str]] | None") -> list[dict]:
    """Derive A2A skill descriptors from toolset names, or a toolset → tool-names
    mapping (tool names become tags, max 10, so peers can match tasks to us)."""
    if not isinstance(toolsets, dict):
        toolsets = {ts: [] for ts in set(toolsets or [])}
    skills = [
        {"id": f"toolset.{name}", "name": name, "description": f"Hermes '{name}' capabilities",
         "tags": [name] + [str(t) for t in (toolsets[name] or [])][:10]}
        for name in sorted(toolsets)
    ]
    return skills or [{"id": "general", "name": "general", "description": "General-purpose conversational agent", "tags": ["general"]}]


# ── JSON-RPC framing + message / part builders ────────────────────────────────

def jsonrpc_result(req_id: Any, result: Any) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "result": result}


def jsonrpc_error(req_id: Any, code: int, message: str) -> dict:
    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}


def send_message_response(payload: dict) -> dict:
    """A2A v1.0 SendMessageResponse oneof wrapper: exactly one of ``task`` /
    ``message``. Legacy methods still return bare payloads."""
    if isinstance(payload, dict) and payload.get("status") and payload.get("id"):
        return {"task": payload}
    return {"message": payload}


def unwrap_send_message_response(result: Any) -> Any:
    """Return the Task/Message inside a v1.0 response, or pass legacy through."""
    if isinstance(result, dict):
        if isinstance(result.get("task"), dict):
            return result["task"]
        if isinstance(result.get("message"), dict):
            return result["message"]
    return result


def stream_task(task: dict) -> dict:
    """v1.0 StreamResponse with a task member."""
    return {"task": task}


def new_task_id() -> str:
    return "task-" + uuid.uuid4().hex[:16]


def new_context_id() -> str:
    return "ctx-" + uuid.uuid4().hex[:16]


def text_part(text: str) -> dict:
    """v1.0 text Part (member-presence discriminated, no ``kind``)."""
    return {"text": text, "mediaType": "text/plain"}


def file_part(url: str = "", raw: str = "", filename: str = "",
              media_type: str = "application/octet-stream") -> dict:
    """v1.0 file Part: ``url`` (reference) or ``raw`` (base64 bytes)."""
    part: dict[str, Any] = {"mediaType": media_type}
    if filename:
        part["filename"] = filename
    if url:
        part["url"] = url
    elif raw:
        part["raw"] = raw
    return part


def data_part(data: Any, media_type: str = "application/json") -> dict:
    """v1.0 data Part (structured data, no ``kind`` field)."""
    return {"data": data, "mediaType": media_type}


def message_with_parts(role: str, parts: list[dict], context_id: str = "") -> dict:
    """A2A v1.0 Message with arbitrary Parts (text, file, data)."""
    msg: dict[str, Any] = {"role": role, "parts": parts, "messageId": uuid.uuid4().hex}
    if context_id:
        msg["contextId"] = context_id
    return msg


def text_message(role: str, text: str, context_id: str = "") -> dict:
    """A2A v1.0 Message with a single text Part."""
    return message_with_parts(role, [text_part(text)], context_id)


def _file_note(fname: str, body: str, mtype: str) -> str:
    label = f"[file: {fname}]" if fname else "[file]"
    return f"{label} {body}" + (f" ({mtype})" if mtype else "")


def _json_or_str(data: Any) -> str:
    try:
        return json.dumps(data, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(data)


def extract_text(message_or_params: dict) -> str:
    """Concatenated text from an A2A Message / Task-result / params payload.

    v1.0, v0.3 (``kind``) and pre-0.3 (``type``) Parts all carry a string ``text``
    member. File Parts render as URL/filename (raw base64 noted, not decoded);
    data Parts render their JSON so the agent sees them."""
    msg = message_or_params.get("message", message_or_params)
    parts = msg.get("parts", []) if isinstance(msg, dict) else []
    chunks = []
    for part in parts:
        if not isinstance(part, dict):
            continue
        if isinstance(txt := part.get("text"), str):
            chunks.append(txt)
        elif isinstance(url := part.get("url"), str) and url:
            chunks.append(_file_note(part.get("filename") or part.get("name") or "", url,
                                     part.get("mediaType") or part.get("mimeType") or ""))
        elif isinstance(v03 := part.get("file"), dict) and isinstance(v03.get("fileWithUri"), str):
            chunks.append(_file_note(v03.get("name") or "", v03["fileWithUri"], v03.get("mimeType") or ""))
        elif isinstance(part.get("raw"), str):
            chunks.append(_file_note(part.get("filename") or "", f"{len(part['raw'])} bytes base64-encoded",
                                     part.get("mediaType") or ""))
        elif (data := part.get("data")) is not None:
            chunks.append(f"[data ({part.get('mediaType') or 'application/json'})]\n{_json_or_str(data)}")
    return "\n".join(chunks).strip()


def extract_context_id(params: dict) -> str:
    """v1.0 puts contextId inside the Message; tolerate legacy top-level."""
    msg = params.get("message") or {}
    ctx = str(msg.get("contextId") or "") if isinstance(msg, dict) else ""
    return ctx or str(params.get("contextId") or "")


def build_task(task_id: str, context_id: str, state: str, agent_text: str = "", *, created_at: str = "") -> dict:
    """A2A v1.0 Task object. ``created_at`` is accepted but NOT serialized: the v1.0
    Task proto has no createdAt and strict ProtoJSON parsers (a2a-sdk) reject unknown fields."""
    task: dict[str, Any] = {"id": task_id, "contextId": context_id, "status": {"state": state, "timestamp": now_iso()}}
    if agent_text:
        task["status"]["message"] = text_message(ROLE_AGENT, agent_text, context_id)
        if state == STATE_COMPLETED:
            task["artifacts"] = [{"artifactId": uuid.uuid4().hex, "parts": [text_part(agent_text)]}]
    return task


# ── Streaming (v1.0 StreamResponse events) ────────────────────────────────────

def status_update(task_id: str, context_id: str, state: str, text: str = "") -> dict:
    """v1.0 StreamResponse with a statusUpdate member."""
    status: dict[str, Any] = {"state": state, "timestamp": now_iso()}
    if text:
        status["message"] = text_message(ROLE_AGENT, text, context_id)
    return {"statusUpdate": {"taskId": task_id, "contextId": context_id, "status": status}}


def artifact_update(task_id: str, context_id: str, text: str) -> dict:
    """v1.0 StreamResponse with an artifactUpdate member."""
    artifact = {"artifactId": uuid.uuid4().hex, "parts": [text_part(text)]}
    return {"artifactUpdate": {"taskId": task_id, "contextId": context_id, "artifact": artifact}}


def sse_data(payload: dict, req_id: Any = None) -> str:
    """One StreamResponse as an SSE data frame. §9.4 requires a full JSON-RPC envelope
    (a2a-sdk breaks on bare StreamResponses); ``req_id=None`` is the legacy no-envelope fallback."""
    envelope = jsonrpc_result(req_id, payload) if req_id is not None else payload
    return f"data: {json.dumps(envelope, ensure_ascii=False)}\n\n"


def sse_done() -> str:
    """Stream-closure marker as an SSE *comment* — ``data: {}`` would make
    JSON-RPC clients try to parse an empty response."""
    return ": done\n\n"


# ── Anti-loop ping-pong protection (per-adapter instance) ─────────────────────

class TurnTracker:
    """Counts inbound turns per context_id; beyond max_pingpong_turns() the
    adapter rejects further messages for that context."""

    _TTL = 3600  # prune contexts idle longer than 1 hour

    def __init__(self) -> None:
        self._counts: dict[str, int] = defaultdict(int)
        self._timestamps: dict[str, float] = {}
        self._lock = threading.Lock()

    def track(self, context_id: str) -> int:
        """Increment and return the turn count; prunes stale contexts."""
        with self._lock:
            now = time.time()
            for cid in [cid for cid, ts in self._timestamps.items() if now - ts > self._TTL]:
                self._counts.pop(cid, None)
                self._timestamps.pop(cid, None)
            self._counts[context_id] += 1
            self._timestamps[context_id] = now
            return self._counts[context_id]

    def reset(self, context_id: str) -> None:
        with self._lock:
            self._counts.pop(context_id, None)
            self._timestamps.pop(context_id, None)


class RateLimiter:
    """Sliding-window request limiter, one bucket per authenticated identity."""

    def __init__(self) -> None:
        self._buckets: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, identity: str) -> bool:
        with self._lock:
            limit = _rate_limit_per_minute()
            now = time.time()
            bucket = self._buckets[identity]
            while bucket and now - bucket[0] > _RATE_WINDOW:
                bucket.popleft()
            if len(bucket) >= limit:
                return False
            bucket.append(now)
            return True


# ── Metrics collection ────────────────────────────────────────────────────────

class Metrics:
    """Simple counters for A2A operations (module singleton ``metrics`` is shared
    by the inbound adapter and outbound tools; not persisted)."""

    _COUNTERS = ("inbound_total", "outbound_total", "streams_started", "push_sent", "push_failed",
                 "tasks_completed", "tasks_failed", "anti_loop_triggers", "rate_limit_triggers")

    def __init__(self) -> None:
        self.inbound_total = self.outbound_total = self.streams_started = self.push_sent = self.push_failed = 0
        self.tasks_completed = self.tasks_failed = self.anti_loop_triggers = self.rate_limit_triggers = 0
        self._start_time = time.time()
        self._latencies: deque[float] = deque(maxlen=100)  # last 100 completed inbound tasks

    def record_latency(self, seconds: float) -> None:
        self._latencies.append(seconds)

    def avg_latency(self) -> float:
        return sum(self._latencies) / len(self._latencies) if self._latencies else 0.0

    def snapshot(self) -> dict[str, Any]:
        return {"uptime_seconds": round(time.time() - self._start_time, 1),
                **{name: getattr(self, name) for name in self._COUNTERS},
                "avg_latency_ms": round(self.avg_latency() * 1000, 1)}


metrics = Metrics()


# ── Task store — pending AND completed tasks (queryable via tasks/get, tasks/list) ───

class TaskStore:
    """In-memory store of A2A tasks, kept after completion for tasks/get. Records carry
    agent slug + tenant; readers pass a scope and get not-found outside it (spec authz rule)."""

    _MAX_TERMINAL = 500

    def __init__(self) -> None:
        self._tasks: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
        self._watchers: dict[str, list[Future]] = {}
        self._lock = threading.Lock()

    @staticmethod
    def _in_scope(rec: dict, agent_slug: str = "", tenant: str = "") -> bool:
        if agent_slug and rec.get("agent_slug", "") != agent_slug:
            return False
        return not (tenant and rec.get("tenant", "") != tenant)

    def _scoped(self, task_id: str, agent_slug: str = "", tenant: str = "") -> Optional[dict]:
        """Live record if visible in scope. Caller holds the lock."""
        rec = self._tasks.get(task_id)
        return rec if rec and self._in_scope(rec, agent_slug, tenant) else None

    def _push_rec(self, task_id: str, config_id: str = "", agent_slug: str = "", tenant: str = "") -> Optional[dict]:
        """Scoped record that has a push config (matching ``config_id`` if given). Caller holds the lock."""
        rec = self._scoped(task_id, agent_slug, tenant)
        if not rec or not rec.get("push_url"):
            return None
        if config_id and rec.get("push_config_id") != config_id:
            return None
        return rec

    @staticmethod
    def _push_config_view(rec: dict) -> dict:
        return {"configId": rec.get("push_config_id") or "", "taskId": rec["task_id"],
                "createdAt": rec.get("created_iso", ""), "pushNotificationConfig": {"url": rec.get("push_url") or ""}}

    def create(self, task_id: str, context_id: str, peer: str, agent_slug: str = "", tenant: str = "") -> dict:
        rec = {
            "task_id": task_id, "context_id": context_id, "peer": peer,
            "agent_slug": agent_slug or "", "tenant": tenant or "", "state": STATE_SUBMITTED, "reply": "",
            "created_at": time.time(), "created_iso": now_iso(), "push_url": "", "push_config_id": "",
        }
        with self._lock:
            self._tasks[task_id] = rec
        return dict(rec)

    def set_state(self, task_id: str, state: str) -> None:
        with self._lock:
            rec = self._tasks.get(task_id)
            if rec and rec["state"] not in TERMINAL_STATES:
                rec["state"] = state

    def set_push_config(self, task_id: str, url: str, agent_slug: str = "", tenant: str = "") -> Optional[dict]:
        """Attach a push notification config; returns the stored config or None."""
        with self._lock:
            rec = self._scoped(task_id, agent_slug, tenant)
            if not rec:
                return None
            rec["push_url"] = url
            rec["push_config_id"] = "cfg-" + uuid.uuid4().hex[:12]
            return self._push_config_view(rec)

    def get_push_config(self, task_id: str, config_id: str = "", agent_slug: str = "", tenant: str = "") -> Optional[dict]:
        with self._lock:
            rec = self._push_rec(task_id, config_id, agent_slug, tenant)
            return self._push_config_view(rec) if rec else None

    def list_push_configs(self, task_id: str, agent_slug: str = "", tenant: str = "") -> list[dict]:
        with self._lock:
            rec = self._push_rec(task_id, "", agent_slug, tenant)
            return [self._push_config_view(rec)] if rec else []

    def delete_push_config(self, task_id: str, config_id: str = "", agent_slug: str = "", tenant: str = "") -> bool:
        with self._lock:
            rec = self._push_rec(task_id, config_id, agent_slug, tenant)
            if not rec:
                return False
            rec["push_url"] = ""
            rec["push_config_id"] = ""
            return True

    def pop_push_url(self, task_id: str) -> str:
        with self._lock:
            rec = self._tasks.get(task_id)
            if not rec:
                return ""
            url, rec["push_url"] = rec["push_url"], ""
            return url

    def get(self, task_id: str, agent_slug: str = "", tenant: str = "") -> Optional[dict]:
        with self._lock:
            rec = self._scoped(task_id, agent_slug, tenant)
            return dict(rec) if rec else None

    def complete(self, task_id: str, state: str, reply: str = "") -> Optional[dict]:
        """Transition a task to a terminal state. Idempotent."""
        with self._lock:
            rec = self._tasks.get(task_id)
            if not rec or rec["state"] in TERMINAL_STATES:
                return None
            rec["state"] = state
            rec["reply"] = reply
            rec["completed_at"] = time.time()
            watchers = self._watchers.pop(task_id, [])
            self._trim_locked()
            out = dict(rec)
        for fut in watchers:
            if not fut.done():
                fut.set_result((state, reply))
        return out

    def watch(self, task_id: str, agent_slug: str = "", tenant: str = "") -> Optional[Future]:
        with self._lock:
            rec = self._scoped(task_id, agent_slug, tenant)
            if not rec:
                return None
            fut: Future = Future()
            if rec["state"] in TERMINAL_STATES:
                fut.set_result((rec["state"], rec.get("reply", "")))
            else:
                self._watchers.setdefault(task_id, []).append(fut)
            return fut

    def list(self, context_id: str = "", state: str = "", page_size: int = 50, offset: int = 0,
             agent_slug: str = "", tenant: str = "", with_total: bool = False):
        """Filtered task page (newest first) as ``(records, next_offset)``, or
        ``(records, next_offset, total)`` with ``with_total`` (v1.0 ListTasks totalSize)."""
        page_size = max(1, min(int(page_size or 50), 100))
        with self._lock:
            recs = [dict(r) for r in reversed(self._tasks.values())]
        if agent_slug or tenant:
            recs = [r for r in recs if self._in_scope(r, agent_slug, tenant)]
        if context_id:
            recs = [r for r in recs if r["context_id"] == context_id]
        if state:
            recs = [r for r in recs if r["state"] == state]
        total = len(recs)
        page = recs[offset:offset + page_size]
        next_offset = offset + page_size if offset + page_size < total else 0
        if with_total:
            return page, next_offset, total
        return page, next_offset

    def fail_orphans(self, timeout_seconds: int = 300) -> list[str]:
        with self._lock:
            now = time.time()
            stale = [tid for tid, rec in self._tasks.items()
                     if rec["state"] not in TERMINAL_STATES and now - rec["created_at"] > timeout_seconds]
        return [tid for tid in stale if self.complete(tid, STATE_FAILED, "[task orphaned — no reply produced]")]

    def _trim_locked(self) -> None:
        terminal = [tid for tid, rec in self._tasks.items() if rec["state"] in TERMINAL_STATES]
        for tid in terminal[:max(0, len(terminal) - self._MAX_TERMINAL)]:
            self._tasks.pop(tid, None)

    @staticmethod
    def to_task(rec: dict, history_length: Optional[int] = None, include_artifacts: bool = True) -> dict:
        """Render a stored record as an A2A v1.0 Task object."""
        task = build_task(rec["task_id"], rec["context_id"], rec["state"], rec.get("reply", ""),
                          created_at=rec.get("created_iso", ""))
        if not include_artifacts:
            task.pop("artifacts", None)
        if history_length == 0:
            task.pop("history", None)
        return copy.deepcopy(task)


# ── Conversation persistence (outside the context-compaction pipeline) ────────

def _conv_dir() -> Path:
    return _hermes_home() / "a2a_conversations"


def _conv_path(context_id: str) -> Path:
    safe = "".join(c for c in (context_id or "default") if c.isalnum() or c in "-_") or "default"
    return _conv_dir() / f"{safe}.jsonl"


def persist_message(context_id: str, role: str, text: str, task_id: str = "") -> None:
    """Append one message to the context's on-disk conversation log."""
    try:
        _conv_dir().mkdir(parents=True, exist_ok=True)
        rec = {"ts": time.time(), "role": role, "text": text, "task_id": task_id}
        with _conv_path(context_id).open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    except Exception:
        pass


def load_conversation(context_id: str, limit: int = 50) -> list[dict]:
    """Load the last *limit* messages for a context (empty list if none)."""
    path = _conv_path(context_id)
    if not path.exists():
        return []
    out: list[dict] = []
    try:
        with path.open("r", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    try:
                        out.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    except Exception:
        return []
    return out[-limit:]


def list_conversations() -> list[str]:
    """Return known context-ids that have persisted conversations."""
    d = _conv_dir()
    if not d.exists():
        return []
    return sorted(p.stem for p in d.glob("*.jsonl"))
