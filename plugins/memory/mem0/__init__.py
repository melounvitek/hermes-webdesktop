"""Mem0 memory plugin — MemoryProvider interface.

Server-side LLM fact extraction, semantic search and deduplication via the Mem0
Platform API (cloud), a self-hosted Mem0 server (MEM0_HOST, HTTP), or OSS Memory.
Secrets live in $HERMES_HOME/.env (MEM0_API_KEY, MEM0_HOST); behavioral settings
in $HERMES_HOME/mem0.json via `hermes memory setup`: mode ("platform"|"oss"), host,
user_id (canonical id across every gateway so one human gets one merged store;
unset → gateway-native id), agent_id. MEM0_MODE/MEM0_USER_ID/MEM0_AGENT_ID env
vars remain a backward-compatible fallback.
"""

from __future__ import annotations

import atexit
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from agent.secret_scope import get_secret
from tools.registry import tool_error

logger = logging.getLogger(__name__)

# Circuit breaker: after _BREAKER_THRESHOLD consecutive failures, pause API
# calls for _BREAKER_COOLDOWN_SECS to avoid hammering a down server.
_BREAKER_THRESHOLD, _BREAKER_COOLDOWN_SECS, _PREFETCH_WAIT_SECS = 5, 120, 3
_CLIENT_ERROR_TYPES = ("MemoryNotFoundError", "ValidationError")
# Placeholder user_id. initialize() treats it as "no operator-configured user_id"
# so legacy mem0.json files written by the wizard don't override gateway-native ids.
_DEFAULT_USER_ID = "hermes-user"


def _is_client_error(exc: Exception) -> bool:
    """True for user-caused errors (bad ID, not found) that should NOT trip circuit breaker."""
    err_str = str(exc).lower()
    return type(exc).__name__ in _CLIENT_ERROR_TYPES or any(s in err_str for s in ("404", "not found", "valid uuid"))


def _read_mem0_json(config_path: Path) -> dict:
    """Best-effort read of mem0.json; missing/corrupt file -> {}."""
    if config_path.exists():
        try:
            return json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _load_config() -> dict:
    """Env vars provide defaults; $HERMES_HOME/mem0.json overrides individual keys.
    Layering avoids a silent failure when the JSON file exists but lacks fields
    like ``api_key`` that the user set in ``.env``."""
    from hermes_constants import get_hermes_home
    config = {
        "mode": os.environ.get("MEM0_MODE", "platform"),
        "api_key": get_secret("MEM0_API_KEY", ""),
        "host": os.environ.get("MEM0_HOST", ""),
        "agent_id": os.environ.get("MEM0_AGENT_ID", "hermes"),
        "oss": {},
    }
    # Only carry user_id when explicitly configured so initialize() can fall
    # back to the gateway-native id.
    if os.environ.get("MEM0_USER_ID"):
        config["user_id"] = os.environ["MEM0_USER_ID"]
    file_cfg = _read_mem0_json(get_hermes_home() / "mem0.json")
    config.update({k: v for k, v in file_cfg.items() if v is not None and v != ""})
    return config


# ---------------------------------------------------------------------------
# Tool schemas
# ---------------------------------------------------------------------------

def _schema(name: str, description: str, properties: dict, required: list[str]) -> dict:
    return {"name": name, "description": description, "parameters": {"type": "object", "properties": properties, "required": required}}


def _param(type_: str, description: str) -> dict:
    return {"type": type_, "description": description}


SEARCH_SCHEMA = _schema(
    "mem0_search",
    "Search the user's memories by meaning; returns facts ranked by "
    "relevance. Use this before answering any question that may depend on "
    "what you know about the user (preferences, facts, history, people, "
    "projects, past decisions). For multi-part or multi-hop questions, "
    "call it several times — vary the wording and run follow-up searches "
    "on what earlier results reveal; one search is rarely enough.",
    {
        "query": _param("string", "What to search for."),
        "top_k": _param("integer", "Max results (default: 10, max: 50)."),
        "rerank": _param("boolean", "Rerank results for relevance (default: false, platform mode only)."),
    },
    ["query"],
)

ADD_SCHEMA = _schema(
    "mem0_add",
    "Store a durable fact about the user, verbatim (no LLM extraction). "
    "Call this the moment the user states a lasting preference, correction, "
    "decision, or personal detail worth recalling on future turns — don't "
    "wait to be asked to remember. Skip transient chit-chat and facts you've "
    "already stored.",
    {"content": _param("string", "The fact to store.")},
    ["content"],
)

UPDATE_SCHEMA = _schema(
    "mem0_update",
    "Replace the text of an existing memory by its ID (take the ID from a "
    "mem0_search result). Use when a stored fact has changed "
    "or was wrong — correct it in place instead of adding a duplicate.",
    {"memory_id": _param("string", "Memory UUID to update."), "text": _param("string", "New text content.")},
    ["memory_id", "text"],
)

DELETE_SCHEMA = _schema(
    "mem0_delete",
    "Delete a memory by its ID (take the ID from a mem0_search "
    "result). Use when a stored fact is obsolete or the user asks you to "
    "forget it; prefer mem0_update if the fact merely changed.",
    {"memory_id": _param("string", "Memory UUID to delete.")},
    ["memory_id"],
)


# ---------------------------------------------------------------------------
# MemoryProvider implementation
# ---------------------------------------------------------------------------

class Mem0MemoryProvider(MemoryProvider):
    """Mem0 memory with server-side extraction and semantic search (platform, self-hosted or OSS)."""

    def __init__(self):
        self._config = self._backend = None
        self._mode, self._api_key, self._host = "platform", "", ""
        self._user_id, self._agent_id = _DEFAULT_USER_ID, "hermes"
        self._rerank_default = False
        self._channel = "cli"  # gateway channel name (cli/telegram/discord/...)
        self._sync_thread = self._prefetch_thread = None
        self._prefetch_query = self._prefetch_result = ""
        self._prefetch_done = self._atexit_registered = False
        self._consecutive_failures, self._breaker_open_until = 0, 0.0  # circuit breaker state
        self._breaker_lock, self._sync_lock, self._prefetch_lock = threading.Lock(), threading.Lock(), threading.Lock()

    @property
    def name(self) -> str:
        return "mem0"

    def is_available(self) -> bool:
        cfg = _load_config()
        if cfg.get("mode", "platform") == "oss":
            return bool(cfg.get("oss", {}).get("vector_store"))
        # Platform needs an api_key; self-hosted needs a host (api_key optional
        # when the server runs with AUTH_DISABLED).
        return bool(cfg.get("api_key") or cfg.get("host"))

    def save_config(self, values, hermes_home):
        """Merge-write config to $HERMES_HOME/mem0.json."""
        from utils import atomic_json_write
        config_path = Path(hermes_home) / "mem0.json"
        existing = _read_mem0_json(config_path)
        existing.update(values)
        atomic_json_write(config_path, existing, mode=0o600)

    def get_config_schema(self):
        api_key_required = _load_config().get("mode", "platform") != "oss"
        return [
            {"key": "api_key", "description": "Mem0 Platform API key", "secret": True, "required": api_key_required, "env_var": "MEM0_API_KEY", "url": "https://app.mem0.ai"},
            {"key": "host", "description": "Self-hosted Mem0 server URL (leave blank for cloud)", "required": False, "env_var": "MEM0_HOST"},
            {"key": "user_id", "description": "User identifier", "default": "hermes-user"},
            {"key": "agent_id", "description": "Agent identifier", "default": "hermes"},
            {"key": "rerank", "description": "Enable reranking for recall", "default": "false", "choices": ["true", "false"]},
        ]

    def post_setup(self, hermes_home: str, config: dict) -> None:
        from ._setup import post_setup
        post_setup(hermes_home, config)

    def _vs_provider(self, default: str) -> str:
        """Configured OSS vector-store provider name (for error hints)."""
        return self._config.get("oss", {}).get("vector_store", {}).get("provider", default)

    def _create_backend(self):
        # Lazy-install the mem0 SDK before either backend imports it. ensure() honors
        # security.allow_lazy_installs and redirects sealed Docker venvs to the durable
        # target; on failure the backend import raises the canonical error, captured below.
        try:
            from tools.lazy_deps import ensure as _lazy_ensure
            _lazy_ensure("memory.mem0", prompt=False)
        except Exception:
            pass
        try:
            if self._mode == "oss":
                from ._backend import OSSBackend
                return OSSBackend(self._config.get("oss", {}))
            if self._host:
                from ._backend import SelfHostedBackend
                return SelfHostedBackend(self._api_key, self._host)
            from ._backend import PlatformBackend
            return PlatformBackend(self._api_key)
        except Exception as e:
            logger.error("Mem0 backend failed to initialize (%s mode): %s", self._mode, e)
            self._init_error = str(e)
            return None

    def _is_breaker_open(self) -> bool:
        """Return True if the circuit breaker is tripped (too many failures)."""
        with self._breaker_lock:
            if self._consecutive_failures < _BREAKER_THRESHOLD:
                return False
            if time.monotonic() >= self._breaker_open_until:
                self._consecutive_failures = 0
                return False
            return True

    def _format_error(self, prefix: str, exc: Exception) -> str:
        msg = f"{prefix}: {exc}"
        if self._mode == "oss":
            err_str = str(exc).lower()
            if "connection" in err_str or "refused" in err_str or "timeout" in err_str:
                msg += f" (check that {self._vs_provider('vector store')} is running)"
        return msg

    def _record_success(self):
        with self._breaker_lock:
            self._consecutive_failures = 0

    def _record_failure(self):
        with self._breaker_lock:
            self._consecutive_failures = count = self._consecutive_failures + 1
            tripped = count >= _BREAKER_THRESHOLD
            if tripped:
                self._breaker_open_until = time.monotonic() + _BREAKER_COOLDOWN_SECS
        if tripped:
            hint = f" Check that your {self._vs_provider('unknown')} vector store is running and reachable." if self._mode == "oss" else ""
            logger.warning(
                "Mem0 circuit breaker tripped after %d consecutive failures. "
                "Pausing API calls for %ds.%s",
                count, _BREAKER_COOLDOWN_SECS, hint,
            )

    def initialize(self, session_id: str, **kwargs) -> None:
        self._config = _load_config()
        self._mode = self._config.get("mode", "platform")
        self._api_key = self._config.get("api_key", "")
        self._host = self._config.get("host", "")
        # user_id precedence: operator-configured (env/mem0.json) > gateway-native id
        # from kwargs > _DEFAULT_USER_ID. The literal placeholder counts as unset so
        # wizard users still get gateway-native ids instead of being bucketed together.
        configured = self._config.get("user_id")
        self._user_id = (None if configured == _DEFAULT_USER_ID else configured) or kwargs.get("user_id") or _DEFAULT_USER_ID
        self._agent_id = self._config.get("agent_id", "hermes")
        # Persisted rerank preference: DEFAULT for mem0_search when the model doesn't
        # pass ``rerank``; per-call args win. Platform-only; other backends ignore it.
        _rr = self._config.get("rerank", False)
        self._rerank_default = _rr.lower() in ("true", "1", "yes") if isinstance(_rr, str) else bool(_rr)
        self._channel = kwargs.get("platform") or "cli"
        self._backend = self._create_backend()
        if self._backend and not self._atexit_registered:
            atexit.register(self._shutdown_backend)
            self._atexit_registered = True

    def _read_filters(self) -> Dict[str, Any]:
        # Scoped to user_id only — by design — so recall surfaces memories from any
        # gateway/agent under this principal; writes attach agent_id and metadata.channel
        # (dashboard per-channel filtering) so narrower views remain possible at query time.
        return {"user_id": self._user_id}

    def _write_metadata(self) -> Dict[str, Any]:
        return {"channel": self._channel} if self._channel else {}

    def system_prompt_block(self) -> str:
        # Mirror _create_backend precedence (oss > host > platform) so the label names
        # the backend that actually runs. Rerank is a Mem0 Platform feature only.
        mode_label = "OSS (self-hosted)" if self._mode == "oss" else "self-hosted (HTTP API)" if self._host else "platform (cloud API)"
        rerank_note = " Rerank is available on search." if (self._mode == "platform" and not self._host) else ""
        return (
            "# Mem0 Memory\n"
            f"Active. Mode: {mode_label}. User: {self._user_id}.\n"
            "You have persistent memory of this user from past conversations. "
            "You should call mem0_search before answering anything that could depend "
            "on prior context (the user's preferences, facts, history, people, "
            "projects, or earlier decisions) — do not rely on the chat window "
            "alone, and do not assume you have no memory.\n"
            "For multi-part or multi-hop questions, run several searches with "
            "different wording/angles and follow-up searches on what the first "
            "results surface; one search is rarely enough. Keep searching until "
            "you have every fact the question needs before you answer.\n"
            "Tools: mem0_search to find memories, mem0_add to store facts, "
            f"mem0_update and mem0_delete to manage by ID.{rerank_note}"
        )

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        self._start_prefetch(message)

    def _consume_prefetch_result(self, query: str) -> str | None:
        with self._prefetch_lock:
            if self._prefetch_query != query or not self._prefetch_done:
                return None
            result, self._prefetch_result, self._prefetch_done = self._prefetch_result, "", False
            return result

    def _start_prefetch(self, query: str) -> None:
        backend = self._backend
        if not query or backend is None or self._is_breaker_open():
            return
        with self._prefetch_lock:
            # Same query already answered or still in flight: don't restart it.
            if self._prefetch_query == query and (
                self._prefetch_done or (self._prefetch_thread and self._prefetch_thread.is_alive())
            ):
                return
            self._prefetch_query, self._prefetch_result, self._prefetch_done = query, "", False

        def _run():
            body = ""
            try:
                results = backend.search(query, filters=self._read_filters(), top_k=10, rerank=False)
                lines = [r.get("memory", "") for r in (results or []) if r.get("memory")]
                if lines:
                    body = "## Mem0 Memory\n" + "\n".join(f"- {l}" for l in lines)
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.debug("Mem0 prefetch failed: %s", e)
            with self._prefetch_lock:
                if self._prefetch_query == query:
                    self._prefetch_result = body
                    self._prefetch_done = True

        t = threading.Thread(target=_run, daemon=True, name="mem0-prefetch")
        with self._prefetch_lock:
            self._prefetch_thread = t
        t.start()

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Recall memories for the CURRENT question with a short hot-path wait."""
        cached = self._consume_prefetch_result(query)
        if cached is not None:
            return cached
        self._start_prefetch(query)
        with self._prefetch_lock:
            thread = self._prefetch_thread if self._prefetch_query == query else None
        if thread:
            thread.join(timeout=_PREFETCH_WAIT_SECS)
        # Slow backend: skip injection; mem0_search tool remains the backstop.
        return self._consume_prefetch_result(query) or ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        """Send the turn to Mem0 for server-side fact extraction (non-blocking)."""
        if self._backend is None or self._is_breaker_open():
            return

        def _sync():
            backend = self._backend
            if backend is None:
                return
            try:
                messages = [{"role": "user", "content": user_content}, {"role": "assistant", "content": assistant_content}]
                backend.add(messages, user_id=self._user_id, agent_id=self._agent_id, infer=True, metadata=self._write_metadata())
                self._record_success()
            except Exception as e:
                self._record_failure()
                logger.warning("Mem0 sync failed: %s", e)

        with self._sync_lock:
            if self._sync_thread and self._sync_thread.is_alive():
                self._sync_thread.join(timeout=5.0)
            # If still alive after timeout, skip to avoid duplicate ingestion.
            if self._sync_thread and self._sync_thread.is_alive():
                return
            self._sync_thread = threading.Thread(target=_sync, daemon=True, name="mem0-sync")
            self._sync_thread.start()

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [SEARCH_SCHEMA, ADD_SCHEMA, UPDATE_SCHEMA, DELETE_SCHEMA]

    # -- tool handlers -------------------------------------------------------

    def _tool_search(self, args: dict) -> str:
        query = args.get("query", "")
        if not query:
            return tool_error("Missing required parameter: query")
        try:
            top_k = max(1, min(int(args.get("top_k", 10)), 50))
            rerank_raw = args.get("rerank", self._rerank_default)
            rerank = rerank_raw.lower() not in ("false", "0", "no") if isinstance(rerank_raw, str) else bool(rerank_raw)
            results = self._backend.search(query, filters=self._read_filters(), top_k=top_k, rerank=rerank)
            self._record_success()
            if not results:
                return json.dumps({"result": "No relevant memories found."})
            items = [{"id": r.get("id"), "memory": r.get("memory", ""), "score": r.get("score", 0)} for r in results]
            return json.dumps({"results": items, "count": len(items)})
        except Exception as e:
            if not _is_client_error(e):
                self._record_failure()
            return tool_error(self._format_error("Search failed", e))

    def _tool_add(self, args: dict) -> str:
        content = args.get("content", "")
        if not content:
            return tool_error("Missing required parameter: content")
        try:
            result = self._backend.add(
                [{"role": "user", "content": content}],
                user_id=self._user_id, agent_id=self._agent_id, infer=False, metadata=self._write_metadata(),
            )
            self._record_success()
            event_id = result.get("event_id") if isinstance(result, dict) else None
            # Cloud add is async (server-side extraction); OSS and self-hosted store synchronously.
            msg = "Fact stored." if (self._mode == "oss" or self._host) else "Fact queued for storage."
            return json.dumps({"result": msg, "event_id": event_id})
        except Exception as e:
            self._record_failure()
            return tool_error(self._format_error("Failed to store", e))

    def _tool_by_id(self, args: dict, required: tuple[str, ...], label: str, method: str) -> str:
        """Shared update/delete shape: required-param check, then backend.<method>(*values)."""
        values = [args.get(k, "") for k in required]
        for k, v in zip(required, values):
            if not v:
                return tool_error(f"Missing required parameter: {k}")
        try:
            result = getattr(self._backend, method)(*values)
            self._record_success()
            return json.dumps(result)
        except Exception as e:
            if _is_client_error(e):
                return tool_error(f"Memory not found: {values[0]}")
            self._record_failure()
            return tool_error(self._format_error(label, e))

    def _tool_update(self, args: dict) -> str:
        return self._tool_by_id(args, ("memory_id", "text"), "Update failed", "update")

    def _tool_delete(self, args: dict) -> str:
        return self._tool_by_id(args, ("memory_id",), "Delete failed", "delete")

    _TOOL_HANDLERS = {
        "mem0_search": _tool_search,
        "mem0_add": _tool_add,
        "mem0_update": _tool_update,
        "mem0_delete": _tool_delete,
    }

    def handle_tool_call(self, tool_name: str, args: dict, **kwargs) -> str:
        if self._backend is None:
            err = getattr(self, "_init_error", "unknown error")
            hint = f" Check that {self._vs_provider('vector store')} is running and reachable." if self._mode == "oss" else ""
            return json.dumps({"error": f"Mem0 backend not initialized: {err}.{hint}"})
        if self._is_breaker_open():
            hint = f" Check that your {self._vs_provider('vector store')} is running." if self._mode == "oss" else ""
            return json.dumps({"error": f"Mem0 temporarily unavailable (multiple consecutive failures). Will retry automatically.{hint}"})
        handler = self._TOOL_HANDLERS.get(tool_name)
        if handler is None:
            return tool_error(f"Unknown tool: {tool_name}")
        return handler(self, args)

    def _shutdown_backend(self):
        try:
            if self._backend:
                self._backend.close()
                self._backend = None
        except Exception:
            pass

    def shutdown(self) -> None:
        for t in (self._prefetch_thread, self._sync_thread):
            if t and t.is_alive():
                t.join(timeout=5.0)
        self._shutdown_backend()


def register(ctx) -> None:
    """Register Mem0 as a memory provider plugin."""
    ctx.register_memory_provider(Mem0MemoryProvider())
