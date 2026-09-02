"""Async LSP client over stdin/stdout.

One :class:`LSPClient` per ``(language_server, workspace_root)`` pair.  It owns
the child process, drives JSON-RPC, and exposes :meth:`open_file`,
:meth:`wait_for_diagnostics`, :meth:`diagnostics_for` and :meth:`shutdown`.
:class:`agent.lsp.manager.LSPService` runs the event loop in a background
thread so the synchronous file_operations layer can call in.

Implementation notes:

- Freshness is tracked with **document versions**, not timestamps: every
  didChange bumps ``version`` and each stored push/pull result is tagged with
  the version it describes.  A result is fresh iff its tag >= the version
  being waited on, so a didChange implicitly invalidates everything older.
  This is what prevents "ghost diagnostics" — a slow server's leftovers from
  the previous edit can never masquerade as a verdict on the current content.
- Whole-document sync: even when the server advertises incremental sync we
  send one ``contentChanges`` entry replacing the whole document.  Every
  major server tolerates this and it saves range bookkeeping.
- Every ``open_file`` also fires ``workspace/didChangeWatchedFiles`` (CREATED
  first, CHANGED after) — some servers (clangd, eslint) only re-scan on it.
- ``ContentModified`` (-32801) errors are retried with exponential backoff.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set
from urllib.parse import quote, unquote

from hermes_cli._subprocess_compat import windows_hide_flags

from agent.lsp.protocol import (
    ERROR_CONTENT_MODIFIED,
    ERROR_METHOD_NOT_FOUND,
    LSPProtocolError,
    LSPRequestError,
    classify_message,
    encode_message,
    make_error_response,
    make_notification,
    make_request,
    make_response,
    read_message,
)

logger = logging.getLogger("agent.lsp.client")

# Timeouts (seconds).
INITIALIZE_TIMEOUT = 45.0
DIAGNOSTICS_DOCUMENT_WAIT = 5.0
DIAGNOSTICS_FULL_WAIT = 10.0
DIAGNOSTICS_REQUEST_TIMEOUT = 3.0
PUSH_DEBOUNCE = 0.15
SHUTDOWN_GRACE = 1.0  # seconds between SIGTERM and SIGKILL

# Retry policy for transient ContentModified errors: 0.5, 1.0, 2.0s.
MAX_CONTENT_MODIFIED_RETRIES = 3
RETRY_BASE_DELAY = 0.5

_WRITE_ERRORS = (BrokenPipeError, ConnectionResetError, OSError)


def file_uri(path: str) -> str:
    """Return a ``file://`` URI for a path (handles spaces, unicode, Windows drive letters)."""
    abs_path = os.path.abspath(path)
    if os.name == "nt":
        # ``C:\foo`` → ``file:///C:/foo``: the drive letter must be a path component.
        abs_path = abs_path.replace("\\", "/")
        if not abs_path.startswith("/"):
            abs_path = "/" + abs_path
    return "file://" + quote(abs_path, safe="/:")


def uri_to_path(uri: str) -> str:
    """Inverse of :func:`file_uri`."""
    if not uri.startswith("file://"):
        return uri
    raw = uri[len("file://"):]
    if os.name == "nt" and raw.startswith("/") and len(raw) > 2 and raw[2] == ":":
        raw = raw[1:]  # strip leading slash before drive letter
    return os.path.normpath(unquote(raw))


def _end_position(text: str) -> Dict[str, int]:
    """LSP Position at the end of ``text`` (for a whole-document replace range)."""
    if not text:
        return {"line": 0, "character": 0}
    lines = text.splitlines(keepends=False)
    # A trailing newline isn't represented by splitlines: the end is then
    # the start of the next (empty) line.
    if text.endswith(("\n", "\r")):
        return {"line": len(lines), "character": 0}
    return {"line": len(lines) - 1, "character": len(lines[-1])}


@dataclass
class _DocState:
    """Per-document state.

    ``version`` is the LSP document version last sent (didOpen=0, +1 per
    didChange) and doubles as the freshness token: ``push_version`` /
    ``pull_version`` tag stored results, which are fresh iff tag >= version.
    Tags start at -1 ("no data yet").  Servers that echo a version in
    publishDiagnostics get exact tagging; others are credited with the
    current version at receipt time.
    """

    version: int = 0
    text: str = ""
    push: List[Dict[str, Any]] = field(default_factory=list)
    pull: List[Dict[str, Any]] = field(default_factory=list)
    push_version: int = -1
    pull_version: int = -1
    seed_seen: bool = False

    def fresh_push(self, version: Optional[int] = None) -> bool:
        return self.push_version >= (self.version if version is None else version)

    def fresh_pull(self, version: Optional[int] = None) -> bool:
        return self.pull_version >= (self.version if version is None else version)


class LSPClient:
    """Async LSP client tied to one server process and one workspace root.

    Lifecycle: ``start()`` → ``open_file()`` → ``wait_for_diagnostics()`` →
    ``diagnostics_for()`` → ``shutdown()``.
    """

    def __init__(
        self,
        *,
        server_id: str,
        workspace_root: str,
        command: List[str],
        env: Optional[Dict[str, str]] = None,
        cwd: Optional[str] = None,
        initialization_options: Optional[Dict[str, Any]] = None,
        seed_diagnostics_on_first_push: bool = False,
    ) -> None:
        self.server_id = server_id
        self.workspace_root = workspace_root
        self._command = list(command)
        self._env = env
        self._cwd = cwd or workspace_root
        self._init_options = initialization_options or {}
        self._seed_first_push = seed_diagnostics_on_first_push

        self._proc: Optional[asyncio.subprocess.Process] = None
        self._stderr_task: Optional[asyncio.Task] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._cleanup_lock = asyncio.Lock()

        self._next_id: int = 0
        self._pending: Dict[int, asyncio.Future] = {}

        # Server → client requests; anything else gets method-not-found.
        self._request_handlers: Dict[str, Callable[[Any], Awaitable[Any]]] = {
            "window/workDoneProgress/create": self._handle_work_done_create,
            "workspace/configuration": self._handle_workspace_configuration,
            "client/registerCapability": self._handle_register_capability,
            "client/unregisterCapability": self._handle_unregister_capability,
            "workspace/workspaceFolders": self._handle_workspace_folders,
            "workspace/diagnostic/refresh": self._handle_diagnostic_refresh,
        }
        # Server → client notifications; others (showMessage, $/progress) are dropped.
        self._notification_handlers: Dict[str, Callable[[Any], None]] = {
            "textDocument/publishDiagnostics": self._handle_publish_diagnostics,
        }

        # Per-document state keyed by absolute path (NOT URI).
        self._docs: Dict[str, _DocState] = {}
        # Only diagnostic capability registrations are tracked.
        self._diagnostic_registrations: Dict[str, Dict[str, Any]] = {}

        self._state: str = "stopped"
        self._sync_kind: int = 1  # 1=Full, 2=Incremental
        self._stopping: bool = False

        # Waiters snapshot ``_push_counter`` and treat any increase as "recheck
        # the predicate" — avoids the asyncio.Event sticky-state trap.
        self._push_event = asyncio.Event()
        self._push_counter = 0
        self._registration_event = asyncio.Event()

    @property
    def is_running(self) -> bool:
        return self._state == "running" and self._connection_is_open()

    def _connection_is_open(self) -> bool:
        proc = self._proc
        reader = self._reader_task
        return (
            self._state in {"starting", "running"}
            and proc is not None
            and proc.returncode is None
            and proc.stdin is not None
            and not proc.stdin.is_closing()
            and reader is not None
            and not reader.done()
        )

    @property
    def state(self) -> str:
        return self._state

    async def start(self) -> None:
        """Spawn the server and complete the initialize handshake.

        On failure the process is killed and state is ``"error"``; re-call to retry.
        """
        if self._state in {"running", "starting"}:
            return
        self._state = "starting"
        try:
            await self._spawn()
            await self._initialize()
            if not self._connection_is_open():
                raise LSPProtocolError("server connection closed during initialization")
            self._state = "running"
        except Exception:
            self._state = "error"
            await self._cleanup_process()
            raise

    @staticmethod
    def _win_wrap_cmd(cmd: List[str]) -> List[str]:
        """On Windows, wrap .cmd/.bat shims so CreateProcess can run them."""
        if cmd[0].lower().endswith((".cmd", ".bat")):
            return ["cmd.exe", "/c", *cmd]
        return cmd

    async def _spawn(self) -> None:
        env = dict(os.environ)
        if self._env:
            env.update(self._env)

        cmd = self._command
        if sys.platform == "win32":
            cmd = self._win_wrap_cmd(cmd)

        try:
            # start_new_session=True gives the server its own process group.
            # Otherwise it inherits the gateway's pgid and mcp_tool's orphan
            # sweeper can killpg() the TUI parent along with it.
            # windows_hide_flags() suppresses the console window a .cmd shim
            # would flash from a console-less host (CREATE_NO_WINDOW; 0 on POSIX).
            self._proc = await asyncio.create_subprocess_exec(
                cmd[0],
                *cmd[1:],
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
                cwd=self._cwd,
                start_new_session=True,
                creationflags=windows_hide_flags(),
            )
        except FileNotFoundError as e:
            raise LSPProtocolError(
                f"LSP server binary not found: {cmd[0]} ({e})"
            ) from e

        # stderr must be drained or the pipe buffer fills and the server hangs.
        self._stderr_task = asyncio.create_task(self._drain_stderr())
        self._reader_task = asyncio.create_task(self._reader_loop())

    async def _drain_stderr(self) -> None:
        if self._proc is None or self._proc.stderr is None:
            return
        try:
            while True:
                line = await self._proc.stderr.readline()
                if not line:
                    break
                text = line.decode("utf-8", errors="replace").rstrip()
                if text:
                    logger.debug("[%s] stderr: %s", self.server_id, text[:1000])
        except (asyncio.CancelledError, OSError):
            pass

    async def _reader_loop(self) -> None:
        if self._proc is None or self._proc.stdout is None:
            return
        try:
            while True:
                msg = await read_message(self._proc.stdout)
                if msg is None:
                    logger.debug("[%s] server closed stdout cleanly", self.server_id)
                    break
                kind, key = classify_message(msg)
                if kind == "response":
                    self._dispatch_response(key, msg)
                elif kind == "request":
                    asyncio.create_task(self._dispatch_request(key, msg))
                elif kind == "notification":
                    self._dispatch_notification(key, msg)
                else:
                    logger.warning("[%s] dropping invalid message: %r", self.server_id, msg)
        except LSPProtocolError as e:
            logger.warning("[%s] protocol error in reader loop: %s", self.server_id, e)
        except (asyncio.CancelledError, OSError):
            pass
        finally:
            unexpected_close = not self._stopping and self._state in {"starting", "running"}
            if unexpected_close:
                self._state = "error"
            # Fail pending requests fast.
            for fut in list(self._pending.values()):
                if not fut.done():
                    fut.set_exception(LSPProtocolError("server connection closed"))
            self._pending.clear()
            if unexpected_close:
                await self._cleanup_process()

    async def _initialize(self) -> None:
        params = {
            "rootUri": file_uri(self.workspace_root),
            "rootPath": self.workspace_root,
            "processId": os.getpid(),
            "workspaceFolders": [
                {"name": "workspace", "uri": file_uri(self.workspace_root)}
            ],
            "initializationOptions": self._init_options,
            "capabilities": {
                "window": {"workDoneProgress": True},
                "workspace": {
                    "configuration": True,
                    "workspaceFolders": True,
                    "didChangeWatchedFiles": {"dynamicRegistration": True},
                    "diagnostics": {"refreshSupport": False},
                },
                "textDocument": {
                    "synchronization": {
                        "dynamicRegistration": False,
                        "didOpen": True,
                        "didChange": True,
                        "didSave": True,
                        "willSave": False,
                        "willSaveWaitUntil": False,
                    },
                    "diagnostic": {
                        "dynamicRegistration": True,
                        "relatedDocumentSupport": True,
                    },
                    "publishDiagnostics": {
                        "relatedInformation": True,
                        "tagSupport": {"valueSet": [1, 2]},
                        "versionSupport": True,
                        "codeDescriptionSupport": True,
                        "dataSupport": False,
                    },
                    "hover": {"contentFormat": ["markdown", "plaintext"]},
                    "definition": {"linkSupport": True},
                    "references": {},
                    "documentSymbol": {"hierarchicalDocumentSymbolSupport": True},
                },
                "general": {"positionEncodings": ["utf-16"]},
            },
        }

        result = await asyncio.wait_for(
            self._send_request("initialize", params),
            timeout=INITIALIZE_TIMEOUT,
        )
        self._sync_kind = self._extract_sync_kind(result.get("capabilities") or {})

        await self._send_notification("initialized", {})
        if self._init_options:
            # Some servers (vtsls, eslint) only pick config up via
            # didChangeConfiguration even when it was in initializationOptions.
            await self._send_notification(
                "workspace/didChangeConfiguration",
                {"settings": self._init_options},
            )

    @staticmethod
    def _extract_sync_kind(capabilities: dict) -> int:
        sync = capabilities.get("textDocumentSync")
        if isinstance(sync, int):
            return sync
        if isinstance(sync, dict):
            change = sync.get("change")
            if isinstance(change, int):
                return change
        return 1  # default to Full

    async def shutdown(self) -> None:
        """Best-effort graceful shutdown: ``shutdown`` + ``exit``, then SIGTERM/SIGKILL.  Idempotent."""
        if self._stopping:
            return
        self._stopping = True
        try:
            if self.is_running:
                try:
                    await asyncio.wait_for(self._send_request("shutdown", None), timeout=2.0)
                except (asyncio.TimeoutError, LSPRequestError, LSPProtocolError):
                    pass
                try:
                    await self._send_notification("exit", None)
                except Exception:
                    pass
        finally:
            self._state = "stopped"
            await self._cleanup_process()

    @staticmethod
    async def _cancel_task(task: Optional[asyncio.Task]) -> None:
        if task is not None and not task.done():
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):  # noqa: BLE001
                pass

    async def _cleanup_process(self) -> None:
        async with self._cleanup_lock:
            reader_task = self._reader_task
            self._reader_task = None
            if reader_task is not asyncio.current_task():
                await self._cancel_task(reader_task)
            stderr_task = self._stderr_task
            self._stderr_task = None
            await self._cancel_task(stderr_task)
            proc = self._proc
            self._proc = None
            if proc is None or proc.returncode is not None:
                return
            try:
                proc.terminate()
                try:
                    await asyncio.wait_for(proc.wait(), timeout=SHUTDOWN_GRACE)
                except asyncio.TimeoutError:
                    try:
                        proc.kill()
                        await proc.wait()
                    except ProcessLookupError:
                        pass
            except ProcessLookupError:
                pass

    # ------------------------------------------------------------------
    # request / notification plumbing
    # ------------------------------------------------------------------

    async def _write(self, msg: dict) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(encode_message(msg))
        await self._proc.stdin.drain()

    async def _send_request(self, method: str, params: Any) -> Any:
        if not self._connection_is_open():
            raise LSPProtocolError(f"cannot send {method!r}: server connection closed")
        loop = asyncio.get_running_loop()
        req_id = self._next_id
        self._next_id += 1
        fut: asyncio.Future = loop.create_future()
        self._pending[req_id] = fut
        try:
            await self._write(make_request(req_id, method, params))
        except _WRITE_ERRORS as e:
            self._pending.pop(req_id, None)
            raise LSPProtocolError(f"send failed for {method!r}: {e}") from e
        try:
            return await fut
        finally:
            self._pending.pop(req_id, None)

    async def _send_request_with_retry(self, method: str, params: Any, *, timeout: float) -> Any:
        """Send a request, retrying ``ContentModified`` (-32801) with backoff; other errors propagate."""
        for attempt in range(MAX_CONTENT_MODIFIED_RETRIES + 1):
            try:
                return await asyncio.wait_for(self._send_request(method, params), timeout=timeout)
            except LSPRequestError as e:
                if e.code == ERROR_CONTENT_MODIFIED and attempt < MAX_CONTENT_MODIFIED_RETRIES:
                    await asyncio.sleep(RETRY_BASE_DELAY * (2 ** attempt))
                    continue
                raise

    async def _send_notification(self, method: str, params: Any) -> None:
        if not self._connection_is_open():
            raise LSPProtocolError(f"cannot send {method!r}: server connection closed")
        try:
            await self._write(make_notification(method, params))
        except _WRITE_ERRORS as e:
            logger.debug("[%s] notify %s failed: %s", self.server_id, method, e)

    async def _send_reply(self, msg: dict) -> None:
        """Send a response to a server→client request; silently no-ops when the pipe is gone."""
        if self._proc is None or self._proc.stdin is None or self._proc.stdin.is_closing():
            return
        try:
            await self._write(msg)
        except _WRITE_ERRORS:
            pass

    def _dispatch_response(self, req_id: int, msg: dict) -> None:
        fut = self._pending.get(req_id)
        if fut is None or fut.done():
            return
        if "error" in msg:
            err = msg["error"] or {}
            fut.set_exception(
                LSPRequestError(
                    code=int(err.get("code", -32000)),
                    message=str(err.get("message", "unknown")),
                    data=err.get("data"),
                )
            )
        else:
            fut.set_result(msg.get("result"))

    async def _dispatch_request(self, req_id: Any, msg: dict) -> None:
        method = msg.get("method", "")
        params = msg.get("params")
        handler = self._request_handlers.get(method)
        if handler is None:
            await self._send_reply(make_error_response(req_id, ERROR_METHOD_NOT_FOUND, f"method not found: {method}"))
            return
        try:
            result = await handler(params)
        except Exception as e:  # noqa: BLE001 — protocol must not blow up
            logger.warning("[%s] request handler %s failed: %s", self.server_id, method, e)
            await self._send_reply(make_error_response(req_id, -32000, f"handler failed: {e}"))
            return
        await self._send_reply(make_response(req_id, result))

    def _dispatch_notification(self, method: str, msg: dict) -> None:
        handler = self._notification_handlers.get(method)
        if handler is None:
            return
        try:
            handler(msg.get("params"))
        except Exception as e:  # noqa: BLE001
            logger.debug("[%s] notification handler %s failed: %s", self.server_id, method, e)

    # ------------------------------------------------------------------
    # built-in server-→-client request handlers
    # ------------------------------------------------------------------

    async def _handle_work_done_create(self, params: Any) -> Any:
        # Acknowledge progress tokens — required by some servers.
        return None

    async def _handle_workspace_configuration(self, params: Any) -> Any:
        # Walk dotted sections through initializationOptions; null when missing.
        if not isinstance(params, dict):
            return [None]
        out: List[Any] = []
        for item in params.get("items") or []:
            if not isinstance(item, dict):
                out.append(None)
                continue
            section = item.get("section")
            if not section or not self._init_options:
                out.append(self._init_options or None)
                continue
            cur: Any = self._init_options
            for part in str(section).split("."):
                if isinstance(cur, dict) and part in cur:
                    cur = cur[part]
                else:
                    cur = None
                    break
            out.append(cur)
        return out

    async def _handle_register_capability(self, params: Any) -> Any:
        if not isinstance(params, dict):
            return None
        for reg in params.get("registrations") or []:
            if not isinstance(reg, dict):
                continue
            reg_id = reg.get("id")
            if reg.get("method") == "textDocument/diagnostic" and reg_id:
                self._diagnostic_registrations[str(reg_id)] = reg
                self._registration_event.set()
        return None

    async def _handle_unregister_capability(self, params: Any) -> Any:
        if not isinstance(params, dict):
            return None
        for unreg in params.get("unregisterations") or []:
            if not isinstance(unreg, dict):
                continue
            reg_id = unreg.get("id")
            if reg_id:
                self._diagnostic_registrations.pop(str(reg_id), None)
        return None

    async def _handle_workspace_folders(self, params: Any) -> Any:
        return [{"name": "workspace", "uri": file_uri(self.workspace_root)}]

    async def _handle_diagnostic_refresh(self, params: Any) -> Any:
        # We don't honour refresh — we re-pull on every touchFile.
        return None

    def _handle_publish_diagnostics(self, params: Any) -> None:
        if not isinstance(params, dict):
            return
        uri = params.get("uri")
        if not isinstance(uri, str):
            return
        diagnostics = params.get("diagnostics") or []
        if not isinstance(diagnostics, list):
            diagnostics = []
        version = params.get("version")

        doc = self._docs.setdefault(uri_to_path(uri), _DocState(version=-1))
        is_seed = self._seed_first_push and not doc.seed_seen
        doc.seed_seen = True
        doc.push = diagnostics
        if is_seed:
            # First push is baseline data only: it predates any didChange we
            # sent, so it's stored WITHOUT a freshness tag and never satisfies a waiter.
            return
        # Tag with the echoed version when provided; otherwise credit the
        # current version (a push observed after our change describes it or
        # newer).  doc.version is -1 for never-opened paths (relatedDocuments
        # spillover), keeping them unfresh.
        doc.push_version = version if isinstance(version, int) else doc.version
        # Keep the Event sticky-set so in-progress waits resolve; waiters
        # compare ``_push_counter`` to detect a genuinely new push.
        self._push_counter += 1
        self._push_event.set()

    # ------------------------------------------------------------------
    # public file-sync API
    # ------------------------------------------------------------------

    async def open_file(self, path: str, *, language_id: str = "plaintext") -> int:
        """Send didOpen (first time) or didChange (subsequent); return the new document version."""
        if not self.is_running:
            raise LSPProtocolError("client not running")

        abs_path = os.path.abspath(path)
        try:
            text = Path(abs_path).read_text(encoding="utf-8", errors="replace")
        except OSError as e:
            raise LSPProtocolError(f"cannot read {abs_path}: {e}") from e

        uri = file_uri(abs_path)
        doc = self._docs.get(abs_path)

        if doc is not None and doc.version >= 0:
            await self._send_notification(
                "workspace/didChangeWatchedFiles",
                {"changes": [{"uri": uri, "type": 2}]},  # 2 = CHANGED
            )
            new_version = doc.version + 1
            content_changes: List[Dict[str, Any]]
            if self._sync_kind == 2:
                content_changes = [
                    {
                        "range": {
                            "start": {"line": 0, "character": 0},
                            "end": _end_position(doc.text),
                        },
                        "text": text,
                    }
                ]
            else:
                content_changes = [{"text": text}]
            await self._send_notification(
                "textDocument/didChange",
                {
                    "textDocument": {"uri": uri, "version": new_version},
                    "contentChanges": content_changes,
                },
            )
            # Bumping the version is the whole invalidation story (see _DocState).
            doc.version = new_version
            doc.text = text
            return new_version

        await self._send_notification(
            "workspace/didChangeWatchedFiles",
            {"changes": [{"uri": uri, "type": 1}]},  # 1 = CREATED
        )
        # Fresh state: anything a pre-open push stashed under this path
        # (relatedDocuments spillover) is discarded.
        self._docs[abs_path] = _DocState(version=0, text=text)
        await self._send_notification(
            "textDocument/didOpen",
            {
                "textDocument": {
                    "uri": uri,
                    "languageId": language_id,
                    "version": 0,
                    "text": text,
                }
            },
        )
        return 0

    async def save_file(self, path: str) -> None:
        """Send didSave for ``path``.  Some linters re-scan only on save."""
        if not self.is_running:
            return
        await self._send_notification(
            "textDocument/didSave",
            {"textDocument": {"uri": file_uri(os.path.abspath(path))}},
        )

    # ------------------------------------------------------------------
    # diagnostics: pull + wait
    # ------------------------------------------------------------------

    async def _pull_document_diagnostics(self, path: str) -> None:
        """Send ``textDocument/diagnostic`` for one file into the pull store.

        Results are tagged with the version captured at send time, so a
        didChange racing past the request makes them stale automatically.
        Silently no-ops on errors (server may not support pull).
        """
        abs_path = os.path.abspath(path)
        doc = self._docs.get(abs_path)
        sent_version = doc.version if doc else -1
        try:
            result = await self._send_request_with_retry(
                "textDocument/diagnostic",
                {"textDocument": {"uri": file_uri(abs_path)}},
                timeout=DIAGNOSTICS_REQUEST_TIMEOUT,
            )
        except (LSPRequestError, LSPProtocolError, asyncio.TimeoutError) as e:
            logger.debug("[%s] document diagnostic pull failed: %s", self.server_id, e)
            return
        if not isinstance(result, dict):
            return
        items = result.get("items")
        if isinstance(items, list):
            doc = self._docs.setdefault(abs_path, _DocState(version=-1))
            doc.pull = items
            doc.pull_version = sent_version
        related = result.get("relatedDocuments")
        if isinstance(related, dict):
            for uri, sub in related.items():
                if not isinstance(sub, dict):
                    continue
                sub_items = sub.get("items")
                if isinstance(sub_items, list):
                    rel = self._docs.setdefault(uri_to_path(uri), _DocState(version=-1))
                    rel.pull = sub_items
                    # Same send-anchored tagging: fresh only if that doc hasn't changed since.
                    rel.pull_version = rel.version

    async def wait_for_diagnostics(
        self,
        path: str,
        version: int,
        *,
        mode: str = "document",
        timeout: Optional[float] = None,
    ) -> bool:
        """Wait for fresh diagnostics for ``path`` at ``version``.

        ``mode`` is ``"document"`` (5s) or ``"full"`` (10s); ``timeout`` overrides
        the budget (this is how ``lsp.wait_timeout`` reaches the loop).  Returns
        True when fresh data arrived (push at/after our didChange, or a pull
        answered after it), False on timeout.  Callers must treat False as
        "no data", NOT "no errors" — the stores may still hold stale entries.
        Never throws for servers lacking pull support; the push side still works.
        """
        if timeout is not None and timeout > 0:
            budget = timeout
        else:
            budget = DIAGNOSTICS_FULL_WAIT if mode == "full" else DIAGNOSTICS_DOCUMENT_WAIT
        deadline = asyncio.get_event_loop().time() + budget
        abs_path = os.path.abspath(path)

        while True:
            if not self._connection_is_open():
                raise LSPProtocolError(
                    "server connection closed while waiting for diagnostics"
                )
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return False

            # Concurrent: document pull + push wait.
            pull_task = asyncio.create_task(self._pull_document_diagnostics(abs_path))
            push_task = asyncio.create_task(self._wait_for_fresh_push(abs_path, version, remaining))
            done, pending = await asyncio.wait(
                {pull_task, push_task},
                timeout=remaining,
                return_when=asyncio.FIRST_COMPLETED,
            )
            for t in pending:
                t.cancel()
            for t in pending:
                try:
                    await t
                except (asyncio.CancelledError, Exception):  # noqa: BLE001
                    pass

            doc = self._docs.get(abs_path)
            if doc and (doc.fresh_push(version) or doc.fresh_pull(version)):
                return True

    async def _wait_for_fresh_push(self, path: str, version: int, timeout: float) -> None:
        """Wait until a fresh publishDiagnostics arrives for ``path`` at ``version``+."""
        deadline = asyncio.get_event_loop().time() + timeout
        baseline = self._push_counter
        while True:
            doc = self._docs.get(path)
            if doc and doc.fresh_push(version):
                # Debounce: TS often emits in pairs.  Snapshot the counter so
                # we wake on a *new* push, not the one that just satisfied us.
                debounce_baseline = self._push_counter
                debounce_deadline = asyncio.get_event_loop().time() + PUSH_DEBOUNCE
                while self._push_counter == debounce_baseline:
                    remaining = debounce_deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        break
                    self._push_event.clear()
                    try:
                        await asyncio.wait_for(self._push_event.wait(), timeout=remaining)
                    except asyncio.TimeoutError:
                        break
                return
            remaining = deadline - asyncio.get_event_loop().time()
            if remaining <= 0:
                return
            if self._push_counter > baseline:
                # New push but predicate still false — re-check without waiting.
                baseline = self._push_counter
                continue
            self._push_event.clear()
            try:
                await asyncio.wait_for(self._push_event.wait(), timeout=min(remaining, 0.5))
            except asyncio.TimeoutError:
                continue

    def diagnostics_for(self, path: str, *, fresh_only: bool = False) -> List[Dict[str, Any]]:
        """Merged + deduped push/pull diagnostics for one file.

        With ``fresh_only=True`` a store only contributes when its version tag
        has caught up to the document's version — report paths must use this
        so "stale errors" and "no errors" aren't conflated.
        """
        doc = self._docs.get(os.path.abspath(path))
        if doc is None:
            return []
        if fresh_only:
            return _dedupe(
                doc.push if doc.fresh_push() else [],
                doc.pull if doc.fresh_pull() else [],
            )
        return _dedupe(doc.push, doc.pull)


def _dedupe(*lists: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[str] = set()
    out: List[Dict[str, Any]] = []
    for lst in lists:
        for d in lst:
            if not isinstance(d, dict):
                continue
            key = _diagnostic_key(d)
            if key in seen:
                continue
            seen.add(key)
            out.append(d)
    return out


def _diagnostic_key(d: Dict[str, Any]) -> str:
    """Content-equality key: severity + code + source + message + range.

    Shared with the manager's cross-edit delta filter (as ``_diag_key``) so
    both layers agree on diagnostic identity.  The range is included so an
    identical error introduced at a second site still surfaces as new; the
    manager line-shifts its baseline into post-edit coordinates before keying.
    """
    rng = d.get("range") or {}
    start = rng.get("start") or {}
    end = rng.get("end") or {}
    code = d.get("code")
    if code is not None and not isinstance(code, str):
        code = str(code)
    return "\x00".join(
        [
            str(d.get("severity") or 1),
            str(code or ""),
            str(d.get("source") or ""),
            str(d.get("message") or "").strip(),
            f"{start.get('line', 0)}:{start.get('character', 0)}-{end.get('line', 0)}:{end.get('character', 0)}",
        ]
    )


__all__ = [
    "LSPClient",
    "file_uri",
    "uri_to_path",
    "INITIALIZE_TIMEOUT",
    "DIAGNOSTICS_DOCUMENT_WAIT",
    "DIAGNOSTICS_FULL_WAIT",
]
