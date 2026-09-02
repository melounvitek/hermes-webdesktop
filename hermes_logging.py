"""Centralized logging setup for Hermes Agent.

Log files produced: agent.log — INFO+, all agent/tool/session activity (the main log) errors.log —
WARNING+, errors and warnings only (quick triage) gateway.log — INFO+, gateway-only events (created
when mode="gateway") gui.log — INFO+, dashboard/websocket/TUI-gateway events (created when
mode="gui")

All files use ``RotatingFileHandler`` with ``RedactingFormatter`` so secrets are never written to
disk.
"""

import atexit
import copy
import io
import logging
import os
import queue
import sys
import threading
from logging.handlers import QueueHandler, QueueListener
from pathlib import Path
from typing import Optional, Sequence

# On Windows, stdlib ``RotatingFileHandler`` calls ``os.rename()`` in
# ``doRollover()`` and fails with ``PermissionError [WinError 32]`` whenever
# another process holds an append-mode handle on ``agent.log`` — which is
# essentially always in Hermes (TUI, gateway, ``hy_memory`` server, MCP
# servers, and on-demand CLI commands all log from separate processes),
# pinning ``agent.log`` at the 5 MiB threshold and spamming stderr with
# a traceback on every emit. ``concurrent-log-handler`` wraps the rename in a
# cross-process file lock (via ``portalocker``: pywin32 on Windows) so only
# one process rotates at a time and the others wait their turn.
#
# This swap is Windows-ONLY and deliberately so:
#   * The bug (WinError 32 on rename-while-open) is specific to Windows file
#     locking semantics — POSIX renames an open file fine, so stdlib already
#     works correctly on Linux/macOS.
#   * On POSIX, managed-mode (NixOS) relies on the exact ``_open()`` /
#     ``doRollover()`` lifecycle of stdlib ``RotatingFileHandler`` (the
#     ``_ManagedRotatingFileHandler`` subclass chmods 0660 after each). CLH
#     opens lazily and rotates differently, which breaks the group-writable
#     guarantee and the eager file-creation those paths depend on.
# Aliasing keeps every existing ``RotatingFileHandler`` reference in this
# module (class declaration, ``isinstance`` checks, docstring) working
# unchanged. See #44873.
if sys.platform == "win32":
    from concurrent_log_handler import (  # noqa: E402
        ConcurrentRotatingFileHandler as RotatingFileHandler,
    )
else:
    from logging.handlers import RotatingFileHandler  # noqa: E402


from hermes_constants import get_config_path, get_hermes_home, mkdir_under_hermes_home

# Sentinel to track whether setup_logging() has already run.  The function
# is idempotent — calling it twice is safe but the second call is a no-op
# unless ``force=True``.
_logging_initialized = False

# Thread-local storage for per-conversation session context.
_session_context = threading.local()

# Default log format — includes timestamp, level, optional session tag,
# logger name, and message.  The ``%(session_tag)s`` field is guaranteed to
# exist on every LogRecord via _install_session_record_factory() below.
_LOG_FORMAT = "%(asctime)s %(levelname)s%(session_tag)s %(name)s: %(message)s"
_LOG_FORMAT_VERBOSE = "%(asctime)s - %(name)s - %(levelname)s%(session_tag)s - %(message)s"


def _safe_stderr():  # type: ignore[return]
    """Return a stderr stream that tolerates Unicode on all platforms.

    We wrap ``sys.stderr`` in a ``TextIOWrapper`` with ``errors='replace'`` so log lines are never
    lost — un-encodable characters are replaced with ``?`` instead of crashing the process.
    """
    stream = sys.stderr
    encoding = getattr(stream, "encoding", None) or "utf-8"
    # Already UTF-8 or surrogate-aware — no wrapping needed.
    if encoding.lower().replace("-", "") in ("utf8", "utf8surrogateescape"):
        return stream
    try:
        wrapped = io.TextIOWrapper(stream.buffer, encoding="utf-8", errors="replace", line_buffering=True)
        # Prevent the wrapper from closing the underlying buffer when it is garbage-collected.
        wrapped.close = lambda: None  # type: ignore[assignment]
        return wrapped
    except Exception:
        return stream  # best-effort: no buffer / wrapping failed -> original stream


def _is_windows_concurrent_log_lock_timeout(exc: BaseException | None) -> bool:
    """Return True for concurrent-log-handler's Windows lock timeout.

    On Windows Desktop, slash-command workers and the gateway can all write to the same rotating log
    files. ``concurrent-log-handler`` serializes rollover with a cross-process lock, but when
    another process holds that lock too long it raises this RuntimeError. Logging failures should
    not escape into Desktop chat output.
    """
    return (
        sys.platform == "win32"
        and isinstance(exc, RuntimeError)
        and "Cannot acquire lock after 20 attempts" in str(exc)
    )


# Third-party loggers that are noisy at DEBUG/INFO level.
_NOISY_LOGGERS = (
    "openai", "openai._base_client", "httpx", "httpcore", "asyncio", "hpack", "hpack.hpack",
    "grpc", "modal", "urllib3", "urllib3.connectionpool", "websockets", "charset_normalizer",
    "markdown_it",
)


def _quiet_noisy_loggers() -> None:
    """Pin noisy third-party loggers at WARNING."""
    for name in _NOISY_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)


# Public session context API

def set_session_context(session_id: str) -> None:
    """Set the session ID for the current thread."""
    _session_context.session_id = session_id


def clear_session_context() -> None:
    """Clear the session ID for the current thread."""
    _session_context.session_id = None


# Record factory — injects session_tag into every LogRecord at creation

def _install_session_record_factory() -> None:
    """Replace the global LogRecord factory with one that adds ``session_tag``.

    Unlike a handler/logger ``Filter``, the record factory runs for EVERY record in the process,
    including propagated and third-party-handled ones, so ``%(session_tag)s`` is always available
    and never KeyErrors. Idempotent: a marker attribute prevents double-wrapping on reload.
    """
    current_factory = logging.getLogRecordFactory()
    if getattr(current_factory, "_hermes_session_injector", False):
        return  # already installed

    def _session_record_factory(*args, **kwargs):
        record = current_factory(*args, **kwargs)
        sid = getattr(_session_context, "session_id", None)
        record.session_tag = f" [{sid}]" if sid else ""  # type: ignore[attr-defined]
        # QueueListener formats records on its own thread, after the
        # profile-scoped ContextVar has gone out of scope. Keep the resolved
        # home on the record so a multiplex desktop ticker can route the log
        # to the job owner's files (#97489).
        try:
            record.hermes_home = str(get_hermes_home().resolve())  # type: ignore[attr-defined]
        except Exception:
            record.hermes_home = ""  # type: ignore[attr-defined]
        return record

    _session_record_factory._hermes_session_injector = True  # type: ignore[attr-defined]
    logging.setLogRecordFactory(_session_record_factory)


# Install immediately on import — session_tag is available on all records
# from this point forward, even before setup_logging() is called.
_install_session_record_factory()


# Filters

class _ComponentFilter(logging.Filter):
    """Only pass records whose logger name starts with one of *prefixes*."""

    def __init__(self, prefixes: Sequence[str]) -> None:
        super().__init__()
        self._prefixes = tuple(prefixes)

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name.startswith(self._prefixes)


# Logger name prefixes that belong to each component.
# Used by _ComponentFilter and exposed for ``hermes logs --component``.
COMPONENT_PREFIXES = {
    # ``plugins.platforms`` covers messaging-platform adapters that migrated
    # out of ``gateway/platforms/`` into bundled plugins (#41112) — they are
    # still gateway components and their logs belong in gateway.log / match
    # ``hermes logs --component gateway``.
    "gateway": ("gateway", "hermes_plugins", "plugins.platforms"),
    "agent": ("agent", "run_agent", "model_tools", "batch_runner"),
    "tools": ("tools",),
    "cli": ("hermes_cli", "cli"),
    "cron": ("cron",),
    "gui": ("hermes_cli.web_server", "hermes_cli.pty_bridge", "tui_gateway", "uvicorn"),
}


# Main setup

def setup_logging(
    *,
    hermes_home: Optional[Path] = None,
    log_level: Optional[str] = None,
    max_size_mb: Optional[int] = None,
    backup_count: Optional[int] = None,
    mode: Optional[str] = None,
    force: bool = False,
) -> Path:
    """Configure the Hermes logging subsystem.

    Safe to call multiple times; the second call is a no-op unless *force* is ``True``. Level and
    rotation defaults come from config.yaml ``logging.*``. ``mode="gateway"`` adds ``gateway.log``
    (gateway components only) and ``mode="gui"`` adds ``gui.log`` (dashboard / TUI-gateway).
    Returns the ``logs/`` directory.
    """
    global _logging_initialized
    home = hermes_home or get_hermes_home()
    log_dir = mkdir_under_hermes_home(home / "logs")

    # Read config defaults (best-effort — config may not be loaded yet).
    cfg_level, cfg_max_size, cfg_backup = _read_logging_config()

    level_name = (log_level or cfg_level or "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    max_bytes = (max_size_mb or cfg_max_size or 5) * 1024 * 1024
    backups = backup_count or cfg_backup or 3

    # Lazy import to avoid circular dependency at module load time.
    from agent.redact import RedactingFormatter

    root = logging.getLogger()

    # (filename, level, max_bytes, backup_count, component) — component gates on ``mode`` and
    # restricts the file to that component's logger prefixes.
    handler_specs = (
        ("agent.log", level, max_bytes, backups, None),
        ("errors.log", logging.WARNING, 2 * 1024 * 1024, 2, None),
        ("gateway.log", logging.INFO, 5 * 1024 * 1024, 3, "gateway"),
        ("gui.log", logging.INFO, 10 * 1024 * 1024, 5, "gui"),
    )
    for filename, lvl, size, count, component in handler_specs:
        if component is not None and mode != component:
            continue
        _add_rotating_handler(
            log_dir / filename, level=lvl, max_bytes=size, backup_count=count,
            formatter=RedactingFormatter(_LOG_FORMAT),
            log_filter=_ComponentFilter(COMPONENT_PREFIXES[component]) if component else None,
        )

    if _logging_initialized and not force:
        return log_dir

    # Ensure root logger level is low enough for the handlers to fire.
    if root.level == logging.NOTSET or root.level > level:
        root.setLevel(level)

    _quiet_noisy_loggers()

    _logging_initialized = True
    return log_dir


def setup_verbose_logging() -> None:
    """Enable DEBUG-level console logging for ``--verbose`` / ``-v`` mode."""
    from agent.redact import RedactingFormatter

    root = logging.getLogger()

    # Avoid adding duplicate stream handlers.
    if any(getattr(h, "_hermes_verbose", False) for h in root.handlers):
        return

    handler = logging.StreamHandler(_safe_stderr())
    handler.setLevel(logging.DEBUG)
    handler.setFormatter(RedactingFormatter(_LOG_FORMAT_VERBOSE, datefmt="%H:%M:%S"))
    handler._hermes_verbose = True  # type: ignore[attr-defined]
    root.addHandler(handler)

    # Lower root logger level so DEBUG records reach all handlers.
    if root.level > logging.DEBUG:
        root.setLevel(logging.DEBUG)

    _quiet_noisy_loggers()
    # rex-deploy at INFO for sandbox status.
    logging.getLogger("rex-deploy").setLevel(logging.INFO)


# Internal helpers

def _quietly(fn) -> None:
    """Call *fn* (a ``close``/``stop`` bound method) swallowing errors — teardown must never raise."""
    try:
        fn()
    except Exception:
        pass


class _ManagedRotatingFileHandler(RotatingFileHandler):
    """RotatingFileHandler that ensures group-writable perms in managed mode

    In managed mode (NixOS) the setgid stateDir needs group-readable files, but ``_open()`` and
    ``doRollover()`` honor the umask (0644), so ``chmod 0660`` is applied after both. Also, a
    rotating handler holds an fd: if the file is rotated externally (logrotate, ``mv``) writes
    silently go to the old inode, so before each emit the path's inode is compared to the open
    stream's and the file reopened on mismatch (the ``WatchedFileHandler`` pattern).
    """

    def __init__(self, *args, **kwargs):
        from hermes_cli.config import is_managed
        self._managed = is_managed()
        super().__init__(*args, **kwargs)
        # Snapshot the inode of the currently open stream so emit() can
        # detect external rotation without an extra fstat per write.
        self._record_stream_stat()

    def _chmod_if_managed(self):
        if self._managed:
            try:
                os.chmod(self.baseFilename, 0o660)
            except OSError:
                pass

    def _record_stream_stat(self, st: Optional[os.stat_result] = None) -> None:
        """Snapshot dev/ino of ``baseFilename`` so we can detect external rotation."""
        try:
            st = st or os.stat(self.baseFilename)
            self._stat_dev, self._stat_ino = st.st_dev, st.st_ino
        except OSError:
            self._stat_dev, self._stat_ino = None, None

    def _reopen_stream(self, stat_result=None) -> None:
        """Close the current stream and open ``baseFilename`` afresh (best-effort).

        On failure the stream is left ``None`` so the next emit bails rather than writing to a
        stale inode.
        """
        if self.stream is not None:
            _quietly(self.stream.close)
        self.stream = None  # type: ignore[assignment]
        try:
            self.stream = self._open()
        except Exception:
            return
        self._record_stream_stat(stat_result)

    def _reopen_if_externally_rotated(self) -> None:
        """Reopen the stream when ``baseFilename`` no longer matches our fd.

        Triggered when ``baseFilename`` was renamed (logrotate), unlinked, or replaced by a
        different inode. Silent + best-effort: any error falls back to the existing (possibly stale)
        stream so logging keeps working instead of dying on a stat failure.
        """
        try:
            st = os.stat(self.baseFilename)
        except FileNotFoundError:
            # File was rotated/unlinked underneath us: reopen so a fresh inode
            # is created at the expected path.
            self._reopen_stream()
            return
        except OSError:
            return  # transient — try again on the next emit

        if self._stat_dev is None or self._stat_ino is None:
            self._record_stream_stat(st)
        elif (st.st_dev, st.st_ino) != (self._stat_dev, self._stat_ino):
            # baseFilename now points at a DIFFERENT inode than the one we hold open.
            self._reopen_stream(st)

    def emit(self, record: logging.LogRecord) -> None:
        # Cheap-ish stat-per-record check; the kernel caches inode metadata
        # so the syscall is sub-microsecond on a hot file.
        if self.stream is not None or os.path.exists(self.baseFilename):
            self._reopen_if_externally_rotated()
        super().emit(record)

    def handleError(self, record: logging.LogRecord) -> None:
        """Suppress the known Windows ``concurrent-log-handler`` lock timeout

        CLH's ``emit()`` catches the ``"Cannot acquire lock after N attempts"`` RuntimeError and
        routes it here, so this override is the single point to silence it before stdlib prints to
        stderr (which the Desktop slash-worker captures and surfaces into chat output).
        """
        if not _is_windows_concurrent_log_lock_timeout(sys.exc_info()[1]):
            super().handleError(record)

    def _open(self):
        stream = super()._open()
        self._chmod_if_managed()
        return stream

    def doRollover(self):
        super().doRollover()
        self._chmod_if_managed()
        # Our own rollover writes a new baseFilename; refresh the snapshot
        # so the next emit doesn't mistake it for external rotation.
        self._record_stream_stat()


def _new_file_handler(
    path: Path, *, level: int, max_bytes: int, backup_count: int, formatter
) -> "_ManagedRotatingFileHandler":
    """Create the ``logs/`` directory and a configured ``_ManagedRotatingFileHandler``."""
    mkdir_under_hermes_home(path.parent)
    handler = _ManagedRotatingFileHandler(
        str(path), maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
    )
    handler.setLevel(level)
    handler.setFormatter(formatter)
    return handler


class _ProfileRoutingFileHandler(logging.Handler):
    """Route queued records to the log file for their Hermes home.

    The handler itself is used only behind the existing QueueListener, so its small routing lock
    never blocks an agent or dashboard event loop. The underlying handlers retain the existing
    rotation, redaction, and managed permission behavior.
    """

    def __init__(self, existing: RotatingFileHandler, profile_homes: Sequence[Path]) -> None:
        """Take over *existing*'s path, level, rotation, formatter and filters."""
        super().__init__(level=existing.level)
        resolved = Path(existing.baseFilename).resolve()
        self.baseFilename = str(resolved)
        self._hermes_routed_log_path = resolved
        self._default_home = resolved.parent.parent.resolve()
        self._profile_homes = {Path(home).expanduser().resolve() for home in profile_homes}
        self._filename = resolved.name
        self._max_bytes = getattr(existing, "maxBytes", 0)
        self._backup_count = getattr(existing, "backupCount", 0)
        self._profile_handlers: dict[Path, _ManagedRotatingFileHandler] = {}
        self._profile_handlers_lock = threading.RLock()
        self.setFormatter(existing.formatter)
        for log_filter in existing.filters:
            self.addFilter(log_filter)

    def _home_for_record(self, record: logging.LogRecord) -> Path:
        raw_home = getattr(record, "hermes_home", "")
        try:
            candidate = Path(raw_home).expanduser().resolve()
        except (TypeError, ValueError, OSError):
            candidate = self._default_home
        return candidate if candidate in self._profile_homes else self._default_home

    def _handler_for_home(self, home: Path) -> _ManagedRotatingFileHandler:
        with self._profile_handlers_lock:
            if home not in self._profile_handlers:
                self._profile_handlers[home] = _new_file_handler(
                    home / "logs" / self._filename, level=self.level, max_bytes=self._max_bytes,
                    backup_count=self._backup_count, formatter=self.formatter,
                )
            return self._profile_handlers[home]

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._handler_for_home(self._home_for_record(record)).handle(record)
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        with self._profile_handlers_lock:
            handlers = list(self._profile_handlers.values())
            self._profile_handlers.clear()
        for handler in handlers:
            _quietly(handler.close)
        super().close()


# Asynchronous file logging — keep the cross-process rotation lock off the loop
#
# The rotating file handlers serialize rollover with a cross-process lock (see
# the module header): when several Hermes processes log to the same file, an
# ``emit`` can block while another process holds that lock.  When the emitting
# thread is an asyncio event loop, that block stalls the loop and drops
# WebSocket clients.  To keep file I/O off the hot path, every file handler is
# driven by a single ``QueueListener`` on a dedicated thread; loggers only touch
# an in-memory queue (a non-blocking enqueue).

_log_queue: "Optional[queue.SimpleQueue]" = None
_queue_listener: Optional[QueueListener] = None
_queued_file_handlers: list = []
_queue_atexit_registered = False
# Guards every read-modify-write of the four globals above. setup_logging()
# holds no lock and its _logging_initialized guard runs AFTER handler
# registration, so _register_queued_handler() can run concurrently with a
# flush/reset from another thread (gateway init racing a plugin/CLI path).
# Without this, two threads can interleave listener.stop()/reassign/start()
# and leave the queue with two live listeners or an orphaned worker thread.
_queue_state_lock = threading.Lock()


class _NonFormattingQueueHandler(QueueHandler):
    """``QueueHandler`` for an in-process queue.

    Stdlib ``prepare()`` formats and strips ``args``/``exc_info`` for pickling across processes;
    our queue is in-process, so the target handlers get an unformatted record and apply their own
    ``RedactingFormatter`` on the listener thread. A shallow copy is returned because the emitting
    thread's synchronous handlers may mutate ``record.message`` while the listener reads it.
    """

    def prepare(self, record: logging.LogRecord) -> logging.LogRecord:
        return copy.copy(record)


def _stop_queue_listener() -> None:
    """Flush and stop the background log listener (idempotent, thread-safe).

    This is the atexit hook, so it must acquire the state lock itself.
    """
    global _queue_listener
    with _queue_state_lock:
        listener, _queue_listener = _queue_listener, None
        if listener is not None:
            _quietly(listener.stop)


def _start_queue_listener_locked() -> None:
    """(Re)build + start a listener over the current handler set (``_queue_state_lock`` held).

    A running listener is stopped first; this only happens while handlers are being added
    (queue empty), so ``stop()`` returns immediately.
    """
    global _queue_listener
    if _queue_listener is not None:
        _queue_listener.stop()
    _queue_listener = QueueListener(_log_queue, *_queued_file_handlers, respect_handler_level=True)
    _queue_listener.start()


def _register_queued_handler(handler: logging.Handler) -> None:
    """Route *handler* through the shared async queue instead of attaching it to *root* directly, so
    emitting threads never block on file I/O or the cross-process rotation lock. The
    ``QueueListener`` applies each handler's own level and filters on its worker thread.
    """
    global _log_queue, _queue_atexit_registered
    with _queue_state_lock:
        if _log_queue is None:
            _log_queue = queue.SimpleQueue()
            qh = _NonFormattingQueueHandler(_log_queue)
            qh._hermes_queue = True  # type: ignore[attr-defined]
            # Always funnel through the root logger so records from any logger
            # (production passes root here; callers may pass a child) reach the
            # queue via propagation.
            logging.getLogger().addHandler(qh)
        _queued_file_handlers.append(handler)
        _start_queue_listener_locked()
        if not _queue_atexit_registered:
            # Runs before logging.shutdown (registered earlier at import time),
            # so the listener stops before its file handlers are closed.
            atexit.register(_stop_queue_listener)
            _queue_atexit_registered = True


def flush_log_queue() -> None:
    """Block until all queued records have been written, then resume.

    Draining is done by stopping the listener (which processes every pending record before joining)
    and restarting it. Used by tests that read a log file right after emitting to it.

    NOTE: ``stop()`` joins the worker thread, so this blocks until the queue is empty. Do NOT call
    this on a hard-exit path where the listener may be wedged on the rotation lock — use
    ``drain_log_queue()`` there instead, which bounds the wait.
    """
    with _queue_state_lock:
        listener = _queue_listener
        if listener is not None:
            listener.stop()
            listener.start()


def drain_log_queue(timeout: float = 1.0) -> None:
    """Best-effort, time-bounded drain for hard-exit paths (no restart).

    Unlike ``flush_log_queue()``, this stops the listener WITHOUT restarting it (the process is
    about to exit) and bounds the drain: if the listener's worker thread is wedged on the cross-
    process rotation lock — the very failure this async-logging change exists to survive — an
    unbounded ``stop()``/join would re-freeze the shutdown path.
    """
    listener = _queue_listener
    if listener is None:
        return
    t = threading.Thread(target=lambda: _quietly(listener.stop), name="hermes-log-drain", daemon=True)
    t.start()
    t.join(timeout)


def enable_profile_log_routing(profile_homes: Sequence[str | Path]) -> bool:
    """Make the queued file logs follow a desktop profile context.

    ``setup_logging`` normally binds handlers to one process home. The desktop dashboard is the
    exception: its embedded cron ticker may run jobs for every profile. Replace the existing static
    file handlers with profile routers after that profile list is known.

    Returns ``True`` when routing is enabled or was already enabled. A single-profile caller is left
    untouched because its existing handlers are already correctly scoped.
    """
    global _queue_listener
    homes: list[Path] = []
    for entry in profile_homes:
        try:
            resolved = Path(entry[1] if isinstance(entry, tuple) else entry).expanduser().resolve()
        except (TypeError, ValueError, OSError):
            continue
        if resolved not in homes:
            homes.append(resolved)
    if len(homes) < 2:
        return False

    with _queue_state_lock:
        if not _queued_file_handlers:
            return False
        if any(isinstance(h, _ProfileRoutingFileHandler) for h in _queued_file_handlers):
            return True

        listener = _queue_listener
        if listener is not None:
            listener.stop()
            _queue_listener = None

        replacement = []
        for existing in _queued_file_handlers:
            if isinstance(existing, RotatingFileHandler):
                replacement.append(_ProfileRoutingFileHandler(existing, homes))
                _quietly(existing.close)
            else:
                replacement.append(existing)

        _queued_file_handlers[:] = replacement
        if listener is not None:
            _start_queue_listener_locked()
        return True


def _reset_queued_handlers() -> None:
    """Tear down the async logging queue + listener (test-isolation helper)."""
    global _log_queue
    _stop_queue_listener()
    with _queue_state_lock:
        root = logging.getLogger()
        for h in list(root.handlers):
            if getattr(h, "_hermes_queue", False):
                root.removeHandler(h)
        for h in list(_queued_file_handlers):
            _quietly(h.close)
        _queued_file_handlers.clear()
        _log_queue = None


def _add_rotating_handler(
    path: Path,
    *,
    level: int,
    max_bytes: int,
    backup_count: int,
    formatter: logging.Formatter,
    log_filter: Optional[logging.Filter] = None,
) -> None:
    """Register a queued ``RotatingFileHandler`` for *path*, skipping if one already exists for the
    same resolved file path (idempotent).
    """
    resolved = path.resolve()
    for existing in _queued_file_handlers:
        # Already attached directly, or already covered by the profile router.
        if getattr(existing, "_hermes_routed_log_path", None) == resolved or (
            isinstance(existing, RotatingFileHandler)
            and Path(getattr(existing, "baseFilename", "")).resolve() == resolved
        ):
            return

    handler = _new_file_handler(
        path, level=level, max_bytes=max_bytes, backup_count=backup_count, formatter=formatter,
    )
    if log_filter is not None:
        handler.addFilter(log_filter)
    # Route through the async queue instead of ``logger.addHandler(handler)`` so
    # the rotation-lock wait never runs on the caller's (often event-loop) thread.
    _register_queued_handler(handler)


def _read_logging_config():
    """Best-effort read of ``logging.*`` from config.yaml."""
    try:
        # Prefer the shared (mtime, size)-keyed raw-config cache so this read
        # reuses the parse hermes_cli.main's early bridge already did (one
        # config.yaml parse per process instead of 3-4). Fall back to a
        # direct parse when hermes_cli.config isn't importable (bare
        # hermes_logging consumers).
        try:
            from hermes_cli.config import read_raw_config as _rrc
            cfg = _rrc() or {}
        except Exception:
            from utils import fast_safe_load
            config_path = get_config_path()
            cfg = {}
            if config_path.exists():
                with open(config_path, "r", encoding="utf-8") as f:
                    cfg = fast_safe_load(f) or {}
        if not cfg:
            return (None, None, None)
        # Managed scope: an administrator can pin logging.* too. Overlay via
        # the shared helper (fail-open) since this reads config.yaml directly.
        try:
            from hermes_cli import managed_scope
            cfg = managed_scope.apply_managed_overlay(cfg)
        except Exception:
            pass
        log_cfg = cfg.get("logging", {})
        if isinstance(log_cfg, dict):
            return (log_cfg.get("level"), log_cfg.get("max_size_mb"), log_cfg.get("backup_count"))
    except Exception:
        pass
    return (None, None, None)
