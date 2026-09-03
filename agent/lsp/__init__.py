"""Language Server Protocol (LSP) integration for Hermes Agent.

Hermes runs real language servers (pyright, gopls, rust-analyzer, ...) as
subprocesses and pipes their ``textDocument/publishDiagnostics`` output into
the post-write lint delta filter used by ``write_file`` and ``patch``.  LSP is
**gated on git workspace detection**: outside a git repository the
file_operations layer falls back to its in-process syntax checks, so users
on user-home cwd's (e.g. Telegram gateway chats) never spawn daemons.
``get_service()`` returns the singleton :class:`LSPService` (or ``None`` when
disabled); wiring: :func:`tools.file_operations.FileOperations._check_lint_delta`.
"""
from __future__ import annotations

import atexit
import logging
import threading
from typing import Optional

from agent.lsp.manager import LSPService

logger = logging.getLogger("agent.lsp")

_service: Optional[LSPService] = None
_atexit_registered = False
_service_lock = threading.Lock()


def _active(svc: Optional[LSPService]) -> Optional[LSPService]:
    return svc if (svc is not None and svc.is_active()) else None


def get_service() -> Optional[LSPService]:
    """Return the process-wide LSP service singleton, or None when disabled.

    Created lazily on first call.  Also registers an :mod:`atexit` hook so a
    clean exit tears down spawned language servers: without it every
    ``hermes chat`` exit leaks pyright processes for a few seconds while
    their stdout buffers drain.  (SIGKILL/os._exit skip atexit — fine, the
    kernel reaps the stateless servers with their parent.)
    """
    global _service, _atexit_registered
    if _service is None:
        with _service_lock:
            if _service is None:
                _service = LSPService.create_from_config()
                if not _atexit_registered:
                    atexit.register(_atexit_shutdown)
                    _atexit_registered = True
    return _active(_service)


def shutdown_service() -> None:
    """Tear down the LSP service if one was started.  Idempotent."""
    global _service
    with _service_lock:
        svc, _service = _service, None
    if svc is not None:
        try:
            svc.shutdown()
        except Exception as e:  # noqa: BLE001
            logger.debug("LSP shutdown error: %s", e)


def _atexit_shutdown() -> None:
    """atexit wrapper; logs at debug since the user has already seen the final output."""
    try:
        shutdown_service()
    except Exception as e:  # noqa: BLE001
        logger.debug("atexit LSP shutdown failed: %s", e)


__all__ = ["get_service", "shutdown_service", "LSPService"]
