"""Transport abstraction for the tui_gateway JSON-RPC server.

A :class:`Transport` accepts a JSON-serialisable dict and forwards it to its peer, so the same
dispatcher runs over stdio (``tui_gateway.entry``) or WebSocket (``tui_gateway.ws``). The active
transport for the current request lives in a ``ContextVar`` so handlers dispatched onto the worker
pool route writes to the right peer. ``server.write_json`` works with nothing bound: it falls back
to the module-level :class:`StdioTransport`, which resolves ``_real_stdout`` lazily through a
callback so tests that monkey-patch ``server._real_stdout`` keep working.
"""

from __future__ import annotations

import contextvars
import errno
import json
import logging
import os
import threading
from typing import Any, Callable, Optional, Protocol, runtime_checkable

# Errno values that mean "the peer is gone" rather than "the host has a real I/O problem". Anything
# outside this set re-raises so it surfaces in the crash log instead of looking like a clean disconnect.
_PEER_GONE_ERRNOS = frozenset({
    errno.EPIPE, errno.ECONNRESET, errno.EBADF, errno.ESHUTDOWN,
    getattr(errno, "WSAECONNRESET", -1), getattr(errno, "WSAESHUTDOWN", -1),  # win32 (no-op on POSIX)
} - {-1})

logger = logging.getLogger(__name__)

# When true, StdioTransport skips ``stream.flush`` after writing: on a half-closed pipe (TUI Node
# parent quit while the gateway still emits) flush can block long enough to starve the worker pool.
# Python text stdout is fully buffered on a pipe, so this ONLY makes sense with ``-u`` /
# ``PYTHONUNBUFFERED=1``; otherwise frames accumulate and the TUI hangs waiting for ``gateway.ready``.
_DISABLE_FLUSH = (os.environ.get("HERMES_TUI_GATEWAY_NO_FLUSH", "") or "").strip().lower() in {"1", "true", "yes", "on"}


@runtime_checkable
class Transport(Protocol):
    """Minimal interface every transport implements."""

    def write(self, obj: dict) -> bool:
        """Emit one JSON frame. Return ``False`` when the peer is gone."""

    def close(self) -> None:
        """Release any resources owned by this transport."""


_current_transport: contextvars.ContextVar[Optional[Transport]] = contextvars.ContextVar(
    "hermes_gateway_transport", default=None
)


def current_transport() -> Optional[Transport]:
    """Return the transport bound for the current request, if any."""
    return _current_transport.get()


def bind_transport(transport: Optional[Transport]):
    """Bind *transport* for the current context. Returns a token for :func:`reset_transport`."""
    return _current_transport.set(transport)


def reset_transport(token) -> None:
    """Restore the transport binding captured by :func:`bind_transport`."""
    _current_transport.reset(token)


def _peer_gone(exc: Exception, what: str) -> bool:
    """True when *exc* from a stream write/flush means the peer is gone; re-raise anything else.

    ``False`` from :meth:`StdioTransport.write` is the dispatcher's "broken stdout pipe" signal
    (``entry.py`` exits cleanly on it), so programming errors and real host I/O bugs (non-JSON-safe
    payloads, UnicodeEncodeError from a misconfigured locale, ENOSPC, EACCES, ...) MUST re-raise so
    the crash log records them instead of masquerading as a clean disconnect. Peer-gone:
    ``BrokenPipeError``, ``ValueError("...closed file...")``, ``OSError`` with errno in
    :data:`_PEER_GONE_ERRNOS`.
    """
    if isinstance(exc, BrokenPipeError):
        return True
    if isinstance(exc, ValueError):
        if isinstance(exc, UnicodeEncodeError) or "closed file" not in str(exc):
            raise exc
        return True
    if isinstance(exc, OSError):
        if exc.errno not in _PEER_GONE_ERRNOS:
            raise exc
        logger.debug("StdioTransport %s peer gone: %s", what, exc)
        return True
    raise exc


class StdioTransport:
    """Writes JSON frames to a stream (usually ``sys.stdout``), resolved via a callable so runtime
    monkey-patches of the underlying stream keep working."""

    __slots__ = ("_stream_getter", "_lock")

    def __init__(self, stream_getter: Callable[[], Any], lock: threading.Lock) -> None:
        self._stream_getter = stream_getter
        self._lock = lock

    def write(self, obj: dict) -> bool:
        """Return ``True`` on success, ``False`` ONLY when the peer is gone (see :func:`_peer_gone`)."""
        # Serialization is OUTSIDE the lock so a large payload can't block other threads emitting
        # their own frames. A non-JSON-safe payload is a programming error: re-raise.
        line = json.dumps(obj, ensure_ascii=False) + "\n"
        with self._lock:
            stream = self._stream_getter()
            try:
                stream.write(line)
            except Exception as e:
                if _peer_gone(e, "write"):
                    return False
            # A flush that *raises* with a peer-gone errno means the dispatcher should exit cleanly.
            # A flush that *hangs* on a half-closed pipe holds the lock until it returns — see
            # ``_DISABLE_FLUSH`` for the "skip flush entirely" escape hatch.
            if not _DISABLE_FLUSH:
                try:
                    stream.flush()
                except Exception as e:
                    if _peer_gone(e, "flush"):
                        return False
        return True

    def close(self) -> None:
        return None


class TeeTransport:
    """Mirrors writes to one primary plus N best-effort secondaries.

    The primary's return value (and exceptions) determine the result — secondaries swallow failures
    so a wedged sidecar never stalls the main IO path. Used by the PTY child so every dispatcher
    emit lands on stdio (Ink) AND on a back-WS feeding the dashboard sidebar.
    """

    __slots__ = ("_primary", "_secondaries")

    def __init__(self, primary: "Transport", *secondaries: "Transport") -> None:
        self._primary = primary
        self._secondaries = secondaries

    def write(self, obj: dict) -> bool:
        # Primary first so a slow sidecar (WS publisher) never delays Ink/stdio.
        ok = self._primary.write(obj)
        for sec in self._secondaries:
            try:
                sec.write(obj)
            except Exception:
                pass
        return ok

    def close(self) -> None:
        try:
            self._primary.close()
        finally:
            for sec in self._secondaries:
                try:
                    sec.close()
                except Exception:
                    pass
