"""Per-thread interrupt signaling for all tools.

Thread-scoped interrupt tracking so that interrupting one agent session does
not kill tools running in other sessions — critical in the gateway where
multiple agents run concurrently in one process.  The agent stores its
execution thread ID at the start of run_conversation() and passes it to
set_interrupt(); tools call is_interrupted(), which checks the CURRENT thread.

Usage in tools:
    from tools.interrupt import is_interrupted
    if is_interrupted():
        return {"output": "[interrupted]", "returncode": 130}
"""

import logging
import os
import threading

logger = logging.getLogger(__name__)

# Opt-in debug tracing — pairs with HERMES_DEBUG_INTERRUPT in
# tools/environments/base.py.  Logs caller thread, target thread, and current
# state per set/check for "interrupt signaled but tool never saw it" reports.
_DEBUG_INTERRUPT = bool(os.getenv("HERMES_DEBUG_INTERRUPT"))

if _DEBUG_INTERRUPT:
    # AIAgent's quiet_mode path forces the `tools` logger to ERROR on CLI
    # startup; force ours back to INFO so the trace is visible in agent.log.
    logger.setLevel(logging.INFO)

# Interrupted thread idents, plus an optional user-safe cause per signal.  The
# cause deliberately never contains an incoming user's message text.
_interrupted_threads: set[int] = set()
_interrupt_reasons: dict[int, str] = {}
_lock = threading.Lock()


def set_interrupt(
    active: bool,
    thread_id: int | None = None,
    *,
    reason: str | None = None,
) -> None:
    """Set (``active=True``) or clear the interrupt for *thread_id*, defaulting
    to the current thread (backward compat for CLI/tests).  ``reason`` is an
    optional user-safe cause."""
    tid = thread_id if thread_id is not None else threading.current_thread().ident
    with _lock:
        if active:
            _interrupted_threads.add(tid)
            if reason:
                _interrupt_reasons[tid] = reason
            else:
                _interrupt_reasons.pop(tid, None)
        else:
            _interrupted_threads.discard(tid)
            _interrupt_reasons.pop(tid, None)
        _snapshot = set(_interrupted_threads) if _DEBUG_INTERRUPT else None
    if _DEBUG_INTERRUPT:
        logger.info(
            "[interrupt-debug] set_interrupt(active=%s, target_tid=%s) "
            "called_from_tid=%s current_set=%s",
            active, tid, threading.current_thread().ident, _snapshot,
        )


def is_interrupted() -> bool:
    """Check if an interrupt has been requested for the current thread."""
    return is_thread_interrupted(threading.current_thread().ident)


def is_thread_interrupted(thread_id: int | None) -> bool:
    """Check whether *thread_id* has an interrupt bit set (``None`` never is).

    Used when a wait is moved onto a deadline worker (``run_bounded_sync``)
    so ``/stop`` targeting the original tool-worker tid still kills the
    subprocess.
    """
    if thread_id is None:
        return False
    with _lock:
        return thread_id in _interrupted_threads


def get_interrupt_reason() -> str | None:
    """Return the user-safe interrupt cause for the current thread, if known."""
    tid = threading.current_thread().ident
    with _lock:
        return _interrupt_reasons.get(tid)


def clear_current_thread_interrupt() -> None:
    """Clear any interrupt bit on the CURRENT thread.

    Gives a user-approved command a clean interrupt slate immediately before
    it spawns its child process, so a stale bit that landed on this thread
    during the blocking approval-wait cannot SIGINT the just-approved run
    (exit 130 + "[Command interrupted]").  Single-thread ordering on this tid
    keeps the DO-NOT-BREAK invariant intact: a *genuine* interrupt arriving
    after this call re-sets the bit on the same thread and is still observed by
    the executor's poll loop.  Call this directly, never via the
    _interrupt_event proxy (its .clear() binds to whatever thread runs it).
    """
    set_interrupt(False)  # thread_id=None -> current thread


# ---------------------------------------------------------------------------
# Backward-compatible _interrupt_event proxy: legacy call sites
# (code_execution_tool, process_registry, tests) import it and call
# .is_set() / .set() / .clear(); the shim maps those to the per-thread API.
# ---------------------------------------------------------------------------

class _ThreadAwareEventProxy:
    """Drop-in proxy that maps threading.Event methods to per-thread state."""

    def is_set(self) -> bool:
        return is_interrupted()

    def set(self) -> None:  # noqa: A003
        set_interrupt(True)

    def clear(self) -> None:
        set_interrupt(False)

    def wait(self, timeout: float | None = None) -> bool:
        """Not truly supported — returns current state immediately."""
        return self.is_set()


_interrupt_event = _ThreadAwareEventProxy()
