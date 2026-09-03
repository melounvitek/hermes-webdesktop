"""Per-thread interrupt signaling for all tools.

Thread-scoped so interrupting one agent session does not kill tools running in
other sessions (the gateway runs many agents in one process). The agent stores
its execution thread id at the start of run_conversation() and passes it to
set_interrupt(); tools call is_interrupted(), which checks the CURRENT thread.
"""

import logging
import os
import threading

logger = logging.getLogger(__name__)

# Opt-in debug tracing — pairs with HERMES_DEBUG_INTERRUPT in
# tools/environments/base.py; logs caller/target thread and state per set/check.
_DEBUG_INTERRUPT = bool(os.getenv("HERMES_DEBUG_INTERRUPT"))

if _DEBUG_INTERRUPT:
    # AIAgent's quiet_mode forces the `tools` logger to ERROR on CLI startup;
    # force ours back to INFO so the trace is visible in agent.log.
    logger.setLevel(logging.INFO)

# Interrupted thread idents, plus an optional user-safe cause per signal.  The
# cause deliberately never contains an incoming user's message text.
_interrupted_threads: set[int] = set()
_interrupt_reasons: dict[int, str] = {}
_lock = threading.Lock()


def set_interrupt(active: bool, thread_id: int | None = None, *, reason: str | None = None) -> None:
    """Set or clear the interrupt for *thread_id* (default: current thread, for
    CLI/tests).  ``reason`` is an optional user-safe cause."""
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
            active, tid, threading.current_thread().ident, _snapshot)


def is_interrupted() -> bool:
    """Check if an interrupt has been requested for the current thread."""
    return is_thread_interrupted(threading.current_thread().ident)


def is_thread_interrupted(thread_id: int | None) -> bool:
    """Whether *thread_id* has an interrupt bit set (``None`` never is). Used when
    a wait moves onto a deadline worker (``run_bounded_sync``) so ``/stop``
    targeting the original tool-worker tid still kills the subprocess."""
    if thread_id is None:
        return False
    with _lock:
        return thread_id in _interrupted_threads


def get_interrupt_reason() -> str | None:
    """User-safe interrupt cause for the current thread, if known."""
    with _lock:
        return _interrupt_reasons.get(threading.current_thread().ident)


def clear_current_thread_interrupt() -> None:
    """Clear any interrupt bit on the CURRENT thread.

    Gives a user-approved command a clean slate right before it spawns its child,
    so a stale bit that landed during the blocking approval-wait cannot SIGINT the
    just-approved run.  Single-thread ordering keeps the invariant: a *genuine*
    interrupt arriving after this call re-sets the bit and is still observed by the
    executor's poll loop.  Call directly, never via the _interrupt_event proxy (its
    .clear() binds to whatever thread runs it).
    """
    set_interrupt(False)


class _ThreadAwareEventProxy:
    """Backward-compatible ``_interrupt_event``: legacy call sites call
    .is_set()/.set()/.clear(); the shim maps those to the per-thread API."""

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
