"""Shared concurrency helpers for plugin authors.

The common plugin footgun is the lazy process-wide singleton (``if _client is None:
_client = Expensive()``): two threads both pass the guard, both build, and the second
write leaks the first's connections/threads. Multi-threaded agent sessions (delegated
tool calls, background workers) make this reachable, so use these instead of hand-rolling
double-checked locking:

* :func:`lazy_singleton` — decorator for the zero-arg accessor case.
* :class:`SingletonSlot` — manual slot when the instance depends on a config/key argument.

Both are stdlib-only (``threading``) so any plugin can import them cheaply.
"""

from __future__ import annotations

import functools
import threading
from typing import Callable, Generic, Optional, TypeVar

__all__ = ["lazy_singleton", "SingletonSlot"]

T = TypeVar("T")


class SingletonSlot(Generic[T]):
    """Thread-safe lazy slot for accessors that take a build argument.

    Caches the first successfully-built instance and ignores the argument afterwards
    ("first config wins", the semantics most plugins rely on). The factory runs at most
    once under concurrent first calls; if it raises, nothing is cached and the next call
    retries. Example::

        _slot: SingletonSlot[Honcho] = SingletonSlot()

        def get_honcho_client(config=None):
            return _slot.get(lambda: Honcho(**resolve(config)))
    """

    __slots__ = ("_lock", "_value", "_set")

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._value: Optional[T] = None
        self._set = False

    def get(self, factory: Callable[[], T]) -> T:
        # Fast path without the lock: a bool + ref read is atomic under the GIL.
        if self._set:
            return self._value  # type: ignore[return-value]
        with self._lock:
            if self._set:
                return self._value  # type: ignore[return-value]
            value = factory()
            self._value = value
            self._set = True
            return value

    def peek(self) -> Optional[T]:
        """Return the cached instance without building it (None if unset)."""
        return self._value if self._set else None

    def reset(self) -> None:
        """Drop the cached instance so the next ``get()`` rebuilds it."""
        with self._lock:
            self._value = None
            self._set = False


def lazy_singleton(factory: Callable[[], T]) -> Callable[[], T]:
    """Wrap a zero-argument factory into a thread-safe lazy singleton accessor.

    The factory runs exactly once even under concurrent first calls (if it raises, the
    next call retries). A ``.reset()`` attribute drops the instance for tests/teardown::

        @lazy_singleton
        def get_client():
            return ExpensiveClient(load_config())
    """
    slot: SingletonSlot[T] = SingletonSlot()

    @functools.wraps(factory)
    def accessor() -> T:
        return slot.get(factory)

    accessor.reset = slot.reset  # type: ignore[attr-defined]
    return accessor
