"""Late-binding dependency seam for extracted dashboard routers.

``web_server`` owns all dashboard state/helpers; routers under ``web_routers/`` cannot
import it at import time (web_server imports them to mount them — a cycle) and must not
copy its state (tests ``monkeypatch.setattr(web_server, ...)`` and expect that to win).
``late(name)`` / ``LateState(name)`` resolve ``web_server.<name>`` *at call time*.
"""

from __future__ import annotations

import sys
from typing import Any


def _server():
    """Return the live ``hermes_cli.web_server`` module (imported on demand)."""
    mod = sys.modules.get("hermes_cli.web_server")
    if mod is None:  # pragma: no cover - routers are only mounted by web_server
        import hermes_cli.web_server as mod  # type: ignore[no-redef]
    return mod


def late(name: str):
    """Late-binding proxy for a callable defined on ``web_server``."""

    def _proxy(*args: Any, **kwargs: Any):
        return getattr(_server(), name)(*args, **kwargs)

    _proxy.__name__ = name
    _proxy.__qualname__ = name
    return _proxy


class LateState:
    """Live proxy for module-level state owned by ``web_server``.

    Forwards attribute/item access, iteration, membership, len/truthiness, ``with`` (locks)
    and rich comparisons to ``web_server.<name>`` resolved at operation time — some state is
    defined *after* the router's ``include_router`` point, so a late import would miss it.
    """

    __slots__ = ("_name",)

    def __init__(self, name: str) -> None:
        object.__setattr__(self, "_name", name)

    def _target(self) -> Any:
        return getattr(_server(), object.__getattribute__(self, "_name"))

    def __getattr__(self, attr: str) -> Any:
        return getattr(self._target(), attr)

    def __getitem__(self, key: Any) -> Any:
        return self._target()[key]

    def __setitem__(self, key: Any, value: Any) -> None:
        self._target()[key] = value

    def __delitem__(self, key: Any) -> None:
        del self._target()[key]

    def __contains__(self, item: Any) -> bool:
        return item in self._target()

    def __iter__(self):
        return iter(self._target())

    def __len__(self) -> int:
        return len(self._target())

    def __bool__(self) -> bool:
        return bool(self._target())

    def __enter__(self):
        return self._target().__enter__()

    def __exit__(self, *exc):
        return self._target().__exit__(*exc)

    def __eq__(self, other: Any) -> bool:
        return self._target() == other

    def __ne__(self, other: Any) -> bool:
        return self._target() != other

    def __lt__(self, other: Any) -> bool:
        return self._target() < other

    def __le__(self, other: Any) -> bool:
        return self._target() <= other

    def __gt__(self, other: Any) -> bool:
        return self._target() > other

    def __ge__(self, other: Any) -> bool:
        return self._target() >= other

    def __hash__(self) -> int:
        return hash(self._target())

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"<LateState {object.__getattribute__(self, '_name')} -> {self._target()!r}>"
