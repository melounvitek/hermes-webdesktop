"""Origin-module lookup shared by the ``tools.browser_tool_*`` extraction modules.

Code extracted from ``tools/browser_tool.py`` must keep reading its origin's
symbols (helpers, module state dicts, ``logger``) *through* ``tools.browser_tool``
so ``patch("tools.browser_tool.X")`` in tests is honoured. The extraction modules
must not import ``tools.browser_tool`` at import time (cycle), so they call
:func:`origin_module` lazily per call.
"""

import sys
from types import ModuleType

_ORIGIN_NAME = "tools.browser_tool"


class _NamespaceView:
    """Live attribute view over a module namespace whose module object is gone.

    Used when a test purged ``sys.modules`` after importing the origin: the old
    module's code still runs with its own globals, so reads must see *those*
    (patched) globals, not a fresh re-import.
    """

    __slots__ = ("_g",)

    def __init__(self, g):
        object.__setattr__(self, "_g", g)

    def __getattr__(self, name):
        try:
            return self._g[name]
        except KeyError:
            raise AttributeError(name) from None

    def __setattr__(self, name, value):
        self._g[name] = value


def _module_for_globals(g: dict):
    """The module object whose namespace is ``g`` (or a live view over it)."""
    mod = sys.modules.get(_ORIGIN_NAME)
    if mod is not None and mod.__dict__ is g:
        return mod
    import tools

    mod = getattr(tools, "browser_tool", None)
    if mod is not None and mod.__dict__ is g:
        return mod
    return _NamespaceView(g)


def origin_module():
    """Return the ``tools.browser_tool`` instance the *calling* moved function belongs to.

    Resolution order, mirroring what in-file code would see:
    1. the nearest enclosing frame executing ``tools.browser_tool`` code — a moved
       helper called from origin code binds to that exact module copy, even after a
       test purged/reloaded ``sys.modules`` (several copies may be alive);
    2. a ``tools.browser_tool`` module referenced from a calling frame's globals
       (``from tools import browser_tool`` in a test that calls the helper directly);
    3. ``sys.modules`` / the ``tools`` package attribute / a fresh import.
    """
    start = sys._getframe(2)
    frame = start
    while frame is not None:
        if frame.f_globals.get("__name__") == _ORIGIN_NAME:
            return _module_for_globals(frame.f_globals)
        frame = frame.f_back
    frame = start
    while frame is not None:
        for value in list(frame.f_globals.values()):
            if isinstance(value, ModuleType) and getattr(value, "__name__", None) == _ORIGIN_NAME:
                return value
        frame = frame.f_back
    mod = sys.modules.get(_ORIGIN_NAME)
    if mod is None:
        import tools

        mod = getattr(tools, "browser_tool", None)
    if mod is None:
        import tools.browser_tool as mod
    return mod
