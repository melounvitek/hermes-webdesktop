"""Lazy forwarding helpers shared by the AIAgent façade and its mixins.

Targets are resolved on every call, so ``patch("<module>.<name>")`` in tests still intercepts and the
heavy agent modules stay off the ``run_agent`` import path.
"""
import importlib


def lazy_attr(module: str, name: str):
    """Resolve ``module.name`` at call time (keeps heavy/cyclic imports off the run_agent import path)."""
    return getattr(importlib.import_module(module), name)


def forward(module: str, name: str, *, static: bool = False):
    """Build an AIAgent method that lazily forwards to ``module.name`` (``target(self, *args, **kwargs)``).

    The target is resolved on every call, so ``patch("<module>.<name>")`` in tests still intercepts and
    run_agent keeps its import-time cost. ``static=True`` drops ``self``.
    """
    if static:
        def forwarder(*args, **kwargs):
            return getattr(importlib.import_module(module), name)(*args, **kwargs)
    else:
        def forwarder(self, *args, **kwargs):
            return getattr(importlib.import_module(module), name)(self, *args, **kwargs)
    forwarder.__name__ = forwarder.__qualname__ = name
    forwarder.__doc__ = f"Forwarder — see ``{module}.{name}``."
    return staticmethod(forwarder) if static else forwarder


def forward_static(module: str, name: str):
    return forward(module, name, static=True)
