"""Platform adapters for messaging integrations (receive, send, auth, media)."""

from .base import BasePlatformAdapter, MessageEvent, SendResult

# QQAdapter / YuanbaoAdapter are exposed lazily (PEP 562 ``__getattr__``): eager
# imports cost ~48 ms / ~8 MB RSS on every CLI invocation and nothing in-tree
# imports them from the package root.
__all__ = [
    "BasePlatformAdapter",
    "MessageEvent",
    "SendResult",
    "QQAdapter",
    "YuanbaoAdapter",
]

_LAZY_ADAPTERS = {"QQAdapter": ".qqbot", "YuanbaoAdapter": ".yuanbao"}


def __getattr__(name):
    module = _LAZY_ADAPTERS.get(name)
    if module is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    return getattr(import_module(module, __name__), name)


def __dir__():
    return sorted(__all__)
