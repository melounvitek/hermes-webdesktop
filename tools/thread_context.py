"""Propagate agent-turn context into worker threads that dispatch Hermes tools.

A bare ``threading.Thread`` / ``ThreadPoolExecutor`` worker starts with an
empty ``contextvars.Context`` and no thread-local approval/sudo callbacks, so
tool dispatch inside it silently loses the approval ContextVars (gateway
sessions then auto-approve dangerous commands) and the CLI approval/sudo
callbacks (``prompt_dangerous_approval`` cannot reach the user,
GHSA-qg5c-hvr5-hjgr). Call :func:`propagate_context_to_thread` **on the parent
thread** (it snapshots at call time) and use the result as the worker target.
Callbacks are installed for the worker's lifetime and always cleared on exit.
"""

from __future__ import annotations

import contextvars
import logging
from typing import Callable

logger = logging.getLogger(__name__)


def _callback_api():
    """Resolve the terminal_tool callback getters/setters.

    Lazy: ``tools.terminal_tool`` imports ``tools.approval`` at module load, so a
    top-level import would risk a cycle for callers in ``tools.approval``.
    """
    from tools.terminal_tool import (
        _get_approval_callback,
        _get_sudo_password_callback,
        set_approval_callback,
        set_sudo_password_callback,
    )
    return (_get_approval_callback, _get_sudo_password_callback, set_approval_callback, set_sudo_password_callback)


def propagate_context_to_thread(target: Callable) -> Callable:
    """Wrap *target* to run with the *current* thread's ContextVars and approval/sudo callbacks.

    Fail-closed: if callback installation raises, the callbacks stay unset
    (``None``) — ``prompt_dangerous_approval`` then denies dangerous commands
    and the gateway approval queue blocks.
    """
    ctx = contextvars.copy_context()
    parent_approval_cb = parent_sudo_cb = None
    setters = None
    try:
        get_approval, get_sudo, set_approval, set_sudo = _callback_api()
        parent_approval_cb = get_approval()
        parent_sudo_cb = get_sudo()
        setters = (set_approval, set_sudo)
    except Exception:
        logger.debug("Could not capture parent approval/sudo callbacks", exc_info=True)

    def _runner(*args, **kwargs):
        def _inner():
            if setters is None:
                return target(*args, **kwargs)
            set_approval, set_sudo = setters
            try:
                if parent_approval_cb is not None:
                    set_approval(parent_approval_cb)
                if parent_sudo_cb is not None:
                    set_sudo(parent_sudo_cb)
            except Exception:
                logger.debug("Failed to install propagated approval/sudo callbacks; "
                             "dangerous-command approval will fail closed", exc_info=True)
            try:
                return target(*args, **kwargs)
            finally:
                try:
                    set_approval(None)
                    set_sudo(None)
                except Exception:
                    logger.debug("Failed to clear propagated approval/sudo callbacks", exc_info=True)

        return ctx.run(_inner)

    return _runner
