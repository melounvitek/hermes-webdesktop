"""Session-scoped context variables for the Hermes gateway.

Replaces the old ``os.environ``-based session state (``HERMES_SESSION_*``)
with ``contextvars.ContextVar``.  The gateway processes messages concurrently
via asyncio; ``os.environ`` is process-global, so message B silently
overwrote message A's thread id before A's agent finished and notifications
routed to the wrong thread.  ContextVar values are task-local (inherited by
``run_in_executor`` threads), so concurrent messages never interfere.

``get_session_env(name, default="")`` is a drop-in for
``os.getenv("HERMES_SESSION_*", default)`` at existing tool call sites.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from typing import Any, Iterator

# Distinguishes "never set in this context" (fall back to os.environ for
# CLI/cron compat) from "explicitly set to empty" by clear_session_vars (no fallback).
_UNSET: Any = object()

# Process-level monotonic latch: has any code in this process bound a session via
# set_session_vars()?  Concurrent multi-session hosts (gateway, ACP, API server,
# TUI, cron) do; a pure single-process CLI/one-shot does not.  The subprocess-env
# bridge (tools/environments/local.py) reads this to pick its leak policy: when
# engaged, the ContextVars are authoritative and an _UNSET var means "no session
# bound in THIS task", so the last-writer-wins os.environ mirror must NOT be
# inherited by a child process.  When never engaged, the os.environ fallback is
# preserved (no concurrency to leak across).
_session_context_engaged: bool = False


def session_context_engaged() -> bool:
    """True if any session has been bound via set_session_vars in this process."""
    return _session_context_engaged


def _var(name: str) -> ContextVar:
    return ContextVar(name, default=_UNSET)


# --- Per-task session variables --------------------------------------------
_SESSION_PLATFORM = _var("HERMES_SESSION_PLATFORM")
_SESSION_SOURCE = _var("HERMES_SESSION_SOURCE")
_SESSION_CHAT_ID = _var("HERMES_SESSION_CHAT_ID")
_SESSION_CHAT_TYPE = _var("HERMES_SESSION_CHAT_TYPE")
_SESSION_CHAT_NAME = _var("HERMES_SESSION_CHAT_NAME")
_SESSION_THREAD_ID = _var("HERMES_SESSION_THREAD_ID")
_SESSION_USER_ID = _var("HERMES_SESSION_USER_ID")
_SESSION_USER_ID_ALT = _var("HERMES_SESSION_USER_ID_ALT")
_SESSION_USER_NAME = _var("HERMES_SESSION_USER_NAME")
# Platform-neutral scope discriminator (Discord guild / Slack workspace / Matrix
# server).  Captured at bind time so async producers (delegate_task
# background=True, terminal watchers) can persist a completion's full routing
# origin: a relay connector's fail-closed egress guard needs scope_id (or a user
# binding) to resolve the tenant for a scoped reply after a restart.
_SESSION_SCOPE_ID = _var("HERMES_SESSION_SCOPE_ID")
_SESSION_KEY = _var("HERMES_SESSION_KEY")
_SESSION_ID = _var("HERMES_SESSION_ID")
# In-process UI tab/window id for multi-session desktop/TUI hosts — deliberately
# separate from the durable HERMES_SESSION_ID.  Background completions use it as
# a precise return address so a stale/rotated durable key cannot be consumed by
# whichever desktop poller wakes first.
_SESSION_UI_SESSION_ID = _var("HERMES_UI_SESSION_ID")
# Triggering message id: reply anchor so background notifications stay inside
# the originating Telegram private-chat topic (routes only with thread id + anchor).
_SESSION_MESSAGE_ID = _var("HERMES_SESSION_MESSAGE_ID")
_SESSION_PROFILE = _var("HERMES_SESSION_PROFILE")
_BROWSER_CONTROL_PRINCIPAL = _var("HERMES_BROWSER_CONTROL_PRINCIPAL")
_BROWSER_CONTROL_TRANSPORT_FAMILY = _var("HERMES_BROWSER_CONTROL_TRANSPORT_FAMILY")
# Per-session cron marker, tri-state: _UNSET keeps the legacy env fallback for
# CLI/tests; "1" marks cron; "" explicitly marks non-cron and masks leaked env.
_CRON_SESSION = _var("HERMES_CRON_SESSION")

# Whether this session's channel can route an ASYNC completion back to the agent
# AFTER the turn ends (wake a fresh turn).  True for long-lived CLI sessions and
# real gateway platforms (persistent outbound channel + watcher/drain loops);
# False for finite runtimes that may exit before a detached completion returns
# (stateless API-server requests, dispatcher-spawned Kanban workers).  Tools that
# promise async delivery (terminal notify_on_complete / watch_patterns,
# delegate_task background=True) read ``async_delivery_supported()`` and refuse a
# promise the channel can't keep.  Default _UNSET => supported, so the CLI (never
# sets a platform) and contextvar-unaware paths keep working; stateless adapters
# opt OUT via ``supports_async_delivery = False`` on the adapter class, which the
# gateway propagates here at session-bind time.
_SESSION_ASYNC_DELIVERY = _var("HERMES_SESSION_ASYNC_DELIVERY")

# Cron auto-delivery vars — set per-job in run_job() so concurrent jobs don't
# clobber each other's delivery targets.
_CRON_AUTO_DELIVER_PLATFORM = _var("HERMES_CRON_AUTO_DELIVER_PLATFORM")
_CRON_AUTO_DELIVER_CHAT_ID = _var("HERMES_CRON_AUTO_DELIVER_CHAT_ID")
_CRON_AUTO_DELIVER_THREAD_ID = _var("HERMES_CRON_AUTO_DELIVER_THREAD_ID")

# Vars bound by set_session_vars / cleared to "" by clear_session_vars, in order.
_SESSION_VARS = (
    _SESSION_PLATFORM,
    _SESSION_SOURCE,
    _SESSION_CHAT_ID,
    _SESSION_CHAT_TYPE,
    _SESSION_CHAT_NAME,
    _SESSION_THREAD_ID,
    _SESSION_USER_ID,
    _SESSION_USER_ID_ALT,
    _SESSION_USER_NAME,
    _SESSION_SCOPE_ID,
    _SESSION_KEY,
    _SESSION_ID,
    _SESSION_UI_SESSION_ID,
    _SESSION_MESSAGE_ID,
    _SESSION_PROFILE,
    _BROWSER_CONTROL_PRINCIPAL,
    _BROWSER_CONTROL_TRANSPORT_FAMILY,
    _CRON_SESSION,
)

# Legacy env-var name -> ContextVar, for get_session_env.  _SESSION_ASYNC_DELIVERY
# is deliberately absent: it is a bool capability read via async_delivery_supported.
_VAR_MAP = {
    var.name: var
    for var in (*_SESSION_VARS, _CRON_AUTO_DELIVER_PLATFORM, _CRON_AUTO_DELIVER_CHAT_ID, _CRON_AUTO_DELIVER_THREAD_ID)
}


def _clear_session_cwd() -> None:
    try:
        from agent.runtime_cwd import clear_session_cwd

        clear_session_cwd()
    except Exception:
        pass


def set_current_session_id(session_id: str) -> None:
    """Synchronize ``HERMES_SESSION_ID`` across ContextVar and ``os.environ``.

    Long-lived single-process entrypoints (CLI) rotate sessions via /new,
    /resume, /branch, or compression splits without rebuilding the agent; tools
    read ``get_session_env("HERMES_SESSION_ID")`` with an os.environ fallback,
    so both stores must move together.

    Delegated subagent children are the exception: they are built in the parent
    process inside ``delegated_child_context()`` and their ``AIAgent.__init__``
    calls this helper.  Writing the child's id to process-global ``os.environ``
    would clobber the parent's id for the rest of the process, so only the
    task-local ContextVar write happens for them.  Root agents keep both paths.
    """
    import os

    _SESSION_ID.set(session_id)
    try:
        from agent.delegation_context import is_delegated_child_context

        if is_delegated_child_context():
            return
    except Exception:
        pass
    os.environ["HERMES_SESSION_ID"] = session_id


@contextmanager
def scoped_current_session_id(session_id: str | None = None) -> Iterator[None]:
    """Bind a task-local session id and restore the prior value on exit.

    ``session_id=None`` makes this a pure save/restore boundary around code that
    may call :func:`set_current_session_id` itself (delegated ``AIAgent``
    construction).  Never mutates ``os.environ``.
    """
    previous = _SESSION_ID.get()
    if session_id is not None:
        _SESSION_ID.set(session_id)
    try:
        yield
    finally:
        _SESSION_ID.set(previous)


def set_session_vars(
    platform: str = "",
    source: str = "",
    chat_id: str = "",
    chat_type: str = "",
    chat_name: str = "",
    thread_id: str = "",
    user_id: str = "",
    user_id_alt: str = "",
    user_name: str = "",
    scope_id: str = "",
    session_key: str = "",
    session_id: str = "",
    message_id: str = "",
    profile: str = "",
    browser_control_principal: str = "",
    browser_control_transport_family: str = "",
    cwd: str = "",
    async_delivery: bool = True,
    ui_session_id: str = "",
    cron_session: Any = _UNSET,
) -> list:
    """Set all session context variables and return reset tokens.

    Call ``clear_session_vars(tokens)`` in a ``finally`` when the handler exits.
    These helpers are not nestable: clearing resets every var to ``""`` rather
    than restoring prior values, and the tokens are accepted only for API compat.

    ``cwd`` pins the logical working directory.  ``async_delivery`` declares
    whether the channel can route a background completion back after the turn
    (stateless adapters such as the API server pass ``False``).  ``cron_session``
    is tri-state; see ``_CRON_SESSION``.
    """
    # Latch the process as engaged — see _session_context_engaged.
    global _session_context_engaged
    _session_context_engaged = True
    values = (
        platform, source, chat_id, chat_type, chat_name, thread_id, user_id,
        user_id_alt, user_name, scope_id, session_key, session_id, ui_session_id,
        message_id, profile, browser_control_principal,
        browser_control_transport_family, cron_session,
    )
    tokens = [var.set(value) for var, value in zip(_SESSION_VARS, values)]
    tokens.append(_SESSION_ASYNC_DELIVERY.set(bool(async_delivery)))
    try:
        from agent.runtime_cwd import set_session_cwd

        set_session_cwd(cwd)
    except Exception:
        pass
    return tokens


def clear_session_vars(tokens: list) -> None:
    """Mark session context variables as explicitly cleared.

    Sets every var to ``""`` (not ``var.reset(token)``) so ``get_session_env``
    returns empty instead of falling back to stale ``os.environ`` values while
    staying distinguishable from "never set" (``_UNSET``).  Async-delivery is
    reset to ``_UNSET`` rather than a falsy value: a cleared context must fall
    back to default-supported, not look like an opted-out stateless adapter.
    """
    for var in _SESSION_VARS:
        var.set("")
    _SESSION_ASYNC_DELIVERY.set(_UNSET)
    _clear_session_cwd()


def reset_session_vars() -> None:
    """Reset every session context variable to ``_UNSET`` for THIS context.

    Unlike :func:`clear_session_vars` (``""`` = "explicitly cleared", used when a
    handler *finishes*), this restores "never bound here" — what a freshly
    spawned task should look like *before* binding its own session.

    Why: ``create_task`` snapshots the current context, so message B's task can
    inherit message A's already-**set** vars.  Until B binds its own session,
    any subprocess it spawns reads A's identity through the subprocess-env
    bridge — whose _UNSET-strip guard cannot help because the vars are set-to-A.
    Calling this at the top of the per-message handler makes that window strip
    safe (no session) instead of leaking the foreign one.  See
    tests/tools/test_local_env_session_leak.py and
    tests/gateway/test_session_context_inheritance.py.

    ``_SESSION_ASYNC_DELIVERY`` is reset explicitly (it lives outside
    ``_VAR_MAP``): otherwise a task spawned from a context where a sibling
    adapter bound ``async_delivery=False`` inherits that ``False`` through the
    pre-bind window and misreports the new channel as unable to deliver.
    """
    for var in _VAR_MAP.values():
        var.set(_UNSET)
    _SESSION_ASYNC_DELIVERY.set(_UNSET)
    _clear_session_cwd()


def get_session_env(name: str, default: str = "") -> str:
    """Read a session context variable by its legacy ``HERMES_SESSION_*`` name.

    Drop-in for ``os.getenv(name, default)``.  Resolution: the ContextVar if it
    was ever set in this context (even to ``""`` — no fallback); else
    ``os.environ`` (CLI, cron scheduler, tests that never bind); else *default*.
    """
    import os

    var = _VAR_MAP.get(name)
    if var is not None and (value := var.get()) is not _UNSET:
        return value
    return os.getenv(name, default)


# Surfaces that are not a human chat channel.  The gateway binds a platform value
# (``telegram``) to HERMES_SESSION_PLATFORM while the CLI/TUI/desktop bind
# HERMES_SESSION_SOURCE and leave platform empty, so both are consulted.
# ``local``, ``api_server``, ``webhook``, ``msgraph_webhook`` are real Platform
# values with no attachment channel behind them.  Default-deny: an unrecognized
# identity counts as messaging so a new chat platform is never treated as a
# private surface before this set is updated.  Mirrors LOCAL_SESSION_SOURCE_IDS
# in apps/desktop/src/lib/session-source.ts; keep roughly in sync.
NON_MESSAGING_SESSION_SURFACES = frozenset(
    {
        "",
        "api_server",
        "cli",
        "codex",
        "desktop",
        "gateway",
        "kanban",
        "local",
        "msgraph_webhook",
        "tool",
        "tui",
        "webhook",
    }
)


def session_is_messaging_surface() -> bool:
    """Whether this turn is delivered over a human messaging channel.

    Decides "user is reading a chat message" vs "user is at a machine they own":
    delivery tags, whether a file must land somewhere the gateway can send from,
    whether narration reads as chat noise.  Checks ``HERMES_PLATFORM``, then the
    session platform, then the session source against
    :data:`NON_MESSAGING_SESSION_SURFACES`.
    """
    import os

    platform = os.getenv("HERMES_PLATFORM") or get_session_env("HERMES_SESSION_PLATFORM", "")
    source = get_session_env("HERMES_SESSION_SOURCE", "")
    return any(
        (ident := str(identity or "").strip().lower()) and ident not in NON_MESSAGING_SESSION_SURFACES
        for identity in (platform, source)
    )


def declare_stateless_channel() -> None:
    """Declare that this session cannot receive an async background completion.

    Binds only the delivery capability.  Use this instead of
    ``set_session_vars(async_delivery=False)`` on a pure single-process runner:
    ``set_session_vars`` also latches ``_session_context_engaged``, which flips
    the subprocess env bridge to ContextVar-authoritative — a one-shot CLI must
    not flip that latch as a side effect of declaring a capability.  Callers that
    build a full context (cron's ``run_job``) pass ``async_delivery=False``.
    ``delegate_task`` then falls through to its inline path so results return
    within the turn instead of going to a channel that never delivers.
    """
    _SESSION_ASYNC_DELIVERY.set(False)


def async_delivery_supported() -> bool:
    """Whether the current session can deliver a background completion later.

    False for finite runtimes: sessions bound by a stateless channel (API
    server, ``hermes -z``, cron — see :func:`declare_stateless_channel`) and
    dispatcher-spawned Kanban workers (``HERMES_KANBAN_TASK``), which are
    one-shot ``chat -q`` subprocesses whose parent disappears after the quiet
    turn, so a later completion has no durable consumer.  Gateway platforms,
    the interactive CLI, and any path that never bound the var return True.
    """
    import os

    # Kanban worker: force tools onto their synchronous/polling fallbacks.
    if os.environ.get("HERMES_KANBAN_TASK"):
        return False
    value = _SESSION_ASYNC_DELIVERY.get()
    return True if value is _UNSET else bool(value)
