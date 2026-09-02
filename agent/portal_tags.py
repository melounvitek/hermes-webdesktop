"""Centralized Nous Portal request tags.

Every Hermes request to the Nous Portal (main loop, auxiliary client, fallback
paths) must carry the same product-attribution tags, sent in OpenAI-compatible
``extra_body['tags']``::

    ["product=hermes-agent", "client=hermes-client-v<__version__>"]

One helper instead of inlined literals: the call sites drifted apart before,
and tests can assert one tag list everywhere. The version is read live from
``hermes_cli.__version__`` (the release script bumps that single string) — do
NOT pre-compute it as a module constant in consumers; it can change at runtime
(editable installs, hot reload).
"""

from __future__ import annotations

from contextvars import ContextVar
from typing import List, Optional

# Ambient conversation id (ATTRIBUTION value, sent as ``conversation=<id>``).
# The agent loop publishes it at turn entry; the dozens of auxiliary call
# sites funnelling through ``auxiliary_client.call_llm`` (no session handle)
# pick it up via ``nous_portal_tags()`` instead of threading a session_id
# parameter everywhere. A ContextVar, not a module global, so concurrent agents
# in one process (gateway sessions, delegate subagents) never see each other's
# id; ``tools.thread_context.propagate_context_to_thread`` workers inherit it,
# bare threads capture it at spawn time.
_conversation_id: ContextVar[Optional[str]] = ContextVar(
    "nous_portal_conversation_id", default=None
)

# Ambient affinity scope (ROUTING value): OpenRouter's sticky ``session_id``,
# Nous Portal's sticky key and xAI's ``x-grok-conv-id`` pin a conversation to
# one backend/prompt cache. Usually equal to the conversation id, but a host
# that mints one physical session per RESPONSE must route on the key it
# declared for the whole chat (``prompt_cache_scope.declared_conversation_scope``).
# Only that declared value is published; unset means consumers fall back to the
# conversation id, so delegate trees keep sharing their parent's sticky key.
_affinity_scope: ContextVar[Optional[str]] = ContextVar(
    "hermes_affinity_scope", default=None
)


def _reset_var(var: ContextVar, token) -> None:
    """Reset ``var``; a token from another Context (reset on a different
    thread) falls back to clearing rather than raising in cleanup paths."""
    try:
        var.reset(token)
    except Exception:
        var.set(None)


def set_affinity_scope(scope: Optional[str]):
    """Publish the declared routing/affinity scope; returns the ContextVar token."""
    return _affinity_scope.set(scope or None)


def reset_affinity_scope(token) -> None:
    """Restore the previous affinity scope (pair with ``set_affinity_scope``)."""
    _reset_var(_affinity_scope, token)


def get_affinity_scope() -> Optional[str]:
    """Return the declared routing/affinity scope, or ``None`` when unset."""
    return _affinity_scope.get()


def set_conversation_context(conversation_id: Optional[str]):
    """Publish the active conversation id for ambient Portal tagging.

    Called by the agent loop at turn entry with the session-lineage ROOT id
    (so the tag survives context-compression rotation). ``None`` clears.
    Returns the ContextVar token for ``reset_conversation_context``.
    """
    return _conversation_id.set(conversation_id or None)


def reset_conversation_context(token) -> None:
    """Restore the previous conversation context (pair with ``set_...``)."""
    _reset_var(_conversation_id, token)


def get_conversation_context() -> Optional[str]:
    """Return the ambient conversation id, or ``None`` when unset."""
    return _conversation_id.get()


def _hermes_version() -> str:
    """Current Hermes release version; ``"unknown"`` if hermes_cli is unimportable."""
    try:
        from hermes_cli import __version__
        return __version__
    except Exception:
        return "unknown"


def hermes_client_tag() -> str:
    """``client=hermes-client-v<MAJOR>.<MINOR>.<PATCH>``."""
    return f"client=hermes-client-v{_hermes_version()}"


def conversation_tag(session_id: str) -> str:
    """``conversation=<session_id>`` — high-cardinality, so only appended when
    a session id is actually available, never in the always-on base set."""
    return f"conversation={session_id}"


def nous_portal_tags(session_id: str | None = None) -> List[str]:
    """Return a fresh list of the canonical Nous Portal tags.

    The ambient conversation context (lineage ROOT id published by the agent
    loop) wins over the explicit ``session_id``, which remains a fallback for
    callers outside any agent turn; with neither, the two-tag base set.
    """
    tags = ["product=hermes-agent", hermes_client_tag()]
    effective = get_conversation_context() or session_id
    if effective:
        tags.append(conversation_tag(effective))
    return tags
