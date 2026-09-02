"""Rotation-stable logical cache scope for prompt_cache_key derivation.

Legacy compression rotation (``compression.in_place: false``) mints a new
physical ``session_id`` mid-conversation, which moved the conversation into a
fresh cache bucket each time. ``resolve_prompt_cache_scope()`` instead maps
the physical id to the ROOT of its compression lineage via
``SessionDB.get_compression_lineage()`` — NOT ``get_conversation_root`` /
``_conversation_root_id`` (the Portal-attribution walk), which follows
``parent_session_id`` blindly and would collapse /branch children and delegate
trees into one id. The two resolvers are intentionally different.

Scope boundaries: rotation children walk back to the original segment; ``/new``
starts a fresh scope; ``/branch`` children, delegate subagents, and tool-tagged
children are explicit fork children with their own isolated scope; cron fires
keep their physical id (the per-fire timestamp is stripped later).

Hosts that mint one physical id per RESPONSE (Studio group chat, ``/v1/responses``
with client-managed history) carry no lineage, so the walk returns the physical
id and the scope moves every reply. Hermes must not infer the conversation from
id SYNTAX (that collides client-supplied ids); the host declares it via
``gateway_session_key`` (``X-Hermes-Session-Key`` / ``build_session_key``),
consumed by ``declared_conversation_scope()``, which wins over the lineage walk.
The declared key is hashed to ``gwk_<sha256[:24]>`` because it embeds
platform/chat/user identifiers and leaves the process as a provider routing key.

Resolution is memoized per (agent, session_id, db-present): the lineage walk
runs once per transcript segment, never per API call.
"""

import hashlib
import logging
from typing import Any, Optional

logger = logging.getLogger(__name__)

_MEMO_ATTR = "_prompt_cache_scope_memo"
_DECLARED_SCOPE_PREFIX = "gwk_"


def _lineage_root(session_id: str, session_db: Any) -> Optional[str]:
    """Compression-lineage root of *session_id*, or None.

    Tolerates non-list results from test doubles / partially built agents.
    """
    if session_db is None:
        return None
    try:
        lineage = session_db.get_compression_lineage(session_id)
    except Exception:
        logger.debug("prompt-cache scope lineage walk failed", exc_info=True)
        return None
    if isinstance(lineage, (list, tuple)) and lineage:
        root = lineage[0]
        if isinstance(root, str) and root:
            return root
    return None


def _agent_source(
    agent: Any, session_id: str, session_db: Any, row_source: Optional[str] = None
) -> str:
    """The ``sessions.source`` this agent's conversation is recorded under.

    ``row_source`` is the row's value when the caller already read it (``""``
    = read, no source; ``None`` = not read yet, do the lookup). Before the row
    lands, use the SAME resolver persistence uses
    (``run_agent._session_source_for_agent``), not ``agent.platform``: they
    diverge under ``HERMES_SESSION_SOURCE``, and the declared scope is memoized
    immediately, so both sides of a ``/new`` would otherwise miss the boundary
    recorded under the override and hash the same scope.
    """
    if row_source is None and session_id and session_db is not None:
        try:
            row = session_db.get_session(session_id)
        except Exception:
            logger.debug("declared-scope source lookup failed", exc_info=True)
            row = None
        row_source = str(row.get("source") or "").strip() if row else ""
    if row_source:
        return row_source
    platform = getattr(agent, "platform", None)
    try:
        # Lazy: run_agent imports this module.
        from run_agent import _session_source_for_agent

        source = str(_session_source_for_agent(platform) or "").strip()
        if source:
            return source
    except Exception:
        logger.debug("declared-scope source authority unavailable", exc_info=True)
    return str(platform or "").strip()


def _conversation_generation(session_key: str, source: str, session_db: Any) -> str:
    """Durable generation for *session_key*'s current conversation (``""`` if none).

    The declared key names a chat and survives ``/new`` and policy resets, so
    hashing it alone would reuse one scope across distinct conversations. The
    ``conversation_generations`` counter advances in the same transaction that
    records a reset boundary and is independent of prunable rows and
    wall-clock, so pruning or clock rollback cannot reissue a generation.
    Compression does not advance it.
    """
    reader = getattr(session_db, "latest_conversation_boundary", None)
    if not callable(reader):
        return ""
    generation = reader(session_key, source)
    if generation is None:
        return ""
    return str(int(generation))


def declared_conversation_scope(agent: Any) -> Optional[str]:
    """Host-declared logical conversation scope (``gwk_<sha256[:24]>``), or None.

    Hashes ``(source, gateway_session_key, generation)`` so no platform/chat/
    user identifier reaches a provider. None — fall back to the physical-id
    scope — when no key is declared, when the agent is a background-review
    fork (``_persist_disabled`` clones the live runtime incl. the key), when
    the row is an explicit fork child, and on any DB error (fail closed rather
    than merge a fork onto its parent's key).
    """
    key = str(getattr(agent, "_gateway_session_key", "") or "").strip()
    if not key or getattr(agent, "_persist_disabled", False):
        return None
    sid = str(getattr(agent, "session_id", None) or "")
    db = getattr(agent, "_session_db", None)
    generation = ""
    row_source: Optional[str] = None
    if sid and db is not None:
        try:
            # One read for both halves of the row identity (fork verdict +
            # source). A SessionDB without the combined view keeps the
            # original call.
            identity = getattr(db, "declared_scope_identity", None)
            if callable(identity):
                is_fork, row_source = identity(sid)
            else:
                is_fork = db.is_explicit_fork_child(sid)
            if is_fork:
                return None
        except Exception:
            logger.debug("declared-scope fork check failed", exc_info=True)
            return None
    source = _agent_source(agent, sid, db, row_source)
    if db is not None:
        try:
            generation = _conversation_generation(key, source, db)
        except Exception:
            logger.debug("declared-scope generation read failed", exc_info=True)
            return None
    # Same identity tuple the peer queries use: two hosts may declare the
    # same key under different sources and must not collapse.
    carrier = f"{source}|{key}|{generation}"
    digest = hashlib.sha256(carrier.encode("utf-8", errors="replace")).hexdigest()[:24]
    return f"{_DECLARED_SCOPE_PREFIX}{digest}"


def resolve_prompt_cache_scope(agent: Any) -> str:
    """Rotation-stable cache-scope id for *agent*'s conversation.

    Declared scope when one applies, else the compression-lineage root of
    ``agent.session_id`` (the physical id when there is no ancestry, no DB, or
    the walk fails). Memoized on the agent keyed by session id.
    """
    sid = str(getattr(agent, "session_id", None) or "")
    if not sid:
        return ""
    db = getattr(agent, "_session_db", None)
    # DB presence is part of the key: an agent that gains a DB handle later
    # must re-resolve instead of staying pinned to the physical id.
    key = (sid, db is not None)
    memo = getattr(agent, _MEMO_ATTR, None)
    if isinstance(memo, tuple) and len(memo) == 2 and memo[0] == key:
        return memo[1]
    root = declared_conversation_scope(agent) or (
        _lineage_root(sid, db) if db is not None else None
    )
    scope = root or sid
    # Memoize on success, with no DB, or when the agent never persists a row
    # (background-review forks hold a DB handle but set _persist_disabled).
    # A failed/empty walk on a persisting agent is NOT memoized: the physical
    # id is right for now (row not yet persisted, transient error) but would
    # stay wrong for the whole segment once the row lands.
    if root is not None or db is None or getattr(agent, "_persist_disabled", False):
        try:
            setattr(agent, _MEMO_ATTR, (key, scope))
        except Exception:
            pass  # frozen/slotted doubles: resolution works, just unmemoized
    return scope


def declared_conversation_scope_safe(agent: Any) -> Optional[str]:
    """Never-raising variant of :func:`declared_conversation_scope`."""
    try:
        return declared_conversation_scope(agent)
    except Exception:
        logger.debug("declared conversation scope resolution failed", exc_info=True)
        return None


def resolve_prompt_cache_scope_safe(agent: Any) -> Optional[str]:
    """Never-raising variant of :func:`resolve_prompt_cache_scope` (None on failure/empty).

    Consumers treat None as "use the physical session_id"; at turn_context's
    call site an exception inside the ``set_runtime_main(...)`` argument list
    would skip the whole runtime binding, not just the cache scope.
    """
    try:
        return resolve_prompt_cache_scope(agent) or None
    except Exception:
        logger.debug("prompt-cache scope resolution failed", exc_info=True)
        return None
