"""
Session mirroring for cross-platform message delivery.

When a message is sent to a platform (send_message or cron delivery), append
a "delivery-mirror" record to the target session's transcript so the
receiving-side agent knows what was sent. Standalone: works from CLI, cron
and gateway contexts without the full SessionStore machinery.
"""

import json
import logging
from datetime import datetime
from typing import Optional

from hermes_cli.config import get_hermes_home

logger = logging.getLogger(__name__)

_SESSIONS_DIR = get_hermes_home() / "sessions"
_SESSIONS_INDEX = _SESSIONS_DIR / "sessions.json"


def _origin_user_id(entry: dict) -> str:
    return str((entry.get("origin") or {}).get("user_id") or "")


def mirror_to_session(
    platform: str, chat_id: str, message_text: str, source_label: str = "cli",
    thread_id: Optional[str] = None, user_id: Optional[str] = None,
    role: str = "assistant", session_id: Optional[str] = None,
) -> bool:
    """Append a delivery-mirror message to the target session's SQLite transcript.

    ``session_id``: pass it when the caller already holds the exact session
    (e.g. the cron in_channel seed that just created the row) to skip the
    origin scan, which refuses to guess on a populated chat (flat session + N
    thread sessions sharing one chat_id) and would silently drop the mirror.

    ``role`` defaults to ``"assistant"`` (the agent's own outgoing reply).  Text
    that is NOT the agent speaking (e.g. a cron brief) must pass ``role="user"``:
    ``mirror``/``mirror_source`` metadata is dropped at the SQLite boundary, so an
    assistant-role mirror replays as a real assistant turn and produces
    assistant→assistant pairs that break strict-alternation providers; a
    user-role mirror collapses safely via the consecutive-user merge.

    Returns True if mirrored, False if no matching session or error. Never raises.
    """
    try:
        if not session_id:
            session_id = _find_session_id(platform, str(chat_id), thread_id=thread_id, user_id=user_id)
        if not session_id:
            logger.warning(
                "Mirror: no session found for %s:%s thread=%s user=%s "
                "(explicit_id=none, origin-scan bailed)",
                platform, chat_id, thread_id, user_id,
            )
            return False

        _append_to_sqlite(session_id, {
            "role": role,
            "content": message_text,
            "timestamp": datetime.now().isoformat(),
            "mirror": True,
            "mirror_source": source_label,
        })
        logger.debug("Mirror: wrote to session %s (from %s)", session_id, source_label)
        return True
    except Exception as e:
        # WARNING, not debug: a silent mirror drop is the cron continuation-amnesia bug.
        logger.warning(
            "Mirror failed for %s:%s thread=%s user=%s session=%s: %s",
            platform, chat_id, thread_id, user_id, session_id, e,
        )
        return False


def _find_session_id(
    platform: str, chat_id: str, thread_id: Optional[str] = None, user_id: Optional[str] = None,
) -> Optional[str]:
    """Find the active session_id for a platform + chat_id pair.

    state.db gateway session rows are primary; sessions.json is the fallback
    for pre-migration databases. DM session keys don't embed the chat_id
    (e.g. "agent:main:telegram:dm"), so matching is on the persisted origin.

    With *user_id*, exact sender matches win. If several same-chat candidates
    exist and none matches the user, return None rather than guess and
    contaminate another participant's session.
    """
    try:
        from hermes_state import get_shared_session_db, release_or_close
        db = get_shared_session_db()
        try:
            finder = getattr(db, "find_session_by_origin", None)
            if callable(finder):
                session_id = finder(platform=platform, chat_id=chat_id, thread_id=thread_id, user_id=user_id)
                if session_id:
                    return str(session_id)
        finally:
            release_or_close(db)
    except Exception as e:
        logger.debug("Mirror state.db session lookup failed: %s", e)

    if not _SESSIONS_INDEX.exists():
        return None
    try:
        with open(_SESSIONS_INDEX, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    platform_lower = platform.lower()
    candidates = []
    for _key, entry in data.items():
        # Keys starting with "_" (e.g. the gateway's "_README") are metadata sentinels.
        if str(_key).startswith("_") or not isinstance(entry, dict):
            continue
        origin = entry.get("origin") or {}
        if (
            (origin.get("platform") or entry.get("platform", "")).lower() != platform_lower
            or str(origin.get("chat_id", "")) != str(chat_id)
            or (thread_id is not None and str(origin.get("thread_id") or "") != str(thread_id))
        ):
            continue
        candidates.append(entry)

    if not candidates:
        return None
    if user_id:
        exact_user_matches = [e for e in candidates if _origin_user_id(e) == str(user_id)]
        if exact_user_matches:
            candidates = exact_user_matches
        elif len(candidates) > 1:
            return None
    elif len(candidates) > 1:
        distinct_user_ids = {uid.strip() for uid in map(_origin_user_id, candidates) if uid.strip()}
        if len(distinct_user_ids) > 1:
            return None

    return max(candidates, key=lambda entry: entry.get("updated_at", "")).get("session_id")


def _append_to_sqlite(session_id: str, message: dict) -> None:
    """Append a message to the SQLite session database."""
    try:
        from hermes_state import get_shared_session_db, release_or_close
        db = get_shared_session_db()
        try:
            db.append_message(session_id=session_id, role=message.get("role", "assistant"), content=message.get("content"))
        finally:
            release_or_close(db)
    except Exception as e:
        logger.debug("Mirror SQLite write failed: %s", e)
