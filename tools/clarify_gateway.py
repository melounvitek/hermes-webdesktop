"""Gateway-side clarify primitive (blocking event-based queue).

In gateway mode the agent runs on a worker thread while the event loop handles
the user's reply, so a pending clarify is stored module-level (same shape as
``tools.approval``) and the agent thread blocks on an ``Event`` until an
adapter button callback or the gateway text-intercept resolves it, or the
timeout fires. Adapters render either inline buttons (with an "Other" row that
flips the entry into text-capture mode) or a numbered-list text fallback.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class _ClarifyEntry:
    """One pending clarify request inside a gateway session."""
    clarify_id: str
    session_key: str
    question: str
    choices: Optional[List[str]]
    multi_select: bool = False
    event: threading.Event = field(default_factory=threading.Event)
    response: Optional[str] = None
    awaiting_text: bool = False  # set when user picked "Other" or clarify is open-ended


_lock = threading.RLock()
_entries: Dict[str, _ClarifyEntry] = {}  # clarify_id -> entry (button callbacks)
_session_index: Dict[str, List[str]] = {}  # session_key -> [clarify_id] FIFO (text intercept, cleanup)
# Per-session notify callbacks (gateway -> adapter bridge); mirrors tools.approval.
_notify_cbs: Dict[str, Callable[[_ClarifyEntry], None]] = {}

# Outcomes for typed clarify replies. Gateway cancels the pending prompt on
# free prose (deadlock break) but keeps it armed for a retryable bad selection.
TEXT_RESOLVED = "resolved"
TEXT_REJECTED_PROSE = "rejected_prose"
TEXT_REJECTED_SELECTION = "rejected_selection"
TEXT_NO_PENDING = "no_pending"


# --- agent-thread side ---------------------------------------------------

def register(
    clarify_id: str,
    session_key: str,
    question: str,
    choices: Optional[List[str]],
    multi_select: bool = False,
) -> _ClarifyEntry:
    """Register a pending clarify request; caller then blocks on ``wait_for_response``."""
    entry = _ClarifyEntry(
        clarify_id=clarify_id,
        session_key=session_key,
        question=question,
        choices=list(choices) if choices else None,
        multi_select=bool(multi_select) and bool(choices),
        # Open-ended (no choices) -> next message IS the response, no buttons needed.
        awaiting_text=not bool(choices),
    )
    with _lock:
        _entries[clarify_id] = entry
        _session_index.setdefault(session_key, []).append(clarify_id)
    return entry


def wait_for_response(clarify_id: str, timeout: float) -> Optional[str]:
    """Block until the entry resolves or ``timeout`` elapses; None on timeout/unknown id.

    Polls in 1s slices so the inactivity heartbeat keeps firing (a single long
    ``Event.wait`` would let the gateway watchdog kill a live prompt).
    ``timeout <= 0`` means an unlimited wait.
    """
    with _lock:
        entry = _entries.get(clarify_id)
    if entry is None:
        return None

    try:
        from tools.environments.base import touch_activity_if_due
    except Exception:  # pragma: no cover - optional
        touch_activity_if_due = None

    unlimited = timeout is None or float(timeout) <= 0.0
    deadline = None if unlimited else time.monotonic() + float(timeout)
    activity_state = {"last_touch": time.monotonic(), "start": time.monotonic()}
    while True:
        if deadline is None:
            slice_s = 1.0
        else:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            slice_s = min(1.0, remaining)
        if entry.event.wait(timeout=slice_s):
            break
        if touch_activity_if_due is not None:
            touch_activity_if_due(activity_state, "waiting for user clarify response")

    with _lock:
        _entries.pop(clarify_id, None)  # regardless of outcome
        ids = _session_index.get(entry.session_key)
        if ids and clarify_id in ids:
            ids.remove(clarify_id)
            if not ids:
                _session_index.pop(entry.session_key, None)

    return entry.response


# --- gateway / adapter side -----------------------------------------------

def resolve_gateway_clarify(clarify_id: str, response: str) -> bool:
    """Unblock the waiter on ``clarify_id``; False if already resolved/expired/unknown."""
    with _lock:
        entry = _entries.get(clarify_id)
        if entry is None or entry.event.is_set():
            return False
        entry.response = str(response) if response is not None else ""
        entry.event.set()
        return True


def get_pending_for_session(
    session_key: str,
    *,
    include_choice_prompts: bool = False,
) -> Optional[_ClarifyEntry]:
    """Oldest pending entry awaiting free text (open-ended, or after "Other").

    ``include_choice_prompts=True`` returns the oldest unresolved entry of any
    kind — for when the user typed directly at an active multi-choice prompt,
    so the text resolves it instead of queueing as an unrelated follow-up turn.
    """
    with _lock:
        for cid in _session_index.get(session_key) or []:
            entry = _entries.get(cid)
            if entry is not None and (include_choice_prompts or entry.awaiting_text):
                return entry
        return None


def _label_matches(text: str, choice: object) -> bool:
    """Case-insensitive label match ignoring the '(Recommended)' suffix the
    first choice carries by the time it reaches adapters."""
    from tools.clarify_tool import strip_recommended

    return strip_recommended(text).casefold() == strip_recommended(str(choice)).casefold()


def _match_label(text: str, choices: List[str]) -> Optional[str]:
    """Canonical (stripped) choice text matching ``text``, or None."""
    for choice in choices:
        if _label_matches(text, choice):
            return str(choice).strip()
    return None


def _split_tokens(text: str) -> Optional[List[str]]:
    """Comma-separated tokens, or space-separated all-numeric tokens ("1 3"); else None."""
    if "," in text:
        return [t.strip() for t in text.split(",") if t.strip()]
    parts = text.split()
    if len(parts) > 1 and all(p.isdigit() for p in parts):
        return parts
    return None


def _selection_attempt_tokens(
    text: str,
    choices: Optional[List[str]] = None,
) -> Optional[List[str]]:
    """Tokens when ``text`` looks like a typed selection (bare int, comma list,
    all-numeric space list); None for free prose so the gateway can release the
    clarify. Comma-list labels may span up to the longest choice's word count."""
    stripped = str(text).strip()
    if not stripped:
        return None

    tokens = _split_tokens(stripped)
    if tokens is not None:
        if "," not in stripped or not tokens:
            return tokens or None
        max_words = max(1, max((len(str(c).split()) for c in choices or []), default=1))
        if all(t.isdigit() or len(t.split()) <= max_words for t in tokens):
            return tokens
        return None

    if stripped.isdigit() or (stripped.startswith("-") and stripped[1:].isdigit()):
        return [stripped]
    try:
        int(stripped)
        return [stripped]
    except ValueError:
        return None


def _coerce_text_response(entry: _ClarifyEntry, response: str) -> Optional[str]:
    """Accepted value for a typed reply, or None on any rejection."""
    return _coerce_text_response_detailed(entry, response)[0]


def _coerce_text_response_detailed(
    entry: _ClarifyEntry,
    response: str,
) -> tuple[Optional[str], Optional[str]]:
    """Map a typed reply to ``(value, None)`` or ``(None, reason)``.

    Reasons: ``"invalid_selection"`` (selection-shaped but out of range /
    unrecognised — keep the clarify armed for a retry) or ``"prose"`` (free
    text on a native choice prompt — gateway may cancel and route normally so a
    redirect-to-steer path cannot deadlock behind the waiting tool).
    Open-ended entries and ``awaiting_text`` mode accept any text; numeric
    picks and exact labels always resolve; multi-select returns a JSON array
    string that ``_parse_multi_select_response`` decodes on the tool side, and
    one bad token rejects the whole reply rather than resolving a partial pick.
    """
    text = str(response).strip()

    if not entry.choices:
        return text, None

    if entry.multi_select:
        coerced = _coerce_multi_select_text(entry, text)
        if coerced is not None:
            return coerced, None
        if entry.awaiting_text:
            return text, None
        if _selection_attempt_tokens(text, entry.choices) is not None:
            return None, "invalid_selection"
        return None, "prose"

    try:
        idx = int(text) - 1
        is_int = True
    except ValueError:
        idx = -1
        is_int = False

    if is_int and 0 <= idx < len(entry.choices):
        return entry.choices[idx], None

    matched = _match_label(text, entry.choices)
    if matched is not None:
        return matched, None

    if entry.awaiting_text:
        return text, None

    # Out-of-range / non-canonical integer is a failed selection, not prose.
    return None, "invalid_selection" if is_int else "prose"


def _coerce_multi_select_text(entry: _ClarifyEntry, text: str) -> Optional[str]:
    """Parse "1,3" / "1 3" / "staging, prod" into a JSON array of choice labels;
    None when any token is out of range or unrecognised (reject the whole reply)."""
    if not text:
        return None
    choices = entry.choices or []
    tokens = _split_tokens(text)
    if tokens is None:
        tokens = [text]

    selected: List[str] = []
    for token in tokens:
        if token.isdigit():
            idx = int(token) - 1
            if not 0 <= idx < len(choices):
                return None
            label = str(choices[idx]).strip()
        else:
            label = _match_label(token, choices)
            if label is None:
                return None
        if label not in selected:
            selected.append(label)

    return json.dumps(selected, ensure_ascii=False) if selected else None


def attempt_text_response_for_session(session_key: str, response: str) -> str:
    """Try to resolve the oldest pending clarify from typed text; returns a TEXT_* outcome."""
    entry = get_pending_for_session(session_key, include_choice_prompts=True)
    if entry is None:
        return TEXT_NO_PENDING

    coerced, reason = _coerce_text_response_detailed(entry, response)
    if coerced is None:
        return TEXT_REJECTED_SELECTION if reason == "invalid_selection" else TEXT_REJECTED_PROSE

    if resolve_gateway_clarify(entry.clarify_id, coerced):
        return TEXT_RESOLVED
    # Lost a race with a button/callback resolution — treat as no work left.
    return TEXT_NO_PENDING


def resolve_text_response_for_session(session_key: str, response: str) -> bool:
    """True only when the typed reply was accepted and the waiter unblocked."""
    return attempt_text_response_for_session(session_key, response) == TEXT_RESOLVED


def mark_awaiting_text(clarify_id: str) -> bool:
    """Flip an entry into text-capture mode (user picked 'Other'); False if unknown."""
    with _lock:
        entry = _entries.get(clarify_id)
        if entry is None:
            return False
        entry.awaiting_text = True
        return True


def has_pending(session_key: str) -> bool:
    """True when this session has at least one pending clarify entry."""
    with _lock:
        ids = _session_index.get(session_key) or []
        return any(_entries.get(cid) is not None for cid in ids)


def clear_session(session_key: str) -> int:
    """Drop every pending clarify for a session; returns how many were cancelled.

    Session-boundary cleanup (``/new``, shutdown, cached-agent eviction) so
    blocked agent threads don't outlive their session; cancelled waiters see "".
    First-writer-wins: an entry whose event is already set was answered for
    real, so it is dropped from the registry but its response is preserved
    rather than overwritten with the "" cancellation sentinel. The whole loop
    stays inside the lock so a button callback cannot slip between pop and check.
    Entries are removed from the registry regardless of state — a cleared
    session must not be resurrected by late callbacks. Callers distinguish the
    "" sentinel from a real reply only by their own timeout bookkeeping; most
    treat any falsy result as "user did not respond".
    """
    with _lock:
        ids = list(_session_index.pop(session_key, []) or [])
        cancelled = 0
        for entry in (_entries.pop(cid, None) for cid in ids):
            if entry is None or entry.event.is_set():
                continue
            entry.response = ""
            entry.event.set()
            cancelled += 1
    return cancelled


# --- config ---------------------------------------------------------------

def resolve_clarify_timeout(config: dict) -> int:
    """Clarify timeout (seconds) from a loaded config: legacy ``clarify.timeout``
    if explicitly set, else ``agent.clarify_timeout``, else 3600. Single source
    of truth for every surface (gateway, CLI, TUI) so the value can't drift.
    ``<= 0`` is preserved verbatim (unlimited); non-numeric falls back to 3600."""
    raw = (config.get("clarify") or {}).get("timeout")
    if raw is None:
        raw = (config.get("agent") or {}).get("clarify_timeout", 3600)
    try:
        return int(raw)
    except (TypeError, ValueError):
        return 3600


def get_clarify_timeout() -> int:
    """Read the clarify timeout from config.yaml; 0/negative = unlimited.

    Default 3600: long enough that a user who stepped away still finds a live
    entry when they tap, short enough that an abandoned prompt eventually
    unblocks the agent thread instead of pinning the running-agent guard.
    """
    try:
        from hermes_cli.config import load_config
        return resolve_clarify_timeout(load_config() or {})
    except Exception:
        return 3600
