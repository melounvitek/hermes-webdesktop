"""Idle deferral for background reviews on the managed local runtime.

When the review runtime IS the managed llama-server, the post-turn review fork monopolizes
the GPU the user's next prompt needs, and the next live turn cancels it — an active
session pays the decode cost AND loses the learning. Only the execution moment moves:
reviews bound for the managed local endpoint are queued and dispatched when the machine
is quiet; everything else runs immediately. Policy ``auxiliary.background_review.defer``:
``auto`` (default) defers exactly when the review runtime targets the managed local
server; ``never`` is the old behavior. Explicit /refine (focus set) never defers.

Queue semantics: one slot per session, newest snapshot wins (a review replays the whole
conversation, so coalescing is deduplication, not loss); preempted reviews are requeued by
the spawn wrapper observing the run token's cancel flag; aged-out events (defer_max_age_s,
default 30 min) dispatch regardless of idleness — deferral may delay learning, never lose
it; in-memory best-effort, dropped on process exit like the immediate fork. Idle truth
comes from the supervisor's /slots (machine-level, sees every client incl. other profiles)
and must hold for a settle window so a review is not launched between two quick prompts.
"""

from __future__ import annotations

import json
import logging
import threading
import time
import urllib.request
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger(__name__)

# Sustained-quiet window before dispatch: two back-to-back prompts must not look idle,
# a coffee break must run the queue.
_IDLE_SETTLE_S = 15.0
# Poll cadence while the queue is non-empty; the thread parks when empty.
_POLL_INTERVAL_S = 5.0
# Age at which a queued review dispatches regardless of idleness.
_MAX_AGE_DEFAULT_S = 30.0 * 60.0


def defer_mode(task_cfg: Optional[Dict[str, Any]]) -> str:
    """'auto' (default) or 'never' from auxiliary.background_review.defer."""
    raw = str((task_cfg or {}).get("defer", "auto")).strip().lower()
    return raw if raw in ("auto", "never") else "auto"


def defer_max_age_s(task_cfg: Optional[Dict[str, Any]]) -> float:
    raw = (task_cfg or {}).get("defer_max_age_s", _MAX_AGE_DEFAULT_S)
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return _MAX_AGE_DEFAULT_S
    return value if value > 0 else _MAX_AGE_DEFAULT_S


def review_targets_managed_local(agent: Any,
                                 task_cfg: Optional[Dict[str, Any]]) -> bool:
    """Would this review fork decode on the llama-server WE manage?

    Resolves the review runtime as the fork will and exact-matches its netloc against
    the supervisor state file (cannot false-positive on external local servers). Any
    failure reads False: immediate spawn is the safe default. The netloc probe (one
    TTL-cached state-file read) runs FIRST so cloud-only installs return False without
    resolving the runtime on the turn's tail.
    """
    try:
        from agent.auxiliary_client import (
            _is_managed_local_endpoint,
            _managed_local_netloc,
        )

        if not _managed_local_netloc():
            return False
        from agent.background_review import _resolve_review_runtime

        runtime = _resolve_review_runtime(agent, task_cfg)
        return _is_managed_local_endpoint(runtime.get("base_url"))
    except Exception:  # noqa: BLE001
        return False


class _PendingReview:
    __slots__ = ("agent", "kwargs", "enqueued_at", "session_key")

    def __init__(self, agent: Any, session_key: str, kwargs: Dict[str, Any], enqueued_at: float):
        self.agent = agent
        self.session_key = session_key
        self.kwargs = kwargs
        self.enqueued_at = enqueued_at


class ReviewIdleQueue:
    """Session-coalescing queue + idle-gated dispatcher thread."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending: Dict[str, _PendingReview] = {}
        self._wake = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._live_turns = 0
        self._quiet_since: Optional[float] = None
        # Test seams — replaced by unit tests, never in production.
        self._now: Callable[[], float] = time.monotonic
        self._server_idle: Callable[[], bool] = _managed_server_idle

    # ── turn liveness (this process) ────────────────────────────

    def note_turn_started(self) -> None:
        with self._lock:
            self._live_turns += 1
            self._quiet_since = None

    def note_turn_finished(self) -> None:
        with self._lock:
            self._live_turns = max(0, self._live_turns - 1)
            if self._live_turns == 0:
                self._quiet_since = self._now()
        self._wake.set()

    # ── queue ────────────────────────────────────────────────────

    def enqueue(self, agent: Any, session_key: str,
                kwargs: Dict[str, Any]) -> None:
        """Add (or replace — newest snapshot wins) a session's pending review.

        Keeps the ORIGINAL enqueue time on coalesce so a busy session cannot push its
        review's age-out forever.
        """
        with self._lock:
            existing = self._pending.get(session_key)
            enqueued_at = existing.enqueued_at if existing is not None else self._now()
            self._pending[session_key] = _PendingReview(agent, session_key, kwargs, enqueued_at)
        self._ensure_thread()
        self._wake.set()
        logger.info("Background review deferred (session=%s, queued=%d)",
                    session_key[-12:], len(self._pending))

    def pending_count(self) -> int:
        with self._lock:
            return len(self._pending)

    # ── dispatcher ───────────────────────────────────────────────

    def _ensure_thread(self) -> None:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run, daemon=True, name="bg-review-idle-queue")
                self._thread.start()

    def _quiet_for(self) -> float:
        """Seconds this process has been turn-free (0 while a turn runs)."""
        with self._lock:
            if self._live_turns > 0 or self._quiet_since is None:
                return 0.0
            return self._now() - self._quiet_since

    def _pop_dispatchable(self) -> Optional[_PendingReview]:
        """Oldest aged-out item, else the oldest item once quiet+idle hold."""
        with self._lock:
            if not self._pending:
                return None
            now = self._now()
            aged = [p for p in sorted(self._pending.values(), key=lambda p: p.enqueued_at)
                    if now - p.enqueued_at >= defer_max_age_s(p.kwargs.get("task_cfg"))]
            candidate = aged[0] if aged else None
        if candidate is None:
            if self._quiet_for() < _IDLE_SETTLE_S or not self._server_idle():
                return None
            with self._lock:
                if not self._pending:
                    return None
                candidate = min(self._pending.values(),
                                key=lambda p: p.enqueued_at)
        with self._lock:
            return self._pending.pop(candidate.session_key, None)

    def _run(self) -> None:
        while True:
            self._wake.wait()
            with self._lock:
                if not self._pending:
                    self._wake.clear()
                    continue
            item = None
            try:
                item = self._pop_dispatchable()
                if item is not None:
                    if not self._still_enabled(item):
                        logger.info(
                            "Deferred background review dropped: reviews "
                            "were disabled while it was queued (session=%s)",
                            item.session_key[-12:])
                        continue
                    logger.info(
                        "Dispatching deferred background review "
                        "(session=%s, waited=%.0fs, queued=%d)",
                        item.session_key[-12:],
                        self._now() - item.enqueued_at,
                        self.pending_count())
                    item.agent._spawn_background_review_now(**item.kwargs)
            except Exception:  # noqa: BLE001 — dispatcher must survive anything
                logger.warning("Deferred review dispatch failed",
                               exc_info=True)
            if item is None:
                time.sleep(_POLL_INTERVAL_S)

    @staticmethod
    def _still_enabled(item: _PendingReview) -> bool:
        """Re-check the enabled gate at DISPATCH time (minutes may pass in the queue;
        disabling reviews meanwhile must not be resurrected). Fail-open like the gate."""
        try:
            from agent.background_review import load_background_review_settings

            return load_background_review_settings()[0]
        except Exception:  # noqa: BLE001
            return True


def _managed_server_idle() -> bool:
    """Machine-level idle: no processing slot on any loaded model of the managed router.
    Unreachable/no state file reads idle (nothing to contend with)."""
    try:
        from hermes_cli.local_runtime.supervisor import state_path
        from urllib.parse import quote

        state = json.loads(state_path().read_text(encoding="utf-8"))
        base = str(state.get("base_url", "")).rsplit("/v1", 1)[0]
        headers = {"Authorization": f"Bearer {state.get('api_key', '')}"}
        if not base:
            return True

        def _get(path: str) -> Any:
            req = urllib.request.Request(f"{base}{path}", headers=headers)
            with urllib.request.urlopen(req, timeout=3) as r:
                return json.loads(r.read())

        loaded = [m["id"] for m in _get("/models").get("data", [])
                  if (m.get("status") or {}).get("value") in ("loaded", "ready")]
        for mid in loaded:
            if any(s.get("is_processing") for s in _get(f"/slots?model={quote(mid)}")
                   if isinstance(s, dict)):
                return False
        return True
    except Exception:  # noqa: BLE001
        return True


# Module singleton — one queue per process, like the load-progress watcher.
QUEUE = ReviewIdleQueue()
