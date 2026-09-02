"""Recurring in-session wakeups — the /loop command (Claude Code parity).

- The agent ends a wakeup reply with ``LOOP_COMPLETE`` on its own line (the wakeup prompt teaches it
to do so when the task is done/moot). - ``--times N`` — stop after N ticks.

Design notes / invariants (same contract as ``hermes_cli/goals.py``):
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, field, fields, asdict
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────────────
# Constants & defaults
# ──────────────────────────────────────────────────────────────────────

# Floor for fixed intervals. Claude Code allows 30s; anything tighter is
# almost always an accident that burns tokens polling state that hasn't
# changed. Overridable via loops.min_interval_seconds (still clamped ≥ 5).
DEFAULT_MIN_INTERVAL_SECONDS = 30

# Backstop tick budget so an unattended loop can't run forever by default.
# 0 = unlimited (Claude Code behavior); config loops.max_ticks.
DEFAULT_MAX_TICKS = 100

# Self-paced mode: start at the floor, double while replies are unchanged,
# cap at the ceiling, snap back to the floor on any change.
DEFAULT_SELF_PACED_FLOOR_SECONDS = 60
DEFAULT_SELF_PACED_CEILING_SECONDS = 15 * 60

# The completion sentinel the wakeup prompt teaches the agent to emit when
# the loop's task is finished or no longer applicable.
LOOP_COMPLETE_MARKER = "LOOP_COMPLETE"

# Matches the marker on its own line (possibly with surrounding whitespace
# or trailing punctuation the model added despite instructions).
_LOOP_COMPLETE_RE = re.compile(
    r"(?im)^\s*" + re.escape(LOOP_COMPLETE_MARKER) + r"\s*[.!]?\s*$"
)

# Interval token: 30s / 5m / 2h / 1h30m (compound units allowed, at least one).
_INTERVAL_TOKEN_RE = re.compile(
    r"^(?=\d)(?:(\d+)h)?(?:(\d+)m)?(?:(\d+)s)?$", re.IGNORECASE
)


WAKEUP_PROMPT_TEMPLATE = (
    "[/loop wakeup #{tick}{cadence}]\n"
    "Recurring task: {prompt}\n\n"
    "This is an automatic wakeup from the /loop the user set. Perform the "
    "task now against the CURRENT state (re-check files, processes, or "
    "services fresh — do not assume anything from earlier iterations still "
    "holds). Report concisely what you found or did this iteration.\n"
    "If the task is now complete, no longer applicable, or the thing you "
    "were watching has finished, say so and end your reply with "
    f"{LOOP_COMPLETE_MARKER} on its own line — that stops the loop."
)

WAKEUP_PROMPT_WITH_UNTIL_TEMPLATE = (
    "[/loop wakeup #{tick}{cadence}]\n"
    "Recurring task: {prompt}\n\n"
    "Stop condition: {until}\n\n"
    "This is an automatic wakeup from the /loop the user set. Perform the "
    "task now against the CURRENT state (re-check files, processes, or "
    "services fresh — do not assume anything from earlier iterations still "
    "holds). Report concisely what you found or did this iteration, and "
    "show concrete evidence of the stop condition's status.\n"
    "If the stop condition is met, or the task is no longer applicable, say "
    f"so and end your reply with {LOOP_COMPLETE_MARKER} on its own line — "
    "that stops the loop."
)


# ──────────────────────────────────────────────────────────────────────
# Interval parsing
# ──────────────────────────────────────────────────────────────────────


def parse_interval_token(token: str) -> Optional[int]:
    """Parse a compact interval token (``30s``/``5m``/``2h``/``1h30m``).

    Returns total seconds, or None when the token is not an interval. A bare number is NOT an
    interval (it collides with prompt text like ``/loop 3 things to check``); units are required.
    """
    if not token:
        return None
    m = _INTERVAL_TOKEN_RE.match(token.strip())
    if not m:
        return None
    h, mnt, s = (int(g) if g else 0 for g in m.groups())
    total = h * 3600 + mnt * 60 + s
    return total if total > 0 else None


def parse_loop_args(text: str) -> Dict[str, Any]:
    """Parse the argument string of ``/loop [interval] <prompt> [flags]``.

    Returns ``{"interval_seconds": int|None, "prompt": str, "times": int, "until": str, "error":
    str|None}``. ``interval_seconds`` None means self-paced. ``error`` is set for unusable input
    (empty prompt, interval-only, bad --times).
    """
    raw = (text or "").strip()
    result: Dict[str, Any] = {"interval_seconds": None, "prompt": "", "times": 0, "until": "", "error": None}
    if not raw:
        result["error"] = "empty"
        return result

    # Pull trailing flags first so an interval-looking token inside the
    # --until clause can't confuse the front parse. Flags may appear in
    # either order at the end of the line; --until consumes to end-of-line
    # (or to a following --times).
    times, until = 0, ""
    m_times = re.search(r"\s--times\s+(\S+)", raw)
    if m_times:
        try:
            times = int(m_times.group(1))
            if times < 1:
                raise ValueError
        except ValueError:
            result["error"] = f"--times expects a positive integer, got {m_times.group(1)!r}"
            return result
        raw = (raw[: m_times.start()] + raw[m_times.end():]).strip()

    m_until = re.search(r"\s--until\s+(.+)$", raw, re.DOTALL)
    if m_until:
        until = m_until.group(1).strip()
        raw = raw[: m_until.start()].strip()

    # Leading "every" sugar: /loop every 5m <prompt>
    tokens = raw.split(None, 1)
    if tokens and tokens[0].lower() == "every" and len(tokens) > 1:
        raw = tokens[1]
        tokens = raw.split(None, 1)

    interval: Optional[int] = None
    if tokens:
        maybe = parse_interval_token(tokens[0])
        if maybe is not None:
            interval = maybe
            raw = tokens[1].strip() if len(tokens) > 1 else ""

    if not raw:
        result["error"] = "missing prompt (usage: /loop [interval] <prompt>)"
        return result

    result.update(interval_seconds=interval, prompt=raw, times=times, until=until)
    return result


def format_interval(seconds: float) -> str:
    """Render seconds as a compact human interval (``90`` → ``1m30s``)."""
    seconds = int(max(0, round(seconds)))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    parts = []
    if h:
        parts.append(f"{h}h")
    if m:
        parts.append(f"{m}m")
    if s or not parts:
        parts.append(f"{s}s")
    return "".join(parts)


# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────


def _loops_config() -> Dict[str, Any]:
    """Read the ``loops:`` config section (cached load_config underneath)."""
    try:
        from hermes_cli.config import load_config

        section = (load_config() or {}).get("loops") or {}
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


def _config_int(key: str, default: int, floor: int) -> int:
    """``loops.<key>`` as an int clamped to ``floor``; ``default`` on any bad value."""
    try:
        return max(floor, int(_loops_config().get(key, default)))
    except Exception:
        return default


def min_interval_seconds() -> int:
    return _config_int("min_interval_seconds", DEFAULT_MIN_INTERVAL_SECONDS, 5)


def max_ticks_default() -> int:
    return _config_int("max_ticks", DEFAULT_MAX_TICKS, 0)


def self_paced_floor_seconds() -> int:
    return _config_int("self_paced_floor_seconds", DEFAULT_SELF_PACED_FLOOR_SECONDS, 10)


def self_paced_ceiling_seconds() -> int:
    floor = self_paced_floor_seconds()
    return _config_int("self_paced_ceiling_seconds", max(floor, DEFAULT_SELF_PACED_CEILING_SECONDS), floor)


# ──────────────────────────────────────────────────────────────────────
# Dataclass
# ──────────────────────────────────────────────────────────────────────


@dataclass
class LoopState:
    """Serializable /loop state stored per session."""

    prompt: str
    status: str = "active"            # active | paused | done | cleared
    mode: str = "interval"            # interval | self_paced
    interval_seconds: float = 0.0     # fixed cadence (mode == "interval")
    current_delay: float = 0.0        # live cadence (self-paced backoff)
    times: int = 0                    # user cap (--times N); 0 = none
    until: str = ""                   # judged stop condition; "" = none
    max_ticks: int = DEFAULT_MAX_TICKS  # config backstop; 0 = unlimited
    ticks_fired: int = 0
    created_at: float = 0.0
    last_fired_at: float = 0.0
    next_due_at: float = 0.0
    # True between "wakeup injected" and "that turn's response evaluated".
    # Keeps a tick from double-firing while its turn is still running and
    # tells the post-turn hook that the turn that just ended was ours.
    awaiting_response: bool = False
    # Self-paced change detection: digest of the previous wakeup's reply.
    last_response_digest: str = ""
    paused_reason: Optional[str] = None
    last_stop_reason: Optional[str] = None
    # Gateway routing captured at creation time (platform / chat_id /
    # chat_type / thread_id) so the idle wakeup watcher can inject the
    # tick back into the right chat. Empty for CLI / TUI sessions, which
    # drive ticks from their own session-local schedulers.
    route: Dict[str, str] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_json(cls, raw: str) -> "LoopState":
        data = json.loads(raw)
        route = data.get("route")
        kwargs: Dict[str, Any] = {
            "prompt": data.get("prompt", ""),
            "status": data.get("status", "active"),
            "mode": data.get("mode", "interval"),
            "awaiting_response": bool(data.get("awaiting_response", False)),
            "paused_reason": data.get("paused_reason"),
            "last_stop_reason": data.get("last_stop_reason"),
            "route": route if isinstance(route, dict) else {},
        }
        for f in fields(cls):
            if f.name not in kwargs:
                # Missing key -> dataclass default; present-but-falsy -> the type's zero.
                cast = {"str": str, "int": int, "float": float}[f.type]
                kwargs[f.name] = cast(data.get(f.name, f.default) or cast())
        return cls(**kwargs)

    # --- helpers -------------------------------------------------------

    def cadence_label(self) -> str:
        if self.mode == "self_paced":
            live = f", currently {format_interval(self.current_delay)}" if self.current_delay else ""
            return f"self-paced{live}"
        return f"every {format_interval(self.interval_seconds)}"

    def remaining_label(self) -> str:
        if self.status != "active":
            return ""
        remaining = self.next_due_at - time.time()
        if remaining <= 0:
            return "due now"
        return f"next in {format_interval(remaining)}"


# ──────────────────────────────────────────────────────────────────────
# Persistence (SessionDB state_meta)
# ──────────────────────────────────────────────────────────────────────

_META_PREFIX = "loop:"


def _meta_key(session_id: str) -> str:
    return f"{_META_PREFIX}{session_id}"


def _get_session_db() -> Optional[Any]:
    """One SessionDB per HERMES_HOME.

    Delegates to the goals module's cached SessionDB so goals, loops, and heartbeats share one
    connection (same pattern as ``hermes_cli/heartbeat.py``). The delegation also inherits the off-
    loop bootstrap and the window logic: a cold cache on the loop thread never runs ``SessionDB()``
    inline.
    """
    try:
        from hermes_cli.goals import _get_session_db as _goals_db
    except Exception as exc:  # pragma: no cover
        logger.debug("LoopManager: SessionDB bootstrap failed (%s)", exc)
        return None
    return _goals_db()


def _db_op(label: str, fn, default=None):
    """Run one SessionDB call; any error is logged at debug and yields ``default``."""
    try:
        return fn()
    except Exception as exc:
        logger.debug("LoopManager: %s failed: %s", label, exc)
        return default


def load_loop(session_id: str) -> Optional[LoopState]:
    """Load the loop for a session, or None if none exists."""
    if not session_id:
        return None
    db = _get_session_db()
    if db is None:
        return None
    raw = _db_op("get_meta", lambda: db.get_meta(_meta_key(session_id)))
    if not raw:
        return None
    try:
        return LoopState.from_json(raw)
    except Exception as exc:
        logger.warning("LoopManager: could not parse stored loop for %s: %s", session_id, exc)
        return None


def save_loop(session_id: str, state: LoopState) -> None:
    """Persist a loop to SessionDB. No-op if DB unavailable."""
    if not session_id:
        return
    db = _get_session_db()
    if db is None:
        from hermes_cli.goals import _warn_dropped_write

        _warn_dropped_write("LoopManager", "loop", session_id)
        return
    _db_op("set_meta", lambda: db.set_meta(_meta_key(session_id), state.to_json()))


def clear_loop(session_id: str) -> None:
    """Mark a loop cleared in the DB (preserved for audit, status=cleared)."""
    state = load_loop(session_id)
    if state is None:
        return
    state.status = "cleared"
    save_loop(session_id, state)


def list_active_loops() -> List[Tuple[str, LoopState]]:
    """Return ``[(session_id, LoopState), ...]`` for every ACTIVE loop.

    Used by the gateway's idle wakeup watcher, which has no per-session scheduler and scans for
    due loops on a coarse tick. Best-effort: any DB error yields ``[]``.
    """
    db = _get_session_db()
    if db is None:
        return []
    out: List[Tuple[str, LoopState]] = []
    for key, raw in _db_op("list_meta_prefix", lambda: db.list_meta_prefix(_META_PREFIX), []):
        session_id = key[len(_META_PREFIX):]
        if not session_id or not raw:
            continue
        try:
            state = LoopState.from_json(raw)
        except Exception:
            continue
        if state.status == "active":
            out.append((session_id, state))
    return out


def migrate_loop_to_session(old_session_id: str, new_session_id: str, *, reason: str = "") -> bool:
    """Carry a persistent /loop from a parent session to its continuation.

    Context compression rotates ``session_id`` to a fresh child session; without this the loop
    silently dies at the compaction boundary (the same hazard /goal hit in #33618). Best-effort and
    never raises.
    """
    if not old_session_id or not new_session_id or old_session_id == new_session_id:
        return False
    try:
        state = load_loop(old_session_id)
        if state is None or state.status == "cleared":
            return False
        if load_loop(new_session_id) is not None:
            return False
        save_loop(new_session_id, state)
        clear_loop(old_session_id)
        logger.debug(
            "LoopManager: migrated loop %s -> %s (%s)",
            old_session_id, new_session_id, reason or "rotation",
        )
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("LoopManager: loop migration failed: %s", exc)
        return False


# ──────────────────────────────────────────────────────────────────────
# Response evaluation helpers
# ──────────────────────────────────────────────────────────────────────


def _ticks_label(n: int) -> str:
    return f"{n} tick{'s' if n != 1 else ''}"


def response_signals_complete(response: str) -> bool:
    """True when the agent ended its reply with the LOOP_COMPLETE marker."""
    return bool(response) and _LOOP_COMPLETE_RE.search(response) is not None


def _digest_response(response: str) -> str:
    """Stable digest for self-paced change detection.

    Normalizes whitespace and strips volatile timestamp-ish tokens so a reply that differs only by
    'checked at 14:02:33' doesn't defeat the backoff.
    """
    text = (response or "").strip().lower()
    # Drop clock/timestamp tokens (14:02:33, 2026-07-26, 1500s, 25m ago...).
    text = re.sub(r"\d{1,2}:\d{2}(:\d{2})?", "", text)
    text = re.sub(r"\d{4}-\d{2}-\d{2}", "", text)
    text = re.sub(r"\b\d+(\.\d+)?\s*(s|sec|secs|seconds|m|min|mins|minutes|h|hr|hrs|hours)\b", "", text)
    text = re.sub(r"\s+", " ", text)
    return hashlib.sha256(text.encode("utf-8", "replace")).hexdigest()


# ──────────────────────────────────────────────────────────────────────
# LoopManager — the orchestration surface CLI + gateway + TUI talk to
# ──────────────────────────────────────────────────────────────────────


class LoopManager:
    """Per-session /loop state + tick decisions.

    Drivers (CLI process_loop, gateway wakeup watcher, TUI ticker) call ``set``/``pause``/
    ``resume``/``clear`` for user controls, ``is_due()`` (cheap, in-memory), ``fire_tick()`` to
    claim a tick and get the wakeup message, ``complete_tick()`` to evaluate the finished turn
    (LOOP_COMPLETE, --until judge, --times caps, next-tick scheduling), and ``status_line()``.
    """

    def __init__(self, session_id: str):
        self.session_id = session_id
        self._state: Optional[LoopState] = load_loop(session_id)

    # --- introspection ------------------------------------------------

    @property
    def state(self) -> Optional[LoopState]:
        return self._state

    def refresh(self) -> None:
        """Re-read state from the DB (cross-process safety for the gateway)."""
        self._state = load_loop(self.session_id)

    def is_active(self) -> bool:
        return self._state is not None and self._state.status == "active"

    def has_loop(self) -> bool:
        return self._state is not None and self._state.status in {"active", "paused"}

    def status_line(self) -> str:
        s = self._state
        if s is None or s.status == "cleared":
            return "No loop set. Start one with /loop [interval] <prompt>."
        fired = _ticks_label(s.ticks_fired)
        if s.times:
            caps = [f"{s.ticks_fired}/{s.times} runs"]
        elif s.max_ticks:
            caps = [f"{s.ticks_fired}/{s.max_ticks} budget"]
        else:
            caps = [fired]
        if s.until:
            caps.append(f"until: {s.until}")
        meta = f"{s.cadence_label()}, {', '.join(caps)}"
        if s.status == "active":
            remaining = s.remaining_label()
            tail = ", wakeup running" if s.awaiting_response else (f", {remaining}" if remaining else "")
            return f"↻ Loop (active, {meta}{tail}): {s.prompt}"
        if s.status == "paused":
            extra = f" — {s.paused_reason}" if s.paused_reason else ""
            return f"⏸ Loop (paused, {meta}{extra}): {s.prompt}"
        if s.status == "done":
            extra = f" — {s.last_stop_reason}" if s.last_stop_reason else ""
            return f"✓ Loop finished ({fired}{extra}): {s.prompt}"
        return f"Loop ({s.status}, {meta}): {s.prompt}"

    # --- mutation -----------------------------------------------------

    def set(
        self,
        prompt: str,
        *,
        interval_seconds: Optional[int] = None,
        times: int = 0,
        until: str = "",
        route: Optional[Dict[str, str]] = None,
    ) -> LoopState:
        """Start a new loop (replaces any existing one for the session)."""
        prompt = (prompt or "").strip()
        if not prompt:
            raise ValueError("loop prompt is empty")

        now = time.time()
        self_paced = interval_seconds is None
        interval = 0.0 if self_paced else float(max(int(interval_seconds), min_interval_seconds()))
        state = LoopState(
            prompt=prompt,
            mode="self_paced" if self_paced else "interval",
            interval_seconds=interval,
            current_delay=float(self_paced_floor_seconds()) if self_paced else interval,
            times=max(0, int(times or 0)),
            until=(until or "").strip(),
            max_ticks=max_ticks_default(),
            created_at=now,
            next_due_at=now,
            route=dict(route or {}),
        )
        self._state = state
        save_loop(self.session_id, state)
        return state

    def pause(self, reason: str = "user-paused") -> Optional[LoopState]:
        if not self._state or self._state.status not in {"active", "paused"}:
            return None
        self._state.status = "paused"
        self._state.paused_reason = reason
        self._state.awaiting_response = False
        save_loop(self.session_id, self._state)
        return self._state

    def resume(self) -> Optional[LoopState]:
        if not self._state or self._state.status == "cleared":
            return None
        self._state.status = "active"
        self._state.paused_reason = None
        self._state.awaiting_response = False
        # Re-arm relative to now so a long pause doesn't fire instantly N times.
        delay = self._state.current_delay or self._state.interval_seconds or self_paced_floor_seconds()
        self._state.next_due_at = time.time() + min(delay, 5.0)
        save_loop(self.session_id, self._state)
        return self._state

    def clear(self) -> bool:
        if self._state is None or self._state.status == "cleared":
            return False
        self._state.status = "cleared"
        save_loop(self.session_id, self._state)
        self._state = None
        return True

    # --- tick lifecycle -------------------------------------------------

    def is_due(self, now: Optional[float] = None) -> bool:
        """Cheap check: active, not mid-wakeup, and the clock has passed."""
        s = self._state
        if s is None or s.status != "active" or s.awaiting_response:
            return False
        return (now if now is not None else time.time()) >= s.next_due_at

    def fire_tick(self) -> Optional[str]:
        """Claim a due tick. Returns the message to inject, or None.

        Returns the wakeup-framed prompt, or the raw command when the loop's prompt is itself a
        slash command (``/loop 10m /recap``) so normal slash dispatch handles it. Marks
        ``awaiting_response`` so the tick can't double-fire; drivers MUST follow up with
        ``complete_tick`` (or ``abandon_tick`` on injection failure).
        """
        s = self._state
        if s is None or not self.is_due():
            return None
        s.ticks_fired += 1
        s.last_fired_at = time.time()
        s.awaiting_response = True
        # Provisionally schedule the next tick from NOW; complete_tick
        # reschedules from turn end (so a 10-minute turn doesn't cause an
        # instant re-fire), but if the process dies mid-turn the provisional
        # schedule keeps the persisted loop from being 'due' in a tight loop.
        delay = s.current_delay or s.interval_seconds or self_paced_floor_seconds()
        s.next_due_at = s.last_fired_at + delay
        save_loop(self.session_id, s)

        if s.prompt.lstrip().startswith("/"):
            return s.prompt.strip()
        cadence = f", {s.cadence_label()}" if s.mode == "interval" else ", self-paced"
        template = WAKEUP_PROMPT_WITH_UNTIL_TEMPLATE if s.until else WAKEUP_PROMPT_TEMPLATE
        return template.format(tick=s.ticks_fired, cadence=cadence, prompt=s.prompt, until=s.until)

    def abandon_tick(self) -> None:
        """Roll back a fired tick whose injection failed (nothing ran)."""
        s = self._state
        if s is None or not s.awaiting_response:
            return
        s.awaiting_response = False
        s.ticks_fired = max(0, s.ticks_fired - 1)
        save_loop(self.session_id, s)

    def _stop(self, status: str, reason: str, message: str) -> Dict[str, Any]:
        """Persist a terminal (``done``) or recoverable (``paused``) stop and build the result."""
        s = self._state
        s.status = status
        if status == "done":
            s.last_stop_reason = reason
        else:
            s.paused_reason = reason
        save_loop(self.session_id, s)
        return {"status": status, "stopped": True, "reason": reason, "message": message}

    def complete_tick(self, last_response: str) -> Dict[str, Any]:
        """Evaluate the finished wakeup turn and schedule what's next.

        Returns ``{"status": "active|done|paused", "stopped": bool, "reason": str,
        "message": str}``; ``message`` is a user-visible one-liner, "" in the common
        still-looping case.
        """
        s = self._state
        if s is None or not s.awaiting_response:
            return {"status": s.status if s else None, "stopped": False, "reason": "no tick in flight", "message": ""}
        s.awaiting_response = False
        now = time.time()
        ticks = _ticks_label(s.ticks_fired)

        # 1. Agent self-stop marker.
        if response_signals_complete(last_response):
            return self._stop("done", "agent signaled the task is complete",
                              f"✓ Loop finished after {ticks} — task complete.")

        # 2. Evidence-based --until judge (reuses the /goal judge; fail-open).
        if s.until and (last_response or "").strip():
            try:
                from hermes_cli.goals import judge_goal

                verdict, reason, _pf, _wait, _tf = judge_goal(s.until, last_response)
            except Exception as exc:
                verdict, reason = "continue", f"judge unavailable: {type(exc).__name__}"
            if verdict == "done":
                return self._stop("done", f"stop condition met: {reason}",
                                  f"✓ Loop finished after {ticks} — {reason}")
            if verdict == "blocked":
                # Judge ruled the stop condition unachievable — don't spin
                # until the tick budget; pause so the user can re-scope.
                why = f"stop condition judged unachievable: {reason}"
                return self._stop("paused", why,
                                  f"⏸ Loop paused — {why}. /loop resume to keep going, /loop stop to end it.")

        # 3. --times user cap.
        if s.times and s.ticks_fired >= s.times:
            return self._stop("done", f"completed the requested {s.times} runs",
                              f"✓ Loop finished — ran {s.times}/{s.times} times.")

        # 4. Config backstop budget → pause (recoverable), not done.
        if s.max_ticks and s.ticks_fired >= s.max_ticks:
            return self._stop(
                "paused", f"tick budget exhausted ({s.ticks_fired}/{s.max_ticks})",
                f"⏸ Loop paused — {s.ticks_fired}/{s.max_ticks} ticks used "
                "(loops.max_ticks). /loop resume to keep going, /loop stop to end it.",
            )

        # 5. Still looping — schedule the next tick from turn end.
        if s.mode == "self_paced":
            digest = _digest_response(last_response)
            floor = self_paced_floor_seconds()
            ceiling = self_paced_ceiling_seconds()
            if digest and digest == s.last_response_digest:
                # Nothing changed — back off.
                s.current_delay = min(max(s.current_delay, floor) * 2, ceiling)
            else:
                s.current_delay = float(floor)
            s.last_response_digest = digest
        else:
            s.current_delay = s.interval_seconds
        s.next_due_at = now + s.current_delay
        save_loop(self.session_id, s)
        return {"status": "active", "stopped": False, "reason": "loop continues", "message": ""}


# ──────────────────────────────────────────────────────────────────────
# /goal mixing
# ──────────────────────────────────────────────────────────────────────


def goal_blocks_loop_tick(session_id: str) -> bool:
    """True when an ACTIVE /goal should defer this session's /loop tick.

    Both features inject synthetic turns at idle boundaries. An active, non-parked goal owns the
    boundary; firing a loop wakeup in between would interleave two synthetic conversations and
    burn the goal's turn budget. Parked, paused, or done goals do NOT block the loop.
    """
    try:
        from hermes_cli.goals import GoalManager

        mgr = GoalManager(session_id=session_id)
        # Parked (waiting) goal → the loop may use the idle time.
        return mgr.is_active() and not mgr.is_waiting()
    except Exception:
        return False


# ──────────────────────────────────────────────────────────────────────
# Shared slash-command dispatch (CLI + gateway + TUI use the same logic)
# ──────────────────────────────────────────────────────────────────────


LOOP_HELP = (
    "Usage: /loop [interval] <prompt> [--times N] [--until <condition>]\n"
    "  /loop 5m check the deploy status      — first run now, then every 5m\n"
    "  /loop every 10m /recap                — loop a slash command\n"
    "  /loop keep fixing tests until green   — self-paced (backs off while output is unchanged)\n"
    "  /loop 2m poll CI --times 30           — stop after 30 runs\n"
    "  /loop 5m watch the queue --until queue is empty\n"
    "Controls: /loop status · /loop pause · /loop resume · /loop stop\n"
    "The loop also stops itself when the agent replies with "
    f"{LOOP_COMPLETE_MARKER}."
)


def _pause_output(mgr: "LoopManager") -> str:
    state = mgr.pause(reason="user-paused")
    return "No loop set." if state is None else f"⏸ Loop paused: {state.prompt}\nUse /loop resume to continue."


def _resume_output(mgr: "LoopManager") -> str:
    state = mgr.resume()
    return "No loop to resume." if state is None else f"▶ Loop resumed ({state.cadence_label()}): {state.prompt}"


def _stop_output(mgr: "LoopManager") -> str:
    return "✓ Loop stopped." if mgr.clear() else "No active loop."


# Control words -> handler returning the output text. Anything else is a new loop spec.
_CONTROL_COMMANDS = {
    **dict.fromkeys(("", "status"), lambda mgr: mgr.status_line()),
    "pause": _pause_output,
    "resume": _resume_output,
    **dict.fromkeys(("stop", "clear", "cancel"), _stop_output),
    **dict.fromkeys(("help", "--help", "-h"), lambda mgr: LOOP_HELP),
}


def dispatch_loop_command(
    mgr: "LoopManager",
    args: str,
    *,
    route: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """Surface-agnostic handler for ``/loop <args>``.

    Returns ``{"output": str, "created": bool}``. ``output`` is ready to print/send verbatim; each
    surface only decorates it (dim colors on the CLI, plain text on messaging platforms). ``route``
    is stored on newly created loops so the gateway's idle watcher can inject wakeups back into the
    right chat; CLI/TUI pass None.
    """
    arg = (args or "").strip()
    control = _CONTROL_COMMANDS.get(arg.lower())
    if control is not None:
        return {"output": control(mgr), "created": False}

    parsed = parse_loop_args(arg)
    if parsed["error"]:
        if parsed["error"] == "empty":
            return {"output": "Usage: /loop [interval] <prompt> — see /loop help.", "created": False}
        return {"output": f"/loop: {parsed['error']}", "created": False}

    replacing = mgr.has_loop()
    try:
        state = mgr.set(
            parsed["prompt"],
            interval_seconds=parsed["interval_seconds"],
            times=parsed["times"],
            until=parsed["until"],
            route=route,
        )
    except ValueError as exc:
        return {"output": f"/loop: {exc}", "created": False}

    lines = [f"↻ Loop set ({state.cadence_label()}): {state.prompt}"]
    if parsed["interval_seconds"] is not None and parsed["interval_seconds"] < state.interval_seconds:
        lines.append(
            f"(interval raised to the {format_interval(state.interval_seconds)} minimum — "
            "loops.min_interval_seconds)"
        )
    if state.mode == "self_paced":
        lines.append(
            f"Self-paced: first check in {format_interval(state.current_delay)}; "
            f"backs off up to {format_interval(self_paced_ceiling_seconds())} while nothing changes."
        )
    if state.times:
        lines.append(f"Runs {state.times} time{'s' if state.times != 1 else ''}, then stops.")
    if state.until:
        lines.append(f"Stops when: {state.until}")
    if not state.times and state.max_ticks:
        lines.append(f"Backstop budget: {state.max_ticks} ticks (loops.max_ticks; 0 = unlimited).")
    if state.status == "active":
        lines.append("First wakeup fires now, then on the cadence above. Controls: /loop status · pause · resume · stop.")
    else:
        lines.append(f"First wakeup {state.remaining_label()}. Controls: /loop status · pause · resume · stop.")
    if replacing:
        lines.insert(1, "(replaced the previous loop for this session)")
    return {"output": "\n".join(lines), "created": True}


__all__ = [
    "LoopState", "LoopManager", "parse_loop_args", "parse_interval_token", "format_interval",
    "response_signals_complete", "goal_blocks_loop_tick", "load_loop", "save_loop", "clear_loop",
    "list_active_loops", "migrate_loop_to_session", "dispatch_loop_command", "LOOP_COMPLETE_MARKER",
    "WAKEUP_PROMPT_TEMPLATE", "WAKEUP_PROMPT_WITH_UNTIL_TEMPLATE", "DEFAULT_MIN_INTERVAL_SECONDS",
    "DEFAULT_MAX_TICKS",
]
