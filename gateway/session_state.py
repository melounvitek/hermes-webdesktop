"""Per-session gateway state consolidated into one container.

GatewayRunner historically carried ~19 separate ``Dict[str, ...]`` attributes
keyed by session_key, each with an ad-hoc lifecycle.  That shape bred three
bug classes, all structurally closed here:

1. Boundary drift — hand-copied pop-lists at conversation boundaries went
   stale when a dict was added.  Now one ``ConversationState.clear()``.
2. Turn-release drift — ad-hoc ``del self._running_agents[key]`` sites popped
   different subsets of the turn dicts.  Now ``TurnState.clear()``.
3. Wholesale-reset races — lazy-init ``self._x = {}`` replaced the ENTIRE dict,
   discarding concurrent sessions' entries.  Resets now touch one field of
   one ``SessionState``.

Scopes follow where each dict was CLEARED: ``turn`` resets at the end of every
running turn; ``conversation`` at conversation boundaries (/new, /resume,
auto-reset, expiry, compression-exhausted reset); ``persistent`` fields have
their own lifecycles and ``run_generation`` is monotonic and NEVER reset.

Entries in ``GatewayRunner._sessions`` are never evicted (matching the old
dicts, which also leaked empty/stale entries for dead sessions).  Eviction of
fully-default SessionStates is possible follow-up work.
"""

from __future__ import annotations

from collections.abc import MutableMapping
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, NamedTuple, Optional, Tuple

# Presence-sensitive sentinel: /fast stores "priority" or None (explicit
# normal), so key PRESENCE — not value truthiness — decides whether the
# override applies.  ``_UNSET_TIER`` means "no override recorded".
_UNSET_TIER = object()
SERVICE_TIER_UNSET = _UNSET_TIER  # public alias


@dataclass
class TurnState:
    """State scoped to one running gateway turn.

    ``clear()`` runs at every site that ends a running turn.  ``lease_token`` /
    ``lease_generation`` are deliberately NOT cleared by it — they are owned by
    ``_release_turn_lease``, which must release the registry lease exactly once
    per acquiring turn.
    """

    # Running AIAgent instance (or _AGENT_PENDING_SENTINEL); None = idle.
    agent: Any = None
    started_ts: float = 0.0  # 0.0 = not running
    lease: Any = None  # cross-process active-session slot lease
    busy_ack_ts: float = 0.0  # debounce; 0.0 = never acked
    # Held turn-lease token + the run generation that acquired it.  The pair
    # replaces the old (session_key, generation)-keyed dict so a stale unwind
    # can never free a newer turn's lease: release/rebind only match when the
    # generation is current.
    lease_token: Any = None
    lease_generation: Optional[int] = None

    def clear(self) -> None:
        """Reset the per-turn slot (agent / start ts / lease / busy-ack).

        The caller pops ``lease`` first so it can call ``lease.release()``.
        """
        self.agent = None
        self.started_ts = 0.0
        self.lease = None
        self.busy_ack_ts = 0.0


@dataclass
class ConversationState:
    """State scoped to one conversation (survives turns, not boundaries)."""

    # /model per-session override (model/provider/api_key/base_url/api_mode).
    model_override: Optional[Dict[str, Any]] = None
    one_turn_restore: Optional[Dict[str, Any]] = None  # /model --once snapshot
    reasoning_override: Optional[Dict[str, Any]] = None  # /reasoning override
    # /fast per-session override: "priority" or None; _UNSET_TIER = absent.
    service_tier_override: Any = _UNSET_TIER
    last_resolved_model: str = ""  # last successfully-resolved non-empty model
    queued_events: List[Any] = field(default_factory=list)  # /queue overflow FIFO (adapter slot holds the head)
    sidecar_notes: List[str] = field(default_factory=list)  # one-shot must-deliver notes
    ephemeral_pin: Optional[Tuple[Any, ...]] = None  # pinned session-context bytes: (change_key, text)
    vc_last: Optional[str] = None  # last voice-channel context delivered

    def clear(self) -> None:
        """Reset every field to its default — adding a field here means every
        conversation boundary clears it automatically."""
        self.__dict__.update(ConversationState().__dict__)


@dataclass
class PersistentState:
    """State with its own lifecycle — NOT cleared wholesale by turn or boundary
    resets (approvals/update prompts ARE cleared by the boundary *security*
    funnel, but individually)."""

    approvals: Optional[Dict[str, Any]] = None  # {"command": ..., "pattern_key": ...}
    update_prompt_pending: bool = False  # /update prompt awaiting a reply
    native_image_paths: List[str] = field(default_factory=list)  # consumed one-shot
    # Legacy runner-level pending message text (write-mostly; flushed to disk on
    # shutdown).  Distinct from the adapter-level ``_pending_messages``
    # (Dict[str, MessageEvent]) in gateway/base.py, which shares the old name.
    pending_command_text: Optional[str] = None
    # Monotonic run-generation counter.  NEVER reset: stale-run detection depends on it.
    run_generation: int = 0
    # Consecutive session-hygiene compression failures.  The in-agent compressor's
    # own timeout ladder is unreachable from the gateway (hygiene builds a FRESH
    # AIAgent per run and bind_session_state() zeroes that counter), so the streak
    # lives here and lets hygiene escalate its cooldown.  Reset on a successful
    # compression only.  PROCESS-LOCAL by design: no disk flush, so a restart drops
    # escalation to rung 1 while the DB-backed deadline survives; gateway.run
    # mirrors it to the DB keyed by session_key (not session_id) so it also holds
    # across compaction ROTATION, where the sid changes but the chat does not.
    hygiene_failure_streak: int = 0


@dataclass
class SessionState:
    """All per-session gateway state, grouped by lifecycle scope."""

    turn: TurnState = field(default_factory=TurnState)
    conversation: ConversationState = field(default_factory=ConversationState)
    persistent: PersistentState = field(default_factory=PersistentState)


# ---------------------------------------------------------------------------
# Legacy dict-view adapters.
#
# Dozens of tests construct bare runners (object.__new__) and read/write the
# old dict attributes directly (``runner._running_agents = {}``, ``assert key
# in runner._pending_approvals``).  Each view is a LIVE MutableMapping over one
# SessionState field across all sessions.  Production code in gateway/run.py
# uses ``self._session_state(key).<scope>.<field>``.
# ---------------------------------------------------------------------------


class _FieldSpec(NamedTuple):
    """One legacy dict: scope attr, field name, default factory, presence test."""

    scope: str
    name: str
    default: Callable[[], Any]
    is_present: Callable[[Any], bool]


def _spec(scope: str, name: str, default: Any) -> _FieldSpec:
    """``default`` is either a type (presence = truthiness) or a sentinel value
    such as ``None`` / ``_UNSET_TIER`` (presence = ``is not`` sentinel)."""
    if isinstance(default, type):
        return _FieldSpec(scope, name, default, bool)
    return _FieldSpec(scope, name, lambda: default, lambda v: v is not default)


class _RunnerView(MutableMapping):
    """Shared plumbing: live view over ``runner._sessions``, dict-comparable."""

    __slots__ = ("_runner",)

    def __init__(self, runner: Any) -> None:
        self._runner = runner

    def _sessions(self) -> Dict[str, SessionState]:
        return self._runner.__dict__.get("_sessions") or {}

    def __len__(self) -> int:
        return sum(1 for _ in self)

    # Mapping doesn't provide __eq__; tests compare against plain dicts.
    def __eq__(self, other: object) -> bool:
        if isinstance(other, (dict, MutableMapping)):
            return dict(self.items()) == dict(other)
        return NotImplemented

    def __ne__(self, other: object) -> bool:
        result = self.__eq__(other)
        return NotImplemented if result is NotImplemented else not result


class SessionFieldView(_RunnerView):
    """Live dict-like view of one SessionState field across sessions."""

    __slots__ = ("_spec",)

    def __init__(self, runner: Any, spec: _FieldSpec) -> None:
        super().__init__(runner)
        self._spec = spec

    def _value(self, state: SessionState) -> Any:
        return getattr(getattr(state, self._spec.scope), self._spec.name)

    def _set(self, state: SessionState, value: Any) -> None:
        setattr(getattr(state, self._spec.scope), self._spec.name, value)

    def __getitem__(self, key: str) -> Any:
        state = self._sessions().get(key)
        if state is None or not self._spec.is_present(value := self._value(state)):
            raise KeyError(key)
        return value

    def __setitem__(self, key: str, value: Any) -> None:
        self._set(self._runner._session_state(key), value)

    def __delitem__(self, key: str) -> None:
        state = self._sessions().get(key)
        if state is None or not self._spec.is_present(self._value(state)):
            raise KeyError(key)
        self._set(state, self._spec.default())

    def __iter__(self) -> Iterator[str]:
        for key, state in list(self._sessions().items()):
            if self._spec.is_present(self._value(state)):
                yield key

    def __contains__(self, key: object) -> bool:
        state = self._sessions().get(key)  # type: ignore[arg-type]
        return state is not None and self._spec.is_present(self._value(state))

    def clear(self) -> None:  # avoid MutableMapping's popitem loop
        for state in list(self._sessions().values()):
            self._set(state, self._spec.default())

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return f"SessionFieldView({self._spec.scope}.{self._spec.name}, {dict(self.items())!r})"


class TurnLeaseTokenView(_RunnerView):
    """Legacy view of ``_turn_lease_tokens``: keyed by (session_key, generation).

    The pair lives on ``TurnState.lease_token`` / ``lease_generation``; the lease
    registry serializes acquisition per session, so at most one held token
    exists per session key and the single slot equals the old tuple-keyed dict.
    """

    __slots__ = ()

    def _held(self, key: Any) -> Tuple[Any, SessionState]:
        """Return (session_key, state) for a currently-held (key, gen) or raise KeyError."""
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError(key)
        state = self._sessions().get(key[0])
        if state is None or state.turn.lease_token is None or state.turn.lease_generation != key[1]:
            raise KeyError(key)
        return key[0], state

    def __getitem__(self, key: Any) -> Any:
        return self._held(key)[1].turn.lease_token

    def __setitem__(self, key: Any, value: Any) -> None:
        if not isinstance(key, tuple) or len(key) != 2:
            raise KeyError(key)
        turn = self._runner._session_state(key[0]).turn
        turn.lease_token, turn.lease_generation = value, key[1]

    def __delitem__(self, key: Any) -> None:
        turn = self._held(key)[1].turn
        turn.lease_token = turn.lease_generation = None

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        for key, state in list(self._sessions().items()):
            if state.turn.lease_token is not None:
                yield (key, state.turn.lease_generation)

    def clear(self) -> None:  # avoid MutableMapping's popitem loop
        for key in list(self):
            del self[key]


# One spec per legacy dict attribute.
LEGACY_FIELD_SPECS: Dict[str, _FieldSpec] = {
    "_running_agents": _spec("turn", "agent", None),
    "_running_agents_ts": _spec("turn", "started_ts", float),
    "_active_session_leases": _spec("turn", "lease", None),
    "_busy_ack_ts": _spec("turn", "busy_ack_ts", float),
    "_session_model_overrides": _spec("conversation", "model_override", None),
    "_pending_one_turn_model_restores": _spec("conversation", "one_turn_restore", None),
    "_session_reasoning_overrides": _spec("conversation", "reasoning_override", None),
    "_session_service_tier_overrides": _spec("conversation", "service_tier_override", _UNSET_TIER),
    "_last_resolved_model": _spec("conversation", "last_resolved_model", str),
    "_queued_events": _spec("conversation", "queued_events", list),
    "_pending_turn_sidecar_notes": _spec("conversation", "sidecar_notes", list),
    "_session_ephemeral_pin": _spec("conversation", "ephemeral_pin", None),
    "_session_vc_last": _spec("conversation", "vc_last", None),
    "_pending_approvals": _spec("persistent", "approvals", None),
    "_update_prompt_pending": _spec("persistent", "update_prompt_pending", bool),
    "_pending_native_image_paths_by_session": _spec("persistent", "native_image_paths", list),
    "_pending_messages": _spec("persistent", "pending_command_text", None),
    "_session_run_generation": _spec("persistent", "run_generation", int),
}


def _legacy_property(make_view: Callable[[Any], MutableMapping], doc: str) -> property:
    """Dict-shaped @property over a live view.

    Getter returns the view; setter accepts a plain dict (the ubiquitous test
    pattern ``runner._X = {...}``), resetting the field on every known session
    and then applying the given entries; ``del runner._X`` (older tests
    simulating a runner without the attribute) means "no entries".
    """

    def fset(self: Any, mapping: Optional[Dict[Any, Any]]) -> None:
        view = make_view(self)
        view.clear()
        for key, value in (mapping or {}).items():
            view[key] = value

    return property(make_view, fset, lambda self: make_view(self).clear(), doc=doc)


def legacy_dict_property(attr_name: str) -> property:
    """Legacy dict-shaped @property for one migrated attribute."""
    spec = LEGACY_FIELD_SPECS[attr_name]
    return _legacy_property(
        lambda self: SessionFieldView(self, spec),
        f"Legacy dict view over SessionState.{spec.scope}.{spec.name} "
        "(kept for tests that access the pre-SessionState attribute).",
    )


def legacy_lease_token_property() -> property:
    """Legacy (session_key, generation)-keyed view of held turn-lease tokens."""
    return _legacy_property(
        TurnLeaseTokenView, "Legacy (session_key, generation)-keyed turn-lease token view."
    )
