"""Per-turn context shared between ``GatewayRunner._run_agent_inner`` and
``TurnRunner`` (gateway/run.py).

``_run_agent_inner`` once defined its tool-progress plumbing and ``run_sync`` as
nested closures over ~20 locals. ``TurnContext`` is the extraction seam: each
closed-over local is a field here, so the closure bodies moved onto ``TurnRunner``
methods unchanged modulo ``name`` -> ``ctx.name`` rewrites.

Invariants:
- Fields are written once by ``_run_agent_inner`` while wiring the turn (a few are
  assigned onto the ctx slightly after construction, at the original binding sites).
- The original closures never rebound captured names except ``message`` (formerly
  ``nonlocal``): rebind sites now write ``ctx.message`` and the outer body reads it.
  Other mutable state keeps the single-element-list containers so mutation stays
  visible to the outer body through the shared objects.
- ``_run_still_current`` stays a callable (captures ``self``/``session_key``/
  ``run_generation``) so the extracted bodies remain byte-identical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional


@dataclass
class TurnContext:
    """Closed-over locals of ``_run_agent_inner`` needed by ``TurnRunner``."""

    # --- read-only turn identity / wiring ---
    source: Any = None
    _run_still_current: Callable[[], bool] = None  # type: ignore[assignment]
    _live_status_adapter: Any = None
    _live_status_mode: str = "off"
    _thinking_enabled: bool = False
    progress_mode: str = "off"
    progress_grouping: str = "grouped"
    tool_progress_enabled: bool = False

    # --- queues ---
    progress_queue: Any = None
    log_queue: Any = None

    # --- mutable single-element containers (shared with the outer body) ---
    last_progress_msg: list = field(default_factory=lambda: [None])
    last_tool: list = field(default_factory=lambda: [None])
    last_was_terminal_block: list = field(default_factory=lambda: [False])
    repeat_count: list = field(default_factory=lambda: [0])
    long_tool_hint_fired: list = field(default_factory=lambda: [False])
    agent_holder: list = field(default_factory=lambda: [None])

    # --- constants / cleanup bookkeeping ---
    _LONG_TOOL_THRESHOLD_S: float = 30.0
    _cleanup_progress: bool = False
    _cleanup_msg_ids: List[str] = field(default_factory=list)

    # --- progress threading metadata (assigned before send_progress_messages runs) ---
    _progress_metadata: Optional[dict] = None
    _progress_reply_to: Optional[Any] = None

    # --- run_sync seam: the ex-``nonlocal`` turn message (rebindable) ---
    message: Optional[str] = None

    # --- turn parameters / config snapshots (read-only in run_sync) ---
    history: Any = None
    context_prompt: Optional[str] = None
    channel_prompt: Optional[str] = None
    session_id: Optional[str] = None
    session_key: Optional[str] = None
    run_generation: Optional[int] = None
    process_task_id: str = ""
    process_baseline: frozenset[str] = field(default_factory=frozenset)
    _interrupt_depth: int = 0
    event_message_id: Optional[str] = None
    # Raw platform id of the INBOUND user message (event.message_id), distinct from
    # event_message_id (the reply/thread anchor, which may be the replied-to message
    # on Slack/Mattermost/Buzz or None for Telegram topics). Stamped as
    # platform_message_id on the persisted user turn.
    inbound_message_id: Optional[str] = None
    moa_config: Optional[dict] = None
    persist_user_message: Optional[Any] = None
    persist_user_timestamp: Optional[float] = None
    # display_kind for the persisted user row when the turn was self-injected
    # (MessageEvent.internal), e.g. "internal_notification". DB-only presentation
    # metadata; never sent to the provider.
    persist_user_display_kind: Optional[str] = None
    user_config: Any = None
    enabled_toolsets: Any = None
    disabled_toolsets: Any = None
    log_mode_enabled: bool = False
    interim_assistant_messages_enabled: bool = False
    needs_progress_queue: bool = False

    # --- lazy-imported callables captured from the outer body ---
    AIAgent: Any = None
    resolve_display_setting: Any = None

    # --- mutable holder cells (shared-list pattern) ---
    result_holder: list = field(default_factory=lambda: [None])
    tools_holder: list = field(default_factory=lambda: [None])
    stream_consumer_holder: list = field(default_factory=lambda: [None])
    streaming_tts_consumer_holder: list = field(default_factory=lambda: [None])

    # --- voice-ack wiring ---
    _voice_ack_fired: list = field(default_factory=lambda: [False])
    _voice_ack_guild: list = field(default_factory=lambda: [None])
    _voice_ack_loop: Any = None

    # --- hook / status bridge wiring (published at original binding sites) ---
    _loop_for_step: Any = None
    _hooks_ref: Any = None
    _status_adapter: Any = None
    _status_chat_id: Any = None
    _status_thread_metadata: Optional[dict] = None

    # --- extracted sibling callbacks (bound TurnRunner methods read via ctx) ---
    progress_callback: Optional[Callable] = None
    voice_ack_callback: Optional[Callable] = None
    _step_callback_sync: Optional[Callable] = None
    _event_callback_sync: Optional[Callable] = None
    _status_callback_sync: Optional[Callable] = None

    # --- Slack-native task-card progress (opt-in via the adapter's
    #     native_task_cards_enabled()). ID-bearing lifecycle callbacks are published
    #     by TurnRunner so tool starts/completions correlate by tool-call ID. ---
    _native_slack_task_cards: bool = False
    native_tool_start_callback: Optional[Callable] = None
    native_tool_complete_callback: Optional[Callable] = None
