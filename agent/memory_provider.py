"""Abstract base class for pluggable memory providers.

Plugins ship in ``plugins/memory/<name>/`` and are activated via ``memory.provider``;
MemoryManager allows only ONE external provider at a time. Lifecycle is driven by
MemoryManager: initialize -> system_prompt_block / prefetch / sync_turn per turn ->
tool dispatch -> shutdown, plus the optional ``on_*`` hooks below.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# v1 = historical implicit contract (best-effort on_pre_compress() with the raw
# message list); v2 = opt-in fail-closed checkpoint (normalized evidence handoff
# + strict-mode failure propagation).
PRE_COMPRESS_CHECKPOINT_API_VERSION = 2

# Default glyph for recall indicators; providers may use their own brand mark.
INDICATOR_GLYPH = "🧠"


@dataclass(frozen=True)
class RecallStatus:
    """What the last prefetch injected, for the deterministic recall indicator
    (``MemoryManager.describe_recall``). ``count == 0`` means content without a
    discrete count (e.g. a synthesized reflect answer) and renders generically."""

    provider_label: str
    count: int
    glyph: str = INDICATOR_GLYPH


# Prompts with no semantic signal. Single source of truth for the core prefetch
# gate (turn_context.py, run_agent.py) and provider-side classifiers (honcho).
# Anchored and followed only by whitespace/punctuation, so "k8s"/"yolo"/"note"
# do NOT match while "hi!"/"thanks :)"/"done???" do.
TRIVIAL_PROMPT_RE = re.compile(
    r'^(yes|no|ok|okay|sure|thanks|thank you|y|n|yep|nope|yeah|nah|'
    r'hi|hey|hello|yo|sup|'
    r'continue|go ahead|do it|proceed|got it|cool|nice|great|done|next|lgtm|k)'
    r'[\s!?.:;,"' + "'" + r'~\u2018\u2019\u201c\u201d\u2014\u2013\u2026()\[\]{}<>*&^%$#@!+=`\u00a0]*$',
    re.IGNORECASE,
)


def is_trivial_prompt(text: Optional[str]) -> bool:
    """True for empty input, slash commands and bare greetings/acknowledgements.

    Skipping recall on these saves a blocking network round-trip and keeps
    stale user-model context from derailing one-word replies.
    """
    stripped = (text or "").strip()
    if not stripped or stripped.startswith("/"):
        return True
    return bool(TRIVIAL_PROMPT_RE.match(stripped))


class MemoryProvider(ABC):
    """Abstract base class for memory providers."""

    # Providers that durably checkpoint every successful on_pre_compress() opt
    # in by setting PRE_COMPRESS_CHECKPOINT_API_VERSION; 1 = best-effort legacy.
    pre_compress_checkpoint_api_version = 1

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for this provider (e.g. 'builtin', 'honcho', 'hindsight')."""

    # -- Core lifecycle (implement these) ------------------------------------

    @abstractmethod
    def is_available(self) -> bool:
        """Configured, credentialed and ready? Gates activation at agent init;
        check config/deps only — no network calls."""

    @abstractmethod
    def initialize(self, session_id: str, **kwargs) -> None:
        """Initialize once at agent startup (connections, resources, threads).

        kwargs always include ``hermes_home`` (use it for profile-scoped storage,
        never hardcode ``~/.hermes``) and ``platform``. May include
        ``agent_context`` ("primary" | "subagent" | "cron" | "flush" — skip
        writes for non-primary contexts, cron prompts would corrupt user
        representations), ``agent_identity`` (profile name), ``agent_workspace``,
        ``parent_session_id``, ``user_id``, ``user_id_alt``.
        """

    def unavailable_reason(self) -> str:
        """Short user-facing hint for the "provider unavailable" warning (e.g.
        which package to install); ``initialize()`` never runs when unavailable."""
        return ""

    def system_prompt_block(self) -> str:
        """STATIC system-prompt text (instructions, status); "" to skip.
        Recalled context goes through prefetch(), not here."""
        return ""

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        """Formatted recall context for the upcoming turn ("" if none).

        Must be fast — do the recall in the background and return cached
        results. ``session_id`` scopes concurrent sessions (gateway, cached agents).
        """
        return ""

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        """Queue a background recall after each turn; prefetch() consumes it next turn."""

    def recall_status(self) -> Optional[RecallStatus]:
        """What the most recent :meth:`prefetch` injected, for a deterministic
        "recalled N memories" indicator. ``None`` = nothing / no indicator.
        Must reflect only the LAST prefetch, never a stale prior count."""
        return None

    def sync_turn(
        self,
        user_content: str,
        assistant_content: str,
        *,
        session_id: str = "",
        messages: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        """Persist a completed turn; should be non-blocking. ``messages`` is the
        OpenAI-style list as of this turn, including tool calls/results."""

    @abstractmethod
    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        """OpenAI function-calling schemas ({"name", "description", "parameters"});
        [] for context-only providers."""

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        """Handle one of this provider's tools; must return a JSON string."""
        raise NotImplementedError(f"Provider {self.name} does not handle tool {tool_name}")

    def shutdown(self) -> None:
        """Clean shutdown — flush queues, close connections."""

    # -- Optional hooks (override to opt in) ---------------------------------

    def on_turn_start(self, turn_number: int, message: str, **kwargs) -> None:
        """Per-turn tick (turn-counting, scope management, maintenance).
        kwargs may include remaining_tokens, model, platform, tool_count."""

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        """End-of-session extraction over the full history. Fires only at real
        session boundaries (CLI exit, /reset, gateway expiry), never per-turn."""

    def on_session_switch(
        self,
        new_session_id: str,
        *,
        parent_session_id: str = "",
        reset: bool = False,
        rewound: bool = False,
        **kwargs,
    ) -> None:
        """session_id reassigned mid-process (/resume, /branch, /reset, /new,
        gateway equivalents, context compression) without a provider teardown.

        Update or reset per-session state cached in ``initialize()`` so later
        writes land in the right record. ``parent_session_id`` carries lineage
        ("" when none). ``reset`` is True only for a genuinely new conversation
        (/reset, /new) — flush per-session buffers. ``rewound``: same id but the
        transcript was truncated, so invalidate per-turn document state.
        """

    def on_pre_compress(self, messages: List[Dict[str, Any]]) -> str:
        """Extract insights from ``messages`` about to be compressed; the returned
        text is fed into the compression summary prompt ("" = nothing)."""
        return ""

    def on_delegation(self, task: str, result: str, *,
                      child_session_id: str = "", **kwargs) -> None:
        """PARENT-side observation of a completed delegation (task prompt + final
        result); the subagent itself has no provider session (skip_memory=True)."""

    def get_config_schema(self) -> List[Dict[str, Any]]:
        """Setup fields for ``hermes memory setup`` ([] if none).

        Each field: ``key``, ``description``, optional ``secret`` (goes to .env),
        ``required``, ``default``, ``choices``, ``type`` (text | integer |
        number | boolean), ``minimum`` / ``maximum`` / ``step`` (numeric),
        ``url`` (where to get the credential), ``env_var`` (explicit secret env
        var; default auto-generated).
        """
        return []

    def save_config(self, values: Dict[str, Any], hermes_home: str) -> None:
        """Write non-secret setup ``values`` (secrets go to .env) to the provider's
        native config location. Plugins MUST either override this or use only
        env vars (every schema field carrying ``env_var``) and keep the no-op."""

    def on_memory_write(
        self,
        action: str,
        target: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Mirror a built-in memory-tool write. ``action`` is add | replace |
        remove, ``target`` is memory | user; ``metadata`` (when available) has
        provenance such as write_origin, execution_context, session_id,
        parent_session_id, platform, tool_name."""

    def backup_paths(self) -> List[str]:
        """Absolute paths of provider state OUTSIDE HERMES_HOME (e.g. ``~/.honcho``)
        so ``hermes backup``/``hermes import`` can capture and restore them; paths
        outside the home dir are skipped. MUST work without ``initialize()`` or
        network — resolve from config/env."""
        return []
