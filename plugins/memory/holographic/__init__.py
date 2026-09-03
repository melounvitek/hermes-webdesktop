"""hermes-memory-store — holographic memory plugin (MemoryProvider): structured fact storage with entity
resolution, trust scoring, and HRR-based compositional retrieval. Original plugin by dusterbloom (PR #2351).
Config in $HERMES_HOME/config.yaml under plugins.hermes-memory-store: db_path ($HERMES_HOME/memory_store.db),
auto_extract (false), default_trust (0.5), min_trust_threshold (0.3), temporal_decay_half_life (0),
hrr_dim (1024), hrr_weight (0.3)."""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List

from agent.memory_provider import MemoryProvider
from tools.registry import tool_error
from utils import is_truthy_value
from .store import MemoryStore
from .retrieval import FactRetriever
from hermes_cli.config import cfg_get

logger = logging.getLogger(__name__)


FACT_STORE_SCHEMA = {
    "name": "fact_store",
    "description": (
        "Deep structured memory with algebraic reasoning. "
        "Use alongside the memory tool — memory for always-on context, "
        "fact_store for deep recall and compositional queries.\n\n"
        "ACTIONS (simple → powerful):\n"
        "• add — Store a fact the user would expect you to remember.\n"
        "• search — Keyword lookup ('editor config', 'deploy process').\n"
        "• probe — Entity recall: ALL facts about a person/thing.\n"
        "• related — What connects to an entity? Structural adjacency.\n"
        "• reason — Compositional: facts connected to MULTIPLE entities simultaneously.\n"
        "• contradict — Memory hygiene: find facts making conflicting claims.\n"
        "• update/remove/list — CRUD operations.\n\n"
        "IMPORTANT: Before answering questions about the user, ALWAYS probe or reason first."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": ["add", "search", "probe", "related", "reason", "contradict", "update", "remove", "list"],
            },
            "content": {"type": "string", "description": "Fact content (required for 'add')."},
            "query": {"type": "string", "description": "Search query (required for 'search')."},
            "entity": {"type": "string", "description": "Entity name for 'probe'/'related'."},
            "entities": {"type": "array", "items": {"type": "string"}, "description": "Entity names for 'reason'."},
            "fact_id": {"type": "integer", "description": "Fact ID for 'update'/'remove'."},
            "category": {"type": "string", "enum": ["user_pref", "project", "tool", "general"]},
            "tags": {"type": "string", "description": "Comma-separated tags."},
            "trust_delta": {"type": "number", "description": "Trust adjustment for 'update'."},
            "min_trust": {"type": "number", "description": "Minimum trust filter (default: 0.3)."},
            "limit": {"type": "integer", "description": "Max results (default: 10)."},
        },
        "required": ["action"],
    },
}

FACT_FEEDBACK_SCHEMA = {
    "name": "fact_feedback",
    "description": (
        "Rate a fact after using it. Mark 'helpful' if accurate, 'unhelpful' if outdated. "
        "This trains the memory — good facts rise, bad facts sink."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "action": {"type": "string", "enum": ["helpful", "unhelpful"]},
            "fact_id": {"type": "integer", "description": "The fact ID to rate."},
        },
        "required": ["action", "fact_id"],
    },
}

# Auto-extraction patterns (on_session_end): user preferences -> user_pref, decisions -> project.
_PREF_PATTERNS = [
    re.compile(r'\bI\s+(?:prefer|like|love|use|want|need)\s+(.+)', re.IGNORECASE),
    re.compile(r'\bmy\s+(?:favorite|preferred|default)\s+\w+\s+is\s+(.+)', re.IGNORECASE),
    re.compile(r'\bI\s+(?:always|never|usually)\s+(.+)', re.IGNORECASE),
]
_DECISION_PATTERNS = [
    re.compile(r'\bwe\s+(?:decided|agreed|chose)\s+(?:to\s+)?(.+)', re.IGNORECASE),
    re.compile(r'\bthe\s+project\s+(?:uses|needs|requires)\s+(.+)', re.IGNORECASE),
]
_EXTRACT_CATEGORIES = ((_PREF_PATTERNS, "user_pref"), (_DECISION_PATTERNS, "project"))


def _load_plugin_config() -> dict:
    try:
        # Canonical loader: honors the managed-scope overlay + ${VAR} expansion.
        from hermes_cli.config import load_config_readonly
        return cfg_get(load_config_readonly(), "plugins", "hermes-memory-store", default={}) or {}
    except Exception:
        return {}


def _results(items: list, key: str = "results") -> str:
    return json.dumps({key: items, "count": len(items)})


def _limit(args: dict) -> int:
    return int(args.get("limit", 10))


def _tool_handler(actions: dict):
    """Return a bound-style handler dispatching on args["action"] over ``actions`` (unknown -> tool_error)."""
    def handle(self, args: dict) -> str:
        action = args["action"]
        handler = actions.get(action)
        return handler(self, args) if handler is not None else tool_error(f"Unknown action: {action}")
    return handle


class HolographicMemoryProvider(MemoryProvider):
    """Holographic memory with structured facts, entity resolution, and HRR retrieval."""

    def __init__(self, config: dict | None = None):
        self._config = config or _load_plugin_config()
        self._store = None
        self._retriever = None
        self._min_trust = float(self._config.get("min_trust_threshold", 0.3))

    @property
    def name(self) -> str:
        return "holographic"

    def is_available(self) -> bool:
        return True  # SQLite is always available, numpy is optional

    def save_config(self, values, hermes_home):
        """Write config to config.yaml under plugins.hermes-memory-store."""
        config_path = Path(hermes_home) / "config.yaml"
        try:
            import yaml
            from hermes_cli.config import read_user_config_raw  # raw read: merged defaults must not be persisted
            existing = read_user_config_raw(config_path)
            existing.setdefault("plugins", {})["hermes-memory-store"] = values
            with open(config_path, "w", encoding="utf-8") as f:
                yaml.dump(existing, f, default_flow_style=False)
        except Exception:
            pass

    def get_config_schema(self):
        from hermes_constants import display_hermes_home
        _default_db = f"{display_hermes_home()}/memory_store.db"
        return [
            {"key": "db_path", "description": "SQLite database path", "default": _default_db},
            {"key": "auto_extract", "description": "Auto-extract facts at session end", "default": "false", "choices": ["true", "false"]},
            {"key": "default_trust", "description": "Default trust score for new facts", "default": "0.5"},
            {"key": "hrr_dim", "description": "HRR vector dimensions", "default": "1024"},
        ]

    def initialize(self, session_id: str, **kwargs) -> None:
        from hermes_constants import get_hermes_home
        _hermes_home = str(get_hermes_home())
        db_path = self._config.get("db_path", _hermes_home + "/memory_store.db")
        if isinstance(db_path, str):  # expand $HERMES_HOME so paths resolve to the active profile
            db_path = db_path.replace("$HERMES_HOME", _hermes_home).replace("${HERMES_HOME}", _hermes_home)
        hrr_dim = int(self._config.get("hrr_dim", 1024))
        self._store = MemoryStore(db_path=db_path, default_trust=float(self._config.get("default_trust", 0.5)), hrr_dim=hrr_dim)
        self._retriever = FactRetriever(
            store=self._store, temporal_decay_half_life=int(self._config.get("temporal_decay_half_life", 0)),
            hrr_weight=float(self._config.get("hrr_weight", 0.3)), hrr_dim=hrr_dim,
        )
        self._session_id = session_id

    def system_prompt_block(self) -> str:
        if not self._store:
            return ""
        try:
            total = self._store._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        except Exception:
            total = 0
        body = (
            "Active. Empty fact store — proactively add facts the user would expect you to remember.\n"
            "Use fact_store(action='add') to store durable structured facts about people, projects, preferences, decisions.\n"
        ) if total == 0 else (
            f"Active. {total} facts stored with entity resolution and trust scoring.\n"
            "Use fact_store to search, probe entities, reason across entities, or add facts.\n"
        )
        return "# Holographic Memory\n" + body + "Use fact_feedback to rate facts after using them (trains trust scores)."

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._retriever or not query:
            return ""
        try:
            results = self._retriever.search(query, min_trust=self._min_trust, limit=5)
            lines = [f"- [{r.get('trust_score', r.get('trust', 0)):.1f}] {r.get('content', '')}" for r in results]
            return "## Holographic Memory\n" + "\n".join(lines) if results else ""
        except Exception as e:
            logger.debug("Holographic prefetch failed: %s", e)
            return ""

    def get_tool_schemas(self) -> List[Dict[str, Any]]:
        return [FACT_STORE_SCHEMA, FACT_FEEDBACK_SCHEMA]

    def handle_tool_call(self, tool_name: str, args: Dict[str, Any], **kwargs) -> str:
        handler = self._TOOL_HANDLERS.get(tool_name)
        if handler is None:
            return tool_error(f"Unknown tool: {tool_name}")
        try:
            return handler(self, args)
        except KeyError as exc:
            return tool_error(f"Missing required argument: {exc}")
        except Exception as exc:
            return tool_error(str(exc))

    def on_session_end(self, messages: List[Dict[str, Any]]) -> None:
        # is_truthy_value: the config schema declares auto_extract as a string enum
        # ("false"/"true"); plain truthiness would treat "false" as enabled.
        if is_truthy_value(self._config.get("auto_extract", False)) and self._store and messages:
            self._auto_extract_facts(messages)

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes as facts."""
        if action == "add" and self._store and content:
            try:
                self._store.add_fact(content, category="user_pref" if target == "user" else "general")
            except Exception as e:
                logger.debug("Holographic memory_write mirror failed: %s", e)

    def shutdown(self) -> None:
        # Release the shared SQLite connection on the caller's thread: leaving it to GC keeps
        # the connection (and its write lock) alive on a long-running gateway. close() is idempotent.
        if self._store is not None:
            try:
                self._store.close()
            except Exception as e:
                logger.debug("Holographic shutdown close() failed: %s", e)
        self._store = None
        self._retriever = None

    # -- Tool handlers: KeyError from args[...] / Exception -> tool_error in handle_tool_call. ---
    # Argument coercion order (and therefore which error surfaces first) mirrors the call order.

    def _entity_query(self, method: str, a: dict) -> str:
        """'probe' / 'related': single-entity retriever queries."""
        return _results(getattr(self._retriever, method)(a["entity"], category=a.get("category"), limit=_limit(a)))

    def _act_reason(self, a: dict) -> str:
        entities = a.get("entities", [])
        if not entities:
            return tool_error("reason requires 'entities' list")
        return _results(self._retriever.reason(entities, category=a.get("category"), limit=_limit(a)))

    def _act_update(self, a: dict) -> str:
        updated = self._store.update_fact(
            int(a["fact_id"]), content=a.get("content"),
            trust_delta=float(a["trust_delta"]) if "trust_delta" in a else None,
            tags=a.get("tags"), category=a.get("category"),
        )
        return json.dumps({"updated": updated})

    _TOOL_HANDLERS = {
        "fact_store": _tool_handler({
            "add": lambda self, a: json.dumps({"fact_id": self._store.add_fact(
                a["content"], category=a.get("category", "general"), tags=a.get("tags", "")), "status": "added"}),
            "search": lambda self, a: _results(self._retriever.search(
                a["query"], category=a.get("category"), min_trust=float(a.get("min_trust", self._min_trust)), limit=_limit(a))),
            "probe": lambda self, a: self._entity_query("probe", a),
            "related": lambda self, a: self._entity_query("related", a),
            "reason": _act_reason,
            "contradict": lambda self, a: _results(self._retriever.contradict(category=a.get("category"), limit=_limit(a))),
            "update": _act_update,
            "remove": lambda self, a: json.dumps({"removed": self._store.remove_fact(int(a["fact_id"]))}),
            "list": lambda self, a: _results(self._store.list_facts(
                category=a.get("category"), min_trust=float(a.get("min_trust", 0.0)), limit=_limit(a)), key="facts"),
        }),
        "fact_feedback": lambda self, a: json.dumps(self._store.record_feedback(int(a["fact_id"]), helpful=a["action"] == "helpful")),
    }

    @staticmethod
    def _harvestable_text(msg: dict):
        """User text eligible for extraction, or None. Compaction handoff summaries arrive as role="user" and
        reliably match the decision patterns; never store the compactor's own output as a durable fact. A
        merge-into-tail row holds genuine prior user text BEFORE _MERGED_SUMMARY_DELIMITER (prefixed with the
        header) and the summary AFTER it — harvest only the pre-delimiter segment."""
        # Local import: the compressor module is heavier than this plugin and only needed here.
        from agent.context_compressor import _MERGED_PRIOR_CONTEXT_HEADER, _MERGED_SUMMARY_DELIMITER, is_compaction_summary_message

        content = msg.get("content", "")
        if isinstance(content, str) and _MERGED_SUMMARY_DELIMITER in content:
            pre = content.split(_MERGED_SUMMARY_DELIMITER, 1)[0]
            if pre.startswith(_MERGED_PRIOR_CONTEXT_HEADER):
                pre = pre[len(_MERGED_PRIOR_CONTEXT_HEADER):]
            if pre.strip():
                content = pre.strip()
            elif is_compaction_summary_message(msg):
                return None
        elif is_compaction_summary_message(msg):
            return None
        return content if isinstance(content, str) and len(content) >= 10 else None

    def _auto_extract_facts(self, messages: list) -> None:
        extracted = 0
        texts = filter(None, (self._harvestable_text(m) for m in messages if m.get("role") == "user"))
        for content in texts:
            for patterns, category in _EXTRACT_CATEGORIES:
                if any(p.search(content) for p in patterns):
                    try:
                        self._store.add_fact(content[:400], category=category)
                        extracted += 1
                    except Exception:
                        pass
        if extracted:
            logger.info("Auto-extracted %d facts from conversation", extracted)


def register(ctx) -> None:
    """Register the holographic memory provider with the plugin system."""
    ctx.register_memory_provider(HolographicMemoryProvider(config=_load_plugin_config()))
