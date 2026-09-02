"""hermes-memory-store — holographic memory plugin using MemoryProvider interface.

Registers as a MemoryProvider plugin, giving the agent structured fact storage
with entity resolution, trust scoring, and HRR-based compositional retrieval.

Original plugin by dusterbloom (PR #2351), adapted to the MemoryProvider ABC.

Config in $HERMES_HOME/config.yaml (profile-scoped):
  plugins:
    hermes-memory-store:
      db_path: $HERMES_HOME/memory_store.db   # omit to use the default
      auto_extract: false
      default_trust: 0.5
      min_trust_threshold: 0.3
      temporal_decay_half_life: 0
"""

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


def _load_plugin_config() -> dict:
    try:
        # Canonical loader: honors the managed-scope overlay + ${VAR} expansion.
        from hermes_cli.config import load_config_readonly
        all_config = load_config_readonly()
        return cfg_get(all_config, "plugins", "hermes-memory-store", default={}) or {}
    except Exception:
        return {}


def _results(results: list) -> str:
    return json.dumps({"results": results, "count": len(results)})


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
            # Raw read for the write-back round-trip: merged defaults must not
            # be persisted into the user's file.
            from hermes_cli.config import read_user_config_raw
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
        # Expand $HERMES_HOME so configured paths resolve to the active profile's directory.
        if isinstance(db_path, str):
            db_path = db_path.replace("$HERMES_HOME", _hermes_home).replace("${HERMES_HOME}", _hermes_home)
        hrr_dim = int(self._config.get("hrr_dim", 1024))

        self._store = MemoryStore(
            db_path=db_path, default_trust=float(self._config.get("default_trust", 0.5)), hrr_dim=hrr_dim,
        )
        self._retriever = FactRetriever(
            store=self._store,
            temporal_decay_half_life=int(self._config.get("temporal_decay_half_life", 0)),
            hrr_weight=float(self._config.get("hrr_weight", 0.3)),
            hrr_dim=hrr_dim,
        )
        self._session_id = session_id

    def system_prompt_block(self) -> str:
        if not self._store:
            return ""
        try:
            total = self._store._conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0]
        except Exception:
            total = 0
        if total == 0:
            return (
                "# Holographic Memory\n"
                "Active. Empty fact store — proactively add facts the user would expect you to remember.\n"
                "Use fact_store(action='add') to store durable structured facts about people, projects, preferences, decisions.\n"
                "Use fact_feedback to rate facts after using them (trains trust scores)."
            )
        return (
            f"# Holographic Memory\n"
            f"Active. {total} facts stored with entity resolution and trust scoring.\n"
            f"Use fact_store to search, probe entities, reason across entities, or add facts.\n"
            f"Use fact_feedback to rate facts after using them (trains trust scores)."
        )

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        if not self._retriever or not query:
            return ""
        try:
            results = self._retriever.search(query, min_trust=self._min_trust, limit=5)
            if not results:
                return ""
            lines = [f"- [{r.get('trust_score', r.get('trust', 0)):.1f}] {r.get('content', '')}" for r in results]
            return "## Holographic Memory\n" + "\n".join(lines)
        except Exception as e:
            logger.debug("Holographic prefetch failed: %s", e)
            return ""

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "") -> None:
        # Facts are stored explicitly via tools; on_session_end handles auto-extraction if configured.
        pass

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
        # is_truthy_value: the config schema declares auto_extract as a string
        # enum ("false"/"true"); plain truthiness would treat "false" as enabled.
        if is_truthy_value(self._config.get("auto_extract", False)) and self._store and messages:
            self._auto_extract_facts(messages)

    def on_memory_write(self, action: str, target: str, content: str) -> None:
        """Mirror built-in memory writes as facts."""
        if action == "add" and self._store and content:
            try:
                category = "user_pref" if target == "user" else "general"
                self._store.add_fact(content, category=category)
            except Exception as e:
                logger.debug("Holographic memory_write mirror failed: %s", e)

    def shutdown(self) -> None:
        # Release the shared SQLite connection on the caller's thread: leaving
        # it to GC keeps the connection (and its write lock) alive on a
        # long-running gateway. close() is idempotent and refcount-guarded.
        if self._store is not None:
            try:
                self._store.close()
            except Exception as e:
                logger.debug("Holographic shutdown close() failed: %s", e)
        self._store = None
        self._retriever = None

    # -- Tool handlers -------------------------------------------------------
    # KeyError from args[...] / Exception are turned into tool_error by handle_tool_call.

    def _handle_fact_store(self, args: dict) -> str:
        action = args["action"]
        handler = self._FACT_STORE_ACTIONS.get(action)
        if handler is None:
            return tool_error(f"Unknown action: {action}")
        return handler(self, args)

    def _act_add(self, args: dict) -> str:
        fact_id = self._store.add_fact(args["content"], category=args.get("category", "general"), tags=args.get("tags", ""))
        return json.dumps({"fact_id": fact_id, "status": "added"})

    def _act_search(self, args: dict) -> str:
        return _results(self._retriever.search(
            args["query"], category=args.get("category"),
            min_trust=float(args.get("min_trust", self._min_trust)), limit=int(args.get("limit", 10)),
        ))

    def _act_entity(self, args: dict, method: str) -> str:
        """Shared body of 'probe' and 'related' (single-entity retriever queries)."""
        return _results(getattr(self._retriever, method)(
            args["entity"], category=args.get("category"), limit=int(args.get("limit", 10)),
        ))

    def _act_reason(self, args: dict) -> str:
        entities = args.get("entities", [])
        if not entities:
            return tool_error("reason requires 'entities' list")
        return _results(self._retriever.reason(entities, category=args.get("category"), limit=int(args.get("limit", 10))))

    def _act_contradict(self, args: dict) -> str:
        return _results(self._retriever.contradict(category=args.get("category"), limit=int(args.get("limit", 10))))

    def _act_update(self, args: dict) -> str:
        updated = self._store.update_fact(
            int(args["fact_id"]), content=args.get("content"),
            trust_delta=float(args["trust_delta"]) if "trust_delta" in args else None,
            tags=args.get("tags"), category=args.get("category"),
        )
        return json.dumps({"updated": updated})

    def _act_remove(self, args: dict) -> str:
        return json.dumps({"removed": self._store.remove_fact(int(args["fact_id"]))})

    def _act_list(self, args: dict) -> str:
        facts = self._store.list_facts(
            category=args.get("category"), min_trust=float(args.get("min_trust", 0.0)), limit=int(args.get("limit", 10)),
        )
        return json.dumps({"facts": facts, "count": len(facts)})

    def _handle_fact_feedback(self, args: dict) -> str:
        return json.dumps(self._store.record_feedback(int(args["fact_id"]), helpful=args["action"] == "helpful"))

    _FACT_STORE_ACTIONS = {
        "add": _act_add, "search": _act_search,
        "probe": lambda self, args: self._act_entity(args, "probe"),
        "related": lambda self, args: self._act_entity(args, "related"),
        "reason": _act_reason, "contradict": _act_contradict,
        "update": _act_update, "remove": _act_remove, "list": _act_list,
    }
    _TOOL_HANDLERS = {"fact_store": _handle_fact_store, "fact_feedback": _handle_fact_feedback}

    # -- Auto-extraction (on_session_end) ------------------------------------

    def _auto_extract_facts(self, messages: list) -> None:
        # Local import: the compressor module is heavier than this plugin and
        # only needed when auto_extract is on.
        from agent.context_compressor import (
            _MERGED_PRIOR_CONTEXT_HEADER,
            _MERGED_SUMMARY_DELIMITER,
            is_compaction_summary_message,
        )

        extracted = 0
        for msg in messages:
            if msg.get("role") != "user":
                continue
            content = msg.get("content", "")
            # Compaction handoff summaries arrive as role="user" and reliably
            # match the decision patterns; skip them so the compactor's own
            # output is never stored as a durable fact. A merge-into-tail row
            # holds genuine prior user text BEFORE _MERGED_SUMMARY_DELIMITER
            # (prefixed with the header) and the summary AFTER it — harvest
            # only the pre-delimiter segment.
            if isinstance(content, str) and _MERGED_SUMMARY_DELIMITER in content:
                pre = content.split(_MERGED_SUMMARY_DELIMITER, 1)[0]
                if pre.startswith(_MERGED_PRIOR_CONTEXT_HEADER):
                    pre = pre[len(_MERGED_PRIOR_CONTEXT_HEADER):]
                if pre.strip():
                    content = pre.strip()
                elif is_compaction_summary_message(msg):
                    continue
            elif is_compaction_summary_message(msg):
                continue
            if not isinstance(content, str) or len(content) < 10:
                continue

            for patterns, category in ((_PREF_PATTERNS, "user_pref"), (_DECISION_PATTERNS, "project")):
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
