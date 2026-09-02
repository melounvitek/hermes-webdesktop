"""Progressive tool disclosure ("tool search"): MCP/plugin tools (and a
curated set of event-triggered core tools) are replaced in the model-visible
array by three bridge tools — tool_search / tool_describe / tool_call.

Invariants:
* Working-set core tools (``toolsets._HERMES_CORE_TOOLS``) and session-gated GUI
  toolsets never defer unless named in the ``defer`` list.
* Tiered disclosure: ANY deferrable tool activates the bridge (tier 1 = bridge +
  listing that fits ``min(threshold_pct% of context, listing_max_tokens)``,
  tier 2 = bare bridge + per-server summary). The listing scales, not activation.
* The catalog is stateless — rebuilt from the live tool-defs list on every
  assembly. A session-keyed catalog drifts from the registry and silently
  drops tools.
* Bridge calls route through ``model_tools.handle_function_call`` so guardrails,
  hooks, approvals, and truncation fire identically; display/trajectory unwrap
  always shows the underlying tool.
"""

from __future__ import annotations

import json
import logging
import math
from collections import Counter
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Literal, Optional, Tuple

from tools.registry import tool_error
from tools.tool_search_names import (  # noqa: F401 — re-exported public names
    BRIDGE_TOOL_NAMES,
    TOOL_CALL_NAME,
    TOOL_DESCRIBE_NAME,
    TOOL_SEARCH_NAME,
)
from tools.tool_search_catalog import (  # noqa: F401 — re-exported public/test names
    CHARS_PER_TOKEN,
    CatalogEntry,
    _classify_source,
    _corpus_stats,
    _entry_search_text,
    _listing_group_label,
    _short_desc,
    _stem,
    _tokenize,
    build_catalog,
    build_catalog_listing_with_form,
    search_catalog,
)

from tools.tool_search_validation import (  # noqa: F401 — re-exported public/test names
    _schema_for_local_validation,
    _schema_has_external_ref,
    validate_deferred_call_args,
)

logger = logging.getLogger("tools.tool_search")

# Bound the work one tool_search bridge call can request.
_MAX_QUERIES_PER_CALL = 10
# Bound the work one tool_describe bridge call can request.
_MAX_DESCRIBE_NAMES_PER_CALL = 10


# ---------------------------------------------------------------------------
# Configuration plumbing
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ToolSearchConfig:
    """Resolved, validated tool-search configuration for a single assembly."""

    enabled: str  # "auto" | "on" | "off" — "auto" is an alias of "on" today
    # Listing budget as % of context. Does NOT gate activation; bounds how much
    # the embedded listing may consume before it degrades (full -> names -> bare).
    threshold_pct: float  # 0..100
    search_default_limit: int
    max_search_limit: int
    # Embedded name + short-description manifest so deferred tools stay
    # discoverable. "auto"/"on" = include when it fits; "off" = bare bridge.
    listing: str = "auto"  # "auto" | "on" | "off"
    # Effective budget = min(listing_max_tokens, threshold_pct% of context).
    listing_max_tokens: int = 4000
    # Core/GUI names deferred behind the bridge. None = curated default; an
    # explicit config list replaces it wholesale ([] = defer no core tools).
    defer_tools: Optional[frozenset] = None

    @property
    def effective_defer_tools(self) -> frozenset:
        return _DEFAULT_DEFERRED_TOOLS if self.defer_tools is None else self.defer_tools

    @classmethod
    def from_raw(cls, raw: Any) -> "ToolSearchConfig":
        """Build a config from a raw dict / legacy bool / None. Every field is
        validated and clamped; unknown values fall back to safe defaults rather
        than raising, so a typo in user config cannot break the agent."""
        if not isinstance(raw, dict):
            return cls(enabled="off" if raw is False else "auto", threshold_pct=5.0,
                       search_default_limit=5, max_search_limit=25)

        enabled = _tri_state(raw.get("enabled", "auto"))
        threshold_pct = max(0.0, min(100.0, _safe_float(raw.get("threshold_pct"), 5.0)))

        max_search_limit = max(1, min(50, _safe_int(raw.get("max_search_limit"), 25)))
        search_default_limit = max(1, min(max_search_limit,
                                          _safe_int(raw.get("search_default_limit"), 5)))

        listing = _tri_state(raw.get("listing", "auto"))
        listing_max_tokens = max(200, min(60000, _safe_int(raw.get("listing_max_tokens"), 4000)))

        defer_raw = raw.get("defer")
        if isinstance(defer_raw, (list, tuple, set)):
            defer_tools = frozenset(
                str(n).strip() for n in defer_raw if str(n).strip()
            )
        else:
            defer_tools = None  # curated default

        return cls(
            enabled=enabled,
            threshold_pct=threshold_pct,
            search_default_limit=search_default_limit,
            max_search_limit=max_search_limit,
            listing=listing,
            listing_max_tokens=listing_max_tokens,
            defer_tools=defer_tools,
        )


_TRI_STATE_ALIASES = {"true": "on", "1": "on", "yes": "on", "false": "off", "0": "off", "no": "off"}


def _tri_state(value: Any) -> str:
    """Normalize an ``auto``/``on``/``off`` setting (bool-ish aliases accepted)."""
    text = str(value).strip().lower()
    text = _TRI_STATE_ALIASES.get(text, text)
    return text if text in ("auto", "on", "off") else "auto"


def _safe_int(value: Any, fallback: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback


def _safe_float(value: Any, fallback: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return fallback


def _config_from_loader(loader_name: str) -> ToolSearchConfig:
    try:
        import hermes_cli.config as _cfg_mod
        cfg = getattr(_cfg_mod, loader_name)() or {}
        tools_cfg = cfg.get("tools") if isinstance(cfg.get("tools"), dict) else {}
        return ToolSearchConfig.from_raw(tools_cfg.get("tool_search"))
    except Exception as e:
        logger.debug("Failed to load tool-search config: %s", e)
        return ToolSearchConfig.from_raw(None)


def load_config() -> ToolSearchConfig:
    """Load tool-search config from the user config file."""
    return _config_from_loader("load_config")


def load_config_readonly() -> ToolSearchConfig:
    """Load tool-search config without copying the cached full config."""
    return _config_from_loader("load_config_readonly")


# ---------------------------------------------------------------------------
# Tool classification
# ---------------------------------------------------------------------------


def _core_tool_names() -> frozenset[str]:
    """Tool names that never defer by default (lazy import: ``toolsets`` imports
    ``tools.registry``, so a module-level import would be a cycle)."""
    try:
        from toolsets import _HERMES_CORE_TOOLS
        return frozenset(_HERMES_CORE_TOOLS)
    except Exception:
        return frozenset()


# Session-gated GUI toolsets: off ``_HERMES_CORE_TOOLS`` so non-GUI clients never
# pay their schema; once enabled they stay direct unless the deferral list names them.
_DIRECT_SURFACE_TOOLSETS = frozenset({"desktop_ui", "project"})

# Curated event-triggered core tools deferred BY DEFAULT — reached for when
# something specific happens, not every-turn working-set tools, so a catalog
# stub suffices. ``tools.tool_search.defer`` replaces this list wholesale
# ([] = legacy everything-eager). Names are POST-rename.
# ``clarify`` is deliberately NOT here: A/B showed deferring it collapsed
# structured-clarify usage (18/18 -> 7/18) — the ask-the-user affordance must
# be ambient to fire; a stub is not enough.
_DEFAULT_DEFERRED_TOOLS = frozenset({
    "computer_use", "session_search", "image_generate",
    "todo_list", "process_manage", "cronjob_manage",
    # Desktop GUI surface (desktop_ui + project toolsets)
    "drive_preview", "gui_tour", "desktop_preview", "annotate_preview",
    "show_tip", "setup_mcp", "desktop_project", "close_terminal",
    "apply_layout", "read_terminal", "read_window_below", "focus_pane",
})


def is_deferrable_tool_name(name: str, defer_tools: Optional[frozenset] = None) -> bool:
    """True if a tool is *eligible* for deferral: named in ``defer_tools``
    (curated core set or user override), OR an MCP tool, OR neither core nor a
    session-gated GUI surface (i.e. a plugin tool). Bridge names never defer."""
    if name in BRIDGE_TOOL_NAMES:
        return False
    if defer_tools is not None and name in defer_tools:
        return True
    if name in _core_tool_names():
        return False
    try:
        from tools.registry import registry
        entry = registry.get_entry(name)
        if entry is None:
            return False
        return entry.toolset.startswith("mcp-") or entry.toolset not in _DIRECT_SURFACE_TOOLSETS
    except Exception:
        return False


def _describe_classification(
    name: str,
    defer_tools: Optional[frozenset] = None,
) -> Literal["available", "not_found", "not_deferrable"]:
    """Classify a describe name without treating unknown names as errors."""
    try:
        from tools.registry import registry
        entry = registry.get_entry(name)
    except Exception:
        return "not_found"
    if entry is None:
        return "not_found"
    if defer_tools is not None and name in defer_tools:
        return "available"
    if (
        name in BRIDGE_TOOL_NAMES
        or name in _core_tool_names()
        or entry.toolset in _DIRECT_SURFACE_TOOLSETS
    ):
        return "not_deferrable"
    return "available"


def classify_tools(
    tool_defs: List[Dict[str, Any]],
    defer_tools: Optional[frozenset] = None,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split a tool-defs list into (visible, deferrable). Bridge tools are
    dropped (they are re-added after classification)."""
    visible: List[Dict[str, Any]] = []
    deferrable: List[Dict[str, Any]] = []
    for td in tool_defs:
        fn = td.get("function") or {}
        name = fn.get("name", "")
        if name in BRIDGE_TOOL_NAMES:
            continue
        if is_deferrable_tool_name(name, defer_tools):
            deferrable.append(td)
        else:
            visible.append(td)
    return visible, deferrable


# ---------------------------------------------------------------------------
# Token estimation and threshold gate
# ---------------------------------------------------------------------------


def estimate_tokens_from_schemas(tool_defs: Iterable[Dict[str, Any]]) -> int:
    """Token cost of a tool-defs list via the chars/4 rule (order-of-magnitude
    precision is all the activation gate needs)."""
    total_chars = 0
    for td in tool_defs:
        try:
            total_chars += len(json.dumps(td, ensure_ascii=False, separators=(",", ":")))
        except (TypeError, ValueError):
            total_chars += len(str(td))
    return int(math.ceil(total_chars / CHARS_PER_TOKEN))


def should_activate(
    config: ToolSearchConfig,
    deferrable_tokens: int,
    context_length: Optional[int],
) -> bool:
    """``"off"`` never activates; ``"on"``/``"auto"`` activate whenever any
    deferrable tool exists. ``"auto"`` is an alias of ``"on"`` reserved for a
    future budget-gated mode — do not distinguish them without that design.
    ``context_length`` is kept in the signature for caller compatibility; the
    threshold governs the listing budget, not activation."""
    return config.enabled != "off" and deferrable_tokens > 0


def listing_token_budget(
    config: ToolSearchConfig,
    context_length: Optional[int],
) -> int:
    """``min(listing_max_tokens, threshold_pct% of context)``; unknown context
    uses a 10K percentage leg (5% of a typical 200K window)."""
    if context_length and context_length > 0:
        pct_leg = int(context_length * (config.threshold_pct / 100.0))
    else:
        pct_leg = 10_000
    return max(0, min(config.listing_max_tokens, pct_leg))


# ---------------------------------------------------------------------------
# Catalog + BM25 retrieval
# ---------------------------------------------------------------------------


def bridge_tool_schemas(
    deferred_count: int,
    listing: Optional[str] = None,
    listing_form: str = "",
) -> List[Dict[str, Any]]:
    """Bridge tool schemas injected in place of deferred tools. Kept short —
    every byte is paid on every turn. ``listing`` is embedded in the
    tool_search description; ``listing_form`` picks the framing (per-tool forms
    say "skip search when you see the exact name", the "groups" summary says
    which domains exist and that search is mandatory)."""
    desc_search = (
        f"Search {deferred_count} additional tools that are loaded on demand. "
        "Takes a list of queries searched in parallel against the same "
        "catalog; send one query per distinct capability you need. Returns "
        "matching tool names grouped per query plus a shared map with each "
        "tool's description. Follow with "
        f"`{TOOL_DESCRIBE_NAME}` to load full parameter schemas, "
        f"then `{TOOL_CALL_NAME}` to invoke. Tools listed at the top of this "
        "system prompt are already available and do not need to be searched."
    )
    if listing and listing_form == "groups":
        desc_search += (
            "\n\nThe servers below are connected and their tools ARE available "
            "through this bridge. For any request in these domains, search "
            "here FIRST — do not claim the capability is unavailable and do "
            "not substitute a generic tool (terminal/browser) without "
            "searching.\n\n" + listing
        )
    elif listing:
        desc_search += (
            "\n\nEvery deferred capability is listed below. If a tool name "
            "appears here, do NOT claim it is unavailable — load it with "
            f"`{TOOL_DESCRIBE_NAME}` (skip `{TOOL_SEARCH_NAME}` when you "
            "already see the exact name)."
        )
        if listing_form == "mixed":
            desc_search += (
                " For servers marked 'names not listed', the tools exist "
                f"too — find them with `{TOOL_SEARCH_NAME}` before "
                "concluding anything is missing."
            )
        desc_search += "\n\n" + listing
    desc_describe = (
        f"Load the full JSON schemas for tools returned by `{TOOL_SEARCH_NAME}`. "
        f"Required before `{TOOL_CALL_NAME}` if a tool's parameters are unknown. "
        "Batch every schema you need into one call."
    )
    desc_call = (
        "Invoke a deferred tool by name with the given arguments. Argument shape "
        f"matches the tool's schema (see `{TOOL_DESCRIBE_NAME}`). Policy, hooks, "
        "and approvals run exactly as for any directly-listed tool."
    )

    return [
        {
            "type": "function",
            "function": {
                "name": TOOL_SEARCH_NAME,
                "description": desc_search,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "queries": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Search queries, each a few keywords describing one capability (e.g. ['create github issue', 'send slack message']). Searched in parallel; results come back grouped per query. A single string is accepted and treated as one query.",
                        },
                        "limit": {
                            "type": "integer",
                            "description": "Maximum number of matches per query. Defaults to 5 and is clamped to the configured maximum (25 by default).",
                        },
                    },
                    "required": ["queries"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": TOOL_DESCRIBE_NAME,
                "description": desc_describe,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "names": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Exact tool names (as returned by tool_search). A single string is accepted and treated as one name.",
                        },
                    },
                    "required": ["names"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": TOOL_CALL_NAME,
                "description": desc_call,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": "Exact tool name to invoke.",
                        },
                        "arguments": {
                            "type": "object",
                            "description": "Arguments for the tool, matching its schema.",
                        },
                    },
                    "required": ["name", "arguments"],
                },
            },
        },
    ]


# ---------------------------------------------------------------------------
# Public entry point: assemble tool-defs with optional tool search
# ---------------------------------------------------------------------------


@dataclass
class AssemblyResult:
    """Outcome of one assembly. Useful for tests and observability."""

    tool_defs: List[Dict[str, Any]]
    activated: bool
    deferred_count: int = 0
    deferred_tokens: int = 0
    threshold_tokens: int = 0
    # 0 = passthrough; 1 = bridge + per-tool listing (full/names/mixed);
    # 2 = bare bridge / server-summary only.
    tier: int = 0
    listing_form: str = "none"  # "full" | "names" | "mixed" | "groups" | "none"


def assemble_tool_defs(
    tool_defs: List[Dict[str, Any]],
    *,
    context_length: Optional[int] = None,
    config: Optional[ToolSearchConfig] = None,
) -> AssemblyResult:
    """Return the tool-defs list the model should actually see: passthrough
    when inactive, otherwise deferrable tools replaced by the three bridge
    tools. Idempotent — bridge tools already present are stripped first."""
    if config is None:
        config = load_config()

    incoming = [td for td in tool_defs
                if (td.get("function") or {}).get("name") not in BRIDGE_TOOL_NAMES]

    visible, deferrable = classify_tools(incoming, config.effective_defer_tools)
    if not deferrable:
        return AssemblyResult(tool_defs=incoming, activated=False)

    deferrable_tokens = estimate_tokens_from_schemas(deferrable)
    if not should_activate(config, deferrable_tokens, context_length):
        return AssemblyResult(
            tool_defs=incoming,
            activated=False,
            deferred_count=len(deferrable),
            deferred_tokens=deferrable_tokens,
            threshold_tokens=int((context_length or 0) * (config.threshold_pct / 100.0)),
            tier=0,
        )

    listing = None
    listing_form = "none"
    listing_budget = listing_token_budget(config, context_length)
    if config.listing != "off":
        listing, listing_form = build_catalog_listing_with_form(
            deferrable, max_tokens=listing_budget)
    bridge = bridge_tool_schemas(len(deferrable), listing=listing,
                                 listing_form=listing_form)
    result = visible + bridge
    tier = 1 if listing_form in ("full", "names", "mixed") else 2

    logger.info(
        "tool_search activated (tier %d): %d core/visible tools kept, %d deferred "
        "(~%d tokens), listing %s (budget ~%d tokens)",
        tier, len(visible), len(deferrable), deferrable_tokens,
        listing_form, listing_budget,
    )

    return AssemblyResult(
        tool_defs=result,
        activated=True,
        deferred_count=len(deferrable),
        deferred_tokens=deferrable_tokens,
        threshold_tokens=listing_budget,
        tier=tier,
        listing_form=listing_form,
    )


# ---------------------------------------------------------------------------
# Bridge tool dispatch
# ---------------------------------------------------------------------------


def is_bridge_tool(name: str) -> bool:
    return name in BRIDGE_TOOL_NAMES


def _shared_tool_record(entry: CatalogEntry) -> Dict[str, Any]:
    """One record for the response's shared ``tools`` map (held once per tool;
    per-query groups carry names only). ``required`` lets the model attempt a
    trivial call without a ``tool_describe`` round-trip."""
    schema = entry.schema if isinstance(entry.schema, dict) else {}
    fn = schema.get("function")
    if not isinstance(fn, dict):
        fn = {}
    params = fn.get("parameters")
    if not isinstance(params, dict):
        params = {}
    required = params.get("required")
    if not isinstance(required, list):
        required = []
    return {
        "source": entry.source,
        "source_name": entry.source_name,
        "description": (entry.description or "")[:400],  # cap chatty MCP descriptions
        "required": [r[:64] for r in required if isinstance(r, str)][:32],
    }


def _available_source_summary(catalog: List[CatalogEntry]) -> List[Dict[str, Any]]:
    """Deterministic ``[{name, tool_count}]`` of connected sources, attached to
    empty query groups so a lexical miss is not read as a missing capability."""
    counts = Counter(_listing_group_label(entry.source_name) for entry in catalog)
    return [{"name": name, "tool_count": counts[name]} for name in sorted(counts)]


def _string_list(raw: Any, *, dedupe: bool) -> Optional[List[str]]:
    """Normalize a list-of-strings argument. A bare string (a common model slip)
    counts as a one-item list; non-list input returns None."""
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return None
    out: List[str] = []
    for item in raw:
        text = str(item or "").strip()
        if text and (not dedupe or text not in out):
            out.append(text)
    return out


def dispatch_tool_search(args: Dict[str, Any],
                         *,
                         current_tool_defs: List[Dict[str, Any]],
                         config: Optional[ToolSearchConfig] = None) -> str:
    """Execute the ``tool_search`` bridge tool. Returns JSON::

        {"queries": [...], "total_available": N,
         "results": [{"query": ..., "matches": [names...]}, ...],
         "tools": {name: {"source", "source_name", "description", "required"}}}

    ``limit`` applies PER QUERY. Empty query groups get ``available_sources`` +
    ``hint`` so a lexical miss is not mistaken for a missing capability.
    """
    if config is None:
        config = load_config()

    queries = _string_list(args.get("queries"), dedupe=False)
    if queries is None:
        return tool_error("queries is required and must be an array of strings")
    if not queries:
        return tool_error("queries is required and must contain at least one non-empty string")
    if len(queries) > _MAX_QUERIES_PER_CALL:
        return tool_error(
            f"too many queries: {len(queries)} > max {_MAX_QUERIES_PER_CALL}. "
            "Retry with fewer, more targeted queries."
        )

    raw_limit = args.get("limit")
    if raw_limit is None:
        limit = config.search_default_limit
    else:
        limit = max(1, min(config.max_search_limit, _safe_int(raw_limit, config.search_default_limit)))

    _, deferrable = classify_tools(
        current_tool_defs, load_config_readonly().effective_defer_tools
    )
    catalog = build_catalog(deferrable)

    results: List[Dict[str, Any]] = []
    tools_map: Dict[str, Dict[str, Any]] = {}
    corpus_stats = _corpus_stats(catalog)
    available_sources = _available_source_summary(catalog) if catalog else []
    for query in queries:
        hits = search_catalog(catalog, query, limit=limit, corpus_stats=corpus_stats)
        for h in hits:
            if h.name not in tools_map:
                tools_map[h.name] = _shared_tool_record(h)
        group: Dict[str, Any] = {"query": query, "matches": [h.name for h in hits]}
        if not hits and catalog:
            group["available_sources"] = available_sources
            group["hint"] = (
                "This query returned no lexical matches, but the sources above "
                "are connected and their tools remain available. Retry "
                "tool_search with the service name plus a concrete action or "
                "object before concluding the capability is unavailable."
            )
        results.append(group)

    result: Dict[str, Any] = {
        "queries": queries,
        "total_available": len(catalog),
        "results": results,
        "tools": tools_map,
    }
    return json.dumps(result, ensure_ascii=False)


def dispatch_tool_describe(args: Dict[str, Any],
                           *,
                           current_tool_defs: List[Dict[str, Any]],
                           config: Optional[ToolSearchConfig] = None) -> str:
    """Execute the ``tool_describe`` bridge tool. Returns JSON::

        {"tools": {name: {"description", "parameters"}},
         "not_found": [...],   # unknown / not in this assembly (never fails the call)
         "errors": {name: msg}}  # registered but non-deferrable names

    Duplicates are deduped silently.
    """
    if config is None:
        config = load_config_readonly()

    names = _string_list(args.get("names"), dedupe=True)
    if names is None:
        return tool_error("names is required and must be an array of strings")
    if not names:
        return tool_error("names is required and must contain at least one non-empty string")
    if len(names) > _MAX_DESCRIBE_NAMES_PER_CALL:
        return tool_error(
            f"too many names: {len(names)} > max {_MAX_DESCRIBE_NAMES_PER_CALL}. "
            "Retry with fewer names per call."
        )

    _, deferrable = classify_tools(
        current_tool_defs, load_config_readonly().effective_defer_tools
    )
    by_name: Dict[str, Dict[str, Any]] = {}
    for td in deferrable:
        fn = td.get("function") or {}
        if fn.get("name"):
            by_name[fn["name"]] = fn

    tools: Dict[str, Dict[str, Any]] = {}
    not_found: List[str] = []
    errors: Dict[str, str] = {}
    for name in names:
        fn = by_name.get(name)
        if fn is not None:
            tools[name] = {
                "description": fn.get("description", ""),
                "parameters": fn.get("parameters", {}),
            }
        elif _describe_classification(
            name, load_config_readonly().effective_defer_tools
        ) == "not_deferrable":
            errors[name] = (
                f"'{name}' is not a deferrable tool. If you see it in the tools list "
                "already, call it directly; otherwise check the spelling against tool_search."
            )
        else:
            not_found.append(name)

    result: Dict[str, Any] = {"tools": tools}
    if not_found:
        result["not_found"] = not_found
        result["hint"] = "Names in not_found are not currently available. Re-run tool_search to refresh."
    if errors:
        result["errors"] = errors
    return json.dumps(result, ensure_ascii=False)


def scoped_deferrable_names(tool_defs: List[Dict[str, Any]]) -> frozenset[str]:
    """Deferrable tool names in the *pre-assembly* ``tool_defs`` of the session's
    toolset scope — the universe ``tool_call`` may reach. Gates both bridge
    dispatch and the executor unwrap so a restricted session cannot invoke an
    out-of-scope tool via the bridge."""
    defer_tools = load_config_readonly().effective_defer_tools
    return frozenset(
        name for name in ((td.get("function") or {}).get("name", "") for td in tool_defs)
        if name and is_deferrable_tool_name(name, defer_tools)
    )


def resolve_underlying_call(args: Dict[str, Any]) -> Tuple[Optional[str], Dict[str, Any], Optional[str]]:
    """Parse a ``tool_call`` invocation into (underlying_name, args, error_msg);
    ``(None, {}, msg)`` on parse error. Shared by dispatch, display, and the
    trajectory recorder so all three agree on the underlying tool."""
    name = str(args.get("name") or "").strip()
    if not name:
        return None, {}, "tool_call requires a 'name' argument"
    if name in BRIDGE_TOOL_NAMES:
        return None, {}, f"tool_call cannot invoke '{name}' (it is itself a bridge tool)"
    raw_args = args.get("arguments")
    if raw_args is None:
        raw_args = {}
    if isinstance(raw_args, str):
        try:
            raw_args = json.loads(raw_args)
        except json.JSONDecodeError as e:
            return None, {}, f"tool_call 'arguments' is not valid JSON: {e}"
    if not isinstance(raw_args, dict):
        return None, {}, "tool_call 'arguments' must be an object"
    if not is_deferrable_tool_name(name, load_config_readonly().effective_defer_tools):
        return None, {}, (
            f"'{name}' is not a deferrable tool. If it appears in the model-facing tools "
            "list already, call it directly instead of via tool_call."
        )
    return name, raw_args, None


__all__ = [
    "TOOL_SEARCH_NAME",
    "TOOL_DESCRIBE_NAME",
    "TOOL_CALL_NAME",
    "BRIDGE_TOOL_NAMES",
    "ToolSearchConfig",
    "CatalogEntry",
    "AssemblyResult",
    "load_config",
    "is_deferrable_tool_name",
    "classify_tools",
    "estimate_tokens_from_schemas",
    "should_activate",
    "build_catalog",
    "build_catalog_listing_with_form",
    "listing_token_budget",
    "search_catalog",
    "bridge_tool_schemas",
    "assemble_tool_defs",
    "is_bridge_tool",
    "dispatch_tool_search",
    "dispatch_tool_describe",
    "resolve_underlying_call",
    "scoped_deferrable_names",
    "validate_deferred_call_args",
]
