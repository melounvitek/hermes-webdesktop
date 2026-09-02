"""Deferred-tool catalog for tool search: BM25 retrieval over deferrable tool
defs plus the budgeted, byte-stable catalog listing embedded in the bridge."""

from __future__ import annotations

import functools
import math
import re
import threading
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import snowballstemmer

from tools.tool_search_names import TOOL_CALL_NAME, TOOL_DESCRIBE_NAME, TOOL_SEARCH_NAME

# Chars-per-token rule of thumb for budget estimates; 4.0 slightly
# underestimates, which is the safer direction (fewer false activations).
CHARS_PER_TOKEN = 4.0


@dataclass
class CatalogEntry:
    """One deferrable tool, in a form the bridge tools can search and serve."""

    name: str
    description: str
    schema: Dict[str, Any]  # The full {"type":"function", "function": {...}} entry.
    source: str  # "mcp" | "plugin" | "other"
    source_name: str  # Toolset name, e.g. "mcp-github" or "kanban"
    _tokens: List[str] = field(default_factory=list)  # pre-tokenized for BM25


_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")

# Snowball stemmers carry mutable parsing state and bridge dispatch runs on
# parallel tool-call threads, so: one stemmer per thread, created lazily.
_thread_local = threading.local()


def _stemmer() -> Any:
    st = getattr(_thread_local, "stemmer", None)
    if st is None:
        st = snowballstemmer.stemmer("english")
        _thread_local.stemmer = st
    return st


@functools.lru_cache(maxsize=16384)
def _stem(token: str) -> str:
    """Stem one token, memoized across stateless catalog rebuilds."""
    return _stemmer().stemWord(token)


def _tokenize(text: str) -> List[str]:
    """Lowercase alphanumeric tokens, Snowball-stemmed (English).

    Shared by the index path and the query path so "issues" matches ``create_issue``.
    """
    if not text:
        return []
    return [_stem(token.lower()) for token in _TOKEN_RE.findall(text)]


def _entry_search_text(td: Dict[str, Any], source_label: str = "") -> str:
    """Search-text blob: split name words + source label + description +
    top-level parameter names. Schema bodies are excluded (noise, no recall
    gain). The ``mcp__`` prefix is dropped — it is in every MCP document, so
    its IDF is ~0. The source label lets a service-name query ("linear") reach
    a tool whose own name omits the vendor.
    """
    fn = td.get("function") or {}
    name = fn.get("name", "")
    if name.startswith("mcp__"):
        name = name[len("mcp__"):]
    desc = fn.get("description", "") or ""
    params = ((fn.get("parameters") or {}).get("properties") or {})
    param_names = " ".join(params.keys())
    name_words = re.sub(r"[_.:-]", " ", name)
    extra = source_label if source_label and source_label not in name_words.split() else ""
    return f"{name_words} {extra} {desc} {param_names}"


def _classify_source(name: str) -> Tuple[str, str]:
    """Return (source_kind, source_name) for a registered tool name."""
    try:
        from tools.registry import registry
        entry = registry.get_entry(name)
        if entry is None:
            return ("other", "")
        if entry.toolset.startswith("mcp-"):
            return ("mcp", entry.toolset)
        return ("plugin", entry.toolset)
    except Exception:
        return ("other", "")


def build_catalog(tool_defs: List[Dict[str, Any]]) -> List[CatalogEntry]:
    """Build the deferred-tool catalog from the deferrable subset of tool-defs."""
    catalog: List[CatalogEntry] = []
    for td in tool_defs:
        fn = td.get("function") or {}
        name = fn.get("name", "")
        if not name:
            continue
        source, source_name = _classify_source(name)
        # Index the human-facing label ("linear", not "mcp-linear").
        source_label = _listing_group_label(source_name) if source_name else ""
        catalog.append(CatalogEntry(
            name=name,
            description=fn.get("description", "") or "",
            schema=td,
            source=source,
            source_name=source_name,
            _tokens=_tokenize(_entry_search_text(td, source_label)),
        ))
    return catalog


def _bm25_score(query_tokens: List[str], doc_tokens: List[str],
                doc_lengths: List[int], avg_dl: float,
                doc_freq: Dict[str, int], n_docs: int,
                k1: float = 1.5, b: float = 0.75) -> float:
    """Standard BM25 score for one query against one document (inlined; the
    catalog is small enough that a dependency is not worth it)."""
    if not doc_tokens:
        return 0.0
    score = 0.0
    dl = len(doc_tokens)
    doc_tf = Counter(doc_tokens)
    for q in query_tokens:
        df = doc_freq.get(q, 0)
        if df == 0:
            continue
        idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
        tf = doc_tf.get(q, 0)
        if tf == 0:
            continue
        norm = tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl / max(avg_dl, 1.0)))
        score += idf * norm
    return score


_CorpusStats = Tuple[List[int], float, Dict[str, int], int]


def _corpus_stats(catalog: List[CatalogEntry]) -> _CorpusStats:
    """Compute the BM25 statistics shared by every query over a catalog."""
    doc_lengths = [len(entry._tokens) for entry in catalog]
    avg_dl = sum(doc_lengths) / max(len(doc_lengths), 1)
    doc_freq: Dict[str, int] = Counter()
    for entry in catalog:
        doc_freq.update(set(entry._tokens))
    return doc_lengths, avg_dl, dict(doc_freq), len(catalog)


def search_catalog(
    catalog: List[CatalogEntry],
    query: str,
    limit: int = 5,
    *,
    corpus_stats: Optional[_CorpusStats] = None,
) -> List[CatalogEntry]:
    """Top-``limit`` catalog entries for ``query`` by BM25 (exact name match
    ranks first). Falls back to a name-substring match only when NO query
    token appears in any document (e.g. "hub" vs ``github_*``); the IDF
    variant is strictly positive, so a hit anywhere suppresses the fallback.
    """
    if not catalog or limit <= 0:
        return []
    query_tokens = _tokenize(query)
    if not query_tokens:
        return []

    if corpus_stats is None:
        corpus_stats = _corpus_stats(catalog)
    doc_lengths, avg_dl, doc_freq, n_docs = corpus_stats

    scored: List[Tuple[float, CatalogEntry]] = []
    exact_name = query.strip().lower()
    for entry in catalog:
        if entry.name.lower() == exact_name:
            scored.append((float("inf"), entry))
            continue
        s = _bm25_score(query_tokens, entry._tokens, doc_lengths, avg_dl,
                        doc_freq, n_docs)
        if s > 0:
            scored.append((s, entry))

    if not scored:
        ql = query.lower()
        for entry in catalog:
            if ql in entry.name.lower():
                scored.append((0.1, entry))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [e for _, e in scored[:limit]]


# Sentence end: ., !, ? followed by whitespace/EOS, not inside e.g./i.e./etc.
_SENTENCE_END_RE = re.compile(r"(?<!\be\.g)(?<!\bi\.e)(?<!\betc)[.!?](?=\s|$)")


def _short_desc(description: str, max_chars: int = 60) -> str:
    """First sentence of a tool description, clipped to ``max_chars`` on a
    word boundary. Linear-time on hostile input."""
    text = " ".join((description or "").split())
    if not text:
        return ""
    m = _SENTENCE_END_RE.search(text)
    if m:
        text = text[:m.end()]
    if len(text) <= max_chars:
        return text
    clipped = text[:max_chars]
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped.rstrip(",;: ") + "…"


def _listing_group_label(source_name: str) -> str:
    """Human-facing group heading for a toolset, e.g. ``mcp-github`` -> ``github``."""
    label = source_name or "other"
    if label.startswith("mcp-"):
        label = label[4:]
    return label


def build_catalog_listing_with_form(
    deferrable: List[Dict[str, Any]],
    *,
    max_tokens: int = 4000,
) -> Tuple[Optional[str], str]:
    """Render the skills-style deferred-catalog manifest: ``- name: short desc``
    lines grouped under a heading per source (MCP server / plugin toolset).

    Returns ``(text, form)``; ``form`` is ``"full"``, ``"names"`` (names-only),
    ``"mixed"`` (oversized servers collapsed to a name + count summary line,
    small ones keep per-tool lines), ``"groups"`` (every server summarized),
    or ``"none"`` (over budget even summarized -> text is None).

    Ordering is deterministic (sorted groups and tools) so the block is
    byte-stable across assemblies — the request prefix stays cacheable.
    Degradation is PER SERVER (largest first): one huge server must not cost
    a small co-attached server its listing.
    """
    if not deferrable:
        return None, "none"

    groups: Dict[str, List[Tuple[str, str]]] = {}
    for td in deferrable:
        fn = td.get("function") or {}
        name = fn.get("name", "")
        if not name:
            continue
        source, source_name = _classify_source(name)
        label = _listing_group_label(source_name if source != "other" else "other")
        groups.setdefault(label, []).append((name, _short_desc(fn.get("description", ""))))

    if not groups:
        return None, "none"

    def render_group(label: str, mode: str) -> str:
        """Render one server's block. mode: 'full' | 'names' | 'summary'."""
        tools = sorted(groups[label])
        if mode == "summary":
            return (f"{label} ({len(tools)} tools — names not listed; "
                    f"discover via `{TOOL_SEARCH_NAME}`)")
        lines = [f"{label} tools ({len(tools)}):"]
        if mode == "full":
            for name, desc in tools:
                lines.append(f"- {name}: {desc}" if desc else f"- {name}")
        else:
            lines.append(", ".join(name for name, _ in tools))
        return "\n".join(lines)

    header = ("Deferred tool catalog (call schemas via "
              f"`{TOOL_DESCRIBE_NAME}`, invoke via `{TOOL_CALL_NAME}`):")

    def assemble(modes: Dict[str, str]) -> str:
        return "\n".join([header] + [render_group(lbl, modes[lbl])
                                     for lbl in sorted(groups)])

    def fits(text: str) -> bool:
        return math.ceil(len(text) / CHARS_PER_TOKEN) <= max_tokens

    # 1. Everything full.
    modes = {lbl: "full" for lbl in groups}
    if fits(assemble(modes)):
        return assemble(modes), "full"

    # 2. Everything names-only.
    modes = {lbl: "names" for lbl in groups}
    if fits(assemble(modes)):
        return assemble(modes), "names"

    # 3. Per-server degradation: collapse the LARGEST rendered groups first
    #    (deterministic: size then label).
    by_size = sorted(groups, key=lambda lbl: (-len(render_group(lbl, "names")), lbl))
    for lbl in by_size:
        modes[lbl] = "summary"
        if fits(assemble(modes)):
            form = "groups" if all(m == "summary" for m in modes.values()) else "mixed"
            return assemble(modes), form

    return None, "none"
