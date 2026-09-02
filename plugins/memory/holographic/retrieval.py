"""Hybrid keyword/BM25 retrieval for the memory store.

Ported from KIK memory_agent.py — combines FTS5 full-text search with
Jaccard similarity reranking and trust-weighted scoring.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .store import MemoryStore

try:
    from . import holographic as hrr
except ImportError:
    import holographic as hrr  # type: ignore[no-redef]

_FACT_COLUMNS = (
    "fact_id, content, category, tags, trust_score, "
    "retrieval_count, helpful_count, created_at, updated_at"
)


class FactRetriever:
    """Multi-strategy fact retrieval with trust-weighted scoring."""

    def __init__(
        self,
        store: MemoryStore,
        temporal_decay_half_life: int = 0,  # days, 0 = disabled
        fts_weight: float = 0.4, jaccard_weight: float = 0.3, hrr_weight: float = 0.3,
        hrr_dim: int = 1024,
    ):
        self.store = store
        self.half_life = temporal_decay_half_life
        self.hrr_dim = hrr_dim

        # Auto-redistribute weights if numpy unavailable
        if hrr_weight > 0 and not hrr._HAS_NUMPY:
            fts_weight, jaccard_weight, hrr_weight = 0.6, 0.4, 0.0
        self.fts_weight, self.jaccard_weight, self.hrr_weight = fts_weight, jaccard_weight, hrr_weight

    def search(self, query: str, category: str | None = None, min_trust: float = 0.3, limit: int = 10) -> list[dict]:
        """Hybrid search: FTS5 candidates (limit*3) → Jaccard + HRR rerank → trust
        weighting → optional temporal decay 0.5^(age_days / half_life).

        Returns fact dicts with a 'score' field, sorted by score desc.
        """
        candidates = self._fts_candidates(query, category, min_trust, limit * 3)
        if not candidates:
            return []

        query_tokens = self._tokenize(query)
        # Query vector is loop-invariant; encode lazily on the first candidate
        # that carries an HRR vector so migrated stores whose hrr_vector was
        # never backfilled don't pay for an encode nothing uses.
        query_vec = None
        for fact in candidates:
            all_tokens = self._tokenize(fact["content"]) | self._tokenize(fact.get("tags", ""))
            jaccard = self._jaccard_similarity(query_tokens, all_tokens)
            hrr_sim = 0.5  # neutral
            if self.hrr_weight > 0 and fact.get("hrr_vector"):
                fact_vec = hrr.bytes_to_phases(fact["hrr_vector"], dim=self.hrr_dim)
                if query_vec is None:
                    query_vec = hrr.encode_text(query, self.hrr_dim)
                hrr_sim = (hrr.similarity(query_vec, fact_vec) + 1.0) / 2.0  # shift to [0,1]
            relevance = (self.fts_weight * fact.get("fts_rank", 0.0)
                         + self.jaccard_weight * jaccard
                         + self.hrr_weight * hrr_sim)
            fact["score"] = relevance * fact["trust_score"]
            if self.half_life > 0:
                fact["score"] *= self._temporal_decay(fact.get("updated_at") or fact.get("created_at"))

        candidates.sort(key=lambda x: x["score"], reverse=True)
        results = candidates[:limit]
        for fact in results:
            fact.pop("hrr_vector", None)  # callers expect JSON-serializable dicts
        return results

    def probe(self, entity: str, category: str | None = None, limit: int = 10) -> list[dict]:
        """Compositional entity query: unbind bind(entity, ROLE_ENTITY) from the
        category bank (or each fact vector) to find facts where the entity plays
        a structural role. Not keyword search. Falls back to FTS5 without numpy.
        """
        if not hrr._HAS_NUMPY:
            return self.search(entity, category=category, limit=limit)

        role_entity = hrr.encode_atom("__hrr_role_entity__", self.hrr_dim)
        entity_vec = hrr.encode_atom(entity.lower(), self.hrr_dim)
        probe_key = hrr.bind(entity_vec, role_entity)

        # Try the category-specific bank first, then individual fact vectors
        if category:
            bank_row = self.store._conn.execute(
                "SELECT vector FROM memory_banks WHERE bank_name = ?",
                (f"cat:{category}",),
            ).fetchone()
            if bank_row:
                extracted = hrr.unbind(hrr.bytes_to_phases(bank_row["vector"], dim=self.hrr_dim), probe_key)
                return self._rank_by_vector(
                    self._vector_rows(category), lambda _f, fact_vec: hrr.similarity(extracted, fact_vec), limit,
                )

        rows = self._vector_rows(category)
        if not rows:
            return self.search(entity, category=category, limit=limit)

        # role_content is loop-invariant — encode once, not per row.
        role_content = hrr.encode_atom("__hrr_role_content__", self.hrr_dim)

        def _sim(fact: dict, fact_vec) -> float:
            # Does unbinding the probe key leave the fact's content signal?
            residual = hrr.unbind(fact_vec, probe_key)
            content_vec = hrr.bind(hrr.encode_text(fact["content"], self.hrr_dim), role_content)
            return hrr.similarity(residual, content_vec)

        return self._rank_by_vector(rows, _sim, limit)

    def related(self, entity: str, category: str | None = None, limit: int = 10) -> list[dict]:
        """Facts structurally connected to an entity (shared context), not just
        facts *about* it as in probe. Falls back to FTS5 without numpy.
        """
        if not hrr._HAS_NUMPY:
            return self.search(entity, category=category, limit=limit)

        # Bare atom, not role-bound — we want ANY structural match
        entity_vec = hrr.encode_atom(entity.lower(), self.hrr_dim)

        rows = self._vector_rows(category)
        if not rows:
            return self.search(entity, category=category, limit=limit)

        # Both role atoms are loop-invariant — encode once, not per row.
        role_entity = hrr.encode_atom("__hrr_role_entity__", self.hrr_dim)
        role_content = hrr.encode_atom("__hrr_role_content__", self.hrr_dim)

        def _sim(fact: dict, fact_vec) -> float:
            # A residual similar to ANY role vector means the entity plays a
            # structural role in the fact; take the max over both roles.
            residual = hrr.unbind(fact_vec, entity_vec)
            return max(hrr.similarity(residual, role_entity), hrr.similarity(residual, role_content))

        return self._rank_by_vector(rows, _sim, limit)

    def reason(self, entities: list[str], category: str | None = None, limit: int = 10) -> list[dict]:
        """Multi-entity compositional query (vector-space JOIN): facts where ALL
        entities play structural roles. Falls back to FTS5 without numpy.
        """
        if not hrr._HAS_NUMPY or not entities:
            return self.search(" ".join(entities), category=category, limit=limit)

        role_entity = hrr.encode_atom("__hrr_role_entity__", self.hrr_dim)
        probe_keys = [
            hrr.bind(hrr.encode_atom(entity.lower(), self.hrr_dim), role_entity)
            for entity in entities
        ]

        rows = self._vector_rows(category)
        if not rows:
            return self.search(" ".join(entities), category=category, limit=limit)

        role_content = hrr.encode_atom("__hrr_role_content__", self.hrr_dim)

        def _sim(fact: dict, fact_vec) -> float:
            # AND semantics via min: high only if EVERY entity is structurally present.
            return min(
                hrr.similarity(hrr.unbind(fact_vec, key), role_content) for key in probe_keys
            )

        return self._rank_by_vector(rows, _sim, limit)

    def contradict(self, category: str | None = None, threshold: float = 0.3, limit: int = 10) -> list[dict]:
        """Memory hygiene: pairs of facts that share entities (same subject) but
        have low content-vector similarity (different claims). Empty without numpy.
        """
        if not hrr._HAS_NUMPY:
            return []

        rows = self._vector_rows(
            category,
            columns="fact_id, content, category, tags, trust_score, created_at, updated_at, hrr_vector",
        )
        if len(rows) < 2:
            return []
        # O(n²) guard: ~125K comparisons at 500 facts is acceptable; above that
        # only compare the most recently updated facts.
        if len(rows) > 500:
            rows = sorted(rows, key=lambda r: r["updated_at"] or r["created_at"], reverse=True)[:500]

        facts = [dict(r) for r in rows]
        for fact in facts:
            entity_rows = self.store._conn.execute(
                "SELECT e.name FROM entities e JOIN fact_entities fe ON fe.entity_id = e.entity_id WHERE fe.fact_id = ?",
                (fact["fact_id"],),
            ).fetchall()
            fact["_entities"] = {r["name"].lower() for r in entity_rows}
            fact["_vec"] = hrr.bytes_to_phases(fact.pop("hrr_vector"), dim=self.hrr_dim)

        def _public(fact: dict) -> dict:
            return {k: v for k, v in fact.items() if k not in ("_entities", "_vec")}

        contradictions = []
        for i, f1 in enumerate(facts):
            for f2 in facts[i + 1:]:
                ents1, ents2 = f1["_entities"], f2["_entities"]
                if not ents1 or not ents2:
                    continue
                entity_overlap = len(ents1 & ents2) / len(ents1 | ents2)
                if entity_overlap < 0.3:
                    continue  # not enough shared subject to be contradictory
                content_sim = hrr.similarity(f1["_vec"], f2["_vec"])
                # High entity overlap + low content similarity = contradiction
                contradiction_score = entity_overlap * (1.0 - (content_sim + 1.0) / 2.0)
                if contradiction_score >= threshold:
                    contradictions.append({
                        "fact_a": _public(f1),
                        "fact_b": _public(f2),
                        "entity_overlap": round(entity_overlap, 3),
                        "content_similarity": round(content_sim, 3),
                        "contradiction_score": round(contradiction_score, 3),
                        "shared_entities": sorted(ents1 & ents2),
                    })

        contradictions.sort(key=lambda x: x["contradiction_score"], reverse=True)
        return contradictions[:limit]

    # -- Vector scoring helpers -----------------------------------------------

    def _vector_rows(self, category: str | None, columns: str = _FACT_COLUMNS + ", hrr_vector") -> list:
        """All facts that carry an HRR vector, optionally filtered by category."""
        where = "WHERE hrr_vector IS NOT NULL"
        params: list = []
        if category:
            where += " AND category = ?"
            params.append(category)
        return self.store._conn.execute(f"SELECT {columns} FROM facts {where}", params).fetchall()

    def _rank_by_vector(self, rows: list, sim_fn: Callable[[dict, object], float], limit: int) -> list[dict]:
        """Score each row as (sim + 1) / 2 * trust_score (sim shifted to [0, 1]), sorted desc."""
        scored = []
        for row in rows:
            fact = dict(row)
            fact_vec = hrr.bytes_to_phases(fact.pop("hrr_vector"), dim=self.hrr_dim)
            fact["score"] = (sim_fn(fact, fact_vec) + 1.0) / 2.0 * fact["trust_score"]
            scored.append(fact)
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:limit]

    # -- FTS / lexical helpers ------------------------------------------------

    def _fts_candidates(self, query: str, category: str | None, min_trust: float, limit: int) -> list[dict]:
        """Raw FTS5 MATCH candidates with rank normalized to [0, 1] as 'fts_rank'."""
        category_clause = "AND f.category = ? " if category else ""
        params = [self._sanitize_fts_query(query)] + ([category] if category else []) + [min_trust, limit]
        sql = (
            "SELECT f.*, facts_fts.rank as fts_rank_raw FROM facts_fts "
            "JOIN facts f ON f.fact_id = facts_fts.rowid "
            f"WHERE facts_fts MATCH ? {category_clause}AND f.trust_score >= ? "
            "ORDER BY facts_fts.rank LIMIT ?"
        )
        try:
            rows = self.store._conn.execute(sql, params).fetchall()
        except Exception:
            return []  # FTS5 MATCH can fail on malformed queries
        if not rows:
            return []

        results = [dict(row) for row in rows]
        # FTS5 rank is negative (lower = better); normalize |rank| / max to [0, 1]
        max_rank = max(max(abs(f["fts_rank_raw"]) for f in results), 1e-6)  # avoid div by zero
        for fact in results:
            fact["fts_rank"] = abs(fact.pop("fts_rank_raw")) / max_rank
        return results

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        """Lowercase whitespace tokens with surrounding punctuation stripped (no stemming)."""
        if not text:
            return set()
        return {c for c in (w.strip(".,;:!?\"'()[]{}#@<>") for w in text.lower().split()) if c}

    # Stopwords dropped before FTS5 OR-expansion: short English function words
    # that carry no retrieval signal and force false-negative AND matches.
    _FTS_STOPWORDS = frozenset("""
        a about above after again all am an and any are as at be because been before being
        between both but by can could did do does doing don down during each few for from
        further had has have having he her here hers herself him himself his how i if in
        into is it its itself just me more most my myself no nor not now of off on once
        only or other our ours ourselves out over own same she should so some such than that
        the their theirs them themselves then there these they this those through to too under
        until up very was we were what when where which while who whom why will with would
        you your yours yourself yourselves
    """.split())

    @classmethod
    def _sanitize_fts_query(cls, query: str) -> str:
        """Natural-language query -> FTS5-safe OR expression of quoted tokens.

        FTS5 AND-joins a multi-word MATCH by default, which tanks recall on prose.
        Drops stopwords and <2-char tokens, strips FTS5 operator chars, and
        phrase-quotes each survivor. If nothing survives, returns the raw query
        (caller gets zero results rather than a SQL error).
        """
        if not query:
            return ""
        strip_special = str.maketrans("", "", '"()*^:-+')
        tokens = [
            f'"{cleaned}"'
            for cleaned in (raw.strip(".,;:!?\"'()[]{}#@<>").translate(strip_special) for raw in query.lower().split())
            if len(cleaned) >= 2 and cleaned not in cls._FTS_STOPWORDS
        ]
        return " OR ".join(tokens) if tokens else query

    @staticmethod
    def _jaccard_similarity(set_a: set, set_b: set) -> float:
        """Jaccard similarity coefficient: |A ∩ B| / |A ∪ B|."""
        return len(set_a & set_b) / len(set_a | set_b) if set_a and set_b else 0.0

    def _temporal_decay(self, timestamp_str: str | None) -> float:
        """0.5^(age_days / half_life); 1.0 if disabled, missing, unparseable, or in the future."""
        if not self.half_life or not timestamp_str:
            return 1.0
        try:
            ts = timestamp_str
            if isinstance(ts, str):
                ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - ts).total_seconds() / 86400
            return 1.0 if age_days < 0 else math.pow(0.5, age_days / self.half_life)
        except (ValueError, TypeError):
            return 1.0
