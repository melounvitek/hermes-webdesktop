"""
Session Insights Engine: aggregates the SQLite state DB into usage insights
(tokens, cost estimates, tool/skill usage, activity, model/platform breakdowns).

    engine = InsightsEngine(db)
    report = engine.generate(days=30)
    print(engine.format_terminal(report))
"""

import json
import sqlite3
import time
from collections import Counter, defaultdict
from datetime import datetime
from decimal import Decimal
from typing import Any, Dict, List, Optional

from agent.usage_pricing import (
    CanonicalUsage,
    estimate_usage_cost,
    format_cost_label,
    format_duration_compact,
    has_known_pricing,
)

_TOKEN_KEYS = ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens")
_SKILL_TOOLS = {"skill_view", "skill_manage"}


def _fmt_est_cost(est_cost: float) -> str:
    """Aggregate cost via the shared label helper so sub-cent totals render at 4dp, not "~$0.00"."""
    return format_cost_label(Decimal(str(est_cost)))


def _estimate_cost(
    session_or_model: Dict[str, Any] | str,
    input_tokens: int = 0,
    output_tokens: int = 0,
    *,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    provider: Optional[str] = None,
    base_url: Optional[str] = None,
) -> tuple[float, str]:
    """Estimate the USD cost for a session row or a model/token tuple."""
    if isinstance(session_or_model, dict):
        s = session_or_model
        model = s.get("model") or ""
        usage = CanonicalUsage(**{k: s.get(k) or 0 for k in _TOKEN_KEYS})
        provider, base_url = s.get("billing_provider"), s.get("billing_base_url")
    else:
        model = session_or_model or ""
        usage = CanonicalUsage(
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cache_read_tokens=cache_read_tokens,
            cache_write_tokens=cache_write_tokens,
        )
    result = estimate_usage_cost(model, usage, provider=provider, base_url=base_url)
    return float(result.amount_usd or 0.0), result.status


def _bar_chart(values: List[int], max_width: int = 20) -> List[str]:
    """Create simple horizontal bar chart strings from values."""
    peak = max(values) if values else 1
    if peak == 0:
        return ["" for _ in values]
    return ["█" * max(1, int(v / peak * max_width)) if v > 0 else "" for v in values]


def _short_model(model: Optional[str]) -> str:
    """Display name: strip the provider prefix; empty → "unknown"."""
    return (model or "unknown").split("/")[-1]


def _parse_calls(raw: Any) -> Optional[list]:
    """tool_calls column → list, or None when not decodable as a JSON list."""
    try:
        if isinstance(raw, str):
            raw = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return raw if isinstance(raw, list) else None


def _hour12(hr: int) -> str:
    return f"{hr % 12 or 12}{'AM' if hr < 12 else 'PM'}"


def _day(ts: Any) -> str:
    return datetime.fromtimestamp(ts).strftime("%b %d") if ts else "?"


def _scoped(before: str, after: str = "", *, src: str = " AND s.source = ?") -> tuple[str, str]:
    """(unfiltered, source-filtered) query pair sharing one body.

    Built once at class definition, so no runtime value can alter query structure.
    """
    return before + after, before + src + after


class InsightsEngine:
    """Analyzes session history from a SessionDB (or raw sqlite3 connection)."""

    _SESSION_COLS = ("id, source, model, started_at, ended_at, "
                     "message_count, tool_call_count, input_tokens, output_tokens, "
                     "cache_read_tokens, cache_write_tokens, billing_provider, "
                     "billing_base_url, billing_mode, estimated_cost_usd, "
                     "actual_cost_usd, cost_status, cost_source, api_call_count")

    _GET_SESSIONS_ALL, _GET_SESSIONS_WITH_SOURCE = _scoped(
        f"SELECT {_SESSION_COLS} FROM sessions WHERE started_at >= ?",
        " ORDER BY started_at DESC",
        src=" AND source = ?",
    )

    # ``INDEXED BY`` pins the partial index so the plan is deterministic on a
    # fresh state.db (before ANALYZE) for both branches; without it the
    # source-filtered probe falls back to idx_messages_session_active and scans
    # each session's non-tool-call rows. The pin is a HARD dependency (SQLite
    # raises ``no such index``): read-only opens skip ``_init_schema``, so an
    # older writer's DB may lack it — ``__init__`` probes once and falls back
    # to the unpinned variants (identical rows, optimizer-chosen plan).
    _MESSAGES_ASSISTANT_CALLS_INDEX = "idx_messages_assistant_calls_by_session"
    _ASSISTANT_CALLS = (
        f" FROM messages m INDEXED BY {_MESSAGES_ASSISTANT_CALLS_INDEX}"
        " JOIN sessions s ON s.id = m.session_id"
        " WHERE s.started_at >= ?"
    )
    _GET_TOOL_CALLS_ALL, _GET_TOOL_CALLS_WITH_SOURCE = _scoped(
        "SELECT m.tool_calls" + _ASSISTANT_CALLS,
        " AND m.role = 'assistant' AND m.tool_calls IS NOT NULL",
    )
    _GET_SKILL_CALLS_ALL, _GET_SKILL_CALLS_WITH_SOURCE = _scoped(
        "SELECT m.tool_calls, m.timestamp" + _ASSISTANT_CALLS,
        " AND m.role = 'assistant' AND m.tool_calls IS NOT NULL"
        " AND (instr(m.tool_calls, 'skill_view') > 0"
        " OR instr(m.tool_calls, 'skill_manage') > 0)",
    )
    _GET_TOOL_NAMES_ALL, _GET_TOOL_NAMES_WITH_SOURCE = _scoped(
        """SELECT m.tool_name, COUNT(*) as count
                   FROM messages m
                   JOIN sessions s ON s.id = m.session_id
                   WHERE s.started_at >= ?""",
        """
                     AND m.role = 'tool' AND m.tool_name IS NOT NULL
                   GROUP BY m.tool_name
                   ORDER BY count DESC""",
    )
    _GET_MESSAGE_STATS_ALL, _GET_MESSAGE_STATS_WITH_SOURCE = _scoped(
        """SELECT
                     COUNT(*) as total_messages,
                     SUM(CASE WHEN m.role = 'user' THEN 1 ELSE 0 END) as user_messages,
                     SUM(CASE WHEN m.role = 'assistant' THEN 1 ELSE 0 END) as assistant_messages,
                     SUM(CASE WHEN m.role = 'tool' THEN 1 ELSE 0 END) as tool_messages
                   FROM messages m
                   JOIN sessions s ON s.id = m.session_id
                   WHERE s.started_at >= ?""",
    )
    _GET_MODEL_USAGE_ALL, _GET_MODEL_USAGE_WITH_SOURCE = _scoped(
        "SELECT u.session_id, u.model, u.billing_provider, u.billing_base_url,"
        " u.api_call_count, u.input_tokens, u.output_tokens,"
        " u.cache_read_tokens, u.cache_write_tokens, u.reasoning_tokens,"
        " u.estimated_cost_usd, u.actual_cost_usd, u.cost_status,"
        " u.cost_source, u.billing_mode"
        " FROM session_model_usage u"
        " JOIN sessions s ON s.id = u.session_id"
        " WHERE s.started_at >= ?",
    )
    _PINNED = ("_GET_TOOL_CALLS", "_GET_SKILL_CALLS")

    def __init__(self, db):
        self.db = db
        self._conn = db._conn
        try:
            self._has_assistant_calls_index = bool(
                self._conn.execute(
                    "SELECT 1 FROM sqlite_master WHERE type='index' AND name=?",
                    (self._MESSAGES_ASSISTANT_CALLS_INDEX,),
                ).fetchone()
            )
        except sqlite3.Error:
            self._has_assistant_calls_index = False
        if not self._has_assistant_calls_index:
            strip = f" INDEXED BY {self._MESSAGES_ASSISTANT_CALLS_INDEX}"
            for base in self._PINNED:
                for suffix in ("_ALL", "_WITH_SOURCE"):
                    setattr(self, base + suffix, getattr(self, base + suffix).replace(strip, ""))

    def _query(self, base: str, cutoff: float, source: Optional[str]):
        """Run ``<base>_WITH_SOURCE`` or ``<base>_ALL`` (instance attrs, so the
        unpinned fallback applies) and return the cursor."""
        if source:
            return self._conn.execute(getattr(self, base + "_WITH_SOURCE"), (cutoff, source))
        return self._conn.execute(getattr(self, base + "_ALL"), (cutoff,))

    def generate(self, days: int = 30, source: str = None) -> Dict[str, Any]:
        """Generate a complete insights report for the last ``days`` days,
        optionally filtered by source platform."""
        cutoff = time.time() - (days * 86400)

        # Drain the SessionDB's async accounting queue so counters are exact
        # (self.db may be a raw sqlite3 connection in tests — guard).
        flush = getattr(self.db, "flush_token_counts", None)
        if callable(flush):
            flush()

        sessions = self._get_sessions(cutoff, source)
        tool_usage = self._get_tool_usage(cutoff, source)
        skill_usage = self._get_skill_usage(cutoff, source)
        message_stats = self._get_message_stats(cutoff, source)

        if not sessions:
            return {
                "days": days,
                "source_filter": source,
                "empty": True,
                "overview": {},
                "models": [],
                "platforms": [],
                "tools": [],
                "skills": self._compute_skill_breakdown([]),
                "activity": {},
                "top_sessions": [],
            }

        models = self._compute_model_breakdown(sessions, cutoff, source)
        return {
            "days": days,
            "source_filter": source,
            "empty": False,
            "generated_at": time.time(),
            "overview": self._compute_overview(sessions, message_stats, models),
            "models": models,
            "platforms": self._compute_platform_breakdown(sessions),
            "tools": self._compute_tool_breakdown(tool_usage),
            "skills": self._compute_skill_breakdown(skill_usage),
            "activity": self._compute_activity_patterns(sessions),
            "top_sessions": self._compute_top_sessions(sessions),
        }

    def get_usage_breakdown(self, days: int = 30, source: str = None) -> Dict[str, Any]:
        """Analytics-usage payload (tools + skills) without a full generate().

        Uses the instr()-prefiltered skill query so only skill_view/skill_manage
        messages are loaded, while keeping the per-tool breakdown the dashboard uses.
        """
        cutoff = time.time() - (days * 86400)
        return {
            "tools": self._compute_tool_breakdown(self._get_tool_usage(cutoff, source)),
            "skills": self._compute_skill_breakdown(self._get_skill_usage(cutoff, source)),
        }

    # ------------------------------------------------------------------ SQL

    def _get_sessions(self, cutoff: float, source: str = None) -> List[Dict]:
        return [dict(row) for row in self._query("_GET_SESSIONS", cutoff, source).fetchall()]

    def _get_tool_usage(self, cutoff: float, source: str = None) -> List[Dict]:
        """Tool call counts from two sources: ``tool_name`` on 'tool' rows (set
        by the gateway) and ``tool_calls`` JSON on assistant rows (covers CLI,
        where tool_name is not populated). Overlapping tools take the max."""
        tool_counts = Counter()
        for row in self._query("_GET_TOOL_NAMES", cutoff, source).fetchall():
            tool_counts[row["tool_name"]] += row["count"]

        tool_calls_counts = Counter()
        for row in self._query("_GET_TOOL_CALLS", cutoff, source).fetchall():
            try:
                for call in _parse_calls(row["tool_calls"]) or []:
                    name = (call.get("function", {}) if isinstance(call, dict) else {}).get("name")
                    if name:
                        tool_calls_counts[name] += 1
            except (TypeError, AttributeError):
                continue

        if tool_calls_counts:
            if tool_counts:
                tool_counts = Counter({
                    tool: max(tool_counts.get(tool, 0), tool_calls_counts.get(tool, 0))
                    for tool in set(tool_counts) | set(tool_calls_counts)
                })
            else:
                tool_counts = tool_calls_counts

        return [{"tool_name": name, "count": count} for name, count in tool_counts.most_common()]

    def _get_skill_usage(self, cutoff: float, source: str = None) -> List[Dict]:
        """Extract per-skill usage from assistant tool calls."""
        skill_counts: Dict[str, Dict[str, Any]] = {}
        for row in self._query("_GET_SKILL_CALLS", cutoff, source).fetchall():
            calls = _parse_calls(row["tool_calls"])
            if calls is None:
                continue
            timestamp = row["timestamp"]
            for call in calls:
                if not isinstance(call, dict):
                    continue
                func = call.get("function", {})
                tool_name = func.get("name")
                if tool_name not in _SKILL_TOOLS:
                    continue
                args = func.get("arguments")
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except (json.JSONDecodeError, TypeError):
                        continue
                if not isinstance(args, dict):
                    continue
                skill_name = args.get("name")
                if not isinstance(skill_name, str) or not skill_name.strip():
                    continue
                entry = skill_counts.setdefault(
                    skill_name,
                    {"skill": skill_name, "view_count": 0, "manage_count": 0, "last_used_at": None},
                )
                entry["view_count" if tool_name == "skill_view" else "manage_count"] += 1
                if timestamp is not None and (
                    entry["last_used_at"] is None or timestamp > entry["last_used_at"]
                ):
                    entry["last_used_at"] = timestamp
        return list(skill_counts.values())

    def _get_message_stats(self, cutoff: float, source: str = None) -> Dict:
        row = self._query("_GET_MESSAGE_STATS", cutoff, source).fetchone()
        return dict(row) if row else {
            "total_messages": 0, "user_messages": 0,
            "assistant_messages": 0, "tool_messages": 0,
        }

    def _get_model_usage(self, cutoff: float, source: str = None) -> List[Dict]:
        """Per-model usage rows; [] when the table is missing (older DB) so the
        caller falls back to the per-session aggregate."""
        try:
            return [dict(row) for row in self._query("_GET_MODEL_USAGE", cutoff, source).fetchall()]
        except sqlite3.OperationalError:
            return []

    # -------------------------------------------------------------- Compute

    def _compute_overview(
        self,
        sessions: List[Dict],
        message_stats: Dict,
        models: Optional[List[Dict]] = None,
    ) -> Dict:
        # Per-model breakdown includes auxiliary usage rows (vision/compression/
        # titles) plus reconciled residuals, while session counters carry
        # main-loop usage only — sum the breakdown when available so overview
        # totals match the per-model table and aux spend isn't undercounted.
        rows = models or sessions
        total_input, total_output, total_cache_read, total_cache_write = (
            sum(int(r.get(k) or 0) for r in rows) for k in _TOKEN_KEYS
        )
        total_tokens = total_input + total_output + total_cache_read + total_cache_write
        total_tool_calls = sum(s.get("tool_call_count") or 0 for s in sessions)
        total_messages = sum(s.get("message_count") or 0 for s in sessions)

        total_cost = actual_cost = 0.0
        models_with_pricing, models_without_pricing = set(), set()
        status_counts = Counter()
        for s in sessions:
            model = s.get("model") or ""
            estimated, status = _estimate_cost(s)
            total_cost += estimated
            actual_cost += s.get("actual_cost_usd") or 0.0
            status_counts[status] += 1
            known = has_known_pricing(model, s.get("billing_provider"), s.get("billing_base_url"))
            (models_with_pricing if known else models_without_pricing).add(_short_model(model))
        if models:
            total_cost = sum(float(m.get("cost") or 0.0) for m in models)

        # Guard against negative durations from clock drift.
        durations = [
            s["ended_at"] - s["started_at"]
            for s in sessions
            if s.get("started_at") and s.get("ended_at") and s["ended_at"] > s["started_at"]
        ]
        started = [s["started_at"] for s in sessions if s.get("started_at")]
        n = len(sessions)
        return {
            "total_sessions": n,
            "total_messages": total_messages,
            "total_tool_calls": total_tool_calls,
            "total_input_tokens": total_input,
            "total_output_tokens": total_output,
            "total_cache_read_tokens": total_cache_read,
            "total_cache_write_tokens": total_cache_write,
            "total_tokens": total_tokens,
            "estimated_cost": total_cost,
            "actual_cost": actual_cost,
            "total_hours": sum(durations) / 3600 if durations else 0,
            "avg_session_duration": sum(durations) / len(durations) if durations else 0,
            "avg_messages_per_session": total_messages / n if sessions else 0,
            "avg_tokens_per_session": total_tokens / n if sessions else 0,
            "user_messages": message_stats.get("user_messages") or 0,
            "assistant_messages": message_stats.get("assistant_messages") or 0,
            "tool_messages": message_stats.get("tool_messages") or 0,
            "date_range_start": min(started) if started else None,
            "date_range_end": max(started) if started else None,
            "models_with_pricing": sorted(models_with_pricing),
            "models_without_pricing": sorted(models_without_pricing),
            "unknown_cost_sessions": status_counts["unknown"],
            "included_cost_sessions": status_counts["included"],
        }

    def _compute_model_breakdown(
        self, sessions: List[Dict], cutoff: float, source: str = None
    ) -> List[Dict]:
        """Tokens/cost per model from session_model_usage, so a session that
        switched models via ``/model`` splits across every model it used.
        Sessions without per-model rows (pre-table data) fall back to their
        single recorded aggregate. Tool calls aren't tied to an API call, so
        they stay attributed to the session's recorded model."""
        model_data = defaultdict(lambda: {
            "sessions": set(), "input_tokens": 0, "output_tokens": 0,
            "cache_read_tokens": 0, "cache_write_tokens": 0,
            "reasoning_tokens": 0, "total_tokens": 0, "api_calls": 0,
            "tool_calls": 0, "cost": 0.0, "actual_cost": 0.0,
        })

        def _accumulate(model, provider, base_url, session_id, inp, out,
                        cache_read, cache_write, reasoning, *,
                        stored_cost=None, actual_cost=None, cost_status=None):
            model = model or "unknown"
            display_model = _short_model(model)
            d: Dict[str, Any] = model_data[display_model]
            d["sessions"].add(session_id)
            d["input_tokens"] += inp
            d["output_tokens"] += out
            d["cache_read_tokens"] += cache_read
            d["cache_write_tokens"] += cache_write
            d["reasoning_tokens"] += reasoning
            d["total_tokens"] += inp + out + cache_read + cache_write
            if stored_cost is None:
                estimate, status = _estimate_cost(
                    model, inp, out,
                    cache_read_tokens=cache_read, cache_write_tokens=cache_write,
                    provider=provider or None, base_url=base_url,
                )
            else:
                estimate, status = float(stored_cost or 0.0), cost_status or "unknown"
            d["cost"] += estimate
            d["actual_cost"] += float(actual_cost or 0.0)
            d["cost_status"] = status
            if has_known_pricing(model, provider or None, base_url):
                d["has_pricing"] = True
            else:
                d.setdefault("has_pricing", False)
            return display_model

        count_keys = _TOKEN_KEYS + ("reasoning_tokens", "api_call_count")
        usage_totals = defaultdict(lambda: dict.fromkeys(count_keys, 0) | {
            "estimated_cost_usd": 0.0, "actual_cost_usd": 0.0,
        })
        for r in self._get_model_usage(cutoff, source):
            totals: Dict[str, Any] = usage_totals[r["session_id"]]
            for key in count_keys:
                totals[key] += r[key] or 0
            totals["estimated_cost_usd"] += r["estimated_cost_usd"] or 0.0
            totals["actual_cost_usd"] += r["actual_cost_usd"] or 0.0
            d = _accumulate(
                r["model"], r["billing_provider"], r.get("billing_base_url"),
                r["session_id"], r["input_tokens"] or 0, r["output_tokens"] or 0,
                r["cache_read_tokens"] or 0, r["cache_write_tokens"] or 0,
                r["reasoning_tokens"] or 0,
                stored_cost=(
                    r["estimated_cost_usd"]
                    if r.get("cost_status") or r.get("cost_source")
                    else None
                ),
                actual_cost=r["actual_cost_usd"],
                cost_status=r.get("cost_status"),
            )
            model_data[d]["api_calls"] += r["api_call_count"] or 0

        # Reconcile against the aggregate row: covers legacy sessions,
        # interrupted migrations, and absolute cumulative updates without
        # double-counting already-attributed route deltas.
        for s in sessions:
            totals = usage_totals[s["id"]]
            inp, out, cache_read, cache_write, residual_calls = (
                max(0, (s.get(k) or 0) - totals[k]) for k in _TOKEN_KEYS + ("api_call_count",)
            )
            residual_cost = max(
                0.0, float(s.get("estimated_cost_usd") or 0.0) - totals["estimated_cost_usd"],
            )
            residual_actual = max(
                0.0, float(s.get("actual_cost_usd") or 0.0) - totals["actual_cost_usd"],
            )
            if not (
                inp or out or cache_read or cache_write or residual_cost
                or residual_actual or residual_calls
            ):
                continue
            d = _accumulate(
                s.get("model"), s.get("billing_provider"),
                s.get("billing_base_url"), s["id"],
                inp, out, cache_read, cache_write, 0,
                stored_cost=residual_cost,
                actual_cost=residual_actual,
                cost_status=s.get("cost_status"),
            )
            model_data[d]["api_calls"] += residual_calls

        for s in sessions:
            tool_calls = s.get("tool_call_count") or 0
            if tool_calls:
                model_data[_short_model(s.get("model"))]["tool_calls"] += tool_calls

        result = []
        for model, data in model_data.items():
            entry = {"model": model, **data, "sessions": len(data["sessions"])}
            # Models seen only via tool-call attribution never hit _accumulate —
            # default these so the output shape is uniform for JSON consumers.
            entry.setdefault("has_pricing", False)
            entry.setdefault("cost_status", "unknown")
            result.append(entry)
        result.sort(key=lambda x: (x["total_tokens"], x["sessions"]), reverse=True)
        return result

    def _compute_platform_breakdown(self, sessions: List[Dict]) -> List[Dict]:
        platform_data = defaultdict(lambda: {
            "sessions": 0, "messages": 0, "input_tokens": 0,
            "output_tokens": 0, "cache_read_tokens": 0,
            "cache_write_tokens": 0, "total_tokens": 0, "tool_calls": 0,
        })
        for s in sessions:
            d = platform_data[s.get("source") or "unknown"]
            d["sessions"] += 1
            d["messages"] += s.get("message_count") or 0
            for k in _TOKEN_KEYS:
                d[k] += s.get(k) or 0
                d["total_tokens"] += s.get(k) or 0
            d["tool_calls"] += s.get("tool_call_count") or 0

        result = [{"platform": platform, **data} for platform, data in platform_data.items()]
        result.sort(key=lambda x: x["sessions"], reverse=True)
        return result

    def _compute_tool_breakdown(self, tool_usage: List[Dict]) -> List[Dict]:
        """Ranked tool list with percentages."""
        total_calls = sum(t["count"] for t in tool_usage) if tool_usage else 0
        return [
            {
                "tool": t["tool_name"],
                "count": t["count"],
                "percentage": (t["count"] / total_calls * 100) if total_calls else 0,
            }
            for t in tool_usage
        ]

    def _compute_skill_breakdown(self, skill_usage: List[Dict]) -> Dict[str, Any]:
        """Per-skill usage → summary + ranked list."""
        total_skill_loads = sum(s["view_count"] for s in skill_usage) if skill_usage else 0
        total_skill_edits = sum(s["manage_count"] for s in skill_usage) if skill_usage else 0
        total_skill_actions = total_skill_loads + total_skill_edits

        top_skills = []
        for skill in skill_usage:
            total_count = skill["view_count"] + skill["manage_count"]
            top_skills.append({
                "skill": skill["skill"],
                "view_count": skill["view_count"],
                "manage_count": skill["manage_count"],
                "total_count": total_count,
                "percentage": (total_count / total_skill_actions * 100) if total_skill_actions else 0,
                "last_used_at": skill.get("last_used_at"),
            })
        top_skills.sort(
            key=lambda s: (
                s["total_count"], s["view_count"], s["manage_count"], s["last_used_at"] or 0, s["skill"],
            ),
            reverse=True,
        )
        return {
            "summary": {
                "total_skill_loads": total_skill_loads,
                "total_skill_edits": total_skill_edits,
                "total_skill_actions": total_skill_actions,
                "distinct_skills_used": len(skill_usage),
            },
            "top_skills": top_skills,
        }

    def _compute_activity_patterns(self, sessions: List[Dict]) -> Dict:
        """Activity by day of week, hour, and active-day streak."""
        day_counts = Counter()  # 0=Monday ... 6=Sunday
        hour_counts = Counter()
        daily_counts = Counter()  # "YYYY-MM-DD" -> count
        for s in sessions:
            ts = s.get("started_at")
            if not ts:
                continue
            dt = datetime.fromtimestamp(ts)
            day_counts[dt.weekday()] += 1
            hour_counts[dt.hour] += 1
            daily_counts[dt.strftime("%Y-%m-%d")] += 1

        day_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        day_breakdown = [{"day": day_names[i], "count": day_counts.get(i, 0)} for i in range(7)]
        hour_breakdown = [{"hour": i, "count": hour_counts.get(i, 0)} for i in range(24)]

        max_streak = 0
        if daily_counts:
            dates = [datetime.strptime(d, "%Y-%m-%d") for d in sorted(daily_counts)]
            current_streak = max_streak = 1
            for prev, cur in zip(dates, dates[1:]):
                current_streak = current_streak + 1 if (cur - prev).days == 1 else 1
                max_streak = max(max_streak, current_streak)

        return {
            "by_day": day_breakdown,
            "by_hour": hour_breakdown,
            "busiest_day": max(day_breakdown, key=lambda x: x["count"]),
            "busiest_hour": max(hour_breakdown, key=lambda x: x["count"]),
            "active_days": len(daily_counts),
            "max_streak": max_streak,
        }

    _TOP_METRICS = (
        ("Most messages", lambda s: s.get("message_count") or 0, "{} msgs"),
        ("Most tokens", lambda s: (s.get("input_tokens") or 0) + (s.get("output_tokens") or 0), "{:,} tokens"),
        ("Most tool calls", lambda s: s.get("tool_call_count") or 0, "{} calls"),
    )

    def _compute_top_sessions(self, sessions: List[Dict]) -> List[Dict]:
        """Notable sessions (longest, most messages, most tokens, most tool calls)."""
        top = []
        timed = [s for s in sessions if s.get("started_at") and s.get("ended_at")]
        if timed:
            longest = max(timed, key=lambda s: s["ended_at"] - s["started_at"])
            top.append({
                "label": "Longest session",
                "session_id": longest["id"][:16],
                "value": format_duration_compact(longest["ended_at"] - longest["started_at"]),
                "date": _day(longest["started_at"]),
            })
        for label, metric, fmt in self._TOP_METRICS:
            best = max(sessions, key=metric)
            value = metric(best)
            if value > 0:
                top.append({
                    "label": label,
                    "session_id": best["id"][:16],
                    "value": fmt.format(value),
                    "date": _day(best.get("started_at")),
                })
        return top

    # ------------------------------------------------------------- Formatting

    @staticmethod
    def _section(title: str) -> List[str]:
        return [f"  {title}", "  " + "─" * 56]

    def format_terminal(self, report: Dict) -> str:
        """Format the insights report for terminal display (CLI)."""
        if report.get("empty"):
            days = report.get("days", 30)
            src = f" (source: {report['source_filter']})" if report.get("source_filter") else ""
            return f"  No sessions found in the last {days} days{src}."

        o = report["overview"]
        period_label = f"Last {report['days']} days"
        if report.get("source_filter"):
            period_label += f" ({report['source_filter']})"
        padding = 58 - len(period_label) - 2
        left_pad = padding // 2
        lines = [
            "",
            "  ╔══════════════════════════════════════════════════════════╗",
            "  ║                    📊 Hermes Insights                    ║",
            f"  ║{' ' * left_pad} {period_label} {' ' * (padding - left_pad)}║",
            "  ╚══════════════════════════════════════════════════════════╝",
            "",
        ]

        if o.get("date_range_start") and o.get("date_range_end"):
            start_str = datetime.fromtimestamp(o["date_range_start"]).strftime("%b %d, %Y")
            end_str = datetime.fromtimestamp(o["date_range_end"]).strftime("%b %d, %Y")
            lines += [f"  Period: {start_str} — {end_str}", ""]

        lines += self._section("📋 Overview")
        lines.append(f"  Sessions:          {o['total_sessions']:<12}  Messages:        {o['total_messages']:,}")
        lines.append(f"  Tool calls:        {o['total_tool_calls']:<12,}  User messages:   {o['user_messages']:,}")
        lines.append(f"  Input tokens:      {o['total_input_tokens']:<12,}  Output tokens:   {o['total_output_tokens']:,}")
        lines.append(f"  Total tokens:      {o['total_tokens']:,}")
        if o["total_hours"] > 0:
            lines.append(f"  Active time:       ~{format_duration_compact(o['total_hours'] * 3600):<11}  Avg session:     ~{format_duration_compact(o['avg_session_duration'])}")
        lines += [f"  Avg msgs/session:  {o['avg_messages_per_session']:.1f}", ""]

        # Cost buckets: show included/unknown sessions instead of collapsing to $0.
        est_cost = o.get("estimated_cost", 0.0)
        included_sessions = o.get("included_cost_sessions", 0)
        unknown_sessions = o.get("unknown_cost_sessions", 0)
        if est_cost > 0 or included_sessions > 0 or unknown_sessions > 0:
            lines += self._section("💰 Cost")
            if est_cost > 0:
                lines.append(f"  Estimated:          {_fmt_est_cost(est_cost)}")
            if included_sessions > 0:
                lines.append(f"  Included:           {included_sessions} session(s) (subscription — no provider invoice)")
            if unknown_sessions > 0:
                lines.append(f"  Unknown:            {unknown_sessions} session(s) (no pricing data)")
            lines.append("")

        if report["models"]:
            lines += self._section("🤖 Models Used")
            lines.append(f"  {'Model':<30} {'Sessions':>8} {'Tokens':>12}")
            for m in report["models"]:
                lines.append(f"  {m['model'][:28]:<30} {m['sessions']:>8} {m['total_tokens']:>12,}")
            lines.append("")

        platforms = report["platforms"]
        if len(platforms) > 1 or (platforms and platforms[0]["platform"] != "cli"):
            lines += self._section("📱 Platforms")
            lines.append(f"  {'Platform':<14} {'Sessions':>8} {'Messages':>10} {'Tokens':>14}")
            for p in platforms:
                lines.append(f"  {p['platform']:<14} {p['sessions']:>8} {p['messages']:>10,} {p['total_tokens']:>14,}")
            lines.append("")

        if report["tools"]:
            lines += self._section("🔧 Top Tools")
            lines.append(f"  {'Tool':<28} {'Calls':>8} {'%':>8}")
            for t in report["tools"][:15]:
                lines.append(f"  {t['tool']:<28} {t['count']:>8,} {t['percentage']:>7.1f}%")
            if len(report["tools"]) > 15:
                lines.append(f"  ... and {len(report['tools']) - 15} more tools")
            lines.append("")

        skills = report.get("skills", {})
        top_skills = skills.get("top_skills", [])
        if top_skills:
            lines += self._section("🧠 Top Skills")
            lines.append(f"  {'Skill':<28} {'Loads':>7} {'Edits':>7} {'Last used':>11}")
            for skill in top_skills[:10]:
                last_used = _day(skill.get("last_used_at")) if skill.get("last_used_at") else "—"
                lines.append(
                    f"  {skill['skill'][:28]:<28} {skill['view_count']:>7,} {skill['manage_count']:>7,} {last_used:>11}"
                )
            summary = skills.get("summary", {})
            lines.append(
                f"  Distinct skills: {summary.get('distinct_skills_used', 0)}  "
                f"Loads: {summary.get('total_skill_loads', 0):,}  "
                f"Edits: {summary.get('total_skill_edits', 0):,}"
            )
            lines.append("")

        act = report.get("activity", {})
        if act.get("by_day"):
            lines += self._section("📅 Activity Patterns")
            bars = _bar_chart([d["count"] for d in act["by_day"]], max_width=15)
            for bar, d in zip(bars, act["by_day"]):
                lines.append(f"  {d['day']}  {bar:<15} {d['count']}")
            lines.append("")

            busy_hours = sorted(act["by_hour"], key=lambda x: x["count"], reverse=True)
            busy_hours = [h for h in busy_hours if h["count"] > 0][:5]
            if busy_hours:
                hour_strs = [f"{_hour12(h['hour'])} ({h['count']})" for h in busy_hours]
                lines.append(f"  Peak hours: {', '.join(hour_strs)}")
            if act.get("active_days"):
                lines.append(f"  Active days: {act['active_days']}")
            if act.get("max_streak") and act["max_streak"] > 1:
                lines.append(f"  Best streak: {act['max_streak']} consecutive days")
            lines.append("")

        if report.get("top_sessions"):
            lines += self._section("🏆 Notable Sessions")
            for ts in report["top_sessions"]:
                lines.append(f"  {ts['label']:<20} {ts['value']:<18} ({ts['date']}, {ts['session_id']})")
            lines.append("")

        return "\n".join(lines)

    def format_gateway(self, report: Dict) -> str:
        """Format the insights report for gateway/messaging (shorter)."""
        if report.get("empty"):
            return f"No sessions found in the last {report.get('days', 30)} days."

        o = report["overview"]
        lines = [
            f"📊 **Hermes Insights** — Last {report['days']} days\n",
            f"**Sessions:** {o['total_sessions']} | **Messages:** {o['total_messages']:,} | **Tool calls:** {o['total_tool_calls']:,}",
            f"**Tokens:** {o['total_tokens']:,} (in: {o['total_input_tokens']:,} / out: {o['total_output_tokens']:,})",
        ]
        if o["total_hours"] > 0:
            lines.append(f"**Active time:** ~{format_duration_compact(o['total_hours'] * 3600)} | **Avg session:** ~{format_duration_compact(o['avg_session_duration'])}")
        lines.append("")

        est_cost = o.get("estimated_cost", 0.0)
        included = o.get("included_cost_sessions", 0)
        unknown = o.get("unknown_cost_sessions", 0)
        cost_parts: list[str] = []
        if est_cost > 0:
            cost_parts.append(f"{_fmt_est_cost(est_cost)} estimated")
        if included > 0:
            cost_parts.append(f"{included} included (subscription)")
        if unknown > 0:
            cost_parts.append(f"{unknown} unknown")
        if cost_parts:
            lines += [f"**Cost:** {' | '.join(cost_parts)}", ""]

        if report["models"]:
            lines.append("**🤖 Models:**")
            for m in report["models"][:5]:
                lines.append(f"  {m['model'][:25]} — {m['sessions']} sessions, {m['total_tokens']:,} tokens")
            lines.append("")

        if len(report["platforms"]) > 1:
            lines.append("**📱 Platforms:**")
            for p in report["platforms"]:
                lines.append(f"  {p['platform']} — {p['sessions']} sessions, {p['messages']:,} msgs")
            lines.append("")

        if report["tools"]:
            lines.append("**🔧 Top Tools:**")
            for t in report["tools"][:8]:
                lines.append(f"  {t['tool']} — {t['count']:,} calls ({t['percentage']:.1f}%)")
            lines.append("")

        skills = report.get("skills", {})
        if skills.get("top_skills"):
            lines.append("**🧠 Top Skills:**")
            for skill in skills["top_skills"][:5]:
                suffix = f", last used {_day(skill['last_used_at'])}" if skill.get("last_used_at") else ""
                lines.append(
                    f"  {skill['skill']} — {skill['view_count']:,} loads, {skill['manage_count']:,} edits{suffix}"
                )
            lines.append("")

        act = report.get("activity", {})
        if act.get("busiest_day") and act.get("busiest_hour"):
            lines.append(f"**📅 Busiest:** {act['busiest_day']['day']}s ({act['busiest_day']['count']} sessions), {_hour12(act['busiest_hour']['hour'])} ({act['busiest_hour']['count']} sessions)")
            if act.get("active_days"):
                lines.append(f"**Active days:** {act['active_days']}")
            if act.get("max_streak", 0) > 1:
                lines.append(f"**Best streak:** {act['max_streak']} consecutive days")

        return "\n".join(lines)
