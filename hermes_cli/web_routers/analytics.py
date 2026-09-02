"""Raw-YAML config and token/cost analytics dashboard routes.

Extracted from ``hermes_cli.web_server``; helpers/state that tests monkeypatch on
``web_server`` stay there and are imported lazily at call time (cycle-safe).
"""

import yaml
import asyncio
import time
from fastapi import APIRouter
from hermes_cli.web_deps import late
from fastapi import HTTPException, Query
from hermes_cli.config import get_config_path, read_raw_config
from hermes_cli.web_models import RawConfigUpdate
from typing import Any, Dict, List, Optional

router = APIRouter()

# web_server helpers, late-bound so monkeypatch.setattr(web_server, ...) stays authoritative.
_approval_mode_of = late("_approval_mode_of")
_aux_task_summary = late("_aux_task_summary")
_aux_usage_rows = late("_aux_usage_rows")
_broadcast_gateway_session_info = late("_broadcast_gateway_session_info")
_is_other_profile = late("_is_other_profile")
_merge_aux_into_by_model = late("_merge_aux_into_by_model")
_open_session_db_for_profile = late("_open_session_db_for_profile")
_profile_scope = late("_profile_scope")
save_config = late("save_config")


# ---------------------------------------------------------------------------
# Computer Use (cua-driver) — cross-platform readiness + macOS permission grant
#
# cua-driver runs on macOS, Windows, and Linux. The desktop card reflects
# per-OS readiness: on macOS the Accessibility + Screen Recording TCC grants
# (which attach to cua-driver's OWN identity, com.trycua.driver — not Hermes,
# so no app entitlement is involved); elsewhere, driver health from
# `cua-driver doctor`. The grant flow is macOS-only (no TCC toggles to request
# on Windows/Linux).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Raw YAML config endpoint
# ---------------------------------------------------------------------------


@router.get("/api/config/raw")
async def get_config_raw(profile: Optional[str] = None):
    """Raw config.yaml text plus its resolved path.

    ``path`` is resolved inside ``_profile_scope`` so the Config page header
    shows the file the switched profile actually reads/writes — /api/status's
    ``config_path`` is machine-global and always reports the dashboard
    process's own profile, which is wrong under the global profile switcher.
    """
    def _run():
        with _profile_scope(profile):
            path = get_config_path()
        if not path.exists():
            return {"yaml": "", "path": str(path)}
        return {"yaml": path.read_text(encoding="utf-8"), "path": str(path)}

    return await asyncio.to_thread(_run)


@router.put("/api/config/raw")
async def update_config_raw(body: RawConfigUpdate, profile: Optional[str] = None):
    def _run():
        parsed = yaml.safe_load(body.yaml_text)
        if not isinstance(parsed, dict):
            raise HTTPException(status_code=400, detail="YAML must be a mapping")
        approvals_mode_changed = False
        with _profile_scope(body.profile or profile):
            # Full-document replacement: the editor owns the whole file; do not
            # merge omitted sections back from disk (#62723).
            approvals_mode_changed = _approval_mode_of(parsed) != _approval_mode_of(read_raw_config())
            save_config(parsed, merge_existing=False)
        # Same indicator refresh as the schema-driven save above.
        if approvals_mode_changed and not _is_other_profile(body.profile or profile):
            _broadcast_gateway_session_info()
        return {"ok": True}

    try:
        return await asyncio.to_thread(_run)
    except yaml.YAMLError as e:
        raise HTTPException(status_code=400, detail=f"Invalid YAML: {e}")


def _get_usage_analytics(days: int = 30, profile: Optional[str] = None):
    from agent.insights import InsightsEngine

    db = _open_session_db_for_profile(profile, read_only=True)
    try:
        cutoff = time.time() - (days * 86400)
        cur = db._conn.execute("""
            SELECT date(started_at, 'unixepoch') as day,
                   SUM(input_tokens) as input_tokens,
                   SUM(output_tokens) as output_tokens,
                   SUM(cache_read_tokens) as cache_read_tokens,
                   SUM(reasoning_tokens) as reasoning_tokens,
                   COALESCE(SUM(estimated_cost_usd), 0) as estimated_cost,
                   COALESCE(SUM(actual_cost_usd), 0) as actual_cost,
                   COUNT(*) as sessions,
                   SUM(COALESCE(api_call_count, 0)) as api_calls
            FROM sessions WHERE started_at > ?
            GROUP BY day ORDER BY day
        """, (cutoff,))
        daily = [dict(r) for r in cur.fetchall()]

        cur2 = db._conn.execute("""
            SELECT model,
                   SUM(input_tokens) as input_tokens,
                   SUM(output_tokens) as output_tokens,
                   COALESCE(SUM(estimated_cost_usd), 0) as estimated_cost,
                   COUNT(*) as sessions,
                   SUM(COALESCE(api_call_count, 0)) as api_calls
            FROM sessions WHERE started_at > ? AND model IS NOT NULL
            GROUP BY model ORDER BY SUM(input_tokens) + SUM(output_tokens) DESC
        """, (cutoff,))
        by_model = [dict(r) for r in cur2.fetchall()]

        # Fold in auxiliary usage (vision, compression, title_generation, ...)
        # recorded per (model, task) in session_model_usage. Aux calls never
        # touch the sessions counters, so this is add-only — no double count.
        # Without it the models list shows only the main agent model even when
        # aux models are actively burning tokens (issue #23270).
        aux_rows = _aux_usage_rows(db, cutoff)
        by_model = _merge_aux_into_by_model(by_model, aux_rows)

        cur3 = db._conn.execute("""
            SELECT SUM(input_tokens) as total_input,
                   SUM(output_tokens) as total_output,
                   SUM(cache_read_tokens) as total_cache_read,
                   SUM(reasoning_tokens) as total_reasoning,
                   COALESCE(SUM(estimated_cost_usd), 0) as total_estimated_cost,
                   COALESCE(SUM(actual_cost_usd), 0) as total_actual_cost,
                   COUNT(*) as total_sessions,
                   SUM(COALESCE(api_call_count, 0)) as total_api_calls
            FROM sessions WHERE started_at > ?
        """, (cutoff,))
        totals = dict(cur3.fetchone())
        usage = InsightsEngine(db).get_usage_breakdown(days=days)

        return {
            "daily": daily,
            "by_model": by_model,
            # Aux-task summary across models (vision, compression, ...). Lets
            # the dashboard answer "what is compression costing me" directly.
            "by_task": _aux_task_summary(aux_rows),
            "totals": totals,
            "period_days": days,
            "skills": usage["skills"],
            # Per-tool-name call counts (already computed by InsightsEngine);
            # the desktop Capabilities page aggregates these per toolset.
            "tools": usage["tools"],
        }
    finally:
        db.close()


@router.get("/api/analytics/usage")
async def get_usage_analytics(
    days: int = Query(30, ge=1, le=365),
    profile: Optional[str] = None,
):
    """``days`` is clamped to 1-365 (idea from #74778): huge or non-positive
    values would force expensive full-history SQL and InsightsEngine work, or
    produce empty/inverted time windows. The UI only offers 7/30/90-day
    presets."""
    return await asyncio.to_thread(_get_usage_analytics, days, profile)


def _get_models_analytics(days: int = 30, profile: Optional[str] = None):
    """Rich per-model analytics for the Models dashboard page.

    Returns token/cost/session breakdown per model plus capability metadata
    from models.dev (context window, vision, tools, reasoning, etc.).
    """
    db = _open_session_db_for_profile(profile, read_only=True)
    try:
        cutoff = time.time() - (days * 86400)

        cur = db._conn.execute("""
            SELECT model,
                   billing_provider,
                   SUM(input_tokens) as input_tokens,
                   SUM(output_tokens) as output_tokens,
                   SUM(cache_read_tokens) as cache_read_tokens,
                   SUM(reasoning_tokens) as reasoning_tokens,
                   COALESCE(SUM(estimated_cost_usd), 0) as estimated_cost,
                   COALESCE(SUM(actual_cost_usd), 0) as actual_cost,
                   COUNT(*) as sessions,
                   SUM(COALESCE(api_call_count, 0)) as api_calls,
                   SUM(tool_call_count) as tool_calls,
                   MAX(started_at) as last_used_at,
                   AVG(input_tokens + output_tokens) as avg_tokens_per_session
            FROM sessions WHERE started_at > ? AND model IS NOT NULL AND model != ''
            GROUP BY model, billing_provider
            ORDER BY SUM(input_tokens) + SUM(output_tokens) DESC
        """, (cutoff,))
        raw_rows = [dict(r) for r in cur.fetchall()]

        # Add auxiliary usage as (model, provider) rows so aux-only models
        # (dedicated vision/compression models) appear on the Models page
        # instead of being invisible (issue #23270). Keyed by
        # model+billing_provider to match the GROUP BY above.
        for aux in _aux_usage_rows(db, cutoff):
            raw_rows.append({
                "model": aux.get("model") or "unknown",
                "billing_provider": aux.get("billing_provider") or "",
                "input_tokens": aux.get("input_tokens") or 0,
                "output_tokens": aux.get("output_tokens") or 0,
                "cache_read_tokens": aux.get("cache_read_tokens") or 0,
                "reasoning_tokens": aux.get("reasoning_tokens") or 0,
                "estimated_cost": aux.get("estimated_cost") or 0,
                "actual_cost": 0,
                "sessions": aux.get("sessions") or 0,
                "api_calls": aux.get("api_calls") or 0,
                "tool_calls": 0,
                "last_used_at": aux.get("last_used_at"),
                "avg_tokens_per_session": 0,
                "aux_task": aux.get("task") or "",
            })

        # Session rows can be created before the first billable provider call
        # finishes. If that early row records only the model name, and a later
        # row for the same model has real accounting + billing_provider, the
        # Models page used to show a duplicate "0 tokens / — API calls" card
        # next to the real provider card. Fold those session-only rows into
        # the single accounted provider row when the ownership is unambiguous.
        rows_by_model: Dict[str, List[Dict[str, Any]]] = {}
        for row in raw_rows:
            rows_by_model.setdefault(row.get("model") or "", []).append(row)

        rows: List[Dict[str, Any]] = []
        for model_rows in rows_by_model.values():
            provider_rows = [r for r in model_rows if r.get("billing_provider")]
            if len(provider_rows) == 1:
                target = provider_rows[0]
                for row in model_rows:
                    if row is target or row.get("billing_provider"):
                        continue
                    has_usage = any(
                        (row.get(key) or 0) != 0
                        for key in (
                            "input_tokens",
                            "output_tokens",
                            "cache_read_tokens",
                            "reasoning_tokens",
                            "estimated_cost",
                            "actual_cost",
                            "api_calls",
                            "tool_calls",
                        )
                    )
                    if has_usage:
                        continue
                    target["sessions"] = (target.get("sessions") or 0) + (row.get("sessions") or 0)
                    target["last_used_at"] = max(target.get("last_used_at") or 0, row.get("last_used_at") or 0)
                    total_tokens = (target.get("input_tokens") or 0) + (target.get("output_tokens") or 0)
                    sessions = target.get("sessions") or 0
                    target["avg_tokens_per_session"] = total_tokens / sessions if sessions else 0
                rows.append(target)
                rows.extend(
                    r for r in model_rows
                    if r is not target
                    and (r.get("billing_provider") or any(
                        (r.get(key) or 0) != 0
                        for key in (
                            "input_tokens",
                            "output_tokens",
                            "cache_read_tokens",
                            "reasoning_tokens",
                            "estimated_cost",
                            "actual_cost",
                            "api_calls",
                            "tool_calls",
                        )
                    ))
                )
            else:
                rows.extend(model_rows)

        rows.sort(
            key=lambda r: (r.get("input_tokens") or 0) + (r.get("output_tokens") or 0),
            reverse=True,
        )

        models = []
        for row in rows:
            provider = row.get("billing_provider") or ""
            model_name = row["model"]
            caps = {}
            try:
                from agent.models_dev import get_model_capabilities
                mc = get_model_capabilities(provider=provider, model=model_name)
                if mc is not None:
                    caps = {
                        "supports_tools": mc.supports_tools,
                        "supports_vision": mc.supports_vision,
                        "supports_reasoning": mc.supports_reasoning,
                        "context_window": mc.context_window,
                        "max_output_tokens": mc.max_output_tokens,
                        "model_family": mc.model_family,
                    }
            except Exception:
                pass

            models.append({
                "model": model_name,
                "provider": provider,
                "input_tokens": row["input_tokens"],
                "output_tokens": row["output_tokens"],
                "cache_read_tokens": row["cache_read_tokens"],
                "reasoning_tokens": row["reasoning_tokens"],
                "estimated_cost": row["estimated_cost"],
                "actual_cost": row["actual_cost"],
                "sessions": row["sessions"],
                "api_calls": row["api_calls"],
                "tool_calls": row["tool_calls"],
                "last_used_at": row["last_used_at"],
                "avg_tokens_per_session": row["avg_tokens_per_session"],
                "capabilities": caps,
            })

        totals_cur = db._conn.execute("""
            SELECT COUNT(DISTINCT model) as distinct_models,
                   SUM(input_tokens) as total_input,
                   SUM(output_tokens) as total_output,
                   SUM(cache_read_tokens) as total_cache_read,
                   SUM(reasoning_tokens) as total_reasoning,
                   COALESCE(SUM(estimated_cost_usd), 0) as total_estimated_cost,
                   COALESCE(SUM(actual_cost_usd), 0) as total_actual_cost,
                   COUNT(*) as total_sessions,
                   SUM(COALESCE(api_call_count, 0)) as total_api_calls
            FROM sessions WHERE started_at > ? AND model IS NOT NULL AND model != ''
        """, (cutoff,))
        totals = dict(totals_cur.fetchone())

        return {
            "models": models,
            "totals": totals,
            "period_days": days,
        }
    finally:
        db.close()


@router.get("/api/analytics/models")
async def get_models_analytics(
    days: int = Query(30, ge=1, le=365),
    profile: Optional[str] = None,
):
    # ``days`` clamped to 1-365 (idea from #74778) — see get_usage_analytics.
    """Return model analytics without blocking the serving event loop."""
    return await asyncio.to_thread(_get_models_analytics, days, profile)
