"""Read-only gateway introspection commands: /status, /context, /usage, /agents, /insights, /topup.

Split out of ``gateway/slash_commands.py``; bound onto ``GatewayRunner`` through
``GatewaySlashCommandsMixin``. Origin internals are imported lazily (``from gateway.slash_commands
import ...``) inside the bodies to avoid the import cycle.
"""

from __future__ import annotations

import logging
import asyncio
import hashlib
import os
import re
import time
from typing import Any

from agent.account_usage import fetch_account_usage, render_account_usage_lines
from agent.i18n import t
from gateway.config import Platform
from gateway.platforms.base import MessageEvent

# Log-record parity with gateway/run.py and the origin module.
logger = logging.getLogger("gateway.run")


def _clean_str(value: Any) -> str:
    """Strip and return a non-empty string value, or empty string."""
    return value.strip() if isinstance(value, str) and value.strip() else ""


def _int_value(value: Any) -> int:
    """Safely coerce to int."""
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _status_model_route(status_agent, persisted_route: dict, session_row: dict, session_entry):
    """``(model, provider, context_used, context_total)`` for /status.

    Order: live/cached agent route -> persisted dominant route -> SessionDB row -> gateway config
    (only loaded when something is still missing).
    """
    from gateway.run import _AGENT_PENDING_SENTINEL, _load_gateway_config, _resolve_gateway_model

    model_name = provider_name = ""
    route_resolved = False
    context_used = context_total = 0
    if status_agent is not None and status_agent is not _AGENT_PENDING_SENTINEL:
        live_model = _clean_str(getattr(status_agent, "model", ""))
        live_provider = _clean_str(getattr(status_agent, "provider", ""))
        if live_model and live_provider:
            model_name, provider_name, route_resolved = live_model, live_provider, True
        ctx = getattr(status_agent, "context_compressor", None)
        if ctx is not None:
            context_used = _int_value(getattr(ctx, "last_prompt_tokens", 0))
            context_total = _int_value(getattr(ctx, "context_length", 0))

    persisted_model = _clean_str(persisted_route.get("model"))
    persisted_provider = _clean_str(persisted_route.get("billing_provider"))
    if not route_resolved and persisted_model and persisted_provider:
        model_name, provider_name, route_resolved = persisted_model, persisted_provider, True
    if not route_resolved:
        model_name = _clean_str(session_row.get("model"))
        provider_name = _clean_str(session_row.get("billing_provider"))
    context_used = context_used or _int_value(getattr(session_entry, "last_prompt_tokens", 0))

    user_config: dict[str, Any] = {}
    if not model_name or not provider_name or not context_total:
        try:
            user_config = _load_gateway_config()
        except Exception:
            user_config = {}
    model_cfg = user_config.get("model", {}) if isinstance(user_config, dict) else {}
    if not isinstance(model_cfg, dict):
        model_cfg = {}
    if not model_name:
        model_name = _resolve_gateway_model(user_config)
    if not provider_name:
        provider_name = _clean_str(model_cfg.get("provider"))
    if not context_total:
        configured_context = model_cfg.get("context_length")
        if isinstance(configured_context, int) and configured_context > 0:
            context_total = configured_context
    return model_name, provider_name, context_used, context_total


def _context_compressor_lines(agent, ctx, used: int) -> list[str]:
    """/context full view: auto-compression threshold/headroom, compression count + last savings,
    and cumulative throughput (labelled as throughput, NOT context size)."""
    lines: list[str] = []
    threshold = getattr(ctx, "threshold_tokens", 0) or 0
    threshold_pct = (getattr(ctx, "threshold_percent", 0) or 0) * 100
    if threshold > 0:
        if used >= threshold:
            lines.append(
                t("gateway.context.over_threshold", threshold=f"{threshold:,}", threshold_pct=f"{threshold_pct:.0f}")
            )
        else:
            lines.append(
                t(
                    "gateway.context.threshold",
                    threshold=f"{threshold:,}",
                    threshold_pct=f"{threshold_pct:.0f}",
                    to_go=f"{threshold - used:,}",
                )
            )
    compressions = getattr(ctx, "compression_count", 0) or 0
    lines.append(t("gateway.context.compressions", count=compressions))
    if compressions:
        savings = getattr(ctx, "_last_compression_savings_pct", None)
        if savings is not None:
            lines.append(t("gateway.context.last_savings", savings=f"{savings:.0f}"))

    def _n(attr):
        return getattr(agent, attr, 0) or 0

    lines.append("")
    lines.append(t("gateway.context.totals_header", calls=_n("session_api_calls")))
    lines.append(
        t(
            "gateway.context.totals_line",
            input=f"{_n('session_input_tokens'):,}",
            output=f"{_n('session_output_tokens'):,}",
            reasoning=f"{_n('session_reasoning_tokens'):,}",
        )
    )
    lines.append(t("gateway.context.total_billed", total=f"{_n('session_total_tokens'):,}"))
    lines.append(t("gateway.context.throughput_note"))
    return lines


def _agents_delegation_lines(d: dict) -> list[str]:
    """/agents rows for one background delegation. Live per-child activity comes from the
    registry's progress sampler: api calls, current tool, seconds since last activity."""
    goal = " ".join(str(d.get("goal") or "").split())
    if len(goal) > 70:
        goal = goal[:67] + "..."
    status = d.get("status", "?")
    row = f"- `{d.get('delegation_id', '?')}` · {status}"
    if status == "stalling":
        quiet = d.get("stalled_after_quiet_seconds")
        if quiet is not None:
            row += f" · no progress {quiet:.0f}s"
    elif d.get("seconds_since_progress", 0) >= 60:
        row += f" · quiet {d['seconds_since_progress']:.0f}s"
    if goal:
        row += f" · {goal}"
    lines = [row]
    for i, child in enumerate(d.get("children_activity") or []):
        if not isinstance(child, dict):
            continue
        tool = child.get("current_tool")
        doing = f"`{tool}`" if tool else "between turns"
        part = f"  - child {i + 1}: {child.get('api_calls', '?')} api calls · {doing}"
        idle = child.get("seconds_since_activity")
        if idle is not None:
            part += f" · active {idle:.0f}s ago"
        lines.append(part)
    return lines


def _usage_agent_stats_lines(agent) -> list[str]:
    """/usage session block for a live agent: rate limits, token breakdown (matches the CLI),
    context window and compression count."""
    lines: list[str] = []
    rl_state = agent.get_rate_limit_state()
    if rl_state and rl_state.has_data:
        from agent.rate_limit_tracker import format_rate_limit_compact
        lines.append(t("gateway.usage.rate_limits", state=format_rate_limit_compact(rl_state)))
        lines.append("")
    input_tokens = getattr(agent, "session_input_tokens", 0) or 0
    output_tokens = getattr(agent, "session_output_tokens", 0) or 0
    lines.append(t("gateway.usage.header_session"))
    lines.append(t("gateway.usage.label_model", model=agent.model))
    lines.append(t("gateway.usage.label_input_tokens", count=f"{input_tokens:,}"))
    lines.append(t("gateway.usage.label_output_tokens", count=f"{output_tokens:,}"))
    lines.append(t("gateway.usage.label_total", count=f"{agent.session_total_tokens:,}"))
    lines.append(t("gateway.usage.label_api_calls", count=agent.session_api_calls))
    ctx = agent.context_compressor
    _lpt = ctx.last_prompt_tokens if ctx.last_prompt_tokens > 0 else 0
    if _lpt:
        pct = min(100, _lpt / ctx.context_length * 100) if ctx.context_length else 0
        lines.append(t("gateway.usage.label_context", used=f"{_lpt:,}", total=f"{ctx.context_length:,}", pct=f"{pct:.0f}"))
    if ctx.compression_count:
        lines.append(t("gateway.usage.label_compressions", count=ctx.compression_count))
    return lines


class GatewayStatusCommandsMixin:
    """Read-only gateway introspection commands: /status, /context, /usage, /agents, /insights, /topup."""

    async def _handle_status_command(self, event: MessageEvent) -> str:
        """Handle /status command."""
        from gateway.run import _AGENT_PENDING_SENTINEL

        source = event.source
        session_entry = await self.async_session_store.get_or_create_session(source)

        connected_platforms = [p.value for p in self.adapters]

        # Check if there's an active agent. Keep the sentinel distinct: a
        # starting/pending run should not be treated as a fully usable agent for
        # model/context display, but it still occupies the session slot.
        session_key = session_entry.session_key
        agent = self._running_agents.get(session_key)
        is_running = agent is not None and agent is not _AGENT_PENDING_SENTINEL

        # Count pending /queue follow-ups (slot + overflow).
        adapter = self.adapters.get(source.platform) if source else None
        queue_depth = self._queue_depth(session_key, adapter=adapter)

        title, session_row, db_total_tokens, persisted_route = await self._status_session_db_facts(
            session_entry.session_id
        )
        # Resolve model/context for cockpit-style status. Prefer the live or cached agent because it
        # carries the actual runtime route and context compressor; fall back to SessionDB metadata +
        # last_prompt_tokens so /status stays useful between turns without billing/account calls.
        status_agent = agent if is_running else self._cached_agent_for(session_key)
        model_name, provider_name, context_used, context_total = _status_model_route(
            status_agent, persisted_route, session_row, session_entry
        )

        model_line = ""
        if model_name:
            if provider_name:
                model_line = t("gateway.status.model_provider", model=model_name, provider=provider_name)
            else:
                model_line = t("gateway.status.model", model=model_name)

        context_line = ""
        if context_total:
            pct = min(100, round((context_used / context_total) * 100)) if context_total else 0
            context_line = t(
                "gateway.status.context",
                used=f"{context_used:,}",
                total=f"{context_total:,}",
                pct=f"{pct}",
            )
        elif context_used:
            context_line = t("gateway.status.context_used", used=f"{context_used:,}")

        lines = [
            t("gateway.status.header"),
            "",
            t("gateway.status.session_id", session_id=session_entry.session_id),
        ]
        if title:
            lines.append(t("gateway.status.title", title=title))
        lines.extend([
            t("gateway.status.created", timestamp=session_entry.created_at.strftime('%Y-%m-%d %H:%M')),
            t("gateway.status.last_activity", timestamp=session_entry.updated_at.strftime('%Y-%m-%d %H:%M')),
        ])
        if model_line:
            lines.append(model_line)
        if context_line:
            lines.append(context_line)
        lines.extend([
            t("gateway.status.tokens", tokens=f"{db_total_tokens:,}"),
            t("gateway.status.agent_running", state=t("gateway.status.state_yes") if is_running else t("gateway.status.state_no")),
        ])
        if queue_depth:
            lines.append(t("gateway.status.queued", count=queue_depth))
        if source.platform == Platform.MATRIX:
            scope = getattr(self.adapters.get(Platform.MATRIX), "_matrix_session_scope", os.getenv("MATRIX_SESSION_SCOPE", "auto"))
            thread = source.thread_id or "none"
            lines.extend([
                "",
                t("gateway.status.matrix_scope_header"),
                t("gateway.status.matrix_scope_room", room=source.chat_name or source.chat_id),
                t("gateway.status.matrix_scope_room_id", room_id=source.chat_id),
                t("gateway.status.matrix_scope_thread", thread_id=thread),
                t("gateway.status.matrix_scope_mode", scope=scope),
                t(
                    "gateway.status.matrix_scope_key",
                    session_key=self._redact_matrix_session_key(session_key),
                ),
            ])
        lines.extend([
            "",
            t("gateway.status.platforms", platforms=', '.join(connected_platforms)),
        ])

        return "\n".join(lines)

    async def _status_session_db_facts(self, session_id: str):
        """``(title, session_row, db_total_tokens, persisted_route)`` for /status; each fail-open.

        Token totals come from the SQLite session DB rather than the in-memory SessionStore: the
        agent's per-turn token deltas are persisted into sessions_db (run_agent.py), not into
        SessionEntry, so session_entry.total_tokens is always 0.
        """
        title = None
        session_row: dict[str, Any] = {}
        db_total_tokens = 0
        persisted_route: dict[str, Any] = {}
        if not self._session_db:
            return title, session_row, db_total_tokens, persisted_route
        try:
            title = await self._session_db.get_session_title(session_id)
        except Exception:
            title = None
        try:
            row = await self._session_db.get_session(session_id)
            if isinstance(row, dict):
                session_row = row
                db_total_tokens = sum(
                    _int_value(row.get(k))
                    for k in ("input_tokens", "output_tokens", "cache_read_tokens", "cache_write_tokens", "reasoning_tokens")
                )
        except Exception:
            db_total_tokens = 0
        try:
            route = await self._session_db.get_dominant_session_model_route(session_id)
            if isinstance(route, dict):
                persisted_route = route
        except Exception:
            persisted_route = {}
        return title, session_row, db_total_tokens, persisted_route

    @staticmethod
    def _redact_matrix_session_key(session_key: str) -> str:
        """Return a stable Matrix session-key fingerprint for shared room status."""
        text = str(session_key or "")
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
        return f"sha256:{digest}"

    async def _handle_context_command(self, event: MessageEvent) -> str:
        """Handle /context — the dedicated context-window view.

        /status shows a one-line ``used / total`` summary; this command is the deep view: a usage
        gauge, auto-compression threshold and headroom, compression count and last savings, and
        cumulative throughput — the last clearly labelled as throughput, NOT context size.
        Resolution order: running agent, cached agent, SessionStore/SessionDB metadata, and a
        transcript estimate only as last resort. ``/context all`` adds per-skill/toolset listings.
        """
        source = event.source
        session_key = self._session_key_for_source(source)
        session_entry = await self.async_session_store.get_or_create_session(source)
        expanded = event.get_command_args().strip().lower() in {"all", "full", "details"}

        # Running agent first (mid-turn), then cached agent (between turns).
        agent = self._resident_agent_for(session_key)
        has_agent = bool(agent)

        ctx = getattr(agent, "context_compressor", None) if has_agent else None
        used, context_length, model_name = await self._resolve_context_figures(
            agent if has_agent else None, ctx, session_entry, source
        )

        # Gauge path: real current-context figure
        if used > 0 and context_length > 0:
            pct = min(100.0, used / context_length * 100)
            headroom = max(0, context_length - used)
            BAR_WIDTH = 24
            filled = int(round(pct / 100 * BAR_WIDTH))
            bar = "█" * max(0, filled) + "░" * max(0, BAR_WIDTH - filled)

            lines = [
                t("gateway.context.header"),
                "",
                t("gateway.context.model", model=model_name or "?"),
                t("gateway.context.window", total=f"{context_length:,}"),
                t(
                    "gateway.context.in_use",
                    used=f"{used:,}",
                    total=f"{context_length:,}",
                    pct=f"{pct:.0f}",
                ),
                t("gateway.context.bar", bar=bar),
                t("gateway.context.headroom", headroom=f"{headroom:,}"),
                "",
            ]
            # Full view — compression / throughput need the live agent.
            if ctx is not None:
                lines.extend(_context_compressor_lines(agent, ctx, used))
            else:
                lines.append(t("gateway.context.detail_after_first"))

            # Per-category estimated breakdown (+ optional expanded listings). Same chars/4 engine
            # the desktop popover and /usage use; plain text (no glyph grid — monospace isn't
            # guaranteed on messaging platforms). Fail-open: rendering errors never break /context.
            if has_agent:
                breakdown = await asyncio.to_thread(
                    self._context_breakdown_block, agent, source, expanded
                )
                if breakdown:
                    lines.append("")
                    lines.extend(breakdown)

            return "\n".join(lines)

        # Last resort: rough estimate from transcript
        history = await self.async_session_store.load_transcript(session_entry.session_id)
        if history:
            from agent.model_metadata import estimate_messages_tokens_rough

            msgs = [
                m
                for m in history
                if m.get("role") in {"user", "assistant"} and m.get("content")
            ]
            approx = estimate_messages_tokens_rough(msgs)
            return "\n".join(
                [
                    t("gateway.context.header"),
                    "",
                    t(
                        "gateway.context.estimated",
                        count=f"{approx:,}",
                        messages=len(msgs),
                    ),
                    t("gateway.context.detail_after_first"),
                ]
            )
        return t("gateway.context.no_data")

    async def _resolve_context_figures(self, agent, ctx, session_entry, source):
        """``(used, context_length, model_name)`` for /context with cascading fallbacks.

        used  : compressor.last_prompt_tokens -> SessionStore.last_prompt_tokens
        model : agent.model -> SessionDB row model
        window: compressor.context_length -> effective gateway model route -> model metadata
        """
        used = context_length = 0
        if ctx is not None:
            used = getattr(ctx, "last_prompt_tokens", 0) or 0
            context_length = getattr(ctx, "context_length", 0) or 0
        model_name = _clean_str(getattr(agent, "model", "")) if agent is not None else ""
        if not used:
            used = _int_value(getattr(session_entry, "last_prompt_tokens", 0))
        if not model_name and self._session_db:
            try:
                row = await self._session_db.get_session(session_entry.session_id) or {}
                if isinstance(row, dict):
                    model_name = _clean_str(row.get("model", ""))
            except Exception:
                model_name = ""
        if not context_length:
            try:
                from gateway.run import _profile_runtime_scope, _resolve_gateway_model_context

                def _resolve_nonresident_context():
                    if getattr(getattr(self, "config", None), "multiplex_profiles", False):
                        profile_home = self._resolve_profile_home_for_source(source)
                        with _profile_runtime_scope(profile_home):
                            return _resolve_gateway_model_context(model_name or None)
                    return _resolve_gateway_model_context(model_name or None)

                resolved = await asyncio.to_thread(_resolve_nonresident_context)
                model_name = model_name or resolved.model
                context_length = _int_value(resolved.context_length)
            except Exception:
                context_length = 0
        if not context_length and model_name:
            try:
                from agent.model_metadata import get_model_context_length

                context_length = _int_value(await asyncio.to_thread(get_model_context_length, model_name))
            except Exception:
                context_length = 0
        return used, context_length, model_name

    async def _handle_agents_command(self, event: MessageEvent) -> str:
        """Handle /agents command - list active agents and running tasks."""
        from gateway.run import _AGENT_PENDING_SENTINEL
        from tools.process_registry import format_uptime_short, process_registry

        now = time.time()
        current_session_key = self._session_key_for_source(event.source)

        running_agents: dict = getattr(self, "_running_agents", {}) or {}
        running_started: dict = getattr(self, "_running_agents_ts", {}) or {}

        agent_rows: list[dict] = []
        for session_key, agent in running_agents.items():
            started = float(running_started.get(session_key, now))
            elapsed = max(0, int(now - started))
            is_pending = agent is _AGENT_PENDING_SENTINEL
            agent_rows.append(
                {
                    "session_key": session_key,
                    "elapsed": elapsed,
                    "state": t("gateway.agents.state_starting") if is_pending else t("gateway.agents.state_running"),
                    "session_id": "" if is_pending else str(getattr(agent, "session_id", "") or ""),
                    "model": "" if is_pending else str(getattr(agent, "model", "") or ""),
                }
            )

        agent_rows.sort(key=lambda row: row["elapsed"], reverse=True)

        running_processes: list[dict] = []
        try:
            running_processes = [
                p for p in process_registry.list_sessions()
                if p.get("status") == "running"
            ]
        except Exception:
            running_processes = []

        background_tasks = [
            t for t in (getattr(self, "_background_tasks", set()) or set())
            if hasattr(t, "done") and not t.done()
        ]

        lines = [
            t("gateway.agents.header"),
            "",
            t("gateway.agents.active_agents", count=len(agent_rows)),
        ]

        if agent_rows:
            for idx, row in enumerate(agent_rows[:12], 1):
                current = t("gateway.agents.this_chat") if row["session_key"] == current_session_key else ""
                sid = f" · `{row['session_id']}`" if row["session_id"] else ""
                model = f" · `{row['model']}`" if row["model"] else ""
                lines.append(
                    f"{idx}. `{row['session_key']}` · {row['state']} · "
                    f"{format_uptime_short(row['elapsed'])}{sid}{model}{current}"
                )
            if len(agent_rows) > 12:
                lines.append(t("gateway.agents.more", count=len(agent_rows) - 12))

        lines.extend(
            [
                "",
                t("gateway.agents.running_processes", count=len(running_processes)),
            ]
        )
        if running_processes:
            for proc in running_processes[:12]:
                cmd = " ".join(str(proc.get("command", "")).split())
                if len(cmd) > 90:
                    cmd = cmd[:87] + "..."
                lines.append(
                    f"- `{proc.get('session_id', '?')}` · "
                    f"{format_uptime_short(int(proc.get('uptime_seconds', 0)))} · `{cmd}`"
                )
            if len(running_processes) > 12:
                lines.append(t("gateway.agents.more", count=len(running_processes) - 12))

        lines.extend(
            [
                "",
                t("gateway.agents.async_jobs", count=len(background_tasks)),
            ]
        )

        # Background (async) delegations — delegate_task(background=true).
        try:
            from tools.async_delegation import list_async_delegations
            delegations = [
                d for d in list_async_delegations()
                if d.get("status") in ("running", "stalling", "finalizing")
            ]
        except Exception:
            delegations = []
        if delegations:
            lines.extend(["", t("gateway.agents.background_delegations", count=len(delegations))])
            for d in delegations[:12]:
                lines.extend(_agents_delegation_lines(d))
            if len(delegations) > 12:
                lines.append(t("gateway.agents.more", count=len(delegations) - 12))

        if (
            not agent_rows
            and not running_processes
            and not background_tasks
            and not delegations
        ):
            lines.append("")
            lines.append(t("gateway.agents.none"))

        return "\n".join(lines)

    async def _handle_topup_command(self, event: MessageEvent) -> str:
        """Handle /topup -- show the Nous balance and hand off to the portal.

        Does NOT charge, confirm, or track payment — that happens in the browser; the next /topup
        shows the new balance. Fetched off the event loop; fail-open.
        """
        from agent.account_usage import build_credits_view

        try:
            view = await asyncio.to_thread(build_credits_view, markdown=True)
        except Exception:
            view = None

        if view is None or not view.logged_in:
            return t("gateway.credits.not_logged_in")

        lines: list[str] = ["💳 **Nous balance**"]
        for line in view.balance_lines:
            if line.lstrip().startswith("📈"):
                continue  # drop the helper's header; we print our own
            lines.append(line)
        if view.identity_line:
            lines.append("")
            lines.append(view.identity_line)
        if view.topup_url:
            lines.append("")
            lines.append(f"Manage billing on the portal: {view.topup_url}")
            lines.append("Top up and manage billing in the browser — your balance updates here after.")
        return "\n".join(lines)

    def _context_breakdown_block(self, agent, source, expanded: bool) -> list[str]:
        """Render the /context per-category block (plain text, no grid).

        Estimated (chars/4), same engine as /usage. Runs in a thread; returns [] and never raises.
        """
        try:
            from agent.context_breakdown import compute_context_details, render_context_breakdown_lines

            payload = self._session_context_breakdown(agent, source)
            if not (payload.get("categories") or []):
                return []

            details = None
            if expanded:
                try:
                    details = compute_context_details(agent)
                except Exception:
                    details = {"skills": [], "toolsets": []}

            return render_context_breakdown_lines(payload, details=details, grid=False)
        except Exception:
            return []

    def _session_context_breakdown(self, agent, source) -> dict:
        """Per-category context estimate (chars/4) for *agent* over the session transcript (sync)."""
        from agent.context_breakdown import compute_session_context_breakdown

        history: list[dict] = []
        try:
            entry = self.session_store.get_or_create_session(source)
            history = self.session_store.load_transcript(entry.session_id) or []
        except Exception:
            history = []
        return compute_session_context_breakdown(agent, history)

    def _context_breakdown_lines(self, agent, source) -> list[str]:
        """Render the per-category context breakdown for /usage.

        Estimated (chars/4). Returns [] and never raises so /usage stays robust.
        """
        try:
            payload = self._session_context_breakdown(agent, source)
            categories = payload.get("categories") or []
            if not categories:
                return []

            total = payload.get("estimated_total") or 0
            out = [t("gateway.usage.breakdown_header")]
            for cat in categories:
                tokens = int(cat.get("tokens") or 0)
                if tokens <= 0:
                    continue
                cat_id = str(cat.get("id") or "")
                label = t(f"gateway.usage.breakdown_cat_{cat_id}")
                # Missing key → t() echoes the key back; fall back to the
                # English label the engine already provides.
                if label.endswith(f"breakdown_cat_{cat_id}"):
                    label = str(cat.get("label") or cat_id)
                pct = round(tokens / total * 100) if total else 0
                out.append(
                    t("gateway.usage.breakdown_line", label=label, count=f"{tokens:,}", pct=pct)
                )
            return out if len(out) > 1 else []
        except Exception:
            return []

    async def _handle_usage_command(self, event: MessageEvent) -> str:
        """Handle /usage command -- show token usage for the current session.

        Checks both _running_agents (mid-turn) and _agent_cache (between turns) so details are
        available whenever the user asks.
        """
        source = event.source
        session_key = self._session_key_for_source(source)

        # `/usage reset [--force]` — redeem one banked Codex rate-limit reset
        # credit. Parsed before the display path so it never mixes with the
        # stats rendering below.
        raw_args = event.get_command_args().strip()
        args = [a.lower() for a in raw_args.split()] if raw_args else []
        wants_reset = bool(args) and args[0] == "reset"
        if args and not wants_reset:
            return t("gateway.usage.unknown_subcommand", args=raw_args)

        # Running agent first (mid-turn), then cached agent (between turns).
        agent = self._resident_agent_for(session_key)

        # Resolve provider/base_url/api_key for the account-usage fetch. Prefer the live agent; fall
        # back to persisted billing data on the SessionDB row so `/usage` still returns account info
        # between turns when no agent is resident.
        provider = getattr(agent, "provider", None) if agent else None
        base_url = getattr(agent, "base_url", None) if agent else None
        api_key = getattr(agent, "api_key", None) if agent else None
        if not provider and getattr(self, "_session_db", None) is not None:
            provider, base_url = await self._persisted_billing_route(source)

        if wants_reset:
            normalized_provider = str(provider or "").strip().lower()
            if normalized_provider != "openai-codex":
                return t("gateway.usage.reset_wrong_provider")
            force = "--force" in args[1:]
            from agent.account_usage import redeem_codex_reset_credit

            result = await asyncio.to_thread(
                redeem_codex_reset_credit,
                base_url=base_url,
                api_key=api_key,
                force=force,
            )
            return result.message

        # Fetch account usage off the event loop so slow provider APIs don't
        # block the gateway. Failures are non-fatal -- account_lines stays [].
        account_lines: list[str] = []
        credits_lines: list[str] = []
        if provider:
            try:
                account_snapshot = await asyncio.to_thread(
                    fetch_account_usage,
                    provider,
                    base_url=base_url,
                    api_key=api_key,
                )
            except Exception:
                account_snapshot = None
            if account_snapshot:
                account_lines = render_account_usage_lines(account_snapshot, markdown=True)

        # ── Nous credits magnitudes + monthly-grant % gauge ─────────────
        # Shared with CLI/TUI via nous_credits_lines(); run off the event loop. Gates on "a Nous
        # account is logged in" — NOT the inference provider, NOT under `if provider:` — so a Nous
        # user inferring elsewhere still sees a balance. No recovery trigger; fail-open.
        try:
            from agent.account_usage import nous_credits_lines

            credits_lines = await asyncio.to_thread(nous_credits_lines, markdown=True)
        except Exception:
            credits_lines = []  # fail-open: never break /usage

        def _with_account_blocks(lines: list[str]) -> str:
            # Each block is preceded by a blank divider only when something precedes it.
            for block in (account_lines, credits_lines):
                if block:
                    if lines:
                        lines.append("")
                    lines.extend(block)
            return "\n".join(lines)

        if agent and hasattr(agent, "session_total_tokens") and agent.session_api_calls > 0:
            lines = _usage_agent_stats_lines(agent)
            # Per-category context breakdown (estimated — chars/4 heuristic). Same engine the
            # desktop popover uses. The system prompt / tools / skills / memory slices read off the
            # live agent; the conversation slice is estimated from the session transcript.
            breakdown_lines = await asyncio.to_thread(self._context_breakdown_lines, agent, source)
            if breakdown_lines:
                lines.append("")
                lines.extend(breakdown_lines)
            return _with_account_blocks(lines)

        # No agent at all -- check session history for a rough count
        session_entry = await self.async_session_store.get_or_create_session(source)
        history = await self.async_session_store.load_transcript(session_entry.session_id)
        if history:
            from agent.model_metadata import estimate_messages_tokens_rough
            msgs = [m for m in history if m.get("role") in {"user", "assistant"} and m.get("content")]
            approx = estimate_messages_tokens_rough(msgs)
            return _with_account_blocks([
                t("gateway.usage.header_session_info"),
                t("gateway.usage.label_messages", count=len(msgs)),
                t("gateway.usage.label_estimated_context", count=f"{approx:,}"),
                t("gateway.usage.detailed_after_first"),
            ])
        if account_lines or credits_lines:
            return _with_account_blocks([])
        return t("gateway.usage.no_data")

    async def _persisted_billing_route(self, source):
        """``(provider, base_url)`` from the SessionDB row / dominant route when no agent is resident."""
        try:
            entry = await self.async_session_store.get_or_create_session(source)
            persisted = await self._session_db.get_session(entry.session_id) or {}
            route = await self._session_db.get_dominant_session_model_route(entry.session_id)
            persisted_route = route if isinstance(route, dict) else {}
        except Exception:
            persisted = {}
            persisted_route = {}
        if persisted_route.get("billing_provider"):
            return persisted_route["billing_provider"], persisted_route.get("billing_base_url")
        return persisted.get("billing_provider"), persisted.get("billing_base_url")

    async def _handle_insights_command(self, event: MessageEvent) -> str:
        """Handle /insights command -- show usage insights and analytics."""
        args = event.get_command_args().strip()

        # Normalize Unicode dashes (Telegram/iOS auto-converts -- to em/en dash)
        args = re.sub(r'[\u2012\u2013\u2014\u2015](days|source)', r'--\1', args)

        days = 30
        source = None

        # Parse simple args: /insights 7  or  /insights --days 7
        if args:
            parts = args.split()
            i = 0
            while i < len(parts):
                if parts[i] == "--days" and i + 1 < len(parts):
                    try:
                        days = int(parts[i + 1])
                    except ValueError:
                        return t("gateway.insights.invalid_days", value=parts[i + 1])
                    i += 2
                elif parts[i] == "--source" and i + 1 < len(parts):
                    source = parts[i + 1]
                    i += 2
                elif parts[i].isdigit():
                    days = int(parts[i])
                    i += 1
                else:
                    i += 1

        try:
            from hermes_state import get_shared_session_db
            from agent.insights import InsightsEngine

            def _run_insights():
                db = get_shared_session_db()
                try:
                    engine = InsightsEngine(db)
                    report = engine.generate(days=days, source=source)
                    result = engine.format_gateway(report)
                    return result
                finally:
                    from hermes_state import release_or_close
                    release_or_close(db)

            # Not a bare hop: ``SessionDB()`` resolves ``get_hermes_home()`` at call time, which is
            # a contextvar set by ``_profile_runtime_scope``; a default-executor hop starts with an
            # EMPTY context and would read the DEFAULT profile's state.db.
            return await self._run_in_executor_with_context(_run_insights)
        except Exception as e:
            logger.error("Insights command error: %s", e, exc_info=True)
            return t("gateway.insights.error", error=e)
