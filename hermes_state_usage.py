"""Token/usage accounting mixin for SessionDB: the coalescing background
token writer, per-model usage rows, and billing-route columns.  Writer thread
state lives on the SessionDB instance."""

from __future__ import annotations

import atexit
import logging
import threading
import time
import weakref
from typing import Any, Dict, List, Optional, Tuple

# caplog tests pin the "hermes_state" logger name.
logger = logging.getLogger("hermes_state")

_TOKEN_UPDATE_ABSOLUTE_SQL = """UPDATE sessions SET
                   input_tokens = ?,
                   output_tokens = ?,
                   cache_read_tokens = ?,
                   cache_write_tokens = ?,
                   reasoning_tokens = ?,
                   estimated_cost_usd = COALESCE(?, 0),
                   actual_cost_usd = CASE
                       WHEN ? IS NULL THEN actual_cost_usd
                       ELSE ?
                   END,
                   cost_status = COALESCE(?, cost_status),
                   cost_source = COALESCE(?, cost_source),
                   pricing_version = COALESCE(?, pricing_version),
                   billing_provider = COALESCE(billing_provider, ?),
                   billing_base_url = COALESCE(billing_base_url, ?),
                   billing_mode = COALESCE(billing_mode, ?),
                   model = COALESCE(model, ?),
                   api_call_count = ?
                   WHERE id = ?"""

_TOKEN_UPDATE_DELTA_SQL = """UPDATE sessions SET
                   input_tokens = input_tokens + ?,
                   output_tokens = output_tokens + ?,
                   cache_read_tokens = cache_read_tokens + ?,
                   cache_write_tokens = cache_write_tokens + ?,
                   reasoning_tokens = reasoning_tokens + ?,
                   estimated_cost_usd = COALESCE(estimated_cost_usd, 0) + COALESCE(?, 0),
                   actual_cost_usd = CASE
                       WHEN ? IS NULL THEN actual_cost_usd
                       ELSE COALESCE(actual_cost_usd, 0) + ?
                   END,
                   cost_status = COALESCE(?, cost_status),
                   cost_source = COALESCE(?, cost_source),
                   pricing_version = COALESCE(?, pricing_version),
                   billing_provider = COALESCE(billing_provider, ?),
                   billing_base_url = COALESCE(billing_base_url, ?),
                   billing_mode = COALESCE(billing_mode, ?),
                   model = COALESCE(model, ?),
                   api_call_count = COALESCE(api_call_count, 0) + ?
                   WHERE id = ?"""

_MODEL_USAGE_UPSERT_SQL = """INSERT INTO session_model_usage (
                   session_id, model, billing_provider, billing_base_url, billing_mode,
                   task, api_call_count, input_tokens, output_tokens,
                   cache_read_tokens, cache_write_tokens, reasoning_tokens,
                   estimated_cost_usd, actual_cost_usd, cost_status, cost_source,
                   first_seen, last_seen
               ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(session_id, model, billing_provider, billing_base_url, billing_mode, task)
               DO UPDATE SET
                   api_call_count = api_call_count + excluded.api_call_count,
                   input_tokens = input_tokens + excluded.input_tokens,
                   output_tokens = output_tokens + excluded.output_tokens,
                   cache_read_tokens = cache_read_tokens + excluded.cache_read_tokens,
                   cache_write_tokens = cache_write_tokens + excluded.cache_write_tokens,
                   reasoning_tokens = reasoning_tokens + excluded.reasoning_tokens,
                   estimated_cost_usd = estimated_cost_usd + excluded.estimated_cost_usd,
                   actual_cost_usd = actual_cost_usd + excluded.actual_cost_usd,
                   cost_status = COALESCE(excluded.cost_status, cost_status),
                   cost_source = COALESCE(excluded.cost_source, cost_source),
                   last_seen = excluded.last_seen"""


class SessionUsageMixin:
    """Coalesced token writer, per-model usage rows, billing route."""

    def update_session_billing_route(
        self, session_id: str, *, provider: str, base_url: str, billing_mode: Optional[str] = None,
    ) -> None:
        """Unconditionally set the billing route (``update_token_counts`` only
        COALESCE-fills NULLs) so the dashboard reflects the latest /model switch. Also
        nulls ``system_prompt`` so the cached snapshot header is rebuilt."""
        # Barrier against queued token deltas — see update_session_model.
        self.flush_token_counts()

        def _do(conn):
            conn.execute(
                """UPDATE sessions SET
                   billing_provider = ?,
                   billing_base_url = ?,
                   billing_mode = COALESCE(?, billing_mode),
                   system_prompt = NULL,
                   system_prompt_hash = NULL
                   WHERE id = ?""",
                (provider, base_url, billing_mode, session_id),
            )
            self._delete_unreferenced_system_prompts(conn)
        self._execute_write(_do)

    def queue_token_counts(self, session_id: str, **kwargs) -> None:
        """Enqueue a token/cost delta for the background writer (same kwargs as
        :meth:`update_token_counts`). After close() has stopped the writer, falls back
        to the synchronous path and may raise."""
        with self._token_queue_cond:
            thread = self._token_writer_thread
            writer_stopped = self._token_writer_stop and (thread is None or not thread.is_alive())
            if not writer_stopped:
                self._token_queue.append((session_id, kwargs))
                if thread is None or not thread.is_alive():
                    # Daemon so exit never hangs on accounting; the atexit hook drains
                    # leftovers. ``not is_alive()`` (not ``is None``) respawns a writer
                    # that died from an unexpected escape.
                    thread = threading.Thread(
                        target=self._token_writer_loop, name="session-db-token-writer", daemon=True,
                    )
                    self._token_writer_thread = thread
                    thread.start()
                    if self._token_atexit_hook is None:
                        self_ref = weakref.ref(self)

                        def _drain_at_exit() -> None:
                            db = self_ref()
                            if db is not None:
                                db._drain_token_queue_at_exit()

                        self._token_atexit_hook = _drain_at_exit
                        atexit.register(_drain_at_exit)
                self._token_queue_cond.notify_all()
        if writer_stopped:
            # close() ran: enqueueing would drop the delta silently, so apply inline.
            self.update_token_counts(session_id, **kwargs)

    def _apply_claimed_batch(self, batch) -> None:
        """Apply a batch whose ``busy`` flag the caller already claimed, then release."""
        try:
            self._apply_token_batch(batch)
        finally:
            with self._token_queue_cond:
                self._token_writer_busy = False
                self._token_queue_cond.notify_all()

    def flush_token_counts(self, timeout: float = 5.0) -> bool:
        """Block until every queued token delta has been applied. False on timeout
        (callers then read totals stale by the queued deltas). Never raises."""
        # Lock-free fast path: reads queue-then-busy (see ordering notes below).
        if not self._token_queue and not self._token_writer_busy:
            return True
        batch = None
        with self._token_queue_cond:
            deadline = time.monotonic() + timeout
            while self._token_queue or self._token_writer_busy:
                # A live writer is authoritative even when stop-flagged: draining here
                # would race its in-flight batch and reorder deltas (breaking last-non-
                # None-wins / first-accounted-route / COALESCE-backfill fields). Only a
                # dead writer lets the caller take leftovers; a claimed busy means
                # "wait", never "drain alongside".
                thread = self._token_writer_thread
                if (thread is None or not thread.is_alive()) and not self._token_writer_busy:
                    self._token_writer_busy = True
                    batch = list(self._token_queue)
                    self._token_queue.clear()
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                self._token_queue_cond.wait(remaining)
        if batch:
            self._apply_claimed_batch(batch)
        return True

    def _token_writer_loop(self) -> None:
        while True:
            with self._token_queue_cond:
                idle_deadline = time.monotonic() + self._TOKEN_WRITER_IDLE_SECONDS
                while not self._token_queue and not self._token_writer_stop:
                    remaining = idle_deadline - time.monotonic()
                    if remaining <= 0:
                        # Retire under the same lock queue_token_counts() uses to decide
                        # to spawn, so no delta strands behind an exiting worker.
                        self._token_writer_thread = None
                        return
                    self._token_queue_cond.wait(remaining)
                if not self._token_queue:
                    self._token_writer_thread = None
                    return  # stop requested and fully drained
                # busy BEFORE clearing the queue: flush's lock-free fast path must never
                # see "empty and idle" while a popped batch is unapplied.
                self._token_writer_busy = True
                batch = list(self._token_queue)
                self._token_queue.clear()
            self._apply_claimed_batch(batch)

    def _apply_token_batch(self, batch: List[Tuple[str, Dict[str, Any]]]) -> None:
        """Apply queued deltas in order, coalescing where safe. Never raises."""
        try:
            coalesced = self._coalesce_token_deltas(batch)
        except Exception as exc:
            # Coalescing must never kill the writer; the merge is only an optimization.
            logger.warning("async token accounting: coalesce failed, applying raw batch: %s", exc)
            coalesced = batch
        for session_id, kwargs in coalesced:
            try:
                self.update_token_counts(session_id, **kwargs)
            except Exception as exc:
                # Accounting loss is logged, never raised into a turn.
                logger.warning("async token accounting: apply failed (session=%s): %s", session_id, exc)

    def _coalesce_token_deltas(self, batch: List[Tuple[str, Dict[str, Any]]]) -> List[Tuple[str, Dict[str, Any]]]:
        """Merge adjacent incremental deltas with an identical route, so ordering across
        sessions and /model switches is preserved exactly. absolute=True never merges."""
        groups: List[Tuple[Optional[tuple], str, Dict[str, Any]]] = []
        for session_id, kwargs in batch:
            key = None
            if not kwargs.get("absolute"):
                key = (session_id,) + tuple(kwargs.get(f) for f in self._TOKEN_DELTA_ROUTE_FIELDS)
            if groups and key is not None and groups[-1][0] == key:
                merged = groups[-1][2]
                for f in self._TOKEN_DELTA_SUM_FIELDS:
                    merged[f] = merged.get(f, 0) + kwargs.get(f, 0)
                for f in self._TOKEN_DELTA_COST_FIELDS:
                    value = kwargs.get(f)
                    if value is not None:
                        # All-None runs stay None so COALESCE keeps the stored value.
                        merged[f] = (merged.get(f) or 0.0) + value
            else:
                groups.append((key, session_id, dict(kwargs)))
        return [(sid, kw) for _, sid, kw in groups]

    def _stop_token_writer(self, join_timeout: float = 10.0) -> None:
        """Stop the writer thread and drain remaining deltas. Never raises."""
        with self._token_queue_cond:
            self._token_writer_stop = True
            self._token_queue_cond.notify_all()
            thread = self._token_writer_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=join_timeout)
            if thread.is_alive():
                # Writer stuck mid-apply: leave deltas unapplied rather than race it.
                logger.warning(
                    "async token accounting: writer did not stop within %.0fs; "
                    "%d queued delta(s) not persisted",
                    join_timeout, len(self._token_queue),
                )
                return
        # Writer gone: apply leftovers synchronously under the same busy protocol. Wait
        # out a flush caller-drain that already claimed busy — close() nulls the
        # connection right after this returns and must not yank it mid-batch.
        with self._token_queue_cond:
            deadline = time.monotonic() + join_timeout
            while self._token_writer_busy:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    logger.warning(
                        "async token accounting: concurrent drain did not "
                        "finish within %.0fs; %d queued delta(s) not persisted",
                        join_timeout, len(self._token_queue),
                    )
                    return
                self._token_queue_cond.wait(remaining)
            # busy BEFORE clearing the queue (same ordering as the writer loop).
            batch = list(self._token_queue)
            if batch:
                self._token_writer_busy = True
                self._token_queue.clear()
        if batch:
            self._apply_claimed_batch(batch)

    def _drain_token_queue_at_exit(self) -> None:
        try:
            self._stop_token_writer()
        except Exception:
            pass  # never fatal at interpreter shutdown

    def update_token_counts(
        self,
        session_id: str,
        input_tokens: int = 0,
        output_tokens: int = 0,
        model: str = None,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        reasoning_tokens: int = 0,
        estimated_cost_usd: Optional[float] = None,
        actual_cost_usd: Optional[float] = None,
        cost_status: Optional[str] = None,
        cost_source: Optional[str] = None,
        pricing_version: Optional[str] = None,
        billing_provider: Optional[str] = None,
        billing_base_url: Optional[str] = None,
        billing_mode: Optional[str] = None,
        api_call_count: int = 0,
        absolute: bool = False,
    ) -> None:
        """Update token counters and backfill model if unset. *absolute*=False
        increments (per-API-call deltas, CLI path); *absolute*=True sets directly
        (gateway path, where the cached agent holds cumulative totals)."""
        # Ensure the row exists: under concurrent load create_session() may have failed
        # on locking, and the UPDATE would silently affect 0 rows.
        self._insert_session_row(session_id, "unknown", model=model)
        sql = _TOKEN_UPDATE_ABSOLUTE_SQL if absolute else _TOKEN_UPDATE_DELTA_SQL
        has_usage = bool(
            input_tokens or output_tokens or cache_read_tokens
            or cache_write_tokens or reasoning_tokens or api_call_count or estimated_cost_usd
        )
        has_accounted_usage = bool(has_usage or actual_cost_usd)
        params = (
            input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens,
            estimated_cost_usd, actual_cost_usd, actual_cost_usd, cost_status, cost_source, pricing_version,
            billing_provider if has_accounted_usage else None,
            billing_base_url if has_accounted_usage else None,
            billing_mode if has_accounted_usage else None,
            model if has_accounted_usage else None,
            api_call_count,
            session_id,
        )
        # Per-model attribution: the sessions row keeps one (model, provider) pair, so a
        # mid-session /model switch would attribute every token to the initial model.
        # Only the incremental path records here — absolute cumulative updates cannot be
        # split back into routes; Insights reconciles the residual instead.
        record_model_usage = (not absolute) and has_usage

        def _do(conn):
            row = conn.execute(
                "SELECT model, billing_provider, api_call_count FROM sessions WHERE id = ?", (session_id,),
            ).fetchone()
            existing_model = row["model"] if row is not None else None
            existing_provider = row["billing_provider"] if row is not None else None
            existing_api_calls = int((row["api_call_count"] if row is not None else 0) or 0)
            # create_session records the requested route before any API call. If that
            # fails and fallback succeeds, the first accounted usage is the authoritative
            # route; after that keep the row as is (one row cannot represent mixed usage).
            first_accounted_route = (
                existing_api_calls == 0
                and has_accounted_usage
                and bool(model)
                and bool(billing_provider)
                and (existing_model != model or existing_provider != billing_provider)
            )
            if first_accounted_route:
                conn.execute(
                    """UPDATE sessions
                       SET model = ?, billing_provider = ?,
                       billing_base_url = ?, billing_mode = ?
                       WHERE id = ?""",
                    (model, billing_provider, billing_base_url, billing_mode, session_id),
                )
            conn.execute(sql, params)
            if record_model_usage:
                self._record_model_usage(
                    conn, session_id, model=model, billing_provider=billing_provider,
                    billing_base_url=billing_base_url, billing_mode=billing_mode,
                    input_tokens=input_tokens, output_tokens=output_tokens,
                    cache_read_tokens=cache_read_tokens, cache_write_tokens=cache_write_tokens,
                    reasoning_tokens=reasoning_tokens, estimated_cost_usd=estimated_cost_usd,
                    actual_cost_usd=actual_cost_usd, cost_status=cost_status, cost_source=cost_source,
                    api_call_count=api_call_count,
                )
        self._execute_write(_do)

    def _record_model_usage(
        self,
        conn,
        session_id: str,
        *,
        model: Optional[str],
        billing_provider: Optional[str],
        billing_base_url: Optional[str],
        billing_mode: Optional[str],
        input_tokens: int,
        output_tokens: int,
        cache_read_tokens: int,
        cache_write_tokens: int,
        reasoning_tokens: int,
        estimated_cost_usd: Optional[float],
        actual_cost_usd: Optional[float],
        cost_status: Optional[str],
        cost_source: Optional[str],
        api_call_count: int,
        task: str = "",
    ) -> None:
        """Accumulate a per-API-call usage delta into session_model_usage, inside the
        caller's write txn after the ``sessions`` UPDATE. A missing model/provider falls
        back to the session row (same COALESCE behaviour as the summary update) — except
        for aux rows (``task`` set), which must NOT inherit the main-loop route (vision
        on gemini while the main loop runs anthropic): missing info stays 'unknown'/empty.
        """
        row = conn.execute(
            "SELECT model, billing_provider, billing_base_url, billing_mode "
            "FROM sessions WHERE id = ?",
            (session_id,),
        ).fetchone()
        sess = dict(row) if (row is not None and not task) else {}
        eff_model = model or sess.get("model") or "unknown"
        eff_provider = billing_provider or sess.get("billing_provider") or ""
        eff_base_url = billing_base_url or sess.get("billing_base_url") or ""
        eff_billing_mode = billing_mode or sess.get("billing_mode") or ""
        counts = [v or 0 for v in (input_tokens, output_tokens, cache_read_tokens, cache_write_tokens, reasoning_tokens)]
        now = time.time()
        conn.execute(
            _MODEL_USAGE_UPSERT_SQL,
            (
                session_id, eff_model, eff_provider, eff_base_url, eff_billing_mode, task or "",
                api_call_count or 0, *counts,
                float(estimated_cost_usd or 0.0), float(actual_cost_usd or 0.0),
                cost_status, cost_source, now, now,
            ),
        )

    def record_auxiliary_usage(
        self,
        session_id: str,
        task: str,
        *,
        model: Optional[str] = None,
        billing_provider: Optional[str] = None,
        billing_base_url: Optional[str] = None,
        input_tokens: int = 0,
        output_tokens: int = 0,
        cache_read_tokens: int = 0,
        cache_write_tokens: int = 0,
        reasoning_tokens: int = 0,
        estimated_cost_usd: Optional[float] = None,
        api_call_count: int = 1,
    ) -> None:
        """Record an auxiliary LLM call's usage (vision, compression, title generation,
        ...) as a per-(model, provider, task) delta in ``session_model_usage`` WITHOUT
        touching the ``sessions`` summary row (the gateway overwrites those counters with
        absolute main-loop totals). ``api_call_count`` may aggregate N calls. Best-effort:
        callers must never fail an aux call over accounting."""
        if not session_id or not task:
            return
        # FK to sessions.id: same INSERT OR IGNORE guard as update_token_counts.
        self._insert_session_row(session_id, "unknown")

        def _do(conn):
            self._record_model_usage(
                conn, session_id, model=model, billing_provider=billing_provider,
                billing_base_url=billing_base_url, billing_mode=None,
                input_tokens=input_tokens or 0, output_tokens=output_tokens or 0,
                cache_read_tokens=cache_read_tokens or 0, cache_write_tokens=cache_write_tokens or 0,
                reasoning_tokens=reasoning_tokens or 0, estimated_cost_usd=estimated_cost_usd,
                actual_cost_usd=None, cost_status=None, cost_source=None,
                api_call_count=1 if api_call_count is None else int(api_call_count),
                task=task,
            )
        self._execute_write(_do)

    def usage_totals(self, *, min_message_count: int = 1, include_archived: bool = False) -> Dict[str, float]:
        """Tokens and spend across the whole store (one scan), so the sidebar total does
        not shrink with paging. Spend prefers the billed figure over the estimate."""
        where = ["parent_session_id IS NULL", "message_count >= ?"]
        params: List[Any] = [min_message_count]
        if not include_archived:
            where.append("COALESCE(archived, 0) = 0")
        row = self._read_one(
            f"""
            SELECT COALESCE(SUM(COALESCE(input_tokens, 0) + COALESCE(output_tokens, 0)), 0),
                   COALESCE(SUM(COALESCE(actual_cost_usd, estimated_cost_usd, 0)), 0)
              FROM sessions
             WHERE {' AND '.join(where)}
            """,
            params,
        )
        return {"tokens": int(row[0] or 0), "cost_usd": float(row[1] or 0.0)}
