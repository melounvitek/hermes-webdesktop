"""Cron job scheduler: tick() runs due jobs (gateway calls it every 60s from a background thread).
A file lock (~/.hermes/cron/.tick.lock) keeps overlapping processes to one tick at a time.
"""

import asyncio
import atexit
import concurrent.futures
import contextlib
import contextvars
import errno
import json
import logging
import os
import re
import shutil
import signal
import subprocess
import sys
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone

# fcntl is Unix-only; Windows uses msvcrt
try:
    import fcntl
except ImportError:
    fcntl = None
    try:
        import msvcrt
    except ImportError:
        msvcrt = None
from pathlib import Path
from typing import Any, Callable, List, Optional, Protocol

# Must precede repo-level imports: standalone invocations (e.g. module reload after
# `hermes update`) otherwise fail with ModuleNotFoundError for hermes_time et al.
sys.path.insert(0, str(Path(__file__).parent.parent))

from hermes_constants import get_hermes_home
from hermes_cli._subprocess_compat import windows_hide_flags
from hermes_cli.config import (
    _expand_env_vars,
    cron_model_drift_axes,
    cron_model_drift_guard_enabled,
    load_config,
    resolve_cron_model_drift_defaults,
)
from hermes_cli.fallback_config import get_fallback_chain
from hermes_time import now as _hermes_now
from agent.interrupt_compat import request_hard_interrupt
from agent.delegation_context import (
    enter_non_dispatcher_owned_context,
    exit_non_dispatcher_owned_context,
)

logger = logging.getLogger(__name__)


def _close_late_session_db_result(future: "concurrent.futures.Future") -> None:
    """Done-callback: close a SessionDB whose constructor finished after run_job's init timeout
    (worker abandoned via ``shutdown(wait=False)``), else its .db/WAL/SHM handles leak to EMFILE.
    """
    with contextlib.suppress(Exception):
        db = future.result()
        if db is not None:
            from hermes_state import release_or_close
            release_or_close(db)


def _set_cron_session_title(session_db, session_id, base_title):
    """Persist a non-blank, unique title for a finished cron session; returns it (None if unset).

    Runs synchronously in the cron finally block BEFORE end_session()/close() so no write races the
    close. Duplicate (unique-title index ValueError) -> get_next_title_in_lineage(); if unavailable,
    raise rather than end up untitled.
    """
    if not session_db or not session_id:
        return None
    title = (base_title or "").strip()
    if not title:
        return None
    try:
        session_db.set_session_title(session_id, title)
        return title
    except ValueError:
        # Unique-title collision: fall back to the next lineage title (base #2, #3, ...).
        next_title_fn = getattr(session_db, "get_next_title_in_lineage", None)
        if next_title_fn is None:
            raise
        deduped = next_title_fn(title)
        if not deduped or deduped == title:
            raise
        session_db.set_session_title(session_id, deduped)
        return deduped


def _fallback_chain_phrase() -> str:
    """Fallback-chain clause for a provider-failure message: "exhausted" vs "none configured" (most
    installs). Fails open to the ambiguous wording if config can't be read — never crash delivery.
    """
    try:
        cfg = load_config() or {}
        chain = get_fallback_chain(cfg)
    except Exception:
        return "Fallback chain was exhausted or unavailable."
    if chain:
        return "Fallback chain was exhausted or unavailable."
    return (
        "No fallback chain configured — add one with `hermes fallback add`, "
        "or set a cron fleet default via `cron.model` + `cron.model_provider` "
        "in config.yaml."
    )


def _failure_streak_nudge(job: dict) -> str:
    """Return a review nudge when a recurring job keeps failing, else "".

    ``failure_streak`` is persisted by ``cron.jobs.mark_job_run`` (reset on success); the failure
    message is delivered BEFORE mark_job_run records this run, hence stored+1.
    Threshold: ``cron.failure_nudge_threshold`` (default 3, 0 disables).
    """
    schedule_kind = (job.get("schedule") or {}).get("kind")
    if schedule_kind not in {"cron", "interval"}:
        return ""
    try:
        cfg = load_config() or {}
        threshold = int(
            ((cfg.get("cron") or {}) if isinstance(cfg, dict) else {}).get(
                "failure_nudge_threshold", 3
            )
        )
    except Exception:
        threshold = 3
    if threshold <= 0:
        return ""
    streak = int(job.get("failure_streak") or 0) + 1  # +1 = this run
    if streak < threshold:
        return ""
    job_ref = job.get("name") or job.get("id") or "this job"
    return (
        f"\nThis job has failed {streak} runs in a row — worth a review. "
        f"Fix its prompt/config, or pause it with `hermes cron pause {job_ref}` "
        "(resume/remove also available) to stop the noise."
    )


def _detect_gateway_code_skew() -> tuple[str, str] | None:
    """Boot-vs-disk revision skew for THIS process, or None. Test seam over
    ``gateway.code_skew.detect_code_skew``; a broken import must never take delivery down."""
    try:
        from gateway.code_skew import detect_code_skew

        return detect_code_skew()
    except Exception:
        return None


class CronTickYielded(RuntimeError):
    """A stale-code ticker yielded this tick to a fresh gateway.

    Raised by ``tick()`` BEFORE the tick lock is acquired when boot fingerprint ≠ disk, this process
    does NOT own the gateway runtime lock, and a fresh process holds it; the stale process must stay
    out of the dispatch race entirely (lock contention would starve the fresh ticker). Skew ``None``
    (non-git, no fingerprint, probe failure) never yields: fail open. Raised, not returned, so
    provider loops record it via ``record_ticker_error`` and ``hermes cron status`` isn't green.
    """

    def __init__(self, boot_rev: str, disk_rev: str) -> None:
        self.boot_rev = boot_rev
        self.disk_rev = disk_rev
        super().__init__(
            f"Cron tick yielded to a fresh gateway process (stale code: "
            f"booted on {boot_rev}, disk is at {disk_rev})"
        )


# Log the yield at most once per episode (reset when the skew changes) to avoid per-interval spam.
_YIELD_LOG_INTERVAL_SECONDS = 3600.0
_last_yield_log: dict[str, object] = {}


def _should_yield_tick_to_fresh_gateway() -> tuple[str, str] | None:
    """Return ``(boot_rev, disk_rev)`` when this tick must yield to a fresher gateway, else None.

    Yields only when ALL hold: code skew, we don't own the runtime lock, another process holds it.
    Every probe failure returns None — yielding is a certainty claim, never a guess.
    """
    skew = _detect_gateway_code_skew()
    if skew is None:
        return None
    try:
        from gateway import status as _gateway_status
    except Exception:
        return None
    try:
        if _gateway_status.owns_gateway_runtime_lock():
            return None
        if not _gateway_status.is_gateway_runtime_lock_active():
            return None
    except Exception:
        return None
    return skew


def _log_tick_yield_once(reason: str) -> None:
    """Log the yield at error level once per episode (skew signature)."""
    global _last_yield_log
    now = time.monotonic()
    last_reason = _last_yield_log.get("reason")
    last_at = _last_yield_log.get("at", 0.0)
    if last_reason != reason or (now - float(last_at)) >= _YIELD_LOG_INTERVAL_SECONDS:
        logger.error(
            "Cron tick yielded: this process is running stale code (%s) and a "
            "fresher gateway owns the runtime lock — jobs will fire from that "
            "process. Restart this one to reclaim its ticks.",
            reason,
        )
    _last_yield_log = {"reason": reason, "at": now}


def _summarize_cron_failure_for_delivery(job: dict, error: str | None) -> str:
    """Compact one-line failure message for chat delivery (full details stay in cron output)."""
    job_name = job.get("name") or job.get("id") or "cron job"
    text = (error or "unknown error").strip()
    lower = text.lower()

    if "skipped to prevent unintended spend: global inference config drifted" in lower:
        if "finite one-shot job is consumed" in lower:
            remediation = (
                "This finite one-shot is consumed; create a new one-shot job at "
                "a future time with an explicit provider and model."
            )
        else:
            job_id = job.get("id") or "<job_id>"
            remediation = (
                "On the host running Hermes, pin it explicitly: "
                f"`hermes cron edit {job_id} --provider <provider> "
                "--model <model>`."
            )
        return (
            f"⚠️ Cron '{job_name}' skipped before inference to prevent "
            f"unintended spend. {remediation}"
        )

    # no_agent jobs never reach a model, so provider errors are structurally impossible for them.
    # Gate on job MODE before substring matching, or a script's own wording ("timed out", "429")
    # would blame the wrong subsystem; the generic cleaner below reports what actually happened.
    provider_reachable = not job.get("no_agent")

    # Script runner contract ("Script timed out after {n}s: {path}") — also for agent jobs with a
    # context script. Must precede generic timeout matching so it never claims a provider fallback.
    if lower.startswith("script timed out"):
        return (
            f"⚠️ Cron '{job_name}' failed: script timed out. "
            "No model was invoked. Full details saved in cron output."
        )

    # Whole-token 429: substrings in job ids/ports/hashes tripped false rate-limit alerts.
    if provider_reachable and (
        re.search(r"\b429\b", text) or "rate limit" in lower or "usage limit" in lower
    ):
        reason = "rate limit"
        if "weekly usage limit" in lower:
            reason = "weekly usage limit"
        elif "quota" in lower:
            reason = "quota limit"
        return (
            f"⚠️ Cron '{job_name}' failed: provider {reason}. "
            f"{_fallback_chain_phrase()} "
            "Full details saved in cron output."
        )

    # Scheduler inactivity watchdog shape ("idle for {n}s (limit {m}s)"). Must precede the generic
    # provider-timeout branch: the job's own tool going quiet involves no provider/fallback chain.
    if re.search(r"idle for \d+s\s*\(limit \d+s\)", lower):
        return (
            f"⚠️ Cron '{job_name}' failed: the job itself stalled — no tool/API "
            "activity for the configured inactivity window. Not a provider or "
            "fallback-chain issue; check what the job was doing when it went "
            "quiet. Full details saved in cron output."
        )

    if provider_reachable and (
        "readtimeout" in lower or "timed out" in lower or "timeout" in lower
    ):
        return (
            f"⚠️ Cron '{job_name}' failed: provider timeout. "
            f"{_fallback_chain_phrase()} "
            "Full details saved in cron output."
        )

    # Whole-token 401/403 and auth wording so "oauth", "4015" etc. don't trip a false auth message.
    if provider_reachable and (
        re.search(r"authenticat|authoriz", lower) or re.search(r"\b(401|403)\b", text)
    ):
        return (
            f"⚠️ Cron '{job_name}' failed: provider authentication error. "
            "Full details saved in cron output."
        )

    # Strip exception wrappers; bound input first so a multi-KB blob can't slow the regexes.
    cleaned = re.sub(r"^(RuntimeError|Exception|ValueError|HTTPStatusError):\s*", "", text[:2000])
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if len(cleaned) > 180:
        cleaned = cleaned[:177].rstrip() + "..."
    message = f"⚠️ Cron '{job_name}' failed: {cleaned}"

    # Import-class failures in a gateway whose checkout changed underneath it (mixed sys.modules)
    # read like code bugs. When boot SHA ≠ disk HEAD, APPEND cause + fix — never replace the raw
    # error, which carries the failing symbol. Fail-safe: skew is None on non-git/no-fingerprint
    # (message unchanged); no_agent jobs excluded via the same mode gate (a fresh subprocess
    # resolves imports against disk, so its ImportError is the script's own problem).
    if provider_reachable and re.search(
        r"cannot import name|modulenotfounderror|importerror", lower
    ):
        try:
            skew = _detect_gateway_code_skew()
        except Exception:
            skew = None  # delivery must never die on a diagnostics probe
        if skew is not None:
            boot_rev, disk_rev = skew
            message += (
                f" Likely cause: the gateway is running stale code (booted "
                f"on {boot_rev}, disk is at {disk_rev}) — run "
                "`hermes gateway restart` to fix it."
            )

    return message


def _upsert_incident_for_failure(
    job: dict, error: str, *, output_file: Optional[Any] = None
) -> tuple[bool, Optional[str]]:
    """Record a durable failure incident (grouped by job + error signature).

    Returns ``(acked, incident_id)``; acked=True when the signature's incident is already
    ``closed`` -> suppress the per-run ping.
    Best-effort: store errors log at debug and the caller delivers as if no incident existed.
    """
    try:
        from cron.incidents import get_incident, upsert_incident

        incident_id, _is_new = upsert_incident(
            job["id"],
            str(error or ""),
            job_name=job.get("name"),
            output_file=output_file,
        )
        incident = get_incident(incident_id)
        acked = bool(incident and incident.get("state") == "closed")
        return acked, incident_id
    except Exception as exc:
        logger.debug(
            "Incident store unavailable for job %s (delivery unaffected): %s",
            job["id"], exc,
        )
        return False, None


def _mark_incident_alerted(incident_id: Optional[str]) -> None:
    """Best-effort: mark incident ``alerted`` (no-op for closed; never resurrects an acked one)."""
    if not incident_id:
        return
    try:
        from cron.incidents import set_incident_state

        set_incident_state(incident_id, "alerted")
    except Exception as exc:
        logger.debug("Failed marking incident %s alerted: %s", incident_id, exc)


class CronPromptInjectionBlocked(Exception):
    """Raised by _build_job_prompt when the assembled prompt (incl. runtime-loaded skill content,
    unseen by create-time scanning) trips the injection scanner; run_job turns it into a clean
    "job blocked" delivery."""


def _resolve_cron_disabled_toolsets(cfg: dict) -> list[str]:
    """Toolsets a cron-spawned agent must never receive.

    ``messaging``/``clarify`` always (interactive). ``cronjob`` by default (loop prevention, not a
    security boundary); ``cron.allow_agent_scheduling: true`` lifts only that, never the user
    denylist. ``agent.disabled_toolsets`` is layered on top so per-job ``enabled_toolsets`` cannot
    widen past config.yaml's denylist.
    """
    cron_cfg = (cfg or {}).get("cron") or {}
    if cron_cfg.get("allow_agent_scheduling"):
        disabled = ["messaging", "clarify"]
    else:
        disabled = ["cronjob", "messaging", "clarify"]
    agent_cfg = (cfg or {}).get("agent") or {}
    from agent.skill_utils import parse_config_string_list

    user_disabled = parse_config_string_list(agent_cfg.get("disabled_toolsets"))
    for name in user_disabled:
        name = str(name).strip()
        if name and name not in disabled:
            disabled.append(name)
    return disabled


def _merge_mcp_into_per_job_toolsets(per_job: list[str], cfg: dict) -> list[str]:
    """Layer enabled MCP servers onto a per-job ``enabled_toolsets`` allowlist.

    Without this a per-job list silently drops every MCP server ("Unknown tool" on mcp_* calls).
    Mirrors ``_get_platform_tools``: ``no_mcp`` sentinel -> none (sentinel stripped); any MCP server
    already listed -> treat as allowlist, add nothing; otherwise union in all globally-enabled.
    """
    result = [t for t in per_job if t != "no_mcp"]
    if "no_mcp" in per_job:
        return result
    # lazy: avoid heavy hermes_cli import at module load; shares MCP-membership with gateway/CLI
    from hermes_cli.tools_config import enabled_mcp_server_names
    enabled_mcp = enabled_mcp_server_names(cfg)
    if set(result) & enabled_mcp:
        return result
    for name in sorted(enabled_mcp):
        if name not in result:
            result.append(name)
    return result


def _resolve_cron_enabled_toolsets(job: dict, cfg: dict) -> list[str] | None:
    """Resolve the toolset list for a cron job.

    Precedence: per-job ``enabled_toolsets`` (+ ``_merge_mcp_into_per_job_toolsets``) > ``cron``
    platform config (``_get_platform_tools``) > ``None`` on any failure (full default set).
    ``_get_platform_tools`` strips _DEFAULT_OFF_TOOLSETS ({moa, homeassistant, rl}) for unconfigured
    platforms, so fresh installs run cron without ``moa``.
    """
    per_job = job.get("enabled_toolsets")
    if per_job:
        return _merge_mcp_into_per_job_toolsets(list(per_job), cfg or {})
    try:
        from hermes_cli.tools_config import _get_platform_tools  # lazy: avoid heavy import at cron module load
        return sorted(_get_platform_tools(cfg or {}, "cron"))
    except Exception as exc:
        logger.warning(
            "Cron toolset resolution failed, falling back to full default toolset: %s",
            exc,
        )
        return None


def _resolve_job_reasoning_config(job: dict, cfg: dict, model: str) -> dict | None:
    """Resolve the effective reasoning config for a cron run.

    Per-job ``reasoning_effort`` pin beats global and per-model config; it is model-independent by
    design (also governs an auth-fallback swap) — clamping stays with provider transports at send
    time. An unparseable pin warns and falls back, never kills the tick. No pin -> config.
    """
    from hermes_constants import parse_reasoning_effort, resolve_reasoning_config

    pinned = job.get("reasoning_effort")
    if pinned is not None:
        parsed = parse_reasoning_effort(pinned)
        if parsed is not None:
            logger.info("Job '%s': using per-job reasoning_effort '%s'", job.get("id", "?"), pinned)
            return parsed
        logger.warning(
            "Job '%s': invalid stored reasoning_effort %r — ignoring the pin "
            "and falling back to config resolution. Fix with `cronjob "
            "action=update job_id=%s reasoning_effort=<level>` (valid: none, "
            "minimal, low, medium, high, xhigh, max, ultra).",
            job.get("id", "?"),
            pinned,
            job.get("id", "?"),
        )
    return resolve_reasoning_config(cfg if isinstance(cfg, dict) else {}, str(model))


# Validates user-supplied delivery platform names, preventing env-var enumeration via crafted names.
_KNOWN_DELIVERY_PLATFORMS = frozenset({
    "telegram", "discord", "slack", "whatsapp", "signal",
    "matrix", "mattermost", "homeassistant", "dingtalk", "feishu",
    "wecom", "wecom_callback", "weixin", "sms", "email", "webhook", "bluebubbles",
    "qqbot", "yuanbao",
})

# Platforms supporting a cron/notification home target -> env var used by gateway config.
_HOME_TARGET_ENV_VARS = {
    "matrix": "MATRIX_HOME_ROOM",
    "telegram": "TELEGRAM_HOME_CHANNEL",
    "discord": "DISCORD_HOME_CHANNEL",
    "slack": "SLACK_HOME_CHANNEL",
    "signal": "SIGNAL_HOME_CHANNEL",
    "mattermost": "MATTERMOST_HOME_CHANNEL",
    "sms": "SMS_HOME_CHANNEL",
    "email": "EMAIL_HOME_ADDRESS",
    "dingtalk": "DINGTALK_HOME_CHANNEL",
    "feishu": "FEISHU_HOME_CHANNEL",
    "wecom": "WECOM_HOME_CHANNEL",
    "weixin": "WEIXIN_HOME_CHANNEL",
    "bluebubbles": "BLUEBUBBLES_HOME_CHANNEL",
    "qqbot": "QQBOT_HOME_CHANNEL",
    "whatsapp": "WHATSAPP_HOME_CHANNEL",
    "whatsapp_cloud": "WHATSAPP_CLOUD_HOME_CHANNEL",
}

# Back-compat: primary env var -> previous name; _get_home_target_chat_id falls back to the legacy
# name when the primary is unset.
_LEGACY_HOME_TARGET_ENV_VARS = {"QQBOT_HOME_CHANNEL": "QQ_HOME_CHANNEL"}

from cron.jobs import (
    _ensure_cron_dir,
    advance_next_runs,
    claim_dispatch,
    claim_job_for_fire,
    fire_claim_fence,
    clear_run_claim,
    get_due_jobs,
    heartbeat_fire_claim,
    heartbeat_run_claim,
    mark_job_run,
    save_job_output,
    use_cron_store,
)
from cron.executions import create_execution, finish_execution, mark_execution_running

# Response marker that suppresses delivery (output is still saved locally for audit).
SILENT_MARKER = "[SILENT]"


def _is_cron_silence_response(text: str) -> bool:
    """Return True when a cron final response should suppress delivery.

    Looser than the gateway's exact-whole-response rule: ``[SILENT]`` (or SILENT / NO_REPLY /
    NO REPLY) counts as the whole response OR its own first/last line — NOT mid-sentence. Shares the
    webhook-lane matcher in :mod:`gateway.response_filters` so the two cannot drift.
    """
    from gateway.response_filters import is_autonomous_silence_response

    return is_autonomous_silence_response(text)

# Persistent pool for parallel cron jobs: tick() submits and returns; long jobs never block it.
_parallel_pool: Optional[concurrent.futures.ThreadPoolExecutor] = None
_parallel_pool_max_workers: Optional[int] = None
_running_job_ids: set = set()
_running_fire_owners: dict[str, dict[object, tuple[Optional[str], Path]]] = {}
_running_lock = threading.Lock()

# Per in-flight id: time.time() claim instant + the future owning its release (``_FUTURE_PENDING``
# until pool.submit returns). Past-allowance with no live future = leak; the sweep force-releases.
_running_since: dict = {}
_running_futures: dict = {}

# Installed in ``_running_futures`` at claim time so a sweep landing before ``pool.submit`` returns
# never sees ``missing`` and releases a claim about to get its future.
_FUTURE_PENDING = object()

# Forced-release count/history for ``get_inflight_guard_stats()``; mirrored to JSONL for probes.
_forced_release_count: int = 0
_forced_releases: list = []
_FORCED_RELEASE_HISTORY = 20

# Stale-allowance floor (minutes); per-job allowance is max(2 * interval, this).
_INFLIGHT_MIN_ALLOWANCE_MINUTES = 30.0


# Execution tokens (``_running_fire_owners`` identity keys) force-interrupted at shutdown; see
# ``mark_running_jobs_interrupted``. ``run_one_job`` checks its OWN token before writing
# ``last_status`` so a still-running agent thread can't overwrite "interrupted" with a false "ok".
# Token keying scopes the flag to one execution (recurring jobs reuse IDs); legacy paths without a
# fire owner fall back to the bare job ID.
_interrupted_job_ids: set = set()


class _CancelEventLike(Protocol):
    """Structural type for cancellation sources (``threading.Event``, ``_CombinedCancelEvent``)."""

    def is_set(self) -> bool: ...
    def set(self) -> None: ...


class _CombinedCancelEvent:
    """Duck-typed ``threading.Event`` ORing several cancellation sources (fire-claim heartbeat
    ``lost_ownership`` + per-transport events). Workers only call is_set()/set(), so no pump thread.
    """

    def __init__(self, *events: Optional["_CancelEventLike"]) -> None:
        self._events = [event for event in events if event is not None]

    def is_set(self) -> bool:
        return any(event.is_set() for event in self._events)

    def set(self) -> None:
        for event in self._events:
            event.set()


def get_running_job_ids() -> "frozenset[str]":
    """Thread-safe snapshot of executing job IDs (dispatch until ``_process_job`` returns). Read by
    the gateway shutdown drain, otherwise blind to cron work (runs outside ``_running_agents``)."""
    with _running_lock:
        return frozenset(_running_job_ids | _running_fire_owners.keys())


def try_register_running_job(job_id: str) -> bool:
    """Atomically add ``job_id`` to the in-flight set; False (caller must skip) if already mid-run.

    Single dedupe owner for ticker + manual runs (the fire claim's 300s TTL is outlived by real
    jobs). Callers MUST pair success with ``release_running_job`` in a ``finally``.
    """
    with _running_lock:
        if job_id in _running_job_ids:
            return False
        _running_job_ids.add(job_id)
        # Same critical section as the add: no window where an in-flight id lacks an age the sweep
        # can bound. Sentinel is replaced by the real future once ``pool.submit`` returns.
        _running_since[job_id] = time.time()
        _running_futures[job_id] = _FUTURE_PENDING
        return True


def release_running_job(job_id: str) -> None:
    """Remove ``job_id`` from the in-flight running set (idempotent)."""
    with _running_lock:
        _running_job_ids.discard(job_id)
        _running_since.pop(job_id, None)
        _running_futures.pop(job_id, None)


def _inflight_min_allowance_minutes() -> float:
    """Stale allowance floor (min): ``cron.inflight_max_minutes``, else env escape hatch/default."""
    with contextlib.suppress(Exception):
        _ucfg = load_config() or {}
        _cfg_val = (
            _ucfg.get("cron", {}) if isinstance(_ucfg, dict) else {}
        ).get("inflight_max_minutes")
        if _cfg_val is not None:
            val = float(_cfg_val)
            if val > 0:
                return val
    raw = os.getenv("HERMES_CRON_INFLIGHT_MAX_MINUTES", "").strip()
    if raw:
        try:
            val = float(raw)
            if val > 0:
                return val
        except (ValueError, TypeError):
            logger.warning(
                "Invalid HERMES_CRON_INFLIGHT_MAX_MINUTES=%r; using default %s",
                raw,
                _INFLIGHT_MIN_ALLOWANCE_MINUTES,
            )
    return _INFLIGHT_MIN_ALLOWANCE_MINUTES


# expr -> minutes; cadence never changes, so avoid re-evaluating croniter every tick.
_cron_interval_cache: dict = {}


def _cron_interval_minutes(expr: str) -> Optional[float]:
    """Cron expression cadence (gap between next two fires) in minutes; None -> floor allowance."""
    if expr in _cron_interval_cache:
        return _cron_interval_cache[expr]
    result = None
    with contextlib.suppress(Exception):
        from cron.jobs import _ensure_croniter

        if _ensure_croniter():
            from cron.jobs import croniter as _croniter
            from datetime import datetime

            base = datetime.now()
            it = _croniter(expr, base)
            first = it.get_next(datetime)
            second = it.get_next(datetime)
            gap = (second - first).total_seconds() / 60.0
            result = gap if gap > 0 else None
    _cron_interval_cache[expr] = result
    return result


def _job_interval_minutes(job: dict) -> Optional[float]:
    """Best-effort job interval in minutes (None if unknown / one-shot -> floor). ``schedule`` is
    persisted as a parsed dict; the string path is only a fallback for programmatic callers."""
    with contextlib.suppress(Exception):
        schedule = job.get("schedule")
        if isinstance(schedule, str) and schedule.strip():
            from cron.jobs import parse_schedule

            schedule = parse_schedule(schedule) or {}
        if isinstance(schedule, dict):
            kind = schedule.get("kind")
            if kind == "interval":
                minutes = schedule.get("minutes")
                return float(minutes) if minutes else None
            if kind == "cron":
                return _cron_interval_minutes(str(schedule.get("expr") or ""))
    return None


def get_inflight_guard_stats() -> dict:
    """Probe-visible snapshot; non-zero ``forced_releases`` means a job wedged and was recovered."""
    now = time.time()
    with _running_lock:
        return {
            "running": sorted(_running_job_ids),
            "running_ages_seconds": {
                jid: round(now - started, 1)
                for jid, started in _running_since.items()
            },
            "forced_releases": _forced_release_count,
            "recent_forced_releases": list(_forced_releases),
        }


def _record_forced_release(job_id: str, name: str, age_seconds: float, allowance_seconds: float) -> None:
    """Persist a countable signal for one forced release (best-effort)."""
    entry = {
        "job_id": job_id,
        "name": name,
        "age_seconds": round(age_seconds, 1),
        "allowance_seconds": round(allowance_seconds, 1),
        "at": _hermes_now().isoformat(),
    }
    with _running_lock:
        _forced_releases.append(entry)
        del _forced_releases[:-_FORCED_RELEASE_HISTORY]
    try:
        path = _get_hermes_home() / "cron" / "inflight_forced_releases.jsonl"
        _ensure_cron_dir(path.parent)
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(entry) + "\n")
    except Exception as e:  # never let telemetry break a tick
        logger.debug("Could not append forced-release record: %s", e)


def sweep_stale_inflight(due_jobs: Optional[list] = None) -> list:
    """Force-release in-flight claims that can no longer be making progress; returns released ids.

    Stale = older than ``max(2 * interval, floor)`` AND (no live future — submit path hung before
    ``pool.submit`` returned — or finished without discarding the id). Each release logs WARNING
    ``event=forced_release``, bumps the probe counter, mirrors JSONL, and writes ``last_error``.
    """
    global _forced_release_count

    by_id = {j.get("id"): j for j in (due_jobs or []) if isinstance(j, dict)}
    floor_seconds = _inflight_min_allowance_minutes() * 60.0
    now = time.time()
    stale: list = []

    # Latest durable execution per releasable-looking claim, one indexed query. A claim whose OWN
    # run's row is terminal is stale regardless of age. Two-phase so the healthy path pays no DB
    # work: only claims with a missing/pending/done future are queried. Snapshot under
    # _running_lock — iterating the set while try_register/release mutate it raises RuntimeError.
    from cron.executions import _TERMINAL_STATES as _terminal_states

    with _running_lock:
        _claim_futures = {job_id: _running_futures.get(job_id) for job_id in _running_job_ids}
    _ledger_candidates = [
        job_id
        for job_id, fut in _claim_futures.items()
        if fut is None or fut is _FUTURE_PENDING or fut.done()
    ]
    _latest: dict = {}
    if _ledger_candidates:
        try:
            from cron.executions import latest_executions as _latest_execs
            _latest = _latest_execs(_ledger_candidates)
        except Exception:
            _latest = {}

    def _row_belongs_to_claim(row: dict, claim_started: float) -> bool:
        """True when the ledger row was claimed at/after this in-memory claim.

        A terminal row older than the claim is the PREVIOUS run's (common for recurring jobs in the
        try_register->create_execution window); releasing on it would double-dispatch. Unparseable
        timestamps fail closed (treated as previous-run; the age path still bounds the claim).
        """
        claimed_at = row.get("claimed_at")
        if not claimed_at:
            return False
        try:
            from cron.jobs import _ensure_aware as _ensure_aware_ts
            row_ts = _ensure_aware_ts(datetime.fromisoformat(claimed_at))
            return row_ts.timestamp() >= claim_started
        except (ValueError, TypeError, OSError):
            return False

    # Compute intervals OUTSIDE _running_lock so croniter doesn't block try_register/release.
    _intervals = {jid: _job_interval_minutes(j) for jid, j in by_id.items()}

    with _running_lock:
        for job_id in list(_running_job_ids):
            started = _running_since.get(job_id)
            if started is None:
                # Claim predates this guard — adopt it; sweepable one allowance from now.
                _running_since[job_id] = now
                continue
            age = now - started
            interval_minutes = _intervals.get(job_id)
            allowance = floor_seconds
            if interval_minutes:
                allowance = max(allowance, 2.0 * interval_minutes * 60.0)
            fut = _running_futures.get(job_id)
            if fut is _FUTURE_PENDING:
                # Submit path hung before ``pool.submit`` returned — the wedge class; release it.
                pass
            elif fut is not None and not fut.done():
                continue  # genuinely still executing
            # Ledger reconciliation: a terminal row belonging to THIS claim proves it stale even
            # inside its age allowance. Row must be this claim's, else a recurring job's previous
            # run would double-dispatch a fresh claim.
            latest = _latest.get(job_id)
            if (
                latest is not None
                and latest.get("status") in _terminal_states
                and _row_belongs_to_claim(latest, started)
            ):
                reason = "ledger-terminal"
            elif age >= allowance:
                reason = "age"
            else:
                continue
            _running_job_ids.discard(job_id)
            _running_since.pop(job_id, None)
            _running_futures.pop(job_id, None)
            _forced_release_count += 1
            stale.append((job_id, age, allowance, fut, reason))

    for job_id, age, allowance, fut, _reason in stale:
        job = by_id.get(job_id) or {}
        name = job.get("name") or job_id
        if fut is _FUTURE_PENDING:
            future_state = "pending"
        elif fut is None:
            future_state = "missing"
        else:
            future_state = "finished"
        logger.warning(
            "cron.inflight.forced_release event=forced_release reason=%s job='%s' "
            "id=%s age=%.0fs allowance=%.0fs future=%s — stale in-flight claim "
            "released; the job was skipping every fire with 'already running'",
            _reason,
            name,
            job_id,
            age,
            allowance,
            future_state,
        )
        _record_forced_release(job_id, name, age, allowance)
        # Ledger already records how the run ended: mark_job_run here would clobber an honest
        # ok status with a synthetic failure or double-write a failure.
        if _reason == "ledger-terminal":
            continue
        # Age release may lack a ledger row, so last_error is how it surfaces. But a forced release
        # is NOT a real run: never consume a finite repeat budget or let mark_job_run auto-delete.
        repeat = job.get("repeat") or {}
        if isinstance(repeat, dict) and repeat.get("times") is not None:
            logger.warning(
                "cron.inflight.forced_release.job_untouched job='%s' id=%s — "
                "finite-repeat job released without mark_job_run (repeat budget "
                "preserved); row left in place so it re-fires normally",
                name,
                job_id,
            )
            continue
        try:
            mark_job_run(
                job_id,
                False,
                f"Stale in-flight claim force-released after {age / 60:.1f}m "
                f"(allowance {allowance / 60:.1f}m); previous run never released "
                f"the scheduler in-flight guard",
            )
        except Exception as e:
            logger.warning("Could not record forced release for job %s: %s", job_id, e)

    return [s[0] for s in stale]


def mark_running_jobs_interrupted(
    reason: str,
    *,
    only_owners: Optional[set] = None,
) -> list:
    """Best-effort: mark every in-flight cron job interrupted; returns the job IDs marked.

    Called by gateway shutdown right after ``process_registry.kill_all()``: a job whose tool was
    killed must never report success even if its agent thread produces a plausible response.
    ``only_owners`` (``(job_id, fire_owner)`` pairs) restricts marking to those executions. Tokens
    go into ``_interrupted_job_ids`` BEFORE ``last_status`` is written so ``run_one_job`` sees them.
    """
    with _running_lock:
        active_fires = [
            (token, job_id, owner, profile_home)
            for job_id, executions in _running_fire_owners.items()
            for token, (owner, profile_home) in executions.items()
        ]
        if only_owners is not None:
            active_fires = [fire for fire in active_fires if (fire[1], fire[2]) in only_owners]
        registered_ids = {job_id for _t, job_id, _o, _p in active_fires}
        if only_owners is None:
            active_fires.extend(
                (None, job_id, None, _get_hermes_home())
                for job_id in _running_job_ids - registered_ids
            )
        _interrupted_job_ids.update(
            token if token is not None else job_id
            for token, job_id, _owner, _profile_home in active_fires
        )
    marked = []
    for _token, job_id, fire_owner, profile_home in active_fires:
        if not fire_owner:
            logger.warning(
                "Job '%s' interrupted before its durable fire owner was registered; "
                "leaving persisted state untouched",
                job_id,
            )
            # Still report it: shutdown uses the returned IDs for the interrupted-cron notice. The
            # in-memory flag WAS recorded above; only the persisted last_status write is skipped.
            marked.append(job_id)
            continue
        try:
            with use_cron_store(profile_home):
                if mark_job_run(
                    job_id,
                    False,
                    reason,
                    expected_fire_owner=fire_owner,
                ):
                    marked.append(job_id)
        except Exception as e:
            logger.warning("Failed to mark job %s interrupted: %s", job_id, e)
    return marked


def _is_interrupted(job_id: str, token: Optional[object] = None) -> bool:
    """Non-destructive peek: has shutdown marked THIS execution interrupted? Used before deciding
    what to deliver; does not clear the flag (the authoritative pre-``last_status`` check needs it).
    ``token`` scopes to one execution so a fresh run reusing the job ID isn't poisoned."""
    with _running_lock:
        if token is not None and token in _interrupted_job_ids:
            return True
        return job_id in _interrupted_job_ids


def _consume_interrupted_flag(job_id: str, token: Optional[object] = None) -> bool:
    """Return True and clear the flag if shutdown marked THIS execution interrupted. Called right
    before ``last_status`` is written; consuming stops the flag leaking into a later run."""
    with _running_lock:
        hit = False
        if token is not None and token in _interrupted_job_ids:
            _interrupted_job_ids.discard(token)
            hit = True
        if job_id in _interrupted_job_ids:
            _interrupted_job_ids.discard(job_id)
            hit = True
        return hit


def _inactivity_watchdog_loop(
    *,
    get_idle_seconds: Callable[[], float],
    limit_s: float,
    poll_s: float,
    stop: threading.Event,
    future_done: Callable[[], bool],
) -> bool:
    """Poll idle time until limit (-> True), stop, or the future completes (-> False). Uses
    ``threading.Event.wait``, not asyncio, so a blocked event loop cannot disable the watchdog."""
    while not stop.wait(poll_s):
        if future_done():
            return False
        try:
            idle = float(get_idle_seconds() or 0.0)
        except Exception:
            idle = 0.0
        if idle >= limit_s:
            return True
    return False


def _cron_inactivity_seconds() -> float:
    """Parse HERMES_CRON_TIMEOUT (seconds). 0 = unlimited; bad input = 600.

    Shared by run_job's inactivity monitor and the cwd-lock bound so they can't drift: the lock
    bound must stay >= the inactivity limit or waiters fail while a healthy holder runs.
    """
    raw = os.getenv("HERMES_CRON_TIMEOUT", "").strip()
    if not raw:
        return 600.0
    try:
        return float(raw)
    except (ValueError, TypeError):
        logger.warning("Invalid HERMES_CRON_TIMEOUT=%r; using default 600s", raw)
        return 600.0


def _get_parallel_pool(max_workers: Optional[int]) -> concurrent.futures.ThreadPoolExecutor:
    """Return (or create) the persistent parallel pool."""
    global _parallel_pool, _parallel_pool_max_workers
    if _parallel_pool is None or _parallel_pool_max_workers != max_workers:
        if _parallel_pool is not None:
            _parallel_pool.shutdown(wait=False, cancel_futures=False)
        _parallel_pool = concurrent.futures.ThreadPoolExecutor(
            max_workers=max_workers,
            thread_name_prefix="cron-parallel",
        )
        _parallel_pool_max_workers = max_workers
    return _parallel_pool


def _shutdown_parallel_pool() -> None:
    """Shut down the persistent pool on process exit."""
    global _parallel_pool, _parallel_pool_max_workers
    if _parallel_pool is not None:
        _parallel_pool.shutdown(wait=True, cancel_futures=False)
        _parallel_pool = None
        _parallel_pool_max_workers = None


atexit.register(_shutdown_parallel_pool)
# Per-fire usage audit log; resolves via _get_hermes_home() so profile-scoped paths work.
def _usage_audit_path() -> Path:
    return _get_hermes_home() / "cron" / "usage_audit.jsonl"


def _utcnow_iso_ms() -> str:
    """RFC3339 UTC timestamp with millisecond precision and 'Z' suffix."""
    now = datetime.now(timezone.utc)
    return now.strftime("%Y-%m-%dT%H:%M:%S.") + f"{now.microsecond // 1000:03d}Z"


def _write_usage_audit(record: dict) -> None:
    """Append one JSONL line to cron/usage_audit.jsonl. NEVER raises — a logger bug must not
    break cron jobs (the whole write is inside one try)."""
    try:
        path = _usage_audit_path()
        _ensure_cron_dir(path.parent)
        line = json.dumps(record, ensure_ascii=False)
        with open(path, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception as e:
        logger.warning("usage_audit write failed: %s", e)


def _interpreter_shutting_down(exc: Optional[BaseException] = None) -> bool:
    """True when the interpreter is finalizing (tick fired during gateway teardown).

    Once finalization starts, concurrent.futures/asyncio refuse new work, so any delivery attempt
    (live adapter, asyncio.run, fresh pool) only pollutes errors.log — callers skip with a warning.
    ``exc`` lets an already-raised scheduling error count as a shutdown signal. Thin wrapper over
    ``tools.interpreter_shutdown`` (shared with the gateway).
    """
    from tools.interpreter_shutdown import interpreter_shutting_down

    return interpreter_shutting_down(exc)


# Module override hook for tests / emergency monkeypatches.
_hermes_home: Path | None = None


def _get_hermes_home() -> Path:
    """Resolve Hermes home at call time (honouring the test override).

    Cron is per-profile: jobs must be stored AND executed under the active profile's home. Do not
    freeze this at import or anchor it at the shared default root — either breaks profile isolation.
    """
    return _hermes_home or get_hermes_home()


def _get_lock_paths() -> tuple[Path, Path]:
    """Resolve cron lock paths at call time so profile/env changes are honored."""
    hermes_home = _get_hermes_home()
    lock_dir = hermes_home / "cron"
    return lock_dir, lock_dir / ".tick.lock"


def _is_lock_contention_errno(err: OSError) -> bool:
    """True when *err* from the lock syscall means another ticker holds the lock.

    POSIX flock: EWOULDBLOCK/EAGAIN (EACCES on some NFS); Windows msvcrt.locking: EACCES/EDEADLK.
    Everything else — notably EMFILE/ENFILE (fd exhaustion) and EACCES on open() — must be
    surfaced, never swallowed as contention.
    """
    if err.errno is None:
        return False
    if fcntl is not None:
        return err.errno in (errno.EWOULDBLOCK, errno.EAGAIN, errno.EACCES)
    if msvcrt is not None:
        return err.errno in (errno.EACCES, errno.EDEADLK)
    return False


def _is_fd_exhaustion_text(text: str) -> bool:
    """Text half of _is_fd_exhaustion (shared with the CLI hint)."""
    lowered = text.lower()
    return "too many open files" in lowered or "emfile" in lowered


def _is_fd_exhaustion(exc: BaseException) -> bool:
    """True when *exc* indicates fd exhaustion: EMFILE/ENFILE errno, or the "Too many open files"
    wording for wrapped exceptions (load_jobs wraps the OSError in a RuntimeError)."""
    if isinstance(exc, OSError) and exc.errno in (errno.EMFILE, errno.ENFILE):
        return True
    return _is_fd_exhaustion_text(str(exc))


def _reclaim_fds_best_effort() -> None:
    """Best-effort fd reclamation: gc.collect() closes file objects stuck in reference cycles;
    apply_nofile_soft_limit() raises the RLIMIT_NOFILE soft limit for headroom. Never raises."""
    with contextlib.suppress(Exception):
        import gc

        gc.collect()
    with contextlib.suppress(Exception):
        from hermes_cli.resource_limits import apply_nofile_soft_limit

        apply_nofile_soft_limit(None)


def _resolve_cron_surface_mode(pconfig, logical_platform_name: str) -> str:
    """Return ``"in_channel"`` or ``"thread"`` (default) for a platform config.

    Native: flat ``platforms.<p>.extra.cron_continuable_surface``. Relay-fronted:
    ``platforms.relay.extra.<logical>.cron_continuable_surface`` (same sub-block as the relay's
    Slack knobs); the sub-block wins over the flat key and is scoped to its logical platform.
    Unlike _relay_slack_extra (all-or-nothing), this falls back to the flat key when the sub-block
    omits the knob — deliberate, the flat key must keep working — so a flat value applies to EVERY
    platform the relay fronts (only the D6 capability gate contains it). Scope it on multi-platform
    relays.
    """
    with contextlib.suppress(Exception):
        extra = getattr(pconfig, "extra", None) or {}
        sub = extra.get(str(logical_platform_name or "").lower())
        if isinstance(sub, dict) and sub.get("cron_continuable_surface") is not None:
            raw = sub.get("cron_continuable_surface")
        else:
            raw = extra.get("cron_continuable_surface")
        if raw is not None and str(raw).strip().lower() == "in_channel":
            return "in_channel"
    return "thread"


def _resolve_origin(job: dict) -> Optional[dict]:
    """Extract origin info from a job. Non-dict origins (provenance strings, hand-edited
    jobs.json) are treated as missing — otherwise every fire crashed on ``origin.get``."""
    origin = job.get("origin")
    if isinstance(origin, dict) and origin.get("platform") and origin.get("chat_id"):
        return origin
    return None


def _cron_mirror_delivery_enabled(job: dict, cfg: Optional[dict] = None) -> bool:
    """Whether a cron delivery is also mirrored into the target chat's session transcript.

    Default OFF (cron deliveries live only in the job's own session unless opted in). Precedence:
    per-job ``attach_to_session`` (bool) → global ``cron.mirror_delivery`` → False.
    CARVE-OUT: the ``in_channel`` surface seeds its target session independently of this knob
    (the seed IS that feature; in_channel is itself opt-in) — this knob governs only the
    default/thread-surface mirror. The mirror uses ``mirror_to_session`` at a turn boundary, so it
    is alternation- and cache-safe.
    """
    per_job = job.get("attach_to_session")
    if isinstance(per_job, bool):
        return per_job
    try:
        if cfg is None:
            cfg = load_config() or {}
        return bool((cfg.get("cron", {}) or {}).get("mirror_delivery", False))
    except Exception:
        return False


def _target_matches_origin(origin: dict, platform_name: str, chat_id: str,
                           thread_id: Optional[str]) -> bool:
    """True when a delivery target is the job's own origin conversation.

    Mirroring is scoped to the origin session (guaranteed to exist — the job was created there).
    Fan-out targets (``all``, explicit other chats) are broadcasts and deliberately NOT mirrored.
    """
    if not origin:
        return False
    if str(origin.get("platform", "")).lower() != str(platform_name).lower():
        return False
    if str(origin.get("chat_id", "")) != str(chat_id):
        return False
    # A pinned origin thread_id must match — a target without it is a different lane.
    origin_thread = origin.get("thread_id")
    return origin_thread is None or str(origin_thread) == str(thread_id or "")


# Provenance rank for the dedup OR-merge in _resolve_delivery_targets (higher = stronger mirror
# claim). Broadcasts rank 0 so "origin,all"/"all,origin" keep the origin tag regardless of order.
_MIRROR_PROVENANCE_RANK = {"origin": 3, "origin_fallback": 2, "explicit": 1}


def _target_mirror_eligible(
    job: dict,
    target: dict,
    *,
    global_mirror: bool,
    origin_match: Optional[bool] = None,
) -> bool:
    """Whether a resolved delivery target may receive the transcript mirror.

    Origin targets: always. ``origin_fallback`` (deliver=origin with no captured origin → home
    channel, standing in for the user's primary conversation): same flags as a true origin.
    ``explicit`` ``platform:chat_id``: ONLY with per-job ``attach_to_session: true`` — the global
    flag must never write transcripts into arbitrary explicitly-addressed chats (shared channels,
    other users' DMs). Untagged broadcasts (``all``, bare-platform home) are never eligible.
    ``origin_match`` may be precomputed by the caller; computed here when ``None``.
    """
    if origin_match is None:
        origin = _resolve_origin(job) or {}
        origin_match = _target_matches_origin(
            origin, target.get("platform", ""), target.get("chat_id", ""),
            target.get("thread_id"),
        )
    if origin_match:
        return True
    resolved_from = target.get("_resolved_from")
    if resolved_from == "origin_fallback":
        # Same precedence as _cron_mirror_delivery_enabled (keep in sync): a per-job False must
        # beat a global True even for callers that don't pre-merge `global_mirror`.
        per_job = job.get("attach_to_session")
        if isinstance(per_job, bool):
            return per_job
        return bool(global_mirror)
    if resolved_from == "explicit":
        return job.get("attach_to_session") is True
    return False


def _inchannel_seed_allowed(*, is_dm: bool, user_id: Optional[str]) -> bool:
    """Whether the flat in_channel seed may run.

    Group keys are user-isolated (``…:group:<chat_id>:<user_id>``): seeding without a real user_id
    creates an orphan session no reply resolves to — worse than no seed. DM keys omit user_id, so
    DMs are always seedable; origin-less group targets fall back to the plain mirror.
    """
    return bool(is_dm or user_id)


def _maybe_mirror_cron_delivery(
    job: dict,
    platform_name: str,
    chat_id: str,
    mirror_text: str,
    thread_id: Optional[str] = None,
    user_id: Optional[str] = None,
    *,
    enabled: bool = False,
) -> None:
    """Best-effort mirror of a cron delivery into the origin chat's session.

    No-op unless ``enabled`` (caller resolves it, scoped to the origin target). Rides the same
    ``mirror_to_session`` path as ``send_message``, passing ``user_id`` so user-isolated group
    chats resolve to the scheduling member. All failures swallowed — a successful delivery must
    never be reported failed because the mirror broke.
    """
    if not enabled:
        return
    text = (mirror_text or "").strip()
    if not text:
        return
    try:
        from gateway.mirror import mirror_to_session

        # USER role + labelled prefix, NOT assistant: an assistant-role mirror lands
        # assistant→assistant and breaks strict alternation; consecutive user turns merge safely.
        ok = mirror_to_session(
            platform_name,
            str(chat_id),
            f"[Cron delivery: {job.get('name') or job.get('id', 'cron')}]\n{text}",
            source_label="cron",
            thread_id=thread_id,
            user_id=user_id,
            role="user",
        )
        if ok:
            logger.info(
                "Job '%s': mirrored delivery into %s:%s session transcript",
                job.get("id", "?"), platform_name, chat_id,
            )
        else:
            logger.debug(
                "Job '%s': delivery mirror skipped for %s:%s "
                "(no matching gateway session — cold start)",
                job.get("id", "?"), platform_name, chat_id,
            )
    except Exception as e:
        logger.debug(
            "Job '%s': delivery mirror failed for %s:%s: %s",
            job.get("id", "?"), platform_name, chat_id, e,
        )


def _open_continuable_cron_thread(
    job: dict,
    adapter,
    chat_id: str,
    loop,
) -> Optional[str]:
    """Open a thread for a continuable cron job via ``adapter.create_handoff_thread``. Returns the
    thread_id, or ``None`` (no thread primitive / failed) = caller falls back to the DM mirror."""
    create_thread = getattr(adapter, "create_handoff_thread", None)
    if not callable(create_thread) or loop is None:
        return None
    task_name = job.get("name") or job.get("id", "cron")
    thread_name = f"Hermes — {task_name}"
    try:
        from agent.async_utils import safe_schedule_threadsafe

        coro = create_thread(str(chat_id), thread_name)
        future = safe_schedule_threadsafe(coro, loop)  # type: ignore[arg-type]
        if future is None:
            return None
        new_thread_id = future.result(timeout=30)
        return str(new_thread_id) if new_thread_id else None
    except Exception as e:
        logger.debug(
            "Job '%s': create_handoff_thread failed on %s — falling back to "
            "DM-session mirror: %s",
            job.get("id", "?"), getattr(adapter, "name", "?"), e,
        )
        return None


def _seed_cron_session(
    job: dict,
    adapter,
    platform_name: str,
    chat_id: str,
    text: str,
    *,
    thread_id: Optional[str],
    chat_type: str,
    user_id: Optional[str],
    user_name: Optional[str] = None,
    chat_name: Optional[str],
    scope_id: Optional[str],
    discord_keys_on_thread: bool = False,
) -> bool:
    """Create the session row (so the mirror has a target) and mirror the brief as a USER turn.
    The seeded key must equal the reply's ``build_session_key``: chat_type, user_id, thread_id and
    scope_id (Slack team id) are all part of it, so callers pass exactly what the reply carries."""
    from gateway.config import Platform
    from gateway.session import SessionSource

    seeded_session_id: Optional[str] = None
    session_store = getattr(adapter, "_session_store", None)
    if session_store is not None:
        try:
            platform_enum = Platform(platform_name.lower())
        except (ValueError, KeyError):
            platform_enum = None
        if platform_enum is not None:
            # Discord keys in-thread messages with chat_id == thread_id; Slack/Telegram use the
            # parent channel.
            seed_chat_id = (
                str(thread_id)
                if discord_keys_on_thread and platform_enum == Platform.DISCORD
                else str(chat_id)
            )
            dest_source = SessionSource(
                platform=platform_enum,
                chat_id=seed_chat_id,
                chat_name=chat_name,
                chat_type=chat_type,
                user_id=user_id,
                user_name=user_name,
                thread_id=thread_id,
                scope_id=str(scope_id) if scope_id else None,
            )
            # Create the row and pass its exact id to the mirror — origin-heuristic rediscovery
            # bails on populated chats.
            _entry = session_store.get_or_create_session(dest_source)
            seeded_session_id = getattr(_entry, "session_id", None)

    from gateway.mirror import mirror_to_session

    return mirror_to_session(
        platform_name,
        str(chat_id),
        f"[Cron delivery: {job.get('name') or job.get('id', 'cron')}]\n{text}",
        source_label="cron",
        thread_id=thread_id,
        user_id=user_id,
        role="user",
        session_id=seeded_session_id,
    )


def _seed_cron_thread_session(
    job: dict,
    adapter,
    platform_name: str,
    chat_id: str,
    thread_id: str,
    mirror_text: str,
    chat_name: Optional[str] = None,
    is_dm: bool = False,
    scope_id: Optional[str] = None,
) -> None:
    """Seed the freshly-opened cron thread's session with the brief (never raises), else the
    user's in-thread reply resolves to a transcript without it. Threads are participant-shared (no
    real user_id); a DM thread must seed ``chat_type="dm"`` — DM-thread replies route through the DM
    arm (``…:dm:<chat>:<thread>``), so a "thread"-typed seed is a row no DM reply ever hits."""
    text = (mirror_text or "").strip()
    if not text:
        return
    try:
        ok = _seed_cron_session(
            job, adapter, platform_name, chat_id, text,
            thread_id=str(thread_id),
            chat_type="dm" if is_dm else "thread",
            user_id="system:cron",
            user_name="Cron",
            chat_name=chat_name,
            scope_id=scope_id,
            discord_keys_on_thread=True,
        )
        if ok:
            logger.info(
                "Job '%s': opened continuable thread %s on %s:%s and seeded the brief",
                job.get("id", "?"), thread_id, platform_name, chat_id,
            )
        else:
            logger.warning(
                "Job '%s': thread seed did NOT land on %s:%s thread=%s — an "
                "in-thread reply will not see this brief",
                job.get("id", "?"), platform_name, chat_id, thread_id,
            )
    except Exception as e:
        # WARNING, not debug: a silent seed failure IS the continuation-amnesia bug.
        logger.warning(
            "Job '%s': seeding cron thread session failed for %s:%s:%s: %s",
            job.get("id", "?"), platform_name, chat_id, thread_id, e,
        )


def _seed_cron_channel_session(
    job: dict,
    adapter,
    platform_name: str,
    chat_id: str,
    mirror_text: str,
    *,
    is_dm: bool,
    user_id: Optional[str],
    chat_name: Optional[str] = None,
    scope_id: Optional[str] = None,
) -> bool:
    """Seed the FLAT (thread_id=None) session for an ``in_channel`` delivery; True on success.
    ``mirror_to_session`` only APPENDS to an existing session and the flat row is only created by an
    inbound human message, so the row must be created first or the brief is silently dropped. Group
    keys are user-isolated (``…:group:<chat_id>:<user_id>``): the seed MUST carry the origin's real
    user_id, not ``system:cron``; DM keys omit user_id. chat_type mirrors the inbound handler."""
    text = (mirror_text or "").strip()
    if not text:
        return False
    try:
        chat_type = "dm" if is_dm else "group"
        ok = _seed_cron_session(
            job, adapter, platform_name, chat_id, text,
            thread_id=None,  # flat — the whole-channel/DM session
            chat_type=chat_type,
            user_id=str(user_id) if user_id else None,
            chat_name=chat_name,
            scope_id=scope_id,
        )
        if ok:
            logger.info(
                "Job '%s': seeded flat in_channel session on %s:%s (chat_type=%s)",
                job.get("id", "?"), platform_name, chat_id, chat_type,
            )
        return bool(ok)
    except Exception as e:
        # WARNING, not debug: a silent seed failure IS the continuation-amnesia bug.
        logger.warning(
            "Job '%s': seeding in_channel session failed for %s:%s: %s",
            job.get("id", "?"), platform_name, chat_id, e,
        )
        return False


def _cron_job_origin_log_suffix(job: dict) -> str:
    """Secret-free provenance suffix (origin platform/chat/source-IP fields) for security warnings
    about a bad stored ``context_from`` reference, where no live request object exists."""
    origin = job.get("origin")
    if not isinstance(origin, dict):
        return ""

    fields = []
    for key in ("platform", "chat_id", "thread_id", "source_ip", "remote", "forwarded_for"):
        value = origin.get(key)
        if value is None:
            continue
        text = str(value).replace("\r", " ").replace("\n", " ").strip()
        if text:
            fields.append(f"origin_{key}={text[:200]!r}")
    return " " + " ".join(fields) if fields else ""


def _plugin_cron_env_var(platform_name: str) -> str:
    """Cron home-channel env var registered by a plugin ``PlatformEntry.cron_deliver_env_var``."""
    with contextlib.suppress(Exception):
        from hermes_cli.plugins import discover_plugins
        discover_plugins()  # idempotent
        from gateway.platform_registry import platform_registry
        entry = platform_registry.get(platform_name.lower())
        if entry and entry.cron_deliver_env_var:
            return entry.cron_deliver_env_var
    return ""


def _is_known_delivery_platform(platform_name: str) -> bool:
    """Valid cron delivery platform: built-in, or plugin with a ``cron_deliver_env_var``."""
    name = platform_name.lower()
    if name in _KNOWN_DELIVERY_PLATFORMS:
        return True
    return bool(_plugin_cron_env_var(name))


def _resolve_home_env_var(platform_name: str) -> str:
    """Env var name for a platform's cron home channel (built-in table, then plugin registry)."""
    name = platform_name.lower()
    env_var = _HOME_TARGET_ENV_VARS.get(name)
    if env_var:
        return env_var
    return _plugin_cron_env_var(name)


def _get_config_home_channel(platform_name: str):
    """Persisted ``HomeChannel`` from gateway config — the canonical store ``/sethome`` writes.

    The ``<PLATFORM>_HOME_CHANNEL`` env var is only a best-effort mirror; relay-fronted platforms
    may exist solely in config.yaml, so reading only the env mirror silently drops their delivery.
    """
    try:
        from gateway.config import load_gateway_config, Platform

        config = load_gateway_config()
        platform = Platform(platform_name.lower())
        return config.get_home_channel(platform)
    except Exception:
        logger.debug(
            "config home_channel lookup failed for platform %r",
            platform_name, exc_info=True,
        )
        return None


def _env_home_target_chat_id(platform_name: str) -> str:
    """Home chat id from the env mirror only (no config).

    Reads via ``get_secret``, not ``os.getenv``: in a multiplex gateway the tick runs with the
    job-owning profile's secret scope (run_one_job sets it), so this resolves the OWNING profile's
    chat id rather than the host process's environ.
    """
    env_var = _resolve_home_env_var(platform_name)
    if not env_var:
        return ""
    try:
        from agent.secret_scope import get_secret
    except Exception:
        get_secret = None  # type: ignore
    if get_secret is not None:
        value = get_secret(env_var, "")
        if not value:
            legacy = _LEGACY_HOME_TARGET_ENV_VARS.get(env_var)
            if legacy:
                value = get_secret(legacy, "")
        return value or ""
    value = os.getenv(env_var, "")
    if not value:
        legacy = _LEGACY_HOME_TARGET_ENV_VARS.get(env_var)
        if legacy:
            value = os.getenv(legacy, "")
    return value


def _get_home_target_chat_id(platform_name: str) -> str:
    """Home target chat id: env var (first, so operator overrides win) → legacy env var →
    config.yaml ``home_channel``."""
    value = _env_home_target_chat_id(platform_name)
    if value:
        return value
    home = _get_config_home_channel(platform_name)
    if home is not None and home.chat_id:
        return str(home.chat_id)
    return ""


def _get_home_target_thread_id(platform_name: str) -> Optional[str]:
    """Optional thread/topic id for a platform home target.

    Telegram: ``TELEGRAM_CRON_THREAD_ID`` overrides ``TELEGRAM_HOME_CHANNEL_THREAD_ID`` — in topic
    mode a root-DM delivery lands in the system-only lobby where the user cannot reply.
    """
    env_var = _resolve_home_env_var(platform_name)
    try:
        from agent.secret_scope import get_secret
    except Exception:
        get_secret = None  # type: ignore

    def _scope_get(name: str) -> str:
        if get_secret is None:
            return ""
        v = get_secret(name, "")
        return v if v is not None else ""

    if platform_name.lower() == "telegram":
        cron_thread = _scope_get("TELEGRAM_CRON_THREAD_ID").strip()
        if cron_thread:
            return cron_thread
    if get_secret is not None:
        value = _scope_get(f"{env_var}_THREAD_ID").strip() if env_var else ""
        if not value and env_var:
            legacy = _LEGACY_HOME_TARGET_ENV_VARS.get(env_var)
            if legacy:
                value = _scope_get(f"{legacy}_THREAD_ID").strip()
    else:
        value = os.getenv(f"{env_var}_THREAD_ID", "").strip() if env_var else ""
        if not value and env_var:
            legacy = _LEGACY_HOME_TARGET_ENV_VARS.get(env_var)
            if legacy:
                value = os.getenv(f"{legacy}_THREAD_ID", "").strip()
    if value:
        return value
    # config.yaml fallback only when the chat id also came from config (an env-provided chat id
    # keeps its env-provided thread semantics).
    if not _env_home_target_chat_id(platform_name):
        home = _get_config_home_channel(platform_name)
        if home is not None and home.thread_id:
            return str(home.thread_id)
    return None


def _iter_home_target_platforms():
    """Iterate built-in + plugin platform names that expose a home channel."""
    for name in _HOME_TARGET_ENV_VARS:
        yield name
    with contextlib.suppress(Exception):
        from hermes_cli.plugins import discover_plugins
        discover_plugins()  # idempotent
        from gateway.platform_registry import platform_registry
        for entry in platform_registry.plugin_entries():
            if entry.cron_deliver_env_var and entry.name not in _HOME_TARGET_ENV_VARS:
                yield entry.name


def _relay_fronted_delivery_platforms(connected: set) -> set:
    """Logical platforms deliverable through a connected relay. ``get_connected_platforms()`` only
    sees native platforms; fronted ones come from the same ``GATEWAY_RELAY_PLATFORMS`` stamp
    fire-time routing uses (validation symmetric with routing). No relay -> empty set."""
    if "relay" not in connected:
        return set()
    try:
        from gateway.relay import relay_fronted_platforms

        return relay_fronted_platforms()
    except Exception:
        logger.debug("relay fronted-platform lookup failed", exc_info=True)
        return set()


def cron_delivery_targets() -> list[dict]:
    """Platforms a cron job can auto-deliver to (single source of truth for UIs).

    Included when a valid delivery platform AND gateway-configured; ``home_target_set`` flags
    whether the home channel exists. Returns ``{"id", "name", "home_target_set", "home_env_var"}``
    dicts in canonical order; callers prepend the implicit ``local`` option themselves.
    """
    targets: list[dict] = []
    try:
        from gateway.config import load_gateway_config

        gateway_config = load_gateway_config()
        connected = {p.value for p in gateway_config.get_connected_platforms()}
        connected |= _relay_fronted_delivery_platforms(connected)
    except Exception:
        logger.debug("cron_delivery_targets: gateway config unavailable", exc_info=True)
        connected = set()

    for name in _iter_home_target_platforms():
        if name not in connected:
            continue
        if not _is_known_delivery_platform(name):
            continue
        env_var = _resolve_home_env_var(name)
        targets.append(
            {
                "id": name,
                "name": name.replace("_", " ").title(),
                "home_target_set": bool(_get_home_target_chat_id(name)),
                "home_env_var": env_var or None,
            }
        )

    # Bot Chat targets: one per local profile (machine-local; no gateway config or home channel).
    try:
        from hermes_cli.profiles import list_profile_names

        for profile_name in list_profile_names():
            targets.append(
                {
                    "id": f"{BOT_CHAT_PLATFORM}:{profile_name}",
                    "name": f"Bot Chat ({profile_name})",
                    "home_target_set": True,
                    "home_env_var": None,
                }
            )
    except Exception:
        logger.debug("cron_delivery_targets: profile listing unavailable", exc_info=True)
    return targets


def _origin_thread_is_stale(origin: dict) -> bool:
    """True when a Slack origin's thread is a stale creation-turn artifact.

    Thread-per-message Slack stamps each top-level message id as the session thread (a KEY, not a
    location); old jobs carry it as ``origin.thread_id``. Heuristic: if the origin chat IS the Slack
    home chat, the pinned thread is that artifact and delivery goes top-level (or to the home
    target's thread). Non-home chats keep their threads.
    """
    if str(origin.get("platform") or "").lower() != "slack" or not origin.get("thread_id"):
        return False
    home_chat = _get_home_target_chat_id("slack")
    return bool(home_chat) and str(origin.get("chat_id")) == str(home_chat)


def _origin_delivery_thread(origin: dict):
    """The thread a deliver=origin job should use, stale stamps dropped."""
    if _origin_thread_is_stale(origin):
        return _get_home_target_thread_id("slack") or None
    return origin.get("thread_id")


def _home_target(platform_name: str, chat_id: str, resolved_from: Optional[str] = None) -> dict:
    """Target dict for a platform's configured home channel (+ optional mirror provenance)."""
    target = {
        "platform": platform_name,
        "chat_id": chat_id,
        "thread_id": _get_home_target_thread_id(platform_name),
    }
    if resolved_from:
        target["_resolved_from"] = resolved_from
    return target


def _resolve_single_delivery_target(job: dict, deliver_value: str) -> Optional[dict]:
    """Resolve one concrete auto-delivery target for a cron job."""

    origin = _resolve_origin(job)

    if deliver_value == "local":
        return None

    # Must precede the generic platform:chat_id split so the profile name isn't parsed as chat_id.
    bot_chat_profile = parse_bot_chat_deliver_token(deliver_value)
    if bot_chat_profile is not None:
        return _resolve_bot_chat_target(job, bot_chat_profile)

    if deliver_value == "origin":
        if origin:
            return {
                "platform": origin["platform"],
                "chat_id": str(origin["chat_id"]),
                "thread_id": _origin_delivery_thread(origin),
                # Provenance for _target_mirror_eligible.
                "_resolved_from": "origin",
            }
        # No origin (API/script job): fall back to a home channel instead of silently dropping.
        for platform_name in _iter_home_target_platforms():
            chat_id = _get_home_target_chat_id(platform_name)
            if chat_id:
                logger.info(
                    "Job '%s' has deliver=origin but no origin; falling back to %s home channel",
                    job.get("name", job.get("id", "?")),
                    platform_name,
                )
                # Stands in for the primary conversation (NOT a broadcast): mirror-eligible.
                return _home_target(platform_name, chat_id, "origin_fallback")
        return None

    if ":" in deliver_value:
        platform_name, rest = deliver_value.split(":", 1)
        platform_key = platform_name.lower()

        from tools.send_message_tool import (
            prepare_send_message_platforms,
            resolve_send_target,
        )

        prepare_send_message_platforms()
        # pass_unresolved_references: no model in the loop to react; an unknown-to-directory target
        # must reach the adapter as written or the job's output is silently lost.
        chat_id, thread_id, resolution_error = resolve_send_target(
            platform_key, rest, pass_unresolved_references=True
        )
        if resolution_error:
            logger.warning("Invalid cron delivery target '%s': %s", deliver_value, resolution_error)
            return None

        if (
            thread_id is None
            and platform_key == "slack"
            and origin
            and str(origin.get("platform") or "").lower() == platform_key
            and str(origin.get("chat_id")) == str(chat_id)
            and origin.get("thread_id")
            and not _origin_thread_is_stale(origin)
        ):
            thread_id = origin.get("thread_id")

        return {
            "platform": platform_name,
            "chat_id": chat_id,
            "thread_id": thread_id,
            # Mirror-eligible only under the job's own attach_to_session opt-in.
            "_resolved_from": "explicit",
        }

    platform_name = deliver_value
    if origin and origin.get("platform") == platform_name:
        chat_id = _get_home_target_chat_id(platform_name)
        if chat_id:
            return _home_target(platform_name, chat_id)
        return {
            "platform": platform_name,
            "chat_id": str(origin["chat_id"]),
            "thread_id": origin.get("thread_id"),
        }

    if not _is_known_delivery_platform(platform_name):
        return None
    chat_id = _get_home_target_chat_id(platform_name)
    return _home_target(platform_name, chat_id) if chat_id else None


def _get_bot_chat_delivery_timeout() -> int:
    """Timeout for one bot-chat delivery turn (a full agent turn — minutes, not seconds).
    ``cron.bot_chat_delivery_timeout_seconds``; default 600."""
    try:
        cfg = load_config()
        value = int(cfg.get("cron", {}).get("bot_chat_delivery_timeout_seconds", 600))
        return value if value > 0 else 600
    except Exception:
        return 600


def _deliver_to_bot_chat(job: dict, content: str, profile: str) -> Optional[str]:
    """Deliver job output into a profile's canonical Bot Chat as a real inbound user turn.

    Runs ``hermes [-p <profile>] chat --in ~ -c "Bot Chat" --create-if-missing -Q --query-file`` —
    the same lane Bot Mode agent-to-agent messages use, so canonical-session rules apply and it is
    alternation-safe (inbound turn, not a transcript splice). ``profile`` is ``""`` for the job's
    own profile. Returns None on success or an error string for ``last_delivery_error``.
    """
    import shutil as _shutil
    import tempfile

    job_id = job.get("id", "?")
    job_name = job.get("name", job_id)

    hermes_bin = _shutil.which("hermes")
    if hermes_bin:
        argv = [hermes_bin]
    else:
        try:
            import importlib.util as _ilu

            if _ilu.find_spec("hermes_cli") is not None:
                argv = [sys.executable, "-m", "hermes_cli.main"]
            else:
                return "bot-chat delivery failed: hermes CLI not resolvable"
        except Exception:
            return "bot-chat delivery failed: hermes CLI not resolvable"

    env = os.environ.copy()
    if profile:
        argv += ["-p", profile]
        # -p owns profile resolution; this scheduler's HERMES_HOME must not shadow it.
        env.pop("HERMES_HOME", None)

    # Prefix marks this as scheduled output, not the human (Bot Mode sender-attribution).
    message = (
        f'[Cronjob "{job_name}" output — scheduled job, not the user. '
        f"Review it, act on anything that needs action, and summarize "
        f"for the chat.]\n\n{content}"
    )

    query_file = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", suffix=".txt", prefix="hermes-cron-botchat-",
            delete=False,
        ) as fh:
            fh.write(message)
            query_file = fh.name

        argv += [
            "chat", "--in", "~", "-c", "Bot Chat", "--create-if-missing",
            "-Q", "--query-file", query_file,
        ]

        result = subprocess.run(
            argv,
            capture_output=True,
            text=True,
            timeout=_get_bot_chat_delivery_timeout(),
            env=env,
            creationflags=windows_hide_flags(),
        )
        if result.returncode != 0:
            tail = (result.stderr or result.stdout or "").strip()[-500:]
            msg = (
                f"bot-chat delivery to profile "
                f"'{profile or '(own)'}' failed (exit {result.returncode})"
                + (f": {tail}" if tail else "")
            )
            logger.warning("Job '%s': %s", job_id, msg)
            return msg
        logger.info("Job '%s': delivered to Bot Chat of profile '%s'", job_id, profile or "(own)")
        return None
    except subprocess.TimeoutExpired:
        msg = (
            f"bot-chat delivery to profile '{profile or '(own)'}' timed out "
            f"after {_get_bot_chat_delivery_timeout()}s (the bot's turn may "
            "still complete; raise cron.bot_chat_delivery_timeout_seconds if "
            "this recurs)"
        )
        logger.warning("Job '%s': %s", job_id, msg)
        return msg
    except Exception as e:
        msg = f"bot-chat delivery failed: {str(e) or type(e).__name__}"
        logger.warning("Job '%s': %s", job_id, msg, exc_info=True)
        return msg
    finally:
        if query_file:
            with contextlib.suppress(OSError):
                os.unlink(query_file)


def _normalize_deliver_value(deliver) -> str:
    """Normalize ``deliver`` to its canonical comma-separated string; ``"local"`` when falsy.

    Lists/tuples (MCP clients, hand-edited jobs.json) are flattened — ``str(["telegram"])`` would
    yield ``"['telegram']"`` and fail resolution silently.
    """
    if deliver is None or deliver == "":
        return "local"
    if isinstance(deliver, (list, tuple)):
        parts = [str(p).strip() for p in deliver if str(p).strip()]
        return ",".join(parts) if parts else "local"
    return str(deliver)


# Routing tokens resolve at fire time (a job outlives platform wiring). ``all`` = platforms with a
# configured home chat_id (_expand_routing_tokens); ``bot-chat`` is NOT in ``all`` (costs a turn).
_ROUTING_TOKENS = frozenset({"all"})

# Pseudo-platform: deliver output as a real inbound turn into a profile's "Bot Chat" (not a mirror).
# ``bot-chat`` = own profile; ``bot-chat:<name>`` = named profile on THIS machine.
BOT_CHAT_PLATFORM = "bot-chat"


def parse_bot_chat_deliver_token(part: str) -> Optional[str]:
    """``bot-chat[:<name>]`` → ``""`` (own profile), the name, or ``None`` if not a bot-chat token.
    Token is case-insensitive; the name is normalized later by the profile layer."""
    raw = (part or "").strip()
    lowered = raw.lower()
    if lowered == BOT_CHAT_PLATFORM:
        return ""
    prefix = BOT_CHAT_PLATFORM + ":"
    if lowered.startswith(prefix):
        return raw[len(prefix):].strip()
    return None


def _resolve_bot_chat_target(job: dict, profile_arg: str) -> Optional[dict]:
    """Resolve a bot-chat token to a delivery target. ``""`` = own profile (no ``-p`` needed);
    otherwise the profile must exist locally — cross-machine delivery is intentionally unsupported
    so same-named profiles on other gateways can never be targeted by accident."""
    if not profile_arg:
        return {"platform": BOT_CHAT_PLATFORM, "chat_id": "", "thread_id": None}
    try:
        from hermes_cli.profiles import normalize_profile_name, profile_exists

        canon = normalize_profile_name(profile_arg)
        if not profile_exists(canon):
            logger.warning(
                "Job '%s': bot-chat delivery profile '%s' not found on this "
                "machine — skipping target",
                job.get("id", "?"), profile_arg,
            )
            return None
        return {"platform": BOT_CHAT_PLATFORM, "chat_id": canon, "thread_id": None}
    except Exception:
        logger.warning(
            "Job '%s': failed to resolve bot-chat profile '%s'",
            job.get("id", "?"), profile_arg, exc_info=True,
        )
        return None


def _expand_routing_tokens(part: str) -> List[str]:
    """Expand ``all`` to every home-target platform with a configured chat_id; non-tokens pass
    through as a single-element list."""
    token = part.lower()
    if token not in _ROUTING_TOKENS:
        return [part]
    expanded: List[str] = []
    for platform_name in _iter_home_target_platforms():
        if _get_home_target_chat_id(platform_name):
            expanded.append(platform_name)
    return expanded


def _delivery_lane_value(job: dict, *, for_failure: bool = False):
    """Raw deliver-lane value for a run outcome: the failure lane when ``for_failure`` and the job
    overrides it, else ``deliver``. Bookkeeping (outcome classification, unresolved-origin, incident
    'alerted' marking) must read the SAME lane the notice was routed through (NS-788)."""
    if for_failure:
        failure_deliver = job.get("failure_deliver")
        if failure_deliver is not None and str(failure_deliver).strip():
            return failure_deliver
    return job.get("deliver", "local")


def _resolve_delivery_targets(job: dict, *, for_failure: bool = False) -> List[dict]:
    """Resolve auto-delivery targets from comma-separated ``deliver``; ``all`` expands to every
    platform with a home channel and combines with explicit targets. Dedup by (platform, chat_id,
    thread_id). ``for_failure=True`` (failure summaries, interrupted-run notices, drift/preflight
    alerts) resolves from ``failure_deliver`` INSTEAD when the job carries one — ``failure_deliver:
    local`` is the structural opt-out for shared channels; absent, failures follow ``deliver``."""
    deliver = _normalize_deliver_value(_delivery_lane_value(job, for_failure=for_failure))
    if deliver == "local":
        return []

    raw_parts = [p.strip() for p in deliver.split(",") if p.strip()]

    parts: List[str] = []
    for raw in raw_parts:
        parts.extend(_expand_routing_tokens(raw))

    seen = {}
    targets = []
    for part in parts:
        target = _resolve_single_delivery_target(job, part)
        if target:
            key = (target["platform"].lower(), str(target["chat_id"]), target.get("thread_id"))
            if key not in seen:
                seen[key] = target
                targets.append(target)
            else:
                # OR-merge provenance on dedup: "origin,all" in either order must keep the
                # origin/origin_fallback tag or mirror eligibility would depend on token order.
                kept = seen[key]
                if _MIRROR_PROVENANCE_RANK.get(str(target.get("_resolved_from") or ""), 0) > \
                        _MIRROR_PROVENANCE_RANK.get(str(kept.get("_resolved_from") or ""), 0):
                    kept["_resolved_from"] = target.get("_resolved_from")
    return targets


def _resolve_delivery_target(job: dict) -> Optional[dict]:
    """Resolve the concrete auto-delivery target for a cron job, if any."""
    targets = _resolve_delivery_targets(job)
    return targets[0] if targets else None


# Audio routing is centralized in gateway.platforms.base.should_send_media_as_audio().
_VIDEO_EXTS = frozenset({'.mp4', '.mov', '.avi', '.mkv', '.webm', '.3gp'})
_IMAGE_EXTS = frozenset({'.jpg', '.jpeg', '.png', '.webp', '.gif'})


def _send_media_via_adapter(
    adapter,
    chat_id: str,
    media_files: list,
    metadata: dict | None,
    loop,
    job: dict,
    platform=None,
) -> list:
    """Send MEDIA files as native attachments (routed by extension, as in
    _process_message_background). Returns per-file error strings so a dropped attachment surfaces
    in run status, not just the gateway log."""
    from gateway.platforms.base import (
        BasePlatformAdapter, should_send_media_as_audio, validate_media_delivery_path,
    )
    from agent.async_utils import safe_schedule_threadsafe

    job_ref = {"id": job.get("id", "?")}
    errors: list = []
    requested = [(str(p), v) for p, v in (media_files or [])]
    media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files)
    # Report paths the safety filter dropped (missing file, denied prefix, strict-mode miss).
    kept = {p for p, _ in media_files}
    for raw_path, _v in requested:
        try:
            dropped = validate_media_delivery_path(raw_path) not in kept
        except Exception:
            dropped = True
        if dropped:
            errors.append(f"attachment dropped by media path policy: {raw_path}")

    route_platform = platform if platform is not None else getattr(adapter, "platform", None)
    for media_path, _is_voice in media_files:
        try:
            ext = Path(media_path).suffix.lower()
            if should_send_media_as_audio(route_platform, ext, is_voice=_is_voice):
                coro = adapter.send_voice(chat_id=chat_id, audio_path=media_path, metadata=metadata)
            elif ext in _VIDEO_EXTS:
                coro = adapter.send_video(chat_id=chat_id, video_path=media_path, metadata=metadata)
            elif ext in _IMAGE_EXTS:
                coro = adapter.send_image_file(chat_id=chat_id, image_path=media_path, metadata=metadata)
            else:
                coro = adapter.send_document(chat_id=chat_id, file_path=media_path, metadata=metadata)

            future = safe_schedule_threadsafe(coro, loop)
            if future is None:
                _note_target_error(
                    job_ref, f"cannot send media {media_path}: gateway loop unavailable", errors,
                )
                return errors
            try:
                # Large attachments can exceed 30s; configurable via _get_media_send_timeout().
                result = future.result(timeout=_get_media_send_timeout())
            except TimeoutError:
                future.cancel()
                raise
            if result and not getattr(result, "success", True):
                _note_target_error(
                    job_ref,
                    f"media send failed for {media_path}: {getattr(result, 'error', 'unknown')}",
                    errors,
                )
        except Exception as e:
            # TimeoutError etc. have an empty str(); fall back to the class name.
            _note_target_error(
                job_ref, f"failed to send media {media_path}: {str(e) or type(e).__name__}", errors,
            )
    return errors


def _confirm_adapter_delivery(send_result, job_id: str = "?", unverified: Optional[list] = None) -> bool:
    """Return True only if ``send_result`` unambiguously confirms delivery.

    ``None`` or no ``success`` attr/key is NOT success (would log "delivered" while nothing was
    sent). ``delivered is False`` REJECTS even with truthy ``success``: the silence-narration filter
    returns ``{"success": True, "delivered": False}`` (dropped). No ``message_id``/``raw_response``
    is still accepted (some adapters return a bare success) but logged at WARNING as UNVERIFIED.
    """
    if send_result is None:
        return False
    if isinstance(send_result, dict):
        if "success" not in send_result:
            return False
        success = bool(send_result.get("success"))
        delivered = send_result.get("delivered")
        message_id = send_result.get("message_id")
        raw_response = send_result.get("raw_response")
    else:
        if not hasattr(send_result, "success"):
            return False
        success = bool(getattr(send_result, "success"))
        delivered = getattr(send_result, "delivered", None)
        message_id = getattr(send_result, "message_id", None)
        raw_response = getattr(send_result, "raw_response", None)
    if not success or delivered is False:
        return False
    if message_id is None and not raw_response:
        logger.warning(
            "Job '%s': live adapter reported success with no delivery evidence "
            "(no message_id, no raw_response) — treating as delivered but "
            "UNVERIFIED",
            job_id,
        )
        if unverified is not None:
            unverified.append(True)
    return True


def _is_channel_dm_topic(
    runtime_adapter: Any,
    chat_id: Any,
    loop: Any,
    job_id: str,
) -> bool:
    """Is an ambiguous ``telegram:<positive_chat_id>:<numeric_thread_id>`` target a channel
    Direct-Messages topic (``direct_messages_topic_id``) rather than a private-chat forum topic
    (``message_thread_id``)? Shape cannot decide; signal is ``get_chat_info`` type == ``channel``.
    Fails SAFE to False (thread routing) without a probe or on any probe error/timeout."""
    # Resolve on the CLASS, not the instance: a MagicMock instance auto-creates a truthy
    # ``get_chat_info``, so an instance-level probe would misclassify test doubles.
    get_chat_info = getattr(type(runtime_adapter), "get_chat_info", None)
    if not callable(get_chat_info):
        return False
    try:
        from agent.async_utils import safe_schedule_threadsafe

        future = safe_schedule_threadsafe(
            get_chat_info(runtime_adapter, str(chat_id)), loop,  # type: ignore[arg-type]
        )
        if future is None:
            return False
        # Metadata-only call, so a shorter bound than the send waits is intentional.
        info = future.result(timeout=10)
    except Exception:
        logger.debug(
            "Job '%s': get_chat_info probe failed for chat=%s — "
            "defaulting to message_thread_id routing",
            job_id, chat_id, exc_info=True,
        )
        return False
    is_channel = isinstance(info, dict) and str(info.get("type") or "").lower() == "channel"
    if is_channel:
        logger.info(
            "Job '%s': chat=%s is a channel — routing via direct_messages_topic_id",
            job_id, chat_id,
        )
    return is_channel


def _cron_delivery_notify_enabled(cfg: Optional[dict]) -> bool:
    """Resolve ``cron.delivery.notify`` (default True). Only an explicit ``False`` disables; a
    missing/malformed section keeps the default so a typo cannot silently mute briefs."""
    try:
        cron_cfg = (cfg or {}).get("cron")
        if not isinstance(cron_cfg, dict):
            return True
        delivery_cfg = cron_cfg.get("delivery")
        if not isinstance(delivery_cfg, dict):
            return True
        return delivery_cfg.get("notify", True) is not False
    except Exception:
        return True


def _record_delivery_verification(job: dict, unverified_targets: list) -> None:
    """Persist ``last_delivery_unverified``: list of ``platform:chat_id`` targets acked with no
    evidence, or None. Skips the write when unchanged; never raises (bookkeeping must not fail a
    delivery)."""
    new_value = list(unverified_targets) or None
    if (job.get("last_delivery_unverified") or None) == new_value:
        return
    try:
        from cron.jobs import update_job

        update_job(job["id"], {"last_delivery_unverified": new_value})
    except Exception as exc:  # pragma: no cover - defensive
        logger.debug("Job '%s': could not record delivery verification: %s", job.get("id"), exc)


@dataclass
class _TargetDelivery:
    """Per-target delivery state shared by the live-adapter and standalone lanes."""

    job: dict
    platform: Any
    platform_name: str
    chat_id: str
    thread_id: Optional[str]
    transport: Any
    pconfig: Any
    runtime_adapter: Any
    target_adapters: Any
    config: Any
    loop: Any
    notify_delivery: bool
    origin: dict
    origin_target: bool
    origin_user_id: Optional[str]
    is_dm_target: bool
    mirror_text: str
    mirror_this_target: bool
    in_channel_surface: bool
    inchannel_continuable: bool
    opened_thread_id: Optional[str]

    @property
    def is_relay(self) -> bool:
        return self.transport is not None and self.transport.is_relay

    @property
    def where(self) -> str:
        return f"{self.platform_name}:{self.chat_id}"


def _note_target_error(job: dict, msg: str, errors: list) -> None:
    """Log a per-target delivery failure as a WARNING and record it in ``errors``."""
    logger.warning("Job '%s': %s", job["id"], msg)
    errors.append(msg)


def _warn_live_lane_failure(job: dict, msg: str, is_relay: bool) -> None:
    """Relay targets have no standalone fallback, so the log line must not promise one."""
    if is_relay:
        logger.warning("Job '%s': %s", job["id"], msg)
    else:
        logger.warning("Job '%s': %s, falling back to standalone", job["id"], msg)


def _resolve_target_transport(job: dict, platform, platform_name: str, target: dict, adapters, config):
    """Resolve ``(transport, pconfig, runtime_adapter, target_adapters)`` for one target, or
    ``(None, error)`` when it cannot be served (relay-fronted with no live transport, or not
    configured/enabled)."""
    from gateway.delivery import resolve_delivery_transport

    target_adapters = adapters
    if isinstance(adapters, SharedRouteAdapters):
        # Credentialless satellite: the primary adapter serves THIS target only when an exact
        # primary route maps it to this profile; a miss fails closed below.
        shared = adapters.get(platform, target)
        target_adapters = {platform: shared} if shared is not None else {}
    transport = resolve_delivery_transport(platform, config, target_adapters)
    if transport is not None:
        pconfig = transport.config
        runtime_adapter = transport.adapter
    else:
        # Relay-fronted platforms have NO standalone fallback (the connector owns the credential),
        # so surface that instead of the native configured/enabled gate, which misdiagnoses them.
        from gateway.relay import relay_fronted_platforms

        if platform_name in relay_fronted_platforms():
            return None, (
                f"platform '{platform_name}' is relay-fronted and has no "
                "live gateway transport; start the gateway (its ticker "
                "owns relay-fronted delivery and will fire the job on "
                "schedule)"
            )
        pconfig = config.platforms.get(platform)
        runtime_adapter = None

    if transport is not None and transport.is_relay:
        # Relay transport carries the RELAY adapter's config (enablement already checked). The
        # logical platform is deliberately NOT natively enabled, so the native gate must not apply.
        if pconfig is None:
            from gateway.config import PlatformConfig
            pconfig = PlatformConfig(enabled=True)
    elif not pconfig or not pconfig.enabled:
        return None, f"platform '{platform_name}' not configured/enabled"
    return (transport, pconfig, runtime_adapter, target_adapters), None


def _inchannel_surface_supported(runtime_adapter, platform_name: str) -> bool:
    """D6 probe: can this adapter deliver a continuable in_channel brief on ``platform_name``?
    Per-platform check first (one RelayAdapter fronts N platforms; the scalar attr only carries
    the PRIMARY identity's bit); native adapters use the class attribute."""
    per_platform_check = getattr(runtime_adapter, "supports_inchannel_continuable_for_platform", None)
    if callable(per_platform_check):
        try:
            return bool(per_platform_check(platform_name))
        except Exception:
            return False
    return bool(getattr(runtime_adapter, "supports_inchannel_continuable", False))


def _live_route_metadata(t: _TargetDelivery) -> tuple[Optional[str], dict, dict]:
    """Compute ``(route_thread_id, route_metadata, media_metadata)`` for a live send, ONCE so text
    and media agree. ``telegram:<positive_chat_id>:<numeric_thread_id>`` is ambiguous (private
    forum topic vs channel DM topic need OPPOSITE routing) — see ``_is_channel_dm_topic``.
    ``thread_id`` rides in ``route_metadata`` to bypass the DeliveryRouter's private-chat
    reply-anchor requirement for anchorless cron sends."""
    from gateway.config import Platform
    from gateway.delivery import _looks_like_int, looks_like_telegram_private_chat_id

    job = t.job
    thread_id = t.thread_id
    is_ambiguous_telegram_topic = (
        t.platform == Platform.TELEGRAM
        and thread_id is not None
        and looks_like_telegram_private_chat_id(str(t.chat_id))
        and _looks_like_int(str(thread_id))
    )
    if is_ambiguous_telegram_topic and _is_channel_dm_topic(
        t.runtime_adapter, t.chat_id, t.loop, job["id"],
    ):
        # Channel DM topic: direct_messages_topic_id, no bare thread_id; media mirrors text.
        route_thread_id = None
        route_metadata = {
            "direct_messages_topic_id": str(thread_id),
            "job_id": job["id"],
            "notify": t.notify_delivery,
        }
        media_metadata = {"direct_messages_topic_id": str(thread_id), "notify": t.notify_delivery}
    else:
        # Forum-style topic or non-topic target: message_thread_id.
        route_thread_id = str(thread_id) if thread_id is not None else None
        route_metadata = {"job_id": job["id"], "notify": t.notify_delivery}
        if route_thread_id:
            route_metadata["thread_id"] = route_thread_id
        media_metadata = {"notify": t.notify_delivery}
        if thread_id:
            media_metadata["thread_id"] = thread_id

    # Relay egress needs metadata.scope_id (fail-closed tenant guard; scope cache is COLD after a
    # restart; router stamps HOME only). Origin targets only: a wrong fan-out scope is worse than
    # none.
    if t.origin_target and t.origin.get("scope_id"):
        route_metadata.setdefault("scope_id", str(t.origin["scope_id"]))
        media_metadata.setdefault("scope_id", str(t.origin["scope_id"]))
    return route_thread_id, route_metadata, media_metadata


def _live_send_text(
    t: _TargetDelivery,
    text_to_send: str,
    route_thread_id: Optional[str],
    route_metadata: dict,
    *,
    target_errors: list,
    delivery_errors: list,
    unverified_targets: list,
) -> tuple[bool, bool, Any]:
    """Schedule the text send on the gateway loop; returns ``(adapter_ok, timed_out, message_id)``.
    Re-raises a real send error so the caller falls through to standalone."""
    from agent.async_utils import safe_schedule_threadsafe
    from gateway.delivery import DeliveryRouter, DeliveryTarget

    job = t.job
    router = DeliveryRouter(t.config, t.target_adapters)
    route_target = DeliveryTarget(
        platform=t.platform,
        chat_id=str(t.chat_id),
        thread_id=route_thread_id,
        is_explicit=True,
    )
    # Thread routing goes via the target, not a bare metadata "thread_id": the router only applies
    # its Telegram DM-topic detection when thread_id/message_thread_id are absent from metadata.
    future = safe_schedule_threadsafe(
        router._deliver_to_platform(route_target, text_to_send, route_metadata),
        t.loop,
    )
    if future is None:
        target_errors.append("live adapter event loop scheduling failed")
        return False, False, None
    try:
        send_result = future.result(timeout=60)
    except TimeoutError:
        # Slow confirmation != failure; future.cancel() disambiguates. False -> already in flight,
        # cannot be un-sent, standalone resend would DUPLICATE: assume delivered. True -> never
        # started (loop wedged): MUST fall through to standalone or it is silently dropped.
        if future.cancel():
            msg = (
                f"live adapter send to {t.where} "
                "timed out before the coroutine was dispatched"
            )
            logger.warning("Job '%s': %s, falling back to standalone", job["id"], msg)
            target_errors.append(msg)
            return False, False, None
        logger.warning(
            "Job '%s': live adapter send to %s:%s timed out "
            "after 60s; already dispatched (in flight), "
            "assuming delivered (skipping standalone fallback "
            "to avoid duplicate)",
            job["id"], t.platform_name, t.chat_id,
        )
        return True, True, None
    except Exception as ex:
        # Real send error (not a slow confirmation): fall through to standalone.
        target_errors.append(f"live adapter send failed: {ex}")
        raise

    # _deliver_to_platform returns a SendResult, or a plain dict {"success": True, "delivered":
    # False, ...} when the silence-narration filter drops the message.
    if isinstance(send_result, dict):
        send_raw_response = send_result.get("raw_response")
        delivered_message_id = send_result.get("message_id")
    else:
        send_raw_response = getattr(send_result, "raw_response", None)
        delivered_message_id = getattr(send_result, "message_id", None)
    _evidence_gap: list = []
    send_success = _confirm_adapter_delivery(send_result, job["id"], _evidence_gap)
    if send_success and _evidence_gap:
        unverified_targets.append(t.where)

    if not send_success:
        if isinstance(send_result, dict):
            # A filtered drop carries no "error" — name the filter instead of reporting "unknown".
            err = send_result.get("error") or send_result.get("filtered") or "unknown"
            shape = "dict"
        elif send_result is not None:
            err = getattr(send_result, "error", None)
            shape = type(send_result).__name__
        else:
            err = "no response from adapter"
            shape = "None"
        msg = f"live adapter send to {t.where} returned unconfirmed result ({shape}, error={err})"
        _warn_live_lane_failure(job, msg, t.is_relay)
        target_errors.append(msg)
        return False, False, None
    if send_raw_response and t.thread_id and send_raw_response.get("thread_fallback"):
        requested_thread_id = send_raw_response.get("requested_thread_id") or t.thread_id
        _note_target_error(
            job,
            f"configured thread_id {requested_thread_id} for "
            f"{t.where} was not found; delivered without thread_id",
            delivery_errors,
        )
    return True, False, delivered_message_id


def _live_send_media(t: _TargetDelivery, media_metadata: dict, media_files: list, delivery_errors: list) -> None:
    """Send extracted media as native attachments with the same routing as the text send."""
    routed_media_metadata = dict(media_metadata or {})
    if t.is_relay:
        routed_media_metadata["_relay_logical_platform"] = t.platform.value
        logical_home = t.config.get_home_channel(t.platform)
        if logical_home is not None and logical_home.chat_id == t.chat_id:
            if logical_home.user_id:
                routed_media_metadata["user_id"] = logical_home.user_id
            if logical_home.scope_id:
                routed_media_metadata["scope_id"] = logical_home.scope_id
    _media_errors = _send_media_via_adapter(
        t.runtime_adapter,
        t.chat_id,
        media_files,
        routed_media_metadata or None,
        t.loop,
        t.job,
        platform=t.platform,
    )
    # Surface per-file failures into run status: text delivered but attachment lost is not ok.
    for _me in _media_errors:
        delivery_errors.append(f"{_me} (target {t.where})")


def _seed_live_delivery_sessions(t: _TargetDelivery, delivered_message_id) -> None:
    """After a confirmed live send, seed continuation session(s) and run the generic mirror.
    Thread seeding is deferred here so open-succeeds/deliver-fails never seeds an unseen brief."""
    job = t.job
    origin = t.origin
    thread_seeded = False
    inchannel_seeded = False
    if t.opened_thread_id:
        _seed_cron_thread_session(
            job, t.runtime_adapter, t.platform_name, t.chat_id,
            t.opened_thread_id, t.mirror_text,
            chat_name=origin.get("chat_name"),
            is_dm=t.is_dm_target,
            scope_id=origin.get("scope_id"),
        )
        thread_seeded = True
    # in_channel: CREATE + seed the flat session (the mirror only APPENDS to an existing one). Same
    # `inchannel_continuable` gate as the flatten in _deliver_result (must not drift). Origin
    # seed without mirror opt-in; others only via _inchannel_seed_allowed (user-less seed = orphan).
    if t.in_channel_surface and t.inchannel_continuable and not thread_seeded:
        inchannel_seeded = _seed_cron_channel_session(
            job, t.runtime_adapter, t.platform_name, t.chat_id,
            t.mirror_text, is_dm=t.is_dm_target,
            user_id=t.origin_user_id,
            chat_name=origin.get("chat_name"),
            scope_id=origin.get("scope_id"),
        )
        if not inchannel_seeded:
            logger.warning(
                "Job '%s': in_channel seed did NOT land on %s:%s "
                "— a plain reply will not see this brief",
                job["id"], t.platform_name, t.chat_id,
            )
        # Companion THREAD seed: a reply in the brief's own thread keys to (chat, thread=<ts>),
        # which the flat seed never touches. Seed it too so BOTH reply surfaces continue the job.
        if delivered_message_id:
            _seed_cron_thread_session(
                job, t.runtime_adapter, t.platform_name, t.chat_id,
                str(delivered_message_id), t.mirror_text,
                chat_name=origin.get("chat_name"),
                is_dm=t.is_dm_target,
                scope_id=origin.get("scope_id"),
            )
    elif t.in_channel_surface and not t.inchannel_continuable:
        logger.warning(
            "Job '%s': in_channel delivery to %s:%s is not a "
            "continuable target (origin=%s:%s thread=%s; not the "
            "origin conversation, and not a mirror-eligible "
            "fallback/opted-in target the seed can key) — seed "
            "skipped; the plain mirror below may still apply",
            job["id"], t.platform_name, t.chat_id,
            origin.get("platform"), origin.get("chat_id"),
            origin.get("thread_id"),
        )
    _maybe_mirror_cron_delivery(
        job, t.platform_name, t.chat_id, t.mirror_text,
        thread_id=t.thread_id, user_id=t.origin_user_id,
        enabled=t.mirror_this_target and not thread_seeded and not inchannel_seeded,
    )


def _deliver_via_live_adapter(
    t: _TargetDelivery,
    cleaned_text: str,
    media_files: list,
    *,
    target_errors: list,
    delivery_errors: list,
    unverified_targets: list,
) -> bool:
    """Deliver one target via the live gateway adapter; True once delivered. ``target_errors`` =
    this lane's soft failures (surfaced only if standalone also fails); ``delivery_errors`` =
    partial failures (media, thread fallback) that surface even on success."""
    job = t.job
    route_thread_id, route_metadata, media_metadata = _live_route_metadata(t)
    delivered = False
    try:
        # Send cleaned text (MEDIA tags stripped) through the gateway's DeliveryRouter so it gets
        # the same platform routing as live messages (Telegram's three-mode topic routing).
        text_to_send = cleaned_text.strip()
        adapter_ok = True
        timed_out = False
        delivered_message_id = None
        if not text_to_send and not media_files:
            # Fail closed so the run reports the empty payload.
            _note_target_error(
                job, f"live adapter send skipped (empty text and no media) for {t.where}", target_errors,
            )
            adapter_ok = False
        elif text_to_send:
            adapter_ok, timed_out, delivered_message_id = _live_send_text(
                t, text_to_send, route_thread_id, route_metadata,
                target_errors=target_errors,
                delivery_errors=delivery_errors,
                unverified_targets=unverified_targets,
            )

        # Media rides the same DM-topic-aware routing as text. Skipped after a confirmation
        # timeout (loop contended, text already assumed delivered) — record the drop instead.
        if adapter_ok and not timed_out and media_files:
            _live_send_media(t, media_metadata, media_files, delivery_errors)
        elif timed_out and media_files:
            _note_target_error(
                job,
                f"{len(media_files)} media attachment(s) not delivered to "
                f"{t.where} (live adapter confirmation timed out)",
                delivery_errors,
            )

        if adapter_ok:
            # Log WHERE it went: a ghost delivery in the wrong lane is otherwise indistinguishable.
            logger.info(
                "Job '%s': delivered to %s:%s via live adapter thread=%s message_id=%s",
                job["id"], t.platform_name, t.chat_id,
                route_thread_id if route_thread_id is not None else "-",
                delivered_message_id if delivered_message_id is not None else "-",
            )
            delivered = True
            _seed_live_delivery_sessions(t, delivered_message_id)
    except Exception as e:
        err_msg = f"live adapter delivery to {t.where} failed: {e}"
        if not any(err_msg in err for err in target_errors):
            target_errors.append(err_msg)
        _warn_live_lane_failure(job, err_msg, t.is_relay)
    return delivered


def _standalone_send(t: _TargetDelivery, content: str, media_files: list) -> tuple[Any, Optional[str]]:
    """Run the standalone sender for one target: ``(result, None)`` or ``(None, error)`` (already
    logged — WARNING for a shutdown race, ERROR with traceback otherwise)."""
    from tools.send_message_tool import _send_to_platform

    job = t.job
    shutdown_msg = f"delivery to {t.where} skipped — interpreter is shutting down"

    def _send():
        return _send_to_platform(
            t.platform, t.pconfig, t.chat_id, content, thread_id=t.thread_id, media_files=media_files,
        )

    def _failed(e) -> tuple[None, str]:
        msg = f"delivery to {t.where} failed: {e}"
        logger.error("Job '%s': %s", job["id"], msg, exc_info=True)
        return None, msg

    # Interpreter finalizing (SIGTERM/restart/OOM): asyncio.run and a fresh ThreadPoolExecutor both
    # raise "cannot schedule new futures after interpreter shutdown" — warn, not ERROR traceback.
    if _interpreter_shutting_down():
        logger.warning("Job '%s': %s", job["id"], shutdown_msg)
        return None, shutdown_msg
    # The live lane failed closed on an empty payload; standalone senders don't (Telegram returns
    # success=True for empty content WITHOUT an API call) — a phantom delivery would result.
    if not content.strip() and not media_files:
        msg = f"standalone send skipped (empty text and no media) for {t.where}"
        logger.warning("Job '%s': %s", job["id"], msg)
        return None, msg
    coro = _send()
    try:
        return asyncio.run(coro), None
    except RuntimeError as run_err:
        # asyncio.run() refuses inside a running loop; close the unstarted coro, retry in a thread.
        coro.close()
        if _interpreter_shutting_down(run_err):
            logger.warning("Job '%s': %s", job["id"], shutdown_msg)
            return None, shutdown_msg
        # The fallback can itself raise (SMTP, result timeout); catch it or remaining targets skip.
        try:
            pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
            try:
                # A fresh thread does NOT inherit the profile ContextVars (home override + secret
                # scope); run in the active context or the sender reads the default bot token.
                _fallback_context = contextvars.copy_context()
                future = pool.submit(_fallback_context.run, asyncio.run, _send())
                return future.result(timeout=30), None
            finally:
                pool.shutdown(wait=False)
        except Exception as e:
            if _interpreter_shutting_down(e):
                logger.warning("Job '%s': %s", job["id"], shutdown_msg)
                return None, shutdown_msg
            return _failed(e)
    except Exception as e:
        return _failed(e)


def _deliver_standalone(
    t: _TargetDelivery, content: str, media_files: list, target_errors: list, delivery_errors: list,
) -> None:
    """Standalone fallback for a target the live lane did not deliver."""
    job = t.job
    if t.is_relay:
        # Relay owns the destination and credential; a native retry could duplicate — fail closed.
        if not target_errors:
            target_errors.append(f"relay delivery to {t.where} failed")
        delivery_errors.extend(target_errors)
        return
    result, err = _standalone_send(t, content, media_files)
    if err is None and result and result.get("error"):
        # Not inside an except block — the error comes from the result dict, no traceback.
        err = f"delivery error: {result['error']} (target {t.where})"
        logger.error("Job '%s': %s", job["id"], err)
    if err is not None:
        target_errors.append(err)
        delivery_errors.extend(target_errors)
        return

    # Standalone senders report per-file attachment failures in ``warnings`` while returning
    # success; surface them so a vanished attachment doesn't mark the run ok.
    _sender_warnings = (result.get("warnings") if isinstance(result, dict) else None) or []
    for _w in _sender_warnings:
        msg = f"delivery warning: {_w} (target {t.where})"
        logger.error("Job '%s': %s", job["id"], msg)
        delivery_errors.append(msg)

    logger.info("Job '%s': delivered to %s:%s", job["id"], t.platform_name, t.chat_id)
    # Thread seeding only happens on the live lane, so no thread_seeded gate applies here.
    _maybe_mirror_cron_delivery(
        job, t.platform_name, t.chat_id, t.mirror_text,
        thread_id=t.thread_id, user_id=t.origin_user_id,
        enabled=t.mirror_this_target,
    )


def _deliver_result(
    job: dict, content: str, adapters=None, loop=None, *, for_failure: bool = False
) -> Optional[str]:
    """Deliver job output to the configured target(s). With ``adapters``/``loop`` (gateway
    running) the live adapter is tried first (E2EE rooms can't use the standalone HTTP path), then
    standalone fallback. ``for_failure=True`` routes failure-category notices through the job's
    ``failure_deliver`` override when present (NS-788). Returns None on success or an error string."""
    targets = _resolve_delivery_targets(job, for_failure=for_failure)
    if not targets:
        deliver_value = _normalize_deliver_value(
            _delivery_lane_value(job, for_failure=for_failure)
        )
        if deliver_value == "local":
            return None  # local-only jobs don't deliver — not a failure
        # deliver=origin with no origin and no home channels: treat as local, not an error — CLI
        # jobs never capture an origin and would emit a spurious error every run.
        if deliver_value == "origin":
            logger.info(
                "Job '%s': deliver=origin but no origin or home channels — "
                "skipping delivery (output saved in last_output)",
                job.get("name", job.get("id", "?")),
            )
            return None
        msg = f"no delivery target resolved for deliver={deliver_value}"
        logger.warning("Job '%s': %s", job["id"], msg)
        return msg

    from gateway.config import load_gateway_config, Platform

    # Wrap with header/footer unless cron.wrap_response: false.
    wrap_response = True
    user_cfg = None
    with contextlib.suppress(Exception):
        user_cfg = load_config()
        wrap_response = user_cfg.get("cron", {}).get("wrap_response", True)

    # Mark live sends FINAL so the platform pushes them (Telegram "important" mode mutes otherwise).
    notify_delivery = _cron_delivery_notify_enabled(user_cfg)
    # Targets acked with NO evidence (bare SendResult(success=True) — Slack/Matrix/Mattermost);
    # persisted as ``last_delivery_unverified`` so `hermes cron list` shows it.
    unverified_targets: list = []

    if wrap_response:
        task_name = job.get("name", job["id"])
        job_id = job.get("id", "")
        delivery_content = (
            f"Cronjob Response: {task_name}\n"
            f"(job_id: {job_id})\n"
            f"-------------\n\n"
            f"{content}\n\n"
            f"To stop or manage this job, send me a new message (e.g. \"stop reminder {task_name}\")."
        )
    else:
        delivery_content = content

    from gateway.platforms.base import BasePlatformAdapter

    # Bridge media-policy config into the env vars the path validator reads. The gateway does this
    # at boot; standalone runs (`hermes cron run`) did not, silently dropping files. Idempotent.
    from gateway.media_policy import apply_media_policy_env

    apply_media_policy_env(user_cfg)

    media_files, cleaned_delivery_content = BasePlatformAdapter.extract_media(delivery_content)
    requested_media = [(str(p), v) for p, v in media_files]
    media_files = BasePlatformAdapter.filter_media_delivery_paths(media_files)
    # Policy-dropped attachments will never be sent on ANY lane — record them in run status.
    _policy_dropped = len(requested_media) - len(media_files)
    policy_drop_errors = (
        [
            f"{_policy_dropped} media attachment(s) dropped by media path "
            "policy (missing file, denied prefix, or strict-mode miss); "
            "see gateway.strict / media_delivery_allow_dirs in config.yaml"
        ]
        if _policy_dropped > 0
        else []
    )

    # Resolve the mirror gate ONCE (default off): successful deliveries are appended to the target
    # chat's session transcript. Mirror the CLEAN, unwrapped output (not the header/footer).
    try:
        mirror_enabled = _cron_mirror_delivery_enabled(job, user_cfg)
    except Exception:
        mirror_enabled = False
    # Independent of the mirror knob: continuable surfaces (in_channel) must seed even when
    # attach_to_session=false and cron.mirror_delivery=false, else the seed gets "" and fails.
    _, mirror_text = BasePlatformAdapter.extract_media(content)
    mirror_text = (mirror_text or "").strip()

    try:
        config = load_gateway_config()
    except Exception as e:
        msg = f"failed to load gateway config: {e}"
        logger.error("Job '%s': %s", job["id"], msg)
        return msg

    delivery_errors = []

    for target in targets:
        platform_name = target["platform"]
        chat_id = target["chat_id"]
        thread_id = target.get("thread_id")

        # bot-chat targets bypass gateway adapters: output becomes an inbound turn in the target
        # profile's Bot Chat via the chat CLI lane. Must precede the Platform enum, which lacks it.
        if platform_name == BOT_CHAT_PLATFORM:
            bot_chat_error = _deliver_to_bot_chat(job, content, chat_id)
            if bot_chat_error:
                delivery_errors.append(bot_chat_error)
            continue

        origin = _resolve_origin(job) or {}
        origin_thread = origin.get("thread_id")
        if origin_thread and not thread_id:
            logger.warning(
                "Job '%s': origin has thread_id=%s but delivery target lost it "
                "(deliver=%s, target=%s)",
                job["id"], origin_thread, job.get("deliver", "local"), target,
            )
        elif thread_id:
            logger.debug(
                "Job '%s': delivering to %s:%s thread_id=%s",
                job["id"], platform_name, chat_id, thread_id,
            )

        # Mirror: origin, home FALLBACK for origin-less deliver=origin, or attach_to_session opt-in.
        origin_target = _target_matches_origin(origin, platform_name, chat_id, thread_id)
        mirror_this_target = mirror_enabled and _target_mirror_eligible(
            job, target, global_mirror=mirror_enabled, origin_match=origin_target,
        )
        # Resolved for ANY origin match (not just mirror-enabled): the in_channel seed needs it too.
        origin_user_id = origin.get("user_id") if origin_target else None

        # DM shape for BOTH the flatten gate and seed chat_type (Slack DM ids start with "D").
        origin_chat_type = str(origin.get("chat_type") or "").lower()
        is_dm_target = origin_chat_type == "dm" or (
            not origin_chat_type and str(chat_id).startswith("D")
        )

        # in_channel gate shared by thread-flatten and flat seed — they MUST match or brief and
        # session land in different places. Origin qualifies unconditionally; others only when the
        # seed can create a resolvable session (_inchannel_seed_allowed).
        inchannel_continuable = origin_target or (
            mirror_this_target
            and _inchannel_seed_allowed(is_dm=is_dm_target, user_id=origin_user_id)
        )

        # Plugin platform names create dynamic members via Platform._missing_().
        try:
            platform = Platform(platform_name.lower())
        except (ValueError, KeyError):
            _note_target_error(job, f"unknown platform '{platform_name}'", delivery_errors)
            continue

        resolved, resolve_err = _resolve_target_transport(
            job, platform, platform_name, target, adapters, config,
        )
        if resolved is None:
            _note_target_error(job, resolve_err, delivery_errors)
            continue
        transport, pconfig, runtime_adapter, target_adapters = resolved

        # Live send needs a RUNNING loop, not just an adapter. Computed ONCE so the in_channel
        # thread_id clear below stays in lockstep with the seed (standalone cannot seed flat).
        live_adapter_ready = (
            runtime_adapter is not None
            and loop is not None
            and getattr(loop, "is_running", lambda: False)()
        )
        target_errors: list = []

        # Continuable surface (D1/D2/D6) from platform config ``extra``; default "thread".
        # ``in_channel`` delivers FLAT so a plain channel reply continues via the shared session
        # ``(platform, chat_id, None)``. Unsupported adapters fail SAFE to thread.
        in_channel_surface = _resolve_cron_surface_mode(pconfig, platform_name) == "in_channel"
        if (
            in_channel_surface
            and runtime_adapter is not None
            and not _inchannel_surface_supported(runtime_adapter, platform_name)
        ):
            logger.debug(
                "Job '%s': cron_continuable_surface=in_channel not supported on "
                "%s, using thread",
                job.get("id", "?"), platform_name,
            )
            in_channel_surface = False

        if in_channel_surface and inchannel_continuable and live_adapter_ready:
            # Force flat (D2): an inherited thread_id would never match the flat seed (None). Gated
            # on `inchannel_continuable` (SAME gate as the seed) AND `live_adapter_ready` (fallback
            # never seeds). Stay AFTER mirror_this_target/origin_user_id (need ORIGINAL thread_id).
            thread_id = None

        # Thread-preferred continuable cron: open a DEDICATED thread; its session is seeded after a
        # successful send. DM-only platforms return None → mirror the origin DM. in_channel SKIPS
        # this: it posts flat and _seed_cron_channel_session CREATES the session.
        opened_thread_id: Optional[str] = None
        if (
            mirror_this_target
            and not in_channel_surface
            and runtime_adapter is not None
            and loop is not None
            and not thread_id  # never override an explicit origin thread/topic
        ):
            opened_thread_id = _open_continuable_cron_thread(
                job, runtime_adapter, chat_id, loop,
            ) or None
            if opened_thread_id:
                thread_id = opened_thread_id

        t = _TargetDelivery(
            job=job,
            platform=platform,
            platform_name=platform_name,
            chat_id=chat_id,
            thread_id=thread_id,
            transport=transport,
            pconfig=pconfig,
            runtime_adapter=runtime_adapter,
            target_adapters=target_adapters,
            config=config,
            loop=loop,
            notify_delivery=notify_delivery,
            origin=origin,
            origin_target=origin_target,
            origin_user_id=origin_user_id,
            is_dm_target=is_dm_target,
            mirror_text=mirror_text,
            mirror_this_target=mirror_this_target,
            in_channel_surface=in_channel_surface,
            inchannel_continuable=inchannel_continuable,
            opened_thread_id=opened_thread_id,
        )
        delivered = live_adapter_ready and _deliver_via_live_adapter(
            t, cleaned_delivery_content, media_files,
            target_errors=target_errors,
            delivery_errors=delivery_errors,
            unverified_targets=unverified_targets,
        )
        if not delivered:
            _deliver_standalone(
                t, cleaned_delivery_content, media_files, target_errors, delivery_errors,
            )

    if policy_drop_errors:
        # Filter-time drops apply to every target; report them once.
        delivery_errors.extend(policy_drop_errors)
    _record_delivery_verification(job, unverified_targets)
    if delivery_errors:
        return "; ".join(delivery_errors)
    return None


_DEFAULT_SCRIPT_TIMEOUT = 3600  # seconds (1 hour)
# Backward-compatible module override used by tests and emergency monkeypatches.
_SCRIPT_TIMEOUT = _DEFAULT_SCRIPT_TIMEOUT
_RUN_CLAIM_HEARTBEAT_SECONDS = 60.0
_FIRE_CLAIM_HEARTBEAT_GRACE_SECONDS = _RUN_CLAIM_HEARTBEAT_SECONDS * 3


def _get_script_timeout() -> int:
    """Resolve cron pre-run script timeout from module/env/config with a safe default."""
    if _SCRIPT_TIMEOUT != _DEFAULT_SCRIPT_TIMEOUT:
        try:
            timeout = int(float(_SCRIPT_TIMEOUT))
            if timeout > 0:
                return timeout
        except Exception:
            logger.warning("Invalid patched _SCRIPT_TIMEOUT=%r; using env/config/default", _SCRIPT_TIMEOUT)

    env_value = os.getenv("HERMES_CRON_SCRIPT_TIMEOUT", "").strip()
    if env_value:
        try:
            timeout = int(float(env_value))
            if timeout > 0:
                return timeout
        except Exception:
            logger.warning("Invalid HERMES_CRON_SCRIPT_TIMEOUT=%r; using config/default", env_value)

    try:
        cfg = load_config() or {}
        cron_cfg = cfg.get("cron", {}) if isinstance(cfg, dict) else {}
        configured = cron_cfg.get("script_timeout_seconds")
        if configured is not None:
            timeout = int(float(configured))
            if timeout > 0:
                return timeout
    except Exception as exc:
        logger.debug("Failed to load cron script timeout from config: %s", exc)

    return _DEFAULT_SCRIPT_TIMEOUT


_DEFAULT_MEDIA_SEND_TIMEOUT = 300


def _get_media_send_timeout() -> int:
    """Per-attachment media-send timeout: HERMES_CRON_MEDIA_SEND_TIMEOUT env, then
    ``cron.media_send_timeout_seconds``, then 300s (long TTS audio can exceed a 30s window)."""
    env_value = os.getenv("HERMES_CRON_MEDIA_SEND_TIMEOUT", "").strip()
    if env_value:
        try:
            timeout = int(float(env_value))
            if timeout > 0:
                return timeout
        except Exception:
            logger.warning(
                "Invalid HERMES_CRON_MEDIA_SEND_TIMEOUT=%r; using config/default",
                env_value,
            )

    try:
        cfg = load_config() or {}
        cron_cfg = cfg.get("cron", {}) if isinstance(cfg, dict) else {}
        configured = cron_cfg.get("media_send_timeout_seconds")
        if configured is not None:
            timeout = int(float(configured))
            if timeout > 0:
                return timeout
    except Exception as exc:
        logger.debug("Failed to load cron media-send timeout from config: %s", exc)

    return _DEFAULT_MEDIA_SEND_TIMEOUT


def _get_session_db_timeout() -> float:
    """Bound on run_job's SessionDB init: HERMES_CRON_SESSION_DB_TIMEOUT env, then
    ``cron.session_db_timeout_seconds`` (in DEFAULT_CONFIG), then 10s. Unlike sibling timeouts,
    0 is meaningful (unlimited, debugging opt-in), so values pass through untouched."""
    env_value = os.getenv("HERMES_CRON_SESSION_DB_TIMEOUT", "").strip()
    if env_value:
        try:
            return float(env_value)
        except (ValueError, TypeError):
            logger.warning(
                "Invalid HERMES_CRON_SESSION_DB_TIMEOUT=%r; using config/default",
                env_value,
            )

    try:
        cfg = load_config() or {}
        cron_cfg = cfg.get("cron", {}) if isinstance(cfg, dict) else {}
        configured = cron_cfg.get("session_db_timeout_seconds")
        if configured is not None:
            return float(configured)
    except Exception as exc:
        logger.debug("Failed to load cron.session_db_timeout_seconds from config: %s", exc)

    return 10.0


def _read_windows_pyvenv_cfg(venv_dir: Path) -> dict[str, str]:
    cfg_path = venv_dir / "pyvenv.cfg"
    try:
        lines = cfg_path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return {}

    parsed: dict[str, str] = {}
    for raw in lines:
        if "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        parsed[key.strip().lower()] = value.strip()
    return parsed


def _windows_cron_python_invocation(python_exe: str) -> tuple[str, dict[str, str]]:
    """Hidden, output-capable Python invocation for Windows cron scripts. ``pythonw.exe`` loses
    captured output; uv venv launchers can re-exec the base console python and flash a window even
    with CREATE_NO_WINDOW, so run the base python directly with venv paths overlaid in env."""
    if sys.platform != "win32":
        return python_exe, {}

    interpreter = Path(python_exe)
    venv_dir = interpreter.parent.parent
    env_overlay: dict[str, str] = {}

    if interpreter.name.lower() == "pythonw.exe":
        sibling = interpreter.with_name("python.exe")
        if sibling.exists():
            interpreter = sibling

    cfg = _read_windows_pyvenv_cfg(venv_dir)
    home = cfg.get("home", "")
    site_packages = venv_dir / "Lib" / "site-packages"
    if "uv" in cfg and home:
        base_python = Path(home) / "python.exe"
        if base_python.exists() and site_packages.exists():
            interpreter = base_python
            env_overlay["VIRTUAL_ENV"] = str(venv_dir)
            pythonpath_entries = [str(Path(__file__).resolve().parents[1]), str(site_packages)]
            existing_pythonpath = os.environ.get("PYTHONPATH", "")
            if existing_pythonpath:
                pythonpath_entries.append(existing_pythonpath)
            env_overlay["PYTHONPATH"] = os.pathsep.join(pythonpath_entries)

    return str(interpreter), env_overlay


def _terminate_cron_script_process(proc: subprocess.Popen) -> None:
    """Best-effort hard stop of a cron script and every child it spawned."""
    if proc.poll() is not None:
        return
    if sys.platform == "win32":
        try:
            subprocess.run(
                ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
                capture_output=True,
                timeout=10,
                creationflags=windows_hide_flags(),
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()
    else:
        try:
            process_group: Optional[int] = os.getpgid(proc.pid)
        except (ProcessLookupError, OSError):
            process_group = None
        if process_group is not None:
            try:
                os.killpg(process_group, signal.SIGTERM)  # windows-footgun: ok — POSIX-only branch (win32 handled above)
            except (ProcessLookupError, PermissionError, OSError):
                process_group = None
            if process_group is not None:
                with contextlib.suppress(subprocess.TimeoutExpired):
                    proc.wait(timeout=1.0)
                # Escalate if ANY group member survived TERM: a survivor holds the pipe write ends
                # open and the caller's communicate() would block on EOF forever.
                try:
                    os.killpg(process_group, 0)  # windows-footgun: ok — POSIX-only branch
                except (ProcessLookupError, OSError):
                    process_group = None
                if process_group is not None:
                    with contextlib.suppress((ProcessLookupError, PermissionError, OSError)):
                        os.killpg(process_group, getattr(signal, "SIGKILL", signal.SIGTERM))
    try:
        proc.wait(timeout=1.0)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait(timeout=1.0)


def _terminate_cron_script_tree(proc: subprocess.Popen) -> None:
    """Terminate a script tree, then fall back to the local process-group path."""
    if proc.poll() is not None:
        # Already reaped: kill_process_tree would log a spurious "no signal" warning.
        return
    pid = getattr(proc, "pid", None)
    if not isinstance(pid, int) or pid <= 0:
        logger.warning(
            "Cron script tree-kill received invalid pid %r; "
            "falling back to process-group termination",
            pid,
        )
        _terminate_cron_script_process(proc)
        return
    try:
        # Function-local (monkeypatchable); separate try so an import problem is not
        # misreported as a kill failure.
        from agent.deadline import kill_process_tree
    except Exception:
        logger.warning(
            "agent.deadline.kill_process_tree unavailable; "
            "falling back to process-group termination",
            exc_info=True,
        )
        _terminate_cron_script_process(proc)
        return
    try:
        if kill_process_tree(pid):
            return
        logger.warning(
            "Cron script tree-kill reported no signal for pid %s; "
            "falling back to process-group termination",
            pid,
        )
    except Exception:
        logger.warning(
            "Cron script tree-kill failed for pid %s; "
            "falling back to process-group termination",
            pid,
            exc_info=True,
        )
    _terminate_cron_script_process(proc)


def _drain_script_pipes(proc: subprocess.Popen) -> None:
    """Reap a terminated script without blocking forever: a surviving descendant can hold the pipe
    write ends open, so bound the drain and abandon the pipes (output is not needed)."""
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.communicate(timeout=5.0)
        return
    with contextlib.suppress(OSError):
        proc.kill()
    for stream in (proc.stdout, proc.stderr):
        with contextlib.suppress(OSError):
            if stream is not None:
                stream.close()
    with contextlib.suppress(subprocess.TimeoutExpired):
        proc.wait(timeout=5.0)


def _windows_cron_bootstrap_argv(
    python_exe: str,
    env_overlay: dict[str, str],
    script_path: str,
) -> list[str]:
    """Bootstrap a cron script under the base interpreter with ``.pth`` support.

    Overlay mode runs base ``python.exe`` (avoids the launcher flashing a console window) with the
    venv on ``PYTHONPATH`` — but ``.pth`` files are only processed by ``site.addsitedir()``, so
    editable installs would be invisible. Bootstrap via addsitedir + ``runpy.run_path`` (keeps
    ``__file__`` and ``sys.path[0]`` semantics); plain invocation if the venv is unresolvable.
    """
    site_packages = Path(env_overlay.get("VIRTUAL_ENV", "")) / "Lib" / "site-packages"
    if not site_packages.is_dir():
        # Warn: silent fallback would make "editable installs invisible" undiagnosable.
        logger.warning(
            "Windows cron script: venv site-packages %s not found; running "
            "without .pth processing (editable installs may be unimportable)",
            site_packages,
        )
        return [python_exe, script_path]
    bootstrap = (
        "import os, runpy, site, sys;"
        f"site.addsitedir({str(site_packages)!r});"
        "script = sys.argv[1];"
        "sys.argv = [script] + sys.argv[2:];"
        "sys.path.insert(0, os.path.dirname(os.path.abspath(script)));"
        "runpy.run_path(script, run_name='__main__')"
    )
    return [python_exe, "-c", bootstrap, script_path]


def _run_job_script(
    script_path: str,
    workdir: Optional[str] = None,
    cancel_event: Optional[_CancelEventLike] = None,
) -> tuple[bool, str]:
    """Execute a cron job's script and return ``(success, output)``; on failure *output* is the
    error message for the LLM to report.

    Scripts MUST resolve inside HERMES_HOME/scripts/ (relative, absolute and ``~`` paths are all
    validated — path traversal / absolute-path injection). Interpreter by extension:
    ``.sh``/``.bash`` → bash, else ``sys.executable``. Env goes through ``build_subprocess_env``
    (SECURITY.md §2.3).
    ``workdir`` sets the subprocess cwd only; the Python process cwd is NEVER mutated (an
    ``os.chdir()`` would leak into concurrent gateway sessions).
    """
    scripts_dir = _get_hermes_home() / "scripts"
    _ensure_cron_dir(scripts_dir)
    scripts_dir_resolved = scripts_dir.resolve()

    # Same contract as cron.lifecycle_guard._expand_candidate_path. Reject NUL eagerly: on Windows
    # Path ops raise ValueError *after* expanduser so the try below would not catch it. str() first
    # so the guard itself cannot raise on a non-str script_path.
    if "\x00" in str(script_path):
        return False, f"Blocked: script path contains a NUL byte: {script_path!r}"

    try:
        raw = Path(script_path).expanduser()
    except (ValueError, RuntimeError, OSError):
        # RuntimeError: unexpandable ``~`` (no resolvable HOME).
        return False, f"Blocked: script path is not a valid filesystem path: {script_path!r}"
    path = raw.resolve() if raw.is_absolute() else (scripts_dir / raw).resolve()

    # Traversal / absolute-path / symlink escape guard — MUST stay inside HERMES_HOME/scripts/.
    try:
        path.relative_to(scripts_dir_resolved)
    except ValueError:
        return False, (
            f"Blocked: script path resolves outside the scripts directory "
            f"({scripts_dir_resolved}): {script_path!r}"
        )

    if not path.exists():
        return False, f"Script not found: {path}"
    if not path.is_file():
        return False, f"Script path is not a file: {path}"

    script_timeout = _get_script_timeout()

    # Interpreter by extension; the shebang is deliberately NOT honoured (small, auditable surface).
    suffix = path.suffix.lower()
    if suffix in {".sh", ".bash"}:
        # which() finds Git Bash on Windows; None there → clear error instead of a "[WinError 2]".
        _bash = shutil.which("bash") or ("/bin/bash" if os.path.isfile("/bin/bash") else None)
        if _bash is None:
            return False, (
                f"Cannot run .sh/.bash script {path.name!r}: bash not found on PATH. "
                "On Windows, install Git for Windows (which ships Git Bash) "
                "or rewrite the script as Python (.py)."
        )
        argv = [_bash, str(path)]
        env_overlay: dict[str, str] = {}
    else:
        python_exe, env_overlay = _windows_cron_python_invocation(sys.executable)
        if env_overlay:
            # Windows uv-venv overlay: needs the .pth bootstrap for editable installs.
            argv = _windows_cron_bootstrap_argv(python_exe, env_overlay, str(path))
        else:
            argv = [python_exe, str(path)]

    try:
        from tools.environments.local import build_subprocess_env

        popen_kwargs: dict[str, Any] = {"start_new_session": True}
        if sys.platform == "win32":
            popen_kwargs = {
                "creationflags": windows_hide_flags()
                | getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0),
                "encoding": "utf-8",
                "errors": "replace",
            }
        env = build_subprocess_env()
        env.update(env_overlay)
        # Subprocess cwd only (default: scripts-dir parent). NEVER os.chdir() the process.
        _script_cwd = workdir or str(path.parent)
        proc = subprocess.Popen(
            argv,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            cwd=_script_cwd,
            env=env,
            **popen_kwargs,
        )
        deadline = time.monotonic() + script_timeout
        while True:
            if cancel_event is not None and cancel_event.is_set():
                # Tree-kill here too: a cancelled fire must not orphan own-session grandchildren.
                _terminate_cron_script_tree(proc)
                _drain_script_pipes(proc)
                return False, "Script cancelled because cron fire ownership was lost"
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                # Timeout must leave ZERO descendants: killpg misses setsid grandchildren
                # (watchdogs, backgrounded shell jobs); kill_process_tree snapshots descendants
                # BEFORE signalling.
                _terminate_cron_script_tree(proc)
                _drain_script_pipes(proc)
                return False, f"Script timed out after {script_timeout}s: {path}"
            try:
                stdout_raw, stderr_raw = proc.communicate(timeout=min(0.1, remaining))
                break
            except subprocess.TimeoutExpired:
                continue

        stdout = (stdout_raw or "").strip()
        stderr = (stderr_raw or "").strip()

        # Redact secrets before ANY return path.
        try:
            from agent.redact import redact_sensitive_text
            stdout = redact_sensitive_text(stdout)
            stderr = redact_sensitive_text(stderr)
        except Exception as e:
            logger.warning("Failed to redact sensitive text from output: %s", e)
            stdout = "[REDACTED - redaction failed]"
            stderr = "[REDACTED - redaction failed]"

        if proc.returncode != 0:
            parts = [f"Script exited with code {proc.returncode}"]
            if stderr:
                parts.append(f"stderr:\n{stderr}")
            if stdout:
                parts.append(f"stdout:\n{stdout}")
            return False, "\n".join(parts)

        return True, stdout

    except Exception as exc:
        return False, f"Script execution failed: {exc}"


def _start_heartbeat_thread(loop_fn, name: str, fail_log) -> Optional[threading.Thread]:
    """Start ``loop_fn`` on a daemon thread inside a copy of the current context (multiplexed
    profile ContextVars). On failure calls ``fail_log()`` inside the except (traceback intact) and
    returns None."""
    thread = threading.Thread(
        target=contextvars.copy_context().run, args=(loop_fn,), name=name, daemon=True,
    )
    try:
        thread.start()
    except Exception:
        fail_log()
        return None
    return thread


def _run_job_script_with_claim_heartbeat(
    job: dict,
    script_path: str,
    workdir: Optional[str] = None,
    cancel_event: Optional[_CancelEventLike] = None,
) -> tuple[bool, str]:
    """Run a cron script while heartbeating its owned one-shot claim.

    A long script can outlive the stale-claim TTL; without a heartbeat another scheduler would
    re-dispatch the one-shot. Recurring/unclaimed runs have no durable claim → no thread. The owner
    is captured from the dispatched job, never re-read, so a stale runner cannot extend a
    replacement owner's claim.
    """
    schedule = job.get("schedule")
    claim = job.get("run_claim")
    owner = str(claim.get("by") or "") if isinstance(claim, dict) else ""
    if not (
        isinstance(schedule, dict)
        and schedule.get("kind") == "once"
        and owner
    ):
        return _run_job_script(script_path, workdir=workdir, cancel_event=cancel_event)

    job_id = str(job.get("id") or "")
    stop = threading.Event()

    def _heartbeat_loop() -> None:
        while not stop.wait(_RUN_CLAIM_HEARTBEAT_SECONDS):
            try:
                heartbeat_run_claim(job_id, expected_owner=owner)
            except Exception:
                logger.debug("Job '%s': script run_claim heartbeat failed", job_id, exc_info=True)

    heartbeat_thread = _start_heartbeat_thread(
        _heartbeat_loop, "cron-script-claim-heartbeat",
        lambda: logger.debug(
            "Job '%s': could not start script run_claim heartbeat", job_id, exc_info=True,
        ),
    )
    if heartbeat_thread is None:
        return _run_job_script(script_path, workdir=workdir, cancel_event=cancel_event)

    try:
        return _run_job_script(script_path, workdir=workdir, cancel_event=cancel_event)
    finally:
        stop.set()
        # Bounded join: the heartbeat may be blocked on another process's jobs-file lock.
        heartbeat_thread.join(timeout=1.0)


def _parse_wake_gate(script_output: str) -> bool:
    """Wake gate: False only if the last non-empty stdout line is JSON ``{"wakeAgent": false}``
    (agent skipped entirely — no LLM run, no delivery); anything else wakes normally."""
    if not script_output:
        return True
    stripped_lines = [line for line in script_output.splitlines() if line.strip()]
    if not stripped_lines:
        return True
    last_line = stripped_lines[-1].strip()
    try:
        gate = json.loads(last_line)
    except (json.JSONDecodeError, ValueError):
        return True
    if not isinstance(gate, dict):
        return True
    return gate.get("wakeAgent", True) is not False


def _prepend_context_block(prompt: str, heading: str, intro: str, body: str) -> str:
    """Prefix ``prompt`` with a fenced ``## heading`` data block."""
    return f"## {heading}\n{intro}\n\n```\n{body}\n```\n\n{prompt}"


_MAX_CONTEXT_CHARS = 8000


def _inject_context_from(job: dict, prompt: str) -> tuple[str, bool]:
    """Prepend the latest output of each ``context_from`` job; returns ``(prompt, injected)``."""
    context_from = job.get("context_from")
    if not context_from:
        return prompt, False
    from cron.jobs import get_cron_output_dir
    output_dir = get_cron_output_dir()
    if isinstance(context_from, str):
        context_from = [context_from]
    injected = False
    for source_job_id in context_from:
        # "self" = the job's own id: continuity across runs without touching session history.
        if isinstance(source_job_id, str) and source_job_id.strip().lower() == "self":
            source_job_id = str(job.get("id") or "")
        is_self = source_job_id == job.get("id")
        # Traversal guard — valid job IDs are hex strings.
        if not source_job_id or not all(c in "0123456789abcdef" for c in source_job_id):
            logger.warning(
                "context_from: skipping invalid job_id %r for job_id=%r name=%r%s",
                source_job_id, job.get("id"), job.get("name"), _cron_job_origin_log_suffix(job),
            )
            continue
        try:
            output_files = sorted(
                (output_dir / source_job_id).glob("*.md"),
                key=lambda f: f.stat().st_mtime,
                reverse=True,
            )
            if not output_files:
                continue  # silent skip — no output yet
            latest_output = output_files[0].read_text(encoding="utf-8").strip()
            if len(latest_output) > _MAX_CONTEXT_CHARS:
                latest_output = latest_output[:_MAX_CONTEXT_CHARS] + "\n\n[... output truncated ...]"
            if not latest_output:
                continue  # silent skip — empty output
            if is_self:
                prompt = _prepend_context_block(
                    prompt, "Your previous run's output",
                    "The following is this job's most recent output from its "
                    "previous run. Use it for continuity: avoid repeating what "
                    "was already reported, and continue where the last run "
                    "left off.",
                    latest_output,
                )
            else:
                prompt = _prepend_context_block(
                    prompt, f"Output from job '{source_job_id}'",
                    "The following is the most recent output from a preceding "
                    "cron job. Use it as context for your analysis.",
                    latest_output,
                )
            injected = True
        except (OSError, PermissionError) as e:
            # silent skip — never put error text into the prompt
            logger.warning("context_from: failed to read output for job %r: %s", source_job_id, e)
    return prompt, injected


def _load_cron_skill_parts(job: dict, skill_names: list[str]) -> list[str]:
    """Load each named skill/bundle into prompt parts; unknown ones are skipped with a user notice."""
    from tools.skills_tool import skill_view
    from tools.skill_usage import bump_use
    from agent.skill_bundles import build_bundle_invocation_message, resolve_bundle_command_key
    from agent.skill_utils import normalize_skill_lookup_name

    job_label = job.get("name", job.get("id"))
    task_id = str(job.get("id") or "") or None
    parts: list[str] = []
    skipped: list[str] = []
    for skill_name in skill_names:
        # Bundles shadow same-slug skills, mirroring the CLI/gateway slash-command path.
        bundle_key = resolve_bundle_command_key(skill_name.lstrip("/"))
        if bundle_key:
            bundle_payload = build_bundle_invocation_message(
                bundle_key, user_instruction="", task_id=task_id,
            )
            if bundle_payload:
                if parts:
                    parts.append("")
                parts.append(bundle_payload[0])
                continue
            logger.warning(
                "Cron job '%s': bundle '%s' could not load any skills, skipping", job_label, skill_name,
            )
            skipped.append(skill_name)
            continue

        try:
            loaded = json.loads(skill_view(normalize_skill_lookup_name(skill_name)))
        except (json.JSONDecodeError, TypeError):
            logger.warning("Cron job '%s': skill '%s' returned invalid JSON, skipping", job_label, skill_name)
            skipped.append(skill_name)
            continue
        if not loaded.get("success"):
            error = loaded.get("error") or f"Failed to load skill '{skill_name}'"
            logger.warning("Cron job '%s': skill not found, skipping — %s", job_label, error)
            skipped.append(skill_name)
            continue

        try:
            bump_use(skill_name, task_id=task_id)
        except Exception:
            logger.debug("Cron job: failed to bump skill usage for '%s'", skill_name, exc_info=True)

        if parts:
            parts.append("")
        parts.extend([
            f'[IMPORTANT: The user has invoked the "{skill_name}" skill, indicating they want you to follow its instructions. The full skill content is loaded below.]',
            "",
            str(loaded.get("content") or "").strip(),
        ])

    if skipped:
        parts.insert(0, (
            f"[IMPORTANT: The following skill(s) were listed for this job but could not be found "
            f"and were skipped: {', '.join(skipped)}. "
            f"Start your response with a brief notice so the user is aware, e.g.: "
            f"'⚠️ Skill(s) not found and skipped: {', '.join(skipped)}']"
        ))
    return parts


def _build_job_prompt(
    job: dict,
    prerun_script: Optional[tuple] = None,
    extra_prompt: Optional[str] = None,
) -> str:
    """Build the effective prompt for a cron job, optionally loading skills first.

    ``prerun_script``: cached ``(success, stdout)`` from a script the caller already ran (wake-gate
    check) — skips re-execution. ``extra_prompt``: per-run ``## Run Context`` for this fire only,
    never persisted to the job.
    """
    user_prompt = str(job.get("prompt") or "")
    if extra_prompt:
        user_prompt = f"{user_prompt}\n\n## Run Context\n{extra_prompt}"
    prompt = user_prompt
    skills = job.get("skills")
    # Runtime DATA (script stdout, upstream output) legitimately quotes command-shape strings, so it
    # must not be scanned with the strict user-prompt set — see _scan_assembled_cron_prompt.
    has_injected_data = False

    script_path = job.get("script")
    if script_path:
        if prerun_script is not None:
            success, script_output = prerun_script
        else:
            success, script_output = _run_job_script(script_path)
        if success and not script_output:
            return None  # no output → nothing to report, skip the AI call
        if success:
            prompt = _prepend_context_block(
                prompt, "Script Output",
                "The following data was collected by a pre-run script. "
                "Use it as context for your analysis.",
                script_output,
            )
        else:
            prompt = _prepend_context_block(
                prompt, "Script Error",
                "The data-collection script failed. Report this to the user.",
                script_output,
            )
        has_injected_data = True

    prompt, _ctx_injected = _inject_context_from(job, prompt)
    has_injected_data = has_injected_data or _ctx_injected

    # Durable per-job notepad; empty renders as "" so unused → byte-identical prompt.
    from cron import notepad as cron_notepad

    notepad_section = cron_notepad.render_notepad_section(str(job.get("id") or ""))
    if notepad_section:
        prompt = f"{notepad_section}{prompt}"
        has_injected_data = True

    cron_hint = (
        "[IMPORTANT: You are running as a scheduled cron job. "
        "DELIVERY: Your final response will be automatically delivered "
        "to the user — do NOT use send_message or try to deliver "
        "the output yourself. Just produce your report/output as your "
        "final response and the system handles the rest. "
        "SILENT: If there is genuinely nothing new to report, respond "
        "with exactly \"[SILENT]\" (nothing else) to suppress delivery. "
        "Never combine [SILENT] with content — either report your "
        "findings normally, or say [SILENT] and nothing more.]\n\n"
    )
    prompt = cron_hint + prompt
    if skills is None:
        legacy = job.get("skill")
        skills = [legacy] if legacy else []
    elif isinstance(skills, str):
        skills = [skills]

    skill_names = [str(name).strip() for name in skills if str(name).strip()]
    if not skill_names:
        return _scan_assembled_cron_prompt(
            prompt,
            job,
            has_skills=False,
            has_injected_data=has_injected_data,
            user_prompt=user_prompt,
        )

    parts = _load_cron_skill_parts(job, skill_names)
    stable_prefix = None
    if prompt:
        from agent.skill_commands import append_user_instruction

        parts.append("")
        # Skill blocks are stable per job config; the appended instruction is volatile per-run.
        # Declare that boundary for the Anthropic cache planner.
        stable_prefix = append_user_instruction(parts, prompt)
    assembled = _scan_assembled_cron_prompt("\n".join(parts), job, has_skills=True)
    if stable_prefix and len(assembled) > len(stable_prefix) and assembled.startswith(stable_prefix):
        # Guarded: the scanner may mutate the bytes; mismatch → whole-message caching.
        from agent.prompt_cache_boundary import register_stable_prefix

        register_stable_prefix(stable_prefix)
    return assembled


def _scan_assembled_cron_prompt(
    assembled: str,
    job: dict,
    *,
    has_skills: bool = False,
    has_injected_data: bool = False,
    user_prompt: Optional[str] = None,
) -> str:
    """Scan the assembled cron prompt for injection; raise ``CronPromptInjectionBlocked`` on a hit.

    Needed because skill content is loaded from disk at runtime (never scanned at create/update)
    and cron auto-approves tool calls. Tier is chosen by what the prompt CONTAINS: user prompt +
    hint only → STRICT ``_scan_cron_prompt``; skills or injected data → LOOSER
    ``_scan_cron_skill_assembled`` (command-shape patterns dropped, invisible unicode sanitized not
    blocked, so a false positive cannot permanently kill a job). With injected data but no skills,
    ``user_prompt`` is additionally scanned STRICT (defense-in-depth for legacy jobs).
    """
    from tools.cronjob_tools import _scan_cron_prompt, _scan_cron_skill_assembled

    if has_skills or has_injected_data:
        # The cleaned (sanitized) prompt is what actually runs.
        cleaned, scan_error = _scan_cron_skill_assembled(assembled)
        assembled = cleaned
        if not scan_error and not has_skills and user_prompt:
            scan_error = _scan_cron_prompt(user_prompt)
    else:
        scan_error = _scan_cron_prompt(assembled)
    if scan_error:
        job_label = job.get("name") or job.get("id") or "<unknown>"
        logger.warning(
            "Cron job '%s': assembled prompt blocked by injection scanner — %s",
            job_label,
            scan_error,
        )
        raise CronPromptInjectionBlocked(scan_error)
    return assembled


def _guard_job_credential_exfil(job: dict) -> None:
    """Fail closed (RuntimeError) if the stored provider/base_url pair could exfiltrate a key.

    Runtime backstop: jobs persisted before the create/update guard, or written directly to the
    store, reach provider resolution unchecked. Fallback providers come from operator config and
    are validated by the caller, not here.
    """
    try:
        from tools.cronjob_tools import _validate_cron_base_url
        err = _validate_cron_base_url(job.get("provider"), job.get("base_url"))
    except Exception as exc:
        # Fail CLOSED on validator/import errors — but only for jobs WITH a base_url override; a job
        # without one cannot exfiltrate via this path, so it still runs.
        if job.get("base_url"):
            err = (
                f"could not validate provider/base_url pair "
                f"({exc.__class__.__name__}: {exc}); refusing to run a job with "
                "an unverified base_url override"
            )
        else:
            err = None
    if err:
        job_id = job.get("id")
        logger.error(
            "Job '%s': refusing to run — unsafe provider/base_url pair could "
            "exfiltrate a stored credential: %s",
            job_id, err,
        )
        raise RuntimeError(f"Cron job '{job_id}' blocked for safety: {err}")


def _block_and_pause_job(
    job_id: str, job_name: str, reason: str
) -> tuple[bool, str, str, Optional[str]]:
    """Fail a run closed and pause the job: an unrunnable job left enabled re-fires every tick
    forever; ``paused_at``/``paused_reason`` give an auditable record."""
    from cron.jobs import pause_job

    logger.error("Job '%s': %s", job_id, reason)
    try:
        pause_job(job_id, f"Auto-paused by scheduler: {reason}")
    except Exception:
        logger.exception("Job '%s': failed to auto-pause unrunnable job", job_id)

    now_iso = _hermes_now().strftime("%Y-%m-%d %H:%M:%S")
    doc = (
        f"# Cron Job: {job_name}\n\n"
        f"**Job ID:** {job_id}\n"
        f"**Run Time:** {now_iso}\n"
        f"**Status:** blocked (unrunnable job) — auto-paused\n\n"
        f"{reason}\n"
    )
    alert = f"⚠ Cron job '{job_name}' was auto-paused\n\n{reason}"
    return False, doc, alert, reason


# Error-string prefixes from ``run_job``; ``run_one_job`` keys off them for last_status and the
# alert-once dedup. ``:silent`` = already alerted on a previous tick — do not deliver again.
BLOCKED_CONFIG_MARKER = "[blocked_config]"
BLOCKED_CONFIG_SILENT_MARKER = "[blocked_config:silent]"

# Drift-guard skip: same contract (drift_alerted bit on the job record).
DRIFT_SKIP_MARKER = "[drift_skip]"
DRIFT_SKIP_SILENT_MARKER = "[drift_skip:silent]"


_TRANSIENT_NET_EXC_NAMES = frozenset({
    "ConnectError", "ConnectTimeout", "ReadTimeout", "WriteTimeout", "PoolTimeout", "NetworkError",
    "TimeoutException", "ClientConnectorError", "ClientConnectorDNSError", "ServerTimeoutError",
    "ClientOSError",
})
_DNS_FAILURE_NEEDLES = ("nodename nor servname", "name or service not known")
_TRANSIENT_OSERROR_NEEDLES = _DNS_FAILURE_NEEDLES + (
    "temporary failure in name resolution", "network is unreachable",
)
_TRANSIENT_HTTP_NEEDLES = _TRANSIENT_OSERROR_NEEDLES + (
    "failed to resolve", "connection refused", "timed out", "timeout",
)
_TRANSIENT_ERRNOS = frozenset({
    errno.ECONNREFUSED, errno.ECONNRESET, errno.EHOSTUNREACH, errno.ENETUNREACH, errno.ENETDOWN,
    errno.ETIMEDOUT, errno.EAGAIN,
})


def _is_transient_provider_resolve_error(exc: BaseException) -> bool:
    """True when primary provider resolution failed for a transient network reason (DNS blip,
    ConnectError...). Must be eligible for ``fallback_providers`` like AuthError, else a healthy
    fallback rung is never tried and the job dies before the first model call."""
    import socket

    # gaierror carries EAI_* codes, plain OSError carries errno — never mix the namespaces (raw
    # literals like {8, 7, 11} are macOS-only and wrong on Linux).
    eai_transient = {
        getattr(socket, n) for n in ("EAI_NONAME", "EAI_AGAIN", "EAI_FAIL", "EAI_NODATA")
        if hasattr(socket, n)
    }
    # Walk the cause chain; the scheduler wraps raw transport errors.
    seen: set[int] = set()
    cur: Optional[BaseException] = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        module = type(cur).__module__ or ""
        msg = str(cur).lower()
        if type(cur).__name__ in _TRANSIENT_NET_EXC_NAMES:
            return True
        if any(m in module for m in ("httpx", "httpcore", "aiohttp")) and any(
            needle in msg for needle in _TRANSIENT_HTTP_NEEDLES
        ):
            return True
        if isinstance(cur, OSError):
            if isinstance(cur, socket.gaierror):
                if cur.errno in eai_transient:
                    return True
            elif getattr(cur, "errno", None) in _TRANSIENT_ERRNOS:
                return True
            if any(needle in msg for needle in _TRANSIENT_OSERROR_NEEDLES):
                return True
        # Bare exceptions that carry the raw DNS text (format_runtime_provider_error).
        if any(needle in msg for needle in _DNS_FAILURE_NEEDLES):
            return True
        cur = cur.__cause__ or cur.__context__
    return False


def _cron_preflight_enabled(cfg: dict) -> bool:
    """Preflight is ON unless ``cron.preflight`` is literally ``false``."""
    cron_cfg = (cfg or {}).get("cron")
    if not isinstance(cron_cfg, dict):
        return True
    return cron_cfg.get("preflight", True) is not False


def _preflight_check_provider_key(job: dict, cfg: dict) -> Optional[str]:
    """READ-ONLY probe: would provider resolution fail for lack of a key? Mirrors run_job's
    requested-provider computation. Skipped when a fallback chain exists — auth-fallback may
    legitimately rescue a missing primary key, so blocking here would break that contract."""
    try:
        if get_fallback_chain(cfg):
            return None
    except Exception:
        return None  # fail-open: never block on a preflight-internal error

    _cron_cfg = cfg.get("cron") if isinstance(cfg.get("cron"), dict) else {}
    requested = (
        job.get("provider")
        or str((_cron_cfg or {}).get("model_provider") or "").strip()
        or None
    )
    model = job.get("model") or os.getenv("HERMES_MODEL") or ""

    from hermes_cli.auth import AuthError

    try:
        from hermes_cli.runtime_provider import resolve_runtime_provider

        kwargs = {"requested": requested, "target_model": model}
        if job.get("base_url"):
            kwargs["explicit_base_url"] = job.get("base_url")
        resolve_runtime_provider(**kwargs)
    except AuthError as exc:
        return (
            f"provider credential missing: {exc}. "
            "Set the provider API key in .env (or `hermes setup`), or pin a "
            "working provider via `hermes cron edit "
            f"{job.get('id')} --provider <p>`."
        )
    except Exception:
        return None  # non-auth errors are not a missing-credential verdict; real path reports them
    return None


def _primary_profile_routes_for_current_home() -> list:
    """Primary gateway ``profile_routes`` targeting the profile being served; ``[]`` if this IS the
    primary home.

    Satellite crons are ticked and delivered by the primary gateway (a satellite holding its own
    token is a ``duplicate_credential`` fatal). Reads the primary config.yaml directly (top-level or
    nested ``gateway.``) instead of ``load_gateway_config()`` so no primary platform config leaks
    into this process. Shared by preflight rescue and delivery-time resolution so they cannot drift.
    """
    try:
        from hermes_constants import get_default_hermes_root, get_hermes_home

        primary_home = get_default_hermes_root()
        current_home = Path(get_hermes_home())
        if (
            primary_home.expanduser().resolve(strict=False)
            == current_home.expanduser().resolve(strict=False)
        ):
            return []  # this IS the primary home — nothing to consult
        config_path = primary_home.expanduser() / "config.yaml"
        if not config_path.exists():
            return []

        import yaml

        with open(config_path, encoding="utf-8") as f:
            raw = yaml.safe_load(f) or {}
        routes_raw = raw.get("profile_routes")
        if routes_raw is None and isinstance(raw.get("gateway"), dict):
            routes_raw = raw["gateway"].get("profile_routes")
        if not isinstance(routes_raw, list):
            return []

        from gateway.profile_routing import parse_profile_routes
        from hermes_cli.profiles import profile_matches_home

        return [
            route
            for route in parse_profile_routes(routes_raw)
            if route.enabled and profile_matches_home(route.profile)
        ]
    except Exception:
        logger.debug("primary-gateway profile-route lookup unavailable", exc_info=True)
        return []


def _delivery_platform_routed_from_primary_gateway(platform_name: str) -> bool:
    """True when the primary gateway routes this platform to the profile being served."""
    platform_key = platform_name.lower()
    return any(
        str(route.platform).lower() == platform_key
        for route in _primary_profile_routes_for_current_home()
    )


class SharedRouteAdapters:
    """Read-only adapter map for a credentialless satellite profile.

    ``get(platform, target)`` resolves the PRIMARY adapter iff the inbound route matcher
    (``ProfileRoute.matches``) accepts the target; anything else (unmatched target, disabled route,
    other profile, or target-less ``get(platform)``) is a miss — fail closed, never the default bot.
    """

    def __init__(self, primary_adapters, routes) -> None:
        self._primary = dict(primary_adapters or {})
        self._routes = list(routes or [])

    def __bool__(self) -> bool:
        return bool(self._primary) and bool(self._routes)

    def get(self, platform, target=None, default=None):
        if not target:
            return default
        adapter = self._primary.get(platform)
        if adapter is None:
            return default
        platform_key = str(getattr(platform, "value", platform)).lower()
        chat_id = str(target.get("chat_id") or "") or None
        thread_id = target.get("thread_id")
        thread_id = str(thread_id) if thread_id else None
        for route in self._routes:
            if str(route.platform).lower() != platform_key:
                continue
            if not (route.chat_id or route.thread_id):
                continue  # guild-only routes are not target-exact
            if route.matches(str(route.platform), chat_id=chat_id, thread_id=thread_id):
                return adapter
        return default


def _preflight_check_delivery(job: dict) -> Optional[str]:
    """Check delivery targets resolve to configured platforms.

    ``local``/``origin``/``all`` are never checked (no gateway-config load). Unknown platform always
    blocks; known platform blocks only if the gateway config loads AND reports it unconnected.
    Config load failures fail OPEN. ``failure_deliver`` is checked with the same rules: a typo'd
    failure platform would otherwise only surface when a failure occurs (NS-788).
    """
    deliver_value = _normalize_deliver_value(job.get("deliver", "local"))
    failure_deliver_value = _normalize_deliver_value(
        _delivery_lane_value(job, for_failure=True)
    )
    lane_values = [deliver_value]
    if failure_deliver_value != deliver_value:
        lane_values.append(failure_deliver_value)
    platform_parts: list[str] = []
    for lane_value in lane_values:
        for part in lane_value.split(","):
            part = part.strip()
            if not part or part.lower() in {"local", "origin", "all"}:
                continue
            # bot-chat targets deliver via a local subprocess; failures surface in last_delivery_error.
            if parse_bot_chat_deliver_token(part) is not None:
                continue
            platform_parts.append(part.split(":", 1)[0].strip())
    if not platform_parts:
        return None

    connected: Optional[set] = None
    for platform_name in platform_parts:
        if not _is_known_delivery_platform(platform_name):
            return (
                f"delivery platform '{platform_name}' is not a known cron "
                "delivery target. Fix the job's `deliver` value or configure "
                "the platform's gateway credentials."
            )
        if connected is None:
            try:
                from gateway.config import load_gateway_config

                gateway_config = load_gateway_config()
                connected = {p.value for p in gateway_config.get_connected_platforms()}
                connected |= _relay_fronted_delivery_platforms(connected)
            except Exception:
                logger.debug(
                    "preflight: gateway config unavailable — skipping "
                    "delivery credential check", exc_info=True,
                )
                return None  # fail-open
        if platform_name.lower() not in connected:
            # Multiplex: a satellite served by the primary's adapters reads unconnected — no block.
            if _delivery_platform_routed_from_primary_gateway(platform_name):
                continue
            return (
                f"delivery platform '{platform_name}' has no gateway "
                "credentials configured (not connected). Configure it via "
                "`hermes setup` or change the job's `deliver` target."
            )
    return None


def _preflight_check_skills(job: dict) -> Optional[str]:
    """Block only on an affirmative ``setup_needed`` verdict from ``skill_view``; skills that fail
    to load fall through to ``_build_job_prompt``'s skipped-skill handling (fail-open)."""
    skills = job.get("skills")
    if skills is None:
        legacy = job.get("skill")
        skills = [legacy] if legacy else []
    elif isinstance(skills, str):
        skills = [skills]
    skill_names = [str(name).strip() for name in skills if str(name).strip()]
    if not skill_names:
        return None

    from tools.skills_tool import skill_view

    for skill_name in skill_names:
        try:
            payload = json.loads(skill_view(skill_name))
        except Exception:
            continue  # unreadable/missing skill → existing skip handling
        if not isinstance(payload, dict) or not payload.get("success"):
            continue
        if (
            payload.get("setup_needed")
            or payload.get("readiness_status") == "setup_needed"
        ):
            missing = [
                f"env ${name}"
                for name in payload.get(
                    "missing_required_environment_variables"
                ) or []
            ]
            missing += [
                f"command '{name}'"
                for name in payload.get("missing_required_commands") or []
            ]
            missing += [
                f"credential file {name}"
                for name in payload.get("missing_credential_files") or []
            ]
            detail = ", ".join(missing) or "required setup incomplete"
            return (
                f"attached skill '{skill_name}' is not ready: missing "
                f"{detail}. Provide the missing prerequisites or detach the "
                "skill from this job."
            )
    return None


def _preflight_job_config(job: dict, cfg: dict) -> Optional[str]:
    """Pre-dispatch validation: return a reason (missing key, unconfigured delivery, unready skill)
    so the caller refuses BEFORE building agent machinery or burning an LLM call. Every check fails
    open — preflight blocks only on an affirmative misconfiguration verdict."""
    for name, check in (
        ("provider_key", lambda: _preflight_check_provider_key(job, cfg)),
        ("skills", lambda: _preflight_check_skills(job)),
        ("delivery", lambda: _preflight_check_delivery(job)),
    ):
        try:
            reason = check()
        except Exception:
            logger.debug("preflight check %s raised — failing open", name, exc_info=True)
            continue
        if reason:
            return reason
    return None


def _cron_cleanup_timeout_seconds() -> float:
    """Return the wall-clock bound for cron post-run cleanup."""
    default = 10.0
    try:
        from hermes_cli.config import load_config

        cfg = load_config() or {}
        cron_cfg = cfg.get("cron", {}) if isinstance(cfg, dict) else {}
        configured = cron_cfg.get("cleanup_timeout_seconds")
        if configured is not None:
            timeout = float(configured)
            if timeout >= 0:
                return timeout
    except Exception as exc:
        logger.debug("Failed to load cron cleanup timeout from config: %s", exc)
    return default


def _run_cron_cleanup_with_timeout(
    cleanup,
    *,
    job_id: str,
    label: str,
    timeout_seconds: Optional[float] = None,
) -> bool:
    """Run fallible post-run cleanup without permanently wedging a cron ID."""
    timeout = (_cron_cleanup_timeout_seconds() if timeout_seconds is None else float(timeout_seconds))
    if timeout <= 0:
        try:
            cleanup()
            return True
        except (Exception, KeyboardInterrupt) as exc:
            logger.debug("Job '%s': %s failed: %s", job_id, label, exc)
            return False

    done = threading.Event()
    error: list[BaseException] = []

    def _runner() -> None:
        try:
            cleanup()
        except BaseException as exc:
            error.append(exc)
        finally:
            done.set()

    # Daemon thread is deliberate: unlike ThreadPoolExecutor workers it is not joined at interpreter
    # exit if cleanup never returns, so the gateway can still shut down.
    worker = threading.Thread(
        target=_runner,
        name=f"cron-cleanup-{job_id}",
        daemon=True,
    )
    worker.start()
    if not done.wait(timeout):
        logger.error(
            "Job '%s': %s exceeded %.1fs; abandoning cleanup so future runs remain dispatchable",
            job_id,
            label,
            timeout,
        )
        return False
    if error:
        logger.debug("Job '%s': %s failed: %s", job_id, label, error[0])
        return False
    return True


class _BoundedCronSessionDB:
    """Proxy SessionDB cleanup calls through the cron cleanup timeout; after the first failure or
    timeout all later calls fail immediately (a damaged connection leaks at most one worker)."""

    def __init__(self, session_db, job_id: str):
        self._session_db = session_db
        self._job_id = job_id
        self._disabled = False

    def __getattr__(self, name):
        target = getattr(self._session_db, name)
        if not callable(target):
            return target

        def _bounded(*args, **kwargs):
            if self._disabled:
                raise RuntimeError("session finalization disabled after prior cleanup failure")

            result = {}

            def _call():
                try:
                    result["value"] = target(*args, **kwargs)
                except BaseException as exc:
                    result["error"] = exc
                    raise

            ok = _run_cron_cleanup_with_timeout(
                _call,
                job_id=self._job_id,
                label=f"session finalization ({name})",
            )
            if not ok:
                error = result.get("error")
                if error is not None:
                    raise error
                # No error yet not complete == timeout: disable so later steps fail fast.
                self._disabled = True
                raise TimeoutError(f"session finalization method {name} timed out")
            return result.get("value")

        return _bounded


def _job_doc_header(job_name: str, job_id: str, now_iso: str, mode: str) -> str:
    """Common markdown header for the short-circuit run docs (no_agent / monitor)."""
    return (
        f"# Cron Job: {job_name}\n\n"
        f"**Job ID:** {job_id}\n"
        f"**Run Time:** {now_iso}\n"
        f"**Mode:** {mode}\n"
    )


def _resolve_job_workdir(job: dict, job_id: str) -> Optional[str]:
    """Configured job workdir, or None when unset / no longer a directory (logged)."""
    workdir = (job.get("workdir") or "").strip() or None
    if workdir and not Path(workdir).is_dir():
        logger.warning(
            "Job '%s': configured workdir %r no longer exists — running without it",
            job_id, workdir,
        )
        return None
    return workdir


def _run_no_agent_job(
    job: dict, job_id: str, job_name: str, cancel_event,
) -> tuple[bool, str, str, Optional[str]]:
    """no_agent short-circuit — the script IS the job (no AIAgent, no tokens). stdout → delivered
    verbatim; empty stdout or wakeAgent=false → silent success; non-zero exit/timeout → error alert.
    """
    # Load .env first so auto-delivery can resolve *_HOME_CHANNEL: the agent path's per-run dotenv
    # reload never runs for no_agent jobs. Does not override existing values.
    try:
        from hermes_cli.env_loader import load_hermes_dotenv

        load_hermes_dotenv(hermes_home=_get_hermes_home())
    except Exception:
        logger.debug("Job '%s': no_agent .env reload failed", job_id, exc_info=True)

    script_path = job.get("script")
    # Legacy/hand-edited no_agent job without a script: pause it, or it re-fires every tick.
    if not str(script_path or "").strip():
        from cron.jobs import NO_AGENT_WITHOUT_SCRIPT_ERROR

        return _block_and_pause_job(job_id, job_name, NO_AGENT_WITHOUT_SCRIPT_ERROR)

    # Pass workdir as subprocess cwd; never os.chdir() (leaks into concurrent gateway sessions).
    _job_workdir = _resolve_job_workdir(job, job_id)
    try:
        ok, output = _run_job_script_with_claim_heartbeat(
            job, script_path, workdir=_job_workdir, cancel_event=cancel_event,
        )
    except Exception as exc:
        logger.exception("Job '%s': script execution raised unexpectedly", job_id)
        ok, output = False, f"Script execution failed: {exc}"

    now_iso = _hermes_now().strftime("%Y-%m-%d %H:%M:%S")
    header = _job_doc_header(job_name, job_id, now_iso, "no_agent (script)")

    if not ok:
        # Deliver the error: a silently broken watchdog is the worst-case outcome.
        alert = (
            f"⚠ Cron watchdog '{job_name}' script failed\n\n"
            f"{output}\n\n"
            f"Time: {now_iso}"
        )
        return False, f"{header}**Status:** script failed\n\n{output}\n", alert, output

    # wakeAgent=false is a silent signal, same as empty stdout.
    if not _parse_wake_gate(output):
        logger.info("Job '%s' (no_agent): wakeAgent=false gate — silent run", job_id)
        return True, f"{header}**Status:** silent (wakeAgent=false)\n", SILENT_MARKER, None

    if not output.strip():
        logger.info("Job '%s' (no_agent): empty stdout — silent run", job_id)
        return True, f"{header}**Status:** silent (empty output)\n", SILENT_MARKER, None

    return True, f"{header}\n---\n\n{output}\n", output, None


def _apply_monitor_gate(
    job: dict, job_id: str, job_name: str, extra_prompt: Optional[str],
) -> tuple[Optional[tuple], Optional[str]]:
    """Monitor gate (hash-suppressed change detection). Must run BEFORE any agent machinery so an
    unchanged tick costs no LLM/delivery. Returns ``(early_result | None, extra_prompt)``; when
    early_result is None, extra_prompt may carry the injected monitor context.
    """
    from cron.monitor import check_monitor, job_has_monitor

    if not job_has_monitor(job):
        return None, extra_prompt
    _mon = check_monitor(job)
    _mon_now = _hermes_now().strftime("%Y-%m-%d %H:%M:%S")
    header = _job_doc_header(job_name, job_id, _mon_now, "monitor")
    if not _mon.ok:
        # Source failure is an ERROR, never a change: alert so a broken monitor can't silently
        # stop watching. Stored hash untouched.
        logger.error("Job '%s': monitor source failed: %s", job_id, _mon.error)
        _mon_alert = (
            f"⚠ Cron monitor '{job_name}' source failed\n\n"
            f"{_mon.error}\n\n"
            f"Time: {_mon_now}"
        )
        return (
            False, f"{header}**Status:** monitor source failed\n\n{_mon.error}\n", _mon_alert, _mon.error,
        ), extra_prompt
    if not _mon.changed:
        # Unchanged: silent no_change tick (ledger doc kept; SILENT_MARKER blocks delivery).
        logger.info("Job '%s': monitor output unchanged — suppressing agent run", job_id)
        return (
            True, f"{header}**Status:** no_change (agent run suppressed)\n", SILENT_MARKER, None,
        ), extra_prompt
    # Changed (or first run): inject monitor context via the per-run seam, then normal agent run.
    if _mon.context_block:
        extra_prompt = (
            f"{_mon.context_block}\n\n{extra_prompt}" if extra_prompt else _mon.context_block
        )
    return None, extra_prompt


@dataclass
class _CronJobConfig:
    """Config-derived inputs for one agent-backed cron run."""

    cfg: dict
    model: str
    model_cfg: Any
    cron_default_provider: str


def _load_cron_job_config(job: dict, job_id: str, job_name: str) -> _CronJobConfig:
    """Load config.yaml and resolve the run's model.

    Precedence: per-job override > cron.model (fleet default) > HERMES_MODEL > config ``model:``.
    Re-read every tick (no cache) so ``hermes cron edit --model`` takes effect next tick. An axis
    resolved from cron.model / cron.model_provider is explicit, so the drift guard skips it.
    """
    model = job.get("model") or os.getenv("HERMES_MODEL") or ""
    _cron_default_provider = ""
    _cfg: dict = {}
    _model_cfg: Any = {}
    try:
        from hermes_cli.config import read_user_config_raw
        _cfg_path = str(_get_hermes_home() / "config.yaml")
        if os.path.exists(_cfg_path):
            _cfg = read_user_config_raw(Path(_cfg_path))
            # Honor administrator-pinned managed scope (fail-open; no-op without managed scope).
            with contextlib.suppress(Exception):
                from hermes_cli import managed_scope
                _cfg = managed_scope.apply_managed_overlay(_cfg)
            _cfg = _expand_env_vars(_cfg)
            # Coerce null to {} so a falsy default never clobbers a resolved env value.
            _model_cfg = _cfg.get("model") or {}
            _cron_cfg_for_model = _cfg.get("cron") or {}
            _cron_default_model = ""
            if isinstance(_cron_cfg_for_model, dict):
                _cron_default_model = str(_cron_cfg_for_model.get("model") or "").strip()
                _cron_default_provider = str(_cron_cfg_for_model.get("model_provider") or "").strip()
            if not job.get("model"):
                if _cron_default_model:
                    model = _cron_default_model
                else:
                    # Shared with Desktop's impact summary so both compare against the same model.
                    _, _global_model = resolve_cron_model_drift_defaults(_cfg)
                    if _global_model:
                        model = _global_model
    except Exception as e:
        logger.warning("Job '%s': failed to load config.yaml, using defaults: %s", job_id, e)

    # Fail fast: an empty model otherwise reaches the provider as an opaque 400.
    if not (isinstance(model, str) and model.strip()):
        raise RuntimeError(
            f"Cron job '{job_name}' has no model configured "
            f"(job.model={job.get('model')!r}, "
            f"HERMES_MODEL={os.getenv('HERMES_MODEL', '')!r}, "
            "config.yaml model.default missing or empty). "
            f"Set a per-job model via "
            f"`hermes cron edit {job_id} --model <name>` or set a "
            "default with `hermes model <name>`."
        )

    with contextlib.suppress(Exception):
        from hermes_constants import apply_ipv4_preference
        _net_cfg = _cfg.get("network", {})
        if isinstance(_net_cfg, dict) and _net_cfg.get("force_ipv4"):
            apply_ipv4_preference(force=True)
    return _CronJobConfig(_cfg, model, _model_cfg, _cron_default_provider)


def _load_prefill_messages(cfg: dict, job_id: str) -> Optional[list]:
    """Prefill messages from env or config.yaml (top-level key canonical; agent.* is legacy)."""
    agent_cfg = cfg.get("agent", {}) if isinstance(cfg.get("agent", {}), dict) else {}
    prefill_file = (
        os.getenv("HERMES_PREFILL_MESSAGES_FILE", "")
        or cfg.get("prefill_messages_file", "")
        or agent_cfg.get("prefill_messages_file", "")
    )
    if not prefill_file:
        return None
    pfpath = Path(prefill_file).expanduser()
    if not pfpath.is_absolute():
        pfpath = _get_hermes_home() / pfpath
    if not pfpath.exists():
        return None
    try:
        with open(pfpath, "r", encoding="utf-8") as _pf:
            prefill_messages = json.load(_pf)
        return prefill_messages if isinstance(prefill_messages, list) else None
    except Exception as e:
        logger.warning("Job '%s': failed to parse prefill messages file '%s': %s", job_id, pfpath, e)
        return None


def _preflight_or_block(job: dict, job_id: str, job_name: str, cfg: dict) -> Optional[tuple]:
    """Pre-dispatch config validation: refuse unrunnable jobs (missing key, unready skill,
    unconfigured delivery) BEFORE AIAgent is built. run_one_job keys off BLOCKED_CONFIG_MARKER to
    record blocked_config and alert once (`preflight_alerted` bit). Must run after the wake gate so
    silent ticks stay silent. Opt-out: `cron.preflight: false`. Returns failure tuple or None.
    """
    _pf_reason = None
    try:
        if _cron_preflight_enabled(cfg):
            _pf_reason = _preflight_job_config(job, cfg)
            if not _pf_reason and job.get("preflight_alerted"):
                # Config healthy again: clear alert-once marker so a future break re-alerts.
                with contextlib.suppress(Exception):
                    from cron.jobs import clear_preflight_alerted
                    clear_preflight_alerted(job_id)
    except Exception:
        # Fail open: the validator must never take down a runnable job.
        logger.debug("Job '%s': preflight validation errored — failing open", job_id, exc_info=True)
        _pf_reason = None
    if not _pf_reason:
        return None

    logger.warning(
        "Job '%s' (ID: %s): BLOCKED by pre-dispatch config "
        "validation — %s (no LLM call was made)",
        job_name, job_id, _pf_reason,
    )
    already_alerted = False
    try:
        from cron.jobs import mark_preflight_alerted
        already_alerted = mark_preflight_alerted(job_id)
    except Exception:
        logger.debug("Job '%s': could not persist preflight alert marker", job_id, exc_info=True)
    marker = BLOCKED_CONFIG_SILENT_MARKER if already_alerted else BLOCKED_CONFIG_MARKER
    blocked_doc = (
        f"# Cron Job: {job_name}\n\n"
        f"**Job ID:** {job_id}\n"
        f"**Run Time:** {_hermes_now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"**Status:** BLOCKED (configuration)\n\n"
        "Pre-dispatch validation found a configuration problem and "
        "the agent was NOT run (no tokens spent).\n\n"
        f"**Reason:** {_pf_reason}\n\n"
        "The job will stay blocked (without re-alerting) until the "
        "configuration is fixed; the next healthy run clears this "
        "state. Set `cron.preflight: false` in config.yaml to "
        "disable this validation."
    )
    return False, blocked_doc, "", f"{marker} {_pf_reason}"


def _resolve_job_runtime(
    job: dict, job_id: str, jc: _CronJobConfig,
) -> tuple[dict, str, Optional[str]]:
    """Resolve the runtime, walking the fallback chain on auth/transient-network errors.

    Returns ``(runtime, model, primary_provider_for_drift)``; provider+model swap atomically (never
    swap only the provider while keeping a paid primary model).
    """
    from hermes_cli.runtime_provider import (
        resolve_runtime_provider,
        format_runtime_provider_error,
    )
    from hermes_cli.auth import AuthError

    model = jc.model
    configured_provider_for_drift = (
        str(jc.model_cfg.get("provider") or "").strip().lower()
        if isinstance(jc.model_cfg, dict)
        else ""
    )
    primary_provider_for_drift = (
        str(job.get("provider") or "").strip().lower()
        or configured_provider_for_drift
        or None
    )
    try:
        # Do NOT pass HERMES_INFERENCE_PROVIDER as `requested`: it would override persisted config
        # and resurrect stale providers for unpinned jobs.
        runtime_kwargs = {
            "requested": job.get("provider") or jc.cron_default_provider or None,
            # api_mode must derive from the model actually run, not the stale persisted default.
            "target_model": model,
        }
        if job.get("base_url"):
            runtime_kwargs["explicit_base_url"] = job.get("base_url")
        runtime = resolve_runtime_provider(**runtime_kwargs)
        primary_provider_for_drift = (
            str(runtime.get("provider") or "").strip().lower() or primary_provider_for_drift
        )
        return runtime, model, primary_provider_for_drift
    except Exception as resolve_exc:
        # Walk the fallback chain on AuthError AND transient network/DNS failures (e.g. during
        # OAuth refresh); anything else re-raises.
        is_auth = isinstance(resolve_exc, AuthError)
        is_transient_net = _is_transient_provider_resolve_error(resolve_exc)
        if not (is_auth or is_transient_net):
            raise RuntimeError(format_runtime_provider_error(resolve_exc)) from resolve_exc

        primary_provider_for_drift = (
            str(getattr(resolve_exc, "provider", "") or "").strip().lower()
            or primary_provider_for_drift
        )
        logger.warning(
            "Job '%s': primary provider resolve failed (%s: %s), trying fallback",
            job_id, "auth" if is_auth else "transient network", resolve_exc,
        )
        for entry in get_fallback_chain(jc.cfg):
            if not isinstance(entry, dict):
                continue
            fb_provider = str(entry.get("provider") or "").strip()
            fb_model = str(entry.get("model") or "").strip()
            if not fb_provider or not fb_model:
                continue
            try:
                from hermes_cli.fallback_config import resolve_entry_api_key

                fb_kwargs = {"requested": fb_provider, "target_model": fb_model}
                if entry.get("base_url"):
                    fb_kwargs["explicit_base_url"] = entry["base_url"]
                fb_api_key = resolve_entry_api_key(entry)
                if fb_api_key:
                    fb_kwargs["explicit_api_key"] = fb_api_key
                runtime = resolve_runtime_provider(**fb_kwargs)
                logger.info(
                    "Job '%s': fallback resolved to %s model %s",
                    job_id, runtime.get("provider"), fb_model,
                )
                return runtime, fb_model, primary_provider_for_drift
            except Exception as fb_exc:
                logger.debug("Job '%s': fallback %s failed: %s", job_id, fb_provider, fb_exc)
        raise RuntimeError(format_runtime_provider_error(resolve_exc)) from resolve_exc


def _check_model_drift(
    job: dict, job_id: str, cfg: dict, runtime: dict,
    primary_provider_for_drift: Optional[str], primary_model_for_drift: str,
) -> None:
    """Fail-closed provider/model drift guard; raises RuntimeError (with drift marker) on drift.

    An unpinned job follows the global default, which may have switched to a paid provider/model
    since creation. For each unpinned axis with a creation snapshot (job["<axis>_snapshot"]) that
    now resolves differently: skip the run, no paid call, alert to pin. No snapshot, pinned axes,
    or resolution from the cron.model fleet default never count as drift.
    """
    if not cron_model_drift_guard_enabled(cfg):
        return
    _current_provider = str(
        primary_provider_for_drift or runtime.get("provider") or ""
    ).strip().lower()
    _current_model = str(primary_model_for_drift or "").strip().lower()
    _drift: list[str] = []
    for _axis in cron_model_drift_axes(
        job, current_provider=_current_provider, current_model=_current_model, config=cfg,
    ):
        _snapshot = str(job.get(f"{_axis}_snapshot") or "").strip().lower()
        _current = _current_provider if _axis == "provider" else _current_model
        _drift.append(f"{_axis} '{_snapshot}' -> '{_current}'")
    if not _drift:
        return
    _changes = "; ".join(_drift)
    # A finite one-shot is consumed by this attempt, so "edit the job" is a dead end for it.
    _repeat = job.get("repeat") if isinstance(job.get("repeat"), dict) else {}
    _finite_oneshot = (
        isinstance(job.get("schedule"), dict)
        and job["schedule"].get("kind") == "once"
        and _repeat.get("times") == 1
    )
    if _finite_oneshot:
        _remediation = (
            "This finite one-shot job is consumed by this attempted run; "
            "create a new one-shot job at a future time with an explicit "
            "provider and model."
        )
    else:
        _remediation = (
            "To run on the new config, on the host running Hermes "
            "pin it explicitly: "
            f"`hermes cron edit {job_id} --provider <provider> "
            "--model <model>` (or pin the original values to keep "
            "them)."
        )
    logger.warning(
        "Job '%s': SKIPPED — global inference config drifted since "
        "creation (%s) and this job is unpinned. Skipped to prevent "
        "unintended spend. %s",
        job_id, _changes, _remediation,
    )
    # Alert-once via drift_alerted bit (silent marker suppresses delivery); a successful run
    # clears it and re-arms the alert.
    _drift_already_alerted = False
    with contextlib.suppress(Exception):
        from cron.jobs import mark_drift_alerted

        _drift_already_alerted = mark_drift_alerted(job_id)
    _drift_marker = DRIFT_SKIP_SILENT_MARKER if _drift_already_alerted else DRIFT_SKIP_MARKER
    raise RuntimeError(
        f"{_drift_marker} Skipped to prevent unintended spend: global "
        f"inference config drifted since this job was created "
        f"({_changes}), and this job is unpinned. No inference call "
        f"was made. {_remediation} "
        f"This alert is sent once; the job stays skipped until the "
        f"config is pinned or restored. See #44585."
    )


def _load_credential_pool(runtime: dict, job_id: str):
    runtime_provider = str(runtime.get("provider") or "").strip().lower()
    if not runtime_provider:
        return None
    try:
        from agent.credential_pool import load_pool
        pool = load_pool(runtime_provider)
        if pool.has_credentials():
            logger.info(
                "Job '%s': loaded credential pool for provider %s with %d entries",
                job_id, runtime_provider, len(pool.entries()),
            )
            return pool
    except Exception as e:
        logger.debug("Job '%s': failed to load credential pool for %s: %s", job_id, runtime_provider, e)
    return None


def _init_cron_mcp_tools(job_id: str) -> None:
    """Register MCP servers for the agent's tool registry. Idempotent across ticks; non-fatal so a
    broken MCP server never kills a working job."""
    try:
        from tools.mcp_tool import discover_mcp_tools
        _mcp_tools = discover_mcp_tools()
        if _mcp_tools:
            logger.info("Job '%s': %d MCP tool(s) available", job_id, len(_mcp_tools))
    except Exception as _mcp_exc:
        logger.warning("Job '%s': MCP initialization failed (non-fatal): %s", job_id, _mcp_exc)


def _open_cron_session_db(job: dict):
    """Open the SQLite session store under its own timeout (HERMES_CRON_TIMEOUT only watches
    run_conversation). A wedged sqlite3.connect returns None (no session store) instead of
    wedging the worker thread."""
    _session_db_timeout = _get_session_db_timeout()
    try:
        from hermes_state import get_shared_session_db

        if _session_db_timeout <= 0:
            return get_shared_session_db()
        _session_db_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        # Copy the context so a profile run resolves ITS OWN home/state.db on the worker thread
        # instead of the process-global default.
        _session_db_context = contextvars.copy_context()
        _session_db_future = _session_db_pool.submit(_session_db_context.run, get_shared_session_db)
        try:
            return _session_db_future.result(timeout=_session_db_timeout)
        except concurrent.futures.TimeoutError:
            # The abandoned worker may still finish; close its late result or its SQLite FDs leak.
            _session_db_future.add_done_callback(_close_late_session_db_result)
            raise
        finally:
            # Abandon a wedged connect() rather than blocking shutdown on it.
            _session_db_pool.shutdown(wait=False)
    except concurrent.futures.TimeoutError:
        logger.error(
            "Job '%s': SessionDB init did not return within %.0fs — proceeding "
            "without a session store for this run instead of blocking it "
            "forever",
            job.get("id", "?"), _session_db_timeout,
        )
    except Exception as e:
        logger.debug("Job '%s': SQLite session store not available: %s", job.get("id", "?"), e)
    return None


def _run_agent_with_watchdog(
    agent, prompt: str, job: dict, job_id: str, job_name: str, task_id: str, cancel_event,
) -> dict:
    """Run ``agent.run_conversation`` on a worker thread under the inactivity watchdog.

    Inactivity (not wall-clock) limit from the agent's activity tracker; default 600s, override
    HERMES_CRON_TIMEOUT, 0 = unlimited.
    """
    _cron_timeout = _cron_inactivity_seconds()
    _cron_inactivity_limit = _cron_timeout if _cron_timeout > 0 else None
    _POLL_INTERVAL = 5.0
    # Heartbeat the one-shot run_claim while alive: without it a long run looks like a dead owner
    # and gets re-dispatched / stale-removed out from under the live run.
    _job_schedule = job.get("schedule")
    _is_oneshot = isinstance(_job_schedule, dict) and _job_schedule.get("kind") == "once"
    _run_claim = job.get("run_claim")
    _run_claim_owner = str(_run_claim.get("by") or "") if isinstance(_run_claim, dict) else ""
    _last_claim_heartbeat = time.monotonic()

    def _abort_if_fire_claim_lost() -> None:
        if cancel_event is None or not cancel_event.is_set():
            return
        if agent is not None and hasattr(agent, "interrupt"):
            agent.interrupt("Cron fire claim ownership was lost")
        raise RuntimeError(f"Cron job '{job_name}' lost its durable fire claim ownership")

    def _heartbeat_run_claim_if_due():
        nonlocal _last_claim_heartbeat
        if not _is_oneshot or not _run_claim_owner:
            return
        _mono = time.monotonic()
        if _mono - _last_claim_heartbeat < _RUN_CLAIM_HEARTBEAT_SECONDS:
            return
        _last_claim_heartbeat = _mono
        try:
            heartbeat_run_claim(job_id, expected_owner=_run_claim_owner)
        except Exception:
            logger.debug("Job '%s': run_claim heartbeat failed", job_name, exc_info=True)

    _cron_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    # Carry scheduler-scoped ContextVar state (e.g. env passthrough) into the worker thread.
    _cron_context = contextvars.copy_context()
    _cron_future = _cron_pool.submit(
        _cron_context.run, agent.run_conversation, prompt, task_id=task_id,
    )
    _inactivity_timeout = False
    _watch_stop = threading.Event()

    def _idle_seconds() -> float:
        if not hasattr(agent, "get_activity_summary"):
            return 0.0
        try:
            _act = agent.get_activity_summary()
            return float(_act.get("seconds_since_activity", 0.0) or 0.0)
        except Exception:
            return 0.0

    def _watch_inactivity() -> None:
        nonlocal _inactivity_timeout
        if _cron_inactivity_limit is None:
            return
        if _inactivity_watchdog_loop(
            get_idle_seconds=_idle_seconds,
            limit_s=_cron_inactivity_limit,
            poll_s=_POLL_INTERVAL,
            stop=_watch_stop,
            future_done=_cron_future.done,
        ):
            _inactivity_timeout = True

    _watch_thread = threading.Thread(
        target=_watch_inactivity,
        name=f"cron-inactivity-{str(job_id)[:8]}",
        daemon=True,
    )
    try:
        if _cron_inactivity_limit is not None:
            # Separate daemon thread so a hung get_activity_summary can't stop the limit firing.
            _watch_thread.start()
        if _cron_inactivity_limit is None and not _is_oneshot and cancel_event is None:
            result = _cron_future.result()
        else:
            result = None
            while True:
                done, _ = concurrent.futures.wait({_cron_future}, timeout=_POLL_INTERVAL)
                if done:
                    _abort_if_fire_claim_lost()
                    result = _cron_future.result()
                    break
                if _inactivity_timeout:
                    break
                _abort_if_fire_claim_lost()
                _heartbeat_run_claim_if_due()
    except Exception:
        _cron_pool.shutdown(wait=False, cancel_futures=True)
        raise
    finally:
        _watch_stop.set()
        _cron_pool.shutdown(wait=False, cancel_futures=True)

    if _inactivity_timeout:
        _activity = {}
        if hasattr(agent, "get_activity_summary"):
            with contextlib.suppress(Exception):
                _activity = agent.get_activity_summary()
        _last_desc = _activity.get("last_activity_desc", "unknown")
        _secs_ago = _activity.get("seconds_since_activity", 0)
        logger.error(
            "Job '%s' idle for %.0fs (inactivity limit %.0fs) "
            "| last_activity=%s | iteration=%s/%s | tool=%s",
            job_name, _secs_ago, _cron_inactivity_limit,
            _last_desc, _activity.get("api_call_count", 0), _activity.get("max_iterations", 0),
            _activity.get("current_tool") or "none",
        )
        request_hard_interrupt(agent, "Cron job timed out (inactivity)")
        raise TimeoutError(
            f"Cron job '{job_name}' idle for "
            f"{int(_secs_ago)}s (limit {int(_cron_inactivity_limit)}s) "
            f"— last activity: {_last_desc}"
        )

    if not isinstance(result, dict):
        raise RuntimeError(
            f"agent.run_conversation returned {type(result).__name__} instead of dict: {result!r}"
        )
    return result


def _final_response_from_result(result: dict, job_id: str, job_name: str, AIAgent) -> str:
    """Turn a ``run_conversation`` result into the deliverable final response.

    Raises RuntimeError on `failed=True`/`completed=False`: the error text may sit in
    `final_response` and would otherwise be delivered as the reply with the job marked ok.
    """
    turn_exit_reason = str(result.get("turn_exit_reason") or "")
    final_response_text = (result.get("final_response") or "").strip()
    max_iteration_summary = (
        result.get("failed") is not True
        and result.get("completed") is False
        and turn_exit_reason.startswith("max_iterations_reached(")
        and bool(final_response_text)
    )
    if result.get("failed") is True or (result.get("completed") is False and not max_iteration_summary):
        raise RuntimeError(result.get("error") or final_response_text or "agent reported failure")
    if max_iteration_summary:
        logger.warning(
            "Job '%s' reached the iteration limit but produced a final fallback response; "
            "delivering the response instead of failing the cron run",
            job_name,
        )

    final_response = result.get("final_response", "") or ""
    # Repair model-mangled computer_use media paths before delivery (fail-open, as in gateway).
    if final_response:
        from gateway.media_repair import repair_explicit_computer_use_media_paths

        final_response = repair_explicit_computer_use_media_paths(
            final_response, result.get("messages", []),
        )
    if final_response.strip() == "(No response generated)":
        final_response = ""
    # The "⚠️ No reply" turn-completion explainer would be delivered as a cron warning; detect it
    # via the same formatter and treat as empty so cron stays silent on abnormal empty turns.
    if final_response.strip() and turn_exit_reason:
        # Render every persistence-cause variant or cause-refined text slips through.
        _explainer_variants = []
        try:
            from hermes_state import PERSISTENCE_ERROR_CAUSES as _causes
        except Exception:
            _causes = ("locked", "disk", "unknown")
        for _cause in (None, *_causes):
            try:
                _variant = AIAgent._format_turn_completion_explanation(turn_exit_reason, _cause)
            except TypeError:
                try:
                    _variant = AIAgent._format_turn_completion_explanation(turn_exit_reason)
                except Exception:
                    _variant = ""
            except Exception:
                _variant = ""
            if _variant:
                _explainer_variants.append(_variant.strip())
        if final_response.strip() in _explainer_variants:
            logger.info(
                "Job '%s': abnormal empty turn (%s) — suppressing explainer for cron delivery",
                job_id, turn_exit_reason,
            )
            final_response = ""
    return final_response


def _finalize_cron_session(session_db, agent, job_id: str, job_name: str, cron_session_id: str) -> None:
    """Title, classify, end and release the cron session after the agent turn has returned."""
    # Bound every DB op so storage failure cannot hold the dispatch guard.
    _session_db = _BoundedCronSessionDB(session_db, job_id)
    # Compression may have rotated the run onto a continuation: finalize that, not the stale cron
    # id. SessionDB lineage is authoritative; agent.session_id is only a fail-safe.
    _final_cron_session_id = cron_session_id
    try:
        _compression_tip = _session_db.get_compression_tip(cron_session_id)
        if _compression_tip:
            _final_cron_session_id = _compression_tip
    except (Exception, KeyboardInterrupt) as e:
        with contextlib.suppress((Exception, KeyboardInterrupt)):
            _agent_session_id = getattr(agent, "session_id", None)
            if _agent_session_id:
                _final_cron_session_id = _agent_session_id
        logger.debug("Job '%s': failed to resolve cron compression tip: %s", job_id, e)
    # Title must persist BEFORE end_session()/close(). Run-time suffix keeps it unique against the
    # sessions.title index; the fallbacks below guarantee a non-blank title.
    try:
        _title_base = " ".join(job_name.split())[:60].strip() or f"cron {job_id}"
        _cron_title = f"{_title_base} · {_hermes_now().strftime('%b %d %H:%M')}"
        if not _set_cron_session_title(_session_db, _final_cron_session_id, _cron_title):
            _set_cron_session_title(_session_db, _final_cron_session_id, f"cron {job_id}")
    except (Exception, KeyboardInterrupt) as e:
        logger.debug("Job '%s': failed to set cron session title: %s", job_id, e)
        # Never leave the session untitled.
        for _fallback in (
            getattr(_session_db, "get_next_title_in_lineage", lambda b: b)(f"cron {job_id}"),
            f"cron {job_id} {_final_cron_session_id[-6:]}",
        ):
            try:
                if _set_cron_session_title(_session_db, _final_cron_session_id, _fallback):
                    break
            except (Exception, KeyboardInterrupt):
                continue
    # Book cron_complete only when the last row is a real assistant reply ([SILENT] counts). Only a
    # POSITIVELY recognized bad status downgrades (keep tuple in sync with
    # session_lifecycle_statuses); unknown values / probe failures fail OPEN.
    _end_reason = "cron_complete"
    try:
        _statuses = _session_db.session_lifecycle_statuses([_final_cron_session_id])
        _lifecycle = _statuses.get(_final_cron_session_id)
        if _lifecycle in ("interrupted", "error", "empty"):
            _end_reason = "cron_incomplete_no_output"
            logger.warning(
                "Job '%s': session ended without a final assistant "
                "message (lifecycle=%s) — booking run as %s",
                job_id, _lifecycle, _end_reason,
            )
    except (Exception, KeyboardInterrupt) as e:
        logger.debug("Job '%s': session lifecycle classification failed: %s", job_id, e)
    try:
        _session_db.end_session(_final_cron_session_id, _end_reason)
    except (Exception, KeyboardInterrupt) as e:
        logger.debug("Job '%s': failed to end session: %s", job_id, e)
    try:
        from hermes_state import release_or_close
        release_or_close(_session_db)
    except (Exception, KeyboardInterrupt) as e:
        logger.debug("Job '%s': failed to close SQLite session store: %s", job_id, e)


def _run_doc_header(job: dict, title: str, job_id: str, prompt: str) -> str:
    """Header of the persisted run document (title, ids, schedule, prompt)."""
    return (
        f"# Cron Job: {title}\n\n"
        f"**Job ID:** {job_id}\n"
        f"**Run Time:** {_hermes_now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"**Schedule:** {job.get('schedule_display', 'N/A')}\n\n"
        f"## Prompt\n\n{prompt}\n\n"
    )


def run_job(
    job: dict,
    *,
    defer_agent_teardown: Optional[list] = None,
    extra_prompt: Optional[str] = None,
    cancel_event: Optional[_CancelEventLike] = None,
    execution_id: Optional[str] = None,
) -> tuple[bool, str, str, Optional[str]]:
    """Execute a single cron job. Returns (success, full_output_doc, final_response, error).

    ``defer_agent_teardown``: if a list, the live agent is appended instead of torn down in
    ``finally``; the caller MUST call ``_teardown_cron_agent(agent)`` AFTER delivery (delivery
    against a torn-down async client fails). ``extra_prompt``: per-fire context, never persisted.
    """
    job_id = job["id"]
    job_name = str(job.get("name") or job.get("prompt") or job_id or "cron job")

    # Fail closed on a corrupt config.yaml: defaults would let auto-detection bill a provider the
    # user never chose. no_agent jobs are exempt. Escape hatch: HERMES_IGNORE_USER_CONFIG=1.
    if not job.get("no_agent"):
        from hermes_cli.config import (
            InvalidUserConfigError,
            require_parseable_user_config,
        )

        try:
            require_parseable_user_config()
        except InvalidUserConfigError as exc:
            logger.error("Job '%s': refusing to run — %s", job_id, exc)
            return (False, f"# Cron Job: {job_name}\n\nError: {exc}\n", "", str(exc))

    # no_agent short-circuits BEFORE importing run_agent / opening SessionDB.
    if job.get("no_agent"):
        return _run_no_agent_job(job, job_id, job_name, cancel_event)

    # Legacy / hand-edited job with nothing to run: pause it instead of waking the LLM every fire.
    from cron.jobs import EMPTY_PAYLOAD_ERROR, job_payload_is_empty

    if job_payload_is_empty(job):
        return _block_and_pause_job(job_id, job_name, EMPTY_PAYLOAD_ERROR)

    _early, extra_prompt = _apply_monitor_gate(job, job_id, job_name, extra_prompt)
    if _early is not None:
        return _early

    from run_agent import AIAgent

    # Wake-gate: run the pre-check script BEFORE building the prompt; its result is passed into
    # _build_job_prompt so the script runs only once.
    prerun_script = None
    script_path = job.get("script")
    if script_path:
        prerun_script = _run_job_script_with_claim_heartbeat(
            job, script_path, cancel_event=cancel_event,
        )
        _ran_ok, _script_output = prerun_script
        if _ran_ok and not _parse_wake_gate(_script_output):
            logger.info("Job '%s' (ID: %s): wakeAgent=false, skipping agent run", job_name, job_id)
            silent_doc = (
                f"# Cron Job: {job_name}\n\n"
                f"**Job ID:** {job_id}\n"
                f"**Run Time:** {_hermes_now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                "Script gate returned `wakeAgent=false` — agent skipped.\n"
            )
            return True, silent_doc, SILENT_MARKER, None

    try:
        prompt = _build_job_prompt(job, prerun_script=prerun_script, extra_prompt=extra_prompt)
    except CronPromptInjectionBlocked as block_exc:
        # Injection scanner tripped: refuse this tick and tell the operator WHY.
        logger.warning(
            "Job '%s' (ID: %s): blocked by prompt-injection scanner — %s",
            job_name, job_id, block_exc,
        )
        blocked_doc = (
            f"# Cron Job: {job_name}\n\n"
            f"**Job ID:** {job_id}\n"
            f"**Run Time:** {_hermes_now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"**Status:** BLOCKED\n\n"
            "The assembled prompt (user prompt + loaded skill content) tripped "
            "the cron injection scanner and the agent was NOT run.\n\n"
            f"**Scanner result:** {block_exc}\n\n"
            "Audit the skill(s) attached to this job for prompt-injection "
            "payloads or invisible-unicode markers. If the skill is legitimate "
            "and the match is a false positive, rephrase the content to avoid "
            "the threat pattern (`tools/cronjob_tools.py::_CRON_THREAT_PATTERNS`)."
        )
        return False, blocked_doc, "", str(block_exc)
    if prompt is None:
        logger.info("Job '%s': script produced no output, skipping AI call.", job_name)
        return True, "", SILENT_MARKER, None
    _cron_session_id = f"cron_{job_id}_{_hermes_now().strftime('%Y%m%d_%H%M%S')}"

    logger.info("Running job '%s' (ID: %s)", job_name, job_id)
    logger.info("Prompt: %s", prompt[:100])

    agent = None
    model = ""

    # ContextVars, not os.environ (process-global), so parallel jobs don't clobber each other.
    from gateway.session_context import set_session_vars, clear_session_vars, _VAR_MAP

    # Do NOT seed HERMES_SESSION_* from job["origin"]: it is delivery metadata, not a sender, and
    # terminal/tts/skills/send_message tools would act as if the origin user were driving the
    # agent. Delivery reads job["origin"] and HERMES_CRON_AUTO_DELIVER_* directly, so blanking is
    # safe. Resolve workdir BEFORE set_session_vars so it owns the _SESSION_CWD set/clear.
    _job_workdir = _resolve_job_workdir(job, job_id)

    _ctx_tokens = set_session_vars(
        platform="",
        chat_id="",
        chat_name="",
        # Cron can't receive completions after its turn; async delegation output could otherwise
        # route to an unrelated chat via the ambient session key. Stateless => inline delegation.
        async_delivery=False,
        cwd=_job_workdir or "",
    )
    _cron_delivery_vars = (
        "HERMES_CRON_AUTO_DELIVER_PLATFORM",
        "HERMES_CRON_AUTO_DELIVER_CHAT_ID",
        "HERMES_CRON_AUTO_DELIVER_THREAD_ID",
    )
    for _var_name in _cron_delivery_vars:
        _VAR_MAP[_var_name].set("")

    # Bind workdir to the per-run task id (tool-layer cwd authority) instead of mutating global
    # TERMINAL_CWD; _SESSION_CWD above remains the prompt/context-file authority.
    _cron_task_id = (
        f"cron:{job_id}:"
        f"{execution_id or job.get('execution_id') or uuid.uuid4().hex}"
    )
    from tools.terminal_tool import clear_session_cwd as _clear_tool_session_cwd
    from tools.terminal_tool import record_session_cwd as _record_tool_session_cwd
    if _job_workdir:
        _record_tool_session_cwd(_cron_task_id, _job_workdir)
    _cron_session_var = _VAR_MAP["HERMES_CRON_SESSION"]
    _cron_session_token = None
    _non_dispatcher_token = None
    _session_db = None
    try:

        # Scope cron approval policy; the finally RESETS via token (pinning "" would suppress the
        # legacy os.environ fallback used by standalone entrypoints/tests).
        _cron_session_token = _cron_session_var.set("1")

        # Mark NOT the kanban worker: a worker's cronjob(action="run") lands here with
        # HERMES_KANBAN_TASK in env, and an unrelated job could close the worker's task. Must be a
        # ContextVar, NOT an os.environ clear (env is shared with the worker heartbeat and
        # concurrent jobs); copy_context() carries it into the agent thread.
        _non_dispatcher_token = enter_non_dispatcher_owned_context()
        if _job_workdir:
            logger.info("Job '%s': using task-scoped workdir %s", job_id, _job_workdir)

        # Re-read .env every run; reset the secret-source cache FIRST or a Bitwarden/BSM-backed
        # secret is never re-resolved (only the placeholder reloads -> 401s).
        from hermes_cli.env_loader import (
            load_hermes_dotenv,
            reset_secret_source_cache,
        )
        reset_secret_source_cache()
        load_hermes_dotenv(hermes_home=_get_hermes_home())

        delivery_target = _resolve_delivery_target(job)
        if delivery_target:
            _VAR_MAP["HERMES_CRON_AUTO_DELIVER_PLATFORM"].set(delivery_target["platform"])
            _VAR_MAP["HERMES_CRON_AUTO_DELIVER_CHAT_ID"].set(str(delivery_target["chat_id"]))
            _VAR_MAP["HERMES_CRON_AUTO_DELIVER_THREAD_ID"].set(
                ""
                if delivery_target.get("thread_id") is None
                else str(delivery_target["thread_id"])
            )

        jc = _load_cron_job_config(job, job_id, job_name)
        _cfg = jc.cfg
        model = jc.model

        prefill_messages = _load_prefill_messages(_cfg, job_id)

        # resolve_turn_limit() honors none/unlimited (sys.maxsize) and explicit 0 / null.
        from hermes_cli.config import resolve_turn_limit as _resolve_turn_limit
        _mt = _cfg.get("agent", {}).get("max_turns")
        if _mt is None:
            _mt = _cfg.get("max_turns")
        max_iterations = _resolve_turn_limit(_mt)

        pr = _cfg.get("provider_routing") or {}

        # Runtime backstop (CWE-200/522): fail closed BEFORE resolution on a provider/base_url pair
        # that would ship a stored credential off-host; hand-written jobs bypass create-time checks.
        _guard_job_credential_exfil(job)

        _blocked = _preflight_or_block(job, job_id, job_name, _cfg)
        if _blocked is not None:
            return _blocked

        primary_model_for_drift = model
        runtime, model, primary_provider_for_drift = _resolve_job_runtime(job, job_id, jc)

        reasoning_config = _resolve_job_reasoning_config(
            job, _cfg if isinstance(_cfg, dict) else {}, str(model)
        )
        _check_model_drift(
            job, job_id, _cfg, runtime, primary_provider_for_drift, primary_model_for_drift,
        )

        fallback_model = get_fallback_chain(_cfg) or None
        credential_pool = _load_credential_pool(runtime, job_id)
        # MCP servers must be registered before AIAgent is constructed.
        _init_cron_mcp_tools(job_id)

        # Open state.db only after every early-return gate has passed.
        _session_db = _open_cron_session_db(job)

        agent = AIAgent(
            model=model,
            api_key=runtime.get("api_key"),
            base_url=runtime.get("base_url"),
            provider=runtime.get("provider"),
            requested_provider=runtime.get("requested_provider"),
            api_mode=runtime.get("api_mode"),
            request_overrides=runtime.get("request_overrides"),
            acp_command=runtime.get("command"),
            acp_args=runtime.get("args"),
            max_iterations=max_iterations,
            reasoning_config=reasoning_config,
            prefill_messages=prefill_messages,
            fallback_model=fallback_model,
            credential_pool=credential_pool,
            providers_allowed=pr.get("only"),
            providers_ignored=pr.get("ignore"),
            providers_order=pr.get("order"),
            provider_sort=pr.get("sort"),
            openrouter_min_coding_score=(_cfg.get("openrouter") or {}).get("min_coding_score"),
            enabled_toolsets=_resolve_cron_enabled_toolsets(job, _cfg),
            disabled_toolsets=_resolve_cron_disabled_toolsets(_cfg),
            quiet_mode=True,
            # Project context files only with a configured workdir; SOUL.md always.
            skip_context_files=not bool(_job_workdir),
            load_soul_identity=True,
            skip_memory=False,
            skip_background_review=True,  # Cron has no human-in-the-loop need for skill/memory review forks (~30K tok/event)
            platform="cron",
            session_id=_cron_session_id,
            session_db=_session_db,
        )

        _audit_fire_id = uuid.uuid4().hex
        _audit_t_start = time.monotonic()

        def _audit(result: dict, error: Optional[str]) -> None:
            """One usage_audit.jsonl line per fire."""
            _write_usage_audit({
                "ts": _utcnow_iso_ms(),
                "job_id": job_id,
                "fire_id": _audit_fire_id,
                "prompt_tokens": result.get("prompt_tokens"),
                "completion_tokens": result.get("completion_tokens"),
                "total_tokens": result.get("total_tokens"),
                "response_silent": bool(result.get("response_silent")),
                "deliver_target": job.get("deliver"),
                "model": model or None,
                "duration_ms": int((time.monotonic() - _audit_t_start) * 1000),
                "error": error,
            })

        result = _run_agent_with_watchdog(
            agent, prompt, job, job_id, job_name, _cron_task_id, cancel_event,
        )
        final_response = _final_response_from_result(result, job_id, job_name, AIAgent)
        # Keep final_response clean for delivery logic (empty = no delivery).
        logged_response = final_response if final_response else "(No response generated)"
        output = _run_doc_header(job, job_name, job_id, prompt) + f"## Response\n\n{logged_response}\n"
        logger.info("Job '%s' completed successfully", job_name)
        _audit(dict(result, response_silent=_is_cron_silence_response(final_response or "")), None)
        return True, output, final_response, None

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.exception("Job '%s' failed: %s", job_name, error_msg)
        # _audit is unbound if we failed before the agent ran; the audit write must never raise.
        if "_audit" in locals():
            _audit({}, error_msg)
        output = (
            _run_doc_header(job, f"{job_name} (FAILED)", job_id, prompt)
            + f"## Error\n\n```\n{error_msg}\n```\n"
        )
        return False, output, "", error_msg

    finally:
        _clear_tool_session_cwd(_cron_task_id)
        # clear_session_vars also clears _SESSION_CWD.
        clear_session_vars(_ctx_tokens)
        if _cron_session_token is not None:
            _cron_session_var.reset(_cron_session_token)
        if _non_dispatcher_token is not None:
            exit_non_dispatcher_owned_context(_non_dispatcher_token)
        for _var_name in _cron_delivery_vars:
            _VAR_MAP[_var_name].set("")
        if _session_db:
            _finalize_cron_session(_session_db, agent, job_id, job_name, _cron_session_id)
        # Tear down the ephemeral agent or the gateway leaks fds per tick (EMFILE). With deferred
        # teardown, hand the live agent back: delivery needs a live async client.
        if defer_agent_teardown is not None:
            if agent is not None:
                defer_agent_teardown.append(agent)
        else:
            _teardown_cron_agent(agent, job_id)


def _teardown_cron_agent(
    agent, job_id: str, *, timeout_seconds: Optional[float] = None
) -> None:
    """Release an ephemeral cron agent's async resources within a hard bound.

    Split out of ``run_job``'s ``finally`` so a caller deferring teardown until after delivery runs
    the identical cleanup. Bounded because this runs outside the agent inactivity watchdog.
    """
    def _cleanup_agent() -> None:
        try:
            if agent is not None:
                agent.close()
        except (Exception, KeyboardInterrupt) as e:
            logger.debug("Job '%s': failed to close agent resources: %s", job_id, e)
        # Worker-thread event loop dies with the executor; reap httpx clients cached under it.
        try:
            from agent.auxiliary_client import cleanup_stale_async_clients
            cleanup_stale_async_clients()
        except Exception as e:
            logger.debug("Job '%s': failed to reap stale auxiliary clients: %s", job_id, e)

    _run_cron_cleanup_with_timeout(
        _cleanup_agent,
        job_id=job_id,
        label="agent resource teardown",
        timeout_seconds=timeout_seconds,
    )


def _run_with_fire_claim_heartbeat(job: dict, run) -> bool:
    """Run ``run`` while keeping this job's owned durable fire claim fresh."""
    claim = job.get("fire_claim")
    owner = str(claim.get("by") or "") if isinstance(claim, dict) else ""
    if not owner:
        return run(None)

    job_id = str(job.get("id") or "")
    stop = threading.Event()
    lost_ownership = threading.Event()

    def _finish_unstarted(error: str) -> None:
        execution_id = job.get("execution_id")
        if not execution_id:
            return
        try:
            finish_execution(execution_id, success=False, error=error)
        except Exception:
            logger.warning(
                "Job '%s': failed to close unstarted execution ledger row",
                job_id,
                exc_info=True,
            )

    try:
        owns_fire_claim = heartbeat_fire_claim(job_id, expected_owner=owner)
    except Exception:
        logger.warning("Job '%s': initial fire_claim validation failed", job_id, exc_info=True)
        _finish_unstarted("Fire claim ownership could not be validated before execution started.")
        return True

    if owns_fire_claim is False:
        logger.warning("Job '%s': fire claim ownership was already lost before execution", job_id)
        _finish_unstarted("Fire claim ownership lost before execution started.")
        return True

    def _heartbeat_loop() -> None:
        last_confirmed = time.monotonic()
        while not stop.wait(_RUN_CLAIM_HEARTBEAT_SECONDS):
            try:
                if not heartbeat_fire_claim(job_id, expected_owner=owner):
                    lost_ownership.set()
                    logger.warning(
                        "Job '%s': fire claim ownership lost; interrupting stale run",
                        job_id,
                    )
                    return
                last_confirmed = time.monotonic()
            except Exception:
                logger.debug("Job '%s': fire_claim heartbeat failed", job_id, exc_info=True)
                if (
                    time.monotonic() - last_confirmed
                    >= _FIRE_CLAIM_HEARTBEAT_GRACE_SECONDS
                ):
                    lost_ownership.set()
                    logger.warning(
                        "Job '%s': fire_claim could not be renewed within %.1fs; "
                        "interrupting uncertain run",
                        job_id,
                        _FIRE_CLAIM_HEARTBEAT_GRACE_SECONDS,
                    )
                    return

    heartbeat_thread = _start_heartbeat_thread(
        _heartbeat_loop, "cron-fire-claim-heartbeat",
        lambda: logger.warning(
            "Job '%s': could not start fire_claim heartbeat", job_id, exc_info=True,
        ),
    )
    if heartbeat_thread is None:
        _finish_unstarted("Fire claim heartbeat could not be started; execution was not run.")
        return True

    try:
        return run(lost_ownership)
    finally:
        stop.set()
        heartbeat_thread.join(timeout=1.0)


def run_one_job(
    job: dict,
    *,
    adapters=None,
    loop=None,
    verbose: bool = False,
    extra_prompt: Optional[str] = None,
    cancel_event: Optional[_CancelEventLike] = None,
) -> bool:
    """Run ONE due job end-to-end: execute → save output → deliver → mark.

    Shared firing body for BOTH the built-in ticker and external providers' ``fire_due``. Does NOT
    decide due-ness or acquire the initial claim (callers use the same store CAS); does keep the
    claim alive. Returns True if processed (job failure is recorded via ``mark_job_run``), False
    only if processing raised. ``cancel_event``: optional transport-level cancel (dashboard drain).
    """
    if extra_prompt is None:
        # Gateway-forwarded manual run stamps its prompt on the job via trigger_job; the fire that
        # consumes the manual occurrence picks it up here. Single-fire: mark_job_run clears it.
        _stamped = job.get("manual_run_prompt")
        if _stamped and job.get("manual_run_at"):
            extra_prompt = str(_stamped)
    claim = job.get("fire_claim")
    fire_owner = str(claim.get("by") or "") if isinstance(claim, dict) else ""
    execution_token = object()
    profile_home = _get_hermes_home().resolve()
    with _running_lock:
        _running_fire_owners.setdefault(job["id"], {})[execution_token] = (
            fire_owner or None,
            profile_home,
        )
    try:
        return _run_with_fire_claim_heartbeat(
            job,
            lambda lost_ownership: _run_one_job_body(
                job,
                adapters=adapters,
                loop=loop,
                verbose=verbose,
                extra_prompt=extra_prompt,
                fire_claim_lost=(
                    _CombinedCancelEvent(lost_ownership, cancel_event)
                    if cancel_event is not None
                    else lost_ownership
                ),
                execution_token=execution_token,
            ),
        )
    finally:
        with _running_lock:
            executions = _running_fire_owners.get(job["id"])
            if executions is not None:
                executions.pop(execution_token, None)
                if not executions:
                    _running_fire_owners.pop(job["id"], None)


_OWNERSHIP_LOST_INTERRUPTED = "Interrupted by shutdown before terminal completion."


def _record_fire_ownership_lost(job_id: str, fire_owner: Optional[str], execution_id: str) -> None:
    """Bookkeeping after fire-claim ownership loss. A transport-level cancel (dashboard drain) is
    not a real loss — we still own the claim, so record the interruption via the owner-fenced
    terminal write instead of leaving fire_claim/last_status stale; otherwise discard."""
    if fire_owner is not None and heartbeat_fire_claim(job_id, expected_owner=fire_owner):
        mark_job_run(job_id, False, _OWNERSHIP_LOST_INTERRUPTED, expected_fire_owner=fire_owner)
        finish_execution(execution_id, success=False, error=_OWNERSHIP_LOST_INTERRUPTED)
    else:
        finish_execution(
            execution_id,
            success=False,
            error="Fire claim ownership lost; stale result was discarded.",
        )


def _classify_delivery_outcome(
    *, delivery_error, should_deliver: bool, unresolved_origin: bool,
    normalized_deliver: str, incident_acked: bool, success: bool,
) -> str:
    if delivery_error:
        return "failed"
    if should_deliver and unresolved_origin:
        return "not_configured"
    if should_deliver and normalized_deliver != "local":
        return "delivered"
    if incident_acked and not success:
        # Failure ping withheld: operator acked this exact signature (vs. plain "suppressed").
        return "suppressed_acked"
    return "suppressed"


def _compose_run_delivery(
    job: dict, *, success: bool, error, final_response: str, output_file,
) -> tuple[str, bool, bool, bool, Optional[str]]:
    """Build the text to deliver for a finished run.

    Returns ``(deliver_content, blocked_config, silent_alert, incident_acked, failure_incident_id)``.
    ``silent_alert``: an alert-once marker says the operator was already told; deliver nothing.
    """
    err = str(error) if error else ""
    # Failed jobs always deliver, except blocked-config / drift-skip runs, which alert exactly ONCE.
    blocked_config_silent = BLOCKED_CONFIG_SILENT_MARKER in err
    blocked_config = blocked_config_silent or BLOCKED_CONFIG_MARKER in err
    drift_skip_silent = DRIFT_SKIP_SILENT_MARKER in err
    drift_skip = drift_skip_silent or DRIFT_SKIP_MARKER in err
    incident_acked = False
    failure_incident_id = None
    if blocked_config and not success:
        # Bypass the generic failure summarizer (its auth/timeout heuristics would mislabel this).
        _pf_text = re.sub(r"\[blocked_config[^\]]*\]\s*", "", err).strip()
        deliver_content = (
            f"⛔ Cron '{job.get('name') or job['id']}' blocked by "
            f"configuration validation (no LLM call was made): "
            f"{_pf_text} "
            "This alert is sent once; the job stays blocked until "
            "the configuration is fixed."
        )
    elif success:
        deliver_content = final_response
    else:
        # Record the job+error signature once; if already acked by the operator, suppress the
        # per-run ping. Best-effort: a ledger failure never breaks delivery.
        incident_acked, failure_incident_id = _upsert_incident_for_failure(
            job, error or "", output_file=output_file
        )
        if incident_acked and not drift_skip:
            deliver_content = ""
        else:
            deliver_content = (
                _summarize_cron_failure_for_delivery(job, error) + _failure_streak_nudge(job)
            )
        if drift_skip:
            # Deliver the guard's message intact (summarizer truncation would eat the remediation
            # command). NOT gated on incident ack: acks silence failure pings, not drift alerts.
            _drift_text = re.sub(r"\[drift_skip[^\]]*\]\s*", "", err).strip()
            deliver_content = f"⚠️ Cron '{job.get('name') or job['id']}' skipped: {_drift_text}"
    return (
        deliver_content, blocked_config, blocked_config_silent or drift_skip_silent,
        incident_acked, failure_incident_id,
    )


def _run_one_job_body(
    job: dict,
    *,
    adapters=None,
    loop=None,
    verbose: bool = False,
    extra_prompt: Optional[str] = None,
    fire_claim_lost: Optional[_CancelEventLike] = None,
    execution_token: Optional[object] = None,
) -> bool:
    claim = job.get("fire_claim")
    fire_owner = str(claim.get("by") or "") if isinstance(claim, dict) else None

    class _FireClaimLostDuringSideEffect(Exception):
        pass

    def _side_effect_fence():
        if fire_owner is None:
            return contextlib.nullcontext(True)
        return fire_claim_fence(job["id"], expected_owner=fire_owner)

    def _fire_claim_ownership_lost() -> bool:
        if fire_claim_lost is not None and fire_claim_lost.is_set():
            return True
        if fire_owner is None:
            return False
        try:
            if heartbeat_fire_claim(job["id"], expected_owner=fire_owner):
                return False
        except Exception:
            logger.debug(
                "Job '%s': fire_claim ownership validation failed",
                job["id"],
                exc_info=True,
            )
            return False
        if fire_claim_lost is not None:
            fire_claim_lost.set()
        return True

    execution_id = job.get("execution_id")
    if not execution_id:
        execution_id = create_execution(job["id"], source="direct")["id"]
    delivery_attempted = False
    delivery_error = None
    incident_acked = False
    failure_incident_id = None
    from agent.secret_scope import (
        build_profile_secret_scope,
        reset_secret_scope,
        set_secret_scope,
    )

    _scope_token = None
    _terminal_scope_token = None
    try:
        # Commit a finite one-shot's dispatch BEFORE its side effect so a tick dying mid-run cannot
        # re-fire it forever on restart. No-op for recurring/infinite jobs (at-most-times).
        if not claim_dispatch(job["id"]):
            logger.info(
                "Job '%s': one-shot dispatch limit reached — skipping",
                job.get("name", job["id"]),
            )
            finish_execution(
                execution_id,
                success=False,
                error="Dispatch claim rejected; execution was not started.",
            )
            return True  # not an error — already handled/removed

        mark_execution_running(execution_id)

        # get_secret() fails closed outside a scope; the ticker thread has none. Delivery adapters
        # resolve credentials, so the scope must span delivery too (reset in the outer finally).
        _scope_token = set_secret_scope(build_profile_secret_scope(_get_hermes_home()))
        # Same for terminal policy (gateway/run.py _profile_runtime_scope): else the ticker reads
        # process-global TERMINAL_* env a concurrent profile pinned. Resolution failure installs a
        # refusal scope — terminal execution raises instead of using the launch process's policy.
        from tools.terminal_scope import (
            install_profile_terminal_scope,
        )

        _terminal_scope_token = install_profile_terminal_scope(_get_hermes_home())
        # Defer agent teardown until AFTER delivery; closing first races the live send against a
        # torn-down async client. run_job hands the agent back via this list.
        _deferred_agents: list = []

        def _teardown_deferred() -> None:
            for _deferred_agent in _deferred_agents:
                _teardown_cron_agent(_deferred_agent, job["id"])

        _run_kwargs = {
            "defer_agent_teardown": _deferred_agents,
            "extra_prompt": extra_prompt,
            "execution_id": execution_id,
        }
        if fire_claim_lost is not None:
            _run_kwargs["cancel_event"] = fire_claim_lost
        try:
            success, output, final_response, error = run_job(job, **_run_kwargs)
        except BaseException:
            # run_job hands back the agent even when raising; tear down so a failed run never leaks.
            # BaseException so KeyboardInterrupt/SystemExit mid-run still trigger teardown.
            _teardown_deferred()
            raise

        if _fire_claim_ownership_lost():
            _teardown_deferred()
            _record_fire_ownership_lost(job["id"], fire_owner, execution_id)
            return True

        # Agent is still live through delivery; wrap ALL of save/compose/deliver in try/finally so a
        # raise anywhere still tears the deferred agent down.
        blocked_config = False
        side_effect_ownership_lost = False
        try:
            with _side_effect_fence() as owns_output:
                if not owns_output:
                    raise _FireClaimLostDuringSideEffect
                output_file = save_job_output(job["id"], output)
            if verbose:
                logger.info("Output saved to: %s", output_file)

            # A shutdown-killed tool subprocess can leave a plausible final_response from truncated
            # output; force the honest "interrupted" failure path. Peek-only (consumed later).
            if success and _is_interrupted(job["id"], execution_token):
                success = False
                error = (
                    "Interrupted by gateway shutdown before the run finished "
                    "(tool subprocess was killed mid-flight)."
                )

            (
                deliver_content, blocked_config, _silent_alert,
                incident_acked, failure_incident_id,
            ) = _compose_run_delivery(
                job, success=success, error=error, final_response=final_response,
                output_file=output_file,
            )
            # Whitespace-only == empty: skip delivery; the guard below marks it a soft failure.
            should_deliver = bool(deliver_content.strip()) and not _silent_alert
            unresolved_origin = False
            # Not a substring check: bare "SILENT"/"NO_REPLY" or a report quoting "[SILENT]" must
            # not be swallowed; bracketed-prefix / trailing-line tolerance is kept.
            if should_deliver and success and _is_cron_silence_response(deliver_content):
                logger.info("Job '%s': agent returned %s — skipping delivery", job["id"], SILENT_MARKER)
                should_deliver = False

            if should_deliver and _fire_claim_ownership_lost():
                should_deliver = False
                logger.warning(
                    "Job '%s': skipping delivery after fire claim ownership loss",
                    job["id"],
                )

            if should_deliver:
                unresolved_origin = (
                    _normalize_deliver_value(_delivery_lane_value(job, for_failure=not success))
                    == "origin"
                    and not _resolve_delivery_targets(job, for_failure=not success)
                )
                try:
                    with _side_effect_fence() as owns_delivery:
                        if not owns_delivery:
                            raise _FireClaimLostDuringSideEffect
                        delivery_attempted = True
                        delivery_error = _deliver_result(
                            job,
                            deliver_content,
                            adapters=adapters,
                            loop=loop,
                            # Failure summaries (and drift/blocked-config alerts
                            # composed into deliver_content on the failure path)
                            # honor the job's failure_deliver override (NS-788).
                            for_failure=not success,
                        )
                except Exception as de:
                    if isinstance(de, _FireClaimLostDuringSideEffect):
                        raise
                    delivery_error = str(de)
                    logger.error("Delivery failed for job %s: %s", job["id"], de)
        except _FireClaimLostDuringSideEffect:
            side_effect_ownership_lost = True
        finally:
            # Every path must tear down deferred agent(s) so they never leak subprocesses/clients.
            _teardown_deferred()

        if side_effect_ownership_lost or _fire_claim_ownership_lost():
            _record_fire_ownership_lost(job["id"], fire_owner, execution_id)
            return True

        # Empty final_response is a soft failure so last_status is not "ok".
        if success and not final_response.strip():
            success = False
            error = "Agent completed but produced empty response (model error, timeout, or misconfiguration)"

        interrupted = _consume_interrupted_flag(job["id"], execution_token)
        if interrupted:
            if delivery_error:
                # Shutdown already wrote last_status so mark_job_run is skipped below (a second call
                # would skip a fire or auto-delete the job); note the unsent notice via update_job.
                try:
                    from cron.jobs import update_job
                    update_job(job["id"], {"last_delivery_error": delivery_error})
                except Exception as _rec_err:
                    logger.debug(
                        "Failed recording delivery_error for interrupted job %s: %s",
                        job["id"], _rec_err,
                    )
            finish_execution(
                execution_id,
                success=False,
                error="Interrupted by gateway shutdown before terminal completion.",
            )
            return True

        mark_kwargs = {"delivery_error": delivery_error}
        if fire_owner is not None:
            mark_kwargs["expected_fire_owner"] = fire_owner
        if blocked_config:
            mark_kwargs["status"] = "blocked_config"
        marked = mark_job_run(job["id"], success, error, **mark_kwargs)
        if fire_owner is not None and not marked:
            finish_execution(
                execution_id,
                success=False,
                error="Fire claim ownership lost before terminal completion.",
            )
            return True
        delivery_outcome = _classify_delivery_outcome(
            delivery_error=delivery_error,
            should_deliver=should_deliver,
            unresolved_origin=unresolved_origin,
            # Read the lane the notice was actually routed through (failure_deliver on failure).
            normalized_deliver=_normalize_deliver_value(_delivery_lane_value(job, for_failure=not success)),
            incident_acked=incident_acked,
            success=success,
        )
        if delivery_outcome in ("delivered", "not_configured") and not success:
            # Failure ping left the process (or had a configured target): mark the incident alerted.
            _mark_incident_alerted(failure_incident_id)
        finish_execution(
            execution_id,
            success=success,
            error=error,
            delivery_outcome=delivery_outcome,
        )
        return True

    except BaseException as e:  # noqa: BLE001 — deliberate: see below
        # BaseException, not Exception: CancelledError/KeyboardInterrupt/SystemExit propagate here.
        # Without mark_job_run(False) a finite one-shot is wedged: claim_dispatch consumed
        # repeat.completed but last_run_at is never written. Record first, then re-raise
        # non-Exception. Owner fencing still applies.
        _err_text = str(e) or type(e).__name__
        logger.error(
            "Error processing job %s: %s",
            job["id"],
            _err_text,
            exc_info=(type(e), e, e.__traceback__),
        )
        delivery_outcome = "suppressed"
        # Owner fencing: a stale worker whose claim was taken over (or transport-cancelled) must not
        # send a failure alert on top of the replacement run's; fall through to fenced bookkeeping.
        if (
            isinstance(e, Exception)
            and not delivery_attempted
            and not isinstance(e, _FireClaimLostDuringSideEffect)
            and not _fire_claim_ownership_lost()
        ):
            normalized_deliver = _normalize_deliver_value(_delivery_lane_value(job, for_failure=True))
            # Same ack gate as the normal failure delivery: acked signatures stay silent here too.
            incident_acked, failure_incident_id = _upsert_incident_for_failure(job, _err_text)
            if incident_acked:
                delivery_outcome = "suppressed_acked"
            else:
                try:
                    delivery_attempted = True
                    delivery_error = _deliver_result(
                        job,
                        # Same text as the normal failure delivery: this run also counts toward
                        # failure_streak, so the nudge must leave through here too.
                        _summarize_cron_failure_for_delivery(job, _err_text)
                        + _failure_streak_nudge(job),
                        adapters=adapters,
                        loop=loop,
                        for_failure=True,
                    )
                except Exception as delivery_exc:
                    delivery_error = str(delivery_exc)
                    logger.error("Delivery failed for job %s: %s", job["id"], delivery_exc)
                unresolved_origin = bool(
                    not delivery_error
                    and normalized_deliver == "origin"
                    and not _resolve_delivery_targets(job, for_failure=True)
                )
                delivery_outcome = _classify_delivery_outcome(
                    delivery_error=delivery_error,
                    should_deliver=True,
                    unresolved_origin=unresolved_origin,
                    normalized_deliver=normalized_deliver,
                    incident_acked=False,
                    success=False,
                )
                if delivery_outcome in ("delivered", "not_configured"):
                    _mark_incident_alerted(failure_incident_id)
        try:
            if not _consume_interrupted_flag(job["id"], execution_token):
                mark_kwargs = {}
                if fire_owner is not None:
                    mark_kwargs["expected_fire_owner"] = fire_owner
                if isinstance(e, Exception):
                    mark_kwargs["delivery_error"] = delivery_error
                mark_job_run(job["id"], False, _err_text, **mark_kwargs)
        except Exception as record_err:
            # Never let bookkeeping mask the original interruption.
            logger.error("Failed to record interrupted run for job %s: %s", job["id"], record_err)
        try:
            finish_execution(
                execution_id,
                success=False,
                error=_err_text,
                delivery_outcome=delivery_outcome,
            )
        except Exception as record_err:
            logger.error("Failed to finish execution record for job %s: %s", job["id"], record_err)
        if not isinstance(e, Exception):
            raise
        return False
    finally:
        # Function-level on purpose: must scope delivery, deferred teardown, claim-loss handling and
        # bookkeeping — not just run_job. Do not move into the run block's finally.
        if _scope_token is not None:
            reset_secret_scope(_scope_token)
        if _terminal_scope_token is not None:
            from tools.terminal_scope import reset_terminal_scope

            reset_terminal_scope(_terminal_scope_token)


def _notify_provider_jobs_changed() -> None:
    """Best-effort: tell the active scheduler provider the job set changed. Call AFTER a successful
    store mutation so an external provider can re-provision/cancel the one-shot; no-op for the
    built-in. Kept out of cron/jobs.py (import cycle). Never raises."""
    try:
        from cron.scheduler_provider import resolve_cron_scheduler
        resolve_cron_scheduler().on_jobs_changed()
    except Exception as e:
        logger.debug("on_jobs_changed notify failed: %s", e)


class CronSchedulerRegistrationError(RuntimeError):
    """A job was persisted but its first external trigger was not registered."""

    def __init__(self, job: dict, cause: Exception) -> None:
        self.job = job
        self.cause = cause
        super().__init__(
            f"Cron job '{job['id']}' was saved, but its first scheduler "
            f"registration failed ({type(cause).__name__}). Do not create a "
            "duplicate. Pause/resume or update the job to retry registration."
        )

    def user_message(self) -> str:
        """Human-facing variant for chat/CLI surfaces (no exception class name)."""
        label = self.job.get("name") or self.job["id"]
        return (
            f"Saved cron job '{label}', but couldn't register it with the "
            "external scheduler yet. The job is kept — don't re-create it; "
            "pause/resume or edit it (e.g. via /cron) to retry registration."
        )

    def to_dict(self) -> dict:
        """Return the public partial-failure contract without provider details."""
        return {
            "error": str(self),
            "job_id": self.job["id"],
            "job_saved": True,
            "scheduler_registered": False,
            "retry_create": False,
        }


def create_job_with_scheduler_registration(**kwargs) -> dict:
    """Persist one job and register its first trigger with the active provider."""
    from cron.jobs import create_job
    from cron.scheduler_provider import resolve_cron_scheduler

    job = create_job(**kwargs)
    try:
        resolve_cron_scheduler().register_job(job)
    except Exception as exc:
        raise CronSchedulerRegistrationError(job, exc) from exc
    return job


# Dead-owner reap is throttled (opens the executions ledger). Tests may reset
# _last_dead_owner_reap_at to None to force a reap next tick.
_DEAD_OWNER_REAP_INTERVAL_SECONDS = 300.0
_last_dead_owner_reap_at: Optional[float] = None

# Worktree prune throttle: the cron tick is the only reliably periodic process on gateway boxes.
_WORKTREE_MAINTENANCE_INTERVAL_SECONDS = 6 * 3600.0
_last_worktree_maintenance_at: Optional[float] = None
_worktree_maintenance_lock = threading.Lock()


def _worktree_maintenance_repos() -> List[str]:
    """Repos whose ``.worktrees/`` to keep pruned: the hermes checkout plus job workdir repo roots,
    filtered to those that actually have a ``.worktrees/`` dir."""
    repos: set = set()

    # Hermes source checkout (git installs only; wheel installs have no .git).
    with contextlib.suppress(Exception):
        install_root = Path(__file__).resolve().parent.parent
        if (install_root / ".git").exists():
            repos.add(str(install_root))

    with contextlib.suppress(Exception):
        from cron.jobs import load_jobs

        for job in load_jobs():
            workdir = str(job.get("workdir") or "").strip()
            if not workdir or not Path(workdir).is_dir():
                continue
            try:
                probe = subprocess.run(
                    ["git", "rev-parse", "--show-toplevel"],
                    capture_output=True, text=True, encoding="utf-8",
                    errors="replace", timeout=5, cwd=workdir,
                )
                if probe.returncode == 0 and probe.stdout.strip():
                    repos.add(probe.stdout.strip())
            except Exception:
                continue

    return [r for r in sorted(repos) if (Path(r) / ".worktrees").is_dir()]


def _maybe_run_worktree_maintenance() -> None:
    """Throttled worktree prune from the cron tick, on a daemon thread so the tick never waits on
    git. Same conservative pruner as ``hermes -w`` startup (dirty/unpushed/locked trees untouched).
    Errors never propagate: GC is hygiene, not scheduling."""
    global _last_worktree_maintenance_at
    now = time.monotonic()
    with _worktree_maintenance_lock:
        if (
            _last_worktree_maintenance_at is not None
            and now - _last_worktree_maintenance_at
            < _WORKTREE_MAINTENANCE_INTERVAL_SECONDS
        ):
            return
        _last_worktree_maintenance_at = now

    def _run() -> None:
        try:
            repos = _worktree_maintenance_repos()
            if not repos:
                return
            from cli import _prune_stale_worktrees

            for repo in repos:
                try:
                    _prune_stale_worktrees(repo)
                except Exception:
                    logger.debug("Cron worktree maintenance failed for %s", repo, exc_info=True)
        except Exception:
            logger.debug("Cron worktree maintenance skipped", exc_info=True)

    threading.Thread(target=_run, name="cron-worktree-prune", daemon=True).start()


def _acquire_tick_lock(lock_file):
    """Open + non-blocking lock the tick file. Returns the fd, or None on genuine contention.

    fcntl on Unix, msvcrt on Windows. A real OSError (esp. EMFILE/ENFILE) must NOT pass as
    contention — the scheduler would look healthy while no job runs — so it is re-raised for the
    ticker loop to record a FAILED tick.
    """
    lock_fd = None
    try:
        lock_fd = open(lock_file, "w", encoding="utf-8")
        if fcntl:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        elif msvcrt:
            msvcrt.locking(lock_fd.fileno(), msvcrt.LK_NBLCK, 1)
        return lock_fd
    except OSError as exc:
        if lock_fd is not None:
            with contextlib.suppress(OSError):
                lock_fd.close()
            if _is_lock_contention_errno(exc):
                logger.debug("Tick skipped — another instance holds the lock")
                return None
        if _is_fd_exhaustion(exc):
            # fd reclamation is the ticker loop's job (scheduler_provider.py); here would double it.
            logger.error(
                "Cron tick could not acquire tick lock: %s — scheduler will "
                "attempt fd reclamation and retry with backoff",
                exc,
            )
        else:
            logger.error("Cron tick could not acquire tick lock: %s", exc)
        raise


def _release_tick_lock(lock_fd) -> None:
    if fcntl:
        with contextlib.suppress((OSError, IOError)):
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
    elif msvcrt:
        with contextlib.suppress((OSError, IOError)):
            msvcrt.locking(lock_fd.fileno(), msvcrt.LK_UNLCK, 1)
    lock_fd.close()


def _maybe_reap_dead_owners() -> None:
    """Dead-owner reclaim: a run that died mid-flight would leave its row 'claimed' forever. Only
    rows whose owner process is proved gone are touched (_owner_is_live). Throttled."""
    global _last_dead_owner_reap_at
    _reap_now = time.monotonic()
    if (
        _last_dead_owner_reap_at is not None
        and _reap_now - _last_dead_owner_reap_at < _DEAD_OWNER_REAP_INTERVAL_SECONDS
    ):
        return
    _last_dead_owner_reap_at = _reap_now
    try:
        from cron.executions import recover_interrupted_executions

        _reclaimed = recover_interrupted_executions()
        if _reclaimed:
            logger.warning(
                "Reclaimed %d cron execution(s) whose owner process died "
                "before reaching a terminal state (marked unknown)",
                _reclaimed,
            )
    except Exception as _reap_exc:
        logger.debug("Dead-owner execution reclaim failed: %s", _reap_exc)


def _sweep_stale_inflight_for_tick(due_jobs: list) -> None:
    """Bound the in-flight set BEFORE the dedup guard so a leaked claim is force-released now
    rather than eating every later fire until restart. Skipped when nothing is in flight."""
    if not _running_job_ids:
        return
    _sweep_jobs = due_jobs
    with contextlib.suppress(Exception):
        _inflight_ids = set(_running_job_ids)
        _due_ids = {j.get("id") for j in due_jobs if isinstance(j, dict)}
        if not _inflight_ids <= _due_ids:
            from cron.jobs import load_jobs as _load_all_jobs

            _sweep_jobs = _load_all_jobs()
    try:
        sweep_stale_inflight(_sweep_jobs)
    except Exception as e:
        logger.warning("Stale in-flight sweep failed: %s", e)


def _resolve_max_parallel_workers() -> Optional[int]:
    """Max workers: env > config.yaml > unbounded (HERMES_CRON_MAX_PARALLEL=1 restores serial)."""
    try:
        _env_par = os.getenv("HERMES_CRON_MAX_PARALLEL", "").strip()
        if _env_par:
            return int(_env_par) or None
    except (ValueError, TypeError):
        logger.warning("Invalid HERMES_CRON_MAX_PARALLEL value; defaulting to unbounded")
    with contextlib.suppress(Exception):
        _ucfg = load_config() or {}
        _cfg_par = (_ucfg.get("cron", {}) if isinstance(_ucfg, dict) else {}).get("max_parallel_jobs")
        if _cfg_par is not None:
            return int(_cfg_par) or None
    return None


def _sweep_mcp_orphans() -> None:
    """Reap MCP stdio orphans (only PIDs flagged by tools.mcp_tool._run_stdio's finally block);
    run AFTER jobs finish so live sessions are never touched."""
    try:
        from tools.mcp_tool import _kill_orphaned_mcp_children
        _kill_orphaned_mcp_children()
    except Exception as _e:
        logger.debug("Post-tick MCP orphan cleanup failed: %s", _e)


def _process_due_job(job: dict, adapters, loop, verbose: bool) -> bool:
    """Run one due job via the shared ``run_one_job`` body."""
    # Claim only when the worker actually starts, so a queued lease can't expire first.
    claimed = claim_job_for_fire(job["id"], return_job=True)
    if not claimed:
        finish_execution(
            job["execution_id"],
            success=False,
            error="Fire claim lost; execution was not started.",
        )
        return True
    # CAS returns the persisted record; bool fallback only for older test doubles.
    claimed_job = dict(claimed) if isinstance(claimed, dict) else dict(job)
    claimed_job["execution_id"] = job["execution_id"]
    return run_one_job(claimed_job, adapters=adapters, loop=loop, verbose=verbose)


def _submit_with_guard(job: dict, pool: concurrent.futures.ThreadPoolExecutor, process_job):
    """Submit with the in-flight dedup guard; None if a prior tick's run is still in flight.
    Running-set membership is released in the worker's finally."""
    job_id = job["id"]
    job_label = job.get("name", job_id)

    def _clear_run_claim_best_effort() -> None:
        """Best-effort claim cleanup on dispatch-failure paths. Only one-shots carry a run_claim;
        clear_run_claim takes _jobs_lock + full load/save and can raise on degraded paths
        (shutdown, EMFILE) — a claim expiring at TTL beats crashing the tick."""
        _schedule = job.get("schedule")
        if not (isinstance(_schedule, dict) and _schedule.get("kind") == "once"):
            return
        try:
            clear_run_claim(job_id)
        except Exception as claim_err:
            logger.warning(
                "Could not clear run_claim for job '%s' after dispatch "
                "failure: %s (claim will expire at TTL)",
                job_label, claim_err,
            )

    def _not_dispatched_shutdown() -> None:
        logger.warning("Job '%s' not dispatched — interpreter is shutting down", job_label)

    # During interpreter shutdown pool.submit raises; skip — the job fires on the next tick.
    if _interpreter_shutting_down():
        _not_dispatched_shutdown()
        _clear_run_claim_best_effort()
        return None
    if not try_register_running_job(job_id):
        logger.info("Job '%s' already running — skipping", job_label)
        return None
    # Record the attempt before dispatch; recovery marks abandoned rows unknown (no retry).
    try:
        execution = create_execution(job_id, source="builtin")
        dispatched_job = dict(job, execution_id=execution["id"])
        _ctx = contextvars.copy_context()
    except Exception as execution_err:
        # Release the claim so the next tick retries instead of wedging "already running".
        release_running_job(job_id)
        _clear_run_claim_best_effort()
        logger.exception(
            "Job '%s' not dispatched: execution creation failed: %s", job_label, execution_err,
        )
        return None

    def _run_and_release(j=dispatched_job, ctx=_ctx):
        try:
            return ctx.run(process_job, j)
        finally:
            release_running_job(j["id"])

    try:
        fut = pool.submit(_run_and_release)
    except Exception as submit_err:
        release_running_job(job_id)
        _clear_run_claim_best_effort()
        finish_execution(
            execution["id"],
            success=False,
            error=f"Executor dispatch failed: {submit_err}",
        )
        if isinstance(submit_err, RuntimeError) and _interpreter_shutting_down(submit_err):
            _not_dispatched_shutdown()
        else:
            logger.error("Job '%s' not dispatched: %s", job_label, submit_err)
        return None

    with _running_lock:
        if job_id in _running_job_ids:
            _running_futures[job_id] = fut
    return fut


def tick(
    verbose: bool = True,
    adapters=None,
    loop=None,
    sync: bool = True,
    *,
    can_dispatch=None,
):
    """Check and run all due jobs. File-locked so only one tick runs at a time (gateway ticker vs
    standalone daemon / manual tick). ``can_dispatch``: optional gate; false leaves due jobs for the
    next allowed tick. Returns the number of jobs executed (0 if another tick holds the lock)."""
    # Stale-code yield gate — BEFORE the lock race. A process whose checkout was updated under it
    # serves mixed sys.modules (jobs die on ImportErrors); if a fresher gateway holds the runtime
    # lock, ITS ticker dispatches. With no fresh holder (desktop-standalone) the tick proceeds.
    _skew = _should_yield_tick_to_fresh_gateway()
    if _skew is not None:
        _log_tick_yield_once(f"boot={_skew[0]} disk={_skew[1]}")
        raise CronTickYielded(_skew[0], _skew[1])

    lock_dir, lock_file = _get_lock_paths()
    _ensure_cron_dir(lock_dir)
    lock_fd = _acquire_tick_lock(lock_file)
    if lock_fd is None:
        return 0

    try:
        # `hermes pause` ESTOP: skip dispatch, never touch in-flight runs; check_paused logs once.
        with contextlib.suppress(ImportError):
            from agent.estop import check_paused as _estop_check_paused
            if _estop_check_paused("cron", logger):
                return 0

        if can_dispatch is not None and not can_dispatch():
            logger.debug("Cron dispatch paused while gateway drains existing work")
            return 0

        _maybe_reap_dead_owners()
        # Periodic worktree GC (6h, threaded) — the only sweep gateway-only boxes get.
        try:
            _maybe_run_worktree_maintenance()
        except Exception as _wt_exc:
            logger.debug("Worktree maintenance dispatch failed: %s", _wt_exc)

        due_jobs = get_due_jobs()
        _sweep_stale_inflight_for_tick(due_jobs)

        if not due_jobs:
            # Idle tick: skip config load + pool setup, but still reap crashed jobs' MCP orphans.
            if verbose:
                logger.info("%s - No jobs due", _hermes_now().strftime('%H:%M:%S'))
            _sweep_mcp_orphans()
            return 0

        if verbose:
            logger.info("%s - %s job(s) due", _hermes_now().strftime('%H:%M:%S'), len(due_jobs))

        # Advance next_run_at for recurring jobs FIRST, under the lock, before any execution
        # (at-most-once). Re-advancing running jobs keeps the grace window alive; mark_job_run
        # overwrites it on completion. Composes with the claim-time advance in claim_job_for_fire.
        advance_next_runs([job["id"] for job in due_jobs])

        _max_workers = _resolve_max_parallel_workers()
        if verbose:
            logger.info(
                "Running %d job(s) in parallel (max_workers=%s)",
                len(due_jobs),
                _max_workers if _max_workers else "unbounded",
            )

        def _process_job(job: dict) -> bool:
            return _process_due_job(job, adapters, loop, verbose)

        # Persistent pool, non-blocking dispatch. Already-running jobs are skipped; mark_job_run
        # re-arms next_run_at on completion, so no catch-up queue is needed.
        _results: list = []
        _all_futures: list = []
        pool = _get_parallel_pool(_max_workers)
        for job in due_jobs:
            fut = _submit_with_guard(job, pool, _process_job)
            if fut is None:
                continue
            _all_futures.append(fut)
            if not sync:
                _results.append(True)  # optimistically counted

        if sync:
            for f in concurrent.futures.as_completed(_all_futures):
                try:
                    _results.append(f.result())
                except Exception as exc:
                    logger.error("Cron job future failed: %s", exc)
                    _results.append(False)
            _sweep_mcp_orphans()
            return sum(_results)

        # Async (gateway ticker): sweep via a done-callback after the LAST job completes.
        if _all_futures:
            _remaining = [len(_all_futures)]

            def _on_done(_f: concurrent.futures.Future) -> None:
                _remaining[0] -= 1
                with contextlib.suppress(Exception):
                    _exc = _f.exception()
                    if _exc is not None:
                        logger.error("Cron job future failed in async mode: %s", _exc, exc_info=(type(_exc), _exc, _exc.__traceback__))
                if _remaining[0] <= 0:
                    _sweep_mcp_orphans()

            for _f in _all_futures:
                _f.add_done_callback(_on_done)
        else:
            # Nothing dispatched (all skipped / no due jobs) — sweep inline.
            _sweep_mcp_orphans()

        return sum(_results)
    finally:
        _release_tick_lock(lock_fd)


if __name__ == "__main__":
    tick(verbose=True)
