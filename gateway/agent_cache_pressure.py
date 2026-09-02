"""Memory-pressure bounds for the gateway's per-session AIAgent cache.

Each cached ``AIAgent`` pins ``_session_messages`` (the full live transcript,
tool outputs included — tens of MB on a tool-heavy session). The cache's LRU
cap counts entries, not bytes, and the idle TTL defers eviction for busy
sessions, so neither sees actual memory use. This module supplies that signal:
the process's own anonymous RSS against a budget derived from the cgroup limit.
``GatewayRunner._sweep_agent_cache_under_pressure`` uses it to shed LRU
transcripts via the soft-eviction path (rebuilt from the persisted session on
the next turn).

Everything here is pure or read-only. Config lives under ``agent.agent_cache``.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, List, Optional, Tuple

# Shed well under the limit: once cgroup ``memory.high`` throttling kicks in
# (swap full), a SIGTERM flush cannot finish inside systemd's stop timeout.
_AUTO_BUDGET_FRACTION = 0.65
# Below this a budget is noise — small containers would evict every pass and
# never keep a warm prefix.
_AUTO_BUDGET_FLOOR_MB = 512

_DEFAULT_MAX_EVICTIONS_PER_PASS = 16
# Never shed the hottest sessions: their prompt cache is worth the most, and
# evicting them just moves the cost to the next turn.
_DEFAULT_PROTECT_RECENT = 8

_BYTES_PER_MB = 1024 * 1024


@dataclass(frozen=True)
class AgentCacheBounds:
    """Operator-facing bounds for the per-session agent cache.

    ``max_size`` / ``idle_ttl_secs`` are ``None`` when unset so ``gateway/run.py``
    keeps its module defaults; ``memory_high_mb`` is ``None`` when pressure
    eviction is off.
    """

    max_size: Optional[int] = None
    idle_ttl_secs: Optional[float] = None
    memory_high_mb: Optional[int] = None
    max_evictions_per_pass: int = _DEFAULT_MAX_EVICTIONS_PER_PASS
    protect_recent: int = _DEFAULT_PROTECT_RECENT


def _positive(value: Any, cast: Callable[[Any], Any]) -> Any:
    """``cast(value)`` if it is a positive number (bools rejected), else None."""
    if isinstance(value, bool) or value is None:
        return None
    try:
        parsed = cast(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _positive_int(value: Any) -> Optional[int]:
    return _positive(value, int)


def _positive_float(value: Any) -> Optional[float]:
    return _positive(value, float)


def _cgroup_limit_bytes() -> Optional[int]:
    """Return the memory limit this process runs under, if cgroup-capped.

    Prefers cgroup v2 ``memory.high`` (the throttling point) over ``memory.max``,
    then cgroup v1. Checks the process's *own* cgroup first (where a systemd
    unit's ``MemoryHigh=``/``MemoryMax=`` lands — the root files read ``max``
    there), then the root for container-style limits. ``max`` and the v1
    near-2^63 sentinel mean unlimited.
    """
    if sys.platform != "linux":
        return None
    candidates: list[str] = []
    try:
        from gateway.cgroup_cleanup import _own_cgroup_path

        own = _own_cgroup_path()
    except Exception:
        own = None
    if own and own != "/":
        candidates += [f"/sys/fs/cgroup{own}/memory.high", f"/sys/fs/cgroup{own}/memory.max"]
    candidates += [
        "/sys/fs/cgroup/memory.high",
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
    ]
    for candidate in candidates:
        try:
            raw = Path(candidate).read_text(encoding="utf-8").strip()
        except OSError:
            continue
        if not raw or raw == "max":
            continue
        try:
            limit = int(raw)
        except ValueError:
            continue
        if limit <= 0 or limit >= (1 << 62):
            continue
        return limit
    return None


def _total_memory_bytes() -> Optional[int]:
    try:
        return int(os.sysconf("SC_PAGE_SIZE")) * int(os.sysconf("SC_PHYS_PAGES"))
    except (OSError, ValueError, AttributeError):
        pass
    try:
        import psutil  # type: ignore

        return int(psutil.virtual_memory().total)
    except Exception:
        return None


def resolve_memory_high_mb(setting: Any) -> Optional[int]:
    """Resolve ``memory_high_mb`` into an absolute MB budget.

    ``"auto"`` derives it from the cgroup limit (or total RAM when uncapped);
    a positive number is literal; anything falsy/off disables the pass.
    """
    if isinstance(setting, str):
        normalized = setting.strip().lower()
        if normalized != "auto":
            return (
                None
                if normalized in ("", "off", "none", "false", "disabled")
                else _positive_int(normalized)
            )
    elif isinstance(setting, bool):
        if not setting:
            return None
    else:
        return _positive_int(setting)

    limit = _cgroup_limit_bytes() or _total_memory_bytes()
    if not limit:
        return None
    budget = int(limit * _AUTO_BUDGET_FRACTION / _BYTES_PER_MB)
    return budget if budget >= _AUTO_BUDGET_FLOOR_MB else None


def resolve_agent_cache_bounds(config: Any) -> AgentCacheBounds:
    """Read ``agent.agent_cache`` out of the *raw* config mapping.

    The gateway's loader does not deep-merge ``DEFAULT_CONFIG``, so an absent
    key stays absent and callers can tell "operator chose 128" from "unset".
    """
    section: Any = None
    if isinstance(config, dict):
        agent_cfg = config.get("agent")
        if isinstance(agent_cfg, dict):
            section = agent_cfg.get("agent_cache")
    if not isinstance(section, dict):
        section = {}

    max_evictions = _positive_int(section.get("max_evictions_per_pass"))
    protect_recent = section.get("protect_recent")
    protect_parsed = _positive_int(protect_recent)
    # 0 means "shed anything" — distinct from unset. The bool guard keeps
    # `protect_recent: false` (False == 0) on the default instead of silently
    # disabling MRU protection.
    if (
        protect_parsed is None
        and isinstance(protect_recent, int)
        and not isinstance(protect_recent, bool)
        and protect_recent == 0
    ):
        protect_parsed = 0

    return AgentCacheBounds(
        max_size=_positive_int(section.get("max_size")),
        idle_ttl_secs=_positive_float(section.get("idle_ttl_secs")),
        memory_high_mb=resolve_memory_high_mb(section.get("memory_high_mb", "auto")),
        max_evictions_per_pass=(
            max_evictions if max_evictions is not None else _DEFAULT_MAX_EVICTIONS_PER_PASS
        ),
        protect_recent=(
            protect_parsed if protect_parsed is not None else _DEFAULT_PROTECT_RECENT
        ),
    )


def read_anon_rss_mb() -> Optional[int]:
    """Return the process's anonymous resident memory in MB, or None.

    Anonymous pages are where cached transcripts live; file-backed pages are
    noise. ``collect_memory_snapshot`` reads ``/proc/self/status`` without a
    dependency; psutil covers other platforms (total RSS only).
    """
    try:
        from hermes_cli.mem_trim import collect_memory_snapshot

        snapshot = collect_memory_snapshot()
        for key in ("rss_anon_kib", "rss_kib"):
            kib = snapshot.get(key)
            if isinstance(kib, int) and kib > 0:
                return kib // 1024
    except Exception:
        pass

    try:
        import psutil  # type: ignore

        return int(psutil.Process(os.getpid()).memory_info().rss / _BYTES_PER_MB)
    except Exception:
        return None


def transcript_persistence_caught_up(agent: Any) -> bool:
    """True when the agent's live transcript is fully on disk.

    Soft eviction drops ``_session_messages`` and rebuilds from the persisted
    session, so it is only safe once ``_last_flushed_db_idx`` (advanced to
    ``len(messages)`` by ``AIAgent._flush_messages_to_session_db`` only on a
    fully successful write) has caught up. Unknown shapes are *not* caught up:
    a skipped eviction costs memory, a wrong one costs the conversation.
    """
    messages = getattr(agent, "_session_messages", None)
    if not isinstance(messages, list):
        return False
    flushed = getattr(agent, "_last_flushed_db_idx", None)
    if not isinstance(flushed, int) or isinstance(flushed, bool):
        return False
    return flushed >= len(messages)


def plan_pressure_evictions(
    ordered_entries: Iterable[Tuple[str, Any]],
    *,
    is_evictable: Callable[[str, Any], bool],
    max_evictions: int,
    protect_recent: int = 0,
) -> List[Tuple[str, Any]]:
    """Choose which cached sessions to shed, least-recently-used first.

    ``ordered_entries`` must be LRU→MRU (the cache OrderedDict is kept that way
    by ``move_to_end`` on every hit). The batch is capped so one pass cannot
    stall the gateway. ``protect_recent`` is clamped to half the cache: a few
    huge transcripts can exhaust the budget alone, and a fixed guard would then
    protect the whole cache with nothing left to shed.
    """
    entries = list(ordered_entries)
    if max_evictions <= 0 or not entries:
        return []
    protect = min(max(protect_recent, 0), len(entries) // 2)
    if protect:
        entries = entries[:-protect]

    plan: List[Tuple[str, Any]] = []
    for key, agent in entries:
        if len(plan) >= max_evictions:
            break
        if is_evictable(key, agent):
            plan.append((key, agent))
    return plan
