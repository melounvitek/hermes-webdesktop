"""Bot Mode cross-connection relay — connections ARE the peer set.

Gateway-side half of the relay that lets agents on ANY Desktop-connected
gateway (local, remote URL, SSH, Hermes Cloud, docker) message agents on ANY
other, with ``message_agent`` as the one send path — connections ARE the peer
set. Plain file plumbing under ``<root>/bot_relay/`` — no network; the gateway
never holds another connection's credentials, the Desktop owns every socket and
does all cross-connection I/O:

- ``roster.json`` — union roster of agents on OTHER connections, pushed by the
  Desktop (``bot_relay.roster.sync``); folded into the Bot Chat protocol section
  and used to resolve cross-connection targets.
- ``outbox/`` — envelopes queued by ``message_agent``; the Desktop drains them
  (``bot_relay.outbox.drain``) and delivers on the target connection.
- ``replies/`` — one JSON per envelope (``bot_relay.reply``); a background
  waiter spawned at send time watches it so the reply wakes the sender via the
  same completion-notification path local DMs use.

Public helpers never raise, except ``enqueue_envelope`` → ``EnvelopeRefusedError``
when the target is definitively offline (fail fast instead of queueing a DM
nobody will drain).
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import re
import shlex
import shutil
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any, Iterator, Optional

from tools.bot_mode_probe import _default_home, _hermes_root

logger = logging.getLogger(__name__)

RELAY_DIR_NAME = "bot_relay"
ROSTER_FILE = "roster.json"
OUTBOX_DIR = "outbox"
CLAIMED_DIR = "claimed"
REPLIES_DIR = "replies"
LOCKS_DIR = "locks"

# Fallback wait budget for a queued delivery turn when config is unreadable
# (real knob: ``bot_mode.turn_wait_seconds``).
TURN_WAIT_SECONDS_FALLBACK = 120

# Waiter give-up budget. Cross-connection turns can be slow (remote model,
# cold gateway) — generous, but bounded.
REPLY_WAIT_SECONDS = 900

# Envelopes/replies older than this are stale artifacts (Desktop closed,
# connection died) and are swept opportunistically.
STALE_AFTER_SECONDS = 6 * 3600

# Fallback envelope TTL when config is unreachable — mirrors the
# ``bot_mode.envelope_ttl_seconds`` default. Older envelopes are refused at
# drain time with a 'queued_expired' error reply instead of delivered late.
DEFAULT_ENVELOPE_TTL_SECONDS = 900

# A roster older than this proves nothing about who is offline: the Desktop
# re-pushes roster.sync on connection-state changes, so only a recent roster
# is authoritative for the fail-fast check.
ROSTER_FRESH_SECONDS = 600


class EnvelopeRefusedError(RuntimeError):
    """``enqueue_envelope`` refused to queue — nothing was written to disk.

    ``reason`` is a stable machine code ('runtime_offline'); ``str(exc)`` is the
    human text.
    """

    def __init__(self, reason: str, message: str):
        super().__init__(message)
        self.reason = reason


# Profile names, handles and connection ids share one shape (also the local
# ``message_agent`` target grammar in ``tools/bot_mode_dm.py``).
_HANDLE_RE = re.compile(r"^[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}$")

# One turn in a profile's canonical Bot Chat: ``hermes -p <profile> *BOT_CHAT_TURN_ARGS``.
# ``-c "Bot Chat"`` must match ``bot_mode_probe.BOT_CHAT_TITLE``.
BOT_CHAT_TURN_ARGS = ("chat", "--in", "~", "-c", "Bot Chat", "--create-if-missing", "-Q")


def relay_root(root: Path | str) -> Path:
    return Path(root) / RELAY_DIR_NAME


def _ensure_dirs(root: Path | str) -> Path:
    base = relay_root(root)
    for sub in (OUTBOX_DIR, CLAIMED_DIR, REPLIES_DIR):
        (base / sub).mkdir(parents=True, exist_ok=True)
    return base


def _atomic_write_json(target: Path, payload: Any, *, prefix: str, sort_keys: bool = False) -> None:
    """Write ``payload`` to ``target`` via tempfile + os.replace (readers never see
    a partial file). The tempfile is removed if the write fails."""
    fd, tmp = tempfile.mkstemp(dir=str(target.parent), prefix=prefix, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, sort_keys=sort_keys)
        os.replace(tmp, target)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ── remote roster ────────────────────────────────────────────────────────────


def _normalize_roster_row(row: Any) -> Optional[dict]:
    """Validated, minimal roster row or None.

    Rows come from the Desktop over RPC — treat as untrusted input. A row names
    an agent on another connection: profile, taggable handle, owning connection
    id/label, and optional title/description for the protocol section.
    """
    if not isinstance(row, dict):
        return None
    profile = str(row.get("profile") or "").strip()
    handle = str(row.get("handle") or "").strip().lstrip("@")
    connection_id = str(row.get("connection_id") or "").strip()
    if not profile or not connection_id:
        return None
    if not handle:
        handle = "hermes" if profile == "default" else profile
    if (
        not _HANDLE_RE.match(handle)
        or not _HANDLE_RE.match(profile)
        or not _HANDLE_RE.match(connection_id)
    ):
        return None
    out = {
        "profile": profile,
        "handle": handle,
        "connection_id": connection_id,
        "connection_label": str(row.get("connection_label") or "").strip()[:80],
        "title": str(row.get("title") or "").strip()[:120],
        "description": " ".join(str(row.get("description") or "").split())[:160],
    }
    # Optional liveness flag, kept only when a real bool so absent stays
    # distinguishable from false: absent == unknown == fail-open on enqueue.
    if isinstance(row.get("online"), bool):
        out["online"] = row["online"]
    return out


def write_remote_roster(root: Path | str, rows: Any) -> int:
    """Atomically persist the Desktop-pushed remote roster. Returns count."""
    base = _ensure_dirs(root)
    by_key: dict[tuple[str, str], dict] = {}
    for row in rows if isinstance(rows, list) else []:
        norm = _normalize_roster_row(row)
        if norm:
            by_key.setdefault((norm["connection_id"], norm["profile"]), norm)
    cleaned = [by_key[k] for k in sorted(by_key)]
    payload = {"updated_at": int(time.time()), "agents": cleaned}
    _atomic_write_json(base / ROSTER_FILE, payload, prefix=".roster-", sort_keys=True)
    return len(cleaned)


def read_remote_roster(root: Path | str) -> list[dict]:
    """The current remote roster (possibly empty). Never raises."""
    try:
        raw = (relay_root(root) / ROSTER_FILE).read_text(encoding="utf-8")
        data = json.loads(raw)
        agents = data.get("agents") if isinstance(data, dict) else None
        if not isinstance(agents, list):
            return []
        return [r for r in (_normalize_roster_row(a) for a in agents) if r]
    except FileNotFoundError:
        return []
    except Exception:
        logger.debug("bot_relay roster read failed", exc_info=True)
        return []


def resolve_remote_target(raw_target: str, roster: list[dict]) -> Any:
    """Resolve ``raw_target`` against the remote roster.

    Accepted forms:
    - bare handle/profile (``moxie``) — must be unique across connections;
    - ``<handle>@<connection-id>`` / ``<profile>@<connection-id>`` — exact.

    Returns the matched row, the string ``"ambiguous"`` when a bare form
    matches agents on several connections, or None for no match.
    """
    want = str(raw_target or "").strip().lstrip("@")
    if not want:
        return None
    conn: Optional[str] = None
    if "@" in want:
        want, _, conn = want.partition("@")
        want = want.strip()
        conn = conn.strip()
        if not want or not conn:
            return None
    matches = [
        row
        for row in roster
        if want.lower() in (row["handle"].lower(), row["profile"].lower())
        and (not conn or row["connection_id"].lower() == conn.lower())
    ]
    if not matches:
        return None
    return matches[0] if len(matches) == 1 else "ambiguous"


def remote_target_forms(roster: list[dict]) -> list[str]:
    """Human/agent-facing target strings: bare handle when unique across
    connections, else ``handle@connection`` (mirrors ``resolve_remote_target``)."""
    handles = [row["handle"].lower() for row in roster]
    return [
        f"{row['handle']}@{row['connection_id']}" if handles.count(h) > 1 else row["handle"]
        for row, h in zip(roster, handles)
    ]


# ── outbox / replies ─────────────────────────────────────────────────────────


def _envelope_ttl_seconds() -> int:
    """Configured drain TTL (``bot_mode.envelope_ttl_seconds``), read per-drain
    (tools/ must not import CLI config at import time); falls back to
    ``DEFAULT_ENVELOPE_TTL_SECONDS``. ``0`` (or negative) disables expiry."""
    try:
        from hermes_cli.config import load_config_readonly

        cfg = load_config_readonly() or {}
        val = (cfg.get("bot_mode") or {}).get("envelope_ttl_seconds")
        if val is not None:
            return int(val)
    except Exception:
        logger.debug("bot_relay TTL config read failed", exc_info=True)
    return DEFAULT_ENVELOPE_TTL_SECONDS


def _target_liveness(root: Path | str, target: dict) -> Optional[bool]:
    """Tri-state liveness for ``target``: True / False / None (unknown).

    'Definitively offline' = explicit ``online: false`` on the row, or the
    target ABSENT from a *fresh* roster (the Desktop re-pushes the whole roster
    on connection-state changes). A missing, unreadable, empty or stale roster
    proves nothing → None, and callers fail open. Never raises.
    """
    try:
        roster_path = relay_root(root) / ROSTER_FILE
        try:
            age = time.time() - roster_path.stat().st_mtime
        except OSError:
            return None  # no roster ever synced — unknown
        if age > ROSTER_FRESH_SECONDS:
            return None  # stale view — unknown
        roster = read_remote_roster(root)
        if not roster:
            return None  # empty/corrupt roster — treat as unknown, fail open
        key = (str(target.get("connection_id") or ""), str(target.get("profile") or ""))
        for row in roster:
            if (row["connection_id"], row["profile"]) == key:
                online = row.get("online")
                if online is False:
                    return False
                return True if online is True else None
        return False  # fresh roster no longer lists the target — offline
    except Exception:
        logger.debug("bot_relay liveness check failed", exc_info=True)
        return None


def enqueue_envelope(
    root: Path | str,
    *,
    target: dict,
    message: str,
    sender_profile: str,
    sender_handle: str,
) -> dict:
    """Queue a cross-connection DM for the Desktop relay. Returns envelope.

    Raises ``EnvelopeRefusedError`` ('runtime_offline') without writing when the
    target is definitively offline; unknown liveness enqueues (fail-open).
    """
    if _target_liveness(root, target) is False:
        label = (
            f"@{target.get('handle') or target.get('profile') or '?'} on "
            f"{target.get('connection_label') or target.get('connection_id') or '?'}"
        )
        raise EnvelopeRefusedError(
            "runtime_offline",
            f"{label} is offline right now — the message was NOT queued. "
            "Try again once that machine reconnects to the Desktop.",
        )
    base = _ensure_dirs(root)
    envelope = {
        "id": uuid.uuid4().hex,
        "created_at": int(time.time()),
        "from_profile": sender_profile,
        "from_handle": sender_handle,
        "target_connection": target["connection_id"],
        "target_profile": target["profile"],
        "target_handle": target["handle"],
        "message": message,
    }
    _atomic_write_json(base / OUTBOX_DIR / f"{envelope['id']}.json", envelope, prefix=".env-")
    return envelope


def _expire_if_stale(root: Path | str, path: Path, ttl: float, now: float) -> bool:
    """True when the outbox envelope at ``path`` is older than ``ttl``; writes the
    'queued_expired' error reply so the sender's waiter resolves (best effort —
    an invalid id still counts as expired). Unreadable envelopes are left for
    the claim attempt to deal with."""
    try:
        env = json.loads(path.read_text(encoding="utf-8"))
        created = float(env.get("created_at") or path.stat().st_mtime)
    except (OSError, ValueError):
        return False
    if now - created <= ttl:
        return False
    handle = str(env.get("target_handle") or "?")
    conn = str(env.get("target_connection") or "?")
    with contextlib.suppress(OSError, ValueError):
        write_reply(
            root,
            str(env.get("id") or ""),
            error=(
                f"queued message to @{handle} on {conn} expired after "
                f"{ttl}s waiting for the Desktop to drain it — it was "
                "NOT delivered. Resend once the Desktop reconnects."
            ),
            reason="queued_expired",
        )
    return True


def claim_pending_envelopes(root: Path | str) -> list[dict]:
    """Drain the outbox (rename → claimed/, so a second drain can't double-
    deliver). Sweeps stale claimed/reply artifacts opportunistically.

    Envelopes older than ``bot_mode.envelope_ttl_seconds`` are NOT delivered:
    each gets a 'queued_expired' error reply (so the sender's waiter resolves)
    and its outbox file is removed.
    """
    base = _ensure_dirs(root)
    _sweep_stale(base)
    ttl = _envelope_ttl_seconds()
    now = time.time()
    out: list[dict] = []
    for path in sorted((base / OUTBOX_DIR).glob("*.json")):
        if ttl > 0 and _expire_if_stale(root, path, ttl, now):
            with contextlib.suppress(OSError):
                path.unlink()
            continue
        claimed = base / CLAIMED_DIR / path.name
        try:
            os.replace(path, claimed)  # atomic claim
            out.append(json.loads(claimed.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    return out


def write_reply(
    root: Path | str, envelope_id: str, *, reply: str = "", error: str = "", reason: str = ""
) -> Path:
    """Persist the relayed reply (or delivery error) for the waiter.

    ``reason`` is an optional typed failure code (``tools.bot_failure_reasons``);
    when omitted and ``error`` is non-empty it is classified from the text. The
    waiter only surfaces the human ``error`` (plus the code as a tag).
    """
    base = _ensure_dirs(root)
    safe = str(envelope_id or "").strip()
    if not re.match(r"^[0-9a-f]{32}$", safe):
        raise ValueError(f"invalid envelope id: {envelope_id!r}")
    err = str(error or "")
    code = str(reason or "")
    if not code and err:
        from tools.bot_failure_reasons import classify_agent_error

        code = classify_agent_error(err)
    path = base / REPLIES_DIR / f"{safe}.json"
    payload = {"id": safe, "at": int(time.time()), "reply": str(reply or ""), "error": err, "reason": code}
    _atomic_write_json(path, payload, prefix=".rep-")
    return path


def unlink_files_older_than(directory: Path, pattern: str, cutoff: float) -> int:
    """Remove regular files under ``directory`` matching ``pattern`` with mtime
    before ``cutoff``; returns the count. Never raises (missing dir → 0)."""
    removed = 0
    try:
        for path in directory.glob(pattern):
            try:
                if path.is_file() and path.stat().st_mtime < cutoff:
                    path.unlink()
                    removed += 1
            except OSError:
                continue
    except OSError:
        pass
    return removed


def _sweep_stale(base: Path, *, now: float | None = None) -> int:
    cutoff = (time.time() if now is None else now) - STALE_AFTER_SECONDS
    return sum(
        unlink_files_older_than(base / sub, "*.json", cutoff)
        for sub in (CLAIMED_DIR, REPLIES_DIR, OUTBOX_DIR)
    )


def cleanup_bot_relay_artifacts(max_age_hours: float | None = None) -> int:
    """Sweep stale relay artifacts (envelopes/replies hold DM plaintext).

    ``_sweep_stale`` otherwise runs only on Desktop drains — if the Desktop never
    reconnects, plaintext would sit on disk forever. Same contract as the
    ``cleanup_*_cache`` helpers (hourly housekeeping). ``max_age_hours`` is
    accepted for signature compatibility; ``STALE_AFTER_SECONDS`` governs.
    """
    del max_age_hours
    try:
        base = relay_root(_hermes_root(Path(_default_home())))
        if not base.is_dir():
            return 0
        return _sweep_stale(base)
    except Exception:
        logger.debug("bot_relay artifact sweep failed", exc_info=True)
        return 0


# ── waiter (runs on the sender gateway via terminal background process) ─────


def waiter_command(root: Path | str, envelope: dict) -> str:
    """Shell command that blocks until the reply file appears, then prints it.

    Spawned via ``terminal_tool(background=True, notify_on_complete=True)`` so
    its stdout — the reply — arrives as the same completion notification local
    DMs use. Stdlib-only; runs under the sender gateway's interpreter.
    """
    reply_path = str(relay_root(root) / REPLIES_DIR / f"{envelope['id']}.json")
    label = (
        f"@{envelope.get('target_handle', '')} "
        f"on {envelope.get('target_connection', '')}"
    )
    # !r keeps roster fields from breaking out of the generated python -c source.
    # The r-prefix keeps Windows paths viable: the Windows execution layer folds
    # repr's "\\" back to "\", turning "\U" into an invalid unicode escape; a
    # raw literal parses the folded backslash literally. No-op on POSIX, and \'
    # still cannot terminate a raw literal, so the injection defense holds.
    code = (
        "import json,os,sys,time\n"
        f"p = r{reply_path!r}\n"
        f"label = r{label!r}\n"
        f"deadline = time.time() + {REPLY_WAIT_SECONDS}\n"
        "while time.time() < deadline:\n"
        "    if os.path.exists(p):\n"
        "        d = json.load(open(p, encoding='utf-8'))\n"
        "        if d.get('error'):\n"
        # Typed reason code rides ahead of the free text so the sender can
        # branch on it without parsing provider prose.
        "            code = str(d.get('reason') or '').strip()\n"
        "            tag = ' [reason: ' + code + ']' if code else ''\n"
        "            print('Delivery to ' + label + ' failed' + tag + ': ' + d['error'])\n"
        "            sys.exit(1)\n"
        "        print('Reply from ' + label + ':')\n"
        "        print(d.get('reply') or '(empty reply)')\n"
        "        sys.exit(0)\n"
        # 250ms cadence: stat is cheap and a longer sleep is pure dead air.
        "    time.sleep(0.25)\n"
        f"print('No reply from ' + label + ' within {REPLY_WAIT_SECONDS}s. The message may "
        "still be delivered when the Desktop reconnects; do not resend blindly.')\n"
        "sys.exit(1)\n"
    )
    return f"{shlex.quote(sys.executable or 'python3')} -c {shlex.quote(code)}"


# ── delivery command (used by the deliver RPC on the TARGET gateway) ────────


def _hermes_cli() -> str:
    """Resolve the hermes CLI beside this gateway's own interpreter.

    Service contexts (systemd, desktop launchers, non-login SSH) lack PATH, so a
    bare "hermes" died with ENOENT; the venv sibling wins, then ``shutil.which``
    (honors whatever PATH exists), then the bare name.
    """
    exe = Path(sys.executable or "")
    sibling = exe.parent / ("hermes.exe" if sys.platform == "win32" else "hermes")
    if sibling.is_file():
        return str(sibling)
    return shutil.which("hermes") or "hermes"


def local_delivery_command(profile: str, query_file: str) -> list[str]:
    """argv that delivers a DM into ``profile``'s Bot Chat on THIS gateway."""
    return [_hermes_cli(), "-p", profile, *BOT_CHAT_TURN_ARGS, "--query-file", query_file]


# ── per-profile turn lock ────────────────────────────────────────────────────
#
# Two deliveries into the SAME profile must never run Bot Chat turns
# concurrently. Deliveries are separate ``hermes`` subprocesses, so the lock is
# a per-profile lockfile under ``<root>/bot_relay/locks/`` held with
# ``fcntl.flock`` for exactly the turn window; the kernel releases it on fd
# close (including process death), so a crashed turn can never wedge the
# profile. Waiters are bounded by ``bot_mode.turn_wait_seconds`` and then fail
# with a structured 'target_busy' refusal.


class TurnBusyError(RuntimeError):
    """A delivery turn is already running for the target profile.

    ``waited_seconds``: roughly how long the caller queued before giving up.
    """

    reason = "target_busy"

    def __init__(self, profile: str, waited_seconds: float):
        self.profile = profile
        self.waited_seconds = waited_seconds
        super().__init__(
            f"target_busy: another delivery turn is already running for "
            f"profile '{profile}' — queued behind it for ~{int(round(waited_seconds))}s "
            "without it finishing. The message was NOT delivered; retry shortly."
        )


def turn_wait_seconds() -> float:
    """Wait budget for a queued delivery turn (config, lazily read)."""
    try:
        from hermes_cli.config import cfg_get, load_config

        val = cfg_get(load_config(), "bot_mode", "turn_wait_seconds", default=None)
        if val is not None:
            return max(0.0, float(val))
    except Exception:
        logger.debug("bot_mode.turn_wait_seconds read failed", exc_info=True)
    return float(TURN_WAIT_SECONDS_FALLBACK)


def turn_lock_path(root: Path | str, profile: str) -> Path:
    """Per-profile lockfile path (short — safe on macOS temp roots)."""
    safe = re.sub(r"[^a-zA-Z0-9_-]", "_", str(profile or ""))[:64] or "_"
    return relay_root(root) / LOCKS_DIR / f"{safe}.lock"


@contextlib.contextmanager
def acquire_turn_lock(
    root: Path | str, profile: str, timeout_seconds: float | None = None
) -> Iterator[Path]:
    """Hold ``profile``'s cross-process turn lock for the ``with`` body.

    Non-blocking flock probe + short-sleep retry up to the budget
    (``bot_mode.turn_wait_seconds`` unless ``timeout_seconds`` is given). No
    ordering among waiters, but every waiter is bounded — no deadlock. Raises
    :class:`TurnBusyError` when the budget is exhausted. Without ``fcntl``
    (Windows) the lock is a no-op — those installs never had this race path
    in production.
    """
    try:
        import fcntl
    except ImportError:  # pragma: no cover — Windows
        logger.debug("bot turn lock disabled: fcntl unavailable on this platform")
        yield turn_lock_path(root, profile)
        return

    budget = turn_wait_seconds() if timeout_seconds is None else max(0.0, float(timeout_seconds))
    path = turn_lock_path(root, profile)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        start = time.monotonic()
        deadline = start + budget
        while True:
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                now = time.monotonic()
                if now >= deadline:
                    raise TurnBusyError(profile, now - start)
                time.sleep(min(0.1, max(0.005, deadline - now)))
        try:
            yield path
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            except OSError:  # pragma: no cover — kernel releases on close anyway
                pass
    finally:
        os.close(fd)
