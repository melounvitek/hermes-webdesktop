import atexit
import concurrent.futures
import contextlib
import contextvars
import copy
import hashlib
import importlib
import inspect
import json
import logging
import os
import queue
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, NamedTuple, Optional

from agent.secret_scope import (build_profile_secret_scope, reset_secret_scope, set_secret_scope)
from hermes_constants import (
    DEFAULT_INDICATOR_STYLE, INDICATOR_STYLES, get_hermes_home, get_hermes_home_override,
    reset_hermes_home_override, set_hermes_home_override,
)
from hermes_cli.env_loader import load_hermes_dotenv
from utils import is_truthy_value
from tools.environments.local import hermes_subprocess_env
from agent.replay_cleanup import sanitize_replay_history
from agent.compaction_display import project_compaction_message_for_display
from agent.skill_commands import describe_skill_invocation
from agent.conversation_loop import INTERRUPT_WAITING_FOR_MODEL_PREFIX
from tui_gateway import git_probe
from tui_gateway._env import env_float, env_int
from tui_gateway.turn_marker import (clear_turn_marker, read_turn_marker, record_turn_start)
from tui_gateway.transport import (
    StdioTransport, Transport, bind_transport, current_transport, reset_transport,
)

logger = logging.getLogger(__name__)

_hermes_home = get_hermes_home()
load_hermes_dotenv(hermes_home=_hermes_home, project_env=Path(__file__).parent.parent / ".env")


# ── Panic logger ─────────────────────────────────────────────────────
# Crashes otherwise leave no forensics (stdout is the JSON-RPC pipe, stderr
# doesn't flush before exit): append every unhandled exception to
# logs/tui_gateway_crash.log and re-emit a one-line stderr summary for Activity.
_CRASH_LOG = os.path.join(_hermes_home, "logs", "tui_gateway_crash.log")


def _panic_hook(exc_type, exc_value, exc_tb):
    import traceback
    trace = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
    with contextlib.suppress(Exception):
        os.makedirs(os.path.dirname(_CRASH_LOG), exist_ok=True)
        with open(_CRASH_LOG, "a", encoding="utf-8") as f:
            f.write(f"\n=== unhandled exception · {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
            f.write(trace)
    # Stderr goes through to the TUI as a gateway.stderr Activity line —
    # the first line here is what the user will see without opening any
    # log files.  Rest of the stack is still in the log for full context.
    first = (
        str(exc_value).strip().splitlines()[0]
        if str(exc_value).strip()
        else exc_type.__name__
    )
    print(f"[gateway-crash] {exc_type.__name__}: {first}", file=sys.stderr, flush=True)
    # Chain to the default hook so the process still terminates normally.
    sys.__excepthook__(exc_type, exc_value, exc_tb)


sys.excepthook = _panic_hook


def _thread_panic_hook(args):
    # threading.excepthook signature: SimpleNamespace(exc_type, exc_value, exc_traceback, thread)
    import traceback
    trace = "".join(traceback.format_exception(args.exc_type, args.exc_value, args.exc_traceback))
    with contextlib.suppress(Exception):
        os.makedirs(os.path.dirname(_CRASH_LOG), exist_ok=True)
        with open(_CRASH_LOG, "a", encoding="utf-8") as f:
            f.write(
                f"\n=== thread exception · {time.strftime('%Y-%m-%d %H:%M:%S')} "
                f"· thread={args.thread.name} ===\n"
            )
            f.write(trace)
    first_line = (
        str(args.exc_value).strip().splitlines()[0]
        if str(args.exc_value).strip()
        else args.exc_type.__name__
    )
    print(
        f"[gateway-crash] thread {args.thread.name} raised {args.exc_type.__name__}: {first_line}",
        file=sys.stderr, flush=True,
    )


threading.excepthook = _thread_panic_hook

with contextlib.suppress(Exception):
    from hermes_cli.banner import prefetch_update_check

    prefetch_update_check()

from tui_gateway.render import make_stream_renderer, render_diff, render_message

_sessions: dict[str, dict] = {}
_methods: dict[str, callable] = {}
_pending: dict[str, tuple[str, threading.Event]] = {}
_pending_prompt_payloads: dict[str, tuple[str, dict]] = {}
_answers: dict[str, str] = {}
# Batch clarify accumulators: rid → {"qids": [...], "answers": {qid: answer}}.
# Written by clarify.respond (per-question lock, update-in-place), read out by
# _block on resolution/timeout so locked answers survive the deadline.
_batch_clarify: dict[str, dict] = {}
_db = None
_db_error: str | None = None
_stdout_lock = threading.Lock()
_cfg_lock = threading.Lock()
# Shared profile UI metadata can be updated concurrently by Desktop, mobile,
# and multiple worker-pool RPCs.  Its compare/check/write transaction needs a
# dedicated lock rather than the unrelated process-config cache lock.
_profile_ui_meta_lock = threading.Lock()
_sessions_lock = threading.RLock()  # reentrant: _close_session_by_id may run under callers that already hold it
_prompt_lock = threading.Lock()
_cfg_cache: dict | None = None
_cfg_mtime: float | None = None
_cfg_path = None
_session_resume_lock = threading.Lock()
_SLASH_WORKER_TIMEOUT_S = max(5.0, env_float("HERMES_TUI_SLASH_TIMEOUT_S", 45.0))

# On WS disconnect ws.py parks the session for a quick reattach, but a browser
# refresh creates a NEW sid and never reattaches the old one (leaking its slash
# worker per refresh). After this grace an orphaned WS session is interrupted
# if running, then reaped once turn finalization settles. 0 = park forever.
def _resolve_ws_orphan_reap_grace() -> float:
    """Resolve the WS-orphan reap grace window (seconds).

    Config-driven via ``dashboard.ws_orphan_reap_grace_s`` (#79635); the
    ``HERMES_TUI_WS_ORPHAN_REAP_GRACE_S`` env var is kept as an internal
    override for backward compatibility and wins when set.
    """
    raw = os.environ.get("HERMES_TUI_WS_ORPHAN_REAP_GRACE_S")
    if raw is None or not str(raw).strip():
        try:
            from hermes_cli.config import load_config
            raw = (load_config().get("dashboard") or {}).get("ws_orphan_reap_grace_s")
        except Exception:
            raw = None
    try:
        grace = float(raw) if raw is not None else 20.0
    except (ValueError, TypeError):
        grace = 20.0
    return max(0.0, grace)


_WS_ORPHAN_REAP_GRACE_S = _resolve_ws_orphan_reap_grace()


def _resolve_ws_orphan_activity_stale() -> float:
    """Resolve the detached-turn activity staleness threshold (seconds).

    A detached RUNNING turn is only interrupted by the WS-orphan reaper once
    its activity clock has been idle at least this long (#98028/#100325);
    while the turn keeps producing (API waits, stream tokens, tool
    heartbeats all stamp the clock) it runs to completion detached.
    Config-driven via ``dashboard.ws_orphan_activity_stale_s``; the
    ``HERMES_TUI_WS_ORPHAN_ACTIVITY_STALE_S`` env var is an internal
    override. Defaults to 600s, matching the turn-liveness watchdog's idle
    bound (``agent.turn_liveness.timeout_s``) so "wedged" means the same
    thing on both paths. ``0`` disables the gate (pre-#98028 behavior:
    interrupt at grace regardless of activity).
    """
    raw = os.environ.get("HERMES_TUI_WS_ORPHAN_ACTIVITY_STALE_S")
    if raw is None or not str(raw).strip():
        try:
            from hermes_cli.config import load_config
            raw = (load_config().get("dashboard") or {}).get("ws_orphan_activity_stale_s")
        except Exception:
            raw = None
    try:
        stale = float(raw) if raw is not None else 600.0
    except (ValueError, TypeError):
        stale = 600.0
    return max(0.0, stale)


_WS_ORPHAN_ACTIVITY_STALE_S = _resolve_ws_orphan_activity_stale()
_WS_ORPHAN_INTERRUPT_REAP_POLL_S = 1.0
# Budget for the interrupt-then-reap poll chain: an interrupted turn that never
# settles (thread hung in a syscall) would reschedule the 1s poll forever. After
# this many polls, log loudly and force-reap.
_WS_ORPHAN_INTERRUPT_REAP_MAX_POLLS = 60
_TURN_SETTLE_BEFORE_CLOSE_SECONDS = 5.0
_DETAIL_SECTION_NAMES = ("thinking", "tools", "subagents", "activity")
_DETAIL_MODES = frozenset({"hidden", "collapsed", "expanded"})

# ── Async RPC dispatch ───────────────────────────────────────────────
# Slow handlers (seconds to minutes) would leave approval.respond and
# session.interrupt unread in the stdin pipe; only those go to a small thread
# pool, everything else stays inline so fast-path ordering stays sane.
# write_json is _stdout_lock-guarded, so concurrent response writes are safe.
_LONG_HANDLERS = frozenset(
    {
        # Billing/usage reads each do a blocking portal HTTP fetch (state + usage
        # is two serial round-trips); keep them off the main stdin loop so a slow
        # portal can't stall approval.respond / session.interrupt / other RPCs.
        "billing.state",
        "subscription.state",
        # Subscription change (V3): preview + the pending-change mutations + upgrade
        # each do a blocking portal round-trip (preview + upgrade also hit Stripe,
        # which can take seconds) — keep them off the main stdin loop.
        "subscription.preview",
        "subscription.change",
        "subscription.resume",
        "subscription.upgrade",
        "usage.bars",
        "session.usage",
        "billing.step_up",
        "browser.manage",
        "cli.exec",
        # complete.path spawns `git ls-files` + fuzzy-ranks the repo;
        # complete.slash does first-call prompt_toolkit imports + a skill scan.
        # Inline either freezes the TUI until the 120s RPC timeout.
        "complete.path",
        "complete.slash",
        "llm.oneshot",
        # model.options: credential pool checks, pricing fetch, tier check,
        # provider probe — seconds inline, blocking the picker on every open.
        "model.options",
        # Pet RPCs hit the network or decode PNG frames; inline they serialize
        # on the reader thread and the animation poll stutters.
        "pet.cells",
        "pet.gallery",
        # Generation is the heaviest pet path by far — multiple image-model
        # round-trips per call — so it must never block the reader thread.
        "pet.generate",
        "pet.hatch",
        "pet.info",
        "pet.select",
        "pet.thumb",
        "learning.frames",
        "plugins.manage",
        # reload.mcp shuts down and rediscovers every server (minutes with a
        # flapping one); concurrent reloads serialize via _mcp_reload_lock.
        "reload.mcp",
        # MCP test/OAuth RPCs block on network (cold npx spawn; oauth.start
        # waits up to ~30s for an authorization URL).
        "mcp.servers.test",
        "mcp.servers.oauth.start",
        "process.list",
        # profiles.list walks every profile's skill tree + opens its state.db;
        # profiles.create copies skill bundles — seconds on cold disks.
        "profiles.configure",
        "profiles.create",
        "profiles.describe",
        "profiles.get_asset",
        "profiles.list",
        "profiles.set_asset",
        # bot_relay.deliver runs a FULL one-turn agent conversation (up to
        # 600s); all four stay off the reader thread together.
        "bot_relay.roster.sync",
        "bot_relay.outbox.drain",
        "bot_relay.deliver",
        "bot_relay.reply",
        # image.generate is a multi-second remote API round-trip.
        "image.generate",
        "projects.discover_repos",
        "projects.record_repos",
        "projects.for_cwd",
        "projects.tree",
        "projects.project_sessions",
        # Setup readiness RPCs (polled by the Desktop) may probe the provider
        # endpoint / scan credential files; under GIL pressure they block the WS
        # read loop and cause false "needs setup".
        "setup.runtime_check",
        "setup.status",
        # Voice RPCs can trigger a SYNCHRONOUS faster-whisper lazy install
        # (300s subprocess); inline that leaves prompt.submit unread for minutes.
        "voice.toggle",
        "voice.record",
        "voice.tts",
        # wake.* hit the same synchronous STT install chain plus lazy_deps for
        # the wake engine; wake.status is polled on every gateway-ready.
        "wake.start",
        "wake.status",
        # Polled every 15s by the Desktop; cheap normally, but under GIL
        # pressure it can stall and block interrupts queued behind it.
        "session.active_list",
        "session.branch",
        "session.compress",
        "session.list",
        "session.resume",
        # Workspace re-home runs git branch/root subprocess probes against an
        # arbitrary folder — inline they'd stall the reader on a slow mount.
        "session.workspace.move",
        "shell.exec",
        "skills.manage",
        "slash.exec",
    }
)

_rpc_pool_workers = max(2, env_int("HERMES_TUI_RPC_POOL_WORKERS", 8))
_pool = concurrent.futures.ThreadPoolExecutor(
    max_workers=_rpc_pool_workers, thread_name_prefix="tui-rpc",
)
atexit.register(lambda: _pool.shutdown(wait=False, cancel_futures=True))

# Exact in-memory session generation executing on the current turn thread.
# Unlike a public session id, this object identity cannot be supplied by RPC.
_current_runtime_session_record: contextvars.ContextVar[dict | None] = (
    contextvars.ContextVar("hermes_gateway_runtime_session_record", default=None)
)

# JSON-RPC method being dispatched on this thread/task. Diagnostic only (names
# WHICH client poll is looping in the 4001 warning); never used for
# authorization — the method string is client-supplied.
_current_rpc_method: contextvars.ContextVar[str] = contextvars.ContextVar(
    "hermes_gateway_rpc_method", default=""
)

# Reserve real stdout for JSON-RPC only; redirect Python's stdout to stderr
# so stray print() from libraries/tools becomes harmless gateway.stderr instead
# of corrupting the JSON protocol.
_real_stdout = sys.stdout
sys.stdout = sys.stderr


class _DropTransport:
    """Detached WS sink: keep sessions resumable without writing stale frames."""

    def write(self, obj: dict) -> bool:
        return False

    def close(self) -> None:
        return None


# Module-level stdio transport — fallback sink when no transport is bound via
# contextvar or session. Stream resolved through a lambda so runtime monkey-
# patches of `_real_stdout` (used extensively in tests) still land correctly.
_stdio_transport = StdioTransport(lambda: _real_stdout, _stdout_lock)

# Detached websocket sessions use a drop sink instead of stdio. Desktop embeds
# the gateway in-process and captures stdout into logs, so stale JSON-RPC frames
# must not fall through there while the session waits for resume or reap.
_detached_ws_transport = _DropTransport()


def _prepend_tool_paths(env: dict[str, str]) -> dict[str, str]:
    """Prepend Hermes' managed bin, the venv bin dir, and the user-local
    bin dir to PATH so slash_worker child processes can resolve
    Hermes-managed CLIs (browser-use, uvx, uv) even when the parent
    gateway was launched with a minimal PATH (e.g. by the
    Desktop/Dashboard app). Managed bin leads, matching the managed-first
    resolution policy for the Browser Use CLI."""
    managed_bin = ""
    with contextlib.suppress(Exception):
        from hermes_constants import get_hermes_home
        managed_bin = str(Path(get_hermes_home()) / "bin")
    venv_bin = str(Path(sys.executable).parent)  # <venv>/bin (POSIX) or <venv>/Scripts (Windows)
    user_bin = str(Path.home() / ".local" / "bin")
    existing = env.get("PATH") or ""
    env["PATH"] = os.pathsep.join(
        [p for p in (managed_bin, venv_bin, user_bin) if p]
        + ([existing] if existing else [])
    )
    return env


class _SlashWorker:
    """Persistent HermesCLI subprocess for slash commands."""

    def __init__(self, session_key: str, model: str, profile_home: str | None = None):
        self._lock = threading.Lock()
        self._seq = 0
        self.stderr_tail: list[str] = []
        self.stdout_queue: queue.Queue[dict | None] = queue.Queue()
        argv = [sys.executable, "-m", "tui_gateway.slash_worker", "--session-key", session_key]
        if model:
            argv += ["--model", model]
        self._closed = False
        from hermes_cli._subprocess_compat import windows_hide_flags

        # The worker runs the agent → needs provider credentials; tier-1 secrets
        # (gateway/GitHub/infra) are still stripped. Multi-profile sessions must
        # resolve against the session's profile home, via the factory's `extra`
        # (applied last, always wins).
        from tools.environments.local import build_subprocess_env
        env = build_subprocess_env(
            hermes_subprocess_env(inherit_credentials=True),
            scrub_secrets=False,
            inherit_profile_home=False,  # base already carries the HOME contract
            extra={"HERMES_HOME": str(profile_home)} if profile_home else None,
        )
        # Hermes venv/user-local bin on PATH so worker children resolve
        # Hermes-managed CLIs under the Desktop's minimal PATH.
        env = _prepend_tool_paths(env)

        # start_new_session=True: otherwise the worker inherits the gateway's
        # pgid and mcp_tool's orphan sweep, racing the spawn, killpg()s the TUI
        # parent itself (see agent/lsp/client.py, mcp_tool._filter_mcp_children).
        self.proc = subprocess.Popen(
            argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            # Lossy UTF-8: bytes invalid in the system locale (GBK Windows)
            # must not raise UnicodeDecodeError in the drain threads.
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            cwd=os.getcwd(),
            env=env,
            creationflags=windows_hide_flags(),
            start_new_session=True,
        )
        threading.Thread(target=self._drain_stdout, daemon=True).start()
        threading.Thread(target=self._drain_stderr, daemon=True).start()

    def _drain_stdout(self):
        for line in self.proc.stdout or []:
            try:
                self.stdout_queue.put(json.loads(line))
            except json.JSONDecodeError:
                continue
        self.stdout_queue.put(None)

    def _drain_stderr(self):
        for line in self.proc.stderr or []:
            if text := line.rstrip("\n"):
                self.stderr_tail = (self.stderr_tail + [text])[-80:]

    def run(self, command: str) -> str:
        if self.proc.poll() is not None:
            raise RuntimeError("slash worker exited")
        with self._lock:
            self._seq += 1
            rid = self._seq
            self.proc.stdin.write(json.dumps({"id": rid, "command": command}) + "\n")
            self.proc.stdin.flush()
            while True:
                try:
                    msg = self.stdout_queue.get(timeout=_SLASH_WORKER_TIMEOUT_S)
                except queue.Empty:
                    raise RuntimeError("slash worker timed out")
                if msg is None:
                    break
                if msg.get("id") != rid:
                    continue
                if not msg.get("ok"):
                    raise RuntimeError(msg.get("error", "slash worker failed"))
                return str(msg.get("output", "")).rstrip()
            raise RuntimeError(
                f"slash worker closed pipe{': ' + chr(10).join(self.stderr_tail[-8:]) if self.stderr_tail else ''}"
            )

    def close(self):
        if getattr(self, "_closed", False):
            return
        self._closed = True
        proc = self.proc
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=1)
                except Exception:
                    proc.kill()
                    with contextlib.suppress(Exception):
                        proc.wait(timeout=1)  # reap the zombie SIGKILL leaves behind
        except Exception:
            with contextlib.suppress(Exception):
                proc.kill()
                proc.wait(timeout=1)
        finally:
            for stream in (proc.stdin, proc.stdout, proc.stderr):
                with contextlib.suppress(Exception):
                    stream.close()


def _display_cfg() -> dict:
    """``display`` section of the behavioral config, or ``{}`` when absent/malformed."""
    display = _load_cfg().get("display")
    return display if isinstance(display, dict) else {}


def _load_busy_input_mode() -> str:
    raw = str(_display_cfg().get("busy_input_mode", "") or "").strip().lower()
    return raw if raw in {"queue", "steer", "interrupt"} else "interrupt"


def _load_interim_assistant_messages() -> bool:
    """Return whether interim assistant commentary should be surfaced to UIs.

    Honors ``display.interim_assistant_messages`` (default true). When false,
    the tui_gateway does not install ``interim_assistant_callback``, so
    interim text from tool-call turns and verify-on-stop candidates is never
    emitted as ``message.interim`` — mirroring the messaging gateway's gating.
    """
    return is_truthy_value(_display_cfg().get("interim_assistant_messages", True))


def _shutdown_sessions() -> None:
    # Durable-first: flush every un-flushed transcript (bounded budget) BEFORE
    # the slow per-session teardown, so a supervisor SIGKILL mid-shutdown
    # can no longer lose them.
    with contextlib.suppress(Exception):
        _flush_sessions_before_exit()
    with contextlib.suppress(Exception):
        _release_gateway_wake_owner()
    with _sessions_lock:
        sids = list(_sessions)
    for sid in sids:
        _close_session_by_id(sid, end_reason="tui_shutdown")


# Session reaping / flushing knobs (implementation: session_reaper.py).
# TTL is the last-resort net for disconnect paths that slip past the WS finally;
# hours-scale because last_active freezes during a long turn and on passive
# viewing — running/pending/starting/live-transport are hard exemptions instead.
_SESSION_TTL_S = max(0.0, env_float("HERMES_TUI_SESSION_TTL_S", float(6 * 3600)))
_REAPER_SCAN_S = 300.0
# Flush-on-kill budget + periodic incremental flush (piggybacks the reaper scan)
# so a SIGTERM/SIGKILL mid-update loses at most one flush interval of session state.
_EXIT_FLUSH_BUDGET_S = max(0.0, env_float("HERMES_TUI_EXIT_FLUSH_BUDGET_S", 5.0))
_INCREMENTAL_FLUSH_INTERVAL_S = max(0.0, env_float("HERMES_TUI_SESSION_FLUSH_INTERVAL_S", _REAPER_SCAN_S))


def _start_idle_reaper() -> None:
    def _loop():
        while True:
            time.sleep(_REAPER_SCAN_S)
            with contextlib.suppress(Exception):
                _reap_idle_sessions()
    threading.Thread(target=_loop, daemon=True).start()


atexit.register(_shutdown_sessions)
_start_idle_reaper()


# ── Plumbing ──────────────────────────────────────────────────────────


def _get_db():
    global _db, _db_error
    if _db is None:
        from hermes_state import get_shared_session_db
        try:
            _db = get_shared_session_db()
            _db_error = None
        except Exception as exc:
            _db_error = str(exc)
            logger.warning(
                "TUI session store unavailable — continuing without state.db features: %s", exc,
            )
            return None
    return _db


def _db_for_profile(profile: str | None = None):
    """Return SessionDB for ``params.profile`` when it differs from launch.

    App-global remote mode passes ``profile`` on session.* RPCs so history/list/
    create operate on that profile's ``state.db``. Launch/own profile → shared
    ``_get_db()`` handle (left open). Non-launch profile → a dedicated handle
    the caller should ``close()`` (see :func:`_profile_db` contextmanager).

    Returns (db, owns_handle). ``db`` is None when unavailable.
    """
    profile_home = _profile_home(profile)
    if profile_home is None:
        return _get_db(), False
    try:
        from hermes_state import get_shared_session_db
        return get_shared_session_db(Path(profile_home) / "state.db"), True
    except Exception as exc:
        logger.warning("TUI profile session store unavailable for %s: %s", profile, exc)
        return None, False


def _transfer_db_to_agent(agent, db) -> bool:
    """Hand a DEDICATED profile handle to *agent*, which closes it on teardown.

    The build sites open a per-profile ``state.db`` handle, pass it to
    ``_make_agent``, and own it until the built agent is the one that will be
    torn down. This marks that transfer: from here ``AIAgent.close()`` (reached
    via :func:`_teardown_session`) releases the handle, so the caller must stop
    closing it.

    Returns True only when the transfer actually happened. It is refused when
    *agent* is not holding *this* handle — the build failed before
    ``_make_agent``, or the agent was given a different db — because a False
    return is what tells the caller the handle is still its own to close.
    Never called for the shared launch handle: that one is opened by
    ``_get_db()``, outlives every agent, and stays at ``_owns_session_db``
    False.
    """
    if agent is None or db is None:
        return False
    try:
        if getattr(agent, "_session_db", None) is not db:
            return False
        # The shared launch handle must never transfer: identity alone passes
        # for it, and ownership would let session.close() tear down the
        # process-wide database every other session shares.
        if db is _get_db():
            logger.warning(
                "Refused transfer of the shared launch SessionDB to a session "
                "agent — the caller's owns_db gate should have prevented this."
            )
            return False
        agent._owns_session_db = True
        return True
    except Exception:
        return False


def _open_profile_session_db(profile_home):
    """Open a DEDICATED handle on ``profile_home``'s ``state.db`` — FAIL CLOSED.

    A named-profile agent whose profile store cannot be opened must surface a
    clear error and never get built against the launch ``state.db``: a silent
    fallback bleeds the session's rows and messages into the wrong profile's
    store exactly when the profile store is briefly unopenable (locked,
    unreadable, mid-restore), and the named profile then looks blank. Callers
    let the raised error abort the agent build (deferred builds route it to
    the build's ``agent_error`` path) instead of swallowing it back onto the
    launch handle.
    """
    from hermes_state import get_shared_session_db
    db_path = Path(profile_home) / "state.db"
    try:
        return get_shared_session_db(db_path)
    except Exception as exc:
        raise RuntimeError(f"profile session store unavailable: {db_path}: {exc}") from exc


@contextlib.contextmanager
def _profile_db(params: dict | None = None):
    """Yield the SessionDB for ``params['profile']`` (app-global remote mode).

    Closes dedicated profile handles; leaves the launch-profile shared handle open.
    Yields None when the db is unavailable.
    """
    profile = None
    if isinstance(params, dict):
        profile = (params.get("profile") or "").strip() or None
    db, owns = _db_for_profile(profile)
    try:
        yield db
    finally:
        if owns and db is not None:
            with contextlib.suppress(Exception):
                db.close()


def _response_profile_name(profile: str | None = None) -> str:
    """Profile name to report on session.* payloads.

    Prefer the RPC's requested profile when it is a real non-launch profile;
    otherwise the process launch profile.
    """
    name = (profile or "").strip()
    if name and _profile_home(name) is not None:
        return name
    return _current_profile_name()


def _db_unavailable_error(rid, *, code: int):
    detail = _db_error or "state.db unavailable"
    return _err(rid, code, f"state.db unavailable: {detail}")


# ── per-session profile scoping (global remote mode) ───────────────────────────
# The desktop's app-global remote mode points every profile at this backend, so
# calls carry ``profile``: open that profile's db and bind its HERMES_HOME
# (ContextVar override) for the call so config/skills/model/persistence resolve
# to it. Omitted/own profile → the launch profile.
def _profile_home(profile: str | None) -> Path | None:
    """Resolve a named profile's home on THIS host, or None for the launch profile."""
    name = (profile or "").strip()
    if not name:
        return None
    try:
        from hermes_cli import profiles as profiles_mod
        home = Path(profiles_mod.get_profile_dir(name))
    except Exception:
        return None
    # Already the launch profile? No override needed.
    if home.resolve() == Path(_hermes_home).resolve():
        return None
    if (home / "state.db").exists() or home.exists():
        # Remember every sibling home this backend was asked to serve so the
        # change watcher stats its store too (#99333 class).
        _served_profile_homes.add(home)
        return home
    return None


# Profile homes served by this process besides the launch home — the only
# extra stores the sessions watcher must probe. Empty on single-profile
# installs, so their watcher stays byte-identical (two stats per tick).
_served_profile_homes: set[Path] = set()


def _profile_scoped(handler):
    """Bind ``params['profile']``'s HERMES_HOME around a handler.

    Pets (config + sprites) and projects (projects.db, discovery policy) both
    resolve via ``get_hermes_home``. The desktop sends ``profile`` so a single
    backend serving every profile in app-global remote mode still hits the
    focused profile's home. No-op for the launch profile.
    """

    def wrapper(rid, params):
        home = _profile_home(params.get("profile") if isinstance(params, dict) else None)
        if home is None:
            return handler(rid, params)
        token = set_hermes_home_override(home)
        try:
            return handler(rid, params)
        finally:
            reset_hermes_home_override(token)
    return wrapper


# Placeholder ``terminal.cwd`` values that don't name a real directory — the
# gateway resolves these to the home dir at runtime, so they must NOT be treated
# as an explicit workspace (mirrors gateway/run.py's config bridge).
_CWD_PLACEHOLDERS = {".", "auto", "cwd"}


def _configured_cwd_from_cfg(cfg: dict | None) -> str | None:
    """Return an absolute, existing ``terminal.cwd`` from a config mapping.

    Returns None for placeholders (``.``/``auto``/``cwd``), missing values, or
    paths that don't resolve to a real directory.
    """
    if not isinstance(cfg, dict):
        return None
    terminal_cfg = cfg.get("terminal")
    if not isinstance(terminal_cfg, dict):
        return None
    raw = str(terminal_cfg.get("cwd") or "").strip()
    if not raw or raw in _CWD_PLACEHOLDERS:
        return None
    resolved = os.path.abspath(os.path.expanduser(raw))
    return resolved if os.path.isdir(resolved) else None


def _profile_configured_cwd(profile_home: Path | None) -> str | None:
    """Resolve a non-launch profile's ``terminal.cwd`` from its own config.yaml.

    The desktop's app-global remote mode serves every profile from one backend,
    so the process-global ``TERMINAL_CWD`` belongs to the *launch* profile. A new
    session bound to another profile must take its workspace from THAT profile's
    config, not the stale env var (issue #40334). Returns an absolute, existing
    directory, or None for placeholders / missing / invalid paths.
    """
    if profile_home is None:
        return None
    try:
        from hermes_cli.config import _expand_env_vars, read_user_config_raw
        p = Path(profile_home) / "config.yaml"
        if not p.exists():
            return None
        # load_config() resolves the ACTIVE profile, so read this profile's
        # file directly + the same read-side pipeline as _load_cfg. Fail-open.
        data = _apply_managed(read_user_config_raw(p))
        expanded = _expand_env_vars(data)
        if isinstance(expanded, dict):
            data = expanded
        return _configured_cwd_from_cfg(data)
    except Exception:
        return None


def _launch_configured_cwd() -> str | None:
    """Resolve the launch profile's ``terminal.cwd`` from config.yaml.

    Dashboard ``/chat`` for the launch profile attaches to the dashboard
    process's in-memory TUI gateway. The Node PTY child receives a bridged
    ``TERMINAL_CWD`` env var, but this in-memory process does not — so reading
    the process env alone leaves a fresh chat starting in ``os.getcwd()``
    (wherever ``hermes dashboard`` was launched) instead of the configured
    ``terminal.cwd``. Read config directly so changing ``terminal.cwd`` affects
    new in-memory TUI sessions too.
    """
    try:
        return _configured_cwd_from_cfg(_load_cfg())
    except Exception:
        return None


def _default_session_cwd() -> str:
    """Fallback cwd for a session with no explicit / stored / profile cwd.

    Mirrors the launch-config-aware tail of :func:`_completion_cwd` so freshly
    created AND resumed sessions land in the configured ``terminal.cwd`` rather
    than ``os.getcwd()`` when the in-memory gateway's process env has no bridged
    ``TERMINAL_CWD``.
    """
    return _launch_configured_cwd() or os.getenv("TERMINAL_CWD") or os.getcwd()


def write_json(obj: dict) -> bool:
    """Emit one JSON frame. Routes via the most-specific transport available.

    Precedence:

    1. Event frames with a session id → the transport stored on that session,
       so async events land with the client that owns the session even if
       the emitting thread has no contextvar binding.
    2. Otherwise the transport bound on the current context (set by
       :func:`dispatch` for the lifetime of a request).
    3. Otherwise the module-level stdio transport, matching the historical
       behaviour and keeping tests that monkey-patch ``_real_stdout`` green.

    Every routed event frame is stamped with a per-session monotonic
    ``seq`` and recorded in the bounded replay ring (tui_gateway.event_replay)
    so a WS client can resume losslessly after a reconnect via
    ``session.events.since``.
    """
    if obj.get("method") == "event":
        params = obj.get("params")
        sid = ((params or {}).get("session_id")) if isinstance(params, dict) else ""
        if sid and (t := (_sessions.get(sid) or {}).get("transport")) is not None:
            from tui_gateway.event_replay import _stamp_event
            _stamp_event(obj)
            return t.write(obj)
    from tui_gateway.event_replay import _stamp_event
    _stamp_event(obj)
    return (current_transport() or _stdio_transport).write(obj)


def _event_frame(event: str, sid: str, payload: dict | None = None) -> dict:
    params: dict = {"type": event, "session_id": sid}
    if payload is not None:
        params["payload"] = payload
    return {"jsonrpc": "2.0", "method": "event", "params": params}


def _emit(event: str, sid: str, payload: dict | None = None):
    write_json(_event_frame(event, sid, payload))


# Live WS peer transports (maintained by tui_gateway.ws): the only route for
# session-less background events, which write_json would otherwise drop on
# stdio. See _broadcast_global_event.
_live_transports: set[Transport] = set()
_live_transports_lock = threading.Lock()


def register_live_transport(transport: Transport | None) -> None:
    """Track a connected client transport for global broadcasts. Idempotent."""
    if transport is None:
        return
    with _live_transports_lock:
        _live_transports.add(transport)


def unregister_live_transport(transport: Transport | None) -> None:
    """Stop tracking a transport (call on disconnect). Idempotent."""
    with _live_transports_lock:
        _live_transports.discard(transport)


def _broadcast_global_event(event: str, payload: dict | None = None) -> None:
    """Fan a session-less, surface-global event (``skin.changed``) to every
    connected client. Emitters like the skin watcher run on background threads
    where ``write_json``'s ladder bottoms out at stdio and WS peers never see
    the frame. No registered transports (stdio TUI, tests) → plain ``_emit``,
    which that path already tees where it needs to go.
    """
    with _live_transports_lock:
        targets = list(_live_transports)
    if not targets:
        _emit(event, "", payload)
        return
    frame = _event_frame(event, "", payload)
    for transport in targets:
        try:
            transport.write(frame)
        except Exception:
            # One wedged peer must not stall the rest; disconnect teardown
            # unregisters it.
            logger.debug("global-event broadcast write failed type=%s", event, exc_info=True)


def _approval_request_payload(data: dict | None) -> dict:
    """Build the client-safe representation of a pending approval."""
    payload = dict(data or {})
    if "choices" not in payload:
        if payload.get("smart_denied"):
            payload["choices"] = ["once", "deny"]
        else:
            choices = ["once"]
            if payload.get("allow_session") is not False:
                choices.append("session")
                if payload.get("allow_permanent") is not False:
                    choices.append("always")
            choices.append("deny")
            payload["choices"] = choices
    if "command" in payload:
        from gateway.run import _redact_approval_command
        payload["command"] = _redact_approval_command(payload.get("command"))
    return payload


def _pending_clarify_request_payload(sid: str) -> dict | None:
    """Read the clarify prompt still blocking a session, if there is one.

    Clarify prompts share `_block()`'s pending registry, so a reconnecting
    client whose transport was detached when `clarify.request` was emitted
    would otherwise never see the question — the agent thread stays parked on
    the Event until timeout. Same replay contract as `pending_approval`: a
    read-only snapshot, the registry stays authoritative and `clarify.respond`
    with the embedded request_id resolves it.
    """
    with _prompt_lock:
        for rid, (owner_sid, _ev) in _pending.items():
            if owner_sid != sid:
                continue
            event, prompt_payload = _pending_prompt_payloads.get(rid, ("", {}))
            if event == "clarify.request":
                snapshot = dict(prompt_payload)
                # Batch clarify: replay the answers locked so far, so a
                # reconnecting client restores its per-question ✓ state
                # instead of presenting every question as unanswered.
                batch = _batch_clarify.get(rid)
                if batch is not None and batch["answers"]:
                    snapshot["answers"] = dict(batch["answers"])
                return snapshot
    session = _sessions.get(sid)
    if session is not None:
        with session.get("history_lock", threading.Lock()):
            pending = session.get("_compute_host_pending_clarify")
            if isinstance(pending, dict):
                return dict(pending)
    return None


def _pending_approval_request_payload(session_key: str) -> dict | None:
    """Read the oldest unresolved approval in a session, if there is one."""
    try:
        from tools.approval import get_pending_gateway_approval
        approval = get_pending_gateway_approval(session_key)
    except Exception:
        logger.debug("failed to read pending approval for %s", session_key, exc_info=True)
        return None
    return _approval_request_payload(approval) if approval else None


def _emit_approval_request(sid: str, data: dict | None) -> None:
    """Emit an ``approval.request`` event to the TUI client with the command
    redacted. The approval payload is built from the RAW command string, so a
    credential-shaped value Tirith flagged would otherwise be echoed verbatim
    to the TUI client (#48456 — third egress transport alongside the chat
    platforms and the SSE/API stream fixed in #50767). Reuse the shared gateway
    seam so all approval transports redact consistently."""
    payload = _approval_request_payload(data)
    _emit("approval.request", sid, payload)


def _status_update(sid: str, kind: str, text: str | None = None):
    body = (text if text is not None else kind).strip()
    if not body:
        return
    out_kind = kind if text is not None else "status"
    # Auto-compaction reaches us as a generic "lifecycle" status. Re-tag it so
    # drivers (TUI / desktop) can show an explicit summarizing indicator —
    # otherwise idle/preflight compaction looks like a hung turn (#97239).
    if out_kind == "lifecycle":
        from agent.conversation_compression import is_compaction_progress_status
        if is_compaction_progress_status(body):
            out_kind = "compacting"
    _emit("status.update", sid, {"kind": out_kind, "text": body})


def _estimate_image_tokens(width: int, height: int) -> int:
    """Very rough UI estimate for image prompt cost.

    Uses 512px tiles at ~85 tokens/tile as a lightweight cross-provider hint.
    This is intentionally approximate and only used for attachment display.
    """
    if width <= 0 or height <= 0:
        return 0
    return max(1, (width + 511) // 512) * max(1, (height + 511) // 512) * 85


def _image_meta(path: Path) -> dict:
    meta = {"name": path.name}
    with contextlib.suppress(Exception):
        from PIL import Image
        with Image.open(path) as img:
            width, height = img.size
        meta["width"] = int(width)
        meta["height"] = int(height)
        meta["token_estimate"] = _estimate_image_tokens(int(width), int(height))
    return meta


def _ok(rid, result: dict) -> dict:
    return {"jsonrpc": "2.0", "id": rid, "result": result}


def _err(rid, code: int, msg: str, data=None) -> dict:
    error = {"code": code, "message": msg}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": rid, "error": error}


def method(name: str):
    def dec(fn):
        _methods[name] = fn
        return fn
    return dec


def _normalize_request(req: Any) -> tuple[Any, str, dict] | dict:
    """Validate a JSON-RPC request enough for safe local dispatch."""
    if not isinstance(req, dict):
        return _err(None, -32600, "invalid request: expected an object")
    rid = req.get("id")
    method = req.get("method")
    if not isinstance(method, str) or not method:
        return _err(rid, -32600, "invalid request: method must be a non-empty string")
    params = req.get("params", {})
    if params is None:
        params = {}
    elif not isinstance(params, dict):
        return _err(rid, -32602, "invalid params: expected an object")
    return rid, method, params


def handle_request(req: dict) -> dict | None:
    normalized = _normalize_request(req)
    if isinstance(normalized, dict):
        return normalized
    rid, method, params = normalized
    fn = _methods.get(method)
    if not fn:
        return _err(rid, -32601, f"unknown method: {method}")
    token = _current_rpc_method.set(method)
    try:
        return fn(rid, params)
    finally:
        _current_rpc_method.reset(token)


def _current_session_steer_authority(session_id: str) -> tuple[Transport | None, dict | None]:
    """Resolve unforgeable steering authority for this exact RPC context.

    The public session id is only a lookup hint. Authority is the identity of
    both the request's ContextVar-bound transport and the live in-memory
    session record currently stored under that id. Session transport rebinding,
    removal, or id reuse therefore invalidates an earlier generation.
    """
    transport = current_transport()
    if transport is None or not session_id:
        return None, None
    expected_session = _current_runtime_session_record.get()
    with _sessions_lock:
        session = _sessions.get(session_id)
        if (
            session is None
            or (expected_session is not None and session is not expected_session)
            or session.get("transport") is not transport
        ):
            return None, None
        return transport, session


def dispatch(req: dict, transport: Optional[Transport] = None) -> dict | None:
    """Route inbound RPCs — long handlers to the pool, everything else inline.

    Returns a response dict when handled inline. Returns None when the
    handler was scheduled on the pool; the worker writes its own response
    via the bound transport when done.

    *transport* (optional): pins every write produced by this request —
    including any events emitted by the handler — to the given transport.
    Omitting it falls back to the module-level stdio transport, preserving
    the original behaviour for ``tui_gateway.entry``.
    """
    t = transport or _stdio_transport
    token = bind_transport(t)
    try:
        normalized = _normalize_request(req)
        if isinstance(normalized, dict):
            return normalized
        _rid, method, _params = normalized
        if method not in _LONG_HANDLERS:
            return handle_request(req)

        # Snapshot the context so the pool worker sees the bound transport.
        ctx = contextvars.copy_context()

        def run():
            try:
                resp = handle_request(req)
            except Exception as exc:
                resp = _err(req.get("id"), -32000, f"handler error: {exc}")
            if resp is not None:
                t.write(resp)
        _pool.submit(lambda: ctx.run(run))
        return None
    finally:
        reset_transport(token)


def _wait_agent(session: dict, rid: str, timeout: float = 30.0) -> dict | None:
    ready = session.get("agent_ready")
    if ready is not None and not ready.wait(timeout=timeout):
        return _err(rid, 5032, "agent initialization timed out")
    err = session.get("agent_error")
    return _err(rid, 5032, err) if err else None


# The deferred prompt path waits in short slices so a cancel is honored
# promptly and a slow build can be reported to the client exactly once.
_AGENT_BUILD_WAIT_SLICE = 5.0
_AGENT_BUILD_SLOW_NOTICE_AFTER = 30.0
_AGENT_BUILD_SLOW_NOTICE_KEY = "agent-build-slow"


def _agent_build_wait_cap() -> float:
    """Upper bound (seconds) a submitted prompt waits for the deferred agent
    build before failing permanently. ``agent.build_wait_timeout`` in
    config.yaml overrides the 600s default (raise it for deployments with
    many slow/unreachable MCP servers or high-latency provider metadata)."""
    with contextlib.suppress(Exception):
        agent_cfg = _load_cfg().get("agent") or {}
        raw = agent_cfg.get("build_wait_timeout")
        if raw is not None:
            value = float(raw)
            if value > 0:
                return value
    return 600.0


def _wait_agent_for_prompt(session: dict, rid: str, sid: str) -> dict | None:
    """Patient variant of ``_wait_agent`` for the deferred prompt.submit path.

    prompt.submit has already answered ``{"status": "streaming"}`` and the
    user's first message IS the turn in flight, while a cold deferred build
    (MCP discovery, model-metadata HTTP, skills scan) routinely outlives the
    flat 30s ceiling — timing out here silently discarded that first message.
    So: keep the prompt attached to this (off-RPC) thread and deliver it when
    the build lands; wait in short slices so a cancel is honored promptly;
    tell the client once (keyed notice) when the build outlives
    ``_AGENT_BUILD_SLOW_NOTICE_AFTER``; fail only when the build thread died
    without signalling ready or the bounded cap (``agent.build_wait_timeout``,
    default 600s) expired on a genuinely hung build.

    Returns ``None`` on success OR when the turn was cancelled mid-wait (the
    caller's cancel branch owns that messaging), an ``_err`` dict otherwise.
    """
    ready = session.get("agent_ready")
    if ready is None:
        return None
    start = time.monotonic()
    cap = _agent_build_wait_cap()
    notified_slow = False
    while not ready.wait(timeout=_AGENT_BUILD_WAIT_SLICE):
        with session["history_lock"]:
            cancelled = session.get("_turn_cancel_requested") or not session.get("running")
        if cancelled:
            # The caller's cancel/not-running branch emits the user-visible
            # event for this — bail without an error of our own.
            return None
        waited = time.monotonic() - start
        if waited >= cap:
            return _err(
                rid,
                5032,
                f"agent initialization timed out after {int(waited)}s — "
                "your message was not sent; retry once the session is ready",
            )
        build_thread = session.get("_agent_build_thread")
        if (build_thread is not None and not build_thread.is_alive() and not ready.is_set()):
            # _build's finally guarantees ready.set(); dead thread + unset
            # ready = the build died hard — don't wait on a corpse.
            return _err(
                rid,
                5032,
                session.get("agent_error")
                or "agent initialization failed before completing",
            )
        if not notified_slow and waited >= _AGENT_BUILD_SLOW_NOTICE_AFTER:
            # One keyed, replace-in-place notice (toast / status bar).
            notified_slow = True
            _emit(
                "notification.show",
                sid,
                {
                    "text": (
                        "Still starting the agent (tool discovery / model "
                        "setup) — your message will be sent as soon as it's "
                        "ready."
                    ),
                    "level": "info",
                    "kind": "agent",
                    "ttl_ms": None,
                    "key": _AGENT_BUILD_SLOW_NOTICE_KEY,
                    "id": _AGENT_BUILD_SLOW_NOTICE_KEY,
                },
            )
    if notified_slow:
        _emit("notification.clear", sid, {"key": _AGENT_BUILD_SLOW_NOTICE_KEY})
    err = session.get("agent_error")
    return _err(rid, 5032, err) if err else None


def _bind_build_profile_scopes(profile_home: str) -> "_TurnScopes":
    """Bind a session profile's HERMES_HOME / secret / terminal scopes for an agent build.

    Fail-open per scope (the build must not die on a scope helper), except that
    the terminal scope installer itself fails closed (malformed policy →
    refusal scope) so _make_agent's terminal probing / cwd hints resolve the
    routed profile, never the launch process.
    """
    scopes = _TurnScopes()
    scopes.home = set_hermes_home_override(profile_home)
    with contextlib.suppress(Exception):
        from agent.secret_scope import build_profile_secret_scope, set_secret_scope

        scopes.secret = set_secret_scope(build_profile_secret_scope(Path(profile_home)))
    try:
        from tools.terminal_scope import install_profile_terminal_scope

        scopes.terminal = install_profile_terminal_scope(Path(profile_home))
    except Exception:
        scopes.terminal = None
    return scopes


def _release_build_profile_scopes(scopes: "_TurnScopes") -> None:
    if scopes.home is not None:
        reset_hermes_home_override(scopes.home)
    if scopes.secret is not None:
        with contextlib.suppress(Exception):
            from agent.secret_scope import reset_secret_scope

            reset_secret_scope(scopes.secret)
    if scopes.terminal is not None:
        with contextlib.suppress(Exception):
            from tools.terminal_scope import reset_terminal_scope

            reset_terminal_scope(scopes.terminal)


def _deferred_build_agent_kwargs(current: dict, session_db) -> dict:
    """_make_agent kwargs for a deferred (first-prompt) build.

    A lazy-resumed (watch) session carries the stored conversation id so the
    upgrade continues that session instead of starting a fresh one under the
    same key.  A cold deferred resume restores the full persisted runtime
    identity exactly as the eager resume path's _stored_session_runtime_overrides
    splat did, so a deferred build can't drop the provider and fail with "No LLM
    provider configured".  When there is no stored runtime, or its provider no
    longer resolves (renamed/removed), fall back to the model/effort/tier the
    desktop picked for THIS session, else the configured default — never sink
    agent init with "Unknown provider".
    """
    kw = {
        "session_db": session_db,
        "context_cwd_is_launch_artifact": _context_cwd_is_launch_artifact(current),
    }
    if resume_sid := current.get("resume_session_id"):
        kw["session_id"] = resume_sid
    kw["platform_override"] = _session_source(current)
    resume_overrides = current.get("resume_runtime_overrides")
    if (
        isinstance(resume_overrides, dict)
        and resume_overrides
        and _overrides_have_routable_provider(resume_overrides)
    ):
        kw.update(resume_overrides)
    else:
        if override := current.get("model_override"):
            kw["model_override"] = override
        if (reasoning := current.get("create_reasoning_override")) is not None:
            kw["reasoning_config_override"] = reasoning
        if (tier := current.get("create_service_tier_override")) is not None:
            kw["service_tier_override"] = tier
    return kw


def _wire_session_agent(sid: str, key: str, agent) -> bool:
    """Common post-build wiring for a session agent; returns whether notify registered.

    Approval prompts route to the client; the self-improvement review's "💾 …"
    summary is emitted as review.summary so the TUI/desktop render it in the
    transcript (the CLI prints it via prompt_toolkit; the TUI has no print
    surface), honoring display.memory_notifications like the gateway and CLI.
    """
    notify_registered = False
    with contextlib.suppress(Exception):
        from tools.approval import load_permanent_allowlist, register_gateway_notify

        register_gateway_notify(key, lambda data: _emit_approval_request(sid, data))
        notify_registered = True
        load_permanent_allowlist()
    _wire_callbacks(sid)
    try:
        agent.background_review_callback = lambda message, _sid=sid: _emit(
            "review.summary", _sid, {"text": str(message)}
        )
        agent.memory_notifications = _load_memory_notifications()
    except Exception:
        pass  # bare agents without the attribute must not break startup
    return notify_registered


def _start_session_services(sid: str, key: str, current: dict) -> None:
    """Start the notification poller and fire the session-reset boundary hook."""
    with _sessions_lock:
        if sid in _sessions:
            _sessions[sid]["_notif_stop"] = _start_notification_poller(sid, _sessions[sid])
    _notify_session_boundary("on_session_reset", key, _session_source(current))


def _start_agent_build(sid: str, session: dict) -> None:
    """Start building the real AIAgent for a TUI session, once.

    Deferred until the first prompt (or any command that needs the agent) so
    the composer is responsive instead of blocked on tool discovery / model
    metadata; the ready/error event contract for the frontend is unchanged.
    """
    ready = session.get("agent_ready")
    if ready is None:
        return
    # A lazy watch session spectating an in-flight child must stay lazy so the
    # subagent live-mirror keeps flowing (the mirror bails once agent is set);
    # incidental RPCs resolve through _sess() and would otherwise upgrade it
    # mid-stream.  Once the child completes the guard lifts.
    if session.get("lazy") and _child_run_active(str(session.get("session_key") or "")):
        return
    lock = session.setdefault("agent_build_lock", threading.Lock())
    with lock:
        if ready.is_set() or session.get("agent_build_started"):
            return
        session["agent_build_started"] = True
        # An upgrading lazy session is now genuinely mid-construction — restore
        # its "still starting" eviction exemption.
        session.pop("lazy", None)
    key = session["session_key"]

    def _build() -> None:
        with _sessions_lock:
            current = _sessions.get(sid)
        if current is None:
            ready.set()
            return

        notify_registered = False
        scopes = None
        session_db = None
        profile_home = current.get("profile_home")
        try:
            history_ready = current.get("resume_history_ready")
            if history_ready is not None:
                if not history_ready.wait(timeout=300.0):
                    raise TimeoutError("session history hydration timed out")
                if history_error := current.get("resume_history_error"):
                    raise RuntimeError(str(history_error))
                with _sessions_lock:
                    if _sessions.get(sid) is not current:
                        return
            tokens = _set_session_context(key)
            # Build against the session's profile (global-remote): bind its
            # HERMES_HOME so config/skills/model resolve to it, and hand the
            # agent that profile's db so turns persist to the right state.db.
            if profile_home:
                scopes = _bind_build_profile_scopes(profile_home)
                # DEDICATED handle, ours until _transfer_db_to_agent in the
                # finally; any non-transfer exit must close it. FAIL CLOSED on
                # open failure (routes to the except) rather than binding the
                # launch DB and bleeding rows into the wrong profile's state.db.
                session_db = _open_profile_session_db(profile_home)

            try:
                from tui_gateway.entry import ensure_mcp_discovery_started

                ensure_mcp_discovery_started()
            except Exception:
                logger.warning("MCP discovery startup failed", exc_info=True)

            try:
                agent = _make_agent(sid, key, **_deferred_build_agent_kwargs(current, session_db))
            finally:
                _clear_session_context(tokens)

            # Bot Mode gate hint: the DB title lands post-first-turn but the
            # system prompt builds at turn START, so hand the agent its title.
            _title_hint = str(current.get("pending_title") or "").strip()
            if _title_hint:
                agent._session_title_hint = _title_hint

            # Session DB row deferred to first run_conversation() call.
            current["agent"] = agent
            _session_todo_state(current)
            # Baseline for the per-turn config sync (profile home override still active).
            current["config_model_seen"] = _config_model_target()

            # No eager slash-worker pre-warm (slash.exec spawns on demand):
            # each worker forks the full stdio MCP fleet (~20 processes), and
            # live-transport sessions are never reaped, so fleets accumulate
            # until the OS refuses new spawns.
            notify_registered = _wire_session_agent(sid, key, agent)
            # Credits notices at session OPEN so depletion / usage-band
            # warnings show at "ready"; after notice_callback is wired. Fail-open.
            with contextlib.suppress(Exception):
                from agent.credits_tracker import seed_credits_at_session_start

                seed_credits_at_session_start(agent)
            _start_session_services(sid, key, current)

            info = _session_info(agent, current)
            cfg_warn = _probe_config_health(_load_cfg())
            if cfg_warn:
                info["config_warning"] = cfg_warn
                logger.warning(cfg_warn)
            _emit("session.info", sid, info)
            # MCP servers slower than _make_agent's bounded discovery wait are
            # missing from the agent's tool list; catch up once they land
            # (cache-safe: pre-first-turn only).
            _schedule_mcp_late_refresh(sid, agent)
        except Exception as e:
            current["agent_error"] = str(e)
            _emit("error", sid, {"message": f"agent init failed: {e}"})
        finally:
            if scopes is not None:
                _release_build_profile_scopes(scopes)
            # _attach_worker already closed the worker if this session was
            # reaped mid-build; only the late notify registration can still
            # leak (session.close unregistered before _build registered it).
            with _sessions_lock:
                replaced = _sessions.get(sid) is not current
            if replaced and notify_registered:
                with contextlib.suppress(Exception):
                    from tools.approval import unregister_gateway_notify

                    unregister_gateway_notify(key)
            # Dedicated profile handle: hand it to the agent that will actually
            # be torn down, or close it when no such agent exists — the except
            # above (nothing holds it) and `replaced` (session reaped mid-build,
            # this agent is discarded and _teardown_session never reaches it).
            if session_db is not None:
                built = None if replaced else current.get("agent")
                if not _transfer_db_to_agent(built, session_db):
                    with contextlib.suppress(Exception):
                        session_db.close()
            ready.set()

    build_thread = threading.Thread(target=_build, daemon=True)
    # Handle for _wait_agent_for_prompt: a dead build thread with agent_ready
    # still unset means the build died hard — waiters must not sit out the
    # full cap on a corpse.
    session["_agent_build_thread"] = build_thread
    build_thread.start()


def _sess_nowait(params, rid):
    sid = params.get("session_id") or ""
    s = _sessions.get(sid)
    if s:
        return (s, None)
    # Stale runtime id (orphan-reaped, LRU-evicted, or idle-TTL torn down); the
    # client should session.resume the STORED id. Log it so a "message vanished"
    # report reads as "arrived and was rejected", not "never arrived".
    logger.warning(
        "session-scoped RPC rejected: method=%s session_id=%r not in memory "
        "(detached/reaped runtime; client should resume the stored session), rid=%r",
        _current_rpc_method.get() or "?",
        sid,
        rid,
    )
    return (None, _err(rid, 4001, "session not found"))


def _sess(params, rid):
    s, err = _sess_building(params, rid)
    if err:
        return (None, err)
    return (s, _wait_agent(s, rid))


def _sess_building(params, rid):
    """Resolve a session and warm its agent build WITHOUT waiting for it.

    For handlers that need the session record but not the agent — the attach
    RPCs (image/file/pdf attach, clipboard.paste, image.detach) only read
    ``cwd``/``profile_home`` and mutate ``attached_images``, all populated at
    record creation.  They run inline on the socket reader thread, so waiting
    on a cold deferred build there stalled the paste AND every RPC queued
    behind it ("text is instant, images hang").  The build is still kicked off
    to warm the agent the following ``prompt.submit`` needs.
    """
    s, err = _sess_nowait(params, rid)
    if err:
        return (None, err)
    _start_agent_build(params.get("session_id") or "", s)
    return (s, None)


# ── Config I/O ────────────────────────────────────────────────────────


_DASHBOARD_TURN_ISOLATION_DEFAULT = False
_DASHBOARD_COMPUTE_HOST_HEARTBEAT_SECS_DEFAULT = 15
_DASHBOARD_COMPUTE_HOST_RESPAWN_MAX_DEFAULT = 3


def _coerce_int_config_value(value: Any, default: int, *, min_value: int) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return default
    return coerced if coerced >= min_value else default


def _load_dashboard_process_isolation_config(cfg: dict | None = None) -> dict[str, Any]:
    """Return dashboard process-isolation config with read-site defaults.

    ``_load_cfg()`` intentionally returns the user ``config.yaml`` plus the
    managed overlay and ``${VAR}`` expansion; it does not deep-merge
    ``hermes_cli.config.DEFAULT_CONFIG``. Keep
    the Phase-0 defaults here so dashboard runtime and the REST editor's
    DEFAULT_CONFIG-backed schema cannot drift.
    """
    root = _load_cfg() if cfg is None else cfg
    dashboard = root.get("dashboard") if isinstance(root, dict) else {}
    if not isinstance(dashboard, dict):
        dashboard = {}
    return {
        "turn_isolation": is_truthy_value(
            dashboard.get("turn_isolation"), default=_DASHBOARD_TURN_ISOLATION_DEFAULT,
        ),
        "compute_host_heartbeat_secs": _coerce_int_config_value(
            dashboard.get("compute_host_heartbeat_secs"),
            _DASHBOARD_COMPUTE_HOST_HEARTBEAT_SECS_DEFAULT, min_value=1,
        ),
        "compute_host_respawn_max": _coerce_int_config_value(
            dashboard.get("compute_host_respawn_max"), _DASHBOARD_COMPUTE_HOST_RESPAWN_MAX_DEFAULT,
            min_value=0,
        ),
    }


def _load_cfg_raw() -> dict:
    """Read the active profile's config.yaml EXACTLY as written (write-back primitive).

    ONLY legal for read→mutate→``_save_cfg`` round-trips (and raw-file
    inspection): merging defaults, the managed overlay, or ``${VAR}``
    expansion here would be persisted into the user's file on the next
    save. Behavioral reads must use :func:`_load_cfg`, which layers the
    managed overlay + env expansion on top of this raw read.
    """
    global _cfg_cache, _cfg_mtime, _cfg_path
    try:
        # Per-session profile override (session.resume) → that profile's
        # config; cache keyed on the resolved path so profiles don't clobber.
        override = get_hermes_home_override()
        home = override if isinstance(override, str) and override else _hermes_home
        p = Path(home) / "config.yaml"
        mtime = p.stat().st_mtime if p.exists() else None
        with _cfg_lock:
            if _cfg_cache is not None and _cfg_mtime == mtime and _cfg_path == p:
                return copy.deepcopy(_cfg_cache)
        if p.exists():
            from hermes_cli.config import read_user_config_raw
            data = read_user_config_raw(p)
        else:
            data = {}
        with _cfg_lock:
            # Cache the RAW config: _save_cfg writes _cfg_cache back to disk,
            # so managed values must be overlaid read-side only.
            _cfg_cache = copy.deepcopy(data)
            _cfg_mtime = mtime
            _cfg_path = p
        return data
    except Exception:
        pass
    return {}


def _load_cfg() -> dict:
    """Behavioral config read: raw user file + managed overlay + ${VAR} expansion.

    Delegates the disk read to :func:`_load_cfg_raw` (shared cache), then
    applies the same read-side pipeline as the canonical
    ``hermes_cli.config.load_config_readonly`` — managed-scope overlay and
    ``${ENV_VAR}`` expansion — minus the DEFAULT_CONFIG merge (callers here
    treat a missing key as "unset" and apply their own defaults; merging
    would also break ``_load_cfg() == {}`` sentinels). Do NOT pass the
    result to ``_save_cfg``: use ``_load_cfg_raw()`` for write-back
    round-trips or expanded/overlaid values get persisted into the user's
    file.
    """
    cfg = _apply_managed(_load_cfg_raw())
    with contextlib.suppress(Exception):
        from hermes_cli.config import _expand_env_vars
        expanded = _expand_env_vars(cfg)
        if isinstance(expanded, dict):
            cfg = expanded
    return cfg


def _apply_managed(cfg: dict) -> dict:
    """Overlay administrator-pinned managed-scope values on a config dict.

    The TUI/desktop backend builds config independently of
    hermes_cli.config.load_config, so without this a managed skin / reasoning_effort
    / service_tier / provider_routing would be silently ignored here. Read-side
    only — the raw user config is what gets cached and saved. Fail-open.
    """
    try:
        from hermes_cli import managed_scope
        return managed_scope.apply_managed_overlay(cfg if isinstance(cfg, dict) else {})
    except Exception:
        return cfg


def _save_cfg(cfg: dict):
    global _cfg_cache, _cfg_mtime, _cfg_path
    from utils import atomic_roundtrip_yaml_save
    override = get_hermes_home_override()
    path = Path(override if isinstance(override, str) and override else _hermes_home) / "config.yaml"
    # Comment-, ordering-, and Unicode-preserving write (a plain safe_dump
    # clobbered hand-written configs). Fails closed on an unreadable existing
    # config.yaml like atomic_config_write does.
    atomic_roundtrip_yaml_save(path, cfg)
    with _cfg_lock:
        _cfg_cache = copy.deepcopy(cfg)
        _cfg_path = path
        try:
            _cfg_mtime = path.stat().st_mtime
        except Exception:
            _cfg_mtime = None


def _cwd_for_session_key(session_key: str) -> str:
    """Reverse-map session_key to the session's logical cwd.

    Snapshots ``_sessions`` first: concurrent RPC handlers mutate it from the
    thread pool, so iterating the live view risks ``RuntimeError: dictionary
    changed size during iteration``.
    """
    if not session_key:
        return ""
    with _sessions_lock:
        for sess in list(_sessions.values()):
            if sess.get("session_key") == session_key:
                return str(sess.get("cwd") or "")
    return ""


def _set_session_context(
    session_key: str, cwd: str | None = None, *, ui_session_id: str = "",
) -> list:
    try:
        from gateway.session_context import set_session_vars

        # Ephemeral task ids aren't in `_sessions` (reverse-map → "" would
        # clear the cwd override); callers that know the workspace pass it.
        resolved = cwd if cwd is not None else _cwd_for_session_key(session_key)
        source = _resolve_session_platform()
        browser_control_principal = ""
        browser_control_transport_family = ""
        # Live conversation id for subprocess HERMES_SESSION_ID: an explicitly
        # empty contextvar is authoritative for the subprocess-env bridge (no
        # os.environ fallback), so never leave it "". Prefer the agent's durable
        # session_id, then session_key (same derivation as session-finalize).
        session_id = session_key
        with _sessions_lock:
            for sess in list(_sessions.values()):
                if sess.get("session_key") == session_key:
                    source = _session_source(sess)
                    session_id = (getattr(sess.get("agent"), "session_id", None) or session_key)
                    transport = sess.get("transport")
                    identity = getattr(transport, "auth_identity", None)
                    if _methods_browser_control._is_authenticated_identity(identity):
                        browser_control_principal = (
                            _methods_browser_control._principal_digest(identity)
                        )
                        browser_control_transport_family = (
                            _methods_browser_control._CLOUD_TRANSPORT_FAMILY
                        )
                    break
        return set_session_vars(
            session_key=session_key, session_id=session_id, source=source,
            browser_control_principal=browser_control_principal,
            browser_control_transport_family=browser_control_transport_family, cwd=resolved,
            ui_session_id=ui_session_id, cron_session="",
        )
    except Exception:
        return []


def _clear_session_context(tokens: list) -> None:
    if not tokens:
        return
    with contextlib.suppress(Exception):
        from gateway.session_context import clear_session_vars
        clear_session_vars(tokens)


def _enable_gateway_prompts() -> None:
    """Route approvals through gateway callbacks instead of CLI input()."""
    os.environ["HERMES_GATEWAY_SESSION"] = "1"
    os.environ["HERMES_EXEC_ASK"] = "1"
    os.environ["HERMES_INTERACTIVE"] = "1"


# ── Blocking prompt factory ──────────────────────────────────────────


def _block(
    event: str, sid: str, payload: dict, timeout: float | None = 300,
    batch_qids: list[str] | None = None,
) -> str:
    rid = uuid.uuid4().hex[:8]
    ev = threading.Event()
    with _prompt_lock:
        _pending[rid] = (sid, ev)
        payload["request_id"] = rid
        _pending_prompt_payloads[rid] = (event, dict(payload))
        if batch_qids:
            # Multi-question clarify: per-question answers accumulate here
            # (update-in-place until every qid is locked). Locked answers
            # survive a timeout — see the batch read-out below.
            _batch_clarify[rid] = {"qids": list(batch_qids), "answers": {}}
    answered = False
    answer = ""
    answer_present = False
    batch_answers: dict | None = None
    try:
        _emit(event, sid, payload)
        # Natural Event semantics: None → wait forever (clarify configured with
        # clarify_timeout <= 0, released only by a real answer or
        # session.interrupt), 0 → return immediately, > 0 → bounded wait.
        answered = ev.wait(timeout)
    finally:
        with _prompt_lock:
            _pending.pop(rid, None)
            _pending_prompt_payloads.pop(rid, None)
            answer_present = rid in _answers
            answer = _answers.pop(rid, "")
            batch_state = _batch_clarify.pop(rid, None)
            if batch_state is not None:
                batch_answers = dict(batch_state["answers"])
    if batch_qids is not None:
        # Cancel-all (respond with no question_id) resolves via _answers with
        # an empty string — that stays a plain cancel, not a partial result.
        if answer_present:
            return answer
        result: dict[str, object] = {"answers": batch_answers or {}}
        if not answered:
            # Deadline hit: keep whatever was locked, tell the tool the rest
            # are absences (not skips), and still fire the expire
            # notification so live cards tear down.
            result["timed_out"] = True
            _emit(f"{event.removesuffix('.request')}.expire", sid, {"request_id": rid})
        return json.dumps(result, ensure_ascii=False)

    # `.expire` on timeout for every blocking bridge whose `*.respond` tolerates
    # a late reply (allow_expired=True): the tool returns empty, but a slow
    # renderer can still answer and would otherwise hit a raw 4009.
    if not answered and not answer_present and event in {
        "secret.request", "sudo.request", "clarify.request", "terminal.read.request",
        "preview.read.request", "preview.act.request", "window.read.request", "mcp.setup.request",
        "tour.request",
    }:
        _emit(f"{event.removesuffix('.request')}.expire", sid, {"request_id": rid})
    return answer


def _clarify_timeout_seconds() -> float | None:
    """Clarify wait (seconds) for the TUI/desktop bridge, from the same
    canonical config the messaging gateway and CLI use. Falls back to the
    historical 300s _block default if config can't be read. ``<= 0`` in config
    means unlimited and is returned as ``None`` (never auto-skip)."""
    try:
        from tools.clarify_gateway import get_clarify_timeout
        timeout = get_clarify_timeout()
        return timeout if timeout > 0 else None
    except Exception:
        return 300


def _clarify_block(sid: str, q, c, multi_select=False, questions=None) -> str:
    """Bridge the clarify tool callback onto _block.

    Single-question calls keep the exact historical payload shape (older
    renderers never see a new field). Batch calls emit one clarify.request
    carrying the question list — only wire fields (qid/question/choices/
    multi_select) are forwarded; the tool-side normalized entries also carry
    result-assembly keys (id, choices_offered) the renderer must not see.
    The tool decodes the JSON reply via its batch answer parser.
    """
    if questions:
        wire = [
            {
                "qid": entry["qid"], "question": entry["question"], "choices": entry["choices"],
                "multi_select": bool(entry["multi_select"]),
            }
            for entry in questions
        ]
        return _block(
            "clarify.request", sid, {"questions": wire}, timeout=_clarify_timeout_seconds(),
            batch_qids=[entry["qid"] for entry in questions],
        )
    # multi_select is a pass-through hint older renderers ignore; emitted
    # only when True so single-select payloads keep their exact shape.
    return _block(
        "clarify.request",
        sid,
        (
            {"question": q, "choices": c, "multi_select": True}
            if multi_select
            else {"question": q, "choices": c}
        ),
        timeout=_clarify_timeout_seconds(),
    )


# A tour action is a DOM operation the renderer performs and answers straight
# back, so a client that implements the bridge replies in milliseconds. The
# generous deadline exists for one case only: a preview tour's first action
# injects the engine into a live page.
_TOUR_TIMEOUT_S = 45
# Until a session's client has proven it answers at all, hold it to a deadline
# a working renderer cannot miss. See _tour_request.
_TOUR_PROBE_TIMEOUT_S = 10

_TOUR_BRIDGE_UNAVAILABLE = json.dumps(
    {
        "success": False,
        "error": (
            "No Hermes Desktop window answered the tour request. The tour is "
            "driven by the desktop app's renderer, which updates separately "
            "from this backend, so an app build older than the tour tool has "
            "nothing listening. Update the Hermes Desktop app and start a new "
            "session. Do not retry tour in this session."
        ),
    }
)


def _tour_request(sid: str, payload: dict) -> str:
    """Bridge the tour tool callback onto _block, without paying for a client
    that cannot answer it.

    The renderer's ``tour.request`` handler and this backend's tool update on
    different clocks: against an older app nobody ever calls ``tour.respond``
    and each action blocks for the full deadline, stacking per turn.  So a
    session's first action gets the short probe deadline; an unanswered probe
    marks the bridge unavailable for that session (later calls return at once
    with what to fix).  Once a client has answered, actions get the full
    deadline and one slow action no longer condemns it.  The verdict lives on
    the session record, so a new session re-probes.
    """
    # A detached caller has no session record; the throwaway keeps it on the
    # plain bridge, unprobed.
    session = _sessions.get(sid)
    if session is None:
        session = {}
    state = session.get("tour_bridge")
    if state == "unanswered":
        return _TOUR_BRIDGE_UNAVAILABLE
    answer = _block(
        "tour.request", sid, dict(payload),
        timeout=_TOUR_TIMEOUT_S if state == "answered" else _TOUR_PROBE_TIMEOUT_S,
    )
    if answer:
        session["tour_bridge"] = "answered"
    elif state != "answered":
        session["tour_bridge"] = "unanswered"
    return answer or _TOUR_BRIDGE_UNAVAILABLE


def _clear_pending(sid: str | None = None) -> None:
    """Release pending prompts with an empty answer.

    When *sid* is provided, only prompts owned by that session are
    released — critical for session.interrupt, which must not
    collaterally cancel clarify/sudo/secret prompts on unrelated
    sessions sharing the same tui_gateway process.  When *sid* is
    None, every pending prompt is released (used during shutdown).
    """
    with _prompt_lock:
        for rid, (owner_sid, ev) in list(_pending.items()):
            if sid is None or owner_sid == sid:
                _answers[rid] = ""
                ev.set()


# ── Agent factory ────────────────────────────────────────────────────


def _resolve_model() -> str:
    env = (
        os.environ.get("HERMES_MODEL", "")
        or os.environ.get("HERMES_INFERENCE_MODEL", "")
    ).strip()
    if env:
        return env
    m = _load_cfg().get("model", "")
    if isinstance(m, dict):
        return str(m.get("default", "") or "").strip()
    if isinstance(m, str) and m:
        return m.strip()
    # No env seed and no config preference: fall back to the cost-safe silent
    # default (catalog-labeled, cache-only read), never an expensive Anthropic
    # flagship the user didn't pick.
    try:
        from hermes_cli.models import get_preferred_silent_default_model
        return get_preferred_silent_default_model()
    except Exception:
        return "z-ai/glm-5.2"


def _resolve_session_platform() -> str:
    """Resolve the platform tag for a tui_gateway-routed session.

    Stamping the desktop chat panel ``platform="tui"`` makes the agent suggest
    TUI-only slash commands to chat-panel users.
      * ``HERMES_DESKTOP=1`` with ``HERMES_DESKTOP_TERMINAL`` unset → "desktop"
      * ``HERMES_DESKTOP_TERMINAL=1`` → "tui" (embedded terminal pane; the tui
        hint's clarifier in system_prompt.py describes the embedding)
      * neither → "tui" (standalone ``hermes --tui``)
    """
    if is_truthy_value(os.environ.get("HERMES_DESKTOP")) and not is_truthy_value(
        os.environ.get("HERMES_DESKTOP_TERMINAL")
    ):
        return "desktop"
    return "tui"


def _resolve_session_source(explicit: str | None) -> str:
    """Default the session DB ``source`` field from the resolved platform.

    A caller that explicitly passes ``source`` (e.g. a plugin session tagged
    ``"telegram"``) keeps its value. Only an empty/None ``source`` falls back
    to the env-resolved platform — so env-driven resolution never silently
    rewrites a caller's intent.
    """
    if explicit:
        return explicit
    return _resolve_session_platform()


def _resolve_agent_platform(source: str | None) -> str:
    return _resolve_session_source(source)


def _config_model_target() -> tuple[str, str]:
    """(model, provider) currently selected by config.yaml — and ONLY config.

    Unlike `_resolve_model()`, this never reads HERMES_MODEL /
    HERMES_INFERENCE_MODEL. Those env vars are a launch-scoped seed
    (`hermes --tui -m <model>`, hosted-instance provisioning); if they
    fed the per-turn sync, the seed would be replayed as a /model switch
    and persisted globally, or would pin the session so dashboard/CLI
    model changes never reach an open chat.
    """
    cfg_model = _load_cfg().get("model")
    model = ""
    provider = ""
    if isinstance(cfg_model, dict):
        model = str(cfg_model.get("default", "") or "").strip()
        provider = str(cfg_model.get("provider") or "").strip()
        if provider.lower() == "auto":
            provider = ""
    elif isinstance(cfg_model, str):
        model = cfg_model.strip()
    # No _resolve_model() fallback: that reads the launch-scoped -m env seed,
    # which the per-turn sync would replay as a /model switch and persist
    # globally. Empty model = "config expresses no preference" → sync is a no-op.
    return model, provider


def _resolve_startup_runtime() -> tuple[str, str | None]:
    model = _resolve_model()
    explicit_provider = os.environ.get("HERMES_TUI_PROVIDER", "").strip()
    if explicit_provider:
        return model, explicit_provider
    explicit_model = (
        os.environ.get("HERMES_MODEL", "")
        or os.environ.get("HERMES_INFERENCE_MODEL", "")
    ).strip()
    if not explicit_model:
        return model, None
    with contextlib.suppress(Exception):
        from hermes_cli.models import detect_static_provider_for_model
        cfg = _load_cfg().get("model") or {}
        current_provider = (
            (str(cfg.get("provider") or "").strip().lower() if isinstance(cfg, dict) else "")
            or os.environ.get("HERMES_INFERENCE_PROVIDER", "").strip().lower()
            or "auto"
        )
        detected = detect_static_provider_for_model(explicit_model, current_provider)
        if detected:
            provider, detected_model = detected
            return detected_model, provider
    return model, None


# Bare billing buckets are not routable provider identities; restoring one as a
# session provider override breaks resume. ``openrouter`` is deliberately NOT in
# this set (fully routable; dropping it resumed OpenRouter sessions on the wrong
# provider) — agent_init's fail-fast gate is a different set that skips it.
from hermes_state import _BARE_BILLING_PROVIDERS


def _overrides_have_routable_provider(overrides: dict) -> bool:
    """Whether persisted runtime overrides still name a routable provider.

    A session row written under a provider that has since been renamed or
    removed would otherwise fail agent init with "Unknown provider".
    Empty provider counts as NOT routable here, so the caller falls back
    to the model the user picked for this session / the configured
    default instead of restoring a provider-less snapshot override.
    """
    provider = str(overrides.get("provider_override") or "").strip()
    if not provider:
        provider = str((overrides.get("model_override") or {}).get("provider") or "").strip()
    if not provider:
        return False
    try:
        from hermes_cli.runtime_provider import is_routable_provider
        return is_routable_provider(provider)
    except Exception:
        return False


def _stored_session_runtime_overrides(row: dict | None) -> dict:
    """Return runtime fields persisted with a stored session.

    ``session.resume`` is session-scoped: reopening an older chat must restore
    the model/provider/reasoning state that chat actually used, not the global
    model most recently selected elsewhere.  The row stores the model directly,
    the billing provider in ``billing_provider``, and richer knobs in JSON
    ``model_config``.

    Plugin-owned Bot-Mode sessions are exempt and always rebuild from the member
    profile's CURRENT config — restoring a stale provider pin is what left room
    bots / bot DMs failing ("out of Nous credits") after the profile switched.
    Signals, in order: the explicit ``room_plumbing`` / ``follow_profile_config``
    markers persisted by session.create consumers; the legacy hidden +
    "Group:" title shape (older desktop builds sent no marker); and the title
    exactly "Bot Chat" (the plugin's own identity rule for the forever-DM,
    UNIQUE(title) makes it exact; pre-policy rows may be visible or hidden).
    """
    if not row:
        return {}
    raw_config = row.get("model_config")
    model_config: dict = {}
    if isinstance(raw_config, dict):
        model_config = raw_config
    elif isinstance(raw_config, str) and raw_config.strip():
        try:
            parsed = json.loads(raw_config)
            if isinstance(parsed, dict):
                model_config = parsed
        except Exception:
            logger.debug("failed to parse stored session model_config", exc_info=True)
    _row_title = str(row.get("title") or "").strip()
    if (
        model_config.get("room_plumbing")
        or (row.get("hidden") and _row_title.startswith("Group:"))
        or model_config.get("follow_profile_config")
        or _row_title == "Bot Chat"
    ):
        return {}
    overrides: dict = {}
    model = str(row.get("model") or model_config.get("model") or "").strip()
    # ``billing_provider`` is only the billing bucket — for a custom endpoint it
    # is the bare class ``"custom"``, which agent_init treats as non-routable, so
    # restoring it as the provider override fails resume with "No LLM provider
    # configured".  Only restore an explicit provider; otherwise leave it unset
    # so resume falls back to the configured default (CLI parity).
    explicit_provider = str(model_config.get("provider") or "").strip()
    billing_provider = str(
        model_config.get("billing_provider") or row.get("billing_provider") or ""
    ).strip()
    provider = explicit_provider
    if not provider and billing_provider.lower() not in _BARE_BILLING_PROVIDERS:
        provider = billing_provider
    base_url = str(model_config.get("base_url") or "").strip()
    api_mode = str(model_config.get("api_mode") or "").strip()
    reasoning_config = model_config.get("reasoning_config")
    service_tier = str(model_config.get("service_tier") or "").strip()

    # Heal a stale/expired provider name persisted by an older build (a renamed
    # or removed custom provider would fail agent init with "Unknown provider").
    # Recover the durable ``custom:<name>`` key from the stored base_url, then
    # from the entry serving the stored model; when nothing names a real entry,
    # drop the provider so resume falls back to the configured default.
    if provider:
        try:
            from hermes_cli.runtime_provider import is_routable_provider
            routable = is_routable_provider(provider)
        except Exception:
            routable = False
        if not routable:
            healed = None
            try:
                from hermes_cli.runtime_provider import canonical_custom_identity
                healed = canonical_custom_identity(base_url=base_url or None, model=model or None)
            except Exception:
                logger.debug("custom provider identity recovery failed", exc_info=True)
            if healed:
                logger.info("healed stale session provider %r to %r", provider, healed)
                provider = healed
                # The healed identity owns a registered endpoint; the snapshot's
                # base_url must not override the registry URL.
                base_url = ""
            else:
                provider = ""
    if model:
        # Same dict-shaped override live /model switches use, so a DB-restored
        # session keeps custom endpoint metadata across resume and rebuilds
        # (/new).  Raw api_key is deliberately never persisted or restored here.
        overrides["model_override"] = {
            "model": model, "provider": provider or None, "base_url": base_url or None,
            "api_mode": api_mode or None,
        }
    if provider:
        overrides["provider_override"] = provider
    if isinstance(reasoning_config, dict):
        overrides["reasoning_config_override"] = reasoning_config
    if service_tier.lower() == "normal":
        # None means "inherit the profile" at _make_agent; "" is a real override
        # meaning "do not request a priority service tier".
        overrides["service_tier_override"] = ""
    elif service_tier:
        overrides["service_tier_override"] = service_tier
    return overrides


def _runtime_model_config(agent, existing: dict | None = None) -> dict:
    """Merge the agent's CURRENT runtime identity onto an existing config.

    ``existing`` is the row's previously-persisted ``model_config`` JSON (may
    be absent on first write). The returned dict must mirror the agent's live
    state: falsy agent attributes DELETE the corresponding key rather than
    merely omit the write, so a stale value from an earlier session state can
    never survive into the merged config. Keeping stale values here is what
    desynced the ``sessions.model`` column (fresh) from ``model_config``
    (stale provider/endpoint): ``_persist_live_session_runtime`` writes the
    model column separately, and on resume ``_stored_session_runtime_overrides``
    reads provider/endpoint from this JSON — so a stale provider would silently
    route the resumed chat to the wrong endpoint while the model column claimed
    the new one.
    """
    config = dict(existing or {})
    model = str(getattr(agent, "model", "") or "").strip()
    provider = str(getattr(agent, "provider", "") or "").strip()
    base_url = str(getattr(agent, "base_url", "") or "").strip()
    if provider.lower() == "custom":
        # ``agent.provider`` resolves every named custom entry to the literal
        # "custom", which loses the entry identity (api_key is never persisted,
        # so resume couldn't re-resolve credentials). Recover the canonical
        # ``custom:<name>`` key from the endpoint URL, else from the configured
        # provider (the no-base_url case that routed to OpenRouter with no key).
        try:
            from hermes_cli.runtime_provider import canonical_custom_identity
            provider = canonical_custom_identity(base_url=base_url, model=model or None) or provider
        except Exception:
            logger.debug("custom provider identity lookup failed", exc_info=True)
    reasoning_config = getattr(agent, "reasoning_config", None)
    live = {
        "model": model,
        "provider": provider,
        "base_url": base_url,
        "api_mode": str(getattr(agent, "api_mode", "") or "").strip(),
        # An empty dict is still a real (present) reasoning config.
        "reasoning_config": reasoning_config if isinstance(reasoning_config, dict) else None,
        "service_tier": getattr(agent, "service_tier", None),
    }
    for key, value in live.items():
        if value or isinstance(value, dict):
            config[key] = value
        else:
            config.pop(key, None)
    return config


def _persist_live_session_runtime(session: dict | None) -> None:
    """Persist active session runtime so future resumes restore the same footer."""
    if not session:
        return
    agent = session.get("agent")
    session_key = str(session.get("session_key") or "").strip()
    if agent is None or not session_key:
        return
    db = getattr(agent, "_session_db", None) or _get_db()
    if db is None:
        return
    try:
        row = db.get_session(session_key) or {}
        raw_config = row.get("model_config")
        existing_config = {}
        if isinstance(raw_config, dict):
            existing_config = raw_config
        elif isinstance(raw_config, str) and raw_config.strip():
            parsed = json.loads(raw_config)
            if isinstance(parsed, dict):
                existing_config = parsed
        model_config = _runtime_model_config(agent, existing_config)
        create_service_tier_override = session.get("create_service_tier_override")
        if create_service_tier_override is not None:
            # _runtime_model_config sees agent.service_tier=None for explicit
            # normal and would otherwise erase the distinction on every live
            # metadata persist.
            model_config["service_tier"] = create_service_tier_override or "normal"
        model = str(getattr(agent, "model", "") or "").strip()
        if hasattr(db, "update_session_meta"):
            db.update_session_meta(session_key, json.dumps(model_config), model or None)
        elif model and hasattr(db, "update_session_model"):
            db.update_session_model(session_key, model)
    except Exception:
        logger.debug("failed to persist live session runtime", exc_info=True)


def _persist_live_session_system_prompt(session: dict | None) -> None:
    """Refresh the stored system prompt after a live runtime identity change."""
    if not session:
        return
    agent = session.get("agent")
    session_key = str(session.get("session_key") or "").strip()
    if agent is None or not session_key or not hasattr(agent, "_build_system_prompt"):
        return
    db = getattr(agent, "_session_db", None) or _get_db()
    if db is None or not hasattr(db, "update_system_prompt"):
        return

    # Re-bind HERMES_HOME to the session's profile: the build's finally already
    # reset it, and the rebuilt prompt would use the root profile's SOUL.md/skills.
    profile_home = session.get("profile_home")
    home_token = (set_hermes_home_override(profile_home) if profile_home else None)
    # Bind the session context too: on the RPC dispatcher thread _SESSION_CWD is
    # unset, so resolve_agent_cwd() falls back to the process TERMINAL_CWD and
    # the rebuilt prompt persists the wrong cwd (later turns reuse the bytes).
    session_tokens = _set_session_context(session_key, cwd=_session_cwd(session))
    try:
        prompt = agent._build_system_prompt(None)
        agent._cached_system_prompt = prompt
        db.update_system_prompt(getattr(agent, "session_id", None) or session_key, prompt)
    except Exception:
        logger.warning(
            "failed to persist live session system prompt for session %s", session_key,
            exc_info=True,
        )
    finally:
        _clear_session_context(session_tokens)
        if home_token is not None:
            reset_hermes_home_override(home_token)


# Stable leading text of the model-switch marker (builder + dedup). Only the
# newest marker is meaningful; stale ones would be re-sent every turn.
_MODEL_SWITCH_MARKER_PREFIX = "[System: The active model for this chat has changed to "


def _is_model_switch_marker(entry: Any) -> bool:
    """Whether a history entry is a (self-replacing) model-switch marker."""
    if not isinstance(entry, dict):
        return False
    content = entry.get("content")
    return isinstance(content, str) and content.startswith(_MODEL_SWITCH_MARKER_PREFIX)


def _is_pivot_marker(entry: Any) -> bool:
    """Whether a history entry is a marker the gateway splices in mid-turn.

    Model switches and personality changes both inject a ``role=user`` pivot
    into the live history from the RPC thread while a turn may be running, so
    either one can be the sole reason turn-start and current history differ.
    Only the model-switch marker is self-replacing, which is why the dedup in
    :func:`_append_model_switch_marker` stays narrower than this.
    """
    if _is_model_switch_marker(entry):
        return True
    return isinstance(entry, dict) and entry.get("display_kind") == "personality_switch"


def _append_model_switch_marker(session: dict | None, *, model: str, provider: str) -> None:
    """Record a real system-history pivot after a live model switch.

    Only the most recent marker is kept: each new switch first strips any
    prior model-switch markers from the live history, so N switches leave one
    marker (naming the active model), not N stale ones accumulating tokens on
    every subsequent API call (#65891). The in-memory history is the payload
    re-sent each turn; the dedup is self-healing across resumes because the
    next switch collapses whatever markers a reload brought back.
    """
    if not session:
        return
    session_key = str(session.get("session_key") or "").strip()
    if not session_key:
        return
    provider_part = f" via provider {provider}" if provider else ""
    marker = (
        f"{_MODEL_SWITCH_MARKER_PREFIX}"
        f"{model}{provider_part}. From this point forward, use this runtime "
        "metadata when answering questions about what model/provider is active.]"
    )
    # A user message, not system: strict OpenAI-compatible providers (vLLM,
    # Qwen) reject non-leading system messages.
    entry = {"role": "user", "content": marker, "display_kind": "model_switch"}

    def _replace_markers() -> None:
        history = session.setdefault("history", [])
        # Drop any earlier markers in place before appending the new one.
        history[:] = [h for h in history if not _is_model_switch_marker(h)]
        history.append(entry)
        session["history_version"] = int(session.get("history_version", 0)) + 1
    lock = session.get("history_lock")
    if lock is not None:
        with lock:
            _replace_markers()
    else:
        _replace_markers()
    try:
        agent = session.get("agent")
        db = getattr(agent, "_session_db", None) if agent is not None else None
        if db is not None:
            db.append_message(
                session_id=session_key, role="user", content=marker, display_kind="model_switch",
            )
            return
        _ensure_session_db_row(session)
        with _session_db(session) as scoped_db:
            if scoped_db is not None:
                scoped_db.append_message(
                    session_id=session_key, role="user", content=marker,
                    display_kind="model_switch",
                )
    except Exception:
        logger.debug("failed to persist model switch marker", exc_info=True)


def _write_config_key(key_path: str, value):
    # Write-back round-trip: raw read is mandatory — saving the managed-
    # overlaid / env-expanded view would persist those values into the file.
    cfg = _load_cfg_raw()
    current = cfg
    keys = key_path.split(".")
    for key in keys[:-1]:
        if key not in current or not isinstance(current.get(key), dict):
            current[key] = {}
        current = current[key]
    current[keys[-1]] = value
    _save_cfg(cfg)


_STATUSBAR_MODES = frozenset({"off", "top", "bottom"})
_APPROVAL_MODES = frozenset({"manual", "smart", "off"})

# Appearance switches the renderer owns but the AGENT must see (each gates a
# tool's `check_fn`), so the toggle must reach whichever gateway the app talks
# to. `config.set` answers 4002 for unlisted keys — a mirrored switch missing
# here writes nothing and its tool stays dark. Add renderer mirrors here too.
_DISPLAY_TOGGLE_KEYS = frozenset(
    {"display.message_reactions", "display.in_app_tips", "display.in_app_tours"}
)
_BOOL_WORDS = {
    "1": True, "on": True, "true": True, "yes": True, "0": False, "off": False, "false": False,
    "no": False,
}


def _load_approval_mode() -> str:
    """Resolve the effective ``approvals.mode`` for the TUI surface.

    Delegates to ``tools.approval._get_approval_mode`` so the mode cannot drift
    from the approval gate's own view (a local raw-config re-read missed the
    managed overlay and ``${VAR}`` expansion).
    """
    from tools.approval import _get_approval_mode
    mode = _get_approval_mode()
    return mode if mode in _APPROVAL_MODES else "manual"


def _coerce_statusbar(raw) -> str:
    if raw is False:
        return "off"
    if isinstance(raw, str) and (s := raw.strip().lower()) in _STATUSBAR_MODES:
        return s
    return "top"


_MOUSE_TRACKING_ALIASES = {
    "0": "off", "1": "all", "all": "all", "any": "all", "button": "buttons", "buttons": "buttons",
    "click": "buttons", "false": "off", "full": "all", "no": "off", "off": "off", "on": "all",
    "scroll": "wheel", "true": "all", "wheel": "wheel", "yes": "all",
}


def _display_mouse_tracking(display: dict) -> str:
    """Resolve display.mouse_tracking to one of ``off|wheel|buttons|all``.

    Boolean values keep their legacy meaning (``True`` → ``all``, ``False`` →
    ``off``). The ``wheel`` preset (DEC 1000+1006) is the tmux-friendly
    subset — wheel + click only, no hover events to trigger prompt-row
    clipboard probes. Legacy ``tui_mouse`` is honored only when
    ``mouse_tracking`` is absent.
    """
    if not isinstance(display, dict):
        return "all"
    if "mouse_tracking" in display:
        raw = display.get("mouse_tracking")
    else:
        raw = display.get("tui_mouse", True)
    if raw is False or raw == 0:
        return "off"
    if raw is True or raw is None:
        return "all"
    if isinstance(raw, (int, float)):
        return "all"
    if isinstance(raw, str):
        return _MOUSE_TRACKING_ALIASES.get(raw.strip().lower(), "all")
    return "all"


def _load_reasoning_config(model: str = "") -> dict | None:
    """Load reasoning effort from config.yaml, respecting per-model overrides.

    Thin wrapper over the shared chokepoint
    :func:`hermes_constants.resolve_reasoning_config` (per-model override >
    global ``agent.reasoning_effort``; YAML boolean False = disabled).
    Closes #21256.
    """
    from hermes_constants import resolve_reasoning_config
    return resolve_reasoning_config(_load_cfg(), model)


def _load_service_tier() -> str | None:
    raw = (str((_load_cfg().get("agent") or {}).get("service_tier", "") or "") .strip() .lower())
    if not raw or raw in {"normal", "default", "standard", "off", "none"}:
        return None
    if raw in {"fast", "priority", "on"}:
        return "priority"
    if raw in {"auto", "cold"}:
        return raw
    return None


def _load_provider_routing() -> dict:
    """OpenRouter provider-routing prefs from config.yaml (``provider_routing``).

    Parity with the messaging gateway (``gateway/run.py::_load_provider_routing``)
    and the classic CLI: without this the desktop/TUI backend builds agents with
    no routing prefs, so OpenRouter falls back to its default (effectively random)
    provider selection even when the user configured ``provider_routing``.
    """
    try:
        return _load_cfg().get("provider_routing", {}) or {}
    except Exception:
        return {}


def _load_show_reasoning() -> bool:
    # Fallback True — keep in sync with DEFAULT_CONFIG display.show_reasoning
    # (this loader reads the raw user YAML without the DEFAULT_CONFIG merge).
    return bool(_display_cfg().get("show_reasoning", True))


def _load_memory_notifications() -> str:
    """Self-improvement review notification mode from config.yaml.

    Parity with the messaging gateway (``gateway/run.py``) and the classic CLI:
    ``display.memory_notifications`` controls whether the background review's
    "💾 Self-improvement review: …" summary is surfaced. Without this the
    TUI/desktop backend always behaved as ``"on"`` and silently ignored a user
    who set ``off``. Accepts ``off`` / ``on`` (default) / ``verbose``; a bool is
    normalized for back-compat.
    """
    raw = _display_cfg().get("memory_notifications")
    if isinstance(raw, bool):
        return "on" if raw else "off"
    return str(raw).lower() if raw else "on"


def _load_tool_progress_mode() -> str:
    env = os.environ.get("HERMES_TUI_TOOL_PROGRESS", "").strip().lower()
    if env in {"off", "new", "all", "verbose"}:
        return env
    raw = _display_cfg().get("tool_progress", "all")
    if raw is False:
        return "off"
    if raw is True:
        return "all"
    mode = str(raw or "all").strip().lower()
    return mode if mode in {"off", "new", "all", "verbose"} else "all"


def _gui_surface_toolsets(platform: str) -> set[str]:
    """Toolsets that exist because of the CLIENT on the other end, not the host.

    Both entries are off ``_HERMES_CORE_TOOLS`` (no other platform should carry
    their schema), so this resolver is the one gate that exposes them.
    ``platform`` is the SESSION's source, never a process env var: the desktop
    may drive a URL/cloud backend where ``HERMES_DESKTOP`` is unset, and keying
    off the env var stripped every pane/browser tool there.  See the
    surface-capability rule in AGENTS.md.
    """
    surfaces = {"project"}
    if platform == "desktop":
        surfaces.add("desktop_ui")
    return surfaces


def _enabled_mcp_server_names() -> tuple[set[str], set[str]]:
    """(enabled, disabled) MCP server names from raw config; empty on any failure."""
    try:
        from hermes_cli.config import read_raw_config
        from hermes_cli.tools_config import _parse_enabled_flag
        raw_cfg = read_raw_config()
        mcp_servers = raw_cfg.get("mcp_servers") if isinstance(raw_cfg.get("mcp_servers"), dict) else {}
        enabled, disabled = set(), set()
        for name, server_cfg in mcp_servers.items():
            if not isinstance(server_cfg, dict):
                continue
            if _parse_enabled_flag(server_cfg.get("enabled", True), default=True):
                enabled.add(str(name))
            else:
                disabled.add(str(name))
        return enabled, disabled
    except Exception:
        return set(), set()


def _resolve_explicit_toolsets(explicit: list[str], validate_toolset) -> list[str] | None | bool:
    """Resolve a HERMES_TUI_TOOLSETS pin: list, None for "all", False when nothing was valid."""
    built_in = [name for name in explicit if validate_toolset(name)]
    unresolved = [name for name in explicit if name not in built_in]
    if unresolved:
        try:
            from hermes_cli.plugins import discover_plugins
            discover_plugins()
            plugin_valid = [name for name in unresolved if validate_toolset(name)]
        except Exception:
            plugin_valid = []
        if plugin_valid:
            built_in.extend(plugin_valid)
            unresolved = [name for name in unresolved if name not in plugin_valid]
    if any(name in {"all", "*"} for name in built_in):
        ignored = [name for name in explicit if name not in {"all", "*"}]
        if ignored:
            print(
                "[tui] HERMES_TUI_TOOLSETS=all enables every toolset; "
                f"ignoring additional entries: {', '.join(ignored)}",
                file=sys.stderr,
                flush=True,
            )
        return None
    if not unresolved:
        return built_in
    mcp_names, mcp_disabled = _enabled_mcp_server_names()
    mcp_valid = [name for name in unresolved if name in mcp_names]
    disabled = [name for name in unresolved if name in mcp_disabled]
    unknown = [name for name in unresolved if name not in mcp_names and name not in mcp_disabled]
    if unknown:
        print(
            f"[tui] ignoring unknown HERMES_TUI_TOOLSETS entries: {', '.join(unknown)}",
            file=sys.stderr, flush=True,
        )
    if disabled:
        print(
            "[tui] ignoring disabled MCP servers in HERMES_TUI_TOOLSETS "
            "(set enabled: true in config.yaml to use): "
            f"{', '.join(disabled)}",
            file=sys.stderr,
            flush=True,
        )
    return (built_in + mcp_valid) or False


def _load_enabled_toolsets(platform: str | None = None) -> list[str] | None:
    """Resolve the agent's toolsets for this desktop/TUI session (None = all).

    Order: an explicit HERMES_TUI_TOOLSETS pin; else the coding posture
    (collapse to the coding toolset + enabled MCP servers when sitting in a code
    workspace — agent/coding_context.py, config loaded lazily there); else the
    configured CLI toolsets.  The client-surface (pane/project) toolsets are off
    _HERMES_CORE_TOOLS so no other platform carries their schema; this resolver
    runs only in the desktop/TUI gateway, so folding them in here is the gate
    that exposes them on exactly the surface that can answer them.
    """
    session_platform = platform or _resolve_session_platform()
    explicit = [
        item.strip()
        for item in os.environ.get("HERMES_TUI_TOOLSETS", "").split(",")
        if item.strip()
    ]
    fallback_notice = None
    if not explicit:
        with contextlib.suppress(Exception):
            from agent.coding_context import coding_selection
            selection = coding_selection(platform=session_platform)
            if selection is not None:
                return sorted({*selection, *_gui_surface_toolsets(session_platform)})
    try:
        from toolsets import validate_toolset
    except Exception:
        validate_toolset = None
    if explicit and validate_toolset is not None:
        resolved = _resolve_explicit_toolsets(explicit, validate_toolset)
        if resolved is not False:
            return resolved
        fallback_notice = (
            "[tui] no valid HERMES_TUI_TOOLSETS entries; using configured CLI toolsets"
        )
    try:
        from hermes_cli.config import load_config
        from hermes_cli.tools_config import _get_platform_tools
        cfg = load_config()
        # include_default_mcp_servers=True is the runtime variant (the agent
        # must be able to call default MCP servers); False is the config-editing
        # variant.  Using the wrong one here silently drops MCP tools from the TUI.
        enabled = _get_platform_tools(cfg, "cli", include_default_mcp_servers=True)
        if fallback_notice is not None:
            print(fallback_notice, file=sys.stderr, flush=True)
        if not enabled:
            return None
        return sorted(enabled | _gui_surface_toolsets(session_platform))
    except Exception:
        if fallback_notice is not None:
            print(
                "[tui] no valid HERMES_TUI_TOOLSETS entries and configured CLI toolsets could not be loaded; enabling all toolsets",
                file=sys.stderr,
                flush=True,
            )
        return None


def _session_tool_progress_mode(sid: str) -> str:
    return str(_sessions.get(sid, {}).get("tool_progress_mode", "all") or "all")


def _session_verbose(sid: str) -> bool:
    return _session_tool_progress_mode(sid) == "verbose"


def _tool_progress_enabled(sid: str) -> bool:
    return _session_tool_progress_mode(sid) != "off"


def _tool_lifecycle_required_for_ui(name: str) -> bool:
    """Return True for tool events that are interactive UI, not optional chrome."""
    # Desktop renders clarify / setup_mcp cards from the tool-call part; with
    # tool progress off, suppressing them would leave only the sidebar dot.
    return name in ("clarify", "setup_mcp")


def _restart_slash_worker(sid: str, session: dict):
    worker = session.get("slash_worker")
    # Nothing to replace for a session that never spawned a worker; spawning
    # here would fork the per-worker MCP fleet for nothing.
    if worker is None:
        return
    with contextlib.suppress(Exception):
        worker.close()
    try:
        new_worker = _SlashWorker(
            session["session_key"], getattr(session.get("agent"), "model", _resolve_model()),
            profile_home=session.get("profile_home"),
        )
    except Exception:
        session["slash_worker"] = None
        return
    # Store-iff-still-mapped: the post-turn restart races a close_on_disconnect
    # reap, and a bare store would orphan the fresh worker.
    _attach_worker(sid, session, new_worker)


def _get_usage(agent) -> dict:
    g = lambda k, fb=None: getattr(agent, k, 0) or (getattr(agent, fb, 0) if fb else 0)
    usage = {
        "model": getattr(agent, "model", "") or "",
        "input": g("session_input_tokens", "session_prompt_tokens"),
        "output": g("session_output_tokens", "session_completion_tokens"),
        "reasoning": g("session_reasoning_tokens"), "prompt": g("session_prompt_tokens"),
        "completion": g("session_completion_tokens"), "total": g("session_total_tokens"),
        "calls": g("session_api_calls"),
    }
    comp = getattr(agent, "context_compressor", None)
    if comp:
        # context_used is *current-window* occupancy. Never fall back to
        # usage["total"] (cumulative lifetime) — an external engine without
        # last_prompt_tokens then showed 1.9m/120k clamped to 100%. A falsy
        # last_prompt_tokens emits NO gauge rather than a fabricated one; the -1
        # "compression just ran" sentinel is clamped to 0 for the same reason
        # (matches cli.py _get_status_bar_snapshot).
        last_prompt = getattr(comp, "last_prompt_tokens", 0) or 0
        if last_prompt < 0:
            last_prompt = 0
        ctx_max = getattr(comp, "context_length", 0) or 0
        if ctx_max and last_prompt:
            usage["context_used"] = last_prompt
            usage["context_max"] = ctx_max
            usage["context_percent"] = max(0, min(100, round(last_prompt / ctx_max * 100)))
        usage["compressions"] = getattr(comp, "compression_count", 0) or 0
    # Cache-hit ratio + rolling latency/tps (CLI status-bar parity):
    # hit = cache_read / prompt_tokens (prompt = input + cache_read + cache_write);
    # latency/tps read the per-call deque history from conversation_loop.
    # Omitted, not fabricated, when there is no data (Codex reports no latency;
    # zero cache reads shows no hit% rather than an alarming 0).
    with contextlib.suppress(Exception):
        _prompt_total = int(getattr(agent, "session_prompt_tokens", 0) or 0)
        _cache_read = int(getattr(agent, "session_cache_read_tokens", 0) or 0)
        if _prompt_total > 0 and _cache_read > 0:
            usage["cache_hit_pct"] = max(0, min(100, round(_cache_read / _prompt_total * 100)))
    try:
        _lhist = list(getattr(agent, "_api_latency_history", []) or [])
        _ohist = list(getattr(agent, "_api_output_history", []) or [])
        _n = min(len(_lhist), len(_ohist))
        if _n:
            _lhist = _lhist[-_n:]
            _ohist = _ohist[-_n:]
            _avg_lat = sum(_lhist) / _n
            _total_lat = sum(_lhist)
            _avg_vel = (sum(_ohist) / _total_lat) if _total_lat > 0 else None
            # Guard NaN/negative/absurd values from odd provider timings.
            if _avg_lat == _avg_lat and 0 < _avg_lat < 1e6:
                usage["avg_latency_s"] = round(float(_avg_lat), 1)
            if _avg_vel is not None and _avg_vel == _avg_vel and 0 < _avg_vel < 1e6:
                usage["avg_tps"] = round(float(_avg_vel), 1)
    except Exception:
        # A status-bar readout must never break usage reporting.
        pass
    # Live count of background/async subagents still running (delegate_task
    # batches + background single delegations). Mirrors the classic CLI status
    # bar's ⛓ indicator; sourced from the same async_delegation registry.
    with contextlib.suppress(Exception):
        from tools.async_delegation import active_count as _async_active_count
        usage["active_subagents"] = _async_active_count()
    # Dev-only live credits-spent readout (L0 usage-aware-credits). Gated on
    # HERMES_DEV_CREDITS so the payload stays clean when the flag is off.
    if is_truthy_value(os.environ.get("HERMES_DEV_CREDITS")):
        with contextlib.suppress(Exception):
            spent = agent.get_credits_spent_micros()
            if spent is not None:
                usage["dev_credits_spent_micros"] = int(spent)
    return usage


def _probe_credentials(agent) -> str:
    """Light credential check at session creation — returns warning or ''.

    ``no-key-required`` is a valid sentinel for keyless custom providers; only
    warn when the key is genuinely missing.
    """
    with contextlib.suppress(Exception):
        key = getattr(agent, "api_key", "") or ""
        provider = getattr(agent, "provider", "") or ""
        if not key:
            return f"No API key configured for provider '{provider}'. First message will fail."
    return ""


def _probe_config_health(cfg: dict) -> str:
    """Flag bare YAML keys (`agent:` with no value → None) that silently
    drop nested settings. Returns warning or ''."""
    if not isinstance(cfg, dict):
        return ""
    warnings: list[str] = []
    null_keys = sorted(k for k, v in cfg.items() if v is None)
    if not null_keys:
        pass
    else:
        keys = ", ".join(f"`{k}`" for k in null_keys)
        warnings.append(
            f"config.yaml has empty section(s): {keys}. "
            f"Remove the line(s) or set them to `{{}}` — "
            f"empty sections silently drop nested settings."
        )
    display_cfg = cfg.get("display")
    agent_cfg = cfg.get("agent")
    if isinstance(display_cfg, dict):
        personality = str(display_cfg.get("personality", "") or "").strip().lower()
        if personality and personality not in {"default", "none", "neutral"}:
            with contextlib.suppress(Exception):
                from hermes_cli.personality import available_personalities
                if personality not in available_personalities(cfg):
                    warnings.append(
                        f"`display.personality: {personality}` does not match any "
                        "built-in or `agent.personalities` entry; personality "
                        "overlay will be skipped."
                    )
    _ = agent_cfg  # retained for shape parity; built-ins exist without config
    return " ".join(warnings).strip()


def _current_profile_name() -> str:
    try:
        from hermes_cli.profiles import get_active_profile_name
        return get_active_profile_name() or "default"
    except Exception:
        return "default"


# Monotonic GUI<->backend contract version: the desktop refuses a backend
# reporting less (or none) with a one-click "update to align" prompt. Bump
# whenever the desktop's backend contract changes.
# v2: adds the file.attach RPC (remote-gateway non-image file upload).
# v3: adds approvals.mode config RPCs and session.info reconciliation.
# v4: session.create fast=false is an explicit per-session normal-tier override.
# v5: uvicorn ws_max_size raised for one-shot base64 file.attach frames (>16 MiB).
# v6: plugins.manage list rows carry the canonical registry key; toggles are
#     key-addressed (keyless rows render read-only in Desktop Settings).
DESKTOP_BACKEND_CONTRACT = 6


def _session_usage_snapshot(session: dict | None) -> dict:
    agent = (session or {}).get("agent")
    mirror_usage = _metadata_mirror(session).get("usage")
    if (session or {}).get("_compute_host_active") and isinstance(mirror_usage, dict):
        return dict(mirror_usage)
    if agent is not None:
        return _get_usage(agent)
    return dict(mirror_usage) if isinstance(mirror_usage, dict) else {}


def _project_info_for_cwd(cwd: str) -> dict | None:
    """Return the first-class Project owning ``cwd`` for UI status surfaces.

    Backed by the per-profile projects.db (the same store the desktop's project
    tree caches), so the TUI status label, the desktop status bar, and ``/status``
    all name the session's workspace identically. Only explicit, named projects
    resolve here — an auto-discovered repo root has no projects.db row, so it
    falls back to the cwd leaf on every surface.
    """
    if not str(cwd or "").strip():
        return None
    try:
        from hermes_cli import projects_db as pdb
        with pdb.connect_closing() as conn:
            project = pdb.project_for_path(conn, cwd)
        if project is None:
            return None
        return {
            "id": project.id, "slug": project.slug, "name": project.name,
            "primary_path": project.primary_path,
        }
    except Exception:
        logger.debug("failed to resolve project for cwd", exc_info=True)
        return None


def _session_info(agent, session: dict | None = None) -> dict:
    if session is None:
        for candidate in _sessions.values():
            if candidate.get("agent") is agent:
                session = candidate
                break
    mirror = _metadata_mirror(session)
    cwd = _display_session_cwd(session)
    session_key = str((session or {}).get("session_key") or getattr(agent, "session_id", "") or "")
    cfg_personality = _display_cfg().get("personality") or ""
    personality = (session or {}).get("personality", cfg_personality)
    reasoning_config = getattr(agent, "reasoning_config", None)
    reasoning_effort = ""
    if isinstance(reasoning_config, dict):
        if reasoning_config.get("enabled") is False:
            # Disabled must differ from unset ("" = provider default), or the
            # desktop adopts "" after the first turn and loses "thinking off".
            reasoning_effort = "none"
        else:
            reasoning_effort = str(reasoning_config.get("effort", "") or "")
    service_tier = getattr(agent, "service_tier", None) or mirror.get("service_tier") or ""
    # Effective approval bypass = the same three sources check_all_command_guards()
    # ORs: approvals.mode=off, the process --yolo env, the per-session flag.
    # Reporting only the session flag would show YOLO "off" while config
    # silently auto-approves every dangerous command.
    yolo = False
    approval_mode = "manual"
    try:
        from tools.approval import _YOLO_MODE_FROZEN, is_session_yolo_enabled
        session_yolo = (bool(is_session_yolo_enabled(session_key)) if session_key else False)
        approval_mode = _load_approval_mode()
        yolo = bool(_YOLO_MODE_FROZEN) or session_yolo or approval_mode == "off"
    except Exception:
        yolo = False
    # A switch queued mid-turn applies at the next turn start, so agent.model
    # still reads the OLD model; report the pending pick so the end-of-turn
    # settle doesn't blip the UI back before the switch lands.
    pending_switch = (session or {}).get("pending_model_switch") or {}
    pending_model = str(pending_switch.get("display_model") or "").strip()
    pending_provider = str(pending_switch.get("display_provider") or "").strip()
    # Epoch seconds the current turn started, or None when idle. Lets the
    # desktop preserve the turn-elapsed timer across session switches (cold
    # resume path) instead of resetting it to 0:00.
    inflight = (session or {}).get("inflight_turn")
    turn_started_at = (
        float(inflight["started_at"])
        if isinstance(inflight, dict) and inflight.get("started_at")
        else None
    )
    info: dict = {
        "model": pending_model or mirror.get("model", getattr(agent, "model", "")),
        "provider": pending_provider
        or mirror.get("provider", getattr(agent, "provider", "")),
        "reasoning_effort": reasoning_effort,
        "service_tier": service_tier,
        "fast": service_tier == "priority",
        "yolo": yolo,
        "approval_mode": approval_mode,
        "tools": dict(mirror.get("tools") or {}) if isinstance(mirror.get("tools"), dict) else {},
        "skills": dict(mirror.get("skills") or {}) if isinstance(mirror.get("skills"), dict) else {},
        "cwd": cwd,
        "branch": _git_branch_for_cwd(cwd),
        "project": _project_info_for_cwd(cwd),
        "terminal_backend": _effective_terminal_backend(),
        "personality": str(personality or ""),
        "running": bool((session or {}).get("running")),
        "turn_started_at": turn_started_at,
        "title": _session_live_title(session or {}, session_key) if session_key else "",
        "stored_session_id": session_key or "",
        "desktop_contract": DESKTOP_BACKEND_CONTRACT,
        "version": "",
        "release_date": "",
        "update_behind": None,
        "update_command": "",
        "usage": _session_usage_snapshot(session),
        "profile_name": (
            _response_profile_name(Path(session["profile_home"]).name)
            if isinstance(session, dict) and session.get("profile_home")
            else _current_profile_name()
        ),
    }
    with contextlib.suppress(Exception):
        from hermes_cli import __version__, __release_date__
        info["version"] = __version__
        info["release_date"] = __release_date__
    live_agent = agent is not None and not (session or {}).get("_compute_host_active")
    if live_agent:
        with contextlib.suppress(Exception):
            from model_tools import get_toolset_for_tool
            info["tools"] = {}
            for t in getattr(agent, "tools", []) or []:
                name = t["function"]["name"]
                info["tools"].setdefault(get_toolset_for_tool(name) or "other", []).append(name)
        with contextlib.suppress(Exception):
            from hermes_cli.banner import get_available_skills
            info["skills"] = get_available_skills()
    try:
        from tools.mcp_tool import get_mcp_status
        info["mcp_servers"] = get_mcp_status()
    except Exception:
        info["mcp_servers"] = []
    with contextlib.suppress(Exception):
        info["system_prompt"] = (
            mirror.get("system_prompt")
            if "system_prompt" in mirror
            else getattr(agent, "_cached_system_prompt", "") or ""
        )
    with contextlib.suppress(Exception):
        from hermes_cli.banner import get_update_result
        from hermes_cli.config import recommended_update_command
        info["update_behind"] = get_update_result(timeout=0.5)
        info["update_command"] = recommended_update_command()
    if live_agent and (warn := _probe_credentials(agent)):
        info["credential_warning"] = warn
    return info


def _tool_ctx(name: str, args: dict) -> str:
    """Argument preview for a tool row — never a phrased label.

    Clients own their own phrasing: the TUI wraps this as ``Terminal("...")``
    and the desktop prepends its own localized verb ("Running"/"Ran"). Sending
    ``build_tool_label`` here instead of the raw preview stutters the verb on
    both surfaces ("Running Running sleep 70 + 2 commands") and leaks a display
    label into the desktop's ``args.context``, where it stands in for the real
    command. The friendly labels belong on the CLI spinner, which builds them
    from ``build_tool_label`` at its own call sites.
    """
    try:
        from agent.display import build_tool_preview
        return build_tool_preview(name, args, max_len=80) or ""
    except Exception:
        return ""


def _emit_session_info_for_session(sid: str, session: dict) -> None:
    agent = session.get("agent")
    if agent is None and not _metadata_mirror(session):
        return
    with contextlib.suppress(Exception):
        _emit("session.info", sid, _session_info(agent, session))


def broadcast_session_info() -> None:
    """Re-emit ``session.info`` to every live session.

    For approvals-config writers that bypass the ``config.set`` RPC (which
    re-emits itself): the REST config saves and the ``/approvals`` slash
    mirror. Only reaches sessions in THIS process; a spawned
    ``tui_gateway.entry`` child gateway has its own ``_sessions``.
    """
    with _sessions_lock:
        sessions = list(_sessions.items())
    for sid, sess in sessions:
        _emit_session_info_for_session(sid, sess)


# Tool Args/Result text shipped to the TUI for the verbose trail line. The TUI
# renders only a small persisted preview (ui-tui VERBOSE_TRAIL_MAX_CHARS), kept
# all session and expanded by default — so shipping more than that is pure pipe
def _schedule_mcp_late_refresh(sid: str, agent) -> None:
    """Refresh a session's tool snapshot when MCP discovery lands late.

    The agent snapshots ``agent.tools`` once at build; ``_make_agent`` only
    waits a bounded ``mcp_discovery_timeout`` (default 1.5s), so a slow server
    (HTTP MCP on first connect) lands after the build and its tools are missing
    for the whole session.  A daemon waits for discovery to finish, then does
    the same rebuild ``/reload-mcp`` performs and re-emits ``session.info``.

    Cache safety: the rebuild runs only while the session is pre-first-turn
    (nothing cached to invalidate).  Once a message was sent the snapshot stays
    frozen — late tools then need an explicit, consent-gated ``/reload-mcp``.
    No-op when discovery already finished before the build.
    """
    try:
        from tui_gateway.entry import mcp_discovery_in_flight, join_mcp_discovery
    except Exception:
        return
    if not mcp_discovery_in_flight():
        return

    def _wait_then_refresh() -> None:
        # Bounded but generous — a server still not connected after this is
        # genuinely slow/dead; the user can /reload-mcp once it recovers.
        if not join_mcp_discovery(timeout=30.0):
            return
        with _sessions_lock:
            session = _sessions.get(sid)
            # Session may have been closed/reset while we waited.
            if session is None or session.get("agent") is not agent:
                return
            # Cache safety: never rebuild the tool list once the conversation
            # has started — that would invalidate the cached prompt prefix.
            if (
                int(getattr(agent, "_user_turn_count", 0) or 0) > 0
                or int(getattr(agent, "_api_call_count", 0) or 0) > 0
            ):
                return
            try:
                from tools.mcp_tool import refresh_agent_mcp_tools
                added = refresh_agent_mcp_tools(agent, quiet_mode=True)
            except Exception as exc:
                logger.warning(
                    "Late MCP refresh: tool snapshot rebuild failed for %s: %s", sid, exc,
                )
                return
            # No new tools landed (discovery added nothing) → don't churn the client.
            if not added:
                return
            info = _session_info(agent, session)
        # Emit outside the lock — write_json must not block under _sessions_lock.
        _emit("session.info", sid, info)
    threading.Thread(
        target=_wait_then_refresh, name=f"tui-mcp-late-refresh-{sid}", daemon=True,
    ).start()


class _RuntimeFallbackResolution(NamedTuple):
    runtime: dict
    selected_model: str | None
    used_fallback: bool


def _resolve_runtime_with_fallback(
    resolve_kwargs: dict | None = None,
) -> _RuntimeFallbackResolution:
    """Resolve the primary runtime or one complete provider/model fallback.

    Setup-time auth fallback only accepts entries with both fields. Provider-
    only entries are skipped so the unavailable primary model can never leak
    into a different runtime. ``used_fallback`` remains explicit rather than
    overloading a nullable model as control flow.
    """
    from hermes_cli.auth import AuthError
    from hermes_cli.runtime_provider import resolve_runtime_provider
    kwargs = resolve_kwargs or {}
    try:
        return _RuntimeFallbackResolution(resolve_runtime_provider(**kwargs), None, False)
    except AuthError as primary_exc:
        fb_chain = _load_fallback_model() or []
        for entry in fb_chain:
            if not isinstance(entry, dict):
                continue
            fb_provider = str(entry.get("provider") or "").strip()
            fb_model = str(entry.get("model") or "").strip()
            if not fb_provider or not fb_model:
                continue
            try:
                from hermes_cli.fallback_config import resolve_entry_api_key
                fb_kwargs: dict = {"requested": fb_provider, "target_model": fb_model}
                if entry.get("base_url"):
                    fb_kwargs["explicit_base_url"] = entry["base_url"]
                fb_api_key = resolve_entry_api_key(entry)
                if fb_api_key:
                    fb_kwargs["explicit_api_key"] = fb_api_key
                runtime = resolve_runtime_provider(**fb_kwargs)
                import logging
                logging.getLogger(__name__).warning(
                    "Primary auth failed (%s), falling back to %s model %s", primary_exc,
                    fb_provider, fb_model,
                )
                return _RuntimeFallbackResolution(runtime, fb_model, True)
            except Exception:
                continue
        raise


def _resolve_agent_model_runtime(model_override, provider_override) -> tuple[str, dict]:
    """Resolve (model, runtime) for a new agent.

    A per-session override (prior in-session /model switch, or the persisted
    runtime of a resumed row) wins over global config/env resolution.  Rows
    persisted before the custom-provider identity fix stored the resolved
    provider "custom", which no named ``providers:`` entry matches — recover
    the entry identity from the persisted base_url (falling back to the
    configured provider) or the rebuild surfaces as "No LLM provider
    configured".  Persisted base_url/api_key/api_mode are honored only while
    the original runtime is used; they must not leak into a fallback pair.
    """
    if isinstance(model_override, dict) and model_override.get("model"):
        model = str(model_override.get("model") or "")
        requested_provider = model_override.get("provider") or provider_override or None
        override_base_url = model_override.get("base_url")
        resolve_kwargs = {}
        if str(requested_provider or "").strip().lower() == "custom":
            from hermes_cli.runtime_provider import canonical_custom_identity
            recovered = canonical_custom_identity(base_url=override_base_url or None, model=model or None)
            if recovered:
                requested_provider = recovered
            if override_base_url:
                # Failing identity recovery, still hand the base_url to the
                # direct-alias branch so pool/env credentials resolve for it.
                resolve_kwargs["explicit_base_url"] = override_base_url
        resolve_kwargs["requested"] = requested_provider
        resolve_kwargs["target_model"] = model or None
        overrides = {
            "base_url": override_base_url, "api_key": model_override.get("api_key"),
            "api_mode": model_override.get("api_mode"),
        }
    else:
        model, requested_provider = _resolve_startup_runtime()
        if isinstance(model_override, str) and model_override:
            model = model_override
        if provider_override:
            requested_provider = provider_override
        resolve_kwargs = {"requested": requested_provider, "target_model": model or None}
        overrides = {}
    resolution = _resolve_runtime_with_fallback(resolve_kwargs)
    runtime = resolution.runtime
    if resolution.used_fallback:
        if not resolution.selected_model:
            raise RuntimeError("Auth fallback resolved without a model")
        return resolution.selected_model, runtime
    for k, v in overrides.items():
        if v:
            runtime[k] = v
    return model, runtime


def _make_agent(
    sid: str, key: str, session_id: str | None = None, session_db=None,
    model_override: dict | str | None = None, provider_override: str | None = None,
    reasoning_config_override: dict | None = None, service_tier_override: str | None = None,
    platform_override: str | None = None, context_cwd_is_launch_artifact: bool | None = None,
):
    # AC-4 test seam: dead unless armed by the isolated certify harness.
    from tui_gateway.synthetic_turn import maybe_build_synthetic_agent
    synthetic = maybe_build_synthetic_agent(session_id or key, model_override)
    if synthetic is not None:
        return synthetic
    from run_agent import AIAgent

    # MCP discovery runs in a background daemon thread so a dead server can't
    # freeze the shell; the agent snapshots its tool list once, so briefly
    # (bounded) wait for in-flight discovery.  Dashboard /api/ws uses
    # hermes_cli.mcp_startup; TUI stdio keeps the tui_gateway.entry thread.
    for _mod in ("hermes_cli.mcp_startup", "tui_gateway.entry"):
        with contextlib.suppress(Exception):
            importlib.import_module(_mod).wait_for_mcp_discovery()
    cfg = _load_cfg()
    from hermes_cli.config import resolve_ephemeral_system_prompt_from_config
    system_prompt = resolve_ephemeral_system_prompt_from_config(cfg)
    startup_skills = _parse_tui_skills_env()
    if startup_skills:
        from agent.skill_commands import build_preloaded_skills_prompt
        skills_prompt, loaded_skills, missing_skills = build_preloaded_skills_prompt(
            startup_skills, task_id=session_id or key,
        )
        if missing_skills:
            missing_display = ", ".join(missing_skills)
            # Hard-fail only when EVERY requested skill is missing (cli.py
            # parity): a typo'd name must not auto-block the Kanban task.
            if loaded_skills:
                logger.warning(
                    "Unknown skill(s) requested, skipping: %s. "
                    "Continuing with: %s. "
                    "List available skills with `hermes skills list`.",
                    missing_display,
                    ", ".join(loaded_skills),
                )
            else:
                raise ValueError(f"Unknown skill(s): {missing_display}")
        if skills_prompt:
            system_prompt = "\n\n".join(
                part for part in (system_prompt, skills_prompt) if part
            ).strip()
    model, runtime = _resolve_agent_model_runtime(model_override, provider_override)
    _pr = _load_provider_routing()
    agent = AIAgent(
        model=model,
        max_iterations=_cfg_max_turns(cfg, 500),
        provider=runtime.get("provider"),
        base_url=runtime.get("base_url"),
        api_key=runtime.get("api_key"),
        api_mode=runtime.get("api_mode"),
        acp_command=runtime.get("command"),
        acp_args=runtime.get("args"),
        credential_pool=runtime.get("credential_pool"),
        quiet_mode=True,
        verbose_logging=False,  # DEBUG agent logging; independent of tool_progress_mode
        reasoning_config=(
            reasoning_config_override
            if reasoning_config_override is not None
            else _load_reasoning_config(str(model or ""))
        ),
        service_tier=(
            service_tier_override
            if service_tier_override is not None
            else _load_service_tier()
        ),
        enabled_toolsets=_load_enabled_toolsets(_resolve_agent_platform(platform_override)),
        # OpenRouter provider_routing prefs (gateway + CLI parity).
        providers_allowed=_pr.get("only"),
        providers_ignored=_pr.get("ignore"),
        providers_order=_pr.get("order"),
        provider_sort=_pr.get("sort"),
        provider_require_parameters=_pr.get("require_parameters", False),
        provider_data_collection=_pr.get("data_collection"),
        platform=_resolve_agent_platform(platform_override),
        session_id=session_id or key,
        session_db=session_db if session_db is not None else _get_db(),
        ephemeral_system_prompt=system_prompt or None,
        checkpoints_enabled=is_truthy_value(os.environ.get("HERMES_TUI_CHECKPOINTS")),
        pass_session_id=is_truthy_value(os.environ.get("HERMES_TUI_PASS_SESSION_ID")),
        skip_context_files=is_truthy_value(os.environ.get("HERMES_IGNORE_RULES")),
        skip_memory=is_truthy_value(os.environ.get("HERMES_IGNORE_RULES")),
        fallback_model=_load_fallback_model(),
        **_agent_cbs(sid),
    )
    if context_cwd_is_launch_artifact is None:
        with _sessions_lock:
            context_session = _sessions.get(sid)
        context_cwd_is_launch_artifact = _context_cwd_is_launch_artifact(context_session)
    agent._context_cwd_is_launch_artifact = bool(context_cwd_is_launch_artifact)
    return agent


def _init_session(
    sid: str, key: str, agent, history: list, cols: int = 80, cwd: str | None = None,
    session_db=None, source: str | None = None, profile_home: str | None = None,
    explicit_cwd: bool = False,
):
    now = time.time()
    with _sessions_lock:
        _sessions[sid] = {
            "agent": agent,
            "session_key": key,
            "history": history,
            "history_lock": threading.Lock(),
            "history_version": 0,
            "inflight_turn": None,
            "created_at": now,
            "last_active": now,
            "running": False,
            "attached_images": [],
            "image_counter": 0,
            "cwd": cwd or _completion_cwd(),
            "explicit_cwd": bool(explicit_cwd),
            "cols": cols,
            "slash_worker": None,
            "show_reasoning": _load_show_reasoning(),
            "source": _resolve_session_source(source),
            "tool_progress_mode": _load_tool_progress_mode(),
            "edit_snapshots": {},
            "tool_started_at": {},
            # Profile-scoped HERMES_HOME (None = launch profile); SessionBranch
            # copies the parent's so the child stays on the same state.db.
            "profile_home": profile_home,
            # In-session /model switch, honored on rebuild (/new, resume) so it
            # never leaks into siblings via process-global env vars.
            "model_override": None,
            # Async events go to the transport that created the session
            # (stdio for Ink, JSON-RPC WS for the dashboard sidebar).
            "transport": current_transport() or _stdio_transport,
        }
        _session_todo_state(_sessions[sid])
    _init_owns_db = False
    if session_db is not None:
        db = session_db
    elif profile_home:
        try:
            db = _open_profile_session_db(profile_home)
            _init_owns_db = True
        except Exception:
            # FAIL CLOSED (same class as the deferred-build bind): a named-profile
            # session must never touch the launch state.db — skip cwd hydration
            # (the row lands on the agent's own lazy-create once the store recovers).
            logger.warning(
                "profile session store unavailable for %s — skipping cwd "
                "hydration instead of touching the launch state.db",
                profile_home,
                exc_info=True,
            )
            db = None
    else:
        db = _get_db()
    try:
        if db is not None:
            row = db.get_session(key) if hasattr(db, "get_session") else None
            if row and row.get("cwd"):
                with _sessions_lock:
                    if sid in _sessions:
                        _sessions[sid]["cwd"] = row["cwd"]
            else:
                try:
                    _cwd = _sessions[sid]["cwd"]
                    if hasattr(db, "update_session_cwd"):
                        _persist_session_cwd_and_schedule_git_meta(_sessions[sid], _cwd, db=db)
                except Exception:
                    logger.debug("failed to persist resumed session cwd", exc_info=True)
    finally:
        if _init_owns_db and db is not None:
            with contextlib.suppress(Exception):
                db.close()
    _register_session_cwd(_sessions[sid])
    # No eager slash-worker pre-warm (see _start_agent_build).
    _wire_session_agent(sid, key, agent)
    _start_session_services(sid, key, _sessions.get(sid, {}))
    _emit("session.info", sid, _session_info(agent, _sessions.get(sid, {})))
    _schedule_mcp_late_refresh(sid, agent)


def _new_session_key() -> str:
    return f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"


def _with_checkpoints(session, fn):
    return fn(session["agent"]._checkpoint_mgr, _session_cwd(session))


def _resolve_checkpoint_hash(mgr, cwd: str, ref: str) -> str:
    try:
        checkpoints = mgr.list_checkpoints(cwd)
        idx = int(ref) - 1
    except ValueError:
        return ref
    if 0 <= idx < len(checkpoints):
        return checkpoints[idx].get("hash", ref)
    raise ValueError(f"Invalid checkpoint number. Use 1-{len(checkpoints)}.")


# ── Methods: session ─────────────────────────────────────────────────


def _lazy_resume_info(
    cwd: str, *, model: str = "", provider: str = "", profile: str | None = None,
) -> dict:
    """session.info for a not-yet-built session (the shape session.create
    returns). tools/skills land later when the deferred build emits session.info."""
    info = {
        "cwd": cwd, "branch": _git_branch_for_cwd(cwd), "project": _project_info_for_cwd(cwd),
        "model": model or _resolve_model(), "tools": {}, "skills": {}, "lazy": True,
        "desktop_contract": DESKTOP_BACKEND_CONTRACT,
        "profile_name": _response_profile_name(profile),
    }
    if provider:
        info["provider"] = provider
    return info


def _deferred_session_record(
    session_key: str, *, cols: int, cwd: str, history: list, lease, source: str = "tui",
    close_on_disconnect: bool = False, display_history_prefix: list | None = None,
    profile_home: Path | None = None, lazy: bool = False, model_override=None,
    resume_runtime_overrides: dict | None = None, todo_state: dict | None = None,
    explicit_cwd: bool = False,
) -> dict:
    """A live-session record whose AIAgent is built later (lazy watch / cold
    resume) — _init_session's shape minus the agent."""
    now = time.time()
    return {
        "agent": None, "agent_error": None, "agent_ready": threading.Event(), "attached_images": [],
        "close_on_disconnect": close_on_disconnect, "active_session_lease": lease, "cols": cols,
        "created_at": now, "cwd": cwd, "display_history_prefix": display_history_prefix or [],
        "edit_snapshots": {}, "explicit_cwd": bool(explicit_cwd), "history": history,
        "history_lock": threading.Lock(), "history_version": 0, "image_counter": 0,
        "inflight_turn": None, "last_active": now, "lazy": lazy, "model_override": model_override,
        "pending_title": None,
        "profile_home": str(profile_home) if profile_home is not None else None,
        "resume_runtime_overrides": resume_runtime_overrides, "resume_session_id": session_key,
        "running": False, "session_key": session_key, "show_reasoning": _load_show_reasoning(),
        "slash_worker": None, "source": source, "tool_progress_mode": _load_tool_progress_mode(),
        "tool_started_at": {}, "todo_state": todo_state,
        "transport": current_transport() or _stdio_transport,
    }


_ANY_PROFILE = object()  # default: match a live session regardless of profile


def _live_profile_matches(session: dict, profile_home) -> bool:
    """True when ``session`` belongs to ``profile_home`` (None = launch profile).

    Same string compare as session.resume's ``_find_live_unpersisted``: a
    record with no ``profile_home`` is the launch profile's. ``_ANY_PROFILE``
    disables the check for callers that have no profile to scope by.
    """
    if profile_home is _ANY_PROFILE:
        return True
    want = str(profile_home) if profile_home else None
    return (session.get("profile_home") or None) == want


def _claim_or_reuse_live(
    sid: str, session_key: str, record: dict, lease
) -> tuple[str, dict] | None:
    """Register ``record`` as the live session for ``session_key`` under the
    resume lock, or — if a concurrent resume already won — release ``lease`` and
    return the winner for the caller to reuse."""
    # The record carries the home this resume resolved; a live runtime of the
    # same stored id under ANOTHER profile is not a winner to reuse (#100029).
    profile_home = record.get("profile_home")
    with _session_resume_lock:
        live = _find_live_session_by_key(session_key, profile_home)
        if live is not None:
            if lease is not None:
                lease.release()
            # The winner is being reattached by this resume: any pending
            # ws-orphan reap for it must not fire against the reclaimed
            # client (storm killer — see _cancel_ws_orphan_reap).
            _cancel_ws_orphan_reap(live[0])
            return live
        with _sessions_lock:
            _sessions[sid] = record
            _register_session_cwd(_sessions[sid])
        # A PRIOR runtime for this stored id may still be sentinel-parked with
        # a reap Timer armed; cancel + finalize it quietly so the reap doesn't
        # broadcast session.reclaimed for a just-re-resumed session (storm).
        _cancel_ws_orphan_reap(sid)
        stale = _claim_parked_runtimes(session_key, keep_sid=sid, profile_home=profile_home)
    # Slow finalization work stays OUTSIDE _session_resume_lock (see
    # _pop_session_by_id) — the stale records are already claimed above.
    _finalize_superseded_runtimes(stale)
    return None


def _claim_parked_runtimes(
    session_key: str, *, keep_sid: str, profile_home=_ANY_PROFILE
) -> list[tuple[str, dict]]:
    """Claim sentinel-parked stale runtimes of ``session_key`` for supersession.

    When a resume mints a fresh runtime for stored session id ``session_key``,
    any older runtime record for the same stored id that is still parked on
    the detached-WS sentinel is superseded: its pending orphan-reap Timer is
    cancelled and the record is atomically popped from ``_sessions`` here
    (under the caller's _session_resume_lock), then finalized by
    :func:`_finalize_superseded_runtimes` after the lock is released.
    """
    stale: list[tuple[str, dict]] = []
    with _sessions_lock:
        candidates = [
            (old_sid, old)
            for old_sid, old in list(_sessions.items())
            if old_sid != keep_sid
            and not old.get("_finalized")
            and _session_lookup_key(old, fallback=old_sid) == session_key
            and _live_profile_matches(old, profile_home)
            and old.get("transport") is _detached_ws_transport
        ]
    for old_sid, _old in candidates:
        _cancel_ws_orphan_reap(old_sid)
        popped = _pop_session_by_id(old_sid)
        if popped is not None:
            stale.append((old_sid, popped))
    return stale


def _finalize_superseded_runtimes(stale: list[tuple[str, dict]]) -> None:
    """Quietly finalize runtimes claimed by :func:`_claim_parked_runtimes`.

    Ends them with end_reason ``superseded_by_resume`` — deliberately NOT in
    _RECLAIM_END_REASONS, so no ``session.reclaimed`` broadcast fires (that
    broadcast triggers client auto-re-resume and fed the
    reap->broadcast->resume feedback loop). ``superseded_by_resume`` IS in
    hermes_state_common._RECOVERABLE_END_REASONS so canonical Bot Chat
    resurrection still applies to the stored session.
    """
    for old_sid, popped in stale:
        try:
            _teardown_popped_session(popped, end_reason="superseded_by_resume")
        except Exception:
            logger.exception("superseded runtime teardown failed sid=%s", old_sid)


def _schedule_agent_build(sid: str, delay: float = 0.05) -> None:
    """Pre-warm a deferred session's agent off the response path (session.create
    and cold resume both build through here; _sess() also builds on demand)."""

    def _run():
        session = _sessions.get(sid)
        if session is not None:
            _start_agent_build(sid, session)
    timer = threading.Timer(delay, _run)
    timer.daemon = True
    timer.start()


def _schedule_resume_hydration(sid: str, stored_id: str, db, *, close_db: bool = False) -> None:
    """Load a cold resume's transcript off the JSON-RPC response path."""

    def _run() -> None:
        session = _sessions.get(sid)
        try:
            if session is None:
                return
            _emit("session.resume_progress", sid, {"phase": "history", "status": "loading"})
            db.reopen_session(stored_id)
            from hermes_state import SessionResumeTooLargeError

            # The ancestor prefix is an in-memory convenience (the transcript
            # is REST-paginated): materialize the full lineage only while it
            # fits sessions.max_resume_messages, else hydrate the tip alone.
            prefix_fits = True
            guard = getattr(db, "assert_resume_safe", None)
            if callable(guard):
                try:
                    guard(stored_id)
                except SessionResumeTooLargeError as exc:
                    prefix_fits = False
                    logger.info(
                        "resume %s: compression lineage exceeds the resume "
                        "limit (%s); hydrating the tip segment only",
                        stored_id, exc,
                    )
                except Exception:
                    logger.debug("resume lineage guard failed; loading full lineage", exc_info=True)
            if prefix_fits:
                raw_history, display_history = db.get_resume_conversations(stored_id)
                prefix = db.get_ancestor_display_prefix(stored_id)
            else:
                raw_history = db.get_messages_as_conversation(
                    stored_id, repair_alternation=True, include_row_ids=True
                )
                display_history = raw_history
                prefix = []
            history = sanitize_replay_history(raw_history)
            if _sessions.get(sid) is not session:
                return
            with session["history_lock"]:
                session["history"] = history
                session["display_history_prefix"] = prefix
                session["resume_hydrating"] = False
                session["resume_message_count"] = len(display_history)
            # Deferred resumes answered before the transcript existed; cache
            # the derived todo snapshot now so later payload attaches carry it.
            todo_state = _todo_state_from_history(history)
            if todo_state is not None and session.get("todo_state") is None:
                session["todo_state"] = todo_state
            session["resume_history_ready"].set()
            _emit(
                "session.resume_progress", sid,
                {"message_count": len(display_history), "phase": "history", "status": "complete"},
            )
            _maybe_schedule_auto_continue(sid, session, stored_id)
            _start_agent_build(sid, session)
        except Exception as exc:
            if _sessions.get(sid) is not session:
                return
            message = f"resume failed: {exc}"
            session["resume_hydrating"] = False
            session["resume_history_error"] = message
            session["agent_error"] = message
            session["resume_history_ready"].set()
            session["agent_ready"].set()
            _emit(
                "session.resume_progress", sid,
                {"message": message, "phase": "history", "status": "failed"},
            )
            _emit("error", sid, {"message": message})
            with _sessions_lock:
                discarded = _sessions.pop(sid, None) if _sessions.get(sid) is session else None
            lease = (discarded or {}).get("active_session_lease")
            if lease is not None:
                lease.release()
        finally:
            if close_db and hasattr(db, "close"):
                try:
                    db.close()
                except Exception:
                    logger.debug("failed to close resume db for %s", sid, exc_info=True)
    threading.Thread(target=_run, daemon=True).start()


def _session_pending_kind(sid: str) -> str:
    for rid, (owner_sid, _ev) in list(_pending.items()):
        if owner_sid != sid:
            continue
        event, _payload = _pending_prompt_payloads.get(rid, ("input.request", {}))
        return str(event).removesuffix(".request")
    return ""


def _session_live_status(sid: str, session: dict) -> str:
    if _session_pending_kind(sid):
        return "waiting"
    ready = session.get("agent_ready")
    # Unset + build never started = a lazy watch session sitting idle, not a
    # session stuck mid-construction.
    if ready is not None and not ready.is_set() and session.get("agent_build_started"):
        return "starting"
    if session.get("running"):
        return "working"
    return "idle"


def _message_preview(history: list) -> str:
    for msg in reversed(history or []):
        text = _content_display_text(msg.get("content", msg.get("text", ""))).strip()
        if text:
            return " ".join(text.split())[:160]
    return ""


def _session_live_title(session: dict, key: str) -> str:
    title = str(session.get("pending_title") or "").strip()
    with contextlib.suppress(Exception):
        with _session_db(session) as db:
            if db is not None:
                title = str(db.get_session_title(key) or title or "").strip()
    return title


def _session_live_item(sid: str, session: dict, current_sid: str = "") -> dict:
    key = _session_lookup_key(session, fallback=sid)
    agent = session.get("agent")
    history = list(session.get("history") or [])
    status = _session_live_status(sid, session)
    inflight = _inflight_snapshot(session)
    queued = _queued_prompt_snapshot(session)
    preview = _message_preview(history)
    if queued:
        preview = queued.get("user") or preview
        preview = " ".join(str(preview).split())[:160]
    elif inflight:
        preview = inflight.get("assistant") or inflight.get("user") or preview
        preview = " ".join(str(preview).split())[:160]
    now = time.time()
    return {
        "current": sid == current_sid, "id": sid,
        "last_active": float(session.get("last_active") or session.get("created_at") or now),
        "message_count": len(history),
        "model": str(getattr(agent, "model", "") or _resolve_model()), "preview": preview,
        "session_key": key, "started_at": float(session.get("created_at") or now), "status": status,
        "title": _session_live_title(session, key),
    }


def _session_lookup_key(session: dict, *, fallback: str = "") -> str:
    agent = session.get("agent")
    return str(getattr(agent, "session_id", None) or session.get("session_key") or fallback or "")


def _find_live_session_by_key(
    session_key: str, profile_home=_ANY_PROFILE
) -> tuple[str, dict] | None:
    # Timestamp-based stored ids can exist in several profiles' stores; a
    # bare-id match would hand profile B's resume profile A's runtime, so
    # profile-aware callers match on (profile_home, session_key).
    for sid, session in list(_sessions.items()):
        if session.get("_finalized"):
            continue
        if _session_lookup_key(session, fallback=sid) == session_key and _live_profile_matches(
            session, profile_home
        ):
            return sid, session
    return None


def _fallback_session_info(session: dict) -> dict:
    agent = session.get("agent")
    if agent is not None:
        return _session_info(agent)
    # The SESSION's own workspace, not the gateway launch dir (that painted the
    # wrong project in the desktop Files pane). `branch` is always emitted (""
    # outside git) so a client clears a stale label — same as _lazy_session_info.
    cwd = _session_cwd(session)
    return {
        "cwd": cwd,
        "branch": _git_branch_for_cwd(cwd),
        "project": _project_info_for_cwd(cwd),
        "lazy": True,
        "model": _resolve_model(),
        "skills": {},
        "tools": {},
        # A lazy session is still served by THIS backend: a missing contract
        # field reads as 0 and flags a current backend "out of date".
        "desktop_contract": DESKTOP_BACKEND_CONTRACT,
    }


def _reconcile_display_with_live(db_display: list[dict], in_memory: list[dict]) -> list[dict]:
    """Merge the persisted DISPLAY lineage with the in-memory live history.

    Two projections of the same session that each hold something the other
    lacks:

    - ``db_display`` — the verbatim persisted lineage. It includes
      *model-invisible* rows (verification candidates, finish_reason
      ``verification_required`` / ``verify_hook_continue``) that the in-memory
      model history collapses out via ``repair_message_sequence`` (#65919), but
      it can lag the newest turn by a flush.
    - ``in_memory`` — ``display_history_prefix + session["history"]``. It is the
      freshest recency authority (a just-appended turn may not be flushed yet)
      but it is the collapsed *model* projection, so it is missing candidates.

    The merge keeps the DB display (candidate-inclusive) as the base and appends
    only the in-memory tail that the DB does not yet cover, anchored on the last
    DB row's ``(role, text)``. This satisfies BOTH invariants at once: the
    substantive verification answer survives a warm/live switch (matching the
    eager resume + REST payloads), and a not-yet-flushed live turn is not
    dropped.
    """
    if not db_display:
        return in_memory
    if not in_memory:
        return db_display

    def _key(msg: dict) -> tuple:
        return (msg.get("role"), _coerce_message_text(msg.get("content")))
    anchor = _key(db_display[-1])
    last_shared = -1
    for idx, msg in enumerate(in_memory):
        if isinstance(msg, dict) and _key(msg) == anchor:
            last_shared = idx
    if last_shared == -1:
        # The DB tail isn't present in memory (DB is ahead, or the histories
        # diverged) — trust the persisted display rather than risk duplicating.
        return db_display
    return list(db_display) + list(in_memory[last_shared + 1 :])


def _live_visible_history(session: dict, db, in_memory_fallback: list[dict]) -> list[dict]:
    """Return the user-visible DISPLAY projection for a live/warm session.

    The raw in-memory *model* history lacks model-invisible rows (verification
    candidates) that the eager ``session.resume`` display lineage shows, so the
    two payloads disagreed ("substantive answer vanishes on switch").  Reconcile
    the persisted display lineage (``get_messages_as_conversation(...,
    include_ancestors=True)``, same read as resume/REST) with the fresh
    in-memory tail; fall back to in-memory when the DB/session_key is
    unavailable or the read fails.
    """
    key = session.get("session_key")
    if db is not None and key:
        try:
            display = db.get_messages_as_conversation(
                key,
                include_ancestors=True,
                include_row_ids=True,
                # Display read: a compacted session's archived turns are still
                # the user's conversation. Without them a warm switch repainted
                # the chat as just the summary + tail while the REST transcript
                # showed everything (#92080).
                include_compacted=True,
            )
            return _reconcile_display_with_live(display, in_memory_fallback)
        except Exception:
            logger.debug("live display projection read failed", exc_info=True)
    return in_memory_fallback


def _live_session_payload(
    sid: str, session: dict, *, cols: int | None = None, touch: bool = False,
    transport: Transport | None = None, omit_messages: bool = False,
) -> dict:
    with session["history_lock"]:
        if cols is not None:
            session["cols"] = cols
        if transport is not None:
            session["transport"] = transport
            # Every transport that has shown this session (pop-out windows
            # resume the same sid); the last viewer becomes the transport on
            # disconnect instead of stranding it on the drop sentinel.
            viewers = session.setdefault("viewers", {})
            viewers[transport] = time.time()
            if transport is not _detached_ws_transport:
                # A live transport rebind means the client is back — any
                # pending ws-orphan reap must not fire (storm killer).
                _cancel_ws_orphan_reap(sid)
        if touch:
            session["last_active"] = time.time()
        in_memory_history = list(session.get("display_history_prefix") or []) + list(
            session.get("history") or []
        )
        inflight = _inflight_snapshot(session)
        queued = _queued_prompt_snapshot(session)
        running = bool(session.get("running"))
        inflight_turn = session.get("inflight_turn")
        turn_started_at = (
            float(inflight_turn["started_at"])
            if isinstance(inflight_turn, dict) and inflight_turn.get("started_at")
            else None
        )
    # Persisted display lineage (candidate-inclusive) so this matches the eager
    # resume + REST transcript; via the session's profile-aware DB, not the
    # launch ``_get_db()`` (remote-profile candidates live in profile_home).
    # The DB has its own lock — read outside the history lock. ``omit_messages``
    # skips the read entirely (fast path for counts/status).
    if omit_messages:
        history = in_memory_history
    else:
        with _session_db(session) as db:
            history = _live_visible_history(session, db, in_memory_history)
    payload = {
        "info": _fallback_session_info(session), "message_count": len(history),
        "messages": [] if omit_messages else _history_to_messages(history),
        "messages_omitted": omit_messages, "running": running, "turn_started_at": turn_started_at,
        "session_id": sid, "session_key": _session_lookup_key(session, fallback=sid),
        "started_at": float(session.get("created_at") or time.time()),
        "status": _session_live_status(sid, session),
    }
    if inflight:
        payload["inflight"] = inflight
    if queued:
        payload["queued"] = queued
    if approval := _pending_approval_request_payload(str(session.get("session_key") or "")):
        payload["pending_approval"] = approval
    if clarify := _pending_clarify_request_payload(sid):
        payload["pending_clarify"] = clarify
    return _attach_todo_state(payload, session)


def _main_runtime_from_agent(agent) -> dict | None:
    """Build an aux-client main_runtime override from a live agent.

    Lets a one-shot inherit the session's provider/model/credentials so its
    output matches the model the user is actually coding with, instead of
    falling back to the cheapest auto-detected backend.
    """
    if agent is None:
        return None
    runtime: dict = {}
    for field in ("provider", "model", "base_url", "api_key", "api_mode", "auth_mode"):
        value = getattr(agent, field, None)
        if isinstance(value, str) and value.strip():
            runtime[field] = value.strip()
        elif field == "api_key" and callable(value):
            runtime[field] = value
    return runtime or None


def _pet_frame_counts(spritesheet) -> dict:
    """Real (padding-trimmed) frame count per state, for the desktop canvas.

    Fail-open: a decode hiccup returns ``{}`` and the canvas falls back to its
    static ``framesPerState`` rather than breaking the (cosmetic) pet.
    """
    try:
        from agent.pet import render
        return render.state_frame_counts(str(spritesheet))
    except Exception:  # noqa: BLE001 - cosmetic, never break the surface
        return {}


_pet_payload_cache_lock = threading.Lock()
_pet_payload_cache: dict[tuple, dict] = {}


def _pet_sheet_revision(spritesheet) -> str:
    """Stable revision id for one spritesheet file."""
    try:
        stat = spritesheet.stat()
        return f"{stat.st_mtime_ns}:{stat.st_size}"
    except Exception:  # noqa: BLE001 - cosmetic, never break the surface
        return "0:0"


def _pet_payload_cache_key(pet, *, scale: float) -> tuple | None:
    """Cache key for the expensive sprite payload build."""
    try:
        stat = pet.spritesheet.stat()
    except Exception:  # noqa: BLE001
        return None
    return (
        str(pet.spritesheet), stat.st_mtime_ns, stat.st_size, pet.slug, pet.display_name,
        round(scale, 4),
    )


def _clone_pet_payload(payload: dict) -> dict:
    """Shallow-clone cached payloads so callers can't mutate shared state."""
    out = dict(payload)
    if isinstance(payload.get("framesByState"), dict):
        out["framesByState"] = dict(payload["framesByState"])
    if isinstance(payload.get("framesByRow"), dict):
        out["framesByRow"] = dict(payload["framesByRow"])
    if isinstance(payload.get("stateRows"), list):
        out["stateRows"] = list(payload["stateRows"])
    return out


def _pet_row_frame_counts(spritesheet) -> dict:
    """Real frame count per concrete spritesheet row name."""
    try:
        from PIL import Image
        from agent.pet import constants, render
        with Image.open(spritesheet) as opened:
            image = opened.convert("RGBA")
        cols = max(1, image.width // constants.FRAME_W)
        row_count = max(1, image.height // constants.FRAME_H)
        rows = constants.state_rows_for_grid(row_count)
        out: dict[str, int] = {}
        for row_idx, name in enumerate(rows[:row_count]):
            top = row_idx * constants.FRAME_H
            count = 0
            for col in range(cols):
                left = col * constants.FRAME_W
                frame = image.crop((left, top, left + constants.FRAME_W, top + constants.FRAME_H))
                if render._frame_is_blank(frame):
                    break
                count += 1
            out[name] = count
        return out
    except Exception:  # noqa: BLE001 - cosmetic, never break the surface
        return {}


def _pet_config_scale() -> float:
    """Configured ``display.pet.scale`` (or the engine default), never raises."""
    from agent.pet import constants
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        display = cfg.get("display", {}) if isinstance(cfg.get("display"), dict) else {}
        pet_cfg = display.get("pet", {}) if isinstance(display.get("pet"), dict) else {}
        return float(pet_cfg.get("scale", constants.DEFAULT_SCALE) or constants.DEFAULT_SCALE)
    except Exception:  # noqa: BLE001
        return constants.DEFAULT_SCALE


def _pet_sprite_payload(pet, *, scale: float) -> dict:
    """Build the renderer payload (spritesheet bytes + geometry) for *pet*.

    Shared by ``pet.info`` (the active mascot) and ``pet.hatch`` (the unadopted
    preview) so both feed the desktop canvas / TUI from one shape.
    """
    import base64
    from agent.pet import constants
    cache_key = _pet_payload_cache_key(pet, scale=scale)
    if cache_key is not None:
        with _pet_payload_cache_lock:
            cached = _pet_payload_cache.get(cache_key)
        if cached is not None:
            return _clone_pet_payload(cached)
    raw = pet.spritesheet.read_bytes()
    suffix = pet.spritesheet.suffix.lower()
    mime = "image/png" if suffix == ".png" else "image/webp"
    payload = {
        "slug": pet.slug, "displayName": pet.display_name, "mime": mime,
        "spritesheetBase64": base64.standard_b64encode(raw).decode("ascii"),
        "spritesheetRevision": _pet_sheet_revision(pet.spritesheet), "frameW": constants.FRAME_W,
        "frameH": constants.FRAME_H, "framesPerState": constants.FRAMES_PER_STATE,
        "framesByState": _pet_frame_counts(pet.spritesheet),
        "framesByRow": _pet_row_frame_counts(pet.spritesheet), "loopMs": constants.LOOP_MS,
        "scale": scale, "stateRows": _pet_state_rows(pet.spritesheet),
    }
    if cache_key is not None:
        with _pet_payload_cache_lock:
            _pet_payload_cache[cache_key] = payload
            while len(_pet_payload_cache) > 8:
                _pet_payload_cache.pop(next(iter(_pet_payload_cache)))
    return _clone_pet_payload(payload)


def _pet_active_selection():
    """Resolve configured active pet + scale from config."""
    from agent.pet import constants, store
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        display = cfg.get("display", {}) if isinstance(cfg.get("display"), dict) else {}
        pet_cfg = display.get("pet", {}) if isinstance(display.get("pet"), dict) else {}
    except Exception:
        pet_cfg = {}
    enabled = is_truthy_value(pet_cfg.get("enabled"), default=False)
    configured_slug = str(pet_cfg.get("slug", "") or "")
    pet = store.resolve_active_pet(configured_slug) if enabled else None
    scale = float(pet_cfg.get("scale", constants.DEFAULT_SCALE) or constants.DEFAULT_SCALE)
    return enabled, pet, scale


def _pet_state_rows(spritesheet) -> list[str]:
    """Row taxonomy for the concrete active pet sheet.

    Hermes has to support both the legacy 8-row petdex atlas and the current
    Codex/petdex 9-row atlas. The desktop canvas gets this list and indexes it
    with the same `PetState` names the Python renderer uses.
    """
    try:
        from PIL import Image
        from agent.pet import constants
        with Image.open(spritesheet) as image:
            row_count = max(1, image.height // constants.FRAME_H)
        return list(constants.state_rows_for_grid(row_count))
    except Exception:  # noqa: BLE001 - cosmetic, never break the surface
        from agent.pet import constants
        return list(constants.STATE_ROWS)


def _pet_gen_root():
    """Profile-scoped staging dir for in-progress generation drafts."""
    from hermes_constants import get_hermes_home
    root = get_hermes_home() / "cache" / "pet-gen"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _pet_gen_sweep(root, *, max_age_s: float = 3600.0) -> None:
    """Drop stale draft staging dirs so cache never grows unbounded."""
    import shutil
    import time
    try:
        now = time.time()
        for child in root.iterdir():
            if child.is_dir() and now - child.stat().st_mtime > max_age_s:
                shutil.rmtree(child, ignore_errors=True)
    except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
        logger.debug("pet-gen sweep failed: %s", exc)


def _pet_png_data_uri(path, *, max_px: int = 160) -> str:
    """Downscaled PNG data URI for a draft image (small preview payload)."""
    import base64
    import io
    from PIL import Image
    with Image.open(path) as opened:
        img = opened.convert("RGBA")
    img.thumbnail((max_px, max_px), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return "data:image/png;base64," + base64.standard_b64encode(buf.getvalue()).decode("ascii")


# Cooperative cancellation for pet generation: Stop aborts the RPC, but the
# pool job keeps running unless pet.cancel flips its token (polled between
# provider calls).
_pet_cancel_lock = threading.Lock()
_pet_cancelled: set[str] = set()
_PET_REFERENCE_MIME_EXT = {"png": "png", "jpeg": "jpg", "jpg": "jpg", "webp": "webp", "gif": "gif"}
try:
    _PET_REFERENCE_MAX_BYTES = max(
        1, int(os.environ.get("HERMES_PET_REFERENCE_MAX_BYTES") or str(16 * 1024 * 1024)),
    )
except (TypeError, ValueError):
    _PET_REFERENCE_MAX_BYTES = 16 * 1024 * 1024


def _pet_reference_images_from_data_url(ref_raw: str, stage) -> list:
    """Decode + validate a reference-image data URL into the stage dir."""
    import base64
    import binascii
    import re as _re
    match = _re.match(r"^data:image/([a-zA-Z0-9.+-]+);base64,(.*)$", ref_raw, _re.DOTALL)
    if not match:
        raise ValueError("invalid reference image format")
    mime = match.group(1).lower()
    ext = _PET_REFERENCE_MIME_EXT.get(mime)
    if ext is None:
        raise ValueError("unsupported reference image type")
    payload = "".join(match.group(2).split())
    approx = (len(payload) * 3) // 4
    if approx > _PET_REFERENCE_MAX_BYTES:
        raise ValueError("reference image too large")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("invalid reference image data") from exc
    if len(raw) > _PET_REFERENCE_MAX_BYTES:
        raise ValueError("reference image too large")
    ref_path = stage / f"reference.{ext}"
    ref_path.write_bytes(raw)
    return [ref_path]


def _pet_cancel_arm(token: str) -> None:
    """Clear a stale cancel flag at the start of a generate/hatch run."""
    with _pet_cancel_lock:
        _pet_cancelled.discard(token)


def _pet_cancel_request(token: str) -> None:
    with _pet_cancel_lock:
        _pet_cancelled.add(token)


def _pet_is_cancelled(token: str) -> bool:
    with _pet_cancel_lock:
        return token in _pet_cancelled


def _pet_cancel_release(token: str) -> None:
    with _pet_cancel_lock:
        _pet_cancelled.discard(token)


# ── Delegation: subagent tree observability + controls ───────────────
# Powers the TUI's /agents overlay (see ui-tui/src/components/agentsOverlay).
# The registry lives in tools/delegate_tool — these handlers are thin
# translators between JSON-RPC and the Python API.


# ── Spawn-tree snapshots: TUI-written, disk-persisted ────────────────
# The TUI owns subagent state; on turn-complete it posts the final tree here,
# /replay fetches by session_id + filename.
# Layout: $HERMES_HOME/spawn-trees/<session_id>/<timestamp>.json


def _spawn_trees_root():
    from hermes_constants import get_hermes_home
    root = get_hermes_home() / "spawn-trees"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _spawn_tree_session_dir(session_id: str):
    safe = ("".join(c if c.isalnum() or c in "-_" else "_" for c in session_id) or "unknown")
    d = _spawn_trees_root() / safe
    d.mkdir(parents=True, exist_ok=True)
    return d


# Per-session append-only index of lightweight snapshot metadata.  Read by
# `spawn_tree.list` so scanning doesn't require reading every full snapshot
# file (Copilot review on #14045).  One JSON object per line.
_SPAWN_TREE_INDEX = "_index.jsonl"


def _append_spawn_tree_index(session_dir, entry: dict) -> None:
    try:
        with (session_dir / _SPAWN_TREE_INDEX).open("a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except OSError as exc:
        # Index is a cache — losing a line just means list() falls back
        # to a directory scan for that entry.  Never block the save.
        logger.debug("spawn_tree index append failed: %s", exc)


def _read_spawn_tree_index(session_dir) -> list[dict]:
    index_path = session_dir / _SPAWN_TREE_INDEX
    if not index_path.exists():
        return []
    out: list[dict] = []
    try:
        with index_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return out


# ── Methods: prompt ──────────────────────────────────────────────────


_GOAL_COMPRESSION_RECOVERY_ATTEMPTS = "_goal_compression_recovery_attempts"
_GOAL_COMPRESSION_RECOVERY_LIMIT = 1




# Captured at import time: tests monkeypatch threading.Thread with a synchronous
# stub, and this ticker only exits once `stop` is set AFTER run_conversation
# returns — inline it would spin forever. Always a real daemon thread.
_RealThread = threading.Thread


def _start_usage_ticker(
    sid: str, agent, interval: float = 1.0
) -> tuple[threading.Event, threading.Thread]:
    """Push live ``session.usage`` snapshots every ``interval`` s while a turn runs.

    Otherwise the status-bar context figure is frozen until ``message.complete``.
    (The codex app-server runtime folds usage in only at turn end, so it gets no
    mid-turn ticks.)  The caller must set the Event AND join the thread before
    emitting ``message.complete``: a late tick would roll the client's final
    usage back to a stale snapshot.
    """
    stop = threading.Event()

    # Dedup baseline sampled BEFORE the thread starts (the client already has
    # the turn-start values); a late-scheduled thread would otherwise absorb the
    # first counter growth and never emit it.
    try:
        baseline: dict | None = _get_usage(agent)
    except Exception:
        baseline = None

    def _loop() -> None:
        last = baseline
        while not stop.wait(interval):
            with contextlib.suppress(Exception):
                usage = _get_usage(agent)
                if usage == last:
                    # Counters frozen (e.g. one long API call in flight) —
                    # skip the redundant frame so idle ticks don't re-render
                    # the client status bar every second.
                    continue
                last = usage
                if stop.is_set():
                    # Turn ended while snapshotting — drop the tick;
                    # message.complete carries the authoritative usage.
                    break
                _emit("session.usage", sid, {"usage": usage})
    thread = _RealThread(target=_loop, daemon=True)
    thread.start()
    return stop, thread




# Byte-upload attach caps. 25 MB matches Anthropic's per-image limit; 50 MB / 25
# pages bounds a single PDF drop so it can't blow the context budget.
# ── Methods: respond ─────────────────────────────────────────────────


def _respond(rid, params, key, *, allow_expired=False):
    r = params.get("request_id", "")
    question_id = str(params.get("question_id") or "")
    with _prompt_lock:
        entry = _pending.get(r)
        if not entry:
            if allow_expired and r:
                return _ok(rid, {"status": "expired"})
            return _err(rid, 4009, f"no pending {key} request")
        _, ev = entry
        batch = _batch_clarify.get(r)
        if batch is not None and question_id:
            # Per-question lock; update-in-place so a locked answer stays
            # editable until every qid is locked (the Confirm click).
            if question_id not in batch["qids"]:
                return _err(rid, 4002, f"unknown question_id {question_id!r}")
            batch["answers"][question_id] = params.get(key, "")
            remaining = [qid for qid in batch["qids"] if qid not in batch["answers"]]
            if not remaining:
                ev.set()
            return _ok(rid, {"status": "ok", "remaining": remaining})
        _answers[r] = params.get(key, "")
        ev.set()
    return _ok(rid, {"status": "ok"})


# ── Methods: config ──────────────────────────────────────────────────


# ---------------------------------------------------------------------------
# Projects — first-class, per-profile, multi-folder workspaces
# ---------------------------------------------------------------------------


# JSON-RPC error codes for the projects surface.
_E_PROJECTS = 5061  # generic failure
_E_NO_PROJECT = 5062  # id resolved to nothing
_E_PROJECT_ARG = 5063  # invalid argument (e.g. bad name/slug)


class _NoProject(Exception):
    """Raised inside a projects handler when ``params['id']`` resolves to None."""


def _projects_payload(conn) -> dict:
    from hermes_cli import projects_db as pdb
    return {
        "projects": [p.to_dict() for p in pdb.list_projects(conn, include_archived=True)],
        "active_id": pdb.get_active_id(conn),
    }


def _projects_method(name: str):
    """Register a projects RPC, injecting (pdb, conn) and unifying error mapping.

    Binds ``params['profile']`` (via ``@_profile_scoped``) so app-global remote
    mode reads that profile's ``projects.db``. Missing id maps to 5062, bad args
    to 5063, everything else to 5061.
    """

    def decorator(fn):
        @method(name)
        @_profile_scoped
        def handler(rid, params: dict) -> dict:
            try:
                from hermes_cli import projects_db as pdb
                with pdb.connect_closing() as conn:
                    return fn(rid, params, pdb, conn)
            except _NoProject:
                return _err(rid, _E_NO_PROJECT, "no such project")
            except ValueError as e:
                return _err(rid, _E_PROJECT_ARG, str(e))
            except Exception as e:
                return _err(rid, _E_PROJECTS, str(e))
        return handler
    return decorator


def _require_project(pdb, conn, params: dict):
    """The project named by ``params['id']`` (or raise ``_NoProject``)."""
    proj = pdb.get_project(conn, str(params.get("id") or ""))
    if proj is None:
        raise _NoProject
    return proj


@_projects_method("projects.list")
def _(rid, params, pdb, conn) -> dict:
    return _ok(rid, _projects_payload(conn))


@_projects_method("projects.get")
def _(rid, params, pdb, conn) -> dict:
    return _ok(rid, {"project": _require_project(pdb, conn, params).to_dict()})


@_projects_method("projects.create")
def _(rid, params, pdb, conn) -> dict:
    pid = pdb.create_project(
        conn, name=str(params.get("name") or ""), slug=params.get("slug"),
        folders=params.get("folders") or [], primary_path=params.get("primary_path"),
        description=params.get("description"), icon=params.get("icon"), color=params.get("color"),
        board_slug=params.get("board_slug"),
    )
    if params.get("use"):
        pdb.set_active(conn, pid)
    proj = pdb.get_project(conn, pid)
    return _ok(rid, {"project": proj.to_dict() if proj else None})


@_projects_method("projects.update")
def _(rid, params, pdb, conn) -> dict:
    proj = _require_project(pdb, conn, params)
    pdb.update_project(
        conn, proj.id, name=params.get("name"), description=params.get("description"),
        icon=params.get("icon"), color=params.get("color"), board_slug=params.get("board_slug"),
    )
    return _ok(rid, {"project": pdb.get_project(conn, proj.id).to_dict()})


@_projects_method("projects.add_folder")
def _(rid, params, pdb, conn) -> dict:
    proj = _require_project(pdb, conn, params)
    pdb.add_folder(
        conn, proj.id, str(params.get("path") or ""), label=params.get("label"),
        is_primary=bool(params.get("is_primary")),
    )
    return _ok(rid, {"project": pdb.get_project(conn, proj.id).to_dict()})


@_projects_method("projects.remove_folder")
def _(rid, params, pdb, conn) -> dict:
    proj = _require_project(pdb, conn, params)
    pdb.remove_folder(conn, proj.id, str(params.get("path") or ""))
    return _ok(rid, {"project": pdb.get_project(conn, proj.id).to_dict()})


@_projects_method("projects.set_primary")
def _(rid, params, pdb, conn) -> dict:
    proj = _require_project(pdb, conn, params)
    pdb.set_primary(conn, proj.id, str(params.get("path") or ""))
    return _ok(rid, {"project": pdb.get_project(conn, proj.id).to_dict()})


@_projects_method("projects.archive")
def _(rid, params, pdb, conn) -> dict:
    proj = _require_project(pdb, conn, params)
    (pdb.restore_project if params.get("restore") else pdb.archive_project)(conn, proj.id)
    return _ok(rid, _projects_payload(conn))


@_projects_method("projects.delete")
def _(rid, params, pdb, conn) -> dict:
    proj = _require_project(pdb, conn, params)
    pdb.delete_project(conn, proj.id)
    return _ok(rid, _projects_payload(conn))


@_projects_method("projects.set_active")
def _(rid, params, pdb, conn) -> dict:
    pdb.set_active(conn, _require_project(pdb, conn, params).id if params.get("id") else None)
    return _ok(rid, {"active_id": pdb.get_active_id(conn)})


@_projects_method("projects.for_cwd")
def _(rid, params, pdb, conn) -> dict:
    cwd = _completion_cwd({"cwd": str(params.get("cwd") or "").strip()} if params.get("cwd") else {})
    proj = pdb.project_for_path(conn, cwd)
    return _ok(rid, {"project": proj.to_dict() if proj else None, "cwd": cwd, "branch": _git_branch_for_cwd(cwd)})


def _non_workspace_dirs() -> set[str]:
    """Directories that are never a workspace, whichever tier proposes them.

    The filesystem root, the user's home, and the directory homes live in —
    ``/home`` on Linux, ``/Users`` on macOS, ``C:\\Users`` on Windows. Both
    POSIX spellings are excluded on every host because both are reachable as a
    cwd anywhere: macOS ships an empty ``/home`` autofs stub, and a container or
    remote shell hands back Linux paths. Promoting one of these mints a
    catch-all project that swallows unplaced sessions, and ``/home`` in
    particular renders as a second row reading "home" next to the Home bucket.
    """
    home = os.path.realpath(os.path.expanduser("~"))
    candidates = (os.sep, home, os.path.dirname(home), "/home", "/Users")
    return {os.path.normcase(os.path.realpath(path)) for path in candidates if path}


def _is_repo_junk(root: str) -> bool:
    """A git root we never auto-surface as a project: a non-workspace dir (see
    :func:`_non_workspace_dirs`) or anything under HERMES_HOME (~/.hermes by
    default) — config/sessions/skills, not a workspace. User-created projects
    pointing there are still honored."""
    if not root:
        return True
    from hermes_constants import get_hermes_home
    real = os.path.realpath(root)
    hermes_home = os.path.realpath(str(get_hermes_home()))
    return (
        os.path.normcase(real) in _non_workspace_dirs()
        or real == hermes_home
        or real.startswith(hermes_home + os.sep)
    )


def _is_session_cwd_junk(cwd: str) -> bool:
    """A non-git cwd that should stay in flat Recents rather than auto-group.

    Unlike discovered git roots, an explicitly selected descendant of
    HERMES_HOME may be an intentional prose/data workspace. The pre-Projects
    desktop surfaced every such cwd, so exclude only the broad defaults that
    would create catch-all projects: HERMES_HOME itself and the dirs in
    :func:`_non_workspace_dirs`.
    """
    if not cwd:
        return True
    from hermes_constants import get_hermes_home
    real = os.path.normcase(os.path.realpath(cwd))
    hermes_home = os.path.normcase(os.path.realpath(str(get_hermes_home())))
    return real in _non_workspace_dirs() or real == hermes_home


def _repo_discovery_policy(raw: dict | None = None) -> dict:
    """Return the effective, profile-local Desktop repository scan policy."""
    from hermes_cli.config import DEFAULT_CONFIG
    defaults = DEFAULT_CONFIG["desktop"]
    source = raw if isinstance(raw, dict) else (_load_cfg().get("desktop") or {})
    if not isinstance(source, dict):
        source = {}
    enabled = source.get("enabled", source.get("repo_scan_enabled", defaults["repo_scan_enabled"]))
    roots = source.get("roots", source.get("repo_scan_roots", defaults["repo_scan_roots"]))
    excludes = source.get(
        "exclude_paths", source.get("repo_scan_exclude_paths", defaults["repo_scan_exclude_paths"]),
    )
    return {
        "enabled": enabled if isinstance(enabled, bool) else defaults["repo_scan_enabled"],
        "roots": [value.strip() for value in roots if isinstance(value, str) and value.strip()]
        if isinstance(roots, list)
        else list(defaults["repo_scan_roots"]),
        "exclude_paths": [
            value.strip()
            for value in excludes
            if isinstance(value, str) and value.strip()
        ]
        if isinstance(excludes, list)
        else list(defaults["repo_scan_exclude_paths"]),
    }


def _repo_discovery_policy_key(policy: dict) -> str:
    def _paths(values: list[str]) -> list[str]:
        normalized = set()
        home = os.path.expanduser("~")
        for value in values:
            expanded = os.path.expanduser(value)
            if not os.path.isabs(expanded):
                expanded = os.path.join(home, expanded)
            normalized.add(os.path.normcase(os.path.abspath(expanded)))
        return sorted(normalized)
    canonical = {
        "enabled": bool(policy["enabled"]), "roots": _paths(policy["roots"]),
        "exclude_paths": _paths(policy["exclude_paths"]),
    }
    return json.dumps(canonical, sort_keys=True, separators=(",", ":"))


def _repo_discovery_policy_is_default(policy: dict) -> bool:
    from hermes_cli.config import DEFAULT_CONFIG
    return _repo_discovery_policy_key(policy) == _repo_discovery_policy_key(
        _repo_discovery_policy(DEFAULT_CONFIG["desktop"])
    )


def _scan_discovered_repos_remote(conn, policy: dict) -> bool:
    """Backend-side disk scan of the discovery policy roots.

    The desktop's native repo scan only runs on the local filesystem. On a
    remote gateway connection the host must scan its own disk so repos with
    zero Hermes sessions still appear in the sidebar (#81723). Mirrors the
    desktop's behavior: walk each root (bounded depth), find `.git`
    directories, record (root, label) pairs into the discovery cache.

    Best-effort: any failure logs and leaves the cache untouched — the
    session-derived repos from `_discover_repos_payload` still surface.

    Returns True when the scan is authoritative (every root was walked to
    completion without error and the per-scan cap was not hit). Only then may
    the caller treat the result as a full replacement and pass ``replace=True``
    to the cache write — a partial or errored scan must merge, never wipe, so
    a failed remote refresh can't blank the previously cached repos into the
    silent, unpopulated sidebar of #81723.
    """
    from hermes_cli import projects_db as pdb
    roots = policy.get("roots") or []
    excludes = policy.get("exclude_paths") or []
    pairs: list[tuple[str, str | None]] = []
    seen: set[str] = set()
    authoritative = True

    def _is_excluded(path: str) -> bool:
        return any(path == ex or path.startswith(ex.rstrip("/\\") + os.sep) for ex in excludes if ex)
    for root in roots:
        if not os.path.isdir(root):
            # `os.walk` on a missing root yields nothing instead of raising; an
            # unmounted volume would look like an empty scan and let the
            # authoritative replace wipe every cached repo under it.
            authoritative = False
            logger.debug("discover_repos scan root missing, skipping: %s", root)
            continue
        try:
            for dirpath, dirnames, _filenames in os.walk(root):
                if _is_excluded(dirpath):
                    dirnames[:] = []
                    continue
                # A `.git` directory marks this directory as a repo root. Check
                # BEFORE pruning hidden dirs — `.git` is itself hidden, so a
                # prune-first order would drop it and never detect any repo.
                if ".git" in dirnames:
                    repo_root = dirpath
                    if repo_root not in seen:
                        seen.add(repo_root)
                        pairs.append((repo_root, os.path.basename(repo_root)))
                    # Don't descend into the repo's own .git to hunt nested repos.
                    dirnames[:] = []
                else:
                    # Not a repo: skip hidden dirs (e.g. .hermes) and node_modules.
                    dirnames[:] = [d for d in dirnames if not d.startswith(".") and d not in ("node_modules",)]
                if len(pairs) >= 500:
                    break
        except Exception:
            # A root that can't be walked yields no authoritative set — fall back
            # to merging, never replacing, so the prior cache survives.
            authoritative = False
            logger.debug("discover_repos scan failed for root %s", root, exc_info=True)
        if len(pairs) >= 500:
            # Cap hit means the walk didn't cover the full roots; the collected
            # set must not be treated as the complete authoritative universe.
            authoritative = False
            break
    if pairs:
        try:
            pdb.record_discovered_repos(
                conn, pairs, replace=authoritative, policy_key=_repo_discovery_policy_key(policy)
            )
        except Exception:
            logger.debug("discover_repos cache write failed", exc_info=True)
            authoritative = False
    return authoritative


def _discover_repos_payload(
    db, *, conn=None, backfill: bool = True, include_cached: bool = True
) -> list[dict]:
    """Merge filesystem-scanned repos (cached) with session-derived repo roots.

    Repo-first: the disk scan (persisted by `projects.record_repos`) surfaces
    repos even with zero hermes sessions. Session-derived roots cover repos
    outside the scan roots. Both are junk-filtered (hermes home subtree + bare
    home) and carry their session totals for the overview.

    ``conn`` reuses an already-open projects.db connection (the tree path holds
    one); ``backfill`` persists resolved roots back onto session rows — kept off
    the per-turn tree path (grouping uses the live git resolver regardless) and
    done only on the explicit discover/record refresh.
    """
    _is_junk = _is_repo_junk
    repos: dict[str, dict] = {}

    def _agg(root: str) -> dict:
        return repos.setdefault(root, {"root": root, "label": "", "sessions": 0, "last_active": 0.0})

    # Session-derived roots (common repo root, folding worktrees; cached) +
    # backfill the column so persisted git_repo_root matches the tree grouping.
    cwd_rows = list(db.distinct_session_cwds())
    # Warm the per-cwd git probes in parallel so a cold first paint doesn't
    # serialize one subprocess per distinct cwd before this loop reads the cache.
    git_probe.warm_roots(str(r.get("cwd") or "") for r in cwd_rows)
    cwd_to_root: dict[str, str] = {}
    for row in cwd_rows:
        cwd = str(row.get("cwd") or "")
        root = _git_common_repo_root_for_cwd(cwd)
        if not root:
            continue
        cwd_to_root[cwd] = root
        if _is_junk(root):
            continue
        agg = _agg(root)
        agg["sessions"] += int(row.get("sessions") or 0)
        agg["last_active"] = max(agg["last_active"], float(row.get("last_active") or 0))
    if backfill:
        try:
            db.backfill_repo_roots(cwd_to_root)
        except Exception:
            logger.debug("failed to backfill repo roots", exc_info=True)
    if include_cached:
        # Filesystem-scanned roots from the cache (may have zero sessions). Reuse
        # the caller's projects.db connection when given, else a short-lived one.
        try:
            from hermes_cli import projects_db as pdb

            def _read(c) -> None:
                for entry in pdb.list_discovered_repos(c):
                    root = str(entry.get("root") or "")
                    if not root or _is_junk(root):
                        continue
                    agg = _agg(root)
                    if entry.get("label"):
                        agg["label"] = entry["label"]
                    # `last_seen` is scan time, not user activity; folding it
                    # into `last_active` made every scanned repo "just now".
            if conn is not None:
                _read(conn)
            else:
                with pdb.connect_closing() as own:
                    _read(own)
        except Exception:
            logger.debug("failed to read discovered repo cache", exc_info=True)
    out = sorted(repos.values(), key=lambda r: r["last_active"], reverse=True)
    for r in out:
        r["label"] = r["label"] or os.path.basename(r["root"].rstrip("/\\")) or r["root"]
    return out


# Not user conversations (cron has its own section; kanban runs are read on
# the board). Subagent/compression children are dropped by include_children=False.
_PROJECT_TREE_EXCLUDED_SOURCES = ["cron", "kanban"]


def _project_tree_row(r: dict) -> dict:
    """Project a SessionDB row to the minimal shape the sidebar renders.

    Keeps the fields the grouping needs (cwd / git_branch / git_repo_root) plus
    everything ``SidebarSessionRow`` reads, and drops the heavy columns
    (system_prompt, model_config, ...) so the tree payload stays lean.
    """
    return {
        "id": r.get("id"),
        "_lineage_root_id": r.get("_lineage_root_id"),
        "_lineage_ids": r.get("_lineage_ids"),
        # The sidebar nests branch/fork sessions under their parent
        # (flattenSessionsWithBranches keys on this); without it, lane rows can't
        # draw the └─ connector the flat Recents list shows.
        "parent_session_id": r.get("parent_session_id"),
        "title": r.get("title"),
        "preview": r.get("preview"),
        "started_at": r.get("started_at") or 0,
        "ended_at": r.get("ended_at"),
        "last_active": r.get("last_active") or r.get("started_at") or 0,
        "source": r.get("source"),
        "archived": bool(r.get("archived")),
        "message_count": r.get("message_count") or 0,
        "tool_call_count": r.get("tool_call_count") or 0,
        "input_tokens": r.get("input_tokens") or 0,
        "output_tokens": r.get("output_tokens") or 0,
        # Cost is one of the fields SidebarSessionRow renders, so a lane row has
        # to carry it too — without it, switching Show → cost filled in every
        # figure in Recents and left the same sessions blank under a project.
        "actual_cost_usd": r.get("actual_cost_usd"),
        "estimated_cost_usd": r.get("estimated_cost_usd"),
        "model": r.get("model"),
        "is_active": False,
        "cwd": r.get("cwd"),
        "git_branch": r.get("git_branch"),
        "git_repo_root": r.get("git_repo_root"),
    }


def _project_tree_inputs(
    db, session_limit: int, *, include_discovered: bool
) -> tuple[list[dict], list[dict], list[dict], str | None]:
    """Gather (sessions, projects, discovered_repos, active_id) for build_tree.

    ``include_discovered`` is the zero-session-repo overview tier; the entered
    view (drill-in) skips it entirely — it only needs the project it's showing,
    which already has sessions — avoiding the distinct-cwd scan + git probes on
    that per-turn path. One projects.db connection serves both reads.
    """
    rows = db.list_sessions_rich(
        limit=session_limit,
        offset=0,
        order_by_last_active=True,
        min_message_count=1,
        include_children=False,
        exclude_sources=_PROJECT_TREE_EXCLUDED_SOURCES,
        include_archived=False,
        # `_project_tree_row` keeps ~18 fields and drops the rest, so selecting
        # the system-prompt blob only to discard it costs tens of MB of B-tree
        # reads per build on a long-lived database.
        compact_rows=True,
    )
    sessions = [_project_tree_row(r) for r in rows]
    # Parallel-warm the git cache so build_tree's resolver reads it instead of
    # cold-probing each cwd in sequence (matters on the drill-in path, which
    # skips the discovery warm-up below).
    git_probe.warm_roots(s["cwd"] for s in sessions if s.get("cwd"))
    from hermes_cli import projects_db as pdb
    policy = _repo_discovery_policy()
    policy_key = _repo_discovery_policy_key(policy)
    with pdb.connect_closing() as conn:
        if include_discovered:
            pdb.reconcile_discovered_repos_policy(
                conn, policy_key, preserve_unversioned=_repo_discovery_policy_is_default(policy),
            )
        projects = [p.to_dict() for p in pdb.list_projects(conn)]
        active_id = pdb.get_active_id(conn)
        # backfill stays off the hot tree path — grouping uses the live resolver.
        discovered = (
            _discover_repos_payload(db, conn=conn, backfill=False, include_cached=policy["enabled"])
            if include_discovered
            else []
        )
    return sessions, projects, discovered, active_id


# Per-build memo for `_dir_exists_cached`. Cleared at the top of every
# `_build_project_tree`, so a dir created or deleted between sidebar refreshes
# is seen on the next one.
_DIR_EXISTS_CACHE: dict[str, bool] = {}


def _dir_exists_cached(path: str) -> bool:
    """``os.path.isdir`` for the project tree, memoized per build.

    ``build_tree`` asks per SESSION, not per distinct path, so a power user with
    hundreds of sessions across a handful of dirs would otherwise fire hundreds
    of redundant stats on every sidebar open. The memo is per build, so a dir
    created or deleted between refreshes is picked up on the next one.
    """
    hit = _DIR_EXISTS_CACHE.get(path)
    if hit is None:
        hit = os.path.isdir(path)
        _DIR_EXISTS_CACHE[path] = hit
    return hit


def _build_project_tree(
    db, *, preview_limit: int, hydrate: bool, session_limit: int, include_discovered: bool
) -> tuple[dict, str | None]:
    """Gather inputs and run the one authoritative builder. Returns (tree, active_id)."""
    from tui_gateway import project_tree
    _DIR_EXISTS_CACHE.clear()
    sessions, projects, discovered, active_id = _project_tree_inputs(
        db, session_limit, include_discovered=include_discovered
    )
    # build_tree resolves every declared project folder and every discovered
    # repo root too, and those paths are not session cwds — without this they
    # are the one part of the build still probing git one directory at a time.
    git_probe.warm_roots(
        [str(f.get("path") or "") for p in projects for f in (p.get("folders") or [])]
        + [str(r.get("root") or "") for r in discovered]
    )
    tree = project_tree.build_tree(
        projects, sessions, discovered, _resolve_cwd_git, preview_limit=preview_limit,
        hydrate=hydrate, is_junk_root=_is_repo_junk, is_junk_cwd=_is_session_cwd_junk,
        exists=_dir_exists_cached,
    )
    return tree, active_id


# ── Methods: tools & system ──────────────────────────────────────────


def _session_processes(session: dict) -> list:
    """Background processes owned by this session (registry session_key match)."""
    from tools.process_registry import process_registry
    key = str(session.get("session_key") or "")
    owned = []
    for entry in process_registry.list_sessions():
        proc = process_registry.get(entry["session_id"])
        if proc is None or str(getattr(proc, "session_key", "") or "") != key:
            continue
        # The 200-char list preview is too thin for the desktop's inline
        # terminal viewer — ship a real tail alongside it.
        entry["output_tail"] = (proc.output_buffer or "")[-4000:]
        owned.append(entry)
    return owned


# Serialize reload.mcp (it runs on the pool): overlapping shutdown+discover
# pairs would leave the registry half-built.
_mcp_reload_lock = threading.Lock()
# Bumped per SUCCESSFUL reload; a follower skips only if it advanced while it
# waited (a leader that threw leaves it unchanged → follower reloads itself).
_mcp_reload_gen = 0
# The mcp_rev the last successful reload actually LOADED (re-hashed after
# discovery). A follower coalesces only when its requested rev matches;
# otherwise the config changed under the leader and it must reload itself.
_mcp_reload_loaded_rev = ""
# Bounded convergence for a config edit racing a slow reload: the leader
# re-hashes after discovery and repeats until the hash is stable.
_MCP_RELOAD_MAX_PASSES = 3


def _compute_mcp_rev() -> str:
    """Hash of the MCP-relevant config sections (server definitions,
    settings, toolset enables). ``config.get mtime`` ships it to the TUI so
    cosmetic writes don't trigger reloads; ``reload.mcp`` uses it for
    revision-aware coalescing. Empty string = unknown (fail open)."""
    try:
        cfg = _load_cfg()
        # mcp_servers (definitions, what the CLI auto-reload watches) + mcp
        # (settings) + tools (enable/disable); omitting mcp_servers meant an
        # edited server never bumped mcp_rev and never connected.
        rev_src = json.dumps(
            {"mcp": cfg.get("mcp"), "mcp_servers": cfg.get("mcp_servers"), "tools": cfg.get("tools")},
            sort_keys=True,
            default=str,
        )
        return hashlib.sha1(rev_src.encode()).hexdigest()[:12]
    except Exception:
        return ""


def _finish_reload(rid, params: dict, *, coalesced: bool) -> dict:
    """Shared tail for both reload paths: honor ``always`` (persist the
    confirm opt-out) and return the ok payload."""
    if bool(params.get("always", False)):
        try:
            from cli import save_config_value as _save_cfg
            _save_cfg("approvals.mcp_reload_confirm", False)
        except Exception as _exc:
            logger.warning("Failed to persist mcp_reload_confirm=false: %s", _exc)
    payload = {"status": "reloaded", "loaded_rev": _mcp_reload_loaded_rev}
    if coalesced:
        payload["coalesced"] = True
    return _ok(rid, payload)


_TUI_HIDDEN: frozenset[str] = frozenset({"sethome", "set-home", "commands", "approve", "deny"})

_TUI_EXTRA: list[tuple[str, str, str]] = [
    ("/density", "Toggle compact display mode", "TUI"),
    ("/logs", "Show recent gateway log lines", "TUI"),
    ("/mouse", "Set mouse tracking preset [on|off|toggle|wheel|buttons|all]", "TUI"),
    ("/sessions", "Switch between live TUI sessions", "TUI"),
]

# Commands that queue onto _pending_input in the CLI; the slash worker has no
# reader for that queue, so slash.exec routes them to command.dispatch instead.
_PENDING_INPUT_COMMANDS: frozenset[str] = frozenset(
    {
        "retry", "queue", "q", "steer", "plan", "goal", "loop", "proactive", "moa", "undo", "learn",
        "init", "compress", "compact",
    }
)

_WORKER_BLOCKED_COMMANDS: frozenset[str] = frozenset({"snapshot", "snap"})


def _skill_usage_lookup():
    """Build ``(usage, origin)`` callables for the skill-command catalog.

    ``usage(name)`` is the skill's observed activity count (use + view +
    patch); ``origin(name)`` is ``"hub"``, ``"bundled"``, or ``"local"`` — the
    same classification ``/api/skills`` reports as ``provenance`` (where
    "local" is spelled "agent"). Both read sidecar files that are cheap and
    already parsed once per catalog build. Any failure degrades to zero usage
    and ``"local"`` so a missing/corrupt sidecar can never break the catalog.
    """
    try:
        from tools.skill_usage import (
            _read_bundled_manifest_names, _read_hub_installed_names, activity_count, load_usage,
        )
        records = load_usage()
        bundled = _read_bundled_manifest_names()
        hub = _read_hub_installed_names()
    except Exception as e:
        logger.debug("skill usage lookup unavailable: %s", e)
        return (lambda _name: 0), (lambda _name: "local")

    def usage(name: str) -> int:
        try:
            return activity_count(records.get(name) or {})
        except Exception:
            return 0

    def origin(name: str) -> str:
        if name in hub:
            return "hub"
        if name in bundled:
            return "bundled"
        return "local"
    return usage, origin


_SLASH_COMPLETION_LIMIT = 30


def _rank_slash_completions(
    items: list[dict], usage, origin_of, *, browsing: bool, score_of=None,
) -> list[dict]:
    """Rank and bound slash completions the way the menu should read.

    ``usage``/``origin_of`` come from :func:`_skill_usage_lookup`. Registry
    commands keep their order; only the skill block is reordered: fuzzy
    ``score_of`` first (a name match beats a description match), then
    most-used, then A-Z.

    The limit is spent PER KIND, not as one flat truncation: commands are
    emitted before the first skill, so a flat cut on a large install offered
    no skill at all and dropped heavily-used skills for never-opened ones.

    ``browsing`` (bare ``/``) drops bundled skills with no recorded activity
    as noise; a typed query is SEARCHING, and a search that hides a match is
    broken — nothing is pruned there, only reordered.
    """

    def name_of(item: dict) -> str:
        return str(item.get("text", "")).strip().lstrip("/").lower()
    commands = [item for item in items if item.get("kind") != "skill"]
    skills = [item for item in items if item.get("kind") == "skill"]
    if browsing:
        skills = [
            item
            for item in skills
            if origin_of(name_of(item)) != "bundled" or usage(name_of(item)) > 0
        ]
    if score_of is not None:
        skills.sort(key=lambda item: (score_of(item), -usage(name_of(item)), name_of(item)))
    else:
        skills.sort(key=lambda item: (-usage(name_of(item)), name_of(item)))
    return commands[:_SLASH_COMPLETION_LIMIT] + skills[:_SLASH_COMPLETION_LIMIT]


def _cli_exec_blocked(argv: list[str]) -> str | None:
    """Return user hint if this argv must not run headless in the gateway process."""
    if not argv:
        return "bare `hermes` is interactive — use `/hermes chat -q …` or run `hermes` in another terminal"
    a0 = argv[0].lower()
    if a0 == "setup":
        return "`hermes setup` needs a full terminal — run it outside the TUI"
    if a0 == "gateway":
        return "`hermes gateway` is long-running — run it in another terminal"
    if a0 == "sessions" and len(argv) > 1 and argv[1].lower() == "browse":
        return "`hermes sessions browse` is interactive — use /resume here, or run browse in another terminal"
    if a0 == "config" and len(argv) > 1 and argv[1].lower() == "edit":
        return "`hermes config edit` needs $EDITOR in a real terminal"
    return None


def _resolve_name(name: str) -> str:
    try:
        from hermes_cli.commands import resolve_command
        r = resolve_command(name)
        return r.name if r else name
    except Exception:
        return name


# ── Methods: paste ────────────────────────────────────────────────────

_paste_counter = 0


# ── Methods: insights ────────────────────────────────────────────────


# ── Methods: rollback ────────────────────────────────────────────────



# mcp.servers.* handlers (methods_tools) resolve these through this namespace.
from .mcp_rpc_helpers import (  # noqa: E402
    reset_profile as _mcp_reset_profile,
    summarize_server as _mcp_summarize_server,
)


# ── Split @method handler modules (see method_ctx.py) ────────────────
# Imported at the end of this module so every global the handlers close
# over already exists; register() rebinds them onto this namespace.
from . import (  # noqa: E402
    methods_voice as _methods_voice,
    methods_browser as _methods_browser,
    methods_slash as _methods_slash,
    methods_complete_helpers as _methods_complete_helpers,
    session_auto_continue as _session_auto_continue,
    agent_callbacks as _agent_callbacks,
    session_history as _session_history,
    prompt_attachments as _prompt_attachments,
    session_notifications as _session_notifications,
    tool_progress as _tool_progress,
    change_watcher as _change_watcher,
    session_compression as _session_compression,
    model_switch as _model_switch,
    compute_host_bridge as _compute_host_bridge,
    session_workdir as _session_workdir,
    session_lifecycle as _session_lifecycle,
    session_reaper as _session_reaper,
    methods_browser_control as _methods_browser_control,
    methods_bot_relay as _methods_bot_relay,
    methods_complete as _methods_complete,
    methods_config as _methods_config,
    methods_config_set as _methods_config_set,
    methods_images as _methods_images,
    methods_profiles as _methods_profiles,
    methods_prompt as _methods_prompt,
    methods_session as _methods_session,
    methods_tools as _methods_tools,
    prompt_turn as _prompt_turn,
    billing_view as _billing_view,
)

for _m in (
    _session_reaper, _session_lifecycle, _session_workdir, _compute_host_bridge, _model_switch,
    _session_compression, _change_watcher, _tool_progress, _session_notifications,
    _prompt_attachments, _session_history, _agent_callbacks, _session_auto_continue,
    _methods_complete_helpers, _methods_slash, _methods_voice, _methods_browser,
    _methods_browser_control, _methods_session, _methods_prompt, _methods_config,
    _methods_config_set, _methods_complete, _methods_tools, _methods_profiles, _methods_images,
    _methods_bot_relay, _prompt_turn, _billing_view,
):
    _m.register(sys.modules[__name__])
del _m
