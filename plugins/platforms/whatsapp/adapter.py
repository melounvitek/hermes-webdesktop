"""
WhatsApp platform adapter (Baileys bridge).

A Node.js bridge process runs the WhatsApp Web client; messages are polled over a
local HTTP API and responses are posted back through the bridge.
"""

import asyncio
import logging
import os
import platform
import re
import signal
import subprocess
from pathlib import Path
from typing import Dict, Optional, Any

from gateway.platforms._shared import get_scoped_secret
from hermes_cli._subprocess_compat import windows_detach_popen_kwargs
from hermes_constants import (find_node_executable, get_hermes_dir, with_hermes_node_path)

_IS_WINDOWS = platform.system() == "Windows"


def _wenv(name: str, default: str = "") -> str:
    """WHATSAPP_* env read through the profile secret scope (multiplexed profiles see
    their own .env; os.getenv would return the process-global value)."""
    return get_scoped_secret(name, default)

logger = logging.getLogger(__name__)

# Inbound owner-typed WhatsApp text is prefixed at MessageEvent construction so
# transcripts stay disambiguated even if downstream plugins fail before silent_ingest.
_OWNER_REPLY_PREFIX = "[owner reply] "

_RUN_TEXT = dict(capture_output=True, text=True, encoding='utf-8', errors='replace', stdin=subprocess.DEVNULL)


def _listener_pids_on_port(port: int) -> list:
    """PIDs of processes *listening* on ``port`` (POSIX) — never clients.

    A bare ``lsof -i :PORT`` also returns clients whose connection involves that
    port (e.g. a browser tab on a local dev server); SIGTERMing those closed the
    user's browser. Restricting to LISTEN state never touches an unrelated client.
    """
    pids: list = []
    try:
        result = subprocess.run(["lsof", "-ti", f"tcp:{port}", "-sTCP:LISTEN"], timeout=5, **_RUN_TEXT)
        for line in result.stdout.strip().splitlines():
            try:
                pids.append(int(line))
            except ValueError:
                pass
        if pids:
            return pids
    except FileNotFoundError:
        pass  # lsof not installed — fall through to ss
    try:
        result = subprocess.run(["ss", "-ltnHp", f"sport = :{port}"], timeout=5, **_RUN_TEXT)
        for m in re.finditer(r"pid=(\d+)", result.stdout):
            pids.append(int(m.group(1)))
    except FileNotFoundError:
        pass
    return pids


def _pid_looks_like_node_bridge(pid: int) -> bool:
    """Fail-closed check that *pid* is plausibly a stale node bridge.

    A scan-time PID can name a stranger (even a critical system process) by the
    time we kill it; require the live process to be a ``node`` executable. Any
    ambiguity (process gone, unreadable cmdline) refuses the kill.
    """
    try:
        import psutil
        proc = psutil.Process(pid)
        name = (proc.name() or "").lower()
        cmdline = " ".join(proc.cmdline() or []).lower()
        return "node" in name or "node" in cmdline.split(" ", 1)[0]
    except Exception:
        return False


def _warn_not_bridge(pid: int, port: int) -> None:
    logger.warning(
        "[whatsapp] Not killing PID %s on port %d: process is not a node bridge "
        "(or identity unverifiable)", pid, port)


def _kill_port_process(port: int) -> None:
    """Kill any node bridge *listening* on the given TCP port (never a client)."""
    try:
        if _IS_WINDOWS:
            from hermes_cli._subprocess_compat import windows_hide_flags
            result = subprocess.run(
                ["netstat", "-ano", "-p", "TCP"], timeout=5, creationflags=windows_hide_flags(), **_RUN_TEXT,
            )
            for line in result.stdout.splitlines():
                parts = line.split()
                if len(parts) >= 5 and parts[3] == "LISTENING" and parts[1].endswith(f":{port}"):
                    try:
                        pid = int(parts[4])
                    except ValueError:
                        continue
                    # taskkill /F on a mistyped or recycled PID is unrecoverable — verify first.
                    if pid <= 0 or not _pid_looks_like_node_bridge(pid):
                        _warn_not_bridge(pid, port)
                        continue
                    try:
                        subprocess.run(
                            ["taskkill", "/PID", str(pid), "/F"],
                            capture_output=True, timeout=5, creationflags=windows_hide_flags(),
                        )
                    except subprocess.SubprocessError:
                        pass
        else:
            for pid in _listener_pids_on_port(port):
                if not _pid_looks_like_node_bridge(pid):
                    _warn_not_bridge(pid, port)
                    continue
                try:
                    os.kill(pid, signal.SIGTERM)
                except (ProcessLookupError, PermissionError, OSError):
                    pass
    except Exception:
        pass


def _bridge_pid_is_ours(pid: int, session_path: Path, expected_start) -> bool:
    """True only if ``pid`` is alive AND still our node bridge for this session.

    The kernel can recycle a pidfile's PID onto an unrelated process (seen landing
    on a desktop browser, which a bare-liveness kill then closed). Identity is the
    recorded kernel start time (definitive); legacy pidfiles without one fall back
    to a cmdline containing ``node`` and this session's unique path.
    """
    from gateway.status import _pid_exists
    if not _pid_exists(pid):
        return False
    if expected_start is not None:
        from gateway.status import get_process_start_time
        return get_process_start_time(pid) == expected_start
    from gateway.status import _read_process_cmdline
    cmdline = _read_process_cmdline(pid)
    if not cmdline:
        return False
    return ("node" in cmdline) and (str(session_path) in cmdline)


def _kill_stale_bridge_by_pidfile(session_path: Path) -> None:
    """Kill an orphaned bridge recorded in ``bridge.pid`` from a previous run.

    The PID is re-validated via :func:`_bridge_pid_is_ours` before any signal.
    """
    pid_file = session_path / "bridge.pid"
    if not pid_file.exists():
        return
    pid = None
    recorded_start = None
    try:
        # Line 1 = pid, optional line 2 = kernel start time (legacy files: pid only).
        lines = pid_file.read_text(encoding="utf-8").split("\n")
        pid = int(lines[0].strip())
        if len(lines) > 1 and lines[1].strip():
            recorded_start = int(lines[1].strip())
    except (ValueError, OSError, TypeError, IndexError):
        try:
            pid_file.unlink()
        except OSError:
            pass
        return
    if _bridge_pid_is_ours(pid, session_path, recorded_start):
        try:
            os.kill(pid, signal.SIGTERM)
            logger.info("[whatsapp] Killed stale bridge PID %d from pidfile", pid)
        except (ProcessLookupError, PermissionError, OSError):
            pass
    else:
        from gateway.status import _pid_exists
        if _pid_exists(pid):
            logger.warning(
                "[whatsapp] Not killing pidfile PID %d: it is no longer the "
                "bridge (recycled onto an unrelated process); skipping to avoid "
                "killing a stranger.", pid,
            )
    try:
        pid_file.unlink()
    except OSError:
        pass


def _write_bridge_pidfile(session_path: Path, pid: int) -> None:
    """Write the bridge PID plus its kernel start time (line 2) for later identity-checked cleanup."""
    try:
        from gateway.status import get_process_start_time
        start = get_process_start_time(pid)
        text = str(pid) if start is None else "{}\n{}".format(pid, start)
        (session_path / "bridge.pid").write_text(text, encoding="utf-8")
    except OSError:
        pass


def _terminate_bridge_process(proc, *, force: bool = False) -> None:
    """Terminate the bridge process using process-tree semantics where possible."""
    if _IS_WINDOWS:
        cmd = ["taskkill", "/PID", str(proc.pid), "/T"]
        if force:
            cmd.append("/F")
        try:
            result = subprocess.run(cmd, timeout=10, **_RUN_TEXT)
        except FileNotFoundError:
            if force:
                proc.kill()
            else:
                proc.terminate()
            return
        if result.returncode != 0:
            details = (result.stderr or result.stdout or "").strip()
            raise OSError(details or f"taskkill failed for PID {proc.pid}")
        return
    import psutil
    action = "kill" if force else "terminate"
    try:
        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            try:
                getattr(child, action)()
            except psutil.NoSuchProcess:
                pass
        getattr(parent, action)()
    except psutil.NoSuchProcess:
        return

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from gateway.config import Platform, PlatformConfig
from gateway.platforms.whatsapp_common import WhatsAppBehaviorMixin
from gateway.whatsapp_identity import to_whatsapp_jid
from gateway.platforms.base import (
    BasePlatformAdapter,
    MessageEvent,
    MessageType,
    SendResult,
    SUPPORTED_DOCUMENT_TYPES,
    cache_image_from_url,
    cache_audio_from_url,
)
from utils import env_int


def _is_allowed_bridge_path(url: str) -> bool:
    """True only when an absolute bridge path resolves inside a Hermes media cache dir.

    A compromised or buggy bridge could hand back ``/etc/passwd``; resolve
    symlinks and require one of the real cache roots (canonical ``cache/<kind>``
    or legacy ``<kind>_cache`` layout).
    """
    try:
        resolved = Path(url).resolve()
    except (OSError, ValueError):
        return False
    # Per-call getters (not import-time constants) so a profile override's cache matches.
    from gateway.platforms.base import (
        get_audio_cache_dir,
        get_document_cache_dir,
        get_image_cache_dir,
        get_video_cache_dir,
    )
    for root in (get_image_cache_dir(), get_audio_cache_dir(), get_video_cache_dir(), get_document_cache_dir()):
        try:
            if resolved.is_relative_to(Path(root).resolve()):
                return True
        except (OSError, ValueError):
            continue
    return False


def _file_content_hash(path: Path) -> str:
    """First 16 hex chars of SHA-256 of *path* ("" if unreadable).

    bridge.js reports its own hash in ``/health`` (``scriptHash``); a mismatch with
    the on-disk file means a long-lived bridge is serving pre-update code.
    """
    import hashlib
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:16]
    except OSError:
        return ""


def check_whatsapp_requirements() -> bool:
    """Node.js (Hermes-managed first, so a bad system Node on PATH can't break Windows) is available."""
    _node = find_node_executable("node")
    if not _node:
        return False
    try:
        result = subprocess.run([_node, "--version"], timeout=5, **_RUN_TEXT)
        return result.returncode == 0
    except Exception:
        return False


# Env vars bridge.js consumes. Under multiplexing the subprocess gets a copy of
# os.environ without the secondary profile's .env, so the resolved values are injected.
_BRIDGE_PASSTHROUGH_ENV = (
    "WHATSAPP_ALLOWED_USERS", "WHATSAPP_ALLOW_FROM",
    "WHATSAPP_DM_POLICY", "WHATSAPP_GROUP_POLICY",
    "WHATSAPP_GROUP_ALLOWED_USERS", "WHATSAPP_GROUP_ALLOW_FROM",
    "WHATSAPP_REQUIRE_MENTION", "WHATSAPP_MENTION_PATTERNS",
    "WHATSAPP_FREE_RESPONSE_CHATS",
    "WHATSAPP_DEBUG", "WHATSAPP_FORWARD_OWNER_MESSAGES",
    "WHATSAPP_REPLY_PREFIX", "WHATSAPP_MAX_MESSAGE_LENGTH",
    "WHATSAPP_CHUNK_DELAY_MS", "WHATSAPP_SEND_TIMEOUT_MS",
)
_TEXT_INJECT_EXTS = {".txt", ".md", ".csv", ".json", ".xml", ".yaml", ".yml", ".log", ".py", ".js", ".ts", ".html", ".css"}
_MAX_TEXT_INJECT_BYTES = 100 * 1024  # matches Telegram/Discord/Slack
_MEDIA_LABEL = {
    MessageType.PHOTO: "image", MessageType.VOICE: "audio", MessageType.AUDIO: "audio",
    MessageType.DOCUMENT: "document", MessageType.VIDEO: "video",
}
_MEDIA_DEFAULT_MIME = {
    MessageType.PHOTO: "image/jpeg", MessageType.VOICE: "audio/ogg",
    MessageType.AUDIO: "audio/mpeg", MessageType.VIDEO: "video/mp4",
}


class WhatsAppAdapter(WhatsAppBehaviorMixin, BasePlatformAdapter):
    """WhatsApp adapter over a local Node.js (Baileys) HTTP bridge; behavior comes
    from ``WhatsAppBehaviorMixin``, only transport lives here.

    Configuration (config.extra):
    - bridge_script / bridge_port (default 3000) / session_path
    - dm_policy, group_policy: "open" | "allowlist" | "disabled" | "pairing" (default "pairing")
    - allow_from / group_allow_from: IDs allowed when the policy is "allowlist"
    - send_read_receipts: Mark accepted inbound WhatsApp messages as read
    """

    _DEFAULT_BRIDGE_DIR = None  # resolved in __init__
    splits_long_messages = True  # send() chunks via truncate_message()

    def __init__(self, config: PlatformConfig):
        super().__init__(config, Platform.WHATSAPP)
        if WhatsAppAdapter._DEFAULT_BRIDGE_DIR is None:
            from gateway.platforms.whatsapp_common import resolve_whatsapp_bridge_dir
            WhatsAppAdapter._DEFAULT_BRIDGE_DIR = resolve_whatsapp_bridge_dir()
        self._bridge_process: Optional[subprocess.Popen] = None
        self._bridge_port: int = config.extra.get("bridge_port", 3000)
        self._bridge_script: Optional[str] = config.extra.get(
            "bridge_script", str(self._DEFAULT_BRIDGE_DIR / "bridge.js"),
        )
        self._session_path: Path = Path(config.extra.get(
            "session_path", get_hermes_dir("platforms/whatsapp/session", "whatsapp/session")
        ))
        self._reply_prefix: Optional[str] = config.extra.get("reply_prefix")
        self._dm_policy = str(config.extra.get("dm_policy") or _wenv("WHATSAPP_DM_POLICY", "pairing")).strip().lower()
        allow_raw = self._select_dm_allowlist(config.extra, ("WHATSAPP_ALLOWED_USERS",), _wenv)
        self._allow_from = self._coerce_allow_list(allow_raw)
        self._group_policy = str(config.extra.get("group_policy") or _wenv("WHATSAPP_GROUP_POLICY", "pairing")).strip().lower()
        self._group_allow_from = self._coerce_allow_list(config.extra.get("group_allow_from") or config.extra.get("groupAllowFrom"))
        read_receipts = config.extra.get("send_read_receipts", False)
        self._send_read_receipts = (
            read_receipts if isinstance(read_receipts, bool)
            else str(read_receipts or "").strip().lower() in {"1", "true", "yes", "on"}
        )
        self._mention_patterns = self._compile_mention_patterns()
        self._message_queue: asyncio.Queue = asyncio.Queue()
        self._bridge_log_fh = None
        self._bridge_log: Optional[Path] = None
        self._poll_task: Optional[asyncio.Task] = None
        self._http_session: Optional["aiohttp.ClientSession"] = None
        # Set by disconnect() before SIGTERMing the child so _check_managed_bridge_exit()
        # can tell an intentional exit (-15 / -2 / 0) from a crash.
        self._shutting_down: bool = False

        # Text debounce batching: WhatsApp delivers rapid bursts (forwards,
        # paste-splits); without debounce each triggers a separate agent turn.
        # Tunable via extra.text_batch_delay_seconds / text_batch_split_delay_seconds.
        self._text_batch_delay_seconds = self._coerce_float_extra("text_batch_delay_seconds", 5.0)
        self._text_batch_split_delay_seconds = self._coerce_float_extra("text_batch_split_delay_seconds", 10.0)
        self._pending_text_batches: Dict[str, MessageEvent] = {}
        self._pending_text_batch_tasks: Dict[str, asyncio.Task] = {}

    def _coerce_float_extra(self, key: str, default: float) -> float:
        """Read a float from ``config.extra``; NaN/Inf/negative/unparseable → ``default`` (fed to asyncio.sleep)."""
        import math
        value = self.config.extra.get(key) if getattr(self.config, "extra", None) else None
        if value is None:
            return float(default)
        try:
            parsed = float(value)
        except (TypeError, ValueError):
            return float(default)
        if not math.isfinite(parsed) or parsed < 0:
            return float(default)
        return parsed

    # ── bridge lifecycle ─────────────────────────────────────────────

    def _bridge_url(self, path: str) -> str:
        return f"http://127.0.0.1:{self._bridge_port}/{path}"

    async def _probe_bridge_health(self) -> tuple[bool, Any]:
        """GET /health with a fresh session. Returns ``(http_200, json)``; a 200 with
        an unparseable body yields ``(True, None)``. Connection errors propagate."""
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(self._bridge_url("health"), timeout=aiohttp.ClientTimeout(total=2)) as resp:
                if resp.status != 200:
                    return False, None
                try:
                    return True, await resp.json()
                except Exception:
                    return True, None

    def _ensure_bridge_deps(self, bridge_dir: Path) -> bool:
        """npm install when node_modules is missing OR package.json changed since the
        last install (stamp file holds the package.json hash). False = fatal error set."""
        _pkg_json = bridge_dir / "package.json"
        _dep_stamp = bridge_dir / "node_modules" / ".hermes-pkg-hash"
        _pkg_hash = _file_content_hash(_pkg_json)
        _deps_fresh = False
        if (bridge_dir / "node_modules").exists():
            try:
                _deps_fresh = (_dep_stamp.read_text(encoding="utf-8").strip() == _pkg_hash) and bool(_pkg_hash)
            except OSError:
                _deps_fresh = False
        if _deps_fresh:
            return True
        print(f"[{self.name}] Installing WhatsApp bridge dependencies...")
        # Hermes-managed portable Node's npm.cmd first (Windows), then PATH.
        _npm_bin = find_node_executable("npm") or "npm"
        try:
            # Default 300s accommodates slow systems like an Unraid NAS.
            npm_install_timeout = env_int("WHATSAPP_NPM_INSTALL_TIMEOUT", 300)
            install_result = subprocess.run(
                [_npm_bin, "install", "--silent"],
                cwd=str(bridge_dir), timeout=npm_install_timeout, env=with_hermes_node_path(), **_RUN_TEXT,
            )
            if install_result.returncode != 0:
                print(f"[{self.name}] npm install failed: {install_result.stderr}")
                self._set_fatal_error(
                    "whatsapp_npm_install_failed",
                    f"WhatsApp bridge npm install failed. Run `cd {bridge_dir} && {_npm_bin} install` manually, then restart `hermes gateway`.",
                    retryable=False,
                )
                return False
            print(f"[{self.name}] Dependencies installed")
            if _pkg_hash:
                try:
                    _dep_stamp.write_text(_pkg_hash, encoding="utf-8")
                except OSError:
                    pass  # Stamp is an optimization; install still succeeded
        except Exception as e:
            print(f"[{self.name}] Failed to install dependencies: {e}")
            self._set_fatal_error(
                "whatsapp_npm_install_failed",
                f"WhatsApp bridge npm install failed ({e}). Run `cd {bridge_dir} && {_npm_bin} install` manually, then restart `hermes gateway`.",
                retryable=False,
            )
            return False
        return True

    def _attach_to_bridge(self, managed_process) -> None:
        import aiohttp
        self._bridge_process = managed_process
        self._http_session = aiohttp.ClientSession()
        self._poll_task = asyncio.create_task(self._poll_messages())

    async def _reuse_running_bridge(self, bridge_path: Path) -> bool:
        """Adopt an already-connected bridge if it serves the on-disk bridge.js and
        the same read-receipt config; otherwise report why it will be restarted.

        A long-lived bridge survives gateway restarts AND `hermes update`, so a
        hash mismatch (or no ``scriptHash`` at all) is stale by definition.
        """
        try:
            ok, data = await self._probe_bridge_health()
            if ok and data is not None:
                bridge_status = data.get("status", "unknown")
                if bridge_status == "connected":
                    running_hash = data.get("scriptHash", "")
                    disk_hash = _file_content_hash(bridge_path)
                    running_read_receipts = bool(data.get("sendReadReceipts", False))
                    config_matches = running_read_receipts == self._send_read_receipts
                    if running_hash and disk_hash and running_hash == disk_hash and config_matches:
                        print(f"[{self.name}] Using existing bridge (status: {bridge_status})")
                        self._mark_connected()
                        self._attach_to_bridge(None)  # Not managed by us
                        self._wire_plugin_handlers(None)
                        return True
                    stale_reason = (
                        f"running={running_hash or 'unversioned'}, disk={disk_hash}"
                        if running_hash != disk_hash
                        else "send_read_receipts config changed"
                    )
                    print(f"[{self.name}] Running bridge is stale ({stale_reason}), restarting")
                else:
                    print(f"[{self.name}] Bridge found but not connected (status: {bridge_status}), restarting")
        except Exception:
            pass  # Bridge not running, start a new one
        return False

    def _bridge_env(self) -> dict:
        """Subprocess env: profile-resolved WHATSAPP_* values + profile-aware cache dirs."""
        # with_hermes_node_path() copies os.environ when called with no arg.
        bridge_env = with_hermes_node_path()
        if self._reply_prefix is not None:
            bridge_env["WHATSAPP_REPLY_PREFIX"] = self._reply_prefix
        bridge_env["WHATSAPP_SEND_READ_RECEIPTS"] = "true" if self._send_read_receipts else "false"
        _profile_wa_mode = _wenv("WHATSAPP_MODE", "self-chat")
        if _profile_wa_mode:
            bridge_env["WHATSAPP_MODE"] = _profile_wa_mode
        for _key in _BRIDGE_PASSTHROUGH_ENV:
            _v = _wenv(_key)
            if _v:
                bridge_env[_key] = _v
        # Without these the bridge hardcodes ~/.hermes/{image,audio,document}_cache,
        # which diverges under HERMES_HOME overrides, profiles, and the cache/ layout.
        from gateway.platforms.base import (
            get_audio_cache_dir as _get_audio_dir,
            get_document_cache_dir as _get_doc_dir,
            get_image_cache_dir as _get_img_dir,
        )
        bridge_env["HERMES_IMAGE_CACHE_DIR"] = str(_get_img_dir())
        bridge_env["HERMES_AUDIO_CACHE_DIR"] = str(_get_audio_dir())
        bridge_env["HERMES_DOCUMENT_CACHE_DIR"] = str(_get_doc_dir())
        return bridge_env

    def _bridge_died(self, detail: str) -> bool:
        print(f"[{self.name}] {detail}")
        print(f"[{self.name}] Check log: {self._bridge_log}")
        self._close_bridge_log()
        return False

    async def _poll_bridge_health(self, died_msg: str) -> tuple[Optional[bool], bool, dict]:
        """Poll /health up to 15×1s. Returns ``(connected, http_ready, data)``; ``connected``
        is False when the process died (already reported), None when time ran out."""
        http_ready = False
        data: dict = {}
        for attempt in range(15):
            await asyncio.sleep(1)
            if self._bridge_process.poll() is not None:
                return self._bridge_died(died_msg.format(code=self._bridge_process.returncode)), http_ready, data
            try:
                ok, d = await self._probe_bridge_health()
                if ok:
                    http_ready = True
                    if d is not None:
                        data = d
                        if data.get("status") == "connected":
                            print(f"[{self.name}] Bridge ready (status: connected)")
                            return True, http_ready, data
            except Exception:
                continue
        return None, http_ready, data

    async def _wait_for_bridge(self) -> bool:
        """Phase 1: HTTP server up (≤15s). Phase 2: WhatsApp ``status: connected`` (≤15s more;
        proceeds with a warning if still connecting — the bridge may auto-reconnect later)."""
        connected, http_ready, data = await self._poll_bridge_health("Bridge process died (exit code {code})")
        if connected is False:
            return False
        if not http_ready:
            return self._bridge_died("Bridge HTTP server did not start in 15s")
        if data.get("status") != "connected":
            print(f"[{self.name}] Bridge HTTP ready, waiting for WhatsApp connection...")
            connected, _, _ = await self._poll_bridge_health("Bridge process died during connection")
            if connected is False:
                return False
            if connected is None:
                print(f"[{self.name}] ⚠ WhatsApp not connected after 30s")
                print(f"[{self.name}]   Bridge log: {self._bridge_log}")
                print(f"[{self.name}]   If session expired, re-pair: hermes whatsapp")
        return True

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        """Start (or adopt) the Node.js bridge and wait for it to be ready."""
        if not check_whatsapp_requirements():
            logger.warning("[%s] Node.js not found. WhatsApp requires Node.js.", self.name)
            self._set_fatal_error(
                "whatsapp_node_missing",
                "Node.js is not installed — install Node.js and re-run `hermes gateway`.",
                retryable=False,
            )
            return False
        bridge_path = Path(self._bridge_script)
        if not bridge_path.exists():
            logger.warning("[%s] Bridge script not found: %s", self.name, bridge_path)
            self._set_fatal_error(
                "whatsapp_bridge_missing",
                f"WhatsApp bridge script missing at {bridge_path}.",
                retryable=False,
            )
            return False

        # Without creds.json the bridge only prints QR codes and never connects, so
        # every restart would pay the 30s timeout; fail non-retryable with a clear message.
        creds_path = self._session_path / "creds.json"
        if not creds_path.exists():
            logger.warning(
                "[%s] WhatsApp is enabled but not paired (no creds.json at %s). "
                "Pair from the dashboard or run `hermes whatsapp`; remove "
                "WHATSAPP_ENABLED from your .env to disable.",
                self.name, creds_path,
            )
            self._set_fatal_error(
                "whatsapp_not_paired",
                "WhatsApp enabled but not paired — pair from the dashboard or run `hermes whatsapp`.",
                retryable=False,
            )
            return False
        logger.info("[%s] Bridge found at %s", self.name, bridge_path)
        lock_acquired = False
        try:
            if not self._acquire_platform_lock('whatsapp-session', str(self._session_path), 'WhatsApp session'):
                return False
            lock_acquired = True
        except Exception as e:
            logger.warning("[%s] Could not acquire session lock (non-fatal): %s", self.name, e)
        try:
            if not self._ensure_bridge_deps(bridge_path.parent):
                return False
            self._session_path.mkdir(parents=True, exist_ok=True)
            if await self._reuse_running_bridge(bridge_path):
                return True
            _kill_stale_bridge_by_pidfile(self._session_path)
            _kill_port_process(self._bridge_port)
            await asyncio.sleep(1)

            # Bridge output goes to a log file so QR codes, errors, and reconnection
            # messages are preserved for troubleshooting.
            whatsapp_mode = _wenv("WHATSAPP_MODE", "self-chat")
            self._bridge_log = self._session_path.parent / "bridge.log"
            bridge_log_fh = open(self._bridge_log, "a", encoding="utf-8")
            self._bridge_log_fh = bridge_log_fh
            bridge_env = self._bridge_env()
            self._bridge_process = subprocess.Popen(
                [
                    find_node_executable("node") or "node",
                    str(bridge_path),
                    "--port", str(self._bridge_port),
                    "--session", str(self._session_path),
                    "--mode", whatsapp_mode,
                ],
                stdout=bridge_log_fh,
                stderr=bridge_log_fh,
                env=bridge_env,
                **windows_detach_popen_kwargs(),
            )
            _write_bridge_pidfile(self._session_path, self._bridge_process.pid)
            if not await self._wait_for_bridge():
                return False
            self._attach_to_bridge(self._bridge_process)
            self._mark_connected()
            print(f"[{self.name}] Bridge started on port {self._bridge_port}")
            self._wire_plugin_handlers(None)
            return True
        except Exception as e:
            logger.error("[%s] Failed to start bridge: %s", self.name, e, exc_info=True)
            return False
        finally:
            if not self._running:
                if lock_acquired:
                    self._release_platform_lock()
                self._close_bridge_log()

    def _close_bridge_log(self) -> None:
        """Close the bridge log file handle if open."""
        if self._bridge_log_fh:
            try:
                self._bridge_log_fh.close()
            except Exception:
                pass
            self._bridge_log_fh = None

    async def _check_managed_bridge_exit(self) -> Optional[str]:
        """Return a fatal error message if the managed bridge child exited."""
        if self._bridge_process is None:
            return None
        returncode = self._bridge_process.poll()
        if returncode is None:
            return None

        # getattr-with-default: tests build the adapter via ``__new__`` without __init__.
        if getattr(self, "_shutting_down", False) and returncode in {0, -2, -15}:
            logger.info("[%s] Bridge exited during shutdown (code %d).", self.name, returncode)
            return None
        message = f"WhatsApp bridge process exited unexpectedly (code {returncode})."
        if not self.has_fatal_error:
            logger.error("[%s] %s", self.name, message)
            self._set_fatal_error("whatsapp_bridge_exited", message, retryable=True)
            self._close_bridge_log()
            await self._notify_fatal_error()
        return self.fatal_error_message or message

    async def disconnect(self) -> None:
        """Stop the WhatsApp bridge and clean up any orphaned processes."""
        # Flip BEFORE signalling so the exit-check path (send(), poll loop) doesn't
        # report the intentional termination as fatal.
        self._shutting_down = True
        if self._bridge_process:
            try:
                try:
                    _terminate_bridge_process(self._bridge_process, force=False)
                except (ProcessLookupError, PermissionError):
                    self._bridge_process.terminate()
                await asyncio.sleep(1)
                if self._bridge_process.poll() is None:
                    try:
                        _terminate_bridge_process(self._bridge_process, force=True)
                    except (ProcessLookupError, PermissionError):
                        self._bridge_process.kill()
            except Exception as e:
                print(f"[{self.name}] Error stopping bridge: {e}")
        else:
            print(f"[{self.name}] Disconnecting (external bridge left running)")
        try:
            (self._session_path / "bridge.pid").unlink(missing_ok=True)
        except OSError:
            pass
        if self._poll_task and not self._poll_task.done():
            self._poll_task.cancel()
            try:
                await self._poll_task
            except (asyncio.CancelledError, Exception):
                pass
        self._poll_task = None
        if self._http_session and not self._http_session.closed:
            await self._http_session.close()
        self._http_session = None
        self._release_platform_lock()
        self._mark_disconnected()
        self._bridge_process = None
        self._close_bridge_log()
        print(f"[{self.name}] Disconnected")

    # ── outbound ─────────────────────────────────────────────────────

    async def _bridge_unavailable(self) -> Optional[str]:
        """Error string when the bridge can't take a request, else None."""
        if not self._running or not self._http_session:
            return "Not connected"
        return await self._check_managed_bridge_exit() or None

    async def _post_bridge_message(self, path: str, payload: Dict[str, Any], *, timeout: float) -> SendResult:
        """POST to the bridge; 200 → SendResult(messageId, raw_response), else the error text."""
        try:
            import aiohttp
            async with self._http_session.post(
                self._bridge_url(path), json=payload, timeout=aiohttp.ClientTimeout(total=timeout),
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return SendResult(success=True, message_id=data.get("messageId"), raw_response=data)
                error = await resp.text()
                return SendResult(success=False, error=error)
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def send(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> SendResult:
        """Format markdown for WhatsApp, chunk preserving code blocks, send sequentially."""
        unavailable = await self._bridge_unavailable()
        if unavailable:
            return SendResult(success=False, error=unavailable)
        if not content or not content.strip():
            return SendResult(success=True, message_id=None)
        chat_id = to_whatsapp_jid(chat_id)
        try:
            formatted = self.format_message(content)
            chunks = self.truncate_message(formatted, self._outgoing_chunk_limit())
            sent_message_ids: list[str] = []
            last_message_id = None
            for idx, chunk in enumerate(chunks):
                payload: Dict[str, Any] = {"chatId": chat_id, "message": chunk}
                if reply_to and idx == 0:
                    # Reply-to on the first chunk only.
                    payload["replyTo"] = reply_to
                result = await self._post_bridge_message("send", payload, timeout=30)
                if not result.success:
                    return SendResult(success=False, error=result.error)
                last_message_id = result.message_id
                if last_message_id:
                    sent_message_ids.append(str(last_message_id))
                # Small delay between chunks to avoid rate limiting
                if len(chunks) > 1:
                    await asyncio.sleep(0.3)
            return SendResult(
                success=True,
                message_id=last_message_id,
                continuation_message_ids=tuple(sent_message_ids[:-1]),
                raw_response={"message_ids": sent_message_ids},
            )
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def edit_message(
        self, chat_id: str, message_id: str, content: str, *, finalize: bool = False,
    ) -> SendResult:
        """Edit a previously sent message via the WhatsApp bridge."""
        unavailable = await self._bridge_unavailable()
        if unavailable:
            return SendResult(success=False, error=unavailable)
        try:
            import aiohttp
            async with self._http_session.post(
                self._bridge_url("edit"),
                json={"chatId": to_whatsapp_jid(chat_id), "messageId": message_id, "message": content},
                timeout=aiohttp.ClientTimeout(total=15)
            ) as resp:
                if resp.status == 200:
                    return SendResult(success=True, message_id=message_id)
                error = await resp.text()
                return SendResult(success=False, error=error)
        except Exception as e:
            return SendResult(success=False, error=str(e))

    async def _send_media_to_bridge(
        self, chat_id: str, file_path: str, media_type: str, caption: Optional[str] = None,
        file_name: Optional[str] = None,
    ) -> SendResult:
        """Send any media file via bridge /send-media endpoint."""
        unavailable = await self._bridge_unavailable()
        if unavailable:
            return SendResult(success=False, error=unavailable)
        if not os.path.exists(file_path):
            return SendResult(success=False, error=f"File not found: {file_path}")
        payload: Dict[str, Any] = {"chatId": to_whatsapp_jid(chat_id), "filePath": file_path, "mediaType": media_type}
        if caption:
            payload["caption"] = caption
        if file_name:
            payload["fileName"] = file_name
        return await self._post_bridge_message("send-media", payload, timeout=120)

    async def send_poll(
        self, chat_id: str, question: str, options: list[str], *, selectable_count: int = 1,
    ) -> SendResult:
        """Native WhatsApp poll (low-level transport primitive; approval UX stays gateway-owned)."""
        unavailable = await self._bridge_unavailable()
        if unavailable:
            return SendResult(success=False, error=unavailable)
        payload: Dict[str, Any] = {
            "chatId": to_whatsapp_jid(chat_id),
            "question": question,
            "options": list(options or []),
            "selectableCount": selectable_count,
        }
        return await self._post_bridge_message("send-poll", payload, timeout=30)

    async def send_clarify(
        self, chat_id: str, question: str, choices: Optional[list], clarify_id: str,
        session_key: str, metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Multiple-choice clarify as a native poll; Baileys emits the selected option
        as message text, which the normal clarify text-intercept resolves. Open-ended
        (or failed poll) falls back to the text prompt."""
        clean_choices = [str(choice).strip() for choice in (choices or []) if str(choice).strip()]
        if 2 <= len(clean_choices) <= 12:
            result = await self.send_poll(chat_id, str(question or "").strip(), clean_choices, selectable_count=1)
            if result.success:
                return result
            logger.warning(
                "[%s] Native WhatsApp clarify poll failed; falling back to text: %s",
                self.name, result.error,
            )
        return await super().send_clarify(
            chat_id=chat_id, question=question, choices=choices,
            clarify_id=clarify_id, session_key=session_key, metadata=metadata,
        )

    async def send_location(
        self, chat_id: str, latitude: float, longitude: float, *, name: Optional[str] = None,
        address: Optional[str] = None, reply_to: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Send a native WhatsApp location pin via the Baileys bridge."""
        unavailable = await self._bridge_unavailable()
        if unavailable:
            return SendResult(success=False, error=unavailable)
        try:
            payload: Dict[str, Any] = {
                "chatId": to_whatsapp_jid(chat_id),
                "latitude": float(latitude),
                "longitude": float(longitude),
            }
        except Exception as e:
            return SendResult(success=False, error=str(e))
        if name:
            payload["name"] = name
        if address:
            payload["address"] = address
        return await self._post_bridge_message("send-location", payload, timeout=30)

    async def send_image(
        self, chat_id: str, image_url: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, metadata: Optional[Dict[str, Any]] = None,
    ) -> SendResult:
        """Download image URL to cache, send natively via bridge (``metadata`` honors the base contract)."""
        try:
            local_path = await cache_image_from_url(image_url)
            return await self._send_media_to_bridge(chat_id, local_path, "image", caption)
        except Exception:
            return await super().send_image(chat_id, image_url, caption, reply_to, metadata)

    async def send_image_file(
        self, chat_id: str, image_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, **kwargs,
    ) -> SendResult:
        return await self._send_media_to_bridge(chat_id, image_path, "image", caption)

    async def send_video(
        self, chat_id: str, video_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, **kwargs,
    ) -> SendResult:
        return await self._send_media_to_bridge(chat_id, video_path, "video", caption)

    async def send_voice(
        self, chat_id: str, audio_path: str, caption: Optional[str] = None,
        reply_to: Optional[str] = None, **kwargs,
    ) -> SendResult:
        return await self._send_media_to_bridge(chat_id, audio_path, "audio", caption)

    async def send_document(
        self, chat_id: str, file_path: str, caption: Optional[str] = None,
        file_name: Optional[str] = None, reply_to: Optional[str] = None, **kwargs,
    ) -> SendResult:
        return await self._send_media_to_bridge(
            chat_id, file_path, "document", caption, file_name or os.path.basename(file_path),
        )

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        """Send typing indicator via bridge (failures ignored)."""
        if await self._bridge_unavailable():
            return
        try:
            import aiohttp

            # ``async with`` — a bare ``await session.post(...)`` leaves the response
            # alive until GC, holding its socket in CLOSE_WAIT.
            async with self._http_session.post(
                self._bridge_url("typing"),
                json={"chatId": to_whatsapp_jid(chat_id)},
                timeout=aiohttp.ClientTimeout(total=5)
            ):
                pass
        except Exception:
            pass

    async def get_chat_info(self, chat_id: str) -> Dict[str, Any]:
        """Get information about a WhatsApp chat."""
        if not self._running or not self._http_session:
            return {"name": "Unknown", "type": "dm"}
        if await self._check_managed_bridge_exit():
            return {"name": chat_id, "type": "dm"}
        try:
            import aiohttp
            async with self._http_session.get(
                self._bridge_url(f"chat/{to_whatsapp_jid(chat_id)}"),
                timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return {
                        "name": data.get("name", chat_id),
                        "type": "group" if data.get("isGroup") else "dm",
                        "participants": data.get("participants", []),
                    }
        except Exception as e:
            logger.debug("Could not get WhatsApp chat info for %s: %s", chat_id, e)
        return {"name": chat_id, "type": "dm"}

    # ── inbound ──────────────────────────────────────────────────────

    async def _poll_messages(self) -> None:
        """Poll the bridge for incoming messages."""
        import aiohttp
        while self._running:
            if not self._http_session:
                break
            bridge_exit = await self._check_managed_bridge_exit()
            if bridge_exit:
                print(f"[{self.name}] {bridge_exit}")
                break
            try:
                async with self._http_session.get(
                    self._bridge_url("messages"), timeout=aiohttp.ClientTimeout(total=30)
                ) as resp:
                    if resp.status == 200:
                        messages = await resp.json()
                        for msg_data in messages:
                            event = await self._build_message_event(msg_data)
                            if event:
                                # Fire-and-forget: a slow bridge /read must not delay dispatch.
                                asyncio.create_task(self._send_read_receipt(msg_data))
                                if event.message_type == MessageType.TEXT:
                                    self._enqueue_text_event(event)
                                else:
                                    await self.handle_message(event)
            except asyncio.CancelledError:
                break
            except Exception as e:
                bridge_exit = await self._check_managed_bridge_exit()
                if bridge_exit:
                    print(f"[{self.name}] {bridge_exit}")
                    break
                print(f"[{self.name}] Poll error: {e}")
                await asyncio.sleep(5)
            await asyncio.sleep(1)  # Poll interval

    async def _send_read_receipt(self, data: Dict[str, Any]) -> None:
        """Mark a policy-accepted inbound message as read via the bridge."""
        if not self._send_read_receipts or not self._http_session:
            return
        key = data.get("readReceiptKey")
        if not isinstance(key, dict):
            return
        try:
            import aiohttp
            async with self._http_session.post(
                self._bridge_url("read"), json={"key": key}, timeout=aiohttp.ClientTimeout(total=5),
            ) as resp:
                if resp.status != 200:
                    logger.warning("[%s] WhatsApp read receipt failed with HTTP %s", self.name, resp.status)
        except Exception as exc:
            logger.warning("[%s] WhatsApp read receipt failed: %s", self.name, exc)

    # ── Text debounce batching ──────────────────────────────────────

    _SPLIT_THRESHOLD = 6000  # WhatsApp supports ~65K chars; generous threshold

    async def _flush_text_batch(self, key: str) -> None:
        """Wait for quiet period then dispatch aggregated text."""
        current_task = asyncio.current_task()
        try:
            pending = self._pending_text_batches.get(key)
            last_len = getattr(pending, "_last_chunk_len", 0) if pending else 0
            if last_len >= self._SPLIT_THRESHOLD:
                delay = self._text_batch_split_delay_seconds
            else:
                delay = self._text_batch_delay_seconds
            await asyncio.sleep(delay)
            event = self._pending_text_batches.pop(key, None)
            if not event:
                return
            await self.handle_message(event)
        finally:
            if self._pending_text_batch_tasks.get(key) is current_task:
                self._pending_text_batch_tasks.pop(key, None)

    @staticmethod
    def _classify_bridge_message(data: Dict[str, Any]) -> MessageType:
        media_type = str(data.get("mediaType", "") or "")
        if media_type in {"location", "live_location"}:
            return MessageType.LOCATION
        if media_type == "sticker":
            return MessageType.STICKER
        if data.get("hasMedia"):
            if "image" in media_type:
                return MessageType.PHOTO
            if "video" in media_type:
                return MessageType.VIDEO
            if "ptt" in media_type:  # ptt = WhatsApp voice note
                return MessageType.VOICE
            if "audio" in media_type:
                return MessageType.AUDIO
            return MessageType.DOCUMENT
        return MessageType.TEXT

    async def _collect_bridge_media(
        self, data: Dict[str, Any], msg_type: MessageType
    ) -> tuple[list, list]:
        """Resolve bridge ``mediaUrls`` into ``(cached_urls, media_types)``.

        Remote image/audio URLs are downloaded to the local cache; absolute paths the
        bridge already downloaded are accepted only inside a Hermes cache dir.
        """
        cached_urls = []
        media_types = []
        label = _MEDIA_LABEL.get(msg_type)
        for url in data.get("mediaUrls", []):
            bridge_mime = str(data.get("mime") or "").strip()
            is_http = url.startswith(("http://", "https://"))
            if msg_type == MessageType.DOCUMENT:
                mime = bridge_mime or SUPPORTED_DOCUMENT_TYPES.get(Path(url).suffix.lower(), "application/octet-stream")
            else:
                mime = bridge_mime or _MEDIA_DEFAULT_MIME.get(msg_type, "")
            if is_http and msg_type in {MessageType.PHOTO, MessageType.VOICE, MessageType.AUDIO}:
                cacher, ext = (
                    (cache_image_from_url, ".jpg") if msg_type == MessageType.PHOTO else (cache_audio_from_url, ".ogg")
                )
                try:
                    cached_path = await cacher(url, ext=ext)
                    cached_urls.append(cached_path)
                    media_types.append(mime)
                    print(f"[{self.name}] Cached user {label}: {cached_path}", flush=True)
                except Exception as e:
                    print(f"[{self.name}] Failed to cache {label}: {e}", flush=True)
                    cached_urls.append(url)
                    media_types.append(mime)
            elif label is not None and os.path.isabs(url):
                if _is_allowed_bridge_path(url):
                    cached_urls.append(url)
                    media_types.append(mime)
                    print(f"[{self.name}] Using bridge-cached {label}: {url}", flush=True)
                else:
                    print(f"[{self.name}] Rejected bridge {label} path outside cache dir: {url}", flush=True)
            else:
                cached_urls.append(url)
                media_types.append("unknown")
        return cached_urls, media_types

    def _inject_document_text(self, cached_urls: list, body: str) -> str:
        """Prepend text-readable document contents (≤100KB) so the agent reads them inline."""
        for doc_path in cached_urls:
            if Path(doc_path).suffix.lower() not in _TEXT_INJECT_EXTS:
                continue
            try:
                file_size = Path(doc_path).stat().st_size
                if file_size > _MAX_TEXT_INJECT_BYTES:
                    print(f"[{self.name}] Skipping text injection for {doc_path} ({file_size} bytes > {_MAX_TEXT_INJECT_BYTES})", flush=True)
                    continue
                content = Path(doc_path).read_text(encoding="utf-8", errors="replace")
                fname = Path(doc_path).name
                # Remove the doc_<hex>_ prefix for display
                display_name = fname
                if "_" in fname:
                    parts = fname.split("_", 2)
                    if len(parts) >= 3:
                        display_name = parts[2]
                injection = f"[Content of {display_name}]:\n{content}"
                body = f"{injection}\n\n{body}" if body else injection
                print(f"[{self.name}] Injected text content from: {doc_path}", flush=True)
            except Exception as e:
                print(f"[{self.name}] Failed to read document text: {e}", flush=True)
        return body

    async def _build_message_event(self, data: Dict[str, Any]) -> Optional[MessageEvent]:
        """Build a MessageEvent from bridge message data, downloading images to cache."""
        try:
            if not self._should_process_message(data):
                return None
            msg_type = self._classify_bridge_message(data)
            source = self.build_source(
                chat_id=data.get("chatId", ""),
                chat_name=data.get("chatName"),
                chat_type="group" if data.get("isGroup", False) else "dm",
                user_id=data.get("senderId"),
                user_name=data.get("senderName"),
            )
            cached_urls, media_types = await self._collect_bridge_media(data, msg_type)
            body = data.get("body", "")
            if data.get("isGroup"):
                body = self._clean_bot_mention_text(body, data)
            if (
                msg_type == MessageType.VOICE
                and cached_urls
                and str(body).strip().lower() == "[ptt received]"
            ):
                # Bridge placeholder for captionless voice notes; the audio is the payload.
                body = ""

            # Quoted message stays in structured fields only — GatewayRunner renders
            # the "[Replying to: ...]" pointer for every platform.
            quoted_text = str(data.get("quotedText") or "").strip()
            reply_to_text = quoted_text or None
            reply_to_message_id = None
            reply_to_author_id = None
            reply_to_is_own_message = False
            if data.get("hasQuotedMessage"):
                raw_reply_id = data.get("quotedMessageId")
                if raw_reply_id is not None:
                    reply_to_message_id = str(raw_reply_id)
                quoted_participant = self._normalize_whatsapp_id(data.get("quotedParticipant"))
                if quoted_participant:
                    reply_to_author_id = quoted_participant
                reply_to_is_own_message = self._message_is_reply_to_bot(data)
            if msg_type == MessageType.DOCUMENT and cached_urls:
                body = self._inject_document_text(cached_urls, body)
            metadata: Dict[str, Any] = {}
            native_type = str(data.get("nativeType") or "").strip()
            native_metadata = data.get("nativeMetadata")
            if native_type:
                metadata["whatsapp_native_type"] = native_type
            if isinstance(native_metadata, dict) and native_metadata:
                metadata["whatsapp_native"] = native_metadata
            # ``fromOwner`` = owner-typed inbound fromMe (linked device), gated by
            # WHATSAPP_FORWARD_OWNER_MESSAGES at the bridge. Surfaced as metadata AND
            # a text prefix so the marker survives downstream failures before silent_ingest.
            if data.get("fromOwner"):
                metadata["whatsapp_from_owner"] = True
                if not body.startswith(_OWNER_REPLY_PREFIX):
                    body = f"{_OWNER_REPLY_PREFIX}{body}"
            return MessageEvent(
                text=body,
                message_type=msg_type,
                source=source,
                raw_message=data,
                message_id=data.get("messageId"),
                media_urls=cached_urls,
                media_types=media_types,
                metadata=metadata,
                reply_to_message_id=reply_to_message_id,
                reply_to_text=reply_to_text,
                reply_to_author_id=reply_to_author_id,
                reply_to_is_own_message=reply_to_is_own_message,
            )
        except Exception as e:
            print(f"[{self.name}] Error building event: {e}")
            return None


# ──────────────────────────────────────────────────────────────────────────
# Plugin glue: register(ctx) plus the hook implementations that replaced the
# per-platform core touchpoints (gateway/run.py, gateway/config.py,
# hermes_cli/gateway.py, tools/send_message_tool.py).
# ──────────────────────────────────────────────────────────────────────────


_WA_IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".gif"}
_WA_VIDEO_EXTS = {".mp4", ".mov", ".avi", ".mkv", ".webm", ".3gp"}
_WA_AUDIO_EXTS = {".ogg", ".opus", ".mp3", ".wav", ".m4a", ".flac"}


def _bridge_media_type(file_path: str, is_voice: bool, force_document: bool) -> str:
    """Map a local file to the bridge ``mediaType`` (image|video|audio|document).

    ``force_document`` (the [[as_document]] directive) forces ``document``.
    """
    if force_document:
        return "document"
    ext = os.path.splitext(file_path)[1].lower()
    if is_voice or ext in _WA_AUDIO_EXTS:
        return "audio"
    if ext in _WA_IMAGE_EXTS:
        return "image"
    if ext in _WA_VIDEO_EXTS:
        return "video"
    return "document"


async def _standalone_send(
    pconfig, chat_id, message, *, thread_id=None, media_files=None, force_document=False,
    caption=None,
):
    """Out-of-process WhatsApp delivery via the bridge HTTP API (standalone_sender_fn
    contract, so deliver=whatsapp cron jobs work when cron runs apart from the gateway).

    With ``caption`` (single-file ``MEDIA:<path> caption`` send) the text rides on
    the media bubble's native caption instead of a separate ``/send``.
    """
    extra = getattr(pconfig, "extra", {}) or {}
    try:
        import aiohttp
    except ImportError:
        return {"error": "aiohttp not installed. Run: pip install aiohttp"}
    try:
        bridge_port = extra.get("bridge_port", 3000)
        normalized_chat_id = to_whatsapp_jid(chat_id)
        media = media_files or []
        text = message or ""
        # A caption only applies to a single media file — never repeat it across a multi-file send.
        media_caption = caption if (caption and len(media) == 1) else None
        last_message_id = None
        async with aiohttp.ClientSession() as session:
            def _post(path, payload, total):
                return session.post(
                    f"http://localhost:{bridge_port}/{path}", json=payload, timeout=aiohttp.ClientTimeout(total=total),
                )

            # 1) Text first (skipped when media-only or when the text rides as the caption).
            if text.strip() and not media_caption:
                async with _post("send", {"chatId": normalized_chat_id, "message": text}, 30) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        return {"error": f"WhatsApp bridge error ({resp.status}): {body}"}
                    data = await resp.json()
                    last_message_id = data.get("messageId")

            # 2) Each media file as a native attachment (mediaType picks the WhatsApp kind).
            for media_path, is_voice in media:
                if not os.path.exists(media_path):
                    # In caption mode the words would be lost with the missing file —
                    # deliver the caption as a plain message so nothing silently disappears.
                    if media_caption:
                        try:
                            async with _post("send", {"chatId": normalized_chat_id, "message": media_caption}, 30) as resp:
                                if resp.status == 200:
                                    last_message_id = (await resp.json()).get("messageId")
                        except Exception:
                            logger.warning("WhatsApp caption-fallback send failed for missing media")
                    return {"error": f"WhatsApp media file not found: {media_path}"}
                media_type = _bridge_media_type(media_path, is_voice, force_document)
                payload: Dict[str, Any] = {
                    "chatId": normalized_chat_id,
                    "filePath": media_path,
                    "mediaType": media_type,
                }
                if media_type == "document":
                    payload["fileName"] = os.path.basename(media_path)
                if media_caption:
                    payload["caption"] = media_caption
                async with _post("send-media", payload, 120) as resp:
                    if resp.status != 200:
                        body = await resp.text()
                        return {"error": f"WhatsApp media error ({resp.status}): {body}"}
                    data = await resp.json()
                    last_message_id = data.get("messageId") or last_message_id
        return {
            "success": True,
            "platform": "whatsapp",
            "chat_id": normalized_chat_id,
            "message_id": last_message_id,
        }
    except Exception as e:
        return {"error": f"WhatsApp send failed: {e}"}


def interactive_setup() -> None:
    """Guide the user through WhatsApp setup (CLI helpers lazy-imported)."""
    from hermes_cli.config import get_env_value, remove_env_value, save_env_value
    from hermes_cli.cli_output import (
        prompt,
        prompt_yes_no,
        print_header,
        print_info,
        print_success,
    )
    print_header("WhatsApp")
    print_info("WhatsApp uses a local Node.js bridge (WhatsApp Web client).")
    print_info("Start the bridge separately; the gateway connects to it over HTTP.")
    existing = get_env_value("WHATSAPP_ENABLED")
    if existing and existing.lower() in {"true", "1", "yes"}:
        print_info("WhatsApp: already enabled")
        if not prompt_yes_no("Reconfigure WhatsApp?", False):
            return
    if prompt_yes_no("Enable WhatsApp?", True):
        save_env_value("WHATSAPP_ENABLED", "true")
        print_success("WhatsApp enabled")
    else:
        save_env_value("WHATSAPP_ENABLED", "false")
        print_info("WhatsApp left disabled")
        return
    allowed_users = prompt("Allowed user IDs (comma-separated, leave empty for no allowlist)")
    if allowed_users:
        save_env_value("WHATSAPP_ALLOWED_USERS", allowed_users.replace(" ", ""))
        print_success("WhatsApp allowlist configured")
    home_channel = prompt("Home chat ID for cron delivery (leave empty to skip)").strip()
    if home_channel:
        save_env_value("WHATSAPP_HOME_CHANNEL", home_channel)
    else:
        if remove_env_value("WHATSAPP_HOME_CHANNEL"):
            print_info("Home channel cleared.")


# config.yaml whatsapp: key → env var. Env vars take precedence over YAML.
_YAML_LOWERCASE_KEYS = (
    ("require_mention", "WHATSAPP_REQUIRE_MENTION"),
    ("dm_policy", "WHATSAPP_DM_POLICY"),
    ("group_policy", "WHATSAPP_GROUP_POLICY"),
)
_YAML_LIST_KEYS = (
    ("free_response_chats", "WHATSAPP_FREE_RESPONSE_CHATS"),
    ("allow_from", "WHATSAPP_ALLOWED_USERS"),
    ("group_allow_from", "WHATSAPP_GROUP_ALLOWED_USERS"),
)


def _apply_yaml_config(yaml_cfg: dict, whatsapp_cfg: dict) -> dict | None:
    """Translate config.yaml whatsapp: keys into WHATSAPP_* env vars (apply_yaml_config_fn contract).

    Returns None — everything flows through env.
    """
    import json as _json
    for key, env in _YAML_LOWERCASE_KEYS:
        if key in whatsapp_cfg and not os.getenv(env):
            os.environ[env] = str(whatsapp_cfg[key]).lower()
    if "mention_patterns" in whatsapp_cfg and not os.getenv("WHATSAPP_MENTION_PATTERNS"):
        os.environ["WHATSAPP_MENTION_PATTERNS"] = _json.dumps(whatsapp_cfg["mention_patterns"])
    for key, env in _YAML_LIST_KEYS:
        val = whatsapp_cfg.get(key)
        if val is not None and not os.getenv(env):
            if isinstance(val, list):
                val = ",".join(str(v) for v in val)
            os.environ[env] = str(val)
    return None


def _is_connected(config) -> bool:
    """Connected == explicitly enabled via WHATSAPP_ENABLED (or an enabled PlatformConfig with extras).

    Auth is handled by the Node.js bridge, so the opt-in flag is the signal; an
    unconditional True would show WhatsApp as configured in ``hermes setup`` always.
    """
    extra = getattr(config, "extra", {}) or {}
    if config is not None and getattr(config, "enabled", False) and extra:
        return True
    # Via hermes_cli.gateway.get_env_value (not os.getenv) so setup-status callers
    # that patch get_env_value observe the same value.
    import hermes_cli.gateway as gateway_mod
    val = (gateway_mod.get_env_value("WHATSAPP_ENABLED") or "").strip().lower()
    return val in {"true", "1", "yes"}


def _build_adapter(config):
    """Factory wrapper that constructs WhatsAppAdapter from a PlatformConfig."""
    return WhatsAppAdapter(config)


def register(ctx) -> None:
    """Plugin entry point — called by the Hermes plugin system."""
    ctx.register_platform(
        name="whatsapp",
        label="WhatsApp",
        adapter_factory=_build_adapter,
        check_fn=check_whatsapp_requirements,
        is_connected=_is_connected,
        required_env=["WHATSAPP_ENABLED"],
        install_hint="WhatsApp requires a Node.js bridge — see the WhatsApp messaging docs",
        setup_fn=interactive_setup,
        apply_yaml_config_fn=_apply_yaml_config,
        allowed_users_env="WHATSAPP_ALLOWED_USERS",
        allow_all_env="WHATSAPP_ALLOW_ALL_USERS",
        cron_deliver_env_var="WHATSAPP_HOME_CHANNEL",
        standalone_sender_fn=_standalone_send,
        max_message_length=4096,
        emoji="💬",
        allow_update_command=True,
    )
