"""Cua-driver backend (macOS, Windows, Linux): MCP over stdio to `cua-driver`.

The async `mcp` SDK runs on a background loop (``cua_backend_session``); the
same tool surface works on all three platforms, and per-host gaps (no DISPLAY,
missing AT-SPI, TCC) surface via `hermes computer-use doctor` instead of
failing silently. Install with `hermes computer-use install`. The macOS path
uses private SkyLight SPIs that can break on OS updates.

Siblings: ``cua_backend_driver`` (binary resolution, runtime contract, update
check), ``cua_backend_capture`` / ``cua_backend_input`` (backend mixins),
``cua_backend_parse`` (pure parsing), ``cua_backend_session`` (bridge + session
+ CLI fallback), ``cua_backend_daemon`` (private daemon + macOS app identity).
Moved names are re-imported here so ``patch("tools.computer_use.cua_backend.X")``
keeps working; siblings look policy helpers up lazily through this module.
"""

from __future__ import annotations

import logging
import os
import shutil  # noqa: F401  (tests patch cua_backend.shutil / .subprocess / .threading)
import subprocess
import sys
import threading
import uuid
from typing import Any, Dict, List, Optional

from hermes_cli._subprocess_compat import windows_hide_flags
from tools.computer_use.backend import ActionResult, ComputerUseBackend
from tools.computer_use.cua_backend_capture import (  # noqa: F401
    _CaptureMixin,
    _linux_x11_active_window_id,
    _select_capture_target,
)
from tools.computer_use.cua_backend_daemon import (  # noqa: F401
    _EmbeddedCuaDaemon,
    _embedded_daemon_spawn_command,
    _resolve_cua_driver_app_path,
    _validate_cua_driver_app_signature,
)
from tools.computer_use.cua_backend_driver import (  # noqa: F401
    _CUA_DRIVER_ARGS,
    _CUA_DRIVER_CMD_ENV,
    _cua_driver_supports_no_overlay,
    _mcp_args_with_overlay_flag,
    _resolve_mcp_invocation,
    _wsl_windows_path_to_posix,
    cua_driver_binary_available,
    cua_driver_install_hint,
    cua_driver_runtime_contract_status,
    cua_driver_update_check,
    cua_driver_update_nudge,
    resolve_cua_driver_cmd,
)
from tools.computer_use.cua_backend_input import _InputMixin
from tools.computer_use.cua_backend_parse import (  # noqa: F401
    _action_result_from,
    _extract_tool_result,
    _image_dimensions_from_bytes,
    _ingest_windows,
    _is_placeholder_id,
    _parse_elements_from_structured,
    _parse_elements_from_tree,
    _parse_key_combo,
    _parse_xprop_net_active_window,
    _windows_from_tool_result,
)
from tools.computer_use.cua_backend_session import _AsyncBridge, _CuaDriverSession  # noqa: F401

logger = logging.getLogger(__name__)

# cua-driver's anonymous PostHog telemetry gate ("0" disables; absent => ON upstream).
_CUA_TELEMETRY_ENV_VAR = "CUA_DRIVER_RS_TELEMETRY_ENABLED"


# ---------------------------------------------------------------------------
# Config-derived policy
# ---------------------------------------------------------------------------

def _computer_use_cfg() -> Dict[str, Any]:
    """The ``computer_use`` config block, or ``{}`` when config is unreadable."""
    try:
        from hermes_cli.config import load_config

        return (load_config() or {}).get("computer_use") or {}
    except Exception:
        return {}

def _cua_no_overlay() -> bool:
    """Pass ``--no-overlay``? ``computer_use.no_overlay`` overrides; else off on
    macOS (cursor-overlay redraw loop can peg a core after a session), headless
    Linux / WSL2 / containers, and Linux X11 (the overlay is a fullscreen
    always-on-top all-workspaces window with no compositor-owned lifecycle, so
    an unclean session end can leave it wedged over every app); on for Windows
    and Linux Wayland (compositor owns the surface)."""
    val = _computer_use_cfg().get("no_overlay")
    if val is not None:
        return bool(val)
    if sys.platform == "darwin":
        return True
    if sys.platform != "linux":
        return False
    if not os.environ.get("DISPLAY"):
        return True
    try:
        with open("/proc/version", encoding="utf-8") as f:
            if "microsoft" in f.read().lower():
                return True
    except Exception:
        pass
    return os.environ.get("XDG_SESSION_TYPE") != "wayland" and not os.environ.get("WAYLAND_DISPLAY")

def _cua_telemetry_disabled() -> bool:
    """True unless ``computer_use.cua_telemetry`` opts in (unreadable config
    fails SAFE toward disabling telemetry)."""
    return not bool(_computer_use_cfg().get("cua_telemetry", False))

def _cua_configured_permission_mode() -> str:
    """``computer_use.permission_mode``: ``standard`` (default) or ``bounded``;
    unknown values fall closed to ``standard``. ``unrestricted`` is deliberately
    NOT a config value — it stays tied to the per-session YOLO toggle so a stale
    config line can never silently bypass approvals."""
    raw = str(_computer_use_cfg().get("permission_mode", "standard") or "").strip().lower()
    return raw if raw in {"standard", "bounded"} else "standard"

def _cua_capability_manifest() -> Optional[str]:
    """``computer_use.capability_manifest`` path, or None. Existence is
    validated by ``_EmbeddedCuaDaemon`` so a missing file fails loudly."""
    raw = _computer_use_cfg().get("capability_manifest")
    return raw.strip() if isinstance(raw, str) and raw.strip() else None

def _manifest_is_mode_independent(path: str) -> bool:
    """True when this manifest may accompany any permission mode: v1/v2 declare
    ``mode: bounded`` and abort startup under an unrestricted runtime; v3 has no
    mode and is the ceiling the driver accepts alongside any mode. Unreadable /
    unparseable -> False (forwarding one would turn a working session into a hard
    startup failure; bounded forwards unconditionally anyway)."""
    try:
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            parsed = yaml.safe_load(handle)
    except Exception:
        logger.debug("could not read capability manifest %s", path, exc_info=True)
        return False
    version = parsed.get("version") if isinstance(parsed, dict) else None
    return isinstance(version, int) and not isinstance(version, bool) and version >= 3

def _computer_use_max_image_dimension() -> Optional[int]:
    """``computer_use.max_image_dimension`` longest-edge cap (default 1456,
    matching the aux-vision downscale); ``0``/negative -> None (unset)."""
    try:
        dim = int(_computer_use_cfg().get("max_image_dimension", 1456))
    except (TypeError, ValueError):
        return 1456
    return dim if dim > 0 else None

def cua_driver_child_env(base_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Env for spawning cua-driver: ``base_env`` (default ``os.environ``) plus
    ``CUA_DRIVER_RS_TELEMETRY_ENABLED=0`` unless the user opted in. Used by
    every spawn site (MCP, status, doctor, install) so the policy is uniform."""
    env = dict(base_env if base_env is not None else os.environ)
    if _cua_telemetry_disabled():
        env[_CUA_TELEMETRY_ENV_VAR] = "0"
    return env

def sanitized_cua_driver_env() -> Dict[str, str]:
    """``cua_driver_child_env()`` with Hermes provider secrets stripped —
    cua-driver is a third-party binary and must never inherit API keys.
    Falls back to the unsanitized telemetry env if the sanitizer can't import."""
    env = cua_driver_child_env()
    try:
        from tools.environments.local import _sanitize_subprocess_env

        return _sanitize_subprocess_env(env)
    except Exception:
        return env

def _run_driver(driver_cmd: str, *args: str, timeout: float) -> subprocess.CompletedProcess:
    """Run a short cua-driver verb with the sanitized env, hidden window and
    stdin=DEVNULL (older drivers fall into a stdin-reading mode on unknown
    verbs; EOF makes them exit fast instead of blocking until the timeout)."""
    return subprocess.run(
        [driver_cmd, *args],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
        stdin=subprocess.DEVNULL,
        creationflags=windows_hide_flags(),
        env=sanitized_cua_driver_env(),
    )


# ---------------------------------------------------------------------------
# Linux display diagnostics
# ---------------------------------------------------------------------------

def _linux_session_locked() -> Optional[bool]:
    """Is the graphical session locked? (Linux; best-effort.) A locked KDE/GNOME
    session freezes renderers and half-disables the AX tree, so discovery
    legitimately returns nothing — which otherwise reads as a driver bug.
    True/False when loginctl answers, None when unavailable (non-Linux, no
    systemd-logind, probe failure)."""
    if sys.platform != "linux":
        return None

    def _loginctl(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(["loginctl", *args], capture_output=True, text=True,
                              timeout=2.0, stdin=subprocess.DEVNULL)

    try:
        proc = _loginctl("list-sessions", "--no-legend")
        if proc.returncode != 0:
            return None
        any_seat = False
        for line in proc.stdout.splitlines():
            parts = line.split()
            if len(parts) < 2 or "seat" not in line:
                continue
            any_seat = True
            if "LockedHint=no" in _loginctl("show-session", parts[0], "-p", "LockedHint").stdout:
                return False
        return True if any_seat else None
    except Exception:
        return None

def _empty_discovery_reason() -> str:
    """One-line diagnosis for 'window discovery found nothing'."""
    if _linux_session_locked() is True:
        return (
            "the desktop session is LOCKED (loginctl LockedHint=yes) — "
            "unlock the screen; a locked compositor hides windows and "
            "freezes app renderers"
        )
    if sys.platform == "linux" and not os.environ.get("DISPLAY"):
        return "no DISPLAY is set — X11/XWayland is not reachable from this process"
    if sys.platform == "darwin":
        # Headless Mac / asleep panel: ScreenCaptureKit has 0 shareable
        # displays while TCC grants look fine.
        return (
            "window discovery returned no windows; on macOS this usually "
            "means no shareable display (headless Mac or panel asleep) — "
            "wake the display or attach a monitor/HDMI dummy, then run "
            "`hermes computer-use doctor`"
        )
    return (
        "window discovery returned no windows; run `hermes computer-use "
        "doctor` (display reachability, AX capability)"
    )


# ---------------------------------------------------------------------------
# One-shot start() helpers: auto-repair + update nudge
# ---------------------------------------------------------------------------

_update_checked = False
# One auto-repair attempt per process: when the runtime-contract gate fails
# for something a reinstall fixes (old version, missing manifest verbs) run
# the standard install path once instead of telling the user to. Guarded so a
# failing installer can't loop — the second start() goes straight to the error.
_contract_repair_attempted = False

def _maybe_repair_runtime_contract(contract: Dict[str, Any]) -> Dict[str, Any]:
    """Try one automatic driver repair; return the post-repair contract (or the
    original when no repair was attempted / it failed). Never raises. An
    explicit ``HERMES_CUA_DRIVER_CMD`` override is authoritative even when
    broken, and a missing binary means installation was never requested."""
    global _contract_repair_attempted
    if (
        contract.get("ready")
        or _contract_repair_attempted
        or os.environ.get(_CUA_DRIVER_CMD_ENV, "").strip()
        or not contract.get("binary")
    ):
        return contract
    _contract_repair_attempted = True
    logger.info(
        "computer_use: installed cua-driver is not usable (%s); "
        "attempting automatic repair",
        contract.get("reason") or "runtime contract is incomplete",
    )
    try:
        from hermes_cli.tools_config import install_cua_driver

        if not install_cua_driver(upgrade=False, show_installer_progress=False):
            return contract
    except Exception as exc:
        logger.warning("computer_use: automatic cua-driver repair failed: %s", exc)
        return contract
    try:
        return cua_driver_runtime_contract_status()
    except Exception:
        return contract

def _maybe_nudge_update() -> None:
    """Emit an update nudge at most once per process, off-thread so the
    (cached, ~20h) GitHub poll never blocks the first computer_use action."""
    global _update_checked
    if _update_checked:
        return
    _update_checked = True

    def _run() -> None:
        try:
            msg = cua_driver_update_nudge()
        except Exception:
            return
        if msg:
            logger.info("computer_use: %s", msg)

    threading.Thread(target=_run, name="cua-driver-update-check", daemon=True).start()


# ---------------------------------------------------------------------------
# The backend itself
# ---------------------------------------------------------------------------

class CuaDriverBackend(_CaptureMixin, _InputMixin, ComputerUseBackend):
    """Default computer-use backend. Cross-platform via cua-driver MCP."""

    def __init__(self, permission_mode: str = "standard") -> None:
        if permission_mode not in {"standard", "bounded", "unrestricted"}:
            raise ValueError(f"unsupported cua-driver permission mode: {permission_mode}")
        self.permission_mode = permission_mode
        self._embedded_daemon: Optional[_EmbeddedCuaDaemon] = None
        if permission_mode != "standard":
            # The manifest is mandatory for bounded (the daemon validates it)
            # and optional for unrestricted, where it still caps what an
            # approval-bypassed run may touch.
            self._embedded_daemon = _EmbeddedCuaDaemon(
                resolve_cua_driver_cmd() or "",
                permission_mode,
                capability_manifest=_cua_capability_manifest(),
            )
        self._bridge = _AsyncBridge()
        self._session = _CuaDriverSession(self._bridge, self._embedded_daemon)
        # Sticky target — set by capture()/focus_app(), used by actions.
        self._active_pid: Optional[int] = None
        self._active_window_id: Optional[int] = None
        self._last_app: Optional[str] = None
        # Exact identity for capture_after: Linux app names may be generic
        # (several unrelated Qt windows can all say Qt6Application).
        self._last_target: Optional[Dict[str, Optional[int]]] = None
        # Per-snapshot `element_index -> element_token`; actions attach it so
        # cua-driver reports "stale" instead of silently re-resolving.
        self._snapshot_tokens: Dict[int, str] = {}
        # Public session label (one per Hermes run) sent as `session` on every
        # call: owns the cursor color and gives config/recording state a stable
        # owner across transport restarts. Part of the 0.20 runtime contract.
        self._session_id: str = f"hermes-{uuid.uuid4().hex[:12]}"
        self._session.set_transport_reset_callback(self._handle_transport_reset)

    def _handle_transport_reset(self) -> None:
        """Invalidate every capability minted by the replaced transport."""
        self._clear_active_target()

    # ── Lifecycle ──────────────────────────────────────────────────
    def start(self) -> None:
        contract = cua_driver_runtime_contract_status()
        if not contract.get("ready"):
            contract = _maybe_repair_runtime_contract(contract)
        if not contract.get("ready"):
            reason = contract.get("reason") or "runtime contract is incomplete"
            repair = (
                "Update the binary selected by HERMES_CUA_DRIVER_CMD or remove that override."
                if os.environ.get(_CUA_DRIVER_CMD_ENV, "").strip()
                else "Run `hermes computer-use install` to repair it."
            )
            raise RuntimeError(f"cua-driver is not ready: {reason}. {repair}")
        _maybe_nudge_update()
        # `mcp` is an optional extra: lazy-install on first use (gated by
        # `security.allow_lazy_installs`); failure raises FeatureUnavailable
        # with the exact `uv pip install` hint.
        from tools.lazy_deps import ensure as _lazy_ensure
        _lazy_ensure("tool.computer_use", prompt=False)
        import importlib
        importlib.invalidate_caches()  # a just-installed package may not be importable yet
        try:
            if self._embedded_daemon is not None:
                self._embedded_daemon.start()
            self._session.start()
        except Exception:
            if self._embedded_daemon is not None:
                self._embedded_daemon.stop()
            raise

        # Declare this run's identity. Non-fatal: cua-driver accepts anonymous
        # calls (the cursor just won't render), so degrade rather than abort.
        self._best_effort("start_session failed (continuing anonymous)",
                          self._session.call_tool, "start_session", {"session": self._session_id})

        # Post-handshake tuning guards on `_started`: before the handshake flips
        # it, call_tool would re-enter session.start() (stubbed start() recurses).
        if self._session._started:
            # Smaller screenshots cost less over the daemon socket and per turn.
            max_dim = _computer_use_max_image_dimension()
            if max_dim:
                self._best_effort("set_config(max_image_dimension) failed",
                                  self.set_config, max_image_dimension=max_dim)
            # Belt-and-suspenders when --no-overlay is unsupported or ignored.
            if _cua_no_overlay():
                self._best_effort("set_agent_cursor_enabled failed",
                                  self.set_agent_cursor_enabled, False, cursor_id=self._session_id)

    def stop(self) -> None:
        # Best-effort end_session so the driver cleans per-session state (cursor
        # overlay, recording ownership, config overrides); the connection drop
        # below releases daemon-side state regardless.
        if self._session._started:
            self._best_effort("end_session failed (continuing teardown)",
                              self._session.call_tool, "end_session", {"session": self._session_id})
        try:
            self._session.stop()
        finally:
            try:
                self._bridge.stop()
            finally:
                if self._embedded_daemon is not None:
                    self._embedded_daemon.stop()

    @staticmethod
    def _best_effort(what: str, fn, *args: Any, **kwargs: Any) -> None:
        """Run a non-fatal driver call, logging (debug) instead of raising."""
        try:
            fn(*args, **kwargs)
        except Exception as e:
            logger.debug("cua-driver %s: %s", what, e)

    def is_available(self) -> bool:
        # Other Unix-likes haven't been exercised end-to-end.
        return sys.platform in ("darwin", "win32", "linux") and cua_driver_binary_available()

    # ── Target state ───────────────────────────────────────────────
    def _clear_active_target(self) -> None:
        """Forget a capture/focus target so a failed lookup cannot misroute input."""
        self._active_pid = None
        self._active_window_id = None
        self._last_app = None
        self._last_target = None
        self._snapshot_tokens = {}

    def _set_active_target(self, target: Dict[str, Any]) -> None:
        self._active_pid = target["pid"]
        self._active_window_id = target["window_id"]
        # Tokens belong to the prior snapshot; disarm before any capture call
        # so an exception cannot pair old tokens with this target.
        self._snapshot_tokens = {}
        self._last_target = {"pid": self._active_pid, "window_id": self._active_window_id}

    # ── App lifecycle / focus ─────────────────────────────────────────
    def launch_app(
        self,
        *,
        bundle_id: Optional[str] = None,
        name: Optional[str] = None,
        urls: Optional[List[str]] = None,
        additional_arguments: Optional[List[str]] = None,
        creates_new_application_instance: bool = False,
    ) -> Dict[str, Any]:
        """Idempotent launch returning ``{pid, bundle_id, name, windows[]}``.
        ``creates_new_application_instance=True`` forces a fresh instance so
        concurrent runs touching the same app get isolated windows."""
        if not bundle_id and not name:
            raise ValueError("launch_app requires either bundle_id or name")
        args: Dict[str, Any] = {"session": self._session_id}
        for key, value in (("bundle_id", bundle_id), ("name", name), ("urls", urls and list(urls)),
                           ("additional_arguments", additional_arguments and list(additional_arguments)),
                           ("creates_new_application_instance", creates_new_application_instance or None)):
            if value:
                args[key] = value
        out = self._session.call_tool("launch_app", args)
        return out["structuredContent"] or {"data": out["data"]}

    def bring_to_front(self, *, pid: int, window_id: Optional[int] = None) -> ActionResult:
        """Activate a window so subsequent foreground-dispatched input lands on it."""
        args: Dict[str, Any] = {"pid": int(pid)}
        if window_id is not None:
            args["window_id"] = int(window_id)
        # The live schema is strict and has no session property: this is a
        # standalone native focus operation, not a session-scoped input action.
        return self._action("bring_to_front", args, inject_session=False)

    # ── Agent cursor / config ────────────────────────────────────────
    def set_agent_cursor_enabled(self, enabled: bool, *,
                                 cursor_id: Optional[str] = None) -> ActionResult:
        """Toggle the agent cursor overlay's visibility for this run."""
        args: Dict[str, Any] = {"enabled": bool(enabled)}
        if cursor_id:
            args["cursor_id"] = cursor_id
        return self._action("set_agent_cursor_enabled", args)

    def set_config(self, **config) -> ActionResult:
        """Set cua-driver config keys (e.g. ``max_image_dimension``). Unknown
        keys pass through verbatim — cua-driver validates its own schema."""
        return self._action("set_config", dict(config))

    def call_tool(self, name: str, args: Optional[Dict[str, Any]] = None,
                  *, timeout: float = 30.0) -> Dict[str, Any]:
        """Generic escape hatch: call any cua-driver MCP tool by name.
        ``session`` is injected via setdefault, so this is the supported path
        for tools the wrapper does not type-wrap (preferred over
        ``self._session.call_tool``)."""
        payload = dict(args) if args else {}
        payload.setdefault("session", self._session_id)
        return self._session.call_tool(name, payload, timeout=timeout)

    # ── Internal ───────────────────────────────────────────────────
    def _maybe_attach_element_token(self, tool: str, args: Dict[str, Any]) -> None:
        """Attach the snapshot's ``element_token`` to an ``element_index`` call so
        a superseded snapshot yields an explicit 'stale' error. Gated on the
        per-tool capability: older drivers (``additionalProperties: false``)
        must never see the field."""
        idx = args.get("element_index")
        token = self._snapshot_tokens.get(idx) if isinstance(idx, int) else None
        if token and self._session.supports_capability("accessibility.element_tokens", tool=tool):
            args["element_token"] = token

    def _action(self, name: str, args: Dict[str, Any], *, inject_session: bool = True) -> ActionResult:
        self._maybe_attach_element_token(name, args)
        # setdefault preserves any explicit session a caller already supplied.
        if inject_session:
            args.setdefault("session", self._session_id)
        try:
            out = self._session.call_tool(name, args)
        except Exception as e:
            logger.exception("cua-driver %s call failed", name)
            return ActionResult(ok=False, action=name, message=f"cua-driver error: {e}")
        data = out["data"]
        structured = out.get("structuredContent") or {}
        message = str(data.get("message", "")) if isinstance(data, dict) else data if isinstance(data, str) else ""
        if not message and isinstance(structured, dict):
            message = str(structured.get("message", ""))
        # Merge data + structuredContent into meta, structured winning on
        # overlap (it is the canonical verdict surface).
        meta: Dict[str, Any] = {}
        for part in (data, structured):
            if isinstance(part, dict):
                meta.update(part)
        return _action_result_from(name, not out["isError"], message, meta, structured,
                                   requested_delivery=args.get("delivery_mode"))
