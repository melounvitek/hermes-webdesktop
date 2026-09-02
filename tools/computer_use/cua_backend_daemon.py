"""Private embedded cua-driver daemon for non-standard permission modes, plus the macOS
CuaDriver.app identity checks its launch path depends on. Driver resolution / policy helpers
are looked up lazily through ``tools.computer_use.cua_backend`` so tests that patch them there
keep working.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from collections import deque
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("tools.computer_use.cua_backend")

# The only bundle identity the private daemon may launch through, and the teams that sign
# official releases. Exact matches only: a suffixed identifier or other team is an impostor.
_CUA_DRIVER_BUNDLE_ID = "com.trycua.driver"
_CUA_DRIVER_TEAM_IDS = ("4YEC26S9KF", "YCK386LBJ7")


def _resolve_cua_driver_app_path(driver_cmd: str) -> Optional[str]:
    """Return the CuaDriver.app bundle that CARRIES *driver_cmd*, if any. Derived from the
    resolved binary path only — no /Applications fallback, which could be a DIFFERENT install
    than the one the manifest resolved, running code the resolution chain never validated."""
    resolved = os.path.realpath(driver_cmd)
    marker_index = resolved.find(".app/Contents/MacOS/")
    if marker_index < 0:
        return None
    candidate = resolved[: marker_index + len(".app")]
    executable = os.path.join(candidate, "Contents", "MacOS", "cua-driver")
    return candidate if os.path.isfile(executable) and os.access(executable, os.X_OK) else None


def _validate_cua_driver_app_signature(app_path: str) -> None:
    """Fail closed unless *app_path* is the genuinely-signed CuaDriver.app. ``/usr/bin/open``
    hands LaunchServices whatever bundle sits at the path, so ``codesign -dv`` must report EXACTLY
    ``Identifier=com.trycua.driver`` and an expected TeamIdentifier. ``TeamIdentifier=not set``
    (ad-hoc dev builds) is allowed only with ``computer_use.allow_unsigned_driver: true``.
    Raises RuntimeError on any mismatch or when codesign is unavailable/fails."""
    from tools.computer_use import cua_backend as _cb

    codesign = shutil.which("codesign")
    if not codesign:
        raise RuntimeError("codesign is required to verify CuaDriver.app before launching it.")
    try:
        proc = subprocess.run([codesign, "-dv", app_path], capture_output=True, text=True, timeout=15)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeError(f"could not verify CuaDriver.app signature: {exc}") from exc
    if proc.returncode != 0:
        raise RuntimeError(f"CuaDriver.app at {app_path} is not code-signed; refusing to launch it "
                           f"({(proc.stderr or '').strip()})")
    fields: Dict[str, str] = {}
    for key, sep, value in (line.partition("=") for line in (proc.stderr or "").splitlines()):
        if sep:  # codesign -dv reports on stderr
            fields.setdefault(key.strip(), value.strip())
    identifier, team = fields.get("Identifier", ""), fields.get("TeamIdentifier", "")
    if identifier != _CUA_DRIVER_BUNDLE_ID:
        raise RuntimeError(f"CuaDriver.app at {app_path} has identifier {identifier!r}, "
                           f"expected {_CUA_DRIVER_BUNDLE_ID!r}; refusing to launch it.")
    if team in _CUA_DRIVER_TEAM_IDS or (
            team in ("", "not set") and _cb._computer_use_cfg().get("allow_unsigned_driver") is True):
        return
    raise RuntimeError(
        f"CuaDriver.app at {app_path} is signed by team {team!r}, expected one of "
        f"{_CUA_DRIVER_TEAM_IDS!r}; refusing to launch it. (Set computer_use.allow_unsigned_driver: "
        "true in config.yaml only for local unsigned driver builds.)")


def _embedded_daemon_spawn_command(driver_cmd: str, serve_args: List[str], *, platform: str,
                                   app_path: Optional[str] = None) -> List[str]:
    """Build the private-daemon launch while preserving macOS TCC identity."""
    if platform != "darwin":
        return [driver_cmd, *serve_args]
    resolved_app = app_path or _resolve_cua_driver_app_path(driver_cmd)
    if not resolved_app:
        raise RuntimeError("CuaDriver.app is required for private computer-use sessions on macOS. "
                           "Run `hermes computer-use install` to restore it.")
    _validate_cua_driver_app_signature(resolved_app)
    return ["/usr/bin/open", "-n", "-g", "-a", resolved_app, "--args", *serve_args]


def _wait_or_kill(process: Any) -> None:
    """Wait 5s for a graceful exit, then terminate (2s), then kill."""
    try:
        process.wait(timeout=5.0)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=2.0)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=2.0)


class _EmbeddedCuaDaemon:
    """Private daemon for a non-standard permission mode.

    cua-driver's permission mode is immutable after daemon startup, so reusing the
    machine-wide daemon would let one Hermes session's YOLO choice affect another. A private
    daemon gives the session its own socket, runtime and launch-time authorization; on macOS
    it is launched through CuaDriver.app so TCC stays attached to ``com.trycua.driver``.

    * ``unrestricted`` — explicit Hermes YOLO (``--dangerously-bypass-approvals``).
    * ``bounded`` — a user-reviewed capability manifest approved at launch is the
      authorization boundary, not a runtime prompt.

    The manifest is a ceiling, not a mode: it "can narrow a profile but never widen it", so a
    configured v3 manifest is forwarded even for ``unrestricted`` (bounding an approval-bypassed
    run). Mandatory for ``bounded``, optional everywhere else.
    """

    _START_TIMEOUT_SECONDS = 15.0

    def __init__(self, driver_cmd: str, permission_mode: str, capability_manifest: Optional[str] = None) -> None:
        from tools.computer_use import cua_backend as _cb

        if permission_mode not in {"unrestricted", "bounded"}:
            raise ValueError("embedded permission override supports unrestricted or bounded only")
        self.capability_manifest: Optional[str] = None
        manifest = str(capability_manifest or "").strip()
        if not manifest and permission_mode == "bounded":
            raise ValueError("bounded permission mode requires computer_use.capability_manifest")
        if manifest:
            manifest = os.path.abspath(os.path.expanduser(manifest))
            if not os.path.isfile(manifest):
                raise ValueError(f"capability manifest not found: {manifest}")
            self.capability_manifest = manifest
        # bounded always forwards (the driver validates it); other modes accept only a v3
        # manifest — a legacy one would abort startup instead.
        self.manifest_applies = bool(self.capability_manifest) and (
            permission_mode == "bounded" or _cb._manifest_is_mode_independent(str(self.capability_manifest)))
        if self.capability_manifest and not self.manifest_applies:
            logger.warning("computer_use.capability_manifest is a legacy (v1/v2) manifest, "
                           "which cua-driver only accepts in bounded mode — it will NOT "
                           "bound this %s session. Migrate the manifest to version 3 to "
                           "keep a ceiling on approval-bypassed runs.", permission_mode)
        self.permission_mode = permission_mode
        self._driver_cmd = self._command = driver_cmd
        self._mcp_args: List[str] = list(_cb._CUA_DRIVER_ARGS)
        self._process: Any = None
        self._owns_runtime = self._running = self._launch_via_app = False
        self._stderr_tail: deque[str] = deque(maxlen=20)
        self._stderr_thread: Optional[threading.Thread] = None
        token = uuid.uuid4().hex[:12]
        self.socket_path = (rf"\\.\pipe\hermes-cua-{token}" if sys.platform == "win32"
                            else os.path.join(tempfile.gettempdir(), f"hc-{token}.sock"))

    def child_env(self) -> Dict[str, str]:
        from tools.computer_use import cua_backend as _cb

        env = _cb.cua_driver_child_env()
        env["CUA_DRIVER_PERMISSION_MODE"] = self.permission_mode
        if self.permission_mode == "unrestricted":
            env["CUA_DRIVER_DANGEROUSLY_BYPASS_APPROVALS"] = "1"
        return env

    def _drain_stderr(self, process: Any) -> None:
        try:
            for line in getattr(process, "stderr", None) or ():
                text = str(line).strip()
                if text:
                    self._stderr_tail.append(text)
                    logger.debug("embedded cua-driver: %s", text)
        except Exception:
            pass

    def _serve_args(self) -> List[str]:
        from tools.computer_use import cua_backend as _cb

        serve_args = ["serve", "--embedded", "--socket", self.socket_path,
                      "--no-permissions-gate", "--permission-mode", self.permission_mode]
        if self.permission_mode == "unrestricted":
            serve_args.append("--dangerously-bypass-approvals")
        if self.manifest_applies:
            serve_args += ["--capability-manifest", str(self.capability_manifest), "--approve-capability-manifest"]
        # The private daemon owns the cursor overlay, so the overlay policy must apply to this
        # long-lived serve process, not only its MCP proxy. Appended BEFORE the macOS app-launch
        # wrapping so the flag travels inside `open ... --args` with the rest of the serve args.
        return _cb._mcp_args_with_overlay_flag(serve_args, driver_cmd=self._command)

    def start(self) -> None:
        if self._running:
            return
        from tools.computer_use import cua_backend as _cb
        from tools.environments.local import _sanitize_subprocess_env

        self._driver_cmd = self._driver_cmd or _cb.resolve_cua_driver_cmd() or ""
        if not self._driver_cmd:
            raise RuntimeError(_cb.cua_driver_install_hint())
        self._command, self._mcp_args = _cb._resolve_mcp_invocation(self._driver_cmd)
        env = _sanitize_subprocess_env(self.child_env())
        self._launch_via_app = sys.platform == "darwin"
        command = _embedded_daemon_spawn_command(self._command, self._serve_args(), platform=sys.platform)
        self._process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                         stderr=subprocess.PIPE, text=True, env=env)
        self._owns_runtime = True
        self._stderr_thread = threading.Thread(target=self._drain_stderr, args=(self._process,),
                                               name="hermes-cua-daemon-stderr", daemon=True)
        self._stderr_thread.start()
        deadline = time.monotonic() + self._START_TIMEOUT_SECONDS
        while time.monotonic() < deadline:
            return_code = self._process.poll()
            # `open` exits 0 once LaunchServices took the request, so on macOS only a
            # non-zero exit means the daemon itself died.
            if return_code is not None and (not self._launch_via_app or return_code != 0):
                detail = "; ".join(self._stderr_tail) or "no diagnostic output"
                raise RuntimeError(f"embedded cua-driver exited during startup: {detail}")
            if self._socket_ready(env):
                self._running = True
                return
            time.sleep(0.1)
        self.stop()
        detail = "; ".join(self._stderr_tail) or "daemon did not become ready"
        raise RuntimeError(f"embedded cua-driver startup timed out: {detail}")

    def _socket_ready(self, env: Dict[str, str]) -> bool:
        """``cua-driver status --socket`` exits 0 once the private daemon accepts connections."""
        try:
            probe = subprocess.run([self._command, "status", "--socket", self.socket_path],
                                   stdin=subprocess.DEVNULL, capture_output=True, text=True,
                                   timeout=2.0, env=env)
        except (OSError, subprocess.SubprocessError):
            return False
        return probe.returncode == 0

    def proxy_invocation(self) -> Tuple[str, List[str]]:
        if not self._running:
            raise RuntimeError("embedded cua-driver daemon is not running")
        return self._command, [*self._mcp_args, "--embedded", "--socket", self.socket_path]

    def stop(self) -> None:
        process, self._process = self._process, None
        owns_runtime, self._owns_runtime = self._owns_runtime, False
        self._running = False
        if owns_runtime:
            from tools.environments.local import _sanitize_subprocess_env
            try:
                subprocess.run([self._command, "stop", "--socket", self.socket_path],
                               stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=3.0,
                               env=_sanitize_subprocess_env(self.child_env()))
            except (OSError, subprocess.SubprocessError):
                pass
        if process is not None:
            _wait_or_kill(process)
        if sys.platform != "win32" and os.path.exists(self.socket_path):
            try:
                os.remove(self.socket_path)
            except OSError:
                pass
