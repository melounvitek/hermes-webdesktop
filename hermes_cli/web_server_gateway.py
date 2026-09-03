"""Gateway/process helpers for the dashboard: per-profile gateway topology (+cache), action subprocess spawning, gateway restart plumbing, system platform display.

Split out of ``hermes_cli.web_server``; every externally used name is re-imported
there, so ``web_server.<name>`` keeps resolving (and monkeypatching) as before.
Helpers that tests patch on ``web_server`` are reached lazily through it.
"""

import logging
import json
import os
import re
import subprocess
import sys
import threading
import time
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from hermes_cli._subprocess_compat import windows_detach_flags
from hermes_cli.config import get_hermes_home

# Same logger the code used before extraction (record parity).
_log = logging.getLogger("hermes_cli.web_server")


# DEPRECATED (scheduled for removal): GATEWAY_HEALTH_URL / GATEWAY_HEALTH_TIMEOUT.
# Cross-container / cross-host gateway liveness detection will be folded into a
# first-class dashboard config key so it's no longer Docker-adjacent lore buried
# in env vars.  The env vars still work for now so existing Compose deployments
# don't break.  Do not add new callers — wire new uses through the planned
# config surface.


def _probe_gateway_health() -> tuple[bool, dict | None]:
    """Probe the gateway via its HTTP health endpoint (cross-container).

    .. deprecated::
        Driven by the deprecated ``GATEWAY_HEALTH_URL`` /
        ``GATEWAY_HEALTH_TIMEOUT`` env vars.  Scheduled for removal alongside
        a move to a first-class dashboard config key.  See
        :data:`_GATEWAY_HEALTH_URL` for context.

    Uses ``/health/detailed`` first (returns full state), falling back to
    the simpler ``/health`` endpoint.  Returns ``(is_alive, body_dict)``.

    Accepts any of these as ``GATEWAY_HEALTH_URL``:
    - ``http://gateway:8642``                (base URL — recommended)
    - ``http://gateway:8642/health``         (explicit health path)
    - ``http://gateway:8642/health/detailed`` (explicit detailed path)

    This is a **blocking** call — run via ``run_in_executor`` from async code.
    """
    from hermes_cli.web_server import _GATEWAY_HEALTH_TIMEOUT, _GATEWAY_HEALTH_URL
    if not _GATEWAY_HEALTH_URL:
        return False, None

    # Normalise to base URL so we always probe the right paths regardless of
    # whether the user included /health or /health/detailed in the env var.
    base = _GATEWAY_HEALTH_URL.rstrip("/")
    if base.endswith("/health/detailed"):
        base = base[: -len("/health/detailed")]
    elif base.endswith("/health"):
        base = base[: -len("/health")]

    for path in (f"{base}/health/detailed", f"{base}/health"):
        try:
            req = urllib.request.Request(path, method="GET")
            with urllib.request.urlopen(req, timeout=_GATEWAY_HEALTH_TIMEOUT) as resp:
                if resp.status == 200:
                    body = json.loads(resp.read())
                    return True, body
        except Exception:
            continue
    return False, None


# Host TCP ports each port-binding gateway platform listens on, as
# ``platform-name -> (config port key, adapter default)``.  Mirrors
# ``PORT_BINDING_PLATFORM_VALUES`` in gateway/config.py and each adapter's
# DEFAULT_PORT / DEFAULT_WEBHOOK_PORT constant.  Used only for the dashboard's
# gateway-topology readout — best-effort display data, not a bind source.
_PORT_BINDING_PLATFORM_PORTS: Dict[str, Tuple[str, int]] = {
    "webhook": ("port", 8644),
    "api_server": ("port", 8642),
    "msgraph_webhook": ("port", 8646),
    "feishu": ("webhook_port", 8765),
    "wecom_callback": ("port", 8645),
    "bluebubbles": ("webhook_port", 8645),
    "sms": ("webhook_port", 8080),
    "whatsapp_cloud": ("webhook_port", 8090),
    "line": ("port", 8646),
    "teams": ("port", 3978),
}

# Platform states that mean the adapter is NOT serving its port right now.
_PLATFORM_DEAD_STATES = frozenset({"fatal", "disconnected", "stopped"})


def _profile_platform_ports(profile_home: Path, runtime: Optional[dict]) -> Dict[str, int]:
    """Best-effort map of ``platform -> host TCP port`` for one profile's gateway.

    Reads the platforms the running gateway reported in its
    ``gateway_state.json`` and resolves each port-binding platform's port from
    the profile's ``config.yaml`` (top-level ``platforms:`` wins over
    ``gateway.platforms:``, matching ``load_gateway_config`` precedence),
    falling back to the adapter default.  Display-only: env-var port overrides
    (e.g. ``WEBHOOK_PORT`` in that profile's .env) are not resolved here.
    """
    platforms = (runtime or {}).get("platforms") or {}
    active = [
        name for name, state in platforms.items()
        if name in _PORT_BINDING_PLATFORM_PORTS
        and isinstance(state, dict)
        and state.get("state") not in _PLATFORM_DEAD_STATES
    ]
    if not active:
        return {}

    blocks: Dict[str, dict] = {}
    try:
        # Multi-profile probe: load_config() targets the ACTIVE profile's
        # home, so read the probed profile's file via the raw primitive.
        from hermes_cli.config import read_user_config_raw
        cfg = read_user_config_raw(profile_home / "config.yaml")
        gateway_cfg = cfg.get("gateway") if isinstance(cfg.get("gateway"), dict) else {}
        # gateway.platforms first, top-level platforms second — later wins,
        # matching the precedence in gateway.config.load_gateway_config().
        for src in ((gateway_cfg or {}).get("platforms"), cfg.get("platforms")):
            if not isinstance(src, dict):
                continue
            for plat_name, plat_block in src.items():
                if isinstance(plat_block, dict):
                    blocks.setdefault(plat_name, {}).update(plat_block)
    except Exception:
        blocks = {}

    ports: Dict[str, int] = {}
    for name in active:
        port_key, default_port = _PORT_BINDING_PLATFORM_PORTS[name]
        block = blocks.get(name) or {}
        extra = block.get("extra") if isinstance(block.get("extra"), dict) else {}
        raw = block.get(port_key, (extra or {}).get(port_key, default_port))
        try:
            ports[name] = int(raw)
        except (TypeError, ValueError):
            ports[name] = default_port
    return ports


def _profile_gateway_writer_identity(
    profile_home: Path, runtime: Optional[dict]
) -> Optional[tuple]:
    """``(pid, start_time)`` identity of the profile's LIVE gateway, or None.

    Reuses the validated-liveness helper — recorded PID checked against the
    live process table, the start-time PID-reuse fingerprint, and the
    profile's home — then reads the live process's fingerprint via the same
    ``_get_process_start_time`` that stamped it, so equality is exact (no
    unit or clock-source mismatch).  None when the record doesn't belong to
    a live gateway; nothing in it is current by definition then.
    """
    try:
        from gateway.status import (
            _get_process_start_time,
            get_runtime_status_running_pid,
        )

        pid = get_runtime_status_running_pid(runtime, expected_home=profile_home)
        if pid is None:
            return None
        start_time = _get_process_start_time(pid)
        if start_time is None:
            return None
        return (pid, start_time)
    except Exception:
        return None


def _owned_profile_platforms(
    writer_identity: Optional[tuple], platforms: dict
) -> dict:
    """Keep only platform entries the profile's CURRENT process wrote.

    Gateway startup deliberately preserves plain platform entries in
    ``gateway_state.json`` across restarts (the dashboard keeps showing
    last-known state while adapters reconnect), and the active-profile
    endpoint compensates by filtering them against the current
    configuration.  The cross-profile aggregation has no equivalent config
    context (a profile's platform set depends on tokens in that profile's
    ``.env`` behind its secret scope), so it demands strict process
    ownership instead: ``write_runtime_status`` stamps every platform write
    with the writer's ``(pid, start_time)`` identity, and an entry is
    aggregatable only when that identity equals the profile's live gateway
    process — exact match, no clock heuristics, so an entry written moments
    before a fast restart can never masquerade as current.  A fatal entry
    left behind by a platform the operator has since disabled/removed thus
    stops degrading fleet health as soon as that profile's gateway restarts
    (a config change requires that restart to take effect anyway).  Fail
    closed: entries without a writer identity (legacy records) or records
    with no live process are excluded — aggregation is a supplement, and a
    false "degraded forever" is the worse failure mode.
    """
    if writer_identity is None:
        return {}
    live_pid, live_start = writer_identity
    owned: Dict[str, dict] = {}
    for key, value in platforms.items():
        if not isinstance(value, dict):
            continue
        if (
            value.get("writer_pid") == live_pid
            and value.get("writer_start_time") == live_start
        ):
            owned[key] = value
    return owned


def _collect_profile_gateway_topology() -> Dict[str, Any]:
    """Enumerate profiles and the gateways serving them for ``/api/status``.

    Returns ``{"profiles": [...], "gateway_mode": ..., "gateways": [...]}``:

    * ``profiles`` — every profile on the host (default + named), from
      ``profiles_to_serve(True)`` (the cheap enumeration chokepoint — no
      per-profile config reads or skill counts).
    * ``gateways`` — one entry per profile with a LIVE gateway process:
      ``{"profile", "ports", "served_profiles"?}``.  Liveness reuses
      ``_check_gateway_running`` so this agrees with the profiles sidebar.
    * ``gateway_mode`` — ``"multiplex"`` when the default gateway serves
      multiple profiles (gateway.multiplex_profiles), ``"single"`` for one
      live gateway, ``"multiple"`` for independent per-profile gateways,
      ``"none"`` when nothing is running.
    * ``profile_platforms`` — ``{profile: platforms}`` runtime platform maps
      for each LIVE gateway, ownership-filtered to entries stamped by that
      profile's current process (stale preserved entries for since-removed
      platforms are excluded — see ``_owned_profile_platforms``).  Internal
      aggregation input for ``/api/status`` (independent per-profile gateways
      write failures to their own ``gateway_state.json``, which the
      unparameterized endpoint would otherwise never see).  Never exposed
      directly.
    """
    from hermes_cli.web_server import _profile_gateway_writer_identity
    try:
        from hermes_cli.profiles import _check_gateway_running, profiles_to_serve
        from gateway.status import read_runtime_status
        homes = profiles_to_serve(True)
    except Exception:
        _log.debug("profile/gateway topology enumeration failed", exc_info=True)
        return {
            "profiles": [],
            "gateway_mode": "unknown",
            "gateways": [],
            "profile_platforms": {},
        }

    profile_names = [name for name, _home in homes]
    gateways: List[Dict[str, Any]] = []
    profile_platforms: Dict[str, dict] = {}
    multiplex = False
    for name, home in homes:
        try:
            if not _check_gateway_running(home):
                continue
        except Exception:
            continue
        try:
            runtime = read_runtime_status(home / "gateway_state.json")
        except Exception:
            runtime = None
        served = [str(p) for p in ((runtime or {}).get("served_profiles") or [])]
        if name == "default" and len(served) > 1:
            multiplex = True
        plats = (runtime or {}).get("platforms")
        if isinstance(plats, dict) and plats:
            # Ownership filter: gateway startup preserves plain platform
            # entries across restarts, so the raw map can carry fatal state
            # for platforms the operator has since disabled/removed.  Only
            # entries stamped with the profile's current live process's
            # writer identity are aggregation candidates (see
            # _owned_profile_platforms).
            owned = _owned_profile_platforms(
                _profile_gateway_writer_identity(home, runtime), plats
            )
            if owned:
                profile_platforms[name] = owned
        entry: Dict[str, Any] = {
            "profile": name,
            "ports": _profile_platform_ports(home, runtime),
        }
        if served:
            entry["served_profiles"] = served
        gateways.append(entry)

    if multiplex:
        mode = "multiplex"
    elif len(gateways) > 1:
        mode = "multiple"
    elif len(gateways) == 1:
        mode = "single"
    else:
        mode = "none"

    return {
        "profiles": profile_names,
        "gateway_mode": mode,
        "gateways": gateways,
        "profile_platforms": profile_platforms,
    }


# /api/status is polled ~1/s by the desktop app while it waits for the backend
# (and again by the dashboard badge). Each uncached call above walks 7+ profile
# homes (yaml.safe_load with the pure-Python loader + psutil process-table
# probes + realpath walks) inside the default executor; concurrent polls pile
# up and hold the GIL for 14-16s, starving the event loop — the desktop WS
# never receives gateway.ready and boot fails ("event loop stalled ... GIL
# pressure suspected"). Topology changes on gateway start/stop, so a short TTL
# cache with a collapse lock keeps the scan to one per window. The cache also
# remembers which collector produced the entry: tests monkeypatch
# _collect_profile_gateway_topology per case, and the identity check keeps
# them hermetic without needing a reset hook (a swapped collector is a miss).
_TOPOLOGY_CACHE: Dict[str, Any] = {"ts": 0.0, "data": None, "fn": None}
_TOPOLOGY_CACHE_LOCK = threading.Lock()
_TOPOLOGY_CACHE_TTL = 10.0


def _topology_cache_get(fn: Any) -> Optional[Dict[str, Any]]:
    if (
        _TOPOLOGY_CACHE["data"] is not None
        and _TOPOLOGY_CACHE["fn"] is fn
        and time.monotonic() - _TOPOLOGY_CACHE["ts"] < _TOPOLOGY_CACHE_TTL
    ):
        return _TOPOLOGY_CACHE["data"]
    return None


def _collect_profile_gateway_topology_cached() -> Dict[str, Any]:
    from hermes_cli.web_server import _collect_profile_gateway_topology
    fn = _collect_profile_gateway_topology
    cached = _topology_cache_get(fn)
    if cached is not None:
        return cached
    with _TOPOLOGY_CACHE_LOCK:
        cached = _topology_cache_get(fn)
        if cached is not None:
            return cached
        data = fn()
        _TOPOLOGY_CACHE["data"] = data
        _TOPOLOGY_CACHE["fn"] = fn
        _TOPOLOGY_CACHE["ts"] = time.monotonic()
        return data


def _load_configured_gateway_platforms() -> set[str]:
    """Load connected platform names away from the asyncio event loop.

    The first ``load_gateway_config()`` call performs platform discovery and
    can take longer than Desktop's WebSocket connect timeout on Windows.  This
    helper is synchronous by design; ``get_status`` runs it in Starlette's
    worker pool so a concurrent ``/api/ws`` handshake can still complete.
    """
    from gateway.config import load_gateway_config

    gateway_config = load_gateway_config()
    return {platform.value for platform in gateway_config.get_connected_platforms()}


_WINDOWS_11_MIN_BUILD = 22000


def _windows_build_number(version: str, platform_label: str) -> Optional[int]:
    """Extract the Windows NT build number from stdlib platform strings."""
    for value in (version or "", platform_label or ""):
        match = re.search(r"(?:^|[^\d])10\.0\.(\d{5,})(?:[^\d]|$)", value)
        if not match:
            continue
        try:
            return int(match.group(1))
        except ValueError:
            continue
    return None


def _display_system_platform(
    *,
    system: str,
    release: str,
    version: str,
    platform_label: str,
) -> Dict[str, str]:
    """Return host OS fields for display while preserving stdlib detail."""
    if system == "Windows" and release == "10":
        build = _windows_build_number(version, platform_label)
        if build is not None and build >= _WINDOWS_11_MIN_BUILD:
            platform_label = re.sub(
                r"^Windows-10(?=-)",
                "Windows-11",
                platform_label,
                count=1,
            )
            release = "11"

    return {
        "os": system,
        "os_release": release,
        "os_version": version,
        "platform": platform_label,
    }


# ---------------------------------------------------------------------------
# Gateway + update actions (invoked from the Status page).
#
# Both commands are spawned as detached subprocesses so the HTTP request
# returns immediately.  stdin is closed (``DEVNULL``) so any stray ``input()``
# calls fail fast with EOF rather than hanging forever.  stdout/stderr are
# streamed to a per-action log file under ``~/.hermes/logs/<action>.log`` so
# the dashboard can tail them back to the user.
# ---------------------------------------------------------------------------

_ACTION_LOG_DIR: Path = get_hermes_home() / "logs"

# Short ``name`` (from the URL) → absolute log file path.
_ACTION_LOG_FILES: Dict[str, str] = {
    "gateway-restart": "gateway-restart.log",
    "gateway-start": "gateway-start.log",
    "gateway-stop": "gateway-stop.log",
    "hermes-update": "hermes-update.log",
    "doctor": "action-doctor.log",
    "security-audit": "action-security-audit.log",
    "backup": "action-backup.log",
    "import": "action-import.log",
    "checkpoints-prune": "action-checkpoints-prune.log",
    "skills-install": "action-skills-install.log",
    "skills-uninstall": "action-skills-uninstall.log",
    "skills-update": "action-skills-update.log",
    "curator-run": "action-curator-run.log",
    "prompt-size": "action-prompt-size.log",
    "dump": "action-dump.log",
    "config-migrate": "action-config-migrate.log",
    "tools-post-setup": "action-tools-post-setup.log",
}

# ``name`` → most recently spawned Popen handle.  Used so ``status`` can
# report liveness and exit code without shelling out to ``ps``.
_ACTION_PROCS: Dict[str, subprocess.Popen] = {}
_ACTION_COMMANDS: Dict[str, Tuple[str, ...]] = {}
_ACTION_IDS: Dict[str, str] = {}

# ``name`` → completed synthetic action result for actions the server handled
# without spawning a subprocess (for example, unsupported Docker updates).
_ACTION_RESULTS: Dict[str, Dict[str, Any]] = {}


def _terminate_desktop_managed_gateway() -> None:
    """Stop a live gateway restart child when its Desktop backend shuts down."""
    from hermes_cli.web_server import _ACTION_PROCS
    proc = _ACTION_PROCS.get("gateway-restart")
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
    except OSError:
        # The child may have exited between poll() and terminate().
        pass


def _dashboard_spawn_executable() -> str:
    """Interpreter for detached dashboard actions.

    Prefers the install's own venv interpreter over ``sys.executable`` when
    they differ. Under an SSH remote backend the web server is launched by
    running the **uv base interpreter** with the venv's site-packages
    injected into ``sys.path`` at startup (``-c "sys.path[:0]=[...];
    runpy.run_module('hermes_cli.main', ...)"``) — so ``sys.executable`` is
    the dependency-less base python and a detached action spawned from it
    dies on the first third-party import (``ModuleNotFoundError: yaml``),
    because the injected path is a startup artifact of the parent and is
    not inherited (#90026). The venv launcher resolves the same dependency
    set on its own.

    Falls back to ``sys.executable`` when no venv interpreter exists next
    to the install (in-process dev runs, exotic layouts). On Windows the
    spawn below carries ``windows_detach_flags()`` (CREATE_NO_WINDOW), so
    the console python owns a single hidden console that its own subprocess
    spawns inherit — the action stays invisible without resorting to
    console-less pythonw.exe, which would make every console-subsystem
    descendant flash its own conhost (#54220/#56747).
    """
    from hermes_cli.web_server import PROJECT_ROOT
    exe = Path(sys.executable)
    try:
        for rel in ("venv/bin/python", "venv/Scripts/python.exe"):
            candidate = PROJECT_ROOT / rel
            if candidate.is_file():
                # Same interpreter → keep sys.executable (preserves the
                # docstring's console-ownership behavior verbatim). Compare
                # UNRESOLVED normalized paths: a venv's bin/python is
                # typically a SYMLINK to the base interpreter, so resolving
                # both sides makes the venv python and the dependency-less
                # base compare equal — exactly the SSH-runtime case this
                # function exists to fix. The unresolved path IS the venv's
                # identity (pyvenv.cfg discovery keys off argv0's location).
                if os.path.normcase(os.path.normpath(str(candidate))) == (
                    os.path.normcase(os.path.normpath(str(exe)))
                ):
                    return sys.executable
                # Return the candidate UNRESOLVED for the same reason:
                # invoking the resolved target would bypass pyvenv.cfg and
                # run the bare base interpreter again.
                return str(candidate)
    except OSError:
        pass
    return sys.executable


def _spawn_hermes_action(
    subcommand: List[str],
    name: str,
    *,
    env_overrides: Optional[Dict[str, str]] = None,
) -> subprocess.Popen:
    """Spawn ``hermes <subcommand>`` detached and record the Popen handle.

    Uses the running interpreter's ``hermes_cli.main`` module so the action
    inherits the same venv/PYTHONPATH the web server is using.
    """
    from hermes_cli.web_server import (
        PROJECT_ROOT,
        _ACTION_COMMANDS,
        _ACTION_IDS,
        _ACTION_LOG_DIR,
        _ACTION_PROCS,
        _ACTION_RESULTS,
    )
    log_file_name = _ACTION_LOG_FILES[name]
    _ACTION_LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = _ACTION_LOG_DIR / log_file_name
    log_file = open(log_path, "ab", buffering=0)
    log_file.write(
        f"\n=== {name} started {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n".encode()
    )

    cmd = [_dashboard_spawn_executable(), "-m", "hermes_cli.main", *subcommand]

    # The dashboard runs *inside* the gateway process, so os.environ carries
    # _HERMES_GATEWAY=1. Inheriting it makes a spawned `hermes gateway restart`
    # trip the in-process restart-loop guard and exit 1 — silently failing the
    # dashboard's auto-restart paths. The gateway's own restart watcher already
    # drops it (gateway/run.py); mirror that here (#52470).
    action_env = {**os.environ, "HERMES_NONINTERACTIVE": "1"}
    action_env.pop("_HERMES_GATEWAY", None)

    popen_kwargs: Dict[str, Any] = {
        "cwd": str(PROJECT_ROOT),
        "stdin": subprocess.DEVNULL,
        "stdout": log_file,
        "stderr": subprocess.STDOUT,
        "env": {**action_env, **(env_overrides or {})},
    }
    if sys.platform == "win32":
        popen_kwargs["creationflags"] = windows_detach_flags()
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(cmd, **popen_kwargs)
    # The child inherits its own duplicated fd for stdout/stderr, so the
    # parent's handle can be released immediately — otherwise we leak one
    # fd per spawned action.
    log_file.close()
    _ACTION_RESULTS.pop(name, None)
    _ACTION_COMMANDS[name] = tuple(subcommand)
    _ACTION_PROCS[name] = proc
    action_id = (env_overrides or {}).get("HERMES_ACTION_ID")
    if action_id:
        _ACTION_IDS[name] = action_id
    else:
        _ACTION_IDS.pop(name, None)
    return proc


def _gateway_subcommand(profile: Optional[str], verb: str) -> List[str]:
    from hermes_cli.web_server import _profile_cli_args
    return _profile_cli_args(profile) + ["gateway", verb]


def _restart_gateway_after(profile: Optional[str], *, what: str, label: str) -> dict[str, Any]:
    """Best-effort gateway restart after a config change (webhooks, onboarding).

    The config save stays authoritative; a failed spawn is reported in the
    result (``restart_started: False`` + ``restart_error``) so the UI can fall
    back to its manual restart banner instead of failing the request.
    """
    from hermes_cli.web_server import _spawn_gateway_restart
    try:
        proc, reused = _spawn_gateway_restart(profile)
    except Exception as exc:
        _log.exception("Failed to auto-restart gateway after %s", what)
        return {"restart_started": False, "restart_error": str(exc)}
    if reused:
        _log.info("%s: reusing in-flight gateway restart (pid %s)", label, proc.pid)
    return {"restart_started": True, "restart_action": "gateway-restart", "restart_pid": proc.pid}


def _split_text_for_speak_stream(text: str, cap: int) -> list:
    """Split *text* into provider-cap-sized pieces on sentence boundaries.

    Deliberately NOT unified with gateway.platforms.helpers'
    split_text_fence_aware: this splitter reflows whitespace (sentences are
    re-joined with single spaces) and has no fence/markdown semantics, so
    expressing it as knobs on the fence-aware core would change behavior.
    """
    from tools.tts_streaming import SENTENCE_BOUNDARY_RE as _SENTENCE_BOUNDARY_RE

    cap = cap if cap and cap > 0 else 4000
    pieces, buf = [], ""
    for sentence in filter(str.strip, _SENTENCE_BOUNDARY_RE.split(text)):
        while len(sentence) > cap:
            pieces.append(sentence[:cap])
            sentence = sentence[cap:]
        if buf and len(buf) + len(sentence) + 1 > cap:
            pieces.append(buf)
            buf = sentence
        else:
            buf = f"{buf} {sentence}" if buf else sentence
    if buf:
        pieces.append(buf)
    return pieces


# Per-row fields that no session LIST consumer reads but that dominate the
# payload. ``system_prompt`` is the fully rendered prompt — tens of KB per
# row — and made a 21-row /api/sessions response 528KB (96% dead weight),
# re-fetched by the desktop sidebar on every refresh. The desktop's
# SessionInfo type doesn't declare either field and the web UI never touches
# them; ``GET /api/sessions/{id}`` detail reads stay complete. List callers
# that genuinely need the full rows can pass ``?full=1``.
_SESSION_LIST_HEAVY_FIELDS = ("system_prompt", "model_config")


def _strip_session_list_rows(sessions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    for s in sessions:
        for key in _SESSION_LIST_HEAVY_FIELDS:
            s.pop(key, None)
    return sessions
