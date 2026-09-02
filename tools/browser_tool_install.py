"""agent-browser / Chromium discovery and install: PATH merging, npx resolution, candidate binaries, Chromium detection + auto-install, requirement checks.

Split out of ``tools/browser_tool.py``; every name is re-imported there so
``tools.browser_tool.<name>`` keeps resolving (and monkeypatching). Origin
symbols and module state are read/written through ``_bt`` (the origin module,
resolved per call by :func:`tools.browser_tool_origin.origin_module`) so
``patch("tools.browser_tool.X")`` is honoured and no import cycle exists.
"""

import functools
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from tools.browser_tool_origin import origin_module as _origin


@functools.lru_cache(maxsize=1)
def _discover_homebrew_node_dirs() -> tuple[str, ...]:
    """Find Homebrew versioned Node.js bin directories (e.g. node@20, node@24).

    When Node is installed via ``brew install node@24`` and NOT linked into
    /opt/homebrew/bin, agent-browser isn't discoverable on the default PATH.
    This function finds those directories so they can be prepended.
    """
    dirs: list[str] = []
    homebrew_opt = "/opt/homebrew/opt"
    if not os.path.isdir(homebrew_opt):
        return tuple(dirs)
    try:
        for entry in os.listdir(homebrew_opt):
            if entry.startswith("node") and entry != "node":
                bin_dir = os.path.join(homebrew_opt, entry, "bin")
                if os.path.isdir(bin_dir):
                    dirs.append(bin_dir)
    except OSError:
        pass
    return tuple(dirs)


def _browser_candidate_path_dirs() -> list[str]:
    """Return ordered browser CLI PATH candidates shared by discovery and execution."""
    _bt = _origin()
    hermes_home = _bt.get_hermes_home()
    hermes_node_bin = str(hermes_home / "node" / "bin")
    hermes_node_root = str(hermes_home / "node")
    hermes_nm_bin = str(hermes_home / "node_modules" / ".bin")
    return [hermes_node_bin, hermes_node_root, hermes_nm_bin, *list(_bt._discover_homebrew_node_dirs()), *_bt._SANE_PATH_DIRS]


def _merge_browser_path(existing_path: str = "") -> str:
    """Prepend browser-specific PATH fallbacks without reordering existing entries."""
    _bt = _origin()
    path_parts = [p for p in (existing_path or "").split(os.pathsep) if p]
    existing_parts = set(path_parts)
    prefix_parts: list[str] = []

    for part in _bt._browser_candidate_path_dirs():
        if not part or part in existing_parts or part in prefix_parts:
            continue
        if os.path.isdir(part):
            prefix_parts.append(part)

    return os.pathsep.join(prefix_parts + path_parts)


def _browser_install_hint() -> str:
    _bt = _origin()
    if _bt._is_termux_environment():
        return "npm install -g agent-browser && agent-browser install"
    return "npm install -g agent-browser && agent-browser install --with-deps"


def _is_npx_agent_browser_sentinel(browser_cmd: str) -> bool:
    _bt = _origin()
    return browser_cmd.strip() == _bt.NPX_AGENT_BROWSER_SENTINEL


def _requires_real_termux_browser_install(browser_cmd: str) -> bool:
    _bt = _origin()
    return _bt._is_termux_environment() and _bt._is_local_mode() and _bt._is_npx_agent_browser_sentinel(browser_cmd)


def _termux_browser_install_error() -> str:
    _bt = _origin()
    return (
        "Local browser automation on Termux cannot rely on the bare npx fallback. "
        f"Install agent-browser explicitly first: {_bt._browser_install_hint()}"
    )


def _agent_browser_candidate_present(path: str | None) -> bool:
    if not path:
        return False
    if " " in path and path.split()[0].endswith("npx"):
        return True
    return os.path.exists(path) and (os.name == "nt" or os.access(path, os.X_OK))


def _resolve_npx_bin() -> Optional[str]:
    """Resolve a runnable npx, preferring the Hermes-managed/Homebrew extended PATH.

    Bare PATH first would let a broken system npx shadow a healthy managed one
    with no recovery, so every candidate is validated with ``node_tool_runnable``
    before being trusted.
    """
    _bt = _origin()
    extended_path = _bt._merge_browser_path("")
    if extended_path:
        extended_npx = shutil.which("npx", path=extended_path)
        if extended_npx and _bt.node_tool_runnable(extended_npx):
            return extended_npx
    npx_path = shutil.which("npx")
    if npx_path and _bt.node_tool_runnable(npx_path):
        return npx_path
    return None


def _agent_browser_candidates(extended_path: str):
    """Yield agent-browser lookup candidates in resolution order (lazily — each is a filesystem probe).

    Order: ambient PATH (global install) → extended PATH (Hermes-managed Node,
    macOS versioned Homebrew, Termux/system dirs) → repo-local node_modules/.bin.
    The local lookup goes through ``shutil.which`` with an explicit path so
    Windows resolves the ``.cmd`` shim (CreateProcess cannot run npm's
    extensionless POSIX shim — WinError 193) while POSIX keeps the plain one.
    """
    yield shutil.which("agent-browser")
    if extended_path:
        yield shutil.which("agent-browser", path=extended_path)
    local_bin_dir = Path(__file__).parent.parent / "node_modules" / ".bin"
    if local_bin_dir.is_dir():
        yield shutil.which("agent-browser", path=str(local_bin_dir))


def _find_agent_browser(*, validate: bool = True) -> str:
    """
    Find the agent-browser CLI executable.

    Checks in order: current PATH, Homebrew/common bin dirs, Hermes-managed
    node, local node_modules/.bin/, npx fallback, then a lazy install.

    Every candidate is validated with ``agent_browser_runnable`` before it is
    cached. A bare ``shutil.which`` hit is NOT trusted: agent-browser's npm
    postinstall re-points a global symlink at our local node_modules binary,
    which disappears on the next ``hermes update`` and leaves a dangling link
    that ``which`` still reports but exec fails on (exit 127). Validating lets a
    dead candidate fall through instead of being cached and killing every
    browser tool. ``validate=False`` (schema-time check_fn) only tests presence
    and never caches.

    Raises:
        FileNotFoundError: If agent-browser is not installed
    """
    _bt = _origin()
    if _bt._agent_browser_resolved:
        if _bt._cached_agent_browser is None:
            raise FileNotFoundError(
                "agent-browser CLI not found (cached). Install it with: "
                f"{_bt._browser_install_hint()}\n"
                "Or ensure npx is available in your PATH."
            )
        return _bt._cached_agent_browser

    def _accept(candidate: str) -> str:
        # _agent_browser_resolved is set at each accept site (not before the
        # search) so a concurrent reader never sees resolved=True with a None cache.
        if validate:
            _bt._cached_agent_browser = candidate
            _bt._agent_browser_resolved = True
        return candidate

    ok = _bt.agent_browser_runnable if validate else _bt._agent_browser_candidate_present
    extended_path = _bt._merge_browser_path("")
    for candidate in _bt._agent_browser_candidates(extended_path):
        if candidate and ok(candidate):
            return _accept(candidate)

    # npx fallback (also searches the extended PATH)
    if _bt._resolve_npx_bin():
        return _accept(_bt.NPX_AGENT_BROWSER_SENTINEL)

    if not validate:
        raise FileNotFoundError("agent-browser CLI not found")

    # Nothing found — try lazy installation before giving up.
    try:
        from hermes_cli.dep_ensure import ensure_dependency
        if ensure_dependency("browser"):
            candidates = [
                shutil.which("agent-browser"),
                shutil.which("agent-browser", path=extended_path) if extended_path else None,
                shutil.which("agent-browser", path=str(_bt.get_hermes_home() / "node_modules" / ".bin")),
                shutil.which("agent-browser", path=str(_bt.get_hermes_home() / "node" / "bin")),
                shutil.which("agent-browser", path=str(_bt.get_hermes_home() / "node")),
            ]
            for recheck in candidates:
                if recheck and _bt.agent_browser_runnable(recheck):
                    return _accept(recheck)
    except Exception:
        pass

    _bt._agent_browser_resolved = True
    raise FileNotFoundError(
        "agent-browser CLI not found. Install it with: "
        f"{_bt._browser_install_hint()}\n"
        "Or ensure npx is available in your PATH."
    )


def warm_agent_browser_npx_cache(timeout: float = 60.0) -> bool:
    """Best-effort pre-fetch of the agent-browser npm package via npx.

    agent-browser resolves lazily via ``npx agent-browser`` (not a root
    package.json dependency), so the first invocation in a session would pay
    npx's registry fetch; ``hermes update`` / ``hermes doctor --fix`` call this
    to warm the cache first. Runs with the credential-scrubbed env every other
    agent-browser spawn uses (registry-fetched npm code must never see the
    operator keyring), in its own process group, and tree-kills on timeout so
    a surviving descendant cannot hold the capture pipe open.
    Never raises; True only when npx actually exited 0.
    """
    _bt = _origin()
    npx_bin = _bt._resolve_npx_bin()
    if not npx_bin:
        return False

    env = _bt._build_browser_env()
    env["PATH"] = _bt._merge_browser_path(env.get("PATH", ""))

    popen_kwargs: dict = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "env": env,
        "creationflags": _bt.windows_hide_flags(),
    }
    if os.name == "posix":
        popen_kwargs["start_new_session"] = True
    else:
        popen_kwargs["creationflags"] |= getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    cmd = [
        npx_bin,
        # --ignore-scripts: AGENT_BROWSER_NPX_SPEC is a floating ^0.26.0
        # range, not an exact pin — a compromised future 0.26.x patch must
        # not get to run its own install-time lifecycle scripts here.
        "--ignore-scripts",
        # --prefer-offline: once cached, repeat `hermes update`/`doctor
        # --fix` runs shouldn't hit the registry just to re-confirm
        # "latest" is still latest — that would defeat the point of
        # warming the cache in the first place.
        "--prefer-offline",
        "-y",
        _bt.AGENT_BROWSER_NPX_SPEC,
        "--version",
    ]
    try:
        proc = subprocess.Popen(cmd, stdin=subprocess.DEVNULL, **popen_kwargs)
    except Exception:
        return False
    try:
        proc.communicate(timeout=timeout)
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        _bt._kill_process_tree(proc)
        try:
            proc.communicate(timeout=5)
        except Exception:
            pass
        return False
    except Exception:
        _bt._kill_process_tree(proc)
        return False


def _chromium_search_roots() -> List[str]:
    """Directories to scan for a Chromium / headless-shell build, in the order
    agent-browser and Playwright probe them: ``PLAYWRIGHT_BROWSERS_PATH``, then
    Playwright's per-OS default cache."""
    roots: List[str] = []
    env_path = os.environ.get("PLAYWRIGHT_BROWSERS_PATH", "").strip()
    if env_path and env_path != "0":
        roots.append(env_path)
    home = os.path.expanduser("~")
    roots.append(os.path.join(home, ".cache", "ms-playwright"))
    if sys.platform == "darwin":
        roots.append(os.path.join(home, "Library", "Caches", "ms-playwright"))
    if sys.platform == "win32":
        local = os.environ.get("LOCALAPPDATA") or os.path.join(
            home, "AppData", "Local"
        )
        roots.append(os.path.join(local, "ms-playwright"))
    return roots


def _chromium_installed() -> bool:
    """Return True when a usable Chromium (or headless-shell) build is on disk.

    Checks ``AGENT_BROWSER_EXECUTABLE_PATH``, then system Chrome/Chromium on
    PATH, then Playwright's cache (``chromium-*`` / ``chromium_headless_shell-*``
    dirs). Without a binary the CLI hangs on first use until the command
    timeout fires, so the tool must not be advertised.
    """
    _bt = _origin()
    if _bt._cached_chromium_installed is not None:
        return _bt._cached_chromium_installed

    # 1. AGENT_BROWSER_EXECUTABLE_PATH — explicit user-configured browser
    ab_path = os.environ.get("AGENT_BROWSER_EXECUTABLE_PATH", "").strip()
    if ab_path and (os.path.isfile(ab_path) or shutil.which(ab_path)):
        _bt._cached_chromium_installed = True
        return True

    # 2. System Chrome/Chromium in PATH (common names)
    system_chrome = (
        shutil.which("google-chrome")
        or shutil.which("chromium")
        or shutil.which("chromium-browser")
        or shutil.which("chrome")
    )
    if system_chrome:
        _bt._cached_chromium_installed = True
        return True

    # 3. Playwright browser cache (legacy — chromium-* / chromium_headless_shell-* dirs)
    for root in _bt._chromium_search_roots():
        if not root or not os.path.isdir(root):
            continue
        try:
            entries = os.listdir(root)
        except OSError:
            continue
        # Playwright names them ``chromium-<build>`` and
        # ``chromium_headless_shell-<build>``; agent-browser accepts either.
        for entry in entries:
            if entry.startswith("chromium-") or entry.startswith(
                "chromium_headless_shell-"
            ):
                _bt._cached_chromium_installed = True
                return True

    _bt._cached_chromium_installed = False
    return False


def _maybe_autoinstall_chromium() -> bool:
    """Best-effort, gated download of the Chromium *binary* on local cold start.

    Binary only (``agent-browser install``), never ``--with-deps`` — that shells
    ``apt`` and needs root, so missing system libraries stay a user action.
    Gated by ``security.allow_lazy_installs``, skipped in Docker (Chromium ships
    in the image), attempted once per process. True only when Chromium is
    present afterwards.
    """
    _bt = _origin()
    if _bt._chromium_autoinstall_attempted:
        return _bt._chromium_installed()
    _bt._chromium_autoinstall_attempted = True

    if _bt._running_in_docker():
        return False

    from tools.lazy_deps import _allow_lazy_installs
    if not _allow_lazy_installs():
        return False

    try:
        browser_cmd = _bt._find_agent_browser()
    except FileNotFoundError:
        return False

    if _bt._is_npx_agent_browser_sentinel(browser_cmd):
        install_cmd = [
            _bt._resolve_npx_bin() or "npx", "--ignore-scripts", "-y", _bt.AGENT_BROWSER_NPX_SPEC, "install",
        ]
    else:
        install_cmd = [browser_cmd, "install"]

    _bt.logger.info(
        "browser: Chromium missing — auto-installing the browser binary "
        "(one-time ~170MB; disable via security.allow_lazy_installs)"
    )
    try:
        proc = subprocess.run(
            install_cmd,
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=600,
            env=_bt._build_browser_env(),
        )
    except (OSError, subprocess.SubprocessError) as e:
        _bt.logger.warning("browser: Chromium auto-install failed to start: %s", e)
        return False

    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip()[-300:]
        _bt.logger.warning(
            "browser: Chromium auto-install exited %s: %s", proc.returncode, tail
        )
        return False

    _bt._cached_chromium_installed = None
    return _bt._chromium_installed()


def _running_in_docker() -> bool:
    """Best-effort detection of whether we're inside a Docker container."""
    if os.path.exists("/.dockerenv"):
        return True
    try:
        with open("/proc/1/cgroup", "rt", encoding="utf-8") as fp:
            return "docker" in fp.read()
    except OSError:
        return False


def check_browser_requirements() -> bool:
    """Whether the browser tools should be advertised.

    Local mode needs the ``agent-browser`` CLI plus a Chromium build (except
    Lightpanda-only text workflows); cloud mode needs the CLI plus provider
    credentials (the provider hosts its own Chromium).
    """
    # Browser Use CLI backend — browser_exec replaces the whole browser_*
    # surface (including browser_cdp/browser_dialog, whose check_fns funnel
    # through here), so hide these tools from the model.
    _bt = _origin()
    if _bt._is_browser_use_cli_mode():
        return False

    # Camofox backend — only needs the server URL, no agent-browser CLI
    if _bt._is_camofox_mode():
        return True

    # CDP override mode can connect to an existing remote/local browser endpoint
    # without requiring the local agent-browser binary on PATH.
    # Raw (no-I/O) check: this runs during tool-schema assembly at startup,
    # where a stale endpoint must not cost a blocking HTTP probe.
    if _bt._get_cdp_override_raw():
        return True

    # The agent-browser CLI is required for local launch and cloud-provider flows.
    # Tool-schema assembly runs during Desktop startup; do not execute
    # ``agent-browser --version`` here, because Windows .cmd shims route through
    # cmd.exe and can flash a console before the user invokes any browser tool.
    # Actual browser execution paths still validate the candidate before use.
    try:
        browser_cmd = _bt._find_agent_browser(validate=False)
    except FileNotFoundError:
        return False

    # On Termux, the bare npx fallback is too fragile to treat as a satisfied
    # local browser dependency. Require a real install (global or local) so the
    # browser tool is not advertised as available when it will likely fail on
    # first use.
    if _bt._requires_real_termux_browser_install(browser_cmd):
        return False

    # In cloud mode, also require provider credentials. Cloud browsers
    # don't need a local Chromium binary.
    provider = _bt._get_cloud_provider()
    if provider is not None:
        return provider.is_configured()

    # Local mode with Lightpanda can provide text/navigation tools without a
    # local Chromium install. Chrome fallback, screenshots, and browser_vision
    # will still return actionable Chromium install errors if invoked.
    if _bt._using_lightpanda_engine():
        return True

    # Local Chrome mode: agent-browser needs a Chromium build on disk. Without
    # it the CLI hangs on first use until the command timeout fires.
    return _bt._chromium_installed()


def check_browser_vision_requirements() -> bool:
    """Advertise ``browser_vision`` only with BOTH a working browser AND a vision
    backend — otherwise it fails at call time with a cryptic provider error."""
    _bt = _origin()
    if not _bt.check_browser_requirements():
        return False
    try:
        from tools.vision_tools import check_vision_requirements
    except ImportError:
        return False
    return check_vision_requirements()
