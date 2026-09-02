"""Shared helpers for attaching Hermes to a local Chromium-family CDP port."""

from __future__ import annotations

import contextlib
import logging
import ntpath
import os
import platform
import posixpath
import re
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)


DEFAULT_BROWSER_CDP_PORT = 9222
DEFAULT_BROWSER_CDP_URL = f"http://127.0.0.1:{DEFAULT_BROWSER_CDP_PORT}"

@dataclass(frozen=True)
class _Browser:
    """Per-browser install/profile locations for one Chromium-family product."""

    key: str
    mac_app: str
    mac_support: tuple[str, ...]          # under ~/Library/Application Support
    win_bins: tuple[str, ...]             # shutil.which names on Windows
    win_install: tuple[tuple[str, ...], ...]  # under Program Files / LOCALAPPDATA
    win_profile: tuple[str, ...]          # under LOCALAPPDATA
    linux_bins: tuple[str, ...]           # shutil.which names on Linux
    linux_paths: tuple[str, ...]          # known absolute install paths
    linux_config: str                     # under $XDG_CONFIG_HOME
    # PATH names tried by chromium_executable() when they differ from
    # linux_bins (channel/alias binaries are launch candidates only).
    linux_exec: tuple[str, ...] | None = None


# Launch-candidate order (chrome, chromium, brave, brave-origin, edge) is the
# tuple order. ``brave-origin`` is Brave's standalone paid build: same Chromium
# core, but a fully distinct install identity (BraveSoftware/Brave-Origin
# product path, ``BraveOHTML`` ProgId, ``com.brave.Browser.origin`` bundle id)
# so it side-by-side installs with regular Brave — its profile is NOT under
# Brave-Browser and must never be conflated with the ``brave`` key (a "brave"
# lookup resolving to the Origin binary would drive the wrong browser's profile).
_BROWSERS = (
    _Browser(
        "chrome",
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        ("Google", "Chrome"),
        ("chrome.exe", "chrome"),
        (("Google", "Chrome", "Application", "chrome.exe"),),
        ("Google", "Chrome", "User Data"),
        ("google-chrome", "google-chrome-stable"),
        ("/opt/google/chrome/chrome", "/usr/bin/google-chrome", "/usr/bin/google-chrome-stable"),
        "google-chrome",
    ),
    _Browser(
        "chromium",
        "/Applications/Chromium.app/Contents/MacOS/Chromium",
        ("Chromium",),
        ("chromium.exe", "chromium"),
        (("Chromium", "Application", "chrome.exe"), ("Chromium", "Application", "chromium.exe")),
        ("Chromium", "User Data"),
        ("chromium-browser", "chromium"),
        ("/usr/bin/chromium-browser", "/usr/bin/chromium"),
        "chromium",
    ),
    _Browser(
        "brave",
        "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
        ("BraveSoftware", "Brave-Browser"),
        ("brave.exe", "brave"),
        (("BraveSoftware", "Brave-Browser", "Application", "brave.exe"),),
        ("BraveSoftware", "Brave-Browser", "User Data"),
        ("brave-browser", "brave-browser-stable", "brave"),
        (
            "/usr/bin/brave-browser",
            "/usr/bin/brave-browser-stable",
            "/usr/bin/brave",
            "/snap/bin/brave",
            "/opt/brave.com/brave/brave-browser",
            "/opt/brave.com/brave/brave",
            "/opt/brave-bin/brave",
        ),
        "BraveSoftware/Brave-Browser",
    ),
    _Browser(
        "brave-origin",
        "/Applications/Brave Origin.app/Contents/MacOS/Brave Origin",
        ("BraveSoftware", "Brave-Origin"),
        ("brave-origin.exe", "brave-origin"),
        (
            ("BraveSoftware", "Brave-Origin", "Application", "brave.exe"),
            ("BraveSoftware", "Brave-Origin", "Application", "brave-origin.exe"),
        ),
        ("BraveSoftware", "Brave-Origin", "User Data"),
        ("brave-origin", "brave-origin-nightly"),
        (
            "/usr/bin/brave-origin",
            "/opt/brave.com/brave-origin/brave-origin",
            "/opt/brave.com/brave-origin-nightly/brave-origin",
        ),
        "BraveSoftware/Brave-Origin",
        linux_exec=("brave-origin",),
    ),
    _Browser(
        "edge",
        "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
        ("Microsoft Edge",),
        ("msedge.exe", "msedge"),
        (("Microsoft", "Edge", "Application", "msedge.exe"),),
        ("Microsoft", "Edge", "User Data"),
        ("microsoft-edge", "microsoft-edge-stable", "msedge"),
        (
            "/usr/bin/microsoft-edge",
            "/usr/bin/microsoft-edge-stable",
            "/opt/microsoft/msedge/microsoft-edge",
            "/opt/microsoft/msedge/msedge",
        ),
        "microsoft-edge",
        linux_exec=("microsoft-edge", "microsoft-edge-stable"),
    ),
)
_BROWSER_BY_KEY = {b.key: b for b in _BROWSERS}


# ---------------------------------------------------------------------------
# Real-profile (default Chromium) resolution
#
# Used by the browser tool's ``browser.use_real_profile`` consent path: when a
# local Chromium is launched, point agent-browser at the user's REAL default
# browser profile (``--profile <user-data-dir>`` + ``--executable-path``) so
# their live logins/cookies are available. Only Chromium-family browsers are
# supported; a non-Chromium default (Firefox, Safari) resolves to None and the
# caller fails closed with a clear message.
# ---------------------------------------------------------------------------

# Windows UserChoice ProgId prefixes → canonical browser key. Matched
# case-insensitively by prefix so version suffixes (e.g. ``ChromeHTML.X``)
# still resolve to STABLE. Pre-release channels have their own ProgIds and
# MUST be matched first (see _WINDOWS_CHANNEL_PROGIDS) so they are never
# swallowed into the stable family — driving the wrong profile is a
# wrong-principal bug (#95549 invariant).
_WINDOWS_PROGID_MAP = (
    ("chromehtml", "chrome"),
    ("msedgehtm", "edge"),
    # Brave Origin stable is ``BraveOHTML`` (brave-core install_static). Listed
    # before ``bravehtml`` for clarity; the prefixes don't collide either way.
    ("braveohtml", "brave-origin"),
    ("bravehtml", "brave"),
    ("chromiumhtm", "chromium"),
)

# Pre-release ProgId prefixes we recognize but do NOT support (their profiles
# live in channel-specific dirs the resolver tables don't carry). Matched
# BEFORE the stable map; a hit fails closed rather than resolving to stable.
# ``ChromeBHTML`` = Beta, ``ChromeDHTML`` = Dev, ``ChromeSSHTML`` = Canary
# (SxS); ``MSEdgeBHTML`` / ``MSEdgeDHTML`` / ``MSEdgeCHTML`` = Edge channels.
_WINDOWS_CHANNEL_PROGIDS = (
    "chromebhtml", "chromedhtml", "chromesshtml", "chromecanaryhtml",
    "msedgebhtml", "msedgedhtml", "msedgechtml",
    "bravebetahtml", "bravenightlyhtml",
    # Brave Origin channels (brave-core install_static): Beta=BraveOBHTML,
    # Dev=BraveODHTML, Nightly/SxS=BraveOSHTM (no trailing L — 10-char cap).
    "braveobhtml", "braveodhtml", "braveoshtm",
)

# Linux xdg default-web-browser .desktop name fragments → canonical STABLE key.
# Includes the Flatpak application ids (``com.google.Chrome.desktop`` etc.),
# which share none of the native package name fragments. Anchored so a channel
# .desktop (``google-chrome-beta``, ``com.google.chrome.beta``) does NOT match
# the stable fragment — channels are caught by _LINUX_CHANNEL_FRAGMENTS first.
_LINUX_DESKTOP_MAP = (
    ("google-chrome", "chrome"),
    ("com.google.chrome", "chrome"),
    ("chromium", "chromium"),
    # ORDER MATTERS: ``brave-origin.desktop`` contains the bare ``brave``
    # fragment, so the substring scan must hit the Origin entry first —
    # otherwise an Origin default resolves to stable Brave and real-profile
    # mode drives a DIFFERENT browser's profile (wrong-principal, #95549).
    ("brave-origin", "brave-origin"),
    ("brave", "brave"),
    ("microsoft-edge", "edge"),
    ("com.microsoft.edge", "edge"),
    ("msedge", "edge"),
)

# Non-stable Linux channel .desktop fragments — recognized, unsupported.
# Checked before the stable map; a hit fails closed.
_LINUX_CHANNEL_FRAGMENTS = (
    "google-chrome-beta", "google-chrome-unstable", "google-chrome-canary",
    "com.google.chrome.beta", "com.google.chrome.dev", "com.google.chrome.canary",
    "microsoft-edge-beta", "microsoft-edge-dev", "microsoft-edge-canary",
    "brave-browser-beta", "brave-browser-nightly", "brave-browser-dev",
    "brave-origin-beta", "brave-origin-nightly", "brave-origin-dev",
)

# Where sandboxed Linux packages keep the profile instead of $XDG_CONFIG_HOME.
_LINUX_FLATPAK_IDS = {
    "chrome": "com.google.Chrome",
    "chromium": "org.chromium.Chromium",
    "brave": "com.brave.Browser",
    "edge": "com.microsoft.Edge",
}
_LINUX_SNAP_PROFILE_PARTS = {
    "chromium": ("snap", "chromium", "common", "chromium"),
    "brave": ("snap", "brave", "current", ".config", "BraveSoftware", "Brave-Browser"),
}

# macOS LaunchServices bundle-id → canonical STABLE key. EXACT match (not
# prefix): ``com.google.chrome.beta`` must not be read as ``com.google.chrome``.
_DARWIN_BUNDLE_MAP = (
    ("com.google.chrome", "chrome"),
    ("com.microsoft.edgemac", "edge"),
    ("com.brave.browser", "brave"),
    # Brave Origin reuses the Brave bundle id with an ``.origin`` suffix
    # (Homebrew cask: com.brave.Browser.origin). Exact matching keeps it from
    # ever being read as plain ``com.brave.browser``.
    ("com.brave.browser.origin", "brave-origin"),
    ("org.chromium.chromium", "chromium"),
)

# Non-stable macOS channel bundle ids — recognized, unsupported. Checked first.
_DARWIN_CHANNEL_BUNDLES = (
    "com.google.chrome.beta", "com.google.chrome.dev", "com.google.chrome.canary",
    "com.microsoft.edgemac.beta", "com.microsoft.edgemac.dev", "com.microsoft.edgemac.canary",
    "com.brave.browser.beta", "com.brave.browser.nightly",
    "com.brave.browser.origin.beta", "com.brave.browser.origin.dev",
    "com.brave.browser.origin.nightly",
)

# Sentinel returned when the OS default is a recognized-but-unsupported
# Chromium CHANNEL (Beta/Dev/Canary). Distinct from None (non-Chromium) so the
# caller fails closed with a channel-specific message instead of driving the
# stable profile of a different account.
UNSUPPORTED_CHANNEL = "__unsupported_channel__"


def real_profile_data_dir(browser: str, system: str | None = None) -> str | None:
    """Return the default user-data-dir for a Chromium ``browser`` on ``system``.

    Returns None for unknown browsers. On Linux the native ($XDG_CONFIG_HOME), snap and Flatpak
    locations are tried and the first existing one wins; the native path is returned when none
    exists so the caller's error names it. Darwin/Windows paths are not stat'ed.
    """
    b = _BROWSER_BY_KEY.get(browser)
    if b is None:
        return None
    system = system or platform.system()
    home = os.path.expanduser("~")
    if system == "Darwin":
        return posixpath.join(home, "Library", "Application Support", *b.mac_support)
    if system == "Windows":
        local = os.environ.get("LOCALAPPDATA") or ntpath.join(home, "AppData", "Local")
        return ntpath.join(local, *b.win_profile)
    # Linux / other POSIX
    config = os.environ.get("XDG_CONFIG_HOME") or posixpath.join(home, ".config")
    linux_parts = b.linux_config.split("/")
    candidates = [posixpath.join(config, *linux_parts)]
    snap_parts = _LINUX_SNAP_PROFILE_PARTS.get(browser)
    if snap_parts:
        candidates.append(posixpath.join(home, *snap_parts))
    flatpak_id = _LINUX_FLATPAK_IDS.get(browser)
    if flatpak_id:
        candidates.append(posixpath.join(home, ".var", "app", flatpak_id, "config", *linux_parts))
    for candidate in candidates:
        if os.path.isdir(candidate):
            return candidate
    return candidates[0]


def _first_present(paths) -> str | None:
    for p in paths:
        if p and os.path.isfile(p):
            return p
    return None


def chromium_executable(browser: str, system: str | None = None) -> str | None:
    """Return the first present executable for a Chromium ``browser``."""
    b = _BROWSER_BY_KEY.get(browser)
    if b is None:
        return None
    system = system or platform.system()
    if system == "Darwin":
        return _first_present((b.mac_app,))
    if system == "Windows":
        bases = [
            os.environ.get("PROGRAMFILES", r"C:\Program Files"),
            os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
            os.environ.get("LOCALAPPDATA", str(Path.home() / "AppData" / "Local")),
        ]
        return _first_present(os.path.join(base, *parts) for base in bases for parts in b.win_install)
    # Linux: PATH lookup first, then the known absolute install paths.
    for name in b.linux_exec or b.linux_bins:
        found = shutil.which(name)
        if found:
            return found
    return _first_present(b.linux_paths)


def _classify_default(value: str, channels, table, match) -> str | None:
    """Map an OS default-browser identifier to a canonical key.

    Channels are checked FIRST: a recognized Beta/Dev/Canary identifier must fail closed
    (UNSUPPORTED_CHANNEL), never fall through to a stable match and drive the stable profile.
    """
    for chan in channels:
        if match(value, chan):
            return UNSUPPORTED_CHANNEL
    for frag, browser in table:
        if match(value, frag):
            return browser
    return None


def _detect_default_windows() -> str | None:
    try:
        import winreg  # type: ignore
    except Exception:
        return None
    try:
        key = winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\Shell\Associations\UrlAssociations\https\UserChoice",
        )
        prog_id, _ = winreg.QueryValueEx(key, "ProgId")
        winreg.CloseKey(key)
    except Exception:
        return None
    low = str(prog_id or "").lower()
    return _classify_default(low, _WINDOWS_CHANNEL_PROGIDS, _WINDOWS_PROGID_MAP, str.startswith)


def _run_stdout(argv: list[str]) -> str | None:
    try:
        return subprocess.run(
            argv, capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=5
        ).stdout
    except Exception:
        return None


_LS_HANDLERS_READER = (
    "defaults",
    "read",
    "com.apple.LaunchServices/com.apple.launchservices.secure",
    "LSHandlers",
)


def _launchservices_https_handler(dump: str) -> str | None:
    """Return the bundle id registered for the ``https`` URL scheme.

    ``dump`` is the ``defaults read … LSHandlers`` output: an array of ``{ … }`` dictionaries, one
    per handler. Only the entry whose ``LSHandlerURLScheme`` is ``https`` counts — a browser
    registered for another scheme or a file type must not be mistaken for the default.
    """
    entries: list[str] = []
    depth = 0
    buf: list[str] = []
    for ch in dump:
        if ch == "{":
            depth += 1
            if depth == 1:
                buf = []
                continue
        elif ch == "}":
            depth -= 1
            if depth == 0:
                entries.append("".join(buf))
                continue
        if depth >= 1:
            buf.append(ch)
    for entry in entries:
        low = entry.lower()
        if not re.search(r'lshandlerurlscheme\s*=\s*"?https"?\s*;', low):
            continue
        # Strip the nested LSHandlerPreferredVersions block first: on macOS 26
        # it carries a VERSION NUMBER (e.g. LSHandlerRoleAll = "7559.97";), not
        # the "-" placeholder older releases used. Left in, the role regex below
        # would match that version before the real bundle id sitting at the
        # entry's own level and return "7559.97" — which maps to no browser, so
        # detection fails on a machine whose default IS Chrome (PR #95620 review).
        low = re.sub(r"lshandlerpreferredversions\s*=\s*\{[^}]*\}\s*;", "", low)
        # The real bundle id is the first non-"-" role value at this level.
        for role in re.findall(r'lshandlerrole(?:all|viewer)\s*=\s*"?([a-z0-9.\-]+)"?\s*;', low):
            if role != "-":
                return role
        return None
    return None


def _detect_default_darwin() -> str | None:
    out = _run_stdout(list(_LS_HANDLERS_READER))
    if out is None:
        return None
    bundle = _launchservices_https_handler(out)
    if not bundle:
        return None
    # Exact match. A non-Chromium https handler (Safari, Firefox, Arc, …) or an
    # unknown channel bundle fails closed: no "first installed Chromium wins"
    # fallback — that would drive a browser the user never made their default.
    return _classify_default(bundle.lower(), _DARWIN_CHANNEL_BUNDLES, _DARWIN_BUNDLE_MAP, str.__eq__)


def _detect_default_linux() -> str | None:
    out = (_run_stdout(["xdg-settings", "get", "default-web-browser"]) or "").strip().lower()
    # Substring match; channels first because ``google-chrome-beta.desktop``
    # contains the stable ``google-chrome`` fragment.
    return _classify_default(out, _LINUX_CHANNEL_FRAGMENTS, _LINUX_DESKTOP_MAP, str.__contains__)


def detect_default_chromium(system: str | None = None) -> str | None:
    """Return the canonical key of the default Chromium browser, or None."""
    system = system or platform.system()
    if system == "Windows":
        return _detect_default_windows()
    if system == "Darwin":
        return _detect_default_darwin()
    return _detect_default_linux()


# ---------------------------------------------------------------------------
# Real-profile SNAPSHOT launch
#
# The consent path (``browser.use_real_profile``) never drives the live
# default user-data-dir. Chromium ≥136 (Google-branded builds) refuses
# remote debugging on the default dir no matter who launches it, and the
# live dir is usually held by the user's running browser (SingletonLock).
# Instead we snapshot the real profile into ``~/.hermes/browser-profile/``
# — a non-default dir Chrome will happily debug, that never contends with
# the user's browser — launch the user's real binary on the copy with a
# devtools port, and hand the CDP URL to whichever browser lane is active
# (Browser Use CLI or the built-in tools). We launch the browser ourselves
# precisely so NO mock-keychain/basic-store switches are added: cookies
# encrypted with the OS keyring (gnome-keyring / kwallet / macOS Keychain)
# decrypt exactly like they do in the user's own browser.
# ---------------------------------------------------------------------------

# Directory names excluded from the profile snapshot: caches/telemetry AND the
# heavy, replay-prone state that hangs a fresh Chromium's renderer (extensions
# and their service workers spin up on launch and wedge JS eval; IndexedDB /
# GPUCache add hundreds of MB for nothing). We keep ONLY auth/login state
# (cookies, Login Data, Web Data, Preferences, Local State) — the point of the
# feature — which turns a multi-hundred-MB profile into a few MB.
_SNAPSHOT_IGNORES = (
    "*Cache*",          # Cache, Code Cache, GPUCache, GrShaderCache, ShaderCache, GraphiteDawnCache, component_crx_cache, ...
    "Extensions",       # wallets/etc.: 100s of MB, and hang the renderer headless
    "Extension*",       # Extension State, Extension Rules, Extension Scripts
    "Local Extension Settings",
    "Service Worker",   # replays on launch → wedges the renderer
    "IndexedDB",
    "Crash Reports",
    "Crashpad",
    "BrowserMetrics*",
    "Snapshots",
    "OptimizationGuide*",
    "optimization_guide_model_store",
    "Safe Browsing",
    "SafetyTips",
    "OnDeviceHeadSuggestModel",
    "segmentation_platform",
    "Sync Data",
    "Shared Dictionary",
    "History*",         # large; not needed for auth
    "Favicons*",
    "Singleton*",       # live-instance symlinks; never valid in a copy
    "RunningChromeVersion",
    "SingletonSocket",
    "*.tmp",
    "*-journal",         # SQLite rollback journals — sidecars of the auth DBs,
    "*-wal",             # which are copied via online-backup; a stale sidecar
    "*-shm",             # next to a backed-up DB corrupts it.
    "BrowserMetrics-spare.pma",
)

# Small, auth-bearing files re-synced from the live profile on EVERY consented
# launch (the full tree is only copied when the snapshot doesn't exist yet).
# Paths here are RELATIVE TO A PROFILE DIR (Default, "Profile 6", …) — the
# caller resolves which source profile is active and mirrors these into the
# copy's ``Default`` so the launched Chromium (which opens ``Default``) lands
# on the user's real signed-in session. No ``-journal``/``-wal`` sidecars: the
# SQLite DBs are copied via the online-backup API (see _copy_auth_file), which
# produces a self-contained DB with committed state folded in — copying a
# stale raw journal on top of that would corrupt it.
_AUTH_REFRESH_PROFILE_FILES = (
    "Cookies",
    "Network/Cookies",
    "Login Data",
    "Login Data For Account",
    "Web Data",
    "Preferences",
)

def real_profile_copy_dir(browser: str) -> str:
    """Return the hermes-owned snapshot dir for ``browser``'s real profile."""
    return str(get_hermes_home() / "browser-profile" / browser)


def _last_used_profile(src: str) -> str:
    """Return the profile dir Chrome last used (``Local State`` → profile.last_used).

    Chromium opens ``Default`` inside a user-data-dir unless told otherwise, but the user's signed-
    in session usually lives in whichever profile they actually browse (``Profile 6`` etc.). We read
    that here and mirror its auth into the copy's ``Default`` so the launched browser is signed in.
    """
    import json

    try:
        with open(os.path.join(src, "Local State"), encoding="utf-8", errors="replace") as fh:
            state = json.load(fh)
        last = ((state.get("profile") or {}).get("last_used")) or "Default"
    except (OSError, ValueError, AttributeError):
        last = "Default"
    if not isinstance(last, str) or not os.path.isdir(os.path.join(src, last)):
        return "Default"
    return last


def _secure_snapshot_root(path: str) -> None:
    """Lock down a snapshot dir through Hermes' canonical secret-store policy.

    The snapshot holds copies of the user's Cookies / Login Data, so it is a credential store and
    must get the same owner-only permissions (and managed-mode / NixOS group-share carve-out,
    HERMES_UID/GID ownership) as every other Hermes secret dir — via
    ``hermes_cli.config._secure_dir``, not a bespoke chmod. Best-effort: never blocks a launch.
    """
    try:
        from hermes_cli.config import _secure_dir

        _secure_dir(path)
    except Exception as e:
        logger.debug("could not secure real-profile snapshot dir %s: %s", path, e)


def _secure_snapshot_contents(dst: str) -> None:
    """Owner-only modes for every file/dir INSIDE the snapshot (#96729).

    ``_secure_snapshot_root`` covers the top-level dirs, but ``copy2`` preserves Chrome's 0644 file
    modes and ``sqlite3`` backups create umask files, so live session cookies landed world-readable
    — a real exposure under the ``HERMES_HOME_MODE`` hatch. Reconciled via ``_secure_dir`` /
    ``_secure_file`` on EVERY pass so older snapshots heal too. Best-effort: never blocks a launch.
    """
    try:
        from hermes_cli.config import _secure_dir, _secure_file

        for root, dirs, files in os.walk(dst):
            for d in dirs:
                _secure_dir(os.path.join(root, d))
            for f in files:
                _secure_file(os.path.join(root, f))
    except Exception as e:
        logger.debug("could not secure real-profile snapshot contents %s: %s", dst, e)


# Auth files that are SQLite databases: on Windows a running Chrome holds these
# with an exclusive lock, so a raw file copy raises WinError 32 ("being used by
# another process") and a naive best-effort skip leaves the copy signed-out.
# These are copied via SQLite's online-backup API instead, which reads a
# consistent committed snapshot while the lock is held. Matched by basename.
_SQLITE_AUTH_DBS = frozenset({
    "Cookies", "Login Data", "Login Data For Account", "Web Data",
})


def _copy_auth_file(src_file: str, dst_file: str) -> bool:
    """Copy one auth file, lock-aware. Returns True on success.

    For SQLite DBs (Cookies/Login Data/…), use the online-backup API so the copy works even while
    the browser holds the file's write lock (Windows). Everything else is a plain copy. A DB whose
    backup fails falls through to a raw copy attempt; only if BOTH fail do we report failure to the
    caller.
    """
    os.makedirs(os.path.dirname(dst_file), exist_ok=True)
    if os.path.basename(src_file) in _SQLITE_AUTH_DBS:
        # On a live Chrome on macOS the profile holds
        # its DBs in a state where mode=ro WITHOUT immutable=1 can hang the
        # connect/backup indefinitely (the sqlite busy-timeout never fires
        # because the block happens inside lock negotiation). immutable=1
        # reads instantly and is correct here: we want a committed snapshot of
        # a file another process owns, not coordinated writes. A torn read
        # raises → falls through to the plain-copy fallback below.
        for uri in (
            f"file:{src_file}?mode=ro&immutable=1",
            f"file:{src_file}?mode=ro",
        ):
            try:
                import sqlite3

                # Short busy timeout so a truly wedged DB fails fast rather
                # than hanging the launch.
                source = sqlite3.connect(uri, uri=True, timeout=5)
                try:
                    out = sqlite3.connect(dst_file)
                    try:
                        with out:
                            source.backup(out)
                    finally:
                        out.close()
                finally:
                    source.close()
                return True
            except Exception as e:
                logger.debug("real-profile: sqlite-backup of %s failed (%s); trying next mode",
                             src_file, e)
    # Non-DB file, or DB whose backup failed: raw copy.
    try:
        shutil.copy2(src_file, dst_file)
        return True
    except OSError as e:
        logger.debug("real-profile: could not copy %s: %s", src_file, e)
        return False


def _mirror_profile_auth(src: str, dst: str, source_profile: str) -> int:
    """Copy ``source_profile``'s auth files into the copy's ``Default`` slot.

    agent-browser launches ``Default`` in the copied user-data-dir, so mirroring the active
    profile's cookies/logins/prefs there is what makes the session actually signed in when the real
    session lives in a non-Default profile. Lock-aware on Windows. Returns the number of DB auth
    files that could NOT be copied (0 = clean).
    """
    dst_default = os.path.join(dst, "Default")
    failed_dbs = 0
    for rel in _AUTH_REFRESH_PROFILE_FILES:
        s = os.path.join(src, source_profile, rel)
        if not os.path.isfile(s):
            continue
        ok = _copy_auth_file(s, os.path.join(dst_default, rel))
        if not ok and os.path.basename(rel) in _SQLITE_AUTH_DBS:
            failed_dbs += 1
    return failed_dbs


_SNAPSHOT_DONE_MARKER = ".hermes-snapshot-complete"

# Prefix stamped on the "profile is locked" error so the calling layer can
# recognize it as the specific needs-the-browser-closed condition (vs a generic
# snapshot failure) and surface the close-with-approval flow.
_PROFILE_LOCKED_PREFIX = "[profile-locked] "


def _profile_cookie_db(src: str, source_profile: str) -> str | None:
    """Path to the active profile's cookie DB (modern Network/ first)."""
    return _first_present(
        os.path.join(src, source_profile, rel) for rel in (os.path.join("Network", "Cookies"), "Cookies")
    )


def _profile_is_locked(src: str, source_profile: str) -> bool:
    """True when the active profile's cookie DB can't be opened (browser running).

    On Windows a running browser holds Cookies with a deny-all share mode, so a plain open raises
    PermissionError. This is a FAST one-open probe used to fail closed BEFORE the heavy snapshot so
    a locked profile can never hang the launch. POSIX has no mandatory locking, so it returns False.
    """
    db = _profile_cookie_db(src, source_profile)
    if not db:
        return False  # nothing to lock; let the copy path handle "no cookies"
    try:
        with open(db, "rb"):
            return False
    except PermissionError:
        return True
    except OSError:
        # Other errors (transient) — don't declare locked; let the copy try.
        return False


def _browser_setting(key: str):
    """Read ``browser.<key>`` from raw config; None when unset/unreadable."""
    try:
        from hermes_cli.config import read_raw_config

        browser_cfg = read_raw_config().get("browser", {})
        if isinstance(browser_cfg, dict):
            return browser_cfg.get(key)
    except Exception as e:
        logger.debug("could not read %s: %s", key, e)
    return None


def _real_profile_pin() -> str | None:
    """Pinned source profile dir name from ``browser.real_profile_pin``.

    Natively the snapshot follows Chrome's ``profile.last_used``, which on a machine with work and
    personal profiles can silently hand the agent the wrong identity. When set (e.g. ``"Profile
    2"``) that profile is ALWAYS copied; unset keeps the native last_used behavior.
    """
    pin = _browser_setting("real_profile_pin")
    if isinstance(pin, str) and pin.strip():
        return pin.strip()
    return None


def _resolve_source_profile(src: str) -> tuple[str | None, str | None]:
    """Resolve which source profile to copy: pin first, else last_used.

    Returns ``(profile_dir_name, error)``. A configured pin that does not exist under ``src`` FAILS
    CLOSED with a fixable message — falling back to last_used would silently browse as the wrong
    identity, which is the exact wrong-principal bug this pin exists to prevent.
    """
    pin = _real_profile_pin()
    if pin:
        if os.path.isdir(os.path.join(src, pin)):
            return pin, None
        return None, (
            f"browser.real_profile_pin is set to '{pin}' but that profile "
            f"directory does not exist under {src!r}. Profile directories are "
            "named like 'Default' or 'Profile 2' — list them with: "
            f"ls {src!r}. Fix the pin, or remove it to fall back to the "
            "last-used profile."
        )
    return _last_used_profile(src), None


def _real_profile_autoclose() -> bool:
    """Whether browser.real_profile_autoclose consent is on (config read)."""
    return bool(_browser_setting("real_profile_autoclose") or False)


def _processes_holding_profile(src: str):
    """Yield (psutil.Process) instances holding the user-data-dir ``src`` open.

    Identity discipline mirrors the daemon reaper: a process qualifies only when it's a Chromium-
    family binary AND its command line references THIS user-data-dir — so we never terminate an
    unrelated same-PID process. Any ambiguity (unreadable cmdline) is skipped, fail-closed.
    """
    try:
        import psutil
    except ImportError:  # hard dep; defensive
        return
    norm = os.path.normcase(os.path.normpath(src))
    browser_bins = (
        "chrome", "chrome.exe", "chromium", "chromium.exe", "chrome_crashpad",
        "brave", "brave.exe", "msedge", "msedge.exe", "google chrome",
    )
    for proc in psutil.process_iter(["name", "cmdline"]):
        try:
            name = (proc.info.get("name") or "").lower()
            cmd = proc.info.get("cmdline") or []
            joined = " ".join(cmd)
        except (psutil.NoSuchProcess, psutil.AccessDenied, OSError):
            continue
        if not any(b in name for b in browser_bins):
            # Some platforms report a generic name; also accept when the binary
            # in argv[0] looks like a browser.
            argv0 = (cmd[0].lower() if cmd else "")
            if not any(b in argv0 for b in browser_bins):
                continue
        # Binding: the exact user-data-dir must appear in the cmdline
        # (--user-data-dir=<src>), normalized for case/separators.
        if norm not in os.path.normcase(os.path.normpath(joined)) and \
           f"--user-data-dir={src}".lower() not in joined.lower():
            continue
        yield proc


def close_browser_holding_profile(src: str, timeout: float = 15.0) -> tuple[bool, str]:
    """Terminate the browser process tree holding ``src`` and wait for release.

    CONSENTED, DESTRUCTIVE. Only call after the user has agreed to close their browser — it
    terminates every Chromium-family process bound to this exact user-data-dir (graceful terminate,
    then kill), so unsaved tab/form state in that browser is lost.
    """
    try:
        import psutil
    except ImportError:
        return False, "psutil unavailable — cannot close the browser automatically."

    procs = list(_processes_holding_profile(src))
    if not procs:
        # Nothing we can see holds it. Either already closed, or the holder is
        # a different user / unreadable — caller re-probes the lock.
        return False, "no matching browser process found holding the profile."

    # Include child processes (renderers, GPU, crashpad) for a full tree kill.
    gone_errs = (psutil.NoSuchProcess, psutil.AccessDenied)
    targets = []
    for p in procs:
        targets.append(p)
        with contextlib.suppress(*gone_errs):
            targets.extend(p.children(recursive=True))
    # Graceful terminate first.
    for p in targets:
        with contextlib.suppress(*gone_errs):
            p.terminate()
    _gone, alive = psutil.wait_procs(targets, timeout=min(timeout, 8.0))
    for p in alive:
        with contextlib.suppress(*gone_errs):
            p.kill()
    psutil.wait_procs(alive, timeout=3.0)

    # The lock releases slightly after the process exits on Windows; poll.
    source_profile = _resolve_source_profile(src)[0] or _last_used_profile(src)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not _profile_is_locked(src, source_profile):
            return True, "closed the browser and the profile lock released."
        time.sleep(0.5)
    return False, (
        "closed the browser processes but the profile is still locked — "
        "another instance may have relaunched (background/tray mode)."
    )


def _sync_local_state(src: str, dst: str, source_profile: str) -> None:
    """Copy ``Local State`` into the snapshot and rewrite it for the single ``Default`` profile.

    Cheap; always re-synced so last_used etc. stay current. The copy contains ONLY the mirrored
    Default dir, but a verbatim Local State still names the SOURCE profile (last_used="Profile 2",
    info_cache listing Profile 2/4/7), so Chrome would open a missing profile dir and start SIGNED
    OUT. CRITICAL: Default's identity entry must be the SOURCE profile's entry (name + Google
    account), not the source's own "Default" entry — the Default DIR holds the source profile's
    cookies. A mismatch makes Chrome demand a "Continue as <name>" reconciliation on every launch.
    """
    ls_src = os.path.join(src, "Local State")
    ls_dst = os.path.join(dst, "Local State")
    if os.path.isfile(ls_src):
        try:
            shutil.copy2(ls_src, ls_dst)
        except OSError as e:
            logger.debug("real-profile snapshot: skipped Local State: %s", e)
    try:
        import json as _json

        with open(ls_dst, encoding="utf-8") as fh:
            state = _json.load(fh)
        prof = state.get("profile")
        if isinstance(prof, dict):
            cache = prof.get("info_cache")
            if isinstance(cache, dict):
                src_entry = cache.get(source_profile) or cache.get("Default")
                if src_entry:
                    prof["info_cache"] = {"Default": src_entry}
            prof["last_used"] = "Default"
            prof["last_active_profiles"] = ["Default"]
        with open(ls_dst, "w", encoding="utf-8") as fh:
            _json.dump(state, fh)
    except (OSError, ValueError) as e:
        logger.debug("real-profile snapshot: could not normalize Local State: %s", e)


def snapshot_real_profile(browser: str, src: str | None = None) -> tuple[str | None, str | None]:
    """Snapshot ``browser``'s real ACTIVE profile into the hermes copy dir.

    Copies only what the launched browser needs: the user-data-dir's ``Local State`` plus the auth-
    bearing files of the profile the user actually browses (``Local State → profile.last_used``,
    e.g. ``Profile 6``), mirrored into the copy's ``Default`` — which is what agent-browser opens.

    A ``.hermes-snapshot-complete`` marker is written only after a copy fully succeeds; a
    torn/interrupted first copy (disk full, Ctrl+C) therefore never looks "already populated" on the
    next run — it is redone from scratch.
    """
    src = src or real_profile_data_dir(browser)
    if not src or not os.path.isdir(src):
        return None, (
            f"profile directory for '{browser}' was not found ({src!r}). "
            "Launch that browser at least once, or turn browser.use_real_profile off."
        )
    source_profile, resolve_err = _resolve_source_profile(src)
    if resolve_err or not source_profile:
        return None, resolve_err
    dst = real_profile_copy_dir(browser)
    # Fast lock probe BEFORE any copy: a running browser holds the cookie DB
    # deny-all (Windows), and a blocking file op on it can hang the launch for
    # minutes. On POSIX this never trips (no mandatory locking) so
    # copy-while-running still works.
    if _profile_is_locked(src, source_profile):
        # NEVER kill from here. Closing the user's browser is destructive and
        # must be an explicit, per-attempt, user-approved step — not a silent
        # side effect of a snapshot. So we always BLOCK when locked and let the
        # agent decide whether to ask the user to close it (only offered when
        # browser.real_profile_autoclose arms the capability). A subsequent
        # attempt that is still locked blocks again — no auto-retry, no loop.
        if _real_profile_autoclose():
            msg = (
                f"{browser} is running and has its profile locked, so its login "
                "data can't be copied yet. Hermes can close it for you "
                "(this quits the browser — you'll lose unsaved tabs). Ask the "
                "user to confirm, then close it and retry; if it's still locked "
                "after that, they must fully quit it (including any "
                "background/tray instance)."
            )
        else:
            msg = (
                f"{browser} is running and has its profile locked, so its login "
                "data can't be copied. Fully quit the browser (including any "
                "background/tray instance) and retry, or turn "
                "browser.use_real_profile off. (Enable "
                "browser.real_profile_autoclose to let Hermes offer to close it "
                "for you.)"
            )
        return None, _PROFILE_LOCKED_PREFIX + msg
    marker = os.path.join(dst, _SNAPSHOT_DONE_MARKER)
    # Only a copy that previously COMPLETED counts as populated. A half-written
    # tree (no marker) is treated as absent and rebuilt — otherwise a torn first
    # copy poisons freshness forever and only ever gets auth overlays.
    populated = os.path.isfile(marker)
    try:
        os.makedirs(dst, exist_ok=True)
        # Secure the snapshot dir AND its browser-profile parent on EVERY
        # launch: a failed first attempt or an older-build dir must still
        # converge to owner-only perms; the parent enumerates every browser we
        # hold cookies for.
        parent = os.path.dirname(dst)
        if parent:
            _secure_snapshot_root(parent)
        _secure_snapshot_root(dst)

        _sync_local_state(src, dst, source_profile)

        if not populated:
            # Fresh (or torn-and-rebuilding): drop any partial Default and copy
            # the ACTIVE profile's full dir (minus caches AND the locked auth
            # DBs) into the copy's Default. The SQLite auth DBs are excluded
            # here because a raw copytree of a file a running Chrome holds open
            # raises on Windows; they are copied lock-aware by
            # _mirror_profile_auth below (sqlite online-backup).
            dst_default = os.path.join(dst, "Default")
            try:
                shutil.rmtree(dst_default, ignore_errors=True)
                shutil.copytree(
                    os.path.join(src, source_profile),
                    dst_default,
                    dirs_exist_ok=True,
                    symlinks=False,
                    ignore=shutil.ignore_patterns(*_SNAPSHOT_IGNORES, *_SQLITE_AUTH_DBS),
                    ignore_dangling_symlinks=True,
                )
            except shutil.Error as multi:
                # Per-file failures (browser mid-write) are non-fatal.
                logger.info(
                    "real-profile snapshot: %d file(s) skipped copying %s/%s",
                    len(multi.args[0]) if multi.args else 0, src, source_profile,
                )

        # Both paths: copy the active profile's auth DBs into Default,
        # lock-aware (sqlite online-backup) so a running Chrome on Windows
        # doesn't block them. This is also the per-launch fresh-login re-sync.
        failed_dbs = _mirror_profile_auth(src, dst, source_profile)
        if failed_dbs:
            # We could not read the user's cookie/login DBs at all — even the
            # online-backup fallback failed. Rather than launch a silently
            # signed-out session, fail closed with an actionable message.
            return None, (
                f"could not read the '{browser}' profile's login data "
                f"({failed_dbs} database(s) locked). Close {browser} and retry, "
                "or turn browser.use_real_profile off."
            )

        # Never carry live-instance leftovers into the copy.
        for leftover in ("SingletonLock", "SingletonSocket", "SingletonCookie"):
            with contextlib.suppress(OSError):
                os.unlink(os.path.join(dst, leftover))
        # Mark complete only after everything above succeeded.
        try:
            with open(marker, "w", encoding="utf-8") as fh:
                fh.write(source_profile)
        except OSError as e:
            logger.debug("real-profile snapshot: could not write done marker: %s", e)
        # Owner-only modes for everything the copies above created — copy2
        # preserves Chrome's 0644 and sqlite backup files land umask-wide;
        # these are the user's session cookies (#96729). Runs AFTER the marker
        # write so the marker itself is covered, and on every pass so
        # snapshots from older builds heal on their next launch.
        _secure_snapshot_contents(dst)
    except OSError as e:
        return None, f"could not snapshot the '{browser}' profile into {dst}: {e}"
    return dst, None


def cleanup_real_profile_snapshots() -> None:
    """Delete the whole real-profile snapshot store (all copied credentials).

    Called when consent is OFF: the copied Cookies / Login Data must not outlive the toggle. Best-
    effort and idempotent — missing dir is fine.
    """
    root = str(get_hermes_home() / "browser-profile")
    try:
        if os.path.isdir(root):
            shutil.rmtree(root, ignore_errors=True)
            logger.info("real-profile: removed snapshot store %s (consent off)", root)
    except OSError as e:
        logger.debug("real-profile cleanup failed for %s: %s", root, e)


def get_chrome_debug_candidates(system: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def add(path: str | None) -> None:
        if not path:
            return
        normalized = os.path.normcase(os.path.normpath(path))
        if normalized in seen or not os.path.isfile(path):
            return
        candidates.append(path)
        seen.add(normalized)

    if system == "Darwin":
        for b in _BROWSERS:
            add(b.mac_app)
        return candidates

    if system == "Windows":
        install_bases = (
            os.environ.get("ProgramFiles"),
            os.environ.get("ProgramFiles(x86)"),
            os.environ.get("LOCALAPPDATA"),
        )
        for b in _BROWSERS:
            for name in b.win_bins:
                add(shutil.which(name))
            for base in filter(None, install_bases):
                for parts in b.win_install:
                    add(os.path.join(base, *parts))
        return candidates

    for b in _BROWSERS:
        for name in b.linux_bins:
            add(shutil.which(name))
        for path in b.linux_paths:
            add(path)
    # WSL: Windows installs under ``/mnt/c/...`` are POSIX paths regardless of
    # the host OS, so join with posixpath (os.path.join would emit backslashes on nt).
    for b in _BROWSERS:
        for base in ("/mnt/c/Program Files", "/mnt/c/Program Files (x86)"):
            for parts in b.win_install:
                add(posixpath.join(base, *parts))
    return candidates


def chrome_debug_data_dir() -> str:
    return str(get_hermes_home() / "chrome-debug")


def _chrome_debug_args(port: int) -> list[str]:
    return [
        f"--remote-debugging-port={port}",
        f"--user-data-dir={chrome_debug_data_dir()}",
        "--no-first-run",
        "--no-default-browser-check",
    ]


def is_browser_debug_ready(url: str, timeout: float = 1.0) -> bool:
    """Return True when ``url`` exposes a reachable Chrome DevTools endpoint."""
    import socket
    import urllib.request
    from urllib.parse import urlparse

    parsed = urlparse(url if "://" in url else f"http://{url}")
    try:
        port = parsed.port or (443 if parsed.scheme in {"https", "wss"} else 80)
    except ValueError:
        return False

    if parsed.scheme in {"ws", "wss"} and parsed.path.startswith("/devtools/browser/"):
        if not parsed.hostname:
            return False
        try:
            with socket.create_connection((parsed.hostname, port), timeout=timeout):
                return True
        except OSError:
            return False

    scheme = {"ws": "http", "wss": "https"}.get(parsed.scheme, parsed.scheme)
    if scheme not in {"http", "https"} or not parsed.netloc:
        return False

    root = f"{scheme}://{parsed.netloc}".rstrip("/")
    for probe in (f"{root}/json/version", f"{root}/json"):
        try:
            with urllib.request.urlopen(probe, timeout=timeout) as resp:
                if 200 <= getattr(resp, "status", 200) < 300:
                    return True
        except Exception:
            continue
    return False


# Both loopback literals: Windows (and some Linux setups) can hand the IPv4
# loopback to one process and the IPv6 loopback to another. Chrome asked to
# bind :9222 while e.g. VS Code's js-debug holds 127.0.0.1:9222 will come up
# on [::1]:9222 only — reachable, but invisible to an IPv4-only probe.
_LOOPBACK_PROBE_HOSTS = ("127.0.0.1", "[::1]")
_LOOPBACK_SOCKET_HOSTS = ("127.0.0.1", "::1")


def discover_local_cdp_url(port: int, timeout: float = 1.0) -> str | None:
    """Return the first loopback URL (IPv4 first, then IPv6) speaking CDP.

    Dual-stack discovery: when another application squats the IPv4 loopback on ``port``, a debug
    browser launched with ``--remote-debugging-port`` may bind only ``[::1]``. Probing both literals
    finds it either way. Returns ``None`` when neither loopback exposes a CDP discovery endpoint.
    """
    for host in _LOOPBACK_PROBE_HOSTS:
        url = f"http://{host}:{port}"
        if is_browser_debug_ready(url, timeout=timeout):
            return url
    return None


def local_port_in_use(port: int, timeout: float = 0.5) -> bool:
    """Return True when either loopback accepts TCP on ``port``.

    Callers use this AFTER a failed CDP probe to distinguish "port is free, we can launch a browser
    on it" from "another application (IDE debugger, dev server) is squatting the port and a launch
    would fight it".
    """
    import socket

    for host in _LOOPBACK_SOCKET_HOSTS:
        try:
            with socket.create_connection((host, port), timeout=timeout):
                return True
        except OSError:
            continue
    return False


def find_free_debug_port(preferred: int = DEFAULT_BROWSER_CDP_PORT, attempts: int = 10) -> int:
    """Return the first port after ``preferred`` bindable on both loopbacks.

    Used when ``preferred`` is occupied by a non-CDP application: rather than launching a browser
    into a bind conflict, pick a nearby free port. Falls back to ``preferred + 1`` if nothing binds
    (the launch will then fail with a clear browser-side error instead of silently doing nothing).
    """
    import socket

    for port in range(preferred + 1, preferred + 1 + attempts):
        bindable = True
        for family, host in ((socket.AF_INET, "127.0.0.1"), (socket.AF_INET6, "::1")):
            try:
                with socket.socket(family, socket.SOCK_STREAM) as sock:
                    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                    sock.bind((host, port))
            except OSError:
                bindable = False
                break
        if bindable:
            return port
    return preferred + 1


def manual_chrome_debug_command(port: int = DEFAULT_BROWSER_CDP_PORT, system: str | None = None) -> str | None:
    system = system or platform.system()
    candidates = get_chrome_debug_candidates(system)

    if candidates:
        argv = [candidates[0], *_chrome_debug_args(port)]
        return subprocess.list2cmdline(argv) if system == "Windows" else shlex.join(argv)

    if system == "Darwin":
        data_dir = chrome_debug_data_dir()
        return (
            f'open -a "Google Chrome" --args --remote-debugging-port={port} '
            f'--user-data-dir="{data_dir}" --no-first-run --no-default-browser-check'
        )

    return None


def _detach_kwargs(system: str) -> dict:
    if system != "Windows":
        return {"start_new_session": True}
    flags = getattr(subprocess, "DETACHED_PROCESS", 0) | getattr(
        subprocess, "CREATE_NEW_PROCESS_GROUP", 0
    )
    return {"creationflags": flags} if flags else {}


def _wait_for_browser_debug_ready_or_exit(
    proc: subprocess.Popen,
    port: int,
    timeout: float = 2.0,
    interval: float = 0.1,
) -> str:
    """Classify a launched browser as ready, exited, or still starting.

    We only need to wait long enough to catch the common failure mode where a candidate binary
    exists but exits immediately before exposing the CDP port. Slower browsers can still finish
    starting after this grace window.
    """
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        # Dual-stack: a squatter on the IPv4 loopback can push the browser
        # to bind [::1] only — check both so a successful launch is seen.
        if discover_local_cdp_url(port, timeout=min(interval, 0.2)):
            return "ready"
        if proc.poll() is not None:
            return "exited"
        time.sleep(interval)

    return "starting"


_LAUNCH_STDERR_LOG = "launch-stderr.log"
_STDERR_TAIL_LIMIT = 2000


@dataclass
class LaunchAttempt:
    """Outcome of one candidate-binary launch attempt."""

    binary: str
    state: str  # "ready" | "starting" | "exited" | "spawn-failed"
    returncode: int | None = None
    stderr_tail: str = ""


@dataclass
class ChromeDebugLaunch:
    """Structured result of ``launch_chrome_debug``.

    ``launched`` mirrors the legacy boolean contract: a launch command was executed and the browser
    is ready or still starting (it does NOT guarantee the CDP port ever opens). ``attempts`` carries
    per-candidate diagnostics so callers can explain *why* nothing came up.
    """

    launched: bool = False
    attempts: list[LaunchAttempt] = field(default_factory=list)

    @property
    def hint(self) -> str | None:
        """Best user-facing explanation for a failed/soft launch, if any."""
        for attempt in self.attempts:
            if attempt.state == "exited" and attempt.returncode == 0:
                name = os.path.basename(attempt.binary)
                return (
                    f"{name} exited immediately without opening the debug port — an already-running "
                    f"{name} instance likely absorbed the launch (Chromium's single-instance "
                    "behavior). Close ALL of its processes (including background/tray instances) "
                    "and retry /browser connect."
                )
        for attempt in self.attempts:
            if attempt.state == "exited" and attempt.stderr_tail:
                return (
                    f"{os.path.basename(attempt.binary)} exited before the debug port opened: "
                    f"{attempt.stderr_tail.splitlines()[-1].strip()}"
                )
        return None


def _read_stderr_tail(path: str) -> str:
    try:
        with open(path, "rb") as fh:
            data = fh.read()
        return data[-_STDERR_TAIL_LIMIT:].decode("utf-8", errors="replace").strip()
    except OSError:
        return ""


def launch_chrome_debug(
    port: int = DEFAULT_BROWSER_CDP_PORT, system: str | None = None
) -> ChromeDebugLaunch:
    """Launch a Chromium-family browser with remote debugging, with diagnostics.

    Tries each detected candidate binary in turn. A candidate that exits before the CDP port opens
    (crash, singleton forward to an existing instance, bad profile dir) is logged — with exit code
    and a stderr tail — and the next candidate is tried.
    """
    system = system or platform.system()
    result = ChromeDebugLaunch()
    candidates = get_chrome_debug_candidates(system)
    if not candidates:
        logger.info("browser debug launch: no Chromium-family binary found (system=%s)", system)
        return result

    data_dir = chrome_debug_data_dir()
    os.makedirs(data_dir, exist_ok=True)
    stderr_path = os.path.join(data_dir, _LAUNCH_STDERR_LOG)

    for candidate in candidates:
        try:
            with open(stderr_path, "wb") as stderr_file:
                proc = subprocess.Popen(
                    [candidate, *_chrome_debug_args(port)],
                    stdout=subprocess.DEVNULL,
                    stderr=stderr_file,
                    **_detach_kwargs(system),
                )
        except Exception as exc:
            result.attempts.append(LaunchAttempt(binary=candidate, state="spawn-failed"))
            logger.info("browser debug launch: failed to spawn %s: %s", candidate, exc)
            continue

        logger.info(
            "browser debug launch: spawned %s (pid=%s) with --remote-debugging-port=%d",
            candidate,
            getattr(proc, "pid", None),
            port,
        )
        state = _wait_for_browser_debug_ready_or_exit(proc, port)
        attempt = LaunchAttempt(binary=candidate, state=state)
        result.attempts.append(attempt)

        if state != "exited":
            result.launched = True
            return result

        attempt.returncode = getattr(proc, "returncode", None)
        attempt.stderr_tail = _read_stderr_tail(stderr_path)
        logger.warning(
            "browser debug launch: %s exited (code=%s) before port %d opened%s",
            candidate,
            attempt.returncode,
            port,
            f"; stderr tail: {attempt.stderr_tail}" if attempt.stderr_tail else "",
        )

    return result


def try_launch_chrome_debug(port: int = DEFAULT_BROWSER_CDP_PORT, system: str | None = None) -> bool:
    return launch_chrome_debug(port, system).launched
