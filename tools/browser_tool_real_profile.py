"""Real-profile local browsing: snapshot the user's default Chromium profile into a
hermes-owned copy, launch the real browser binary on it, and attach agent-browser.

State (``_REAL_PROFILE_SESSION``, ``_real_profile_cdp_lock``, ``_real_profile_cdp_cache``,
``_real_profile_chrome_procs``) lives in ``tools.browser_tool``. Origin-module symbols are
resolved lazily through ``tools.browser_tool`` (``_bt``) so ``patch("tools.browser_tool.X")``
keeps working; never import ``tools.browser_tool`` at import time (cycle).
"""

import os
import re
import subprocess
import sys
import time
from typing import Optional, Tuple
from tools.browser_tool_origin import origin_module as _origin


def _terminate_real_profile_chrome() -> None:
    """Terminate real-browser processes launched for real-profile sessions (idempotent, atexit-safe).

    agent-browser only ATTACHED to them, so its own session cleanup never kills them.
    """
    _bt = _origin()
    while _bt._real_profile_chrome_procs:
        proc = _bt._real_profile_chrome_procs.pop()
        try:
            if proc.poll() is None:
                proc.terminate()
                try:
                    proc.wait(timeout=5)
                except Exception:
                    proc.kill()
        except Exception as e:
            _bt.logger.debug("real-profile chrome terminate failed: %s", e)


def _cdp_http_ready(http_cdp: str) -> bool:
    """True when an ``http://host:port`` CDP discovery root answers."""
    try:
        from hermes_cli.browser_connect import is_browser_debug_ready

        return is_browser_debug_ready(http_cdp, timeout=1.0)
    except Exception:
        return False


def _agent_browser_get_cdp(session_name: str) -> Optional[str]:
    """HTTP CDP discovery root of an agent-browser session (converted from its
    ``ws://`` cdp-url), or None when it isn't running / unparsable."""
    _bt = _origin()
    try:
        browser_cmd = _bt._find_agent_browser()
    except FileNotFoundError:
        return None
    try:
        proc = subprocess.run(
            [*_bt._agent_browser_argv(browser_cmd), "--session", session_name, "get", "cdp-url"],
            capture_output=True, text=True, timeout=15, env=_bt._build_browser_env(),
        )
    except (subprocess.SubprocessError, OSError) as e:
        _bt.logger.debug("real-profile get cdp-url failed: %s", e)
        return None
    out = (proc.stdout or "").strip()
    m = re.search(r"ws://127\.0\.0\.1:(\d+)/", out)
    if not m:
        return None
    return f"http://127.0.0.1:{m.group(1)}"


def _cdp_on_data_dir(http_cdp: str, data_dir: str) -> bool:
    """True when the CDP endpoint's browser runs on ``data_dir``.

    Chrome writes its live debug port to ``DevToolsActivePort`` in the
    user-data-dir; a port match proves the browser is our profile copy, not a
    throwaway temp dir a raced/stale launch fell back to.
    """
    m = re.search(r":(\d+)", http_cdp or "")
    if not m:
        return False
    try:
        with open(os.path.join(data_dir, "DevToolsActivePort"), encoding="utf-8") as fh:
            port_line = fh.readline().strip()
        return port_line == m.group(1)
    except OSError:
        return False


def _agent_browser_close_session(session_name: str) -> None:
    """Best-effort close of an agent-browser session (stale/wrong-dir cleanup)."""
    _bt = _origin()
    try:
        browser_cmd = _bt._find_agent_browser()
    except FileNotFoundError:
        return
    try:
        subprocess.run(
            [*_bt._agent_browser_argv(browser_cmd), "--session", session_name, "close"],
            capture_output=True, text=True, timeout=15, env=_bt._build_browser_env(),
        )
    except (subprocess.SubprocessError, OSError) as e:
        _bt.logger.debug("real-profile session close failed: %s", e)


_REAL_PROFILE_CHROME_FLAGS = (
    "--remote-debugging-port=0",
    "--no-first-run",
    "--no-default-browser-check",
    "--disable-background-networking",
    "--disable-component-update",
    "--disable-default-apps",
    "--disable-hang-monitor",
    "--disable-popup-blocking",
    "--disable-prompt-on-repost",
    "--disable-sync",
    "--disable-features=Translate",
    "--no-startup-window",
)


def _real_profile_unsupported_reason(browser) -> Optional[str]:
    """Fail-closed message when the detected default browser can't be used, else None.

    A recognized pre-release channel (Beta/Dev/Canary) lives in a
    channel-specific profile dir we don't resolve; normalizing it to the stable
    family would drive a DIFFERENT profile/account (wrong-principal bug), so
    refuse rather than guess.
    """
    from hermes_cli.browser_connect import UNSUPPORTED_CHANNEL

    if browser is None:
        return (
            "browser.use_real_profile is on, but your default browser is not a "
            "supported Chromium browser (Chrome, Edge, Brave, Brave Origin, "
            "Chromium). "
            "Real-profile browsing requires a Chromium default; set one or turn "
            "the toggle off."
        )
    if browser == UNSUPPORTED_CHANNEL:
        return (
            "browser.use_real_profile is on, but your default browser is a "
            "pre-release Chromium channel (Beta / Dev / Canary), which "
            "real-profile browsing does not support. Set your default to a "
            "stable Chrome / Edge / Brave / Brave Origin / Chromium, or turn "
            "the toggle off."
        )
    return None


def _real_profile_snapshot_error(err: str) -> str:
    """User-facing message for a failed profile snapshot.

    A locked profile surfaces the guidance verbatim (it already says whether
    closing is armed) plus the exact approved-close command; the agent must ASK
    the user before running it — it quits their browser.
    """
    from hermes_cli.browser_connect import _PROFILE_LOCKED_PREFIX

    if err and err.startswith(_PROFILE_LOCKED_PREFIX):
        body = err[len(_PROFILE_LOCKED_PREFIX):]
        return (
            body + " To close it (only after the user approves — it "
            "quits their browser and loses unsaved tabs), run: "
            "`hermes browser close-profile`, then retry."
        )
    return f"browser.use_real_profile is on, but {err}"


def _launch_real_profile_chrome(real_binary: str, copy_dir: str) -> Tuple[Optional[int], Optional[str]]:
    """Launch the user's REAL browser binary on the profile COPY; return (debug_port, error).

    agent-browser's own launch path force-adds --use-mock-keychain /
    --password-store=basic, which makes macOS Chrome drop every
    keychain-encrypted cookie — the copy would launch signed out. Launching the
    real binary ourselves with NO mock-keychain switches keeps the OS keychain
    path intact; agent-browser attaches afterwards via ``--cdp <port>``.

    Headless by default: real-profile browsing is a background capability and a
    focus-stealing window defeats it. Chrome's NEW headless mode shares the
    profile's normal cookie store (legacy --headless does not), and cookie
    decryption is unaffected by headless (the drop comes from mock-keychain, not
    headless). Users opt into a window via browser.headed / AGENT_BROWSER_HEADED;
    on a display-less Linux host we force headless regardless so the launch
    doesn't die at startup. Waits for Chrome to write DevToolsActivePort.
    """
    _bt = _origin()
    port_file = os.path.join(copy_dir, "DevToolsActivePort")
    try:
        os.unlink(port_file)  # stale port from a previous launch confuses reuse probes
    except OSError:
        pass
    chrome_argv = [real_binary, f"--user-data-dir={copy_dir}", *_REAL_PROFILE_CHROME_FLAGS]
    _has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    _want_headed = _bt._is_headed_mode() and (_has_display or not sys.platform.startswith("linux"))
    if not _want_headed:
        chrome_argv.append("--headless=new")
    try:
        chrome_proc = subprocess.Popen(
            chrome_argv,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
            env=_bt._build_browser_env(),
        )
    except (subprocess.SubprocessError, OSError) as e:
        return None, f"browser.use_real_profile is on, but the launch failed: {e}"
    _bt._real_profile_chrome_procs.append(chrome_proc)

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        try:
            with open(port_file, encoding="utf-8") as fh:
                line = fh.readline().strip()
            if line.isdigit():
                return int(line), None
        except OSError:
            pass
        if chrome_proc.poll() is not None:
            _bt._terminate_real_profile_chrome()
            return None, (
                "browser.use_real_profile is on, but Chrome exited during "
                "startup (another instance may hold the profile copy)."
            )
        time.sleep(0.25)
    _bt._terminate_real_profile_chrome()
    return None, (
        "browser.use_real_profile is on, but the real-profile browser "
        "did not expose a debug port in time. Retry, or turn the toggle off."
    )


def _attach_agent_browser_to_real_profile(port: int, copy_dir: str) -> Tuple[Optional[str], Optional[str]]:
    """Make agent-browser ATTACH to the running Chrome (never launch its own).

    Returns ``(http_cdp, error)``. The daemon may answer with the endpoint of a
    browser IT spawned (throwaway temp profile) instead of the Chrome we
    launched; the DevToolsActivePort file OUR Chrome wrote is authoritative, so
    on disagreement we trust ours.
    """
    _bt = _origin()
    try:
        browser_cmd = _bt._find_agent_browser()
    except FileNotFoundError as e:
        return None, (
            "browser.use_real_profile is on, but the local browser engine "
            f"(agent-browser) is not installed: {e}"
        )
    argv = [
        *_bt._agent_browser_argv(browser_cmd),
        "--session", _bt._REAL_PROFILE_SESSION,
        "--cdp", str(port),
        "open", "about:blank",
    ]
    try:
        proc = subprocess.run(
            argv, capture_output=True, text=True,
            timeout=_bt._get_open_command_timeout(first_open=True),
            env=_bt._build_browser_env(),
        )
    except subprocess.TimeoutExpired:
        return None, (
            "browser.use_real_profile is on, but the real-profile browser "
            "took too long to start. Retry, or turn the toggle off."
        )
    except (subprocess.SubprocessError, OSError) as e:
        return None, f"browser.use_real_profile is on, but the launch failed: {e}"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()
        reason = tail[-1] if tail else f"exit {proc.returncode}"
        return None, (
            f"browser.use_real_profile is on, but the real-profile browser "
            f"failed to start: {reason}"
        )

    cdp = _bt._agent_browser_get_cdp(_bt._REAL_PROFILE_SESSION)
    try:
        with open(os.path.join(copy_dir, "DevToolsActivePort"), encoding="utf-8") as fh:
            our_port = fh.readline().strip()
        m = re.search(r":(\d+)", cdp or "")
        if m and m.group(1) != our_port:
            cdp = f"http://127.0.0.1:{our_port}"
    except (OSError, ValueError):
        pass
    if not cdp:
        return None, (
            "browser.use_real_profile is on, but the real-profile browser "
            "started without exposing a devtools endpoint. Retry, or turn "
            "the toggle off."
        )
    return cdp, None


def _real_profile_cdp() -> tuple:
    """Resolve ``(cdp_url, error)`` for consented real-profile browsing.

    Snapshots the user's default-Chromium profile into a hermes-owned copy
    (auth/login state only), launches the real browser binary on that copy and
    returns the HTTP CDP endpoint for agent-browser / the browser-use harness to
    attach to. The copy is a non-default dir, so it sidesteps the Chrome ≥136
    default-profile remote-debugging block and never contends with the user's
    running browser.

    A single shared agent-browser session is reused across calls (its CDP URL
    is cached and re-validated). Returns ``(None, message)`` fail-closed when
    the default browser is non-Chromium or the snapshot/launch fails;
    ``(None, None)`` when consent is off.
    """
    _bt = _origin()
    if not _bt._use_real_profile():
        # Consent is off. A snapshot store from a previous consented run holds
        # copies of the user's cookies/logins — delete it so revoking consent
        # actually removes the credential copies. Cheap and idempotent.
        try:
            from hermes_cli.browser_connect import cleanup_real_profile_snapshots

            cleanup_real_profile_snapshots()
        except Exception as e:
            _bt.logger.debug("real-profile cleanup-on-consent-off failed: %s", e)
        _bt._real_profile_cdp_cache.pop("cdp", None)
        return None, None

    # Lightpanda rejects ``--profile`` outright. Detect it BEFORE default-browser
    # detection so even a host with no Chromium default reports the actionable
    # conflict (the engine setting) rather than a generic launch failure.
    if _bt._using_lightpanda_engine():
        return None, (
            "browser.use_real_profile is on, but browser.engine is set to "
            "'lightpanda', which cannot load a real Chromium profile. Set "
            "browser.engine to 'auto' or 'chrome' to use real-profile browsing, "
            "or turn the toggle off."
        )

    from hermes_cli.browser_connect import (
        chromium_executable, detect_default_chromium, real_profile_copy_dir, snapshot_real_profile
    )

    with _bt._real_profile_cdp_lock:
        # Reuse a live copy-browser from an earlier call this process made.
        cached = _bt._real_profile_cdp_cache.get("cdp")
        if cached and _bt._cdp_http_ready(cached):
            return cached, None
        _bt._real_profile_cdp_cache.pop("cdp", None)

        browser = detect_default_chromium()
        unsupported = _real_profile_unsupported_reason(browser)
        if unsupported:
            return None, unsupported

        # Reuse BEFORE writing anything. A shared copy-browser may already be up
        # from a previous hermes process; if it is driving OUR copy dir, hand it
        # back untouched. CRITICAL: the snapshot overlay (which truncates and
        # rewrites Cookies / Login Data) must NOT run while that browser holds
        # the user-data-dir open — doing so corrupts the live databases. So
        # resolve the copy dir as a PATH only (no copy), probe reuse, and return
        # early on a hit; the overlay happens solely on the relaunch path below.
        copy_dir = real_profile_copy_dir(browser)
        existing = _bt._agent_browser_get_cdp(_bt._REAL_PROFILE_SESSION)
        if existing and _bt._cdp_http_ready(existing) and _bt._cdp_on_data_dir(existing, copy_dir):
            _bt._real_profile_cdp_cache["cdp"] = existing
            return existing, None
        if existing:
            # Stale/wrong-dir session (throwaway-temp fallback, or an old copy):
            # close it so nothing holds the dir open before we overlay + relaunch.
            _bt._agent_browser_close_session(_bt._REAL_PROFILE_SESSION)

        # No live browser owns the dir now — safe to (re)snapshot + overlay.
        snap_dir, err = snapshot_real_profile(browser)
        if err or not snap_dir:
            return None, _real_profile_snapshot_error(err)
        copy_dir = snap_dir

        real_binary = chromium_executable(browser)
        if real_binary is None:
            return None, (
                "browser.use_real_profile is on, but the real browser binary for "
                f"'{browser}' could not be found. Reinstall it or turn the toggle off."
            )
        port, err = _launch_real_profile_chrome(real_binary, copy_dir)
        if port is None:
            return None, err
        cdp, err = _attach_agent_browser_to_real_profile(port, copy_dir)
        if not cdp:
            return None, err
        _bt._real_profile_cdp_cache["cdp"] = cdp
        _bt.logger.info("real-profile browser ready for %s at %s (%s)", browser, cdp, copy_dir)
        return cdp, None
