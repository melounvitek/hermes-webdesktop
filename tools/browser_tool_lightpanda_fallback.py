"""Lightpanda engine status and the automatic Chrome fallback for browser commands
that Lightpanda cannot serve (screenshots, empty snapshots, failed commands).

Origin-module symbols are resolved lazily through ``tools.browser_tool`` (``_bt``)
so ``patch("tools.browser_tool.X")`` keeps working; never import ``tools.browser_tool``
at import time (cycle).
"""

import json
import os
import shutil
import subprocess
from typing import Any, Dict, List, Optional, Tuple
from tools.browser_tool_origin import origin_module as _origin


def _using_lightpanda_engine() -> bool:
    """Return True when local browser commands are configured for Lightpanda."""
    _bt = _origin()
    return _bt._get_browser_engine() == "lightpanda"


def lightpanda_engine_status() -> Tuple[bool, str]:
    """Whether ``browser.engine: lightpanda`` is actually in effect, and why.

    ``(False, "")`` when the engine isn't lightpanda; otherwise the reason names
    the setting shadowing it or the driver running it. Mirrors the precedence
    of ``_should_inject_engine`` / ``browser_use_cli._resolve_backend_cdp``
    with config-only gates (no network I/O) so ``/browser status`` and
    ``hermes doctor`` can call it.
    """
    _bt = _origin()
    if not _bt._using_lightpanda_engine():
        return False, ""
    if _bt._get_cdp_override_raw():
        return False, "a CDP override is active (/browser connect or browser.cdp_url)"
    if _bt._is_camofox_mode():
        return False, "Camofox is the selected browser (CAMOFOX_URL)"
    # Real-profile is checked before the cloud provider: in browser_exec the
    # real-profile resolution runs before backend resolution, so with both
    # set it is the real-profile toggle that actually claims the session.
    if _bt._use_real_profile():
        return False, "browser.use_real_profile is on (Lightpanda cannot load a Chromium profile)"
    try:
        provider = _bt._get_cloud_provider()
    except Exception:
        provider = None
    if provider is not None:
        try:
            name = provider.provider_name()
        except Exception:
            name = type(provider).__name__
        return False, (
            f"cloud provider {name} is selected (browser.cloud_provider, or "
            "auto-detected from credentials)"
        )
    bu_mode = _bt._is_browser_use_cli_mode()
    if bu_mode:
        try:
            from tools.browser_use_cli import (
                _read_browser_cfg,
                is_legacy_browser_use_cloud_config,
            )

            if is_legacy_browser_use_cloud_config(_read_browser_cfg()):
                return False, "Browser Use cloud (BROWSER_USE_API_KEY) is selected"
        except Exception as e:
            _bt.logger.debug("legacy Browser Use cloud check failed: %s", e)
    if bu_mode:
        return True, "Browser Use mode: Hermes spawns `lightpanda serve` per session"
    return True, "built-in browser tools: agent-browser --engine lightpanda"


def _lightpanda_fallback_reason(engine: str, command: str, result: Dict[str, Any]) -> Optional[str]:
    """User-visible reason a Lightpanda result needs the Chrome fallback, or None.

    The string is copied into the fallback result so users can see when Hermes
    silently switched engines.
    """
    _bt = _origin()
    if engine != "lightpanda":
        return None

    # Only retry commands where Chrome can meaningfully produce a different
    # result. Session-management commands (close, record) are tied to the
    # engine's daemon and can't be retried on a different engine.
    _FALLBACK_ELIGIBLE = {"open", "snapshot", "screenshot", "eval", "click",
                          "fill", "scroll", "back", "press", "console", "errors"}
    if command not in _FALLBACK_ELIGIBLE:
        return None

    # Explicit failure
    if not result.get("success"):
        error = str(result.get("error") or "command failed").strip()
        return f"Lightpanda {command!r} failed ({error}); retried with Chrome."

    data = result.get("data", {})

    if command == "snapshot":
        snap = data.get("snapshot", "")
        # Empty or near-empty snapshots indicate Lightpanda couldn't render
        if not snap or len(snap.strip()) < 20:
            return "Lightpanda returned an empty/too-short snapshot; retried with Chrome."

    if command == "screenshot":
        # Lightpanda returns a placeholder PNG with its panda logo.
        # Since Lightpanda resized it to 1920x1080, the placeholder is
        # ~17 KB.  Real Chromium screenshots are typically 100 KB+.
        path = data.get("path", "")
        if path:
            try:
                size = os.path.getsize(path)
                if size < 20480:
                    _bt.logger.debug("Lightpanda screenshot is suspiciously small (%d bytes), "
                                 "triggering Chrome fallback", size)
                    return (
                        f"Lightpanda screenshot was suspiciously small ({size} bytes); "
                        "retried with Chrome."
                    )
            except OSError:
                return "Lightpanda screenshot file was missing/unreadable; retried with Chrome."

    return None


def _needs_lightpanda_fallback(engine: str, command: str, result: Dict[str, Any]) -> bool:
    """Check if a Lightpanda result should trigger an automatic Chrome fallback."""
    return _lightpanda_fallback_reason(engine, command, result) is not None


def _annotate_lightpanda_fallback(result: Dict[str, Any], reason: str) -> Dict[str, Any]:
    """Add a user-visible Chrome fallback warning to a browser command result."""
    warning = (
        "⚠ Lightpanda fallback: Chrome was used for this browser action. "
        f"{reason}"
    )
    annotated = dict(result)
    annotated["fallback_warning"] = warning
    annotated["browser_engine"] = "chrome"
    annotated["browser_engine_fallback"] = {
        "from": "lightpanda",
        "to": "chrome",
        "reason": reason,
    }
    data = annotated.get("data")
    if isinstance(data, dict):
        data = dict(data)
        data.setdefault("fallback_warning", warning)
        data.setdefault("browser_engine", "chrome")
        data.setdefault(
            "browser_engine_fallback",
            {"from": "lightpanda", "to": "chrome", "reason": reason},
        )
        annotated["data"] = data
    return annotated


def _copy_fallback_warning(target: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
    """Copy browser fallback metadata from an internal result into a tool response."""
    if result.get("fallback_warning"):
        target["fallback_warning"] = result["fallback_warning"]
        target["browser_engine"] = result.get("browser_engine")
        target["browser_engine_fallback"] = result.get("browser_engine_fallback")
    return target


def _run_chrome_fallback_command(
    task_id: str,
    command: str,
    args: List[str],
    timeout: int,
) -> Dict[str, Any]:
    """Run a browser command in a temporary Chrome session at the current URL.

    agent-browser locks the engine when a named daemon starts, so ``--engine
    chrome`` on the Lightpanda session is ignored: use a fresh temp Chrome
    session, navigate it to the current URL, run ``command``, tear it down.
    """
    _bt = _origin()
    import uuid

    # 1. Grab the current URL from the Lightpanda session. ``get url`` is not
    # fallback-eligible, so an error cannot recursively trigger this helper.
    # Keep the explicit Lightpanda override so Chromium-only environment flags
    # are stripped while querying the already-running Lightpanda daemon.
    url_result = _bt._run_browser_command(
        task_id, "get", ["url"], timeout=10, _engine_override="lightpanda"
    )
    current_url = None
    if url_result.get("success"):
        current_url = str(url_result.get("data", {}).get("url", "")).strip()
    if not current_url:
        _bt.logger.warning("Chrome fallback: could not determine current URL from LP session")
        return {"success": False, "error": "Chrome fallback failed: could not determine current URL"}

    # 2. Create a temporary Chrome session (bypasses _get_session_info's cache).
    tmp_session = f"h_cfb_{uuid.uuid4().hex[:8]}"
    try:
        browser_cmd = _bt._find_agent_browser()
    except FileNotFoundError as e:
        return {"success": False, "error": str(e)}

    if not _bt._chromium_installed():
        if _bt._running_in_docker():
            hint = (
                "Chrome fallback requires Chromium, but it is missing. "
                "You're running in Docker — pull the latest image: "
                "docker pull ghcr.io/nousresearch/hermes-agent:latest"
            )
        else:
            hint = (
                "Chrome fallback requires Chromium, but it is missing. Install it with: "
                "npx agent-browser install --with-deps "
                "(or: npx playwright install --with-deps chromium)"
            )
        return {"success": False, "error": hint}

    base_args = _bt._agent_browser_argv(browser_cmd) + ["--engine", "chrome", "--session", tmp_session, "--json"]
    task_socket_dir = _bt._prepare_session_socket_dir(tmp_session)
    # Bypasses _run_browser_command, so apply the same Chromium sandbox policy explicitly.
    browser_env = _bt._agent_browser_command_env(task_socket_dir)
    _bt._apply_chromium_sandbox_args(browser_env)

    def _run_tmp(cmd: str, cmd_args: List[str]) -> Dict[str, Any]:
        proc = _bt._popen_agent_browser(base_args + [cmd] + cmd_args, browser_env, task_socket_dir, cmd)
        stdout_path = os.path.join(task_socket_dir, f"_stdout_{cmd}")
        stderr_path = os.path.join(task_socket_dir, f"_stderr_{cmd}")
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            return {"success": False, "error": f"Chrome fallback '{cmd}' timed out"}
        try:
            with open(stdout_path, "r", encoding="utf-8") as f:
                stdout = f.read().strip()
            if stdout:
                return json.loads(stdout.split("\n")[-1])
        except Exception as exc:
            _bt.logger.debug("Chrome fallback tmp cmd '%s' error: %s", cmd, exc)
        finally:
            _bt._unlink_command_output_files(stdout_path, stderr_path)
        return {"success": False, "error": f"Chrome fallback '{cmd}' failed"}

    try:
        # 3. Navigate Chrome to the same URL, then 4. run the requested command.
        nav = _run_tmp("open", [current_url])
        if not nav.get("success"):
            _bt.logger.warning("Chrome fallback: navigate failed: %s", nav.get("error"))
            return {"success": False, "error": f"Chrome fallback navigate failed: {nav.get('error')}"}
        return _run_tmp(command, args)
    finally:
        # 5. Tear down the temporary Chrome session and its socket directory.
        try:
            _run_tmp("close", [])
        except Exception:
            pass
        shutil.rmtree(task_socket_dir, ignore_errors=True)


def _chrome_fallback_screenshot(
    task_id: str,
    args: List[str],
    timeout: int,
) -> Dict[str, Any]:
    """Take a screenshot using a temporary Chrome session."""
    _bt = _origin()
    return _bt._run_chrome_fallback_command(task_id, "screenshot", args, timeout)
