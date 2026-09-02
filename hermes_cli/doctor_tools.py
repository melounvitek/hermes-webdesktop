"""External-tool checks for hermes doctor: terminal backends, git/rg, Node + agent-browser, npm audit, tool availability.

Split out of ``hermes_cli/doctor.py``; every moved name is re-imported there, so
``hermes_cli.doctor.<name>`` keeps resolving (and monkeypatching) as before.
"""

from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from hermes_cli.doctor_platform import _system_package_install_cmd
from hermes_cli.doctor_report import Finding, _fail_and_issue, check_info, check_ok, check_warn
from hermes_cli.vercel_auth import describe_vercel_auth
from hermes_constants import agent_browser_runnable, is_termux as _is_termux


def _safe_which(cmd: str) -> str | None:
    """shutil.which wrapper resilient to platform monkeypatching in tests."""
    try:
        return shutil.which(cmd)
    except Exception:
        return None


def _termux_browser_setup_steps(node_installed: bool) -> list[str]:
    steps: list[str] = []
    step = 1
    if not node_installed:
        steps.append(f"{step}) pkg install nodejs")
        step += 1
    steps.append(f"{step}) npm install -g agent-browser")
    steps.append(f"{step + 1}) agent-browser install")
    return steps


def _termux_install_all_fallback_notes() -> list[str]:
    return [
        "Termux install profile: use .[termux-all] for broad compatibility (installer default on Termux).",
        "Matrix E2EE extra is excluded on Termux (python-olm currently fails to build).",
        "Local faster-whisper extra is excluded on Termux (ctranslate2/av build path unavailable).",
        "STT fallback: use Groq Whisper (set GROQ_API_KEY) or OpenAI Whisper (set VOICE_TOOLS_OPENAI_KEY).",
    ]


def _is_kanban_worker_env_gate(item: dict) -> bool:
    """Return True when Kanban is unavailable only because this is not a worker process."""
    if item.get("name") != "kanban":
        return False
    if os.environ.get("HERMES_KANBAN_TASK"):
        return False

    tools = item.get("tools") or []
    return bool(tools) and all(str(tool).startswith("kanban_") for tool in tools)


def _doctor_tool_availability_detail(toolset: str) -> str:
    """Optional explanatory suffix for toolsets whose doctor status needs context."""
    if toolset == "kanban" and not os.environ.get("HERMES_KANBAN_TASK"):
        return "(runtime-gated; loaded only for dispatcher-spawned workers)"
    return ""


def _doctor_web_capability_rows() -> list[tuple[str, str, str]]:
    """Return doctor rows for web search/extract provider readiness (#78412).

    Each row is ``(status, label, detail)`` where *status* is ``ok`` or ``warn``.
    Uses the same active-provider resolvers as the tools, but reports readiness
    from ``is_available()`` so an explicitly selected but unconfigured backend
    does not look healthy.
    """
    rows: list[tuple[str, str, str]] = []
    try:
        from agent.web_search_registry import (
            get_active_extract_provider,
            get_active_search_provider,
        )
        from tools.web_tools import _ensure_web_plugins_loaded, _provider_is_ready

        # Doctor runs in a fresh process — bundled web providers register
        # during plugin discovery, which nothing has triggered yet here.
        # Without this the registry is empty and every row reads
        # "no provider selected or registered" (idempotent, cheap on rerun).
        _ensure_web_plugins_loaded()
    except Exception:
        return rows

    for capability, getter in (
        ("web search", get_active_search_provider),
        ("web extract", get_active_extract_provider),
    ):
        try:
            provider = getter()
        except Exception:
            provider = None
        if provider is None:
            rows.append(
                (
                    "warn",
                    capability,
                    "(no provider selected or registered)",
                )
            )
            continue
        name = getattr(provider, "name", None) or type(provider).__name__
        if _provider_is_ready(provider):
            rows.append(("ok", capability, f"({name})"))
        else:
            rows.append(
                (
                    "warn",
                    capability,
                    f"({name} selected; provider not configured)",
                )
            )
    return rows


def _apply_doctor_tool_availability_overrides(available: list[str], unavailable: list[dict]) -> tuple[list[str], list[dict]]:
    """Adjust runtime-gated tool availability for doctor diagnostics."""
    from hermes_cli.doctor import _honcho_is_configured_for_doctor
    updated_available = list(available)
    updated_unavailable = []
    for item in unavailable:
        name = item.get("name")
        if _is_kanban_worker_env_gate(item):
            if "kanban" not in updated_available:
                updated_available.append("kanban")
            continue
        if name == "honcho" and _honcho_is_configured_for_doctor():
            if "honcho" not in updated_available:
                updated_available.append("honcho")
            continue
        updated_unavailable.append(item)
    return updated_available, updated_unavailable


def _enabled_cli_toolsets_for_doctor() -> set[str] | None:
    """Return toolsets enabled for the CLI, or None if config resolution fails."""
    try:
        from hermes_cli.config import load_config
        from hermes_cli.tools_config import _get_platform_tools

        return {str(toolset) for toolset in _get_platform_tools(load_config() or {}, "cli")}
    except Exception:
        return None


def _missing_api_key_toolsets_for_summary(unavailable: list[dict]) -> list[dict]:
    """Filter unavailable API-key toolsets to those enabled for the CLI."""
    from hermes_cli.doctor import _enabled_cli_toolsets_for_doctor
    api_key_unavailable = [
        item for item in unavailable
        if item.get("missing_vars") or item.get("env_vars")
    ]
    enabled_toolsets = _enabled_cli_toolsets_for_doctor()
    if enabled_toolsets is None:
        return api_key_unavailable
    return [
        item for item in api_key_unavailable
        if str(item.get("name") or "") in enabled_toolsets
    ]


def _check_git_and_rg(should_fix: bool) -> Finding:
    f = Finding()
    # Git
    if _safe_which("git"):
        check_ok("git")
    else:
        check_warn("git not found", "(optional)")
    
    # ripgrep (optional, for faster file search)
    if _safe_which("rg"):
        check_ok("ripgrep (rg)", "(faster file search)")
    else:
        check_warn("ripgrep (rg) not found", "(file search uses grep fallback)")
        check_info(f"Install for faster search: {_system_package_install_cmd('ripgrep')}")
    return f


def _check_terminal_backend(should_fix: bool) -> Finding:
    """Docker/SSH/Daytona/Vercel/plugin terminal backends, gated on TERMINAL_ENV."""
    f = Finding()
    issues = f.issues
    # Docker (optional)
    terminal_env = os.getenv("TERMINAL_ENV", "local")
    try:
        from hermes_constants import is_container as _is_container
        running_in_container = _is_container()
    except Exception:
        running_in_container = False

    if running_in_container:
        # Inside our container the Docker terminal backend is not
        # configured by default (Docker-in-Docker isn't set up); the
        # local backend is the intended one. Skip the noisy "docker
        # not found" warning. If the user has explicitly chosen
        # TERMINAL_ENV=docker inside the container they likely mounted
        # /var/run/docker.sock, so fall through to the normal check.
        if terminal_env != "docker":
            check_info(
                "Running inside a container — using local terminal backend "
                "(docker-in-docker is not configured by default)"
            )
            # Skip to next section; Docker isn't relevant here.
            terminal_env = "local"
    if terminal_env == "docker":
        if _safe_which("docker"):
            # Check if docker daemon is running
            try:
                result = subprocess.run(["docker", "info"], capture_output=True, timeout=10)
            except subprocess.TimeoutExpired:
                result = None
            if result is not None and result.returncode == 0:
                check_ok("docker", "(daemon running)")
            else:
                _fail_and_issue("docker daemon not running", "", "Start Docker daemon", issues)
        else:
            _fail_and_issue(
                "docker not found",
                "(required for TERMINAL_ENV=docker)",
                "Install Docker or change TERMINAL_ENV",
                issues,
            )
    elif _safe_which("docker"):
        check_ok("docker", "(optional)")
    elif _is_termux():
        check_info("Docker backend is not available inside Termux (expected on Android)")
    elif running_in_container:
        pass  # already explained above
    else:
        check_warn("docker not found", "(optional)")
    
    # SSH (if using ssh backend)
    if terminal_env == "ssh":
        ssh_host = os.getenv("TERMINAL_SSH_HOST")
        if ssh_host:
            ssh_user = os.getenv("TERMINAL_SSH_USER")
            ssh_port = os.getenv("TERMINAL_SSH_PORT")
            ssh_key = os.getenv("TERMINAL_SSH_KEY")
            target = f"{ssh_user}@{ssh_host}" if ssh_user else ssh_host
            cmd = ["ssh", "-o", "ConnectTimeout=5", "-o", "BatchMode=yes"]
            if ssh_port:
                cmd += ["-p", ssh_port]
            if ssh_key:
                cmd += ["-i", os.path.expanduser(ssh_key)]
            cmd += [target, "echo ok"]
            # Try to connect
            try:
                result = subprocess.run(
                    cmd,
                    capture_output=True,
                    text=True, encoding='utf-8', errors='replace',
                    timeout=15
                )
            except subprocess.TimeoutExpired:
                result = None
            if result is not None and result.returncode == 0:
                check_ok(f"SSH connection to {ssh_host}")
            else:
                _fail_and_issue(f"SSH connection to {ssh_host}", "", f"Check SSH configuration for {ssh_host}", issues)
        else:
            _fail_and_issue(
                "TERMINAL_SSH_HOST not set",
                "(required for TERMINAL_ENV=ssh)",
                "Set TERMINAL_SSH_HOST in .env",
                issues,
            )
    
    # Daytona (if using daytona backend)
    if terminal_env == "daytona":
        daytona_key = os.getenv("DAYTONA_API_KEY")
        if daytona_key:
            check_ok("Daytona API key", "(configured)")
        else:
            _fail_and_issue(
                "DAYTONA_API_KEY not set",
                "(required for TERMINAL_ENV=daytona)",
                "Set DAYTONA_API_KEY environment variable",
                issues,
            )
        try:
            from daytona import Daytona  # noqa: F401 — SDK presence check
            check_ok("daytona SDK", "(installed)")
        except ImportError:
            _fail_and_issue(
                "daytona SDK not installed",
                "(pip install daytona)",
                "Install daytona SDK: pip install daytona",
                issues,
            )

    # Vercel Sandbox (if using vercel_sandbox backend)
    if terminal_env == "vercel_sandbox":
        runtime = os.getenv("TERMINAL_VERCEL_RUNTIME", "node24").strip() or "node24"
        from tools.terminal_tool import _SUPPORTED_VERCEL_RUNTIMES
        if runtime in _SUPPORTED_VERCEL_RUNTIMES:
            check_ok("Vercel runtime", f"({runtime})")
        else:
            supported = ", ".join(_SUPPORTED_VERCEL_RUNTIMES)
            _fail_and_issue(
                "Vercel runtime unsupported",
                f"({runtime}; use {supported})",
                f"Set TERMINAL_VERCEL_RUNTIME to one of: {supported}",
                issues,
            )

        disk = os.getenv("TERMINAL_CONTAINER_DISK", "51200").strip()
        if disk in {"", "0", "51200"}:
            check_ok("Vercel disk setting", "(uses platform default)")
        else:
            _fail_and_issue(
                "Vercel custom disk unsupported",
                "(reset terminal.container_disk to 51200)",
                "Vercel Sandbox does not support custom container_disk; use the shared default 51200",
                issues,
            )

        if importlib.util.find_spec("vercel") is not None:
            check_ok("vercel SDK", "(installed)")
        else:
            _fail_and_issue(
                "vercel SDK not installed",
                "(pip install 'hermes-agent[vercel]')",
                "Install the Vercel optional dependency: pip install 'hermes-agent[vercel]'",
                issues,
            )

        auth_status = describe_vercel_auth()
        if auth_status.ok:
            check_ok("Vercel auth", f"({auth_status.label})")
        elif auth_status.label.startswith("partial"):
            _fail_and_issue(
                "Vercel auth incomplete",
                f"({auth_status.label})",
                "Set VERCEL_TOKEN, VERCEL_PROJECT_ID, and VERCEL_TEAM_ID together",
                issues,
            )
        else:
            _fail_and_issue(
                "Vercel auth not configured",
                f"({auth_status.label})",
                "Configure Vercel Sandbox auth with VERCEL_TOKEN, VERCEL_PROJECT_ID, and VERCEL_TEAM_ID",
                issues,
            )
        for line in auth_status.detail_lines:
            check_info(f"Vercel auth {line}")

        persistent = os.getenv("TERMINAL_CONTAINER_PERSISTENT", "true").lower() in {"1", "true", "yes", "on"}
        if persistent:
            check_info("Vercel persistence: snapshot filesystem only; live processes do not survive sandbox recreation")
        else:
            check_info("Vercel persistence: ephemeral filesystem")

    # Plugin-registered terminal backends (if one is the active backend)
    if terminal_env not in {
        "local", "docker", "singularity", "modal", "managed_modal",
        "daytona", "vercel_sandbox", "ssh",
    }:
        try:
            from hermes_cli.plugins import discover_plugins

            discover_plugins()
            from agent.terminal_env_registry import get_provider

            _provider = get_provider(terminal_env)
        except Exception:
            _provider = None
        if _provider is None:
            _fail_and_issue(
                f"Unknown terminal backend '{terminal_env}'",
                "(no built-in or plugin backend by that name)",
                "Fix terminal.backend in config.yaml, or install/enable the plugin that provides it",
                issues,
            )
        else:
            for _ok, _label, _detail in _provider.doctor_checks():
                if _ok:
                    check_ok(_label, _detail)
                else:
                    _fail_and_issue(_label, _detail, _detail.strip("()"), issues)
    return f


def _check_node_and_browser(should_fix: bool) -> Finding:
    """Node.js, agent-browser resolution, Playwright Chromium, Lightpanda engine."""
    from hermes_cli.doctor import PROJECT_ROOT
    f = Finding()
    # Node.js + agent-browser (for browser automation tools)
    if _safe_which("node"):
        check_ok("Node.js")
        # agent-browser is no longer a root package.json dependency (#43564)
        # — it resolves lazily via npx (or a global/Hermes-managed install)
        # at first use. Mirror tools.browser_tool._find_agent_browser's own
        # resolution cascade here so doctor can't diverge from what browser
        # tools will actually find; validate=False keeps this a cheap
        # existence check with no subprocess spawn or install side effects.
        agent_browser_ok = False
        try:
            from tools.browser_tool import _find_agent_browser, _is_npx_agent_browser_sentinel
            _resolved_ab = _find_agent_browser(validate=False)
        except Exception:
            _resolved_ab = None

        if _resolved_ab and _is_npx_agent_browser_sentinel(_resolved_ab):
            check_ok("agent-browser", "(resolves via npx on first use)")
            agent_browser_ok = True
            if should_fix:
                # Doctor can't tell from here whether npx's cache already
                # has agent-browser warm — just fire the same warm-up
                # `hermes update` does, so a session's first browser call
                # doesn't pay the registry fetch either way.
                from tools.browser_tool import warm_agent_browser_npx_cache
                if warm_agent_browser_npx_cache():
                    check_info("  Warmed npx cache for agent-browser")
                else:
                    check_info("  Could not warm npx cache (offline or npx unavailable)")
        elif _resolved_ab and agent_browser_runnable(_resolved_ab):
            check_ok("agent-browser", "(browser automation)")
            agent_browser_ok = True
        elif _resolved_ab:
            # Found on PATH but won't run — almost always a dangling global
            # symlink left behind by agent-browser's npm postinstall after a
            # `hermes update` wiped node_modules (issue #48521).
            check_warn(
                "agent-browser found but not runnable",
                f"(broken symlink at {_resolved_ab}? run: npx agent-browser --version)",
            )
        elif _is_termux():
            check_info("agent-browser is not installed (expected in the tested Termux path)")
            check_info("Install it manually later with: npm install -g agent-browser && agent-browser install")
            check_info("Termux browser setup:")
            for step in _termux_browser_setup_steps(node_installed=True):
                check_info(step)
        else:
            check_warn("agent-browser not installed", "(requires npm/npx on PATH)")

        # Chromium presence — the browser tools silently fail to register when
        # agent-browser is found but no Playwright-managed Chromium is on disk
        # (tools/browser_tool.py::check_browser_requirements filters them out
        # before the agent ever sees them).  Reuse the exact predicate it uses
        # so the two checks cannot diverge.  Skip on Termux (not a tested
        # path).
        if agent_browser_ok and not _is_termux():
            try:
                # Lazy import: browser_tool is a ~150KB module we don't want
                # to eagerly load in every `hermes doctor` invocation.
                from tools.browser_tool import (
                    _chromium_installed,
                    _is_camofox_mode,
                    _get_cloud_provider,
                    _get_cdp_override_raw,
                    _using_lightpanda_engine,
                )
            except Exception:
                # If browser_tool can't even import, that's a separate bug
                # surfaced elsewhere; don't crash doctor.
                pass
            else:
                # Only warn about Chromium if the installed engine actually
                # requires it: Camofox, CDP override, a cloud provider, or
                # Lightpanda all bypass the local Chromium requirement.
                skip_chromium_check = (
                    _is_camofox_mode()
                    or bool(_get_cdp_override_raw())
                    or _get_cloud_provider() is not None
                    or _using_lightpanda_engine()
                )
                if not skip_chromium_check:
                    if _chromium_installed():
                        check_ok("Playwright Chromium", "(browser engine)")
                    else:
                        check_warn(
                            "Playwright Chromium not installed",
                            "(browser_* tools will be hidden from the agent)",
                        )
                        if sys.platform == "win32":
                            check_info(
                                f"Install with: cd {PROJECT_ROOT} && "
                                "npx playwright install chromium"
                            )
                        else:
                            check_info(
                                f"Install with: cd {PROJECT_ROOT} && "
                                "npx playwright install --with-deps chromium"
                            )
    elif _is_termux():
        check_info("Node.js not found (browser tools are optional in the tested Termux path)")
        check_info("Install Node.js on Termux with: pkg install nodejs")
        check_info("Termux browser setup:")
        for step in _termux_browser_setup_steps(node_installed=False):
            check_info(step)
    else:
        check_warn("Node.js not found", "(optional, needed for browser tools)")

    # Lightpanda engine (browser.engine / AGENT_BROWSER_ENGINE). Independent
    # of Node: Browser Use mode spawns ``lightpanda serve`` itself.
    try:
        from tools.browser_tool import _using_lightpanda_engine, lightpanda_engine_status
        from tools.browser_lightpanda import LIGHTPANDA_INSTALL_HINT, find_lightpanda_binary
    except Exception:
        pass
    else:
        # _using_lightpanda_engine() is a cached config read — a failure
        # there would be exceptional, not something to silently hide.
        if _using_lightpanda_engine():
            try:
                _lp_used, _lp_reason = lightpanda_engine_status()
            except Exception as e:
                _lp_used, _lp_reason = False, f"status check failed: {e}"
            if not _lp_used:
                check_warn("browser.engine=lightpanda is shadowed", f"({_lp_reason})")
                check_info(
                    "Fix: pick Lightpanda in `hermes tools` → Browser Automation, "
                    "or set browser.engine: auto"
                )
            elif find_lightpanda_binary():
                check_ok("Lightpanda", f"({_lp_reason})")
            else:
                check_warn(
                    "Lightpanda selected but binary not found",
                    "(browser tools will fail until it is installed)",
                )
                check_info(LIGHTPANDA_INSTALL_HINT)
    return f


def _check_npm_audit(should_fix: bool) -> Finding:
    """npm audit per Node package tree (root, web/ui-tui workspaces, WhatsApp bridge)."""
    from hermes_cli.doctor import PROJECT_ROOT
    f = Finding()
    issues = f.issues
    # npm audit for all Node.js packages
    _npm_bin = _safe_which("npm")
    if _npm_bin:
        # Each entry: (cwd, label, extra_audit_args)
        # PROJECT_ROOT is audited with --workspaces=false so that the apps/*
        # glob (which pulls in Electron, node-pty, etc.) is never resolved
        # for a routine security check. The web and ui-tui workspaces are
        # audited separately via --workspace flags. See #38772.
        # The WhatsApp bridge may live under a writable HERMES_HOME mirror
        # instead of the (possibly read-only) install tree in Docker — resolve
        # it through the shared helper so we audit the dir that actually holds
        # node_modules. See #49561.
        try:
            from gateway.platforms.whatsapp_common import resolve_whatsapp_bridge_dir
            _whatsapp_bridge_dir = resolve_whatsapp_bridge_dir()
        except Exception:
            _whatsapp_bridge_dir = PROJECT_ROOT / "scripts" / "whatsapp-bridge"
        npm_audit_targets = [
            (PROJECT_ROOT, "Browser tools (agent-browser)", ["--workspaces=false"]),
            (PROJECT_ROOT, "web workspace", ["--workspace", "web"]),
            (PROJECT_ROOT, "ui-tui workspace", ["--workspace", "ui-tui"]),
            (_whatsapp_bridge_dir, "WhatsApp bridge", []),
        ]
        for npm_dir, label, audit_extra in npm_audit_targets:
            # For workspace-scoped audits run from PROJECT_ROOT the
            # node_modules check must use the workspace root; standalone dirs
            # (whatsapp-bridge) check their own node_modules.
            check_dir = PROJECT_ROOT if audit_extra else npm_dir
            if not (check_dir / "node_modules").exists():
                continue
            try:
                # Use resolved absolute path so Windows can execute
                # npm.cmd (CreateProcessW can't run bare .cmd names).
                audit_result = subprocess.run(
                    [_npm_bin, "audit", "--json", *audit_extra],
                    cwd=str(npm_dir),
                    capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=30,
                )
                import json as _json
                audit_data = _json.loads(audit_result.stdout) if audit_result.stdout.strip() else {}
                vuln_count = audit_data.get("metadata", {}).get("vulnerabilities", {})
                critical = vuln_count.get("critical", 0)
                high = vuln_count.get("high", 0)
                moderate = vuln_count.get("moderate", 0)
                total = critical + high + moderate
                # Determine a scoped fix command for the remediation hint.
                if audit_extra and audit_extra[0] == "--workspace":
                    # Detection (`npm audit --workspace <name>`) is read-only and
                    # safe, but `npm audit fix --workspace <name>` crashes on
                    # current npm with "Cannot read properties of null (reading
                    # 'edgesOut')" — an arborist bug with workspace-filtered
                    # audit fix. The root-level `npm audit fix` can crash on the
                    # same tree with "isDescendantOf", so do not hand the user a
                    # manual fix command for these build-tool advisories.
                    fix_cmd = None
                elif audit_extra == ["--workspaces=false"]:
                    fix_cmd = f"cd {npm_dir} && npm audit fix --workspaces=false"
                else:
                    fix_cmd = f"cd {npm_dir} && npm audit fix"
                if total == 0:
                    check_ok(f"{label} deps", "(no known vulnerabilities)")
                elif critical > 0 or high > 0:
                    if fix_cmd:
                        vuln_detail = (
                            f"{critical} critical, {high} high, {moderate} moderate — run: {fix_cmd}"
                        )
                    else:
                        vuln_detail = (
                            f"{critical} critical, {high} high, {moderate} moderate — "
                            "build-tool advisory; clears via lockfile bump"
                        )
                    check_warn(
                        f"{label} deps",
                        f"({vuln_detail})"
                    )
                    if audit_extra and audit_extra[0] == "--workspace":
                        # The web/ui-tui workspace advisories are in build-time
                        # tooling (esbuild/vite, etc.), not runtime code that ships
                        # to users. Manual npm remediation may error with a known
                        # arborist crash (edgesOut / isDescendantOf) on this monorepo
                        # tree — in that case it is an npm bug, not a Hermes one.
                        check_info(
                            "  ^ build-time tooling (not runtime); if manual npm remediation "
                            "errors with an arborist crash it's a known npm bug — clears "
                            "via a lockfile bump"
                        )
                    issues.append(
                        f"{label} has {total} npm "
                        f"{'vulnerability' if total == 1 else 'vulnerabilities'}"
                    )
                else:
                    check_ok(
                        f"{label} deps",
                        f"({moderate} moderate "
                        f"{'vulnerability' if moderate == 1 else 'vulnerabilities'})",
                    )
            except Exception:
                pass

    if _is_termux():
        check_info("Termux compatibility fallbacks:")
        for note in _termux_install_all_fallback_notes():
            check_info(note)
    return f


def _check_tool_availability(should_fix: bool) -> Finding:
    from hermes_cli.doctor import PROJECT_ROOT, _doctor_web_capability_rows
    f = Finding()
    issues = f.issues
    try:
        # Add project root to path for imports
        sys.path.insert(0, str(PROJECT_ROOT))
        from model_tools import check_tool_availability, TOOLSET_REQUIREMENTS
        
        available, unavailable = check_tool_availability()
        available, unavailable = _apply_doctor_tool_availability_overrides(available, unavailable)

        # Web is split into search/extract readiness rows so an explicitly
        # selected but unconfigured backend cannot look healthy (#78412).
        web_rows = []
        if "web" in available or any(item.get("name") == "web" for item in unavailable):
            web_rows = _doctor_web_capability_rows()
            if web_rows:
                available = [tid for tid in available if tid != "web"]
                unavailable = [item for item in unavailable if item.get("name") != "web"]

        for tid in available:
            info = TOOLSET_REQUIREMENTS.get(tid, {})
            check_ok(info.get("name", tid), _doctor_tool_availability_detail(tid))

        for status, label, detail in web_rows:
            if status == "ok":
                check_ok(label, detail)
            else:
                check_warn(label, detail)

        for item in unavailable:
            env_vars = item.get("missing_vars") or item.get("env_vars") or []
            if env_vars:
                vars_str = ", ".join(env_vars)
                check_warn(item["name"], f"(missing {vars_str})")
            else:
                check_warn(item["name"], "(system dependency not met)")

        # Count missing API-key requirements only for toolsets enabled in the
        # current CLI platform. Default-off or explicitly disabled toolsets may
        # still show warnings above, but should not pollute the final summary.
        api_disabled = _missing_api_key_toolsets_for_summary(unavailable)
        web_not_ready = any(status != "ok" for status, _, _ in web_rows)
        if api_disabled or web_not_ready:
            issues.append("Run 'hermes setup' to configure missing API keys for full tool access")
    except Exception as e:
        check_warn("Could not check tool availability", f"({e})")
    return f
