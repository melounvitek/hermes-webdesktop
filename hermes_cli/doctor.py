"""
Doctor command for hermes CLI.

Diagnoses issues with Hermes Agent setup.
"""

# stdlib modules stay bound here: tests patch doctor.shutil.which / doctor.subprocess.run /
# doctor.importlib.util.find_spec / doctor.Path.home / doctor.sys.platform / doctor.os.listdir.
import os
import sys
import subprocess  # noqa: F401
import shutil  # noqa: F401
import importlib.util  # noqa: F401
from pathlib import Path  # noqa: F401

from hermes_cli.config import (  # noqa: F401  (detect_install_method: tests patch doctor.detect_install_method)
    detect_install_method,
    get_env_path,
    get_hermes_home,
    get_project_root,
)
from hermes_cli.env_loader import load_hermes_dotenv
from hermes_constants import display_hermes_home

PROJECT_ROOT = get_project_root()
HERMES_HOME = get_hermes_home()
_DHH = display_hermes_home()  # user-facing display path (e.g. ~/.hermes or ~/.hermes/profiles/coder)

# Load environment variables from ~/.hermes/.env so API key checks work
_env_path = get_env_path()
load_hermes_dotenv(hermes_home=_env_path.parent, project_env=PROJECT_ROOT / ".env")

from hermes_cli.colors import Colors, color
from hermes_cli.doctor_config import (  # noqa: F401  (re-exported; tests use hermes_cli.doctor.<name>)
    _DEPRECATED_COMPRESSION_SUMMARY_KEYS,
    _DEPRECATED_CONFIG_KEYS,
    _DEPRECATED_ENV_VARS,
    _check_config_drift,
    _check_config_file,
    _check_env_file,
    _check_mcp_security,
    _check_xai_retirement,
    _has_provider_env_config,
    collect_deprecated_config_keys,
    collect_deprecated_env_vars,
    collect_relay_plugin_cutover_findings,
    managed_scope_check,
    report_deprecated_config_and_env,
)
from hermes_cli.doctor_platform import (  # noqa: F401  (re-exported; tests use hermes_cli.doctor.<name>)
    _SQLITE_HEADER_MAGIC,
    _check_certificates,
    _check_command_installation,
    _check_gateway_service_linger,
    _check_gateway_supervision,
    _check_python_environment,
    _check_required_packages,
    _check_s6_supervision,
    _check_security_advisories,
    _check_version_consistency,
    _desktop_app_bundle,
    _format_db_size,
    _hermes_database_paths,
    _macos_desktop_dr,
    _python_install_cmd,
    _read_journal_mode,
    _read_pyproject_version,
    _report_database_journal_modes,
    _sqlite_upgrade_hint,
    _system_package_install_cmd,
    _unreadable_reason,
    check_certificates,
    check_macos_full_disk_access,
    check_macos_tcc_anchor,
    check_macos_tcc_grants,
)
from hermes_cli.doctor_tools import (  # noqa: F401  (re-exported; tests use hermes_cli.doctor.<name>)
    _apply_doctor_tool_availability_overrides,
    _check_git_and_rg,
    _check_node_and_browser,
    _check_npm_audit,
    _check_terminal_backend,
    _check_tool_availability,
    _doctor_tool_availability_detail,
    _doctor_web_capability_rows,
    _enabled_cli_toolsets_for_doctor,
    _is_kanban_worker_env_gate,
    _missing_api_key_toolsets_for_summary,
    _safe_which,
    _termux_browser_setup_steps,
    _termux_install_all_fallback_notes,
)
from hermes_cli.doctor_state import (  # noqa: F401  (re-exported; tests use hermes_cli.doctor.<name>)
    STATE_DB_SIZE_WARN_BYTES,
    _check_directory_structure,
    _check_memory_provider,
    _check_profiles,
    _check_skills_hub,
    _check_state_db,
    _doctor_memory_config,
    _honcho_is_configured_for_doctor,
    _memory_store_flags,
    _render_state_db_stats,
)
from hermes_cli.doctor_report import (  # noqa: F401  (re-exported for doctor_live and tests)
    Finding,
    _fail_and_issue,
    _section,
    check_fail,
    check_info,
    check_ok,
    check_warn,
)
from hermes_cli.doctor_connectivity import (  # noqa: F401  (re-exported; tests import from hermes_cli.doctor)
    _build_apikey_providers_list,
    _has_healthy_oauth_fallback_for_apikey_provider,
    build_probes,
    run_probes,
)


_PROVIDER_ENV_HINTS = (
    "DEEPINFRA_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "ANTHROPIC_TOKEN",
    "OPENAI_BASE_URL",
    "NOUS_API_KEY",
    "GLM_API_KEY",
    "ZAI_API_KEY",
    "Z_AI_API_KEY",
    "KIMI_API_KEY",
    "KIMI_CN_API_KEY",
    "GMI_API_KEY",
    "FIREWORKS_API_KEY",
    "ACTUAL_API_KEY",
    "ACTUAL_BASE_URL",
    "MINIMAX_API_KEY",
    "MINIMAX_CN_API_KEY",
    "KILOCODE_API_KEY",
    "DEEPSEEK_API_KEY",
    "DASHSCOPE_API_KEY",
    "HF_TOKEN",
    "AI_GATEWAY_API_KEY",
    "OPENCODE_ZEN_API_KEY",
    "OPENCODE_GO_API_KEY",
    "COMMANDCODE_API_KEY",
    "XIAOMI_API_KEY",
    "TOKENHUB_API_KEY",
    "TOKENPLAN_API_KEY",
)


from hermes_constants import is_termux as _is_termux  # noqa: F401  (tests call doctor._is_termux)


# Shared byte formatter, aliased to the name this module's three rendering
# call sites already use.
from hermes_cli.sizefmt import format_bytes as _human_bytes  # noqa: F401  (tests import doctor._human_bytes)


def _check_auth_providers(should_fix: bool) -> Finding:
    """Refresh-free OAuth status snapshot (doctor must never trigger a token refresh)."""
    f = Finding()
    try:
        from hermes_cli.auth import (
            get_nous_auth_status_local,
            get_codex_auth_status,
            get_minimax_oauth_auth_status,
        )

        # Read-only display: refresh-free snapshot — doctor must never
        # trigger an OAuth refresh as a side effect of a health check.
        nous_status = get_nous_auth_status_local()
        if nous_status.get("logged_in"):
            check_ok("Nous Portal auth", "(logged in)")
        else:
            check_warn("Nous Portal auth", "(not logged in)")

        codex_status = get_codex_auth_status()
        if codex_status.get("logged_in"):
            check_ok("OpenAI Codex auth", "(logged in)")
        else:
            check_warn("OpenAI Codex auth", "(not logged in)")
            if codex_status.get("error"):
                check_info(codex_status["error"])
            # Native OAuth uses Hermes' own device-code flow — the Codex CLI is
            # only needed to import existing tokens from ~/.codex/auth.json.
            # Attach the hint to the Codex auth row so it doesn't read as
            # remediation for whichever provider happens to print next (#27975).
            if not _safe_which("codex"):
                check_info(
                    "codex CLI not installed "
                    "(optional — only required to import tokens "
                    "from an existing Codex CLI login)"
                )

        minimax_status = get_minimax_oauth_auth_status()
        if minimax_status.get("logged_in"):
            region = minimax_status.get("region", "global")
            check_ok("MiniMax OAuth", f"(logged in, region={region})")
        else:
            check_warn("MiniMax OAuth", "(not logged in)")
    except Exception as e:
        check_warn("Auth provider status", f"(could not check: {e})")

    # xAI OAuth — separate try/except so an import failure here cannot
    # disrupt the already-printed Nous/Codex/Gemini/MiniMax rows above.
    try:
        from hermes_cli.auth import get_xai_oauth_auth_status
        xai_oauth_status = get_xai_oauth_auth_status() or {}
        if xai_oauth_status.get("logged_in"):
            check_ok("xAI OAuth", "(logged in)")
        else:
            check_warn("xAI OAuth", "(not logged in)")
            if xai_oauth_status.get("error"):
                check_info(xai_oauth_status["error"])
    except Exception:
        pass
    return f


def _check_api_connectivity(should_fix: bool) -> Finding:
    """Parallel HTTP/SDK probes for every configured provider; results printed in submission order."""
    f = Finding()
    probes = build_probes()
    # Single status line so users see something happening; ``\r`` clears it
    # once the first real result line lands.
    print(f"  {color(f'Running {len(probes)} connectivity checks in parallel…', Colors.DIM)}",
          end="", flush=True)
    results = run_probes(probes)
    print("\r" + " " * 70 + "\r", end="")
    for r in results:
        for glyph, label, detail in r.lines:
            print(f"  {glyph} {label} {detail}" if detail else f"  {glyph} {label}")
        if r.issues and not _has_healthy_oauth_fallback_for_apikey_provider(r.label):
            f.issues.extend(r.issues)
    return f


# Ordered (section title, check). A None title means the check prints its own
# header (or none) — order is the user-visible output order, keep it.
DOCTOR_CHECKS = (
    ('Security Advisories', _check_security_advisories),
    ('MCP Server Security', _check_mcp_security),
    ('Python Environment', _check_python_environment),
    ('SSL / CA Certificates', _check_certificates),
    ('Required Packages', _check_required_packages),
    ('Configuration Files', _check_env_file),
    (None, _check_config_file),
    (None, _check_config_drift),
    ('xAI Model Retirement (May 15, 2026)', _check_xai_retirement),
    ('Auth Providers', _check_auth_providers),
    ('Directory Structure', _check_directory_structure),
    (None, _check_state_db),
    (None, _check_gateway_supervision),
    (None, _check_command_installation),
    ('External Tools', _check_git_and_rg),
    (None, _check_terminal_backend),
    (None, _check_node_and_browser),
    (None, _check_npm_audit),
    ('API Connectivity', _check_api_connectivity),
    ('Tool Availability', _check_tool_availability),
    ('Skills Hub', _check_skills_hub),
    ('Memory Provider', _check_memory_provider),
    (None, _check_profiles),
)


def _ack_advisory(ack_target: str) -> None:
    """`hermes doctor --ack <id>`: persist the ack and return without running diagnostics."""
    from hermes_cli.security_advisories import (
        ADVISORIES,
        ack_advisory,
    )
    valid_ids = {a.id for a in ADVISORIES}
    if ack_target not in valid_ids:
        print(color(
            f"Unknown advisory ID: {ack_target!r}. Known IDs: "
            f"{', '.join(sorted(valid_ids)) or '(none)'}",
            Colors.RED,
        ))
        sys.exit(2)
    if ack_advisory(ack_target):
        print(color(
            f"  ✓ Acknowledged advisory {ack_target}. "
            f"It will no longer trigger startup banners.",
            Colors.GREEN,
        ))
    else:
        print(color(
            f"  ✗ Failed to persist ack for {ack_target}. "
            f"Check ~/.hermes/config.yaml is writable.",
            Colors.RED,
        ))
        sys.exit(1)


def _print_summary(should_fix: bool, total: Finding) -> None:
    print()
    remaining_issues = total.issues + total.manual_issues
    fixed_count = total.fixed
    if should_fix and fixed_count > 0:
        print(color("─" * 60, Colors.GREEN))
        print(color(f"  Fixed {fixed_count} issue(s).", Colors.GREEN, Colors.BOLD), end="")
        if remaining_issues:
            print(color(f" {len(remaining_issues)} issue(s) require manual intervention.", Colors.YELLOW, Colors.BOLD))
        else:
            print()
        print()
        if remaining_issues:
            for i, issue in enumerate(remaining_issues, 1):
                print(f"  {i}. {issue}")
            print()
    elif remaining_issues:
        print(color("─" * 60, Colors.YELLOW))
        print(color(f"  Found {len(remaining_issues)} issue(s) to address:", Colors.YELLOW, Colors.BOLD))
        print()
        for i, issue in enumerate(remaining_issues, 1):
            print(f"  {i}. {issue}")
        print()
        if not should_fix:
            print(color("  Tip: run 'hermes doctor --fix' to auto-fix what's possible.", Colors.DIM))
    else:
        print(color("─" * 60, Colors.GREEN))
        print(color("  All checks passed! 🎉", Colors.GREEN, Colors.BOLD))
    
    print()


def run_doctor(args):
    """Run diagnostic checks."""
    should_fix = getattr(args, 'fix', False)
    ack_target = getattr(args, 'ack', None)

    # Doctor runs from the interactive CLI, so CLI-gated tool availability
    # checks (like cronjob management) should see the same context as `hermes`.
    os.environ.setdefault("HERMES_INTERACTIVE", "1")

    if ack_target:
        return _ack_advisory(ack_target)

    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.CYAN))
    print(color("│                 🩺 Hermes Doctor                        │", Colors.CYAN))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.CYAN))

    total = Finding()
    for title, check in DOCTOR_CHECKS:
        if title:
            _section(title)
        total.merge(check(should_fix))

    # Opt-in live backend probes run AFTER all static checks, only with
    # `hermes doctor --live` (real network calls; bounded + read-only).
    try:
        from hermes_cli.doctor_live import maybe_run_live_checks
        maybe_run_live_checks(args, total.manual_issues)
    except Exception:
        pass

    _print_summary(should_fix, total)
