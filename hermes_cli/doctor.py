"""
Doctor command for hermes CLI.

Diagnoses issues with Hermes Agent setup.
"""

import os
import sys
import subprocess
import shutil
import importlib.util
from pathlib import Path

from hermes_cli.config import (  # noqa: F401  (detect_install_method: tests patch doctor.detect_install_method)
    detect_install_method,
    get_env_path,
    get_hermes_home,
    get_project_root,
)
from hermes_cli.env_loader import load_hermes_dotenv
from hermes_constants import display_hermes_home
from hermes_constants import agent_browser_runnable

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
from hermes_cli.vercel_auth import describe_vercel_auth


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


from hermes_constants import is_termux as _is_termux


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


def _honcho_is_configured_for_doctor() -> bool:
    """Return True when Honcho is configured, even if this process has no active session."""
    try:
        from plugins.memory.honcho.client import HonchoClientConfig

        cfg = HonchoClientConfig.from_global_config()
        return bool(cfg.enabled and (cfg.api_key or cfg.base_url))
    except Exception:
        return False


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


def _doctor_memory_config(hermes_home: Path | None = None) -> dict:
    """Return the effective memory section used by doctor diagnostics."""
    home = hermes_home if hermes_home is not None else HERMES_HOME
    try:
        from hermes_cli.config import _expand_env_vars, read_user_config_raw

        config_path = home / "config.yaml"
        if not config_path.exists():
            return {}
        config = _expand_env_vars(read_user_config_raw(config_path))
        try:
            from hermes_cli import managed_scope

            config = managed_scope.apply_managed_overlay(config)
        except Exception:
            pass
        section = config.get("memory") if isinstance(config, dict) else None
        return section if isinstance(section, dict) else {}
    except Exception:
        return {}


# ── state.db health/stats thresholds (advisory only — module constants,
# deliberately NOT config: doctor warnings are guidance, not policy) ──
STATE_DB_SIZE_WARN_BYTES = 1 * 1024 * 1024 * 1024   # 1 GiB logical size


# Shared byte formatter, aliased to the name this module's three rendering
# call sites already use.
from hermes_cli.sizefmt import format_bytes as _human_bytes


def _render_state_db_stats(stats: dict, holders=None) -> list:
    """Turn a collect_state_db_stats() dict into doctor output lines.

    Returns a list of ``(kind, text, detail)`` tuples where kind is one of
    'info' / 'warn'. Pure formatting — no I/O — so it is unit-testable
    without spawning the doctor CLI. Tolerates None in every field.
    """
    lines: list = []
    stats = stats or {}

    logical = stats.get("logical_size_bytes")
    wal = stats.get("wal_size_bytes")
    freelist = stats.get("freelist_count")

    size_bits = []
    if logical is not None:
        size_bits.append(f"logical size {_human_bytes(logical)}")
    if stats.get("page_count") is not None:
        size_bits.append(f"{stats['page_count']:,} pages")
    if freelist is not None:
        size_bits.append(f"{freelist:,} free")
    if wal is not None:
        size_bits.append(f"WAL {_human_bytes(wal)}")
    if size_bits:
        lines.append(("info", "state.db " + ", ".join(size_bits), ""))

    row_bits = []
    if stats.get("messages") is not None:
        row_bits.append(f"{stats['messages']:,} messages")
    if stats.get("sessions") is not None:
        row_bits.append(f"{stats['sessions']:,} sessions")
    if stats.get("journal_mode"):
        row_bits.append(f"journal_mode={stats['journal_mode']}")
    if holders is not None:
        row_bits.append(f"{holders} process(es) holding the DB open")
    if row_bits:
        lines.append(("info", ", ".join(row_bits), ""))

    fts = stats.get("fts_tables")
    if fts:
        present = [t for t, ok in fts.items() if ok]
        lines.append((
            "info",
            "FTS tables: " + (", ".join(present) if present else "none"),
            "",
        ))

    deferral = stats.get("fts_rebuild_deferral")
    if isinstance(deferral, dict):
        attempts = deferral.get("attempts")
        pids = deferral.get("holder_pids") or []
        lines.append((
            "warn",
            f"state.db FTS repair is blocked after {attempts or '?'} "
            f"deferral(s) by PID(s) {pids or 'unknown'}",
            "(stop the listed processes, then run 'hermes sessions "
            "optimize-storage' with the gateway stopped)",
        ))

    # Advisory: oversized database. Suggest auto_prune, and — when the v23
    # FTS rebuild is pending OR the DB still carries the legacy inline
    # trigram layout (fts_storage_version marker absent) — the offline
    # optimize-storage pass that migrates/compacts the FTS indexes.
    if logical is not None and logical > STATE_DB_SIZE_WARN_BYTES:
        detail = (
            "consider enabling sessions.auto_prune in config.yaml "
            "to bound growth"
        )
        legacy_trigram = (
            fts is not None
            and fts.get("messages_fts_trigram")
            and stats.get("fts_storage_version") is None
        )
        if stats.get("fts_rebuild_pending") or legacy_trigram:
            detail += (
                "; run 'hermes sessions optimize-storage' offline "
                "(with the gateway stopped) to compact FTS storage"
            )
        lines.append((
            "warn",
            f"state.db is large ({_human_bytes(logical)})",
            f"({detail})",
        ))

    # WAL runaway is deliberately NOT warned here: the pre-existing WAL
    # check later in the state.db section already warns above 50 MB and
    # offers a checkpoint via --fix; a second warning at a higher threshold
    # would only duplicate it.

    return lines


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


def _memory_store_flags(hermes_home: Path) -> tuple:
    from tools.memory_tool import get_builtin_memory_store_flags

    return get_builtin_memory_store_flags({"memory": _doctor_memory_config(hermes_home)})


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


def _check_directory_structure(should_fix: bool) -> Finding:
    """HERMES_HOME, expected subdirs, SOUL.md, and the enabled built-in memory files."""
    f = Finding()
    hermes_home = HERMES_HOME
    if hermes_home.exists():
        check_ok(f"{_DHH} directory exists")
    elif should_fix:
        hermes_home.mkdir(parents=True, exist_ok=True)
        check_ok(f"Created {_DHH} directory")
        f.fixed += 1
    else:
        check_warn(f"{_DHH} not found", "(will be created on first use)")
    
    _memory_enabled, _user_profile_enabled = _memory_store_flags(hermes_home)

    # Check expected subdirectories. The built-in file store does not create or
    # consume memories/ when both targets are disabled, so stale migration files
    # are not an active diagnostic surface.
    expected_subdirs = ["cron", "sessions", "logs", "skills"]
    if _memory_enabled or _user_profile_enabled:
        expected_subdirs.append("memories")
    for subdir_name in expected_subdirs:
        subdir_path = hermes_home / subdir_name
        if subdir_path.exists():
            check_ok(f"{_DHH}/{subdir_name}/ exists")
        elif should_fix:
            subdir_path.mkdir(parents=True, exist_ok=True)
            check_ok(f"Created {_DHH}/{subdir_name}/")
            f.fixed += 1
        else:
            check_warn(f"{_DHH}/{subdir_name}/ not found", "(will be created on first use)")
    
    # Check for SOUL.md persona file
    soul_path = hermes_home / "SOUL.md"
    if soul_path.exists():
        content = soul_path.read_text(encoding="utf-8").strip()
        # Check if it's just the template comments (no real content)
        lines = [l for l in content.splitlines() if l.strip() and not l.strip().startswith(("<!--", "-->", "#"))]
        if lines:
            check_ok(f"{_DHH}/SOUL.md exists (persona configured)")
        else:
            check_info(f"{_DHH}/SOUL.md exists but is empty — edit it to customize personality")
    else:
        check_warn(f"{_DHH}/SOUL.md not found", "(create it to give Hermes a custom personality)")
        if should_fix:
            soul_path.parent.mkdir(parents=True, exist_ok=True)
            soul_path.write_text(
                "# Hermes Agent Persona\n\n"
                "<!-- Edit this file to customize how Hermes communicates. -->\n\n"
                "You are Hermes, a helpful AI assistant.\n",
                encoding="utf-8",
            )
            check_ok(f"Created {_DHH}/SOUL.md with basic template")
            f.fixed += 1
    
    # Check only enabled built-in stores. External providers are additive, but
    # users can explicitly disable either legacy file target; stale files left
    # by a migration must not be presented as active memory usage.
    memories_dir = hermes_home / "memories"
    if not (_memory_enabled or _user_profile_enabled):
        check_info("Built-in memory files disabled by config")
    elif memories_dir.exists():
        check_ok(f"{_DHH}/memories/ directory exists")
        memory_file = memories_dir / "MEMORY.md"
        user_file = memories_dir / "USER.md"
        if _memory_enabled:
            if memory_file.exists():
                size = len(memory_file.read_text(encoding="utf-8").strip())
                check_ok(f"MEMORY.md exists ({size} chars)")
            else:
                check_info("MEMORY.md not created yet (will be created when the agent first writes a memory)")
        if _user_profile_enabled:
            if user_file.exists():
                size = len(user_file.read_text(encoding="utf-8").strip())
                check_ok(f"USER.md exists ({size} chars)")
            else:
                check_info("USER.md not created yet (will be created when the agent first writes a memory)")
    else:
        check_warn(f"{_DHH}/memories/ not found", "(will be created on first use)")
        if should_fix:
            memories_dir.mkdir(parents=True, exist_ok=True)
            check_ok(f"Created {_DHH}/memories/")
            f.fixed += 1
    return f


def _check_state_db(should_fix: bool) -> Finding:
    """state.db session count, FTS write health, schema repair, stats snapshot, WAL size."""
    f = Finding()
    issues = f.issues
    hermes_home = HERMES_HOME
    # Check SQLite session store
    state_db_path = hermes_home / "state.db"
    if state_db_path.exists():
        try:
            import sqlite3
            conn = sqlite3.connect(str(state_db_path))
            cursor = conn.execute("SELECT COUNT(*) FROM sessions")
            count = cursor.fetchone()[0]
            conn.close()
            check_ok(f"{_DHH}/state.db exists ({count} sessions)")

            # FTS write-health probe (#50502): `SELECT COUNT(*)` above succeeds
            # even when the FTS index is corrupt and every message write fails
            # through the triggers. `_db_opens_cleanly` now drives a rolled-back
            # write so this otherwise-silent corruption class is surfaced (and
            # repaired in place with --fix).
            from hermes_state import _db_opens_cleanly, repair_state_db_schema

            _write_reason = _db_opens_cleanly(state_db_path)
            if _write_reason is not None:
                check_warn(
                    f"{_DHH}/state.db fails a write-health probe (FTS index may be corrupt)",
                    f"({_write_reason})",
                )
                if should_fix:
                    report = repair_state_db_schema(state_db_path)
                    if report.get("repaired"):
                        backup_name = (
                            Path(report["backup_path"]).name
                            if report.get("backup_path") else "n/a"
                        )
                        check_ok(
                            "Repaired state.db FTS write health",
                            f"(strategy: {report.get('strategy')}; backup: {backup_name})",
                        )
                        f.fixed += 1
                    else:
                        check_warn(
                            "state.db FTS write-health repair did not recover automatically",
                            f"({report.get('error')}; backup: {report.get('backup_path')})",
                        )
                        issues.append(
                            "state.db FTS write corruption and auto-repair failed — "
                            "restore from the backup copy beside state.db"
                        )
                else:
                    issues.append(
                        "state.db FTS write corruption — run 'hermes doctor --fix' "
                        "(or 'hermes sessions repair') to rebuild the FTS index"
                    )
        except Exception as e:
            from hermes_state import is_malformed_db_error, repair_state_db_schema

            if is_malformed_db_error(e):
                # sqlite_master itself is malformed (e.g. duplicate
                # messages_fts) — every statement fails before it runs, so
                # this is NOT a plain FTS-index rebuild. Repair sqlite_master
                # in place (backup first; sessions/messages preserved).
                check_warn(
                    f"{_DHH}/state.db schema is malformed (sessions hidden until repaired)",
                    f"({e})",
                )
                if should_fix:
                    report = repair_state_db_schema(state_db_path)
                    if report.get("repaired"):
                        try:
                            conn = sqlite3.connect(str(state_db_path))
                            count = conn.execute(
                                "SELECT COUNT(*) FROM sessions"
                            ).fetchone()[0]
                            conn.close()
                        except Exception:
                            count = "?"
                        backup_name = (
                            Path(report["backup_path"]).name
                            if report.get("backup_path") else "n/a"
                        )
                        check_ok(
                            f"Repaired state.db schema ({count} sessions recovered)",
                            f"(strategy: {report.get('strategy')}; backup: {backup_name})",
                        )
                        f.fixed += 1
                    else:
                        check_warn(
                            "state.db schema repair did not recover automatically",
                            f"({report.get('error')}; backup: {report.get('backup_path')})",
                        )
                        issues.append(
                            "state.db schema malformed and auto-repair failed — "
                            "restore from the backup copy beside state.db"
                        )
                else:
                    issues.append(
                        "state.db schema malformed — run 'hermes doctor --fix' "
                        "(or 'hermes sessions repair') to recover hidden sessions"
                    )
            else:
                check_warn(f"{_DHH}/state.db exists but has issues: {e}")

        # Health/stats snapshot (#statedb-visibility): a multi-GB state.db
        # with a runaway WAL was previously invisible to every Hermes
        # surface. Strictly read-only (mode=ro) so it is safe against a
        # live DB held by the gateway; any failure degrades to one info
        # line rather than failing doctor.
        try:
            from hermes_state import collect_state_db_stats, count_db_holders

            _db_stats = collect_state_db_stats(state_db_path)
            _db_holders = count_db_holders(state_db_path)
            for _kind, _text, _detail in _render_state_db_stats(
                _db_stats, holders=_db_holders
            ):
                if _kind == "warn":
                    check_warn(_text, _detail)
                    if "auto_prune" in _detail:
                        issues.append(
                            "state.db is large — enable sessions.auto_prune "
                            "in config.yaml"
                            + (
                                " and run 'hermes sessions optimize-storage' "
                                "offline (gateway stopped)"
                                if "optimize-storage" in _detail else ""
                            )
                        )
                else:
                    check_info(_text + (f" {_detail}" if _detail else ""))
        except Exception as _stats_exc:
            check_info(f"state.db stats unavailable ({_stats_exc})")
    else:
        check_info(f"{_DHH}/state.db not created yet (will be created on first session)")

    # Check WAL file size (unbounded growth indicates missed checkpoints)
    wal_path = hermes_home / "state.db-wal"
    if wal_path.exists():
        try:
            wal_size = wal_path.stat().st_size
            if wal_size > 50 * 1024 * 1024:  # 50 MB
                check_warn(
                    f"WAL file is large ({wal_size // (1024*1024)} MB)",
                    "(may indicate missed checkpoints)"
                )
                if should_fix:
                    import sqlite3
                    conn = sqlite3.connect(str(state_db_path))
                    conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
                    conn.close()
                    new_size = wal_path.stat().st_size if wal_path.exists() else 0
                    check_ok(f"WAL checkpoint performed ({wal_size // 1024}K → {new_size // 1024}K)")
                    f.fixed += 1
                else:
                    issues.append("Large WAL file — run 'hermes doctor --fix' to checkpoint")
            elif wal_size > 10 * 1024 * 1024:  # 10 MB
                check_info(f"WAL file is {wal_size // (1024*1024)} MB (normal for active sessions)")
        except Exception:
            pass
    return f


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


def _check_tool_availability(should_fix: bool) -> Finding:
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


def _check_skills_hub(should_fix: bool) -> Finding:
    f = Finding()
    hub_dir = HERMES_HOME / "skills" / ".hub"
    if hub_dir.exists():
        check_ok("Skills Hub directory exists")
        lock_file = hub_dir / "lock.json"
        if lock_file.exists():
            try:
                import json
                lock_data = json.loads(lock_file.read_text(encoding="utf-8"))
                count = len(lock_data.get("installed", {}))
                check_ok(f"Lock file OK ({count} hub-installed skill(s))")
            except Exception:
                check_warn("Lock file", "(corrupted or unreadable)")
        quarantine = hub_dir / "quarantine"
        q_count = sum(1 for d in quarantine.iterdir() if d.is_dir()) if quarantine.exists() else 0
        if q_count > 0:
            check_warn(f"{q_count} skill(s) in quarantine", "(pending review)")
    else:
        check_warn("Skills Hub directory not initialized", "(run: hermes skills list)")

    from hermes_cli.config import get_env_value

    def _gh_authenticated() -> bool:
        """Check if gh CLI is authenticated via token file or device flow."""
        try:
            result = subprocess.run(
                ["gh", "auth", "status", "--json", "authenticated"],
                capture_output=True, timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired):
            return False

    github_token = get_env_value("GITHUB_TOKEN") or get_env_value("GH_TOKEN")
    if github_token:
        check_ok("GitHub token configured (authenticated API access)")
    elif _gh_authenticated():
        check_ok("GitHub authenticated via gh CLI", "(full API access — no GITHUB_TOKEN needed)")
    else:
        check_warn("No GITHUB_TOKEN", f"(60 req/hr rate limit — set in {_DHH}/.env for better rates)")
    return f


def _check_memory_provider(should_fix: bool) -> Finding:
    f = Finding()
    issues = f.issues
    _active_memory_provider = _doctor_memory_config(HERMES_HOME).get("provider", "")

    if not _active_memory_provider:
        check_ok("Built-in memory active", "(no external provider configured — this is fine)")
    elif _active_memory_provider == "honcho":
        try:
            from plugins.memory.honcho.client import HonchoClientConfig, resolve_config_path
            hcfg = HonchoClientConfig.from_global_config()
            _honcho_cfg_path = resolve_config_path()

            if not _honcho_cfg_path.exists():
                # Config file missing — but env var fallback may have resolved it.
                # Only warn if the config didn't actually resolve from env vars.
                if hcfg.api_key or hcfg.base_url:
                    check_ok(
                        "Honcho configured via environment variables",
                        f"config file {_honcho_cfg_path} not found, using HONCHO_API_KEY env var",
                    )
                else:
                    check_warn("Honcho config not found", "run: hermes memory setup")
            elif not hcfg.enabled:
                check_info(f"Honcho disabled (set enabled: true in {_honcho_cfg_path} to activate)")
            elif not (hcfg.api_key or hcfg.base_url):
                _fail_and_issue(
                    "Honcho API key or base URL not set",
                    "run: hermes memory setup",
                    "No Honcho API key — run 'hermes memory setup'",
                    issues,
                )
            else:
                from plugins.memory.honcho.client import get_honcho_client, reset_honcho_client
                reset_honcho_client()
                try:
                    get_honcho_client(hcfg)
                    check_ok(
                        "Honcho connected",
                        f"workspace={hcfg.workspace_id} mode={hcfg.recall_mode} freq={hcfg.write_frequency}",
                    )
                except Exception as _e:
                    _fail_and_issue("Honcho connection failed", str(_e), f"Honcho unreachable: {_e}", issues)
        except ImportError:
            _fail_and_issue(
                "honcho-ai not installed",
                "pip install honcho-ai",
                "Honcho is set as memory provider but honcho-ai is not installed",
                issues,
            )
        except Exception as _e:
            check_warn("Honcho check failed", str(_e))
    elif _active_memory_provider == "mem0":
        try:
            from plugins.memory.mem0 import _load_config as _load_mem0_config
            mem0_cfg = _load_mem0_config()
            mem0_key = mem0_cfg.get("api_key", "")
            if mem0_key:
                check_ok("Mem0 API key configured")
                check_info(f"user_id={mem0_cfg.get('user_id', '?')}  agent_id={mem0_cfg.get('agent_id', '?')}")
            else:
                _fail_and_issue(
                    "Mem0 API key not set",
                    "(set MEM0_API_KEY in .env or run hermes memory setup)",
                    "Mem0 is set as memory provider but API key is missing",
                    issues,
                )
        except ImportError:
            _fail_and_issue(
                "Mem0 plugin not loadable",
                "pip install mem0ai",
                "Mem0 is set as memory provider but mem0ai is not installed",
                issues,
            )
        except Exception as _e:
            check_warn("Mem0 check failed", str(_e))
    else:
        # Generic check for other memory providers (openviking, hindsight, etc.)
        try:
            from plugins.memory import load_memory_provider
            _provider = load_memory_provider(_active_memory_provider)
            if _provider and _provider.is_available():
                check_ok(f"{_active_memory_provider} provider active")
            elif _provider:
                check_warn(f"{_active_memory_provider} configured but not available", "run: hermes memory status")
            else:
                check_warn(f"{_active_memory_provider} plugin not found", "run: hermes memory setup")
        except Exception as _e:
            check_warn(f"{_active_memory_provider} check failed", str(_e))
    return f


def _check_profiles(should_fix: bool) -> Finding:
    f = Finding()
    try:
        from hermes_cli.profiles import list_profiles, _get_wrapper_dir, profile_exists
        import re as _re

        named_profiles = [p for p in list_profiles() if not p.is_default]
        if named_profiles:
            _section("Profiles")
            check_ok(f"{len(named_profiles)} profile(s) found")
            wrapper_dir = _get_wrapper_dir()
            for p in named_profiles:
                parts = []
                if p.gateway_running:
                    parts.append("gateway running")
                if p.model:
                    parts.append(p.model[:30])
                if not (p.path / "config.yaml").exists():
                    parts.append("⚠ missing config")
                if not (p.path / ".env").exists():
                    parts.append("no .env")
                wrapper = wrapper_dir / p.name
                if not wrapper.exists():
                    parts.append("no alias")
                status = ", ".join(parts) if parts else "configured"
                check_ok(f"  {p.name}: {status}")

            # Check for orphan wrappers
            if wrapper_dir.is_dir():
                for wrapper in wrapper_dir.iterdir():
                    if not wrapper.is_file():
                        continue
                    try:
                        content = wrapper.read_text(encoding="utf-8")
                        if "hermes -p" in content:
                            _m = _re.search(r"hermes -p (\S+)", content)
                            if _m and not profile_exists(_m.group(1)):
                                check_warn(f"Orphan alias: {wrapper.name} → profile '{_m.group(1)}' no longer exists")
                    except Exception:
                        pass
    except ImportError:
        pass
    except Exception:
        pass
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
