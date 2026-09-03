"""HERMES_HOME state checks for hermes doctor: directories, memory files, state.db health, skills hub, memory provider, profiles.

Split out of ``hermes_cli/doctor.py``; every moved name is re-imported there, so
``hermes_cli.doctor.<name>`` keeps resolving (and monkeypatching) as before.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from hermes_cli.doctor_report import (
    Finding, _fail_and_issue, _section, check_bool, check_info, check_ok, check_warn, doctor_check, ensure_dir,
)
from hermes_cli.sizefmt import format_bytes as _human_bytes


def _honcho_is_configured_for_doctor() -> bool:
    """Return True when Honcho is configured, even if this process has no active session."""
    try:
        from plugins.memory.honcho.client import HonchoClientConfig
        cfg = HonchoClientConfig.from_global_config()
        return bool(cfg.enabled and (cfg.api_key or cfg.base_url))
    except Exception:
        return False


def _doctor_memory_config(hermes_home: Path | None = None) -> dict:
    """Return the effective memory section used by doctor diagnostics."""
    from hermes_cli.doctor import HERMES_HOME
    try:
        from hermes_cli.config import _expand_env_vars, read_user_config_raw
        config_path = (hermes_home if hermes_home is not None else HERMES_HOME) / "config.yaml"
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


# state.db size threshold (advisory only — deliberately a module constant, not config:
# doctor warnings are guidance, not policy).
STATE_DB_SIZE_WARN_BYTES = 1 * 1024 * 1024 * 1024   # 1 GiB logical size


def _bits(*pairs) -> list:
    """``[fmt() for value, fmt in pairs if value is not None]`` — present-only stat fragments."""
    return [fmt() for value, fmt in pairs if value is not None]


def _render_state_db_stats(stats: dict, holders=None) -> list:
    """Turn a collect_state_db_stats() dict into ``(kind, text, detail)`` rows, kind 'info' / 'warn'.

    Pure formatting — no I/O — so it is unit-testable without the doctor CLI. Tolerates None in every field.
    """
    lines: list = []
    stats = stats or {}
    logical, wal, freelist = (stats.get(k) for k in ("logical_size_bytes", "wal_size_bytes", "freelist_count"))
    size_bits = _bits(
        (logical, lambda: f"logical size {_human_bytes(logical)}"),
        (stats.get("page_count"), lambda: f"{stats['page_count']:,} pages"),
        (freelist, lambda: f"{freelist:,} free"),
        (wal, lambda: f"WAL {_human_bytes(wal)}"),
    )
    if size_bits:
        lines.append(("info", "state.db " + ", ".join(size_bits), ""))
    row_bits = _bits(
        (stats.get("messages"), lambda: f"{stats['messages']:,} messages"),
        (stats.get("sessions"), lambda: f"{stats['sessions']:,} sessions"),
        (stats.get("journal_mode") or None, lambda: f"journal_mode={stats['journal_mode']}"),
        (holders, lambda: f"{holders} process(es) holding the DB open"),
    )
    if row_bits:
        lines.append(("info", ", ".join(row_bits), ""))
    fts = stats.get("fts_tables")
    if fts:
        present = [t for t, ok in fts.items() if ok]
        lines.append(("info", "FTS tables: " + (", ".join(present) if present else "none"), ""))
    deferral = stats.get("fts_rebuild_deferral")
    if isinstance(deferral, dict):
        lines.append(("warn", f"state.db FTS repair is blocked after {deferral.get('attempts') or '?'} deferral(s) "
                      f"by PID(s) {deferral.get('holder_pids') or [] or 'unknown'}",
                      "(stop the listed processes, then run 'hermes sessions optimize-storage' with the gateway stopped)"))

    # Advisory: oversized database. Suggest auto_prune, and — when the v23 FTS rebuild is pending OR
    # the DB still carries the legacy inline trigram layout (fts_storage_version marker absent) —
    # the offline optimize-storage pass that migrates/compacts the FTS indexes.
    if logical is not None and logical > STATE_DB_SIZE_WARN_BYTES:
        detail = "consider enabling sessions.auto_prune in config.yaml to bound growth"
        legacy_trigram = fts is not None and fts.get("messages_fts_trigram") and stats.get("fts_storage_version") is None
        if stats.get("fts_rebuild_pending") or legacy_trigram:
            detail += "; run 'hermes sessions optimize-storage' offline (with the gateway stopped) to compact FTS storage"
        lines.append(("warn", f"state.db is large ({_human_bytes(logical)})", f"({detail})"))

    # WAL runaway is deliberately NOT warned here: the WAL check later in the state.db section already
    # warns above 50 MB and offers a checkpoint via --fix; a second warning would only duplicate it.
    return lines


def _memory_store_flags(hermes_home: Path) -> tuple:
    from tools.memory_tool import get_builtin_memory_store_flags
    return get_builtin_memory_store_flags({"memory": _doctor_memory_config(hermes_home)})


def _check_directory_structure(should_fix: bool) -> Finding:
    """HERMES_HOME, expected subdirs, SOUL.md, and the enabled built-in memory files."""
    from hermes_cli.doctor import HERMES_HOME, _DHH
    f = Finding()
    hermes_home = HERMES_HOME
    ensure_dir(f, should_fix, hermes_home, f"{_DHH} directory exists", f"Created {_DHH} directory", f"{_DHH} not found")

    _memory_enabled, _user_profile_enabled = _memory_store_flags(hermes_home)
    memory_on = _memory_enabled or _user_profile_enabled

    # Expected subdirectories. The built-in file store does not create or consume memories/ when
    # both targets are disabled, so stale migration files are not an active diagnostic surface.
    for subdir_name in ["cron", "sessions", "logs", "skills"] + (["memories"] if memory_on else []):
        ensure_dir(f, should_fix, hermes_home / subdir_name, f"{_DHH}/{subdir_name}/ exists",
                   f"Created {_DHH}/{subdir_name}/", f"{_DHH}/{subdir_name}/ not found")

    # SOUL.md persona file
    soul_path = hermes_home / "SOUL.md"
    if soul_path.exists():
        content = soul_path.read_text(encoding="utf-8").strip()
        # Template comments only (no real content)?
        if any(l.strip() and not l.strip().startswith(("<!--", "-->", "#")) for l in content.splitlines()):
            check_ok(f"{_DHH}/SOUL.md exists (persona configured)")
        else:
            check_info(f"{_DHH}/SOUL.md exists but is empty — edit it to customize personality")
    else:
        check_warn(f"{_DHH}/SOUL.md not found", "(create it to give Hermes a custom personality)")
        if should_fix:
            soul_path.parent.mkdir(parents=True, exist_ok=True)
            soul_path.write_text("# Hermes Agent Persona\n\n<!-- Edit this file to customize how Hermes communicates. -->\n\n"
                                 "You are Hermes, a helpful AI assistant.\n", encoding="utf-8")
            check_ok(f"Created {_DHH}/SOUL.md with basic template")
            f.fixed += 1

    # Check only enabled built-in stores. External providers are additive, but users can explicitly
    # disable either legacy file target; stale migration files must not read as active memory usage.
    memories_dir = hermes_home / "memories"
    if not memory_on:
        check_info("Built-in memory files disabled by config")
    elif not memories_dir.exists():
        check_warn(f"{_DHH}/memories/ not found", "(will be created on first use)")
        if should_fix:
            memories_dir.mkdir(parents=True, exist_ok=True)
            check_ok(f"Created {_DHH}/memories/")
            f.fixed += 1
    else:
        check_ok(f"{_DHH}/memories/ directory exists")
        for enabled, fname in ((_memory_enabled, "MEMORY.md"), (_user_profile_enabled, "USER.md")):
            mem_file = memories_dir / fname
            if enabled and mem_file.exists():
                check_ok(f"{fname} exists ({len(mem_file.read_text(encoding='utf-8').strip())} chars)")
            elif enabled:
                check_info(f"{fname} not created yet (will be created when the agent first writes a memory)")
    return f


def _session_count(state_db_path: Path):
    import sqlite3
    conn = sqlite3.connect(str(state_db_path))
    try:
        return conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
    finally:
        conn.close()


# Corruption class -> (ok label, not-fixed label, failed issue, fix hint). ``{count}`` = recovered sessions.
_STATE_DB_REPAIRS = {
    "fts": ("Repaired state.db FTS write health",
            "state.db FTS write-health repair did not recover automatically",
            "state.db FTS write corruption and auto-repair failed — restore from the backup copy beside state.db",
            "state.db FTS write corruption — run 'hermes doctor --fix' (or 'hermes sessions repair') to rebuild the FTS index"),
    "schema": ("Repaired state.db schema ({count} sessions recovered)",
               "state.db schema repair did not recover automatically",
               "state.db schema malformed and auto-repair failed — restore from the backup copy beside state.db",
               "state.db schema malformed — run 'hermes doctor --fix' (or 'hermes sessions repair') to recover hidden sessions"),
}


def _repair_state_db(f: Finding, should_fix: bool, state_db_path: Path, kind: str) -> None:
    """Shared --fix path for both state.db corruption classes (FTS write health, malformed schema)."""
    ok_label, not_fixed_label, failed_issue, fix_hint = _STATE_DB_REPAIRS[kind]
    if not should_fix:
        f.issues.append(fix_hint)
        return
    from hermes_state import repair_state_db_schema
    report = repair_state_db_schema(state_db_path)
    if not report.get("repaired"):
        check_warn(not_fixed_label, f"({report.get('error')}; backup: {report.get('backup_path')})")
        f.issues.append(failed_issue)
        return
    if "{count}" in ok_label:
        try:
            ok_label = ok_label.format(count=_session_count(state_db_path))
        except Exception:
            ok_label = ok_label.format(count="?")
    backup_name = Path(report["backup_path"]).name if report.get("backup_path") else "n/a"
    check_ok(ok_label, f"(strategy: {report.get('strategy')}; backup: {backup_name})")
    f.fixed += 1


def _state_db_health(f: Finding, should_fix: bool, state_db_path: Path, _DHH: str) -> None:
    """Session count + FTS write-health probe; malformed-schema path when even COUNT(*) fails."""
    try:
        check_ok(f"{_DHH}/state.db exists ({_session_count(state_db_path)} sessions)")
        # `SELECT COUNT(*)` succeeds even when the FTS index is corrupt and every message write
        # fails through the triggers; _db_opens_cleanly drives a rolled-back write to surface that.
        from hermes_state import _db_opens_cleanly
        _write_reason = _db_opens_cleanly(state_db_path)
        if _write_reason is not None:
            check_warn(f"{_DHH}/state.db fails a write-health probe (FTS index may be corrupt)", f"({_write_reason})")
            _repair_state_db(f, should_fix, state_db_path, "fts")
    except Exception as e:
        from hermes_state import is_malformed_db_error
        if not is_malformed_db_error(e):
            check_warn(f"{_DHH}/state.db exists but has issues: {e}")
            return
        # sqlite_master itself is malformed (e.g. duplicate messages_fts): every statement fails
        # before it runs, so this is NOT a plain FTS rebuild — repair sqlite_master in place (backup first).
        check_warn(f"{_DHH}/state.db schema is malformed (sessions hidden until repaired)", f"({e})")
        _repair_state_db(f, should_fix, state_db_path, "schema")


def _state_db_stats(issues: list, state_db_path: Path) -> None:
    """Health/stats snapshot: strictly read-only (mode=ro) so it is safe against a live DB held by
    the gateway; any failure degrades to one info line rather than failing doctor."""
    try:
        from hermes_state import collect_state_db_stats, count_db_holders
        rows = _render_state_db_stats(collect_state_db_stats(state_db_path), holders=count_db_holders(state_db_path))
        for _kind, _text, _detail in rows:
            if _kind != "warn":
                check_info(_text + (f" {_detail}" if _detail else ""))
                continue
            check_warn(_text, _detail)
            if "auto_prune" in _detail:
                issues.append("state.db is large — enable sessions.auto_prune in config.yaml"
                              + (" and run 'hermes sessions optimize-storage' offline (gateway stopped)" if "optimize-storage" in _detail else ""))
    except Exception as _stats_exc:
        check_info(f"state.db stats unavailable ({_stats_exc})")


def _state_db_wal(f: Finding, should_fix: bool, state_db_path: Path) -> None:
    """WAL file size (unbounded growth indicates missed checkpoints)."""
    wal_path = state_db_path.parent / "state.db-wal"
    try:
        wal_size = wal_path.stat().st_size if wal_path.exists() else 0
        if wal_size > 50 * 1024 * 1024:  # 50 MB
            check_warn(f"WAL file is large ({wal_size // (1024*1024)} MB)", "(may indicate missed checkpoints)")
            if not should_fix:
                f.issues.append("Large WAL file — run 'hermes doctor --fix' to checkpoint")
                return
            import sqlite3
            conn = sqlite3.connect(str(state_db_path))
            conn.execute("PRAGMA wal_checkpoint(PASSIVE)")
            conn.close()
            new_size = wal_path.stat().st_size if wal_path.exists() else 0
            check_ok(f"WAL checkpoint performed ({wal_size // 1024}K → {new_size // 1024}K)")
            f.fixed += 1
        elif wal_size > 10 * 1024 * 1024:  # 10 MB
            check_info(f"WAL file is {wal_size // (1024*1024)} MB (normal for active sessions)")
    except Exception:
        pass


def _check_state_db(should_fix: bool) -> Finding:
    """state.db session count, FTS write health, schema repair, stats snapshot, WAL size."""
    from hermes_cli.doctor import HERMES_HOME, _DHH
    f = Finding()
    state_db_path = HERMES_HOME / "state.db"
    if state_db_path.exists():
        _state_db_health(f, should_fix, state_db_path, _DHH)
        _state_db_stats(f.issues, state_db_path)
    else:
        check_info(f"{_DHH}/state.db not created yet (will be created on first session)")
    _state_db_wal(f, should_fix, state_db_path)
    return f


def _gh_authenticated() -> bool:
    """Check if gh CLI is authenticated via token file or device flow."""
    try:
        result = subprocess.run(["gh", "auth", "status", "--json", "authenticated"], capture_output=True, timeout=10)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _check_skills_hub(should_fix: bool) -> Finding:
    from hermes_cli.doctor import HERMES_HOME, _DHH
    f = Finding()
    hub_dir = HERMES_HOME / "skills" / ".hub"
    if not hub_dir.exists():
        check_warn("Skills Hub directory not initialized", "(run: hermes skills list)")
    else:
        check_ok("Skills Hub directory exists")
        lock_file = hub_dir / "lock.json"
        if lock_file.exists():
            try:
                import json
                count = len(json.loads(lock_file.read_text(encoding="utf-8")).get("installed", {}))
                check_ok(f"Lock file OK ({count} hub-installed skill(s))")
            except Exception:
                check_warn("Lock file", "(corrupted or unreadable)")
        quarantine = hub_dir / "quarantine"
        q_count = sum(1 for d in quarantine.iterdir() if d.is_dir()) if quarantine.exists() else 0
        if q_count > 0:
            check_warn(f"{q_count} skill(s) in quarantine", "(pending review)")

    from hermes_cli.config import get_env_value
    if get_env_value("GITHUB_TOKEN") or get_env_value("GH_TOKEN"):
        check_ok("GitHub token configured (authenticated API access)")
    else:
        check_bool(_gh_authenticated(), ("GitHub authenticated via gh CLI", "(full API access — no GITHUB_TOKEN needed)"),
                   ("No GITHUB_TOKEN", f"(60 req/hr rate limit — set in {_DHH}/.env for better rates)"))
    return f


def _memory_provider_honcho(issues: list) -> None:
    from plugins.memory.honcho.client import HonchoClientConfig, resolve_config_path
    hcfg = HonchoClientConfig.from_global_config()
    cfg_path = resolve_config_path()
    if not cfg_path.exists():
        # Config file missing — env-var fallback may still have resolved it.
        check_bool(hcfg.api_key or hcfg.base_url,
                   ("Honcho configured via environment variables", f"config file {cfg_path} not found, using HONCHO_API_KEY env var"),
                   ("Honcho config not found", "run: hermes memory setup"))
    elif not hcfg.enabled:
        check_info(f"Honcho disabled (set enabled: true in {cfg_path} to activate)")
    elif not (hcfg.api_key or hcfg.base_url):
        _fail_and_issue("Honcho API key or base URL not set", "run: hermes memory setup",
                        "No Honcho API key — run 'hermes memory setup'", issues)
    else:
        from plugins.memory.honcho.client import get_honcho_client, reset_honcho_client
        reset_honcho_client()
        try:
            get_honcho_client(hcfg)
            check_ok("Honcho connected",
                     f"workspace={hcfg.workspace_id} mode={hcfg.recall_mode} freq={hcfg.write_frequency}")
        except Exception as _e:
            _fail_and_issue("Honcho connection failed", str(_e), f"Honcho unreachable: {_e}", issues)


def _memory_provider_mem0(issues: list) -> None:
    from plugins.memory.mem0 import _load_config as _load_mem0_config
    mem0_cfg = _load_mem0_config()
    if mem0_cfg.get("api_key", ""):
        check_ok("Mem0 API key configured")
        check_info(f"user_id={mem0_cfg.get('user_id', '?')}  agent_id={mem0_cfg.get('agent_id', '?')}")
    else:
        _fail_and_issue("Mem0 API key not set", "(set MEM0_API_KEY in .env or run hermes memory setup)",
                        "Mem0 is set as memory provider but API key is missing", issues)


# provider -> (checker, ImportError row, ImportError issue, label for "check failed")
_MEMORY_PROVIDER_CHECKS = {
    "honcho": (_memory_provider_honcho, ("honcho-ai not installed", "pip install honcho-ai"),
               "Honcho is set as memory provider but honcho-ai is not installed", "Honcho"),
    "mem0": (_memory_provider_mem0, ("Mem0 plugin not loadable", "pip install mem0ai"),
             "Mem0 is set as memory provider but mem0ai is not installed", "Mem0"),
}


def _memory_provider_generic(name: str) -> None:
    """Generic check for other memory providers (openviking, hindsight, etc.)."""
    from plugins.memory import load_memory_provider
    _provider = load_memory_provider(name)
    if _provider and _provider.is_available():
        check_ok(f"{name} provider active")
    elif _provider:
        check_warn(f"{name} configured but not available", "run: hermes memory status")
    else:
        check_warn(f"{name} plugin not found", "run: hermes memory setup")


def _check_memory_provider(should_fix: bool) -> Finding:
    from hermes_cli.doctor import HERMES_HOME
    f = Finding()
    name = _doctor_memory_config(HERMES_HOME).get("provider", "")
    if not name:
        check_ok("Built-in memory active", "(no external provider configured — this is fine)")
        return f
    checker, missing_row, missing_issue, label = _MEMORY_PROVIDER_CHECKS.get(name, (None, None, None, name))
    try:
        checker(f.issues) if checker else _memory_provider_generic(name)
    except ImportError as _e:
        if missing_row is None:
            check_warn(f"{label} check failed", str(_e))
        else:
            _fail_and_issue(*missing_row, missing_issue, f.issues)
    except Exception as _e:
        check_warn(f"{label} check failed", str(_e))
    return f


@doctor_check()
def _check_profiles(should_fix: bool, f: Finding) -> None:
    from hermes_cli.profiles import list_profiles, _get_wrapper_dir, profile_exists
    import re as _re
    named_profiles = [p for p in list_profiles() if not p.is_default]
    if not named_profiles:
        return
    _section("Profiles")
    check_ok(f"{len(named_profiles)} profile(s) found")
    wrapper_dir = _get_wrapper_dir()
    for p in named_profiles:
        parts = []
        if p.gateway_running:
            parts.append("gateway running")
        if p.model:
            parts.append(p.model[:30])
        for missing, text in (("config.yaml", "⚠ missing config"), (".env", "no .env")):
            if not (p.path / missing).exists():
                parts.append(text)
        if not (wrapper_dir / p.name).exists():
            parts.append("no alias")
        check_ok(f"  {p.name}: {', '.join(parts) if parts else 'configured'}")

    # Orphan wrappers
    if wrapper_dir.is_dir():
        for wrapper in wrapper_dir.iterdir():
            if not wrapper.is_file():
                continue
            try:
                _m = _re.search(r"hermes -p (\S+)", wrapper.read_text(encoding="utf-8"))
                if _m and not profile_exists(_m.group(1)):
                    check_warn(f"Orphan alias: {wrapper.name} → profile '{_m.group(1)}' no longer exists")
            except Exception:
                pass
