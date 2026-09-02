"""HERMES_HOME state checks for hermes doctor: directories, memory files, state.db health, skills hub, memory provider, profiles.

Split out of ``hermes_cli/doctor.py``; every moved name is re-imported there, so
``hermes_cli.doctor.<name>`` keeps resolving (and monkeypatching) as before.
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from hermes_cli.doctor_report import (
    Finding,
    _fail_and_issue,
    _section,
    check_info,
    check_ok,
    check_warn,
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


def _memory_store_flags(hermes_home: Path) -> tuple:
    from tools.memory_tool import get_builtin_memory_store_flags

    return get_builtin_memory_store_flags({"memory": _doctor_memory_config(hermes_home)})


def _check_directory_structure(should_fix: bool) -> Finding:
    """HERMES_HOME, expected subdirs, SOUL.md, and the enabled built-in memory files."""
    from hermes_cli.doctor import HERMES_HOME, _DHH
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
    from hermes_cli.doctor import HERMES_HOME, _DHH
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


def _check_skills_hub(should_fix: bool) -> Finding:
    from hermes_cli.doctor import HERMES_HOME, _DHH
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
    from hermes_cli.doctor import HERMES_HOME
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
