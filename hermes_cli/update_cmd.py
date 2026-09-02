"""Hermes update pipeline — extracted from ``hermes_cli/main.py``.

Mechanical move (main.py decomposition): ``_cmd_update_impl``, ``_cmd_update_check``
and every module-level helper used only by the update path, plus the update-only
constants they read. Function bodies are lifted verbatim; the only mechanical
change is that references to helpers/constants that STAY in ``hermes_cli.main``
(and to moved-but-test-patched siblings) are routed through ``_m()`` — a lazy
``hermes_cli.main`` reference — so existing call sites and test monkeypatches
that target ``hermes_cli.main.<name>`` (``PROJECT_ROOT``, ``_is_windows``,
``_run_pre_update_backup``, ...) keep working unchanged. ``main.py`` re-imports
every public-ish name from here (``# noqa: F401``) so the argparse wiring and
the test-patch surface still resolve on ``hermes_cli.main``.

The closures that used to be nested inside ``_cmd_update_impl`` (``_print_items``,
``_wait_for_service_active``, ``_service_restart_sec``, ``_resolve_manage_cmd``,
``_restart_one_systemd_gateway_unit``) now live at module level or inside the
phase helpers ``_cmd_update_impl`` calls in order: ``_pull_updates`` ->
``_sync_python_dependencies_after_pull`` -> ``_run_post_update_maintenance`` ->
``_restart_gateway_fleet_after_update`` -> ``_verify_fleet_after_update``.

Imports are one-way: ``hermes_cli.main`` imports this module, never the reverse
at import time (``_m()`` resolves lazily at call time, when main.py is fully
loaded, so there is no import cycle).
"""

import logging
import os
import shlex
import subprocess
import sys
import time as _time
from dataclasses import dataclass
from pathlib import Path

from hermes_constants import get_default_hermes_root, venv_python_path

# Abort recovery lives in its own bounded module (review on #96235). Re-exported
# here because `hermes_cli.main` and the split update modules address these
# names through `update_cmd` (tests patch them here).
from hermes_cli.update_abort_recovery import (  # noqa: F401
    _abort_recovery_is_complete,
    _qualified_serve_skips,
    _recover_gateway_restart_after_abort,
    _serve_unit_recovery_available,
    _surviving_pre_update_serve_runtimes,
    _warn_stale_serve_runtimes,
)
from hermes_cli.update_cmd_windows import (  # noqa: F401  (re-exported; tests patch hermes_cli.update_cmd.<name>)
    _HOLDER_VALUE_FLAGS_FALLBACK,
    _clear_windows_venv_holders_or_exit,
    _cold_start_windows_gateway_after_update,
    _desktop_owns_gateway_lifecycle,
    _detect_venv_python_processes,
    _format_venv_python_holders_message,
    _handoff_reapable_backend_pids,
    _hermes_holder_subcommand,
    _holder_value_flags,
    _holder_value_flags_cache,
    _ledger_manual_serve_holders,
    _ledger_reapable_backend_pids,
    _leftover_pausable_gateway_pids,
    _looks_like_desktop_control_plane,
    _orphaned_desktop_backend_pids,
    _pause_windows_gateways_for_update,
    _refresh_bootstrap_cache_scripts,
    _refresh_windows_gateway_launchers,
    _refuse_gateway_ancestor_tree_kill,
    _relaunch_stopped_serves,
    _restore_windows_gateway_service,
    _resume_windows_gateways_after_update,
    _resume_windows_gateways_and_merge_outcome,
    _self_and_non_gateway_ancestor_pids,
    _serve_relaunch_commands,
    _start_windows_gateway_service,
    _stop_process_trees,
    _stop_windows_gateway_service,
    _venv_launcher_ancestors,
    _wait_for_windows_update_gateway_exit,
    _write_update_planned_stop_marker,
)
from hermes_cli.update_cmd_fleet import (  # noqa: F401  (re-exported; tests patch hermes_cli.update_cmd.<name>)
    _FLEET_RESTART_PENDING_NAME,
    _FRESH_RESTART_SUPERVISORS,
    _GatewayRestartOutcome,
    _apply_pending_fleet_restart_catchup,
    _clear_fleet_restart_pending_marker,
    _current_checkout_sha,
    _drain_or_signal_gateway_for_update,
    _fleet_probe_expected_runtimes,
    _fleet_restart_pending_marker_path,
    _for_each_systemd_gateway_unit,
    _gateway_recovery_partition,
    _gateway_service_matches_profile,
    _pending_fleet_restart_needed,
    _receipt_looks_unfinished,
    _receipt_reports_stale_runtime,
    _resolve_manage_cmd,
    _restart_gateway_fleet_after_update,
    _restart_launchd_gateway_after_update,
    _restart_macos_launchd_gateways,
    _restart_phase_failure_is_incomplete,
    _restart_systemd_gateway_units,
    _restart_systemd_gateway_units_best_effort,
    _run_pending_fleet_restart,
    _service_restart_sec,
    _service_unit_supports_graceful_sigusr1_restart,
    _surviving_gateway_pids_after_failed_restart,
    _systemctl,
    _systemctl_reset_and_restart,
    _verify_fleet_after_update,
    _wait_for_service_active,
    _warn_gateway_restart_phase_aborted,
    _warn_incomplete_gateway_fleet_restart,
    _warn_pending_fleet_restart,
    _warn_pending_fleet_restart_on_startup,
    _write_fleet_restart_pending_marker,
    _write_gateway_update_exit_code,
)
from hermes_cli.update_cmd_zip import (  # noqa: F401  (re-exported; tests patch hermes_cli.update_cmd.<name>)
    _ZIP_PRESERVED_TOP_LEVEL,
    _ZIP_STAGING_ARTIFACT_SUFFIXES,
    _abort_zip_update_if_dirty_tree,
    _atomic_replace_dir,
    _commit_staged_replacements,
    _discard_staged,
    _is_zip_preserved_entry_status_line,
    _is_zip_staging_artifact_status_line,
    _stage_replacement,
    _update_via_zip,
    _zip_overlay_block_reason,
)
from hermes_cli.update_cmd_stash import (  # noqa: F401  (re-exported; tests patch hermes_cli.update_cmd.<name>)
    _AUTOSTASH_NAME_PREFIX,
    _AUTOSTASH_WARN_AGE_DAYS,
    _discard_stashed_changes,
    _git_untracked_paths,
    _park_stashed_changes,
    _print_stash_cleanup_guidance,
    _reject_unsafe_stash_restore,
    _resolve_stash_selector,
    _restore_stashed_changes,
    _restored_python_paths,
    _stash_apply_failed_only_on_existing_untracked,
    _stash_local_changes_if_needed,
    _warn_orphaned_update_autostashes,
)
from hermes_cli.update_cmd_config import (  # noqa: F401  (re-exported; tests patch hermes_cli.update_cmd.<name>)
    _LAST_SIBLING_SNAPSHOTS,
    _check_and_apply_config_migration,
    _migrate_sibling_profile_configs,
    _print_items,
    _reload_config_modules,
    _run_config_check_fresh,
    _run_migrate_config_fresh,
)
from hermes_cli.update_cmd_deps import (  # noqa: F401  (re-exported; tests patch hermes_cli.update_cmd.<name>)
    _INSTALL_DEFINING_FILES,
    _SELF_LOCKING_NATIVE_MODULES,
    _UPDATE_CRITICAL_MODULES,
    _abort_dependency_sync_if_self_locked,
    _capture_active_lazy_features,
    _capture_active_tool_dependencies,
    _critical_module_import_failures,
    _defer_update_for_self_lock,
    _dependency_sync_would_rewrite,
    _desktop_app_present,
    _detect_self_loaded_native_modules,
    _editable_install_is_current,
    _ensure_uv_for_termux,
    _ensure_venv_pip,
    _install_psutil_android_compat,
    _is_android_python,
    _npm_bin_exists,
    _npm_lockfile_changed,
    _npm_manifest_paths,
    _npm_manifests_digest,
    _path_uid,
    _rebuild_desktop_after_update,
    _record_npm_lockfile_hash,
    _refresh_active_lazy_features,
    _refresh_active_memory_provider_dependencies,
    _refuse_update_if_venv_foreign_owned,
    _repair_node_deps_on_current_checkout,
    _restore_active_tool_dependencies,
    _sync_python_dependencies_after_pull,
    _update_node_dependencies,
    _upgrade_pip_before_lazy_refresh,
    _validate_critical_modules_import,
    _venv_core_imports_healthy,
    _venv_foreign_owned_paths,
    _web_build_toolchain_ready,
    _web_toolchain_roots,
)
from hermes_cli.update_cmd_git import (  # noqa: F401  (re-exported; tests patch hermes_cli.update_cmd.<name>)
    OFFICIAL_REPO_URL,
    OFFICIAL_REPO_URLS,
    SKIP_UPSTREAM_PROMPT_FILE,
    _ORPHAN_RESCUE_REFS_TO_KEEP,
    _ORPHAN_RESCUE_REF_MAX_AGE_DAYS,
    _add_upstream_remote,
    _assess_parked_branch_switch,
    _branch_head_label,
    _branch_head_suffix,
    _classify_fetch_failure,
    _count_commits_between,
    _discard_lockfile_churn,
    _ensure_non_trampoline_git,
    _get_origin_url,
    _git_is_trampoline,
    _has_upstream_remote,
    _is_fork,
    _locate_real_git,
    _mark_skip_upstream_prompt,
    _normalize_managed_eol,
    _portable_git_candidates,
    _print_fetch_failure,
    _print_parked_branch_kept_notice,
    _print_parked_branch_skip_warning,
    _prune_orphan_rescue_refs,
    _should_skip_upstream_prompt,
    _sync_fork_with_upstream,
    _sync_with_upstream_if_needed,
)
from hermes_cli.update_cmd_maint import (  # noqa: F401  (re-exported; tests patch hermes_cli.update_cmd.<name>)
    _PRE_UPDATE_SNAPSHOT_KEEP,
    _PRE_UPDATE_SNAPSHOT_MAX_FILE_SIZE,
    _STALE_PURGE_PREFIXES,
    _STALE_PURGE_PROTECTED,
    _UPDATE_RUNTIME_RELOAD_MODULES,
    _clear_stale_sqlite_sidecars,
    _ensure_acp_launcher,
    _ensure_fhs_path_guard,
    _finish_dashboard_update_cleanup,
    _format_time_ago,
    _post_update_sqlite_runtime_status,
    _print_bundled_skills_sync_report,
    _print_curator_first_run_notice,
    _print_curator_recent_run_notice,
    _print_fts_optimize_available_notice,
    _print_update_completion,
    _print_update_summary,
    _print_verified_update_completion,
    _purge_stale_hermes_modules,
    _read_project_version,
    _reload_process_scan_modules,
    _reload_updated_runtime_modules,
    _resolve_pre_update_backup_mode,
    _restore_state_db_from_snapshot,
    _run_post_update_maintenance,
    _run_pre_update_backup,
    _sweep_bytecode_after_update,
    _update_complete_message,
    _verify_and_restore_one_state_db,
    _verify_and_restore_state_dbs_post_update,
)
logger = logging.getLogger(__name__)


def _m():
    """Lazy ``hermes_cli.main`` reference.

    Keeps ``hermes_cli.main.<helper>`` test patches effective in this code
    path and keeps the ``main`` -> ``update_cmd`` import one-way at import time.
    """
    from hermes_cli import main

    return main


def _no_prompt_git_kwargs() -> dict:
    """``subprocess.run`` kwargs for the updater's network git calls.

    GitHub answers anonymous fetches with HTTP 401 during outages (and for
    unreachable repos); git then prompts ``Username for 'https://github.com':``
    on the inherited terminal and the update sits there forever. Disable the
    prompt so the fetch fails fast into ``_classify_fetch_failure``. Only the
    *prompt* is disabled — a configured credential helper / askpass still
    runs, so a private-fork origin keeps authenticating non-interactively.
    """
    env = dict(os.environ)
    env["GIT_TERMINAL_PROMPT"] = "0"
    env["GCM_INTERACTIVE"] = "Never"
    return {"stdin": subprocess.DEVNULL, "env": env}


# Files that must parse right after an update/install (CLI startup imports;
# ``web_server.py`` is the desktop backend a fresh Windows install launches).
# The post-pull syntax guard validates these and auto-rolls-back on failure.
_UPDATE_CRITICAL_FILES = (
    "hermes_cli/main.py",
    "hermes_cli/config.py",
    "hermes_cli/__init__.py",
    "hermes_cli/web_server.py",
    "cli.py",
    "run_agent.py",
    "model_tools.py",
    "toolsets.py",
    "hermes_constants.py",
)


def _record_update_step(step: str, ok: bool, detail: str = "") -> None:
    """Best-effort ``update_receipt.record_step``; the receipt must never break an update."""
    try:
        from hermes_cli.update_receipt import record_step

        record_step(step, ok, detail)
    except Exception:
        pass


def _git_run(git_cmd, args, cwd=None, *, check=False, network=False):
    """Run ``git_cmd + args`` (default cwd: the checkout), capturing utf-8 text.

    ``network=True`` (fetch/pull/push) disables git's terminal prompt so an
    HTTP 401 fails fast instead of hanging on ``Username for ...``.
    """
    return subprocess.run(
        git_cmd + args,
        cwd=_m().PROJECT_ROOT if cwd is None else cwd,
        capture_output=True,
        text=True, encoding="utf-8", errors="replace",
        check=check,
        **(_no_prompt_git_kwargs() if network else {}),
    )


def _capture_head_sha(git_cmd, cwd) -> str | None:
    """Return the current HEAD SHA, or None if it can't be resolved."""
    try:
        result = _git_run(git_cmd, ["rev-parse", "HEAD"], cwd, check=True)
        return result.stdout.strip() or None
    except (subprocess.CalledProcessError, OSError):
        return None


def _validate_python_files_syntax(
    root, relpaths
) -> tuple[bool, str | None, str | None]:
    """Compile *relpaths* under *root* without writing bytecode into the tree."""
    import py_compile
    import tempfile

    root = Path(root)
    with tempfile.TemporaryDirectory(prefix="hermes-syntax-check-") as tmpdir:
        for relpath in relpaths:
            path = root / relpath
            if not path.exists():
                continue
            cfile = Path(tmpdir) / (str(relpath).replace("/", "__") + "c")
            try:
                py_compile.compile(str(path), cfile=str(cfile), doraise=True)
            except py_compile.PyCompileError as exc:
                return False, str(path), str(exc)
            except OSError as exc:
                return False, str(path), f"could not read: {exc}"
    return True, None, None


def _validate_critical_files_syntax(root) -> tuple[bool, str | None, str | None]:
    """Compile each file in ``_UPDATE_CRITICAL_FILES`` to catch SyntaxErrors.

    These are imported on every ``hermes`` startup; a syntax error (orphan
    conflict markers, etc.) means the CLI can't bootstrap, so we validate
    after ``git pull`` and auto-roll-back instead of leaving a bricked install.

    The ``.pyc`` goes to a temp dir, not the tree's ``__pycache__/``: avoids
    racing concurrent test workers and leaving a stale pyc behind when the
    next interpreter run uses a different Python. Only the compile-or-not
    signal matters.

    Returns ``(ok, failing_path, error_message)``.
    """
    return _validate_python_files_syntax(root, _UPDATE_CRITICAL_FILES)


def _gateway_prompt(prompt_text: str, default: str = "", timeout: float = 300.0) -> str:
    """File-based IPC prompt for gateway mode.

    Writes a prompt marker file for the gateway to forward to the user, then
    polls for a response file; falls back to *default* on timeout. Lets
    ``hermes update --gateway`` forward prompts (stash restore, config
    migration) to the messenger instead of silently skipping them.
    """
    import json as _json
    import uuid as _uuid
    from hermes_constants import get_hermes_home

    home = get_hermes_home()
    prompt_path = home / ".update_prompt.json"
    response_path = home / ".update_response"

    response_path.unlink(missing_ok=True)

    payload = {
        "prompt": prompt_text,
        "default": default,
        "id": str(_uuid.uuid4()),
    }
    tmp = prompt_path.with_suffix(".tmp")
    tmp.write_text(_json.dumps(payload), encoding="utf-8")
    tmp.replace(prompt_path)

    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        if response_path.exists():
            try:
                answer = response_path.read_text(encoding="utf-8").strip()
                response_path.unlink(missing_ok=True)
                prompt_path.unlink(missing_ok=True)
                return answer if answer else default
            except (OSError, ValueError):
                pass
        _time.sleep(0.5)

    prompt_path.unlink(missing_ok=True)
    response_path.unlink(missing_ok=True)
    print(f"  (no response after {int(timeout)}s, using default: {default!r})")
    return default


def _called_process_error_cmd_parts(exc: subprocess.CalledProcessError) -> list[str]:
    """Normalize ``CalledProcessError.cmd`` into argv-style tokens."""
    cmd = exc.cmd
    if cmd is None:
        return []
    if isinstance(cmd, (str, bytes)):
        text = cmd.decode("utf-8", "replace") if isinstance(cmd, bytes) else cmd
        try:
            return shlex.split(text, posix=os.name != "nt")
        except ValueError:
            return text.split()
    return [str(part) for part in cmd]


def _called_process_error_is_git(exc: subprocess.CalledProcessError) -> bool:
    """True when the failed subprocess was git itself."""
    parts = _called_process_error_cmd_parts(exc)
    if not parts:
        return False
    # Windows argv may use backslashes; basename() on POSIX would otherwise
    # keep the whole path. Normalize separators before taking the name.
    name = os.path.basename(parts[0].replace("\\", "/")).lower()
    return name in {"git", "git.exe"}


def _called_process_error_is_python_dep_install(
    exc: subprocess.CalledProcessError,
) -> bool:
    """True when the failed subprocess was a uv/pip (or ensurepip) install."""
    parts = [part.lower() for part in _called_process_error_cmd_parts(exc)]
    if not parts:
        return False
    exe = os.path.basename(parts[0].replace("\\", "/"))
    if "ensurepip" in parts:
        return True
    if "install" in parts and (
        "pip" in parts or exe in {"pip", "pip.exe", "pip3", "pip3.exe", "uv", "uv.exe"}
    ):
        return True
    return False


def _format_update_failure_stage(exc: subprocess.CalledProcessError) -> str:
    """Name the update stage that actually failed.

    The git pull and the Python-dependency install share one ``try`` in
    ``_cmd_update_impl``. Calling every ``CalledProcessError`` a git failure
    (the historical Windows message) sent users hunting in the wrong place
    and, worse, keyed the ZIP overlay on exception *type* rather than on git
    actually having failed (#87304, #85840).
    """
    if _called_process_error_is_python_dep_install(exc):
        return "Python dependency install failed"
    if _called_process_error_is_git(exc):
        return "Git update failed"
    return "Update step failed"


def _shim_quarantine_error_type() -> "type[BaseException]":
    """The strict-quarantine refusal type, resolved lazily through ``_m()``.

    Falls back to a never-raised private type when main.py lacks it (torn
    mid-update tree), so the ``except`` clause stays valid.
    """
    cls = getattr(_m(), "ShimQuarantineError", None)
    if isinstance(cls, type) and issubclass(cls, BaseException):
        return cls

    class _Never(Exception):
        pass

    return _Never


def _refuse_update_for_contended_shims(exc: BaseException) -> None:
    """Refuse the dependency sync when live shims could not be quarantined.

    #87331 fail-closed half: a shim rename that failed every retry proves a
    process holds the venv without FILE_SHARE_DELETE — running the installer
    anyway is exactly how the venv ends up stranded between versions. The
    code swap (when one happened) is already committed; only the dependency
    install is deferred, via the update-incomplete marker, to the next fresh
    launch after the holder exits. Exits 2 (refused) so the command-boundary
    receipt net records it as a refusal, not a failure.
    """
    print("✗ Cannot continue the update: live Hermes launcher(s) could not be")
    print("  moved aside:")
    for name in getattr(exc, "failed_shims", []) or ["hermes.exe"]:
        print(f"    {name}")
    print("  Another process is holding this install's venv — typically Hermes")
    print("  Desktop, a gateway, or another hermes REPL — and mutating the venv")
    print("  now would strand it half-updated.")
    print("  The dependency install has been deferred: close the process(es)")
    print("  above, then run any `hermes` command to finish it automatically.")
    # Idempotent: the git path already dropped the marker before the sync;
    # this covers the ZIP/repair paths so the deferral is never silent.
    _write_update_incomplete_marker()
    sys.exit(2)


def _should_zip_fallback_on_update_error(exc: BaseException) -> bool:
    """ZIP fallback is for Windows git file-I/O breakage, not later stages.

    A dependency-install failure (locked ``hermes.exe`` / ``uv pip install``
    exit 2) is not a git failure. The pull has already succeeded by then, so
    re-downloading the source ZIP cannot fix the install and would replace
    every top-level entry except ``venv`` / ``node_modules`` / ``.git`` /
    ``.env`` — permanently deleting uncommitted edits and untracked files.
    """
    return (
        isinstance(exc, subprocess.CalledProcessError)
        and _m()._is_windows()
        and _called_process_error_is_git(exc)
    )


def _print_called_process_error_tail(
    exc: subprocess.CalledProcessError, *, limit: int = 12
) -> None:
    """Print a captured stderr/stdout tail when the failing call recorded one."""
    blob = exc.stderr or exc.stdout or ""
    if isinstance(blob, bytes):
        blob = blob.decode("utf-8", "replace")
    lines = [line for line in str(blob).splitlines() if line.strip()]
    if not lines:
        return
    print("  Last output:")
    for line in lines[-limit:]:
        print(f"    {line}")


def _invalidate_update_cache():
    """Delete the update-check cache for ALL profiles.

    The git repo is shared, so one profile's update makes every profile
    current; a per-profile cache would show a stale "commits behind" banner.
    """
    homes = []
    # Default profile home (Docker-aware — uses /opt/data in Docker)

    default_home = get_default_hermes_root()
    homes.append(default_home)
    # Named profiles under <root>/profiles/
    profiles_root = default_home / "profiles"
    if profiles_root.is_dir():
        for entry in profiles_root.iterdir():
            if entry.is_dir():
                homes.append(entry)
    for home in homes:
        try:
            cache_file = home / ".update_check"
            if cache_file.exists():
                cache_file.unlink()
        except Exception:
            pass


def _write_marker_file(path: Path, *, label: str) -> None:
    """Drop an update-recovery breadcrumb. Never raises."""
    if _m()._pytest_owns_live_checkout(path.parent):
        logger.debug("Skipping %s marker under pytest (live checkout)", label)
        return
    try:
        path.write_text(
            f"started={_time.time()}\npid={os.getpid()}\n", encoding="utf-8"
        )
    except OSError as exc:
        logger.debug("Could not write %s marker: %s", label, exc)


def _write_update_incomplete_marker() -> None:
    """Drop the interrupted core-install breadcrumb. Never raises."""
    _write_marker_file(_m()._update_marker_path(), label="update-incomplete")


def _write_lazy_refresh_incomplete_marker() -> None:
    """Drop the interrupted lazy-refresh breadcrumb. Never raises."""
    _write_marker_file(_m()._lazy_refresh_marker_path(), label="lazy-refresh-incomplete")


def _format_concurrent_instances_message(
    matches: list[tuple[int, str]], scripts_dir: Path
) -> str:
    """Build a human-readable explanation + remediation hint for the user."""
    shim = scripts_dir / "hermes.exe"
    lines = ["✗ Another hermes.exe is running:"]
    for pid, name in matches:
        lines.append(f"    PID {pid}  {name}")
    lines.append("")
    lines.append(f"  Updating now would fail to overwrite {shim} because")
    lines.append("  Windows blocks REPLACE on a running executable.")
    lines.append("")
    lines.append("  Close Hermes Desktop, exit any open `hermes` REPLs, and")
    lines.append("  stop the gateway (`hermes gateway stop`) before retrying.")
    lines.append("")
    if matches:
        pid_args = " ".join(f"/PID {pid}" for pid, _ in matches)
        lines.append("  If you've already closed everything and these PIDs are")
        lines.append("  stale, terminate them directly, then retry the update:")
        lines.append(f"      taskkill {pid_args} /F")
        lines.append("")
    lines.append("  Override with `hermes update --force` if you've already")
    lines.append("  confirmed those processes will not write to the venv.")
    return "\n".join(lines)


def _classify_concurrent_instance(pid: int) -> str:
    """Return ``"gateway"`` when ``pid``'s command line is a gateway runtime.

    Delegates to ``_is_pausable_gateway`` — the same canonical ``gateway run``
    matcher used by the Desktop preflight exemption and the venv-holder guard
    — so a PID classified ``"gateway"`` here is exactly the set the downstream
    pause/kill+restart machinery will stop. That symmetry lets the pre-update
    concurrent gate skip the abort for gateway-only matches instead of making
    the user kill a gateway that is about to be paused anyway.

    Returns ``"non-gateway"`` when the cmdline doesn't match and ``"unknown"``
    when psutil can't read it; the gate treats ``"unknown"`` as non-gateway
    (better to block than proceed against an unidentified process).
    """
    try:
        import psutil  # noqa: PLC0415
    except Exception:
        return "unknown"

    try:
        proc = psutil.Process(int(pid))
        cmdline_list = proc.cmdline()
    except Exception:
        return "unknown"

    from hermes_cli._scan_venv_blockers import _is_pausable_gateway  # noqa: PLC0415

    cmdline = " ".join(cmdline_list or [])
    if _is_pausable_gateway(cmdline):
        return "gateway"
    return "non-gateway"


def _filter_non_gateway_concurrent_instances(
    matches: list[tuple[int, str]],
) -> list[tuple[int, str]]:
    """Return only the concurrent-instance matches that are NOT the gateway.

    If every concurrent instance is a gateway, the pause machinery and the
    post-update kill+restart handle it and the update proceeds. Anything else
    (TUI shell, Desktop backend child, another ``hermes`` REPL) has no pause
    machinery downstream, so the gate still aborts.
    """
    non_gateway: list[tuple[int, str]] = []
    for pid, name in matches:
        if _classify_concurrent_instance(pid) != "gateway":
            non_gateway.append((pid, name))
    return non_gateway


def _log_only_write(text: str) -> None:
    """Write ``text`` to ``~/.hermes/logs/update.log`` only, never the terminal.

    During ``hermes update`` ``sys.stdout`` is an ``_UpdateOutputStream``
    mirroring to terminal and log; this reaches past it to the log handle so
    loud, low-signal subprocess output (npm, Electron/vite, cua-driver "Next
    steps") stays debuggable without flooding the terminal.
    """
    if not text:
        return
    stream = _m().sys.stdout
    log_file = getattr(stream, "_log", None)
    if log_file is None:
        return
    try:
        log_file.write(text if text.endswith("\n") else text + "\n")
        log_file.flush()
    except Exception:
        pass


def _run_logged_subprocess(cmd, *, cwd=None, env=None):
    """Run ``cmd`` capturing combined output into update.log (not the terminal).

    Returns the ``CompletedProcess`` (with ``stdout`` populated) so the caller
    can decide whether to surface the captured output on failure.
    """
    result = subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    _log_only_write(result.stdout or "")
    return result


def _cmd_update_check(branch: str = "main", *, branch_explicit: bool = False):
    """Implement ``hermes update --check``: fetch and report without installing.

    ``branch`` selects which branch the check compares against. Default is
    "main"; callers can pass another branch to ask "are there new commits
    on origin/<branch>?" without performing the update.

    ``branch_explicit`` is True iff the caller passed --branch on the CLI.
    Installs that can't honor non-default branches (e.g. Docker) surface a
    one-line notice instead of silently dropping the flag.
    """
    # Shared admission gate (#91277 Phase 3): same marker-first decision as
    # the apply path, so --check can never report git state for an install
    # whose real update mechanism is an image pull.
    from hermes_cli.update_contract import (
        evaluate_update_admission,
        record_refusal_receipt,
    )

    refusal = evaluate_update_admission(_m().PROJECT_ROOT)
    if refusal is not None:
        print(refusal.message)
        record_refusal_receipt(refusal)
        sys.exit(2)

    git_dir = _m().PROJECT_ROOT / ".git"
    if not git_dir.exists():
        print("✗ Not a git repository — cannot check for updates.")
        sys.exit(1)

    git_cmd = ["git"]
    if sys.platform == "win32":
        git_cmd = ["git", "-c", "windows.appendAtomically=false"]

    # An interrupted fetch can leave .git/shallow.lock (or another lock) behind,
    # making every later fetch fail with "File exists". Self-heal before fetching.
    from hermes_cli.gitlock import clear_stale_git_locks, clear_stale_tmp_packs

    cleared = clear_stale_git_locks(_m().PROJECT_ROOT)
    for lock_path in cleared:
        print(f"  (removed stale git lock: {lock_path})")
    # Aborted fetches on flaky lines also strand tmp_pack_* debris in
    # .git/objects/pack — unchecked it reached 6 GB and corrupted the pack
    # dir outright (#93732). Same age+process safety contract as the locks.
    swept = clear_stale_tmp_packs(_m().PROJECT_ROOT)
    if swept:
        print(f"  (removed {len(swept)} aborted-fetch pack temp file(s))")

    # Fetch only <branch>: a bare `git fetch <remote>` pulls thousands of
    # auto-generated branches. Prefer upstream as canonical, but only for main
    # (a fork's non-default branch has no upstream counterpart). Installer
    # checkouts are shallow (`--depth 1`); a plain fetch would unshallow them
    # and rev-list would report a huge bogus "behind" count, so fetch with
    # --depth 1 and report presence-only.
    is_shallow = (
        _git_run(git_cmd, ["rev-parse", "--is-shallow-repository"]).stdout.strip()
        == "true"
    )
    depth_args = ["--depth", "1"] if is_shallow else []

    if branch == "main":
        # Probe locally (~6 ms) for an 'upstream' remote before spending a
        # network fetch (~0.3-1 s) that non-fork installs would always fail.
        has_upstream_remote = (
            _git_run(git_cmd, ["remote", "get-url", "upstream"]).returncode
            == 0
        )
        fetch_result = None
        if has_upstream_remote:
            print("→ Fetching from upstream...")
            fetch_result = _git_run(git_cmd, ["fetch"] + depth_args + ["upstream", branch], network=True)
        if fetch_result is not None and fetch_result.returncode == 0:
            compare_branch = f"upstream/{branch}"
        else:
            # No upstream remote, or the upstream fetch failed — use origin.
            print("→ Fetching from origin...")
            fetch_result = _git_run(git_cmd, ["fetch"] + depth_args + ["origin", branch], network=True)
            compare_branch = f"origin/{branch}"
    else:
        # Non-default branch: compare against origin/<branch> directly.
        print("→ Fetching from origin...")
        fetch_result = _git_run(git_cmd, ["fetch"] + depth_args + ["origin", branch], network=True)
        compare_branch = f"origin/{branch}"

    if fetch_result.returncode != 0:
        _print_fetch_failure(fetch_result.stderr)
        sys.exit(1)

    # Verify the compare ref exists first: rev-list on a bogus ref exits 128
    # and (with check=True) would surface a Python traceback.
    verify_result = _git_run(git_cmd, ["rev-parse", "--verify", "--quiet", compare_branch])
    if verify_result.returncode != 0:
        print(f"✗ Branch '{branch}' not found on {compare_branch.split('/', 1)[0]}.")
        sys.exit(1)

    if is_shallow:
        # No history across the shallow boundary: compare tip SHAs (like the
        # banner's _check_via_local_git), then recover the exact count via the
        # GitHub compare API, whose graph is complete.
        head_sha = _git_run(git_cmd, ["rev-parse", "HEAD"]).stdout.strip()
        target_sha = _git_run(git_cmd, ["rev-parse", compare_branch]).stdout.strip()
        if head_sha and target_sha and head_sha == target_sha:
            print("✓ Already up to date.")
        else:
            from hermes_cli.banner import _github_compare_behind
            from hermes_cli.config import recommended_update_command

            counted = _github_compare_behind(head_sha, target_sha)
            if counted == 0:
                # Local commits on top of the remote tip — not behind.
                print("✓ Already up to date.")
                return
            if counted is not None:
                commits_word = "commit" if counted == 1 else "commits"
                print(f"⚕ Update available: {counted} {commits_word} behind {compare_branch}.")
            else:
                print(f"⚕ Update available (behind {compare_branch}).")
            print(f"  Run '{recommended_update_command()}' to install.")
        return

    rev_result = _git_run(git_cmd, ["rev-list", f"HEAD..{compare_branch}", "--count"], check=True)
    behind = int(rev_result.stdout.strip())

    if behind == 0:
        print("✓ Already up to date.")
    else:
        commits_word = "commit" if behind == 1 else "commits"
        print(f"⚕ Update available: {behind} {commits_word} behind {compare_branch}.")
        from hermes_cli.config import recommended_update_command

        print(f"  Run '{recommended_update_command()}' to install.")


def _repair_current_checkout(
    *,
    assume_yes,
    gateway_mode,
    pre_update_snapshot_id,
    desktop_dir,
    had_desktop_app_before_update,
    active_lazy_features,
    active_tool_dependencies,
    upstream_checked,
    _windows_gateway_resume,
) -> bool:
    """Already-up-to-date path: keep the managed runtime current and repair a broken venv.

    A current checkout does not imply a healthy install (a prior dependency
    sync may have died partway), and the Windows shim hand-off child lands
    here BY DESIGN to run the sync its parent could not. Returns whether the
    checkout can be reported as complete.
    """
    # "No new commits" does not mean the managed interpreter is safe.
    # uv can retain the same CPython patch while python-build-standalone
    # refreshes the embedded SQLite underneath it. Keep the existing
    # update-boundary hook active on this retry path too.
    from hermes_cli.managed_uv import ensure_uv, update_managed_uv

    runtime_repairs = []
    update_managed_uv(repair_observer=runtime_repairs.append)
    ensure_uv(repair_observer=runtime_repairs.append)
    runtime_repaired = next(
        (result for result in runtime_repairs if result.repaired),
        None,
    )

    # A current checkout does NOT imply a healthy install: a prior sync may
    # have died partway (Windows: locked .pyd → uv/pip access-denied, venv
    # stranded between versions). Probe core imports and repair, or
    # "Already up to date!" hides a bricked install.
    healthy, detail = _venv_core_imports_healthy()
    # The Windows shim hand-off child exists to run the sync its parent
    # could not; the checkout is current BY DESIGN, so the pending sync —
    # not venv health — is the question. Without this it would print
    # "Already up to date!" and skip its one job.
    handed_off_sync = os.environ.get(_m()._UPDATE_REEXEC_ENV) == "1"
    current_checkout_complete = True
    if handed_off_sync:
        print("→ Finishing the dependency install handed off by hermes.exe...")
    elif not healthy:
        print("⚠ Checkout is current, but the venv is unhealthy:")
        print(f"  {detail}")
        print("→ Repairing Python dependencies...")
    if handed_off_sync or not healthy:
        # Self-lock deferral (#86735): the repair rewrites the venv
        # too — same mapped-extension hazard as the update sync.
        _m()._abort_dependency_sync_if_self_locked(_windows_gateway_resume)
        _write_update_incomplete_marker()
        from hermes_cli.managed_uv import ensure_uv

        repair_uv = ensure_uv()
        # A managed install whose venv is gone entirely (interrupted
        # repair after the old venv was moved aside) needs the venv
        # recreated before dependencies can be installed into it.
        venv_python_missing = not (
            venv_python_path(
                _m().PROJECT_ROOT / "venv", windows=_m()._is_windows()
            )
        ).exists()
        if venv_python_missing and repair_uv:
            print("→ Recreating virtual environment...")
            subprocess.run(
                [repair_uv, "venv", "venv"],
                cwd=_m().PROJECT_ROOT,
                check=False,
            )
        if repair_uv:
            # Isolated from third-party UV env vars (#83914), same as
            # the main-path and git-path dependency syncs.
            from hermes_cli.managed_uv import managed_python_env

            repair_env = managed_python_env()
            repair_env["VIRTUAL_ENV"] = str(_m().PROJECT_ROOT / "venv")
            _m()._install_python_dependencies_with_optional_fallback(
                [repair_uv, "pip"], env=repair_env, group="all"
            )
            _m()._refresh_active_lazy_features(
                [repair_uv, "pip"],
                env=repair_env,
                features=active_lazy_features,
            )
            _m()._restore_active_tool_dependencies(
                active_tool_dependencies,
                [repair_uv, "pip"],
                env=repair_env,
            )
        else:
            _m()._install_python_dependencies_with_optional_fallback(
                [sys.executable, "-m", "pip"], group="all"
            )
            _m()._refresh_active_lazy_features(
                [sys.executable, "-m", "pip"],
                features=active_lazy_features,
            )
            _m()._restore_active_tool_dependencies(
                active_tool_dependencies,
                [sys.executable, "-m", "pip"],
            )
        _m()._clear_update_incomplete_marker()
        healthy_after, detail_after = _venv_core_imports_healthy()
        if healthy_after:
            print("✓ Dependencies repaired!")
            _check_and_apply_config_migration(
                assume_yes=assume_yes,
                gateway_mode=gateway_mode,
                pre_update_snapshot_id=pre_update_snapshot_id,
            )
            # The hand-off child never reaches the commits-pulled rebuild,
            # so rebuild the Desktop app here or it stays on the old build (#97343).
            if _rebuild_desktop_after_update(
                desktop_dir,
                had_desktop_app_before_update=had_desktop_app_before_update,
            ):
                current_checkout_complete = _print_verified_update_completion(
                    "✓ Update complete!"
                )
            else:
                current_checkout_complete = False
                _print_update_completion(
                    "⚠ Update partially complete — the desktop app was "
                    "not rebuilt and is still on the previous build."
                )
        else:
            current_checkout_complete = False
            print(f"⚠ Venv still unhealthy after repair: {detail_after}")
            print("  Close all Hermes windows/gateways and re-run: hermes update")
    else:
        current_checkout_complete = _repair_node_deps_on_current_checkout(
            _print_verified_update_completion,
            assume_yes=assume_yes,
            gateway_mode=gateway_mode,
            pre_update_snapshot_id=pre_update_snapshot_id,
            completion_message=(
                "✓ Already up to date!"
                if upstream_checked
                else "✓ Up to date with your fork (official repo not checked)."
            ),
            had_desktop_app_before_update=had_desktop_app_before_update,
        )
    if runtime_repaired is not None and not _m()._is_windows():
        print()
        print(
            "⚠ Restart required to finish the managed Python runtime repair."
        )
        print(
            "  Any running Hermes gateways, Desktop backends, or other "
            "long-lived processes still use the previous runtime."
        )
        print("  Restart each of them to pick up the repaired runtime.")
    return current_checkout_complete


def _pull_updates(
    git_cmd,
    branch,
    auto_stash_ref,
    *,
    prompt_for_restore,
    gw_input_fn,
    discard_local_changes,
    keep_stash,
):
    """Fast-forward the checkout onto ``origin/<branch>`` and settle the autostash.

    Divergence is handled by shape (custom branch -> merge, same branch ->
    reset, orphan history -> rescue ref first); a post-pull syntax error in a
    critical file rolls back to the pre-pull SHA. Exits the process on
    failure. Returns the pre-pull HEAD SHA (or None).
    """
    update_succeeded = False
    # Pre-pull SHA for auto-rollback when pulled code has a syntax error in
    # a critical file (PR #28452: stray conflict markers in config.py
    # bricked every updater for 7 minutes).
    pre_pull_sha = _capture_head_sha(git_cmd, _m().PROJECT_ROOT)
    try:
        # Merge the ref we already fetched above (→ Fetching updates...)
        # instead of `git pull`, which performs a SECOND network fetch of
        # the same branch (~0.5-1.5 s of redundant round-trip per update).
        # `merge --ff-only origin/<branch>` is byte-identical in effect to
        # `pull --ff-only origin <branch>` given the fresh tracking ref;
        # the divergence fallback below is unchanged.
        pull_result = _git_run(git_cmd, ["merge", "--ff-only", f"origin/{branch}"])
        if pull_result.returncode != 0:
            # ff-only failed — local and remote have diverged. Before
            # assuming an upstream force-push, check WHY: a checkout on a
            # custom branch (local commits on top of origin/<branch>) also
            # cannot fast-forward, and `reset --hard` here would silently
            # discard that work. Merge instead and stop cleanly on
            # conflict — an update must never destroy local commits.
            _cur_branch = (
                _git_run(git_cmd, ["branch", "--show-current"]).stdout
                or ""
            ).strip()
            if _cur_branch and _cur_branch != branch:
                print(
                    f"  ⚠ Checkout is on custom branch '{_cur_branch}' — "
                    f"merging origin/{branch} instead of resetting so local commits survive..."
                )
                # Best-effort safety tag; recovery anchor if anything goes wrong.
                subprocess.run(
                    git_cmd
                    + ["tag", f"pre-update-{_time.strftime('%Y%m%d-%H%M%S')}"],
                    cwd=_m().PROJECT_ROOT,
                    capture_output=True,
                    check=False,
                )
                merge_result = _git_run(git_cmd, ["merge", "--no-edit", f"origin/{branch}"])
                if merge_result.returncode != 0:
                    subprocess.run(
                        git_cmd + ["merge", "--abort"],
                        cwd=_m().PROJECT_ROOT,
                        capture_output=True,
                        check=False,
                    )
                    print(
                        "✗ Merge conflict between local commits and upstream — "
                        "update stopped, nothing was changed."
                    )
                    print(
                        f"  Resolve manually: cd {_m().PROJECT_ROOT} && "
                        f"git merge origin/{branch}"
                    )
                    print(
                        "  Then re-run the update. Local work is untouched."
                    )
                    sys.exit(1)
            else:
                # Same branch as the target — a true upstream force-push/
                # rebase; local changes are stashed, so reset to the remote.
                # Orphan divergence (no common ancestor: corrupted HEAD, repo
                # re-init — #87694) would lose the whole local commit graph,
                # so park pre_pull_sha behind a rescue ref first.
                merge_base_result = _git_run(git_cmd, ["merge-base", "HEAD", f"origin/{branch}"])
                has_common_ancestor = bool(
                    merge_base_result.returncode == 0
                    and merge_base_result.stdout.strip()
                )
                if not has_common_ancestor and pre_pull_sha:
                    from datetime import datetime as _dt, timezone

                    # SHA suffix (not just a 1s timestamp) so two updates in
                    # the same second get distinct refs instead of overwriting.
                    rescue_ref = (
                        f"refs/hermes-update-backups/orphan-{branch}-"
                        f"{_dt.now(timezone.utc).strftime('%Y%m%d-%H%M%S')}"
                        f"-{pre_pull_sha[:12]}"
                    )
                    update_ref_result = _git_run(git_cmd, ["update-ref", rescue_ref, pre_pull_sha])
                    if update_ref_result.returncode == 0:
                        print(
                            "  ⚠ Local history shares no common ancestor with "
                            f"origin/{branch} (orphan divergence) — backed up "
                            f"current HEAD to {rescue_ref} before resetting. "
                            f"This backup expires after "
                            f"{_ORPHAN_RESCUE_REF_MAX_AGE_DAYS} days."
                        )
                    else:
                        # update-ref's return code is intentionally not
                        # fatal (disk full, permissions) — but don't tell
                        # the user a backup exists when the write failed.
                        print(
                            "  ⚠ Local history shares no common ancestor with "
                            f"origin/{branch} (orphan divergence) — attempted "
                            f"to back up current HEAD to {rescue_ref} before "
                            "resetting, but the backup write failed "
                            f"(pre-reset SHA was {pre_pull_sha})."
                        )
                    _prune_orphan_rescue_refs(git_cmd, _m().PROJECT_ROOT, branch)
                print(
                    "  ⚠ Fast-forward not possible (history diverged), resetting to match remote..."
                )
                reset_result = _git_run(git_cmd, ["reset", "--hard", f"origin/{branch}"])
                if reset_result.returncode != 0:
                    print(f"✗ Failed to reset to origin/{branch}.")
                    if reset_result.stderr.strip():
                        print(f"  {reset_result.stderr.strip()}")
                    print(
                        f"  Try manually: git fetch origin && git reset --hard origin/{branch}"
                    )
                    sys.exit(1)

        # Post-pull syntax guard: a bad commit that slipped past CI
        # (admin-merge bypass) is caught here and rolled back so the CLI
        # stays bootable until a fix lands.
        syntax_ok, failing_path, syntax_error = _validate_critical_files_syntax(
            _m().PROJECT_ROOT
        )
        if not syntax_ok:
            print()
            print("✗ Pulled code has a syntax error in a critical file:")
            print(f"  {failing_path}")
            if syntax_error:
                # py_compile errors can be multi-line; show the first
                # ~6 lines so the user sees the actual SyntaxError text.
                for line in str(syntax_error).splitlines()[:6]:
                    print(f"    {line}")
            if pre_pull_sha:
                print()
                print(f"→ Rolling back to {pre_pull_sha[:10]}...")
                rollback_result = _git_run(git_cmd, ["reset", "--hard", pre_pull_sha])
                if rollback_result.returncode == 0:
                    print("  ✓ Rollback complete — your install is unchanged.")
                    print("  Try ``hermes update`` again later once a fix lands.")
                else:
                    print("  ✗ Rollback failed. Recover manually with:")
                    print(f"    cd {_m().PROJECT_ROOT} && git reset --hard {pre_pull_sha}")
                    if rollback_result.stderr.strip():
                        print(f"    ({rollback_result.stderr.strip().splitlines()[0]})")
            else:
                print()
                print("  Could not capture pre-pull SHA — recover manually with:")
                print(f"    cd {_m().PROJECT_ROOT} && git reflog && git reset --hard <prev-sha>")
            sys.exit(1)

        update_succeeded = True
    finally:
        if auto_stash_ref is not None:
            # Don't attempt stash restore if the code update itself failed —
            # working tree is in an unknown state.
            if not update_succeeded:
                print(
                    f"  ℹ️  Local changes preserved in stash (ref: {auto_stash_ref})"
                )
                print("  Restore manually with: git stash apply")
            elif discard_local_changes:
                # Non-interactive update + user opted into discarding local
                # source edits (updates.non_interactive_local_changes:
                # discard). Throw the stash away instead of re-applying it.
                _m()._discard_stashed_changes(
                    git_cmd,
                    _m().PROJECT_ROOT,
                    auto_stash_ref,
                )
            elif keep_stash:
                # --keep-stash (desktop updater): the update landed; leave
                # local edits parked in the stash instead of silently
                # re-applying them onto the updated code.
                _m()._park_stashed_changes(auto_stash_ref)
            else:
                _m()._restore_stashed_changes(
                    git_cmd,
                    _m().PROJECT_ROOT,
                    auto_stash_ref,
                    prompt_user=prompt_for_restore,
                    input_fn=gw_input_fn,
                )
    return pre_pull_sha


@dataclass
class _CheckoutPlan:
    """What the pre-pull checkout phase decided (see ``_prepare_checkout_for_update``)."""

    auto_stash_ref: "str | None"
    commit_count: int
    in_place_update: bool
    parked_branch_switched: bool
    prompt_for_restore: bool
    switch_block_reason: "str | None"
    upstream_checked: bool


def _prepare_checkout_for_update(
    git_cmd,
    branch,
    current_branch,
    *,
    is_fork,
    assume_yes,
    gateway_mode,
    gw_input_fn,
    switch_branch,
    _windows_gateway_resume,
):
    """Apply the parked-branch guard, land on the update target, stash, and count new commits.

    Exits the process when the checkout is unsafe to move or the target branch
    does not exist. ``commit_count`` is 0 when up to date, -1 when tips differ
    but the shallow count is unrecoverable.
    """
    switch_block_reason = None  # only meaningful when parked_branch_switched
    # Parked-branch guard: a checkout parked on a stale feature branch
    # used to stash-switch-pull-switch-back, "updating" main while the
    # running code stayed behind. Routing by branch contents +
    # updates.parked_branch_strategy:
    #   fully merged  -> switch back to the target.
    #   unmerged: N   -> "switch" (default): switch anyway (commits are
    #                    safe on the branch) with a loud "kept" notice;
    #                    deterministic for non-interactive callers.
    #                    "update_in_place": merge origin/<target> INTO
    #                    the branch — checkout never moves, local commits
    #                    survive. --switch-branch overrides for one run.
    #   anything else -> dirty/unverifiable/opted out: touch nothing,
    #                    warn, mark the code update SKIPPED, stop.
    parked_branch_switched = False
    in_place_update = False
    if current_branch != branch and current_branch != "HEAD":
        switch_safe, switch_block_reason = _m()._assess_parked_branch_switch(
            git_cmd, _m().PROJECT_ROOT, current_branch, branch
        )
        if not switch_safe:
            _m()._print_parked_branch_skip_warning(
                git_cmd,
                _m().PROJECT_ROOT,
                current_branch,
                branch,
                switch_block_reason,
            )
            print()
            print(
                "⚠ Update finished — code update SKIPPED"
                f"{_branch_head_suffix(git_cmd, _m().PROJECT_ROOT)}"
            )
            _m()._resume_windows_gateways_after_update(
                _windows_gateway_resume
            )
            sys.exit(1)
        if switch_block_reason.startswith("unmerged:"):
            _in_place_configured = False
            try:
                from hermes_cli.config import load_config as _load_cfg

                _upd_cfg = (_load_cfg() or {}).get("updates", {})
                _in_place_configured = (
                    isinstance(_upd_cfg, dict)
                    and _upd_cfg.get("parked_branch_strategy", "switch")
                    == "update_in_place"
                )
            except Exception as exc:
                logger.debug(
                    "Could not read updates.parked_branch_strategy: %s", exc
                )
            if _in_place_configured and not switch_branch:
                # The merge source must exist upstream; --branch typos
                # previously surfaced through the checkout failing, which
                # does not run on this path.
                verify_ref = _git_run(git_cmd, ["rev-parse", "--verify", "--quiet", f"origin/{branch}"])
                if verify_ref.returncode != 0:
                    print(f"✗ Branch '{branch}' does not exist locally or on origin.")
                    sys.exit(1)
                in_place_update = True
                print(
                    f"  ℹ On branch '{current_branch}' — updating it in place from "
                    f"origin/{branch} (no branch switch; local commits preserved)."
                )
            else:
                parked_branch_switched = True
                _m()._print_parked_branch_kept_notice(
                    current_branch,
                    branch,
                    switch_block_reason.split(":", 1)[1],
                )
        else:
            parked_branch_switched = True
            print(
                f"  ⚠ Checkout was parked on '{current_branch}' "
                f"(fully merged) — switching back to {branch}..."
            )

    if not in_place_update and current_branch != branch:
        if current_branch == "HEAD":
            print(
                f"  ⚠ Currently on detached HEAD — switching to {branch} "
                "for update..."
            )
        # Stash before checkout so uncommitted work isn't lost
        auto_stash_ref = _m()._stash_local_changes_if_needed(git_cmd, _m().PROJECT_ROOT)
        checkout_result = _git_run(git_cmd, ["checkout", branch])
        if checkout_result.returncode != 0:
            # Branch not local yet — set it up tracking origin/<branch>.
            track_result = _git_run(git_cmd, ["checkout", "-B", branch, f"origin/{branch}"])
            if track_result.returncode != 0:
                # Restore the user's prior stash before bailing
                # so we don't leave them stranded in a weird state.
                if auto_stash_ref is not None:
                    _m()._restore_stashed_changes(
                        git_cmd,
                        _m().PROJECT_ROOT,
                        auto_stash_ref,
                        prompt_user=False,
                        input_fn=gw_input_fn,
                    )
                print(f"✗ Branch '{branch}' does not exist locally or on origin.")
                if track_result.stderr.strip():
                    print(f"  {track_result.stderr.strip().splitlines()[0]}")
                sys.exit(1)
    else:
        auto_stash_ref = _m()._stash_local_changes_if_needed(git_cmd, _m().PROJECT_ROOT)

    prompt_for_restore = (
        auto_stash_ref is not None
        and not assume_yes
        and (gateway_mode or (sys.stdin.isatty() and sys.stdout.isatty()))
    )

    # Check if there are updates. On shallow checkouts `rev-list --count`
    # walks the truncated graph and can report the entire remote ancestry
    # (e.g. "Found 9980 new commit(s)" on a depth-1 install — #53479).
    # The zero/nonzero gate is still sound (HEAD == origin/<branch> counts
    # 0), so keep it, but treat the shallow NUMBER as unknown and recover
    # the real one via the GitHub compare API when possible.
    result = _git_run(git_cmd, ["rev-list", f"HEAD..origin/{branch}", "--count"], check=True)
    commit_count = int(result.stdout.strip())

    apply_is_shallow = (
        _git_run(git_cmd, ["rev-parse", "--is-shallow-repository"]).stdout.strip()
        == "true"
    )
    if commit_count > 0 and apply_is_shallow:
        from hermes_cli.banner import _github_compare_behind

        head_sha = _git_run(git_cmd, ["rev-parse", "HEAD"]).stdout.strip()
        target_sha = _git_run(git_cmd, ["rev-parse", f"origin/{branch}"]).stdout.strip()
        counted = _github_compare_behind(head_sha, target_sha)
        # counted == 0 means local-ahead (remote tip reachable from HEAD):
        # not behind, fall through to the up-to-date path.
        commit_count = counted if counted is not None else -1

    # A fork can match origin yet trail upstream, so the upstream sync can
    # move HEAD with commit_count == 0. Detect that BEFORE the no-update
    # return so deps, restarts AND the fleet matrix still run (#73108 —
    # the sync used to live inside the early-return branch and verified
    # nothing). Non-forks have no upstream question.
    upstream_checked = True
    if commit_count == 0 and is_fork and branch == "main":
        pre_sync_sha = _capture_head_sha(git_cmd, _m().PROJECT_ROOT)
        upstream_checked = _m()._sync_with_upstream_if_needed(
            git_cmd,
            _m().PROJECT_ROOT,
            assume_yes=assume_yes,
            input_fn=gw_input_fn,
        )
        post_sync_sha = _capture_head_sha(git_cmd, _m().PROJECT_ROOT)
        if pre_sync_sha and post_sync_sha and pre_sync_sha != post_sync_sha:
            synced_count = _count_commits_between(
                git_cmd,
                _m().PROJECT_ROOT,
                pre_sync_sha,
                post_sync_sha,
            )
            # HEAD moving is itself proof of an update. Keep the update
            # path active even if the informational count cannot be read.
            commit_count = max(1, synced_count)

    return _CheckoutPlan(
        auto_stash_ref=auto_stash_ref,
        commit_count=commit_count,
        in_place_update=in_place_update,
        parked_branch_switched=parked_branch_switched,
        prompt_for_restore=prompt_for_restore,
        switch_block_reason=switch_block_reason,
        upstream_checked=upstream_checked,
    )


def _cmd_update_impl(args, gateway_mode: bool):
    """Body of ``cmd_update`` — kept separate so the wrapper can always
    restore stdio even on ``sys.exit``."""
    # A managed-runtime refresh can replace site-packages before the normal
    # ``.[all]`` install runs. Snapshot while the old environment can still
    # prove which optional backends the user had activated.
    active_lazy_features = _m()._capture_active_lazy_features()
    active_tool_dependencies = _m()._capture_active_tool_dependencies()

    # Snapshot the pre-update version before any code is pulled so the
    # completion line can report the transition (prime-agent#630 port).
    pre_update_version = _read_project_version()
    # In gateway mode, use file-based IPC for prompts instead of stdin
    gw_input_fn = (
        (lambda prompt, default="": _gateway_prompt(prompt, default))
        if gateway_mode
        else None
    )
    assume_yes = bool(getattr(args, "yes", False))
    # --keep-stash (desktop updater): stash local changes so the update can
    # proceed, but never re-apply them afterward — they stay parked in git
    # stash. Only applies when an update actually landed; abort/no-op paths
    # still restore, since the tree they restore onto is unchanged.
    keep_stash = bool(getattr(args, "keep_stash", False))
    # --switch-branch: on a branch carrying unmerged commits, prefer switching
    # to the update target over an in-place merge, so the branch's history is
    # never written to by an update (#89507 review feedback). Only meaningful
    # when updates.parked_branch_strategy is "update_in_place".
    switch_branch = bool(getattr(args, "switch_branch", False))

    # Whether this update is running without a human at the keyboard.
    # Interactive terminal updates always stash-and-ask (unchanged behavior);
    # only non-interactive updates (desktop/chat app, gateway, `--yes`) consult
    # the `updates.non_interactive_local_changes` config setting to decide
    # whether to auto-restore stashed local source changes or throw them away.
    _non_interactive_update = (
        gateway_mode
        or assume_yes
        or not (sys.stdin.isatty() and sys.stdout.isatty())
    )
    discard_local_changes = False
    if _non_interactive_update:
        try:
            from hermes_cli.config import load_config

            _update_cfg = (load_config() or {}).get("updates", {})
            if isinstance(_update_cfg, dict):
                _mode = str(_update_cfg.get("non_interactive_local_changes", "stash")).lower()
                discard_local_changes = _mode == "discard"
        except Exception as exc:
            # Never let a config read failure change the safe default.
            logger.debug("Could not read updates.non_interactive_local_changes: %s", exc)
            discard_local_changes = False

    print("⚕ Updating Hermes Agent...")
    print()

    # Phase 1 (#91277): structured update receipt — record what this run
    # discovers, does, and skips, so silent-failure classes (#88848,
    # #74973, #85753, #81193) become diagnosable from disk.
    try:
        from hermes_cli.update_receipt import begin_update_receipt

        begin_update_receipt()
    except Exception as _receipt_exc:
        logger.debug("Update receipt unavailable: %s", _receipt_exc)

    # Plan phase (#91277): snapshot every running runtime, supervisor and
    # code version into the receipt (read-only; probe failure records
    # nothing). Re-read AFTER the restart phase to reconcile planned
    # runtimes against bookkeeping — the plan is the worklist.
    _pre_update_plan = None
    try:
        from hermes_cli.update_inventory import (
            collect_runtime_inventory,
            record_plan_in_receipt,
        )

        _pre_update_plan = collect_runtime_inventory()
        record_plan_in_receipt(_pre_update_plan)
        if _pre_update_plan.runtimes:
            _n = len(_pre_update_plan.runtimes)
            _profiles = ", ".join(
                sorted({r.profile for r in _pre_update_plan.runtimes})
            )
            print(f"→ Fleet: {_n} running service(s) across profiles: {_profiles}")
    except Exception as _plan_exc:
        logger.debug("Update plan phase failed: %s", _plan_exc)

    # Windows: abort if another hermes.exe holds the venv shim — continuing
    # yields WinError 32 spam and a deferred-rename leftover or silent ZIP
    # fallback (#26670). Exception (#37039): instances positively identified
    # as gateways are paused by ``_pause_windows_gateways_for_update`` below
    # and restarted afterwards; anything else (TUI, Desktop backend,
    # unreadable cmdline) still aborts.
    if _m()._is_windows() and not getattr(args, "force", False):
        scripts_dir = _m()._venv_scripts_dir()
        if scripts_dir is not None:
            concurrent = _m()._detect_concurrent_hermes_instances(scripts_dir)
            if concurrent:
                non_gateway = _m()._filter_non_gateway_concurrent_instances(
                    concurrent
                )
                if non_gateway:
                    print(
                        _format_concurrent_instances_message(
                            non_gateway, scripts_dir
                        )
                    )
                    sys.exit(2)

    # Pre-update backup — runs before any git/file mutation so users can
    # always roll back to the exact state they had before this update.
    # Returns the quick-snapshot id (or None when disabled/failed); the
    # post-update cron-jobs safety net uses it to detect job loss.
    pre_update_snapshot_id = _m()._run_pre_update_backup(args)
    _record_update_step(
        "pre_update_backup",
        pre_update_snapshot_id is not None,
        f"snapshot={pre_update_snapshot_id}" if pre_update_snapshot_id else "disabled or failed",
    )

    _windows_gateway_resume = _m()._pause_windows_gateways_for_update()
    if _windows_gateway_resume:
        import atexit as _atexit

        _atexit.register(
            _m()._resume_windows_gateways_after_update,
            _windows_gateway_resume,
        )

    # With gateways paused, any venv python still running (typically the
    # Desktop `hermes serve` backend) keeps .pyd files locked and would
    # corrupt the sync; refuse rather than race (the app respawns a killed
    # backend). NOT bypassed by --force: the desktop updater passes it to
    # skip the shim guard but only probes the shim and app.asar.
    # --force-venv is the explicit escape hatch.
    if _m()._is_windows() and not getattr(args, "force_venv", False):
        _clear_windows_venv_holders_or_exit(args, gateway_mode, _windows_gateway_resume)

    # Self-lock deferral moved: the venv-holder sweep above excludes this
    # process by design (a CLI `hermes update` IS the venv python), and an
    # updater that has imported a native venv extension cannot rewrite its
    # own mapped .pyd (#83569). That check used to run HERE — before the
    # fetch — but firing pre-fetch meant a deferral stranded the user on the
    # OLD checkout, and any startup path that eagerly loaded cryptography
    # turned every Windows update into an exit-2 loop (#86735/#86780/#86781).
    # It now runs via _abort_dependency_sync_if_self_locked() after the code
    # swap, immediately before the dependency sync — the only phase the lock
    # can actually break — and only when the sync would truly rewrite the
    # loaded distribution.

    # Capture this after every fail-closed venv guard, but before either
    # update path can remove the ignored release tree.
    desktop_dir = _m().PROJECT_ROOT / "apps" / "desktop"
    had_desktop_app_before_update = _desktop_app_present(desktop_dir)

    # Try git-based update first, fall back to ZIP download on Windows
    # when git file I/O is broken (antivirus, NTFS filter drivers, etc.)
    use_zip_update = False
    git_dir = _m().PROJECT_ROOT / ".git"

    if not git_dir.exists():
        if sys.platform == "win32":
            use_zip_update = True
        else:
            print("✗ Not a git repository. Please reinstall:")
            print(
                "  curl -fsSL https://hermes-agent.nousresearch.com/install.sh | bash"
            )
            sys.exit(1)

    # On Windows, git can fail with "unable to write loose object file: Invalid argument"
    # due to filesystem atomicity issues. Set the recommended workaround.
    if sys.platform == "win32" and git_dir.exists():
        subprocess.run(
            [
                "git",
                "-c",
                "windows.appendAtomically=false",
                "config",
                "windows.appendAtomically",
                "false",
            ],
            cwd=_m().PROJECT_ROOT,
            check=False,
            capture_output=True,
        )

    # Build git command once — reused for fork detection and the update itself.
    git_cmd = ["git"]
    if sys.platform == "win32":
        git_cmd = ["git", "-c", "windows.appendAtomically=false"]
    # A broken Git-for-Windows trampoline refuses every git call with a
    # "BUG (fork bomb)" guard instead of running; swap in a real binary up
    # front so the normal git path survives instead of degrading to ZIP
    # (#87876).
    git_cmd = _ensure_non_trampoline_git(git_cmd)

    # Discard npm lockfile churn before stash/branch logic: npm rewrites
    # package-lock.json non-deterministically, which is never an intentional
    # edit on a managed install but forces an autostash every update.
    _discard_lockfile_churn(git_cmd, _m().PROJECT_ROOT)
    # Same rationale, different generator: line-ending churn is machine-made
    # dirt on a managed checkout, so clear it (and stop generating it) before
    # the stash/branch logic rather than autostashing the entire tree.
    _normalize_managed_eol(git_cmd, _m().PROJECT_ROOT)

    # Detect if we're updating from a fork (before any branch logic)
    origin_url = _m()._get_origin_url(git_cmd, _m().PROJECT_ROOT)
    is_fork = _is_fork(origin_url)

    if is_fork:
        print("⚠ Updating from fork:")
        print(f"  {origin_url}")
        print()

    if use_zip_update:
        # ZIP-based update for Windows when git is broken
        try:
            desktop_build_ok = _update_via_zip(
                args,
                had_desktop_app_before_update=had_desktop_app_before_update,
            )
        finally:
            _m()._resume_windows_gateways_after_update(_windows_gateway_resume)
        if gateway_mode:
            _write_gateway_update_exit_code(desktop_build_ok)
        return

    # Fetch and pull
    try:

        # Resolve the branch first so the fetch is scoped: a bare `git fetch
        # origin` pulls thousands of auto-generated branches and can stall for minutes.
        branch = _m()._resolve_update_branch(args)

        # Self-heal abandoned git lock files (e.g. .git/shallow.lock left by a
        # crashed fetch) before the fetch — otherwise the update fails with
        # "Unable to create .../shallow.lock: File exists" and never reaches
        # the network.
        from hermes_cli.gitlock import clear_stale_git_locks, clear_stale_tmp_packs

        cleared = clear_stale_git_locks(_m().PROJECT_ROOT)
        if cleared:
            print("  (removed stale git lock(s): %s)" % ", ".join(cleared))
        swept = clear_stale_tmp_packs(_m().PROJECT_ROOT)
        if swept:
            print("  (removed %d aborted-fetch pack temp file(s))" % len(swept))

        # Surface autostash entries left behind by earlier updates (#63717
        # problem 6) — parked --keep-stash runs and failed restores preserve
        # the stash but nothing ever mentioned it again.
        _m()._warn_orphaned_update_autostashes(git_cmd, _m().PROJECT_ROOT)

        print("→ Fetching updates...")
        fetch_result = _git_run(git_cmd, ["fetch", "origin", branch], network=True)
        if fetch_result.returncode != 0:
            _print_fetch_failure(fetch_result.stderr)
            sys.exit(1)

        # Get current branch (returns literal "HEAD" when detached)
        result = _git_run(git_cmd, ["rev-parse", "--abbrev-ref", "HEAD"], check=True)
        current_branch = result.stdout.strip()

        _plan = _prepare_checkout_for_update(
            git_cmd,
            branch,
            current_branch,
            is_fork=is_fork,
            assume_yes=assume_yes,
            gateway_mode=gateway_mode,
            gw_input_fn=gw_input_fn,
            switch_branch=switch_branch,
            _windows_gateway_resume=_windows_gateway_resume,
        )
        auto_stash_ref = _plan.auto_stash_ref
        commit_count = _plan.commit_count
        in_place_update = _plan.in_place_update
        parked_branch_switched = _plan.parked_branch_switched
        prompt_for_restore = _plan.prompt_for_restore
        switch_block_reason = _plan.switch_block_reason
        upstream_checked = _plan.upstream_checked

        if commit_count == 0:
            _invalidate_update_cache()

            # Restore stash and switch back to original branch if we moved.
            # EXCEPTION: a parked feature branch we verified clean + fully
            # merged stays on the target — re-parking the checkout on the
            # stale branch is the 2026-08-17 incident all over again.
            if auto_stash_ref is not None:
                _m()._restore_stashed_changes(
                    git_cmd,
                    _m().PROJECT_ROOT,
                    auto_stash_ref,
                    prompt_user=prompt_for_restore,
                    input_fn=gw_input_fn,
                )
            if parked_branch_switched:
                if switch_block_reason.startswith("unmerged:"):
                    _count = switch_block_reason.split(":", 1)[1]
                    print(
                        f"  ✓ Checkout was parked on '{current_branch}' — "
                        f"switched back to {branch}; {_count} unmerged "
                        f"commit(s) kept on '{current_branch}'."
                    )
                else:
                    print(
                        f"  ✓ Checkout was parked on '{current_branch}' (fully "
                        f"merged) — switched back to {branch}."
                    )
            elif current_branch not in {branch, "HEAD"}:
                _git_run(git_cmd, ["checkout", current_branch])

            current_checkout_complete = _repair_current_checkout(
                assume_yes=assume_yes,
                gateway_mode=gateway_mode,
                pre_update_snapshot_id=pre_update_snapshot_id,
                desktop_dir=desktop_dir,
                had_desktop_app_before_update=had_desktop_app_before_update,
                active_lazy_features=active_lazy_features,
                active_tool_dependencies=active_tool_dependencies,
                upstream_checked=upstream_checked,
                _windows_gateway_resume=_windows_gateway_resume,
            )
            _m()._resume_windows_gateways_after_update(_windows_gateway_resume)
            # A prior pull may still owe the fleet a restart (#95294); catch
            # up even on the "Already up to date" path, and BEFORE the exit
            # gate below so a partial outcome can't strand the fleet on stale
            # code (#91277 fleet contract).
            _apply_pending_fleet_restart_catchup()
            if not current_checkout_complete:
                if gateway_mode:
                    _write_gateway_update_exit_code(False)
                try:
                    from hermes_cli.update_receipt import finalize_update_receipt

                    finalize_update_receipt("partial")
                except Exception as _receipt_exc:
                    logger.debug(
                        "Update receipt finalize (current checkout) failed: %s",
                        _receipt_exc,
                    )
                sys.exit(1)
            return

        if commit_count > 0:
            print(f"→ Found {commit_count} new commit(s)")
        else:
            # Shallow checkout, exact count unrecoverable (offline/rate-limited
            # compare API) — the tips differ, so there IS an update.
            print("→ Updates available (commit count unknown on this shallow checkout)")

        print("→ Pulling updates...")
        pre_pull_sha = _pull_updates(
            git_cmd,
            branch,
            auto_stash_ref,
            prompt_for_restore=prompt_for_restore,
            gw_input_fn=gw_input_fn,
            discard_local_changes=discard_local_changes,
            keep_stash=keep_stash,
        )

        _invalidate_update_cache()

        # Verify HEAD moved (#79678): a detached checkout pinned to a SHA can
        # report "N new commit(s)" and a successful ``merge --ff-only`` yet
        # stay on the old commit, so the old code reinstalled deps and
        # claimed "✓ Code updated!". Surface the no-op instead.
        post_pull_sha = _capture_head_sha(git_cmd, _m().PROJECT_ROOT)
        if pre_pull_sha and post_pull_sha == pre_pull_sha:
            print()
            print("✗ Code did not move — update was a no-op.")
            print(
                f"  HEAD is pinned to {pre_pull_sha[:10]} (detached checkout); "
                f"origin/{branch} advanced but the working tree stayed put."
            )
            print(
                "  Reattach to the branch and retry: "
                f"git -C {_m().PROJECT_ROOT} checkout {branch} && hermes update"
            )
            _m()._resume_windows_gateways_after_update(_windows_gateway_resume)
            sys.exit(1)

        # Verify HEAD is on the target branch; otherwise "✓ Code updated!"
        # would be a lie. An IN-PLACE update is the one legitimate way to end
        # elsewhere: origin/<target> was merged INTO the checked-out branch,
        # so the running code *is* current.
        post_pull_branch = _git_run(git_cmd, ["rev-parse", "--abbrev-ref", "HEAD"]).stdout.strip()
        if (
            not in_place_update
            and post_pull_branch
            and post_pull_branch not in {branch, "HEAD"}
        ):
            print()
            print(
                f"✗ Update pulled origin/{branch}, but the checkout is on "
                f"'{post_pull_branch}' — not claiming success."
            )
            print(
                "  Switch to the target branch and retry: "
                f"git -C {_m().PROJECT_ROOT} checkout {branch} && hermes update"
            )
            _m()._resume_windows_gateways_after_update(_windows_gateway_resume)
            sys.exit(1)

        # #95294: HEAD advanced; running gateways still serve pre-pull
        # modules until the restart phase below. Any interrupt between here
        # and a completed (or no-op) restart leaves this marker so the next
        # ``hermes update`` can catch up even when git is already up to date.
        # Distinct from ``.update-incomplete`` (venv/install repair).
        _write_fleet_restart_pending_marker(expected_sha=post_pull_sha or "")

        # Clear stale .pyc bytecode cache — prevents ImportError on gateway
        # restart when updated source references names that didn't exist in
        # the old bytecode (e.g. get_hermes_home added to hermes_constants).
        _sweep_bytecode_after_update(branch)

        # Fork upstream sync logic (only for main branch on forks)
        if is_fork and branch == "main":
            _m()._sync_with_upstream_if_needed(
                git_cmd,
                _m().PROJECT_ROOT,
                assume_yes=assume_yes,
                input_fn=gw_input_fn,
            )

        # Reinstall deps: .[all], falling back to base + remaining extras
        # individually so a broken extra doesn't strip working capabilities.
        # Ownership preflight (#83529) refuses first if the venv has
        # foreign-owned (sudo-pip) files that would brick the install mid-mutation.
        _sync_python_dependencies_after_pull(
            git_cmd,
            branch,
            pre_pull_sha,
            active_lazy_features=active_lazy_features,
            active_tool_dependencies=active_tool_dependencies,
            _windows_gateway_resume=_windows_gateway_resume,
        )

        node_failures = _update_node_dependencies()
        _m()._build_web_ui(_m().PROJECT_ROOT / "web")

        desktop_build_ok = _rebuild_desktop_after_update(
            desktop_dir,
            had_desktop_app_before_update=had_desktop_app_before_update,
        )

        print()
        print(f"✓ Code updated!{_branch_head_suffix(git_cmd, _m().PROJECT_ROOT)}")

        update_complete = _run_post_update_maintenance(
            assume_yes=assume_yes,
            gateway_mode=gateway_mode,
            pre_update_snapshot_id=pre_update_snapshot_id,
            had_desktop_app_before_update=had_desktop_app_before_update,
            node_failures=node_failures,
            desktop_build_ok=desktop_build_ok,
            pre_update_version=pre_update_version,
        )

        # Write the exit code *before* the restart attempt: under ``update
        # --gateway`` this process lives in the gateway's systemd cgroup, and
        # the ``systemctl restart`` fallback SIGKILLs the cgroup (KillMode=
        # mixed) — us and the wrapping shell included — so the marker would
        # never be written and the new gateway's watcher would poll 30 min
        # and report a spurious timeout. The verified summary already folds
        # in Desktop and SQLite-runtime health (gateway/run.py).
        if gateway_mode:
            _write_gateway_update_exit_code(update_complete)

        _restart = _restart_gateway_fleet_after_update(_pre_update_plan, gateway_mode)
        _resume_windows_gateways_and_merge_outcome(_restart, _windows_gateway_resume, gateway_mode)
        _verify_fleet_after_update(
            _restart,
            _pre_update_plan=_pre_update_plan,
            _windows_gateway_resume=_windows_gateway_resume,
            node_failures=node_failures,
            update_complete=update_complete,
        )

    except _shim_quarantine_error_type() as e:
        # Fail-closed shim contention (#87331): strict quarantine refused
        # BEFORE any installer ran — defer via marker, exit 2, no ZIP.
        _refuse_update_for_contended_shims(e)
    except subprocess.CalledProcessError as e:
        stage = _format_update_failure_stage(e)
        if _should_zip_fallback_on_update_error(e):
            print(f"⚠ {stage}: {e}")
            print("→ Falling back to ZIP download...")
            print()
            desktop_build_ok = _update_via_zip(
                args,
                had_desktop_app_before_update=had_desktop_app_before_update,
            )
            if gateway_mode:
                _write_gateway_update_exit_code(desktop_build_ok)
        else:
            print(f"✗ {stage}: {e}")
            _print_called_process_error_tail(e)
            if _called_process_error_is_python_dep_install(e):
                print(
                    "  The git update already finished. Re-downloading the source "
                    "ZIP cannot fix a dependency install error and would overwrite "
                    "local files."
                )
                if _m()._is_windows():
                    print("  Retry through the venv interpreter:")
                    print(
                        '    venv\\Scripts\\python.exe -c '
                        '"from hermes_cli.main import main; main()" update --yes'
                    )
            try:
                from hermes_cli.update_receipt import finalize_update_receipt

                finalize_update_receipt("failed")
            except Exception:
                pass
            sys.exit(1)

# --- Hoisted from the body of _cmd_update_impl (self-contained, no closure state) ---


