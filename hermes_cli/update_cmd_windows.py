"""Windows gateway lifecycle for ``hermes update``: pause/resume/cold-start the service, sweep venv holders, reap orphaned backends.

Split out of ``update_cmd.py``; names are re-imported there so ``hermes_cli.update_cmd.<name>`` still resolves/monkeypatches.
Origin helpers are imported lazily per function (no cycle; test patches on the origin stay effective).
"""

import logging
from contextlib import suppress
import os
import subprocess
import sys
import time as _time
from datetime import datetime
from pathlib import Path

from hermes_cli.update_cmd_common import _best_effort

# Log-record parity with the origin module.
logger = logging.getLogger("hermes_cli.update_cmd")


def _write_update_planned_stop_marker(profile_path: Path, pid: int) -> bool:
    """Write a planned-stop marker into a specific profile home."""
    try:
        from datetime import timezone

        from gateway.status import _get_process_start_time
        from utils import atomic_json_write

        record = {
            "target_pid": pid,
            "target_start_time": _get_process_start_time(pid),
            "stopper_pid": os.getpid(),
            "written_at": datetime.now(timezone.utc).isoformat(),
        }
        atomic_json_write(
            Path(profile_path) / ".gateway-planned-stop.json",
            record,
            indent=None,
            separators=(",", ":"),
        )
        return True
    except (OSError, PermissionError):
        return False


def _wait_for_windows_update_gateway_exit(
    pids: list[int], *, timeout: float
) -> set[int]:
    """Wait for the given gateway PIDs to exit, returning survivors."""
    if not pids:
        return set()

    from gateway.status import _pid_exists

    remaining = set(pids)
    deadline = _time.monotonic() + max(timeout, 0.0)
    while remaining and _time.monotonic() < deadline:
        for pid in list(remaining):
            try:
                if not _pid_exists(pid):
                    remaining.discard(pid)
            except Exception:
                remaining.discard(pid)
        if remaining:
            _time.sleep(0.25)

    survivors: set[int] = set()
    for pid in remaining:
        with suppress(Exception):
            if _pid_exists(pid):
                survivors.add(pid)
    return survivors


def _self_and_non_gateway_ancestor_pids(psutil) -> set[int]:
    """PIDs a venv-holder scan must never nominate: this process and its non-gateway ancestry.

    Do NOT blanket-exclude ancestors: under ``/update`` the updater is a CHILD of the gateway, and hiding it
    dead-ends the update on ``venv-blocked``. Keep GATEWAY ancestors visible (the pause path stops them
    gracefully; a detached child survives on Windows); never nominate interactive ancestry as a blocker.
    """
    try:
        from gateway.status import looks_like_gateway_command_line as _is_gw
    except Exception:
        _is_gw = None
    skip: set[int] = {os.getpid()}
    with suppress(Exception):
        for anc in psutil.Process().parents():
            try:
                anc_cmdline = " ".join(anc.cmdline() or [])
            except Exception:
                anc_cmdline = ""
            if _is_gw is not None and anc_cmdline and _is_gw(anc_cmdline):
                continue
            skip.add(int(anc.pid))
    return skip


def _lower_dir_prefix(path: Path) -> str:
    """``str(path)`` lower-cased with one trailing separator, resolved when possible (prefix matching)."""
    try:
        raw = str(path.resolve())
    except OSError:
        raw = str(path)
    return raw.lower().rstrip(os.sep) + os.sep


def _detect_venv_python_processes(
    *, exclude_pids: set[int] | None = None
) -> list[tuple[int, str, str]]:
    """Live processes running from the project venv's interpreter as ``(pid, name, cmdline)``; never raises.

    The hermes.exe shim guard misses the Desktop backend and anything off ``venv\\Scripts\\python(w).exe``;
    they keep ``.pyd`` files mapped so a mid-update dependency sync dies half-way. Killing is pointless (Desktop
    respawns its backend) so callers should refuse. Empty off-Windows / without psutil; self+ancestors excluded.
    """
    from hermes_cli.update_cmd import _m
    if not _m()._is_windows():
        return []
    try:
        import psutil
    except Exception:
        return []

    venv_prefix = _lower_dir_prefix(_m().PROJECT_ROOT / "venv")
    root_prefix = _lower_dir_prefix(_m().PROJECT_ROOT)

    skip: set[int] = set(exclude_pids or set())
    skip |= _self_and_non_gateway_ancestor_pids(psutil)

    matches: list[tuple[int, str, str]] = []
    try:
        # cmdline/cwd are expensive per-process on Windows (500+ procs can blow the
        # Desktop preflight watchdog): fetch them lazily for plausible candidates only.
        proc_iter = psutil.process_iter(["pid", "exe", "name"])
    except Exception:
        return []
    for proc in proc_iter:
        try:
            info = proc.info
        except Exception:
            continue
        pid = info.get("pid")
        exe = info.get("exe")
        if not exe or pid is None or int(pid) in skip:
            continue
        try:
            exe_norm = str(Path(exe).resolve()).lower()
        except (OSError, ValueError):
            exe_norm = str(exe).lower()
        # Primary match: exe lives under this venv (desktop backend / gateway case).
        is_holder = exe_norm.startswith(venv_prefix)
        name = str(info.get("name") or Path(exe).name)
        name_low = name.lower()

        if not is_holder and not (
            name_low.startswith(("python", "pypy"))
            or name_low in {"uv.exe", "uvx.exe", "hermes.exe"}
        ):
            continue

        try:
            cmdline_raw = " ".join(proc.cmdline() or [])
        except Exception:
            cmdline_raw = ""
        cmdline_low = cmdline_raw.lower()
        # Fallback: uv/base-interpreter trampolines have an exe OUTSIDE the venv yet hold
        # its .pyd files — match cmdline (venv path, or `-m hermes_cli.main` + root/cwd).
        if not is_holder and venv_prefix in cmdline_low:
            is_holder = True
        if not is_holder and "hermes_cli.main" in cmdline_low:
            try:
                cwd_low = str(proc.cwd() or "").lower().rstrip(os.sep) + os.sep
            except Exception:
                cwd_low = os.sep
            if root_prefix in cmdline_low or cwd_low.startswith(root_prefix):
                is_holder = True
        if not is_holder:
            continue
        name = info.get("name") or Path(exe).name
        # FULL cmdline: callers parse it (pausable-gateway exemption looks for `gateway run`);
        # truncating here misreported autostarted gateways as blockers. Truncate at display time.
        matches.append((int(pid), str(name), cmdline_raw))
    return matches


_HOLDER_VALUE_FLAGS_FALLBACK = frozenset(
    {
        "--profile", "-p", "--config",
        "--model", "-m", "--provider", "--reasoning",
        "--toolsets", "-t", "--skills", "-s",
        "--continue", "-c", "--resume", "-r",
        "--oneshot", "-z", "--in", "--usage-file",
    }
)


_holder_value_flags_cache: frozenset | None = None


def _holder_value_flags() -> frozenset:
    """Top-level CLI flags that consume a value, introspected from the REAL parser (nargs != 0); cached per process.

    Derived so the holder classifier can't drift from argparse (a handwritten subset misparsed ``--reasoning high
    serve``). Pre-argparse profile selectors are added explicitly (stripped before argparse sees argv). Falls back
    to a static snapshot when the parser can't import — the updater must classify holders even on a broken tree.
    """
    global _holder_value_flags_cache
    if _holder_value_flags_cache is not None:
        return _holder_value_flags_cache
    flags: set[str] = {"--profile", "-p", "--config"}
    try:
        from hermes_cli._parser import build_top_level_parser

        parser = build_top_level_parser()[0]
        for action in parser._actions:
            if action.option_strings and action.nargs != 0:
                flags.update(action.option_strings)
        _holder_value_flags_cache = frozenset(flags)
    except Exception:
        _holder_value_flags_cache = _HOLDER_VALUE_FLAGS_FALLBACK
    return _holder_value_flags_cache


def _hermes_holder_subcommand(cmdline: str) -> str | None:
    """The actual Hermes SUBCOMMAND a venv-holder argv runs, or None (callers must NOT guess a label).

    Token-based, never substring (``kanban --preserve-cache`` contains "serve"): find the ``hermes_cli.main`` /
    ``hermes(.exe)`` entry token, return the first following token that isn't a flag or a flag's value.
    """
    try:
        import shlex

        tokens = shlex.split(cmdline, posix=False)
    except Exception:
        tokens = cmdline.split()

    entry_idx: int | None = None
    for i, token in enumerate(tokens):
        low = token.lower().strip('"')
        if low.endswith("hermes_cli.main") and i > 0 and tokens[i - 1] == "-m":
            entry_idx = i
            break
        base = low.rsplit("\\", 1)[-1].rsplit("/", 1)[-1]
        if base in ("hermes", "hermes.exe"):
            entry_idx = i
            break
    if entry_idx is None:
        return None

    value_flags = _holder_value_flags()
    i = entry_idx + 1
    while i < len(tokens):
        token = tokens[i]
        if token in value_flags or token.split("=", 1)[0] in value_flags:
            # --flag value consumes two tokens; --flag=value consumes one.
            i += 1 if "=" in token else 2
            continue
        if token.startswith("-"):
            i += 1
            continue
        return token.lower()
    return None


def _format_venv_python_holders_message(matches: list[tuple[int, str, str]]) -> str:
    """Explain which venv processes block the update and how to clear them.

    Labels come from the parsed SUBCOMMAND, never substring: a standalone ``hermes dashboard`` must not be
    called the Desktop backend, ``--preserve-cache`` must not match "serve". Unknown argv gets no hint.
    """
    lines = [
        "✗ Other Hermes processes are running from this install's venv:",
    ]
    hint_by_subcommand = {
        "serve": "  ← Hermes backend (if the Desktop app is open, close it)",
        "dashboard": "  ← hermes dashboard (stop it: hermes dashboard stop, or close that terminal)",
        "gateway": "  ← gateway",
    }
    for pid, name, cmdline in matches[:6]:
        sub = _hermes_holder_subcommand(cmdline)
        hint = hint_by_subcommand.get(sub or "", "")
        lines.append(f"  PID {pid}  {name}  {cmdline[:120]}{hint}")
    if len(matches) > 6:
        lines.append(f"  ... and {len(matches) - 6} more")
    lines.append("")
    lines.append(
        "  On Windows these keep native extension files (.pyd) locked, so the"
    )
    lines.append(
        "  dependency update would fail partway and leave a broken install."
    )
    lines.append(
        "  Close the Hermes desktop app / other Hermes terminals, then re-run:"
    )
    lines.append("    hermes update")
    lines.append("  (or use `hermes update --force-venv` to proceed anyway at your own risk)")
    return "\n".join(lines)


def _venv_launcher_ancestors(pids: list[int]) -> list[int]:
    """Venv-interpreter parents of *pids* that hold the install open; never raises.

    A shim-started gateway is a chain: ``venv\\Scripts\\python.exe`` launcher (keeps ``.pyd`` mapped) -> uv
    CPython worker (writes the PID file). The pause set sees the worker, the venv scan sees the launcher, so a
    paused gateway still tripped the guard. One hop up only, venv-prefixed only (bounds blast radius).
    """
    from hermes_cli.update_cmd import _m
    if not _m()._is_windows() or not pids:
        return []
    try:
        import psutil
    except Exception:
        return []

    venv_prefix = _lower_dir_prefix(_m().PROJECT_ROOT / "venv")

    skip = _self_and_non_gateway_ancestor_pids(psutil)

    found: list[int] = []
    for pid in pids:
        try:
            parent = psutil.Process(int(pid)).parent()
        except Exception:
            continue
        if parent is None:
            continue
        ppid = int(parent.pid)
        if ppid in skip or ppid in found or ppid in set(pids):
            continue
        try:
            exe = (parent.exe() or "").lower()
        except Exception:
            continue
        if exe.startswith(venv_prefix):
            found.append(ppid)
    return found


def _leftover_pausable_gateway_pids(
    matches: list[tuple[int, str, str]],
) -> list[int] | None:
    """PIDs from *matches* when EVERY remaining venv holder is a pausable gateway, else ``None`` (keep refusing).

    A gateway respawned inside the pause->guard window (or via an unmapped spawn path) still holds ``.pyd`` files.
    Uses the Desktop preflight's ``_is_pausable_gateway`` so exemption and tolerance cannot drift; live argv is
    re-read via psutil when possible since the scan may hold only a cmdline prefix.
    """
    from hermes_cli._scan_venv_blockers import _is_pausable_gateway

    try:
        import psutil  # type: ignore
    except Exception:
        psutil = None

    pids: list[int] = []
    for pid, _name, cmdline in matches:
        argv = cmdline
        if psutil is not None:
            with suppress(Exception):
                argv = " ".join(psutil.Process(int(pid)).cmdline()) or cmdline
        if not _is_pausable_gateway(argv):
            return None
        pids.append(int(pid))
    return pids


def _refuse_gateway_ancestor_tree_kill(
    pids: list[int], *, gateway_mode: bool
) -> bool:
    """Refuse a plain Windows update that would tree-kill its own ancestry.

    A chat agent running plain ``hermes update`` is a child of the gateway; ``taskkill /T /F`` on it kills the
    updater first. ``/update`` (``--gateway``) is exempt (detached, file-based delivery). Refuse only when a
    nominated gateway is positively an ancestor; unknown ancestry keeps existing recovery.
    """
    if gateway_mode or not pids:
        return False

    try:
        from hermes_cli.gateway import _is_pid_ancestor_of_current_process

        ancestors = [
            int(pid)
            for pid in pids
            if _is_pid_ancestor_of_current_process(int(pid))
        ]
    except Exception as exc:
        logger.debug("Could not inspect gateway ancestry before tree-kill: %s", exc)
        return False

    if not ancestors:
        return False

    rendered = ", ".join(str(pid) for pid in ancestors)
    print(
        "✗ Refusing to stop the gateway process tree because this updater "
        f"is running inside it (gateway PID(s): {rendered})."
    )
    print(
        "  On Windows, taskkill /T would terminate the updater before the "
        "update can run."
    )
    print("  From a chat platform, use `/update` instead.")
    print("  Otherwise, run `hermes update` from a separate terminal.")
    return True


def _ledger_manual_serve_holders(
    matches: list[tuple[int, str, str]],
) -> list[dict]:
    """Full ledger entries for venv holders that are MANUAL serve/dashboard backends.

    Positive identity only: self-registered purpose serve/dashboard, live (pid, create_time), recorded spawner
    NOT alive (a Desktop-owned backend keeps its live Electron spawner and must keep the refusal — the app would
    respawn what we kill). Full entries let the relauncher rebuild from host/port/profile, not argv.
    """
    try:
        from hermes_cli.process_identity import ledger_entries, spawner_is_dead
    except Exception:
        return []
    holder_pids = {int(pid) for pid, _name, _cmd in matches}
    out: list[dict] = []
    for entry in ledger_entries():
        if entry.get("purpose") not in ("serve", "dashboard"):
            continue
        pid = entry.get("pid")
        if not isinstance(pid, int) or pid not in holder_pids:
            continue
        if spawner_is_dead(entry) is False:
            continue  # live Desktop supervisor owns it — keep refusing
        out.append(entry)
    return out


def _serve_relaunch_commands(entries: list[dict]) -> list[list[str]]:
    """Rebuild launch commands for stopped serves from ledger host/port/profile — never argv parsing
    (joined argv cannot round-trip Windows paths with spaces). Entries without a port are skipped.
    """
    from hermes_cli.update_cmd import _m
    commands: list[list[str]] = []
    hermes = None
    try:
        scripts_dir = _m()._venv_scripts_dir()
        if scripts_dir is not None:
            for name in ("hermes.exe", "hermes"):
                candidate = scripts_dir / name
                if candidate.is_file():
                    hermes = str(candidate)
                    break
    except Exception:
        hermes = None
    if hermes is None:
        hermes = "hermes"
    for entry in entries:
        port = entry.get("port")
        if not isinstance(port, int) or port <= 0:
            continue
        cmd = [hermes]
        profile = str(entry.get("profile") or "")
        if profile and profile != "default":
            cmd += ["--profile", profile]
        cmd.append(str(entry.get("purpose")))
        host = str(entry.get("host") or "")
        if host:
            cmd += ["--host", host]
        cmd += ["--port", str(port)]
        commands.append(cmd)
    return commands


def _relaunch_stopped_serves(token: dict) -> None:
    """Idempotent atexit relaunch of manual serves stopped by the venv guard.

    `pending` flips False on first invocation so explicit call + atexit registration cannot double-spawn.
    """
    from hermes_cli.update_cmd import _m, _record_update_step
    if not token.get("pending"):
        return
    token["pending"] = False
    entries = token.get("entries") or []
    if not entries:
        return
    commands = _serve_relaunch_commands(entries)
    skipped = len(entries) - len(commands)
    failed: list = []
    if commands:
        print("  ⟲ Relaunching stopped serve/dashboard backend(s)")
        failed = _m()._respawn_dashboard_processes(commands)
    if skipped or failed:
        print(
            "  ⚠ Some stopped backends could not be relaunched automatically; "
            "restart them manually (hermes serve --host <ip> --port <port>)."
        )
    _record_update_step(
        "serve_relaunch",
        not failed and not skipped,
        f"relaunched={len(commands) - len(failed)} failed={len(failed)} skipped={skipped}",
    )


def _is_backend_argv(argv_low: str) -> bool:
    """Whether a lower-cased argv is a Desktop backend (``hermes_cli.main`` running ``serve``/``dashboard``)."""
    return "hermes_cli.main" in argv_low and (
        " serve" in argv_low or " dashboard" in argv_low
    )


def _live_argv_low(psutil, pid, cmdline: str) -> str | None:
    """Current lower-cased argv of *pid* (falls back to the scanned *cmdline*); ``None`` if it exited."""
    argv = cmdline
    try:
        argv = " ".join(psutil.Process(int(pid)).cmdline()) or cmdline
    except psutil.NoSuchProcess:
        return None
    except Exception:
        pass
    return argv.lower()


def _orphaned_desktop_backend_pids(
    matches: list[tuple[int, str, str]],
) -> list[tuple[int, int]] | None:
    """``(pid, start_time)`` roots from *matches* when every remaining holder is an ORPHANED backend, else ``None``.

    Killing a Desktop-owned ``serve`` is futile (the app respawns it), but after the Desktop exited (GUI hand-off
    contract: it tree-kills backends, the marker parks relaunch) a straggler whose supervisor is gone would
    dead-end the update with "Hermes is still running" and zero open windows.
    Qualifies only if cmdline is a Hermes backend (``hermes_cli.main`` + serve/dashboard) AND the parent is
    demonstrably gone (PID missing or reused: parent created *after* child). Tree-aware: holders inside an
    accepted root's tree fold into it; only roots are returned (``taskkill /T`` reaps descendants). Any other
    live-parent backend, unjustified non-backend, unprovable case, or no psutil -> ``None``. Never raises.
    """
    try:
        import psutil  # type: ignore
    except Exception:
        return None

    # Pass 1: find orphaned backend ROOTS among the holders.
    roots: list[tuple[int, int]] = []
    remaining: list[tuple[int, str]] = []  # (pid, argv_low) still to justify
    for pid, _name, cmdline in matches:
        low = _live_argv_low(psutil, pid, cmdline)
        if low is None:
            continue  # exited between scan and classification — nothing to reap
        if not _is_backend_argv(low):
            remaining.append((int(pid), low))
            continue
        try:
            proc = psutil.Process(int(pid))
            # Fingerprint from the SAME psutil handle, centisecond-quantized like
            # gateway.status.get_process_start_time so pid_is_hermes round-trips at kill time.
            process_start_time = int(round(proc.create_time() * 100))
        except psutil.NoSuchProcess:
            continue  # exited during classification — nothing to reap
        except Exception:
            return None

        try:
            ppid = proc.ppid()
            parent = psutil.Process(ppid) if ppid else None
            if parent is not None and parent.is_running():
                # PID-reuse check: a "parent" created after its child is a recycled PID.
                if parent.create_time() <= proc.create_time():
                    # Live parent: not a root, maybe an orphan root's descendant (the venv
                    # trampoline re-execs uv python with the SAME argv). Defer to pass 2.
                    remaining.append((int(pid), low))
                    continue
        except psutil.NoSuchProcess:
            pass  # parent gone → orphan
        except Exception:
            return None
        roots.append((int(pid), process_start_time))

    # Pass 2: every non-backend holder must descend from an accepted orphan root
    # (dies with the tree reap); anything else keeps the refusal.
    root_set = {pid for pid, _start_time in roots}
    for pid, _low in remaining:
        if not root_set:
            return None
        try:
            ancestors = {int(a.pid) for a in psutil.Process(pid).parents()}
        except psutil.NoSuchProcess:
            continue  # exited already
        except Exception:
            return None
        if not (ancestors & root_set):
            return None
    return roots


def _ledger_reapable_backend_pids(
    matches: list[tuple[int, str, str]],
) -> list[int]:
    """PIDs the spawn ledger positively identifies as orphaned backends; never raises.

    Strongest rung (no PPID/cmdline inference): qualifies when ``(pid, create_time)`` matches a live ledger entry
    (PID reuse can't forge it), purpose is a REAPABLE kind (never interactive), and the recorded SPAWNER is
    provably dead. Safe in ANY context. Unlisted holders fall to later rungs and never disqualify identified ones.
    """
    try:
        from hermes_cli.process_identity import (
            REAPABLE_PURPOSES,
            ledger_entries,
            spawner_is_dead,
        )

        entries = ledger_entries()
    except Exception:
        return []
    by_pid = {e.get("pid"): e for e in entries if isinstance(e.get("pid"), int)}
    roots: list[int] = []
    for pid, _name, _cmdline in matches:
        entry = by_pid.get(int(pid))
        if not entry:
            continue
        if entry.get("purpose") not in REAPABLE_PURPOSES:
            continue
        if spawner_is_dead(entry) is True:
            roots.append(int(pid))
    return roots


def _handoff_reapable_backend_pids(
    matches: list[tuple[int, str, str]],
) -> list[int] | None:
    """Backend PIDs safe to tree-reap during a GUI-updater hand-off, INCLUDING ones with a live parent; never raises.

    ``_orphaned_desktop_backend_pids`` bails on ANY live parent (mid-teardown Electron, launcher->worker chain),
    which hung a hand-off for 12 minutes. With the update-incomplete marker + ``--gateway`` + no live
    ``hermes.exe`` shim, nothing legitimate supervises or respawns a ``serve`` from this venv, so survivors are
    leaks. Only Hermes backends qualify — any non-backend holder, or no psutil -> ``None``. The CALLER must
    have confirmed the hand-off gate; outside it the stricter orphan-only path stands.
    """
    try:
        import psutil  # type: ignore
    except Exception:
        return None

    roots: list[int] = []
    for pid, _name, cmdline in matches:
        low = _live_argv_low(psutil, pid, cmdline)
        if low is None:
            continue  # exited — nothing to reap
        if not _is_backend_argv(low):
            return None  # unexpected non-backend holder: refuse the whole set
        roots.append(int(pid))

    return roots or None


def _stop_process_trees(
    pids: list[int] | list[tuple[int, int]],
) -> None:
    """Force-stop each PID with its full child tree (Windows); best effort, never raises.

    ``taskkill /T /F``: stopping only the parent can leave a ``.hermes-runtime`` child holding the install open.
    """
    from gateway.status import get_process_start_time
    from hermes_cli._subprocess_compat import pid_is_hermes, windows_hide_flags

    for entry in pids:
        if isinstance(entry, tuple):
            pid, expected_start_time = entry
        else:
            pid = int(entry)
            expected_start_time = get_process_start_time(pid)
        try:
            if expected_start_time is None:
                logger.debug(
                    "Skipping taskkill of PID %s: process identity unavailable",
                    pid,
                )
                continue
            if not pid_is_hermes(
                pid,
                expected_start_time=expected_start_time,
            ):
                logger.debug(
                    "Skipping taskkill of non-Hermes or changed PID %s",
                    pid,
                )
                continue
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/T", "/F"],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                stdin=subprocess.DEVNULL,
                creationflags=windows_hide_flags(),
            )
        except Exception as exc:
            logger.debug("Could not stop process tree %s: %s", pid, exc)


def _looks_like_desktop_control_plane(cmdline: str) -> bool:
    """True for this-install ``hermes serve`` / ``hermes dashboard`` argv (Desktop control plane).

    Not the messaging gateway — don't feed into ``looks_like_gateway_command_line``. Token-based via the
    parser-derived classifier, never substring (``kanban --preserve-cache``, ``-m dashboard chat``).
    Undeterminable subcommand is NOT a control plane.
    """
    if "hermes_cli.main" not in (cmdline or "").lower():
        return False
    return _hermes_holder_subcommand(cmdline) in ("serve", "dashboard")


def _desktop_owns_gateway_lifecycle() -> bool:
    """True when Desktop currently supervises this install's control plane (updater must not steal gateway start).

    Not proof messaging is served: serve is the control plane, the gateway a detached sibling. Prefer the spawn
    ledger; fall back to the venv-holder scan. An orphaned control plane (supervisor gone) does not count.
    """
    from hermes_cli.update_cmd import _m
    with _best_effort('Desktop-lifecycle ledger probe failed: %s'):
        from hermes_cli.process_identity import ledger_entries, spawner_is_dead

        for entry in ledger_entries():
            if entry.get("purpose") not in ("serve", "dashboard"):
                continue
            if spawner_is_dead(entry) is False:
                return True

    try:
        import psutil
    except Exception:
        psutil = None

    try:
        holders = _m()._detect_venv_python_processes()
    except Exception as exc:
        logger.debug("Desktop-lifecycle holder scan failed: %s", exc)
        return False

    for pid, _name, cmdline in holders:
        if not _looks_like_desktop_control_plane(cmdline):
            continue
        if psutil is None:
            return True  # cannot prove orphanhood; a live control plane suffices
        try:
            proc = psutil.Process(int(pid))
            parent = proc.parent()
            if parent is None or not parent.is_running():
                continue
            if parent.create_time() > proc.create_time():
                continue
            return True
        except Exception:
            continue
    return False


def _stop_windows_gateway_service(
    name: str,
    *,
    expected_processes: tuple[tuple[int, float], ...] = (),
    expected_service_identity: tuple[int, float] | None = None,
    expected_gateway_identity: tuple[int, float] | None = None,
    timeout: float = 30.0,
) -> None:
    """Stop one verified Windows service and wait until SCM reports it down."""
    import psutil  # noqa: PLC0415

    service = psutil.win_service_get(name)
    if expected_service_identity is not None:
        try:
            current_status = str(service.status())
            current_service_pid = int(service.pid() or 0)
        except Exception as exc:
            raise RuntimeError(
                f"Windows service {name} SCM identity is unavailable before stop"
            ) from exc
        if current_status != "running":
            raise RuntimeError(
                f"Windows service {name} is not stably running before stop: {current_status}"
            )
        if current_service_pid != int(expected_service_identity[0]):
            raise RuntimeError(
                f"Windows service {name} SCM process identity changed before stop"
            )
    for label, identity in (
        ("service", expected_service_identity),
        ("gateway", expected_gateway_identity),
    ):
        if identity is None:
            continue
        pid, create_time = identity
        try:
            current = float(psutil.Process(int(pid)).create_time())
        except Exception as exc:
            raise RuntimeError(
                f"Windows {label} process identity is unavailable before stop"
            ) from exc
        if abs(current - float(create_time)) > 0.001:
            raise RuntimeError(
                f"Windows {label} process identity changed before stop"
            )
    if expected_service_identity is not None and expected_gateway_identity is not None:
        service_pid = int(expected_service_identity[0])
        gateway_pid = int(expected_gateway_identity[0])
        try:
            ancestor_pids = {
                int(parent.pid) for parent in psutil.Process(gateway_pid).parents()
            }
        except Exception as exc:
            raise RuntimeError(
                "Windows gateway ancestry is unavailable before service stop"
            ) from exc
        if service_pid not in ancestor_pids:
            raise RuntimeError(
                f"Windows gateway is no longer owned by service {name}"
            )
    result = subprocess.run(
        ["sc.exe", "stop", name],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
        check=False,
    )
    if result.returncode != 0 and service.status() != "stopped":
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(detail or f"sc.exe stop failed with {result.returncode}")

    def _original_process_is_alive(pid: int, create_time: float) -> bool:
        try:
            current = float(psutil.Process(pid).create_time())
        except (psutil.NoSuchProcess, psutil.ZombieProcess):
            return False
        except Exception:
            return True  # AccessDenied/unknown: fail closed, venv may still be locked
        return abs(current - create_time) <= 0.001

    alive = [
        pid
        for pid, create_time in expected_processes
        if _original_process_is_alive(pid, create_time)
    ]
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        service_stopped = service.status() == "stopped"
        alive = [
            pid
            for pid, create_time in expected_processes
            if _original_process_is_alive(pid, create_time)
        ]
        if service_stopped and not alive:
            return
        _time.sleep(0.2)
    if service.status() == "stopped":
        # Lingering matching-identity processes make venv mutation unsafe — fail closed.
        alive_after_stop = [
            pid
            for pid, create_time in expected_processes
            if _original_process_is_alive(pid, create_time)
        ]
        if alive_after_stop:
            raise RuntimeError(
                f"Windows service {name} stopped but its process tree is still alive: "
                f"{alive_after_stop}"
            )
        return
    raise RuntimeError(
        f"Windows service {name} did not stop within {timeout:.0f}s; venv mutation unsafe."
    )


def _start_windows_gateway_service(name: str, *, timeout: float = 30.0) -> None:
    """Start one previously paused Windows service and verify it is running."""
    import psutil  # noqa: PLC0415

    service = psutil.win_service_get(name)
    result = subprocess.run(
        ["sc.exe", "start", name],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=10,
        check=False,
    )
    if result.returncode != 0 and service.status() != "running":
        detail = (result.stderr or result.stdout).strip()
        raise RuntimeError(detail or f"sc.exe start failed with {result.returncode}")
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        if service.status() == "running":
            return
        _time.sleep(0.2)
    raise RuntimeError(f"Windows service {name} did not start within {timeout:.0f}s")


def _restore_windows_gateway_service(name: str, *, timeout: float = 60.0) -> None:
    """Restore a service after an uncertain stop, including STOP_PENDING."""
    from hermes_cli.update_cmd import _start_windows_gateway_service
    import psutil  # noqa: PLC0415

    service = psutil.win_service_get(name)
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        status = service.status()
        if status == "running":
            return
        if status == "stopped":
            _start_windows_gateway_service(name)
            return
        _time.sleep(0.2)
    raise RuntimeError(
        f"Windows service {name} did not reach a restorable state within {timeout:.0f}s"
    )


def _pause_windows_gateways_for_update() -> dict | None:
    """Stop running Windows gateways before mutating the checkout or venv.

    Scheduled/startup gateways run via pythonw.exe, invisible to the hermes.exe instance guard, yet keep files
    locked during ``git``/``uv``. Stop only PIDs the gateway discovery code identifies.
    """
    from hermes_cli.update_cmd import (
        _desktop_owns_gateway_lifecycle,
        _m,
        _restore_windows_gateway_service,
        _stop_windows_gateway_service,
    )
    if not _m()._is_windows():
        return None

    try:
        from gateway.status import get_process_start_time, terminate_pid
        from hermes_cli.gateway import (
            _capture_gateway_argv,
            _get_restart_drain_timeout,
            find_gateway_pids,
            find_profile_gateway_processes,
            find_windows_gateway_services,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not prepare Windows gateway pause for update: {exc}"
        ) from exc

    try:
        profile_process_list = find_profile_gateway_processes(strict=True)
        profile_processes = {proc.pid: proc for proc in profile_process_list}
    except Exception as exc:
        raise RuntimeError(
            f"Could not map Windows gateway PIDs to profiles: {exc}"
        ) from exc

    try:
        service_gateways = find_windows_gateway_services(
            profile_processes=profile_process_list
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not determine Windows gateway service ownership: {exc}"
        ) from exc

    service_gateway_pids = {int(service.gateway_pid) for service in service_gateways}
    try:
        running_pids = list(
            dict.fromkeys(
                [
                    *find_gateway_pids(all_profiles=True),
                    *sorted(profile_processes),
                    *sorted(service_gateway_pids),
                ]
            )
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not discover Windows gateway PIDs before update: {exc}"
        ) from exc
    if not running_pids:
        # No gateway running, but an installed autostart entry is an explicit "I want a
        # gateway" signal; a gateway that died between updates would otherwise stay down
        # until next login (resume only relaunches what was running). Cold-start after update.
        # Exception: Desktop owns the lifecycle — spawning ``gateway run`` beside it races
        # ports/state. The skip is ownership, not liveness.
        with _best_effort('Could not check Desktop gateway-lifecycle ownership before update: %s'):
            if _desktop_owns_gateway_lifecycle():
                logger.debug(
                    "Skipping Windows gateway cold-start plan: "
                    "Desktop owns gateway lifecycle"
                )
                return None
        with _best_effort('Could not check Windows gateway autostart state before update: %s'):
            from hermes_cli import gateway_windows

            if gateway_windows.is_installed():
                return {
                    "resume_needed": True,
                    "profiles": {},
                    "unmapped_pids": [],
                    "unmapped": [],
                    "cold_start_if_installed": True,
                }
        return None

    profiles: dict[str, int] = {}
    mapped_pids = []
    socket_acks: list[dict] = []
    for pid in running_pids:
        if pid in service_gateway_pids:
            continue
        proc = profile_processes.get(pid)
        if proc is None:
            continue
        profiles[str(proc.profile)] = int(pid)
        mapped_pids.append(int(pid))
        _write_update_planned_stop_marker(Path(proc.path), int(pid))
        # Socket-first pause: ask the gateway to drain and exit itself (ACK = its own
        # graceful path). No answer (older gateway) -> marker poll / force-kill ladder below.
        try:
            from gateway.control_socket import pause_gateway_for_update

            ack = pause_gateway_for_update(Path(proc.path))
            if ack and (ack.get("pausing") or ack.get("already_stopping")):
                socket_acks.append(ack)
        except Exception as exc:
            logger.debug(
                "Socket pause unavailable for gateway %s: %s", pid, exc
            )

    # Resolve venv-side launchers BEFORE draining: a dead worker's parent cannot be
    # recovered (NoSuchProcess). The launcher keeps ``.pyd`` mapped and would trip the
    # venv-holder guard after the gateway stopped; it is killed with the survivors.
    launcher_pids = _m()._venv_launcher_ancestors(mapped_pids)

    print("→ Stopping Windows gateway process(es) before updating Hermes...")
    try:
        drain_timeout = max(float(_get_restart_drain_timeout()), 1.0)
    except Exception:
        drain_timeout = 10.0
    if socket_acks:
        # A socket-paused gateway drains its ACTIVE TURN first; honor its declared
        # budget (+ teardown grace) so it isn't force-killed mid-turn.
        with suppress(Exception):
            declared = max(
                float(a.get("drain_timeout") or 0.0) for a in socket_acks
            )
            drain_timeout = max(drain_timeout, declared + 10.0)
        print(
            f"  → {len(socket_acks)} gateway(s) ACKed socket pause; "
            f"waiting up to {int(drain_timeout)}s for graceful exit"
        )
    survivors = _m()._wait_for_windows_update_gateway_exit(
        mapped_pids,
        timeout=drain_timeout,
    )
    unmapped_pids = [
        pid
        for pid in running_pids
        if pid not in profile_processes and pid not in service_gateway_pids
    ]

    # Snapshot unmapped gateways' argv *before* force-killing so resume can replay it.
    # Unmapped = no profile->PID-file mapping (e.g. Scheduled Task ``pythonw.exe -m ...``).
    unmapped: list[dict] = []
    for pid in unmapped_pids:
        argv = None
        try:
            argv = _capture_gateway_argv(int(pid))
        except Exception as exc:
            logger.debug("Could not capture argv for unmapped gateway %s: %s", pid, exc)
        unmapped.append({"pid": int(pid), "argv": argv})

    # Tree-kill survivors, unmapped gateways, and pre-drain launchers; a launcher
    # already gone with its worker raises ProcessLookupError and is skipped.
    force_killed = []
    for pid in sorted(set(survivors).union(unmapped_pids).union(launcher_pids)):
        with suppress(ProcessLookupError, PermissionError, OSError):
            pid_int = int(pid)
            terminate_pid(
                pid_int,
                force=True,
                expected_start_time=get_process_start_time(pid_int),
            )
            force_killed.append(pid_int)

    if profiles:
        print(f"  ✓ Paused gateway profile(s): {', '.join(sorted(profiles))}")
    if force_killed:
        print(f"  → Force-stopped {len(force_killed)} gateway process(es)")

    if unmapped_pids:
        respawnable = sum(1 for u in unmapped if u.get("argv"))
        print(
            f"  → Stopped {len(unmapped_pids)} gateway process(es) without profile mapping"
        )
        if respawnable < len(unmapped_pids):
            # No recoverable cmdline (psutil missing, access denied, gone): manual restart.
            print("    Restart manually after update: hermes gateway run")

    token = {
        "resume_needed": True,
        "profiles": profiles,
        "unmapped_pids": unmapped_pids,
        "unmapped": unmapped,
    }

    # Stop SCM services only after every fallible ordinary-gateway step; from here any
    # error restores attempted services and already-paused gateways before aborting.
    paused_services = []
    current_service_name = None
    try:
        for service in service_gateways:
            current_service_name = str(service.name)
            _stop_windows_gateway_service(
                current_service_name,
                expected_processes=tuple(
                    getattr(service, "descendant_identities", ())
                ),
                expected_service_identity=(
                    int(service.service_pid),
                    float(service.service_create_time),
                ),
                expected_gateway_identity=(
                    int(service.gateway_pid),
                    float(service.gateway_create_time),
                ),
            )
            paused_services.append(current_service_name)
            current_service_name = None
        if paused_services:
            token["services"] = paused_services
            token["expected_services"] = list(paused_services)
            token["restarted_services"] = []
            token["service_profiles"] = {
                str(service.name): str(service.profile)
                for service in service_gateways
                if str(service.name) in paused_services
            }
            print(
                "  ✓ Paused Windows gateway service(s): "
                + ", ".join(paused_services)
            )
        return token
    except Exception as exc:
        restore_names = []
        if current_service_name:
            restore_names.append(current_service_name)
        restore_names.extend(reversed(paused_services))
        rollback_failures = []
        for service_name in dict.fromkeys(restore_names):
            try:
                _restore_windows_gateway_service(service_name)
            except Exception as restore_exc:
                rollback_failures.append(f"{service_name}: {restore_exc}")
        if profiles or unmapped:
            try:
                _resume_windows_gateways_after_update(token)
            except Exception as restore_exc:
                rollback_failures.append(f"ordinary gateways: {restore_exc}")
        failed_service = current_service_name or "unknown"
        detail = f"Could not stop Windows gateway service {failed_service}: {exc}"
        if rollback_failures:
            detail += "; rollback failures: " + "; ".join(rollback_failures)
        raise RuntimeError(detail) from exc


def _cold_start_windows_gateway_after_update() -> bool:
    """Direct-spawn a detached gateway after update for the ``cold_start_if_installed`` case (installed but down).

    Uses ``gateway_windows._spawn_detached`` (same hidden-console + breakaway path as ``hermes gateway start``).
    Idempotent: re-checks nothing is running so a concurrent autostart can't duplicate. A successful Popen
    doesn't prove survival (a job object denying breakaway kills it), so success is gated on the liveness poll.
    """
    from hermes_cli.update_cmd import _desktop_owns_gateway_lifecycle, _m
    if not _m()._is_windows():
        return True
    try:
        from hermes_cli import gateway_windows
        from hermes_cli.gateway import find_gateway_pids
    except Exception as exc:
        raise RuntimeError(
            f"Could not load Windows gateway cold-start helpers: {exc}"
        ) from exc

    # Re-check liveness right before spawning: autostart may have brought one up. Don't double-start.
    try:
        if list(find_gateway_pids(all_profiles=True)):
            return True
    except Exception as exc:
        raise RuntimeError(
            f"Could not re-check gateway liveness before cold-start: {exc}"
        ) from exc

    try:
        if _desktop_owns_gateway_lifecycle():
            logger.debug(
                "Skipping Windows gateway cold-start: Desktop owns gateway lifecycle"
            )
            return True
    except Exception as exc:
        raise RuntimeError(
            "Could not re-check Desktop gateway-lifecycle ownership before cold-start: "
            f"{exc}"
        ) from exc

    try:
        pid = gateway_windows._spawn_detached()
    except Exception as exc:
        raise RuntimeError(f"Could not cold-start Windows gateway after update: {exc}") from exc

    if not pid:
        raise RuntimeError("Windows gateway cold-start did not return a process ID")
    ready_pids = gateway_windows._wait_for_gateway_ready()
    if not ready_pids:
        raise RuntimeError(
            f"Windows gateway cold-start PID {pid} did not become ready"
        )
    print()
    print(
        "✓ Gateway started via cold-start after update "
        f"(PID: {', '.join(map(str, ready_pids))})"
    )
    # Persist vouched PIDs so a death AFTER updater exit (Job Object teardown) is
    # reported by the next CLI invocation. Best-effort.
    with suppress(Exception):
        gateway_windows._write_start_attestation(
            ready_pids, "cold-start after update"
        )
    return True


def _refresh_windows_gateway_launchers() -> None:
    """Regenerate installed Windows gateway launcher scripts after update; best-effort, never fails the update.

    Launchers are written once at install, so old installs kept launching via ``pythonw.exe`` (conhost flashes,
    ``sys.stderr is None`` death). The task's /TR points at a stable path, so rewriting in place retargets it
    without schtasks/UAC. ``_write_task_script`` is idempotent.
    """
    from hermes_cli.update_cmd import _m
    if not _m()._is_windows():
        return
    with _best_effort('Could not refresh Windows gateway launchers after update: %s'):
        from hermes_cli import gateway_windows

        if not gateway_windows.is_installed():
            return
        gateway_windows._write_task_script()
        print("  ✓ Refreshed Windows gateway launcher scripts")


def _refresh_bootstrap_cache_scripts(branch: str = "main") -> None:
    """Overwrite ``$HERMES_HOME/bootstrap-cache/install-<ref>.{ps1,sh}`` for *branch* from the fresh checkout.

    Old ``hermes-setup.exe`` builds NEVER re-download a cached branch-ref script (and have no self-update), so a
    stale one runs months-old code forever; refreshing turns that reuse into a feature (newer installers
    re-download anyway). Guards mirror ``install_script.rs``: only the sanitized *branch* key is rewritten
    (sibling refs untouched); commit-SHA pins (7-40 hex, incl. abbreviated) are immutable and skipped.
    The .ps1 copy gets a UTF-8 BOM to match the cache format. Best-effort: never fails the update.
    """
    from hermes_cli.update_cmd import _m
    with _best_effort('Could not refresh bootstrap-cache scripts after update: %s'):
        import re as _re

        cache_dir = Path(_m().get_hermes_home()) / "bootstrap-cache"
        if not cache_dir.is_dir():
            return
        # Mirror install_script.rs::sanitize_ref().
        safe_ref = _re.sub(r"[^A-Za-z0-9._-]", "_", str(branch or "main"))
        # Mirror install_script.rs::is_valid_commit(): immutable commit pin, never rewrite.
        if _re.fullmatch(r"[0-9a-fA-F]{7,40}", safe_ref):
            return
        refreshed = []
        for kind, src_name in (("ps1", "install.ps1"), ("sh", "install.sh")):
            src = _m().PROJECT_ROOT / "scripts" / src_name
            if not src.is_file():
                continue
            cached = cache_dir / f"install-{safe_ref}.{kind}"
            if not cached.is_file():
                continue  # this ref was never bootstrap-cached — nothing to heal
            data = src.read_bytes()
            if kind == "ps1" and not data.startswith(b"\xef\xbb\xbf"):
                # PowerShell needs the BOM or localized/em-dash text mis-decodes.
                data = b"\xef\xbb\xbf" + data
            if cached.read_bytes() == data:
                continue  # already current
            tmp = cached.with_suffix(cached.suffix + ".tmp")
            tmp.write_bytes(data)
            os.replace(tmp, cached)
            refreshed.append(cached.name)
        if refreshed:
            print(
                "  ✓ Refreshed installer bootstrap-cache script(s): "
                + ", ".join(sorted(refreshed))
            )


def _resume_windows_gateways_after_update(token: dict | None) -> None:
    """Restart Windows profile gateways previously paused for update."""
    from hermes_cli.update_cmd import _m, _start_windows_gateway_service
    if not token or not token.get("resume_needed"):
        return
    if not _m()._is_windows():
        token["resume_needed"] = False
        return

    # Regenerate launcher scripts before respawning so a legacy pythonw-era
    # autostart entry comes back on the current design at next login too.
    _m()._refresh_windows_gateway_launchers()

    services = list(token.get("services") or [])
    token.setdefault("expected_services", list(services))
    verified_restarts = list(token.get("restarted_services") or [])
    restarted_services = []
    failed_services = []
    for service_name in services:
        try:
            _start_windows_gateway_service(str(service_name))
            restarted_services.append(str(service_name))
            if str(service_name) not in verified_restarts:
                verified_restarts.append(str(service_name))
        except Exception as exc:
            logger.warning(
                "Could not restart Windows gateway service %s after update: %s",
                service_name,
                exc,
            )
            print(f"  ⚠ Could not restart Windows gateway service: {service_name}")
            failed_services.append(str(service_name))

    if failed_services:
        token["services"] = failed_services
        token["restarted_services"] = verified_restarts
        raise RuntimeError(
            "Could not restart Windows gateway service(s): "
            + ", ".join(failed_services)
        )
    token["services"] = []
    token["restarted_services"] = verified_restarts
    if restarted_services:
        print()
        print(
            "  ✓ Restarted Windows gateway service(s): "
            + ", ".join(restarted_services)
        )

    profiles = token.get("profiles") or {}
    unmapped = token.get("unmapped") or []
    cold_start = bool(token.get("cold_start_if_installed"))
    if not profiles and not any(u.get("argv") for u in unmapped):
        if cold_start:
            if not _m()._cold_start_windows_gateway_after_update():
                raise RuntimeError("Windows gateway cold-start was not verified")
            token["cold_start_if_installed"] = False
        token["resume_needed"] = False
        return

    try:
        from hermes_cli.gateway import (
            launch_detached_gateway_restart_by_cmdline,
            launch_detached_profile_gateway_restart,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Could not load Windows gateway restart helper: {exc}"
        ) from exc

    relaunched = []
    failed_profiles = {}
    for profile, old_pid in sorted(profiles.items()):
        try:
            if launch_detached_profile_gateway_restart(str(profile), int(old_pid)):
                relaunched.append(str(profile))
            else:
                failed_profiles[str(profile)] = int(old_pid)
        except Exception as exc:
            logger.debug(
                "Could not restart Windows gateway profile %s after update: %s",
                profile,
                exc,
            )
            failed_profiles[str(profile)] = int(old_pid)

    # Feed the plan-vs-execution reconciliation (else a relaunched gateway is reported
    # "unaccounted", exit 1). Failed relaunches are deliberately left off so they
    # still surface (Windows has no watcher to recover them).
    token["relaunched_profiles"] = relaunched

    # Respawn unmapped gateways by replaying the argv snapshotted before the kill.
    unmapped_relaunched = 0
    failed_unmapped = []
    for entry in unmapped:
        argv = entry.get("argv")
        old_pid = entry.get("pid")
        if not argv or not old_pid:
            failed_unmapped.append(entry)
            continue
        try:
            if launch_detached_gateway_restart_by_cmdline(int(old_pid), list(argv)):
                unmapped_relaunched += 1
            else:
                failed_unmapped.append(entry)
        except Exception as exc:
            logger.debug(
                "Could not restart unmapped Windows gateway (pid %s) after update: %s",
                old_pid,
                exc,
            )
            failed_unmapped.append(entry)

    token["profiles"] = failed_profiles
    token["unmapped"] = failed_unmapped
    if failed_profiles or failed_unmapped:
        raise RuntimeError("Could not restart every paused Windows gateway")

    # A truthy launch only proves the watcher was created; a parent Job Object denying
    # CREATE_BREAKAWAY_FROM_JOB can kill the gateway on updater teardown. Verify with
    # the same liveness poll every spawn path uses; all_profiles=True covers the fleet.
    if relaunched or unmapped_relaunched:
        try:
            from hermes_cli import gateway_windows
        except Exception as exc:
            raise RuntimeError(
                f"Could not load Windows gateway liveness helpers: {exc}"
            ) from exc
        ready_pids = gateway_windows._wait_for_gateway_ready(
            timeout_s=30.0, all_profiles=True
        )
        if not ready_pids:
            token["profiles"] = dict(profiles)
            token["unmapped"] = list(unmapped)
            print()
            print(
                "  ⚠ Windows gateway restart could not be verified — no stable "
                "gateway process appeared after relaunch."
            )
            print(
                "    (The respawned gateway may have been killed by a parent "
                "Job Object during updater teardown, #48820.)"
            )
            print("    Recover with: hermes gateway restart")
            raise RuntimeError(
                "Windows gateway relaunch after update was not verified alive"
            )
        # Persist vouched PIDs so a death AFTER updater exit is reported by the
        # next CLI invocation. Best-effort.
        with suppress(Exception):
            gateway_windows._write_start_attestation(
                ready_pids, "post-update relaunch"
            )

    token["resume_needed"] = False

    if relaunched:
        print()
        print(f"  ✓ Restarting Windows gateway profile(s): {', '.join(relaunched)}")
    if unmapped_relaunched:
        if not relaunched:
            print()
        print(
            f"  ✓ Restarting {unmapped_relaunched} unmapped Windows gateway process(es)"
        )


def _resume_windows_gateways_and_merge_outcome(outcome, _windows_gateway_resume, gateway_mode: bool):
    """Resume gateways paused for a Windows update and fold the token into ``outcome``'s systemd/launchd-style
    bookkeeping so reconciliation never reports a healthy gateway as unaccounted. Must never abort the update.
    """
    from hermes_cli.update_cmd import _m, _write_gateway_update_exit_code
    try:
        _m()._resume_windows_gateways_after_update(_windows_gateway_resume)
    except Exception as _windows_resume_exc:
        outcome.incomplete = True
        outcome.phase_errors.append(str(_windows_resume_exc))
        print(
            "  ⚠ Windows gateway service restart incomplete: "
            f"{_windows_resume_exc}"
        )
        if gateway_mode:
            _write_gateway_update_exit_code(False)

    if isinstance(_windows_gateway_resume, dict):
        # Failed relaunches are absent from the token so they still surface. Best-effort.
        with _best_effort('Could not merge Windows relaunch outcome into fleet reconciliation bookkeeping: %s'):
            for _win_profile in _windows_gateway_resume.get("relaunched_profiles") or []:
                if _win_profile not in outcome.relaunched_profiles:
                    outcome.relaunched_profiles.append(_win_profile)
        windows_restarted = list(
            _windows_gateway_resume.get("restarted_services") or []
        )
        for service_name in windows_restarted:
            if service_name not in outcome.restarted_services:
                outcome.restarted_services.append(service_name)
        service_profiles = _windows_gateway_resume.get("service_profiles") or {}
        for service_name in windows_restarted:
            profile_name = service_profiles.get(service_name)
            if profile_name and profile_name not in outcome.relaunched_profiles:
                outcome.relaunched_profiles.append(profile_name)
        pending_services = list(_windows_gateway_resume.get("services") or [])
        for service_name in pending_services:
            label = str(service_profiles.get(service_name) or service_name)
            if label not in outcome.failed_or_stale_units:
                outcome.failed_or_stale_units.append(label)

        with suppress(Exception):
            from hermes_cli.update_receipt import record_gateway_restart

            record_gateway_restart(
                restarted_services=outcome.restarted_services,
                relaunched_profiles=outcome.relaunched_profiles,
                externally_supervised_profiles=outcome.externally_supervised_profiles,
                killed_pids=sorted(outcome.killed_pids),
                failed_units=outcome.failed_or_stale_units,
                incomplete=(
                    outcome.incomplete
                    or bool(outcome.failed_or_stale_units)
                ),
                phase_error="; ".join(outcome.phase_errors) or None,
            )


def _clear_windows_venv_holders_or_exit(args, gateway_mode: bool, _windows_gateway_resume):
    """Windows: stop every venv-python holder we can positively identify, else resume paused gateways and exit 2.

    Rungs in order: leftover pausable gateways -> ledger orphaned backends -> orphaned Desktop backends ->
    ledger manual serve (relaunched at exit on the same bind) -> GUI hand-off leaks. Remaining holders are
    refused (the sync would corrupt against a locked .pyd).
    """
    from hermes_cli.update_cmd import _m, _record_update_step, _refuse_gateway_ancestor_tree_kill
    _venv_holders = _m()._detect_venv_python_processes()
    if _venv_holders:
        _gateway_holders = _m()._leftover_pausable_gateway_pids(_venv_holders)
        if _gateway_holders is not None:
            if _refuse_gateway_ancestor_tree_kill(
                _gateway_holders, gateway_mode=gateway_mode
            ):
                _m()._resume_windows_gateways_after_update(
                    _windows_gateway_resume
                )
                sys.exit(2)
            # Gateways the pause machinery owns (respawned in the pause->guard window or
            # unmapped spawn path): stop and re-check; post-update resume brings them back.
            from gateway.status import get_process_start_time, terminate_pid

            print(
                f"  ⚠ {len(_gateway_holders)} gateway process(es) still "
                "hold the venv after the pause; stopping them"
            )
            for _pid in _gateway_holders:
                try:
                    pid_int = int(_pid)
                    terminate_pid(
                        pid_int,
                        force=True,
                        expected_start_time=get_process_start_time(pid_int),
                    )
                except Exception as exc:
                    logger.debug(
                        "Could not stop leftover gateway %s: %s", _pid, exc
                    )
            _time.sleep(1.0)
            _venv_holders = _m()._detect_venv_python_processes()
    if _venv_holders:
        # Positive-identity rung (any context): spawn ledger proves the holder is an
        # orphaned backend (self-registered, spawner provably dead). No PPID archaeology.
        _ledger_backends = _m()._ledger_reapable_backend_pids(_venv_holders)
        if _ledger_backends:
            print(
                f"  ⚠ {len(_ledger_backends)} ledger-identified orphaned "
                "Hermes backend process(es) hold the venv; stopping their trees"
            )
            _m()._stop_process_trees(_ledger_backends)
            _time.sleep(1.0)
            _venv_holders = _m()._detect_venv_python_processes()
    if _venv_holders:
        _orphan_backends = _m()._orphaned_desktop_backend_pids(_venv_holders)
        if _orphan_backends:
            # Desktop `serve` backends whose app is GONE: nothing respawns an orphan, so
            # reap the tree. Live-Desktop backends return None and keep the refusal.
            print(
                f"  ⚠ {len(_orphan_backends)} orphaned Desktop backend "
                "process(es) still hold the venv; stopping their trees"
            )
            _m()._stop_process_trees(_orphan_backends)
            _time.sleep(1.0)
            _venv_holders = _m()._detect_venv_python_processes()
    if _venv_holders:
        # Manual serve/dashboard rung (e.g. `hermes serve --host <ip>` for a REMOTE Desktop):
        # ledger identity only (spawner dead; Desktop-owned keep the refusal). Stop and
        # register an idempotent atexit relaunch on the SAME host/port/profile — success or failure.
        _serve_entries = _m()._ledger_manual_serve_holders(_venv_holders)
        if _serve_entries:
            print(
                f"  ⚠ {len(_serve_entries)} manual serve/dashboard "
                "backend(s) hold the venv; stopping them for the update "
                "(they will be relaunched on their recorded endpoints)"
            )
            _m()._stop_process_trees(
                [int(e["pid"]) for e in _serve_entries]
            )
            _serve_resume_token = {
                "pending": True,
                "entries": _serve_entries,
            }
            _record_update_step(
                "serve_pause",
                True,
                f"stopped={len(_serve_entries)}",
            )
            import atexit as _serve_atexit

            _serve_atexit.register(
                _m()._relaunch_stopped_serves, _serve_resume_token
            )
            _time.sleep(1.0)
            _venv_holders = _m()._detect_venv_python_processes()
    if _venv_holders:
        # Final rung: in a GUI hand-off (`--gateway` + update-incomplete marker) the Desktop
        # is contractually gone; surviving `serve` backends are leaks even with a live
        # parent (which made the orphan-only rung bail and hang) — reap by cmdline.
        _handoff = False
        try:
            _handoff = bool(getattr(args, "gateway", False)) and _m()._update_marker_path().exists()
        except Exception:
            _handoff = False
        # Fail closed: unverifiable shim state is treated as a live shim (keep refusing).
        _no_live_shim = False
        try:
            _scripts_dir = _m()._venv_scripts_dir()
            if _scripts_dir is not None:
                _no_live_shim = not _m()._detect_concurrent_hermes_instances(_scripts_dir)
        except Exception:
            _no_live_shim = False
        if _handoff and _no_live_shim:
            _handoff_backends = _m()._handoff_reapable_backend_pids(_venv_holders)
            if _handoff_backends:
                print(
                    f"  ⚠ {len(_handoff_backends)} Hermes backend process(es) "
                    "still hold the venv after the Desktop hand-off; "
                    "stopping their trees"
                )
                _m()._stop_process_trees(_handoff_backends)
                _time.sleep(1.0)
                _venv_holders = _m()._detect_venv_python_processes()
    if _venv_holders:
        print(_format_venv_python_holders_message(_venv_holders))
        _m()._resume_windows_gateways_after_update(_windows_gateway_resume)
        sys.exit(2)
