"""Windows gateway lifecycle for ``hermes update``: pause/resume/cold-start the gateway service, sweep venv-holding python processes, reap orphaned Desktop backends.

Split out of ``hermes_cli/update_cmd.py``; every moved name is re-imported there, so
``hermes_cli.update_cmd.<name>`` keeps resolving (and monkeypatching) as before.
Origin-internal helpers are imported lazily inside each function (no import cycle;
test patches on ``hermes_cli.update_cmd.<name>`` stay effective).
"""

import logging
import os
import subprocess
import sys
import time as _time
from datetime import datetime
from pathlib import Path

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
        try:
            if _pid_exists(pid):
                survivors.add(pid)
        except Exception:
            pass
    return survivors


def _self_and_non_gateway_ancestor_pids(psutil) -> set[int]:
    """PIDs a venv-holder scan must never nominate: this process and its ancestry.

    #87594: do NOT blanket-exclude ancestors. When ``/update`` runs from a
    messaging platform the updater is a CHILD of the gateway; hiding it means
    the pause machinery never sees the one process it exists to stop and the
    update dead-ends on ``venv-blocked``. Keep a GATEWAY ancestor visible (the
    pause path stops it gracefully; a detached child survives on Windows) and
    exclude every other ancestor — an updater must never nominate its own
    interactive ancestry as a blocker.
    """
    try:
        from gateway.status import looks_like_gateway_command_line as _is_gw
    except Exception:
        _is_gw = None
    skip: set[int] = {os.getpid()}
    try:
        for anc in psutil.Process().parents():
            try:
                anc_cmdline = " ".join(anc.cmdline() or [])
            except Exception:
                anc_cmdline = ""
            if _is_gw is not None and anc_cmdline and _is_gw(anc_cmdline):
                continue
            skip.add(int(anc.pid))
    except Exception:
        pass
    return skip


def _detect_venv_python_processes(
    *, exclude_pids: set[int] | None = None
) -> list[tuple[int, str, str]]:
    """Find live processes running from the project venv's interpreter.

    The hermes.exe shim guard misses the biggest Windows lock-holder class:
    the Desktop backend (``python.exe -m hermes_cli.main serve``) and anything
    running off ``venv\\Scripts\\python(w).exe``. They keep native ``.pyd``
    files mapped, so a mid-update dependency sync dies with access-denied and
    strands the venv half-updated.

    Killing them is pointless (the Desktop app respawns its backend), so the
    caller should refuse and ask the user to close the app. Returns
    ``(pid, name, cmdline)`` tuples; empty off-Windows / without psutil / no
    matches. This process and its ancestors are excluded. Never raises.
    """
    from hermes_cli.update_cmd import _m
    if not _m()._is_windows():
        return []
    try:
        import psutil
    except Exception:
        return []

    venv_dir = _m().PROJECT_ROOT / "venv"
    try:
        venv_prefix = str(venv_dir.resolve()).lower().rstrip(os.sep) + os.sep
    except OSError:
        venv_prefix = str(venv_dir).lower().rstrip(os.sep) + os.sep
    try:
        root_prefix = str(_m().PROJECT_ROOT.resolve()).lower().rstrip(os.sep) + os.sep
    except OSError:
        root_prefix = str(_m().PROJECT_ROOT).lower().rstrip(os.sep) + os.sep

    skip: set[int] = set(exclude_pids or set())
    skip |= _self_and_non_gateway_ancestor_pids(psutil)

    matches: list[tuple[int, str, str]] = []
    try:
        # On Windows cmdline/cwd are expensive per-process queries; with 500+
        # processes prefetching them can exceed the Desktop preflight watchdog.
        # Collect cheap identity fields first, fetch cmdline/cwd lazily for
        # plausible Python/uv/Hermes candidates.
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
        # Primary match: the executable itself lives under this venv
        # (venv\Scripts\python(w).exe — the desktop backend / gateway case).
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
        # Fallback: uv/base-interpreter trampolines run a python whose exe is
        # OUTSIDE the venv yet still holds its .pyd files. Match on the cmdline
        # instead: this venv's path, or `-m hermes_cli.main` tied to this
        # install (root in the cmdline or as cwd).
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
        # Return the FULL cmdline: callers parse it (the Desktop preflight's
        # pausable-gateway exemption looks for `gateway run`). Truncating here
        # once cut long interpreter paths before the argv, so autostarted
        # gateways were misreported as blockers. Truncate only at display time.
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
    """Top-level CLI flags that consume a value — derived from the REAL parser.

    Introspects ``build_top_level_parser()`` (every option with nargs != 0) so
    the holder classifier can't drift from argparse (#91869: a handwritten
    subset misparsed ``--reasoning high serve`` as subcommand ``high``). The
    pre-argparse profile selectors (``--profile``/``-p``, ``--config``) are
    added explicitly since they're stripped before argparse sees argv. Falls
    back to a static snapshot when the parser can't be imported (the updater
    must classify holders even on a broken tree). Cached per process.
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
    """The actual Hermes SUBCOMMAND a venv-holder argv runs, or None.

    Token-based, never substring (#90778: ``kanban --preserve-cache`` contains
    \"serve\" and got labeled as the Desktop backend). Finds the
    ``hermes_cli.main`` / ``hermes(.exe)`` entry token, then returns the first
    following token that isn't a flag or a flag's value (profile selectors
    skipped). None when undeterminable — callers must NOT guess a label.
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

    Holder labels come from the parsed SUBCOMMAND, never substring matching
    (#90778): a standalone ``hermes dashboard`` must not be labeled as the
    Desktop backend (advice to close an app that isn't running), and flags
    like ``--preserve-cache`` must not match \"serve\". Unknown argv gets no
    hint rather than a wrong one.
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
    """Return venv-interpreter ancestors of *pids* that hold the install open.

    On Windows a gateway started through the venv shim is a two-process chain:
    ``venv\\Scripts\\python.exe`` (the launcher, which keeps venv ``.pyd``
    files mapped) spawns the real interpreter from uv's managed CPython. The
    PID file is written by the *child*, so ``find_gateway_pids()`` / the pause
    set only see the uv-side worker, while ``_detect_venv_python_processes()``
    (venv path prefix) sees the *launcher*. The sets are disjoint, so a paused
    gateway still tripped the venv-holder guard and aborted the update.

    Walk one hop up from each mapped gateway PID and keep only ancestors under
    the project venv; unrelated ancestors (the Scheduled Task's ``cmd.exe``,
    an operator's shell) are ignored to bound the blast radius. Never raises.
    """
    from hermes_cli.update_cmd import _m
    if not _m()._is_windows() or not pids:
        return []
    try:
        import psutil
    except Exception:
        return []

    venv_dir = _m().PROJECT_ROOT / "venv"
    try:
        venv_prefix = str(venv_dir.resolve()).lower().rstrip(os.sep) + os.sep
    except OSError:
        venv_prefix = str(venv_dir).lower().rstrip(os.sep) + os.sep

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
    """PIDs from *matches* when every remaining venv holder is a pausable gateway.

    ``_pause_windows_gateways_for_update()`` stops the gateways its discovery
    finds, but the venv-holder guard sees the process table as it is *now*: a
    gateway respawned by its supervisor inside the pause→guard window, or one
    started through an unmapped spawn path, still holds venv ``.pyd`` files and
    would dead-end the update on exactly the process the pause exists to stop.

    Holders are classified with the same matcher the Desktop preflight uses
    (``_is_pausable_gateway``) so exemption and tolerance cannot drift apart.
    The scan keeps only a 120-char cmdline prefix, so live argv is re-read via
    psutil when possible, falling back to the prefix.

    Returns ``None`` when any holder is not a pausable gateway (operator REPL,
    stray script, Desktop backend) — nothing downstream can pause it, so the
    guard must keep refusing.
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
            try:
                argv = " ".join(psutil.Process(int(pid)).cmdline()) or cmdline
            except Exception:
                pass
        if not _is_pausable_gateway(argv):
            return None
        pids.append(int(pid))
    return pids


def _refuse_gateway_ancestor_tree_kill(
    pids: list[int], *, gateway_mode: bool
) -> bool:
    """Refuse a plain Windows update that would kill its own process tree.

    A chat agent can run plain ``hermes update`` via its terminal tool, making
    the updater a child of the gateway; leftover-holder recovery uses
    ``taskkill /T /F``, so force-stopping that gateway kills the updater before
    it mutates the checkout (#98814). ``/update`` (``--gateway`` hand-off) is
    exempt: it detaches the updater with file-based progress/result delivery.
    Otherwise refuse only when a nominated gateway is positively an ancestor of
    this process; if ancestry cannot be established, keep existing recovery.
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
    """Ledger entries for venv holders that are MANUAL serve/dashboard backends.

    Positive identity only (#63206): the process self-registered in the spawn
    ledger with purpose serve/dashboard, its (pid, create_time) still matches
    a live process, and its recorded spawner is NOT alive (a Desktop-owned
    backend keeps its live Electron spawner and must keep the refusal — the
    app would respawn what we kill; a PowerShell-launched serve has no live
    Hermes spawner). Returns the full ledger entries so the relauncher can
    rebuild the launch command from structured host/port/profile instead of
    parsing argv.
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
    """Rebuild launch commands for stopped serves from structured identity.

    Uses the ledger's host/port/profile fields — never argv parsing (a
    joined argv string cannot round-trip Windows paths with spaces). Entries
    without a recorded port are skipped; the caller prints the manual hint
    for those.
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

    Mirrors the gateway resume token contract: `pending` flips False on the
    first invocation so the explicit call and the atexit registration cannot
    double-spawn (#63206).
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


def _orphaned_desktop_backend_pids(
    matches: list[tuple[int, str, str]],
) -> list[tuple[int, int]] | None:
    """PIDs from *matches* when every remaining holder is an ORPHANED backend.

    The venv-holder guard refuses on the Desktop app's ``serve`` backend by
    design: while the Desktop is open, killing it is futile (the app respawns
    it within seconds). But in the GUI-updater hand-off the Desktop has
    *already exited* — by contract it tree-kills its backends before spawning
    hermes-setup, and the update-in-progress marker parks any relaunched
    Desktop (#50238). A ``serve`` backend still holding the venv then is a
    straggler whose supervisor is gone (SIGTERM raced its spawn, or a crashed
    window); refusing on it dead-ends the update with "Hermes is still
    running" while the user sees zero open windows.

    A holder qualifies only when BOTH hold:

    - its cmdline is a Hermes backend (``hermes_cli.main`` + ``serve`` /
      ``dashboard``), and
    - its supervising parent is demonstrably gone: the parent PID no longer
      exists, or was reused (parent created *after* the child).

    Tree-aware: the scanner may also return an orphan's managed-runtime child
    (the ``.hermes-runtime`` interpreter), which has a live parent and is not
    a ``serve`` cmdline. Holders inside an accepted orphan root's tree are
    folded into that root; only roots are returned (``taskkill /T`` reaps
    descendants).

    Any other live-parent backend, non-backend holder outside an orphan tree,
    or unprovable case disqualifies the whole set → ``None`` (keep refusing).
    Also ``None`` when psutil is unavailable. Never raises.
    """
    try:
        import psutil  # type: ignore
    except Exception:
        return None

    def _is_backend(argv_low: str) -> bool:
        return "hermes_cli.main" in argv_low and (
            " serve" in argv_low or " dashboard" in argv_low
        )

    # Pass 1: find orphaned backend ROOTS among the holders.
    roots: list[tuple[int, int]] = []
    remaining: list[tuple[int, str]] = []  # (pid, argv_low) still to justify
    for pid, _name, cmdline in matches:
        argv = cmdline
        try:
            argv = " ".join(psutil.Process(int(pid)).cmdline()) or cmdline
        except psutil.NoSuchProcess:
            # Holder exited between scan and classification — nothing to
            # reap, nothing blocking. Skip it.
            continue
        except Exception:
            pass
        low = argv.lower()
        if not _is_backend(low):
            remaining.append((int(pid), low))
            continue
        try:
            proc = psutil.Process(int(pid))
            # Fingerprint from the SAME psutil handle, quantized to centiseconds
            # like gateway.status.get_process_start_time on Windows, so it
            # round-trips through pid_is_hermes at kill time (/proc/<pid>/stat
            # would read the HOST table in different units).
            process_start_time = int(round(proc.create_time() * 100))
        except psutil.NoSuchProcess:
            # The candidate itself exited during classification; there is
            # nothing left to reap and no identity to pass to taskkill.
            continue
        except Exception:
            return None

        try:
            ppid = proc.ppid()
            parent = psutil.Process(ppid) if ppid else None
            if parent is not None and parent.is_running():
                # PID-reuse check: a "parent" created after its child is a
                # recycled PID, not the real (dead) supervisor.
                if parent.create_time() <= proc.create_time():
                    # Live parent — not a root, but possibly an orphan root's
                    # descendant (the venv python.exe trampoline re-execs the
                    # uv interpreter with the SAME argv). Defer to pass 2.
                    remaining.append((int(pid), low))
                    continue
        except psutil.NoSuchProcess:
            pass  # parent gone → orphan
        except Exception:
            return None
        roots.append((int(pid), process_start_time))

    # Pass 2: every non-backend holder must be a descendant of an accepted
    # orphan root — then it dies with the root's tree reap. Anything else
    # (operator REPL, stray script) keeps the refusal.
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
    """PIDs positively identified by the spawn ledger as orphaned backends.

    The strongest rung: look each venv holder up in the machine spawn ledger
    (``hermes_cli.process_identity``) instead of inferring lineage from PPIDs
    or cmdline shape. A holder qualifies when ALL of:

    - its ``(pid, create_time)`` matches a live ledger entry (PID reuse
      cannot forge this pair);
    - the entry's purpose is a reapable backend kind (serve/dashboard/
      gateway — never interactive processes);
    - the entry's recorded SPAWNER is provably dead (``spawner_is_dead``).

    Safe in ANY update context — the process itself declared its supervisor
    and that supervisor is gone. Holders not in the ledger fall through to
    later rungs and never disqualify identified ones. Never raises.
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
    """PIDs of Hermes ``serve``/``dashboard`` backends safe to reap during a
    GUI-updater hand-off, INCLUDING ones with a still-live parent.

    Complements ``_orphaned_desktop_backend_pids``, which returns ``None``
    (keep refusing) the moment ANY holder has a live parent. That produced a
    field incident: a Windows Desktop hand-off (``update --yes --gateway
    --force``) left a swarm of per-profile ``serve`` backends holding
    ``cryptography\\_rust.pyd``, several with a lingering parent (tearing-down
    Electron, or the launcher→worker chain mid-exit), so the orphan check
    disqualified the WHOLE set and the update hung for 12 minutes.

    The hand-off is the safe signal: with the update-incomplete marker claimed
    AND a ``--gateway`` run AND no live Desktop shim (``hermes.exe``), nothing
    legitimate supervises or respawns a ``serve`` backend from this venv (the
    Desktop tree-kills its backends and parks relaunch behind the marker,
    #50238). Any ``serve`` backend still holding the venv is a leak, live
    parent or not, and tree-reaping it is correct rather than a race.

    Guards: only Hermes backends (``hermes_cli.main`` + ``serve``/``dashboard``)
    qualify — a non-backend holder disqualifies the whole set → ``None``; the
    CALLER must have confirmed the hand-off gate above (outside it the stricter
    orphan-only path stands); psutil unavailable → ``None``. Returns backend
    root PIDs to tree-reap, or ``None`` to leave the decision to the caller's
    other rungs. Never raises.
    """
    try:
        import psutil  # type: ignore
    except Exception:
        return None

    def _is_backend(argv_low: str) -> bool:
        return "hermes_cli.main" in argv_low and (
            " serve" in argv_low or " dashboard" in argv_low
        )

    roots: list[int] = []
    for pid, _name, cmdline in matches:
        argv = cmdline
        try:
            argv = " ".join(psutil.Process(int(pid)).cmdline()) or cmdline
        except psutil.NoSuchProcess:
            # Exited between scan and classification — nothing to reap.
            continue
        except Exception:
            pass
        if not _is_backend(argv.lower()):
            # A non-backend holder during a hand-off is unexpected; refuse the
            # whole set rather than reap something we cannot justify.
            return None
        roots.append(int(pid))

    return roots or None


def _stop_process_trees(
    pids: list[int] | list[tuple[int, int]],
) -> None:
    """Force-stop each PID with its full child tree (Windows).

    ``taskkill /T /F`` mirrors the Desktop's ``forceKillProcessTree`` and
    install.ps1's venv sweep: stopping only the parent can leave a managed
    ``.hermes-runtime`` interpreter child alive and holding the install open
    (#70026). Best effort; never raises.
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
    """True for this-install ``hermes serve`` / ``hermes dashboard`` argv.

    That is the Desktop control plane, not the messaging gateway (serve and
    dashboard host no platform adapters, #92091); do not feed this into
    ``looks_like_gateway_command_line``. Token-based via the parser-derived
    subcommand classifier — never substring (#90778/#91869: ``kanban
    --preserve-cache`` contains "serve", ``-m dashboard chat`` contains
    " dashboard"). An undeterminable subcommand is NOT a control plane.
    """
    if "hermes_cli.main" not in (cmdline or "").lower():
        return False
    return _hermes_holder_subcommand(cmdline) in ("serve", "dashboard")


def _desktop_owns_gateway_lifecycle() -> bool:
    """True when Desktop currently supervises this install's control plane.

    The updater must not steal gateway start in that case: Desktop owns
    start/stop via ``/api/gateway/*``. This is *not* proof messaging is
    already served — a live serve process is the control plane, and the
    gateway is a detached sibling (#76129 / #92091).

    Prefer the spawn ledger (owned identity). Fall back to the install-scoped
    venv-holder scan already used by the lock guard; an orphaned control-plane
    process (supervisor gone) does not count.
    """
    from hermes_cli.update_cmd import _m
    try:
        from hermes_cli.process_identity import ledger_entries, spawner_is_dead

        for entry in ledger_entries():
            if entry.get("purpose") not in ("serve", "dashboard"):
                continue
            if spawner_is_dead(entry) is False:
                return True
    except Exception as exc:
        logger.debug("Desktop-lifecycle ledger probe failed: %s", exc)

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
            # Cannot prove orphanhood; a live this-install control plane is
            # enough to refuse stealing gateway start.
            return True
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
            # A vanished process is clear.
            return False
        except Exception:
            # AccessDenied or any unknown probe failure stays fail-closed
            # because the venv may still be locked.
            return True
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
        # Return only if the original processes are gone too; a lingering
        # matching-identity process means venv mutation is unsafe — fail closed.
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
    # Timeout with the original descendants still alive — fail closed; venv mutation is unsafe.
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

    Windows scheduled/startup gateways run through pythonw.exe, so the generic
    hermes.exe concurrent-instance guard does not see them. They still import
    from the checkout and can keep files locked while ``git`` or ``uv`` updates
    the install. Stop only PIDs that the gateway discovery code identifies.
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
        # No gateway is running, but an installed autostart entry (Scheduled
        # Task / Startup-folder login item) is an explicit "I want a gateway"
        # signal. A gateway that died between updates (e.g. its spawning
        # terminal closed) would otherwise stay down until next login, since
        # the resume path only relaunches gateways that were running. Cold-start
        # one after the update; gateway-less users get nothing forced on them.
        #
        # Exception: Desktop owns this install's gateway lifecycle (live
        # supervised serve/dashboard); a vestigial autostart entry is not the
        # owner, and spawning ``gateway run`` beside Desktop races ports/state
        # (#76129). The skip is ownership, not liveness (#92091).
        try:
            if _desktop_owns_gateway_lifecycle():
                logger.debug(
                    "Skipping Windows gateway cold-start plan: "
                    "Desktop owns gateway lifecycle"
                )
                return None
        except Exception as exc:
            logger.debug(
                "Could not check Desktop gateway-lifecycle ownership before update: %s",
                exc,
            )
        try:
            from hermes_cli import gateway_windows

            if gateway_windows.is_installed():
                return {
                    "resume_needed": True,
                    "profiles": {},
                    "unmapped_pids": [],
                    "unmapped": [],
                    "cold_start_if_installed": True,
                }
        except Exception as exc:
            logger.debug(
                "Could not check Windows gateway autostart state before update: %s",
                exc,
            )
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
        # Socket-first pause (#92091 step 2): ask the gateway to drain and exit
        # itself. A positive ACK means it runs its own graceful restart path
        # (same drain as SIGUSR1/service restarts) and releases venv handles on
        # exit. No answer (older gateway, no socket) → the marker poll /
        # force-kill ladder below behaves exactly as before.
        try:
            from gateway.control_socket import pause_gateway_for_update

            ack = pause_gateway_for_update(Path(proc.path))
            if ack and (ack.get("pausing") or ack.get("already_stopping")):
                socket_acks.append(ack)
        except Exception as exc:
            logger.debug(
                "Socket pause unavailable for gateway %s: %s", pid, exc
            )

    # Resolve each mapped worker's venv-side launcher BEFORE draining: a
    # gracefully drained worker is gone when the wait returns, and a dead
    # pid's parent cannot be recovered (psutil raises NoSuchProcess). The
    # snapshot is stopped after the drain alongside the survivors.
    #
    # The drain targets the PID-file writer (uv-side worker); its parent is
    # usually the venv ``python.exe`` launcher, which keeps venv ``.pyd``
    # files mapped and is what ``_detect_venv_python_processes()`` reports.
    # Left alive, it trips the venv-holder guard though the gateway is stopped.
    launcher_pids = _m()._venv_launcher_ancestors(mapped_pids)

    print("→ Stopping Windows gateway process(es) before updating Hermes...")
    try:
        drain_timeout = max(float(_get_restart_drain_timeout()), 1.0)
    except Exception:
        drain_timeout = 10.0
    if socket_acks:
        # A socket-paused gateway drains its ACTIVE TURN before exiting; honor
        # the budget it declared (plus teardown grace) so a mid-turn gateway
        # isn't force-killed by a too-short wait.
        try:
            declared = max(
                float(a.get("drain_timeout") or 0.0) for a in socket_acks
            )
            drain_timeout = max(drain_timeout, declared + 10.0)
        except Exception:
            pass
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

    # Snapshot each unmapped gateway's argv *before* force-killing it so
    # ``_resume_windows_gateways_after_update`` can replay it. Unmapped = no
    # profile→PID-file mapping (e.g. a Scheduled Task running ``pythonw.exe -m
    # hermes_cli.main gateway run``); without this they were never restarted (#50090).
    unmapped: list[dict] = []
    for pid in unmapped_pids:
        argv = None
        try:
            argv = _capture_gateway_argv(int(pid))
        except Exception as exc:
            logger.debug("Could not capture argv for unmapped gateway %s: %s", pid, exc)
        unmapped.append({"pid": int(pid), "argv": argv})

    # Stop drain survivors, unmapped gateways, and the pre-drain launcher
    # snapshot. ``terminate_pid(force=True)`` is a tree kill; a launcher that
    # already exited with its worker raises ProcessLookupError and is skipped.
    force_killed = []
    for pid in sorted(set(survivors).union(unmapped_pids).union(launcher_pids)):
        try:
            pid_int = int(pid)
            terminate_pid(
                pid_int,
                force=True,
                expected_start_time=get_process_start_time(pid_int),
            )
            force_killed.append(pid_int)
        except (ProcessLookupError, PermissionError, OSError):
            pass

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
            # Some had no recoverable command line (psutil missing, access
            # denied, already gone): those still need a manual restart.
            print("    Restart manually after update: hermes gateway run")

    token = {
        "resume_needed": True,
        "profiles": profiles,
        "unmapped_pids": unmapped_pids,
        "unmapped": unmapped,
    }

    # Stop SCM-supervised gateways only after every fallible step for ordinary
    # gateways is done; from here on any error restores attempted services and
    # already-paused ordinary gateways before aborting.
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
    """Start a fresh detached gateway after update when one is installed but down.

    Called from ``_resume_windows_gateways_after_update`` for the
    ``cold_start_if_installed`` case: no gateway was running at update start,
    but an autostart entry is installed. Unlike the relaunch paths (watch an
    old PID, respawn on exit) this is a direct spawn via the same
    hidden-console + breakaway path as ``hermes gateway start``
    (``gateway_windows._spawn_detached``).

    Best-effort and idempotent: re-checks that nothing is running first so a
    concurrent start (e.g. the autostart entry firing) can't duplicate.

    A successful ``Popen`` only proves the process was created, not that it
    survived (a job object denying breakaway kills it before it logs, #84185),
    so the success line is gated on the same post-spawn liveness poll every
    other ``_spawn_detached`` caller uses (``_report_gateway_start``).
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

    # Re-check liveness right before spawning — between pause and resume the
    # autostart entry may have already brought a gateway up, or a leftover
    # process may have re-registered. Don't double-start.
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
    # Persist the PIDs this ✓ vouched for so a death AFTER the updater exits
    # (parent Job Object teardown, #91675) is reported by the next CLI
    # invocation instead of staying silent. Best-effort.
    try:
        gateway_windows._write_start_attestation(
            ready_pids, "cold-start after update"
        )
    except Exception:
        pass
    return True


def _refresh_windows_gateway_launchers() -> None:
    """Regenerate installed Windows gateway launcher scripts after update.

    The Scheduled Task / Startup-folder launchers (``gateway.cmd`` +
    ``gateway.vbs``) are written once at install time, so installs predating
    the hidden-console rework (aa2ae36c3f) kept launching via ``pythonw.exe``:
    conhost flashes (#54220/#56747) and, since #70344, a startup death with
    ``RuntimeError: sys.stderr is None`` (#71671).

    The task's /TR points at a stable script path, so rewriting in place
    retargets it without schtasks (no UAC). ``_write_task_script`` is
    idempotent — a no-op for modern installs. Best-effort: never fails the update.
    """
    from hermes_cli.update_cmd import _m
    if not _m()._is_windows():
        return
    try:
        from hermes_cli import gateway_windows

        if not gateway_windows.is_installed():
            return
        gateway_windows._write_task_script()
        print("  ✓ Refreshed Windows gateway launcher scripts")
    except Exception as exc:
        logger.debug("Could not refresh Windows gateway launchers after update: %s", exc)


def _refresh_bootstrap_cache_scripts(branch: str = "main") -> None:
    """Sync the installer's bootstrap-cache scripts from the fresh checkout.

    The Desktop GUI updater (``hermes-setup.exe``) runs
    ``$HERMES_HOME/bootstrap-cache/install-<ref>.ps1`` (or ``.sh``) for its
    repair/bootstrap stages. Installers built before the #67193 cache-refresh
    fix NEVER re-download a cached branch-ref script, so a stale
    ``install-main.ps1`` runs months-old code forever (the 2026-08-09
    incident: a cached script lacked the #81327 process-tree sweep and died on
    ``Access denied``). The binary has no self-update path.

    Overwriting the cached script for *branch* with the freshly pulled
    ``scripts/install.ps1`` / ``scripts/install.sh`` on every update turns
    that unconditional reuse into a feature. Post-#67193 installers
    re-download anyway, so for them this is a harmless pre-seed.

    Scope guards, mirroring ``install_script.rs``:

    - Only the cache key for the update-target *branch* is rewritten
      (``sanitize_ref``: non ``[A-Za-z0-9._-]`` chars become ``_``, so
      ``bb/gui`` → ``install-bb_gui.ps1``); sibling refs cache other
      branches' scripts and must not be clobbered.
    - Commit-SHA pins are immutable and never touched. ``is_valid_commit()``
      accepts **7–40** hex chars, so abbreviated pins count too; the sanitized
      *branch* must also not itself look like a pin (defense in depth).

    The .ps1 copy gets a UTF-8 BOM to match the installer's cache format
    (#67193). Best-effort: a failed refresh must never fail the update.
    """
    from hermes_cli.update_cmd import _m
    try:
        import re as _re

        cache_dir = Path(_m().get_hermes_home()) / "bootstrap-cache"
        if not cache_dir.is_dir():
            return
        # Mirror install_script.rs::sanitize_ref().
        safe_ref = _re.sub(r"[^A-Za-z0-9._-]", "_", str(branch or "main"))
        # Mirror install_script.rs::is_valid_commit(): 7-40 hex chars is an
        # immutable commit pin — abbreviated SHAs included. Never rewrite.
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
                # Match the installer's cache format: PowerShell needs the
                # UTF-8 BOM or localized/em-dash text mis-decodes (#67193).
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
    except Exception as exc:
        logger.debug("Could not refresh bootstrap-cache scripts after update: %s", exc)


def _resume_windows_gateways_after_update(token: dict | None) -> None:
    """Restart Windows profile gateways previously paused for update."""
    from hermes_cli.update_cmd import _m, _start_windows_gateway_service
    if not token or not token.get("resume_needed"):
        return
    if not _m()._is_windows():
        token["resume_needed"] = False
        return

    # Regenerate the persisted launcher scripts before respawning anything,
    # so a legacy pythonw-era Scheduled Task / Startup entry comes back on
    # current hidden-console design at the next login too.
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

    # Surface the outcome on the token (#91277 plan-vs-execution
    # reconciliation): the git-update fleet reconciliation cross-checks every
    # planned runtime against relaunched_profiles etc., which this Windows
    # pause/resume never fed, so a correctly relaunched gateway was reported
    # "unaccounted" (loud warning + exit 1). The caller merges this into the
    # shared list. A profile whose relaunch failed is deliberately left off so
    # it still surfaces as unaccounted (Windows has no watcher to recover it).
    token["relaunched_profiles"] = relaunched

    # Respawn unmapped gateways (no profile→PID-file mapping, e.g. a Scheduled
    # Task) by replaying the argv we snapshotted before force-killing them.
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

    # A truthy launch result only proves the detached watcher was created, not
    # that the respawned gateway survived: a parent Job Object denying
    # CREATE_BREAKAWAY_FROM_JOB kills it on updater teardown before it logs
    # anything, yet "✓ Restarting" was printed (#48820). Verify a stable
    # gateway exists with the same provisional-hit + confirmation-window poll
    # every other spawn path uses (#91675); all_profiles=True because the
    # resume covers the fleet.
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
        # Persist the PIDs this ✓ vouches for so a death AFTER the updater
        # exits (parent Job Object teardown, #91675) is reported by the next
        # CLI invocation instead of staying silent. Best-effort.
        try:
            gateway_windows._write_start_attestation(
                ready_pids, "post-update relaunch"
            )
        except Exception:
            pass

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
    """Resume the gateways paused for a Windows update and fold that outcome into ``outcome``.

    Feeds the pause/resume token's relaunched/restarted/pending lists into
    the same bookkeeping the systemd/launchd phase populates so the
    plan-vs-execution reconciliation never reports a healthy Windows gateway
    as unaccounted. Best-effort: must never itself abort the update.
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
        # Fold the Windows pause/resume outcome into the same bookkeeping
        # the systemd/launchd phase fills, so the #91277 reconciliation
        # doesn't flag a healthy Windows gateway as "unaccounted". Genuinely
        # failed relaunches are left out of the token so they still surface.
        # Best-effort: must never abort the update.
        try:
            for _win_profile in _windows_gateway_resume.get("relaunched_profiles") or []:
                if _win_profile not in outcome.relaunched_profiles:
                    outcome.relaunched_profiles.append(_win_profile)
        except Exception as _win_reconcile_exc:
            logger.debug(
                "Could not merge Windows relaunch outcome into fleet "
                "reconciliation bookkeeping: %s",
                _win_reconcile_exc,
            )
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

        try:
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
        except Exception:
            pass


def _clear_windows_venv_holders_or_exit(args, gateway_mode: bool, _windows_gateway_resume):
    """Windows: stop every venv-python holder we can positively identify, else exit 2.

    Rungs, in order: leftover pausable gateways -> ledger-identified orphaned
    backends -> orphaned Desktop backends -> ledger-identified manual serve
    (relaunched at exit on the same bind) -> GUI-updater hand-off leaks.
    Anything still holding the venv afterwards is refused (the sync would
    corrupt against a locked .pyd) and the paused gateways are resumed.
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
            # Remaining holders are gateways the pause machinery owns
            # (supervisor respawn in the pause→guard window, or an unmapped
            # spawn path). Stop and re-check; the post-update resume brings them back.
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
        # Positive-identity rung (runs FIRST, any update context): holders
        # the spawn ledger proves are orphaned Hermes backends — the
        # process self-registered (pid, create_time, purpose, spawner) at
        # startup and its recorded spawner is provably dead. No PPID
        # archaeology, no hand-off contract required.
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
            # Remaining holders are Desktop `serve` backends whose app is
            # GONE (Electron lost the SIGTERM race on teardown). Nothing
            # respawns an orphan, so reap the tree and re-check. Backends
            # with a live Desktop never reach here (returns None) — the app
            # would just respawn what we kill, so that path keeps refusing.
            print(
                f"  ⚠ {len(_orphan_backends)} orphaned Desktop backend "
                "process(es) still hold the venv; stopping their trees"
            )
            _m()._stop_process_trees(_orphan_backends)
            _time.sleep(1.0)
            _venv_holders = _m()._detect_venv_python_processes()
    if _venv_holders:
        # Manual serve/dashboard rung (#63206): a `hermes serve --host <ip>`
        # powering a REMOTE Desktop used to dead-end the update with exit 2.
        # Ledger identity only (spawner not alive; Desktop-owned backends
        # keep the refusal). Stop them and register an idempotent atexit
        # relaunch on the SAME host/port/profile — success or failure.
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
        # Final rung: a GUI-updater hand-off (`update --gateway --force` with
        # the update-incomplete marker) means the Desktop is contractually
        # gone and nothing legitimate respawns a `serve` backend. The
        # orphan-only reap bails on ANY live parent (mid-teardown Electron,
        # launcher→worker chain), which hung updates; in hand-off context
        # surviving backends are leaks regardless — reap by cmdline.
        _handoff = False
        try:
            _handoff = bool(getattr(args, "gateway", False)) and _m()._update_marker_path().exists()
        except Exception:
            _handoff = False
        # Fail closed: if we cannot positively verify the shim state
        # (scripts dir unresolvable, detection raised), assume a live
        # shim exists and keep refusing rather than reap.
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
