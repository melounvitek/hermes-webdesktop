"""Web UI (dashboard frontend) build: content-hash stamps, npm install/build with idle timeout, bytecode sweep.

Split out of ``hermes_cli/main.py``; every moved name is re-imported there, so
``hermes_cli.main.<name>`` keeps resolving (and monkeypatching) as before.
Names that stay in main are imported lazily inside the functions that use them
(call-time resolution keeps ``hermes_cli.main.<name>`` patches effective and
avoids an import cycle).
"""

import logging
import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time as _time

from pathlib import Path
from typing import Callable
from hermes_cli.main_tui_launch import (
    _npm_lifecycle_env, _termux_workspace_install_context, _workspace_root
)

# Log-record parity with the origin module.
logger = logging.getLogger("hermes_cli.main")

# Checkout fingerprint the bytecode cache was last validated against. Lives next
# to the checkout (NOT in HERMES_HOME): __pycache__ is per-checkout state shared
# by every profile.
_BYTECODE_FINGERPRINT_FILE = ".bytecode-fingerprint"


def _record_bytecode_fingerprint() -> None:
    """Persist the current checkout fingerprint after a bytecode sweep. Never raises."""
    from hermes_cli.main import PROJECT_ROOT, _BYTECODE_FINGERPRINT_FILE, _read_git_revision_fingerprint
    try:
        fingerprint = _read_git_revision_fingerprint(PROJECT_ROOT)
        if not fingerprint:
            return
        stamp_path = PROJECT_ROOT / _BYTECODE_FINGERPRINT_FILE
        tmp_path = stamp_path.with_name(stamp_path.name + ".tmp")
        tmp_path.write_text(fingerprint, encoding="utf-8")
        tmp_path.replace(stamp_path)
    except OSError as exc:
        logger.debug("Could not record bytecode fingerprint: %s", exc)


def _sweep_stale_bytecode_if_checkout_changed() -> None:
    """Clear ``__pycache__`` at launch when the checkout changed underneath us.

    Stale-bytecode bug class: the checkout's ``.py`` files change (git pull inside
    ``hermes update``, a manual pull, a ZIP update, a file-sync restore) while
    ``__pycache__`` keeps bytecode from the previous revision. Update-time clears
    can't close it — ``hermes update`` runs the PRE-pull updater code, and manual
    pulls never run it — so every entry point compares the checkout fingerprint
    (cheap file reads, no git subprocess) against the last-validated stamp and
    sweeps once when they diverge. Never raises.
    """
    from hermes_cli.main import PROJECT_ROOT, _BYTECODE_FINGERPRINT_FILE, _clear_bytecode_cache, _read_git_revision_fingerprint, _record_bytecode_fingerprint
    try:
        fingerprint = _read_git_revision_fingerprint(PROJECT_ROOT)
        if not fingerprint:
            return  # non-git install — the ZIP update path clears explicitly
        stamp_path = PROJECT_ROOT / _BYTECODE_FINGERPRINT_FILE
        try:
            recorded = stamp_path.read_text(encoding="utf-8").strip()
        except OSError:
            recorded = ""
        if recorded == fingerprint:
            return
        removed = _clear_bytecode_cache(PROJECT_ROOT)
        if removed:
            logger.info(
                "Checkout changed since last launch (%s -> %s): cleared %d stale __pycache__ director%s",
                recorded or "unknown", fingerprint, removed, "y" if removed == 1 else "ies",
            )
        _record_bytecode_fingerprint()
    except Exception as exc:
        logger.debug("Stale-bytecode launch sweep failed: %s", exc)


def _web_project_root(web_dir: Path) -> Path:
    """Repo root for a frontend dir (``web/`` or ``apps/<name>/``)."""
    return web_dir.parent.parent if web_dir.parent.name == "apps" else web_dir.parent


def _web_dist_dir(web_dir: Path) -> Path:
    """Vite outputs to ``hermes_cli/web_dist/`` (vite.config.ts outDir), NOT ``web/dist/``."""
    return _web_project_root(web_dir) / "hermes_cli" / "web_dist"


def _hash_source_tree(project_root: Path, tree_dir: Path) -> str:
    """SHA-256 over *tree_dir* plus the root ``package.json`` / ``package-lock.json``.

    Ignored paths (``node_modules/``, ``dist/``, ``*.pyc``, ...) are skipped via
    the repo-root ``.gitignore`` (pathspec) so build output never feeds back into
    its own staleness check. Filenames are sorted for a deterministic digest.
    """
    h = hashlib.sha256()

    def _hash_file(path: Path) -> None:
        h.update(str(path.relative_to(project_root)).encode())
        h.update(b"\0")
        try:
            with open(path, "rb") as f:
                for chunk in iter(lambda: f.read(65536), b""):
                    h.update(chunk)
        except OSError:
            pass
        h.update(b"\0")

    from pathspec import PathSpec

    gitignore = project_root / ".gitignore"
    lines = gitignore.read_text(encoding="utf-8").splitlines() if gitignore.is_file() else []
    spec = PathSpec.from_lines("gitignore", lines)

    def _ignored(path: Path) -> bool:
        return spec.match_file(str(path.relative_to(project_root)))

    for name in ("package.json", "package-lock.json"):
        p = project_root / name
        if p.is_file() and not _ignored(p):
            _hash_file(p)

    # Prune ignored directories in place so we never descend into them.
    for dirpath, dirnames, filenames in os.walk(tree_dir, topdown=True):
        dirnames[:] = [d for d in dirnames if not _ignored(Path(dirpath) / d)]
        for fn in sorted(filenames):
            fp = Path(dirpath) / fn
            if not _ignored(fp):
                _hash_file(fp)

    return h.hexdigest()


def _stamp_is_current(stamp_file: Path, current_hash: Callable[[], str], **expect) -> bool:
    """True when *stamp_file* parses, every ``expect`` key matches, and the hash matches.

    ``current_hash`` is only evaluated once the cheaper checks pass (it walks the
    source tree).
    """
    if not stamp_file.is_file():
        return False
    try:
        stamp_data = json.loads(stamp_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if not isinstance(stamp_data, dict):
        return False
    if any(stamp_data.get(k) != v for k, v in expect.items()):
        return False
    saved_hash = stamp_data.get("contentHash")
    return bool(saved_hash) and current_hash() == saved_hash


def _write_build_stamp(stamp_file: Path, label: str, current_hash: Callable[[], str], **extra) -> None:
    """Write ``{contentHash, **extra, builtAt}``; never lets stamp-writing fail a build."""
    try:
        stamp_file.parent.mkdir(parents=True, exist_ok=True)
        content_hash = current_hash()
        from datetime import datetime, timezone
        stamp_data = {"contentHash": content_hash, **extra, "builtAt": datetime.now(timezone.utc).isoformat()}
        stamp_file.write_text(json.dumps(stamp_data, indent=2) + "\n", encoding="utf-8")
    except Exception as exc:
        logger.debug("Failed to write %s build stamp: %s", label, exc)


def _web_ui_build_needed(web_dir: Path) -> bool:
    """True if the web UI dist is missing or its source content changed.

    Content hash, NOT mtime: ``git checkout`` / ``hermes update`` rewrite source
    mtimes without changing content, which made an mtime check unreliable in
    both directions.
    """
    project_root = _web_project_root(web_dir)
    dist_dir = _web_dist_dir(web_dir)
    sentinel = dist_dir / ".vite" / "manifest.json"
    if not sentinel.exists():
        sentinel = dist_dir / "index.html"
    if not sentinel.exists():
        return True
    return not _stamp_is_current(
        _web_ui_stamp_path(), lambda: _compute_web_ui_content_hash(project_root, web_dir)
    )


def _compute_web_ui_content_hash(project_root: Path, web_dir: Path) -> str:
    """SHA-256 of the web UI source tree plus root workspace config."""
    return _hash_source_tree(project_root, web_dir)


def _web_ui_stamp_path() -> Path:
    """Path of the web UI build stamp under $HERMES_HOME."""
    from hermes_constants import get_hermes_home
    return get_hermes_home() / "web-ui-build-stamp.json"


def _write_web_ui_build_stamp(project_root: Path, web_dir: Path) -> None:
    """Write the web UI build stamp after a successful build."""
    _write_build_stamp(
        _web_ui_stamp_path(), "web UI", lambda: _compute_web_ui_content_hash(project_root, web_dir)
    )


def _console_print(text: str) -> None:
    """print() that survives cp1252-style consoles (arrow/check glyphs) via errors="replace"."""
    try:
        print(text)
    except UnicodeEncodeError:
        encoding = getattr(sys.stdout, "encoding", None) or "ascii"
        print(text.encode(encoding, errors="replace").decode(encoding, errors="replace"))


def _run_with_idle_timeout(
    cmd: list[str], cwd: Path, *, idle_timeout_seconds: int = 180, indent: str = "    ",
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Run a subprocess that streams output, killing it after *idle_timeout_seconds* of silence.

    A silent, captured ``npm run build`` on a low-memory host looks like a hang and
    users reboot mid-install. Stdout is streamed, and idle output terminates the
    process with a non-zero returncode (124 if terminate raced a clean exit).
    Returns merged stdout (text) and empty stderr; never raises on idle timeout.
    """
    merged_chunks: list[str] = []
    last_output_ts = _time.monotonic()
    lock = threading.Lock()

    try:
        proc = subprocess.Popen(
            cmd, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace", bufsize=1, env=env,
        )
    except OSError as exc:
        # E.g. npm not on PATH between the which() check and now.
        return subprocess.CompletedProcess(cmd, 127, stdout="", stderr=str(exc))

    def _reader() -> None:
        nonlocal last_output_ts
        assert proc.stdout is not None
        for line in proc.stdout:
            _console_print(f"{indent}{line.rstrip()}")
            sys.stdout.flush()
            with lock:
                merged_chunks.append(line)
                last_output_ts = _time.monotonic()

    reader_thread = threading.Thread(target=_reader, daemon=True)
    reader_thread.start()

    idle_killed = False
    while True:
        try:
            rc = proc.wait(timeout=5)
            break
        except subprocess.TimeoutExpired:
            with lock:
                idle = _time.monotonic() - last_output_ts
            if idle > idle_timeout_seconds:
                idle_killed = True
                proc.terminate()
                try:
                    rc = proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    rc = proc.wait()
                break

    # Drain reader so we don't leak the stdout file descriptor.
    reader_thread.join(timeout=2)

    combined = "".join(merged_chunks)
    if idle_killed:
        combined += (
            f"\n  ⚠ Build produced no output for {idle_timeout_seconds}s — terminated.\n"
            "    Common causes: out-of-memory on a low-RAM host (WSL/container),\n"
            "    a stuck Node process, or an antivirus scan stalling I/O.\n"
        )
        if rc == 0:
            rc = 124  # GNU `timeout` convention
    return subprocess.CompletedProcess(cmd, rc, stdout=combined, stderr="")


def _nixos_build_env() -> dict[str, str] | None:
    """Extra env for native module builds on NixOS, or None when not needed.

    node-gyp's ``find-python.js`` does a bare PATH lookup for python3, which fails
    on NixOS outside a nix-shell. Tier 1: the hermes venv python3; tier 2: resolve
    via ``nix-shell`` (a self-contained Nix store binary, valid after the shell exits).
    """
    from hermes_cli.main import PROJECT_ROOT
    import re

    try:
        os_release = Path("/etc/os-release").read_text(encoding="utf-8")
    except OSError:
        return None
    if not re.search(r"^ID=nixos$", os_release, re.M) or shutil.which("python3"):
        return None

    for venv_name in ("venv", ".venv"):
        venv_python = PROJECT_ROOT / venv_name / "bin" / "python3"
        if venv_python.exists():
            return {**os.environ, "PYTHON": str(venv_python)}

    try:
        result = subprocess.run(
            ["nix-shell", "-p", "python3", "--run", "which python3"],
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=False, timeout=15,
        )
        if result.returncode == 0:
            python3_path = result.stdout.strip()
            if python3_path and Path(python3_path).exists():
                return {**os.environ, "PYTHON": python3_path}
    except Exception:
        pass  # nix-shell not available — caller will get None

    return None


def _run_npm_install_deterministic(
    npm: str, cwd: Path, *, extra_args: tuple[str, ...] = (), capture_output: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess:
    """Deterministic npm install that never mutates ``package-lock.json``.

    ``npm ci`` when a lockfile is present, else/on failure ``npm install --no-save``
    (lockfile may be out of sync on a WIP checkout; ``--no-save`` keeps the
    contract — a rewritten lockfile makes every future ``npm ci`` fail).
    ``--include=dev`` is forced: callers are frontend builds whose toolchain is in
    devDependencies, and an inherited ``NODE_ENV=production`` / ``omit=dev`` would
    silently skip them (exit 0) and the build then dies with ``tsc: not found``.
    An npm outside the root ``engines.npm`` range fails every command identically,
    so that failure gets exactly one ``maybe_repair_npm_engine`` retry.
    """
    # CI=1 no-ops unicode-animations' postinstall that animates to /dev/tty.
    run_env = _npm_lifecycle_env(env)

    def _attempt(npm_exe: str) -> subprocess.CompletedProcess:
        def _run(args: list[str]) -> subprocess.CompletedProcess:
            return _run_npm_watching_for_engine_failure(
                [npm_exe, *args, "--include=dev", *extra_args], cwd=cwd, env=run_env, capture_output=capture_output,
            )
        if (cwd / "package-lock.json").exists():
            ci_result = _run(["ci"])
            if ci_result.returncode == 0:
                return ci_result
        return _run(["install", "--no-save"])

    result = _attempt(npm)
    if result.returncode == 0:
        return result

    from hermes_cli.npm_engine import maybe_repair_npm_engine

    repaired_npm = maybe_repair_npm_engine(npm, f"{result.stdout or ''}\n{result.stderr or ''}")
    if not repaired_npm:
        return result
    # A freshly provisioned managed npm resolves `node` from PATH — put the
    # managed tree first so it finds the managed Node, not a mismatched system one.
    from hermes_constants import with_hermes_node_path

    run_env["PATH"] = with_hermes_node_path(run_env)["PATH"]
    return _attempt(repaired_npm)


def _run_npm_watching_for_engine_failure(
    cmd: list[str], *, cwd: Path, env: dict[str, str], capture_output: bool
) -> subprocess.CompletedProcess:
    """Run *cmd*, always retaining stderr so ``EBADENGINE`` stays detectable.

    ``capture_output=False`` callers stream npm's progress live; tee stderr so it
    is both forwarded as it arrives and accumulated for the engine-repair check.
    """
    if capture_output:
        return subprocess.run(
            cmd, cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
        )

    captured: list[str] = []
    with subprocess.Popen(
        cmd, cwd=cwd, env=env, stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
    ) as proc:
        if proc.stderr is not None:
            for line in proc.stderr:
                captured.append(line)
                sys.stderr.write(line)
            sys.stderr.flush()
        returncode = proc.wait()
    return subprocess.CompletedProcess(cmd, returncode, None, "".join(captured))


def _missing_web_build_tool(output: str) -> str | None:
    """The build tool a failed ``npm run build`` could not resolve (dash/bash/cmd.exe phrasings)."""
    lowered = output.lower()
    for tool in ("tsc", "vite"):
        phrases = (f"{tool}: not found", f"{tool}: command not found", f"'{tool}' is not recognized")
        if any(phrase in lowered for phrase in phrases):
            return tool
    return None


def _build_web_ui(web_dir: Path, *, fatal: bool = False) -> bool:
    """Build the web UI frontend if npm is available, serializing across processes.

    Concurrent dashboard boots used to each spawn ``npm install`` + ``vite build``
    over the same tree and starve each other. One process builds under an
    exclusive flock; the rest serve the existing dist (stale is acceptable) or,
    when none exists yet, block until the builder finishes. Staleness is checked
    once inside :func:`_do_build_web_ui`, after the lock is held.
    """
    if not (web_dir / "package.json").exists():
        return True
    try:
        import fcntl
    except ImportError:
        # Windows: no flock — fall through to the unserialized build.
        return _do_build_web_ui(web_dir, fatal=fatal)
    project_root = _web_project_root(web_dir)
    try:
        lock_file = open(project_root / ".web_ui_build.lock", "a", encoding="utf-8")
    except OSError:
        return _do_build_web_ui(web_dir, fatal=fatal)
    try:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            if (_web_dist_dir(web_dir) / "index.html").exists():
                return True  # another process is building — serve the current dist
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)  # first-ever build: wait
        return _do_build_web_ui(web_dir, fatal=fatal)
    finally:
        lock_file.close()


def _relay_npm_output(result: subprocess.CompletedProcess) -> None:
    """Print captured npm output so users can see *why* a step failed."""
    for blob in (result.stdout, result.stderr):
        if not blob:
            continue
        text = blob.decode("utf-8", errors="replace").rstrip() if isinstance(blob, bytes) else blob.rstrip()
        if text:
            _console_print(text)


def _web_npm_install_context(web_dir: Path) -> tuple[Path, tuple[str, ...]]:
    """``(cwd, workspace_args)`` for installing the web workspace's deps.

    Scoped to ``--workspace web`` so the root ``apps/*`` glob never pulls desktop
    (Electron + node-pty) into a web build; no args when ``web/`` has its own
    lockfile (``_workspace_root`` returns web_dir and ``--workspace`` would fail).
    From the workspace root this must name the SAME closure as ``hermes update``'s
    ``_update_node_dependencies()`` (ui-tui + web + root): ``npm ci`` deletes
    node_modules before reifying, so a narrower closure silently prunes what the
    update step just installed while still exiting 0. ui-tui is only named when
    present (prebuilt/partial checkouts lack it and npm fails hard otherwise).
    """
    from hermes_cli.main import _is_termux_startup_environment
    if _is_termux_startup_environment():
        return _termux_workspace_install_context(web_dir)
    npm_cwd = _workspace_root(web_dir)
    if npm_cwd == web_dir:
        return npm_cwd, ()
    args: tuple[str, ...] = ("--workspace", "web", "--include-workspace-root")
    if (npm_cwd / "ui-tui" / "package.json").exists():
        args = ("--workspace", "ui-tui", *args)
    return npm_cwd, args


def _report_web_build_failure(step: str, result: subprocess.CompletedProcess, *, fatal: bool) -> bool:
    """Print the standard ``Web UI <step> failed`` block + manual hint; returns False."""
    _console_print(f"  {'✗' if fatal else '⚠'} Web UI {step} failed" + ("" if fatal else " (hermes web will not be available)"))
    _relay_npm_output(result)
    if fatal:
        _console_print("  Run manually:  npm install --workspace web && npm run build -w web")
    return False


def _do_build_web_ui(web_dir: Path, *, fatal: bool = False) -> bool:
    """Build the web UI frontend if npm is available.

    ``fatal`` prints error guidance and returns False on failure instead of a
    soft warning (used by ``hermes web``). Returns True when the build succeeded
    or was skipped (no package.json / up to date / stale dist served as fallback).
    """
    from hermes_cli.main import _resolve_node_runtime_npm, _run_npm_install_deterministic, _run_with_idle_timeout, _web_ui_build_needed, _write_web_ui_build_stamp
    if not (web_dir / "package.json").exists() or not _web_ui_build_needed(web_dir):
        return True

    from hermes_constants import with_hermes_node_path

    npm = _resolve_node_runtime_npm()
    if not npm:
        if fatal:
            _console_print("Web UI frontend not built and npm is not available.")
            _console_print("Install Node.js, then run:  cd web && npm install && npm run build")
        return not fatal
    build_env = _npm_lifecycle_env(with_hermes_node_path())
    _console_print("→ Building web UI...")

    npm_cwd, npm_workspace_args = _web_npm_install_context(web_dir)

    def _install_web_deps(*, silent: bool) -> subprocess.CompletedProcess:
        extra = (*npm_workspace_args, "--silent", "--prefer-offline") if silent else (*npm_workspace_args, "--prefer-offline")
        return _run_npm_install_deterministic(npm, npm_cwd, extra_args=extra, env=build_env)

    def _build() -> subprocess.CompletedProcess:
        # Streamed + idle-killed (never capture_output on a long Vite build: it
        # looks identical to a hang and users reboot mid-install).
        return _run_with_idle_timeout([npm, "run", "build"], cwd=web_dir, env=build_env)

    r1 = _install_web_deps(silent=True)
    if r1.returncode != 0:
        return _report_web_build_failure("npm install", r1, fatal=fatal)
    r2 = _build()
    if r2.returncode != 0:
        # The install can exit 0 over a half-installed tree (lockfile-hash skip,
        # interrupted link step); a plain retry would keep `tsc: not found`
        # forever. Reinstall non-silently first, then one delayed retry for
        # boot-time races (antivirus scanning Node, npm cache not ready).
        missing_tool = _missing_web_build_tool((r2.stdout or "") + (r2.stderr or ""))
        if missing_tool:
            _console_print(f"  ⚠ Build could not resolve {missing_tool} — reinstalling web dependencies...")
            _install_web_deps(silent=False)
            r2 = _build()
        if r2.returncode != 0:
            _time.sleep(3)
            r2 = _build()

    if r2.returncode != 0:
        # A stale dist is far better than no UI for non-interactive callers
        # (Windows Scheduled Tasks, CI): serve it as a fallback instead of failing.
        if (_web_dist_dir(web_dir) / "index.html").exists():
            _console_print("  ⚠ Web UI build failed — serving stale dist as fallback")
            # Idle-timeout merges stderr into stdout; subprocess.run keeps them split.
            preview = ((r2.stderr or "") + (r2.stdout or "")).strip()
            if preview:
                _console_print("  Build error:\n  " + "\n  ".join(preview.splitlines()[-10:]))
            return True
        return _report_web_build_failure("build", r2, fatal=fatal)
    _console_print("  ✓ Web UI built")
    _write_web_ui_build_stamp(_web_project_root(web_dir), web_dir)
    return True
