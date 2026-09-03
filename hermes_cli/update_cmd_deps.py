"""Post-``hermes update`` dependency sync: venv preflight, editable reinstall, lazy refresh, npm/Desktop rebuilds, self-lock deferral.

Split out of ``update_cmd.py``; names are re-imported there so ``hermes_cli.update_cmd.<name>``
still resolves/monkeypatches. Origin helpers are imported lazily per function (no cycle; patches hold).
"""

import logging
from contextlib import suppress
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Optional
from hermes_constants import venv_python_path

# Log-record parity with the origin module.
logger = logging.getLogger("hermes_cli.update_cmd")

# Files defining the editable install; a pull touching none of them cannot invalidate it.
_INSTALL_DEFINING_FILES = "pyproject.toml", "setup.py", "setup.cfg", "MANIFEST.in", "uv.lock"


def _editable_install_is_current(git_cmd, cwd, pre_pull_sha: str | None) -> bool:
    """True when the pulled commits cannot have invalidated the editable install.

    ``uv pip install -e .`` always rewrites console-script shims; on Windows that rewrite is
    why ``hermes.exe`` must be quarantined (lost race = ``os error 32``), so skip it when it
    provably changes nothing. Safe because the editable finder uses a *static* module list:
    only a new top-level module/package (needing a ``pyproject.toml`` diff, like deps and
    scripts) can stale it. Fails closed: unresolvable pre-pull SHA or failed diff -> False.
    """
    if not pre_pull_sha:
        return False
    try:
        result = subprocess.run(
            git_cmd
            + ["diff", "--name-only", f"{pre_pull_sha}..HEAD", "--"]
            + list(_INSTALL_DEFINING_FILES),
            cwd=cwd,
            capture_output=True,
            text=True, encoding="utf-8", errors="replace")
    except OSError:
        return False
    return result.returncode == 0 and not result.stdout.strip()


# Modules imported on every startup. Unlike _UPDATE_CRITICAL_FILES (only parsed) these are
# *imported*, catching cross-module breakage (a name pulled from a sibling no longer exists).
_UPDATE_CRITICAL_MODULES = "hermes_cli.main", "run_agent", "model_tools", "toolsets"


def _critical_module_import_failures(
    root, *, report_runtime_errors: bool = False) -> dict[str, tuple[str, str]]:
    """Import each ``_UPDATE_CRITICAL_MODULES`` entry in a subprocess; return failures in probe order.

    Syntax validation only *parses*, so a partially-updated tree (reachable via the Windows
    ZIP copy loop) parses yet dies with ``ImportError: cannot import name``. A subprocess
    keeps the half-updated tree's import side effects out of the updater's ``sys.modules``;
    uses the venv interpreter when present since ``hermes update`` may run under another Python.
    Generic import-time exceptions are tolerated by default (may depend on local config);
    ``report_runtime_errors=True`` exposes them so two checkout states can be compared.
    """
    from hermes_cli.update_cmd import _UPDATE_CRITICAL_MODULES, _m
    from hermes_constants import FIRST_PARTY_MODULE_ROOTS

    import secrets

    marker = f"__HERMES_IMPORT_HEALTH_{secrets.token_hex(16)}__"
    probe = (
        "import importlib, json, sys\n"
        "failures = []\n"
        "for name in %r:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except ModuleNotFoundError as exc:\n"
        # A missing *third-party* module means deps aren't installed, not a skewed checkout;
        # only our own packages count. Roots come from hermes_constants so the user hint can't drift.
        "        missing = (getattr(exc, 'name', '') or '').split('.')[0]\n"
        "        if missing in %r or missing.startswith('hermes_') or %r:\n"
        "            failures.append((name, type(exc).__name__, str(exc)))\n"
        "    except ImportError as exc:\n"
        "        failures.append((name, type(exc).__name__, str(exc)))\n"
        "    except Exception as exc:\n"
        "        if %r:\n"
        "            failures.append((name, type(exc).__name__, str(exc)))\n"
        "    except BaseException as exc:\n"
        "        failures.append((name, type(exc).__name__, str(exc)))\n"
        "sys.stdout.write('\\n%s' + json.dumps(failures))\n"
        % (
            _UPDATE_CRITICAL_MODULES,
            tuple(sorted(FIRST_PARTY_MODULE_ROOTS)),
            report_runtime_errors,
            report_runtime_errors,
            marker))
    try:
        interpreter = sys.executable
        with suppress(Exception):
            venv_python = venv_python_path(Path(root) / "venv", windows=_m()._is_windows())
            if venv_python.exists():
                interpreter = str(venv_python)
        result = subprocess.run(
            [interpreter, "-c", probe], cwd=str(root), capture_output=True, text=True,
            encoding="utf-8", errors="replace", timeout=120,
        )
    except subprocess.TimeoutExpired:
        return _probe_failure("TimeoutExpired", "timed out before reporting import health")
    except (OSError, subprocess.SubprocessError):
        # Can't run the probe — don't block the update on our own tooling.
        return {}
    output = result.stdout or ""
    if marker not in output:
        return _probe_failure(
            "ProbeTerminated",
            f"terminated before reporting import health (exit code {result.returncode})")
    try:
        failures = json.loads(output.rsplit(marker, 1)[1])
        if not isinstance(failures, list) or any(
            not isinstance(item, list)
            or len(item) != 3
            or not all(isinstance(value, str) for value in item)
            for item in failures):
            raise ValueError("invalid import-health payload")
        return {str(module): (str(kind), str(detail)) for module, kind, detail in failures}
    except (TypeError, ValueError):
        return _probe_failure("MalformedPayload", "reported malformed import health data")


def _probe_failure(kind: str, detail: str) -> dict[str, tuple[str, str]]:
    """Failure row for the probe itself (as opposed to a module it imported)."""
    return {"critical-module probe": (kind, detail)}


def _validate_critical_modules_import(
    root, *, report_runtime_errors: bool = False) -> tuple[bool, str | None, str | None]:
    """Return the first critical-module import failure, if any."""
    failures = _critical_module_import_failures(root, report_runtime_errors=report_runtime_errors)
    if failures:
        module = next(iter(failures))
        return False, module, failures[module][1]
    return True, None, None


def _npm_bin_exists(bin_dir: Path, name: str) -> bool:
    """True when an npm bin shim for *name* exists (POSIX or Windows)."""
    return any(
        (bin_dir / candidate).exists()
        for candidate in (name, f"{name}.cmd", f"{name}.ps1", f"{name}.exe"))


def _web_build_toolchain_ready(*roots: Path) -> bool:
    """True when ``tsc`` and ``vite`` shims are reachable from any of *roots*.
    Callers must pass every root the build would search, or a healthy tree reads as broken."""
    bin_dirs = [
        bin_dir
        for bin_dir in (root / "node_modules" / ".bin" for root in roots)
        if bin_dir.is_dir()]
    return bool(bin_dirs) and all(
        any(_npm_bin_exists(bin_dir, tool) for bin_dir in bin_dirs) for tool in ("tsc", "vite"))


def _web_toolchain_roots(web_dir: Path) -> tuple[Path, ...]:
    """Roots whose ``node_modules/.bin`` can satisfy the web build: ``npm run build`` searches the
    package and each ancestor, so hoisted and package-local shims are equally valid."""
    return (web_dir, web_dir.parent)


def _ensure_venv_pip(pip_cmd: list, python_exe: str) -> None:
    """Bootstrap pip back into the venv via ensurepip when ``pip --version`` fails
    (some environments lose it); call before the editable install."""
    from hermes_cli.update_cmd import _m
    try:
        subprocess.run(
            pip_cmd + ["--version"], cwd=_m().PROJECT_ROOT, check=True, capture_output=True)
    except subprocess.CalledProcessError:
        subprocess.run(
            [python_exe, "-m", "ensurepip", "--upgrade", "--default-pip"],
            cwd=_m().PROJECT_ROOT,
            check=True)


def _upgrade_pip_before_lazy_refresh(
    install_cmd_prefix: list[str], *, env: dict[str, str] | None = None) -> None:
    """Upgrade pip before lazy refreshes: older pip can fail setuptools source builds and
    leave a partially-written venv. Never raises."""
    from hermes_cli.update_cmd import _m
    try:
        _m()._run_package_only_install(
            install_cmd_prefix + ["install", "--upgrade", "pip"], env=env)
    except subprocess.CalledProcessError as exc:
        logger.debug("pip upgrade before lazy refresh failed: %s", exc)


def _capture_active_lazy_features() -> list[str]:
    """Snapshot active lazy backends before a managed runtime is replaced."""
    try:
        from tools import lazy_deps

        return lazy_deps.active_features()
    except Exception as exc:
        logger.debug("Could not snapshot active lazy features: %s", exc)
        return []


def _capture_active_tool_dependencies() -> list[str]:
    """Snapshot Python dependencies installed explicitly through ``hermes tools``."""
    try:
        from hermes_cli import tools_config

        return tools_config.active_restorable_python_tool_dependencies()
    except Exception as exc:
        logger.debug("Could not snapshot active Hermes Tools dependencies: %s", exc)
        return []


def _restore_active_tool_dependencies(
    dependencies: list[str], install_cmd_prefix: list[str], *, env: dict[str, str] | None = None
) -> None:
    """Restore allowlisted ``hermes tools`` dependencies (from a pre-rebuild probe) into a rebuilt venv.
    Never raises: a failed optional tool must not block the update, but must be reported."""
    from hermes_cli.update_cmd import _m
    if not dependencies:
        return

    try:
        from hermes_cli import tools_config
    except Exception as exc:
        logger.debug("Hermes Tools dependency restore skipped (import failed): %s", exc)
        return

    target_python = _m()._resolve_install_target_python(install_cmd_prefix, env)
    missing: list[tuple[str, tuple[str, ...]]] = []
    for name in dependencies:
        spec = tools_config.restorable_python_tool_dependency(name)
        if spec is None:
            continue
        module_name, install_args = spec
        if target_python is not None:
            try:
                probe = subprocess.run(
                    [
                        str(target_python),
                        "-c",
                        "import importlib.util,sys; "
                        "raise SystemExit(0 if importlib.util.find_spec(sys.argv[1]) else 1)",
                        module_name],
                    capture_output=True,
                    env=env,
                    check=False)
                if probe.returncode == 0:
                    continue
            except (subprocess.SubprocessError, OSError):
                # Indeterminate probe: safer to repair than assume it survived.
                pass
        missing.append((name, install_args))

    if not missing:
        return

    print()
    print(f"→ Restoring {len(missing)} Hermes Tools dependency set(s)...")
    restored: list[str] = []
    failed: list[tuple[str, str]] = []
    for name, install_args in missing:
        try:
            _m()._run_package_only_install(
                install_cmd_prefix + ["install", *install_args, "--quiet"], env=env)
            restored.append(name)
        except Exception as exc:
            # Best-effort: surface failures without aborting the update.
            failed.append((name, str(exc)))

    if restored:
        print(f"  ✓ {len(restored)} restored: {', '.join(restored)}")
    for name, reason in failed:
        print(f"  ⚠ {name} failed to restore: {_clip(reason)}")


def _clip(reason: str, limit: int = 200) -> str:
    """Bound a failure reason for the one-line report."""
    return reason if len(reason) <= limit else reason[:limit] + "..."


def _refresh_active_lazy_features(
    install_cmd_prefix: list[str] | None = None,
    *,
    env: dict[str, str] | None = None,
    features: list[str] | None = None) -> bool:
    """Refresh previously-activated lazy backends (cold ones untouched) after a code update.

    The core install never touches ``lazy_deps`` backends, so a bumped :data:`LAZY_DEPS` pin
    would otherwise leave them stale forever. Returns True when the venv is safe to use
    (refreshed, nothing active, or import repair succeeded); False when a failed lazy install
    left broken core imports repair couldn't fix. Never raises.
    """
    from hermes_cli.update_cmd import _m
    try:
        from tools import lazy_deps
    except Exception as exc:
        logger.debug("Lazy refresh skipped (import failed): %s", exc)
        return True

    active = features
    if active is None:
        try:
            active = lazy_deps.active_features()
        except Exception as exc:
            logger.debug("Lazy refresh skipped (active_features failed): %s", exc)
            return True
    if not active:
        return True

    print()
    print(f"→ Refreshing {len(active)} active lazy backend(s)...")

    unexpected_failure = False
    try:
        if features is None:
            results = lazy_deps.refresh_active_features(prompt=False)
        else:
            results = lazy_deps.restore_features(active)
    except Exception as exc:
        # refresh_active_features is never-raise by contract; defend anyway.
        print(f"  ⚠ Lazy refresh failed unexpectedly: {exc}")
        results = {}
        unexpected_failure = True

    refreshed = [f for f, s in results.items() if s in {"refreshed", "restored"}]
    current = [f for f, s in results.items() if s == "current"]
    failed = [(f, s) for f, s in results.items() if s.startswith("failed:")]
    skipped = [(f, s) for f, s in results.items() if s.startswith("skipped:")]

    if refreshed:
        print(f"  ↑ {len(refreshed)} refreshed: {', '.join(refreshed)}")
    if current:
        print(f"  ✓ {len(current)} already current")
    if skipped:
        # Usually security.allow_lazy_installs=false; informational, not an error.
        names = ", ".join(f for f, _ in skipped)
        reason = skipped[0][1].split(": ", 1)[-1]
        print(f"  · {len(skipped)} skipped ({reason}): {names}")

    if not failed and not unexpected_failure:
        return True

    for feature, status in failed:
        print(f"  ⚠ {feature} failed to refresh: {_clip(status.split(': ', 1)[-1])}")

    if install_cmd_prefix is None:
        print("  ⚠ Lazy refresh failed; rerun `hermes update` once resolved.")
        return False

    # Import-based recovery: metadata-only verifiers miss dist-info intact but import files
    # wiped. Unavailable probes are indeterminate, not healthy — keep the lazy marker.
    status = _m()._repair_venv_via_import_probes(install_cmd_prefix, env=env)
    if status == "repaired":
        print("  Lazy backend(s) keep their previous version until refresh succeeds.")
        return True
    if status == "healthy":
        print("  Lazy backend(s) keep their previous version; probed packages look intact.")
        print("  Rerun `hermes update` once the upstream issue is resolved.")
        return True
    if status == "indeterminate":
        print("  ⚠ Leaving `.lazy-refresh-incomplete` until import probes can confirm health.")
    return False


def _refresh_active_memory_provider_dependencies() -> None:
    """Refresh pip deps for the configured external memory provider.

    Bridge packages live in each provider's ``plugin.yaml``, not Hermes extras or ``LAZY_DEPS``,
    so the core reinstall can strip/downgrade them; re-run the ACTIVE provider's install after
    core install and lazy refresh so its writes land last. Never raises.
    """
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
    except Exception as exc:
        logger.debug("Memory provider refresh skipped (config load failed): %s", exc)
        return

    provider = ""
    if isinstance(cfg, dict):
        memory_cfg = cfg.get("memory")
        if isinstance(memory_cfg, dict):
            if memory_cfg.get("enabled") is False:
                return
            provider = str(memory_cfg.get("provider") or "").strip()

    # "default"/empty is the built-in file store — no pip deps.
    if not provider or provider in {"default", "builtin", "none"}:
        return

    try:
        from hermes_cli.memory_setup import _install_dependencies
    except Exception as exc:
        logger.debug("Memory provider refresh skipped (import failed): %s", exc)
        return

    print()
    print(f"→ Refreshing active memory provider dependencies ({provider})...")

    try:
        _install_dependencies(provider, force=True)
    except Exception as exc:
        print(f"  ⚠ {provider} dependencies failed to refresh: {exc}")


def _is_android_python() -> bool:
    from hermes_cli.update_cmd import _m
    return _m().sys.platform == "android"


def _install_psutil_android_compat(
    install_cmd_prefix: list[str], *, env: dict[str, str] | None = None) -> None:
    """Install psutil on Android by patching its platform detection: setup gates Linux sources on
    ``sys.platform.startswith('linux')`` but Termux reports ``'android'`` though the Linux path
    compiles fine. Only this attempt's build tree is patched. Stopgap until psutil PR 2762 ships."""
    from hermes_cli.update_cmd import _m
    import tempfile
    import urllib.request
    from hermes_cli.psutil_android import PSUTIL_URL, prepare_patched_psutil_sdist

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        archive = tmp_path / "psutil.tar.gz"
        urllib.request.urlretrieve(PSUTIL_URL, archive)
        src_root = prepare_patched_psutil_sdist(archive, tmp_path)

        _m()._run_install_with_heartbeat(
            install_cmd_prefix + ["install", "--no-build-isolation", str(src_root)], env=env)


def _ensure_uv_for_termux(pip_cmd: list[str]) -> str | None:
    """Best-effort uv bootstrap on Termux (official installer may fail: glibc vs bionic). Prefer a
    PATH uv; else wheel-only ``pip install uv`` so the Rust crate is never source-built."""
    from hermes_cli.update_cmd import _m
    from hermes_cli.managed_uv import resolve_uv

    existing = resolve_uv()
    if existing:
        return existing
    if not _m()._is_termux_env():
        return None
    # Termux-packaged uv is on PATH but not the managed bin dir, so resolve_uv() misses it;
    # prefer it over pip, which has no Android wheel and would source-build on a small device.
    system_uv = shutil.which("uv")
    if system_uv:
        return system_uv
    with suppress(Exception):
        print("  → Termux detected: trying to install uv for faster dependency updates...")
        result = subprocess.run(
            pip_cmd + ["install", "uv", "--only-binary", ":all:"],
            cwd=_m().PROJECT_ROOT,
            check=False)
        if result.returncode != 0:
            return None
    return resolve_uv() or shutil.which("uv")


def _npm_manifest_paths() -> tuple[Path, ...]:
    """Manifests whose changes must defeat the update-skip.

    The lockfile alone isn't enough: a package.json can be edited without running npm, and
    `hermes update` is the step expected to sync node_modules. Workspaces come from the root
    `workspaces` globs so a new one can't escape the key; every workspace manifest counts
    (desktop too) because the single lockfile spans the whole graph. Falls back to root
    manifests only if package.json is unreadable.
    """
    from hermes_cli.update_cmd import _m
    root_pkg = _m().PROJECT_ROOT / "package.json"
    paths = [_m().PROJECT_ROOT / "package-lock.json", root_pkg]
    with suppress(OSError, json.JSONDecodeError, TypeError):
        workspaces = json.loads(root_pkg.read_text(encoding="utf-8")).get("workspaces", [])
        if isinstance(workspaces, dict):  # legacy {"packages": [...]} form
            workspaces = workspaces.get("packages", [])
        for pattern in workspaces:
            for match in sorted(_m().PROJECT_ROOT.glob(str(pattern))):
                manifest = match / "package.json"
                if manifest.is_file():
                    paths.append(manifest)
    return tuple(paths)


def _npm_manifests_digest() -> str | None:
    """Combined sha256 over lockfile + all workspace package.json; None when lockfile missing (never skip)."""
    from hermes_cli.update_cmd import _m
    if not (_m().PROJECT_ROOT / "package-lock.json").exists():
        return None
    h = hashlib.sha256()
    for p in _npm_manifest_paths():
        h.update(str(p.relative_to(_m().PROJECT_ROOT)).encode())
        try:
            h.update(p.read_bytes())
        except OSError:
            h.update(b"<missing>")
    return h.hexdigest()


def _npm_lockfile_changed(hermes_root: Path) -> bool:
    from hermes_cli.update_cmd import _m
    current = _npm_manifests_digest()
    if current is None:
        return True
    # Matching hash but no node_modules: cache was recorded by another checkout.
    if not (_m().PROJECT_ROOT / "node_modules").is_dir():
        return True
    # Never skip when the web toolchain never landed, or later updates build on a half-installed tree.
    web_dir = _m().PROJECT_ROOT / "web"
    if (web_dir / "package.json").is_file() and not _web_build_toolchain_ready(
        *_web_toolchain_roots(web_dir)):
        return True
    try:
        cache_file = _npm_lock_cache_file(hermes_root)
        if not cache_file.exists():
            return True
        return cache_file.read_text(encoding="utf-8").strip() != current
    except OSError:
        return True


def _npm_lock_cache_file(hermes_root: Path) -> Path:
    """Per-checkout cache path: keyed by PROJECT_ROOT so parallel worktrees don't collide."""
    from hermes_cli.update_cmd import _m
    cache_key = hashlib.sha256(str(_m().PROJECT_ROOT).encode()).hexdigest()[:12]
    return hermes_root / f".npm_lock_hash_{cache_key}"


def _record_npm_lockfile_hash(hermes_root: Path) -> None:
    digest = _npm_manifests_digest()
    if digest is None:
        return
    try:
        _npm_lock_cache_file(hermes_root).write_text(digest, encoding="utf-8")
    except OSError:
        logger.debug("Could not write npm lockfile hash cache")


def _repair_node_deps_on_current_checkout(
    print_completion,
    *,
    assume_yes: bool = False,
    gateway_mode: bool = False,
    pre_update_snapshot_id: str | None = None,
    completion_message: str = "✓ Already up to date!",
    had_desktop_app_before_update: bool = False) -> bool:
    """Repair Node deps on the ``commit_count == 0`` path.

    A current checkout doesn't imply healthy Node deps (a failed npm install says "re-run
    hermes update" but the early return skipped the refresh). ``_update_node_dependencies``
    self-gates on the hash recorded only after a SUCCESSFUL install, so this is a cheap
    no-op when healthy and a real repair otherwise.
    """
    from hermes_cli.update_cmd import (
        _check_and_apply_config_migration,
        _m,
        _rebuild_desktop_after_update,
        _update_node_dependencies)
    node_failures = _update_node_dependencies()
    if node_failures:
        print(f"  ⚠ Node.js refresh failed for: {', '.join(node_failures)}")
        print("    Fix npm and re-run `hermes update`.")
        print_completion("⚠ Checkout is current, but Node.js dependencies could not be repaired.")
        return False
    # Pair with the web build like every other call site; it staleness-checks internally.
    _m()._build_web_ui(_m().PROJECT_ROOT / "web")
    _check_and_apply_config_migration(
        assume_yes=assume_yes,
        gateway_mode=gateway_mode,
        pre_update_snapshot_id=pre_update_snapshot_id)
    # A current checkout can still owe a Desktop rebuild (e.g. the Windows hand-off child
    # never reaches the commits-pulled rebuild). Self-gates on the build stamp.
    if not _rebuild_desktop_after_update(
        _m().PROJECT_ROOT / "apps" / "desktop",
        had_desktop_app_before_update=had_desktop_app_before_update):
        # Retry hint already printed; withhold success rather than claim completion.
        print_completion(
            "⚠ Update partially complete — the desktop app was not rebuilt "
            "and is still on the previous build.")
        return False
    return bool(print_completion(completion_message))


def _update_node_dependencies() -> list[str]:
    """Refresh Node deps for ui-tui and web. Returns labels whose npm install failed (empty on
    success) so the caller reports a partial update instead of ``Update complete!``."""
    from hermes_cli.update_cmd import _m
    if not (_m().PROJECT_ROOT / "package.json").exists():
        return []

    npm = _m()._resolve_node_runtime_npm()
    if not npm:
        # Only a Windows npm reachable from WSL: flag loudly — skipping silently leaves
        # deps stale, running it would corrupt the tree.
        from hermes_constants import is_wsl

        path_npm = shutil.which("npm")
        if is_wsl() and path_npm and _m()._is_windows_npm_path(path_npm):
            print("→ Updating Node.js dependencies...")
            print("  ⚠ Skipped: only a Windows npm is reachable from this WSL shell.")
            print("    Install Node.js inside the WSL distro (nvm, or your distro's")
            print("    package manager), then re-run `hermes update`.")
            has_workspace = any(
                (_m().PROJECT_ROOT / ws / "package.json").exists() for ws in ("ui-tui", "web"))
            return ["ui-tui, web workspaces"] if has_workspace else []
        return []

    from hermes_constants import get_default_hermes_root

    # node_modules is shared by every profile on this checkout: one per-checkout cache.
    shared_hermes_root = get_default_hermes_root()

    # Best-effort npx cache warm before the lockfile-unchanged early return. Can block
    # ~11s on a cold cache — print first so it doesn't look like a hang.
    print("→ Warming npx cache for agent-browser...")
    with suppress(Exception):
        from tools.browser_tool import warm_agent_browser_npx_cache
        warm_agent_browser_npx_cache()

    if not _m()._npm_lockfile_changed(shared_hermes_root):
        logger.info("npm lockfile unchanged, skipping npm install")
        return []

    # Root package.json has no deps of its own, so a workspace-scoped install prunes nothing
    # root-only. apps/desktop is deliberately never named: its Electron devDependency has a
    # ~200MB postinstall, so desktop deps install on demand (see _desktop_build_needed).
    print("→ Updating Node.js dependencies...")

    def _partial_update_failure(*labels: str) -> list[str]:
        print()
        print("  ⚠ Node.js dependency refresh did not complete cleanly; the")
        print("    installation may be in a mixed state (updated code, stale Node")
        print("    deps). Fix npm and re-run `hermes update`.")
        return list(labels)

    install_args = [
        "--no-fund", "--no-audit", "--prefer-offline", "--progress=false",
        "--workspace", "ui-tui", "--workspace", "web",
        # Root devDependencies (shared ESLint config) would otherwise be pruned by the
        # scoped install; apps/desktop stays excluded since it is never named above.
        "--include-workspace-root"]

    from hermes_constants import with_hermes_node_path

    nixos_env = with_hermes_node_path(_m()._nixos_build_env())

    # capture_output=False is deliberate: postinstall scripts print download progress and
    # capturing makes a long download look hung.
    result = _m()._run_npm_install_deterministic(
        npm, _m().PROJECT_ROOT, extra_args=tuple(install_args), capture_output=False, env=nixos_env)
    if result.returncode == 0:
        _record_npm_lockfile_hash(shared_hermes_root)
        print("  ✓ ui-tui, web workspaces installed (desktop skipped)")
        return []
    print("  ⚠ npm install failed")
    stderr = (result.stderr or "").strip()
    if stderr:
        print(f"    {stderr.splitlines()[-1]}")
    return _partial_update_failure("ui-tui, web workspaces")


def _venv_core_imports_healthy() -> tuple[bool, str]:
    """Probe the venv (in ITS interpreter — updater may run under another Python) for core imports.

    Catches a half-updated venv where the checkout is current but a dependency sync died
    partway; otherwise "Already up to date!" never re-syncs. Returns ``(healthy, detail)``.
    Never raises; unknown states report healthy so a probe failure can't force reinstalls.
    """
    from hermes_cli.update_cmd import _m
    venv_dir = _m().PROJECT_ROOT / "venv"
    venv_python = venv_python_path(venv_dir, windows=_m()._is_windows())
    if not venv_python.exists():
        # No venv: normal for a dev checkout (healthy), but on a MANAGED install (bootstrap
        # stamp or `.update-incomplete`) the venv IS the install — absence means an interrupted repair.
        managed_markers = (
            _m().PROJECT_ROOT / ".hermes-bootstrap-complete", _m()._update_marker_path())
        if any(m.exists() for m in managed_markers):
            return False, f"venv python missing ({venv_python})"
        return True, ""

    # Import (not just metadata): dist-info can be intact with modules missing after an
    # interrupted uninstall/install.
    check = (
        "import importlib\n"
        "mods = ['fastapi', 'uvicorn', 'pydantic', 'openai', 'yaml']\n"
        "missing = []\n"
        "for m in mods:\n"
        "    try: importlib.import_module(m)\n"
        "    except Exception as e: missing.append(f'{m}: {e}')\n"
        "print('\\n'.join(missing))\n")
    try:
        result = subprocess.run(
            [str(venv_python), "-c", check], capture_output=True, text=True, encoding="utf-8",
            errors="replace", timeout=60, cwd=_m().PROJECT_ROOT,
        )
    except Exception as exc:
        logger.debug("venv health probe failed to run: %s", exc)
        return True, ""

    missing = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    if result.returncode != 0 and not missing:
        # Interpreter itself is broken — that IS unhealthy.
        detail = (result.stderr or "").strip().splitlines()
        return False, detail[0] if detail else "venv python failed to run"
    if missing:
        return False, "; ".join(missing[:4])
    return True, ""


# Native extensions that pin venv files once imported: if the updater holds one, Windows blocks
# REPLACE on the mapped ``.pyd`` and the sync dies with ``os error 5`` mid-reinstall. PyYAML's
# ``_yaml`` is loaded by every CLI process, so the guard must be HONEST (an always-firing
# preflight bricked the flow it protected): (1) fire only when the sync would actually
# REWRITE the dist (``_dependency_sync_would_rewrite``); (2) run AFTER the code swap, right
# before the venv rewrite, so a deferral leaves new code with only the install pending.
# Keys are ``sys.modules`` prefixes; values are ``(display name, PyPI dist)``.
_SELF_LOCKING_NATIVE_MODULES: dict[str, tuple[str, str]] = {
    "cryptography.hazmat.bindings._rust": ("cryptography (_rust.pyd)", "cryptography"),
    "yaml._yaml": ("PyYAML (_yaml.pyd)", "pyyaml")}


def _dependency_sync_would_rewrite(dist_name: str) -> bool | None:
    """Whether the ``.[all]`` install would replace *dist_name*'s files, judged against every
    applicable pin in on-disk ``pyproject.toml`` (base + extras). False: all pins satisfied;
    True: pin unsatisfied or dist missing; None: undeterminable. Never raises. Callers treat
    None as fail-OPEN — PyYAML is in every process, so deferring on uncertainty always fires."""
    from hermes_cli.update_cmd import _m
    try:
        from importlib import metadata as _ilmd

        installed = _ilmd.version(dist_name)
    except Exception:
        return True  # not installed → the sync will definitely install it
    try:
        import tomllib

        from packaging.requirements import Requirement
        from packaging.utils import canonicalize_name
        from packaging.version import Version

        pyproject = _m().PROJECT_ROOT / "pyproject.toml"
        data = tomllib.loads(pyproject.read_text(encoding="utf-8"))
        project = data.get("project") or {}
        req_strings: list[str] = list(project.get("dependencies") or [])
        for extra_reqs in (project.get("optional-dependencies") or {}).values():
            req_strings.extend(extra_reqs or [])

        target = canonicalize_name(dist_name)
        installed_v = Version(installed)
        saw_pin = False
        for req_str in req_strings:
            try:
                req = Requirement(req_str)
            except Exception:
                continue
            if canonicalize_name(req.name) != target:
                continue
            if req.marker is not None and not req.marker.evaluate():
                continue
            saw_pin = True
            if installed_v not in req.specifier:
                return True
        if saw_pin:
            return False
        # Not pinned in pyproject: the resolver may still move it as a transitive — unknown.
        return None
    except Exception:
        return None


def _detect_self_loaded_native_modules() -> list[str]:
    """Display names of native venv extensions loaded into THIS process that the sync would rewrite.
    Empty off Windows (POSIX keeps an unlinked inode usable). Modules whose installed version
    already satisfies the pins are NOT reported — no swap at risk. Never raises."""
    from hermes_cli.update_cmd import _m
    if not _m()._is_windows():
        return []
    # Defer ONLY on a CONFIRMED rewrite; unknown fails OPEN (PyYAML is in every process, so
    # unknown-as-at-risk always fires). A missed deferral only yields the mid-sync os error 5
    # that marker recovery already handles — far less harmful than an update that never runs.
    return sorted({
        display for prefix, (display, dist) in _SELF_LOCKING_NATIVE_MODULES.items()
        if prefix in sys.modules and _m()._dependency_sync_would_rewrite(dist) is True})


def _abort_dependency_sync_if_self_locked(gateway_resume=None) -> None:
    """Defer the venv rewrite when THIS process holds something it must replace (runs after the
    code swap, so a deferral leaves NEW code with only the install pending). Two hazards:
    a mapped ``.pyd`` -> exit 2, next launch's marker recovery finishes; the ``hermes.exe`` shim
    we run from -> every launch is the shim so the marker would defer forever: hand the
    install to a child under the venv interpreter and exit 0."""
    from hermes_cli.update_cmd import _m
    locked = _m()._detect_self_loaded_native_modules()
    if locked:
        _m()._defer_update_for_self_lock(locked)
        exit_code = 2
    elif _m()._reexec_dependency_sync_off_windows_shim():
        exit_code = 0
    else:
        return
    if gateway_resume is not None:
        _m()._resume_windows_gateways_after_update(gateway_resume)
    sys.exit(exit_code)


def _defer_update_for_self_lock(loaded: list[str]) -> None:
    """Bail out before the sync when the updater holds a lock: the install can't win from inside
    the locked process (killing threads won't unmap the image), so drop the update-incomplete
    marker (next fresh launch completes it), explain, and let the caller exit 2."""
    from hermes_cli.update_cmd import _m
    print("✗ This updater process has already loaded native venv modules that")
    print("  the dependency sync must replace:")
    for name in loaded:
        print(f"    {name}")
    print()
    print("  On Windows a mapped extension cannot be replaced by the process")
    print("  holding it. The code update has been applied; only the dependency")
    print("  sync has been deferred: the next `hermes` launch will complete it")
    print("  in a fresh process before anything imports these modules.")
    _m()._write_update_incomplete_marker()


def _desktop_app_present(desktop_dir: Path) -> bool:
    """Return whether a packaged or source Desktop build exists."""
    from hermes_cli.update_cmd import _m
    return (
        _m()._desktop_packaged_executable(desktop_dir) is not None
        or _m()._desktop_dist_exists(desktop_dir))


def _rebuild_desktop_after_update(
    desktop_dir: Path, *, had_desktop_app_before_update: bool) -> bool:
    """Rebuild an installed Desktop app when its source or artifact changed. Returns ``False``
    only when a rebuild was attempted and failed (caller withholds ``✓ Update complete!`` and
    writes a failing ``.update_exit_code`` in gateway mode); every other outcome is ``True``."""
    from hermes_cli.update_cmd import _m
    # The release tree is git-ignored and can vanish mid-update; pre-update presence suffices.
    # Never make people who never used Desktop pay for an Electron build.
    has_desktop_app = had_desktop_app_before_update or _desktop_app_present(desktop_dir)
    if not (
        (desktop_dir / "package.json").exists()
        and _m()._resolve_node_runtime_npm()
        and has_desktop_app):
        return True

    print("→ Checking if desktop app needs rebuilding...")
    # Check the content-hash stamp IN-PROCESS first (the subprocess spends ~1-3 s importing the
    # CLI to reach the same check). Update never passes --source, so source_mode=False.
    # Any pre-check error falls through to the subprocess.
    try:
        skip_desktop_build = not _m()._desktop_build_needed(
            desktop_dir, _m().PROJECT_ROOT, source_mode=False)
    except Exception:
        skip_desktop_build = False
    if skip_desktop_build:
        print("  ✓ Desktop app up to date")
        return True

    desktop_build_cmd = [sys.executable, "-m", "hermes_cli.main", "desktop", "--build-only"]
    # Capture the loud build output into update.log; retry once on failure (still-settling
    # rebuild window), then surface the tail. Put Hermes-managed Node on PATH: the desktop
    # updater chain loses shell PATH customizations, so a bare-PATH child hits `node: not found`.
    from hermes_constants import with_hermes_node_path

    build_env = with_hermes_node_path()
    build_result = _m()._run_logged_subprocess(
        desktop_build_cmd, cwd=_m().PROJECT_ROOT, env=build_env)
    if build_result.returncode != 0:
        build_result = _m()._run_logged_subprocess(
            desktop_build_cmd, cwd=_m().PROJECT_ROOT, env=build_env)
    if build_result.returncode != 0:
        print("  ⚠ Desktop build failed (run `hermes desktop` to retry)")
        tail = "\n".join((build_result.stdout or "").strip().splitlines()[-15:])
        if tail:
            print(tail)
        from hermes_constants import display_hermes_home as _dhh

        print(f"  Full build log: {_dhh()}/logs/update.log")
        return False
    print("  ✓ Desktop app up to date")
    return True


def _path_uid(path) -> Optional[int]:
    """Owner uid of ``path`` (``None`` when unreadable). Separate seam so tests can simulate
    root-owned files without chown. Never raises."""
    try:
        return os.stat(path, follow_symlinks=False).st_uid
    except OSError:
        return None


def _venv_foreign_owned_paths(venv_root, limit: int = 5) -> list:
    """Bounded scan for venv entries not owned by the current user; returns up to ``limit``
    ``(path_str, uid)`` tuples.

    A venv touched by ``sudo pip``/``sudo hermes`` has root-owned files, and a later normal
    update dies mid-mutation with ``venv/bin/hermes`` already deleted — never mutate a venv we
    can't safely mutate. Deliberately BOUNDED (venv root, ``venv/bin``, top-level of first
    site-packages, children of each ``*.dist-info``; ~2000 stats max). POSIX-only: ``[]`` on
    Windows and as root. Swallows per-entry ``OSError``; ``[]`` on any structural surprise —
    must NEVER raise or add noticeable latency.
    """
    from hermes_cli.update_cmd import _path_uid
    try:
        if not hasattr(os, "geteuid"):
            return []  # windows-footgun: ok — POSIX ownership concept only
        euid = os.geteuid()  # windows-footgun: ok — guarded by hasattr above
        if euid == 0:
            return []  # root can rewrite anything; nothing to refuse

        venv_root = Path(venv_root)
        budget = 2000  # max stat() calls — hard bound on preflight cost
        foreign: list = []

        def _check(p) -> bool:
            """stat one path; True while scan should continue."""
            nonlocal budget
            if budget <= 0 or len(foreign) >= limit:
                return False
            budget -= 1
            uid = _path_uid(p)
            if uid is not None and uid != euid:
                foreign.append((str(p), uid))
            return budget > 0 and len(foreign) < limit

        def _scan_dir(d, recurse_dist_info: bool = False) -> None:
            try:
                entries = list(os.scandir(d))
            except OSError:
                return
            for entry in entries:
                if not _check(entry.path):
                    return
                if recurse_dist_info and entry.name.endswith(".dist-info"):
                    try:
                        children = list(os.scandir(entry.path))
                    except OSError:
                        continue
                    for child in children:
                        if not _check(child.path):
                            return

        if not _check(venv_root):
            return foreign[:limit]
        _scan_dir(venv_root / "bin")

        # First lib/python*/site-packages (POSIX venv layout).
        site_packages = next(iter(sorted(venv_root.glob("lib/python*/site-packages"))), None)
        if site_packages is not None:
            _scan_dir(site_packages, recurse_dist_info=True)

        return foreign[:limit]
    except Exception:
        # Advisory preflight: structural surprise = "no verdict", never a blocked update.
        return []


def _refuse_update_if_venv_foreign_owned(project_root) -> None:
    """Refuse-before-mutate ownership gate, run after the pull and before the first venv mutation:
    foreign-owned files would brick the install mid-mutation, so refuse with the recovery command
    while the venv is intact. No subprocess calls — tests mock ``subprocess.run`` with sequenced effects."""
    foreign = _venv_foreign_owned_paths(Path(project_root) / "venv")
    if not foreign:
        return
    print("\n✗ Update stopped: this install's venv contains files owned by another user.")
    print("  Updating now would fail midway (Permission denied) and leave Hermes broken.")
    print("  This usually happens after running hermes or pip with sudo. Offending paths:")
    for p, uid in foreign:
        print(f"    - {p} (owner uid {uid})")
    print("\n  Fix ownership, then re-run the update:")
    print(f"    sudo chown -R $(id -un): {project_root}")
    print("    hermes update")
    print("\n  Nothing in the venv was modified.")
    sys.exit(1)


def _sync_python_dependencies_after_pull(
    git_cmd,
    branch,
    pre_pull_sha,
    *,
    active_lazy_features,
    active_tool_dependencies,
    _windows_gateway_resume):
    """Reinstall Python deps for the pulled checkout. Order matters: ownership preflight ->
    self-lock deferral -> core marker -> ``.[all]`` -> bytecode sweep -> lazy/tool refresh (own
    marker) -> memory-provider deps -> critical-import probe (warn only; stale bytecode self-heals)."""
    from hermes_cli.update_cmd import (
        _m,
        _sweep_bytecode_after_update,
        _validate_critical_modules_import,
        _write_lazy_refresh_incomplete_marker,
        _write_update_incomplete_marker)
    _refuse_update_if_venv_foreign_owned(_m().PROJECT_ROOT)
    # Self-lock deferral: if THIS process holds a native extension the sync must rewrite, defer
    # NOW (after the code swap) so only the install is pending for the next launch's marker.
    _m()._abort_dependency_sync_if_self_locked(_windows_gateway_resume)
    # Drop the core-install breadcrumb BEFORE touching the venv so a killed install is finished
    # by the next launch (``_recover_from_interrupted_install``). Lazy refresh uses its own marker.
    _write_update_incomplete_marker()
    deps_current = _editable_install_is_current(git_cmd, _m().PROJECT_ROOT, pre_pull_sha)
    if deps_current:
        print("→ Python dependencies unchanged — skipping reinstall")
    else:
        print("→ Updating Python dependencies...")
    from hermes_cli.managed_uv import ensure_uv, update_managed_uv

    # `uv self update` if we already have a managed uv.
    update_managed_uv()

    uv_bin = ensure_uv()

    # sys.executable -m pip avoids PEP 668 'externally-managed-environment' errors.
    pip_cmd = [sys.executable, "-m", "pip"]
    if not uv_bin:
        uv_bin = _ensure_uv_for_termux(pip_cmd)
    install_group = "all"

    if uv_bin:
        # managed_python_env() isolation so a third-party UV_PYTHON_INSTALL_DIR can't hijack uv.
        from hermes_cli.managed_uv import managed_python_env

        install_prefix, lazy_env = [uv_bin, "pip"], managed_python_env()
        lazy_env["VIRTUAL_ENV"] = str(_m().PROJECT_ROOT / "venv")
        termux_note = "  → Termux detected: using uv + curated termux-all optional profile..."
    else:
        _ensure_venv_pip(pip_cmd, sys.executable)
        install_prefix, lazy_env = pip_cmd, None
        termux_note = "  → Termux detected: using curated termux-all optional profile..."
    is_termux = _m()._is_termux_env(lazy_env)
    if is_termux:
        if lazy_env is not None:
            lazy_env.pop("PYTHONPATH", None)
            lazy_env.pop("PYTHONHOME", None)
        install_group = "termux-all"
        print(termux_note)
    if not deps_current:
        if is_termux and _is_android_python():
            print("  → Termux/Android detected: prebuilding psutil with Linux source path compatibility...")
            _install_psutil_android_compat(install_prefix, env=lazy_env)
        _m()._install_python_dependencies_with_optional_fallback(
            install_prefix, env=lazy_env, group=install_group)

    if deps_current:
        # Verification normally runs inside the skipped install; run it here so a wrong skip
        # self-heals (both verifiers reinstall what they find missing).
        _m()._verify_core_dependencies_installed(install_prefix, env=lazy_env, group=install_group)
        _m()._verify_console_scripts_installed(install_prefix, env=lazy_env)

    # Clear the core breadcrumb before lazy refresh, which uses its own marker so a lazy
    # failure can't be "healed" by a narrow core import probe.
    _m()._clear_update_incomplete_marker()

    # Still the old interpreter process: refresh caches/modules before lazy refresh imports
    # newly-pulled modules. The install may have regenerated bytecode from build-cache
    # copies — this second sweep catches those stragglers.
    _sweep_bytecode_after_update(branch)
    _m()._reload_updated_runtime_modules()

    # Stale pip can fail source builds and leave partially-written packages.
    _write_lazy_refresh_incomplete_marker()
    _m()._upgrade_pip_before_lazy_refresh(install_prefix, env=lazy_env)

    # Clear the lazy marker only when refresh/repair is confirmed healthy.
    lazy_ok = _m()._refresh_active_lazy_features(
        install_prefix, env=lazy_env, features=active_lazy_features)
    if lazy_ok:
        _m()._clear_lazy_refresh_incomplete_marker()
    else:
        print(
            "  ⚠ Lazy-refresh recovery incomplete — run `hermes` again "
            "to finish import-based venv repair.")

    _m()._restore_active_tool_dependencies(active_tool_dependencies, install_prefix, env=lazy_env)

    # Heal memory-provider bridge packages last — the steps above may have stripped them.
    _m()._refresh_active_memory_provider_dependencies()

    # Remaining import failures are real breakage. Warn only — never roll back: `cannot import
    # name X` is also the stale-bytecode signature, which self-heals next launch.
    import_ok, failing_module, import_error = _validate_critical_modules_import(_m().PROJECT_ROOT)
    if not import_ok:
        print()
        print(f"  ⚠ {failing_module} still fails to import after updating:")
        print(f"      {import_error}")
        print("    Run `hermes update` again — if it persists, reinstall:")
        print("    https://hermes-agent.nousresearch.com")
