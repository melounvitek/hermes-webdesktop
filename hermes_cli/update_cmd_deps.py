"""Dependency sync after ``hermes update``: venv health/ownership preflight, editable reinstall, lazy-feature refresh, npm/Desktop rebuilds, self-lock deferral.

Split out of ``hermes_cli/update_cmd.py``; every moved name is re-imported there, so
``hermes_cli.update_cmd.<name>`` keeps resolving (and monkeypatching) as before.
Origin-internal helpers are imported lazily inside each function (no import cycle;
test patches on ``hermes_cli.update_cmd.<name>`` stay effective).
"""

import logging
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


# Files that define the editable install. A pull that touches none of them
# cannot have invalidated it.
_INSTALL_DEFINING_FILES = (
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "MANIFEST.in",
    "uv.lock",
)


def _editable_install_is_current(git_cmd, cwd, pre_pull_sha: str | None) -> bool:
    """True when the pulled commits cannot have invalidated the editable install.

    ``uv pip install -e .`` reinstalls unconditionally and rewrites the
    console-script shims every time. On Windows that rewrite is the only
    reason the running ``hermes.exe`` must be quarantined, and a lost
    quarantine race is the whole ``os error 32`` family — so skip the
    reinstall when it provably cannot change anything.

    Safe because Hermes pins its editable finder to a *static* module list
    (``[tool.setuptools] py-modules`` + ``packages.find.include``): only a
    new top-level module/package can stale it, and that needs a
    ``pyproject.toml`` diff (as do dependencies and ``[project.scripts]``).
    New submodules under an already-mapped package need no reinstall.

    Fails closed: an unresolvable pre-pull SHA (shallow checkout, ZIP swap)
    or a failed ``git diff`` returns False and the install runs as before.
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
            text=True, encoding="utf-8", errors="replace",
        )
    except OSError:
        return False
    if result.returncode != 0:
        return False
    return not result.stdout.strip()


# Modules imported on every agent startup. Unlike _UPDATE_CRITICAL_FILES (which
# is only parsed), these are actually *imported* so that cross-module breakage
# is caught — a file can be syntactically perfect and still fail to import
# because a name it pulls from a sibling module no longer exists.
_UPDATE_CRITICAL_MODULES = (
    "hermes_cli.main",
    "run_agent",
    "model_tools",
    "toolsets",
)


def _critical_module_import_failures(
    root, *, report_runtime_errors: bool = False
) -> dict[str, tuple[str, str]]:
    """Import each module in ``_UPDATE_CRITICAL_MODULES`` in a subprocess.

    ``_validate_critical_files_syntax`` only *parses*, so a partially-updated
    tree (new ``agent/``, old ``tools/``) parses fine yet dies at startup
    with ``ImportError: cannot import name ...``. That skew is reachable on
    the Windows ZIP-update path, whose copy loop replaces top-level entries
    one at a time in ``os.listdir`` order.

    Runs in a subprocess (~0.4s) so the half-updated tree's import-time side
    effects don't pollute the updater's ``sys.modules``. Uses the project
    venv's interpreter when present (like ``_venv_core_imports_healthy``):
    ``hermes update`` may be driven by a different Python than the install's.

    Returns every failing module in probe order. Generic import-time
    exceptions are tolerated by default (they can depend on local config);
    ``report_runtime_errors=True`` exposes them so a caller can compare two
    states of the same checkout without one failure masking another.
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
        # A missing *third-party* module means dependencies aren't installed
        # yet, not a skewed checkout. Only our own packages count as breakage.
        # The root set is injected from hermes_constants so this can't drift
        # from the hint the user is shown (they disagreed once already).
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
            marker,
        )
    )
    try:
        interpreter = sys.executable
        try:
            venv_python = venv_python_path(
                Path(root) / "venv", windows=_m()._is_windows()
            )
            if venv_python.exists():
                interpreter = str(venv_python)
        except Exception:
            pass  # fall back to the running interpreter
        result = subprocess.run(
            [interpreter, "-c", probe],
            cwd=str(root),
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=120,
        )
    except subprocess.TimeoutExpired:
        return {
            "critical-module probe": (
                "TimeoutExpired",
                "timed out before reporting import health",
            )
        }
    except (OSError, subprocess.SubprocessError):
        # Can't run the probe — don't block the update on our own tooling.
        return {}
    output = result.stdout or ""
    if marker not in output:
        return {
            "critical-module probe": (
                "ProbeTerminated",
                "terminated before reporting import health "
                f"(exit code {result.returncode})",
            )
        }
    try:
        import json

        failures = json.loads(output.rsplit(marker, 1)[1])
        if not isinstance(failures, list) or any(
            not isinstance(item, list)
            or len(item) != 3
            or not all(isinstance(value, str) for value in item)
            for item in failures
        ):
            raise ValueError("invalid import-health payload")
        return {
            str(module): (str(kind), str(detail))
            for module, kind, detail in failures
        }
    except (TypeError, ValueError):
        return {
            "critical-module probe": (
                "MalformedPayload",
                "reported malformed import health data",
            )
        }


def _validate_critical_modules_import(
    root, *, report_runtime_errors: bool = False
) -> tuple[bool, str | None, str | None]:
    """Return the first critical-module import failure, if any."""
    failures = _critical_module_import_failures(
        root, report_runtime_errors=report_runtime_errors
    )
    if failures:
        module = next(iter(failures))
        return False, module, failures[module][1]
    return True, None, None


def _npm_bin_exists(bin_dir: Path, name: str) -> bool:
    """True when an npm bin shim for *name* exists (POSIX or Windows)."""
    return any(
        (bin_dir / candidate).exists()
        for candidate in (name, f"{name}.cmd", f"{name}.ps1", f"{name}.exe")
    )


def _web_build_toolchain_ready(*roots: Path) -> bool:
    """True when ``tsc`` and ``vite`` shims are reachable from any of *roots*.

    Callers must pass every root the build would search; checking only one
    reports a healthy tree as broken.
    """
    bin_dirs = [
        bin_dir
        for bin_dir in (root / "node_modules" / ".bin" for root in roots)
        if bin_dir.is_dir()
    ]
    return bool(bin_dirs) and all(
        any(_npm_bin_exists(bin_dir, tool) for bin_dir in bin_dirs)
        for tool in ("tsc", "vite")
    )


def _web_toolchain_roots(web_dir: Path) -> tuple[Path, ...]:
    """Roots whose ``node_modules/.bin`` can satisfy the web build.

    ``npm run build`` prepends ``node_modules/.bin`` for the package and each
    of its ancestors, so shims hoisted to the workspace root and shims nested
    under a package that owns its lockfile (#42973) are equally valid.
    """
    return (web_dir, web_dir.parent)


def _ensure_venv_pip(pip_cmd: list, python_exe: str) -> None:
    """Bootstrap pip back into the venv via ensurepip when ``pip --version`` fails
    (some environments lose it); call before the editable install."""
    from hermes_cli.update_cmd import _m
    try:
        subprocess.run(
            pip_cmd + ["--version"],
            cwd=_m().PROJECT_ROOT,
            check=True,
            capture_output=True,
        )
    except subprocess.CalledProcessError:
        subprocess.run(
            [python_exe, "-m", "ensurepip", "--upgrade", "--default-pip"],
            cwd=_m().PROJECT_ROOT,
            check=True,
        )


def _upgrade_pip_before_lazy_refresh(
    install_cmd_prefix: list[str],
    *,
    env: dict[str, str] | None = None,
) -> None:
    """Upgrade pip before lazy-backend refreshes.

    Older pip (e.g. 24.0 on Python 3.11) can fail setuptools-backed source
    builds during lazy installs and leave a partially-written venv (#57828).
    Never raises.
    """
    from hermes_cli.update_cmd import _m
    try:
        _m()._run_package_only_install(
            install_cmd_prefix + ["install", "--upgrade", "pip"],
            env=env,
        )
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
    dependencies: list[str],
    install_cmd_prefix: list[str],
    *,
    env: dict[str, str] | None = None,
) -> None:
    """Restore allowlisted ``hermes tools`` dependencies into a rebuilt venv.

    The dependency names came from a pre-rebuild import probe and are resolved
    through a static package allowlist. Never raises: a failed optional tool
    must not block the core update, but the user must be told what stayed
    unavailable.
    """
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
                        module_name,
                    ],
                    capture_output=True,
                    env=env,
                    check=False,
                )
                if probe.returncode == 0:
                    continue
            except (subprocess.SubprocessError, OSError):
                # An indeterminate probe is safer to repair than to treat as
                # proof that a pre-rebuild dependency survived.
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
                install_cmd_prefix + ["install", *install_args, "--quiet"],
                env=env,
            )
            restored.append(name)
        except Exception as exc:
            # Best-effort optional tooling: surface failures without aborting
            # the core update.
            failed.append((name, str(exc)))

    if restored:
        print(f"  ✓ {len(restored)} restored: {', '.join(restored)}")
    for name, reason in failed:
        if len(reason) > 200:
            reason = reason[:200] + "..."
        print(f"  ⚠ {name} failed to restore: {reason}")


def _refresh_active_lazy_features(
    install_cmd_prefix: list[str] | None = None,
    *,
    env: dict[str, str] | None = None,
    features: list[str] | None = None,
) -> bool:
    """Refresh lazy-installed backends after a code update.

    ``uv pip install -e .[all]`` never touches ``tools/lazy_deps.py`` backends,
    so a bumped :data:`LAZY_DEPS` pin (CVE, transitive fix) would otherwise
    leave already-activated backends stale forever. Reinstalls only the
    features the user previously activated; cold backends stay untouched.

    Returns True when the venv is safe to use (refresh succeeded, nothing
    active, or post-failure import repair succeeded); False when a failed
    lazy install left broken core imports that repair could not fix (#57828).

    Never raises. A failure here must not block the rest of the update.
    """
    from hermes_cli.update_cmd import _m
    try:
        from tools import lazy_deps
    except Exception as exc:
        logger.debug("Lazy refresh skipped (import failed): %s", exc)
        return True

    if features is None:
        try:
            active = lazy_deps.active_features()
        except Exception as exc:
            logger.debug("Lazy refresh skipped (active_features failed): %s", exc)
            return True
    else:
        active = features

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
        # refresh_active_features is documented as never-raise, but defend
        # the update flow against future regressions.
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
        # Most common reason: security.allow_lazy_installs=false. Show one
        # line so the user knows why; not an error.
        names = ", ".join(f for f, _ in skipped)
        reason = skipped[0][1].split(": ", 1)[-1]
        print(f"  · {len(skipped)} skipped ({reason}): {names}")

    if not failed and not unexpected_failure:
        return True

    for feature, status in failed:
        reason = status.split(": ", 1)[-1]
        # Clip noisy pip stderr to keep update output legible.
        if len(reason) > 200:
            reason = reason[:200] + "..."
        print(f"  ⚠ {feature} failed to refresh: {reason}")

    if install_cmd_prefix is None:
        print("  ⚠ Lazy refresh failed; rerun `hermes update` once resolved.")
        return False

    # Immediate import-based recovery — metadata-only verifiers miss the case
    # where DISTRIBUTION-INFO remains but import files were wiped (#57828).
    # Unavailable probes are indeterminate, not healthy — keep the lazy marker.
    status = _m()._repair_venv_via_import_probes(install_cmd_prefix, env=env)
    if status == "repaired":
        print(
            "  Lazy backend(s) keep their previous version until refresh succeeds."
        )
        return True
    if status == "healthy":
        print(
            "  Lazy backend(s) keep their previous version; probed packages look intact."
        )
        print("  Rerun `hermes update` once the upstream issue is resolved.")
        return True
    if status == "indeterminate":
        print(
            "  ⚠ Leaving `.lazy-refresh-incomplete` until import probes can confirm health."
        )
    return False


def _refresh_active_memory_provider_dependencies() -> None:
    """Refresh pip dependencies for the configured external memory provider.

    Provider bridge packages are declared in each provider's ``plugin.yaml``
    (plus mode extras like Hindsight's ``hindsight-all``), not in Hermes'
    extras or ``LAZY_DEPS``, so the core reinstall can strip or downgrade
    them (#53272, #70636). Re-run the ACTIVE provider's install after the
    core install and lazy refresh so its writes to shared packages land last.

    Never raises. A failure here must not block the rest of the update.
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

    # "default" / empty is the built-in file-backed store — no pip deps.
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
    install_cmd_prefix: list[str],
    *,
    env: dict[str, str] | None = None,
) -> None:
    """Install psutil on Android by patching upstream platform detection.

    psutil's setup gates Linux sources behind ``sys.platform.startswith('linux')``;
    Termux reports ``'android'``, so setup aborts although the Linux source path
    compiles fine. Only the extracted build tree for this attempt is patched.

    Stopgap: remove (together with the standalone installer's use of the same
    helper) once https://github.com/giampaolo/psutil/pull/2762 ships.
    """
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
            install_cmd_prefix + ["install", "--no-build-isolation", str(src_root)],
            env=env,
        )


def _ensure_uv_for_termux(pip_cmd: list[str]) -> str | None:
    """Best-effort uv bootstrap on Termux for faster update installs.

    The official uv installer may not work on Termux (glibc vs bionic). Prefer
    a uv already on PATH (``pkg install uv``); otherwise fall back to a
    wheel-only ``pip install uv`` so the Rust crate is never source-built.
    """
    from hermes_cli.update_cmd import _m
    from hermes_cli.managed_uv import resolve_uv

    existing = resolve_uv()
    if existing:
        return existing
    if not _m()._is_termux_env():
        return None
    # A Termux-packaged uv lands on PATH but not in the managed bin dir, so
    # resolve_uv() misses it. Use it before pip, which has no Android wheel and
    # would otherwise build uv from source on a low-memory device.
    system_uv = shutil.which("uv")
    if system_uv:
        return system_uv
    try:
        print("  → Termux detected: trying to install uv for faster dependency updates...")
        result = subprocess.run(
            pip_cmd + ["install", "uv", "--only-binary", ":all:"],
            cwd=_m().PROJECT_ROOT,
            check=False,
        )
        if result.returncode != 0:
            return None
    except Exception:
        pass
    return resolve_uv() or shutil.which("uv")


def _npm_manifest_paths() -> tuple[Path, ...]:
    """Manifests whose changes must defeat the update-skip.

    The lockfile alone is not a sufficient key: a dev can edit a package.json
    (root or workspace) without running npm, and `hermes update` is exactly
    the step expected to sync node_modules (`npm install` fallback in
    _run_npm_install_deterministic).

    Workspaces come from the root package.json's `workspaces` globs so a new
    workspace can never escape the key. Every workspace manifest counts —
    desktop included, though the install names only ui-tui and web — because
    the single lockfile spans the whole workspace graph. Falls back to root
    manifests only if package.json is unreadable (never skips more than main
    would have installed).
    """
    from hermes_cli.update_cmd import _m
    root_pkg = _m().PROJECT_ROOT / "package.json"
    paths = [_m().PROJECT_ROOT / "package-lock.json", root_pkg]
    try:
        workspaces = json.loads(root_pkg.read_text(encoding="utf-8")).get(
            "workspaces", []
        )
        if isinstance(workspaces, dict):  # legacy {"packages": [...]} form
            workspaces = workspaces.get("packages", [])
        for pattern in workspaces:
            for match in sorted(_m().PROJECT_ROOT.glob(str(pattern))):
                manifest = match / "package.json"
                if manifest.is_file():
                    paths.append(manifest)
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return tuple(paths)


def _npm_manifests_digest() -> str | None:
    """Combined sha256 over the lockfile + all workspace package.json files.

    Returns None when the lockfile is missing (never skip then).
    """
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
    # Also check that node_modules exists; a matching hash with missing
    # node_modules means the cache was recorded by another checkout.
    if not (_m().PROJECT_ROOT / "node_modules").is_dir():
        return True
    # A matching hash must NOT skip the reinstall when the web build toolchain
    # never landed, or every later update rebuilds against a half-installed tree.
    web_dir = _m().PROJECT_ROOT / "web"
    if (web_dir / "package.json").is_file() and not _web_build_toolchain_ready(
        *_web_toolchain_roots(web_dir)
    ):
        return True
    try:
        # Key the cache by PROJECT_ROOT so parallel worktrees don't collide.
        cache_key = hashlib.sha256(str(_m().PROJECT_ROOT).encode()).hexdigest()[:12]
        cache_file = hermes_root / f".npm_lock_hash_{cache_key}"
        if not cache_file.exists():
            return True
        return cache_file.read_text(encoding="utf-8").strip() != current
    except OSError:
        return True


def _record_npm_lockfile_hash(hermes_root: Path) -> None:
    from hermes_cli.update_cmd import _m
    digest = _npm_manifests_digest()
    if digest is None:
        return
    try:
        cache_key = hashlib.sha256(str(_m().PROJECT_ROOT).encode()).hexdigest()[:12]
        cache_file = hermes_root / f".npm_lock_hash_{cache_key}"
        cache_file.write_text(digest, encoding="utf-8")
    except OSError:
        logger.debug("Could not write npm lockfile hash cache")


def _repair_node_deps_on_current_checkout(
    print_completion,
    *,
    assume_yes: bool = False,
    gateway_mode: bool = False,
    pre_update_snapshot_id: str | None = None,
    completion_message: str = "✓ Already up to date!",
    had_desktop_app_before_update: bool = False,
) -> bool:
    """Repair Node deps on the ``commit_count == 0`` path (#77211).

    A current checkout does not imply healthy Node deps: a failed npm install
    (EBADENGINE, network timeout, interrupt) says "re-run hermes update", but
    the early return never reached the Node refresh. ``_update_node_dependencies``
    self-gates on the lockfile hash, recorded only after a SUCCESSFUL install
    (and re-tripped when node_modules or the web toolchain is missing), so this
    is a cheap no-op on healthy installs and a real repair after a failed one.
    """
    from hermes_cli.update_cmd import (
        _check_and_apply_config_migration,
        _m,
        _rebuild_desktop_after_update,
        _update_node_dependencies,
    )
    node_failures = _update_node_dependencies()
    if node_failures:
        print(f"  ⚠ Node.js refresh failed for: {', '.join(node_failures)}")
        print("    Fix npm and re-run `hermes update`.")
        print_completion(
            "⚠ Checkout is current, but Node.js dependencies could not be repaired."
        )
        return False
    # Pair the refresh with the web build like every other
    # _update_node_dependencies call site; it staleness-checks internally,
    # so this is a no-op when nothing changed.
    _m()._build_web_ui(_m().PROJECT_ROOT / "web")
    _check_and_apply_config_migration(
        assume_yes=assume_yes,
        gateway_mode=gateway_mode,
        pre_update_snapshot_id=pre_update_snapshot_id,
    )
    # A current checkout can still owe a Desktop rebuild (#97343) — e.g. the
    # Windows hand-off child never reaches the commits-pulled rebuild — leaving
    # a stale app behind a successful-looking update. Self-gates on the build stamp.
    if not _rebuild_desktop_after_update(
        _m().PROJECT_ROOT / "apps" / "desktop",
        had_desktop_app_before_update=had_desktop_app_before_update,
    ):
        # _rebuild_desktop_after_update already printed the retry hint; withhold
        # success rather than claiming the update finished (#88251).
        print_completion(
            "⚠ Update partially complete — the desktop app was not rebuilt "
            "and is still on the previous build."
        )
        return False
    return bool(print_completion(completion_message))


def _update_node_dependencies() -> list[str]:
    """Refresh Node deps for the ui-tui and web workspaces.

    Returns the list of labels whose npm install failed (empty on success),
    so the caller can treat a Node refresh failure as a partial update rather
    than silently reporting ``Update complete!`` (#30271).
    """
    from hermes_cli.update_cmd import _m
    if not (_m().PROJECT_ROOT / "package.json").exists():
        return []

    npm = _m()._resolve_node_runtime_npm()
    if not npm:
        # If the only npm reachable inside this WSL shell is the Windows one,
        # flag it loudly: silently skipping leaves ui-tui deps stale while the
        # rest of the update proceeds, and running it would corrupt the tree.
        from hermes_constants import is_wsl

        path_npm = shutil.which("npm")
        if is_wsl() and path_npm and _m()._is_windows_npm_path(path_npm):
            print("→ Updating Node.js dependencies...")
            print("  ⚠ Skipped: only a Windows npm is reachable from this WSL shell.")
            print("    Install Node.js inside the WSL distro (nvm, or your distro's")
            print("    package manager), then re-run `hermes update`.")
            failed = []
            if any(
                (_m().PROJECT_ROOT / workspace / "package.json").exists()
                for workspace in ("ui-tui", "web")
            ):
                failed.append("ui-tui, web workspaces")
            return failed
        return []

    from hermes_constants import get_default_hermes_root

    # node_modules is shared by every profile on this checkout, so keep one
    # per-checkout cache under the shared root instead of one per profile.
    shared_hermes_root = get_default_hermes_root()

    # Best-effort npx cache warm for agent-browser (#43564), before the
    # lockfile-unchanged early return (the common case). Can block ~11s on a
    # cold cache — print first so it doesn't look like a hang.
    print("→ Warming npx cache for agent-browser...")
    try:
        from tools.browser_tool import warm_agent_browser_npx_cache
        warm_agent_browser_npx_cache()
    except Exception:
        pass

    if not _m()._npm_lockfile_changed(shared_hermes_root):
        logger.info("npm lockfile unchanged, skipping npm install")
        return []

    # Root package.json has no dependencies of its own (#43564: agent-browser
    # resolves via `npx` at runtime, @streamdown/math moved to apps/desktop),
    # so a workspace-scoped install prunes nothing root-only. apps/desktop is
    # deliberately never named: its Electron devDependency has a ~200MB
    # postinstall download, so desktop deps install on demand
    # (see _desktop_build_needed).
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
        # Root's own devDependencies (the shared ESLint flat config every
        # workspace imports) would otherwise be pruned by this scoped install
        # and have nowhere else to live. apps/desktop is still excluded since
        # it is never named above.
        "--include-workspace-root",
    ]

    from hermes_constants import with_hermes_node_path

    nixos_env = with_hermes_node_path(_m()._nixos_build_env())

    # capture_output=False is deliberate (#18840): optional postinstall scripts
    # print download progress, and capturing it makes a long download look
    # hung. The npm-deprecation noise comes from the desktop build (captured
    # to update.log), not this step.
    result = _m()._run_npm_install_deterministic(
        npm,
        _m().PROJECT_ROOT,
        extra_args=tuple(install_args),
        capture_output=False,
        env=nixos_env,
    )
    if result.returncode == 0:
        _record_npm_lockfile_hash(shared_hermes_root)
        print("  ✓ ui-tui, web workspaces installed (desktop skipped)")
        failures: list[str] = []
    else:
        print("  ⚠ npm install failed")
        stderr = (result.stderr or "").strip() if result.stderr else ""
        if stderr:
            print(f"    {stderr.splitlines()[-1]}")
        failures = _partial_update_failure("ui-tui, web workspaces")

    return failures


def _venv_core_imports_healthy() -> tuple[bool, str]:
    """Probe the project venv for the core imports the backend needs to boot.

    Runs inside the venv interpreter (NOT this process — ``hermes update`` may
    run under a different Python). Catches a half-updated venv: checkout
    current but a dependency sync failed or was killed partway (e.g. Windows
    access-denied on a loaded .pyd). Without it, a current checkout prints
    "Already up to date!" and never re-syncs, so the install stays broken.

    Returns ``(healthy, detail)``. Never raises; unknown states report
    healthy so a probe failure can't force needless reinstalls.
    """
    from hermes_cli.update_cmd import _m
    venv_dir = _m().PROJECT_ROOT / "venv"
    venv_python = venv_python_path(venv_dir, windows=_m()._is_windows())
    if not venv_python.exists():
        # No venv interpreter. Normal for a dev checkout (report healthy to
        # avoid forced reinstalls), but on a MANAGED install (bootstrap stamp
        # or `.update-incomplete` present) the venv IS the install — its
        # absence means a repair was interrupted after the old venv was moved
        # aside, and "Already up to date!" would be a lie.
        managed_markers = (
            _m().PROJECT_ROOT / ".hermes-bootstrap-complete",
            _m()._update_marker_path(),
        )
        if any(m.exists() for m in managed_markers):
            return False, f"venv python missing ({venv_python})"
        return True, ""

    # Core web/serve imports plus their newest transitive deps. Import (not
    # just metadata) — a package can have intact dist-info but a missing
    # module after an interrupted uninstall/install cycle.
    check = (
        "import importlib\n"
        "mods = ['fastapi', 'uvicorn', 'pydantic', 'openai', 'yaml']\n"
        "missing = []\n"
        "for m in mods:\n"
        "    try: importlib.import_module(m)\n"
        "    except Exception as e: missing.append(f'{m}: {e}')\n"
        "print('\\n'.join(missing))\n"
    )
    try:
        result = subprocess.run(
            [str(venv_python), "-c", check],
            capture_output=True,
            text=True, encoding="utf-8", errors="replace",
            timeout=60,
            cwd=_m().PROJECT_ROOT,
        )
    except Exception as exc:
        logger.debug("venv health probe failed to run: %s", exc)
        return True, ""

    missing = [line.strip() for line in (result.stdout or "").splitlines() if line.strip()]
    if result.returncode != 0 and not missing:
        # Interpreter itself is broken (e.g. deleted stdlib) — that IS unhealthy.
        detail = (result.stderr or "").strip().splitlines()
        return False, detail[0] if detail else "venv python failed to run"
    if missing:
        return False, "; ".join(missing[:4])
    return True, ""


# Native-extension modules that pin files inside the venv once imported. If
# the updater itself has one loaded, Windows blocks REPLACE on the mapped
# ``.pyd``/``.dll`` and the sync dies with ``os error 5`` between uninstall
# and reinstall, stranding the venv half-updated (#83569). ``cryptography``
# is the canonical case; PyYAML's ``_yaml`` is loaded by every CLI process.
# Kept as defence-in-depth against future eager imports, but the guard must
# be HONEST (#86735/#86780/#86781: a preflight firing on every run, before
# the fetch, re-bricked the flow it protected). Two honesty gates:
#
# 1. Fire only when the sync would actually REWRITE the loaded distribution
#    (``_dependency_sync_would_rewrite``); a satisfied pin means uv/pip
#    never touch the mapped ``.pyd``.
# 2. Run AFTER the code swap, right before the venv rewrite — so gate 1
#    compares against the NEW pyproject and a deferral leaves the user on
#    new code with only the dependency install pending for the next launch's
#    marker recovery.
#
# Keys are ``sys.modules`` prefixes; values are ``(display name, PyPI dist)``.
_SELF_LOCKING_NATIVE_MODULES: dict[str, tuple[str, str]] = {
    "cryptography.hazmat.bindings._rust": ("cryptography (_rust.pyd)", "cryptography"),
    "yaml._yaml": ("PyYAML (_yaml.pyd)", "pyyaml"),
}


def _dependency_sync_would_rewrite(dist_name: str) -> bool | None:
    """Whether ``uv pip install -e .[all]`` would replace *dist_name*'s files.

    Compares the installed version against every applicable requirement in
    the on-disk ``pyproject.toml`` (base deps plus all extras). ``False`` —
    every pin satisfied, a mapped extension is NOT at risk; ``True`` — some
    pin unsatisfied or dist missing; ``None`` — undeterminable.

    Never raises. Callers treat ``None`` as fail-OPEN (no deferral): PyYAML
    is loaded by every process, so deferring on uncertainty would recreate
    the #86735 always-firing loop.
    """
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
        # Not pinned anywhere in pyproject: the resolver may still move it
        # as a transitive — we cannot cheaply predict that, so stay honest
        # about the uncertainty.
        return None
    except Exception:
        return None


def _detect_self_loaded_native_modules() -> list[str]:
    """Native venv extensions loaded into THIS process that the sync would rewrite.

    Returns display names (empty off Windows — POSIX lets a running process
    keep using an unlinked inode, so self-locking is a Windows-only hazard).
    A loaded module whose installed version already satisfies the on-disk
    pyproject pins is NOT reported: the dependency sync will not touch its
    files, so there is no swap at risk (#86735 — the always-firing variant
    of this preflight bricked every Windows update).  Never raises.
    """
    from hermes_cli.update_cmd import _m
    if not _m()._is_windows():
        return []
    found = []
    for prefix, (display, dist) in _SELF_LOCKING_NATIVE_MODULES.items():
        if prefix not in sys.modules:
            continue
        # Defer ONLY on a CONFIRMED pending rewrite; "unknown" must fail OPEN,
        # since PyYAML is loaded in every CLI process and treating unknown as
        # at-risk recreated the always-firing loop (#86735). A missed deferral
        # only yields the pre-existing mid-sync os error 5, which marker
        # recovery already handles — far less harmful than an update that
        # can never run.
        if _m()._dependency_sync_would_rewrite(dist) is not True:
            continue
        found.append(display)
    return sorted(set(found))


def _abort_dependency_sync_if_self_locked(gateway_resume=None) -> None:
    """Defer the venv rewrite when THIS process holds something it must replace.

    Runs after the code swap, right before the venv rewrite, so a deferral
    leaves the user on NEW code with only the dependency install pending.
    No-op when nothing at-risk is held. Two hazards with different recoveries:

    - A mapped native extension (``.pyd``): exit 2 and let the next launch's
      marker recovery finish the install before importing anything heavy.
    - The ``hermes.exe`` shim we were launched from (#88838, #89599): every
      future launch is also the shim, so the marker would defer forever.
      Hand the install to a child under the venv interpreter and exit.
    """
    from hermes_cli.update_cmd import _m
    locked = _m()._detect_self_loaded_native_modules()
    if locked:
        _m()._defer_update_for_self_lock(locked)
        if gateway_resume is not None:
            _m()._resume_windows_gateways_after_update(gateway_resume)
        sys.exit(2)

    if _m()._reexec_dependency_sync_off_windows_shim():
        if gateway_resume is not None:
            _m()._resume_windows_gateways_after_update(gateway_resume)
        sys.exit(0)


def _defer_update_for_self_lock(loaded: list[str]) -> None:
    """Bail out before the dependency sync when the updater holds a lock.

    The install cannot win this race from inside the locked process — even
    killing threads would not unmap the image — so defer it: drop the
    update-incomplete marker (next launch's fresh process completes the
    install before importing anything heavy), explain, and exit 2 like the
    other preflight refusals.
    """
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
        or _m()._desktop_dist_exists(desktop_dir)
    )


def _rebuild_desktop_after_update(
    desktop_dir: Path, *, had_desktop_app_before_update: bool
) -> bool:
    """Rebuild an installed Desktop app when its source or artifact changed.

    Returns ``False`` only when a rebuild was attempted and failed, so the
    caller can withhold ``✓ Update complete!`` and (in gateway mode) write
    a failing ``.update_exit_code`` (#88251). Every other outcome — nothing
    to rebuild, up to date, build succeeded, Desktop never installed —
    returns ``True``.
    """
    from hermes_cli.update_cmd import _m
    # The release tree is ignored by git and can disappear during an update.
    # Its pre-update presence is enough to restore it; do not make people who
    # have never used Desktop pay for an Electron build.
    has_desktop_app = had_desktop_app_before_update or _desktop_app_present(desktop_dir)
    if not (
        (desktop_dir / "package.json").exists()
        and _m()._resolve_node_runtime_npm()
        and has_desktop_app
    ):
        return True

    print("→ Checking if desktop app needs rebuilding...")
    # Consult the content-hash stamp IN-PROCESS first: the spawned
    # `hermes desktop --build-only` re-imports the whole CLI stack (~1-3 s)
    # just to reach the same _m()._desktop_build_needed check. The update path
    # never passes --source, so mirror source_mode=False. Any pre-check error
    # falls through to the subprocess.
    skip_desktop_build = False
    try:
        skip_desktop_build = not _m()._desktop_build_needed(
            desktop_dir, _m().PROJECT_ROOT, source_mode=False
        )
    except Exception:
        skip_desktop_build = False
    if skip_desktop_build:
        print("  ✓ Desktop app up to date")
        return True

    desktop_build_cmd = [sys.executable, "-m", "hermes_cli.main", "desktop", "--build-only"]
    # Capture the (very loud) Electron/vite build output into update.log. On
    # a nonzero exit, retry once (covers a still-settling rebuild window), then
    # surface the captured tail so the failure is debuggable.
    #
    # Put the Hermes-managed Node on PATH: inside the desktop updater chain
    # (Desktop → hermes-setup → hermes update) shell PATH customizations are
    # lost, so a bare-PATH child fails with `node: not found` before cmd_gui
    # can self-heal.
    from hermes_constants import with_hermes_node_path

    build_env = with_hermes_node_path()
    build_result = _m()._run_logged_subprocess(
        desktop_build_cmd, cwd=_m().PROJECT_ROOT, env=build_env
    )
    if build_result.returncode != 0:
        build_result = _m()._run_logged_subprocess(
            desktop_build_cmd, cwd=_m().PROJECT_ROOT, env=build_env
        )
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
    """Owner uid of ``path`` via ``os.stat`` — ``None`` when unreadable.

    Separate seam so tests can simulate root-owned files without chown
    (which needs root). Never raises.
    """
    try:
        return os.stat(path, follow_symlinks=False).st_uid
    except OSError:
        return None


def _venv_foreign_owned_paths(venv_root, limit: int = 5) -> list:
    """Bounded scan for venv entries not owned by the current user (#83529).

    A venv ever touched by ``sudo pip`` / ``sudo hermes`` contains root-owned
    files (classically ``*.dist-info/INSTALLER``); a later normal ``hermes
    update`` then dies mid-mutation inside ``uv pip install -e .`` with
    ``venv/bin/hermes`` already deleted — the CLI is bricked. Same philosophy
    as the contended-venv gate (#87331): never mutate a venv we cannot safely mutate.

    Deliberately BOUNDED (no full recursion): the venv root, direct entries of
    ``venv/bin``, top-level entries of the first ``lib/python*/site-packages``,
    and direct children of each ``*.dist-info`` there; ~2000 stat calls max,
    at most ``limit`` paths returned. POSIX-only: ``[]`` on Windows and as
    root. Swallows every per-entry ``OSError`` and returns ``[]`` on any
    structural surprise — must NEVER raise or add noticeable latency.

    Returns ``(path_str, uid)`` tuples, at most ``limit`` long.
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
        site_packages = next(
            iter(sorted(venv_root.glob("lib/python*/site-packages"))), None
        )
        if site_packages is not None:
            _scan_dir(site_packages, recurse_dist_info=True)

        return foreign[:limit]
    except Exception:
        # Preflight is advisory: any structural surprise means "no verdict",
        # never a crashed or blocked update.
        return []


def _refuse_update_if_venv_foreign_owned(project_root) -> None:
    """Refuse-before-mutate ownership gate for the dependency install (#83529).

    Runs after the code pull and immediately before the first venv mutation:
    foreign-owned venv files would make ``uv pip install -e .`` die
    mid-mutation and brick the install, so refuse up front with the exact
    recovery command while the venv is intact. No subprocess calls here —
    update tests mock ``subprocess.run`` with sequenced side effects.
    """
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
    _windows_gateway_resume,
):
    """Reinstall Python dependencies for the freshly pulled checkout.

    Order matters: ownership preflight -> self-lock deferral -> core-install
    marker -> ``.[all]`` (uv or pip) -> bytecode sweep -> lazy-feature and
    tool-dependency refresh (own marker) -> memory-provider bridge deps ->
    critical-import probe (warn only; stale-bytecode self-heals next launch).
    """
    from hermes_cli.update_cmd import (
        _m,
        _sweep_bytecode_after_update,
        _validate_critical_modules_import,
        _write_lazy_refresh_incomplete_marker,
        _write_update_incomplete_marker,
    )
    _refuse_update_if_venv_foreign_owned(_m().PROJECT_ROOT)
    #
    # Self-lock deferral (relocated preflight — #86735): if THIS process
    # holds a native extension the sync must rewrite, defer NOW — after
    # the code swap, so only the dependency install is pending and the
    # next fresh launch completes it via the marker.
    _m()._abort_dependency_sync_if_self_locked(_windows_gateway_resume)
    #
    # Drop the core-install breadcrumb BEFORE touching the venv. If the
    # install is killed mid-flight (Ctrl-C, terminal close, WSL OOM), the
    # marker survives and the next ``hermes`` launch finishes the install
    # via ``_recover_from_interrupted_install``. Cleared after the core
    # ``.[all]`` install completes — lazy refresh uses a separate marker.
    _write_update_incomplete_marker()
    deps_current = _editable_install_is_current(
        git_cmd, _m().PROJECT_ROOT, pre_pull_sha
    )
    if deps_current:
        print("→ Python dependencies unchanged — skipping reinstall")
    else:
        print("→ Updating Python dependencies...")
    from hermes_cli.managed_uv import ensure_uv, update_managed_uv

    # Keep managed uv current — runs `uv self update` if we already have one.
    update_managed_uv()

    uv_bin = ensure_uv()

    pip_cmd = [sys.executable, "-m", "pip"]
    if not uv_bin:
        uv_bin = _ensure_uv_for_termux(pip_cmd)
    install_group = "all"

    if uv_bin:
        # Use official managed_python_env() isolation so third-party
        # UV_PYTHON_INSTALL_DIR (e.g. WorkBuddy) cannot hijack uv; then
        # point VIRTUAL_ENV at this install's venv.
        from hermes_cli.managed_uv import managed_python_env

        uv_env = managed_python_env()
        uv_env["VIRTUAL_ENV"] = str(_m().PROJECT_ROOT / "venv")
        if _m()._is_termux_env(uv_env):
            uv_env.pop("PYTHONPATH", None)
            uv_env.pop("PYTHONHOME", None)
            install_group = "termux-all"
            print("  → Termux detected: using uv + curated termux-all optional profile...")
        if not deps_current:
            if _m()._is_termux_env(uv_env) and _is_android_python():
                print("  → Termux/Android detected: prebuilding psutil with Linux source path compatibility...")
                _install_psutil_android_compat([uv_bin, "pip"], env=uv_env)
            _m()._install_python_dependencies_with_optional_fallback(
                [uv_bin, "pip"], env=uv_env, group=install_group
            )
    else:
        # sys.executable -m pip avoids PEP 668 'externally-managed-environment' errors.
        pip_cmd = [sys.executable, "-m", "pip"]
        _ensure_venv_pip(pip_cmd, sys.executable)
        if _m()._is_termux_env():
            install_group = "termux-all"
            print("  → Termux detected: using curated termux-all optional profile...")
        if not deps_current:
            if _m()._is_termux_env() and _is_android_python():
                print("  → Termux/Android detected: prebuilding psutil with Linux source path compatibility...")
                _install_psutil_android_compat(pip_cmd)
            _m()._install_python_dependencies_with_optional_fallback(pip_cmd, group=install_group)

    install_prefix = [uv_bin, "pip"] if uv_bin else pip_cmd
    lazy_env = uv_env if uv_bin else None

    if deps_current:
        # The verification normally runs inside the install we just
        # skipped. Run it here so a wrong skip self-heals into a real
        # install (both verifiers reinstall what they find missing)
        # instead of leaving a venv nobody checked.
        _m()._verify_core_dependencies_installed(
            install_prefix, env=lazy_env, group=install_group
        )
        _m()._verify_console_scripts_installed(install_prefix, env=lazy_env)

    # Core ``.[all]`` install finished. Clear the generic core breadcrumb
    # before the lazy-refresh phase — that phase uses its own marker so a
    # later lazy failure cannot be "healed" by clearing the core marker
    # based on a narrow 7-package import probe (#58004 review).
    _m()._clear_update_incomplete_marker()

    # The update process is still the old Python interpreter process. Run
    # one final cache/module refresh immediately before lazy backend
    # refresh, which imports newly-pulled modules that may depend on fresh
    # symbols in hermes_constants or lazy_deps. The dependency install
    # above may also have regenerated bytecode from build-cache copies —
    # this second sweep catches those stragglers (#60242, #65240).
    _sweep_bytecode_after_update(branch)
    _m()._reload_updated_runtime_modules()

    # Upgrade pip before lazy refreshes — stale pip can fail source builds
    # and leave partially-written packages (#57828).
    _write_lazy_refresh_incomplete_marker()
    _m()._upgrade_pip_before_lazy_refresh(install_prefix, env=lazy_env)

    # Lazy refresh can corrupt the venv when a backend install fails.
    # Clear the lazy marker only when refresh/repair is confirmed healthy.
    lazy_ok = _m()._refresh_active_lazy_features(
        install_prefix,
        env=lazy_env,
        features=active_lazy_features,
    )
    if lazy_ok:
        _m()._clear_lazy_refresh_incomplete_marker()
    else:
        print(
            "  ⚠ Lazy-refresh recovery incomplete — run `hermes` again "
            "to finish import-based venv repair."
        )

    _m()._restore_active_tool_dependencies(
        active_tool_dependencies,
        install_prefix,
        env=lazy_env,
    )

    # Heal the active memory provider's bridge packages last — the core
    # reinstall + lazy refresh above may have stripped or downgraded
    # plugin.yaml-declared deps that aren't in extras (#53272, #70636).
    _m()._refresh_active_memory_provider_dependencies()

    # All transient-ImportError sources have run, so a module that still
    # won't import is real breakage. Warn only — never roll back: `cannot
    # import name X` is also the stale-bytecode signature (#6207, #60242),
    # which _sweep_stale_bytecode_if_checkout_changed() self-heals next launch.
    import_ok, failing_module, import_error = _validate_critical_modules_import(
        _m().PROJECT_ROOT
    )
    if not import_ok:
        print()
        print(f"  ⚠ {failing_module} still fails to import after updating:")
        print(f"      {import_error}")
        print("    Run `hermes update` again — if it persists, reinstall:")
        print("    https://hermes-agent.nousresearch.com")
