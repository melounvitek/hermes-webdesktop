"""Hermes-owned PYTHONPATH stripping for child processes.

Launchers prepend the repo root and the Hermes venv's site-packages so the
backend can ``import tools``; leaked into a child Python of a DIFFERENT version
they load the backend's C extensions and crash (numpy, PIL, cryptography). Only
entries proven Hermes-owned by *path provenance* are removed — never by a
cross-version heuristic — so user entries survive. Module state
(``_hermes_repo_root_aliases``, ``_in_venv``, ``_hermes_site_packages``) stays
in ``tools.environments.local`` and is read via :func:`_state` so tests that
monkeypatch it there keep working.
"""

import logging
import os
import platform
import sys
from pathlib import Path

from tools.environments.local_env_policy import _ACTIVE_VENV_MARKER_VARS

_IS_WINDOWS = platform.system() == "Windows"

logger = logging.getLogger("tools.environments.local")


def _state():
    """Return the ``tools.environments.local`` module (owner of the caches)."""
    from tools.environments import local

    return local


def _same_path(left: Path, right: Path) -> bool:
    """Compare path spellings with host filesystem case semantics."""
    left_parts = [os.path.normcase(part) for part in left.parts]
    right_parts = [os.path.normcase(part) for part in right.parts]
    return left_parts == right_parts


def _build_hermes_repo_root_aliases(
    resolved_root: Path,
    lexical_root: Path,
    configured_home: Path,
) -> tuple[Path, ...]:
    """Exact repo-root spellings emitted by Hermes launchers. Mirrors
    ``gateway_windows._preserve_hermes_home_path`` (physical path under the
    resolved HERMES_HOME -> configured spelling) so a junction-backed install
    matches without treating arbitrary HERMES_HOME descendants as Hermes-owned.
    A repo-level junction (possibly cross-drive) is accepted only when a strict
    resolve proves <root>/<repo dirname> is the physical root (fail-closed)."""
    aliases: list[Path] = []

    def add(candidate: Path) -> None:
        if not any(_same_path(candidate, existing) for existing in aliases):
            aliases.append(candidate)

    add(resolved_root)
    add(lexical_root)

    # Profile re-home: with --profile the configured home is <root>/profiles/<name>
    # and the repo lives beside the profiles dir, so derive the root lexically the
    # same way get_default_hermes_root() does and map against it too.
    home_candidates = [configured_home]
    if configured_home.parent.name == "profiles":
        home_candidates.append(configured_home.parent.parent)

    for home in home_candidates:
        try:
            resolved_home = home.resolve()
            home_key = os.path.normcase(str(resolved_home))
            root_key = os.path.normcase(str(resolved_root))
            if os.path.commonpath([home_key, root_key]) == home_key:
                relative_root = os.path.relpath(str(resolved_root), str(resolved_home))
                add(home / relative_root)
        except (OSError, ValueError):
            pass

    # Repo-level junction recovery (commonpath raises across drives, so the
    # home-relative mapping above cannot express a cross-drive link).
    for home in home_candidates:
        repo_candidate = home / resolved_root.name
        try:
            if repo_candidate.resolve(strict=True) == resolved_root.resolve(strict=True):
                add(repo_candidate)
        except OSError:
            pass

    return tuple(aliases)


def _validated_runtime_venv(env: dict) -> Path | None:
    """Producer-owned runtime venv identified by VIRTUAL_ENV, or None. The
    variable alone is not provenance (users carry unrelated venvs): require the
    legacy Windows base-Python producer's exact ``<repo>/venv`` layout AND a
    real ``pyvenv.cfg``."""
    value = env.get("VIRTUAL_ENV")
    if not value:
        return None

    candidate = Path(value)
    aliases = _state()._hermes_repo_root_aliases
    if not any(_same_path(candidate, repo_root / "venv") for repo_root in aliases):
        return None

    try:
        if not (candidate / "pyvenv.cfg").is_file():
            return None
    except OSError:
        return None

    return candidate


def _get_hermes_site_packages(env: dict) -> list[Path]:
    """Exact site-packages dirs owned by the Hermes runtime (cached):
    ``site.getsitepackages()`` with a ``sys.prefix`` fallback, plus a validated
    Windows base-interpreter launch's ``VIRTUAL_ENV/Lib/site-packages``."""
    local = _state()
    if local._hermes_site_packages is not None:
        result = list(local._hermes_site_packages)
    else:
        result = []
        if local._in_venv:
            try:
                import site
                for sp in site.getsitepackages():
                    result.append(Path(sp))
            except Exception:
                pass

            if not result:
                if _IS_WINDOWS:
                    result.append(Path(sys.prefix) / "Lib" / "site-packages")
                else:
                    pyver = f"python{sys.version_info[0]}.{sys.version_info[1]}"
                    result.append(Path(sys.prefix) / "lib" / pyver / "site-packages")

        local._hermes_site_packages = list(result)

    runtime_venv = _validated_runtime_venv(env)
    if runtime_venv is not None:
        runtime_site_packages = runtime_venv / "Lib" / "site-packages"
        if not any(_same_path(runtime_site_packages, existing) for existing in result):
            result.append(runtime_site_packages)

    return result


def _strip_hermes_owned_pythonpath_and_runtime_markers(env: dict) -> None:
    """Strip Hermes-owned PYTHONPATH entries, then the runtime marker vars.

    Ordering is load-bearing: PYTHONPATH filtering must run BEFORE the markers
    are removed so a validated Windows base-interpreter launch
    (VIRTUAL_ENV -> <repo>/venv) can still prove ownership.
    """
    _strip_hermes_owned_pythonpath(env)
    for _marker in _ACTIVE_VENV_MARKER_VARS:
        env.pop(_marker, None)


def _strip_hermes_owned_pythonpath(env: dict) -> None:
    """Remove Hermes-owned PYTHONPATH entries. Only exact matches of the repo
    root (any launcher spelling) and runtime site-packages are stripped — never
    children/descendants, which are user paths. Empty components (= cwd) and
    everything else are preserved byte-for-byte."""
    pp = env.get("PYTHONPATH")
    if not pp:
        return

    hermes_site_packages = _get_hermes_site_packages(env)
    repo_roots = _state()._hermes_repo_root_aliases

    kept: list[str] = []
    stripped: list[str] = []

    for entry in pp.split(os.pathsep):
        if entry == "":
            kept.append(entry)
            continue

        entry_path = Path(entry)
        owned = any(_same_path(entry_path, sp) for sp in hermes_site_packages) or any(
            _same_path(entry_path, repo_root) for repo_root in repo_roots
        )
        (stripped if owned else kept).append(entry)

    if kept:
        env["PYTHONPATH"] = os.pathsep.join(kept)
    else:
        env.pop("PYTHONPATH", None)

    if stripped:
        logger.debug("Stripped Hermes-owned entries from PYTHONPATH: %s", stripped)
