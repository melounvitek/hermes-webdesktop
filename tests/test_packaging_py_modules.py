"""Invariant: every root-level module the packaged code imports ships in the wheel.

The repo keeps a handful of single-file modules at the repository root
(``hermes_state.py``, ``run_agent.py``, ``cli.py``, ...). setuptools'
``packages.find`` never picks those up, so ``[tool.setuptools] py-modules``
in pyproject.toml has to name each one explicitly. When a root module is
split into siblings (``hermes_state_*``) and the list is not updated, the
source tree still works (cwd is on ``sys.path``) but the built wheel/sdist
and the uv2nix sealed venv raise ``ModuleNotFoundError`` on the very first
``import hermes_state``.

This test derives the required set from the code rather than freezing a
list: parse pyproject, walk every import (top-level *and* lazy/function-
scoped) in the packaged root modules and packaged packages, and require
that any import resolving to a root-level ``*.py`` file is declared.
"""

from __future__ import annotations

import ast
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    return tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def _declared_py_modules() -> set[str]:
    return set(_pyproject()["tool"]["setuptools"]["py-modules"])


def _packaged_package_roots() -> list[Path]:
    include = _pyproject()["tool"]["setuptools"]["packages"]["find"]["include"]
    roots = []
    for pattern in include:
        top = pattern.split(".", 1)[0]
        if "*" in top:
            continue
        path = REPO_ROOT / top
        if path.is_dir() and path not in roots:
            roots.append(path)
    return roots


def _root_module_names() -> set[str]:
    return {p.stem for p in REPO_ROOT.glob("*.py")}


def _imported_top_names(path: Path) -> set[str]:
    """All top-level names imported anywhere in ``path`` (absolute imports only)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name.split(".", 1)[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            names.add(node.module.split(".", 1)[0])
    return names


def _packaged_source_files() -> list[Path]:
    files = [REPO_ROOT / f"{name}.py" for name in _declared_py_modules()]
    for root in _packaged_package_roots():
        files.extend(root.rglob("*.py"))
    return [f for f in files if f.is_file()]


def test_declared_py_modules_exist_at_repo_root():
    missing = sorted(n for n in _declared_py_modules() if not (REPO_ROOT / f"{n}.py").is_file())
    assert not missing, f"py-modules names files that do not exist at the repo root: {missing}"


def test_every_root_module_imported_by_packaged_code_is_in_py_modules():
    declared = _declared_py_modules()
    root_modules = _root_module_names()

    # importer -> set of undeclared root modules it imports
    offenders: dict[str, set[str]] = {}
    for src in _packaged_source_files():
        needed = _imported_top_names(src) & root_modules
        undeclared = needed - declared
        if undeclared:
            offenders[str(src.relative_to(REPO_ROOT))] = undeclared

    assert not offenders, (
        "Root-level modules are imported by packaged code but missing from "
        "[tool.setuptools] py-modules in pyproject.toml. The source tree hides "
        "this (cwd is on sys.path); the built wheel / uv2nix sealed venv will "
        "fail with ModuleNotFoundError. Add them to py-modules:\n"
        + "\n".join(f"  {importer}: {sorted(mods)}" for importer, mods in sorted(offenders.items()))
    )
