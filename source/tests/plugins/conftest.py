"""The optional plugin tests require an explicitly selected, unmodified backend."""

import importlib.util
import os
from pathlib import Path
import subprocess
import sys

import pytest


SOURCE = Path(__file__).resolve().parents[2]
STOCK_MODULES = {
    "hermes_cli.web_server": "hermes_cli/web_server.py",
    "hermes_cli.pty_bridge": "hermes_cli/pty_bridge.py",
    "hermes_cli.pty_session": "hermes_cli/pty_session.py",
    "hermes_constants": "hermes_constants.py",
    "plugins.dashboard_auth.basic": "plugins/dashboard_auth/basic/__init__.py",
    "tools.environments.local": "tools/environments/local.py",
}


def pytest_addoption(parser):
    parser.addoption(
        "--backend-root", help="Absolute path to a clean stock Hermes Git checkout"
    )


@pytest.fixture(scope="session")
def backend_root(pytestconfig):
    value = pytestconfig.getoption("--backend-root")
    if not value or not Path(value).is_absolute():
        pytest.fail(
            "Plugin tests require --backend-root=/absolute/path/to/clean/stock-checkout"
        )
    root = Path(value).resolve()
    if root == SOURCE or root.is_relative_to(SOURCE):
        pytest.fail(
            "--backend-root must be a separate stock checkout, not bundled source/"
        )
    if (root / ".env").exists() or (root / ".env").is_symlink():
        pytest.fail("Stock checkout must not contain a root .env (including symlinks)")

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args],
            text=True,
            stderr=subprocess.STDOUT,
            timeout=15,
        ).strip()

    try:
        if Path(git("rev-parse", "--show-toplevel")).resolve() != root:
            pytest.fail("--backend-root must be the stock Git checkout root")
        if git("status", "--porcelain", "--untracked-files=all"):
            pytest.fail("--backend-root must be clean (tracked and untracked files)")
        git("ls-files", "--error-unmatch", *STOCK_MODULES.values())
    except (OSError, subprocess.SubprocessError) as exc:
        pytest.fail(f"Cannot validate stock backend: {exc}")
    return root


@pytest.fixture
def backend(backend_root, tmp_path, monkeypatch):
    # No checkout dotenv, user site, credentials or source/ fallback. Imports are
    # delayed until this fixture runs, after the installer's lightweight conftest.
    home = tmp_path / "backend-home"
    hermes = home / ".hermes"
    hermes.mkdir(parents=True)
    path = os.environ.get("PATH", os.defpath)
    for name in list(os.environ):
        monkeypatch.delenv(name)
    for name, value in {
        "HOME": str(home),
        "HERMES_HOME": str(hermes),
        "PATH": path,
        "LANG": "C.UTF-8",
        "TZ": "UTC",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONNOUSERSITE": "1",
        "HERMES_DISABLE_LAZY_INSTALLS": "1",
        "AWS_EC2_METADATA_DISABLED": "true",
        "HF_HUB_OFFLINE": "1",
        "HTTP_PROXY": "http://127.0.0.1:9",
        "HTTPS_PROXY": "http://127.0.0.1:9",
        "ALL_PROXY": "http://127.0.0.1:9",
        "NO_PROXY": "127.0.0.1,localhost,::1",
    }.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, "dont_write_bytecode", True)
    monkeypatch.setattr(
        sys,
        "path",
        [entry for entry in sys.path if entry and Path(entry).resolve() != SOURCE],
    )
    monkeypatch.syspath_prepend(str(backend_root))
    monkeypatch.chdir(home)
    for name, relative in STOCK_MODULES.items():
        spec = importlib.util.find_spec(name)
        expected = backend_root / relative
        if spec is None or not spec.origin or Path(spec.origin).resolve() != expected:
            pytest.fail(f"Stock import escaped --backend-root: {name}: {spec}")
    return backend_root
