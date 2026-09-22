"""The browser harness must work without backend imports or live user state."""

import os
from pathlib import Path
import shutil
import shlex
import subprocess
import sys

import pytest


@pytest.fixture
def standalone_conftest(tmp_path):
    target = tmp_path / "project/tests/conftest.py"
    target.parent.mkdir(parents=True)
    shutil.copyfile(Path(__file__).with_name("conftest.py"), target)
    return target


def test_collection_isolates_home_and_credentials_without_backend(
    standalone_conftest, tmp_path
):
    assert Path.home().is_relative_to(tmp_path)
    assert os.environ["HERMES_TEST_ISOLATION"] == os.environ["HERMES_HOME"]
    home = tmp_path / "operator"
    home.mkdir()
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            """
import os, pathlib, runpy, sys
original = pathlib.Path.home()
runpy.run_path(sys.argv[1])
home = pathlib.Path.home()
assert home != original
assert home.is_dir()
assert pathlib.Path(os.environ['HERMES_HOME']).is_relative_to(home)
assert os.environ['HERMES_TEST_ISOLATION'] == os.environ['HERMES_HOME']
assert 'OPENAI_API_KEY' not in os.environ
assert 'HERMES_STATE_DB_GUARD_BYPASS' not in os.environ
assert 'HERMES_REAL_HOME' not in os.environ
assert not any(name.split('.')[0] in {'agent', 'hermes_cli', 'gateway', 'hermes_constants', 'hermes_state'} for name in sys.modules)
""",
            str(standalone_conftest),
        ],
        env={
            "HOME": str(home),
            "PATH": os.defpath,
            "OPENAI_API_KEY": "fake-sentinel",
            "HERMES_HOME": str(home / "custom-profile"),
            "HERMES_REAL_HOME": str(home),
            "HERMES_STATE_DB_GUARD_BYPASS": "1",
        },
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert list(home.iterdir()) == []


@pytest.mark.linux_only
def test_explicit_python_takes_precedence_over_local_venv(standalone_conftest):
    root = standalone_conftest.parent.parent
    source = Path(__file__).resolve().parents[1]
    (root / "scripts").mkdir()
    for name in ("run_tests.sh", "run_tests_parallel.py"):
        shutil.copyfile(source / "scripts" / name, root / "scripts" / name)
    (root / "tests/test_probe.py").write_text("def test_probe(): pass\n")
    local = root / ".venv/bin"
    local.mkdir(parents=True)
    (local / "activate").touch()
    marker = root / "local-python-used"
    wrapper = local / "python"
    wrapper.write_text(
        f"#!/bin/sh\nprintf used > {shlex.quote(str(marker))}\n"
        f'exec {shlex.quote(sys.executable)} "$@"\n'
    )
    wrapper.chmod(0o755)
    result = subprocess.run(
        ["bash", str(root / "scripts/run_tests.sh"), "-j", "1", "tests/test_probe.py"],
        env={
            "HOME": os.environ["HOME"],
            "PATH": os.defpath,
            "HERMES_PYTHON": sys.executable,
            "HERMES_TEST_FILE_RETRIES": "0",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert not marker.exists(), (
        "The installer venv overrode the explicitly selected plugin-test interpreter"
    )


@pytest.mark.linux_only
@pytest.mark.live_system_guard_bypass
def test_signal_and_os_guards_through_runner(standalone_conftest):
    child = subprocess.Popen(
        [sys.executable, "-I", "-c", "import time; time.sleep(30)"]
    )
    try:
        child.terminate()
        child.wait(timeout=10)
    finally:
        if child.poll() is None:
            child.kill()
            child.wait(timeout=10)

    root = standalone_conftest.parent.parent
    source = Path(__file__).resolve().parents[1]
    (root / "scripts").mkdir()
    for name in ("run_tests.sh", "run_tests_parallel.py"):
        shutil.copyfile(source / "scripts" / name, root / "scripts" / name)
    shutil.copyfile(source / "pyproject.toml", root / "pyproject.toml")
    (root / "tests/test_probe.py").write_text(
        """
import os
import signal
import pytest

# Capture the downstream primitives BEFORE fixtures wrap them: no destructive
# probe can reach a foreign process even if the guard regresses.
calls = []
os.kill = lambda *args: calls.append(args)
os.killpg = lambda *args: calls.append(args)

@pytest.mark.linux_only
@pytest.mark.live_system_guard_bypass
def test_signals():
    for pid in (os.getppid(), 0, -1, -os.getpgrp()):
        with pytest.raises(RuntimeError, match='outside the test process subtree'):
            os.kill(pid, signal.SIGTERM)
    with pytest.raises(RuntimeError, match='outside the test process subtree'):
        os.killpg(os.getpgrp(), signal.SIGTERM)
    assert calls == []
    os.kill(os.getppid(), 0)
    assert calls == [(os.getppid(), 0)]

@pytest.mark.macos_only
def test_macos():
    pytest.fail('macOS-only test ran on Linux')

@pytest.mark.windows_only
def test_windows():
    pytest.fail('Windows-only test ran on Linux')
""",
        encoding="utf-8",
    )
    result = subprocess.run(
        ["bash", str(root / "scripts/run_tests.sh"), "-j", "1", "tests/test_probe.py"],
        env={
            "HOME": os.environ["HOME"],
            "PATH": os.defpath,
            "HERMES_PYTHON": sys.executable,
            "HERMES_TEST_FILE_RETRIES": "0",
        },
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "1 tests passed, 0 failed" in result.stdout
    assert "2 skipped" in result.stdout
