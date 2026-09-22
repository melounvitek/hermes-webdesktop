"""Browser-test isolation only; no backend imports or runtime patches.

Use scripts/run_tests.sh for clean subprocess environments and process-group
cleanup. These fixtures prevent accidental home/credential inheritance and
foreign signals in the test interpreter; they are not a subprocess sandbox.
"""

import atexit
import os
from pathlib import Path
import sys
import tempfile

import psutil
import pytest


# Keep only interpreter/runner necessities, not credentials or Hermes settings.
_ENV_ALLOWLIST = {
    "PATH",
    "SYSTEMROOT",
    "WINDIR",
    "PYTEST_CURRENT_TEST",
    "PYTEST_VERSION",
    "PYTEST_DEBUG_TEMPROOT",
}


def _isolate_environment(home, patch):
    for name in list(os.environ):
        if name not in _ENV_ALLOWLIST:
            patch.delenv(name)
    hermes = home / ".hermes"
    hermes.mkdir(parents=True)
    for name, value in {
        "HOME": home,
        "USERPROFILE": home,
        "XDG_CONFIG_HOME": home / ".config",
        "XDG_CACHE_HOME": home / ".cache",
        "XDG_DATA_HOME": home / ".local/share",
        "XDG_STATE_HOME": home / ".local/state",
        "APPDATA": home / "AppData/Roaming",
        "LOCALAPPDATA": home / "AppData/Local",
        "HERMES_HOME": hermes,
        "HERMES_TEST_ISOLATION": hermes,
        "HERMES_DISABLE_LAZY_INSTALLS": "1",
        "AWS_EC2_METADATA_DISABLED": "true",
        "TZ": "UTC",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "PYTHONHASHSEED": "0",
    }.items():
        patch.setenv(name, str(value))


# Collection imports run before fixtures. On the supported installer host, also
# ignore TMPDIR: it may point inside an inherited live profile.
_session_home = tempfile.TemporaryDirectory(
    prefix="hermes-browser-tests-",
    dir="/tmp" if sys.platform.startswith("linux") else None,
)
atexit.register(_session_home.cleanup)
_isolate_environment(Path(_session_home.name), pytest.MonkeyPatch())


@pytest.fixture(autouse=True)
def _isolated_home(tmp_path, monkeypatch):
    _isolate_environment(tmp_path / "browser-test-home", monkeypatch)


@pytest.fixture(autouse=True)
def _guard_signals(monkeypatch):
    """Allow child cleanup, never broadcast or signal unrelated processes.

    The inherited live_system_guard_bypass marker intentionally has no effect.
    Optional integration fixtures must own and track their child lifetimes.
    """
    owner = os.getpid()
    real_kill = os.kill

    def owned(pid):
        if pid <= 0 or pid == owner:
            return False
        try:
            return any(parent.pid == owner for parent in psutil.Process(pid).parents())
        except psutil.NoSuchProcess:
            raise ProcessLookupError(pid) from None
        except psutil.AccessDenied:
            return False

    def kill(pid, sig):
        # On Windows signal 0 terminates; only POSIX uses it as a liveness probe.
        if not (sys.platform != "win32" and sig == 0 and pid > 0) and not owned(pid):
            raise RuntimeError(
                f"blocked signal to PID {pid}: outside the test process subtree"
            )
        return real_kill(pid, sig)

    monkeypatch.setattr(os, "kill", kill)
    if hasattr(os, "killpg"):
        real_killpg = os.killpg

        def killpg(pgid, sig):
            # A descendant group leader alone does not prove every member is ours.
            members = []
            if pgid > 0 and pgid != os.getpgrp():
                for process in psutil.process_iter():
                    try:
                        if os.getpgid(process.pid) == pgid:
                            members.append(process.pid)
                    except ProcessLookupError:
                        continue
            if not members or not all(owned(pid) for pid in members):
                raise RuntimeError(
                    f"blocked signal to PGID {pgid}: outside the test process subtree"
                )
            return real_killpg(pgid, sig)

        monkeypatch.setattr(os, "killpg", killpg)


def pytest_collection_modifyitems(items):
    for item in items:
        for marker, host in (
            ("linux_only", "linux"),
            ("macos_only", "darwin"),
            ("windows_only", "win32"),
        ):
            if item.get_closest_marker(marker) and not sys.platform.startswith(host):
                item.add_marker(
                    pytest.mark.skip(reason=f"{marker}; host is {sys.platform}")
                )
