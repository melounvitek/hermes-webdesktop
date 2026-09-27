"""Native user-systemd lifecycle proof; only disposable protocol fixtures run."""

import fcntl
import json
import os
from pathlib import Path
import select
import shutil
import socket
import subprocess
import sys
import uuid

import pytest

from tests.scripts.install.test_browser_maintenance import snapshot
from tests.scripts.install.test_browser_offline import (
    CLI,
    dashboard,  # noqa: F401
    lifecycle,
    pidfd_open,
    release,  # noqa: F401
    sha,
    state_is,
    wait_for,
)

# All mutating systemctl calls name our UUID transient unit, never installed units.
pytestmark = [pytest.mark.linux_only, pytest.mark.live_system_guard_bypass]


@pytest.fixture
def user_systemd():
    if not all(shutil.which(tool) for tool in ("systemctl", "systemd-run")):
        pytest.skip("Native user systemd tools unavailable")
    runtime = Path(f"/run/user/{os.getuid()}")
    if not (runtime / "bus").is_socket():
        pytest.skip("Native user systemd bus unavailable")
    # The canonical runner strips DBUS variables. Restore only subprocess-local
    # routing, not os.environ or the user manager's activation environment.
    env = {
        "PATH": "/usr/bin:/bin",
        "XDG_RUNTIME_DIR": str(runtime),
        "DBUS_SESSION_BUS_ADDRESS": f"unix:path={runtime}/bus",
        "LC_ALL": "C",
    }

    def command(*args):
        return subprocess.run(args, env=env, capture_output=True, text=True, timeout=30)

    probe = command("systemctl", "--user", "show", "--property=Version")
    if probe.returncode:
        pytest.skip(f"Native user systemd manager unavailable: {probe.stderr}")
    return command


@pytest.fixture
def systemd_controller(user_systemd, dashboard, tmp_path, record_property):
    r = dashboard
    unit = f"hermes-browser-test-{uuid.uuid4().hex}.service"
    before = {key: snapshot(r[key]) for key in ("backend", "home")}
    artifacts = set()
    pidfds = []
    created = False

    def ctl(*args):
        return user_systemd("systemctl", "--user", *args, unit)

    def properties():
        result = ctl("show")
        assert result.returncode == 0, result.stderr
        return dict(line.split("=", 1) for line in result.stdout.splitlines())

    def start():
        nonlocal created
        # Transient units are unloaded after stop; explicitly recreate the same
        # disposable definition for a stopped->started cycle, not a disk unit.
        wait_for(lambda: properties()["LoadState"] == "not-found")
        with socket.socket() as probe:
            probe.bind(("127.0.0.1", 0))
            port = probe.getsockname()[1]
        marker = r["home"] / f"launch-{port}.json"
        handshake = r["home"] / f"release-{port}"
        artifacts.update((marker, handshake, r["home"] / "launch.json"))
        marker.unlink(missing_ok=True)
        handshake.unlink(missing_ok=True)
        created = True
        result = user_systemd(
            "systemd-run",
            "--user",
            f"--unit={unit}",
            "--property=Type=simple",
            "--property=Restart=no",
            "--property=KillMode=mixed",
            "--property=TimeoutStopSec=15s",
            f"--property=WorkingDirectory={tmp_path}",
            f"--property=StandardOutput=append:{tmp_path / 'controller.stdout'}",
            f"--property=StandardError=append:{tmp_path / 'controller.stderr'}",
            # Do not import credentials/config from the host manager environment.
            "/usr/bin/env",
            "-i",
            f"HOME={tmp_path}",
            "PATH=/usr/bin:/bin",
            "LANG=C.UTF-8",
            "TZ=UTC",
            sys.executable,
            "-I",
            "-S",
            str(CLI),
            "start",
            "--install-root",
            str(r["dest"]),
            "--port",
            str(port),
            "--timeout",
            "15",
        )
        assert result.returncode == 0, result.stderr
        wait_for(marker.exists)
        launch = json.loads(marker.read_text())
        fd = pidfd_open(launch["pid"])
        pidfds.append(fd)
        props = properties()
        assert props["Type"] == "simple"
        assert props["Restart"] == "no"
        assert props["KillMode"] == "mixed"
        assert props["ActiveState"] == "active"
        assert launch["module"] == str(r["backend"] / "hermes_cli/main.py")
        proc = Path(f"/proc/{launch['pid']}/stat").read_text().rsplit(")", 1)[1].split()
        assert int(proc[1]) == int(props["MainPID"]) > 0
        # Pin the actual fixture child before allowing it to serve readiness.
        handshake.touch()
        wait_for(lambda: state_is(r, "ready"), timeout=20)
        status = json.loads(lifecycle(r, "status").stdout)
        assert status["pid"] == launch["pid"]
        return status, fd

    assert properties()["LoadState"] == "not-found"
    try:
        yield r, start, ctl, properties
    finally:
        if created:
            props = properties()
            if props["LoadState"] != "not-found":
                stopped = ctl("stop")
                assert stopped.returncode == 0, stopped.stderr
                if properties()["ActiveState"] == "failed":
                    reset = ctl("reset-failed")
                    assert reset.returncode == 0, reset.stderr
            wait_for(lambda: properties()["LoadState"] == "not-found")
        for fd in pidfds:
            try:
                assert select.select([fd], [], [], 5)[0], (
                    "Fixture child survived cleanup"
                )
            finally:
                os.close(fd)
        # Only the protocol fixture's explicit handshake/output files are exempt
        # from preservation. Any other new file or changed config fails equality.
        for path in artifacts:
            path.unlink(missing_ok=True)
        after = {key: snapshot(r[key]) for key in before}
        for key in before:
            record_property(f"{key}_before_sha256", sha(repr(before[key]).encode()))
            record_property(f"{key}_after_sha256", sha(repr(after[key]).encode()))
        assert after == before


def test_systemd_foreground_stop_releases_lock_and_explicit_restart(systemd_controller):
    r, start, ctl, properties = systemd_controller
    first, child = start()
    control = r["dest"].with_name(r["dest"].name + ".run")
    with control.open("rb") as retained:
        inode = os.fstat(retained.fileno()).st_ino
        with pytest.raises(BlockingIOError):
            fcntl.flock(retained, fcntl.LOCK_EX | fcntl.LOCK_NB)
        duplicate = lifecycle(r, "start", "--timeout", "2")
        assert duplicate.returncode != 0
        assert (
            json.loads(lifecycle(r, "status").stdout)["generation"]
            == first["generation"]
        )
        assert not select.select([child], [], [], 0)[0]

        stopped = ctl("stop")
        assert stopped.returncode == 0, stopped.stderr
        assert state_is(r, "stopped")
        assert select.select([child], [], [], 5)[0]
        fcntl.flock(retained, fcntl.LOCK_EX | fcntl.LOCK_NB)
        fcntl.flock(retained, fcntl.LOCK_UN)

        second, child = start()
        assert second["generation"] != first["generation"]
        assert control.stat().st_ino == inode
        with pytest.raises(BlockingIOError):
            fcntl.flock(retained, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert properties()["NRestarts"] == "0"
        stopped = ctl("stop")
        assert stopped.returncode == 0, stopped.stderr
        assert state_is(r, "stopped")
        assert select.select([child], [], [], 5)[0]
        fcntl.flock(retained, fcntl.LOCK_EX | fcntl.LOCK_NB)


def test_systemd_controller_crash_retains_unknown_ownership_without_restart(
    systemd_controller,
):
    r, start, ctl, properties = systemd_controller
    ready, child = start()
    control = r["dest"].with_name(r["dest"].name + ".run")
    record = control.read_bytes()
    killed = ctl("kill", "--kill-whom=main", "--signal=SIGKILL")
    assert killed.returncode == 0, killed.stderr
    wait_for(lambda: properties()["ActiveState"] == "failed", timeout=20)
    props = properties()
    assert props["Result"] == "signal"
    assert props["Restart"] == "no" and props["NRestarts"] == "0"
    assert props["MainPID"] == "0"
    # mixed lets the controller own graceful shutdown, while systemd contains
    # remaining fixture children after a crash. That does not repair ownership.
    assert select.select([child], [], [], 5)[0]
    assert state_is(r, "unknown")
    assert lifecycle(r, "stop").returncode != 0
    duplicate = lifecycle(r, "start", "--timeout", "2")
    assert duplicate.returncode != 0
    assert "Ownership unknown" in duplicate.stderr
    assert control.read_bytes() == record
    assert json.loads(record)["generation"] == ready["generation"]
    assert properties()["ActiveState"] == "failed"
    assert properties()["NRestarts"] == "0"
