"""Owned friendly commands against HTTPS and dashboard protocol fixtures, not Hermes."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys

import pytest

from tests.scripts.install.test_browser_distribution import (
    distribution,
    https,
    entry,
    terminal,  # noqa: F401
)
from tests.scripts.install.test_browser_setup import layout  # noqa: F401
from tests.scripts.install.test_browser_offline import (
    DASHBOARD_FIXTURE,
    release,
    sha,
    wait_for,
    pidfd_open,
    kill_fixture,  # noqa: F401
)
from tests.scripts.install.test_browser_maintenance import candidate, snapshot

pytestmark = [pytest.mark.linux_only, pytest.mark.live_system_guard_bypass]


def command(d, action, *args):
    return subprocess.run(
        [str(d["command"]), action, *args],
        env=d["env"],
        capture_output=True,
        text=True,
        timeout=20,
    )


def confirmed(d, action, *args, answer="yes\n"):
    return terminal([str(d["command"]), action, *args], d["env"], answer=answer)


def serve(d, archive, launcher=None):
    descriptor = json.loads((d["root"] / "CURRENT.json").read_bytes())
    for key, data in (
        ("archive", archive.read_bytes()),
        (
            "launcher",
            launcher.read_bytes()
            if launcher
            else (
                Path(__file__).resolve().parents[3] / "scripts/hermes-browser.py"
            ).read_bytes(),
        ),
    ):
        (d["root"] / descriptor[key]["name"]).write_bytes(data)
        descriptor[key].update(size=len(data), sha256=sha(data))
    (d["root"] / "CURRENT.json").write_text(json.dumps(descriptor))


@pytest.fixture
def installed(distribution):
    d = distribution
    code, output = terminal(entry(d), d["env"])
    assert code == 0, output
    return d


def test_repeat_verifies_original_selection_without_update(
    installed, release, tmp_path
):
    d = installed
    before = snapshot(d["base"])
    original_command = d["command"].read_bytes()
    serve(d, candidate(release, tmp_path, "later"))
    (d["home"] / ".hermes/active_profile").write_text("beta\n")
    requests = len(d["requests"])
    code, output = terminal(entry(d), d["env"])
    assert code == 0, output
    assert "Already installed" in output and "Type yes" not in output
    assert "/CURRENT.json" not in d["requests"][requests:]
    assert snapshot(d["base"]) == before
    assert d["command"].read_bytes() == original_command
    rejected = command(d, "setup", "--profile", "beta")
    assert rejected.returncode == 1
    assert snapshot(d["base"]) == before


def test_confirmed_update_rollback_uninstall_and_stable_reinstall(
    installed, release, tmp_path
):
    d = installed
    root = d["base"] / "installation"
    control = snapshot(d["base"] / "control")
    baseline = snapshot(root)
    with (d["home"] / ".hermes/hermes-agent/hermes_cli/main.py").open("a") as source:
        source.write("# Compatible fixture update; no receipt refresh.\n")
    preserved = snapshot(d["home"] / ".hermes")
    with (d["base"] / "installation.run").open("rb") as lock:
        inode = os.fstat(lock.fileno()).st_ino
        serve(d, candidate(release, tmp_path, "next", backend_match=False))
        for answer in ("no\n", "\x04", "\x03"):
            code, output = confirmed(d, "update", answer=answer)
            assert "Type yes" in output
            assert snapshot(root) == baseline
        code, output = confirmed(d, "update")
        assert code == 0, output
        assert output.count("Type yes") == 1
        assert snapshot(d["base"] / "control") == control
        assert (
            snapshot(
                d["base"]
                / "installation.history"
                / sha(release["archive"].read_bytes())
            )
            == baseline
        )
        assert "next" in command(d, "inspect").stdout
        code, output = confirmed(d, "rollback")
        assert code == 0, output
        assert snapshot(root) == baseline
        code, output = confirmed(d, "uninstall")
        assert code == 0, output
        assert not d["command"].exists()
        assert {p.name for p in d["base"].iterdir()} == {
            "command.lock",
            "installation.run",
        }
        assert (d["base"] / "installation.run").stat().st_ino == inode
        code, output = terminal(entry(d), d["env"])
        assert code == 0, output
        assert (d["base"] / "installation.run").stat().st_ino == inode
    assert snapshot(d["home"] / ".hermes") == preserved


@pytest.mark.parametrize("foreign", ["command", "control", "history", "installation"])
def test_uninstall_preserves_modified_or_foreign_files(installed, foreign):
    d = installed
    if foreign == "command":
        path = d["command"]
        path.write_bytes(path.read_bytes() + b"# modified\n")
    else:
        name = {
            "control": "control",
            "history": "installation.history",
            "installation": "installation",
        }[foreign]
        path = d["base"] / name / "foreign"
        path.parent.mkdir(exist_ok=True)
        path.write_text("mine")
    before = snapshot(d["base"])
    code, output = confirmed(d, "uninstall")
    assert code == 1, output
    assert "Type yes" not in output
    assert snapshot(d["base"]) == before
    assert path.exists()


@pytest.mark.parametrize("state", ["unknown", "no-tty"])
def test_maintenance_refuses_without_mutation(installed, release, tmp_path, state):
    d = installed
    archive = candidate(release, tmp_path, "next")
    serve(d, archive)
    if state == "unknown":
        path = d["base"] / "installation.run"
        record = json.loads(path.read_bytes())
        record.update(state="unknown", generation="a" * 32)
        path.write_text(json.dumps(record))
    before = snapshot(d["base"])
    requests = len(d["requests"])
    if state == "no-tty":
        result = subprocess.run(
            [str(d["command"]), "update"],
            env=d["env"],
            input="yes\n",
            capture_output=True,
            text=True,
            start_new_session=True,
        )
        code, output = result.returncode, result.stdout + result.stderr
        assert "terminal" in output.lower()
    else:
        code, output = confirmed(d, "update")
        assert "Type yes" not in output
    assert code == 1, output
    assert snapshot(d["base"]) == before
    if state == "unknown":
        assert len(d["requests"]) == requests


def test_current_controller_runs_restored_launcher_and_stop_survives_changed_ui(
    distribution, release, tmp_path
):
    d = distribution
    data = d["home"] / ".hermes"
    backend = data / "hermes-agent"
    (backend / "hermes_cli/main.py").write_text(DASHBOARD_FIXTURE)
    receipt = json.loads(release["receipt"].read_bytes())
    receipt["tested_backend"]["reference_files"]["hermes_cli/main.py"] = sha(
        DASHBOARD_FIXTURE.encode()
    )
    release["receipt"].write_text(json.dumps(receipt))
    marker = tmp_path / "old-launcher-executed"
    old = tmp_path / "legacy.py"
    old.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('old snapshot must never execute')\n"
    )
    archive = candidate(release, tmp_path, "legacy", launcher=old)
    serve(d, archive, old)
    # This fake backend has no dependencies/venv; explicitly select the test
    # interpreter whose existing YAML parser is used by startup preflight.
    code, output = terminal(
        ["sh", str(d["root"] / "install.sh"), "--python", sys.executable], d["env"]
    )
    assert code == 0, output
    root = d["base"] / "installation"
    legacy = snapshot(root)
    serve(d, candidate(release, tmp_path, "current"))
    assert confirmed(d, "update")[0] == 0
    assert confirmed(d, "rollback")[0] == 0
    assert snapshot(root) == legacy
    assert (root / "hermes-browser.py").read_bytes() == old.read_bytes()
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    stdout = tmp_path / "foreground.out"
    stderr = tmp_path / "foreground.err"
    pidfd = None
    with stdout.open("w") as out, stderr.open("w") as err:
        process = subprocess.Popen(
            [str(d["command"]), "start", "--port", str(port)],
            env=d["env"],
            stdout=out,
            stderr=err,
        )
        try:
            launch = data / f"launch-{port}.json"
            wait_for(lambda: launch.exists() or process.poll() is not None)
            assert launch.exists(), stderr.read_text()
            pid = json.loads(launch.read_text())["pid"]
            pidfd = pidfd_open(pid)
            assert os.readlink(f"/proc/{pid}/cwd") == str(backend)
            assert (
                int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
                == process.pid
            )
            (data / f"release-{port}").touch()
            wait_for(lambda: "ready" in command(d, "status").stdout)
            wait_for(
                lambda: f"Browser ready: http://127.0.0.1:{port}/" in stderr.read_text()
            )
            for action in ("update", "rollback", "uninstall"):
                code, output = confirmed(d, action)
                assert code == 1, output
                assert "Type yes" not in output
                assert "ready" in command(d, "status").stdout
            (next((root / "versions").iterdir()) / "web/index.html").write_text(
                "modified"
            )
            stopped = command(d, "stop")
            assert stopped.returncode == 0, stopped.stderr
            assert "stopped" in stopped.stdout
            assert process.wait(timeout=10) == 0
            assert not marker.exists()
        finally:
            if pidfd is not None:
                kill_fixture(pidfd)
                os.close(pidfd)
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=10)
