"""Updates requested through the mailbox of the real, running launcher.

Only a certificate-verified loopback issuer and the disposable protocol
dashboard run; the installation lives in a temporary home.
"""

import http.client
import http.server
import json
import os
from pathlib import Path
import select
import socket
import stat
import subprocess
import sys
import threading
import time

import pytest

from tests.scripts.install.test_browser_commands import command, serve
from tests.scripts.install.test_browser_distribution import (
    distribution,  # noqa: F401
    https,  # noqa: F401
    terminal,
)
from tests.scripts.install.test_browser_maintenance import candidate, snapshot
from tests.scripts.install.test_browser_offline import (
    DASHBOARD_FIXTURE,
    kill_fixture,
    pidfd_open,
    release,  # noqa: F401
    run,
    sha,
    wait_for,
)
from tests.scripts.install.test_browser_setup import layout  # noqa: F401

pytestmark = pytest.mark.linux_only


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


@pytest.fixture
def ready(distribution):
    d = distribution
    (d["home"] / ".hermes/hermes-agent/hermes_cli/main.py").write_text(
        DASHBOARD_FIXTURE
    )
    # This fake backend has no dependencies/venv; explicitly select the test
    # interpreter whose existing YAML parser is used by startup preflight.
    code, output = terminal(
        ["sh", str(d["root"] / "install.sh"), "--python", sys.executable], d["env"]
    )
    assert code == 0, output
    return {
        **d,
        "installation": d["base"] / "installation",
        "mailbox": d["base"] / "installation.updates",
    }


@pytest.fixture
def leftover(ready):
    """What a crashed launcher leaves behind; request it before `live`."""
    mailbox = ready["mailbox"]
    mailbox.mkdir(mode=0o700)
    (mailbox / "request.json").write_bytes(request("0" * 32))
    (mailbox / "status.json").write_text(
        json.dumps({"id": "0" * 32, "state": "running"})
    )


@pytest.fixture
def live(ready, tmp_path):
    d = ready
    data = d["home"] / ".hermes"
    port = free_port()
    stderr = tmp_path / "foreground.err"
    pidfd = None
    with (tmp_path / "foreground.out").open("w") as out, stderr.open("w") as err:
        process = subprocess.Popen(
            [str(d["command"]), "start", "--port", str(port)],
            env=d["env"],
            cwd=d["home"],
            stdout=out,
            stderr=err,
        )
        try:
            launch = data / f"launch-{port}.json"
            wait_for(lambda: launch.exists() or process.poll() is not None)
            assert launch.exists(), stderr.read_text()
            pid = json.loads(launch.read_text())["pid"]
            pidfd = pidfd_open(pid)
            assert (
                int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])
                == process.pid
            )
            (data / f"release-{port}").touch()
            wait_for(lambda: "ready" in command(d, "status").stdout)
            yield {**d, "process": process, "pid": pid, "pidfd": pidfd, "port": port}
        finally:
            if pidfd is not None:
                kill_fixture(pidfd)
                os.close(pidfd)
            if process.poll() is None:
                process.terminate()
                process.wait(timeout=20)


@pytest.fixture
def issuer(monkeypatch):
    """Hold the issuer's answers while cleared; an update then stays in progress."""
    gate = threading.Event()
    gate.set()
    answer = http.server.SimpleHTTPRequestHandler.do_GET

    def held(handler):
        gate.wait(60)
        answer(handler)

    monkeypatch.setattr(http.server.SimpleHTTPRequestHandler, "do_GET", held)
    yield gate
    gate.set()


def ask(d, content):
    # Stock Hermes writes a sibling temporary file and renames it into place.
    staged = d["mailbox"] / ".request.json.tmp"
    staged.write_bytes(content)
    os.replace(staged, d["mailbox"] / "request.json")


def request(job):
    return json.dumps({"id": job}).encode()


def status(d):
    path = d["mailbox"] / "status.json"
    return json.loads(path.read_bytes()) if path.exists() else None


def finished(d, job):
    def final():
        current = status(d)
        return current and current["id"] == job and current["state"] != "running"

    wait_for(final)
    return status(d)


def served(d, path):
    connection = http.client.HTTPConnection("127.0.0.1", d["port"], timeout=5)
    try:
        connection.request("GET", path)
        return connection.getresponse().read()
    finally:
        connection.close()


def controller(d):
    result = run("status", "--install-root", d["installation"])
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_request_switches_the_served_build_without_restarting_the_dashboard(
    live, issuer, release, tmp_path
):
    d = live
    root, mailbox = d["installation"], d["mailbox"]
    history = d["base"] / "installation.history"
    assert stat.S_IMODE(mailbox.stat().st_mode) == 0o700
    assert not list(mailbox.iterdir())
    original = snapshot(root)
    archive = candidate(release, tmp_path, "next")
    serve(d, archive)

    issuer.clear()
    ask(d, request("a" * 32))
    wait_for(lambda: status(d) == {"id": "a" * 32, "state": "running"})
    assert not (mailbox / "request.json").exists()
    assert snapshot(root) == original
    issuer.set()
    assert finished(d, "a" * 32) == {
        "id": "a" * 32,
        "state": "updated",
        "release": "next",
    }
    assert stat.S_IMODE((mailbox / "status.json").stat().st_mode) == 0o600
    assert {path.name for path in mailbox.iterdir()} == {"status.json"}

    receipt = json.loads((root / "installation.json").read_bytes())
    assert receipt["archive_sha256"] == sha(archive.read_bytes())
    assert (root / "web/index.html").read_bytes() == b"<h1>next</h1>"
    assert snapshot(history / sha(release["archive"].read_bytes())) == original
    # The same dashboard process serves the new build and is reported with it.
    assert served(d, "/index.html") == b"<h1>next</h1>"
    assert not select.select([d["pidfd"]], [], [], 0)[0]
    info = controller(d)
    assert (info["state"], info["pid"]) == ("ready", d["pid"])
    assert info["receipt"] == receipt

    before = snapshot(root), snapshot(history)
    ask(d, request("b" * 32))
    assert finished(d, "b" * 32) == {
        "id": "b" * 32,
        "state": "current",
        "release": "next",
    }
    assert (snapshot(root), snapshot(history)) == before
    assert not select.select([d["pidfd"]], [], [], 0)[0]

    stopped = command(d, "stop")
    assert stopped.returncode == 0, stopped.stderr
    assert d["process"].wait(timeout=10) == 0
    assert {path.name for path in d["base"].iterdir()} == {
        "command.lock",
        "control",
        "installation",
        "installation.history",
        "installation.run",
    }


def test_invalid_requests_are_ignored_and_failures_change_nothing(
    leftover, live, release, tmp_path
):
    d = live
    root, mailbox = d["installation"], d["mailbox"]
    assert not list(mailbox.iterdir())
    original = snapshot(root)
    serve(d, candidate(release, tmp_path, "next"))
    outside = tmp_path / "outside.json"
    outside.write_bytes(request("c" * 32))
    for content in (
        b"{",
        request("C" * 32),
        json.dumps({"id": "c" * 32, "force": True}).encode(),
        b" " * 4096 + request("c" * 32),
    ):
        ask(d, content)
        wait_for(lambda: not (mailbox / "request.json").exists())
    (mailbox / "request.json").symlink_to(outside)
    wait_for(lambda: not os.path.lexists(mailbox / "request.json"))
    assert outside.read_bytes() == request("c" * 32)
    # Nor may a request that cannot be removed cost a descriptor every second.
    descriptors = Path(f"/proc/{d['process'].pid}/fd")
    opened = len(list(descriptors.iterdir()))
    (mailbox / "request.json").mkdir()
    time.sleep(2.2)
    assert len(list(descriptors.iterdir())) == opened
    (mailbox / "request.json").rmdir()

    descriptor = json.loads((d["root"] / "CURRENT.json").read_bytes())
    published = descriptor["archive"]["sha256"]
    descriptor["archive"]["sha256"] = "0" * 64
    (d["root"] / "CURRENT.json").write_text(json.dumps(descriptor))
    ask(d, request("d" * 32))
    failed = finished(d, "d" * 32)
    assert failed == {"id": "d" * 32, "state": "failed", "error": failed["error"]}
    assert "SHA-256 mismatch" in failed["error"]
    descriptor["archive"]["sha256"] = published
    (d["root"] / "CURRENT.json").write_text(json.dumps(descriptor))

    source = d["base"] / "control/source.json"
    trusted = source.read_bytes()
    source.write_text(json.dumps({"source": "https://localhost:1/CURRENT.json"}))
    ask(d, request("e" * 32))
    assert "Modified controller" in finished(d, "e" * 32)["error"]
    source.write_bytes(trusted)
    # Updates never overlap, so the leftover, an invalid request or the modified
    # controller would have fetched as well: only setup and one request did.
    assert d["requests"].count("/CURRENT.json") == 2

    (d["home"] / ".hermes/.env").write_text("TOKEN=no-newline")
    ask(d, request("f" * 32))
    failed = finished(d, "f" * 32)
    assert failed == {"id": "f" * 32, "state": "failed", "error": failed["error"]}
    assert "Dotenv normalization required" in failed["error"]

    assert snapshot(root) == original
    assert {path.name for path in d["base"].iterdir()} == {
        "command.lock",
        "control",
        "installation",
        "installation.run",
        "installation.updates",
    }
    assert served(d, "/index.html") == (release["web"] / "index.html").read_bytes()
    assert not select.select([d["pidfd"]], [], [], 0)[0]
    assert controller(d)["pid"] == d["pid"]


def test_stop_waits_for_the_update_in_progress(live, issuer, release, tmp_path):
    d = live
    root = d["installation"]
    archive = candidate(release, tmp_path, "next")
    serve(d, archive)
    issuer.clear()
    ask(d, request("f" * 32))
    wait_for(lambda: status(d) == {"id": "f" * 32, "state": "running"})

    # The dashboard exits, yet the lock is kept until the switch has finished.
    stopping = command(d, "stop", "--timeout", "1")
    assert stopping.returncode == 1, stopping.stdout
    assert "stopping" in stopping.stdout
    assert select.select([d["pidfd"]], [], [], 5)[0]
    assert d["process"].poll() is None
    issuer.set()
    assert d["process"].wait(timeout=20) == 0

    assert json.loads((root / "installation.json").read_bytes())[
        "archive_sha256"
    ] == sha(archive.read_bytes())
    assert "stopped" in command(d, "status").stdout
    assert command(d, "inspect").returncode == 0
    assert {path.name for path in d["base"].iterdir()} == {
        "command.lock",
        "control",
        "installation",
        "installation.history",
        "installation.run",
    }


@pytest.mark.parametrize("unsafe", ["mode", "symlink"])
def test_start_refuses_a_mailbox_that_is_not_private(ready, tmp_path, unsafe):
    d = ready
    kept = tmp_path / "elsewhere"
    if unsafe == "symlink":
        kept.mkdir(mode=0o700)
        d["mailbox"].symlink_to(kept, target_is_directory=True)
    else:
        kept = d["mailbox"]
        kept.mkdir()
        kept.chmod(0o750)
    (kept / "request.json").write_bytes(request("0" * 32))
    # The short timeout only bounds a regression that would launch the dashboard.
    refused = command(d, "start", "--port", str(free_port()), "--timeout", "2")
    assert refused.returncode == 1, refused.stdout
    assert "Unsafe update mailbox" in refused.stderr
    assert (kept / "request.json").read_bytes() == request("0" * 32)
    assert not list((d["home"] / ".hermes").glob("launch*.json"))
