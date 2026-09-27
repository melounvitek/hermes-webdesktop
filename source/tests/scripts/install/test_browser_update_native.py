"""Native Round 2: real updater/controller, disposable protocol backend and units.

No installed service, backend, plugin, or manager environment is changed. This
proves lifecycle/update wiring, not compatibility with stock Hermes APIs.
"""

from contextlib import closing
from copy import deepcopy
from dataclasses import replace
import importlib
import json
import os
from pathlib import Path
import select
import socket
import sys
import threading
from types import SimpleNamespace
import uuid

import pytest

from tests.scripts.install.test_browser_maintenance import candidate, history, snapshot
from tests.scripts.install.test_browser_offline import (
    CLI,
    DASHBOARD_FIXTURE,
    kill_fixture,
    pidfd_open,
    release,  # noqa: F401
    run,
    sha,
    wait_for,
)
from tests.scripts.install.test_browser_systemd import user_systemd  # noqa: F401

# The bypass is limited to this file's UUID units and temporary fixture children.
pytestmark = [pytest.mark.linux_only, pytest.mark.live_system_guard_bypass]
ACTOR = ("fixture", None, "administrator")


@pytest.fixture
def native_update(user_systemd, release, tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(CLI.parent))
    updater = importlib.import_module("browser_updater")
    jobs = importlib.import_module("browser_update_jobs")
    service = importlib.import_module("browser_update_service")
    r = release
    # Configure failure before packing the base; never edit a backend during a job.
    source = DASHBOARD_FIXTURE.replace(
        "mode = (root/'mode').read_text() if (root/'mode').exists() else 'normal'",
        "mode = 'exit' if (Path(os.environ['HERMES_WEB_DIST'])/'index.html').read_text() == '<h1>fails-to-start</h1>' else 'normal'",
    ).replace("('release-' + str(a.port))", "('release-' + str(os.getpid()))")
    (r["backend"] / "hermes_cli/main.py").write_text(source)
    receipt = json.loads(r["receipt"].read_bytes())
    receipt["tested_backend"]["reference_files"]["hermes_cli/main.py"] = sha(
        source.encode()
    )
    r["receipt"].write_text(json.dumps(receipt))
    r["archive"].unlink()
    packed = run(
        "pack",
        "--web-dir",
        r["web"],
        "--receipt",
        r["receipt"],
        "--output",
        r["archive"],
    )
    assert packed.returncode == 0, packed.stderr
    r["args"][3] = sha(r["archive"].read_bytes())
    installed = run("install", *r["args"], input="yes\n")
    assert installed.returncode == 0, installed.stderr
    protected = {key: snapshot(r[key]) for key in ("backend", "home")}
    unit = f"hermes-browser-round2-{uuid.uuid4().hex}.service"
    anchor = f"hermes-browser-round2-anchor-{uuid.uuid4().hex}.service"
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    config_path = tmp_path / "updater.json"
    value = {
        "public_origin": "https://browser.invalid",
        "listen_port": 0,
        "backend_port": port,
        "administrators": [
            {"provider": ACTOR[0], "org_id": ACTOR[1], "user_id": ACTOR[2]}
        ],
        "maintenance": {
            "installation": str(r["dest"]),
            "state_dir": str(tmp_path / "state"),
            "source": "https://publisher.invalid/CURRENT.json",
            "unit": unit,
            "unit_sha256": "0" * 64,
            "working_directory": str(tmp_path),
        },
    }
    config_path.write_text(json.dumps(value))
    config_path.chmod(0o600)
    # --dashboard refuses an absent journal. Construction performs no unit action.
    with closing(jobs.Updates(updater.load_config(config_path))) as initial:
        assert initial.status()["phase"] == "idle"

    # Only pytest's environment changes, so the real adapter can reach the user
    # bus stripped by the canonical runner. Never import/set manager environment.
    runtime = Path(f"/run/user/{os.getuid()}")
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime))
    monkeypatch.setenv("DBUS_SESSION_BUS_ADDRESS", f"unix:path={runtime}/bus")

    def command(*args):
        result = user_systemd(*args)
        assert result.returncode == 0, result.stderr
        return result.stdout

    def properties(owned=unit):
        return dict(
            line.split("=", 1)
            for line in command("systemctl", "--user", "show", owned).splitlines()
        )

    # UnsetEnvironment is applied last by systemd; clear all inherited manager
    # keys except the four explicitly replaced non-secret fixture values.
    safe = {"HOME", "PATH", "LANG", "TZ"}
    inherited = {
        line.split("=", 1)[0]
        for line in command("systemctl", "--user", "show-environment").splitlines()
    }
    unset = inherited - safe
    assert properties()["LoadState"] == "not-found"
    assert properties(anchor)["LoadState"] == "not-found"
    created, children, artifacts, errors = [], [], set(), []
    finished = threading.Event()
    marker = r["home"] / f"launch-{port}.json"
    artifacts.update((marker, marker.with_suffix(".tmp"), r["home"] / "launch.json"))

    def handshakes():
        seen = set()
        try:
            while not finished.wait(0.02):
                if not marker.exists():
                    continue
                launch = json.loads(marker.read_bytes())
                pid = launch["pid"]
                if pid in seen:
                    continue
                fd = pidfd_open(pid)
                try:
                    assert launch["module"] == str(r["backend"] / "hermes_cli/main.py")
                    proc = (
                        Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                    )
                    parent = int(proc[1])
                    assert parent == int(properties()["MainPID"]) > 1
                    assert not (unset & launch["env"].keys())
                    children.append((fd, launch, parent))
                except BaseException:
                    os.close(fd)
                    raise
                seen.add(pid)
                handshake = r["home"] / f"release-{pid}"
                artifacts.add(handshake)
                handshake.touch()
        except BaseException as error:
            errors.append(error)

    watcher = threading.Thread(target=handshakes, name="native-fixture-handshakes")
    watcher.start()
    try:
        created.append(unit)
        command(
            "systemd-run",
            "--user",
            f"--unit={unit}",
            "--property=Type=simple",
            "--property=Restart=no",
            "--property=KillMode=mixed",
            "--property=SendSIGKILL=no",
            "--property=CollectMode=inactive-or-failed",
            f"--property=WorkingDirectory={tmp_path}",
            f"--property=UnsetEnvironment={' '.join(sorted(unset))}",
            f"--property=Environment=HOME={tmp_path} PATH=/usr/bin:/bin LANG=C.UTF-8 TZ=UTC",
            f"--property=StandardOutput=append:{tmp_path / 'controller.stdout'}",
            f"--property=StandardError=append:{tmp_path / 'controller.stderr'}",
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(CLI.with_name("browser_updater.py")),
            "--config",
            str(config_path),
            "--dashboard",
        )
        created.append(anchor)
        command(
            "systemd-run",
            "--user",
            f"--unit={anchor}",
            "--property=Type=oneshot",
            "--property=RemainAfterExit=yes",
            f"--property=UnsetEnvironment={' '.join(sorted(unset))}",
            f"--property=Environment=HOME={tmp_path} PATH=/usr/bin:/bin LANG=C.UTF-8 TZ=UTC",
            f"--property=Wants={unit}",
            "/usr/bin/true",
        )
        # Do not rewrite config while the first controller may still be reading it.
        wait_for(lambda: children or errors)
        assert not errors, errors
        fragment = Path(properties()["FragmentPath"])
        value["maintenance"]["unit_sha256"] = sha(fragment.read_bytes())
        config_path.write_text(json.dumps(value))
        config = updater.load_config(config_path)
        adapter = service.Dashboard(config)

        def ready():
            assert not errors, errors
            try:
                info = jobs.E.control_request(
                    r["dest"], "status", expected_pid=int(properties()["MainPID"])
                )
            except (OSError, ValueError):
                return None
            return info if info["state"] == "ready" else None

        wait_for(ready, timeout=15)
        yield SimpleNamespace(
            r=r,
            jobs=jobs,
            updater=updater,
            config=config,
            adapter=adapter,
            properties=properties,
            children=children,
            errors=errors,
            fragment=fragment,
            value=value,
        )
    finally:
        # Kill authority is held only by pinned, identified fixture pidfds, and
        # used solely as teardown fallback. The product/unit never force-kills.
        try:
            if unit in created:
                command("systemctl", "--user", "stop", "--no-block", unit)
                wait_for(
                    lambda: properties()["ActiveState"] in ("inactive", "failed"),
                    timeout=10,
                )
        finally:
            finished.set()
            watcher.join(5)
            assert not watcher.is_alive()
            for fd, _, _ in children:
                if not select.select([fd], [], [], 0)[0]:
                    kill_fixture(fd)
            try:
                for owned in created:
                    command("systemctl", "--user", "stop", owned)
                for owned in created:
                    wait_for(lambda: properties(owned)["LoadState"] == "not-found")
            finally:
                for fd, _, _ in children:
                    try:
                        assert select.select([fd], [], [], 5)[0], (
                            "Fixture child survived cleanup"
                        )
                    finally:
                        os.close(fd)
            for path in artifacts:
                path.unlink(missing_ok=True)
            assert {key: snapshot(r[key]) for key in protected} == protected
            assert not errors, errors


def assert_live(s, receipt):
    info = s.adapter.status()
    props = s.properties()
    required = importlib.import_module("browser_update_service").PROPERTIES
    shown = s.adapter._systemctl(
        "show", "--no-pager", "--property=" + ",".join(required)
    )
    missing = set(required) - {line.partition("=")[0] for line in shown.splitlines()}
    assert info["state"] == "ready", (info, {"missing_properties": sorted(missing)})
    assert info["receipt"] == receipt
    assert info["installation"] == str(s.r["dest"])
    assert info["url"] == f"http://127.0.0.1:{s.config.backend_port}/"
    assert props["Restart"] == "no" and props["NRestarts"] == "0"
    assert props["KillMode"] == "mixed" and props["SendSIGKILL"] == "no"
    assert sha(s.fragment.read_bytes()) == s.config.maintenance["unit_sha256"]
    assert info == s.jobs.E.control_request(
        s.r["dest"], "status", expected_pid=int(props["MainPID"])
    )
    with pytest.raises(ValueError, match="PID differs"):
        s.jobs.E.control_request(s.r["dest"], "status", expected_pid=os.getpid())
    child = next(entry for entry in s.children if entry[1]["pid"] == info["pid"])
    assert child[2] == int(props["MainPID"])
    assert not select.select([child[0]], [], [], 0)[0]
    return info


@pytest.mark.parametrize(
    "label,phase", [("native-success", "succeeded"), ("fails-to-start", "rolled_back")]
)
def test_native_update_and_failed_activation_restore(
    native_update, tmp_path, monkeypatch, label, phase
):
    s = native_update
    old = s.jobs.E.installed(s.r["dest"])[0]
    original = snapshot(s.r["dest"])
    first = assert_live(s, old)
    actions = []
    real_systemctl = s.adapter._systemctl

    def observed(*args, **kwargs):
        if args[0] in ("start", "stop"):
            # Validate every installed byte at each native service boundary.
            receipt = s.jobs.E.installed(s.r["dest"])[0]
            actions.append((args[0], receipt, snapshot(s.r["dest"])))
        return real_systemctl(*args, **kwargs)

    monkeypatch.setattr(s.adapter, "_systemctl", observed)
    invalid = deepcopy(s.value)
    del invalid["maintenance"]["unit_sha256"]
    invalid_path = tmp_path / "invalid.json"
    invalid_path.write_text(json.dumps(invalid))
    invalid_path.chmod(0o600)
    with pytest.raises(ValueError):
        s.updater.load_config(invalid_path)
    wrong = replace(
        s.config, maintenance={**s.config.maintenance, "unit_sha256": "0" * 64}
    )
    with monkeypatch.context() as wrong_pin:
        wrong_pin.setattr(s.adapter, "config", wrong)
        assert s.adapter.status()["state"] == "unknown"
        for method in (s.adapter.start, s.adapter.stop):
            with pytest.raises(ValueError, match="hash mismatch"):
                method(old)
    assert actions == []
    assert assert_live(s, old)["generation"] == first["generation"]

    archive = candidate(s.r, tmp_path, label)
    launcher = CLI.read_bytes()
    descriptor = {"schema": 1}
    responses = {}
    for key, name, data in (
        ("archive", "release.tar.gz", archive.read_bytes()),
        ("launcher", "launcher.py", launcher),
    ):
        descriptor[key] = {"name": name, "size": len(data), "sha256": sha(data)}
        responses[f"https://publisher.invalid/{name}"] = data
    responses[s.config.maintenance["source"]] = json.dumps(descriptor).encode()
    requests = []

    def fetch(url, limit, expected_size=None):
        requests.append(url)
        data = responses[url]  # No network fallback, including unexpected issuers.
        assert len(data) <= limit
        assert expected_size is None or len(data) == expected_size
        return data

    monkeypatch.setattr(importlib.import_module("browser_download"), "fetch", fetch)
    _, payload = s.jobs.E.archive_payload(
        str(archive), sha(archive.read_bytes()), launcher
    )
    target, files = s.jobs.E.installation_files(
        sha(archive.read_bytes()), old["selection"], payload, launcher
    )
    with closing(s.jobs.Updates(s.config, dashboard=s.adapter)) as updates:
        updates.offer(ACTOR)
        updates.worker.join(15)
        offered = updates.status()
        assert offered["phase"] == "offered", offered
        assert not actions and snapshot(s.r["dest"]) == original
        checked_requests = list(requests)
        updates.apply(ACTOR, offered["offer"]["id"], confirm_restart_and_rollback=True)
        updates.worker.join(45)
        done = updates.status()
        assert done["phase"] == phase, (
            done,
            (tmp_path / "controller.stderr").read_text(),
        )
        assert not updates.worker.is_alive()
        assert requests == checked_requests
        assert done["job"]["id"] == offered["offer"]["id"]
        journal = s.jobs.read_state(Path(s.config.maintenance["state_dir"]))
        assert journal["current"] == old and journal["target"] == target
        assert journal["confirmed"] is True

    expected = target if phase == "succeeded" else old
    assert s.jobs.E.installed(s.r["dest"])[0] == expected
    live = assert_live(s, expected)
    assert live["generation"] != first["generation"]
    assert live["pid"] != first["pid"]
    assert all(
        select.select([fd], [], [], 5)[0]
        for fd, launch, _ in s.children
        if launch["pid"] != live["pid"]
    )
    assert snapshot(history(s.r["dest"]) / old["archive_sha256"]) == original
    assert [(action, receipt) for action, receipt, _ in actions] == (
        [("stop", old), ("start", target)]
        if phase == "succeeded"
        else [("stop", old), ("start", target), ("start", old)]
    )
    target_snapshot = actions[1][2]
    assert {
        name: entry[1] for name, entry in target_snapshot.items() if entry[0] == "file"
    } == files
    if phase == "succeeded":
        assert snapshot(s.r["dest"]) == target_snapshot
    else:
        assert snapshot(s.r["dest"]) == original
        assert (
            snapshot(history(s.r["dest"]) / target["archive_sha256"]) == target_snapshot
        )
    assert not s.errors
