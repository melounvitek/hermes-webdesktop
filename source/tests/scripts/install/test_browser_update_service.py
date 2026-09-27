"""Fixed-unit admission with real lifecycle records and disposable controllers."""

from copy import deepcopy
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
import uuid

import pytest

from tests.scripts.install.test_browser_offline import (
    CLI,
    controllers,  # noqa: F401
    dashboard,  # noqa: F401
    release,  # noqa: F401
    sha,
    wait_for,
)

from tests.scripts.install.test_browser_systemd import user_systemd  # noqa: F401

pytestmark = [pytest.mark.linux_only, pytest.mark.live_system_guard_bypass]


@pytest.fixture
def service(monkeypatch):
    monkeypatch.syspath_prepend(str(CLI.parent))
    return importlib.import_module("browser_update_service")


@pytest.fixture
def managed(service, dashboard, controllers, tmp_path, monkeypatch):
    r = dashboard
    unit = "fixture-dashboard.service"
    config = SimpleNamespace(
        config_path=tmp_path / "updater.json",
        backend_port=9119,
        maintenance={
            "installation": str(r["dest"]),
            "state_dir": str(tmp_path / "state"),
            "source": "https://example.invalid/CURRENT.json",
            "unit": unit,
            "working_directory": str(tmp_path),
        },
    )
    argv = [
        sys.executable,
        "-I",
        "-S",
        "-B",
        str(CLI.with_name("browser_updater.py")),
        "--config",
        str(config.config_path),
        "--dashboard",
    ]
    fragment = tmp_path / unit
    fragment.write_text(
        "[Service]\nExecStart=" + " ".join(f'"{a}"' for a in argv) + "\n"
    )
    fragment.chmod(0o600)
    config.maintenance["unit_sha256"] = sha(fragment.read_bytes())
    props = dict(
        Id=unit,
        LoadState="loaded",
        ActiveState="inactive",
        MainPID="0",
        FragmentPath=str(fragment),
        DropInPaths="",
        NeedDaemonReload="no",
        Type="simple",
        Restart="no",
        KillMode="mixed",
        SendSIGKILL="no",
        KillSignal="15",
        TimeoutStopFailureMode="terminate",
        WorkingDirectory=str(tmp_path),
        ExecStartPre="",
        ExecStartPost="",
        ExecStop="",
        ExecStopPost="",
        ExecCondition="",
        ExecReload="",
        ExecStart=(
            f"{{ path={argv[0]} ; argv[]={' '.join(argv)} ; ignore_errors=no ; "
            "start_time=[n/a] ; stop_time=[n/a] ; pid=0 ; code=(null) ; status=0/0 }"
        ),
        ExecStartEx=(
            f"{{ path={argv[0]} ; argv[]={' '.join(argv)} ; flags= ; "
            "start_time=[n/a] ; stop_time=[n/a] ; pid=0 ; code=(null) ; status=0/0 }"
        ),
    )
    actions = []
    child = None

    def run(command, **kwargs):
        nonlocal child
        assert command[:2] == ["systemctl", "--user"]
        assert command[-1] == unit
        assert 0 < kwargs["timeout"] <= service.SERVICE_TIMEOUT
        if "show" in command:
            return subprocess.CompletedProcess(
                command, 0, "".join(f"{k}={v}\n" for k, v in props.items()), ""
            )
        action = next(item for item in command if item in ("start", "stop"))
        actions.append(action)
        if action == "start":
            child, port = controllers(r)
            config.backend_port = port
            props.update(ActiveState="active", MainPID=str(child.pid))
        else:
            child.terminate()  # This fixture owns the Popen, never a recorded child PID.
            assert child.wait(timeout=10) == 0
            props.update(ActiveState="inactive", MainPID="0")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(
        service,
        "subprocess",
        SimpleNamespace(run=run, TimeoutExpired=subprocess.TimeoutExpired),
    )
    adapter = service.Dashboard(config)
    return SimpleNamespace(
        adapter=adapter,
        config=config,
        props=props,
        actions=actions,
        root=r["dest"],
        fragment=fragment,
        argv=argv,
    )


def test_exact_unit_and_live_generation_fence_every_action(service, managed):
    m = managed
    receipt = service.E.installed(m.root)[0]
    assert m.adapter.status()["state"] == "stopped"
    assert m.adapter.start(receipt)["state"] == "ready"
    assert m.adapter.start(receipt)["receipt"] == receipt
    assert m.actions == ["start"]
    first = m.adapter.status()
    assert first["receipt"] == receipt
    changed = deepcopy(receipt)
    changed["selection"]["profile"] = "beta"
    with pytest.raises(ValueError):
        m.adapter.stop(changed)
    assert m.actions == ["start"]

    # A disk receipt is not the launched controller's identity.
    (m.root / "installation.json").write_bytes(service.E.json_bytes(changed))
    assert m.adapter.status()["receipt"] == receipt
    (m.root / "installation.json").write_bytes(service.E.json_bytes(receipt))
    for key, bad in {
        "MainPID": str(os.getpid()),
        "Id": "other.service",
        "LoadState": "not-found",
        "DropInPaths": "/tmp/override.conf",
        "NeedDaemonReload": "yes",
        "Type": "forking",
        "Restart": "always",
        "KillMode": "control-group",
        "SendSIGKILL": "yes",
        "KillSignal": "9",
        "TimeoutStopFailureMode": "abort",
        "WorkingDirectory": "/",
        "ExecStartPre": "foreign",
        "ExecStartPost": "foreign",
        "ExecStop": "foreign",
        "ExecStopPost": "foreign",
        "ExecCondition": "foreign",
        "ExecReload": "foreign",
        "ExecStart": m.props["ExecStart"] + " ; foreign",
        "ExecStartEx": m.props["ExecStartEx"].replace(
            "flags= ;", "flags=ignore-failure ;"
        ),
    }.items():
        original = m.props[key]
        m.props[key] = bad
        assert m.adapter.status()["state"] == "unknown", key
        for method in (m.adapter.stop, m.adapter.start):
            with pytest.raises(ValueError):
                method(receipt)
        assert m.actions == ["start"], key
        m.props[key] = original

    original = m.fragment.read_bytes()
    m.fragment.write_bytes(original + b"# changed\n")
    assert m.adapter.status()["state"] == "unknown"
    with pytest.raises(ValueError):
        m.adapter.stop(receipt)
    m.fragment.write_bytes(original)
    # systemctl flattens argv: a single quoted argument must not pass as two flags.
    m.fragment.write_bytes(original.replace(b'"-I" "-S"', b'"-I -S"'))
    m.config.maintenance["unit_sha256"] = sha(m.fragment.read_bytes())
    assert m.adapter.status()["state"] == "unknown"
    m.fragment.write_bytes(original)
    m.config.maintenance["unit_sha256"] = sha(original)

    stopped = m.adapter.stop(receipt)
    assert stopped["state"] == "stopped"
    assert m.actions == ["start", "stop"]
    assert m.adapter.start(receipt)["generation"] != first["generation"]
    assert m.adapter.stop(receipt)["state"] == "stopped"


def test_inactive_is_not_stopped_and_running_requires_locked_generation(
    service, managed
):
    m = managed
    receipt = service.E.installed(m.root)[0]
    path = m.root.with_name(m.root.name + ".run")
    original = path.read_bytes()
    path.unlink()
    assert m.adapter.status()["state"] == "unknown"
    assert not path.exists()
    with pytest.raises(ValueError):
        m.adapter.start(receipt)
    assert not path.exists() and not m.actions
    path.write_bytes(original)
    path.chmod(0o600)
    record = json.loads(original)
    record.update(state="unknown", generation="a" * 32, pid=None)
    path.write_bytes(service.E.json_bytes(record))
    for method in (m.adapter.start, m.adapter.stop):
        with pytest.raises(ValueError):
            method(receipt)
    assert m.adapter.status()["state"] == "unknown"
    assert path.read_bytes() == service.E.json_bytes(record) and not m.actions
    path.write_bytes(original)
    ready = m.adapter.start(receipt)
    live = path.read_bytes()
    record = json.loads(live)
    record["generation"] = "b" * 32
    path.write_bytes(service.E.json_bytes(record))
    assert m.adapter.status()["state"] == "unknown"
    with pytest.raises(ValueError):
        m.adapter.stop(receipt)
    assert m.actions == ["start"]
    path.write_bytes(live)
    assert m.adapter.status()["generation"] == ready["generation"]
    assert m.adapter.stop(receipt)["state"] == "stopped"
    # A stale active unit/socket cannot override an unlocked record.
    m.props.update(ActiveState="active", MainPID=str(os.getpid()))
    assert m.adapter.status()["state"] == "unknown"


def test_native_anchor_retains_exact_transient_definition_across_stop_start(
    user_systemd,
):
    # Infrastructure acceptance only: the product adapter's command is never
    # replaced with sleep. End-to-end --dashboard acceptance belongs to its CLI.
    unit = f"hermes-browser-adapter-{uuid.uuid4().hex}.service"
    anchor = f"hermes-browser-anchor-{uuid.uuid4().hex}.service"

    def show():
        result = user_systemd("systemctl", "--user", "show", unit)
        assert result.returncode == 0, result.stderr
        return dict(line.split("=", 1) for line in result.stdout.splitlines())

    def command(*args):
        result = user_systemd(*args)
        assert result.returncode == 0, result.stderr

    assert show()["LoadState"] == "not-found"
    created = []
    try:
        command(
            "systemd-run",
            "--user",
            f"--unit={unit}",
            "--property=Type=simple",
            "--property=Restart=no",
            "--property=KillMode=mixed",
            "--property=SendSIGKILL=no",
            "--property=CollectMode=inactive-or-failed",
            "/usr/bin/sleep",
            "120",
        )
        created.append(unit)
        command(
            "systemd-run",
            "--user",
            f"--unit={anchor}",
            "--property=Type=oneshot",
            "--property=RemainAfterExit=yes",
            f"--property=Wants={unit}",
            "/usr/bin/true",
        )
        created.append(anchor)
        initial = show()
        assert initial["ActiveState"] == "active"
        fragment = Path(initial["FragmentPath"])
        digest = sha(fragment.read_bytes())
        command("systemctl", "--user", "stop", unit)
        stopped = show()
        assert stopped["LoadState"] == "loaded" and stopped["ActiveState"] == "inactive"
        assert stopped["MainPID"] == "0" and sha(fragment.read_bytes()) == digest
        command("systemctl", "--user", "start", unit)
        restarted = show()
        assert restarted["ActiveState"] == "active"
        assert int(restarted["MainPID"]) > 1
        assert restarted["MainPID"] != initial["MainPID"]
        assert sha(fragment.read_bytes()) == digest
    finally:
        for owned in created:
            command("systemctl", "--user", "stop", owned)
        wait_for(lambda: show()["LoadState"] == "not-found")
