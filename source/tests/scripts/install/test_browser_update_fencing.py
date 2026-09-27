"""Updater fences against disposable installations and protocol controllers only."""

from argparse import Namespace
from copy import deepcopy
import fcntl
import json
import os
import socket

import pytest

from tests.scripts.install.test_browser_maintenance import (
    candidate,
    control,
    history,
    installed_receipt,
    maintained,  # noqa: F401
    snapshot,
    update,
)
from tests.scripts.install.test_browser_offline import (
    controllers,  # noqa: F401
    dashboard,  # noqa: F401
    installer_module,  # noqa: F401
    lifecycle,
    release,  # noqa: F401
    sha,
    state_is,
    wait_for,
)


# Only tmp_path browser assets and the disposable protocol fixture are operated on.
pytestmark = [pytest.mark.linux_only, pytest.mark.live_system_guard_bypass]


def assert_locked(root):
    with control(root).open("r+b") as contender:
        with pytest.raises(BlockingIOError):
            fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)


@pytest.mark.parametrize(
    "operation", ["update", "update-noop", "rollback", "rollback-noop"]
)
def test_maintenance_checks_full_receipts_before_confirmation_or_noop(
    maintained, installer_module, tmp_path, monkeypatch, operation
):
    r, engine = maintained, installer_module
    root = r["dest"]
    original = installed_receipt(root)
    archive = candidate(r, tmp_path, "next")
    if operation.startswith("rollback"):
        assert update(r, archive).returncode == 0
        target = original if operation == "rollback" else installed_receipt(root)
        args = Namespace(
            command="rollback", install_root=str(root), to=target["archive_sha256"]
        )
    else:
        if operation == "update-noop":
            archive = r["archive"]
        digest = sha(archive.read_bytes())
        launcher = engine.read_regular(engine.Path(engine.__file__))
        _, payload = engine.archive_payload(archive, digest, launcher)
        target, _ = engine.installation_files(
            digest, original["selection"], payload, launcher
        )
        args = Namespace(
            command="update",
            install_root=str(root),
            archive=archive,
            sha256=digest,
            launcher=None,
        )
    current = installed_receipt(root)
    before = snapshot(root), snapshot(history(root)), control(root).read_bytes()
    confirmations = []
    real_installed = engine.installed
    reads = []

    def checked_installed(path):
        if path == root:
            if reads:
                assert_locked(root)
            reads.append(path)
        return real_installed(path)

    monkeypatch.setattr(engine, "installed", checked_installed)

    def confirm(preview):
        assert_locked(root)
        confirmations.append(preview)
        return True

    for side in ("current", "target"):
        for field in current:
            expected = {
                "expected_current": deepcopy(current),
                "expected_target": deepcopy(target),
            }
            changed = expected["expected_" + side]
            if field == "selection":
                changed[field]["profile"] = "alpha"
            else:
                changed[field] = "0" * 64
            reads.clear()
            with pytest.raises(ValueError, match=f"expected {side} receipt"):
                engine.maintenance(args, confirm=confirm, **expected)
            assert len(reads) >= 2
            assert not confirmations
            assert (
                snapshot(root),
                snapshot(history(root)),
                control(root).read_bytes(),
            ) == before
    reads.clear()
    engine.maintenance(
        args, confirm=confirm, expected_current=current, expected_target=target
    )
    assert real_installed(root)[0] == target
    assert len(confirmations) == (0 if operation.endswith("noop") else 1)
    if operation.endswith("noop"):
        assert (
            snapshot(root),
            snapshot(history(root)),
            control(root).read_bytes(),
        ) == before


def test_start_admission_uses_fresh_receipt_under_lock_before_any_mutation(
    maintained, installer_module, monkeypatch
):
    r, engine = maintained, installer_module
    root = r["dest"]
    record = json.loads(control(root).read_bytes())
    record["owner"] = engine.OWNER  # Admission must not fence a legacy controller.
    control(root).write_bytes(engine.json_bytes(record))
    before = control(root).read_bytes()
    fresh = installed_receipt(root)
    fresh["selection"]["profile"] = "alpha"
    acquire = engine.acquire_control

    def switch_before_lock_validation(stream):
        acquired = acquire(stream)
        assert acquired
        (root / "installation.json").write_bytes(engine.json_bytes(fresh))
        return acquired

    monkeypatch.setattr(engine, "acquire_control", switch_before_lock_validation)
    monkeypatch.setattr(
        engine,
        "run_foreground",
        lambda *args: pytest.fail("Admission spawned a controller"),
    )
    admitted = []

    def deny(receipt):
        admitted.append(receipt)
        assert receipt == fresh
        assert_locked(root)
        assert control(root).read_bytes() == before
        raise ValueError("Update owns start admission")

    args = Namespace(command="start", install_root=str(root), port=9119, timeout=2)
    with pytest.raises(ValueError, match="Update owns start admission"):
        engine.lifecycle(args, admit_start=deny)
    assert admitted == [fresh]
    assert control(root).read_bytes() == before
    assert not list(r["home"].glob("launch*.json"))


def test_status_reports_launch_receipt_and_peer_pid_fences_stop(
    dashboard, controllers, installer_module
):
    r, engine = dashboard, installer_module
    root = r["dest"]
    launched = installed_receipt(root)
    owner, _ = controllers(r)
    wait_for(lambda: state_is(r, "ready"))
    # Disk selection is not evidence of the identity already launched.
    changed = deepcopy(launched)
    changed["selection"]["profile"] = "beta"
    (root / "installation.json").write_bytes(engine.json_bytes(changed))
    status = json.loads(lifecycle(r, "status").stdout)
    assert status["receipt"] == launched
    with socket.socket(socket.AF_UNIX) as connection:
        connection.connect(engine.control_address(root))
        assert engine.same_user(connection) == owner.pid
    with pytest.raises(ValueError, match="(?i)pid"):
        engine.control_request(root, "stop", expected_pid=os.getpid())
    # A subsequent served status proves the rejected request sent no stop command.
    status = engine.control_request(root, "status", expected_pid=owner.pid)
    assert status["state"] == "ready"
    assert status["receipt"] == launched
    engine.control_request(root, "stop", expected_pid=owner.pid)
    assert owner.wait(timeout=10) == 0
