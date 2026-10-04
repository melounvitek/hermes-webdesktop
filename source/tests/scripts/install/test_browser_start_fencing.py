"""Start and status fences; disposable installations and protocol controllers only."""

from argparse import Namespace
from copy import deepcopy
import json

import pytest

from tests.scripts.install.test_browser_maintenance import (
    control,
    installed_receipt,
    maintained,  # noqa: F401
)
from tests.scripts.install.test_browser_offline import (
    controllers,  # noqa: F401
    dashboard,  # noqa: F401
    installer_module,  # noqa: F401
    lifecycle,
    release,  # noqa: F401
    state_is,
    wait_for,
)


# Only tmp_path browser assets and the disposable protocol fixture are operated on.
pytestmark = [pytest.mark.linux_only, pytest.mark.live_system_guard_bypass]


def test_start_checks_fresh_receipt_under_lock_before_any_mutation(
    maintained, installer_module, monkeypatch
):
    r, engine = maintained, installer_module
    root = r["dest"]
    record = json.loads(control(root).read_bytes())
    record["owner"] = engine.OWNER  # Refusal must not fence a legacy controller.
    control(root).write_bytes(engine.json_bytes(record))
    before = control(root).read_bytes()
    expected = installed_receipt(root)["selection"]
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
        lambda *args: pytest.fail("Refused start spawned a controller"),
    )
    args = Namespace(command="start", install_root=str(root), port=9119, timeout=2)
    with pytest.raises(ValueError, match="selection differs from controller"):
        engine.lifecycle(args, selection=expected)
    assert control(root).read_bytes() == before
    assert not list(r["home"].glob("launch*.json"))


def test_status_reports_launch_receipt(dashboard, controllers, installer_module):
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
    engine.control_request(root, "stop")
    assert owner.wait(timeout=10) == 0
