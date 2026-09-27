"""Durable browser updates with real archives, no backend, network, or live units."""

from contextlib import closing
from copy import deepcopy
import importlib
import json
import multiprocessing
from pathlib import Path
import threading
from types import SimpleNamespace

import pytest

from tests.scripts.install.test_browser_maintenance import (
    CLI,
    candidate,
    history,
    installed_receipt,
    maintained,  # noqa: F401
    release,  # noqa: F401
    sha,
    snapshot,
    wait_for,
)

# Only disposable portable-CLI fixtures and our own crash-test worker run here.
pytestmark = [pytest.mark.linux_only, pytest.mark.live_system_guard_bypass]
ACTOR = ("fixture", None, "administrator")
TERMINAL = {
    "offered",
    "current",
    "failed",
    "recovery_required",
    "succeeded",
    "rolled_back",
}


class Dashboard:
    """Model only service identity and activation failure; never launch a backend."""

    def __init__(self, root):
        self.root = root
        self.receipt = installed_receipt(root)
        self.state = "ready"
        self.actions = []
        self.fail_start = None
        self.before_stop = None

    def status(self):
        result = {"state": self.state}
        if self.state == "ready":
            result["receipt"] = deepcopy(self.receipt)
        return result

    def stop(self, expected):
        assert self.state != "unknown"
        if self.state == "ready":
            assert expected == self.receipt
        self.actions.append(("stop", deepcopy(expected)))
        if self.before_stop:
            self.before_stop()
        self.state = "stopped"
        return self.status()

    def start(self, expected):
        assert self.state == "stopped"
        assert installed_receipt(self.root) == expected
        self.actions.append(("start", deepcopy(expected)))
        if expected["archive_sha256"] == self.fail_start:
            raise ValueError("Fixture activation failed; service remains stopped")
        self.receipt = deepcopy(expected)
        self.state = "ready"
        return self.status()


@pytest.fixture
def jobs(monkeypatch):
    monkeypatch.syspath_prepend(str(CLI.parent))
    return importlib.import_module("browser_update_jobs")


@pytest.fixture
def scene(maintained, tmp_path, monkeypatch):
    r = maintained
    protected = {path: snapshot(path) for path in (r["backend"], r["home"])}
    config = SimpleNamespace(
        config_path=tmp_path / "updater.json",
        backend_port=9119,
        maintenance={
            "installation": str(r["dest"]),
            "state_dir": str(tmp_path / "state"),
            "source": "https://publisher.invalid/CURRENT.json",
            "unit": "fixture-dashboard.service",
            "unit_sha256": sha(b"fixture unit"),
            "working_directory": str(tmp_path),
        },
    )
    config.config_path.write_text(
        json.dumps({
            "public_origin": "https://browser.invalid",
            "listen_port": 9120,
            "backend_port": config.backend_port,
            "administrators": [
                {"provider": ACTOR[0], "org_id": ACTOR[1], "user_id": ACTOR[2]}
            ],
            "maintenance": config.maintenance,
        })
    )
    config.config_path.chmod(0o600)
    config.configuration_sha256 = sha(config.config_path.read_bytes())
    responses, requests = {}, []

    def publish(archive, launcher=None):
        payloads = {
            "archive": archive.read_bytes(),
            "launcher": (launcher or CLI).read_bytes(),
        }
        descriptor = {"schema": 1}
        responses.clear()
        for key, data in payloads.items():
            name = "release.tar.gz" if key == "archive" else "launcher.py"
            descriptor[key] = {"name": name, "size": len(data), "sha256": sha(data)}
            responses[f"https://publisher.invalid/{name}"] = data
        responses[config.maintenance["source"]] = json.dumps(descriptor).encode()

    def fetch(url, limit, expected_size=None):
        requests.append(url)
        # An unexpected URL is a test failure, never a real network fallback.
        data = responses[url]
        assert len(data) <= limit
        if expected_size is not None:
            assert len(data) == expected_size
        return data

    monkeypatch.syspath_prepend(str(CLI.parent))
    monkeypatch.setattr(importlib.import_module("browser_download"), "fetch", fetch)
    publish(r["archive"])
    yield SimpleNamespace(
        r=r,
        config=config,
        dashboard=Dashboard(r["dest"]),
        publish=publish,
        responses=responses,
        requests=requests,
        state=Path(config.maintenance["state_dir"]),
    )
    assert all(snapshot(path) == before for path, before in protected.items())


def settled(updates, phase):
    def terminal():
        state = updates.status()
        return state if state["phase"] in TERMINAL else None

    state = wait_for(terminal, timeout=20)
    assert state["phase"] == phase, state
    return state


def offered(updates):
    updates.offer(ACTOR)
    return settled(updates, "offered")["offer"]["id"]


def record(scene):
    return json.loads((scene.state / "state.json").read_bytes())


def test_pinned_offer_is_durable_idempotent_and_never_executes_downloaded_launcher(
    jobs, scene, tmp_path
):
    s = scene
    original = snapshot(s.r["dest"])
    old = installed_receipt(s.r["dest"])
    marker = tmp_path / "downloaded-launcher-executed"
    launcher = tmp_path / "untrusted-launcher.py"
    launcher.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\n"
        "raise RuntimeError('do not execute')\n"
    )
    archive = candidate(s.r, tmp_path, "offered", launcher=launcher)
    s.publish(archive, launcher)
    entered, release_stop = threading.Event(), threading.Event()
    durable = []

    def stopping():
        durable.append(record(s))
        entered.set()
        assert release_stop.wait(20), "Test did not release the stopped worker"

    s.dashboard.before_stop = stopping
    with closing(jobs.Updates(s.config, dashboard=s.dashboard)) as updates:
        assert "install" in updates.status()["capabilities"]
        offer_id = offered(updates)
        assert (s.state.stat().st_mode & 0o777) == 0o700
        assert record(s)["actor"] == list(ACTOR)
        assert record(s)["current"] == old
        assert (s.state / "candidate/archive").read_bytes() == archive.read_bytes()
        assert (s.state / "candidate/launcher").read_bytes() == launcher.read_bytes()
        later = candidate(s.r, tmp_path, "publisher-moved")
        s.publish(later)
        requests = list(s.requests)
        try:
            accepted = updates.apply(ACTOR, offer_id, confirm_restart_and_rollback=True)
            job_id = accepted["job"]["id"]
            assert entered.wait(20), updates.status()
            assert durable[0]["phase"] == "stopping"
            assert durable[0]["id"] == job_id
            assert durable[0]["target"]["archive_sha256"] == sha(archive.read_bytes())
            duplicate = updates.apply(
                ACTOR, offer_id, confirm_restart_and_rollback=True
            )
            assert duplicate["job"]["id"] == job_id
        finally:
            release_stop.set()
        done = settled(updates, "succeeded")
        assert done["job"]["id"] == job_id
        repeated = updates.apply(ACTOR, offer_id, confirm_restart_and_rollback=True)
        assert repeated["job"]["id"] == job_id
        # Validate all selected bytes, not just the newly written receipt.
        target, _ = importlib.import_module("browser_setup").E.installed(s.r["dest"])
        assert target == durable[0]["target"]
        assert target["selection"] == old["selection"]
        assert s.dashboard.actions == [("stop", old), ("start", target)]
        assert s.requests == requests
        assert snapshot(history(s.r["dest"]) / old["archive_sha256"]) == original
        assert (s.r["dest"] / "hermes-browser.py").read_bytes() == launcher.read_bytes()
        assert not marker.exists()


def test_apply_requires_bound_actor_exact_offer_and_explicit_consent(
    jobs, scene, tmp_path
):
    s = scene
    before = snapshot(s.r["dest"]), snapshot(history(s.r["dest"]))
    with closing(jobs.Updates(s.config, dashboard=s.dashboard)) as updates:
        updates.offer(ACTOR)
        settled(updates, "current")
        with pytest.raises(ValueError):
            updates.apply(ACTOR, "0" * 32, confirm_restart_and_rollback=True)
        s.publish(candidate(s.r, tmp_path, "candidate"))
        offer_id = offered(updates)
        for actor, identifier, consent in (
            ((ACTOR[0], "", ACTOR[2]), offer_id, True),
            ((ACTOR[0], None, "someone-else"), offer_id, True),
            (ACTOR, "0" * 32, True),
            (ACTOR, offer_id, False),
        ):
            with pytest.raises(ValueError):
                updates.apply(actor, identifier, confirm_restart_and_rollback=consent)
        with pytest.raises(ValueError):
            updates.apply(ACTOR, offer_id)
        assert updates.status()["offer"]["id"] == offer_id
        assert (snapshot(s.r["dest"]), snapshot(history(s.r["dest"]))) == before
        assert s.dashboard.actions == []


def test_download_corruption_and_changed_current_are_refused_before_stop(
    jobs, scene, tmp_path
):
    s = scene
    archive = candidate(s.r, tmp_path, "candidate")
    s.publish(archive)
    before = snapshot(s.r["dest"]), snapshot(history(s.r["dest"]))
    url = "https://publisher.invalid/release.tar.gz"
    raw = s.responses[url]
    s.responses[url] = bytes([raw[0] ^ 1]) + raw[1:]
    with closing(jobs.Updates(s.config, dashboard=s.dashboard)) as updates:
        updates.offer(ACTOR)
        settled(updates, "failed")
        assert (snapshot(s.r["dest"]), snapshot(history(s.r["dest"]))) == before
        assert not s.dashboard.actions
        s.publish(archive)
        offer_id = offered(updates)
        installed = s.r["dest"] / "hermes-browser.py"
        installed.write_bytes(installed.read_bytes() + b"\n# local modification\n")
        changed = snapshot(s.r["dest"]), snapshot(history(s.r["dest"]))
        updates.apply(ACTOR, offer_id, confirm_restart_and_rollback=True)
        settled(updates, "failed")
        assert not s.dashboard.actions
        assert (snapshot(s.r["dest"]), snapshot(history(s.r["dest"]))) == changed


def test_failed_activation_restores_complete_retained_installation(
    jobs, scene, tmp_path
):
    s = scene
    original = snapshot(s.r["dest"])
    old = installed_receipt(s.r["dest"])
    archive = candidate(s.r, tmp_path, "fails-to-start")
    target_sha = sha(archive.read_bytes())
    s.publish(archive)
    s.dashboard.fail_start = target_sha
    with closing(jobs.Updates(s.config, dashboard=s.dashboard)) as updates:
        offer_id = offered(updates)
        updates.apply(ACTOR, offer_id, confirm_restart_and_rollback=True)
        settled(updates, "rolled_back")
        assert snapshot(s.r["dest"]) == original
        assert snapshot(history(s.r["dest"]) / old["archive_sha256"]) == original
        assert s.dashboard.status() == {"state": "ready", "receipt": old}
        starts = [
            receipt for action, receipt in s.dashboard.actions if action == "start"
        ]
        assert [receipt["archive_sha256"] for receipt in starts] == [
            target_sha,
            old["archive_sha256"],
        ]
        assert starts[0]["selection"] == old["selection"]
        assert s.dashboard.actions[0] == ("stop", old)


def test_unknown_service_never_switches_or_attempts_stop(jobs, scene, tmp_path):
    s = scene
    s.publish(candidate(s.r, tmp_path, "candidate"))
    before = snapshot(s.r["dest"]), snapshot(history(s.r["dest"]))
    with closing(jobs.Updates(s.config, dashboard=s.dashboard)) as updates:
        offer_id = offered(updates)
        s.dashboard.state = "unknown"
        updates.apply(ACTOR, offer_id, confirm_restart_and_rollback=True)
        settled(updates, "failed")
        assert s.dashboard.actions == []
        assert (snapshot(s.r["dest"]), snapshot(history(s.r["dest"]))) == before


def test_state_directory_publication_failure_refuses_initialization(
    jobs, scene, monkeypatch
):
    sync = jobs.E.sync_directory

    def fail_parent(path):
        if Path(path) == scene.state.parent:
            raise OSError("Fixture parent-directory fsync failure")
        return sync(path)

    monkeypatch.setattr(jobs.E, "sync_directory", fail_parent)
    with pytest.raises(OSError):
        with closing(jobs.Updates(scene.config, dashboard=scene.dashboard)):
            pass
    assert not scene.dashboard.actions and not scene.requests


def test_configuration_replacement_requires_reload_before_an_offer(
    jobs, scene, tmp_path
):
    s = scene
    s.publish(candidate(s.r, tmp_path, "candidate"))
    with closing(jobs.Updates(s.config, dashboard=s.dashboard)) as updates:
        changed = json.loads(s.config.config_path.read_bytes())
        changed["maintenance"]["source"] = "https://other.invalid/CURRENT.json"
        s.config.config_path.write_text(json.dumps(changed))
        with pytest.raises(ValueError):
            updates.offer(ACTOR)
        assert updates.status()["phase"] == "idle"
        assert not s.requests and not s.dashboard.actions


def test_success_record_failure_keeps_healthy_target_and_blocks_further_actions(
    jobs,
    scene,
    tmp_path,
    monkeypatch,
):
    s = scene
    archive = candidate(s.r, tmp_path, "candidate")
    s.publish(archive)
    replace = jobs.os.replace

    def fail_terminal(source, destination):
        if (
            Path(source) == s.state / "state.next"
            and json.loads(Path(source).read_bytes())["phase"] == "succeeded"
        ):
            raise OSError("Fixture journal persistence failure")
        return replace(source, destination)

    with closing(jobs.Updates(s.config, dashboard=s.dashboard)) as updates:
        identifier = offered(updates)
        monkeypatch.setattr(jobs.os, "replace", fail_terminal)
        updates.apply(ACTOR, identifier, True)
        settled(updates, "recovery_required")
        assert installed_receipt(s.r["dest"])["archive_sha256"] == sha(
            archive.read_bytes()
        )
        assert s.dashboard.state == "ready"
        assert [action for action, _ in s.dashboard.actions] == ["stop", "start"]
        with pytest.raises(ValueError):
            updates.offer(ACTOR)
    with closing(jobs.Updates(s.config, dashboard=s.dashboard)) as updates:
        settled(updates, "recovery_required")
        assert [action for action, _ in s.dashboard.actions] == ["stop", "start"]


def interrupted_worker(jobs, config, offer_id, entered, release_stop):
    dashboard = Dashboard(Path(config.maintenance["installation"]))

    def stopping():
        entered.set()
        assert release_stop.wait(30), "Crash-test parent did not terminate its worker"

    dashboard.before_stop = stopping
    with closing(jobs.Updates(config, dashboard=dashboard)) as updates:
        updates.apply(ACTOR, offer_id, confirm_restart_and_rollback=True)
        release_stop.wait(30)


def test_restart_preserves_offer_and_worker_loss_requires_manual_recovery(
    jobs, scene, tmp_path
):
    s = scene
    s.publish(candidate(s.r, tmp_path, "candidate"))
    before = snapshot(s.r["dest"]), snapshot(history(s.r["dest"]))
    with closing(jobs.Updates(s.config, dashboard=s.dashboard)) as updates:
        offer_id = offered(updates)
        offered_record = record(s)
    artifacts = snapshot(s.state / "candidate")
    with closing(jobs.Updates(s.config, dashboard=s.dashboard)) as updates:
        assert updates.status()["phase"] == "offered"
        assert updates.status()["offer"]["id"] == offer_id
        assert record(s) == offered_record
        assert s.dashboard.actions == []

    # A separate process owns both its lock and worker thread. Kill only that
    # process after the durable stopping boundary, never a recorded service PID.
    context = multiprocessing.get_context("fork")
    entered, release_stop = context.Event(), context.Event()
    child = context.Process(
        target=interrupted_worker,
        args=(jobs, s.config, offer_id, entered, release_stop),
    )
    child.start()
    try:
        assert entered.wait(20), f"Worker did not reach stop (exit={child.exitcode})"
        interrupted = record(s)
        assert interrupted["phase"] == "stopping"
        child.kill()
        child.join(10)
        assert child.exitcode is not None
    finally:
        # A killed process may hold an Event's internal lock; never reuse its IPC.
        if child.is_alive():
            child.kill()
        child.join(10)
        child.close()
    assert record(s) == interrupted
    with closing(jobs.Updates(s.config, dashboard=s.dashboard)) as updates:
        recovered = updates.status()
        assert recovered["phase"] == "recovery_required"
        assert recovered["job"]["id"] == interrupted["id"]
        persisted = record(s)
        for key in ("id", "actor", "current", "target"):
            assert persisted[key] == interrupted[key]
        assert s.dashboard.actions == []
        assert snapshot(s.state / "candidate") == artifacts
        assert (snapshot(s.r["dest"]), snapshot(history(s.r["dest"]))) == before
