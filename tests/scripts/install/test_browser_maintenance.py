"""Round 3 maintenance contracts; only the existing protocol fixture may run."""

from contextlib import contextmanager
import fcntl
import io
import json
import os
import select
import shutil
import subprocess
import sys
import tarfile

import pytest

from tests.scripts.install.test_browser_offline import (
    CLI,
    controllers,  # noqa: F401 -- registers the retained-pidfd cleanup fixture
    dashboard,  # noqa: F401 -- depends on the imported release fixture
    installer_module,  # noqa: F401
    lifecycle,
    release,  # noqa: F401
    run,
    sha,
    state_is,
    wait_for,
)


# The guard matches any Hermes-named command with an "update" argument.
# This standalone browser CLI only targets tmp_path installations, never the
# backend updater; runtime children use the retained-pidfd protocol fixture.
pytestmark = [pytest.mark.linux_only, pytest.mark.live_system_guard_bypass]


def history(root):
    return root.with_name(root.name + ".history")


def control(root):
    return root.with_name(root.name + ".run")


def snapshot(root):
    """Include empty directories and links, not just the known inventory."""
    if not os.path.lexists(root):
        return None
    result = {}
    for path in [root, *sorted(root.rglob("*"))]:
        name = str(path.relative_to(root))
        if path.is_symlink():
            result[name] = ("link", os.readlink(path))
        elif path.is_dir():
            result[name] = ("directory",)
        else:
            result[name] = ("file", path.read_bytes())
    return result


def rewrite_archive(archive, change):
    with tarfile.open(archive) as source:
        payload = {member.name: source.extractfile(member).read() for member in source}
    change(payload)
    with tarfile.open(archive, "w:gz") as target:
        for name, data in payload.items():
            member = tarfile.TarInfo(name)
            member.size = len(data)
            target.addfile(member, io.BytesIO(data))


def candidate(r, tmp_path, label, *, launcher=None, backend_match=True):
    folder = tmp_path / label
    folder.mkdir()
    web = folder / "web"
    shutil.copytree(r["web"], web)
    (web / "index.html").write_bytes(f"<h1>{label}</h1>".encode())
    (web / "assets/app.js").write_bytes(f"console.log({label!r});".encode())
    receipt = json.loads(r["receipt"].read_bytes())
    receipt["release"] = label
    receipt["files"] = {
        str(path.relative_to(web)): {
            "size": path.stat().st_size,
            "sha256": sha(path.read_bytes()),
        }
        for path in sorted(web.rglob("*"))
        if path.is_file()
    }
    if not backend_match:
        receipt["tested_backend"]["reference_files"]["hermes_cli/main.py"] = sha(
            b"another backend"
        )
    receipt_path = folder / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    archive = folder / "release.tar.gz"
    packed = run(
        "pack", "--web-dir", web, "--receipt", receipt_path, "--output", archive
    )
    assert packed.returncode == 0, packed.stderr
    if launcher is not None:

        def pair(payload):
            manifest = json.loads(payload["manifest.json"])
            manifest["launcher_sha256"] = sha(launcher.read_bytes())
            payload["manifest.json"] = json.dumps(manifest).encode()

        rewrite_archive(archive, pair)
    return archive


def update(r, archive, *, input="yes\n", launcher=None, digest=None):
    args = ["--archive", archive, "--sha256", digest or sha(archive.read_bytes())]
    if launcher is not None:
        args += ["--launcher", launcher]
    return maintenance(r, "update", *args, input=input)


# The shared lifecycle helper deliberately has no stdin argument.
# Keep maintenance on the same external CLI path, with confirmation explicit.
def maintenance(r, command, *args, input="yes\n"):
    return run(command, "--install-root", r["dest"], *args, input=input)


def installed_receipt(root):
    return json.loads((root / "installation.json").read_bytes())


def assert_retained(r, expected):
    result = lifecycle(r, "inspect")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["retained_versions"] == expected
    archive_root = history(r["dest"])
    if expected:
        assert json.loads((archive_root / "owner.json").read_bytes()) == {
            "owner": "hermes-browser-history-v1",
            "installation": str(r["dest"]),
        }
        assert {path.name for path in archive_root.iterdir()} == {
            "owner.json",
            *expected,
        }


def assert_refused(result):
    # argparse's unsupported-command exit must not make a security test green.
    assert result.returncode == 1, (result.stdout, result.stderr)


@pytest.fixture
def maintained(release):
    release["args"][release["args"].index("--profile") + 1] = "beta"
    result = run("install", *release["args"], input="yes\n")
    assert result.returncode == 0, result.stderr
    return release


@contextmanager
def pending(r, command, *args):
    process = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-S",
            str(CLI),
            command,
            "--install-root",
            str(r["dest"]),
            *map(str, args),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        preview = bytearray()

        def preview_ready():
            if select.select([process.stdout], [], [], 0)[0]:
                chunk = os.read(process.stdout.fileno(), 65536)
                assert chunk, process.stderr.read()
                preview.extend(chunk)
            return b"\n}\n" in preview

        wait_for(preview_ready)
        yield process
    finally:
        if process.poll() is None:
            process.kill()
        process.wait(timeout=10)
        process.stdin.close()
        process.stdout.close()
        process.stderr.close()


def test_update_chain_and_rollback_preserve_complete_installations(
    maintained, tmp_path
):
    r = maintained
    root = r["dest"]
    original = snapshot(root)
    selection = installed_receipt(root)["selection"]
    a = sha(r["archive"].read_bytes())
    retained = {a: json.loads(r["receipt"].read_bytes())["release"]}
    snapshots = {a: original}
    for label in ("release-B", "release-C"):
        archive = candidate(r, tmp_path, label)
        result = update(r, archive)
        assert result.returncode == 0, result.stderr
        assert_retained(r, retained)
        for digest, expected in snapshots.items():
            assert snapshot(history(root) / digest) == expected
        receipt = installed_receipt(root)
        digest = sha(archive.read_bytes())
        assert receipt["archive_sha256"] == digest
        assert receipt["owner"] == "hermes-browser-offline-v1"
        assert receipt["selection"] == selection
        with tarfile.open(archive) as source:
            for member in source:
                assert (
                    root / "versions" / digest / member.name
                ).read_bytes() == source.extractfile(member).read()
        assert {path.name for path in root.iterdir()} == {
            "installation.json",
            "hermes-browser.py",
            "versions",
        }
        assert {path.name for path in (root / "versions").iterdir()} == {digest}
        snapshots[digest] = snapshot(root)
        retained[digest] = label
    result = maintenance(r, "rollback", "--to", a)
    assert result.returncode == 0, result.stderr
    assert snapshot(root) == original
    assert_retained(r, retained)
    for digest, expected in snapshots.items():
        assert snapshot(history(root) / digest) == expected
    assert state_is(r, "stopped")
    assert not list(r["home"].glob("launch*.json"))


@pytest.mark.parametrize("failure", ["digest", "manifest", "backend"])
def test_bad_candidate_leaves_current_and_history_untouched(
    maintained, tmp_path, failure
):
    r = maintained
    archive = candidate(r, tmp_path, "candidate", backend_match=failure != "backend")
    if failure == "manifest":

        def corrupt(payload):
            payload["web/index.html"] = b"not the verified bytes"

        rewrite_archive(archive, corrupt)
    before = snapshot(r["dest"]), snapshot(history(r["dest"]))
    result = update(
        r, archive, digest=sha(b"wrong archive") if failure == "digest" else None
    )
    assert_refused(result)
    assert (snapshot(r["dest"]), snapshot(history(r["dest"]))) == before


@pytest.mark.parametrize("answer", ["no\n", "YES\n", " yes \n"])
def test_only_exact_yes_confirms_any_maintenance(maintained, tmp_path, answer):
    r = maintained
    archive = candidate(r, tmp_path, "release-B")
    assert update(r, archive).returncode == 0
    before = snapshot(r["dest"]), snapshot(history(r["dest"]))
    for command, args in (
        (
            "update",
            ("--archive", r["archive"], "--sha256", sha(r["archive"].read_bytes())),
        ),
        ("rollback", ("--to", sha(r["archive"].read_bytes()))),
        ("uninstall", ()),
    ):
        result = maintenance(r, command, *args, input=answer)
        assert result.returncode == 0, result.stderr
        assert (snapshot(r["dest"]), snapshot(history(r["dest"]))) == before


@pytest.mark.parametrize(
    "damage",
    [
        "root-modified",
        "root-foreign",
        "root-empty",
        "history-modified",
        "history-empty",
        "history-foreign-owner",
    ],
)
def test_unowned_or_changed_content_blocks_all_maintenance(
    maintained, tmp_path, damage
):
    r = maintained
    archive = candidate(r, tmp_path, "release-B")
    assert update(r, archive).returncode == 0
    a = sha(r["archive"].read_bytes())
    target = r["dest"] if damage.startswith("root-") else history(r["dest"]) / a
    if damage.endswith("modified"):
        (target / "hermes-browser.py").write_bytes(b"user edits")
    elif damage.endswith("empty"):
        (target / "foreign-empty").mkdir()
    elif damage.endswith("owner"):
        (history(r["dest"]) / "owner.json").write_text(
            json.dumps({"owner": "somebody-else", "installation": str(r["dest"])})
        )
    else:
        (target / "foreign-file").write_bytes(b"keep me")
    before = snapshot(r["dest"]), snapshot(history(r["dest"]))
    for command, args in (
        ("update", ("--archive", r["archive"], "--sha256", a)),
        ("rollback", ("--to", a)),
        ("uninstall", ()),
    ):
        assert_refused(maintenance(r, command, *args))
        assert (snapshot(r["dest"]), snapshot(history(r["dest"]))) == before
    assert_refused(lifecycle(r, "inspect"))


def test_same_candidate_is_noop_without_confirmation(maintained, tmp_path):
    r = maintained
    archive = candidate(r, tmp_path, "release-B")
    assert update(r, archive).returncode == 0
    before = snapshot(r["dest"]), snapshot(history(r["dest"]))
    timestamps = {
        path: path.stat().st_mtime_ns
        for root in (r["dest"], history(r["dest"]))
        for path in [root, *root.rglob("*")]
    }
    result = update(r, archive, input="")
    assert result.returncode == 0, result.stderr
    assert (snapshot(r["dest"]), snapshot(history(r["dest"]))) == before
    assert all(path.stat().st_mtime_ns == stamp for path, stamp in timestamps.items())


@pytest.mark.parametrize("controller_state", ["running", "unknown"])
def test_maintenance_never_stops_or_signals_a_controller(
    dashboard, controllers, tmp_path, controller_state
):
    r = dashboard
    archive = candidate(r, tmp_path, "release-B")
    assert update(r, archive).returncode == 0
    owner, _ = controllers(r)
    wait_for(lambda: state_is(r, "ready"))
    if controller_state == "unknown":
        owner.kill()
        owner.wait(timeout=10)
        assert state_is(r, "unknown")
    before = (
        snapshot(r["dest"]),
        snapshot(history(r["dest"])),
        control(r["dest"]).read_bytes(),
    )
    for command, args in (
        (
            "update",
            ("--archive", r["archive"], "--sha256", sha(r["archive"].read_bytes())),
        ),
        ("rollback", ("--to", sha(r["archive"].read_bytes()))),
        ("uninstall", ()),
    ):
        assert_refused(maintenance(r, command, *args))
        assert not select.select([owner.fixture_pidfd], [], [], 2)[0]
        assert (
            snapshot(r["dest"]),
            snapshot(history(r["dest"])),
            control(r["dest"]).read_bytes(),
        ) == before
        if controller_state == "running":
            assert owner.poll() is None
            assert state_is(r, "ready")


@pytest.mark.parametrize("command", ["update", "rollback", "uninstall"])
def test_confirmation_holds_lock_against_start_and_other_mutations(
    dashboard, controllers, tmp_path, command
):
    r = dashboard
    archive = candidate(r, tmp_path, "release-B")
    assert update(r, archive).returncode == 0
    a = sha(r["archive"].read_bytes())
    commands = {
        "update": ("--archive", r["archive"], "--sha256", a),
        "rollback": ("--to", a),
        "uninstall": (),
    }
    before = snapshot(r["dest"]), snapshot(history(r["dest"]))
    with pending(r, command, *commands[command]) as holder:
        contender, _ = controllers(r)
        assert contender.wait(timeout=10) == 1
        assert contender.fixture_pidfd is None
        for other, args in commands.items():
            assert_refused(maintenance(r, other, *args))
        assert (snapshot(r["dest"]), snapshot(history(r["dest"]))) == before
        _, stderr = holder.communicate("no\n", timeout=10)
        assert holder.returncode == 0, stderr
    assert not list(r["home"].glob("launch*.json"))


def test_initial_publication_respects_existing_run_lock(release, installer_module):
    r = release
    with installer_module.open_control(r["dest"], True) as holder:
        fcntl.flock(holder, fcntl.LOCK_EX | fcntl.LOCK_NB)
        inode = control(r["dest"]).stat().st_ino
        assert_refused(run("install", *r["args"], input="yes\n"))
        assert not r["dest"].exists()
        assert control(r["dest"]).stat().st_ino == inode


def test_uninstall_removes_only_owned_files_and_reinstall_keeps_control_inode(
    maintained, tmp_path, installer_module
):
    r = maintained
    root = r["dest"]
    plugin = r["home"] / "plugins/terminal"
    plugin.mkdir(parents=True)
    (plugin / "plugin.py").write_bytes(b"raise RuntimeError('never execute plugin')\n")
    sibling = tmp_path / "unrelated"
    sibling.mkdir()
    (sibling / "keep").write_bytes(b"unrelated")
    protected = {path: snapshot(path) for path in (r["backend"], r["home"], sibling)}
    with installer_module.open_control(root, True):
        pass
    with control(root).open("r+b") as retained_handle:
        inode = os.fstat(retained_handle.fileno()).st_ino
        assert (
            json.loads(control(root).read_bytes())["owner"]
            == "hermes-browser-control-v2"
        )
        archive = candidate(r, tmp_path, "release-B")
        assert update(r, archive).returncode == 0
        assert (
            maintenance(
                r, "rollback", "--to", sha(r["archive"].read_bytes())
            ).returncode
            == 0
        )
        result = maintenance(r, "uninstall")
        assert result.returncode == 0, result.stderr
        assert not root.exists()
        assert not history(root).exists()
        assert control(root).stat().st_ino == inode
        assert json.loads(control(root).read_bytes())["state"] == "stopped"
        assert all(snapshot(path) == expected for path, expected in protected.items())
        result = run("install", *r["args"], input="yes\n")
        assert result.returncode == 0, result.stderr
        assert control(root).stat().st_ino == inode
        assert (
            json.loads(control(root).read_bytes())["owner"]
            == "hermes-browser-control-v2"
        )
        assert installed_receipt(root)["owner"] == "hermes-browser-offline-v1"
        assert all(snapshot(path) == expected for path, expected in protected.items())
        fcntl.flock(retained_handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_refused(maintenance(r, "uninstall"))
        assert root.is_dir()


def test_legacy_launcher_pair_is_preserved_never_executed_and_current_cli_can_start(
    dashboard, controllers, tmp_path
):
    r = dashboard
    marker = tmp_path / "candidate-launcher-executed"
    launcher = tmp_path / "separately-trusted-launcher.py"
    launcher.write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).touch()\nraise RuntimeError('must not execute')\n"
    )
    archive = candidate(r, tmp_path, "legacy-pair", launcher=launcher)
    result = update(r, archive, launcher=launcher)
    assert result.returncode == 0, result.stderr
    legacy = snapshot(r["dest"])
    assert (r["dest"] / "hermes-browser.py").read_bytes() == launcher.read_bytes()
    result = lifecycle(r, "inspect")
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["runtime"]["profile"] == "alpha"
    owner, _ = controllers(r)
    wait_for(lambda: state_is(r, "ready"))
    assert lifecycle(r, "stop").returncode == 0
    assert owner.wait(timeout=10) == 0
    current = candidate(r, tmp_path, "current-pair")
    assert update(r, current).returncode == 0
    assert snapshot(history(r["dest"]) / sha(archive.read_bytes())) == legacy
    result = maintenance(r, "rollback", "--to", sha(archive.read_bytes()))
    assert result.returncode == 0, result.stderr
    assert snapshot(r["dest"]) == legacy
    assert not marker.exists()
    (r["dest"] / "hermes-browser.py").write_bytes(b"tampered original launcher")
    assert_refused(lifecycle(r, "inspect"))
    refused, _ = controllers(r)
    assert refused.wait(timeout=10) == 1
    assert refused.fixture_pidfd is None
    assert not marker.exists()


def test_missing_or_wrong_separately_trusted_launcher_refused(maintained, tmp_path):
    r = maintained
    launcher = tmp_path / "trusted.py"
    launcher.write_bytes(b"raise RuntimeError('not executed')\n")
    archive = candidate(r, tmp_path, "legacy-pair", launcher=launcher)
    before = snapshot(r["dest"]), snapshot(history(r["dest"]))
    assert_refused(update(r, archive))
    launcher.write_bytes(b"raise RuntimeError('different launcher')\n")
    assert_refused(update(r, archive, launcher=launcher))
    assert (snapshot(r["dest"]), snapshot(history(r["dest"]))) == before


@pytest.mark.parametrize("boundary", ["before", "after"])
def test_process_loss_at_exchange_keeps_complete_selection_and_history(
    maintained, tmp_path, boundary
):
    r = maintained
    original = snapshot(r["dest"])
    old_sha = sha(r["archive"].read_bytes())
    archive = candidate(r, tmp_path, "release-B")
    new_sha = sha(archive.read_bytes())
    # Terminate only this test's maintenance process at the filesystem boundary.
    # No backend code, cleanup hook, or production fault flag is involved.
    runner = """
import importlib.util, os, sys
from argparse import Namespace
spec = importlib.util.spec_from_file_location("installer", sys.argv[1])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
rename = m.atomic_rename
def interrupt(source, destination, exchange=False):
    if exchange and sys.argv[5] == "before":
        os._exit(73)
    rename(source, destination, exchange=exchange)
    if exchange and sys.argv[5] == "after":
        os._exit(73)
m.atomic_rename = interrupt
m.maintenance(Namespace(command="update", install_root=sys.argv[2], archive=sys.argv[3], sha256=sys.argv[4], launcher=None))
"""
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            runner,
            str(CLI),
            str(r["dest"]),
            str(archive),
            new_sha,
            boundary,
        ],
        input="yes\n",
        text=True,
        capture_output=True,
        timeout=20,
    )
    assert result.returncode == 73, result.stderr
    assert installed_receipt(r["dest"])["archive_sha256"] == (
        old_sha if boundary == "before" else new_sha
    )
    assert lifecycle(r, "inspect").returncode == 0
    assert snapshot(history(r["dest"]) / old_sha) == original
    assert state_is(r, "stopped")
    # Crash leftovers are private and preserved, not automatically scavenged.
    leftovers = list(r["dest"].parent.glob(".hermes-browser-switch-*"))
    assert len(leftovers) == 1
    result = maintenance(r, "rollback", "--to", old_sha)
    assert result.returncode == 0, result.stderr
    assert snapshot(r["dest"]) == original


@pytest.mark.parametrize("protected", ["hermes_root", "backend_root"])
def test_history_namespace_cannot_be_a_preserved_runtime_or_data_root(
    maintained, tmp_path, protected
):
    r = maintained
    archive = candidate(r, tmp_path, "release-B")
    assert update(r, archive).returncode == 0
    old = history(r["dest"]) / sha(r["archive"].read_bytes())
    for root in (r["dest"], old):
        receipt = installed_receipt(root)
        receipt["selection"][protected] = str(history(r["dest"]))
        receipt["selection"]["profile"] = "default"
        (root / "installation.json").write_text(json.dumps(receipt))
    before = snapshot(r["dest"]), snapshot(history(r["dest"]))
    assert_refused(maintenance(r, "uninstall"))
    assert (snapshot(r["dest"]), snapshot(history(r["dest"]))) == before


@pytest.mark.parametrize("sidecar_exists", [True, False])
@pytest.mark.parametrize("action", ["decline", "invalid", "noop"])
def test_unconfirmed_maintenance_does_not_fence_legacy_start(
    maintained, tmp_path, sidecar_exists, action
):
    r = maintained
    path = control(r["dest"])
    if sidecar_exists:
        record = json.loads(path.read_bytes())
        record["owner"] = "hermes-browser-offline-v1"
        path.write_text(json.dumps(record))
    else:
        path.unlink()  # Model a legacy installation that has never been started.
    archive = r["archive"] if action == "noop" else candidate(r, tmp_path, "release-B")
    result = update(
        r, archive, input="no\n", digest=sha(b"wrong") if action == "invalid" else None
    )
    assert result.returncode == (1 if action == "invalid" else 0), result.stderr
    assert json.loads(path.read_bytes())["owner"] == "hermes-browser-offline-v1"
    assert not history(r["dest"]).exists()


def test_rollback_rechecks_candidate_backend_references(release, tmp_path):
    r = release
    # Initial inspect/install can record an untested backend; maintenance cannot
    # activate that old release merely because the retained installation is valid.
    archive = candidate(r, tmp_path, "old-untested", backend_match=False)
    args = r["args"].copy()
    args[1], args[3] = archive, sha(archive.read_bytes())
    result = run("install", *args, input="yes\n")
    assert result.returncode == 0, result.stderr
    current = candidate(r, tmp_path, "current-tested")
    assert update(r, current).returncode == 0
    before = snapshot(r["dest"]), snapshot(history(r["dest"]))
    assert_refused(maintenance(r, "rollback", "--to", sha(archive.read_bytes())))
    assert (snapshot(r["dest"]), snapshot(history(r["dest"]))) == before
