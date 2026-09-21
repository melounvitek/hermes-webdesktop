"""Published controller migration; throwing backend fixtures, never real Hermes.

fixtures/browser_controller_fd244e9.pyz is the unmodified published installer from
/home/vitek/Work/hermes-webdesktop/installer.pyz, export source baseline fd244e9
(standalone main 2ac521f is a README-only change). Its SHA-256 is
619296ca659db909c6b28f25d033fd73dfad500cea2b209c15f40d01fbf713b1.
All four controller code modules were verified byte-for-byte against hermes-agent
73803796b1906495b635e7a77ed5a113fb73f52b before check-in. No Git objects are needed
at test time. The current preparation tool runs beside those genuine old modules
and rebinds only source.json to certificate-verified loopback HTTPS.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest

from tests.scripts.install.test_browser_commands import command, confirmed, serve
from tests.scripts.install.test_browser_distribution import (
    SCRIPTS,
    distribution,  # noqa: F401
    entry,
    https,  # noqa: F401
    terminal,
)
from tests.scripts.install.test_browser_maintenance import candidate, snapshot
from tests.scripts.install.test_browser_offline import release, sha  # noqa: F401
from tests.scripts.install.test_browser_setup import layout  # noqa: F401

# Only the browser's tmp_path asset updater runs, not the Hermes updater.
pytestmark = [pytest.mark.linux_only, pytest.mark.live_system_guard_bypass]


def test_published_controller_requires_explicit_stopped_reinstall(
    distribution, release, tmp_path, record_property
):
    d = distribution
    old = tmp_path / "published-controller"
    old.mkdir()
    fixture = Path(__file__).with_name("fixtures") / "browser_controller_fd244e9.pyz"
    with zipfile.ZipFile(fixture) as archive:
        archive.extractall(old)
    # Packaging must import the old siblings, not today's engine with old labels.
    prepare = old / "prepare-browser-distribution.py"
    shutil.copy2(SCRIPTS / prepare.name, prepare)
    old_launcher = old / "hermes-browser.py"
    initial = candidate(release, tmp_path, "published", launcher=old_launcher)
    prepared = tmp_path / "old-distribution"
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(prepare),
            "--archive",
            str(initial),
            "--sha256",
            sha(initial.read_bytes()),
            "--launcher",
            str(old_launcher),
            "--source",
            d["url"],
            "--output",
            str(prepared),
        ],
        env=d["env"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    assert result.returncode == 0, result.stderr
    for path in prepared.iterdir():
        shutil.copy2(path, d["root"] / path.name)
    code, output = terminal(entry(d), d["env"])
    assert code == 0, output
    assert "Nothing started" in output
    root = d["base"] / "installation"
    controller = snapshot(d["base"] / "control")
    original = snapshot(root)
    facade = d["command"].read_bytes()
    assert (
        d["base"] / "control/hermes-browser.py"
    ).read_bytes() == old_launcher.read_bytes()
    inspected = command(d, "inspect")
    assert inspected.returncode == 0, inspected.stderr
    assert json.loads(inspected.stdout)["runtime"]["compatibility"] == "reference-match"

    # Publish today's bootstrap and asset/launcher pair at the same trusted issuer.
    for path in (d["root"] / "distribution").iterdir():
        shutil.copy2(path, d["root"] / path.name)
    current = candidate(release, tmp_path, "current-assets")
    serve(d, current)
    before = snapshot(d["base"])
    requests = len(d["requests"])
    code, output = terminal(entry(d), d["env"])
    assert code == 0 and "Already installed" in output, output
    assert "Type yes" not in output
    assert "/CURRENT.json" not in d["requests"][requests:]
    assert snapshot(d["base"]) == before
    assert d["command"].read_bytes() == facade

    with (
        (d["base"] / "command.lock").open("rb") as command_lock,
        (d["base"] / "installation.run").open("rb") as run_lock,
    ):
        locks = {
            Path(handle.name): os.fstat(handle.fileno()).st_ino
            for handle in (command_lock, run_lock)
        }
        code, output = confirmed(d, "update")
        assert code == 0 and output.count("Type yes") == 1, output
        history = d["base"] / "installation.history"
        assert snapshot(history / sha(initial.read_bytes())) == original
        assert json.loads((root / "installation.json").read_bytes())[
            "archive_sha256"
        ] == sha(current.read_bytes())
        assert (root / "hermes-browser.py").read_bytes() == (
            d["root"] / "hermes-browser.py"
        ).read_bytes()
        assert snapshot(d["base"] / "control") == controller
        assert d["command"].read_bytes() == facade

        data = d["home"] / ".hermes"
        backend = data / "hermes-agent/hermes_cli/main.py"
        with backend.open("a") as source:
            source.write("# Same throwing fixture, different backend identity.\n")
        plugin = data / "plugins/terminal/plugin.py"
        plugin.parent.mkdir(parents=True)
        plugin.write_text("raise RuntimeError('never execute plugin')\n")
        preserved = snapshot(data)
        before = snapshot(d["base"])
        inspected = command(d, "inspect")
        assert inspected.returncode == 0, inspected.stderr
        assert json.loads(inspected.stdout)["runtime"]["compatibility"] == "untested"
        rejected = command(d, "start")
        assert rejected.returncode == 1, rejected.stderr
        assert "Untested backend references; startup unsupported" in rejected.stderr
        record_property("old_start_after_asset_update", rejected.stderr)
        rejected = command(d, "setup")
        assert rejected.returncode == 1, rejected.stderr
        assert "bundle reference files do not match" in rejected.stderr
        record_property("old_repeat_setup", rejected.stderr)

        # A downloaded setup can verify with newer rules, but must not replace the
        # persistent controller. The baseline itself rejects before Already installed.
        requests = len(d["requests"])
        code, output = terminal(entry(d), d["env"])
        if code == 0:
            assert "Already installed" in output, output
        else:
            assert code == 1 and "bundle reference files do not match" in output, output
        record_property("downloaded_repeat_setup", output)
        assert "Type yes" not in output
        assert "/CURRENT.json" not in d["requests"][requests:]
        assert snapshot(d["base"]) == before
        assert d["command"].read_bytes() == facade
        assert snapshot(data) == preserved
        assert "stopped" in command(d, "status").stdout

        code, output = confirmed(d, "uninstall", answer="no\n")
        assert code == 0 and "Type yes" in output, output
        assert snapshot(d["base"]) == before
        assert d["command"].read_bytes() == facade
        code, output = confirmed(d, "uninstall")
        assert code == 0 and output.count("Type yes") == 1, output
        assert not d["command"].exists()
        assert {path.name for path in d["base"].iterdir()} == {
            "command.lock",
            "installation.run",
        }
        assert not history.exists()
        assert (
            json.loads((d["base"] / "installation.run").read_bytes())["state"]
            == "stopped"
        )
        assert all(path.stat().st_ino == inode for path, inode in locks.items())
        assert snapshot(data) == preserved

        code, output = terminal(entry(d), d["env"])
        record_property("explicit_reinstall", output)
        assert code == 0 and output.count("Type yes") == 1, output
        assert "Nothing started" in output
        assert snapshot(d["base"] / "control") != controller
        inspected = command(d, "inspect")
        assert inspected.returncode == 0, inspected.stderr
        assert (
            json.loads(inspected.stdout)["runtime"]["compatibility"] == "not-exercised"
        )
        repeated = command(d, "setup")
        assert repeated.returncode == 0, repeated.stderr
        assert "Already installed" in repeated.stdout
        assert not history.exists()
        assert all(path.stat().st_ino == inode for path, inode in locks.items())
        assert snapshot(data) == preserved
        assert "stopped" in command(d, "status").stdout
