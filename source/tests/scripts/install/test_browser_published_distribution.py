"""Install the committed distribution through a fresh, verified local HTTPS issuer."""

import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile

import pytest

from tests.scripts.install.test_browser_distribution import (
    entry,
    https,  # noqa: F401
    terminal,
)
from tests.scripts.install.test_browser_maintenance import snapshot
from tests.scripts.install.test_browser_offline import release  # noqa: F401
from tests.scripts.install.test_browser_setup import layout  # noqa: F401

pytestmark = pytest.mark.linux_only
ROOT = Path(__file__).resolve().parents[4]


@pytest.mark.parametrize("shell", ["sh", "bash", "zsh"])
def test_published_distribution_install_inspect_and_uninstall(
    https, layout, monkeypatch, record_property, shell
):
    home, data, backend = layout
    (data / "active_profile").write_text("alpha\n")
    preserved = snapshot(data)
    # No ambient credentials, proxies, shell hooks, or Hermes selection overrides.
    env = {
        "HOME": str(home),
        "PATH": "/usr/bin:/bin",
        "LANG": "C.UTF-8",
        "LC_ALL": "C.UTF-8",
        "TZ": "UTC",
        "HERMES_TEST_ISOLATION": str(data),
        "SSL_CERT_FILE": str(https["cert"]),
        "CURL_CA_BUNDLE": str(https["cert"]),
    }
    monkeypatch.chdir(https["root"].parent)
    archive = ROOT / "hermes-browser.tar.gz"
    prepared = subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            str(ROOT / "prepare-browser-distribution.py"),
            "--archive",
            str(archive),
            "--sha256",
            hashlib.sha256(archive.read_bytes()).hexdigest(),
            "--launcher",
            str(ROOT / "hermes-browser.py"),
            "--source",
            https["url"],
            "--output",
            str(https["root"] / "distribution"),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    assert prepared.returncode == 0, prepared.stderr
    for path in (https["root"] / "distribution").iterdir():
        shutil.copy2(path, https["root"] / path.name)
    bootstrap = prepared.stdout.splitlines()[-1]
    record_property("local_bootstrap_command", bootstrap)
    code, output = terminal(entry({"bootstrap": bootstrap}, shell), env, stdin_pipe=True)
    record_property("pty_transcript", output)
    assert code == 0, output
    assert "Nothing started" in output
    assert set(https["requests"]) >= {
        "/install.sh",
        "/installer.pyz",
        "/CURRENT.json",
        "/hermes-browser.tar.gz",
        "/hermes-browser.py",
    }

    base = home / ".local/lib/hermes-browser"
    command = home / ".local/bin/hermes-browser"
    assert command.is_file()
    assert (base / "control").is_dir()
    inspected = subprocess.run(
        [str(command), "inspect"], env=env, capture_output=True, text=True, timeout=30
    )
    assert inspected.returncode == 0, inspected.stderr
    runtime = json.loads(inspected.stdout)["runtime"]
    assert runtime["compatibility"] == "not-exercised"
    assert runtime["backend_root"] == str(backend)
    assert runtime["profile_home"] == str(data / "profiles/alpha")
    web = next((base / "installation/versions").iterdir()) / "web"
    with tarfile.open(archive) as bundle:
        expected = {
            member.name.removeprefix("web/"): bundle.extractfile(member).read()
            for member in bundle
            if member.isfile() and member.name.startswith("web/")
        }
    assert {
        path.relative_to(web).as_posix(): path.read_bytes()
        for path in web.rglob("*")
        if path.is_file()
    } == expected

    code, output = terminal([str(command), "uninstall"], env)
    assert code == 0, output
    assert not command.exists()
    # Stable ownership locks intentionally survive removal of the owned payload.
    assert {path.name for path in base.iterdir()} == {
        "command.lock",
        "installation.run",
    }
    assert snapshot(data) == preserved
