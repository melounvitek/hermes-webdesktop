"""Permission failures explain safe, manual repairs before installation writes."""

import grp
import importlib.util
import os
from pathlib import Path
import pwd
import shlex
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from tests.scripts.install.test_browser_distribution import (
    distribution,  # noqa: F401
    entry,
    https,  # noqa: F401
    terminal,
)
from tests.scripts.install.test_browser_maintenance import snapshot
from tests.scripts.install.test_browser_offline import release  # noqa: F401
from tests.scripts.install.test_browser_setup import layout  # noqa: F401

pytestmark = pytest.mark.linux_only
SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"


@pytest.mark.parametrize("mode,repair", [(0o757, "o-w"), (0o777, "go-w")])
def test_packaged_installer_reports_all_permission_repairs(distribution, mode, repair):
    d = distribution
    home = d["home"].with_name("user's $(touch SHOULD_NOT_EXIST)")
    d["home"].rename(home)
    d.update(
        home=home,
        base=home / ".local/lib/hermes-browser",
        command=home / ".local/bin/hermes-browser",
    )
    d["env"]["HOME"] = str(home)
    paths = [home / ".local", home / ".local/lib", home / ".local/bin"]
    for path in paths:
        path.mkdir(mode=0o700)
        path.chmod(mode)
    before = snapshot(home)
    modes = {path: stat.S_IMODE(path.stat().st_mode) for path in [home, *paths]}
    result = subprocess.run(
        entry(d), env=d["env"], capture_output=True, text=True, start_new_session=True
    )
    assert result.returncode != 0
    assert "Installation stopped" in result.stderr
    assert "writable by other users" in result.stderr
    assert "not intentionally shared" in result.stderr
    assert "README" in result.stderr and "rerun" in result.stderr
    assert "No permissions were changed automatically" in result.stderr
    commands = [
        line.strip()
        for line in result.stderr.splitlines()
        if line.strip().startswith("chmod ")
    ]
    assert sorted(shlex.split(command) for command in commands) == sorted(
        ["chmod", repair, "--", str(path)] for path in paths
    )
    assert snapshot(home) == before
    assert {path: stat.S_IMODE(path.stat().st_mode) for path in modes} == modes
    assert "/CURRENT.json" not in d["requests"]
    assert "Type yes" not in result.stderr

    # Execute the actual printed shell commands, only inside the scratch home.
    subprocess.run(["sh", "-c", "\n".join(commands)], cwd=home, check=True)
    assert snapshot(home) == before
    assert {path: stat.S_IMODE(path.stat().st_mode) for path in modes} == {
        path: original & ~0o022 for path, original in modes.items()
    }
    code, output = terminal(entry(d), d["env"])
    assert code == 0, output
    assert d["command"].is_file()
    assert (d["base"] / "installation/installation.json").is_file()


@pytest.mark.parametrize("shared_by", [None, "member", "primary group", "name", "acl"])
def test_group_write_stops_installation_only_when_the_group_is_shared(
    tmp_path, monkeypatch, shared_by
):
    # Debian and Ubuntu default to a umask of 002 and one group per user, so
    # ~/.local/lib and ~/.local/bin are usually group-writable there.
    home = tmp_path / "home"
    paths = [home / ".local", home / ".local/lib", home / ".local/bin"]
    home.mkdir(mode=0o700)
    for path in paths:
        path.mkdir()
        path.chmod(0o775)
    monkeypatch.setenv("HOME", str(home))
    spec = importlib.util.spec_from_file_location(
        "group_install", SCRIPTS / "browser_install.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    gid = home.stat().st_gid
    user = SimpleNamespace(pw_name="me", pw_uid=os.getuid(), pw_gid=gid)
    other = SimpleNamespace(
        pw_name="other",
        pw_uid=os.getuid() + 1,
        pw_gid=gid if shared_by == "primary group" else gid + 1,
    )
    group = SimpleNamespace(
        gr_name="staff" if shared_by == "name" else "me",
        gr_mem=["me", "other"] if shared_by == "member" else ["me"],
    )
    monkeypatch.setattr(pwd, "getpwuid", lambda uid: user)
    monkeypatch.setattr(pwd, "getpwall", lambda: [user, other])
    monkeypatch.setattr(grp, "getgrgid", lambda gid: group)
    if shared_by == "acl":
        monkeypatch.setattr(os, "getxattr", lambda *args, **kwargs: b"acl")
    if shared_by is None:
        assert module.locations() == (
            home / ".local/lib/hermes-browser",
            home / ".local/bin/hermes-browser",
        )
    else:
        with pytest.raises(ValueError) as stopped:
            module.locations()
        assert "writable by other users" in str(stopped.value)
        for path in paths:
            assert f"chmod g-w -- {path}" in str(stopped.value)


@pytest.mark.parametrize("mode", [0o700, 0o777])
def test_foreign_ownership_has_no_chmod_remedy(tmp_path, monkeypatch, capsys, mode):
    home = tmp_path / "home"
    home.mkdir(mode=0o700)
    home.chmod(mode)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("HERMES_HOME", str(home / ".hermes"))
    spec = importlib.util.spec_from_file_location(
        "permission_install", SCRIPTS / "browser_install.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", ["hermes-browser", "setup"])
    uid = os.getuid()
    monkeypatch.setattr(os, "getuid", lambda: uid + 1)
    before = snapshot(home)
    mask = os.umask(0o077)
    try:
        assert module.main() == 1
    finally:
        os.umask(mask)
    error = capsys.readouterr().err
    assert "not owned by your user" in error
    assert str(home) in error
    assert "Inspect" in error
    assert "chmod " not in error
    assert "chown " not in error
    assert snapshot(home) == before
    assert stat.S_IMODE(home.stat().st_mode) == mode
