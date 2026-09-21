"""Friendly installer contracts; no real Hermes imports or services."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import subprocess

import yaml
from types import SimpleNamespace

import pytest

from tests.scripts.install.test_browser_offline import release  # noqa: F401

pytestmark = pytest.mark.linux_only
SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"


@pytest.fixture
def setup_module():
    spec = importlib.util.spec_from_file_location(
        "browser_setup", SCRIPTS / "browser_setup.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def layout(release, tmp_path, monkeypatch):
    home = tmp_path / "user"
    home.mkdir(mode=0o700)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("HERMES_HOME", raising=False)
    monkeypatch.delenv("HERMES_INSTALL_DIR", raising=False)
    monkeypatch.setenv("PATH", "/usr/bin:/bin")
    data = home / ".hermes"
    shutil.copytree(release["home"], data)
    (data / "active_profile").unlink()
    backend = data / "hermes-agent"
    shutil.copytree(release["backend"], backend)
    subprocess.run(
        [
            sys.executable,
            "-I",
            "-B",
            "-m",
            "venv",
            "--without-pip",
            str(backend / "venv"),
        ],
        check=True,
    )
    site = next((backend / "venv/lib").glob("python*/site-packages"))
    (site / "yaml").symlink_to(Path(yaml.__file__).parent, target_is_directory=True)
    return home, data, backend


def options(**kwargs):
    return SimpleNamespace(
        backend_root=None, python=None, hermes_home=None, profile=None, **kwargs
    )


def test_detection_is_read_only_and_preserves_profile_and_venv(
    setup_module, layout, monkeypatch, release
):
    m = setup_module
    home, data, backend = layout
    manifest = json.loads(release["receipt"].read_bytes())
    args = options()
    selection = m.detect(args)
    assert selection == dict(
        python=str(backend / "venv/bin/python"),
        backend_root=str(backend),
        hermes_root=str(data),
        profile="default",
    )
    before = {str(p): p.stat().st_mtime_ns for p in home.rglob("*")}
    assert m.preflight(selection, manifest)["compatibility"] == "reference-match"
    assert before == {str(p): p.stat().st_mtime_ns for p in home.rglob("*")}
    for name in ("alpha", "beta", "alpha"):
        (data / "active_profile").write_text(name + "\n")
        assert m.detect(args)["profile"] == name
    monkeypatch.setenv("HERMES_HOME", str(data / "profiles/beta"))
    assert m.detect(args)["profile"] == "beta"
    args.profile = "default"
    assert m.detect(args)["profile"] == "default"
    custom = home / "custom"
    shutil.copytree(data, custom)
    monkeypatch.setenv("HERMES_HOME", str(custom / "profiles/alpha"))
    args.profile = None
    with pytest.raises(ValueError, match="Ambiguous"):
        m.detect(args)
    args.backend_root = str(custom / "hermes-agent")
    assert m.detect(args)["hermes_root"] == str(custom)
    assert m.detect(args)["profile"] == "alpha"
    args.hermes_home = str(data)
    assert m.detect(args)["hermes_root"] == str(data)


@pytest.mark.parametrize(
    "kind",
    [
        "missing",
        "python",
        "ambiguous-python",
        "active",
        "deleted",
        "incompatible",
        "wrapper",
    ],
)
def test_detection_refuses_without_mutation(
    setup_module, layout, release, monkeypatch, kind
):
    m = setup_module
    home, data, backend = layout
    args = options()
    manifest = json.loads(release["receipt"].read_bytes())
    if kind in ("missing", "wrapper"):
        shutil.rmtree(backend)
        if kind == "wrapper":
            bin_dir = home / "bin"
            bin_dir.mkdir()
            wrapper = bin_dir / "hermes"
            wrapper.write_text("#!/bin/sh\ntouch " + str(home / "executed") + "\n")
            wrapper.chmod(0o700)
            monkeypatch.setenv("PATH", str(bin_dir))
    elif kind == "python":
        (backend / "venv/bin/python").unlink()
    elif kind == "ambiguous-python":
        shutil.copytree(backend / "venv", backend / ".venv", symlinks=True)
    elif kind == "active":
        (data / "active_profile").write_text("not-there\n")
    elif kind == "deleted":
        (data / "active_profile").write_text("alpha\n")
        (data / "profiles/.deleted").mkdir()
        (data / "profiles/.deleted/alpha").touch()
    else:
        (backend / "hermes_cli/main.py").write_text("different\n")
    before = {str(p): p.lstat().st_mtime_ns for p in home.rglob("*")}
    with pytest.raises((ValueError, OSError)):
        m.preflight(m.detect(args), manifest)
    assert before == {str(p): p.lstat().st_mtime_ns for p in home.rglob("*")}
    assert not (home / "executed").exists()
