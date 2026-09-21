"""Review regressions using only owned PTYs, HTTPS and disposable fake Hermes."""

import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from tests.scripts.install.test_browser_distribution import (
    SCRIPTS,
    distribution,
    https,
    entry,
    terminal,  # noqa: F401
)
from tests.scripts.install.test_browser_setup import layout, options  # noqa: F401
from tests.scripts.install.test_browser_offline import release  # noqa: F401
from tests.scripts.install.test_browser_commands import installed, command, serve  # noqa: F401
from tests.scripts.install.test_browser_maintenance import snapshot, candidate

pytestmark = pytest.mark.linux_only


@pytest.mark.parametrize("layer", ["outer", "inner"])
def test_curlrc_cannot_disable_bootstrap_tls(distribution, layer):
    d = distribution
    (d["home"] / ".curlrc").write_text("insecure\n")
    env = dict(d["env"])
    env.pop("CURL_CA_BUNDLE")
    env.pop("SSL_CERT_FILE")
    argv = entry(d) if layer == "outer" else ["sh", str(d["root"] / "install.sh")]
    code, output = terminal(argv, env)
    assert code != 0, output
    assert not d["requests"], "Untrusted TLS must fail before executable bytes transfer"
    assert not d["base"].exists()


def test_release_asset_source_refused_with_pages_guidance(
    distribution, release, tmp_path
):
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(SCRIPTS / "prepare-browser-distribution.py"),
            "--archive",
            str(release["archive"]),
            "--sha256",
            "0" * 64,
            "--launcher",
            str(SCRIPTS / "hermes-browser.py"),
            "--output",
            str(tmp_path / "bad"),
            "--source",
            "https://github.com/OWNER/REPO/releases/latest/download/CURRENT.json",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "https://OWNER.github.io/REPO/CURRENT.json" in result.stderr
    assert not (tmp_path / "bad").exists()


@pytest.mark.parametrize(
    "answer", ["2\n2\nyes\n", "q\n", "99\n", "\x04", "2\nq\n", "2\n99\n", "2\n\x04"]
)
def test_ambiguous_selection_uses_tty_before_separate_confirmation(
    distribution, answer
):
    d = distribution
    original = d["home"] / ".hermes/hermes-agent"
    custom = d["home"] / "custom-backend"
    shutil.copytree(original, custom, symlinks=True)
    shutil.copytree(original / "venv", original / ".venv", symlinks=True)
    d["env"]["HERMES_INSTALL_DIR"] = str(custom)
    # Custom installer location is first; select the default checkout, then .venv.
    code, output = terminal(
        entry(d), d["env"], answer=answer, prompt="Choose backend-root"
    )
    if answer.endswith("yes\n"):
        assert code == 0, output
        owner = json.loads((d["base"] / "control/owner.json").read_bytes())
        assert owner["selection"]["backend_root"] == str(original)
        assert owner["selection"]["python"] == str(original / ".venv/bin/python")
        assert output.count("Type yes") == 1
        assert "Choose python" in output
    else:
        assert not d["base"].exists(), output
        assert "Type yes" not in output
        assert not d["command"].exists()


def test_custom_installer_venv_bootstraps_without_system_python(distribution, tmp_path):
    d = distribution
    custom = d["home"] / "custom-backend"
    (d["home"] / ".hermes/hermes-agent").rename(custom)
    d["env"]["HERMES_INSTALL_DIR"] = str(custom)
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name in ("sh", "curl", "sha256sum", "mktemp", "rm"):
        (bin_dir / name).symlink_to(shutil.which(name))
    d["env"]["PATH"] = str(bin_dir)
    code, output = terminal(entry(d), d["env"])
    assert code == 0, output
    owner = json.loads((d["base"] / "control/owner.json").read_bytes())
    assert owner["selection"]["python"] == str(custom / "venv/bin/python")


def test_explicit_profile_home_challenges_stored_profile(distribution):
    d = distribution
    (d["home"] / ".hermes/active_profile").write_text("alpha\n")
    code, output = terminal(entry(d), d["env"])
    assert code == 0, output
    before = snapshot(d["base"])
    result = command(
        d, "setup", "--hermes-home", str(d["home"] / ".hermes/profiles/beta")
    )
    assert result.returncode != 0, result.stdout
    assert "selection differs" in result.stderr
    assert snapshot(d["base"]) == before


@pytest.mark.parametrize(
    "fault",
    ["dotenv", "secrets", "parser", ".update-incomplete", ".lazy-refresh-incomplete"],
)
def test_unstartable_configuration_refused_before_confirmation(distribution, fault):
    d = distribution
    data = d["home"] / ".hermes"
    backend = data / "hermes-agent"
    secret = "PRIVATE_VALUE_MUST_NOT_BE_PRINTED"
    if fault == "dotenv":
        (data / ".env").write_text("TOKEN=" + secret)
    elif fault == "secrets":
        (data / "config.yaml").write_text("secrets:\n  provider: " + secret + "\n")
    elif fault == "parser":
        next((backend / "venv/lib").glob("python*/site-packages/yaml")).unlink()
    else:
        (backend / fault).touch()
    before = snapshot(data)
    code, output = terminal(entry(d), d["env"])
    assert code != 0, output
    assert "Type yes" not in output
    assert secret not in output
    assert not d["base"].exists()
    assert snapshot(data) == before


@pytest.mark.parametrize("checkpoint", ["control", "installation"])
def test_owned_interrupted_publication_recovery(distribution, monkeypatch, checkpoint):
    d = distribution
    (d["home"] / ".hermes/active_profile").write_text("alpha\n")
    spec = importlib.util.spec_from_file_location(
        "review_install", SCRIPTS / "browser_install.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "packaged_source", lambda: d["url"])
    original_read = module.E.read_regular
    monkeypatch.setattr(
        module.E,
        "read_regular",
        lambda path, *args: (
            json.dumps({"source": d["url"]}).encode()
            if path.name == "source.json"
            else original_read(path, *args)
        ),
    )
    monkeypatch.setenv("SSL_CERT_FILE", str(d["cert"]))
    monkeypatch.setattr(module, "confirm", lambda preview: True)
    rename = module.E.atomic_rename
    publish = module.publish_file

    def after_control(stage, target, *args, **kwargs):
        rename(stage, target, *args, **kwargs)
        if checkpoint == "control" and target == d["base"] / "control":
            raise KeyboardInterrupt

    def before_command(path, *args):
        if checkpoint == "installation" and path == d["command"]:
            raise KeyboardInterrupt
        publish(path, *args)

    mask = os.umask(0o077)
    try:
        with monkeypatch.context() as patch:
            patch.setattr(module.E, "atomic_rename", after_control)
            patch.setattr(module, "publish_file", before_command)
            with pytest.raises(KeyboardInterrupt):
                module.setup(options())
    finally:
        os.umask(mask)
    control = snapshot(d["base"] / "control")
    assert not d["command"].exists()
    foreign = d["base"] / checkpoint / "foreign"
    foreign.write_text("not owned")
    damaged = snapshot(d["base"])
    code, output = terminal(entry(d), d["env"])
    assert code != 0 and "Type yes" not in output, output
    assert snapshot(d["base"]) == damaged
    foreign.unlink()
    (d["home"] / ".hermes/active_profile").write_text("beta\n")
    requests = len(d["requests"])
    if checkpoint == "installation":
        inode = (d["base"] / "installation.run").stat().st_ino
        for answer in ("no\n", "yes\n"):
            code, output = terminal(entry(d), d["env"], answer=answer)
            assert code == 0, output
            assert output.count("Type yes") == 1
            assert d["command"].exists() == (answer == "yes\n")
        assert snapshot(d["base"] / "control") == control
        assert (d["base"] / "installation.run").stat().st_ino == inode
        assert command(d, "inspect").returncode == 0
        assert "Profile: alpha" in output
        assert "/CURRENT.json" not in d["requests"][requests:]
    else:
        # No pinned archive survived control publication. Explicitly remove only
        # the verified owned controller; do not silently fetch today's release.
        code, output = terminal(entry(d), d["env"])
        assert code != 0 and "uninstall" in output, output
        assert snapshot(d["base"] / "control") == control
        assert "/CURRENT.json" not in d["requests"][requests:]
        for answer in ("no\n", "yes\n"):
            code, output = terminal(
                ["sh", str(d["root"] / "install.sh"), "uninstall"],
                d["env"],
                answer=answer,
            )
            assert code == 0, output
            assert output.count("Type yes") == 1
            assert (d["base"] / "control").exists() == (answer == "no\n")
            assert "Profile: alpha" in output
        inode = (d["base"] / "installation.run").stat().st_ino
        code, output = terminal(entry(d), d["env"])
        assert code == 0, output
        assert (d["base"] / "installation.run").stat().st_ino == inode


@pytest.mark.parametrize("action", ["setup", "update"])
def test_confirmation_revalidates_startup_config(
    distribution, release, tmp_path, action
):
    d = distribution
    if action == "update":
        code, output = terminal(entry(d), d["env"])
        assert code == 0, output
        serve(d, candidate(release, tmp_path, "next"))
    before = snapshot(d["base"]) if d["base"].exists() else None
    argv = entry(d) if action == "setup" else [str(d["command"]), "update"]
    code, output = terminal(
        argv,
        d["env"],
        before_answer=lambda: (d["home"] / ".hermes/.env").write_text(
            "TOKEN=private-no-newline"
        ),
    )
    assert code != 0, output
    assert "Type yes" in output and "private-no-newline" not in output
    assert (snapshot(d["base"]) if d["base"].exists() else None) == before


@pytest.mark.live_system_guard_bypass
def test_update_help_names_asset_only_scope(installed):
    result = command(installed, "update", "--help")
    assert result.returncode == 0
    assert "assets" in result.stdout and "controller" in result.stdout
