"""Offline browser distribution: exercise the portable CLI without running Hermes."""

import ctypes
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest


pytestmark = pytest.mark.linux_only

CLI = Path(__file__).resolve().parents[3] / "scripts" / "hermes-browser.py"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def run(*args, input="", env=None):
    return subprocess.run(
        [sys.executable, "-I", "-S", str(CLI), *map(str, args)],
        input=input,
        text=True,
        capture_output=True,
        timeout=20,
        env=env,
    )


@pytest.fixture
def release(tmp_path):
    web = tmp_path / "web"
    (web / "assets").mkdir(parents=True)
    (web / "index.html").write_bytes(b'<script src="/assets/app.js"></script>')
    (web / "assets/app.js").write_bytes(b"console.log('verified');")
    backend = tmp_path / "backend"
    (backend / "hermes_cli").mkdir(parents=True)
    (backend / "hermes_cli/main.py").write_text(
        "raise RuntimeError('never import Hermes')\n"
    )
    home = tmp_path / "custom-data"
    home.mkdir()
    (home / "config.yaml").write_text("private: do not load\n")
    (home / "active_profile").write_text("other\n")
    for name in ("alpha", "beta"):
        profile = home / "profiles" / name
        profile.mkdir(parents=True)
        (profile / "config.yaml").write_text("{}\n")
    receipt = {
        "schema": 1,
        "release": "fixture-1",
        "ui_commit": "a" * 40,
        "upstream_commit": "b" * 40,
        "verification": {
            "sha256": "c" * 64,
            "scope": "fixture browser checks; not a release certification",
        },
        "tested_backend": {
            "revision": "d" * 40,
            "reference_files": {
                "hermes_cli/main.py": sha((backend / "hermes_cli/main.py").read_bytes())
            },
        },
        "files": {
            str(p.relative_to(web)): {
                "size": p.stat().st_size,
                "sha256": sha(p.read_bytes()),
            }
            for p in sorted(web.rglob("*"))
            if p.is_file()
        },
    }
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt))
    archive = tmp_path / "release.tar.gz"
    result = run(
        "pack", "--web-dir", web, "--receipt", receipt_path, "--output", archive
    )
    assert result.returncode == 0, result.stderr
    destination = tmp_path / "offline-install"
    args = [
        "--archive",
        archive,
        "--sha256",
        sha(archive.read_bytes()),
        "--python",
        sys.executable,
        "--backend-root",
        backend,
        "--hermes-root",
        home,
        "--profile",
        "default",
        "--install-root",
        destination,
    ]
    return {
        "args": args,
        "archive": archive,
        "dest": destination,
        "web": web,
        "backend": backend,
        "home": home,
        "receipt": receipt_path,
    }


def test_preview_decline_install_repeat_and_inspect(release):
    r = release
    result = run("inspect", *r["args"])
    assert result.returncode == 0, result.stderr
    assert "reference-match" in result.stdout and "not exercised" in result.stdout
    assert str(r["home"]) in result.stdout and "web/index.html" in result.stdout
    assert not r["dest"].exists()
    assert run("install", *r["args"], input="no\n").returncode == 0
    assert not r["dest"].exists()
    result = run("install", *r["args"], input="yes\n")
    assert result.returncode == 0, result.stderr
    installed_web = next((r["dest"] / "versions").iterdir()) / "web"
    assert (installed_web / "index.html").read_bytes() == (
        r["web"] / "index.html"
    ).read_bytes()
    assert (r["dest"].stat().st_mode & 0o777) == 0o700
    assert (installed_web / "index.html").stat().st_mode & 0o777 == 0o600
    before = {str(p): p.stat().st_mtime_ns for p in r["dest"].rglob("*")}
    result = run("install", *r["args"], input="yes\n")
    assert result.returncode == 0 and "Already installed" in result.stdout, (
        result.stderr
    )
    assert before == {str(p): p.stat().st_mtime_ns for p in r["dest"].rglob("*")}
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-S",
            str(r["dest"] / "hermes-browser.py"),
            "inspect",
            "--install-root",
            str(r["dest"]),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "reference-match" in result.stdout


@pytest.mark.parametrize("profile", ["default", "alpha", "beta"])
def test_explicit_profile_and_isolated_probe(release, tmp_path, profile):
    trap = tmp_path / "sitecustomize.py"
    marker = tmp_path / "imported"
    trap.write_text(f"open({str(marker)!r}, 'w').write('BAD')\n")
    args = release["args"].copy()
    args[args.index("--profile") + 1] = profile
    env = {
        **os.environ,
        "PYTHONPATH": str(tmp_path),
        "HERMES_HOME": "/nonexistent/ambient",
        "PYTHONSTARTUP": str(trap),
    }
    result = run("inspect", *args, env=env)
    assert result.returncode == 0, result.stderr
    plan = json.loads(result.stdout)
    expected = (
        release["home"]
        if profile == "default"
        else release["home"] / "profiles" / profile
    )
    assert plan["runtime"]["profile_home"] == str(expected)
    assert not marker.exists()
    assert not list(release["backend"].rglob("__pycache__"))


@pytest.mark.parametrize(
    "option,value",
    [
        ("--python", "/missing/python"),
        ("--profile", "missing"),
        ("--profile", "../escape"),
        ("--sha256", "0" * 64),
    ],
)
def test_invalid_selection_writes_nothing(release, option, value):
    args = release["args"].copy()
    args[args.index(option) + 1] = value
    result = run("install", *args, input="yes\n")
    assert result.returncode != 0
    assert not release["dest"].exists()


@pytest.mark.parametrize("kind", ["empty", "foreign", "symlink", "data", "backend"])
def test_foreign_destinations_are_not_adopted(release, kind):
    dest = release["dest"]
    if kind == "symlink":
        dest.symlink_to(release["web"], target_is_directory=True)
    elif kind in ("data", "backend"):
        dest = release["home" if kind == "data" else "backend"] / "new-install"
    else:
        dest.mkdir()
        if kind == "foreign":
            (dest / "keep").write_text("mine")
    args = release["args"].copy()
    args[-1] = dest
    result = run("install", *args, input="yes\n")
    assert result.returncode != 0
    assert not (dest / "installation.json").exists()
    if kind == "foreign":
        assert (dest / "keep").read_text() == "mine"


@pytest.mark.parametrize(
    "kind",
    [
        "traversal",
        "absolute",
        "symlink",
        "hardlink",
        "fifo",
        "duplicate",
        "extra",
        "missing",
        "hash",
        "truncated",
    ],
)
def test_reject_unsafe_archives_even_with_matching_digest(release, tmp_path, kind):
    archive = release["archive"]
    with tarfile.open(archive) as src:
        members = [(m, src.extractfile(m).read()) for m in src.getmembers()]
    bad = tmp_path / "bad.tar.gz"
    with tarfile.open(bad, "w:gz") as dst:
        for m, data in members:
            if kind == "missing" and m.name == "web/index.html":
                continue
            if kind == "hash" and m.name == "web/index.html":
                data = b"x" * len(data)
            dst.addfile(m, io.BytesIO(data))
        if kind in {
            "traversal",
            "absolute",
            "symlink",
            "hardlink",
            "fifo",
            "duplicate",
            "extra",
        }:
            name = {
                "traversal": "../escaped",
                "absolute": str(tmp_path / "escaped"),
                "duplicate": "web/index.html",
            }.get(kind, "web/unexpected")
            m = tarfile.TarInfo(name)
            m.type = {
                "symlink": tarfile.SYMTYPE,
                "hardlink": tarfile.LNKTYPE,
                "fifo": tarfile.FIFOTYPE,
            }.get(kind, tarfile.REGTYPE)
            m.linkname = "/tmp/escape" if kind in {"symlink", "hardlink"} else ""
            dst.addfile(m, io.BytesIO(b""))
    if kind == "truncated":
        bad.write_bytes(bad.read_bytes()[:60])
    args = release["args"].copy()
    args[1] = bad
    args[3] = sha(bad.read_bytes())
    result = run("install", *args, input="yes\n")
    assert result.returncode != 0
    assert not release["dest"].exists()
    assert not (tmp_path / "escaped").exists()


def test_modified_install_and_changed_selection_refused(release):
    assert run("install", *release["args"], input="yes\n").returncode == 0
    args = release["args"].copy()
    args[args.index("--profile") + 1] = "alpha"
    assert run("install", *args, input="yes\n").returncode != 0
    web = next((release["dest"] / "versions").iterdir()) / "web"
    (web / "index.html").write_text("user changes")
    assert run("install", *release["args"], input="yes\n").returncode != 0
    assert run("inspect", "--install-root", release["dest"]).returncode != 0
    assert (web / "index.html").read_text() == "user changes"


def test_packer_requires_exact_verified_bytes(release, tmp_path):
    (release["web"] / "index.html").write_text("not verified")
    out = tmp_path / "other.tar.gz"
    result = run(
        "pack",
        "--web-dir",
        release["web"],
        "--receipt",
        release["receipt"],
        "--output",
        out,
    )
    assert result.returncode != 0
    assert not out.exists()


@pytest.mark.parametrize("target", ["backend", "home", "live"])
def test_double_slash_cannot_bypass_protected_paths(release, tmp_path, target):
    env = {**os.environ, "HOME": str(tmp_path)}
    protected = (
        release[target]
        if target != "live"
        else tmp_path / ".local/share/hermes-browser"
    )
    protected.mkdir(parents=True, exist_ok=True)
    args = release["args"].copy()
    args[-1] = "/" + str(protected / "new-install")
    result = run("install", *args, input="yes\n", env=env)
    assert result.returncode != 0
    assert not (protected / "new-install").exists()


def test_probe_ignores_protected_tmpdir(release):
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.inotify_init1(os.O_NONBLOCK | os.O_CLOEXEC)
    assert fd >= 0
    try:
        assert (
            libc.inotify_add_watch(fd, os.fsencode(release["backend"]), 0x100 | 0x200)
            >= 0
        )
        result = run(
            "inspect",
            *release["args"],
            env={**os.environ, "TMPDIR": str(release["backend"])},
        )
        assert result.returncode == 0, result.stderr
        with pytest.raises(BlockingIOError):
            os.read(fd, 65536)
    finally:
        os.close(fd)


def test_long_verified_filenames_pack_successfully(release, tmp_path):
    path = release["web"] / ("x" * 101 + ".js")
    path.write_bytes(b"long-name")
    receipt = json.loads(release["receipt"].read_text())
    receipt["files"][path.name] = {
        "size": path.stat().st_size,
        "sha256": sha(path.read_bytes()),
    }
    release["receipt"].write_text(json.dumps(receipt))
    out = tmp_path / "long.tar.gz"
    result = run(
        "pack",
        "--web-dir",
        release["web"],
        "--receipt",
        release["receipt"],
        "--output",
        out,
    )
    assert result.returncode == 0, result.stderr
    args = release["args"].copy()
    args[1], args[3] = out, sha(out.read_bytes())
    assert run("install", *args, input="yes\n").returncode == 0


def pending_install(release):
    process = subprocess.Popen(
        [sys.executable, "-I", "-S", str(CLI), "install", *map(str, release["args"])],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    # The complete JSON preview is flushed before reading confirmation.
    while True:
        line = process.stdout.readline()
        assert line, process.stderr.read()
        if line == "}\n":
            return process


def test_archive_replacement_during_confirmation_cannot_change_payload(release):
    process = pending_install(release)
    try:
        release["archive"].write_bytes(b"replaced after verification")
        _, stderr = process.communicate("yes\n", timeout=15)
        assert process.returncode == 0, stderr
        web = next((release["dest"] / "versions").iterdir()) / "web"
        assert (web / "index.html").read_bytes() == (
            release["web"] / "index.html"
        ).read_bytes()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


@pytest.mark.parametrize(
    "change", ["foreign-destination", "backend", "profile-deleted"]
)
def test_confirmation_revalidates_destination_and_runtime(release, change):
    process = pending_install(release)
    try:
        if change == "foreign-destination":
            release["dest"].mkdir()
            (release["dest"] / "keep").write_text("foreign")
        elif change == "backend":
            (release["backend"] / "hermes_cli/main.py").write_text("changed")
        else:
            release["home"].rename(release["home"].with_name("moved-data"))
        _, stderr = process.communicate("yes\n", timeout=15)
        assert process.returncode != 0, stderr
        assert not (release["dest"] / "installation.json").exists()
        assert not list(release["dest"].parent.glob(".hermes-browser-stage-*"))
        if change == "foreign-destination":
            assert (release["dest"] / "keep").read_text() == "foreign"
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_concurrent_installers_publish_one_complete_install(release):
    processes = [pending_install(release), pending_install(release)]
    try:
        for process in processes:
            process.stdin.write("yes\n")
            process.stdin.flush()
        for process in processes:
            process.communicate(timeout=15)
        assert sorted(p.returncode for p in processes) == [0, 1]
        result = run("inspect", "--install-root", release["dest"])
        assert result.returncode == 0, result.stderr
        assert not list(release["dest"].parent.glob(".hermes-browser-stage-*"))
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()


def test_named_tombstone_and_ghost_profile_are_refused(release):
    args = release["args"].copy()
    args[args.index("--profile") + 1] = "alpha"
    tombstone = release["home"] / "profiles/.deleted/alpha"
    tombstone.parent.mkdir()
    tombstone.write_text("deleted")
    assert run("inspect", *args).returncode != 0
    tombstone.unlink()
    (release["home"] / "profiles/alpha/config.yaml").unlink()
    assert run("inspect", *args).returncode != 0


def test_selected_venv_does_not_execute_site_hooks(release, tmp_path):
    venv = tmp_path / "venv"
    subprocess.run(
        [sys.executable, "-I", "-B", "-m", "venv", "--without-pip", str(venv)],
        check=True,
    )
    site = next((venv / "lib").glob("python*/site-packages"))
    marker = tmp_path / "site-executed"
    (site / "probe.pth").write_text(
        f"import pathlib; pathlib.Path({str(marker)!r}).touch()\n"
    )
    (site / "sitecustomize.py").write_text(f"open({str(marker)!r}, 'w').close()\n")
    args = release["args"].copy()
    args[args.index("--python") + 1] = venv / "bin/python"
    result = run("inspect", *args)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["runtime"]["python"] == str(venv / "bin/python")
    assert not marker.exists()
    assert not list(site.rglob("__pycache__"))


def test_separate_python_venv_is_not_an_install_destination(release, tmp_path):
    venv = tmp_path / "separate-venv"
    subprocess.run(
        [sys.executable, "-I", "-B", "-m", "venv", "--without-pip", str(venv)],
        check=True,
    )
    args = release["args"].copy()
    args[args.index("--python") + 1] = venv / "bin/python"
    args[-1] = venv / "browser-install"
    result = run("install", *args, input="yes\n")
    assert result.returncode != 0
    assert not (venv / "browser-install").exists()


@pytest.mark.parametrize("budget", ["tar", "manifest"])
def test_packer_enforces_reader_limits_before_publication(
    release, tmp_path, monkeypatch, budget
):
    spec = importlib.util.spec_from_file_location("browser_installer", CLI)
    installer = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(installer)
    if budget == "tar":
        # The verified payload fits; tar headers and end padding exceed this budget.
        monkeypatch.setattr(installer, "MAX_EXPANDED", 4096)
        error = "Expanded archive exceeds size limit"
    else:
        # The receipt fits; the formatted manifest plus launcher digest does not.
        monkeypatch.setattr(installer, "MAX_JSON", release["receipt"].stat().st_size)
        error = "JSON exceeds size limit"
    output = tmp_path / "too-large.tar.gz"
    from argparse import Namespace

    with pytest.raises(ValueError, match=error):
        installer.pack(
            Namespace(
                web_dir=str(release["web"]),
                receipt=str(release["receipt"]),
                output=str(output),
            )
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".hermes-browser-pack-*"))


def test_backend_difference_is_not_called_compatible(release):
    (release["backend"] / "hermes_cli/main.py").write_text("changed")
    result = run("inspect", *release["args"])
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["runtime"]["compatibility"] == "untested"
