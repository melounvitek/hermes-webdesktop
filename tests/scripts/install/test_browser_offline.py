"""Offline browser distribution: exercise the portable CLI without running Hermes."""

import ctypes
import fcntl
import hashlib
import importlib.util
import io
import json
import os
from pathlib import Path
import select
import signal
import socket
import time
import subprocess
import sys
import tarfile
import textwrap

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
    assert json.loads(result.stdout)["runtime"]["compatibility"] == "not-exercised"
    assert "not exercised" in result.stdout
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
    assert json.loads(result.stdout)["runtime"]["compatibility"] == "not-exercised"


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
    "change", ["foreign-destination", "missing-source-entry", "profile-deleted"]
)
def test_confirmation_revalidates_destination_and_runtime(release, change):
    process = pending_install(release)
    try:
        if change == "foreign-destination":
            release["dest"].mkdir()
            (release["dest"] / "keep").write_text("foreign")
        elif change == "missing-source-entry":
            (release["backend"] / "hermes_cli/main.py").unlink()
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


# This protocol fixture is not Hermes; stock acceptance runs separately in a
# read-only mount/network namespace with temporary homes and the real backend.
DASHBOARD_FIXTURE = "def main():\n" + textwrap.indent(
    """
import argparse, json, os, signal, sys, time
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
root = Path(os.environ['HERMES_HOME'])
mode = (root/'mode').read_text() if (root/'mode').exists() else 'normal'
p = argparse.ArgumentParser()
p.add_argument('-p')
p.add_argument('command')
p.add_argument('--host')
p.add_argument('--port', type=int)
for flag in ('isolated', 'skip-build', 'no-open'):
    if mode != 'reject-flags' or flag != 'skip-build':
        p.add_argument('--' + flag, action='store_true')
a, unknown = p.parse_known_args()
home = root if a.p == 'default' else root/'profiles'/a.p
record = root/('launch-' + str(a.port) + '.json')
record.with_suffix('.tmp').write_text(json.dumps({'argv': vars(a), 'env': dict(os.environ), 'pid': os.getpid(), 'home': str(home)}))
record.with_suffix('.tmp').replace(record)
while not (root/('release-' + str(a.port))).exists():
    time.sleep(.01)
(root/'launch.json').write_text(record.read_text())
if unknown:
    p.error('unrecognized arguments: ' + ' '.join(unknown))
if mode == 'exit':
    sys.exit(7)
if mode == 'stubborn':
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.environ['HERMES_WEB_DIST'], **kwargs)
    def do_GET(self):
        if self.path == '/api/health':
            body = {'bad-health': b'{"ok":false}', 'malformed-health': b'{', 'missing-health': b'{}'}.get(mode, b'{"ok":true}')
            self.send_response(503 if mode == 'health-error' else 200)
            self.end_headers(); self.wfile.write(body)
        elif mode == 'wrong':
            self.send_response(200); self.end_headers(); self.wfile.write(b'wrong bytes')
        else:
            super().do_GET()
server = HTTPServer((a.host, a.port), Handler)
if mode != 'no-sentinel':
    print('HERMES_DASHBOARD_READY port=' + str(a.port), flush=True)
server.serve_forever()
""",
    "    ",
)


@pytest.fixture
def dashboard(release):
    (release["backend"] / "hermes_cli/main.py").write_text(DASHBOARD_FIXTURE)
    receipt = json.loads(release["receipt"].read_text())
    receipt["tested_backend"]["reference_files"]["hermes_cli/main.py"] = sha(
        DASHBOARD_FIXTURE.encode()
    )
    release["receipt"].write_text(json.dumps(receipt))
    release["archive"].unlink()
    packed = run(
        "pack",
        "--web-dir",
        release["web"],
        "--receipt",
        release["receipt"],
        "--output",
        release["archive"],
    )
    assert packed.returncode == 0, packed.stderr
    release["args"][3] = sha(release["archive"].read_bytes())
    release["args"][release["args"].index("--profile") + 1] = "alpha"
    result = run("install", *release["args"], input="yes\n")
    assert result.returncode == 0, result.stderr
    return release


def lifecycle(release, command, *args):
    return run(command, "--install-root", release["dest"], *args)


def wait_for(check, timeout=12):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = check()
        if value:
            return value
        time.sleep(0.05)
    raise AssertionError("Timed out waiting for observable lifecycle state")


def state_is(release, state):
    result = lifecycle(release, "status")
    return result.returncode == 0 and json.loads(result.stdout)["state"] == state


def pidfd_open(pid):
    # The shared standalone Python lacks os.pidfd_open; use the host libc API.
    libc = ctypes.CDLL(None, use_errno=True)
    fd = libc.pidfd_open(pid, 0)
    if fd < 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    return fd


def kill_fixture(fd):
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.pidfd_send_signal(fd, signal.SIGKILL, None, 0) < 0:
        error = ctypes.get_errno()
        if error != 3:  # The fixture may already have exited.
            raise OSError(error, os.strerror(error))


@pytest.fixture
def controllers(tmp_path):
    children = []

    def start(release, mode="normal", env=None, port=None):
        (release["home"] / "mode").write_text(mode)
        if port is None:
            with socket.socket() as probe:
                probe.bind(("127.0.0.1", 0))
                port = probe.getsockname()[1]
        stdout = (tmp_path / f"controller-{len(children)}.stdout").open("w+")
        stderr = (tmp_path / f"controller-{len(children)}.stderr").open("w+")
        child = subprocess.Popen(
            [
                sys.executable,
                "-I",
                "-S",
                str(CLI),
                "start",
                "--install-root",
                str(release["dest"]),
                "--port",
                str(port),
                "--timeout",
                "2",
            ],
            stdout=stdout,
            stderr=stderr,
            env=env,
        )
        entry = [child, stdout, stderr, None]
        children.append(entry)
        marker = release["home"] / f"launch-{port}.json"
        wait_for(lambda: marker.exists() or child.poll() is not None)
        if marker.exists():
            pid = json.loads(marker.read_text())["pid"]
            fd = pidfd_open(pid)
            try:
                # Pin while the fixture waits for our handshake, then establish
                # identity. Never acquire kill authority from a stale PID alone.
                assert os.readlink(f"/proc/{pid}/cwd") == str(release["backend"])
                proc_stat = (
                    Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
                )
                assert int(proc_stat[1]) == child.pid
                entry[3] = fd
            except BaseException:
                os.close(fd)
                raise
            (release["home"] / f"release-{port}").touch()
        child.fixture_pidfd = entry[
            3
        ]  # Lifetime/cleanup remains owned by this fixture.
        return child, port

    yield start
    for child, stdout, stderr, fd in children:
        # Retained pidfds remain safe after reaping, controller crash, or PID reuse.
        if fd is not None:
            kill_fixture(fd)
            os.close(fd)
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        stdout.close()
        stderr.close()


def test_foreground_start_status_stop_and_changed_disk(dashboard, controllers):
    with (dashboard["backend"] / "hermes_cli/main.py").open("a") as source:
        source.write("# Same dashboard protocol, different source bytes.\n")
    assert state_is(dashboard, "stopped")
    env = {
        **os.environ,
        "HERMES_HOME": "/wrong",
        "HERMES_DESKTOP": "1",
        "HERMES_PARENT_PID": "1",
        "HERMES_DESKTOP_READY_FILE": "/must-not-write",
        "HERMES_SERVE_HEADLESS": "1",
        "HERMES_LAZY_INSTALL_TARGET": "/must-not-install",
        "HERMES_DASHBOARD_SESSION_TOKEN": "preserved-auth",
    }
    (dashboard["home"] / "profiles/alpha/.env").write_text(
        'HERMES_DASHBOARD_SESSION_TOKEN="ordinary-auth"\n'
    )
    (dashboard["home"] / "profiles/beta/.env").write_text("HERMES_HOME=/unselected\n")
    process, port = controllers(dashboard, env=env)
    wait_for(lambda: state_is(dashboard, "ready"))
    launch = json.loads((dashboard["home"] / "launch.json").read_text())
    assert launch["argv"] == {
        "p": "alpha",
        "command": "dashboard",
        "host": "127.0.0.1",
        "port": port,
        "isolated": True,
        "skip_build": True,
        "no_open": True,
    }
    assert launch["home"] == str(dashboard["home"] / "profiles/alpha")
    assert launch["env"]["HERMES_WEB_DIST"] == str(
        next((dashboard["dest"] / "versions").iterdir()) / "web"
    )
    assert launch["env"]["HERMES_DISABLE_LAZY_INSTALLS"] == "1"
    assert launch["env"]["HERMES_DASHBOARD_SESSION_TOKEN"] == "preserved-auth"
    assert not any(
        k in launch["env"]
        for k in (
            "HERMES_DESKTOP",
            "HERMES_PARENT_PID",
            "HERMES_DESKTOP_READY_FILE",
            "HERMES_SERVE_HEADLESS",
            "HERMES_LAZY_INSTALL_TARGET",
        )
    )
    # Startup and current disk identity are not conflated; changed UI cannot
    # prevent stopping the child the controller already owns.
    (dashboard["backend"] / "hermes_cli/main.py").unlink()
    status = json.loads(lifecycle(dashboard, "status").stdout)
    assert status["startup"]["compatibility"] == "not-exercised"
    assert status["current_disk"]["compatibility"] == "unavailable"
    (next((dashboard["dest"] / "versions").iterdir()) / "web/index.html").write_text(
        "changed"
    )
    stopped = lifecycle(dashboard, "stop")
    assert stopped.returncode == 0, stopped.stderr
    assert json.loads(stopped.stdout)["state"] == "stopped"
    assert process.wait(timeout=10) == 0
    assert state_is(dashboard, "stopped")


@pytest.mark.parametrize(
    "mode",
    [
        "wrong",
        "no-sentinel",
        "exit",
        "reject-flags",
        "bad-health",
        "malformed-health",
        "missing-health",
        "health-error",
    ],
)
def test_failed_readiness_is_not_ready_and_cleans_child(dashboard, controllers, mode):
    process, _ = controllers(dashboard, mode=mode)
    assert process.wait(timeout=12) != 0
    assert state_is(dashboard, "stopped")
    assert select.select([process.fixture_pidfd], [], [], 0)[0]


def test_occupied_port_and_duplicate_start_do_not_replace_owners(
    dashboard, controllers
):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen()
        first, _ = controllers(dashboard, port=listener.getsockname()[1])
        assert first.wait(timeout=10) != 0
        assert not (dashboard["home"] / "launch.json").exists()
        assert listener.getsockname()[1] > 0
    owner, _ = controllers(dashboard)
    wait_for(lambda: state_is(dashboard, "ready"))
    duplicate, _ = controllers(dashboard)
    assert duplicate.wait(timeout=10) != 0
    assert owner.poll() is None
    assert lifecycle(dashboard, "stop").returncode == 0
    assert owner.wait(timeout=10) == 0


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM])
def test_controller_signals_stop_only_owned_child(dashboard, controllers, sig):
    with subprocess.Popen([
        sys.executable,
        "-I",
        "-S",
        "-c",
        "import time; time.sleep(30)",
    ]) as unrelated:
        try:
            owner, _ = controllers(dashboard)
            wait_for(lambda: state_is(dashboard, "ready"))
            owner.send_signal(sig)
            assert owner.wait(timeout=10) == 0
            assert state_is(dashboard, "stopped")
            assert unrelated.poll() is None
        finally:
            unrelated.terminate()
            unrelated.wait()


def test_crashed_controller_and_stale_pid_never_authorize_kill(dashboard, controllers):
    owner, _ = controllers(dashboard)
    wait_for(lambda: state_is(dashboard, "ready"))
    owner.kill()
    owner.wait()
    assert state_is(dashboard, "unknown")
    assert lifecycle(dashboard, "stop").returncode != 0
    duplicate, _ = controllers(dashboard)
    assert duplicate.wait(timeout=10) != 0
    # The same pinned fixture generation survives the controller's exit.
    assert not select.select([owner.fixture_pidfd], [], [], 0)[0]


def test_uncooperative_child_is_not_force_killed(dashboard, controllers):
    owner, _ = controllers(dashboard, mode="stubborn")
    wait_for(lambda: state_is(dashboard, "ready"))
    result = lifecycle(dashboard, "stop", "--timeout", "2")
    assert result.returncode != 0
    assert json.loads(result.stdout)["state"] == "stopping"
    assert owner.poll() is None
    assert state_is(dashboard, "stopping")


@pytest.mark.parametrize("marker", [".update-incomplete", ".lazy-refresh-incomplete"])
def test_pending_backend_repair_is_refused(dashboard, controllers, marker):
    (dashboard["backend"] / marker).write_text("pending")
    process, _ = controllers(dashboard)
    assert process.wait(timeout=10) != 0
    assert not (dashboard["home"] / "launch.json").exists()


@pytest.fixture
def installer_module():
    spec = importlib.util.spec_from_file_location("browser_installer", CLI)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_stop_revalidates_record_after_controller_disconnect(
    dashboard, installer_module, monkeypatch, capsys
):
    from argparse import Namespace

    path = dashboard["dest"].with_name(dashboard["dest"].name + ".run")
    path.write_text(
        json.dumps({
            "owner": "foreign",
            "installation": "/another/install",
            "state": "stopped",
        })
    )
    path.chmod(0o600)
    with path.open("r+b") as holder:
        fcntl.flock(holder, fcntl.LOCK_EX)

        def disconnect(*_):
            fcntl.flock(holder, fcntl.LOCK_UN)
            raise ConnectionRefusedError()

        monkeypatch.setattr(installer_module, "control_request", disconnect)
        result = installer_module.lifecycle(
            Namespace(command="stop", install_root=str(dashboard["dest"]), timeout=2)
        )
    assert result != 0
    assert json.loads(capsys.readouterr().out)["state"] == "unknown"


@pytest.mark.parametrize(
    "content",
    [
        'export HERMES_HOME="/wrong"\n',
        "'HERMES_DESKTOP'=1\n",
        "\ufeffHERMES_DISABLE_LAZY_INSTALLS=0\n",
        "HERMES_PARENT_PID=1\n",
        "HERMES_LAZY_INSTALL_TARGET=/wrong\n",
        "HERMES_WEB_DIST=/wrong\n",
        " SECRET=synthetic \n",
        "SECRET=synthetic",
    ],
)
@pytest.mark.parametrize("source", ["profile", "op", "backend"])
def test_configuration_cannot_override_launch_controls(
    dashboard, controllers, source, content
):
    path = (
        dashboard["backend"] / ".env"
        if source == "backend"
        else dashboard["home"]
        / "profiles/alpha"
        / (".op.env" if source == "op" else ".env")
    )
    path.write_text(content)
    process, _ = controllers(dashboard)
    assert process.wait(timeout=12) != 0
    assert not (dashboard["home"] / "launch.json").exists()
    assert path.read_text() == content


@pytest.mark.parametrize(
    "config",
    [
        "secrets: {command: {enabled: true}}\n",
        '"secr\\u0065ts": {command: {enabled: true}}\n',
        "secrets: {command: {enabled: false}}\n",
        "base: &s {secrets: {command: {enabled: true}}}\n<<: *s\n",
    ],
)
def test_external_secret_sources_are_unsupported_not_executed(
    dashboard, controllers, config
):
    path = dashboard["home"] / "profiles/alpha/config.yaml"
    path.write_text(config)
    process, _ = controllers(dashboard)
    assert process.wait(timeout=12) != 0
    assert not (dashboard["home"] / "launch.json").exists()
    assert path.read_text() == config


def test_container_routing_is_refused_before_spawn(dashboard, controllers):
    marker = dashboard["home"] / "profiles/alpha/.container-mode"
    marker.write_text('{"runtime":"docker","container_name":"unrelated"}\n')
    process, _ = controllers(dashboard)
    assert process.wait(timeout=12) != 0
    assert not (dashboard["home"] / "launch.json").exists()
    assert marker.exists()


def test_managed_configuration_is_not_silently_bypassed(
    dashboard, controllers, tmp_path
):
    managed = tmp_path / "managed"
    managed.mkdir()
    process, _ = controllers(
        dashboard, env={**os.environ, "HERMES_MANAGED_DIR": str(managed)}
    )
    assert process.wait(timeout=12) != 0
    assert not (dashboard["home"] / "launch.json").exists()


def test_concurrent_state_publication_is_initialized_and_locked(
    dashboard, installer_module
):
    from concurrent.futures import ThreadPoolExecutor

    # Installation now creates its own control file. Exercise publication in a
    # fresh namespace, rather than opening that existing, unlocked inode.
    root = dashboard["dest"].with_name("unpublished")
    with ThreadPoolExecutor(max_workers=2) as pool:
        streams = list(
            pool.map(lambda _: installer_module.open_control(root, True), range(2))
        )
    try:
        assert len({os.fstat(stream.fileno()).st_ino for stream in streams}) == 1
        for stream in streams:
            assert installer_module.read_control(stream, root)["state"] == "stopped"
        with installer_module.open_control(root, False) as contender:
            with pytest.raises(BlockingIOError):
                fcntl.flock(contender, fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        for stream in streams:
            stream.close()
    assert not list(root.parent.glob(".hermes-browser-state-*"))


@pytest.mark.parametrize(
    "field,value",
    [("pid", True), ("pid", -1), ("generation", "invalid"), ("state", "ready")],
)
def test_invalid_record_never_authorizes_stop(
    dashboard, installer_module, field, value
):
    root = dashboard["dest"]
    record = {
        "owner": installer_module.OWNER,
        "installation": str(root),
        "state": "unknown",
        "generation": "f" * 32,
        "pid": 12345,
    }
    record[field] = value
    path = root.with_name(root.name + ".run")
    path.write_text(json.dumps(record))
    path.chmod(0o600)
    result = lifecycle(dashboard, "stop")
    assert result.returncode != 0
    assert json.loads(path.read_text()) == record


def test_inspection_reports_prerequisites_not_source_identity_or_api_compatibility(
    release,
):
    before = run("inspect", *release["args"])
    assert before.returncode == 0, before.stderr
    with (release["backend"] / "hermes_cli/main.py").open("a") as source:
        source.write("# Behavior-preserving difference from the receipt.\n")
    result = run("inspect", *release["args"])
    assert result.returncode == 0, result.stderr
    runtime = json.loads(result.stdout)["runtime"]
    assert runtime == json.loads(before.stdout)["runtime"]
    assert runtime["compatibility"] == "not-exercised"
    assert (
        runtime["tested_revision"]
        == json.loads(release["receipt"].read_bytes())["tested_backend"]["revision"]
    )
    assert "not exercised" in runtime["limitations"]
