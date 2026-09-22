"""Exercise packaged entry points over certificate-verified loopback HTTPS."""

import functools
import http.server
import hashlib
import json
import os
from pathlib import Path
import pty
import select
import shutil
import signal
import ssl
import subprocess
import sys
import threading
import time

import pytest

from tests.scripts.install.test_browser_offline import release  # noqa: F401
from tests.scripts.install.test_browser_setup import layout  # noqa: F401

pytestmark = pytest.mark.linux_only
SCRIPTS = Path(__file__).resolve().parents[3] / "scripts"


@pytest.fixture
def https(tmp_path):
    root = tmp_path / "served"
    root.mkdir()
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(
        [
            "openssl",
            "req",
            "-x509",
            "-newkey",
            "rsa:2048",
            "-nodes",
            "-days",
            "1",
            "-keyout",
            str(key),
            "-out",
            str(cert),
            "-subj",
            "/CN=localhost",
            "-addext",
            "subjectAltName=DNS:localhost,IP:127.0.0.1",
        ],
        check=True,
        capture_output=True,
    )
    requests = []
    faults = {}

    class Handler(http.server.SimpleHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_GET(self):
            requests.append(self.path)
            if self.path in faults:
                status, headers, body = faults[self.path]
                self.send_response(status)
                for name, value in headers.items():
                    self.send_header(name, str(value))
                self.end_headers()
                self.wfile.write(body)
                self.close_connection = True
                return
            super().do_GET()

    server = http.server.ThreadingHTTPServer(
        ("127.0.0.1", 0), functools.partial(Handler, directory=str(root))
    )
    context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    context.load_cert_chain(cert, key)
    server.socket = context.wrap_socket(server.socket, server_side=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield dict(
            root=root,
            url=f"https://localhost:{server.server_port}/CURRENT.json",
            cert=cert,
            requests=requests,
            faults=faults,
        )
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


@pytest.fixture
def distribution(https, release, layout, monkeypatch):
    result = subprocess.run(
        [
            sys.executable,
            "-B",
            str(SCRIPTS / "prepare-browser-distribution.py"),
            "--archive",
            str(release["archive"]),
            "--sha256",
            hashlib.sha256(release["archive"].read_bytes()).hexdigest(),
            "--launcher",
            str(SCRIPTS / "hermes-browser.py"),
            "--source",
            https["url"],
            "--output",
            str(https["root"] / "distribution"),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    for path in (https["root"] / "distribution").iterdir():
        shutil.copy2(path, https["root"] / path.name)
    home, _, _ = layout
    monkeypatch.chdir(https["root"].parent)
    env = {
        **os.environ,
        "HOME": str(home),
        "SSL_CERT_FILE": str(https["cert"]),
        "CURL_CA_BUNDLE": str(https["cert"]),
        "PATH": "/usr/bin:/bin",
    }
    env.pop("HERMES_HOME", None)
    return {
        **https,
        "env": env,
        "home": home,
        "base": home / ".local/lib/hermes-browser",
        "command": home / ".local/bin/hermes-browser",
        "bootstrap": result.stdout.splitlines()[-1],
    }


def terminal(
    argv, env, answer="yes\n", prompt="Type yes", stdin_pipe=False, before_answer=None
):
    pid, fd = pty.fork()
    if pid == 0:
        if stdin_pipe:
            read, write = os.pipe()
            os.close(write)
            os.dup2(read, 0)
            os.close(read)
        os.execvpe(argv[0], list(map(str, argv)), env)
    output = bytearray()
    sent = False
    deadline = time.monotonic() + 30
    try:
        while time.monotonic() < deadline:
            if select.select([fd], [], [], 0.1)[0]:
                try:
                    block = os.read(fd, 65536)
                except OSError:
                    break
                if not block:
                    break
                output.extend(block)
                if not sent and prompt.encode() in output:
                    if before_answer is not None:
                        before_answer()
                    os.write(fd, answer.encode())
                    sent = True
            done, status = os.waitpid(pid, os.WNOHANG)
            if done:
                return os.waitstatus_to_exitcode(status), output.decode(
                    errors="replace"
                )
        done, status = os.waitpid(pid, os.WNOHANG)
        # PTY EOF can precede child exit; retain the original wait deadline.
        while not done and time.monotonic() < deadline:
            time.sleep(0.01)
            done, status = os.waitpid(pid, os.WNOHANG)
        if not done:
            os.kill(pid, signal.SIGKILL)
            _, status = os.waitpid(pid, 0)
            pytest.fail("PTY timed out: " + output.decode(errors="replace"))
        return os.waitstatus_to_exitcode(status), output.decode(errors="replace")
    finally:
        os.close(fd)


def entry(d, shell="sh"):
    # Exercise exactly the command printed by the maintainer preparation tool.
    executable = shutil.which(shell)
    if executable is None:
        pytest.skip(f"{shell} is not installed")
    return [executable, "-c", d["bootstrap"]]


@pytest.mark.parametrize("shell", ["sh", "bash", "zsh"])
def test_complete_download_then_tty_install_and_verify(
    distribution, record_property, shell
):
    d = distribution
    record_property("local_bootstrap_command", d["bootstrap"])
    with (d["home"] / ".hermes/hermes-agent/hermes_cli/main.py").open("a") as source:
        source.write("# Compatible fixture differs from the packaged reference.\n")
    code, output = terminal(entry(d, shell), d["env"], stdin_pipe=True)
    record_property("pty_transcript", output)
    assert code == 0, output
    assert "Nothing started" in output
    assert d["command"].is_file()
    assert (d["base"] / "control").is_dir()
    assert (d["base"] / "installation/installation.json").is_file()
    inspected = subprocess.run(
        [str(d["command"]), "inspect"], env=d["env"], text=True, capture_output=True
    )
    assert inspected.returncode == 0, inspected.stderr
    assert json.loads(inspected.stdout)["runtime"]["compatibility"] == "not-exercised"
    assert "/CURRENT.json" in d["requests"]


@pytest.mark.parametrize("shell", ["sh", "bash", "zsh"])
@pytest.mark.parametrize("fault", ["404", "500", "truncated", "tls", "connection"])
def test_bootstrap_download_failure_does_not_execute(distribution, shell, fault):
    d = distribution
    marker = Path.cwd() / "executed"
    payload = b"printf executed > executed\n"
    # A failed retry must not run a previously downloaded installer either.
    Path("hermes-browser-install.sh").write_bytes(payload)
    if fault in ("404", "500"):
        d["faults"]["/install.sh"] = (int(fault), {}, payload)
    elif fault == "truncated":
        d["faults"]["/install.sh"] = (
            200,
            {"Content-Length": len(payload) + 100},
            payload,
        )
    elif fault == "tls":
        d["env"].pop("CURL_CA_BUNDLE")
        d["env"].pop("SSL_CERT_FILE")
    else:
        d["bootstrap"] = d["bootstrap"].replace(
            d["url"].rsplit("/", 1)[0], "https://127.0.0.1:1"
        )
    result = subprocess.run(
        entry(d, shell), env=d["env"], text=True, capture_output=True, timeout=15
    )
    assert result.returncode != 0, result.stdout + result.stderr
    assert not marker.exists()
    assert "/installer.pyz" not in d["requests"]
    assert not d["base"].exists()


@pytest.mark.parametrize("answer", ["no\n", "\x04", "\x03"])
def test_decline_eof_interrupt_leave_no_final_files(distribution, answer):
    d = distribution
    code, output = terminal(entry(d), d["env"], answer=answer)
    assert "Type yes" in output
    assert not d["base"].exists()
    assert not d["command"].exists()


def test_no_tty_explains_safe_retry(distribution):
    d = distribution
    result = subprocess.run(
        entry(d),
        env=d["env"],
        input="yes\n",
        text=True,
        capture_output=True,
        start_new_session=True,
    )
    assert result.returncode != 0
    assert "terminal" in result.stderr.lower()
    assert not d["base"].exists()


@pytest.mark.parametrize("missing", ["hermes", "python", "curl", "sha256sum"])
def test_missing_prerequisite_installs_nothing(distribution, missing, tmp_path):
    d = distribution
    if missing == "hermes":
        shutil.rmtree(d["home"] / ".hermes")
    elif missing == "python":
        (d["home"] / ".hermes/hermes-agent/venv/bin/python").unlink()
    else:
        bin_dir = tmp_path / "bin"
        bin_dir.mkdir()
        for name in ("sh", "curl", "sha256sum", "mktemp", "rm"):
            if name != missing:
                (bin_dir / name).symlink_to(shutil.which(name))
        d["env"]["PATH"] = str(bin_dir)
    code, output = terminal(entry(d), d["env"])
    assert code != 0, output
    assert "Type yes" not in output
    assert not d["base"].exists()


@pytest.mark.parametrize("fault", ["hash", "redirect", "oversize", "truncated"])
def test_installer_download_is_not_executed_on_failure(distribution, fault):
    d = distribution
    path = d["root"] / "installer.pyz"
    if fault == "hash":
        path.write_bytes(b"not an installer")
    else:
        status = 302 if fault == "redirect" else 200
        headers = (
            {"Location": "https://localhost:1/no"}
            if fault == "redirect"
            else {
                "Content-Length": 9999999
                if fault == "oversize"
                else path.stat().st_size
            }
        )
        d["faults"]["/installer.pyz"] = (status, headers, b"partial")
    code, output = terminal(entry(d), d["env"])
    assert code != 0, output
    assert "/CURRENT.json" not in d["requests"]
    assert not d["base"].exists()


@pytest.mark.parametrize(
    "fault",
    ["404", "redirect", "hash", "truncated", "oversize", "pair", "extra", "insecure"],
)
def test_download_failures_never_publish(distribution, fault):
    d = distribution
    descriptor = json.loads((d["root"] / "CURRENT.json").read_bytes())
    name = descriptor["archive"]["name"]
    if fault == "404":
        (d["root"] / name).unlink()
    elif fault == "redirect":
        d["faults"]["/" + name] = (302, {"Location": "http://127.0.0.1:1/stolen"}, b"")
    elif fault == "hash":
        descriptor["archive"]["sha256"] = "0" * 64
    elif fault == "truncated":
        d["faults"]["/" + name] = (
            200,
            {"Content-Length": descriptor["archive"]["size"]},
            b"partial",
        )
    elif fault == "oversize":
        d["faults"]["/" + name] = (200, {"Content-Length": 999999999}, b"")
    elif fault == "pair":
        path = d["root"] / descriptor["launcher"]["name"]
        path.write_bytes(path.read_bytes() + b"\n# different paired engine\n")
        descriptor["launcher"].update(
            size=path.stat().st_size,
            sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        )
    elif fault == "extra":
        descriptor["variant"] = "other"
    else:
        descriptor["archive"]["name"] = "http://127.0.0.1:1/bad"
    (d["root"] / "CURRENT.json").write_text(json.dumps(descriptor))
    code, output = terminal(entry(d), d["env"])
    assert code != 0, output
    assert "Type yes" not in output
    assert not d["base"].exists()
    assert not d["command"].exists()


@pytest.mark.parametrize("collision", ["command", "control", "symlink", "data-overlap"])
def test_foreign_paths_are_preserved(distribution, collision):
    d = distribution
    target = d["command"] if collision == "command" else d["base"] / "control/foreign"
    if collision == "data-overlap":
        d["env"]["XDG_DATA_HOME"] = str(d["base"].parent)
        code, output = terminal(entry(d), d["env"])
        assert code == 1, output
        assert "Type yes" not in output
        assert not d["base"].exists()
        return
    target.parent.mkdir(parents=True)
    target.write_text("mine")
    if collision == "symlink":
        shutil.rmtree(d["base"])
        d["base"].symlink_to(d["home"] / ".hermes", target_is_directory=True)
    code, output = terminal(entry(d), d["env"])
    assert code != 0, output
    assert "Type yes" not in output
    assert not (d["base"] / "installation").exists()
    if collision != "symlink":
        assert target.read_text() == "mine"
