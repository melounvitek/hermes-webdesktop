"""Optional host-shell plugin through stock discovery/auth, real HTTP/WS and Linux PTYs."""
import asyncio
import contextlib
import importlib.util
import json
import os
from pathlib import Path
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import psutil
import pytest
from websockets.sync.client import connect
from websockets.exceptions import ConnectionClosed, InvalidStatus


ROOT = Path(__file__).resolve().parents[2]
PLUGIN = ROOT / "apps/desktop/browser-terminal-plugin"
API = "/api/plugins/browser-terminal"
TOKEN = "test-browser-terminal-token"
# Capture before the suite's per-test signal guard: cleanup may need to kill a
# child after a deliberately broken plugin has orphaned it. Only retained,
# verified descendant identities may use this narrow cleanup path.
_kill = os.kill


def owned_child(pid):
    child = psutil.Process(pid)
    assert os.getpid() in [parent.pid for parent in child.parents()]
    return child


def kill_child(child):
    if child.is_running():  # psutil also checks creation time, not just PID reuse.
        with contextlib.suppress(ProcessLookupError):
            _kill(child.pid, signal.SIGKILL)


SERVER = """
import sys
from pathlib import Path
import uvicorn
from hermes_cli import web_server
assert Path(web_server.__file__).resolve() == Path(sys.argv[4]) / 'hermes_cli/web_server.py'
web_server.app.state.bound_host = sys.argv[2]
web_server.app.state.auth_required = sys.argv[3] == 'gated'
if web_server.app.state.auth_required:
    from plugins.dashboard_auth.basic import BasicAuthProvider, hash_password
    from hermes_cli.dashboard_auth.registry import register_global_provider
    register_global_provider(BasicAuthProvider(username='alice', password_hash=hash_password('pw'),
                                               secret=b'isolated-browser-terminal-test'))
uvicorn.run(web_server.app, host='127.0.0.1', port=int(sys.argv[1]), log_level='info', proxy_headers=False)
"""


def wait_for(predicate, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.05)
    raise AssertionError("condition did not become true")


@contextlib.contextmanager
def server(tmp_path, backend, *, settings=None, installed=True, enabled=True, bound_host="127.0.0.1", gated=False):
    home = tmp_path / "home"
    hermes = home / ".hermes"
    hermes.mkdir(parents=True)
    (home / ".zshrc").touch()  # No first-run wizard in the isolated test account home.
    config = {"plugins": {"enabled": ["browser-terminal"] if enabled else [], "entries": {
        "browser-terminal": {"settings": settings or {}}}},
        "model_catalog": {"enabled": False}, "curator": {"enabled": False},
        "security": {"tirith_enabled": False}}
    config_path = hermes / "config.yaml"
    config_path.write_text(json.dumps(config))
    if installed:
        shutil.copytree(PLUGIN, hermes / "plugins/browser-terminal")
    for name in ("a", "b"):
        profile = hermes / "profiles" / name
        profile.mkdir(parents=True)
        work = home / ("work-" + name)
        work.mkdir()
        (profile / "config.yaml").write_text(json.dumps({"terminal": {"cwd": str(work)}}))
        (profile / ".env").write_text(f"OPENAI_API_KEY=secret-{name}\n")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    log_path = tmp_path / "server.log"
    # backend has already replaced the ambient environment with a safe allowlist.
    env = {**os.environ, "HOME": str(home), "HERMES_HOME": str(hermes),
           "PYTHONPATH": str(backend), "HERMES_DASHBOARD_SESSION_TOKEN": TOKEN,
           "OPENAI_API_KEY": "must-not-reach-shell", "UNRELATED_SECRET": "also-not-for-shell"}
    # This is a disposable runtime, not a nested pytest process. Its HOME is
    # deliberately also its real platform home; don't activate stock test guards.
    env.pop("PYTEST_CURRENT_TEST", None)
    env.pop("PYTEST_VERSION", None)
    with log_path.open("w") as log:
        process = subprocess.Popen([sys.executable, "-B", "-u", "-c", SERVER, str(port), bound_host,
                                    "gated" if gated else "loopback", str(backend)],
                                   cwd=home, env=env, stdout=log, stderr=log)
        try:
            url = f"http://127.0.0.1:{port}"
            token = user_token("alice") if gated else TOKEN
            with httpx.Client(base_url=url, headers={"Authorization": f"Bearer {token}", "Origin": url}, timeout=15) as client:
                def ready():
                    if process.poll() is not None:
                        raise AssertionError(log_path.read_text())
                    try:
                        return client.get("/api/status").status_code == 200
                    except httpx.TransportError:
                        return False
                wait_for(ready, timeout=60)
                yield client, url.replace("http:", "ws:"), hermes, config, log_path
        finally:
            # Retain process identities before shutdown can orphan any PTYs.
            children = psutil.Process(process.pid).children(recursive=True) if process.poll() is None else []
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired as exc:
                process.kill()
                process.wait()
                raise AssertionError("Server did not finish lifespan cleanup\n" + log_path.read_text()) from exc
            finally:
                # Clean up red runs, but don't hide plugin lifecycle bugs.
                _, alive = psutil.wait_procs(children, timeout=2)
                for child in reversed(alive):
                    kill_child(child)
                psutil.wait_procs(alive, timeout=2)
                assert not alive, f"Server left child processes running: {alive}"


def user_token(username):
    from plugins.dashboard_auth.basic import BasicAuthProvider, hash_password
    provider = BasicAuthProvider(username=username, password_hash=hash_password("pw"),
                                 secret=b"isolated-browser-terminal-test")
    return provider.complete_password_login(username=username, password="pw").access_token


def create(client, profile="current", **body):
    response = client.post(f"{API}/sessions", params={"profile": profile}, json=body)
    assert response.status_code == 200, response.text
    return response.json()["id"]


def ticket(client, sid, profile="current"):
    response = client.post(f"{API}/sessions/{sid}/ticket", params={"profile": profile})
    assert response.status_code == 200, response.text
    assert response.headers["cache-control"] == "no-store"
    return response.json()["ticket"]


def dial(client, ws_url, value, **kwargs):
    return connect(f"{ws_url}{API}/ws?ticket={value}", origin=str(client.base_url).rstrip("/"),
                   open_timeout=10, close_timeout=2, **kwargs)


def until(ws, marker):
    output = b""
    deadline = time.monotonic() + 15
    while marker not in output:
        try:
            frame = ws.recv(timeout=max(0.01, deadline - time.monotonic()))
        except TimeoutError as exc:
            raise AssertionError(repr(output)) from exc
        assert isinstance(frame, bytes)
        output += frame
    return output


def shell_pid(ws):
    ws.send(b"printf '\\nPID=%s\\n' $$\r")
    output = until(ws, b"\r\nPID=")
    while not output.split(b"\r\nPID=")[-1].split(b"\r\n")[0].isdigit():
        output += ws.recv(timeout=15)
    return int(output.split(b"\r\nPID=")[-1].split(b"\r\n")[0])


@pytest.mark.linux_only
def test_stock_mount_auth_reconnect_and_process_lifecycle(tmp_path, backend):
    pids = []
    values = []
    with server(tmp_path, backend) as (client, ws_url, hermes, config, log):
        assert client.post(f"{API}/sessions", json={}, headers={"Authorization": ""}).status_code == 401
        assert client.post(f"{API}/sessions", json={}, headers={"Origin": "https://evil.invalid"}).status_code == 403
        assert client.post(f"{API}/sessions", json={"env": {"EVIL": "1"}}).status_code == 422
        assert client.post(f"{API}/sessions", json={"cwd": "/missing/browser-terminal-test"}).status_code == 400
        sid = create(client)
        value = ticket(client, sid)
        values.append(value)
        with pytest.raises(InvalidStatus):
            connect(f"{ws_url}{API}/ws?ticket={value}", origin="https://evil.invalid")
        with dial(client, ws_url, value) as ws:
            pid = shell_pid(ws)
            pids.append(pid)
            ws.send(b"SAVED=still-here; printf '\\n%s\\n' READY\r")
            until(ws, b"\r\nREADY\r\n")
            with pytest.raises(InvalidStatus):
                dial(client, ws_url, value)
        value = ticket(client, sid)
        values.append(value)
        with dial(client, ws_url, value) as ws:
            until(ws, b"\r\nREADY\r\n")
            assert shell_pid(ws) == pid
            ws.send(json.dumps({"type": "resize", "cols": 91, "rows": 37}))
            ws.send(b"stty size; printf '\\n%s\\n' \"$SAVED\"\r")
            output = until(ws, b"\r\nstill-here\r\n")
            assert b"37 91" in output
            with dial(client, ws_url, ticket(client, sid)) as replacement:
                with pytest.raises(ConnectionClosed) as closed:
                    while True:
                        ws.recv(timeout=15)
                assert closed.value.rcvd.code == 4409
                replacement.send(b"exit\r")
                with pytest.raises(ConnectionClosed) as exited:
                    while True:
                        replacement.recv(timeout=15)
                assert exited.value.rcvd.code == 4410
        wait_for(lambda: not psutil.pid_exists(pid))
        assert client.post(f"{API}/sessions/{sid}/ticket").status_code == 404
        sid = create(client)
        with dial(client, ws_url, ticket(client, sid)) as ws:
            pid = shell_pid(ws)
            pids.append(pid)
            assert client.delete(f"{API}/sessions/{sid}").json() == {"ok": True}
        wait_for(lambda: not psutil.pid_exists(pid))
        sid = create(client)
        with dial(client, ws_url, ticket(client, sid)) as ws:
            pids.append(shell_pid(ws))
        value = ticket(client, sid)
        values.append(value)
        config["plugins"]["disabled"] = ["browser-terminal"]
        (hermes / "config.yaml").write_text(json.dumps(config))
        assert client.post(f"{API}/sessions/{sid}/ticket").status_code == 404
        with pytest.raises(InvalidStatus):
            dial(client, ws_url, value)
    wait_for(lambda: all(not psutil.pid_exists(pid) for pid in pids))
    assert all(value not in log.read_text() for value in values)


@pytest.mark.linux_only
def test_profile_ownership_limits_expiry_and_shutdown(tmp_path, backend):
    settings = {"max_sessions": 3, "max_tickets": 2, "ticket_ttl": 1,
                "detached_ttl": 3, "buffer_bytes": 4096}
    with server(tmp_path, backend, settings=settings) as (client, ws_url, hermes, config, log):
        sessions = []
        sockets = []
        pids = []
        try:
            for profile in ("a", "b", "a"):
                sid = create(client, profile)
                sessions.append(sid)
                assert client.post(f"{API}/sessions/{sid}/ticket?profile=current").status_code == 404
                ws = dial(client, ws_url, ticket(client, sid, profile))
                sockets.append(ws)
                pids.append(shell_pid(ws))
                ws.send(b"printf '\\nSCOPE=%s|%s|%s|%s\\n' \"$HERMES_HOME\" \"$PWD\" \"$OPENAI_API_KEY\" \"$UNRELATED_SECRET\"\r")
                expected = f"\r\nSCOPE={hermes}/profiles/{profile}|{hermes.parent}/work-{profile}||\r\n".encode()
                until(ws, expected)
                ws.send(b"printf '\\nHOST_PATH=%s\\n' \"$PATH\"\r")
                until(ws, f"\r\nHOST_PATH={os.environ['PATH']}\r\n".encode())
            sockets[-1].send(b"printf '%12000s\\n' X; printf '\\n%s\\n' REPLAY_TAIL\r")
            until(sockets[-1], b"\r\nREPLAY_TAIL\r\n")
            sockets[-1].close()
            sockets[-1] = dial(client, ws_url, ticket(client, sessions[-1], "a"))
            replay = sockets[-1].recv(timeout=15)
            assert b"\r\nREPLAY_TAIL\r\n" in replay
            assert len(replay) <= settings["buffer_bytes"]
            assert b"12000" not in replay
            assert client.post(f"{API}/sessions", json={}).status_code == 429
            assert client.delete(f"{API}/sessions/{sessions[0]}?profile=b").status_code == 404
            t1 = ticket(client, sessions[0], "a")
            ticket(client, sessions[0], "a")
            assert client.post(f"{API}/sessions/{sessions[0]}/ticket?profile=a").status_code == 429
            time.sleep(1.1)
            with pytest.raises(InvalidStatus):
                dial(client, ws_url, t1)
            # A disconnected shell, unlike a live attached one, is reclaimed.
            sockets[0].close()
            wait_for(lambda: not psutil.pid_exists(pids[0]))
            assert client.post(f"{API}/sessions/{sessions[0]}/ticket?profile=a").status_code == 404
            unattached = create(client, "a")
            wait_for(lambda: client.post(f"{API}/sessions/{unattached}/ticket?profile=a").status_code == 404)
            assert all(psutil.pid_exists(pid) for pid in pids[1:])
        finally:
            for ws in sockets:
                ws.close()
    wait_for(lambda: all(not psutil.pid_exists(pid) for pid in pids))


@pytest.mark.linux_only
@pytest.mark.parametrize("options,status", [
    ({"installed": False}, 405),  # Stock SPA fallback owns unknown HTTP paths.
    ({"enabled": False}, 404),
    ({"bound_host": "0.0.0.0"}, 401),
    ({"settings": {"max_sessions": 0}}, 503),
])
def test_unavailable_deployments_fail_closed(tmp_path, backend, options, status):
    with server(tmp_path, backend, **options) as (client, ws_url, hermes, config, log):
        assert client.post(f"{API}/sessions", json={}).status_code == status
        with pytest.raises(InvalidStatus):
            dial(client, ws_url, "not-a-ticket")


@pytest.mark.linux_only
def test_verified_principals_cannot_take_over_each_others_shells(tmp_path, backend):
    with server(tmp_path, backend, gated=True) as (client, ws_url, hermes, config, log):
        sid = create(client, "a")
        value = ticket(client, sid, "a")
        alice = client.headers["Authorization"]
        client.headers["Authorization"] = f"Bearer {user_token('bob')}"
        assert client.post(f"{API}/sessions/{sid}/ticket?profile=a").status_code == 404
        assert client.delete(f"{API}/sessions/{sid}?profile=a").status_code == 404
        assert client.post(f"{API}/sessions/{sid}/heartbeat?profile=a").status_code == 404
        other = create(client, "a")
        # WS uses only the one-time capability, never a user-supplied principal.
        with dial(client, ws_url, value) as ws:
            pid = shell_pid(ws)
        client.headers["Authorization"] = alice
        assert client.post(f"{API}/sessions/{other}/ticket?profile=a").status_code == 404
        assert client.post(f"{API}/sessions", json={}, headers={"Origin": ""}).status_code == 403
        assert client.delete(f"{API}/sessions/{sid}?profile=a").status_code == 200
        wait_for(lambda: not psutil.pid_exists(pid))


@pytest.mark.linux_only
@pytest.mark.parametrize("cleanup", ["delete", "shutdown"])
def test_cleanup_kills_stubborn_foreground_job(tmp_path, backend, cleanup):
    pids = []
    handles = []
    try:
        with server(tmp_path, backend) as (client, ws_url, *_):
            sid = create(client)
            with dial(client, ws_url, ticket(client, sid)) as ws:
                pids.append(shell_pid(ws))
                handles.append(owned_child(pids[-1]))
                program = (
                    "import os,signal,time; "
                    "signal.signal(signal.SIGHUP, signal.SIG_IGN); "
                    "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
                    "print('JOB=' + str(os.getpid()), flush=True); time.sleep(300)"
                )
                ws.send(f"{shlex.quote(sys.executable)} -c {shlex.quote(program)}\r".encode())
                output = until(ws, b"\r\nJOB=")
                while not output.split(b"\r\nJOB=")[-1].split(b"\r\n")[0].isdigit():
                    output += ws.recv(timeout=15)
                child = int(output.split(b"\r\nJOB=")[-1].split(b"\r\n")[0])
                pids.append(child)
                handles.append(owned_child(child))
                assert os.getsid(child) == pids[0]
                assert os.getpgid(child) != os.getpgid(pids[0])
                if cleanup == "delete":
                    assert client.delete(f"{API}/sessions/{sid}").status_code == 200
                    wait_for(lambda: all(not psutil.pid_exists(pid) for pid in pids))
        wait_for(lambda: all(not psutil.pid_exists(pid) for pid in pids))
    finally:
        # A red run must not leave the deliberately HUP/TERM-immune job running.
        for handle in reversed(handles):
            kill_child(handle)


@pytest.fixture
def plugin_api(monkeypatch, backend):
    spec = importlib.util.spec_from_file_location("browser_terminal_lease_test", PLUGIN / "dashboard/plugin_api.py")
    api = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, api)
    spec.loader.exec_module(api)
    return api


def test_http_lease_renewal_expiry_and_detached_retention(tmp_path, monkeypatch, plugin_api):
    api = plugin_api
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(api, "time", SimpleNamespace(monotonic=lambda: clock.now, time=lambda: clock.now))
    monkeypatch.setattr(api, "installation_config", lambda: {})

    async def scenario():
        baseline = asyncio.all_tasks()
        manager = api.Terminals(api.Limits())
        session = SimpleNamespace(alive=True, attached=True, last_detached_at=None, close=AsyncMock())
        ws = SimpleNamespace(close=AsyncMock())
        def detach(socket):
            assert socket is ws
            session.attached = False
            session.last_detached_at = clock.now
        session.detach = detach
        shell = api.Shell(session, ("session", "test", "org", "alice"), tmp_path)
        manager.shells["sid"] = shell
        await manager.renew("sid", shell, clock.now + 30)
        shell.ws = ws
        clock.now += 20
        # A new verified HTTP expiry replaces the original handshake identity's
        # expiry; the WS must not be timed out by its original ticket.
        await manager.renew("sid", shell, clock.now + 120)
        clock.now += 20
        await manager.reap_once()
        ws.close.assert_not_awaited()
        clock.now = shell.lease_until
        manager.tickets["outstanding"] = api.Ticket("sid", shell.owner, tmp_path, "origin", clock.now + 30, clock.now + 120)
        await manager.reap_once()
        ws.close.assert_awaited_once_with(code=4401)
        assert not manager.tickets and not session.attached
        assert manager.shells["sid"] is shell
        session.close.assert_not_awaited()
        # A fresh HTTP principal can recover the same shell. Its actual verified
        # expiry, even if shorter than the lease, still bounds access.
        await manager.renew("sid", shell, clock.now + 10)
        shell.ws = ws
        session.attached = True
        session.last_detached_at = None
        clock.now += 10
        await manager.reap_once()
        assert ws.close.await_count == 2
        clock.now += manager.limits.detached_ttl + 1
        await manager.reap_once()
        session.close.assert_awaited_once()
        assert not manager.shells
        assert not (asyncio.all_tasks() - baseline)

    asyncio.run(scenario())


@pytest.mark.linux_only
@pytest.mark.parametrize("cleanup_phase", ["socket", "process"])
@pytest.mark.parametrize("expiry_bound", ["lease", "auth"])
def test_lease_stops_pty_output_while_other_cleanup_holds_lock(
    tmp_path, monkeypatch, plugin_api, cleanup_phase, expiry_bound,
):
    from hermes_cli.pty_session import PtySession

    api = plugin_api
    clock = SimpleNamespace(now=1000.0)
    monkeypatch.setattr(api, "time", SimpleNamespace(monotonic=lambda: clock.now, time=lambda: clock.now))
    monkeypatch.setattr(api, "AUTH_LEASE_SECONDS", 0.05 if expiry_bound == "lease" else 60)
    monkeypatch.setattr(api, "installation_config", lambda: {})

    async def scenario():
        baseline = asyncio.all_tasks()
        manager = api.Terminals(api.Limits())
        cleanup_started, release_cleanup = asyncio.Event(), asyncio.Event()
        expired, release_close = asyncio.Event(), asyncio.Event()
        output = bytearray()

        async def blocked_cleanup(*args, **kwargs):
            cleanup_started.set()
            await release_cleanup.wait()

        async def expired_close(*, code):
            assert code == 4401
            expired.set()
            await release_close.wait()

        other_session = SimpleNamespace(close=AsyncMock(side_effect=blocked_cleanup), detach=lambda ws: None)
        other_ws = SimpleNamespace(close=AsyncMock())
        if cleanup_phase == "socket":
            other_ws.close.side_effect = blocked_cleanup
        manager.shells["other"] = api.Shell(other_session, (), tmp_path, ws=other_ws)
        bridge = api.spawn_shell(["/bin/sh", "-i"], cwd=str(tmp_path), env={"PATH": os.defpath})
        session = PtySession("live", bridge, buffer_cap=4096, read_timeout=0.01)
        shell = api.Shell(session, (), tmp_path)
        manager.shells["live"] = shell
        ws = SimpleNamespace(close=AsyncMock(side_effect=expired_close),
                             send_bytes=AsyncMock(side_effect=output.extend))

        async def wait_buffer(marker):
            async with asyncio.timeout(5):
                while marker not in session.buffer.snapshot():
                    await asyncio.sleep(0.01)

        async def delete_other():
            async with manager.lock:
                await manager.remove("other")

        deletion = reaper = None
        try:
            await session.start()
            await manager.renew("live", shell, clock.now + 0.01)
            await asyncio.sleep(0)  # Start the old timer before a fresh HTTP renewal.
            await manager.renew("live", shell, clock.now + (0.05 if expiry_bound == "auth" else 120))
            shell.ws = ws
            await session.attach(ws)
            await bridge.write(b"printf 'BEFORE\\n'\n")
            await wait_buffer(b"BEFORE\r\n")
            assert b"BEFORE\r\n" in output
            deletion = asyncio.create_task(delete_other())
            await asyncio.wait_for(cleanup_started.wait(), 5)
            reaper = asyncio.create_task(manager.reap_once())
            clock.now += 0.05
            await asyncio.wait_for(expired.wait(), 5)
            assert manager.lock.locked() and not deletion.done()
            assert not session.attached and shell.ws is None
            before = bytes(output)
            await bridge.write(b"printf 'AFTER\\n'\n")
            await wait_buffer(b"AFTER\r\n")
            assert bytes(output) == before  # PTY still drains, but cannot reach expired WS.
            release_cleanup.set()
            release_close.set()
            await deletion
            await reaper
            # Fresh HTTP auth recovers the retained PTY; removal below must also
            # cancel this still-pending expiry task, not just completed timers.
            async with manager.lock:
                await manager.renew("live", shell, clock.now + 120)
                replacement = SimpleNamespace(close=AsyncMock(), send_bytes=AsyncMock())
                shell.ws = replacement
                await session.attach(replacement)
            assert b"AFTER\r\n" in replacement.send_bytes.await_args.args[0]
        finally:
            release_cleanup.set()
            release_close.set()
            if deletion is not None:
                await deletion
            if reaper is not None:
                await reaper
            for sid in list(manager.shells):
                await manager.remove(sid)
        assert not (asyncio.all_tasks() - baseline)

    asyncio.run(scenario())


@pytest.mark.linux_only
@pytest.mark.parametrize("error", [PermissionError, RuntimeError])
def test_job_cleanup_skips_only_inaccessible_proc_stats(tmp_path, monkeypatch, plugin_api, error):
    api = plugin_api
    bridge = api.spawn_shell(["/bin/sh", "-i"], cwd=str(tmp_path), env={"PATH": os.defpath})
    child_process = None
    try:
        program = (
            "import os,signal,time; signal.signal(signal.SIGHUP, signal.SIG_IGN); "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "print('JOB=' + str(os.getpid()), flush=True); time.sleep(300)"
        )
        asyncio.run(bridge.write(f"{shlex.quote(sys.executable)} -c {shlex.quote(program)}\n".encode()))
        output = bytearray()
        def job_ready():
            output.extend(bridge.read(0.1) or b"")
            return bytes(output).split(b"\r\nJOB=")[-1].split(b"\r\n")[0].isdigit()
        wait_for(job_ready)
        child = int(bytes(output).split(b"\r\nJOB=")[-1].split(b"\r\n")[0])
        child_process = owned_child(child)
        assert os.getpgid(child) != os.getpgid(bridge.pid)
        inaccessible = Path("/proc/0/stat")
        original_glob, original_read = Path.glob, Path.read_text
        def glob(path, pattern):
            if path == Path("/proc"):
                yield inaccessible
            yield from original_glob(path, pattern)
        def read(path, *args, **kwargs):
            if path == inaccessible:
                raise error("unreadable stat")
            return original_read(path, *args, **kwargs)
        with monkeypatch.context() as patch:
            patch.setattr(Path, "glob", glob)
            patch.setattr(Path, "read_text", read)
            if error is PermissionError:
                bridge.close()
                wait_for(lambda: not psutil.pid_exists(child))
            else:
                with pytest.raises(error, match="unreadable stat"):
                    bridge.close()
    finally:
        if child_process is not None:
            kill_child(child_process)
        bridge.close()


@pytest.mark.linux_only
def test_cookie_logout_blocks_heartbeat_and_expires_attached_lease(tmp_path, backend):
    with server(tmp_path, backend, gated=True) as (client, ws_url, *_):
        del client.headers["Authorization"]
        def login():
            response = client.post("/auth/password-login", json={
                "provider": "basic", "username": "alice", "password": "pw",
            })
            assert response.status_code == 200, response.text
        login()
        sid = create(client, "a")
        with dial(client, ws_url, ticket(client, sid, "a")) as ws:
            pid = shell_pid(ws)
            endpoint = f"{API}/sessions/{sid}/heartbeat?profile=a"
            assert client.post(endpoint, headers={"Origin": "https://evil.invalid"}).status_code == 403
            assert client.post(f"{API}/sessions/{sid}/heartbeat?profile=b").status_code == 404
            renewed = client.post(endpoint)
            assert renewed.status_code == 200
            assert renewed.headers["cache-control"] == "no-store"
            assert client.post("/auth/logout").status_code == 302
            assert client.post(endpoint).status_code == 401
            assert client.post(f"{API}/sessions/{sid}/ticket?profile=a").status_code == 401
            # Stock cookies are stateless: logout does not instantly revoke the
            # WS. Without an authenticated HTTP renewal it expires in 60 seconds.
            with pytest.raises(ConnectionClosed) as closed:
                while True:
                    ws.recv(timeout=65)
            assert closed.value.rcvd.code == 4401
        assert psutil.pid_exists(pid)
        login()
        with dial(client, ws_url, ticket(client, sid, "a")) as ws:
            assert shell_pid(ws) == pid
        assert client.delete(f"{API}/sessions/{sid}?profile=a").status_code == 200
        wait_for(lambda: not psutil.pid_exists(pid))


@pytest.mark.linux_only
def test_harness_cleanup_tracks_only_owned_processes():
    with pytest.raises(AssertionError):
        owned_child(os.getpid())
    with pytest.raises(AssertionError):
        owned_child(os.getppid())
    process = subprocess.Popen([sys.executable, "-B", "-c", "import time; time.sleep(60)"])
    try:
        child = owned_child(process.pid)
        kill_child(child)
        assert process.wait(timeout=5) == -signal.SIGKILL
        kill_child(child)  # Already reaped; no signal to a reused PID.
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
