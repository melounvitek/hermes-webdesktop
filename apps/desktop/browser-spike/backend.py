#!/usr/bin/env python3
"""Real dashboard + loopback model fixture. No agent/transport monkeypatches.

Run with --web-dist <built SPA directory>.
Every launch gets a fresh home. runtime.json records URLs, PIDs and log paths.
With --restartable, SIGUSR1 (or a killed child) restarts on the same port/home
with a new token. 'spike: hold' streams once, then waits for hold-stream removal.
Prompts: 'spike: clarify' calls the real clarify tool; 'spike: approval' asks
permission to remove a disposable directory inside the isolated home. Anything
else streams an echo numbered by user turns in the actual model request history.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import secrets
import shlex
import signal
import subprocess
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit
from urllib.request import ProxyHandler, build_opener

ROOT = Path(__file__).resolve().parents[3]
MODEL = "browser-spike-local"


class ModelFixture(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        print("model:", fmt % args, flush=True)

    def do_CONNECT(self):
        self.send_error(403, "External network disabled by fixture proxy")

    def do_GET(self):
        if self.path != "/v1/models":
            self.send_error(404)
            return
        self.send_json({"object": "list", "data": [
            {"id": MODEL, "object": "model", "owned_by": "local-fixture"}]})

    def send_json(self, value):
        data = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        if self.path != "/v1/chat/completions":
            self.send_error(404)
            return
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        messages = request["messages"]
        users = [m for m in messages if m["role"] == "user"]
        prompt = users[-1]["content"] if users else ""
        if isinstance(prompt, list):
            prompt = "\n".join(p.get("text", "") for p in prompt)
        turn = len(users)
        last_user = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=-1)
        results = [m for m in messages[last_user + 1:] if m["role"] == "tool"]
        tool_call = None
        action = next((name for name in ("clarify", "approval") if f"spike: {name}" in prompt.lower()), None)
        if action and not results:
            name = "clarify" if action == "clarify" else "terminal"
            names = {t["function"]["name"] for t in request.get("tools", [])}
            if name not in names:
                self.send_error(400, f"Real {name} tool missing from agent schema")
                return
            if action == "clarify":
                arguments = {"questions": [{
                    "question": "Which browser-spike color?", "choices": ["Blue", "Green"]}]}
            else:
                target = self.server.run_dir / "home" / "approval-target"
                target.mkdir(exist_ok=True)
                arguments = {"command": "rm -rf " + shlex.quote(str(target))}
            tool_call = {"id": f"call_spike_{turn}", "type": "function", "function": {
                "name": name, "arguments": json.dumps(arguments)}}
        text = (f"Spike turn {turn}: {action} result: {results[-1]['content']}"
                if results else f"Spike turn {turn}: {prompt}")
        with (self.server.run_dir / "model-requests.jsonl").open("a") as log:
            log.write(json.dumps(request) + "\n")
        base = {"id": f"chatcmpl-spike-{turn}", "created": 1, "model": MODEL}
        finish = "tool_calls" if tool_call else "stop"
        if not request.get("stream"):
            message = {"role": "assistant", "content": None if tool_call else text}
            if tool_call:
                message["tool_calls"] = [tool_call]
            self.send_json({**base, "object": "chat.completion", "choices": [
                {"index": 0, "message": message, "finish_reason": finish}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120}})
            return
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()

        def chunk(delta, reason=None):
            payload = {**base, "object": "chat.completion.chunk", "choices": [
                {"index": 0, "delta": delta, "finish_reason": reason}]}
            self.wfile.write(("data: " + json.dumps(payload) + "\n\n").encode())
            self.wfile.flush()

        try:
            chunk({"role": "assistant"})
            if tool_call:
                chunk({"tool_calls": [{"index": 0, **tool_call}]})
            else:
                for i in range(0, len(text), 8):
                    chunk({"content": text[i:i + 8]})
                    if i == 0 and "spike: hold" in prompt.lower():
                        deadline = time.monotonic() + 90
                        while (self.server.run_dir / "hold-stream").exists():
                            if time.monotonic() >= deadline:
                                raise TimeoutError("Harness did not release hold-stream")
                            time.sleep(0.05)
                    time.sleep(0.08)
            chunk({}, finish)
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # Interrupting a real streaming turn closes the model connection.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--web-dist", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path.home() / ".hermes/hermes-agent/venv/bin/python")
    parser.add_argument("--port", type=int, default=0)
    parser.add_argument("--restartable", action="store_true", help="Restart child on SIGUSR1 or child exit")
    parser.add_argument("--public-url", help="HTTPS origin for a tailnet proxy; generates a password login")
    args = parser.parse_args()
    if args.public_url:
        origin = urlsplit(args.public_url)
        if (origin.scheme != "https" or not origin.hostname or origin.username or origin.password
                or origin.path not in ("", "/") or origin.query or origin.fragment):
            parser.error("--public-url must be an HTTPS origin without a path or credentials")
        args.public_url = args.public_url.rstrip("/")
    # Keep the venv path: resolving its python symlink would select the bare interpreter.
    args.python = args.python.expanduser().absolute()
    web_dist = args.web_dist.expanduser().resolve()
    if not (web_dist / "index.html").is_file():
        parser.error("--web-dist must contain index.html (build is owned by the browser spike)")
    # The CLI loads project dotenv even with HERMES_HOME set. Refuse rather than
    # silently inheriting checkout credentials or editing the user's checkout.
    if (ROOT / ".env").exists():
        parser.error("checkout .env exists; use a credential-free worktree")
    run_dir = Path(tempfile.mkdtemp(prefix="hermes-browser-spike-"))
    home = run_dir / "home"
    hermes_home = home / ".hermes"
    hermes_home.mkdir(parents=True)
    fixture = ThreadingHTTPServer(("127.0.0.1", 0), ModelFixture)
    fixture.run_dir = run_dir
    threading.Thread(target=fixture.serve_forever, daemon=True).start()
    model_url = f"http://127.0.0.1:{fixture.server_port}"
    config = {
        "model": {"default": MODEL, "provider": "custom", "base_url": model_url + "/v1",
                  "context_length": 131072, "api_mode": "chat_completions"},
        "agent": {"max_turns": 6},
        "platform_toolsets": {"cli": ["clarify", "terminal"]},
        "approvals": {"mode": "manual"},
        "security": {"tirith_enabled": False},  # No optional scanner auto-download; real pattern approval stays on.
        "model_catalog": {"enabled": False},
        "terminal": {"backend": "local", "cwd": str(home)},
        "auxiliary": {"title_generation": {"enabled": False, "model_upgrade_enabled": False},
                      "background_review": {"enabled": False}},
        "curator": {"enabled": False},
    }
    # JSON is valid YAML and keeps the launcher stdlib-only.
    (hermes_home / "config.yaml").write_text(json.dumps(config, indent=2))
    token = secrets.token_urlsafe(32)
    env = {
        "HOME": str(home), "HERMES_HOME": str(hermes_home),
        "PATH": f"{args.python.parent}:/usr/bin:/bin", "LANG": "C.UTF-8", "TZ": "UTC",
        "PYTHONPATH": str(ROOT), "PYTHONNOUSERSITE": "1", "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1", "HERMES_WEB_DIST": str(web_dist),
        "HERMES_DASHBOARD_SESSION_TOKEN": token,
        "HTTP_PROXY": model_url, "HTTPS_PROXY": model_url, "ALL_PROXY": model_url,
        "NO_PROXY": "127.0.0.1,localhost,::1", "HF_HUB_OFFLINE": "1",
    }
    if args.public_url:
        login = {"username": "spike", "password": secrets.token_urlsafe(18)}
        login_file = run_dir / "login.json"
        with open(login_file, "x", opener=lambda path, flags: os.open(path, flags, 0o600)) as out:
            json.dump(login, out)
        hash_code = "import sys; from plugins.dashboard_auth.basic import hash_password; print(hash_password(sys.stdin.read()))"
        password_hash = subprocess.check_output(
            [str(args.python), "-c", hash_code], input=login["password"], cwd=home, env=env, text=True).strip()
        # Basic auth otherwise generates a per-process key, intentionally signing
        # everyone out on restart. Keep this disposable home's login valid while
        # the separate loopback/bootstrap token still rotates with each child.
        config["dashboard"] = {"public_url": args.public_url, "basic_auth": {
            "username": login["username"], "password_hash": password_hash,
            "secret": secrets.token_hex(32)}}
        (hermes_home / "config.yaml").write_text(json.dumps(config, indent=2))
    # Verify the installed venv resolves application modules from THIS worktree.
    probe = "import importlib.util,json; print(json.dumps({n:importlib.util.find_spec(n).origin for n in ['hermes_cli.main','run_agent','tui_gateway.server']}))"
    origins = json.loads(subprocess.check_output([str(args.python), "-c", probe], cwd=home, env=env, text=True))
    assert all(Path(p).is_relative_to(ROOT) for p in origins.values()), origins
    command = [str(args.python), "-m", "hermes_cli.main", "dashboard", "--no-open",
               "--host", "127.0.0.1", "--port", str(args.port), "--skip-build"]
    log_path = run_dir / "backend.log"
    with log_path.open("w") as log:
        child = subprocess.Popen(command, cwd=home, env=env, stdout=log, stderr=subprocess.STDOUT)
    runtime = {"run_dir": str(run_dir), "home": str(home), "hermes_home": str(hermes_home),
               "harness_pid": os.getpid(), "backend_pid": child.pid, "backend_log": str(log_path),
               "model_url": model_url + "/v1", "command": command, "imports": origins,
               "public_url": args.public_url}
    print(json.dumps(runtime, indent=2), flush=True)

    def stop(_signum, _frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    restart = threading.Event()
    if args.restartable:
        signal.signal(signal.SIGUSR1, lambda _signum, _frame: restart.set())
    try:
        # The real backend's sentinel contains the kernel-selected port for --port 0.
        import re
        deadline = time.monotonic() + 90
        generation = 0
        log_offset = 0
        http = build_opener(ProxyHandler({}))
        while True:
            if child.poll() is not None:
                raise RuntimeError(f"Backend exited {child.returncode}: {log_path.read_text()}")
            match = re.search(r"HERMES_DASHBOARD_READY port=(\d+)", log_path.read_text()[log_offset:])
            if match:
                url = f"http://127.0.0.1:{match.group(1)}"
                try:
                    with http.open(url + ("/api/status" if args.public_url else ""), timeout=2) as response:
                        assert response.status == 200
                        if args.public_url:
                            status = json.load(response)
                            assert status["auth_required"] and "basic" in status["auth_providers"]
                        else:
                            assert token in response.read().decode(), "SPA must receive the real backend session token"
                except OSError:
                    pass
                else:
                    runtime.update(url=url, token=token, generation=generation,
                                   restartable=args.restartable, backend_pid=child.pid,
                                   ws_url=url.replace("http:", "ws:") + "/api/ws?token=" + token)
                    pending = run_dir / "runtime.pending.json"
                    pending.write_text(json.dumps(runtime, indent=2))
                    pending.replace(run_dir / "runtime.json")
                    print("READY " + json.dumps({"url": url, "run_dir": str(run_dir),
                                                "generation": generation}), flush=True)
                    if not args.restartable:
                        child.wait()
                        break
                    while child.poll() is None and not restart.wait(0.1):
                        pass
                    restart.clear()
                    # Abrupt loss is intentional: regression cases must not rely
                    # on a graceful shutdown persisting an interrupted turn.
                    child.kill()
                    child.wait()
                    command[command.index("--port") + 1] = match.group(1)
                    token = secrets.token_urlsafe(32)
                    env["HERMES_DASHBOARD_SESSION_TOKEN"] = token
                    log_offset = len(log_path.read_text())
                    with log_path.open("a") as log:
                        child = subprocess.Popen(command, cwd=home, env=env, stdout=log,
                                                 stderr=subprocess.STDOUT)
                    generation += 1
                    deadline = time.monotonic() + 90
            if time.monotonic() > deadline:
                raise TimeoutError(f"Backend not ready; inspect {log_path}")
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        child.terminate()
        try:
            child.wait(timeout=15)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait()
        fixture.shutdown()
        fixture.server_close()


if __name__ == "__main__":
    main()
