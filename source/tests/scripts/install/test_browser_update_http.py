"""Authenticated HTTP offer/consent flow with real downloaded archive validation."""

from contextlib import closing
import importlib
import json
import socket
import time

import pytest

from tests.scripts.install.test_browser_update_jobs import (
    candidate,
    jobs,
    maintained,
    release,
    scene,  # noqa: F401
    settled,
    offered,
    snapshot,
    wait_for,
)
from tests.scripts.install.test_browser_updater import (
    IDENTITY,
    ORIGIN,
    TOKEN,
    backend,
    updater,  # noqa: F401
    request,
    serving,
)

pytestmark = [pytest.mark.linux_only, pytest.mark.live_system_guard_bypass]


def test_http_requires_exact_consent_and_continues_after_lost_response(
    jobs,
    scene,
    updater,
    backend,
    tmp_path,
    monkeypatch,
):
    s = scene
    s.publish(candidate(s.r, tmp_path, "http-target"))
    config = json.loads(s.config.config_path.read_text())
    config.update(
        public_origin=ORIGIN,
        backend_port=backend["port"],
        listen_port=0,
        administrators=[IDENTITY, {**IDENTITY, "user_id": "second-admin"}],
    )
    s.config.config_path.write_text(json.dumps(config))
    monkeypatch.setattr(jobs, "Dashboard", lambda _: s.dashboard)
    with serving(
        updater.UpdaterServer(updater.load_config(s.config.config_path))
    ) as server:
        status, _, _ = request(server, path="/browser-updater/api/offer")
        assert status == 202

        def state():
            code, _, raw = request(server)
            assert code == 200
            return json.loads(raw)

        def reached(phase):
            value = state()
            return value if value["phase"] == phase else None

        prepared = wait_for(lambda: reached("offered"))
        offer = prepared["offer"]
        assert offer["current_release"] != offer["target_release"]
        assert "disconnects all browser users" in offer["warning"]
        assert offer["compatibility"] == "not-exercised"
        body = {"offer_id": offer["id"], "confirm_restart_and_rollback": True}
        for invalid in (
            {"offer_id": offer["id"]},
            {**body, "confirm_restart_and_rollback": False},
            {**body, "source": "https://attacker.invalid/CURRENT.json"},
        ):
            assert (
                request(
                    server,
                    path="/browser-updater/api/apply",
                    body=json.dumps(invalid).encode(),
                )[0]
                == 400
            )
        backend["body"] = json.dumps({
            **IDENTITY,
            "user_id": "second-admin",
            "expires_at": time.time() + 60,
        }).encode()
        assert (
            request(
                server,
                path="/browser-updater/api/apply",
                body=json.dumps(body).encode(),
            )[0]
            == 409
        )
        assert not s.dashboard.actions
        backend["body"] = None
        # Send valid consent, then lose the response. The job must outlive this socket.
        raw = json.dumps(body).encode()
        with socket.create_connection(
            ("127.0.0.1", server.server_port), timeout=5
        ) as client:
            client.sendall(
                b"POST /browser-updater/api/apply HTTP/1.1\r\nHost: updater.localhost\r\n"
                + f"Origin: {ORIGIN}\r\nCookie: __Host-hermes_session_at={TOKEN}\r\n".encode()
                + f"Content-Type: application/json\r\nContent-Length: {len(raw)}\r\n\r\n".encode()
                + raw
            )
            client.shutdown(socket.SHUT_WR)
        done = wait_for(lambda: reached("succeeded"))
        code, _, raw = request(
            server, path="/browser-updater/api/apply", body=json.dumps(body).encode()
        )
        assert code == 202 and json.loads(raw)["job"]["id"] == done["job"]["id"]
        assert [action for action, _ in s.dashboard.actions] == ["stop", "start"]


def test_expiry_artifact_tampering_and_worker_lock_do_not_authorize_stop(
    jobs, scene, tmp_path, monkeypatch
):
    s = scene
    s.publish(candidate(s.r, tmp_path, "candidate"))
    before = snapshot(s.r["dest"])
    with closing(jobs.Updates(s.config, dashboard=s.dashboard)) as updates:
        with pytest.raises(BlockingIOError):
            jobs.Updates(s.config, dashboard=s.dashboard)
        monkeypatch.setattr(jobs, "OFFER_TTL", -1)
        identifier = offered(updates)
        from tests.scripts.install.test_browser_update_jobs import ACTOR

        with pytest.raises(ValueError):
            updates.apply(ACTOR, identifier, True)
        monkeypatch.setattr(jobs, "OFFER_TTL", 600)
        identifier = offered(updates)
        (s.state / "candidate/launcher").write_bytes(b"unverified replacement")
        updates.apply(ACTOR, identifier, True)
        settled(updates, "failed")
        assert not s.dashboard.actions and snapshot(s.r["dest"]) == before
        assert importlib.import_module("browser_setup").E.installed(s.r["dest"])
