"""One durable, explicitly confirmed browser-only maintenance job.

A lost HTTP connection does not cancel work. After worker-process loss, retain the
journal and require operator recovery rather than replaying service actions.
"""

from argparse import Namespace
import fcntl
import os
from pathlib import Path
import secrets
import stat
import threading
import time

from browser_download import current
from browser_setup import E, preflight
from browser_update_service import Dashboard


OFFER_TTL = 600
ACTIVE = frozenset({
    "stopping",
    "switching",
    "starting",
    "rollback_stopping",
    "rollback_switching",
    "rollback_starting",
})
PHASES = ACTIVE | {
    "idle",
    "preparing",
    "offered",
    "current",
    "failed",
    "succeeded",
    "rolled_back",
    "recovery_required",
}
WARNING = (
    "Restarting disconnects all browser users and may interrupt dashboard chats "
    "and close terminals. Finish your work first. If activation fails, restore "
    "and restart the previous browser release. Hermes itself is not upgraded. "
    "Signing in again may be required."
)


def configuration(config):
    actual = E.digest(E.read_regular(config.config_path, 16384))
    E.require(
        actual == config.configuration_sha256,
        "Configuration changed; reload the updater before checking releases",
    )
    return actual


def read_state(directory):
    path = directory / "state.json"
    E.no_links(directory)
    meta = directory.stat()
    E.require(
        meta.st_uid == os.getuid() and stat.S_IMODE(meta.st_mode) == 0o700,
        "Unsafe updater state directory",
    )
    value = E.load_json(E.read_regular(path, E.MAX_JSON))
    E.require(
        isinstance(value, dict) and value.get("phase") in PHASES,
        "Invalid updater journal",
    )
    return value


def admit_start(config, receipt):
    """Called by the trusted controller under its existing installation lock."""
    state = read_state(Path(config.maintenance["state_dir"]))
    phase = state["phase"]
    if phase in ("starting", "rollback_starting"):
        E.require(
            state["configuration"] == configuration(config),
            "Updater configuration changed",
        )
        expected = state["target"] if phase == "starting" else state["current"]
        E.require(receipt == expected, "Dashboard start differs from confirmed release")
    else:
        E.require(
            phase not in ACTIVE | {"recovery_required"},
            "Updater recovery/maintenance blocks dashboard start",
        )


class Updates:
    def __init__(self, config, dashboard=None):
        self.config = config
        configuration(config)
        self.root = E.absolute_path(config.maintenance["installation"])
        self.directory = E.absolute_path(config.maintenance["state_dir"])
        installed, _ = E.installed(self.root)
        E.safe_destination(self.root, installed["selection"])
        protected = [
            self.root,
            self.root.with_name(self.root.name + ".history"),
            self.root.with_name(self.root.name + ".run"),
            Path(__file__).parent,
            Path(installed["selection"]["backend_root"]),
            Path(installed["selection"]["hermes_root"]),
            Path(installed["selection"]["python"]).parent.parent,
        ]
        E.require(
            all(
                not (
                    self.directory.is_relative_to(path)
                    or path.is_relative_to(self.directory)
                )
                for path in protected
            ),
            "Updater state overlaps protected files",
        )
        E.no_links(self.directory)
        self.directory.mkdir(mode=0o700, parents=False, exist_ok=True)
        meta = self.directory.stat()
        E.require(
            meta.st_uid == os.getuid() and stat.S_IMODE(meta.st_mode) == 0o700,
            "Updater state must be an owned mode-0700 directory",
        )
        E.sync_directory(self.directory.parent)
        had_lock = os.path.lexists(self.directory / "worker.lock")
        fd = os.open(
            self.directory / "worker.lock",
            os.O_RDWR | os.O_CREAT | os.O_NOFOLLOW,
            0o600,
        )
        self.lock = os.fdopen(fd, "r+b")
        try:
            meta = os.fstat(fd)
            E.require(
                stat.S_ISREG(meta.st_mode)
                and meta.st_uid == os.getuid()
                and meta.st_nlink == 1
                and stat.S_IMODE(meta.st_mode) == 0o600,
                "Unsafe updater worker lock",
            )
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.guard = threading.RLock()
            self.closed = False
            self.recording_failed = False
            self.worker = None
            self.dashboard = dashboard if dashboard is not None else Dashboard(config)
            if not had_lock and not os.path.lexists(self.directory / "state.json"):
                self._save({"phase": "idle"})
            state = read_state(self.directory)
            if state["phase"] in ACTIVE:
                self._save({
                    **state,
                    "phase": "recovery_required",
                    "error": "Updater interrupted during maintenance; inspect the retained journal before manual recovery.",
                })
            elif state["phase"] == "preparing":
                self._save({
                    **state,
                    "phase": "failed",
                    "error": "Release preparation was interrupted; retry the check.",
                })
        except BaseException:
            self.lock.close()
            raise

    def close(self):
        with self.guard:
            self.closed = True
            worker = self.worker
        if worker is not None:
            worker.join()
        self.lock.close()

    def _save(self, state):
        with self.guard:
            try:
                # Only our private journal temp is replaceable; the lock inode survives.
                path = self.directory / "state.next"
                E.no_links(path)
                with path.open("wb") as stream:
                    os.fchmod(stream.fileno(), 0o600)
                    stream.write(E.json_bytes(state))
                    stream.flush()
                    os.fsync(stream.fileno())
                os.replace(path, self.directory / "state.json")
                E.sync_directory(self.directory)
            except OSError:
                self.recording_failed = True
                raise

    def _phase(self, phase, error=None):
        with self.guard:
            state = read_state(self.directory)
            state["phase"] = phase
            state["error"] = error
            self._save(state)

    def status(self):
        with self.guard:
            state = read_state(self.directory)
            if self.recording_failed:
                state = {
                    **state,
                    "phase": "recovery_required",
                    "error": "Update state could not be recorded. No further service actions will be taken; inspect the retained journal.",
                }
            value = {"capabilities": ["install"], "phase": state["phase"]}
            if "target" in state:
                value["offer"] = {
                    "id": state["id"],
                    "current_release": state["current_release"],
                    "target_release": state["target_release"],
                    "expires_at": state["expires_at"],
                    "warning": WARNING,
                    "tested_backend": state["tested_backend"],
                    "compatibility": "not-exercised",
                }
            if state.get("confirmed"):
                value["job"] = {"id": state["id"], "phase": state["phase"]}
            if state.get("error"):
                value["error"] = state["error"]
            return value

    def _start(self, function):
        self.worker = threading.Thread(
            target=function, name="browser-update", daemon=False
        )
        self.worker.start()

    def offer(self, actor):
        with self.guard:
            E.require(not self.closed, "Updater is closing")
            E.require(not self.recording_failed, "Updater journal requires recovery")
            fingerprint = configuration(self.config)
            state = read_state(self.directory)
            E.require(
                state["phase"] not in ACTIVE | {"recovery_required"},
                "Finish or recover the current update first",
            )
            if (
                state["phase"] in ("preparing", "offered")
                and state.get("expires_at", time.time() + 1) > time.time()
            ):
                E.require(
                    state["actor"] == list(actor),
                    "Another administrator owns this offer",
                )
                return self.status()
            E.require(
                self.worker is None or not self.worker.is_alive(),
                "Updater worker is busy",
            )
            self._save({
                "phase": "preparing",
                "id": secrets.token_hex(16),
                "actor": list(actor),
                "configuration": fingerprint,
            })
            self._start(self._prepare)
            return self.status()

    def _prepare(self):
        try:
            state = read_state(self.directory)
            receipt, manifest = E.installed(self.root)
            service = self.dashboard.status()
            E.require(
                service["state"] == "ready" and service["receipt"] == receipt,
                "Dashboard is not running the verified installation",
            )
            folder = self.directory / "candidate"
            E.no_links(folder)
            folder.mkdir(mode=0o700, exist_ok=True)
            E.require(
                set(p.name for p in folder.iterdir()) <= {"archive", "launcher"},
                "Unknown candidate files",
            )
            for path in folder.iterdir():
                E.read_regular(path)
                path.unlink()
            descriptor, target_manifest = current(
                self.config.maintenance["source"], folder
            )
            launcher = E.read_regular(folder / "launcher")
            _, payload = E.archive_payload(
                str(folder / "archive"), descriptor["archive"]["sha256"], launcher
            )
            target, _ = E.installation_files(
                descriptor["archive"]["sha256"], receipt["selection"], payload, launcher
            )
            preflight(receipt["selection"], target_manifest)
            for path in folder.iterdir():
                path.chmod(0o600)
                with path.open("rb") as stream:
                    os.fsync(stream.fileno())
            E.sync_directory(folder)
            E.sync_directory(self.directory)
            E.require(
                E.installed(self.root)[0] == receipt
                and configuration(self.config) == state["configuration"],
                "Installation or configuration changed while checking",
            )
            self._save({
                **state,
                "phase": "current" if target == receipt else "offered",
                "current": receipt,
                "target": target,
                "descriptor": descriptor,
                "current_release": manifest["release"],
                "target_release": target_manifest["release"],
                "tested_backend": target_manifest["tested_backend"]["revision"],
                "expires_at": time.time() + OFFER_TTL,
            })
        except Exception:
            if not self.recording_failed:
                self._phase(
                    "failed",
                    "Release verification failed; no service action was taken.",
                )

    def apply(self, actor, offer_id, confirm_restart_and_rollback=False):
        with self.guard:
            E.require(not self.closed, "Updater is closing")
            E.require(not self.recording_failed, "Updater journal requires recovery")
            state = read_state(self.directory)
            E.require(
                confirm_restart_and_rollback is True,
                "Explicit restart and rollback consent is required",
            )
            E.require(
                state.get("id") == offer_id and state.get("actor") == list(actor),
                "Offer does not belong to this request",
            )
            if state.get("confirmed"):
                return self.status()
            E.require(
                state["phase"] == "offered" and state["expires_at"] > time.time(),
                "Offer is unavailable or expired",
            )
            E.require(
                state["configuration"] == configuration(self.config),
                "Updater configuration changed",
            )
            E.require(
                self.worker is None or not self.worker.is_alive(),
                "Updater worker is busy",
            )
            self._save({
                **state,
                "phase": "stopping",
                "confirmed": True,
                "consent": WARNING,
            })
            self._start(self._execute)
            return self.status()

    def _candidate(self, state):
        folder = self.directory / "candidate"
        launcher = E.read_regular(folder / "launcher")
        E.require(
            E.digest(launcher) == state["descriptor"]["launcher"]["sha256"],
            "Candidate launcher changed",
        )
        manifest, payload = E.archive_payload(
            str(folder / "archive"), state["target"]["archive_sha256"], launcher
        )
        target, _ = E.installation_files(
            state["target"]["archive_sha256"],
            state["current"]["selection"],
            payload,
            launcher,
        )
        E.require(target == state["target"], "Candidate differs from confirmed receipt")
        preflight(target["selection"], manifest)
        return folder

    def _execute(self):
        attempted_stop = False
        try:
            state = read_state(self.directory)
            E.require(
                state["configuration"] == configuration(self.config),
                "Updater configuration changed",
            )
            E.require(
                E.installed(self.root)[0] == state["current"],
                "Installed browser changed",
            )
            folder = self._candidate(state)
            observed = self.dashboard.status()
            E.require(
                observed["state"] == "ready"
                and observed["receipt"] == state["current"],
                "Dashboard ownership changed",
            )
            attempted_stop = True
            self.dashboard.stop(state["current"])
            self._phase("switching")
            E.maintenance(
                Namespace(
                    command="update",
                    install_root=str(self.root),
                    archive=str(folder / "archive"),
                    launcher=str(folder / "launcher"),
                    sha256=state["target"]["archive_sha256"],
                ),
                confirm=lambda _: True,
                validate=preflight,
                expected_current=state["current"],
                expected_target=state["target"],
            )
            self._phase("starting")
            self.dashboard.start(state["target"])
            self._phase("succeeded")
        except Exception:
            if self.recording_failed:
                return
            if not attempted_stop:
                self._phase(
                    "failed", "Update preflight failed; no service action was taken."
                )
                return
            self._restore()

    def _restore(self):
        try:
            state = read_state(self.directory)
            E.require(
                state["configuration"] == configuration(self.config),
                "Updater configuration changed",
            )
            receipt, _ = E.installed(self.root)
            E.require(
                receipt in (state["current"], state["target"]),
                "Installation is not part of the confirmed update",
            )
            self._phase("rollback_stopping")
            self.dashboard.stop(receipt)
            self._phase("rollback_switching")
            E.maintenance(
                Namespace(
                    command="rollback",
                    install_root=str(self.root),
                    to=state["current"]["archive_sha256"],
                ),
                confirm=lambda _: True,
                validate=preflight,
                expected_current=receipt,
                expected_target=state["current"],
            )
            self._phase("rollback_starting")
            self.dashboard.start(state["current"])
            self._phase(
                "rolled_back",
                "Update did not activate; the previous browser release was restored.",
            )
        except Exception:
            if self.recording_failed:
                return
            self._phase(
                "recovery_required",
                "Automatic recovery could not establish a safe running installation. Inspect the retained journal; no ownership lock was cleared.",
            )
