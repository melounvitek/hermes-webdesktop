"""Admission for one explicitly approved user-systemd dashboard controller."""

import os
from pathlib import Path
import re
import shlex
import stat
import subprocess
import sys
import time

from browser_setup import E


SERVICE_TIMEOUT = 30
HOOKS = (
    "ExecStartPre",
    "ExecStartPost",
    "ExecStop",
    "ExecStopPost",
    "ExecCondition",
    "ExecReload",
)
PROPERTIES = (
    "Id",
    "LoadState",
    "ActiveState",
    "MainPID",
    "FragmentPath",
    "DropInPaths",
    "NeedDaemonReload",
    "Type",
    "Restart",
    "KillMode",
    "SendSIGKILL",
    "KillSignal",
    "TimeoutStopFailureMode",
    "WorkingDirectory",
    "ExecStart",
    "ExecStartEx",
    *HOOKS,
)


class Dashboard:
    def __init__(self, config):
        self.config = config
        self.root = E.absolute_path(config.maintenance["installation"])
        self.unit = config.maintenance["unit"]
        E.require(
            re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.service", self.unit),
            "Expected a fixed service basename",
        )

    def _systemctl(self, *args, deadline=None):
        timeout = SERVICE_TIMEOUT if deadline is None else deadline - time.monotonic()
        E.require(timeout > 0, "Dashboard service deadline expired")
        result = subprocess.run(
            ["systemctl", "--user", *args, self.unit],
            capture_output=True,
            text=True,
            timeout=min(timeout, SERVICE_TIMEOUT),
            env={**os.environ, "LC_ALL": "C", "SYSTEMD_COLORS": "0"},
        )
        E.require(result.returncode == 0, "Dashboard systemctl command failed")
        return result.stdout

    def _unit(self, deadline=None):
        output = self._systemctl(
            "show",
            "--all",
            "--no-pager",
            "--property=" + ",".join(PROPERTIES),
            deadline=deadline,
        )
        props = {}
        for line in output.splitlines():
            key, separator, value = line.partition("=")
            E.require(separator and key not in props, "Invalid systemctl show response")
            props[key] = value
        E.require(
            set(PROPERTIES) - set(HOOKS) <= props.keys(),
            "Incomplete systemctl show response",
        )
        # systemctl omits empty Exec* arrays even with --all (systemd 259).
        for hook in HOOKS:
            props.setdefault(hook, "")
        required = dict(
            Id=self.unit,
            LoadState="loaded",
            DropInPaths="",
            NeedDaemonReload="no",
            Type="simple",
            Restart="no",
            KillMode="mixed",
            SendSIGKILL="no",
            KillSignal="15",
            TimeoutStopFailureMode="terminate",
            WorkingDirectory=self.config.maintenance["working_directory"],
            **dict.fromkeys(HOOKS, ""),
        )
        E.require(
            all(props[key] == value for key, value in required.items()),
            "Unsupported dashboard unit definition",
        )
        fragment = E.absolute_path(props["FragmentPath"])
        E.no_links(fragment)
        fd = os.open(fragment, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(fd, "rb") as stream:
            meta = os.fstat(stream.fileno())
            E.require(
                stat.S_ISREG(meta.st_mode)
                and meta.st_uid == os.getuid()
                and meta.st_nlink == 1
                and not meta.st_mode & 0o022,
                "Dashboard unit must be owned and not writable by others",
            )
            raw = stream.read(E.MAX_JSON + 1)
        E.require(
            len(raw) <= E.MAX_JSON
            and E.digest(raw) == self.config.maintenance["unit_sha256"],
            "Dashboard unit hash mismatch",
        )
        argv = [
            sys.executable,
            "-I",
            "-S",
            "-B",
            str(Path(__file__).with_name("browser_updater.py")),
            "--config",
            str(self.config.config_path),
            "--dashboard",
        ]
        # show flattens argv boundaries. Also check the pinned fragment's token
        # boundaries; support literal commands only, not systemd expansion syntax.
        commands = []
        section = None
        for line in raw.decode("utf-8").splitlines():
            line = line.strip()
            if line.startswith(("#", ";")) or not line:
                continue
            E.require(not line.endswith("\\"), "Unit continuations are unsupported")
            if line.startswith("["):
                section = line
            elif section == "[Service]":
                key, _, value = line.partition("=")
                if key.strip() == "ExecStart":
                    if not value.strip():
                        commands.clear()
                    else:
                        E.require(
                            not any(char in value for char in "\\$%"),
                            "Unit command expansion is unsupported",
                        )
                        commands.append(shlex.split(value))
        E.require(
            commands == [argv], "Dashboard ExecStart differs from approved command"
        )
        for key, flags in (
            ("ExecStart", "ignore_errors=no"),
            ("ExecStartEx", "flags="),
        ):
            prefix = f"{{ path={argv[0]} ; argv[]={' '.join(argv)} ; {flags} ; "
            # Execution timestamps/PIDs vary, but exactly one complete command is required.
            E.require(
                re.fullmatch(
                    re.escape(prefix)
                    + r"start_time=\[[^\]\n]*\] ; stop_time=\[[^\]\n]*\] ; pid=[0-9]+ ; code=[^;{}\n]+ ; status=[^;{}\n]+ \}",
                    props[key],
                ),
                "Unsupported loaded dashboard ExecStart",
            )
        return props

    def _observe(self, deadline=None):
        props = self._unit(deadline)
        E.no_links(self.root)
        E.require(
            props["MainPID"].isascii() and props["MainPID"].isdigit(),
            "Invalid unit MainPID",
        )
        pid = int(props["MainPID"])
        stream = E.open_control(self.root, False)
        E.require(stream is not None, "Missing lifecycle ownership record")
        with stream:
            if props["ActiveState"] in ("inactive", "failed") and pid == 0:
                # Inactive is not evidence of reaping. Only the engine's locked,
                # persistent stopped record permits maintenance or another start.
                with E.stopped_control(self.root) as (locked, record):
                    E.require(
                        os.fstat(stream.fileno()).st_ino
                        == os.fstat(locked.fileno()).st_ino,
                        "Lifecycle record replaced",
                    )
                    return {"state": "stopped", "installation": str(self.root)}
            E.require(
                props["ActiveState"] == "active" and pid > 1,
                "Dashboard unit ownership unknown",
            )
            E.require(
                not E.acquire_control(stream),
                "Live controller does not own lifecycle lock",
            )
            record = E.read_control(stream, self.root)
            info = E.control_request(self.root, "status", expected_pid=pid)
            E.require(
                record["state"] == "unknown"
                and record["generation"] is not None
                and info.get("generation") == record["generation"]
                and info.get("pid") == record["pid"]
                and info.get("state") in ("ready", "starting")
                and isinstance(info.get("receipt"), dict)
                and info.get("url") == f"http://127.0.0.1:{self.config.backend_port}/",
                "Controller and persistent lifecycle identity differ",
            )
            return info

    def status(self):
        try:
            return self._observe()
        except (OSError, ValueError, subprocess.TimeoutExpired) as error:
            return {"state": "unknown", "error": str(error)}

    def _wait(self, state, expected_receipt, deadline):
        while time.monotonic() < deadline:
            try:
                info = self._observe(deadline)
            except (OSError, ValueError, subprocess.TimeoutExpired):
                info = {"state": "unknown"}
            if info["state"] == state:
                if state == "ready":
                    E.require(
                        info["receipt"] == expected_receipt,
                        "Unexpected running dashboard receipt",
                    )
                return info
            time.sleep(0.1)
        raise ValueError("Dashboard service deadline expired; ownership not recovered")

    def stop(self, expected_receipt):
        deadline = time.monotonic() + SERVICE_TIMEOUT
        info = self._observe(deadline)
        if info["state"] == "stopped":
            E.require(
                E.installed(self.root)[0] == expected_receipt,
                "Unexpected stopped dashboard receipt",
            )
            return info
        E.require(
            info["receipt"] == expected_receipt, "Unexpected running dashboard receipt"
        )
        self._systemctl("stop", "--no-block", deadline=deadline)
        return self._wait("stopped", expected_receipt, deadline)

    def start(self, expected_receipt):
        deadline = time.monotonic() + SERVICE_TIMEOUT
        info = self._observe(deadline)
        if info["state"] == "ready":
            E.require(
                info["receipt"] == expected_receipt,
                "Unexpected running dashboard receipt",
            )
            return info
        E.require(info["state"] == "stopped", "Dashboard is not known stopped")
        with E.stopped_control(self.root):
            E.require(
                E.installed(self.root)[0] == expected_receipt,
                "Unexpected stopped dashboard receipt",
            )
        # The --dashboard entry point performs journal admission under the engine
        # lock. Never hold that lock while waiting for the controller to acquire it.
        self._systemctl("start", "--no-block", deadline=deadline)
        return self._wait("ready", expected_receipt, deadline)
