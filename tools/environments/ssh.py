"""SSH remote execution environment with ControlMaster connection persistence."""

import hashlib
import logging
import os
import shlex
import shutil
import subprocess
import tempfile
from pathlib import Path

from tools.environments.base import BaseEnvironment, EnvironmentConnectionError, _popen_bash
from tools.environments.file_sync import (
    FileSyncManager, iter_sync_files, quoted_mkdir_command, quoted_rm_command, unique_parent_dirs,
)
from tools.environments.remote_common import bash_argv, run_capture

logger = logging.getLogger(__name__)

# Windows OpenSSH has no Unix-socket ControlMaster: ControlPath/ControlMaster options
# fail the connection outright ('getsockname failed: Not a socket'). Skip multiplexing there.
_SSH_MULTIPLEX = os.name != "nt"


def _ensure_ssh_available() -> None:
    """Fail fast with a clear error when the SSH client is unavailable."""
    if not shutil.which("ssh"):
        raise RuntimeError("SSH is not installed or not in PATH. Install OpenSSH client: apt install openssh-client")
    if not shutil.which("scp"):
        raise RuntimeError("SCP is not installed or not in PATH. Install OpenSSH client: apt install openssh-client")


class SSHEnvironment(BaseEnvironment):
    """Run commands on a remote machine over SSH.

    Spawn-per-call: every execute() spawns a fresh ``ssh ... bash -c`` process.
    Session snapshot preserves env vars across calls; CWD persists via in-band
    stdout markers. Uses SSH ControlMaster for connection reuse.
    """

    def __init__(self, host: str, user: str, cwd: str = "~",
                 timeout: int = 60, port: int = 22, key_path: str = ""):
        super().__init__(cwd=cwd, timeout=timeout)
        self.host = host
        self.user = user
        self.port = port
        self.key_path = key_path

        self.control_dir = Path(tempfile.gettempdir()) / "hermes-ssh"
        self.control_dir.mkdir(parents=True, exist_ok=True)
        # Short, deterministic socket name: the full path must stay under macOS's
        # 104-byte sun_path limit (raw user@host:port + SSH's 16-byte random suffix
        # under a deep $TMPDIR exceeds it), and stability across reconnects keeps
        # ControlMaster reuse working.
        _socket_id = hashlib.sha256(f"{user}@{host}:{port}".encode()).hexdigest()[:16]
        self.control_socket = self.control_dir / f"{_socket_id}.sock"
        _ensure_ssh_available()
        self._establish_connection()
        self._remote_home = self._detect_remote_home()

        self._ensure_remote_dirs()
        self._sync_manager = FileSyncManager(
            get_files_fn=lambda: iter_sync_files(f"{self._remote_home}/.hermes"),
            upload_fn=self._scp_upload,
            delete_fn=self._ssh_delete,
            bulk_upload_fn=self._ssh_bulk_upload,
            bulk_download_fn=self._ssh_bulk_download,
        )
        self._sync_manager.sync(force=True)
        self.init_session()

    def _build_ssh_command(self, extra_args: list | None = None) -> list:
        cmd = ["ssh"]
        if _SSH_MULTIPLEX:
            cmd.extend(["-o", f"ControlPath={self.control_socket}",
                        "-o", "ControlMaster=auto", "-o", "ControlPersist=300"])
        cmd.extend(["-o", "BatchMode=yes", "-o", "StrictHostKeyChecking=accept-new",
                    "-o", "ConnectTimeout=10"])
        if self.port != 22:
            cmd.extend(["-p", str(self.port)])
        if self.key_path:
            cmd.extend(["-i", self.key_path])
        if extra_args:
            cmd.extend(extra_args)
        cmd.append(f"{self.user}@{self.host}")
        return cmd

    def _run_ssh(self, remote_cmd: str, timeout: float) -> subprocess.CompletedProcess:
        """Run one remote shell command over the multiplexed connection, capturing output."""
        cmd = self._build_ssh_command()
        cmd.append(remote_cmd)
        return run_capture(cmd, timeout=timeout)

    def _establish_connection(self):
        try:
            result = self._run_ssh("echo 'SSH connection established'", timeout=15)
            if result.returncode != 0:
                error_msg = result.stderr.strip() or result.stdout.strip()
                raise EnvironmentConnectionError(
                    f"SSH connection failed: {error_msg}",
                    retry_hint=(f"Verify {self.user}@{self.host}:{self.port} is reachable "
                                "(host up, sshd running, key/agent auth working), then "
                                "retry — the connection is re-established automatically."),
                )
        except subprocess.TimeoutExpired:
            raise EnvironmentConnectionError(
                f"SSH connection to {self.user}@{self.host} timed out",
                retry_hint=(f"Check network connectivity to {self.host}:{self.port} "
                            "and that sshd is accepting connections, then retry."),
            )

    def _detect_remote_home(self) -> str:
        """Detect the remote user's home directory."""
        try:
            result = self._run_ssh("echo $HOME", timeout=10)
            home = result.stdout.strip()
            if home and result.returncode == 0:
                logger.debug("SSH: remote home = %s", home)
                return home
        except Exception:
            pass
        return "/root" if self.user == "root" else f"/home/{self.user}"

    # -- File sync (via FileSyncManager) --------------------------------

    def _ensure_remote_dirs(self) -> None:
        """Create base ~/.hermes directory tree on remote in one SSH call."""
        base = f"{self._remote_home}/.hermes"
        dirs = [base, f"{base}/skills", f"{base}/credentials", f"{base}/cache"]
        self._run_ssh(quoted_mkdir_command(dirs), timeout=10)

    def _scp_upload(self, host_path: str, remote_path: str) -> None:
        """Upload a single file via scp over ControlMaster."""
        parent = str(Path(remote_path).parent)
        self._run_ssh(f"mkdir -p {shlex.quote(parent)}", timeout=10)

        scp_cmd = ["scp"]
        if _SSH_MULTIPLEX:
            scp_cmd.extend(["-o", f"ControlPath={self.control_socket}"])
        if self.port != 22:
            scp_cmd.extend(["-P", str(self.port)])
        if self.key_path:
            scp_cmd.extend(["-i", self.key_path])
        scp_cmd.extend([host_path, f"{self.user}@{self.host}:{remote_path}"])
        result = run_capture(scp_cmd, timeout=30)
        if result.returncode != 0:
            raise EnvironmentConnectionError(
                f"scp failed: {result.stderr.strip()}",
                retry_hint=(f"File sync to {self.user}@{self.host} failed — verify the "
                            "SSH connection is healthy, then retry."),
            )

    def _ssh_bulk_upload(self, files: list[tuple[str, str]]) -> None:
        """Upload many files in a single tar-over-SSH stream.

        Local ``tar c`` is piped through one SSH connection to remote ``tar x``;
        directory creation is batched into a single ``mkdir -p`` beforehand.
        """
        if not files:
            return

        base = f"{self._remote_home}/.hermes"
        parents = unique_parent_dirs(files)
        if parents:
            result = self._run_ssh(quoted_mkdir_command(parents), timeout=30)
            if result.returncode != 0:
                raise EnvironmentConnectionError(
                    f"remote mkdir failed: {result.stderr.strip()}",
                    retry_hint=(f"Remote directory setup on {self.host} failed — verify "
                                "the SSH connection is healthy, then retry."),
                )

        # Symlink staging avoids fragile GNU tar --transform rules. On Windows
        # without Developer Mode symlink creation raises OSError winerror 1314;
        # only that case falls back to a plain copy, other OSErrors re-raise.
        with tempfile.TemporaryDirectory(prefix="hermes-ssh-bulk-") as staging:
            for host_path, remote_path in files:
                try:
                    rel_remote = os.path.relpath(remote_path, base)
                except ValueError as exc:
                    raise RuntimeError(f"remote path {remote_path!r} is not under sync base {base!r}") from exc
                if rel_remote == "." or rel_remote.startswith("../"):
                    raise RuntimeError(f"remote path {remote_path!r} escapes sync base {base!r}")

                staged = os.path.join(staging, rel_remote)
                os.makedirs(os.path.dirname(staged), exist_ok=True)
                try:
                    os.symlink(os.path.abspath(host_path), staged)
                except OSError as e:
                    if getattr(e, "winerror", None) == 1314:
                        shutil.copy2(host_path, staged)
                    else:
                        raise

            tar_cmd = ["tar", "-chf", "-", "-C", staging, "."]
            ssh_cmd = self._build_ssh_command()
            # --no-overwrite-dir keeps tar from stamping the staging dir's mode onto
            # existing dirs (e.g. /home/<user>); a umask-002 0775 home breaks sshd StrictModes.
            ssh_cmd.append(f"tar xf - --no-overwrite-dir -C {shlex.quote(base)}")

            tar_proc = subprocess.Popen(tar_cmd, stdin=subprocess.DEVNULL,
                                        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            try:
                ssh_proc = subprocess.Popen(ssh_cmd, stdin=tar_proc.stdout,
                                            stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            except Exception:
                tar_proc.kill()
                tar_proc.wait()
                raise

            # Allow tar_proc to receive SIGPIPE if ssh_proc exits early
            tar_proc.stdout.close()

            try:
                _, ssh_stderr = ssh_proc.communicate(timeout=120)
                # communicate() (not wait()) drains stderr so tar can't deadlock on >PIPE_BUF errors.
                tar_stderr_raw = b""
                if tar_proc.poll() is None:
                    _, tar_stderr_raw = tar_proc.communicate(timeout=10)
                else:
                    tar_stderr_raw = tar_proc.stderr.read() if tar_proc.stderr else b""
            except subprocess.TimeoutExpired:
                tar_proc.kill()
                ssh_proc.kill()
                tar_proc.wait()
                ssh_proc.wait()
                raise EnvironmentConnectionError(
                    "SSH bulk upload timed out",
                    retry_hint=f"Bulk file sync to {self.host} timed out — check the connection and retry.",
                )

            if tar_proc.returncode != 0:
                raise RuntimeError(f"tar create failed (rc={tar_proc.returncode}): "
                                   f"{tar_stderr_raw.decode(errors='replace').strip()}")
            if ssh_proc.returncode != 0:
                raise EnvironmentConnectionError(
                    f"tar extract over SSH failed (rc={ssh_proc.returncode}): "
                    f"{ssh_stderr.decode(errors='replace').strip()}",
                    retry_hint=(f"File sync over SSH to {self.host} failed — verify the "
                                "connection is healthy, then retry."),
                )

        logger.debug("SSH: bulk-uploaded %d file(s) via tar pipe", len(files))

    def _ssh_bulk_download(self, dest: Path) -> None:
        """Download remote .hermes/ as a tar archive."""
        # Tar from / with the full path so archive entries keep absolute paths
        # (home/user/.hermes/skills/f.py), matching _pushed_hashes keys.
        rel_base = f"{self._remote_home}/.hermes".lstrip("/")
        ssh_cmd = self._build_ssh_command()
        ssh_cmd.append(f"tar cf - -C / {shlex.quote(rel_base)}")
        with open(dest, "wb") as f:
            result = subprocess.run(ssh_cmd, stdin=subprocess.DEVNULL, stdout=f,
                                    stderr=subprocess.PIPE, timeout=120)
        if result.returncode != 0:
            raise EnvironmentConnectionError(
                f"SSH bulk download failed: {result.stderr.decode(errors='replace').strip()}",
                retry_hint=(f"File sync from {self.host} failed — verify the SSH "
                            "connection is healthy, then retry."),
            )

    def _ssh_delete(self, remote_paths: list[str]) -> None:
        """Batch-delete remote files in one SSH call."""
        result = self._run_ssh(quoted_rm_command(remote_paths), timeout=10)
        if result.returncode != 0:
            raise EnvironmentConnectionError(
                f"remote rm failed: {result.stderr.strip()}",
                retry_hint=(f"Remote file cleanup on {self.host} failed — verify the "
                            "SSH connection is healthy, then retry."),
            )

    def _before_execute(self) -> None:
        """Sync files to remote via FileSyncManager (rate-limited internally)."""
        self._sync_manager.sync()

    # -- Execution ------------------------------------------------------

    def _run_bash(self, cmd_string: str, *, login: bool = False, timeout: int = 120,
                  stdin_data: str | None = None) -> subprocess.Popen:
        """Spawn an SSH process that runs bash on the remote host."""
        cmd = self._build_ssh_command()
        cmd.extend(bash_argv(shlex.quote(cmd_string), login))
        return _popen_bash(cmd, stdin_data)

    def cleanup(self):
        if self._sync_manager:
            logger.info("SSH: syncing files from sandbox...")
            self._sync_manager.sync_back()

        if self.control_socket.exists():
            try:
                cmd = ["ssh", "-o", f"ControlPath={self.control_socket}", "-O", "exit", f"{self.user}@{self.host}"]
                subprocess.run(cmd, capture_output=True, timeout=5, stdin=subprocess.DEVNULL)
            except (OSError, subprocess.SubprocessError):
                pass
            try:
                self.control_socket.unlink()
            except OSError:
                pass
