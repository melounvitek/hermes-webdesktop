"""Shared runner for user-configured shell ("command") TTS/STT providers.

Both ``tools.tts_tool`` and ``tools.transcription_tools`` let users declare a
provider as a shell command template with ``{placeholders}``. This module owns
the shell-quote-aware template rendering and the idle-timeout process runner
they share; each origin module re-imports these under its historical private
names.
"""

from __future__ import annotations

import os
import queue
import re
import shlex
import subprocess
import threading
import time
from typing import Any, Dict, Optional


def shell_quote_context(command_template: str, position: int) -> Optional[str]:
    """Return the shell quote char (``'``/``"``) active right before *position*, or None."""
    quote: Optional[str] = None
    escaped = False
    i = 0
    while i < position:
        char = command_template[i]
        if quote == "'":
            if char == "'":
                quote = None
        elif quote == '"':
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                quote = None
        elif char == "'":
            quote = "'"
        elif char == '"':
            quote = '"'
        elif char == "\\":
            i += 1
        i += 1
    return quote


def quote_command_placeholder(value: str, quote_context: Optional[str]) -> str:
    """Quote a placeholder value for its position in a shell command template."""
    if quote_context == "'":
        return value.replace("'", r"'\''")
    if quote_context == '"':
        return (
            value
            .replace("\\", "\\\\")
            .replace('"', r'\"')
            .replace("$", r"\$")
            .replace("`", r"\`")
        )
    if os.name == "nt":
        return subprocess.list2cmdline([value])
    return shlex.quote(value)


def render_command_template(
    command_template: str,
    placeholders: Dict[str, str],
) -> str:
    """Replace ``{name}`` placeholders (quote-aware) while preserving ``{{``/``}}``."""
    names = "|".join(re.escape(name) for name in placeholders)
    pattern = re.compile(
        rf"(?<!\$)(?:\{{\{{(?P<double>{names})\}}\}}|\{{(?P<single>{names})\}})"
    )
    replacements: list[tuple[str, str]] = []

    def replace_match(match: re.Match[str]) -> str:
        name = match.group("double") or match.group("single")
        token = f"__HERMES_CMD_PLACEHOLDER_{len(replacements)}__"
        replacements.append((
            token,
            quote_command_placeholder(
                placeholders[name],
                shell_quote_context(command_template, match.start()),
            ),
        ))
        return token

    rendered = pattern.sub(replace_match, command_template)
    rendered = rendered.replace("{{", "{").replace("}}", "}")
    for token, value in replacements:
        rendered = rendered.replace(token, value)
    return rendered


def _signal_process_tree(psutil: Any, proc: subprocess.Popen, method: str) -> None:
    """Apply ``terminate``/``kill`` to *proc* and all descendants (best effort)."""
    try:
        parent = psutil.Process(proc.pid)
        for child in parent.children(recursive=True):
            try:
                getattr(child, method)()
            except psutil.NoSuchProcess:
                pass
        getattr(parent, method)()
    except psutil.NoSuchProcess:
        return
    except Exception:
        getattr(proc, method)()


def terminate_command_process_tree(proc: subprocess.Popen) -> None:
    """Best-effort termination of a shell process and all of its children."""
    if proc.poll() is not None:
        return

    if os.name == "nt":
        try:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(proc.pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=5,
                stdin=subprocess.DEVNULL,
            )
        except Exception:
            proc.kill()
        return

    try:
        import psutil  # type: ignore
    except ImportError:
        proc.terminate()
        try:
            proc.wait(timeout=2)
        except subprocess.TimeoutExpired:
            proc.kill()
        return

    _signal_process_tree(psutil, proc, "terminate")
    try:
        proc.wait(timeout=2)
        return
    except subprocess.TimeoutExpired:
        pass
    _signal_process_tree(psutil, proc, "kill")


def command_env_passthrough(config: Dict[str, Any]) -> list:
    """Return the provider's ``env_passthrough`` allowlist.

    The child env is scrubbed of Hermes secrets by default; this list names
    variables copied back from the parent env so a trusted template (e.g. a
    curl one-liner using its own API key) keeps working.
    """
    raw = config.get("env_passthrough")
    if not isinstance(raw, (list, tuple)):
        return []
    return [str(item).strip() for item in raw if str(item).strip()]


def run_command_provider(
    command: str,
    timeout: float,
    env_passthrough: Optional[list] = None,
) -> subprocess.CompletedProcess:
    """Run a command-provider shell command with process-tree idle cleanup.

    ``timeout`` is an IDLE timeout, reset whenever the command emits output —
    a slow-but-alive provider survives, a silently stalled one is killed.
    Child env is scrubbed of Hermes secrets while propagating delegated-child
    lineage markers.
    """
    from agent.delegation_context import delegated_child_subprocess_env
    from tools.environments.local import hermes_subprocess_env

    scrubbed = hermes_subprocess_env(inherit_credentials=False)
    for key in env_passthrough or []:
        value = os.environ.get(key)
        if value is not None:
            scrubbed[key] = value
    popen_kwargs: Dict[str, Any] = {
        "shell": True,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        # Lossy UTF-8 decode: locale-mismatched bytes must not raise in the
        # reader threads on non-UTF-8 Windows.
        "encoding": "utf-8",
        "errors": "replace",
        "env": delegated_child_subprocess_env(scrubbed),
    }
    if os.name == "nt":
        popen_kwargs["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)
    else:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(command, **popen_kwargs, stdin=subprocess.DEVNULL)
    output_queue: "queue.Queue[tuple[str, Optional[str]]]" = queue.Queue()
    chunks: Dict[str, list[str]] = {"stdout": [], "stderr": []}
    open_streams = {"stdout", "stderr"}

    def read_stream(name: str, stream: Any) -> None:
        encoding = getattr(stream, "encoding", None) or "utf-8"
        read1 = getattr(getattr(stream, "buffer", None), "read1", None)
        try:
            while True:
                if read1 is None:
                    chunk = stream.read(65536)
                else:
                    chunk = read1(65536).decode(encoding, errors="replace")
                if not chunk:
                    break
                output_queue.put((name, chunk))
        finally:
            output_queue.put((name, None))

    readers = [
        threading.Thread(target=read_stream, args=(name, stream), daemon=True)
        for name, stream in (("stdout", proc.stdout), ("stderr", proc.stderr))
    ]
    for reader in readers:
        reader.start()

    deadline = time.monotonic() + timeout
    timed_out = False
    while open_streams:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            timed_out = True
            break
        try:
            name, chunk = output_queue.get(timeout=min(0.05, remaining))
        except queue.Empty:
            continue
        if chunk is None:
            open_streams.discard(name)
            continue
        chunks[name].append(chunk)
        deadline = time.monotonic() + timeout

    if not timed_out:
        try:
            proc.wait(timeout=max(0.0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            timed_out = True

    if timed_out:
        terminate_command_process_tree(proc)
        for reader in readers:
            reader.join(timeout=0.5)
        while True:
            try:
                name, chunk = output_queue.get_nowait()
            except queue.Empty:
                break
            if chunk:
                chunks[name].append(chunk)
        stdout = "".join(chunks["stdout"])
        stderr = "".join(chunks["stderr"])
        try:
            raise subprocess.TimeoutExpired(command, timeout)
        except subprocess.TimeoutExpired as exc:
            raise subprocess.TimeoutExpired(
                command, timeout, output=stdout, stderr=stderr,
            ) from exc

    stdout = "".join(chunks["stdout"])
    stderr = "".join(chunks["stderr"])

    if proc.returncode:
        raise subprocess.CalledProcessError(
            proc.returncode, command, output=stdout, stderr=stderr,
        )
    return subprocess.CompletedProcess(command, proc.returncode, stdout, stderr)
