#!/usr/bin/env python3
"""Parent-death watchdog supervisor for stdio MCP subprocesses.

If Hermes dies hard (kill -9, crash, force-quit) its graceful teardown never
runs and the stdio child plus its descendants are orphaned (macOS has no
``PR_SET_PDEATHSIG``); piled-up orphans then race the legitimate new connection
for the same upstream session. So the MCP command is spawned via this
supervisor, which (1) runs the real command as its own child in a new process
group so the whole tree can be killpg'd, (2) passes stdin/stdout/stderr straight
through — the MCP stdio protocol talks over those pipes, so this must be a
no-op relay, not a proxy — and (3) polls ``getppid()`` against the recorded
parent PID and, once the parent is gone, SIGTERMs the child's group, waits,
then SIGKILLs. Standard-library only so it starts fast and cannot itself leak.

Usage (see ``_wrap_command_with_watchdog``)::

    python3 -m tools.mcp_stdio_watchdog \\
        --ppid <original_parent_pid> -- <real_command> <arg1> <arg2> ...
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import threading
import time

_POLL_INTERVAL_S = 2.0
_TERM_GRACE_S = 3.0


def _is_orphaned(original_ppid: int, getppid=os.getppid) -> bool:
    """Return whether this process no longer has its original POSIX parent."""
    return getppid() != original_ppid


def _terminate_process_group(proc: subprocess.Popen) -> None:
    """Best-effort SIGTERM-then-SIGKILL of the child's process group; guards the
    POSIX-only primitives so an accidental Windows run degrades to a plain child
    kill instead of AttributeError."""
    killpg = getattr(os, "killpg", None)
    if killpg is None:  # windows-footgun: ok — non-POSIX fallback
        try:
            proc.terminate()
            proc.wait(timeout=_TERM_GRACE_S)
        except (OSError, subprocess.TimeoutExpired):
            proc.kill()
        return
    try:
        pgid = os.getpgid(proc.pid)
    except (ProcessLookupError, OSError):
        return
    sigkill = getattr(signal, "SIGKILL", signal.SIGTERM)
    for sig in (signal.SIGTERM, sigkill):
        try:
            killpg(pgid, sig)
        except (ProcessLookupError, PermissionError, OSError):
            return
        try:
            proc.wait(timeout=_TERM_GRACE_S)
            return
        except subprocess.TimeoutExpired:
            continue


def _watchdog_loop(proc: subprocess.Popen, original_ppid: int) -> None:
    while proc.poll() is None:
        if _is_orphaned(original_ppid):
            _terminate_process_group(proc)
            return
        time.sleep(_POLL_INTERVAL_S)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Parent-death watchdog for a stdio MCP subprocess.")
    parser.add_argument("--ppid", type=int, required=True)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)

    real_argv = list(args.command)
    if real_argv and real_argv[0] == "--":
        real_argv = real_argv[1:]
    if not real_argv:
        print("mcp_stdio_watchdog: no command given after '--'", file=sys.stderr)
        return 2

    # New process group: killpg() reaches the whole tree the real command may
    # spawn without touching our own group or the original parent's.
    proc = subprocess.Popen(real_argv, stdin=sys.stdin, stdout=sys.stdout, stderr=sys.stderr, start_new_session=True)

    # The server lives in its OWN group, so the parent's shutdown killpg of *our*
    # group no longer reaches it: forward SIGTERM/SIGINT to the child's group so
    # graceful teardown still kills a wedged server that ignores stdin EOF.
    def _forward_shutdown(signum, frame):  # noqa: ARG001
        _terminate_process_group(proc)
        sys.exit(128 + signum)

    signal.signal(signal.SIGTERM, _forward_shutdown)
    signal.signal(signal.SIGINT, _forward_shutdown)

    threading.Thread(target=_watchdog_loop, args=(proc, args.ppid), daemon=True).start()

    try:
        return proc.wait()
    except KeyboardInterrupt:
        _terminate_process_group(proc)
        return 130


if __name__ == "__main__":
    sys.exit(main())
