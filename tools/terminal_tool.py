#!/usr/bin/env python3
"""Terminal tool: run shell commands in the configured backend.

Backends (``TERMINAL_ENV``): local (default), docker, singularity, modal
(direct or managed gateway), daytona, vercel_sandbox, ssh, plus
plugin-registered backends. Handles background processes, sandbox lifecycle
(per-task cache, idle reaper, atexit teardown) and sudo password plumbing.
Cloud-sandbox persistent filesystems preserve working state across sandbox
recreation but do NOT guarantee the same live sandbox or long-running
processes survive cleanup, idle reaping, or Hermes exit.

Companion modules: ``terminal_tool_guards`` (pre-exec blocks),
``terminal_tool_background`` (background spawn), ``terminal_tool_result``
(foreground result post-processing).
"""

import importlib.util
import inspect
import json
import logging
import os
import platform
import re
import sys
import time
import threading
import atexit
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Dict, Any, List

from utils import env_var_enabled

logger = logging.getLogger(__name__)


def _redact_terminal_error_text(value: Any) -> str:
    """Force-redact text before serializing a terminal error envelope."""
    from agent.redact import redact_sensitive_text

    return redact_sensitive_text("" if value is None else str(value), force=True)


# Interrupt event set by the agent on user interrupt; executors poll it to
# kill long-running subprocesses instead of blocking until timeout.
from tools.interrupt import is_interrupted, _interrupt_event  # noqa: F401 — re-exported
from tools.registry import tool_error
# display_hermes_home imported lazily at call site (stale-module safety during hermes update)
from tools.environments.singularity import _get_scratch_dir
from tools.tool_backend_helpers import (
    coerce_modal_mode,
    has_direct_modal_credentials,
    managed_nous_tools_enabled,
    nous_tool_gateway_unavailable_message,
    resolve_modal_backend_state,
)


def _safe_parse_import_env(
    name: str,
    default: Any,
    converter,
    type_label: str,
):
    """Parse a module-level numeric env var; a malformed value must never make
    the module unloadable at import time (CLI, ACP, tests, tool discovery)."""
    raw = os.getenv(name)
    if raw is None or raw == "":
        return default
    try:
        return converter(raw)
    except (TypeError, ValueError):
        logger.warning(
            "Invalid value for %s: %r (expected %s). Falling back to %r.",
            name,
            raw,
            type_label,
            default,
        )
        return default


# Hard cap on foreground timeout; override via TERMINAL_MAX_FOREGROUND_TIMEOUT env var.
FOREGROUND_MAX_TIMEOUT = _safe_parse_import_env(
    "TERMINAL_MAX_FOREGROUND_TIMEOUT",
    600,
    int,
    "integer",
)

# Disk usage warning threshold (in GB)
DISK_USAGE_WARNING_THRESHOLD_GB = _safe_parse_import_env(
    "TERMINAL_DISK_WARNING_GB",
    500.0,
    float,
    "number",
)
_VERCEL_SANDBOX_DEFAULT_CWD = "/vercel/sandbox"
_SUPPORTED_VERCEL_RUNTIMES = ("node24", "node22", "python3.13")


def _is_supported_vercel_runtime(runtime: str) -> bool:
    return not runtime or runtime in _SUPPORTED_VERCEL_RUNTIMES


def _check_vercel_sandbox_requirements(config: dict[str, Any]) -> bool:
    """Validate Vercel Sandbox terminal backend requirements."""
    runtime = (config.get("vercel_runtime") or "").strip()
    if not _is_supported_vercel_runtime(runtime):
        supported = ", ".join(_SUPPORTED_VERCEL_RUNTIMES)
        logger.error(
            "Vercel Sandbox runtime %r is not supported. "
            "Set TERMINAL_VERCEL_RUNTIME to one of: %s.",
            runtime,
            supported,
        )
        return False

    disk = config.get("container_disk", 51200)
    if disk not in {0, 51200}:
        logger.error(
            "Vercel Sandbox does not support custom TERMINAL_CONTAINER_DISK=%s. "
            "Use the default shared setting (51200 MB).",
            disk,
        )
        return False

    if importlib.util.find_spec("vercel") is None:
        logger.error(
            "vercel is required for the Vercel Sandbox terminal backend: pip install vercel"
        )
        return False

    from agent.secret_scope import get_secret

    has_oidc = bool(get_secret("VERCEL_OIDC_TOKEN"))
    has_token = bool(get_secret("VERCEL_TOKEN"))
    has_project = bool(get_secret("VERCEL_PROJECT_ID"))
    has_team = bool(get_secret("VERCEL_TEAM_ID"))

    if has_oidc:
        return True

    if has_token or has_project or has_team:
        if has_token and has_project and has_team:
            return True
        logger.error(
            "Vercel Sandbox backend selected with token auth, but "
            "VERCEL_TOKEN, VERCEL_PROJECT_ID, and VERCEL_TEAM_ID must all "
            "be set together. VERCEL_OIDC_TOKEN is supported for one-off "
            "local development only."
        )
        return False

    logger.error(
        "Vercel Sandbox backend selected but no supported auth configuration "
        "was found. Set VERCEL_TOKEN, VERCEL_PROJECT_ID, and VERCEL_TEAM_ID "
        "for normal use. VERCEL_OIDC_TOKEN is supported for one-off local "
        "development only."
    )
    return False


# Advisory disk-usage check; cached so the recursive scan doesn't run on
# every command (a result up to 5 minutes stale is harmless).
_disk_usage_cache: dict = {"timestamp": 0.0, "result": False}
_DISK_USAGE_CACHE_TTL = 300.0  # seconds


def _check_disk_usage_warning():
    """True when hermes scratch dirs exceed the warning threshold (cached, advisory)."""
    import time as _time_mod
    now = _time_mod.monotonic()
    if now - _disk_usage_cache["timestamp"] < _DISK_USAGE_CACHE_TTL:
        return _disk_usage_cache["result"]
    try:
        scratch_dir = _get_scratch_dir()
        total_bytes = 0
        import glob
        for path in glob.glob(str(scratch_dir / "hermes-*")):
            for f in Path(path).rglob('*'):
                if f.is_file():
                    try:
                        total_bytes += f.stat().st_size
                    except OSError as e:
                        logger.debug("Could not stat file %s: %s", f, e)
        total_gb = total_bytes / (1024 ** 3)
        exceeded = total_gb > DISK_USAGE_WARNING_THRESHOLD_GB
        if exceeded:
            logger.warning("Disk usage (%.1fGB) exceeds threshold (%.0fGB). Consider running cleanup_all_environments().",
                           total_gb, DISK_USAGE_WARNING_THRESHOLD_GB)
        _disk_usage_cache["timestamp"] = _time_mod.monotonic()
        _disk_usage_cache["result"] = exceeded
        return exceeded
    except Exception as e:
        logger.debug("Disk usage warning check failed: %s", e, exc_info=True)
        # Don't update cache on error so the next call retries.
        return False


# Interactive sudo password cache, scoped to the session key when present,
# else callback identity (ACP / CLI), else the current thread — so one
# session can never reuse another's cached password in a long-lived process.
_sudo_password_cache: dict[str, str] = {}
_sudo_password_cache_lock = threading.Lock()

# Approval / sudo-prompt UI callbacks (CLI registers prompt_toolkit-aware
# ones). Thread-local so overlapping ACP sessions, each on its own executor
# thread, can't stomp on each other (GHSA-qg5c-hvr5-hjgr). Gateway mode
# resolves approvals via the per-session queue in tools.approval instead.
_callback_tls = threading.local()


def _get_sudo_password_callback():
    return getattr(_callback_tls, "sudo_password", None)


def _current_session_key() -> str:
    """Active gateway/WebUI session key, or "" outside sessions (ContextVar with
    the ``get_session_env`` os.environ fallback for CLI/cron/tests)."""
    from gateway.session_context import get_session_env

    return get_session_env("HERMES_SESSION_KEY", "")


def _get_approval_callback():
    return getattr(_callback_tls, "approval", None)


def set_sudo_password_callback(cb):
    """Register the CLI's sudo password prompt callback (per-thread slot)."""
    _callback_tls.sudo_password = cb


def set_approval_callback(cb):
    """Register the dangerous-command approval prompt callback (per-thread slot)."""
    _callback_tls.approval = cb


def _get_sudo_password_cache_scope() -> str:
    """Return the cache scope for interactive sudo passwords."""
    session_key = _current_session_key()
    if session_key:
        return f"session:{session_key}"

    callback = _get_sudo_password_callback()
    if callback is not None:
        owner = getattr(callback, "__self__", None)
        func = getattr(callback, "__func__", None)
        if owner is not None and func is not None:
            return f"callback-owner:{id(owner)}:{id(func)}"
        return f"callback:{id(callback)}"

    return f"thread:{threading.get_ident()}"


def _get_cached_sudo_password() -> str:
    """Return the cached sudo password for the current scope."""
    scope = _get_sudo_password_cache_scope()
    with _sudo_password_cache_lock:
        return _sudo_password_cache.get(scope, "")


def _set_cached_sudo_password(password: str) -> None:
    """Persist a sudo password for the current scope."""
    scope = _get_sudo_password_cache_scope()
    with _sudo_password_cache_lock:
        if password:
            _sudo_password_cache[scope] = password
        else:
            _sudo_password_cache.pop(scope, None)


def _reset_cached_sudo_passwords() -> None:
    """Clear all cached sudo passwords (tests / process teardown)."""
    with _sudo_password_cache_lock:
        _sudo_password_cache.clear()

from tools.approval import (
    check_all_command_guards as _check_all_guards_impl,
)


def _docker_volume_uses_host_path(volume_spec: str) -> bool:
    """Return True when a docker volume spec bind-mounts a host path."""
    if not isinstance(volume_spec, str):
        return False

    vol = volume_spec.strip()
    return bool(vol) and (
        vol.startswith(("/", "~", "./", "../")) or
        (len(vol) >= 3 and vol[1] == ":" and vol[2] in ("/", "\\"))
    )


def _docker_has_host_access(config: Dict[str, Any]) -> bool:
    """Return True when a Docker sandbox exposes host paths through bind mounts."""
    if config.get("env_type") != "docker":
        return False
    if config.get("host_cwd") and config.get("docker_mount_cwd_to_workspace"):
        return True
    return any(_docker_volume_uses_host_path(vol) for vol in config.get("docker_volumes", []))


def _check_all_guards(command: str, env_type: str,
                      has_host_access: bool = False) -> dict:
    """Delegate to consolidated guard (tirith + dangerous cmd) with CLI callback."""
    return _check_all_guards_impl(command, env_type,
                                  approval_callback=_get_approval_callback(),
                                  has_host_access=has_host_access)




def _in_delegated_child_context() -> bool:
    """True while running inside a delegate_task child.

    Subagents run on parent-process worker threads and inherit process-wide
    ``HERMES_INTERACTIVE=1``, which does NOT mean this context can reach the
    user: a raw ``/dev/tty`` sudo prompt from a child races the TUI for the
    tty and blocks the child for the full timeout. Children must always be
    headless for sudo prompting. The ContextVar is set by
    ``delegated_child_context()`` and propagates via ``copy_context``.
    """
    try:
        from agent.delegation_context import is_delegated_child_context

        return is_delegated_child_context()
    except Exception:
        return False


def _handle_sudo_failure(output: str, env_type: str) -> str:
    """Append a SUDO_PASSWORD tip when sudo failed in a headless context
    (gateway session or delegate_task child); otherwise return *output* as is."""
    is_gateway = env_var_enabled("HERMES_GATEWAY_SESSION")
    is_delegated_child = _in_delegated_child_context()

    if not is_gateway and not is_delegated_child:
        return output

    sudo_failures = [
        "sudo: a password is required",
        "sudo: no tty present",
        "sudo: a terminal is required",
    ]
    
    for failure in sudo_failures:
        if failure in output:
            from hermes_constants import display_hermes_home as _dhh
            if is_delegated_child:
                return output + (
                    "\n\n💡 Tip: Subagents cannot prompt for a sudo password. "
                    f"Add SUDO_PASSWORD to {_dhh()}/.env on the agent machine, "
                    "or run the command without sudo."
                )
            return output + f"\n\n💡 Tip: To enable sudo over messaging, add SUDO_PASSWORD to {_dhh()}/.env on the agent machine."
    
    return output


# sudo -S rejects a bad cached/interactive password with these messages.
_SUDO_WRONG_PASSWORD_MARKERS = (
    "sudo: authentication failed",
    "sudo: incorrect password attempt",
    "sudo: maximum 3 incorrect authentication attempts",
    "sudo: 3 incorrect password attempts",
)


def _sudo_wrong_password_failure(output: str) -> bool:
    """Return True when sudo rejected a piped password."""
    if not output:
        return False
    lowered = output.lower()
    return any(marker in lowered for marker in _SUDO_WRONG_PASSWORD_MARKERS)


def _invalidate_cached_sudo_on_auth_failure(
    command: str | None, output: str
) -> bool:
    """Drop a session-cached sudo password after sudo rejects it.

    Env-configured ``SUDO_PASSWORD`` is left alone — that is an explicit
    operator choice, not an interactive cache entry.
    """
    if "SUDO_PASSWORD" in os.environ:
        return False
    if not _sudo_wrong_password_failure(output):
        return False
    if _count_real_sudo_invocations(command or "") == 0:
        return False
    if not _get_cached_sudo_password():
        return False
    _set_cached_sudo_password("")
    return True


def _prompt_for_sudo_password(timeout_seconds: int = 45) -> str:
    """Prompt for a sudo password; "" on skip (empty Enter), timeout, or error.

    Prefers the CLI-registered callback (prompt_toolkit-integrated); otherwise
    reads /dev/tty (msvcrt on Windows) with echo disabled. Time spent waiting
    on the human is excluded from tool deadlines via ``human_wait_window``.
    """
    _sudo_cb = _get_sudo_password_callback()
    if _sudo_cb is not None:
        try:
            from tools.approval import human_wait_window
            with human_wait_window():
                return _sudo_cb() or ""
        except Exception:
            return ""

    result = {"password": None, "done": False}
    
    def read_password_thread():
        """Read password with echo disabled. Uses msvcrt on Windows, /dev/tty on Unix."""
        tty_fd = None
        old_attrs = None
        try:
            if platform.system() == "Windows":
                import msvcrt
                chars = []
                while True:
                    c = msvcrt.getwch()
                    if c in {"\r", "\n"}:
                        break
                    if c == "\x03":
                        raise KeyboardInterrupt
                    chars.append(c)
                result["password"] = "".join(chars)
            else:
                import termios
                tty_fd = os.open("/dev/tty", os.O_RDONLY)
                old_attrs = termios.tcgetattr(tty_fd)
                new_attrs = termios.tcgetattr(tty_fd)
                new_attrs[3] = new_attrs[3] & ~termios.ECHO
                termios.tcsetattr(tty_fd, termios.TCSAFLUSH, new_attrs)
                chars = []
                while True:
                    b = os.read(tty_fd, 1)
                    if not b or b in {b"\n", b"\r"}:
                        break
                    chars.append(b)
                result["password"] = b"".join(chars).decode("utf-8", errors="replace")
        except (EOFError, KeyboardInterrupt, OSError):
            result["password"] = ""
        except Exception:
            result["password"] = ""
        finally:
            if tty_fd is not None and old_attrs is not None:
                try:
                    import termios as _termios
                    _termios.tcsetattr(tty_fd, _termios.TCSAFLUSH, old_attrs)
                except Exception as e:
                    logger.debug("Failed to restore terminal attributes: %s", e)
            if tty_fd is not None:
                try:
                    os.close(tty_fd)
                except Exception as e:
                    logger.debug("Failed to close tty fd: %s", e)
            result["done"] = True
    
    try:
        os.environ["HERMES_SPINNER_PAUSE"] = "1"
        time.sleep(0.2)
        
        print()
        print("┌" + "─" * 58 + "┐")
        print("│  🔐 SUDO PASSWORD REQUIRED" + " " * 30 + "│")
        print("├" + "─" * 58 + "┤")
        print("│  Enter password below (input is hidden), or:            │")
        print("│    • Press Enter to skip (command fails gracefully)     │")
        print(f"│    • Wait {timeout_seconds}s to auto-skip" + " " * 27 + "│")
        print("└" + "─" * 58 + "┘")
        print()
        print("  Password (hidden): ", end="", flush=True)
        
        password_thread = threading.Thread(target=read_password_thread, daemon=True)
        password_thread.start()
        from tools.approval import human_wait_window
        with human_wait_window():
            password_thread.join(timeout=timeout_seconds)
        
        if result["done"]:
            password = result["password"] or ""
            print()  # newline after hidden input
            if password:
                print("  ✓ Password received (cached for this session)")
            else:
                print("  ⏭ Skipped - continuing without sudo")
            print()
            sys.stdout.flush()
            return password
        else:
            print("\n  ⏱ Timeout - continuing without sudo")
            print("    (Press Enter to dismiss)")
            print()
            sys.stdout.flush()
            return ""
            
    except (EOFError, KeyboardInterrupt):
        print()
        print("  ⏭ Cancelled - continuing without sudo")
        print()
        sys.stdout.flush()
        return ""
    except Exception as e:
        print(f"\n  [sudo prompt error: {e}] - continuing without sudo\n")
        sys.stdout.flush()
        return ""
    finally:
        if "HERMES_SPINNER_PAUSE" in os.environ:
            del os.environ["HERMES_SPINNER_PAUSE"]


def _looks_like_env_assignment(token: str) -> bool:
    """Return True when *token* is a leading shell environment assignment."""
    if "=" not in token or token.startswith("="):
        return False
    name, _value = token.split("=", 1)
    return bool(re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name))


def _read_shell_token(command: str, start: int) -> tuple[str, int]:
    """Read one shell token, preserving quotes/escapes, starting at *start*."""
    i = start
    n = len(command)

    while i < n:
        ch = command[i]
        if ch.isspace() or ch in ";|&()":
            break
        if ch == "'":
            i += 1
            while i < n and command[i] != "'":
                i += 1
            if i < n:
                i += 1
            continue
        if ch == '"':
            i += 1
            while i < n:
                inner = command[i]
                if inner == "\\" and i + 1 < n:
                    i += 2
                    continue
                if inner == '"':
                    i += 1
                    break
                i += 1
            continue
        if ch == "\\" and i + 1 < n:
            i += 2
            continue
        i += 1

    return command[start:i], i


def _rewrite_real_sudo_invocations(command: str) -> tuple[str, int]:
    """Rewrite only real unquoted sudo command words, not plain text mentions.

    Walks the command with the shared tokenizer, tracking whether the cursor
    sits at a command-start position (after newline, ``;``/``|``/``&``/``(``,
    ``&&``/``||``/``;;``, or a leading ``VAR=val`` assignment); comments at
    command start are copied through verbatim. Returns the rewritten command
    and the number of sudo invocations rewritten.
    """
    out: list[str] = []
    i = 0
    n = len(command)
    command_start = True
    sudo_count = 0

    while i < n:
        ch = command[i]

        if ch.isspace():
            out.append(ch)
            if ch == "\n":
                command_start = True
            i += 1
            continue

        if ch == "#" and command_start:
            comment_end = command.find("\n", i)
            if comment_end == -1:
                out.append(command[i:])
                break
            out.append(command[i:comment_end])
            i = comment_end
            continue

        if command.startswith(("&&", "||", ";;"), i):
            out.append(command[i:i + 2])
            i += 2
            command_start = True
            continue

        if ch in ";|&(":
            out.append(ch)
            i += 1
            command_start = True
            continue

        if ch == ")":
            out.append(ch)
            i += 1
            command_start = False
            continue

        token, next_i = _read_shell_token(command, i)
        if command_start and token == "sudo":
            out.append("sudo -S -p ''")
            sudo_count += 1
        else:
            out.append(token)

        command_start = bool(command_start and _looks_like_env_assignment(token))
        i = next_i

    return "".join(out), sudo_count


def _count_real_sudo_invocations(command: str) -> int:
    """Return how many real sudo command words appear in *command*."""
    return _rewrite_real_sudo_invocations(command)[1]


def _sudo_nopasswd_works() -> bool:
    """True when local sudo currently works without prompting.

    Local backend only — Docker/SSH/Modal must not inherit host sudo state.
    Re-probes every call (no cache) so an expired sudo timestamp can't make a
    later command silently block waiting for a password.
    """
    terminal_env = _tenv("TERMINAL_ENV", "local").strip().lower() or "local"
    if terminal_env != "local":
        return False

    try:
        probe = subprocess.run(
            ["sudo", "-n", "true"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=3,
            check=False,
        )
        return probe.returncode == 0
    except Exception:
        return False


def _rewrite_compound_background(command: str) -> str:
    """Wrap `A && B &` (or `A || B &`) to `A && { B & }` at depth 0.

    Bash binds `&&` tighter than `&`, so `A && B &` backgrounds a subshell
    that runs B in the foreground and waits for it; a long-running B leaves
    that subshell stuck in ``wait4`` forever, and its open stdout pipe can
    keep the terminal tool from returning. The brace group keeps `&&`'s
    skip-B-on-failure semantics without a fork: bash backgrounds B as a
    simple command and exits immediately, orphaning B normally.

    Handles redirects (``&>``, ``2>&1``) and skips quoted strings and
    parenthesised subshells. Simple ``cmd &`` is left alone — it doesn't
    have the subshell-wait bug.
    """
    n = len(command)
    i = 0
    paren_depth = 0
    brace_depth = 0
    # Position in *command* just after the most recent `&&` / `||` at depth 0
    # in the current statement; -1 when no chain operator is active.
    last_chain_op_end = -1
    rewrites: list[tuple[int, int]] = []  # (chain_op_end, amp_pos)

    while i < n:
        ch = command[i]

        # Newline terminates a statement at depth 0 — reset chain state.
        # Checked before the whitespace skip so we don't miss it.
        if ch == "\n" and paren_depth == 0 and brace_depth == 0:
            last_chain_op_end = -1
            i += 1
            continue

        if ch.isspace():
            i += 1
            continue

        # Comments (only at statement start — conservative: any `#` not inside
        # a token ends the line). `_read_shell_token` handles quoted strings
        # below so `#` inside quotes is safe.
        if ch == "#":
            nl = command.find("\n", i)
            if nl == -1:
                break
            i = nl
            continue

        if ch == "\\" and i + 1 < n:
            i += 2
            continue

        # Quoted tokens — consume whole string via the shared tokenizer.
        if ch in {"'", '"'}:
            _, next_i = _read_shell_token(command, i)
            i = max(next_i, i + 1)
            continue

        if ch == "(":
            paren_depth += 1
            i += 1
            continue

        if ch == ")":
            paren_depth = max(0, paren_depth - 1)
            i += 1
            continue

        # Brace groups: `{ ... }` is a group (no subshell fork), and bash
        # requires whitespace after `{`. We track depth so already-rewritten
        # output (`A && { B & }`) is idempotent — the inner `&` is part of
        # the group, not a new compound to rewrite. Also skip content inside
        # the group since `A && B &` there is separately well-formed.
        if ch == "{" and i + 1 < n and (command[i + 1].isspace() or command[i + 1] == "\n"):
            brace_depth += 1
            i += 1
            continue
        if ch == "}" and brace_depth > 0:
            brace_depth -= 1
            # Closing a group completes a compound statement; reset chain.
            last_chain_op_end = -1
            i += 1
            continue

        # Inside parens or brace groups, skip operators — they parse in their
        # own scope. `(...)` subshells have the same bug class but are not the
        # common agent pattern; leave for a follow-up.
        if paren_depth > 0 or brace_depth > 0:
            i += 1
            continue

        # Chain operators at depth 0
        if command.startswith("&&", i) or command.startswith("||", i):
            last_chain_op_end = i + 2
            i += 2
            continue

        # Statement terminators reset the chain state
        if ch == ";":
            last_chain_op_end = -1
            i += 1
            continue

        # Single `|` (pipe) starts a new pipeline stage; don't rewrite
        # across it. `||` handled above.
        if ch == "|":
            last_chain_op_end = -1
            i += 1
            continue

        # `&` handling: distinguish `&&`, `&>`, fd redirect (`>&`, `<&`),
        # and a true backgrounding `&`.
        if ch == "&":
            # `&&` handled above; won't reach here
            if i + 1 < n and command[i + 1] == ">":
                # `&>` redirect — consume
                i += 2
                continue
            # `>&` / `<&` fd target — look back past whitespace
            j = i - 1
            while j >= 0 and command[j].isspace():
                j -= 1
            if j >= 0 and command[j] in "<>":
                i += 1
                continue
            # Real background operator
            if last_chain_op_end >= 0:
                rewrites.append((last_chain_op_end, i))
            last_chain_op_end = -1
            i += 1
            continue

        # Regular unquoted token — advance past it via the shared tokenizer
        _, next_i = _read_shell_token(command, i)
        i = max(next_i, i + 1)

    if not rewrites:
        return command

    # Apply rewrites back-to-front so earlier indices remain valid.
    result = command
    for chain_end, amp_pos in reversed(rewrites):
        # Skip whitespace right after the `&&`/`||` so the brace group
        # opens flush against the inner command.
        insert_pos = chain_end
        while insert_pos < amp_pos and result[insert_pos].isspace():
            insert_pos += 1
        prefix = result[:insert_pos]
        middle = result[insert_pos:amp_pos]  # inner command + trailing space
        suffix = result[amp_pos + 1 :]
        # `{` needs a trailing space in bash; the closing `}` needs to be
        # preceded by `;` or `&` — we're providing `&` from the backgrounding.
        result = prefix + "{ " + middle + "& }" + suffix

    return result


def _transform_sudo_command(command: str | None) -> tuple[str | None, str | None]:
    """Rewrite bare ``sudo`` to ``sudo -S -p ''`` when a password is available.

    Shared by every execution environment. Returns ``(command, sudo_stdin)``:
    ``sudo_stdin`` is one password line per sudo invocation that the caller
    must PREPEND to the process stdin (sudo -S consumes exactly one line and
    passes the rest through, so it's safe alongside the caller's own
    stdin_data). Backends that can't pipe stdin (modal, daytona,
    vercel_sandbox) embed the password in the command string themselves.
    With no password available the command is returned unchanged and
    ``sudo_stdin`` is None, so it fails gracefully with
    "sudo: a password is required". Password sources, in order: configured
    SUDO_PASSWORD, the session cache, then an interactive prompt (45s
    timeout, cached on success) when a UI is reachable.
    """
    if command is None:
        return None, None
    transformed, sudo_count = _rewrite_real_sudo_invocations(command)
    if sudo_count == 0:
        return command, None

    # Scope-aware read: under multiplex the process env may hold another
    # profile's SUDO_PASSWORD; unscoped callers keep the os.environ read.
    try:
        from agent.secret_scope import UnscopedSecretError, get_secret

        try:
            _configured_password = get_secret("SUDO_PASSWORD")
        except UnscopedSecretError:
            _configured_password = os.environ.get("SUDO_PASSWORD")
    except Exception:
        _configured_password = os.environ.get("SUDO_PASSWORD")
    has_configured_password = _configured_password is not None
    sudo_password = (
        _configured_password
        if has_configured_password
        else _get_cached_sudo_password()
    )

    # sudoers NOPASSWD hosts must not be forced through the prompt or the
    # -S password pipe (local backend only; re-probed every call).
    if not has_configured_password and not sudo_password and _sudo_nopasswd_works():
        return command, None

    has_sudo_prompt_callback = _get_sudo_password_callback() is not None
    # delegate_task children inherit HERMES_INTERACTIVE=1 (and possibly a
    # stale thread-local callback on a recycled worker) but have no user on
    # the other side — they always behave as headless; configured password,
    # session cache and the NOPASSWD probe still apply.
    should_prompt_for_sudo = (
        env_var_enabled("HERMES_INTERACTIVE") or has_sudo_prompt_callback
    ) and not _in_delegated_child_context()
    if not has_configured_password and not sudo_password and should_prompt_for_sudo:
        sudo_password = _prompt_for_sudo_password(timeout_seconds=45)
        if sudo_password:
            _set_cached_sudo_password(sudo_password)

    if has_configured_password or sudo_password:
        # Trailing newline is required: sudo -S reads one line per invocation.
        # Compound commands (`sudo a && sudo b`) need one password line each.
        password_line = sudo_password + "\n"
        return transformed, password_line * sudo_count

    return command, None


from tools.environments.base import EnvironmentConnectionError
from tools.environments.local import LocalEnvironment as _LocalEnvironment
from tools.environments.singularity import SingularityEnvironment as _SingularityEnvironment
from tools.environments.ssh import SSHEnvironment as _SSHEnvironment
from tools.environments.docker import DockerEnvironment as _DockerEnvironment
from tools.environments.modal import ModalEnvironment as _ModalEnvironment
from tools.environments.managed_modal import ManagedModalEnvironment as _ManagedModalEnvironment
from tools.managed_tool_gateway import is_managed_tool_gateway_ready


# Tool description for LLM
TERMINAL_TOOL_DESCRIPTION = """Execute shell commands. The host OS, shell, and terminal backend are stated in your environment section — write commands for THAT platform. Filesystem, current working directory, and exported environment variables persist between calls.

Do NOT use cat/head/tail (use read_file), grep/rg/find/ls (use search_files), sed/awk (use patch), or echo/heredoc file creation (use write_file). Reserve terminal for: builds, installs, git, processes, scripts, network, package managers — anything that needs a shell. Output is auto-truncated with the full text saved to a file — never pipe through tail/head to shorten it.
Environment state persists: activate a virtualenv or export variables once per session, not before every command.

Foreground (default): returns INSTANTLY when the command finishes, even with a high timeout — set timeout generously for long builds.
Background: set background=true (returns a session_id); add notify=true for bounded tasks, leave silent only for servers/daemons that never exit. After starting a server, verify readiness with a health check in a separate call (no blind sleep loops); manage with process(action="poll"/"wait").
Working directory: use 'workdir' for per-command cwd; when a command changes the session cwd (cd, pushd), trust the result's "cwd" field instead of prefixing every command with 'cd'.
PTY: pty=true + background=true for interactive CLIs (they hang without a terminal); drive them with process(action="write"/"submit"). Local backend only.
"""

# Environment lifecycle state.
_active_environments: Dict[str, Any] = {}
_last_activity: Dict[str, float] = {}
_env_lock = threading.Lock()
_creation_locks: Dict[str, threading.Lock] = {}  # Per-task locks for sandbox creation
_creation_locks_lock = threading.Lock()  # Protects _creation_locks dict itself
_cleanup_thread = None
_cleanup_running = False

# Once-per-process guard for the docker orphan reaper.
_docker_orphan_reaper_ran = False
_docker_orphan_reaper_lock = threading.Lock()


def _maybe_reap_docker_orphans(container_config: Dict[str, Any]) -> None:
    """Run the docker orphan reaper once per process, if enabled.

    Sweeps Exited containers labeled ``hermes-agent=1`` for the current
    profile — leftovers of Hermes processes that died without firing
    ``atexit`` (SIGKILL, OOM, closed terminal). Conservative: only containers
    older than ``2 × lifetime_seconds``, profile-scoped. Gates:
    ``terminal.docker_orphan_reaper: false`` (operator opt-out, e.g. several
    Hermes processes sharing a profile) and the once-per-interpreter flag so
    parallel subagent / RL-rollout calls don't re-sweep.
    """
    global _docker_orphan_reaper_ran
    if not container_config.get("docker_orphan_reaper", True):
        return
    if _docker_orphan_reaper_ran:  # double-checked locking
        return
    with _docker_orphan_reaper_lock:
        if _docker_orphan_reaper_ran:
            return
        _docker_orphan_reaper_ran = True

    # 2 × lifetime gives sibling processes a grace window; floor at 60s so
    # TERMINAL_LIFETIME_SECONDS=0 can't instant-reap a sibling's own setup.
    # container_config only carries container_* keys, so read the env var.
    try:
        lifetime = int(_tenv("TERMINAL_LIFETIME_SECONDS", "300"))
    except (TypeError, ValueError):
        lifetime = 300
    lifetime = max(60, lifetime)
    max_age = lifetime * 2

    try:
        from tools.environments.docker import reap_orphan_containers, _container_identity
    except ImportError:
        return
    try:
        profile = _container_identity(container_config.get("docker_shared_container_key", ""))
        removed = reap_orphan_containers(
            max_age_seconds=max_age, profile_filter=profile,
        )
        if removed:
            logger.info(
                "Docker orphan reaper removed %d stale container(s) for profile %s",
                removed, profile,
            )
    except Exception as e:
        # Never fail the env-creation path because of a janitor problem.
        logger.debug("Docker orphan reaper raised: %s", e)


# Per-task environment overrides (never exposed to the model). RL/benchmark
# envs and ACP register a custom image / cwd for a task_id BEFORE the agent
# loop; sandbox creation consults this first, then the TERMINAL_* env vars.
_task_env_overrides: Dict[str, Dict[str, Any]] = {}

# Per-session cwd records: the durable source of truth for "which directory
# is THIS session in". Keyed by the raw session/task key, NOT the collapsed
# container id — the env is shared across sessions, so cwd state stored on
# it is a global mutable timeshared between sessions (the wrong-worktree bug
# class). Written after every completed command and on cwd-override
# registration; readers resolve against it before any env-side cwd.
_session_cwd: Dict[str, str] = {}
_session_cwd_lock = threading.Lock()


def record_session_cwd(session_key: Optional[str], cwd: Optional[str]) -> None:
    """Record *cwd* as *session_key*'s working directory (after a completed
    command, or on workspace-override registration). None/empty keys collapse
    to ``"default"``; non-string / empty cwds are ignored."""
    if not isinstance(cwd, str) or not cwd.strip():
        return
    key = str(session_key or "default")
    with _session_cwd_lock:
        if _session_cwd.get(key) != cwd:
            _session_cwd[key] = cwd


def get_session_cwd(session_key: Optional[str]) -> Optional[str]:
    """Recorded cwd for *session_key*, or None. No fallback chain on purpose:
    callers decide what an absent record means. None/empty keys read ``"default"``."""
    key = str(session_key or "default")
    with _session_cwd_lock:
        return _session_cwd.get(key)


def clear_session_cwd(session_key: str) -> None:
    """Drop a session's cwd record (session teardown)."""
    with _session_cwd_lock:
        _session_cwd.pop(session_key, None)


def register_task_env_overrides(task_id: str, overrides: Dict[str, Any]):
    """Register per-task sandbox overrides (``docker_image``/``modal_image``/
    ``singularity_image``/``daytona_image``, ``env_type``, ``cwd``) before the
    agent loop runs.

    A ``cwd`` override takes effect immediately: it becomes the session's
    recorded cwd (until a ``cd`` changes it) and any live env's cwd is updated
    too, so env-side seeding stays consistent (ACP switching project root
    mid-session via ``session/load``).
    """
    _task_env_overrides[task_id] = overrides

    new_cwd = overrides.get("cwd")
    if isinstance(new_cwd, str) and new_cwd.strip():
        record_session_cwd(task_id, new_cwd)
        # Live env may be cached under the raw task_id (per-session surfaces)
        # or the collapsed container id (isolation-keyed rollouts); try both so
        # a CWD-only override (which collapses to "default") still finds it.
        container_id = _resolve_container_task_id(task_id)
        with _env_lock:
            env = _active_environments.get(task_id) or _active_environments.get(container_id)
        if env is not None and getattr(env, "cwd", None) is not None:
            env.cwd = new_cwd


def clear_task_env_overrides(task_id: str):
    """Drop a task's overrides, cwd record and container alias (rollout cleanup)."""
    _task_env_overrides.pop(task_id, None)
    clear_session_cwd(task_id)
    with _container_alias_lock:
        _container_aliases.pop(task_id, None)


# Subagent → parent container aliasing. delegate_task children have their own
# task_id but must share the PARENT's container; under per-session isolation
# the collapse-to-"default" shortcut no longer provides that, so the spawn
# site registers an explicit alias.
_container_aliases: Dict[str, str] = {}
_container_alias_lock = threading.Lock()


def register_container_alias(child_task_id: str, parent_task_id: Optional[str]) -> None:
    """Make *child_task_id* resolve to *parent_task_id*'s container (called at
    delegate_task spawn). A missing parent id aliases to ``"default"``."""
    if not child_task_id:
        return
    with _container_alias_lock:
        _container_aliases[child_task_id] = str(parent_task_id or "default")


def _resolve_container_alias(task_id: str) -> str:
    """Follow the child→parent alias chain (cycle-safe) for *task_id*."""
    seen = set()
    key = task_id
    with _container_alias_lock:
        while key in _container_aliases and key not in seen:
            seen.add(key)
            key = _container_aliases[key]
    return key


def _session_isolation_enabled() -> bool:
    """True when non-persistent sandboxes get per-session identities.

    ``container_persistent: false`` means state must not survive or be shared
    across sessions, so one shared sandbox contradicts it. Applies to docker
    and to plugin backends declaring ``session_isolated_when_nonpersistent``
    (e.g. sandboxes resumed by name, where a shared deterministic name would
    let two ephemeral runs attach one VM and delete it under each other).
    """
    _ensure_terminal_env_bridged()
    env_type = _tenv("TERMINAL_ENV", "local")
    if env_type != "docker" and not _plugin_env_flag(
        env_type, "session_isolated_when_nonpersistent"
    ):
        return False
    return not _tenv_bool("TERMINAL_CONTAINER_PERSISTENT", "true")


def _docker_session_isolation_enabled() -> bool:
    """Docker-only view of :func:`_session_isolation_enabled` — the workspace
    mount and session-scoped teardown paths must not fire for other backends."""
    if _tenv("TERMINAL_ENV", "local") != "docker":
        return False
    return _session_isolation_enabled()


def _docker_persistent_profile_scoped() -> bool:
    """True when the persistent Docker container is shared per PROFILE.

    Contract for docker + ``container_persistent: true``: ONE long-lived
    container per profile, shared by every session (CLI, gateway, WebUI).
    The session-key fallback in :func:`_resolve_container_task_id` exists to
    stop cross-profile SSH reuse; ungated it fragmented persistent Docker
    into one container per gateway session, so this predicate restores
    profile scoping for exactly this backend/mode.
    """
    _ensure_terminal_env_bridged()
    if _tenv("TERMINAL_ENV", "local") != "docker":
        return False
    return _tenv_bool("TERMINAL_CONTAINER_PERSISTENT", "true")


def _current_session_profile() -> str:
    """Active session's Hermes profile name, or "" (same lookup discipline as
    :func:`_current_session_key`)."""
    from gateway.session_context import get_session_env

    return get_session_env("HERMES_SESSION_PROFILE", "")


_ISOLATION_OVERRIDE_KEYS = frozenset({
    "docker_image", "modal_image", "singularity_image",
    "daytona_image", "env_type",
})


def _has_isolation_overrides(task_id: Optional[str]) -> bool:
    """True when *task_id* registered image/env_type overrides — the single
    "isolated RL/benchmark rollout" predicate shared by key resolution and
    container creation so the two can't drift."""
    if not task_id or task_id not in _task_env_overrides:
        return False
    return bool(set(_task_env_overrides[task_id].keys()) & _ISOLATION_OVERRIDE_KEYS)


def _resolve_container_task_id(task_id: Optional[str]) -> str:
    """Map a tool-call ``task_id`` to the ``_active_environments`` key.

    Order matters — earlier branches are authoritative where they apply:

    1. Task ids with image/``env_type`` overrides (RL/benchmark rollouts) key
       their own sandbox. CWD-only overrides (ACP workspace tracking) are NOT
       isolation signals.
    2. Per-session isolation (docker + ``container_persistent: false``): each
       session's task_id is its own key, so a fresh chat gets a fresh sandbox
       with only ITS mounts; delegate_task children follow the alias registry
       to the parent's container.
    3. With a session key present (WebUI per-session, gateway per-message):
       persistent Docker is PROFILE-scoped (``shared:<key>`` opt-in, else
       ``profile:<name>``, with the default profile staying literally
       ``"default"`` so CLI and default-profile gateway sessions share ONE
       container); other backends key ``session:<key>`` so switching profiles
       can't reuse another profile's SSHEnvironment on the wrong host.
    4. No session key (CLI): ``shared:<key>`` when opted in — or a CLI run of
       a keyed profile would split from its gateway sessions — else
       ``"default"``, which subagent ids collapse onto so they share the
       parent's long-lived container.
    """
    if task_id and _has_isolation_overrides(task_id):
        return task_id
    if task_id and _session_isolation_enabled():
        return _resolve_container_alias(task_id)
    session_key = _current_session_key()
    if session_key:
        if _docker_persistent_profile_scoped():
            shared = _tenv("TERMINAL_DOCKER_SHARED_CONTAINER_KEY", "").strip()
            if shared:
                return f"shared:{shared}"
            profile = _current_session_profile() or "default"
            if profile == "default":
                return "default"
            return f"profile:{profile}"
        return f"session:{session_key}"
    if _docker_persistent_profile_scoped():
        shared = _tenv("TERMINAL_DOCKER_SHARED_CONTAINER_KEY", "").strip()
        if shared:
            return f"shared:{shared}"
    return "default"


def resolve_task_overrides(task_id: Optional[str]) -> Dict[str, Any]:
    """Return the env overrides for *task_id*, raw key first then collapsed.

    ``register_task_env_overrides`` writes under the *raw* task/session id, but
    a CWD-only override collapses (:func:`_resolve_container_task_id`) to the
    shared ``"default"`` container. Callers must therefore read the raw id
    FIRST and only fall back to the collapsed container id, or the originating
    session's override is silently dropped. Single source of that lookup so
    the terminal and file layers can't drift apart.
    """
    raw = task_id or "default"
    return (
        _task_env_overrides.get(raw)
        or _task_env_overrides.get(_resolve_container_task_id(raw))
        or {}
    )


# Backends that take an image, keyed to the override/config key carrying it.
_IMAGE_KEY_BY_BACKEND = {
    "docker": "docker_image",
    "singularity": "singularity_image",
    "modal": "modal_image",
    "daytona": "daytona_image",
}


def _select_image(env_type: str, overrides: Dict[str, Any], config: Dict[str, Any]) -> str:
    """Image for *env_type*: per-task override first, then config; "" for imageless backends."""
    key = _IMAGE_KEY_BY_BACKEND.get(env_type)
    if key is None:
        return ""
    return overrides.get(key) or config[key]


def _lookup_active_env(effective_task_id: str, task_id: Optional[str]):
    """Return the cached env for the collapsed id, else for the raw task_id, else None.

    Caller holds ``_env_lock``. Per-session surfaces (ACP/gateway/dashboard)
    with a CWD-only override collapse to ``"default"`` for container sharing,
    yet an env may already be cached under the originating task_id; honor it
    instead of spawning a duplicate. Refreshes ``_last_activity`` on a hit.
    """
    if effective_task_id in _active_environments:
        key = effective_task_id
    elif task_id and task_id in _active_environments:
        key = task_id
    else:
        return None
    _last_activity[key] = time.time()
    return _active_environments[key]


def _resolve_task_host_cwd(config: Dict[str, Any], task_id: Optional[str]) -> Optional[str]:
    """Host directory to bind-mount at ``/workspace`` for *task_id*'s container.

    Single owner of the cwd-mount policy for every creation site. Shared-
    container mode: the ``TERMINAL_CWD``-derived ``config["host_cwd"]``.
    Per-session isolation (docker + ``container_persistent: false``): only
    the SESSION's own registered workspace may mount — the process env var is
    a launch artifact that outlives the session that set it, so deriving a
    fresh session's mount from it would leak the previous session's directory.
    Overrides tagged ``cwd_source: "process"`` are refused for the same reason;
    ``cwd_source: "session"`` or untagged (ACP/RL) overrides mount.
    """
    if config.get("env_type") != "docker":
        return None
    if not config.get("docker_mount_cwd_to_workspace"):
        return None
    if not _docker_session_isolation_enabled():
        return config.get("host_cwd")
    if _resolve_container_task_id(task_id) == "default":
        # Top-level CLI parent — single-session process, legacy behavior.
        return config.get("host_cwd")
    overrides = resolve_task_overrides(task_id)
    if overrides.get("cwd_source") == "process":
        return None
    candidate = overrides.get("cwd")
    if not isinstance(candidate, str) or not candidate.strip():
        return None
    candidate = os.path.abspath(os.path.expanduser(candidate))
    if not os.path.isdir(candidate):
        return None
    if candidate.startswith(("/workspace", "/root")):
        # Already an in-container path, not a host workspace.
        return None
    return candidate


def _parse_env_var(name: str, default: str, converter: Any = int, type_label: str = "integer"):
    """Parse an env var with *converter*, raising a clear ValueError on bad
    values (e.g. TERMINAL_TIMEOUT=5m) instead of an opaque crash. TERMINAL_*
    names are read scope-aware via :func:`_tenv`."""
    raw = os.getenv(name, default)
    if name.startswith("TERMINAL_"):
        raw = _tenv(name, default)
    try:
        return converter(raw)
    except (ValueError, json.JSONDecodeError):
        raise ValueError(
            f"Invalid value for {name}: {raw!r} (expected {type_label}). "
            f"Check ~/.hermes/.env or environment variables."
        )


def _safe_getcwd() -> str:
    """``os.getcwd()`` tolerant of a deleted cwd (FileNotFoundError) or a macOS
    TCC-protected one without Full Disk Access (PermissionError); falls back
    to TERMINAL_CWD, then the home directory."""
    try:
        return os.getcwd()
    except (FileNotFoundError, PermissionError):
        return _tenv("TERMINAL_CWD") or os.path.expanduser("~")


# Host-cwd prefixes that cannot exist inside a container sandbox (POSIX user
# dirs and Windows drive paths as they leak toward a Linux ``-w`` flag).
_HOST_CWD_PREFIXES = ("/Users/", "/home/", "C:\\", "C:/")

_CONTAINER_BACKENDS = frozenset({"docker", "singularity", "modal", "daytona", "vercel_sandbox"})


def _plugin_env_flag(env_type: str, attr: str, default=False):
    """Classification flag of a plugin-registered backend. Fail-soft: *default*
    when the registry is unavailable, the backend unknown, or the provider
    raises — a misbehaving plugin must never take the terminal tool down."""
    if not env_type or env_type in _CONTAINER_BACKENDS or env_type in {"local", "ssh", "managed_modal"}:
        return default
    try:
        from agent.terminal_env_registry import provider_flag

        return provider_flag(env_type, attr, default)
    except Exception:
        return default


def _is_container_backend(env_type: str) -> bool:
    """True for built-in container backends and plugins declaring ``is_container``."""
    return env_type in _CONTAINER_BACKENDS or _plugin_env_flag(env_type, "is_container")


def _get_plugin_env_provider(env_type: str):
    """Return the registered plugin provider for *env_type*, or None."""
    if not env_type or env_type in _CONTAINER_BACKENDS or env_type in {"local", "ssh", "managed_modal"}:
        return None
    try:
        from agent.terminal_env_registry import get_provider

        return get_provider(env_type)
    except Exception:
        return None


def _is_unusable_container_cwd(cwd: str) -> bool:
    """True if *cwd* is a host or relative path that can't be a container
    workdir: ``docker run -w`` needs an absolute in-sandbox path, otherwise the
    container fails to start (exit 125). Windows drive paths aren't ``isabs``
    on POSIX, so they're caught by the prefix check."""
    if not cwd:
        return False
    return cwd.startswith(_HOST_CWD_PREFIXES) or not os.path.isabs(cwd)


def _tenv(name: str, default: str = "") -> str:
    """Scope-aware read of a ``TERMINAL_*`` variable. Every terminal setting
    must go through this: under gateway multiplexing the active profile's
    config arrives via a per-turn scope, and a raw ``os.getenv`` would read
    whatever a previous turn pinned into the process env (cross-profile leak)."""
    from tools.terminal_scope import terminal_env

    return terminal_env(name, default)


def _tenv_bool(name: str, default: str) -> bool:
    """Scope-aware boolean ``TERMINAL_*`` read: true/1/yes (case-insensitive)."""
    return _tenv(name, default).lower() in {"true", "1", "yes"}


# One-shot guard for the config-fallback bridge: after the first attempt
# either TERMINAL_ENV is set or the import failed, so retrying is wasted work.
_terminal_config_bridge_attempted = False


def _ensure_terminal_env_bridged() -> None:
    """Backfill TERMINAL_* env vars from config.yaml when no launcher did.

    The CLI, gateway and TUI/dashboard PTY launches bridge ``terminal.*`` into
    env vars at startup; processes that skip those paths (``hermes serve``,
    Desktop in-process agents, desktop cron ticker, ACP) would otherwise fall
    back to the local backend even when config selects docker, running
    commands on the host the user meant to sandbox.

    Explicit keys in config.yaml's ``terminal`` section override matching
    env values (which may be stale from ``hermes setup``); env values for
    omitted keys are preserved. Without a terminal section, an existing
    TERMINAL_ENV selection is kept and defaults are backfilled only when none
    is set. A per-turn terminal scope suppresses the bridge entirely: writing
    the scope's values into the process-global env would re-create the
    first-writer-wins cross-profile leak the scope exists to fix.
    """
    from tools.terminal_scope import get_terminal_scope

    if get_terminal_scope() is not None:
        return
    global _terminal_config_bridge_attempted
    if _terminal_config_bridge_attempted:
        return
    _terminal_config_bridge_attempted = True
    try:
        from hermes_cli.config import apply_terminal_config_to_env, read_raw_config

        raw_config = read_raw_config()
        if isinstance(raw_config.get("terminal"), dict):
            apply_terminal_config_to_env(env=None, override=True)
        elif "TERMINAL_ENV" not in os.environ:
            apply_terminal_config_to_env(env=None, override=False)
    except Exception:
        # Never let a config problem take the terminal tool down.
        logger.debug("terminal config → env fallback bridge failed", exc_info=True)


def _get_env_config() -> Dict[str, Any]:
    """Resolve the terminal configuration dict from TERMINAL_* env vars."""
    default_image = "nikolaik/python-nodejs:python3.11-nodejs20"
    _ensure_terminal_env_bridged()
    env_type = _tenv("TERMINAL_ENV", "local")

    mount_docker_cwd = _tenv_bool("TERMINAL_DOCKER_MOUNT_CWD_TO_WORKSPACE", "false")
    container_backend = _is_container_backend(env_type)
    docker_backend = env_type == "docker"

    # Container/docker-only payloads are parsed only when such a backend is
    # selected: a stale or invalid Docker value bridged from config.yaml must
    # not make local terminal/execute_code unusable.
    if container_backend:
        container_cpu = _parse_env_var("TERMINAL_CONTAINER_CPU", "1", float, "number")
        container_memory = _parse_env_var("TERMINAL_CONTAINER_MEMORY", "5120")
        container_disk = _parse_env_var("TERMINAL_CONTAINER_DISK", "51200")
    else:
        container_cpu = 1.0
        container_memory = 5120
        container_disk = 51200

    if docker_backend:
        docker_forward_env = _parse_env_var("TERMINAL_DOCKER_FORWARD_ENV", "[]", json.loads, "valid JSON")
        docker_volumes = _parse_env_var("TERMINAL_DOCKER_VOLUMES", "[]", json.loads, "valid JSON")
        docker_env = _parse_env_var("TERMINAL_DOCKER_ENV", "{}", json.loads, "valid JSON")
        docker_extra_args = _parse_env_var("TERMINAL_DOCKER_EXTRA_ARGS", "[]", json.loads, "valid JSON")
        docker_shm_size = _tenv("TERMINAL_DOCKER_SHM_SIZE", "1g")
    else:
        docker_forward_env = []
        docker_volumes = []
        docker_env = {}
        docker_extra_args = []
        docker_shm_size = "1g"

    if env_type == "local":
        default_cwd = _safe_getcwd()
    elif env_type == "ssh":
        default_cwd = "~"
    elif env_type == "vercel_sandbox":
        default_cwd = _VERCEL_SANDBOX_DEFAULT_CWD
    else:
        default_cwd = "/root"

    # TERMINAL_CWD, sanity-checked for container backends: with Docker cwd
    # passthrough the host path is remapped to /workspace and tracked as
    # host_cwd; otherwise host paths are discarded.
    cwd = _tenv("TERMINAL_CWD", default_cwd)
    from hermes_cli.config import _is_ssh_remote_tilde_cwd
    if cwd and not _is_ssh_remote_tilde_cwd(env_type, cwd):
        cwd = os.path.expanduser(cwd)
    host_cwd = None
    if env_type == "docker" and mount_docker_cwd:
        docker_cwd_source = _tenv("TERMINAL_CWD") or _safe_getcwd()
        candidate = os.path.abspath(os.path.expanduser(docker_cwd_source))
        if (
            any(candidate.startswith(p) for p in _HOST_CWD_PREFIXES)
            or (os.path.isabs(candidate) and os.path.isdir(candidate) and not candidate.startswith(("/workspace", "/root")))
        ):
            host_cwd = candidate
            cwd = "/workspace"
    elif container_backend and cwd:
        if _is_unusable_container_cwd(cwd) and cwd != default_cwd:
            logger.info("Ignoring TERMINAL_CWD=%r for %s backend "
                        "(host/relative path won't work in sandbox). Using %r instead.",
                        cwd, env_type, default_cwd)
            cwd = default_cwd

    return {
        "env_type": env_type,
        "modal_mode": coerce_modal_mode(_tenv("TERMINAL_MODAL_MODE", "auto")),
        "docker_image": _tenv("TERMINAL_DOCKER_IMAGE", default_image),
        "docker_forward_env": docker_forward_env,
        "singularity_image": _tenv("TERMINAL_SINGULARITY_IMAGE", f"docker://{default_image}"),
        "modal_image": _tenv("TERMINAL_MODAL_IMAGE", default_image),
        "daytona_image": _tenv("TERMINAL_DAYTONA_IMAGE", default_image),
        "vercel_runtime": _tenv("TERMINAL_VERCEL_RUNTIME", "").strip(),
        "cwd": cwd,
        "host_cwd": host_cwd,
        "docker_mount_cwd_to_workspace": mount_docker_cwd,
        "timeout": _parse_env_var("TERMINAL_TIMEOUT", "180"),
        "lifetime_seconds": _parse_env_var("TERMINAL_LIFETIME_SECONDS", "300"),
        # SSH-specific config
        "ssh_host": _tenv("TERMINAL_SSH_HOST", ""),
        "ssh_user": _tenv("TERMINAL_SSH_USER", ""),
        "ssh_port": _parse_env_var("TERMINAL_SSH_PORT", "22"),
        "ssh_key": _tenv("TERMINAL_SSH_KEY", ""),
        # Persistent shell: SSH defaults to the config-level persistent_shell
        # setting; local is always opt-in. Per-backend env vars override.
        "ssh_persistent": _tenv_bool(
            "TERMINAL_SSH_PERSISTENT", _tenv("TERMINAL_PERSISTENT_SHELL", "true"),
        ),
        "local_persistent": _tenv_bool("TERMINAL_LOCAL_PERSISTENT", "false"),
        # Container resources (MB); ignored for local/ssh.
        "container_cpu": container_cpu,
        "container_memory": container_memory,
        "container_disk": container_disk,
        "container_persistent": _tenv_bool("TERMINAL_CONTAINER_PERSISTENT", "true"),
        "docker_volumes": docker_volumes,
        "docker_env": docker_env,
        "docker_run_as_host_user": _tenv_bool("TERMINAL_DOCKER_RUN_AS_HOST_USER", "false"),
        "docker_network": _tenv_bool("TERMINAL_DOCKER_NETWORK", "true"),
        "docker_extra_args": docker_extra_args,
        "docker_shm_size": docker_shm_size,
        # Cross-process reuse: attach to a labeled container at startup
        # instead of starting fresh; false = per-process isolation.
        "docker_persist_across_processes": _tenv_bool("TERMINAL_DOCKER_PERSIST_ACROSS_PROCESSES", "true"),
        "docker_shared_container_key": _tenv(
            "TERMINAL_DOCKER_SHARED_CONTAINER_KEY", ""
        ).strip(),
        "docker_orphan_reaper": _tenv_bool("TERMINAL_DOCKER_ORPHAN_REAPER", "true"),
    }


def _get_modal_backend_state(modal_mode: object | None) -> Dict[str, Any]:
    """Resolve direct vs managed Modal backend selection."""
    return resolve_modal_backend_state(
        modal_mode,
        has_direct=has_direct_modal_credentials(),
        managed_ready=is_managed_tool_gateway_ready("modal"),
    )


def _ssh_config_from_config(config: Dict[str, Any]) -> dict:
    """``ssh_config`` for :func:`_create_environment` (shared by terminal_tool
    and the lazy :func:`ensure_task_env` bring-up)."""
    return {
        "host": config.get("ssh_host", ""),
        "user": config.get("ssh_user", ""),
        "port": config.get("ssh_port", 22),
        "key": config.get("ssh_key", ""),
        "persistent": config.get("ssh_persistent", False),
    }


def _container_config_from_config(config: Dict[str, Any]) -> dict:
    """``container_config`` for :func:`_create_environment` (shared by
    terminal_tool and the lazy :func:`ensure_task_env` bring-up)."""
    return {
        "container_cpu": config.get("container_cpu", 1),
        "container_memory": config.get("container_memory", 5120),
        "container_disk": config.get("container_disk", 51200),
        "container_persistent": config.get("container_persistent", True),
        "modal_mode": config.get("modal_mode", "auto"),
        "vercel_runtime": config.get("vercel_runtime", ""),
        "docker_volumes": config.get("docker_volumes", []),
        "docker_mount_cwd_to_workspace": config.get("docker_mount_cwd_to_workspace", False),
        "docker_forward_env": config.get("docker_forward_env", []),
        "docker_env": config.get("docker_env", {}),
        "docker_run_as_host_user": config.get("docker_run_as_host_user", False),
        "docker_extra_args": config.get("docker_extra_args", []),
        "docker_shm_size": config.get("docker_shm_size", "1g"),
        "docker_network": config.get("docker_network", True),
        "docker_persist_across_processes": config.get("docker_persist_across_processes", True),
        "docker_shared_container_key": config.get("docker_shared_container_key", ""),
        "docker_orphan_reaper": config.get("docker_orphan_reaper", True),
    }


def _resources(cc: Dict[str, Any]) -> dict:
    """Common sandbox resource kwargs (cpu/memory in MB/disk in MB/persistence)."""
    return {
        "cpu": cc.get("container_cpu", 1),
        "memory": cc.get("container_memory", 5120),
        "disk": cc.get("container_disk", 51200),
        "persistent_filesystem": cc.get("container_persistent", True),
    }


def _build_local_env(*, cwd, timeout, **_):
    return _LocalEnvironment(cwd=cwd, timeout=timeout)


def _build_docker_env(*, image, cwd, timeout, cc, task_id, host_cwd, **_):
    # One-shot orphan reaper for labeled containers left behind by prior
    # Hermes processes that died before atexit (SIGKILL / OOM / closed
    # terminal); once per process, ``terminal.docker_orphan_reaper: false``
    # disables it.
    _maybe_reap_docker_orphans(cc)
    # Per-session container isolation: a session-keyed container must not
    # outlive its session, so cross-process reuse/persist is disabled for it —
    # cleanup_vm()/the idle reaper stop+rm it. The shared "default" container
    # and RL/benchmark override sandboxes keep their existing lifecycle.
    session_scoped = (
        _docker_session_isolation_enabled()
        and task_id != "default"
        and not _has_isolation_overrides(task_id)
    )
    docker_env_obj = _DockerEnvironment(
        image=image, cwd=cwd, timeout=timeout, task_id=task_id,
        **_resources(cc),
        volumes=cc.get("docker_volumes", []),
        host_cwd=host_cwd,
        auto_mount_cwd=cc.get("docker_mount_cwd_to_workspace", False),
        forward_env=cc.get("docker_forward_env", []),
        env=cc.get("docker_env", {}),
        run_as_host_user=cc.get("docker_run_as_host_user", False),
        network=cc.get("docker_network", True),
        extra_args=cc.get("docker_extra_args", []),
        persist_across_processes=(
            False if session_scoped
            else cc.get("docker_persist_across_processes", True)
        ),
        shared_container_key=cc.get("docker_shared_container_key", ""),
        shm_size=cc.get("docker_shm_size", "1g"),
    )
    # Marker read by is_persistent_env(): a session-scoped container survives
    # BETWEEN turns (skip per-turn teardown) but is removed at session close /
    # idle timeout. Guarded: test doubles may not accept attributes.
    if session_scoped:
        try:
            docker_env_obj._session_scoped = True
        except AttributeError:
            pass
    return docker_env_obj


def _build_singularity_env(*, image, cwd, timeout, cc, task_id, **_):
    return _SingularityEnvironment(
        image=image, cwd=cwd, timeout=timeout, task_id=task_id, **_resources(cc),
    )


def _build_modal_env(*, image, cwd, timeout, cc, task_id, **_):
    res = _resources(cc)
    persistent = res["persistent_filesystem"]
    sandbox_kwargs = {k: res[k] for k in ("cpu", "memory") if res[k] > 0}
    if res["disk"] > 0:
        try:
            import modal
            if "ephemeral_disk" in inspect.signature(modal.Sandbox.create).parameters:
                sandbox_kwargs["ephemeral_disk"] = res["disk"]
        except Exception:
            pass

    modal_state = _get_modal_backend_state(cc.get("modal_mode"))

    if modal_state["selected_backend"] == "managed":
        return _ManagedModalEnvironment(
            image=image, cwd=cwd, timeout=timeout,
            modal_sandbox_kwargs=sandbox_kwargs,
            persistent_filesystem=persistent, task_id=task_id,
        )

    if modal_state["selected_backend"] != "direct":
        if modal_state["managed_mode_blocked"]:
            raise ValueError(
                "Modal backend is configured for managed mode, but "
                "Nous Tool Gateway access is not currently available and no direct "
                "Modal credentials/config were found. "
                + nous_tool_gateway_unavailable_message(
                    "managed Modal execution",
                )
                + " Choose TERMINAL_MODAL_MODE=direct/auto to use direct Modal credentials."
            )
        if modal_state["mode"] == "managed":
            raise ValueError(
                "Modal backend is configured for managed mode, but the managed tool gateway is unavailable. "
                + nous_tool_gateway_unavailable_message(
                    "managed Modal execution",
                )
            )
        if modal_state["mode"] == "direct":
            raise ValueError(
                "Modal backend is configured for direct mode, but no direct Modal credentials/config were found."
            )
        message = "Modal backend selected but no direct Modal credentials/config was found."
        if managed_nous_tools_enabled():
            message = (
                "Modal backend selected but no direct Modal credentials/config or managed tool gateway was found."
            )
        raise ValueError(message)

    return _ModalEnvironment(
        image=image, cwd=cwd, timeout=timeout,
        modal_sandbox_kwargs=sandbox_kwargs,
        persistent_filesystem=persistent, task_id=task_id,
    )


def _build_daytona_env(*, image, cwd, timeout, cc, task_id, **_):
    # Lazy import so daytona SDK is only required when backend is selected.
    from tools.environments.daytona import DaytonaEnvironment as _DaytonaEnvironment
    res = _resources(cc)
    res["cpu"] = int(res["cpu"])
    return _DaytonaEnvironment(image=image, cwd=cwd, timeout=timeout, task_id=task_id, **res)


def _build_vercel_env(*, cwd, timeout, cc, task_id, **_):
    from tools.environments.vercel_sandbox import (
        VercelSandboxEnvironment as _VercelSandboxEnvironment,
    )
    return _VercelSandboxEnvironment(
        runtime=cc.get("vercel_runtime") or None,
        cwd=cwd, timeout=timeout, task_id=task_id, **_resources(cc),
    )


def _build_ssh_env(*, cwd, timeout, ssh_config, **_):
    if not ssh_config or not ssh_config.get("host") or not ssh_config.get("user"):
        raise ValueError("SSH environment requires ssh_host and ssh_user to be configured")
    return _SSHEnvironment(
        host=ssh_config["host"],
        user=ssh_config["user"],
        port=ssh_config.get("port", 22),
        key_path=ssh_config.get("key", ""),
        cwd=cwd,
        timeout=timeout,
    )


def _build_plugin_env(*, env_type, image, cwd, timeout, cc, task_id, **_):
    provider = _get_plugin_env_provider(env_type)
    if provider is not None:
        env_obj = provider.create_environment(
            cwd=cwd, timeout=timeout, task_id=task_id,
            image=image, container_config=cc,
        )
        # Stamp the backend name so path-resolution and progress surfaces
        # can identify plugin backends without class-name sniffing.
        try:
            env_obj._hermes_backend_name = provider.name.strip().lower()
        except AttributeError:
            pass  # test doubles may reject attributes
        return env_obj
    try:
        from agent.terminal_env_registry import plugin_backend_names

        plugin_names = plugin_backend_names()
    except Exception:
        plugin_names = []
    extra = (
        ", " + ", ".join(f"'{n}'" for n in plugin_names) if plugin_names else ""
    )
    raise ValueError(
        f"Unknown environment type: {env_type}. Use 'local', 'docker', "
        f"'singularity', 'modal', 'daytona', 'vercel_sandbox', 'ssh'{extra}"
    )


# Built-in backend -> builder. Anything else is looked up in the plugin registry.
_ENV_BUILDERS = {
    "local": _build_local_env,
    "docker": _build_docker_env,
    "singularity": _build_singularity_env,
    "modal": _build_modal_env,
    "daytona": _build_daytona_env,
    "vercel_sandbox": _build_vercel_env,
    "ssh": _build_ssh_env,
}


def _create_environment(env_type: str, image: str, cwd: str, timeout: int,
                        ssh_config: dict = None, container_config: dict = None,
                        local_config: dict = None,
                        task_id: str = "default",
                        host_cwd: Optional[str] = None):
    """Create an execution environment (instance with ``execute()``) for *env_type*.

    ``image`` is ignored for local/ssh/vercel; ``container_config`` carries the
    container_*/docker_* resource keys; ``host_cwd`` is the host directory to
    bind into Docker when cwd mounting is explicitly enabled. Unknown
    ``env_type`` values fall through to plugin-registered backends.
    """
    builder = _ENV_BUILDERS.get(env_type, _build_plugin_env)
    return builder(
        env_type=env_type, image=image, cwd=cwd, timeout=timeout,
        cc=container_config or {}, task_id=task_id,
        ssh_config=ssh_config, host_cwd=host_cwd,
    )


def _teardown_env(env: Any, task_id: str, *, force_remove: Optional[bool] = None, done_msg: str = "Cleaned up inactive environment for task: %s") -> None:
    """Stop *env* via cleanup()/stop()/terminate(), whichever it has; log the outcome.

    ``force_remove`` is forwarded to ``cleanup()`` only when given and the
    backend's signature accepts it (DockerEnvironment; others don't). A
    404/"not found" error means the sandbox is already gone — logged at info.
    """
    try:
        if hasattr(env, 'cleanup'):
            if force_remove is not None and "force_remove" in inspect.signature(env.cleanup).parameters:
                env.cleanup(force_remove=force_remove)
            else:
                env.cleanup()
        elif hasattr(env, 'stop'):
            env.stop()
        elif hasattr(env, 'terminate'):
            env.terminate()
        logger.info(done_msg, task_id)
    except Exception as e:
        error_str = str(e)
        if "404" in error_str or "not found" in error_str.lower():
            logger.info("Environment for task %s already cleaned up", task_id)
        else:
            logger.warning("Error cleaning up environment for task %s: %s", task_id, e)


def _clear_file_ops_cache(task_id: str) -> None:
    """Invalidate the file_ops cache entry so ShellFileOperations can't reference a dead sandbox."""
    try:
        from tools.file_tools import clear_file_ops_cache
        clear_file_ops_cache(task_id)
    except ImportError:
        pass


def _cleanup_inactive_envs(lifetime_seconds: int = 300):
    """Clean up environments that have been inactive for longer than lifetime_seconds."""
    current_time = time.time()

    # Sandboxes with active background processes stay alive (refresh activity).
    try:
        from tools.process_registry import process_registry
        for task_id in list(_last_activity.keys()):
            if process_registry.has_active_processes(task_id):
                _last_activity[task_id] = current_time
    except ImportError:
        pass

    # Phase 1: unregister stale entries under the lock. Do NOT call
    # env.cleanup() inside the lock — Modal/Docker teardown can block 10-15s
    # and would stall every concurrent terminal/file tool call.
    envs_to_stop = []  # list of (task_id, env) pairs

    with _env_lock:
        for task_id, last_time in list(_last_activity.items()):
            if current_time - last_time > lifetime_seconds:
                env = _active_environments.pop(task_id, None)
                _last_activity.pop(task_id, None)
                if env is not None:
                    envs_to_stop.append((task_id, env))

        with _creation_locks_lock:
            for task_id, _ in envs_to_stop:
                _creation_locks.pop(task_id, None)

    # Phase 2: stop the sandboxes outside the lock.
    for task_id, env in envs_to_stop:
        _clear_file_ops_cache(task_id)
        _teardown_env(env, task_id)


def _cleanup_thread_worker():
    """Background thread worker that periodically cleans up inactive environments."""
    while _cleanup_running:
        try:
            config = _get_env_config()
            _cleanup_inactive_envs(config["lifetime_seconds"])
        except Exception as e:
            logger.warning("Error in cleanup thread: %s", e, exc_info=True)

        for _ in range(60):
            if not _cleanup_running:
                break
            time.sleep(1)


def _start_cleanup_thread():
    """Start the background cleanup thread if not already running."""
    global _cleanup_thread, _cleanup_running

    with _env_lock:
        if _cleanup_thread is None or not _cleanup_thread.is_alive():
            _cleanup_running = True
            _cleanup_thread = threading.Thread(target=_cleanup_thread_worker, daemon=True)
            _cleanup_thread.start()


def _stop_cleanup_thread():
    """Stop the background cleanup thread."""
    global _cleanup_running
    _cleanup_running = False
    if _cleanup_thread is not None:
        try:
            _cleanup_thread.join(timeout=5)
        except (SystemExit, KeyboardInterrupt):
            pass


def get_active_env(task_id: str):
    """Return the active BaseEnvironment for *task_id*, or None."""
    lookup = _resolve_container_task_id(task_id)
    with _env_lock:
        return _active_environments.get(lookup) or _active_environments.get(task_id)


def ensure_task_env(task_id: Optional[str] = None):
    """Lazily create and cache the sandbox env for *task_id* if none is active.

    Lets non-terminal callers (``tools.image_source`` reading container-only
    paths) bring the sandbox up on demand with the same machinery as the
    terminal tool. No-op on local. Returns the env, or ``None`` when local or
    when creation fails (best-effort; the caller's fail-closed path stays intact).
    """
    config = _get_env_config()
    env_type = config["env_type"]
    if env_type == "local":
        return None

    effective_task_id = _resolve_container_task_id(task_id)

    existing = get_active_env(effective_task_id)
    if existing is not None:
        with _env_lock:
            _last_activity[effective_task_id] = time.time()
        return existing

    overrides = resolve_task_overrides(task_id)
    image = _select_image(env_type, overrides, config)

    _start_cleanup_thread()

    with _creation_locks_lock:
        task_lock = _creation_locks.setdefault(effective_task_id, threading.Lock())

    with task_lock:
        existing = get_active_env(effective_task_id)
        if existing is not None:
            return existing
        try:
            new_env = _create_environment(
                env_type=env_type,
                image=image,
                cwd=config["cwd"],
                timeout=config["timeout"],
                ssh_config=_ssh_config_from_config(config) if env_type == "ssh" else None,
                container_config=(
                    _container_config_from_config(config)
                    if _is_container_backend(env_type) else None
                ),
                local_config=None,
                task_id=effective_task_id,
                host_cwd=_resolve_task_host_cwd(config, task_id),
            )
        except Exception as exc:  # noqa: BLE001 — best-effort bring-up
            logger.warning(
                "Lazy %s environment init failed for task %s: %s",
                env_type, effective_task_id[:8], exc,
            )
            return None

        with _env_lock:
            _active_environments[effective_task_id] = new_env
            _last_activity[effective_task_id] = time.time()
        logger.info(
            "%s environment lazily initialized for task %s",
            env_type, effective_task_id[:8],
        )
        return new_env


def is_persistent_env(task_id: str) -> bool:
    """True if *task_id*'s active env persists across turns.

    The agent loop skips per-turn teardown for these (persistent docker,
    daytona, modal, …); non-persistent backends are torn down at end of turn
    to prevent leakage, and the idle reaper handles the rest. Session-scoped
    docker containers count as persistent HERE: their lifetime is the session
    (removed by ``AIAgent.close()`` → ``cleanup_vm`` and the idle reaper).
    """
    env = get_active_env(task_id)
    if env is None:
        return False
    if getattr(env, "_session_scoped", False):
        return True
    return bool(getattr(env, "_persistent", False))




def cleanup_all_environments():
    """Clean up ALL active environments. Use with caution."""
    task_ids = list(_active_environments.keys())
    cleaned = 0
    
    for task_id in task_ids:
        try:
            cleanup_vm(task_id)
            cleaned += 1
        except Exception as e:
            logger.error("Error cleaning %s: %s", task_id, e, exc_info=True)
    
    # Also clean any orphaned directories
    scratch_dir = _get_scratch_dir()
    import glob
    for path in glob.glob(str(scratch_dir / "hermes-*")):
        try:
            shutil.rmtree(path, ignore_errors=True)
            logger.info("Removed orphaned: %s", path)
        except OSError as e:
            logger.debug("Failed to remove orphaned path %s: %s", path, e)
    
    if cleaned > 0:
        logger.info("Cleaned %d environments", cleaned)
    return cleaned


def cleanup_vm(task_id: str, *, force_remove: bool = False):
    """Manually clean up a specific environment by task_id.

    *force_remove* is forwarded to backends that accept it (currently only
    ``DockerEnvironment``). Default False matches session-lifecycle semantics:
    callers (``AIAgent.close()`` on TUI/gateway session teardown, the per-turn
    cleanup of non-persistent envs) must honor the user's persist-mode
    preference — stopping the container here would break the "ONE long-lived
    container shared across sessions" contract. Pass ``force_remove=True``
    only for user-initiated teardown. The idle reaper calls ``env.cleanup()``
    directly, so persist-mode idle envs are likewise no-op'd; only the orphan
    reaper at next startup reclaims them.
    """
    # Unregister under the lock; run the (slow) cleanup outside it.
    with _env_lock:
        env = _active_environments.pop(task_id, None)
        _last_activity.pop(task_id, None)

    with _creation_locks_lock:
        _creation_locks.pop(task_id, None)

    _clear_file_ops_cache(task_id)

    if env is None:
        return
    _teardown_env(
        env, task_id, force_remove=force_remove,
        done_msg="Manually cleaned up environment for task: %s",
    )


def _atexit_cleanup():
    """Stop the cleanup thread and shut down all remaining sandboxes on exit."""
    _stop_cleanup_thread()
    if _active_environments:
        count = len(_active_environments)
        logger.info("Shutting down %d remaining sandbox(es)...", count)
        # Snapshot BEFORE cleanup_all_environments empties the dict, then
        # block briefly so docker stop/rm completes before the interpreter
        # exits — otherwise daemon cleanup threads die mid-`docker stop` and
        # Exited containers pile up on the host.
        envs_to_wait = list(_active_environments.values())
        cleanup_all_environments()
        for env in envs_to_wait:
            wait_fn = getattr(env, "wait_for_cleanup", None)
            if wait_fn is None:
                continue
            try:
                wait_fn(timeout=15.0)
            except Exception as e:  # never block shutdown on a bad backend
                logger.debug("wait_for_cleanup raised on exit: %s", e)

atexit.register(_atexit_cleanup)




def _command_requires_pipe_stdin(command: str) -> bool:
    """True when PTY mode would break a stdin-driven command: `gh auth login
    --with-token` waits for EOF on piped stdin, and under a PTY
    `process.submit()` only sends a newline, so it hangs forever."""
    normalized = " ".join(command.lower().split())
    return (
        normalized.startswith("gh auth login")
        and "--with-token" in normalized
    )


from tools.terminal_tool_guards import (  # noqa: F401 — re-exported (tests, plugins)
    _LONG_LIVED_FOREGROUND_PATTERNS,
    _SHELL_LEVEL_BACKGROUND_RE,
    _WORKDIR_SAFE_ASCII_CHARS,
    _foreground_background_guidance,
    _is_safe_workdir_char,
    _looks_like_help_or_version_command,
    _safe_command_preview,
    _strip_quotes,
    _validate_workdir,
    gateway_lifecycle_block,
    self_repo_block,
)
from tools.terminal_tool_background import spawn_background_process
from tools.terminal_tool_result import (  # noqa: F401 — re-exported (tests)
    _SIGNAL_EXIT_NOTES,
    _interpret_exit_code,
    _interpret_signal_exit,
    finalize_foreground_result,
)


def _resolve_notification_flag_conflict(
    *,
    notify_on_complete: bool,
    watch_patterns,
    background: bool,
) -> tuple:
    """Resolve notify_on_complete + watch_patterns both set: drop watch_patterns
    (combined they produce duplicate async notifications — one per match plus
    one on exit — that can spam the user long after the process ends).
    Returns ``(watch_patterns_to_use, conflict_note)``; note is "" without conflict."""
    if background and notify_on_complete and watch_patterns:
        note = (
            "watch_patterns ignored because notify_on_complete=True; "
            "these two flags produce duplicate notifications when combined"
        )
        return None, note
    return watch_patterns, ""


def _resolve_command_cwd(
    *,
    workdir: Optional[str],
    default_cwd: str,
    session_key: Optional[str] = None,
    env_type: Optional[str] = None,
) -> str:
    """cwd for a command: explicit ``workdir`` > the session's own cwd record >
    ``default_cwd``.

    The record is written after every completed command of THIS session, so
    it is the session's ``cd`` state with no shared-env ambiguity. On
    container backends a recorded HOST path (a desktop/TUI surface registering
    its workspace) is unusable in the sandbox — ``cd <host path>`` fails with
    exit 126 — so it is discarded in favor of ``default_cwd``.
    """
    if workdir:
        return workdir
    recorded = get_session_cwd(session_key)
    if (
        recorded
        and _is_container_backend(env_type)
        and _is_unusable_container_cwd(recorded)
    ):
        logger.info(
            "Ignoring recorded session cwd %r for %s backend "
            "(host/relative path won't work in sandbox). Using %r instead.",
            recorded, env_type, default_cwd,
        )
        return default_cwd
    return recorded or default_cwd


@dataclass
class _ApprovalVerdict:
    """Outcome of the pre-exec guard pass.

    ``blocked_json`` is the finished tool result when the command may not run
    (denied, or pending gateway approval). ``approved_run`` is True when the
    user explicitly approved (or pre-confirmed via ``force``); it drives the
    clean-interrupt-slate clear before ``env.execute`` so an approved command
    can't be SIGINT-killed by a bit that landed during the approval-wait.
    """
    blocked_json: Optional[str] = None
    note: Optional[str] = None
    approved_run: bool = False


def _run_approval_guards(command: str, env_type: str, config: Dict[str, Any], *, force: bool) -> _ApprovalVerdict:
    """Run tirith + dangerous-command guards; ``force`` skips them entirely."""
    if force:
        return _ApprovalVerdict(approved_run=True)
    approval = _check_all_guards(
        command, env_type,
        has_host_access=_docker_has_host_access(config),
    )
    if not approval["approved"]:
        if approval.get("status") == "pending_approval":  # gateway ask mode
            return _ApprovalVerdict(blocked_json=json.dumps({
                "output": "",
                "exit_code": -1,
                "error": "",
                "status": "pending_approval",
                "approval_pending": True,
                "command": approval.get("command", command),
                "description": approval.get("description", "command flagged"),
                "pattern_key": approval.get("pattern_key", ""),
                "smart_denied": approval.get("smart_denied", False),
                "allow_permanent": approval.get("allow_permanent", True),
            }, ensure_ascii=False))
        desc = approval.get("description", "command flagged")
        fallback_msg = (
            f"Command denied: {desc}. "
            "Use the approval prompt to allow it, or rephrase the command."
        )
        return _ApprovalVerdict(blocked_json=json.dumps({
            "output": "",
            "exit_code": -1,
            "error": approval.get("message", fallback_msg),
            "status": "blocked"
        }, ensure_ascii=False))
    if approval.get("user_approved"):
        desc = approval.get("description", "flagged as dangerous")
        return _ApprovalVerdict(
            note=f"Command required approval ({desc}) and was approved by the user.",
            approved_run=True,
        )
    if approval.get("smart_approved"):
        desc = approval.get("description", "flagged as dangerous")
        return _ApprovalVerdict(
            note=f"Command was flagged ({desc}) and auto-approved by smart approval.",
        )
    return _ApprovalVerdict()


def _fatal_error_json(e: BaseException) -> str:
    """Log the traceback and return the redacted error+traceback envelope.

    Exception text can embed the failing command line (and any secrets inline
    in it), so both fields are force-redacted before reaching the model.
    """
    import traceback
    tb_str = traceback.format_exc()
    logger.error("terminal_tool exception:\n%s", tb_str)
    return json.dumps({
        "output": "",
        "exit_code": -1,
        "error": _redact_terminal_error_text(f"Failed to execute command: {e}"),
        "traceback": _redact_terminal_error_text(tb_str),
        "status": "error"
    }, ensure_ascii=False)


def terminal_tool(
    command: str,
    background: bool = False,
    timeout: Optional[int] = None,
    task_id: Optional[str] = None,
    session_id: Optional[str] = None,
    force: bool = False,
    workdir: Optional[str] = None,
    pty: bool = False,
    notify_on_complete: bool = False,
    watch_patterns: Optional[List[str]] = None,
    _host_local: bool = False,
) -> str:
    """Execute *command* in the configured terminal environment; returns a JSON string.

    ``force`` (internal, not in the model schema) skips the dangerous-command
    check after the user confirmed. ``workdir`` is per-command and never
    recorded as the session cwd. ``pty`` applies to the local backend only.
    ``notify_on_complete`` and ``watch_patterns`` are mutually exclusive
    background-only flags: on conflict watch_patterns is dropped. watch_patterns
    is hard rate-limited (1 notification / 15s / process) and auto-disabled
    after repeated strikes or a lifetime cap, promoting to notify_on_complete —
    use it only for rare one-shot signals on long-lived processes.
    ``_host_local`` forces the local backend for Hermes-owned control-plane
    children (kept in a separate env cache from the configured backend).
    """
    try:
        if not isinstance(command, str):
            logger.warning(
                "Rejected invalid terminal command value: %s",
                type(command).__name__,
            )
            return json.dumps({
                "output": "",
                "exit_code": -1,
                "error": f"Invalid command: expected string, got {type(command).__name__}",
                "status": "error",
            }, ensure_ascii=False)

        config = _get_env_config()
        env_type = "local" if _host_local else config["env_type"]

        # Fail closed under a refusal scope: the routed profile's terminal
        # policy could not be resolved, so running with the launch process's
        # ambient policy is forbidden.
        if not _host_local:
            from tools.terminal_scope import enforce_no_refusal

            enforce_no_refusal()

        effective_task_id = _resolve_container_task_id(task_id)
        if _host_local:
            # Control-plane children run beside this interpreter, never inside
            # the configured Docker/SSH backend; keep their env cache separate.
            effective_task_id = f"host-local-{effective_task_id}"

        # Per-task overrides (RL/benchmark envs, ACP workspace cwd) win over
        # the global env-var config; ``resolve_task_overrides`` reads the raw
        # task id first, then the collapsed container id.
        overrides = resolve_task_overrides(task_id)
        image = _select_image(env_type, overrides, config)

        cwd = overrides.get("cwd") or get_session_cwd(task_id) or config["cwd"]
        host_cwd = _resolve_task_host_cwd(config, task_id)
        # config["cwd"] was sanitized for container backends in _get_env_config
        # but an override / session record is raw: a host path would reach
        # `docker run -w` and fail with exit 125. Re-apply the guard to the
        # resolved cwd; when the host path IS this session's mounted workspace,
        # remap to /workspace instead of discarding it.
        if _is_container_backend(env_type) and _is_unusable_container_cwd(cwd):
            remapped = "/workspace" if host_cwd else config["cwd"]
            if cwd != remapped:
                logger.info(
                    "Remapping host/relative cwd override %r for %s backend "
                    "(won't exist in sandbox). Using %r instead.",
                    cwd, env_type, remapped,
                )
            cwd = remapped
        # Reject non-positive timeouts before deadline math: ``timeout or
        # default`` would silently turn 0 into the default, and a negative
        # value is truthy and would fire an immediate "-Ns" timeout.
        if timeout is not None and timeout <= 0:
            return tool_error(
                f"timeout must be a positive number of seconds (got {timeout})."
            )
        effective_timeout = timeout or config["timeout"]

        if not background and timeout and timeout > FOREGROUND_MAX_TIMEOUT:
            return tool_error(
                f"Foreground timeout {timeout}s exceeds the maximum of "
                f"{FOREGROUND_MAX_TIMEOUT}s. Use background=true with "
                f"notify_on_complete=true for long-running commands."
            )

        if not background:
            guidance = _foreground_background_guidance(command)
            if guidance:
                return json.dumps({
                    "output": "",
                    "exit_code": -1,
                    "error": guidance,
                    "status": "error",
                }, ensure_ascii=False)

        _start_cleanup_thread()

        # Get or create environment. A per-task creation lock makes concurrent
        # calls for the same task_id wait for the first sandbox instead of each
        # creating their own; the cache is re-checked under that lock.
        with _env_lock:
            env: Any = _lookup_active_env(effective_task_id, task_id)

        if env is None:
            with _creation_locks_lock:
                task_lock = _creation_locks.setdefault(effective_task_id, threading.Lock())

            with task_lock:
                with _env_lock:
                    env = _lookup_active_env(effective_task_id, task_id)

                if env is None:
                    if env_type == "singularity":
                        _check_disk_usage_warning()
                    logger.info("Creating new %s environment for task %s...", env_type, effective_task_id[:8])
                    try:
                        new_env = _create_environment(
                            env_type=env_type,
                            image=image,
                            cwd=cwd,
                            timeout=effective_timeout,
                            ssh_config=_ssh_config_from_config(config) if env_type == "ssh" else None,
                            container_config=(
                                _container_config_from_config(config)
                                if _is_container_backend(env_type) else None
                            ),
                            local_config=(
                                {"persistent": config.get("local_persistent", False)}
                                if env_type == "local" else None
                            ),
                            task_id=effective_task_id,
                            host_cwd=host_cwd,
                        )
                    except ImportError as e:
                        return json.dumps({
                            "output": "",
                            "exit_code": -1,
                            "error": _redact_terminal_error_text(
                                f"Terminal tool disabled: environment creation failed ({e})"
                            ),
                            "status": "disabled"
                        }, ensure_ascii=False)

                    with _env_lock:
                        _active_environments[effective_task_id] = new_env
                        _last_activity[effective_task_id] = time.time()
                        env = new_env
                    logger.info("%s environment ready for task %s", env_type, effective_task_id[:8])

        assert env is not None  # all creation failure paths return above

        # Session key for cwd records: the contextvar doesn't cross tool-worker
        # threads, so fall back to the raw task_id (the top-level agent's
        # session_key) as a stable anchor.
        from tools.approval import get_current_session_key

        session_key = get_current_session_key(default="") or (task_id or "")

        blocked = gateway_lifecycle_block(
            command=command, env=env, env_type=env_type, cwd=cwd,
            workdir=workdir, session_key=session_key,
        )
        if blocked:
            return blocked

        if workdir:
            workdir_error = _validate_workdir(workdir)
            if workdir_error:
                logger.warning("Blocked dangerous workdir: %s (command: %s)",
                               workdir[:200], _safe_command_preview(command))
                return json.dumps({
                    "output": "",
                    "exit_code": -1,
                    "error": workdir_error,
                    "status": "blocked"
                }, ensure_ascii=False)

        if env_type == "local":
            blocked = self_repo_block(
                command=command, cwd=cwd, workdir=workdir, session_key=session_key,
            )
            if blocked:
                return blocked

        # Pre-exec security checks (tirith + dangerous command detection);
        # force=True means the user already confirmed.
        verdict = _run_approval_guards(command, env_type, config, force=force)
        if verdict.blocked_json:
            return verdict.blocked_json
        approval_note = verdict.note

        pty_disabled_reason = None
        effective_pty = pty
        if pty and _command_requires_pipe_stdin(command):
            effective_pty = False
            pty_disabled_reason = (
                "PTY disabled for this command because it expects piped stdin/EOF "
                "(for example gh auth login --with-token). For local background "
                "processes, call process(action='close') after writing so it receives "
                "EOF."
            )

        if background:
            return spawn_background_process(
                command=command,
                env=env,
                env_type=env_type,
                effective_task_id=effective_task_id,
                task_id=task_id,
                session_key=session_key,
                workdir=workdir,
                cwd=cwd,
                effective_pty=effective_pty,
                notify_on_complete=notify_on_complete,
                watch_patterns=watch_patterns,
                approval_note=approval_note,
                pty_disabled_reason=pty_disabled_reason,
            )
        # Foreground: run with retry on transient errors.
        max_retries = 3
        retry_count = 0
        result = None
        command_cwd = None

        # Clean interrupt slate for an approved command, ONCE before the retry
        # loop: drop a stale bit that landed during the approval-wait so it
        # can't SIGINT the just-approved run. Do NOT re-clear inside the loop —
        # a genuine interrupt during the backoff sleep must survive and abort
        # the next attempt (rc 130).
        if verdict.approved_run:
            from tools.interrupt import clear_current_thread_interrupt
            clear_current_thread_interrupt()

        while retry_count <= max_retries:
            try:
                command_cwd = _resolve_command_cwd(
                    workdir=workdir,
                    default_cwd=cwd,
                    session_key=session_key,
                    env_type=env_type,
                )
                # bounded_capture: model-facing output keeps a head/tail window
                # while streaming so a verbose command can't OOM the gateway;
                # internal env.execute() consumers stay unbounded.
                result = env.execute(
                    command, timeout=effective_timeout, cwd=command_cwd,
                    bounded_capture=True,
                )
            except Exception as e:
                error_str = str(e).lower()
                if "timeout" in error_str:
                    return json.dumps({
                        "output": "",
                        "exit_code": 124,
                        "error": f"Command timed out after {effective_timeout} seconds"
                    }, ensure_ascii=False)

                # Retry on transient errors
                if retry_count < max_retries:
                    retry_count += 1
                    wait_time = 2 ** retry_count
                    logger.warning("Execution error, retrying in %ds (attempt %d/%d) - Command: %s - Error: %s: %s - Task: %s, Backend: %s",
                                   wait_time, retry_count, max_retries, _safe_command_preview(command), type(e).__name__, e, effective_task_id, env_type)
                    time.sleep(wait_time)
                    continue

                logger.error("Execution failed after %d retries - Command: %s - Error: %s: %s - Task: %s, Backend: %s",
                             max_retries, _safe_command_preview(command), type(e).__name__, e, effective_task_id, env_type)
                return json.dumps({
                    "output": "",
                    "exit_code": -1,
                    "error": _redact_terminal_error_text(
                        f"Command execution failed: {type(e).__name__}: {e}"
                    )
                }, ensure_ascii=False)

            break

        return finalize_foreground_result(
            command=command,
            result=result,
            env=env,
            env_type=env_type,
            effective_task_id=effective_task_id,
            task_id=task_id,
            session_id=session_id,
            session_key=session_key,
            workdir=workdir,
            command_cwd=command_cwd,
            approval_note=approval_note,
        )

    except EnvironmentConnectionError as e:
        # Infrastructure failure (SSH host down, Docker daemon unreachable),
        # distinct from a nonzero exit. ``terminal.degraded_mode``: warn
        # (default) returns a structured degraded result with a retry hint;
        # fail preserves the historical error+traceback result.
        degraded_mode = _tenv("TERMINAL_DEGRADED_MODE", "warn").strip().lower()
        if degraded_mode == "fail":
            return _fatal_error_json(e)

        logger.warning("terminal backend degraded: %s", e.reason)
        # Evict the possibly-broken backend so the next call re-creates it.
        try:
            _evict_environment_for_task(task_id)
        except Exception:
            logger.debug("degraded-env eviction failed", exc_info=True)
        return json.dumps({
            "output": "",
            "exit_code": -1,
            "status": "degraded",
            "reason": e.reason,
            "retry_hint": e.retry_hint,
            "error": f"Terminal backend degraded: {e.reason}",
        }, ensure_ascii=False)

    except Exception as e:
        return _fatal_error_json(e)


def _evict_environment_for_task(task_id: Optional[str]) -> None:
    """Drop any cached env for *task_id* (and its collapsed key) after an
    infrastructure failure, so later calls don't reuse a dead connection."""
    keys = {_resolve_container_task_id(task_id)}
    if task_id:
        keys.add(task_id)
    evicted = []
    with _env_lock:
        for key in keys:
            env = _active_environments.pop(key, None)
            _last_activity.pop(key, None)
            if env is not None:
                evicted.append(env)
    for env in evicted:
        try:
            env.cleanup()
        except Exception:
            logger.debug("cleanup of degraded environment failed", exc_info=True)


def _check_docker_requirements(config: Dict[str, Any]) -> bool:
    from tools.environments.docker import find_docker
    docker = find_docker()
    if not docker:
        logger.error("Docker executable not found in PATH or common install locations")
        return False
    result = subprocess.run([docker, "version"], capture_output=True, timeout=5, stdin=subprocess.DEVNULL)
    return result.returncode == 0


def _check_singularity_requirements(config: Dict[str, Any]) -> bool:
    executable = shutil.which("apptainer") or shutil.which("singularity")
    if executable:
        result = subprocess.run([executable, "--version"], capture_output=True, timeout=5, stdin=subprocess.DEVNULL)
        return result.returncode == 0
    return False


def _check_ssh_requirements(config: Dict[str, Any]) -> bool:
    if not config.get("ssh_host") or not config.get("ssh_user"):
        logger.error(
            "SSH backend selected but TERMINAL_SSH_HOST and TERMINAL_SSH_USER "
            "are not both set. Configure both or switch TERMINAL_ENV to 'local'."
        )
        return False
    return True


def _check_modal_requirements(config: Dict[str, Any]) -> bool:
    modal_state = _get_modal_backend_state(config.get("modal_mode"))
    if modal_state["selected_backend"] == "managed":
        return True

    if modal_state["selected_backend"] != "direct":
        if modal_state["managed_mode_blocked"]:
            logger.error(
                "Modal backend selected with TERMINAL_MODAL_MODE=managed, but "
                "Nous Tool Gateway access is not currently available and no direct "
                "Modal credentials/config were found. %s Choose "
                "TERMINAL_MODAL_MODE=direct/auto to use direct Modal credentials.",
                nous_tool_gateway_unavailable_message(
                    "managed Modal execution",
                ),
            )
            return False
        if modal_state["mode"] == "managed":
            logger.error(
                "Modal backend selected with TERMINAL_MODAL_MODE=managed, but the managed "
                "tool gateway is unavailable. %s",
                nous_tool_gateway_unavailable_message(
                    "managed Modal execution",
                ),
            )
            return False
        elif modal_state["mode"] == "direct":
            if managed_nous_tools_enabled():
                logger.error(
                    "Modal backend selected with TERMINAL_MODAL_MODE=direct, but no direct "
                    "Modal credentials/config were found. Configure Modal or choose "
                    "TERMINAL_MODAL_MODE=managed/auto."
                )
            else:
                logger.error(
                    "Modal backend selected with TERMINAL_MODAL_MODE=direct, but no direct "
                    "Modal credentials/config were found. Configure Modal or choose "
                    "TERMINAL_MODAL_MODE=auto."
                )
            return False
        else:
            if managed_nous_tools_enabled():
                logger.error(
                    "Modal backend selected but no direct Modal credentials/config or managed "
                    "tool gateway was found. Configure Modal, set up the managed gateway, "
                    "or choose a different TERMINAL_ENV."
                )
            else:
                logger.error(
                    "Modal backend selected but no direct Modal credentials/config was found. "
                    "Configure Modal or choose a different TERMINAL_ENV."
                )
            return False

    if importlib.util.find_spec("modal") is None:
        logger.error("modal is required for direct modal terminal backend: pip install modal")
        return False

    return True


def _check_daytona_requirements(config: Dict[str, Any]) -> bool:
    from daytona import Daytona  # noqa: F401 — SDK presence check
    from agent.secret_scope import get_secret
    return get_secret("DAYTONA_API_KEY") is not None


def _check_plugin_requirements(config: Dict[str, Any]) -> bool:
    env_type = config["env_type"]
    provider = _get_plugin_env_provider(env_type)
    if provider is not None:
        return bool(provider.check_requirements(config))
    logger.error(
        "Unknown TERMINAL_ENV '%s'. Use one of: local, docker, singularity, "
        "modal, daytona, vercel_sandbox, ssh, or a plugin-registered backend.",
        env_type,
    )
    return False


# Built-in backend -> requirements checker; unknown backends go to the plugin registry.
_REQUIREMENT_CHECKERS = {
    "local": lambda config: True,
    "docker": _check_docker_requirements,
    "singularity": _check_singularity_requirements,
    "ssh": _check_ssh_requirements,
    "modal": _check_modal_requirements,
    "vercel_sandbox": _check_vercel_sandbox_requirements,
    "daytona": _check_daytona_requirements,
}


def check_terminal_requirements() -> bool:
    """Check if all requirements for the terminal tool are met."""
    try:
        config = _get_env_config()
        checker = _REQUIREMENT_CHECKERS.get(config["env_type"], _check_plugin_requirements)
        return checker(config)
    except Exception as e:
        logger.error("Terminal requirements check failed: %s", e, exc_info=True)
        return False


from tools.registry import registry

TERMINAL_SCHEMA = {
    "name": "terminal",
    "description": TERMINAL_TOOL_DESCRIPTION,
    "parameters": {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to execute"
            },
            "background": {
                "type": "boolean",
                "description": "Run in the background, returning a session_id. Pair with notify=true for anything with a defined end (tests, builds, deploys) — without it the process runs silently. Only servers/watchers/daemons that never exit should stay silent. Short commands: prefer foreground with a generous timeout.",
                "default": False
            },
            "timeout": {
                "type": "integer",
                "description": f"Max seconds to wait (default: 180, foreground max: {FOREGROUND_MAX_TIMEOUT}). Returns INSTANTLY when command finishes — set high for long tasks, you won't wait unnecessarily. Foreground timeout above {FOREGROUND_MAX_TIMEOUT}s is rejected; use background=true for longer commands.",
                "minimum": 1
            },
            "workdir": {
                "type": "string",
                "description": "Working directory for this command (absolute path). Defaults to the session working directory."
            },
            "pty": {
                "type": "boolean",
                "description": "With background=true: run in a pseudo-terminal for interactive CLI tools (Codex, Claude Code, Python REPL). Local backend only. Default: false.",
                "default": False
            },
            "notify": {
                "description": "With background=true: notify=true fires exactly one notification when the process exits (the right choice for nearly every bounded task — builds, tests, deploys). notify=['pattern', ...] instead notifies when a line matches a pattern — ONLY for one-shot readiness signals on processes that never exit (e.g. ['Application startup complete']); rate-limited and auto-disabled if it over-fires. Omit for silent daemons.",
                "anyOf": [
                    {"type": "boolean"},
                    {"type": "array", "items": {"type": "string"}}
                ]
            }
            # Legacy aliases (unadvertised, still accepted): notify_on_complete
            # (bool) and watch_patterns (list). notify=true|[...] maps onto
            # them in the dispatch wrapper; explicit notify wins on conflict.
        },
        "required": ["command"]
    }
}


def _handle_terminal(args, **kw):
    # Models sometimes send execute_code's ``code`` here; name the stray
    # argument and the right tool instead of failing on command=None.
    if "command" not in args and "code" in args:
        return tool_error(
            "terminal received a 'code' parameter, but it requires a shell "
            "command in 'command'. Use execute_code(code=...) for Python; "
            "for shell, retry as terminal(command=...)."
        )
    # `notify` is the advertised interface (true → notify_on_complete,
    # [...] → watch_patterns); the legacy args stay accepted, explicit
    # `notify` wins. Background-only modifiers on a foreground call fail
    # with the corrected call instead of being silently ignored.
    notify = args.get("notify")
    notify_on_complete = args.get("notify_on_complete", False)
    watch_patterns = args.get("watch_patterns")
    if not args.get("background", False):
        if notify or watch_patterns or notify_on_complete:
            return tool_error(
                "notify only applies to background commands (foreground "
                "results return directly). Either drop notify, or run as "
                "terminal(command=..., background=true, notify=...)."
            )
        if args.get("pty", False):
            return tool_error(
                "pty requires background=true (a PTY session is interacted "
                "with via process(action='write'/'submit'), which needs a "
                "tracked background process). Retry as terminal(command=..., "
                "background=true, pty=true)."
            )
    if notify is not None:
        if isinstance(notify, bool):
            notify_on_complete = notify
            watch_patterns = None
        elif isinstance(notify, list):
            watch_patterns = notify
            notify_on_complete = False
        else:
            return tool_error(
                "notify must be true/false (notify on exit) or a list of "
                "strings (notify on output pattern match)."
            )
    return terminal_tool(
        command=args.get("command"),
        background=args.get("background", False),
        timeout=args.get("timeout"),
        task_id=kw.get("task_id"),
        session_id=kw.get("session_id"),
        workdir=args.get("workdir"),
        pty=args.get("pty", False),
        notify_on_complete=notify_on_complete,
        watch_patterns=watch_patterns,
    )


registry.register(
    name="terminal",
    toolset="terminal",
    schema=TERMINAL_SCHEMA,
    handler=_handle_terminal,
    check_fn=check_terminal_requirements,
    emoji="💻",
    max_result_size_chars=100_000,
)

