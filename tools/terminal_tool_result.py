"""Foreground result post-processing for the terminal tool.

Everything that happens between ``env.execute`` returning and the JSON
result leaving the tool: session-cwd dual-write, sudo failure handling,
output transform hook, truncation, ANSI strip, redaction, exit-code notes
and failure hints, spill-file redaction, verification evidence. Also owns
the exit-code interpretation tables. Split out of tools/terminal_tool.py;
lazy ``from tools.terminal_tool import ...`` keeps the origin module's
monkeypatch points authoritative.
"""

import json
import logging
import os
import re
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("tools.terminal_tool")


# Signal-death notes keyed by signum, used for both ``-signum`` (subprocess)
# and ``128+signum`` (shell) encodings. Curated, not exhaustive, so a
# legitimate application exit code is never mislabeled; 130/SIGINT is owned
# by the executor's interrupt-marker path and excluded.
_SIGNAL_EXIT_NOTES: dict[int, str] = {
    3:  "SIGQUIT (quit from keyboard)",
    4:  "SIGILL (illegal instruction — corrupt binary or wrong architecture)",
    6:  "SIGABRT (abort — assertion failure, fatal runtime error, or glibc abort)",
    7:  "SIGBUS (bus error — misaligned or unmapped memory access)",
    8:  "SIGFPE (fatal arithmetic error, e.g. integer division by zero)",
    9:  "SIGKILL — often the kernel OOM killer on memory exhaustion, "
        "or an explicit kill -9",
    11: "SIGSEGV (segmentation fault — the program crashed)",
    13: "SIGPIPE (wrote to a closed pipe — e.g. output piped to a reader that exited)",
    15: "SIGTERM (terminated — kill/timeout or shutdown requested it to stop)",
    24: "SIGXCPU (CPU time limit exceeded)",
    25: "SIGXFSZ (file size limit exceeded)",
}


def _interpret_signal_exit(exit_code: int) -> str | None:
    """Note for a signal-termination exit code, or None. Negative codes are
    definite (subprocess semantics); 128+signum is the shell convention and a
    program *can* exit 139 itself, so those notes hedge with "usually"."""
    if exit_code < 0:
        signum = -exit_code
        if signum == 2:  # SIGINT — executor's interrupt-marker path owns it
            return None
        note = _SIGNAL_EXIT_NOTES.get(signum)
        if note:
            return f"Command terminated by signal {signum}: {note}"
        try:
            import signal as _signal
            name = _signal.Signals(signum).name
        except (ValueError, ImportError):
            name = f"signal {signum}"
        return f"Command terminated by {name} (signal {signum})"

    if exit_code > 128:
        signum = exit_code - 128
        note = _SIGNAL_EXIT_NOTES.get(signum)
        if note:
            return (
                f"Exit code {exit_code} usually means the command was "
                f"terminated by signal {signum}: {note}"
            )

    return None


_NO_MATCH = "No matches found (not an error)"
_FILES_DIFFER = "Files differ (expected, not an error)"
_COND_FALSE = "Condition evaluated to false (expected, not an error)"

# Informational non-zero exit codes per base command.
_EXIT_CODE_SEMANTICS: dict[str, dict[int, str]] = {
    "grep": {1: _NO_MATCH},
    "egrep": {1: _NO_MATCH},
    "fgrep": {1: _NO_MATCH},
    "rg": {1: _NO_MATCH},
    "ag": {1: _NO_MATCH},
    "ack": {1: _NO_MATCH},
    "diff": {1: _FILES_DIFFER},
    "colordiff": {1: _FILES_DIFFER},
    "find": {1: "Some directories were inaccessible (partial results may still be valid)"},
    "test": {1: _COND_FALSE},
    "[": {1: _COND_FALSE},
    "curl": {
        6: "Could not resolve host",
        7: "Failed to connect to host",
        22: "HTTP response code indicated error (e.g. 404, 500)",
        28: "Operation timed out",
    },
    "git": {1: "Non-zero exit (often normal — e.g. 'git diff' returns 1 when files differ)"},
}


def _interpret_exit_code(command: str, exit_code: int) -> str | None:
    """Note for a non-zero exit code that is informational rather than an
    error (grep=1 "no matches", diff=1 "files differ", signal deaths), so the
    model doesn't burn turns investigating it. None when 0 or a real error.
    """
    if exit_code == 0:
        return None

    signal_note = _interpret_signal_exit(exit_code)
    if signal_note is not None:
        return signal_note

    # The last command of a pipeline/chain determines the exit code.
    segments = re.split(r'\s*(?:\|\||&&|[|;])\s*', command)
    last_segment = (segments[-1] if segments else command).strip()

    # Base command name: first word that isn't a VAR=val assignment, basename'd.
    base_cmd = ""
    for w in last_segment.split():
        if "=" in w and not w.startswith("-"):
            continue
        base_cmd = w.split("/")[-1]
        break

    return _EXIT_CODE_SEMANTICS.get(base_cmd, {}).get(exit_code)


def finalize_foreground_result(
    *,
    command: str,
    result: dict,
    env: Any,
    env_type: str,
    effective_task_id: str,
    task_id: Optional[str],
    session_id: Optional[str],
    session_key: str,
    workdir: Optional[str],
    command_cwd: Optional[str],
    approval_note: Optional[str],
) -> str:
    """Turn a raw ``env.execute`` result into the tool's JSON result string."""
    from tools.terminal_tool import (
        _get_sudo_password_callback,
        _handle_sudo_failure,
        _in_delegated_child_context,
        _invalidate_cached_sudo_on_auth_failure,
        _sudo_wrong_password_failure,
        env_var_enabled,
        record_session_cwd,
    )

    # Record the cwd this command finished in as THIS session's durable cwd —
    # but only when the command actually reported it (an interrupted/killed
    # command emits no marker, and env.cwd then holds another session's
    # directory), and never for a transient per-command ``workdir``, which
    # would hijack the session cwd for every later command.
    observed_cwd = None
    if (result or {}).get("cwd_observed"):
        # Prefer the result's own cwd; env.cwd is shared mutable compat state
        # kept as fallback for third-party providers.
        observed_cwd = (result or {}).get("cwd") or getattr(env, "cwd", None)
    if not workdir and observed_cwd:
        record_session_cwd(session_key, observed_cwd)

    output = result.get("output", "")
    returncode = result.get("returncode", 0)
    # Spill metadata: present only when output overflowed the capture window.
    spill_total_chars = result.get("output_total_chars")
    spill_file_path = result.get("full_output_path")

    output = _handle_sudo_failure(output, env_type)

    sudo_auth_failed = _sudo_wrong_password_failure(output)
    sudo_cache_cleared = _invalidate_cached_sudo_on_auth_failure(
        command, output
    )
    if sudo_cache_cleared:
        has_sudo_prompt_callback = _get_sudo_password_callback() is not None
        can_reprompt = (
            has_sudo_prompt_callback or env_var_enabled("HERMES_INTERACTIVE")
        ) and not _in_delegated_child_context()
        if can_reprompt:
            output += (
                "\n\n⚠️ Sudo authentication failed — cached password "
                "cleared. You will be prompted again on the next sudo "
                "command."
            )

    # Plugin output-transform seam (fail-open; first string result wins).
    # Replacements are still subject to the output limit below.
    try:
        from hermes_cli.lifecycle import invoke_hook
        hook_results = invoke_hook(
            "transform_terminal_output",
            command=command,
            output=output,
            returncode=returncode,
            task_id=effective_task_id or "",
            env_type=env_type,
        )
        for hook_result in hook_results:
            if isinstance(hook_result, str):
                output = hook_result
                break
    except Exception:
        pass

    # Truncate keeping head (errors often appear early) and tail (most recent).
    from tools.tool_output_limits import get_max_bytes
    MAX_OUTPUT_CHARS = get_max_bytes()
    if len(output) > MAX_OUTPUT_CHARS:
        head_chars = int(MAX_OUTPUT_CHARS * 0.4)
        tail_chars = MAX_OUTPUT_CHARS - head_chars
        omitted = len(output) - head_chars - tail_chars
        truncated_notice = (
            f"\n\n... [OUTPUT TRUNCATED - {omitted} chars omitted "
            f"out of {len(output)} total] ...\n\n"
        )
        output = output[:head_chars] + truncated_notice + output[-tail_chars:]

    # Strip ANSI so the model never copies escapes into file writes.
    from tools.ansi_strip import strip_ansi
    output = strip_ansi(output)

    # Redact secrets; redact_terminal_output is command-aware (env-dump
    # commands get the KEY=value pass, source/config dumps skip it).
    from agent.redact import redact_terminal_output
    output = redact_terminal_output(output.strip(), command) if output else ""

    exit_note = _interpret_exit_code(command, returncode)

    # Recovery hints for well-known failure shapes (tools/terminal_hints.py);
    # on rc=0, warn when a pipeline tail / `|| echo` may mask an upstream
    # failure and the output carries strong failure indicators (advisory only).
    failure_hint = None
    if returncode != 0 and not exit_note:
        try:
            from tools.terminal_hints import annotate_failure
            failure_hint = annotate_failure(command, returncode, output)
        except Exception:
            failure_hint = None
    elif returncode == 0:
        try:
            from tools.terminal_hints import annotate_masked_success
            failure_hint = annotate_masked_success(command, output)
        except Exception:
            failure_hint = None

    result_dict = {
        "output": output,
        "exit_code": returncode,
        "error": None,
    }
    # cwd echo when the command changed directory (gated on the observation
    # flag above so an interrupted command can't echo another session's cwd).
    try:
        if observed_cwd and command_cwd and os.path.realpath(str(observed_cwd)) != os.path.realpath(str(command_cwd)):
            result_dict["cwd"] = str(observed_cwd)
    except Exception:
        pass
    # Spill handle so the model can read the omitted middle instead of
    # re-running. The collector wrote it raw; redact it with the same pass
    # so no secret persists unmasked on disk.
    if spill_file_path:
        try:
            _sp = Path(spill_file_path)
            raw_spill = _sp.read_text(encoding="utf-8", errors="replace")
            from tools.spill_safety import write_text_exclusive

            # lstat-checked unlink + exclusive create: the redacted copy can't
            # be diverted through a symlink planted since the collector's write.
            write_text_exclusive(
                _sp,
                redact_terminal_output(strip_ansi(raw_spill), command),
                private=True,
                overwrite=True,
                errors="replace",
            )
            result_dict["output_total_chars"] = spill_total_chars
            result_dict["full_output_path"] = spill_file_path
            result_dict["truncation_note"] = (
                "Output exceeded the capture window (head+tail shown). "
                f"Full output ({spill_total_chars:,} chars) saved to "
                f"{spill_file_path} — search it with search_files or page it "
                "with read_file instead of re-running the command."
            )
        except Exception:
            logger.debug("spill redaction failed; dropping spill handle", exc_info=True)
            try:
                Path(spill_file_path).unlink()
            except OSError:
                pass
    try:
        from agent.verification_evidence import record_terminal_result

        evidence = record_terminal_result(
            command=command,
            cwd=command_cwd,
            session_id=session_id or task_id or effective_task_id or "default",
            exit_code=returncode,
            output=output,
        )
        if evidence:
            result_dict["verification_evidence"] = {
                "status": evidence.get("status"),
                "kind": evidence.get("kind"),
                "scope": evidence.get("scope"),
                "canonical_command": evidence.get("canonical_command"),
            }
    except Exception:
        logger.debug("verification evidence recording failed", exc_info=True)
    if approval_note:
        # rc=130 is an interrupt only with the executor's marker — a command
        # can legitimately `exit 130` itself. An interrupted approved run keeps
        # the audit note but must never imply success.
        if returncode == 130 and "[Command interrupted]" in output:
            result_dict["approval"] = approval_note.rstrip(".") + ", then interrupted."
        else:
            result_dict["approval"] = approval_note
    if exit_note:
        result_dict["exit_code_meaning"] = exit_note
    if failure_hint:
        result_dict["hint"] = failure_hint
    if sudo_auth_failed:
        result_dict["sudo_auth_failed"] = True
    if sudo_cache_cleared:
        result_dict["sudo_cache_cleared"] = True

    return json.dumps(result_dict, ensure_ascii=False)
