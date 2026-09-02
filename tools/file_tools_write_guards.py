"""Write-side safety guards for write_file / patch.

Every guard returns ``None`` when the write may proceed, else an error string
the tool returns verbatim. ``tools.file_tools`` re-imports every name here.

Guards (in the order the tools apply them):
  * ``_check_sensitive_path``            — system paths + the Hermes config.yaml (hard deny).
  * ``_check_binary_document_write``     — text write would corrupt an Office/PDF container.
  * ``_check_protected_instruction_write`` — AGENTS.md-style files: ALWAYS ask, no yolo bypass.
  * ``_check_approval_required_write``   — ~/.ssh/config-style files: normal approval gate.
  * ``_check_cross_profile_path``        — sandbox-mirror writes the host never reads (lost-work guard).
  * ``_is_internal_file_tool_content``   — refuse to persist read_file display text as a file.
"""

import fnmatch
import os
from pathlib import Path

from tools.binary_extensions import has_opaque_document_extension, is_pdf_path
from tools.file_tools_paths import _expand_tilde, _resolve_path_for_task

# Prefixes matched after realpath. macOS: /private/var mirrors /var — block the
# sensitive subtrees only; a blanket "/private/var/" refuses every temp-file
# write because $TMPDIR, /tmp and /var/folders all realpath there.
_SENSITIVE_PATH_PREFIXES = (
    "/etc/", "/boot/", "/usr/lib/systemd/",
    "/private/etc/",
    "/private/var/db/", "/private/var/root/",
)
_SENSITIVE_EXACT_PATHS = {"/var/run/docker.sock", "/run/docker.sock"}

_hermes_config_resolved: str | None = None
_hermes_config_resolved_loaded = False


def _get_hermes_config_resolved() -> str | None:
    """Return the resolved absolute path of the Hermes config file (cached)."""
    global _hermes_config_resolved, _hermes_config_resolved_loaded
    if _hermes_config_resolved_loaded:
        return _hermes_config_resolved
    _hermes_config_resolved_loaded = True
    try:
        from hermes_cli.config import get_config_path
        _hermes_config_resolved = str(get_config_path().resolve())
    except Exception:
        try:
            _hermes_config_resolved = str(Path(_expand_tilde("~/.hermes/config.yaml")).resolve())
        except Exception:
            _hermes_config_resolved = None
    return _hermes_config_resolved


def _resolved_or_raw(filepath: str, task_id: str) -> str:
    """Task-resolved path string, falling back to the raw input on resolution failure."""
    try:
        return str(_resolve_path_for_task(filepath, task_id))
    except (OSError, ValueError):
        return filepath


def _check_sensitive_path(filepath: str, task_id: str = "default") -> str | None:
    """Return an error message if the path targets a sensitive system location."""
    resolved = _resolved_or_raw(filepath, task_id)
    normalized = os.path.normpath(_expand_tilde(filepath))
    _err = (
        f"Refusing to write to sensitive system path: {filepath}\n"
        "Use the terminal tool with sudo if you need to modify system files."
    )
    for prefix in _SENSITIVE_PATH_PREFIXES:
        if resolved.startswith(prefix) or normalized.startswith(prefix):
            return _err
    if resolved in _SENSITIVE_EXACT_PATHS or normalized in _SENSITIVE_EXACT_PATHS:
        return _err
    # approvals.mode and other security settings live in config.yaml; a
    # prompt-injected agent could silently disable exec approval by editing it.
    hermes_config = _get_hermes_config_resolved()
    if hermes_config and (resolved == hermes_config or normalized == hermes_config):
        return (
            f"Refusing to write to Hermes config file: {filepath}\n"
            "Agent cannot modify security-sensitive configuration. "
            "Edit ~/.hermes/config.yaml directly or use 'hermes config' instead."
        )
    return None


# ---------------------------------------------------------------------------
# Protected agent-instruction files (always-ask approval gate)
# ---------------------------------------------------------------------------
# Files that steer FUTURE agent behavior are a prompt-injection persistence
# vector: an injected edit to AGENTS.md / CLAUDE.md / SOUL.md / .cursorrules
# (or a project-local .hermes tree) outlives the turn and poisons every later
# session. Writes ALWAYS require human approval — even under --yolo — and fail
# closed when no human channel exists. (Ported from Roo-Code's
# RooProtectedController; the terminal-tool vector is gated separately.)
#
# Basenames match in ANY directory (instruction files load from cwd trees) and
# case-insensitively (case-insensitive filesystems; loaders probe variants).
_PROTECTED_INSTRUCTION_BASENAMES = frozenset({
    "agents.md", "claude.md", "soul.md", ".cursorrules",
})

_real_hermes_home_cached: str | None = None
_real_hermes_home_loaded = False


def _get_real_hermes_home() -> str | None:
    """Return the realpath of the authoritative Hermes home (cached)."""
    global _real_hermes_home_cached, _real_hermes_home_loaded
    if _real_hermes_home_loaded:
        return _real_hermes_home_cached
    _real_hermes_home_loaded = True
    try:
        from hermes_constants import get_hermes_home
        _real_hermes_home_cached = os.path.realpath(str(get_hermes_home()))
    except Exception:
        try:
            _real_hermes_home_cached = os.path.realpath(_expand_tilde("~/.hermes"))
        except Exception:
            _real_hermes_home_cached = None
    return _real_hermes_home_cached


def _protected_instruction_config() -> tuple[bool, list[str]]:
    """Return ``(enabled, extra_patterns)`` from ``security.protected_instruction_files`` /
    ``security.protected_instruction_extra_patterns`` (fnmatch on basename).

    Config read failures keep the gate ON — fail-safe for a security boundary.
    """
    try:
        from hermes_cli.config import load_config, cfg_get
        cfg = load_config()
        enabled = cfg_get(cfg, "security", "protected_instruction_files",
                          default=True)
        extra = cfg_get(cfg, "security", "protected_instruction_extra_patterns",
                        default=[])
    except Exception:
        return True, []
    if not isinstance(enabled, bool):
        enabled = True
    if not isinstance(extra, list):
        extra = []
    return enabled, [str(p) for p in extra if p]


def _protected_instruction_reason(filepath: str, task_id: str = "default",
                                  *, enabled: bool | None = None,
                                  extra_patterns: list[str] | None = None) -> str | None:
    """Return a short label when ``filepath`` targets a protected instruction file, else ``None``.

    Matches BOTH the normalized input and its realpath so neither a symlink
    pointing AT a protected file nor a protected name that is itself a symlink
    escapes; ``..`` traversal is neutralized by normpath/realpath first.
    """
    if enabled is None or extra_patterns is None:
        enabled, extra_patterns = _protected_instruction_config()
    if not enabled:
        return None

    normalized = os.path.normpath(_expand_tilde(filepath))
    try:
        resolved = os.path.realpath(str(_resolve_path_for_task(filepath, task_id)))
    except (OSError, ValueError, RuntimeError):
        resolved = os.path.realpath(normalized)

    # ~/.hermes itself is governed by its own guards (config.yaml hard-block,
    # mirror guard, write_approval); this gate targets PROJECT-LOCAL files only.
    # Must run before the ``.hermes`` component rule, which would match the home.
    real_home = _get_real_hermes_home()
    if real_home and (resolved == real_home
                      or resolved.startswith(real_home + os.sep)):
        return None

    for candidate in (normalized, resolved):
        base = os.path.basename(candidate)
        base_lower = base.lower()
        if base_lower in _PROTECTED_INSTRUCTION_BASENAMES:
            return base
        for pattern in extra_patterns:
            if fnmatch.fnmatch(base_lower, pattern.lower()):
                return base
        # Project-local .hermes config dirs (<repo>/.hermes/config.yaml) steer
        # behavior too. Only the IMMEDIATE parent counts — matching any ancestor
        # would gate every write inside a checkout living under ~/.hermes.
        parts = candidate.replace("\\", "/").rstrip("/").split("/")
        if len(parts) >= 2 and parts[-2] == ".hermes":
            return candidate
    return None


def _request_protected_instruction_approval(
        reasons: list[str], task_id: str = "default") -> str | None:
    """Ask the human to approve a write to protected instruction file(s).

    Returns ``None`` when approved, else a BLOCKED error string. Deliberately
    NOT routed through ``_run_approval_gate``: that honors --yolo and
    session/permanent allowlists, and this gate is one-operation approval EVERY
    time with no persisted scope. Fail-closed when no human channel exists.
    """
    targets = ", ".join(dict.fromkeys(reasons))
    description = (
        f"Write to protected agent-instruction file(s): {targets}. "
        "These files steer future agent behavior; approval is always "
        "required (not bypassed by auto-approve)."
    )
    display = f"<write to {targets}>"
    blocked = (
        f"BLOCKED: write to protected agent-instruction file(s) ({targets}) "
        "{why} The user has NOT consented to this write. Do NOT retry it or "
        "attempt the same edit via another path (terminal, execute_code, "
        "etc.)."
    )
    timed_out = blocked.format(
        why="approval prompt timed out without a user response. "
            "Silence is not consent.")
    denied = blocked.format(why="was denied by the user.")

    try:
        import tools.approval as _approval
    except Exception:
        return blocked.format(why="requires approval but the approval "
                                  "subsystem is unavailable.")

    # Gateway surface: block on the button round-trip when a notify callback
    # is registered for this session. One-operation only — no scope buttons.
    session_key = _approval.get_current_session_key()
    notify_cb = None
    try:
        with _approval._lock:
            notify_cb = _approval._gateway_notify_cbs.get(session_key)
    except Exception:
        notify_cb = None

    if notify_cb is not None:
        approval_data = {
            "command": display,
            "pattern_key": "protected_instruction_file",
            "pattern_keys": ["protected_instruction_file"],
            "description": description,
            "allow_permanent": False,
            "allow_session": False,
        }
        decision = _approval._await_gateway_decision(
            session_key, notify_cb, approval_data, surface="gateway",
        )
        if decision.get("notify_failed"):
            return blocked.format(
                why="requires approval but the approval request could not "
                    "be delivered.")
        choice = decision.get("choice")
        # Any tapped scope is a one-operation grant; nothing is persisted.
        if decision.get("resolved") and choice in {"once", "session", "always"}:
            return None
        if not decision.get("resolved"):
            return timed_out
        return denied

    # CLI surface: per-thread approval callback (prompt_toolkit panel).
    callback = None
    try:
        from tools.terminal_tool import _get_approval_callback
        callback = _get_approval_callback()
    except Exception:
        callback = None

    if callback is not None:
        choice = _approval.prompt_dangerous_approval(
            display, description,
            allow_permanent=False,
            allow_session=False,
            approval_callback=callback,
        )
        if choice in {"once", "session", "always"}:
            return None
        if choice == "timeout":
            return timed_out
        return denied

    # No human channel (script, cron, background thread): fail closed —
    # auto-approving here would recreate the persistence vector.
    return blocked.format(
        why="requires approval but no interactive user or gateway is "
            "present to approve it.")


def _check_protected_instruction_write(paths: list[str],
                                       task_id: str = "default") -> str | None:
    """Gate a write/patch touching protected instruction files.

    ONE protected file gates the ENTIRE multi-file patch: a single prompt lists
    every protected target and a deny applies nothing (atomic all-or-nothing
    beats a partially-applied patch).
    """
    enabled, extra = _protected_instruction_config()
    if not enabled:
        return None
    reasons = [
        r for r in (
            _protected_instruction_reason(p, task_id, enabled=enabled, extra_patterns=extra)
            for p in paths
        ) if r
    ]
    if not reasons:
        return None
    return _request_protected_instruction_approval(reasons, task_id)


def _check_approval_required_write(paths: list[str],
                                   task_id: str = "default") -> str | None:
    """Gate a write/patch touching an approval-required path (``~/.ssh/config``).

    Not credentials and not hard-denied, but they can steer process execution
    (SSH ``ProxyCommand`` / ``Match exec``). Unlike the protected-instruction
    gate this is a routine user edit: the prompt offers once/session/always and
    honors --yolo. Fail-closed when no interactive/gateway channel exists.
    """
    try:
        from agent.file_safety import is_write_approval_required
    except Exception:
        return None

    targets = [p for p in paths if is_write_approval_required(p)]
    if not targets:
        return None

    display_targets = ", ".join(dict.fromkeys(targets))
    description = (
        f"Write to SSH client config file(s): {display_targets}. "
        "The SSH config can carry ProxyCommand / Match exec directives that "
        "run commands, so writes require your approval."
    )
    blocked = (
        f"BLOCKED: write to SSH config file(s) ({display_targets}) "
        "{why} Do NOT retry it via another path (terminal, execute_code) "
        "without the user's explicit consent."
    )

    try:
        import tools.approval as _approval
    except Exception:
        return blocked.format(why="requires approval but the approval "
                                  "subsystem is unavailable.")

    result = _approval._run_approval_gate(
        pattern_key="ssh_config_write",
        description=description,
        display_target=f"<write to {display_targets}>",
        cron_deny_message=blocked.format(
            why="requires approval but this cron session denies it."),
        single_query_deny_message=blocked.format(
            why="requires approval but single-query (-q) sessions run "
                "without a user present to approve it. To allow flagged "
                "actions in single-query mode, set approvals.single_query_mode: "
                "approve in config.yaml."),
        autoapprove_log_prefix="ssh_config_write",
        fail_closed_when_no_human=True,
        no_human_block_message=blocked.format(
            why="requires approval but no interactive user or gateway is "
                "present to approve it."),
    )
    if result.get("approved"):
        return None
    return result.get("message") or blocked.format(why="was denied.")


def _get_container_mirror_prefix_for_task(task_id: str = "default") -> str | None:
    """Return the container-side Hermes mirror prefix for persistent Docker file tools."""
    try:
        from tools.terminal_tool import (
            _active_environments,
            _env_lock,
            _get_env_config,
            _resolve_container_task_id,
        )

        container_key = _resolve_container_task_id(task_id)
    except Exception:
        return None

    try:
        with _env_lock:
            env = _active_environments.get(container_key) or _active_environments.get(task_id)

        if env is not None:
            if env.__class__.__name__ == "DockerEnvironment" and bool(
                getattr(env, "_persistent", False)
            ):
                return "/root/.hermes"
            return None

        config = _get_env_config()
    except Exception:
        return None

    if config.get("env_type") == "docker" and config.get("container_persistent", True):
        return "/root/.hermes"
    return None


def _check_cross_profile_path(filepath: str, task_id: str = "default") -> str | None:
    """Soft-guard: warn when ``filepath`` lands on a host-side or Docker sandbox
    MIRROR of Hermes state — a write the host process never reads (lost work).

    Not profile isolation: the former cross-PROFILE guard was removed by
    maintainer decision (profiles were never isolated). ``cross_profile=True``
    on the tools still bypasses these mirror guards (name kept for replay compat).
    Fails open on import error — the sensitive-path guard and denylist still apply.
    """
    try:
        from agent.file_safety import (
            get_container_mirror_warning,
            get_sandbox_mirror_warning,
        )
    except Exception:
        return None

    resolved = _resolved_or_raw(filepath, task_id)

    warning = get_sandbox_mirror_warning(resolved)
    if warning is not None:
        return warning

    return get_container_mirror_warning(
        resolved,
        mirror_prefix=_get_container_mirror_prefix_for_task(task_id),
    )


def _check_binary_document_write(filepath: str, task_id: str = "default") -> str | None:
    """Reject text-tool writes that would corrupt a binary document.

    ``read_file`` auto-extracts Office/PDF to text, so the model plausibly
    believes it holds the file's bytes and writes edited text back — which can
    never form a valid container. Opaque formats (.docx/.xlsx/.pptx/.odt/...)
    are always rejected; .pdf only when OVERWRITING an existing regular file
    (raw PDF syntax is text-authorable, so new-file creation stays allowed).
    """
    if has_opaque_document_extension(filepath):
        ext = filepath[filepath.rfind("."):].lower()
        return (
            f"Refusing to write plain text to binary document '{filepath}' ({ext}). "
            "A text write cannot produce a valid document container and would "
            "corrupt the file (read_file showed you EXTRACTED text, not the real "
            "bytes). Use the docx/xlsx/powerpoint skills or a library like "
            "python-docx/openpyxl/python-pptx via the terminal to create or edit "
            "this document."
        )
    if is_pdf_path(filepath):
        try:
            resolved = Path(_resolve_path_for_task(filepath, task_id))
        except Exception:
            resolved = Path(_expand_tilde(filepath))
        try:
            if resolved.is_file():
                return (
                    f"Refusing to overwrite existing PDF '{filepath}' with plain text. "
                    "read_file showed you EXTRACTED text, not the real bytes — writing "
                    "text back would destroy the document. Use the pdf skill or a PDF "
                    "library via the terminal to modify it. (Creating a NEW .pdf file "
                    "is allowed.)"
                )
        except OSError:
            pass
    return None


# ---------------------------------------------------------------------------
# Internal display text must never be persisted as file content
# ---------------------------------------------------------------------------
_READ_DEDUP_STATUS_MESSAGE = (
    "File unchanged since last read. The content from "
    "the earlier read_file result in this conversation is "
    "still current — refer to that instead of re-reading."
)


def _is_internal_file_status_text(content: str) -> bool:
    """True when content is the read_file dedup status message (verbatim or lightly framed).

    Models echo the message verbatim OR wrap it with short framing ("Note:",
    a trailing comment). Any write whose stripped body contains the full
    message and is <=2x its length is status-dominated — a real file quoting
    this message would be dramatically longer.
    """
    if not isinstance(content, str):
        return False
    stripped = content.strip()
    if not stripped:
        return False
    if stripped == _READ_DEDUP_STATUS_MESSAGE:
        return True
    return (_READ_DEDUP_STATUS_MESSAGE in stripped
            and len(stripped) <= 2 * len(_READ_DEDUP_STATUS_MESSAGE))


def _looks_like_read_file_line_numbered_content(content: str) -> bool:
    """True for content dominated by read_file's ``LINE_NUM|CONTENT`` display.

    Rejects writes whose non-empty lines are mostly (>=60%) consecutive
    numbered lines, while allowing sparse literal pipe content such as a
    single ``1|value`` line.
    """
    if not isinstance(content, str):
        return False

    lines = [line for line in content.splitlines() if line.strip()]
    if len(lines) < 2:
        return False

    numbered: list[int] = []
    for line in lines:
        prefix, sep, _rest = line.lstrip().partition("|")
        if sep and prefix.isdigit():
            numbered.append(int(prefix))

    if len(numbered) < 2 or len(numbered) / len(lines) < 0.6:
        return False

    consecutive_pairs = sum(
        1 for prev, current in zip(numbered, numbered[1:])
        if current == prev + 1
    )
    return consecutive_pairs >= len(numbered) - 1


def _is_internal_file_tool_content(content: str) -> bool:
    """Return True when content is file-tool display text, not intended file bytes."""
    return (
        _is_internal_file_status_text(content)
        or _looks_like_read_file_line_numbered_content(content)
    )
