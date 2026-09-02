"""Gateway lifecycle guard for cron job creation.

A cron job that restarts/stops the gateway from inside the gateway (``hermes gateway restart``,
``launchctl kickstart ai.hermes.gateway``, ``systemctl restart hermes-gateway``) kills the process,
the supervisor revives it, auto-resume re-runs the turn, and a SIGTERM-respawn loop results.
This module rejects such specs in ``cron.jobs.create_job`` (every creation path: CLI and the
``cronjob`` tool). Patterns are command-shaped — anchored on concrete command identifiers — so
they cannot fire on prose. Defence-in-depth layer.
"""

from __future__ import annotations

import logging
import os
import re
import shlex
import stat
from pathlib import Path
from typing import Callable, Iterator, Optional

logger = logging.getLogger(__name__)


class GatewayLifecycleBlocked(ValueError):
    """Raised when a cron job spec contains a gateway-lifecycle command."""


# Shell-level command shapes that target the gateway lifecycle; each branch is anchored on a
# concrete command identifier so it fires only on command-shaped strings, never prose.
_GATEWAY_LIFECYCLE_PATTERN = re.compile(
    r"(?i)"
    # Branch A: destructive `hermes gateway` ops (restart/stop/uninstall). `start` is excluded:
    # starting from inside a gateway is benign, and a job may legitimately start a sibling profile.
    # Lookbehind: `hermes` must not be a path component or word tail (excluding `/`, word chars, `.`,
    # `-`), so `/docs/hermes gateway restart-notes.md` does not match while every real command
    # position (text start, whitespace, `;`/`&`/`|`, `$(`, backtick, U+FFFD) still does.
    r"(?:(?<![/\w.\-])hermes\s+gateway\s+(?:restart|stop|uninstall)\b)"
    # Branch B: launchctl ops anchored on a hermes-gateway label so unrelated hermes services
    # (`launchctl unload ai.hermes.update-checker.plist`) stay unblocked. `submit`/`bootstrap` are
    # included because they register a NEW keepalive job wrapping an arbitrary helper — a blocked
    # restart laundered into a persistent loop; neutral-label submissions are caught separately by
    # `contains_launchctl_submit_command`. `bootout`/`remove`/`disable` are the modern/legacy/durable
    # forms of `unload`; without them the bypassable approval layer (skipped on force=True) was the
    # only cover.
    r"|(?:launchctl\s+(?:kickstart|unload|load|stop|restart|submit|bootstrap|bootout|remove|disable)\b[^\n]*\bhermes[.\-]?gateway)"
    # Branch C: systemctl ops on a hermes-gateway unit.
    r"|(?:systemctl\s+(?:-\S+\s+)*(?:restart|stop|start)\b[^\n]*\bhermes[.\-]?gateway)"
    # Branch D: pkill/kill of the gateway process, both token orders. Leading \b keeps "skill" from
    # matching as "kill".
    r"|(?:\bp?kill\b[^\n]*\bhermes\b[^\n]*\bgateway)"
    r"|(?:\bp?kill\b[^\n]*\bgateway\b[^\n]*\bhermes)"
)


# Backslash-newline is a POSIX line continuation. Every branch uses `[^\n]*` between verb and label so
# matches cannot span unrelated lines, which would let a continuation-split invocation slip past.
# Collapse continuations to a space before matching (as the shell does) rather than loosening `[^\n]*`.
_SHELL_LINE_CONTINUATION = re.compile(r"\\\r?\n[ \t]*")

# Python argv-list punctuation (`subprocess.run(["launchctl", "bootout", ...])`) separates exec'd words
# with brackets/commas. Stripped only for the token-join re-scan, never from raw text (prose stays
# governed by the primary pattern).
_ARGV_LIST_PUNCTUATION = re.compile(r"[\[\],]+")


# Branch A2: `hermes -p <profile> gateway restart|stop` (also `--profile <name>` / `--profile=<name>`).
# The selector breaks Branch A's adjacency. Not unconditionally self-targeting — a sibling-profile
# restart is a legitimate fleet operation — so the profile name is captured and
# `contains_gateway_lifecycle_command` blocks only when it equals the profile running the guard.
# `start` stays excluded as in Branch A.
_PROFILE_FLAG_LIFECYCLE_PATTERN = re.compile(
    r"(?i)"
    r"hermes\s+"
    # Any global flags before the profile selector (each may carry a value).
    r"(?:-{1,2}\S+(?:\s+\S+)?\s+)*"
    # The selector: exactly the shapes the CLI's `_apply_profile_override` accepts.
    r"(?:--profile=([^\s]+)|(?:-p|--profile)\s+([^\s]+))"
    # Any global flags between the selector and the subcommand.
    r"(?:\s+-{1,2}\S+(?:\s+\S+)?)*"
    r"\s+gateway\s+(?:restart|stop)"
)


def _current_profile_name() -> Optional[str]:
    """Profile running the guard: ``HERMES_PROFILE_NAME``/``HERMES_PROFILE`` env first, then
    ``hermes_cli.profiles.get_active_profile_name`` (from ``HERMES_HOME``); ``None`` if neither."""
    for env_name in ("HERMES_PROFILE_NAME", "HERMES_PROFILE"):
        value = os.environ.get(env_name)
        if value and value.strip():
            return value.strip()
    try:
        from hermes_cli.profiles import get_active_profile_name

        return get_active_profile_name() or None
    except Exception:
        return None


def _named_profile_is_current(named: str) -> bool:
    """True when *named* is the profile executing the guard (self-targeting)."""
    current = _current_profile_name()
    if not current:
        # No profile identity: cannot prove self-targeting, so do not block (sibling restarts stay allowed).
        return False
    return named.strip().casefold() == current.strip().casefold()


# Branch B needs the label AFTER the verb in one `[^\n]*` span. A loop that builds the label in an
# EARLIER `;`-segment (`label=${item%%:*}; launchctl bootout "gui/$uid/$label"`) leaves only the
# unexpanded `$label` next to the verb. These verbs act on an EXISTING job, so anchoring to the
# hermes-gateway label stays correct (unrelated labels must stay unblocked) — but the check is
# "verb anywhere AND label anywhere", not "label right after verb".
_LAUNCHCTL_LIFECYCLE_VERBS_RE = re.compile(
    r"(?i)\blaunchctl\s+(?:kickstart|unload|load|stop|restart|bootout|kill|disable|remove)\b"
)
_HERMES_GATEWAY_LABEL_RE = re.compile(r"(?i)\bhermes[.\-]?gateway\b")


def _contains_launchctl_gateway_lifecycle(normalized_text: str) -> bool:
    """Order-independent companion to Branch B — see comment above."""
    return bool(_LAUNCHCTL_LIFECYCLE_VERBS_RE.search(normalized_text)) and bool(
        _HERMES_GATEWAY_LABEL_RE.search(normalized_text)
    )


def contains_gateway_lifecycle_command(text: str) -> bool:
    """Return True if *text* contains a gateway lifecycle command pattern.

    Two passes: the raw-text regex (cheap; the only pass that fires on inputs shlex cannot tokenize,
    e.g. Python source), then the same pattern on each shell-tokenized segment where quotes/escapes
    are resolved — closing splice bypasses like ``kick"start"`` / ``kick\\start`` that execute as
    ``kickstart``. Single choke point for every recursion level of ``_contains_unsafe_gateway_action``.
    """
    if not text:
        return False
    # Provably inert heredoc bodies (quoted delimiter, data-sink consumer like `cat > f <<'EOF'`) are
    # documentation, not commands, and are masked first. The stripper fails open on ANY ambiguity
    # (unquoted delimiter, shell consumer, unterminated body), so executable heredocs are still scanned.
    from tools.shell_heredoc import strip_inert_heredoc_bodies

    text = strip_inert_heredoc_bodies(text)
    normalized = _SHELL_LINE_CONTINUATION.sub(" ", text)
    if _GATEWAY_LIFECYCLE_PATTERN.search(normalized):
        return True
    # Profile-flag form (`hermes -p <profile> gateway restart|stop`): blocked only when the named
    # profile IS the one running the guard — sibling-profile restarts are legitimate and stay allowed.
    profile_match = _PROFILE_FLAG_LIFECYCLE_PATTERN.search(normalized)
    if profile_match:
        named = profile_match.group(1) or profile_match.group(2)
        if named:
            # Profile ids cannot contain quotes (`^[a-z0-9][a-z0-9_-]{0,63}$`), so a shell-quoted
            # `-p 'zeus'` compares equal to the bare name.
            named = named.strip().strip("\"'")
            if _named_profile_is_current(named):
                return True
    # Token-aware second pass: quotes/escapes resolved, closing splice bypasses like `kick"start"`.
    # Runs after the profile-flag check so both apply independently. Tokens are also re-joined with
    # Python argv-list punctuation stripped, since `subprocess.run(["launchctl", "bootout", ...])`
    # separates argv words with commas/brackets rather than spaces.
    for segment in _iter_command_segments(normalized):
        joined = " ".join(segment)
        if joined and _GATEWAY_LIFECYCLE_PATTERN.search(joined):
            return True
        stripped = _ARGV_LIST_PUNCTUATION.sub(" ", joined)
        if stripped != joined and _GATEWAY_LIFECYCLE_PATTERN.search(stripped):
            return True
    # Order-independent launchctl pass: the label may be built in an earlier `;`-segment, so neither
    # the same-span regex nor same-segment tokenization sees verb and label together.
    return _contains_launchctl_gateway_lifecycle(normalized)


_SHELL_EXECUTABLES = frozenset({"sh", "bash", "dash", "ksh", "zsh"})
_SHELL_OPTIONS_WITH_VALUES = frozenset({"-O", "+O", "-o", "+o"})
_MAX_REFERENCED_SCRIPT_BYTES = 1024 * 1024
_MAX_REFERENCED_SCRIPT_DEPTH = 8
_CONTROL_CHARS = frozenset(";&|()")


# Directory names directly under `Library` that mark a FileProvider-backed subtree: `Mobile Documents`
# is iCloud Drive; `CloudStorage` hosts third-party providers (Dropbox, OneDrive, Google Drive, ...).
_CLOUD_PLACEHOLDER_MARKERS = frozenset({"Mobile Documents", "CloudStorage"})


def _resolve_lenient(path: Path) -> Path:
    """``path.resolve(strict=False)``, falling back to *path* on OSError (unreadable/long) or
    ValueError (embedded NUL from decoded binary tokenized as a path) — never crash the guard."""
    try:
        return path.resolve(strict=False)
    except (OSError, ValueError):
        return path


def _on_cloud_path(path: Path) -> bool:
    """Lexical OR resolved cloud check: covers direct cloud paths and local symlinks into one."""
    return _is_cloud_placeholder_path(path) or _is_cloud_placeholder_path(_resolve_lenient(path))


def _is_cloud_placeholder_path(path: Path) -> bool:
    """True for paths inside a macOS FileProvider-backed subtree.

    ``O_NONBLOCK`` does not make regular-file reads non-blocking, so opening an evicted placeholder
    can wait indefinitely for hydration — and the guard runs before any command timeout starts. The
    boundary must be identified from the path alone and fail closed without opening the file.
    """
    parts = path.parts
    return any(
        parts[index - 1] == "Library" and part in _CLOUD_PLACEHOLDER_MARKERS
        for index, part in enumerate(parts)
        if index
    )

# Executables whose arguments are DATA (search patterns, SQL, log filters) and cannot execute their
# argument text, so a lifecycle-shaped string there is diagnostics, not a command. Deliberately
# conservative: no `awk` (system()), no `sed` (`s///e`), no `echo`/`printf` (routinely piped into a
# shell), no `mysql` (`\\!` and `system` escapes).
_DATA_SINK_EXECUTABLES = frozenset(
    {"grep", "egrep", "fgrep", "rg", "ag", "ack", "journalctl", "sqlite3", "psql"}
)
# Argument shapes that smuggle execution back INTO a data sink (command/process substitution, psql
# `\!`). Any hit disables masking for the whole segment — fail closed to the plain regex verdict.
_UNSAFE_DATA_ARG_MARKERS = ("`", "$(", "<(", ">(", "\\!")
# sqlite3 dot-commands (`.shell`, `.system`) also disable masking. Dot must be followed by a NAME
# character so relative paths (`.`, `./x`) stay paths — `grep -r <pattern> .` is far more common.
_DOT_COMMAND_ARGUMENT = re.compile(r"^\.[A-Za-z]")
# A data sink piped into a shell/interpreter can feed matched lines to execution; never mask such a line.
_PIPE_TO_INTERPRETER = re.compile(
    r"\|\s*&?\s*(?:sudo\s+)?(?:sh|bash|dash|ksh|zsh|xargs|eval|source)\b"
)

# Bytes sniffed before reading a referenced file in full (see _BINARY_MAGICS).
_BINARY_SNIFF_BYTES = 4096

_ReadRemoteScriptFn = Callable[[str], Optional[str]]


def _split_logical_lines(text: str) -> list[str]:
    """Split on newlines outside quotes (a quoted newline is data, not a separator); honors escapes."""
    lines = []
    current = []
    in_single = False
    in_double = False
    escape = False

    for ch in text:
        if escape:
            current.append(ch)
            escape = False
            continue
        if ch == "\\":
            escape = True
            current.append(ch)
            continue
        if ch == "'" and not in_double:
            in_single = not in_single
            current.append(ch)
            continue
        if ch == '"' and not in_single:
            in_double = not in_double
            current.append(ch)
            continue
        if ch == "\n" and not in_single and not in_double:
            lines.append("".join(current))
            current = []
            continue
        current.append(ch)

    if current:
        lines.append("".join(current))
    return lines


def _shlex_tokens(line: str) -> list[str]:
    """POSIX-tokenize one shell line, honoring quotes and `#` comments; raises ValueError."""
    lexer = shlex.shlex(line, posix=True, punctuation_chars=";&|()")
    lexer.whitespace_split = True
    lexer.commenters = "#"
    return list(lexer)


def _split_segments(tokens: list[str], *, keep_controls: bool = False) -> Iterator[list[str]]:
    """Yield non-empty runs of *tokens* between control-operator tokens; with *keep_controls* each
    control token is also yielded as its own segment so the line can be rebuilt in order."""
    segment: list[str] = []
    for token in tokens:
        if token and set(token) <= _CONTROL_CHARS:
            if segment:
                yield segment
                segment = []
            if keep_controls:
                yield [token]
            continue
        segment.append(token)
    if segment:
        yield segment


def _iter_command_segments(command: str) -> Iterator[list[str]]:
    """Yield shell-tokenized command segments, honoring quotes and comments.

    Split on logical lines (newlines outside quotes), shlex-tokenize each; on failure (unbalanced
    quotes) fall back to per-physical-line tokenization for that logical line.
    """
    normalized = command.replace("\\\n", "")
    for line in _split_logical_lines(normalized):
        try:
            tokens = _shlex_tokens(line)
        except ValueError:
            for physical_line in line.splitlines():
                try:
                    yield from _split_segments(_shlex_tokens(physical_line))
                except ValueError:
                    continue
            continue
        yield from _split_segments(tokens)
def _executable_name(token: str) -> str:
    """Return the command name for a tokenized executable token.

    ``Path(token).name`` is "" for ``.``, ``..`` and ``/``; the POSIX dot-source builtin is spelled
    ``.``, so fall back to the raw token or ``. ./helper.sh`` would escape the sourced-script scan.
    """
    return Path(token).name or token


# Wrappers that hand execution to their argument tail: the real command sits further right, so a
# first-token-only guard would let `sudo bash ~/restart.sh` or `sudo launchctl submit ...` walk past.
_TRANSPARENT_COMMAND_PREFIXES = frozenset({
    "sudo", "doas", "env", "nohup", "setsid", "nice", "ionice", "stdbuf",
    "timeout", "exec", "command", "builtin", "eatmydata",
    # Privilege and namespace wrappers: options, then the command they run.
    "pkexec", "su", "runuser", "setpriv", "systemd-run", "nsenter", "unshare",
})

# Wrapper options that consume the NEXT token, so a value is never mistaken for the command.
_TRANSPARENT_PREFIX_VALUE_OPTIONS = {
    "sudo": {"-u", "-g", "-U", "-C", "-p", "-r", "-t", "-T",
             "--user", "--group", "--prompt"},
    "doas": {"-u", "-C"},
    "env": {"-u", "--unset", "-S", "--split-string", "-C", "--chdir"},
    "nice": {"-n", "--adjustment"},
    "ionice": {"-c", "-n", "-p", "--class", "--classdata"},
    "stdbuf": {"-i", "-o", "-e", "--input", "--output", "--error"},
    "timeout": {"-s", "-k", "--signal", "--kill-after"},
    "pkexec": {"--user"},
    "su": {"-s", "--shell", "-g", "--group", "-G", "--supp-group"},
    "runuser": {"-u", "--user", "-s", "--shell", "-g", "--group",
                "-G", "--supp-group"},
    "setpriv": {"--reuid", "--regid", "--groups", "--inh-caps",
                "--ambient-caps", "--bounding-set", "--selinux-label",
                "--apparmor-profile"},
    "systemd-run": {"-u", "--unit", "-p", "--property", "-E", "--setenv",
                    "--slice", "--description", "--uid", "--gid",
                    "--on-calendar", "--service-type"},
    "nsenter": {"-t", "--target", "-S", "--setuid", "-G", "--setgid",
                "-r", "--root", "-w", "--wd"},
    "unshare": {"--map-user", "--map-group", "--setgroups", "-R", "--root",
                "-w", "--wd"},
}

# Wrapper options carrying a COMMAND STRING (shell source, re-scanned like `sh -c`); treating it as
# an opaque value would hide whatever it runs (`env -S 'bash ~/restart.sh'`).
_STRING_COMMAND_OPTIONS = {
    "env": ("-S", "--split-string"),
    "su": ("-c", "--command"),
    "runuser": ("-c", "--command"),
}

# Wrappers whose first non-option operand is a VALUE, not the command (`timeout 60 bash x.sh`).
_TRANSPARENT_PREFIX_OPERANDS = {"timeout": 1}

_ENV_ASSIGNMENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*=")

# Bound the walk: a pathological token run must not spin here.
_MAX_PREFIX_PEELS = 8


def _peel_transparent_prefixes(segment: list[str], index: int) -> int:
    """Index of the command a wrapper chain actually executes. Unchanged if not a wrapper; may be
    ``len(segment)`` when a wrapper has no operand — callers must bounds-check."""
    for _ in range(_MAX_PREFIX_PEELS):
        if index >= len(segment):
            return index
        name = _executable_name(segment[index])
        if name not in _TRANSPARENT_COMMAND_PREFIXES:
            return index
        value_options = _TRANSPARENT_PREFIX_VALUE_OPTIONS.get(name, frozenset())
        index += 1
        while index < len(segment):
            token = segment[index]
            if token == "--":
                # POSIX end-of-options: the command starts at the next token.
                index += 1
                break
            if token in value_options:
                index += 2
                continue
            if token.startswith("-") or _ENV_ASSIGNMENT.match(token):
                index += 1
                continue
            break
        for _ in range(_TRANSPARENT_PREFIX_OPERANDS.get(name, 0)):
            if index < len(segment) and not segment[index].startswith("-"):
                index += 1
    return index


def _command_token_index(segment: list[str]) -> Optional[int]:
    """Return the executable token index after simple env assignments."""
    for index, token in enumerate(segment):
        if _ENV_ASSIGNMENT.match(token):
            continue
        return index
    return None


def _executed_command_index(segment: list[str]) -> Optional[int]:
    """Index of the command a segment actually executes (env assignments and wrappers peeled)."""
    index = _command_token_index(segment)
    if index is None:
        return None
    index = _peel_transparent_prefixes(segment, index)
    return index if index < len(segment) else None


def contains_launchctl_submit_command(command: str) -> bool:
    """Detect an executed ``launchctl submit``/``bootstrap``, not quoted text.

    Label-independent by design: a NEW job's label is attacker-chosen, so a neutral name defeats any
    label-anchored regex. Both verbs register a persistent launchd job (KeepAlive / arbitrary plist),
    never safe from inside the gateway.
    """
    for segment in _iter_command_segments(command):
        index = _executed_command_index(segment)
        if index is not None and _executable_name(segment[index]) == "launchctl":
            arguments = segment[index + 1 :]
            if arguments and arguments[0].lower() in {"submit", "bootstrap"}:
                return True
    return False


def _mask_data_sink_arguments(text: str) -> str:
    """Replace data-sink executables' arguments with a neutral placeholder.

    The regex cannot tell an EXECUTED lifecycle command from the same characters as *data* (a grep
    pattern, a SQL literal). For segments whose executable is in ``_DATA_SINK_EXECUTABLES`` every
    argument becomes ``arg``; a match that survives masking is a real command. Strictly fail-closed:
    masking is skipped when the line pipes into a shell/interpreter, any argument carries an
    execution marker, or the line cannot be tokenized. Masking can only ever ALLOW a command the plain
    regex would block, never block one it would allow — so it runs solely as a second-pass exemption.
    """
    lines_out: list[str] = []
    changed = False
    for line in text.splitlines() or [text]:
        if _PIPE_TO_INTERPRETER.search(line):
            lines_out.append(line)
            continue
        try:
            tokens = _shlex_tokens(line)
        except ValueError:
            lines_out.append(line)
            continue

        rebuilt: list[str] = []
        for segment in _split_segments(tokens, keep_controls=True):
            index = _command_token_index(segment)
            if index is not None and Path(segment[index]).name in _DATA_SINK_EXECUTABLES:
                arguments = segment[index + 1 :]
                if not any(
                    _DOT_COMMAND_ARGUMENT.match(argument)
                    or any(marker in argument for marker in _UNSAFE_DATA_ARG_MARKERS)
                    for argument in arguments
                ):
                    changed = True
                    rebuilt.extend(segment[: index + 1])
                    rebuilt.extend("arg" for _ in arguments)
                    continue
            rebuilt.extend(segment)
        lines_out.append(" ".join(rebuilt))
    if not changed:
        return text
    return "\n".join(lines_out)


def _lifecycle_command_scan_with_data_exemption(text: str) -> bool:
    """Lifecycle scan exempting matches inside data arguments: cheap regex first (no-match pays
    nothing), then re-scan with data-sink arguments masked; only a surviving match blocks."""
    if not contains_gateway_lifecycle_command(text):
        return False
    normalized = _SHELL_LINE_CONTINUATION.sub(" ", text)
    return contains_gateway_lifecycle_command(_mask_data_sink_arguments(normalized))


def _direct_lifecycle_scan(command: str) -> bool:
    """Pure-string direct scans: lifecycle regex (data-exempted) + submit."""
    return _lifecycle_command_scan_with_data_exemption(command) or contains_launchctl_submit_command(command)


def _expand_candidate_path(candidate: str) -> Optional[Path]:
    """Sanitize a tokenized path candidate at the ingestion boundary.

    Tokens from shlex-splitting arbitrary (possibly binary-decoded) text can carry NUL or junk; each
    downstream ``Path`` op raises a different exception for it (ValueError, RuntimeError when HOME
    is unset under launchd, OSError). Reject once here. ``None`` = not a real path, nothing to scan.
    """
    if not candidate or "\x00" in candidate:
        return None
    try:
        return Path(candidate).expanduser()
    except (ValueError, RuntimeError, OSError):
        return None


def _resolve_terminal_script_path(candidate: str, cwd: Optional[str]) -> Optional[Path]:
    path = _expand_candidate_path(candidate)
    if path is None:
        return None
    if not path.is_absolute():
        try:
            path = Path(cwd or Path.cwd()) / path
        except OSError:
            # Path.cwd() can raise when the process cwd was deleted.
            return None
    return path


def _resolved_or_nothing(candidate: str, cwd: Optional[str]) -> Iterator[Path]:
    """Yield the resolved terminal script path for *candidate*, or nothing when it is not a path."""
    resolved = _resolve_terminal_script_path(candidate, cwd)
    if resolved is not None:
        yield resolved


def _iter_option_values(
    segment: list[str], start: int, option: str
) -> Iterator[str]:
    """Yield values given to *option*, in both ``--opt v`` and ``--opt=v`` form."""
    prefix = option + "="
    for position in range(start + 1, len(segment)):
        token = segment[position]
        if token == option and position + 1 < len(segment):
            yield segment[position + 1]
        elif token.startswith(prefix):
            yield token[len(prefix):]


def _references_at(
    segment: list[str], index: int, cwd: Optional[str]
) -> Iterator[Path]:
    """Yield the scripts the token at *index* executes, if any."""
    if index >= len(segment):
        return
    executable = segment[index]
    executable_name = _executable_name(executable)

    if executable_name in {".", "source"}:
        if len(segment) > index + 1:
            yield from _resolved_or_nothing(segment[index + 1], cwd)
        return

    if executable_name in _SHELL_EXECUTABLES:
        arguments = segment[index + 1 :]
        arg_index = 0
        while arg_index < len(arguments):
            argument = arguments[arg_index]
            if argument == "--":
                arg_index += 1
                break
            if argument in {"-c", "--command"}:
                break
            if argument in _SHELL_OPTIONS_WITH_VALUES:
                arg_index += 2
                continue
            if argument.startswith("-"):
                arg_index += 1
                continue
            break
        if arg_index < len(arguments) and arguments[arg_index] not in {"-c", "--command"}:
            yield from _resolved_or_nothing(arguments[arg_index], cwd)
        return

    # A bare "/" is pathlib's division operator in Python sources, not an executable; resolving it hits
    # the filesystem root and fails the regular-file check, hard-blocking innocent .py scripts.
    if executable.strip("/") and ("/" in executable or executable.endswith((".sh", ".bash", ".zsh"))):
        yield from _resolved_or_nothing(executable, cwd)


def _iter_referenced_shell_scripts(
    command: str,
    *,
    cwd: Optional[str] = None,
) -> Iterator[Path]:
    """Yield scripts executed directly or through a POSIX shell.

    Each segment is read at the original token AND at the peeled wrapper target. Additive on purpose:
    peeling must never REMOVE a reference (a local ``./timeout`` is a script, not the coreutils wrapper).
    """
    for segment in _iter_command_segments(command):
        index = _command_token_index(segment)
        if index is None:
            continue
        yield from _references_at(segment, index, cwd)
        peeled = _peel_transparent_prefixes(segment, index)
        if peeled != index:
            yield from _references_at(segment, peeled, cwd)


def _iter_shell_command_payloads(command: str) -> Iterator[str]:
    """Yield code passed through ``sh|bash|... -c`` for recursive scanning."""
    for segment in _iter_command_segments(command):
        index = _command_token_index(segment)
        if index is None:
            continue
        # Read at the ORIGINAL token: peeling past `su`/`env` would discard the option carrying the command.
        for option in _STRING_COMMAND_OPTIONS.get(
            _executable_name(segment[index]), ()
        ):
            yield from _iter_option_values(segment, index, option)
        index = _executed_command_index(segment)
        if index is None or _executable_name(segment[index]) not in _SHELL_EXECUTABLES:
            continue
        arguments = segment[index + 1 :]
        for arg_index, argument in enumerate(arguments[:-1]):
            if argument in {"-c", "--command"}:
                yield arguments[arg_index + 1]
                break


def _resolve_script_directory(script_path: str) -> Optional[str]:
    """Return the directory *script_path* resolves to, handling relative names."""
    try:
        path = _resolve_script_path(script_path)
        if path is not None and path.is_absolute():
            return str(path.parent)
    except Exception:
        pass
    return None


_BINARY_MAGICS = (
    b"\x7fELF",              # ELF — Linux/BSD executables and shared objects
    b"\xfe\xed\xfa\xce",     # Mach-O 32-bit
    b"\xfe\xed\xfa\xcf",     # Mach-O 64-bit
    b"\xce\xfa\xed\xfe",     # Mach-O 32-bit, byte-swapped
    b"\xcf\xfa\xed\xfe",     # Mach-O 64-bit, byte-swapped
    b"\xca\xfe\xba\xbe",     # Mach-O universal ("fat") binary
    b"MZ",                   # PE/COFF — Windows .exe/.dll
    b"!<arch>",              # static archive (.a)
    b"\x1f\x8b",             # gzip
    b"PK\x03\x04",           # zip (also .jar/.whl/.egg)
)


def _has_binary_magic(data: bytes) -> bool:
    """True when *data* starts with a known compiled-binary signature.

    Deliberately narrower than "contains a NUL": ``bash`` still executes a NUL-bearing script, so a
    padded script must not bypass the scan. A shebang always wins (interpreted, never binary). File
    extensions are not consulted: a suffixless script must still be scanned and fail closed if oversized.
    """
    if data.startswith(b"#!"):
        return False
    return data.startswith(_BINARY_MAGICS)


def _read_referenced_script(path: Path) -> tuple[Optional[str], bool]:
    """Return ``(text, unsafe)`` using bounded, regular-file-only reads.

    Shared choke point for every local script read, so the cloud-placeholder refusal lives here: a
    FileProvider path is never opened — not even to check hydration — because an evicted placeholder's
    ``open()`` can hang preflight. Lexical check covers direct paths; resolved check covers symlinks.
    """
    if _on_cloud_path(path):
        return None, True
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except (OSError, ValueError):
        # OSError: unreadable/missing/over-long. ValueError: embedded NUL in *path* (decoded binary
        # tokenized as a path). Never crash the guard — either is "nothing to scan".
        return None, False
    try:
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            # Directories are not scripts (e.g. `fpath=(~/.docker/completions …)` in ~/.zshrc must not
            # block `source ~/.zshrc`). Devices/sockets stay fail-closed.
            if stat.S_ISDIR(metadata.st_mode):
                return None, False
            return None, True
        # Sniff a small prefix first: compiled binaries (executable magic) are never shell scripts, so
        # skip them WITHOUT reading the rest or feeding decoded garbage into the recursion.
        data = os.read(descriptor, _BINARY_SNIFF_BYTES)
        if _has_binary_magic(data):
            return None, False
        # Read the remainder (bounded); loop because os.read may return short.
        while len(data) <= _MAX_REFERENCED_SCRIPT_BYTES:
            chunk = os.read(
                descriptor, _MAX_REFERENCED_SCRIPT_BYTES + 1 - len(data)
            )
            if not chunk:
                break
            data += chunk
    except OSError:
        return None, False
    finally:
        os.close(descriptor)
    # Identify binaries by MAGIC NUMBER, not by a NUL: `bash` executes a text script straight past an
    # embedded NUL, so a single pad byte must not make the scan skip a file that still runs.
    if _has_binary_magic(data):
        return None, False
    # Size check BEFORE NUL stripping: stripping shrinks the buffer and would let an oversized file
    # slip under the threshold past this fail-closed branch.
    if len(data) > _MAX_REFERENCED_SCRIPT_BYTES:
        return None, True
    if b"\x00" in data:
        data = data.replace(b"\x00", b"")
    return data.decode("utf-8", errors="replace"), False


def _sanitize_remote_script_text(text: Optional[str]) -> tuple[Optional[str], bool]:
    """Apply the local-read contract to text from a ``read_remote_script`` callback.

    Callbacks are untrusted: any backend can return decoded binary or huge output. Mirror
    ``_read_referenced_script``: NUL means binary (nothing to scan, checked first); oversized fails
    closed. Size compares *bytes* (re-encoded, matching the ``head -c`` wire bound): a >1 MiB multibyte
    file truncated at the byte cap decodes to fewer chars, and a char count would scan instead of failing.
    """
    if not text:
        return None, False
    if "\x00" in text:
        return None, False
    if len(text.encode("utf-8", errors="replace")) > _MAX_REFERENCED_SCRIPT_BYTES:
        return None, True
    return text, False


def _contains_unsafe_gateway_action(
    command: str,
    *,
    cwd: Optional[str],
    depth: int,
    visited: set[Path],
    read_remote_script: Optional[_ReadRemoteScriptFn] = None,
) -> bool:
    if _direct_lifecycle_scan(command):
        return True
    if depth >= _MAX_REFERENCED_SCRIPT_DEPTH:
        return True

    for payload in _iter_shell_command_payloads(command):
        if _contains_unsafe_gateway_action(
            payload,
            cwd=cwd,
            depth=depth + 1,
            visited=visited,
            read_remote_script=read_remote_script,
        ):
            return True

    for script_path in _iter_referenced_shell_scripts(command, cwd=cwd):
        # Do not touch a FileProvider path even to discover whether the file is hydrated.
        if _on_cloud_path(script_path):
            return True
        resolved = _resolve_lenient(script_path)
        if resolved in visited:
            continue
        visited.add(resolved)
        script_text, unsafe = _read_referenced_script(script_path)
        if unsafe:
            return True
        if script_text is None and read_remote_script is not None:
            # Local path missing; try the remote backend. Its output crosses the same trust boundary
            # as a local read — sanitize identically (binary skip + size fail-closed).
            script_text, unsafe = _sanitize_remote_script_text(
                read_remote_script(str(script_path))
            )
            if unsafe:
                return True
        if not script_text:
            continue
        # Relative references inside a script resolve against that script's directory, not the cwd.
        script_dir = _resolve_script_directory(str(resolved)) or cwd
        if _contains_unsafe_gateway_action(
            script_text,
            cwd=script_dir,
            depth=depth + 1,
            visited=visited,
            read_remote_script=read_remote_script,
        ):
            return True
    return False


def contains_gateway_lifecycle_command_or_referenced_script(
    command: str,
    *,
    cwd: Optional[str] = None,
    read_remote_script: Optional[_ReadRemoteScriptFn] = None,
) -> bool:
    """Detect lifecycle/submit commands, including bounded nested scripts.

    Total by construction: never raises. Direct scans are pure string ops; the referenced-script walk
    (filesystem, remote backends, shlex on decoded bytes) is best-effort defense-in-depth — an
    unexpected failure is logged and treated as "walk found nothing".
    """
    try:
        # Includes the direct regex/submit scans at depth 0.
        return _contains_unsafe_gateway_action(
            command,
            cwd=cwd,
            depth=0,
            visited=set(),
            read_remote_script=read_remote_script,
        )
    except Exception:
        logger.warning(
            "lifecycle guard referenced-script walk failed; "
            "falling back to direct-scan verdict",
            exc_info=True,
        )
        # Pure string scans of the top-level command — cannot raise.
        try:
            return _direct_lifecycle_scan(command)
        except Exception:
            # If even the data-argument masker fails, fall to raw regex + submit scan so the guard stays total.
            return contains_gateway_lifecycle_command(command) or contains_launchctl_submit_command(command)


def _resolve_script_path(script_path: str) -> Optional[Path]:
    """Resolve a cron ``script`` value the same way the scheduler does.

    ``cron.scheduler`` resolves relative paths under ``<HERMES_HOME>/scripts/``; we MUST mirror it so
    the guard scans the file that will actually run, not a nonexistent relative path.
    """
    from hermes_constants import get_hermes_home

    raw = _expand_candidate_path(script_path)
    if raw is None:
        return None
    if raw.is_absolute():
        return raw
    try:
        return get_hermes_home() / "scripts" / raw
    except (RuntimeError, OSError):
        # get_hermes_home() falls back to Path.home(), which raises when neither HERMES_HOME nor HOME
        # is resolvable (launchd/systemd) — same ingestion contract: nothing to scan.
        return None


def _read_script_for_scanning(script_path: str) -> str:
    """Read a cron script with the bounded scanner. Non-regular/oversized inputs fail closed via a
    lifecycle-shaped sentinel; missing/unreadable paths stay empty so scheduler validation reports them."""
    resolved = _resolve_script_path(script_path)
    if resolved is None:
        return ""
    script_text, unsafe = _read_referenced_script(resolved)
    if unsafe:
        return "hermes gateway restart"
    return script_text or ""


def check_gateway_lifecycle(
    prompt: Optional[str],
    script: Optional[str] = None,
) -> None:
    """Raise ``GatewayLifecycleBlocked`` if *prompt* or *script* contains a gateway-lifecycle command.

    The script is read from disk and concatenated with the prompt so a command cannot slip through
    by being split across the two. Callers let the ``ValueError``-shaped exception propagate.
    """
    combined = prompt or ""
    python_script = False
    if script:
        resolved_script = _resolve_script_path(script)
        # Attribute the refusal correctly: not a known lifecycle command, but a cloud path the guard refuses to open.
        if resolved_script is not None and _on_cloud_path(resolved_script):
            raise GatewayLifecycleBlocked(
                "Blocked: the cron script lives on a cloud-synced path "
                "(iCloud Drive / ~/Library/CloudStorage). Opening an "
                "evicted FileProvider placeholder can hang the guard's "
                "preflight scan indefinitely, so it is refused without "
                "being read. Move the script to a local, non-cloud path "
                "(e.g. ~/.hermes/scripts/) and recreate the job."
            )
        python_script = resolved_script is not None and resolved_script.suffix == ".py"
        script_text = _read_script_for_scanning(script)
        if script_text:
            combined = f"{combined}\n{script_text}"

    if python_script:
        # Python runs via the interpreter, never a POSIX shell, and the shell reference walk is a
        # false-positive generator on Python sources (pathlib "/" resolves to the filesystem root).
        # The regex still scans the full text; non-regular/oversized files still fail closed via the sentinel.
        unsafe = _lifecycle_command_scan_with_data_exemption(combined)
    else:
        script_dir = _resolve_script_directory(script) if script else None
        unsafe = contains_gateway_lifecycle_command_or_referenced_script(
            combined,
            cwd=script_dir,
        )
    if unsafe:
        raise GatewayLifecycleBlocked(
            "Blocked: cron job contains a gateway lifecycle command or persistent "
            "launchctl submit operation. This is blocked to prevent agent-driven "
            "SIGTERM-respawn loops under launchd/systemd supervision "
            "(#30719). Run `hermes gateway restart` from a shell outside "
            "the running gateway instead."
        )
