"""Dangerous-command detection: normalization, tokenizing, and pattern tables.

Pure command classification for :mod:`tools.approval` — no approval state,
config reads, or prompting live here. ``tools.approval`` re-exports every
public and private name so ``from tools.approval import X`` and
``patch("tools.approval.X")`` keep working.
"""

import functools
import logging
import os
import re
import shlex
import tempfile
import unicodedata
from typing import Optional

logger = logging.getLogger("tools.approval")

# Sensitive write targets that should trigger approval even when referenced
# via shell expansions like $HOME or $HERMES_HOME, or by the resolved absolute
# active profile home path such as /home/hermes/.hermes/config.yaml. The
# resolved-absolute form is folded into the ~/.hermes/ patterns at detection
# time by _normalize_command_for_detection() — see the rewrite step there — so
# these static patterns stay free of any import-time path snapshot (which would
# go stale when HERMES_HOME is set after this module is imported, e.g. under the
# hermetic test conftest or any deferred-profile-resolution path).
_SSH_SENSITIVE_PATH = r'(?:~|\$home|\$\{home\})/\.ssh(?:/|$)'
_HERMES_ENV_PATH = (
    r'(?:~\/\.hermes/|'
    r'(?:\$home|\$\{home\})/\.hermes/|'
    r'(?:\$hermes_home|\$\{hermes_home\})/)'
    r'\.env\b'
)
# ~/.hermes/config.yaml IS the security policy: approvals.mode, yolo, and the
# permanent-approval allowlist live here, and the config cache is mtime-keyed
# so a write takes effect mid-session (the agent could flip approvals.mode=off
# and immediately bypass the gate). Pair the write_file/patch deny (file_tools
# _check_sensitive_path) with terminal-side coverage so `sed -i`, `tee`, `>`,
# `cp`, etc. targeting it are gated too — otherwise the deny is unpaired
# theater. Mirrors _HERMES_ENV_PATH; matches the HERMES_HOME override form as
# well as ~/.hermes/.
_HERMES_CONFIG_PATH = (
    r'(?:~\/\.hermes/|'
    r'(?:\$home|\$\{home\})/\.hermes/|'
    r'(?:\$hermes_home|\$\{hermes_home\})/)'
    r'config\.yaml\b'
)
_PROJECT_ENV_PATH = r'(?:(?:/|\.{1,2}/)?(?:[^\s/"\'`]+/)*\.env(?:\.[^/\s"\'`]+)*)'
_PROJECT_CONFIG_PATH = r'(?:(?:/|\.{1,2}/)?(?:[^\s/"\'`]+/)*config\.yaml)'
_SHELL_RC_FILES = (
    r'(?:~|\$home|\$\{home\})/\.'
    r'(?:bashrc|zshrc|profile|bash_profile|zprofile)\b'
)
_CREDENTIAL_FILES = (
    r'(?:~|\$home|\$\{home\})/\.'
    r'(?:netrc|pgpass|npmrc|pypirc)\b'
)
# macOS: /etc, /var, /tmp, /home are symlinks to /private/{etc,var,tmp,home}.
# A command written to target /private/etc/sudoers works identically to
# /etc/sudoers on macOS but bypasses a plain "/etc/" pattern check. Match
# both forms. Inspired by Claude Code 2.1.113's "dangerous path protection".
_MACOS_PRIVATE_SYSTEM_PATH = r'/private/(?:etc|var|tmp|home)/'
# System-config paths that should trigger approval for any write/edit,
# collapsing /etc, its macOS /private/etc mirror, and /etc/sudoers.d/ into
# one shared fragment so new DANGEROUS_PATTERNS stay consistent.
_SYSTEM_CONFIG_PATH = (
    rf'(?:/etc/|{_MACOS_PRIVATE_SYSTEM_PATH})'
)
_SENSITIVE_WRITE_TARGET = (
    rf'(?:{_SYSTEM_CONFIG_PATH}|/dev/sd|'
    rf'{_SSH_SENSITIVE_PATH}|'
    rf'{_HERMES_ENV_PATH}|'
    rf'{_HERMES_CONFIG_PATH}|'
    rf'{_SHELL_RC_FILES}|'
    rf'{_CREDENTIAL_FILES})'
)
_USER_SENSITIVE_WRITE_TARGET = (
    rf'(?:{_SSH_SENSITIVE_PATH}|'
    rf'{_SHELL_RC_FILES}|'
    rf'{_CREDENTIAL_FILES})'
)
_PROJECT_SENSITIVE_WRITE_TARGET = rf'(?:{_PROJECT_ENV_PATH}|{_PROJECT_CONFIG_PATH})'
# Anchor for the cp/mv/install rule, where the sensitive path is only a write
# target when it is the LAST argument (the destination). Requiring end-of-line
# (or a command separator) keeps `cp config.yaml backup.yaml` — config.yaml as
# the SOURCE — out of the deny.
_COMMAND_TAIL = r'(?:\s*(?:&&|\|\||;).*)?$'
# Boundary for stream-write rules (`>`/`>>` redirection and `tee`), where the
# sensitive path is ALWAYS a write target no matter what follows it. We only
# need the path token to END at a shell word boundary — whitespace, a quote, a
# command separator, a redirection operator, or end-of-line.
# Using _COMMAND_TAIL here was too strict: it required the rest of the line to
# be empty or a command separator, so `echo x > .env extra` (extra arg to echo)
# and `echo x > .env # note` (trailing comment) slipped past the deny even
# though the shell still overwrites `.env`. Mirrors the looser system-path
# redirection rule, which never had this restriction.
#
# `#` is deliberately NOT a boundary char: a real trailing comment always has
# whitespace before the `#` (already covered by `\s`), whereas a `#` glued to
# the path is part of the filename. `echo x > .env#backup` writes to the
# distinct file `.env#backup`, not `.env`, so it must stay OUT of the deny —
# the same reasoning that keeps `config.yaml.bak` safe.
_WRITE_TARGET_BOUNDARY = r'(?=[\s;&|<>"\']|$)'

# =========================================================================
# Hardline (unconditional) blocklist
# =========================================================================
#
# Commands so catastrophic they should NEVER run via the agent, regardless
# of --yolo, /yolo, approvals.mode=off, or cron approve mode.  This is a
# floor below yolo: opting into yolo is the user trusting the agent with
# their files and services, not trusting it to wipe the disk or power the
# box off.
#
# Hardline only applies to environments that can actually damage the host
# (local, ssh, container-host cron).  Containerized backends (docker,
# singularity, modal, daytona) already bypass the dangerous-command layer
# because nothing they do can touch the host, so we leave that behavior
# alone.
#
# The list is deliberately tiny — only things with no recovery path:
# filesystem destruction rooted at /, raw block device overwrites, kernel
# shutdown/reboot, and denial-of-service commands that take the host down.
# Recoverable-but-costly operations (git reset --hard, rm -rf /tmp/x,
# chmod -R 777, curl|sh) stay in DANGEROUS_PATTERNS where yolo can pass
# them through — that's what yolo is for.
#
# Inspired by Mercury Agent's permission-hardened blocklist
# (https://github.com/cosmicstack-labs/mercury-agent).

# Regex fragment matching the *start* of a command (i.e. positions where
# a shell would begin parsing a new command).  Used by shutdown/reboot
# patterns so they don't fire on "echo reboot" or "grep 'shutdown' log".
# Matches: start of string, after command separators (; && || | newline),
# after subshell openers ( `$(` or backtick ), optionally consuming
# leading wrapper commands (sudo, env VAR=VAL, exec, nohup, setsid).
_CMDPOS = (
    # Real ;/&/| separators are converted to newlines by the quote-aware
    # _mark_command_starts pass. Keeping them in this flat regex mistakes
    # quoted regex/data (for example grep '(safe|rm -rf /)') for commands.
    r'(?:^|[\n`]|\$\()'            # start position
    r'\s*'                          # optional whitespace
    r'(?:sudo\s+(?:-[^\s]+\s+)*)?'  # optional sudo with flags
    r'(?:env\s+(?:\w+=\S*\s+)*)?'   # optional env with VAR=VAL pairs
    r'(?:(?:exec|nohup|setsid|time)\s+)*'  # optional wrapper commands
    r'\s*'
)

# Destructive-path argument matcher for the rm hardline rules.
#
# The path token in `rm -rf /` is almost always written quoted in real
# shells — `rm -rf "/"`, `rm -rf "$HOME"` — and `${HOME}` is the universal
# brace form. A bare-token anchor (`(/...)(\s|$)`) silently misses all of
# these: the surrounding quote breaks both the leading position (the flag
# group can't consume `"`) and the trailing `(\s|$)` terminator, letting
# `rm -rf "/"` slip past the unconditional floor entirely.
#
# Accept the path either fully wrapped in a matching quote pair OR bare with
# a terminator. The matching-quote branch catches `rm -rf "/"` (path quoted
# on its own). The bare branch's terminator accepts whitespace, end-of-string
# OR a shell metacharacter (`) ` ; | &`) so a real root wipe inside a command
# substitution — `$(rm -rf /)`, `` `rm -rf /` `` — whose `/` is terminated by
# `)`/backtick is still caught.
def _hardline_rm_path(path_alt: str, tail: str = r'(?:\s|$|[)`;|&])') -> str:
    return rf'(?:["\'](?:{path_alt})["\']|(?:{path_alt}){tail})'


# Protected system roots whose recursive deletion has no recovery path.
_HARDLINE_SYSTEM_DIRS = (
    r'/home|/home/\*|/root|/root/\*|/etc|/etc/\*|/usr|/usr/\*|'
    r'/var|/var/\*|/bin|/bin/\*|/sbin|/sbin/\*|/boot|/boot/\*|/lib|/lib/\*'
)

# `rm` plus its flag group, shared by the three rm hardline rules. Kept as a
# plain concatenation (not an f-string) so the regex backslashes never live
# inside an f-string replacement field — unsupported on the Python 3.11 floor.
#
# Anchored to _CMDPOS (start of line, after a command separator ; && || |,
# after a subshell opener $(/backtick, or after sudo/env/exec wrappers) so the
# rule fires only when `rm` is an actual command word — not when the literal
# string "rm -rf /" appears as DATA inside another command's argument, e.g.
# `gh pr create --title "block rm -rf / spellings"` or `git commit -m "…rm -rf
# /…"`. Those tripped the unconditional floor and could not run at all before
# the anchor. A real wipe at any command position (bare, chained, in $()/`…`,
# under sudo) still matches; the quoted-path branch in _hardline_rm_path keeps
# catching `rm -rf "/"`.
_RM_FLAG_PREFIX = _CMDPOS + r'rm\s+(-[^\s]*\s+)*'

HARDLINE_PATTERNS = [
    # rm recursive targeting the root filesystem or protected roots.
    # `${HOME}` brace form and quoted paths (`rm -rf "/"`, `rm -rf "$HOME"`)
    # are handled via _hardline_rm_path so the floor cannot be bypassed with
    # the ordinary quoting/brace shell idioms.
    #
    # The path token matches any root-anchored path whose components collapse
    # back to "/" in the shell: a bare "/", repeated slashes ("//"), and
    # "."/".." current/parent segments ("/.", "/./", "/..", "/../..") all
    # resolve to root, optionally followed by a trailing glob ("/*", "//*").
    # Each inter-slash segment must be exactly "." or "..", so a longer dot
    # run or any real name is a literal directory, NOT root — "/tmp", "/home",
    # "/.ssh", "/.config" and even "/..." (a dir literally named "...") fall
    # through to the softer DANGEROUS_PATTERNS / system-directory rules
    # instead of being unconditionally hardline-blocked. The explicit "/ \*"
    # alt preserves the slash-space-glob spelling (`rm -rf / *`, which the
    # shell sees as two args: "/" plus the "*" glob).
    (_RM_FLAG_PREFIX + _hardline_rm_path(r'/(?:(?:\.\.?)?/)*(?:\.\.?)?\**|/ \*'), "recursive delete of root filesystem"),
    (_RM_FLAG_PREFIX + _hardline_rm_path(_HARDLINE_SYSTEM_DIRS), "recursive delete of system directory"),
    (_RM_FLAG_PREFIX + _hardline_rm_path(r'(?:~|\$\{?HOME\}?)(?:/?|/\*)?'), "recursive delete of home directory"),
    # Filesystem format — anchor to command position like every other
    # hardline entry so quoted prose ("echo \"does this workflow use mkfs
    # anywhere?\"") does not trip the unconditional floor (#93392).
    (_CMDPOS + r'mkfs(\.[a-z0-9]+)?\b', "format filesystem (mkfs)"),
    # Raw block device overwrites (dd + redirection). `dd` is a command-name
    # token, so anchor it to command position like mkfs/rm/shutdown (#93392):
    # quoted prose such as `git commit -m "never dd of=/dev/sda"` is an
    # argument, not a command. The argument tail ([^\n]*of=/dev/...) is kept
    # so flag order doesn't matter.
    (_CMDPOS + r'dd\b[^\n]*\bof=/dev/(sd|nvme|hd|mmcblk|vd|xvd)[a-z0-9]*', "dd to raw block device"),
    # The redirect rule has no command-name token to anchor (`>` appears
    # mid-command: `cat f > /dev/sda`), so command-position anchoring is the
    # wrong tool. It is instead matched against a QUOTE-MASKED variant of the
    # command (see _QUOTE_MASKED_HARDLINE / _mask_quoted_strings) so quoted
    # prose (`echo "cat f > /dev/sda"`) cannot trip it, while shell-carrying
    # wrappers (sh -c / bash -c / eval) still surface their payload as a raw
    # detection variant — quoting is not a bypass (#93392).
    (r'>\s*/dev/(sd|nvme|hd|mmcblk|vd|xvd)[a-z0-9]*\b', "redirect to raw block device"),
    # Fork bomb (classic shell form). Also positionless (the trigger is the
    # function definition itself, valid anywhere in a command line), so it is
    # quote-masked like the redirect rule above rather than _CMDPOS-anchored.
    (r':\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:', "fork bomb"),
    # Kill every process on the system — anchor the command-name token so
    # `echo "kill -1 sends SIGHUP to everything"` doesn't trip (#93392).
    (_CMDPOS + r'kill\s+(-[^\s]+\s+)*-1\b', "kill all processes"),
    # System shutdown / reboot — anchor to command position (start of line,
    # after a command separator, or after sudo/env wrappers) so we don't
    # false-positive on "echo reboot" or "grep 'shutdown' logs".
    # _CMDPOS matches start-of-command positions.
    (_CMDPOS + r'(shutdown|reboot|halt|poweroff)\b', "system shutdown/reboot"),
    (_CMDPOS + r'init\s+[06]\b', "init 0/6 (shutdown/reboot)"),
    (_CMDPOS + r'systemctl\s+(poweroff|reboot|halt|kexec)\b', "systemctl poweroff/reboot"),
    (_CMDPOS + r'telinit\s+[06]\b', "telinit 0/6 (shutdown/reboot)"),
]

# Pre-compiled variant used by the hot-path matcher. Building these at module
# load eliminates the ~2.6 ms cold-cache re.compile fan-out on the first
# terminal() call per process (12 HARDLINE + 47 DANGEROUS patterns, each
# potentially evicted from Python's 512-entry ``re._cache`` by unrelated
# regex work elsewhere in the agent). DANGEROUS_PATTERNS_COMPILED is built
# at the end of this module after DANGEROUS_PATTERNS is defined.
_RE_FLAGS = re.IGNORECASE | re.DOTALL

# Hardline rules whose trigger has no command-name token to anchor (the
# redirect target / fork-bomb definition are valid anywhere in a command
# line). These are matched against QUOTE-MASKED variants of the command so
# quoted prose (`echo "cat f > /dev/sda"`, `git commit -m "fork bomb
# :(){ :|:& };:"`) cannot trip the unconditional floor, while the raw
# payloads of shell-carrying wrappers (sh -c, bash -c, eval) are still
# scanned unmasked — quoting is not a bypass (#93392).
_QUOTE_MASKED_HARDLINE_DESCRIPTIONS = frozenset({
    "redirect to raw block device",
    "fork bomb",
})

HARDLINE_PATTERNS_COMPILED = [
    (
        re.compile(pattern, _RE_FLAGS),
        description,
        description in _QUOTE_MASKED_HARDLINE_DESCRIPTIONS,
    )
    for pattern, description in HARDLINE_PATTERNS
]


# Command names that hand a quoted argument to another shell/parser to
# EXECUTE. For these, quoted text is code, not prose, so the quote-masked
# hardline rules must scan the raw string (see detect_hardline_command).
_SHELL_CARRIER_NAMES = frozenset({
    "eval", "sh", "bash", "zsh", "ksh", "dash", "source", ".",
})


def _contains_shell_carrier(command: str) -> bool:
    """Return whether any command-position word is a shell-carrying command."""
    for _, _, word in _iter_shell_command_word_spans(command):
        name = os.path.basename(
            _deobfuscate_shell_word_for_detection(word)
        ).lower()
        if name in _SHELL_CARRIER_NAMES:
            return True
    return False


def _mask_quoted_prose(command: str) -> str:
    """Blank out quoted string CONTENT for positionless hardline matching.

    Detection-only rewrite used by the quote-masked hardline rules
    (redirect-to-block-device, fork bomb): text inside single or double
    quotes is data the shell passes as an argument, so `echo "cat f >
    /dev/sda"` must not trip the unconditional floor (#93392). Structure is
    preserved: the quote characters themselves stay, and inside double
    quotes `$(...)` command substitutions and backtick spans are kept RAW
    because the shell really executes them (`echo "$(cat f > /dev/sda)"`
    remains a true positive). Unquoted text is untouched. Quote tracking
    mirrors _mask_quoted_newlines; an unclosed quote masks to end-of-string,
    which cannot hide a runnable command (the shell would not run it
    either).
    """
    out: list[str] = []
    quote: str | None = None
    i = 0
    n = len(command)
    while i < n:
        ch = command[i]
        if quote == "'":
            if ch == "'":
                quote = None
                out.append(ch)
            else:
                out.append(" ")
            i += 1
            continue
        if quote == '"':
            if ch == "\\" and i + 1 < n:
                out.append("  ")
                i += 2
                continue
            if ch == '"':
                quote = None
                out.append(ch)
                i += 1
                continue
            if ch == "$" and i + 1 < n and command[i + 1] == "(":
                end = _scan_dollar_paren_end(command, i)
                if end is not None:
                    out.append(command[i:end])
                    i = end
                    continue
            if ch == "`":
                close = command.find("`", i + 1)
                if close != -1:
                    out.append(command[i:close + 1])
                    i = close + 1
                    continue
            out.append(" ")
            i += 1
            continue
        if ch == "\\" and i + 1 < n:
            out.append(command[i:i + 2])
            i += 2
            continue
        if ch in ("'", '"'):
            quote = ch
        out.append(ch)
        i += 1
    return "".join(out)


# =========================================================================
# Sudo stdin guard — block password guessing via "sudo -S"
# =========================================================================
# When SUDO_PASSWORD is not configured, any explicit "sudo -S" in the
# command is the LLM piping a guessed password via stdin.  This is a
# brute-force attack vector: the model iterates through candidate
# passwords, inspects sudo's "Sorry, try again" output, and refines.
# Treat this as an unconditional block — there is never a legitimate
# reason for the agent to pipe passwords to sudo -S when no password
# has been configured.
_SUDO_STDIN_RE = re.compile(
    r'(?:^|[;&|`\n]|&&|\|\||\$\()\s*sudo\s+-S\b',
    re.IGNORECASE)


def _check_sudo_stdin_guard(command: str) -> tuple:
    """Detect ``sudo -S`` (stdin password) without configured SUDO_PASSWORD.

    When SUDO_PASSWORD is set, ``_transform_sudo_command`` injects ``-S``
    internally — that path is legitimate and handled elsewhere.  This guard
    only fires when SUDO_PASSWORD is *not* set, meaning the LLM explicitly
    wrote ``sudo -S`` to pipe a guessed password.

    Returns:
        (is_blocked: bool, description: str | None)
    """
    if "SUDO_PASSWORD" in os.environ:
        return (False, None)
    normalized = _normalize_command_for_detection(command).lower()
    if _SUDO_STDIN_RE.search(normalized):
        return (True, "sudo password guessing via stdin (sudo -S)")
    return (False, None)


def detect_hardline_command(command: str) -> tuple:
    """Check if a command matches hardline blocklist patterns.

    Hardline patterns are NEVER bypassable, even in YOLO mode.

    Returns:
        (is_hardline, description) or (False, None)
    """
    if _command_parser_limit_exceeded(command):
        return (True, _PARSER_LIMIT_DESCRIPTION)
    normalized = _normalize_command_for_detection(command)
    _, malformed_grep = _grep_safe_detection_variant(normalized)
    if malformed_grep:
        return (True, _MALFORMED_EXEC_DESCRIPTION)
    for command_variant in _command_detection_variants(command):
        variant_lower = command_variant.lower()
        masked_lower: str | None = None
        for pattern_re, description, quote_masked in HARDLINE_PATTERNS_COMPILED:
            if quote_masked:
                # Positionless rules (redirect-to-block-device, fork bomb)
                # match a quote-masked variant so quoted prose in echo /
                # git commit -m / gh --body arguments is DATA (#93392).
                # Shell-carrying commands (sh/bash -c, eval, source) hand
                # their quoted argument to another parser, so those scan
                # the raw variant — quoting is not a bypass. bash/sh -c
                # payloads additionally surface as their own raw variants
                # via _execution_flag_findings.
                if masked_lower is None:
                    if _contains_shell_carrier(command_variant):
                        masked_lower = variant_lower
                    else:
                        masked_lower = _mask_quoted_prose(command_variant).lower()
                haystack = masked_lower
            else:
                haystack = variant_lower
            if pattern_re.search(haystack):
                return (True, description)
    return (False, None)


# =========================================================================
# Dangerous command patterns
# =========================================================================

DANGEROUS_PATTERNS = [
    (r'\brm\s+(-[^\s]*\s+)*/', "delete in root path"),
    (r'\brm\s+-[^\s]*r', "recursive delete"),
    (r'\brm\s+--recursive\b', "recursive delete (long flag)"),
    # GNU rm permutes options, so a recursive flag group may legally FOLLOW
    # the operands: `rm build/ -rf`, `rm build/ -r -f`, and `rm build/
    # --recursive --force` are all equivalent to the flags-first spellings the
    # two patterns above catch — without this rule they run with no approval
    # prompt at all. The operand run is tempered: it cannot cross a command
    # separator (`;`, `|`, `&`, newline — so a later pipeline segment's flags,
    # e.g. `rm foo | grep -r bar`, are not attributed to `rm`), cannot cross a
    # quote (so `git commit -m "rm x" --amend` style data can't bridge an `rm`
    # word to an unrelated dash token), and cannot cross a bare ` -- `
    # end-of-options separator (after `--`, POSIX rm treats `-rf` as a literal
    # filename, not flags; guarded both leading and mid-run). The flag token
    # itself must start right after whitespace so the `r` inside long options
    # like `--registry` (preceded by `-`, not whitespace) does not count.
    # Port of openai/codex#33464 ("recognize force options when they follow
    # operands").
    (r'\brm\s+(?!--(?:\s|$))(?:(?!\s--(?:\s|$))[^\n"\';|&])*\s'
     r'(?:-[a-z]*r[a-z]*\b|--recursive\b)',
     "recursive delete (flags after operands)"),
    # Windows shell front-ends have destructive built-ins that do not look like
    # Unix `rm`. Gate only when they are executed through cmd/powershell so
    # ordinary prose or filenames containing "del"/"rd" do not trip the guard.
    (r'\bcmd(?:\.exe)?\s+/(?:c|k)\s+.*\b(?:del|erase|rd|rmdir)\b', "Windows cmd destructive delete"),
    # PowerShell/pwsh: the destructive verb runs as the default positional
    # argument, so `powershell Remove-Item ...` needs NO explicit -Command.
    # Anchor the verb to the command position (right after the shell name,
    # after any leading `-Flag` switches, and optionally after -Command/-c)
    # so bare invocations are caught while a benign path arg containing
    # "del"/"rm" (e.g. `-File c:\del-logs\run.ps1`) is not.
    (r'\b(?:powershell|pwsh)(?:\.exe)?\b(?:\s+-\S+)*\s+(?:-(?:command|c)\s+)?["\']?(?:remove-item|rmdir|erase|del|rd|ri|rm)\b', "Windows PowerShell destructive delete"),
    (r'\b(?:powershell|pwsh)(?:\.exe)?\b.*\s-(?:encodedcommand|enc|e)\b', "PowerShell encoded command execution"),
    # ── Windows destructive tier (#69472) ────────────────────────────────
    # These are native Windows EXEs / cmdlets reachable from ANY Hermes
    # terminal backend on a Windows host — including the default git-bash
    # backend (taskkill.exe, icacls.exe, reg.exe, vssadmin.exe, bcdedit.exe,
    # cipher.exe are ordinary PATH executables there). Detection input is
    # lowercased by the variant loop, so patterns are written lowercase.
    # Each pattern requires the destructive flag/verb so benign usage
    # (`taskkill /IM app.exe` graceful kill, `reg query`, `icacls file`)
    # does NOT prompt.
    # Bare PowerShell destructive delete: Remove-Item/ri with -Recurse or
    # -Force. The cmd/powershell-prefixed forms are covered above; this
    # catches the bare form (ACP clients, pwsh-default SSH hosts, or
    # `powershell` invoked earlier in a compound command).
    (r'\bremove-item\b[^\n;|&]*\s-(?:recurse|force)\b', "PowerShell destructive delete (Remove-Item)"),
    # cmd builtins with destructive switches, bare form: del/erase/rd/rmdir
    # with /s (recurse) or /q (quiet). Requires the switch so `del file.txt`
    # inside a cmd /c string stays covered by the prefixed rule only.
    (r'\b(?:del|erase|rd|rmdir)\s+(?:/[a-z]\s+)*/[sq]\b', "Windows destructive delete (recursive/quiet switch)"),
    # Remote content piped to Invoke-Expression — PowerShell's `curl | sh`.
    (r'\b(?:iwr|invoke-webrequest|invoke-restmethod|irm|curl|wget)\b[^\n]*\|\s*(?:iex|invoke-expression)\b', "pipe remote content to PowerShell (iwr | iex)"),
    (r'\b(?:iex|invoke-expression)\s*\(\s*(?:iwr|invoke-webrequest|invoke-restmethod|irm)\b', "execute remote content via Invoke-Expression"),
    # Force process kills — Windows analogue of pkill -9.
    (r'\btaskkill\b[^\n]*\s/f\b', "force kill processes (taskkill /F)"),
    (r'\bstop-process\b[^\n]*\s-force\b', "force kill processes (Stop-Process -Force)"),
    # Volume/disk destruction — Windows analogue of mkfs / dd.
    (r'\bformat-volume\b', "format filesystem (Format-Volume)"),
    (r'\bclear-disk\b', "wipe disk (Clear-Disk)"),
    (r'\bdiskpart\b', "disk partitioning (diskpart)"),
    (r'\bformat(?:\.com)?\s+[a-z]:', "format drive (format.com)"),
    (r'\bcipher\s+/w\b', "wipe free space (cipher /w)"),
    # ACL destruction — Windows analogue of chmod 777.
    (r'\bicacls\b[^\n]*\s/grant\b[^\n]*\b(?:everyone|todos|jeder|tout\s+le\s+monde|\*s-1-1-0)\b', "grant Everyone access (icacls)"),
    (r'\bicacls\b[^\n]*\s/reset\b', "reset ACLs recursively (icacls /reset)"),
    # Backup/recovery destruction — classic ransomware prep, no benign
    # agent use case.
    (r'\bvssadmin\b[^\n]*\bdelete\s+shadows\b', "delete volume shadow copies (vssadmin)"),
    (r'\bwbadmin\b[^\n]*\bdelete\b', "delete backups (wbadmin)"),
    (r'\bbcdedit\b[^\n]*\s/set\b', "modify boot configuration (bcdedit /set)"),
    # Registry deletion with force flag.
    (r'\breg(?:\.exe)?\s+delete\b', "registry delete (reg delete)"),
    (r'\bremove-itemproperty\b[^\n]*\s-force\b', "registry value delete (Remove-ItemProperty -Force)"),
    # Windows service/system stop — analogue of systemctl stop.
    (r'\bstop-service\b[^\n]*\s-force\b', "force stop service (Stop-Service -Force)"),
    (r'\bsc(?:\.exe)?\s+(?:stop|delete)\b', "stop/delete service (sc)"),
    # Credential/key paths in Windows form — the POSIX ~/.ssh patterns never
    # match drive-letter or backslash spellings. Match both separators.
    (r'\busers[\\/][^\\/\s]+[\\/]\.ssh\b', "access to SSH keys (Windows path)"),
    (r'\bappdata[\\/](?:local|roaming)[\\/]hermes[^\n]*\.env\b', "access to Hermes secrets (Windows path)"),
    # ─────────────────────────────────────────────────────────────────────
    (r'\bchmod\s+(-[^\s]*\s+)*(777|666|o\+[rwx]*w|a\+[rwx]*w)\b', "world/other-writable permissions"),
    (r'\bchmod\s+--recursive\b.*(777|666|o\+[rwx]*w|a\+[rwx]*w)', "recursive world/other-writable (long flag)"),
    (r'\bchown\s+(-[^\s]*)?R\s+root', "recursive chown to root"),
    (r'\bchown\s+--recur[a-z]*\b.*root', "recursive chown to root (long flag)"),
    # Anchored to command position like the hardline twins (#93392):
    # quoted prose mentioning mkfs/dd must not require approval to echo.
    (_CMDPOS + r'mkfs\b', "format filesystem"),
    (_CMDPOS + r'dd\s+.*if=', "disk copy"),
    (r'>\s*/dev/sd', "write to block device"),
    (r'\bDROP\s+(TABLE|DATABASE)\b', "SQL DROP"),
    # Use [^\n]* instead of .* so DOTALL mode does not cause a WHERE clause on the
    # *next* line to satisfy the negative lookahead, silently allowing DELETE without WHERE.
    (r'\bDELETE\s+FROM\b(?![^\n]*\bWHERE\b)', "SQL DELETE without WHERE"),
    (r'\bTRUNCATE\s+(TABLE)?\s*\w', "SQL TRUNCATE"),
    (rf'>\s*{_SYSTEM_CONFIG_PATH}', "overwrite system config"),
    (r'\bsystemctl\s+(-[^\s]+\s+)*(stop|restart|disable|mask)\b', "stop/restart system service"),
    (r'\bkill\s+-9\s+-1\b', "kill all processes"),
    (r'\bpkill\s+-9\b', "force kill processes"),
    # killall with SIGKILL (parallel to pkill -9). Catches -9 / -KILL /
    # -s KILL / -SIGKILL forms, and also `killall -r <regex>` broad sweeps
    # that can wipe out unrelated processes by accident.
    # Inspired by Claude Code 2.1.113 expanded deny rules.
    (r'\bkillall\s+(-[^\s]*\s+)*-(9|KILL|SIGKILL)\b', "force kill processes (killall -KILL)"),
    (r'\bkillall\s+(-[^\s]*\s+)*-s\s+(KILL|SIGKILL|9)\b', "force kill processes (killall -s KILL)"),
    (r'\bkillall\s+(-[^\s]*\s+)*-r\b', "kill processes by regex (killall -r)"),
    (r':\(\)\s*\{\s*:\s*\|\s*:\s*&\s*\}\s*;\s*:', "fork bomb"),
    # Shell -c is parsed structurally by _execution_flag_findings(). A regex
    # that merely searched a dash-token for "c" also matched --norc,
    # --rcfile, and --restricted.
    (r'\b(curl|wget)\b.*\|\s*(?:[/\w]*/)?(?:ba)?sh(?:\s|$|-c)', "pipe remote content to shell"),
    (r'\b(bash|sh|zsh|ksh)\s+<\s*<?\s*\(\s*(curl|wget)\b', "execute remote script via process substitution"),
    # Remote content executed via command substitution: eval/source/. $(curl ...)
    # or `wget ...`. Equivalent to piping remote content to a shell.
    (r'(?:\beval\b|\bsource\b|\.)\s*(?:\$\(\s*|`\s*)(?:curl|wget)\b', "execute remote content via command substitution"),
    # Decode-and-execute: encoded/transformed content piped to a shell. Without
    # these, `echo <base64> | base64 -d | bash` silently runs `rm -rf /` or any
    # other command because the raw text carries no dangerous keywords.
    (r'\b(base64|base32|base16)\s+(?:-[dD]|--decode)\b.*\|\s*\b(bash|sh|zsh|ksh|dash)\b',
     "pipe decoded content to shell (possible command obfuscation)"),
    # xxd reverse hex dump to shell (xxd uses -r for decode, not -d).
    (r'\bxxd\s+-r\b.*\|\s*\b(bash|sh|zsh|ksh|dash)\b',
     "pipe xxd-decoded content to shell (possible command obfuscation)"),
    # Character transformation via tr piped to shell:
    # `echo 'eq -pe v/' | tr 'eqv' 'rmf' | bash` decodes to `rm -rf /`.
    (r'\becho\b[^|]*\|\s*\btr\b[^|]*\|\s*\b(bash|sh|zsh|ksh|dash)\b',
     "pipe tr-transformed output to shell (possible command obfuscation)"),
    # openssl decode piped to shell:
    # `echo <base64> | openssl base64 -d | bash` decodes arbitrary commands.
    (r'\bopenssl\b.*\b(?:base64|enc)\b[^|]*\s+-[dD]\b[^|]*\|\s*\b(bash|sh|zsh|ksh|dash)\b',
     "pipe openssl-decoded content to shell (possible command obfuscation)"),
    (rf'\btee\b.*["\']?{_SENSITIVE_WRITE_TARGET}', "overwrite system file via tee"),
    (rf'>>?\s*["\']?{_SENSITIVE_WRITE_TARGET}', "overwrite system file via redirection"),
    (rf'\btee\b.*["\']?{_PROJECT_SENSITIVE_WRITE_TARGET}["\']?{_WRITE_TARGET_BOUNDARY}', "overwrite project env/config via tee"),
    (rf'>>?\s*["\']?{_PROJECT_SENSITIVE_WRITE_TARGET}["\']?{_WRITE_TARGET_BOUNDARY}', "overwrite project env/config via redirection"),
    (r'\bxargs\s+.*\brm\b', "xargs with rm"),
    # find -exec rm / -execdir rm — the -execdir variant (same semantics,
    # runs in the directory of each match) was previously missed. Claude
    # Code 2.1.113 tightened their equivalent find rule to stop auto-
    # approving -exec / -delete flags.
    (r'\bfind\b.*-exec(?:dir)?\s+(/\S*/)?rm\b', "find -exec/-execdir rm"),
    (r'\bfind\b.*-delete\b', "find -delete"),
    # Gateway lifecycle protection: prevent the agent from killing its own
    # gateway process.  These commands trigger a gateway restart/stop that
    # terminates all running agents mid-work.  Allow global flags between
    # `hermes` and `gateway` (e.g. `hermes -p ade gateway restart`) so a
    # profile flag can't slip the agent past the guard.
    (r'\bhermes\s+(?:-{1,2}\S+(?:\s+\S+)?\s+)*gateway\s+(stop|restart)\b', "stop/restart hermes gateway (kills running agents)"),
    (r'\bhermes\s+update\b', "hermes update (restarts gateway, kills running agents)"),
    # Docker container lifecycle — any user with docker.sock mounted (a common
    # Docker Compose pattern) gives the agent the ability to restart/stop/kill
    # containers without approval.  These are agent-initiated lifecycle operations
    # that should always require user consent, just like `hermes gateway restart`
    # already does for the gateway process.
    # Docker/Podman daemon redirect — global flags or env prefixes that point
    # the CLI at a DIFFERENT daemon, often a remote host over ssh/tcp.  A
    # command that looks local (`docker -H ssh://prod stop app`) silently
    # operates on remote infrastructure, so any docker/podman invocation
    # carrying a redirect requires approval regardless of subcommand.  The
    # redirect flag must appear in the global-flag position (before the
    # subcommand) and -H/--host/--context must carry a value, which keeps
    # `docker -h` (help) and subcommand flags like `docker run -h <hostname>`
    # out of the deny.  Listed BEFORE the lifecycle rules so a redirected
    # lifecycle command surfaces the more specific "remote daemon" reason.
    # Inspired by Claude Code 2.1.214, which added permission prompts for
    # docker/podman commands carrying daemon-redirect flags (--url,
    # --connection, --identity, remote mode).
    (r'\bdocker\s+(?:-{1,2}\S+(?:[=\s]\S+)?\s+)*(?:-h|--host)[=\s]+\S+',
     "docker with remote daemon redirect (-H/--host)"),
    (r'\bdocker\s+(?:-{1,2}\S+(?:[=\s]\S+)?\s+)*(?:-c|--context)[=\s]+\S+',
     "docker with daemon redirect (--context: alternate daemon)"),
    (r'\bdocker\s+context\s+use\b',
     "docker context use (switches default daemon for future commands)"),
    (r'\bpodman\s+(?:-{1,2}\S+(?:[=\s]\S+)?\s+)*(?:--url|--connection|--identity)[=\s]+\S+',
     "podman with remote daemon redirect (--url/--connection/--identity)"),
    (r'\bpodman\s+(?:-{1,2}\S+(?:[=\s]\S+)?\s+)*(?:-r\b|--remote\b)',
     "podman remote mode (-r/--remote: remote daemon)"),
    (r'\b(?:docker_host|docker_context|container_host|container_connection)=\S+',
     "docker/podman daemon redirect via environment (DOCKER_HOST/CONTAINER_HOST)"),
    # Allow global flags between `docker`/`compose` and the verb (e.g.
    # `docker compose -f prod.yml down`, `docker --log-level debug stop app`)
    # and the legacy hyphenated `docker-compose` binary, so a flag can't slip
    # a lifecycle command past the guard — same treatment as the `hermes ...
    # gateway` pattern above.
    (r'\bdocker(?:-compose|\s+compose)\s+(?:-{1,2}\S+(?:[=\s]\S+)?\s+)*(restart|stop|kill|down)\b',
     "docker compose restart/stop/kill/down (container lifecycle)"),
    (r'\bdocker\s+(?:-{1,2}\S+(?:[=\s]\S+)?\s+)*(restart|stop|kill)\b',
     "docker restart/stop/kill (container lifecycle)"),
    # Gateway protection: never start gateway outside systemd management
    (r'gateway\s+run\b.*(&\s*$|&\s*;|\bdisown\b|\bsetsid\b)', "start gateway outside systemd (use 'systemctl --user restart hermes-gateway')"),
    (r'\bnohup\b.*gateway\s+run\b', "start gateway outside systemd (use 'systemctl --user restart hermes-gateway')"),
    # Self-termination protection: prevent agent from killing its own process
    (r'\b(pkill|killall)\b.*\b(hermes|gateway|cli\.py)\b', "kill hermes/gateway process (self-termination)"),
    # Self-termination via kill + command substitution (pgrep/pidof).
    # The name-based pattern above catches `pkill hermes` but not
    # `kill -9 $(pgrep -f hermes)` because the substitution is opaque
    # to regex at detection time. Catch the structural pattern instead.
    # `pidof` is the BSD/Linux alternative to `pgrep` and is equally
    # opaque, so include it in the same alternation.
    (r'\bkill\b.*\$\(\s*(pgrep|pidof)\b', "kill process via pgrep/pidof expansion (self-termination)"),
    (r'\bkill\b.*`\s*(pgrep|pidof)\b', "kill process via backtick pgrep/pidof expansion (self-termination)"),
    # launchctl-driven gateway stop/restart on macOS. The agent can bypass
    # the `hermes gateway stop|restart` pattern above by driving launchd
    # directly against the service label (commonly `ai.hermes.gateway`).
    # Catch the operations that stop, restart, or unload it.
    #
    # Order-independent (2026-08-02 incident): the previous version required
    # "hermes"/"ai.hermes" to appear AFTER the launchctl verb in the same
    # string (`.*` only scans forward). A shell for-loop that builds the
    # label from a list defined earlier in the command — e.g. `for item in
    # 'ai.hermes.gateway-apollo:...' ...; do label=${item%%:*}; launchctl
    # bootout "$label"; done` — never has the literal text "hermes" appear
    # after "bootout" (only the expanded variable does), so it slipped past
    # undetected and restarted 4 gateways with zero approval. Two
    # independent lookaheads instead of one sequential match: both
    # substrings must appear SOMEWHERE in the command, in either order.
    # This is intentionally broader (a launchctl-verb command anywhere near
    # an unrelated "hermes" mention now also matches) — for an approval gate
    # that's the correct direction to err: an extra approval prompt is
    # cheap, a missed one took down the whole gateway fleet.
    (r'(?=[\s\S]*\blaunchctl\s+(?:stop|kickstart|bootout|unload|kill|disable|remove)\b)(?=[\s\S]*\b(?:hermes|ai\.hermes)\b)', "stop/restart hermes launchd service (kills running agents)"),
    # File copy/move/edit into sensitive system paths (/etc/ and macOS
    # /private/etc/ mirror).
    (rf'\b(cp|mv|install)\b.*\s{_SYSTEM_CONFIG_PATH}', "copy/move file into system config path"),
    (rf'\b(cp|mv|install)\b.*\s["\']?{_PROJECT_SENSITIVE_WRITE_TARGET}["\']?{_COMMAND_TAIL}', "overwrite project env/config file"),
    # cp/mv/install OVERWRITING a sensitive credential/SSH/shell-rc/Hermes file.
    # The tee/redirection patterns above already gate _SENSITIVE_WRITE_TARGET
    # (~/.ssh/*, ~/.netrc/.pgpass/.npmrc/.pypirc, shell rc files,
    # ~/.hermes/config.yaml/.env), but cp/mv/install was only paired for /etc and
    # project-relative env/config — so `cp evil ~/.ssh/authorized_keys` (key
    # implant), `cp creds ~/.netrc`, and `cp evil ~/.bashrc` (login-time command
    # injection) slipped through with auto-approve. Same unpaired-door rationale
    # as #14639 / the sed-tee-redirect pairing on these targets.
    # Anchor the sensitive target to the command tail so this fires on the
    # DESTINATION (last arg) only — `cp evil ~/.ssh/authorized_keys` is gated,
    # but reading OUT of a sensitive path (`cp ~/.ssh/config /tmp/x`) stays safe.
    # The trailing `[^\s"\']*` consumes the rest of the destination filename
    # (e.g. `authorized_keys` after the `~/.ssh/` fragment).
    (rf'\b(cp|mv|install)\b.*\s["\']?{_SENSITIVE_WRITE_TARGET}[^\s"\']*["\']?{_COMMAND_TAIL}', "copy/move file into sensitive credential/SSH/shell-rc path"),
    # In-place edits mutate the target file directly, bypassing redirection,
    # tee, and copy/move/install coverage. Gate the same user-controlled
    # startup/credential files so `sed -i ... ~/.bashrc` and `perl -i ...
    # ~/.ssh/authorized_keys` cannot silently plant login commands or keys.
    (rf'\bsed\s+-[^\s]*i.*(?:{_USER_SENSITIVE_WRITE_TARGET})[^\s"\']*', "in-place edit of sensitive credential/SSH/shell-rc path"),
    (rf'\bsed\s+--in-place\b.*(?:{_USER_SENSITIVE_WRITE_TARGET})[^\s"\']*', "in-place edit of sensitive credential/SSH/shell-rc path (long flag)"),
    (rf'\b(?:perl|ruby)\b.*(?:^|\s)-[^\s]*i\b.*(?:{_USER_SENSITIVE_WRITE_TARGET})[^\s"\']*', "in-place edit of sensitive credential/SSH/shell-rc path (perl/ruby)"),
    (rf'\bsed\s+-[^\s]*i.*\s{_SYSTEM_CONFIG_PATH}', "in-place edit of system config"),
    (rf'\bsed\s+--in-place\b.*\s{_SYSTEM_CONFIG_PATH}', "in-place edit of system config (long flag)"),
    # In-place edit of a Hermes-managed security file (~/.hermes/config.yaml or
    # .env). sed -i bypasses the redirection/tee patterns above because it
    # mutates the file directly. Pairs the file_tools write_file/patch deny so
    # the terminal side is not an open door. See #14639.
    (rf'\bsed\s+-[^\s]*i.*(?:{_HERMES_CONFIG_PATH}|{_HERMES_ENV_PATH})', "in-place edit of Hermes config/env"),
    (rf'\bsed\s+--in-place\b.*(?:{_HERMES_CONFIG_PATH}|{_HERMES_ENV_PATH})', "in-place edit of Hermes config/env (long flag)"),
    # perl -i and ruby -i perform the same in-place mutation as sed -i but are
    # not caught by the -e/-c script-execution pattern above (which targets code
    # evaluation, not file mutation). Pairs the sed -i coverage from #14639.
    # The -i flag can appear as its own token after other flags
    # (`perl -p -i -e ... config.yaml`), combined (`perl -pi -e`), or with a
    # backup suffix (`perl -i.bak`). Match any flag token containing `i`
    # anywhere in the args, not just the first token — `perl -e '...'` (code
    # eval, no -i) does not trip because it has no `-...i` flag token.
    (rf'\b(?:perl|ruby)\b.*(?:^|\s)-[^\s]*i\b.*(?:{_HERMES_CONFIG_PATH}|{_HERMES_ENV_PATH})', "in-place edit of Hermes config/env (perl/ruby)"),
    # Interpreter heredocs are handled by _execution_flag_findings() alongside
    # inline-exec flags; keep only shell heredocs regex-based here.
    # Shell execution via heredoc — `bash <<'EOF' ... EOF` runs arbitrary
    # shell commands without triggering the `bash -c` pattern above. The
    # inner commands may not individually match any dangerous pattern (e.g.
    # data-exfiltration pipelines using curl/cat) yet are still executed in
    # a full shell context.
    (r'\b(bash|sh|zsh|ksh)\s+<<', "shell execution via heredoc"),
    # Git destructive operations that can lose uncommitted work or rewrite
    # shared history. Not captured by rm/chmod/etc patterns.
    # `git reset --hard` accepts any unambiguous long-flag prefix (--h,
    # --ha, --har, --hard) because git's own option parser resolves
    # abbreviated long flags -- `--hard` is the only `git reset` mode
    # starting with "h" (siblings are --soft/--mixed/--merge/--keep), so
    # this cannot collide with another reset mode. It also does not match
    # `--help`, which git special-cases before mode resolution.
    (r'\bgit\s+reset\s+--h(?:a(?:r(?:d)?)?)?\b', "git reset --hard (destroys uncommitted changes)"),
    (r'\bgit\s+push\b.*--forc[a-z]*\b', "git force push (rewrites remote history)"),
    (r'\bgit\s+push\b.*-f\b', "git force push short flag (rewrites remote history)"),
    (r'\bgit\s+clean\s+-[^\s]*f', "git clean with force (deletes untracked files)"),
    (r'\bgit\s+branch\s+-D\b', "git branch force delete"),
    # `-D` is shorthand for `-d --force`; the long-flag spellings
    # (`--delete`, `--force`) are different tokens entirely, so they slip
    # past the `-D\b` pattern above even though `git branch -d --force`
    # and `git branch --delete --force` delete an unmerged branch exactly
    # like `-D` does. Match delete+force in either order, bounded to the
    # same command segment (not spanning `;`/`|`/`&`/newline) the same
    # way the sudo patterns below do, to avoid contaminating an unrelated
    # later command in the same script.
    (r'\bgit\s+branch\b[^;|&\n]*?(?:-d\b|--delete\b)[^;|&\n]*?(?:-f\b|--force\b)', "git branch force delete (long flags)"),
    (r'\bgit\s+branch\b[^;|&\n]*?(?:-f\b|--force\b)[^;|&\n]*?(?:-d\b|--delete\b)', "git branch force delete (long flags, force-first)"),
    # Script execution after chmod +x — catches the two-step pattern where
    # a script is first made executable then immediately run. The script
    # content may contain dangerous commands that individual patterns miss.
    (r'\bchmod\s+\+x\b.*[;&|]+\s*\./', "chmod +x followed by immediate execution"),
    # Sudo with stdin / askpass / shell / list-privs flags. An LLM-driven
    # agent has no TTY, so sudo invocations that succeed without human
    # interaction are those reading the password from stdin (-S/--stdin)
    # or via an askpass helper (-A/--askpass). The shell-launch (-s) and
    # list-privileges (-a) flags are also gated since they are
    # privilege-relevant invocations the agent can chain after acquiring
    # the password (e.g. read SUDO_PASSWORD from .env -> sudo -S -s ->
    # root shell). Plain `sudo cmd` (no flag) is TTY-bound and excluded.
    # `_normalize_command_for_detection` lowercases input before pattern
    # matching, so case variants of S/s and A/a collapse — both forms
    # are gated below. Lazy `[^;|&\n]*?` allows flag arguments (e.g.
    # `sudo -u root -S whoami`) without spanning command separators. See
    # #17873 category 4.
    # sudo's own option parser (like git's) resolves unambiguous
    # long-flag prefixes, so `sudo --stdi` runs identically to
    # `sudo --stdin` and `sudo --ask` to `sudo --askpass` -- confirmed
    # against a live sudo binary. `--st[a-z]*` and `--a[a-z]*` are safe
    # to match broadly: per `man sudo`, `--stdin` is the only long option
    # starting with "st" (siblings are --shell/--set-home) and
    # `--askpass` is the only one starting with "a" at all.
    (r'\bsudo\b[^;|&\n]*?\s+(?:-s\b|--st[a-z]*\b|-a\b|--a[a-z]*\b)',
     "sudo with privilege flag (stdin/askpass/shell/list)"),
    # Combined short-flag form: -nS, -ns, -sa, -las — sudo flags packed
    # into a single -X token. Catches the same threat class.
    (r'\bsudo\b[^;|&\n]*?\s+-[a-z]*[sa][a-z]*\b',
     "sudo with combined-flag privilege escalation"),
]


# Pre-compiled variant (same rationale as HARDLINE_PATTERNS_COMPILED above).
DANGEROUS_PATTERNS_COMPILED = [
    (re.compile(pattern, _RE_FLAGS), description)
    for pattern, description in DANGEROUS_PATTERNS
]


def _legacy_pattern_key(pattern: str) -> str:
    """Reproduce the old regex-derived approval key for backwards compatibility."""
    return pattern.split(r'\b')[1] if r'\b' in pattern else pattern[:20]


_PATTERN_KEY_ALIASES: dict[str, set[str]] = {}
for _pattern, _description in DANGEROUS_PATTERNS:
    _legacy_key = _legacy_pattern_key(_pattern)
    _canonical_key = _description
    _PATTERN_KEY_ALIASES.setdefault(_canonical_key, set()).update({_canonical_key, _legacy_key})
    _PATTERN_KEY_ALIASES.setdefault(_legacy_key, set()).update({_legacy_key, _canonical_key})

# Preserve approvals stored under the removed interpreter regex rules.
_REMOVED_PATTERN_KEY_ALIASES = {
    "script execution via -e/-c flag": "(python[23]?|perl|ruby|node)\\s+-[ec]\\s+",
    "script execution via heredoc": "(python[23]?|perl|ruby|node)\\s+<<",
}
for _canonical_key, _legacy_key in _REMOVED_PATTERN_KEY_ALIASES.items():
    _PATTERN_KEY_ALIASES.setdefault(_canonical_key, set()).update(
        {_canonical_key, _legacy_key}
    )
    _PATTERN_KEY_ALIASES.setdefault(_legacy_key, set()).update(
        {_legacy_key, _canonical_key}
    )


def _approval_key_aliases(pattern_key: str) -> set[str]:
    """Return all approval keys that should match this pattern.

    New approvals use the human-readable description string, but older
    command_allowlist entries and session approvals may still contain the
    historical regex-derived key.
    """
    return _PATTERN_KEY_ALIASES.get(pattern_key, {pattern_key})


# =========================================================================
# Detection
# =========================================================================

def _normalize_command_for_detection(command: str) -> str:
    """Normalize a command string before dangerous-pattern matching.

    Strips ANSI escape sequences (full ECMA-48 via tools.ansi_strip),
    null bytes, and normalizes Unicode fullwidth characters so that
    obfuscation techniques cannot bypass the pattern-based detection.
    """
    from tools.ansi_strip import strip_ansi

    # Strip all ANSI escape sequences (CSI, OSC, DCS, 8-bit C1, etc.)
    command = strip_ansi(command)
    # Strip null bytes
    command = command.replace('\x00', '')
    # Normalize Unicode (fullwidth Latin, halfwidth Katakana, etc.)
    command = unicodedata.normalize('NFKC', command)
    # Collapse shell line continuations (backslash-newline). The shell removes
    # BOTH characters and joins the tokens, so `rm -rf \<newline>/` executes as
    # `rm -rf /`. This must run BEFORE the generic backslash-escape strip below,
    # whose [^\n] class deliberately skips newlines and would otherwise leave
    # the dangling backslash wedged between tokens — defeating the structured
    # rm/mkfs/dd patterns (notably the HARDLINE root-delete floor, which cannot
    # be bypassed even with yolo). Handles both \n and \r\n line endings. Line
    # continuations carry no path separator, so this is a no-op on the Windows
    # home-prefix folds below (which match C:\Users\alice\... — no newline).
    command = re.sub(r'\\\r?\n', '', command)
    # Fold absolute home / active-profile-home prefixes into their canonical
    # ~/ and ~/.hermes/ forms so static user-sensitive patterns catch
    # /home/alice/.bashrc and C:\Users\alice\.bashrc the same way they catch
    # ~/.bashrc. Resolve at detection time (not via an import-time snapshot) so
    # it tracks HOME / HERMES_HOME even when those are set after this module is
    # imported — as the hermetic test conftest and profile/session launchers do.
    #
    # This MUST run before the backslash-escape strip below: on Windows the home
    # prefix is separated by backslashes (C:\Users\alice\...), which that strip
    # would otherwise dissolve (-> C:Usersalice) and make the fold impossible.
    # The fold matches either separator, so POSIX paths are unaffected by order.
    #
    # Fold the (more specific) Hermes home first: on Windows it nests under the
    # user home (C:\Users\alice\AppData\...\hermes), so folding the user home
    # first would eat the prefix the Hermes-home fold needs.
    command = _rewrite_resolved_hermes_home(command)
    command = _rewrite_resolved_user_home(command)
    # Strip shell backslash-escapes: r\m → rm. Prevents \-injection bypass.
    command = re.sub(r'\\([^\n])', r'\1', command)
    # Strip empty-string literals that split tokens: r''m → rm, r"\"m → rm.
    command = re.sub(r"''|\"\"", '', command)
    # Collapse $IFS / ${IFS} word-separator expansions to a literal space.
    # In any POSIX shell the IFS variable defaults to <space><tab><newline>,
    # so `rm${IFS}-rf${IFS}/` is executed as `rm -rf /`. Because the dangerous
    # and hardline patterns anchor on literal whitespace (\s) between a command
    # and its arguments, leaving the unexpanded `${IFS}` token in place lets an
    # attacker slip past EVERY pattern — including the unconditional hardline
    # floor (rm -rf /, mkfs, dd to raw device, shutdown/reboot). Substituting a
    # space here mirrors the shell's own expansion so the patterns fire. The
    # brace form also covers bash substring expansions like `${IFS:0:1}` (a
    # single space). Same de-obfuscation class as the backslash/empty-quote
    # handling above.
    command = re.sub(r'\$\{IFS\b[^}]*\}|\$IFS\b', ' ', command)
    return command


# Shell metacharacters, quotes, and whitespace that terminate a filesystem
# path token on a command line. Used to bound the path tail we normalize.
_PATH_TOKEN_STOP = r"""\s'"`;|&<>()"""
# One path segment (no separators, no terminators) preceded by a separator.
_PATH_TAIL = r"(?P<tail>(?:[/\\][^/\\" + _PATH_TOKEN_STOP + r"]*)+)"


@functools.lru_cache(maxsize=64)
def _home_prefix_fold_regex(path: str):
    """Compile a regex matching *path* used as an absolute directory prefix.

    The home components are matched with either separator (``/`` or ``\\``)
    between them, followed by the rest of the path token (the ``tail`` group),
    so a Windows native path (``C:\\Users\\alice\\.ssh\\authorized_keys``), its
    forward-slash form, and mixed-separator forms all fold — and the tail's
    backslashes get normalized to ``/`` by the caller so multi-segment static
    patterns (``~/.ssh/authorized_keys``) still match. The trailing tail is
    required (``+``), so a bare home with no path under it is not folded.

    Returns ``None`` for an unset or degenerate path — one with fewer than two
    components below the root — so a stray HOME / HERMES_HOME such as ``/``,
    ``C:\\`` or ``""`` cannot rewrite unrelated filesystem prefixes. Cached
    because the resolved home is stable across calls on this hot path.
    """
    if not path:
        return None
    components = [c for c in re.split(r"[/\\]+", path) if c]
    # Require at least two non-empty components below the root. For POSIX this
    # mirrors the historical ``count("/") >= 2`` guard (``/home/alice`` folds,
    # ``/home`` does not); for Windows it rejects a bare drive root (``C:\\``)
    # while accepting a real home (``C:\\Users\\alice``).
    if len(components) < 2:
        return None
    body = r"[/\\]+".join(re.escape(c) for c in components)
    # Optional leading root separator (POSIX ``/`` or UNC ``\\``); a Windows
    # drive letter is captured as the first component.
    return re.compile(r"[/\\]*" + body + _PATH_TAIL)


def _fold_home_prefixes(command: str, paths, replacement: str) -> str:
    """Fold each resolved home *path* prefix in *command* to *replacement*.

    *replacement* has no trailing separator (``~`` / ``~/.hermes``); the matched
    path tail (with its backslashes normalized to ``/``) supplies it. Longest
    candidate first so a deeper home (e.g. an explicit HOME under USERPROFILE)
    folds before a shorter overlapping one that would otherwise clobber it.
    """
    seen: set[str] = set()
    for path in sorted((p for p in paths if p), key=len, reverse=True):
        if path in seen:
            continue
        seen.add(path)
        pattern = _home_prefix_fold_regex(path)
        if pattern is not None:
            command = pattern.sub(
                lambda m: replacement + m.group("tail").replace("\\", "/"),
                command,
            )
    return command


def _rewrite_resolved_user_home(command: str) -> str:
    """Rewrite the current user's absolute home prefix to ``~/``.

    Resolves the home at detection time — its expanduser form, symlink-resolved
    form, and an explicitly set ``HOME`` — so absolute home paths are checked by
    the same static patterns as tilde and ``$HOME`` forms. ``HOME`` is consulted
    directly because Windows' ``os.path.expanduser`` resolves ``~`` from
    ``USERPROFILE`` and ignores ``HOME``, unlike POSIX. Matches both POSIX
    (``/home/alice``) and Windows (``C:\\Users\\alice`` or ``C:/Users/alice``)
    separators. No-op when the home is unset or degenerate.
    """
    try:
        home = os.path.expanduser("~")
        candidates = [
            home,
            os.path.realpath(home),
            os.environ.get("HOME", ""),
        ]
    except Exception:
        return command
    return _fold_home_prefixes(command, candidates, "~")


def _rewrite_resolved_hermes_home(command: str) -> str:
    """Rewrite the resolved absolute Hermes home prefix to ``~/.hermes/``.

    Resolves the active ``HERMES_HOME`` at call time (and its symlink-resolved
    form) and folds an occurrence of ``<home>/`` in *command* into
    ``~/.hermes/`` so the static ``_HERMES_CONFIG_PATH`` / ``_HERMES_ENV_PATH``
    patterns match. In Docker and gateway deployments the agent often references
    the resolved absolute path directly (e.g. ``sed -i ...
    /home/hermes/.hermes/config.yaml``) rather than ``~``, ``$HOME``, or
    ``$HERMES_HOME``. Matches both POSIX and Windows separators. No-op when the
    path can't be resolved or doesn't appear.
    """
    try:
        from hermes_constants import get_hermes_home
        home = get_hermes_home().expanduser()
        candidates = [
            str(home),
            str(home.resolve(strict=False)),
        ]
    except Exception:
        return command
    return _fold_home_prefixes(command, candidates, "~/.hermes")


_PARAM_REPLACEMENT_RE = re.compile(r"\$\{[^}/\s]+/[^}/]*/(?P<replacement>[^}]*)\}")
_PARAM_DEFAULT_RE = re.compile(r"\$\{[^}:}\s]+:-(?P<default>[^}]*)\}")
_SIMPLE_SHELL_LITERAL_RE = re.compile(r"^[A-Za-z0-9_./:@%+=,-]+$")
_ENV_ASSIGNMENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*")
_COMMAND_WRAPPER_WORDS = {
    "sudo",
    "env",
    "exec",
    "nohup",
    "setsid",
    "time",
    "command",
    "builtin",
}
_SUDO_OPTIONS_WITH_ARG = {
    "-c", "--close-from",
    "-g", "--group",
    "-h", "--host",
    "-p", "--prompt",
    "-u", "--user",
}

_INTERPRETER_EXEC_FLAGS = {
    "python": {"-c"},
    "node": {"-e", "--eval", "-p", "--print"},
    "perl": {"-e", "--eval"},
    "ruby": {"-e"},
    "php": {"-r"},
    "powershell": {"-command", "-c", "-file", "-f"},
}
_INTERPRETER_WITH_ARG = {
    "python": {"-W", "-X", "--check-hash-based-pycs"},
    "node": {"-C", "--conditions", "--cpu-prof-dir", "--diagnostic-dir", "--icu-data-dir", "--import", "--loader", "--openssl-config", "--require", "--title"},
    "perl": {"-0", "-F", "-I", "-M", "-m", "-x"},
    "ruby": {"-C", "-E", "-F", "-I", "-K", "-r"},
    "php": {"-c", "-d", "-z"},
    "powershell": {"-configurationname", "-custompipename", "-executionpolicy", "-inputformat", "-outputformat", "-settingsfile", "-version", "-windowstyle", "-workingdirectory"},
}
_READ_TOOL_EXEC_FLAGS = {
    "sort": {"--compress-program"},
    "rg": {"--pre", "--hostname-bin"},
    "ag": {"--pager"},
    "man": {"--pager", "--html", "-P", "-H"},
}
# Required-argument options are ownership boundaries: an option-looking next
# token is data, not another option. These sets mirror the invocation grammar
# of the supported binaries (ripgrep 14, GNU sort, man-db, and ag 2.2).
_READ_TOOL_LONG_OPTIONS_WITH_ARG = {
    "rg": {
        "--after-context", "--before-context", "--color", "--colors",
        "--context", "--context-separator", "--dfa-size-limit", "--encoding",
        "--engine", "--field-context-separator", "--field-match-separator",
        "--file", "--generate", "--glob", "--hostname-bin",
        "--hyperlink-format", "--iglob", "--ignore-file", "--max-columns",
        "--max-count", "--max-depth", "--max-filesize", "--path-separator",
        "--pre", "--pre-glob", "--regex-size-limit", "--regexp", "--replace",
        "--sort", "--sortr", "--threads", "--type", "--type-add",
        "--type-clear", "--type-not",
    },
    "sort": {
        "--batch-size", "--buffer-size", "--compress-program",
        "--field-separator", "--files0-from", "--key", "--output",
        "--parallel", "--random-source", "--sort", "--temporary-directory",
    },
    "man": {
        "--config-file", "--encoding", "--extension", "--locale",
        "--manpath", "--pager", "--preprocessor", "--prompt", "--recode",
        "--sections", "--systems",
    },
    "ag": {
        "--ackmate-dir-filter", "--color-line-number", "--color-match",
        "--color-path", "--depth", "--filename-pattern", "--file-search-regex",
        "--ignore", "--ignore-dir", "--max-count", "--pager",
        "--path-to-ignore", "--width", "--workers",
    },
}
_READ_TOOL_SHORT_OPTIONS_WITH_ARG = {
    "rg": frozenset("efEmjgdtTABCMr"),
    "sort": frozenset("koStT"),
    "man": frozenset("CRLmMSserEPp"),
    "ag": frozenset("gGmpW"),
}
_SHELL_PUNCTUATION = {";", "&", "&&", "|", "||", "(", ")", "{", "}"}
_MAX_DETECTION_COMMAND_CHARS = 128_000
_MAX_SEPARATOR_FREE_COMMAND_CHARS = 4_096
_MAX_DETECTION_SEGMENTS = 25_000
_PARSER_LIMIT_DESCRIPTION = "command parser limit exceeded"
_MALFORMED_EXEC_DESCRIPTION = "command parser limit or malformed executable payload"



def _command_parser_limit_exceeded(command: str) -> bool:
    """Bound all parser work before normalization/tokenization.

    Counting separator characters is deliberately conservative: quoted
    separators can over-count, but crossing this very high ceiling fails
    closed rather than allowing an uninspected suffix to execute.
    """
    if len(command) > _MAX_DETECTION_COMMAND_CHARS:
        return True
    # Long separator-free input has no compound-command utility and otherwise
    # makes every legacy regex inspect one giant token. Reject it before any
    # normalization, tokenization, or regex work.
    if (
        len(command) > _MAX_SEPARATOR_FREE_COMMAND_CHARS
        and not any(char in command for char in ";&|\n")
    ):
        return True
    separators = 0
    for char in command:
        if char in ";&|\n":
            separators += 1
            if separators >= _MAX_DETECTION_SEGMENTS:
                return True
    return False


def _shell_tokens_with_spans(segment: str, start: int):
    """Return shell words as ``(value, start, end, quoted)`` or ``None``.

    This deliberately small lexer never expands shell syntax.  It exists to
    preserve source spans, which ``shlex`` does not expose, while deciding
    which *quoted* grep operand is data rather than another command.
    """
    tokens = []
    i = start
    while i < len(segment):
        while i < len(segment) and segment[i].isspace():
            i += 1
        if i >= len(segment):
            break
        token_start = i
        value = []
        quote = None
        while i < len(segment) and (quote or not segment[i].isspace()):
            char = segment[i]
            if quote:
                if char == quote:
                    quote = None
                    i += 1
                elif char == "\\" and quote == '"' and i + 1 < len(segment):
                    value.append(segment[i + 1])
                    i += 2
                else:
                    value.append(char)
                    i += 1
            elif char in {"'", '"'}:
                quote = char
                i += 1
            elif char == "\\":
                if i + 1 >= len(segment):
                    return None
                value.append(segment[i + 1])
                i += 2
            else:
                value.append(char)
                i += 1
        if quote:
            return None
        raw = segment[token_start:i]
        # Only a wholly single-quoted operand is inert shell data. Double
        # quotes still execute $() and backticks; unquoted substitutions do too.
        inert_single_quoted = (
            (raw.startswith("'") and raw.endswith("'"))
            or ("='" in raw and raw.endswith("'"))
        )
        tokens.append(("".join(value), token_start, i, inert_single_quoted))
    return tokens


_GREP_OPTIONS_WITH_ARG = {
    "--after-context", "--before-context", "--binary-files", "--context",
    "--directories", "--devices", "--exclude", "--exclude-dir",
    "--exclude-from", "--include", "--label", "--max-count",
    "--regexp", "--file",
}
_GREP_SHORT_OPTIONS_WITH_ARG = {"A", "B", "C", "D", "d", "e", "f", "m"}


def _quoted_grep_pattern_spans(command: str) -> tuple[list[tuple[int, int]], bool]:
    """Structurally locate quoted grep PCRE operands.

    The returned boolean means the grep parse was ambiguous or malformed.  In
    that case callers fail closed and, critically, use the original command:
    no text is hidden on an uncertain parse.
    """
    spans: list[tuple[int, int]] = []
    offset = 0
    for segment in _iter_top_level_shell_segments(command):
        segment_at = command.find(segment, offset)
        offset = segment_at + len(segment)
        for start, _, word in _iter_shell_command_word_spans(segment):
            if os.path.basename(_deobfuscate_shell_word_for_detection(word)).lower() not in {
                "grep", "egrep",
            }:
                continue
            tokens = _shell_tokens_with_spans(segment, start)
            if tokens is None:
                return [], True
            args = tokens[1:]
            pcre = False
            explicit_patterns = False
            pattern_indexes: list[int] = []
            operand_index = None
            i = 0
            options = True
            while i < len(args):
                token = args[i][0]
                if options and token == "--":
                    options = False
                    i += 1
                    continue
                if options and token.startswith("--"):
                    option, equals, _ = token.partition("=")
                    if option == "--perl-regexp":
                        pcre = True
                    if option in {"--regexp", "--file"}:
                        explicit_patterns = True
                    if option in _GREP_OPTIONS_WITH_ARG and not equals:
                        if i + 1 >= len(args):
                            return [], True
                        if option == "--regexp":
                            pattern_indexes.append(i + 1)
                        i += 2
                        continue
                    if option == "--regexp" and equals:
                        pattern_indexes.append(i)
                    i += 1
                    continue
                if options and token.startswith("-") and token != "-":
                    chars = token[1:]
                    j = 0
                    while j < len(chars):
                        char = chars[j]
                        if char == "P":
                            pcre = True
                        if char in {"e", "f"}:
                            explicit_patterns = True
                        if char in _GREP_SHORT_OPTIONS_WITH_ARG:
                            if j + 1 < len(chars):
                                if char == "e":
                                    pattern_indexes.append(i)
                            else:
                                if i + 1 >= len(args):
                                    return [], True
                                if char == "e":
                                    pattern_indexes.append(i + 1)
                                i += 1
                            break
                        j += 1
                    i += 1
                    continue
                if operand_index is None:
                    operand_index = i
                i += 1
            if not explicit_patterns:
                if operand_index is None:
                    return [], bool(pcre)
                pattern_indexes.append(operand_index)
            if pcre:
                for index in pattern_indexes:
                    _, token_start, token_end, quoted = args[index]
                    if quoted:
                        spans.append((segment_at + token_start, segment_at + token_end))
    return spans, False


def _grep_safe_detection_variant(command: str) -> tuple[str, bool]:
    spans, malformed = _quoted_grep_pattern_spans(command)
    if malformed or not spans:
        return command, malformed
    parts = []
    previous = 0
    for start, end in spans:
        parts.extend((command[previous:start], " " * (end - start)))
        previous = end
    parts.append(command[previous:])
    return "".join(parts), False


def _interpreter_family(executable: str) -> str | None:
    name = os.path.basename(executable).lower()
    if re.fullmatch(r"py(?:\.exe)?|python[23]?(?:\.\d+)*(?:\.exe)?", name):
        return "python"
    if re.fullmatch(r"node(?:js)?(?:\.exe)?", name):
        return "node"
    if re.fullmatch(r"perl[0-9]*(?:\.\d+)*(?:\.exe)?", name):
        return "perl"
    if re.fullmatch(r"ruby[0-9.]*(?:\.exe)?", name):
        return "ruby"
    if re.fullmatch(r"php(?:\.exe)?", name):
        return "php"
    if re.fullmatch(r"powershell(?:\.exe)?|pwsh(?:\.exe)?", name):
        return "powershell"
    return None


def _shell_segment_tokens(segment: str, start: int) -> list[str] | None:
    """Tokenize an already-bounded command segment.

    ``None`` distinguishes malformed quoting from an empty segment so callers
    can fail closed for a program-bearing option rather than silently skip it.
    """
    try:
        lexer = shlex.shlex(segment[start:], posix=True, punctuation_chars="<>")
        lexer.whitespace_split = True
        lexer.commenters = ""
        return list(lexer)
    except ValueError:
        return None


def _iter_top_level_shell_segments(command: str):
    """Yield top-level command segments in one left-to-right pass."""
    start = 0
    quote: str | None = None
    escaped = False
    index = 0
    while index < len(command):
        char = command[index]
        if escaped:
            escaped = False
        elif char == "\\" and quote != "'":
            escaped = True
        elif quote:
            if char == quote:
                quote = None
        elif char in {"'", '"'}:
            quote = char
        elif char in ";&|\n":
            if start < index:
                yield command[start:index]
            # Consume a doubled && / || separator as one boundary.
            if char in "&|" and index + 1 < len(command) and command[index + 1] == char:
                index += 1
            start = index + 1
        index += 1
    if start < len(command):
        yield command[start:]


def _split_option(token: str) -> tuple[str, str | None]:
    if "=" in token:
        option, value = token.split("=", 1)
        return option, value
    return token, None


def _interpreter_exec_flag(family: str, args: list[str]) -> str | None:
    """Return an execution-bearing interpreter option, if present."""
    flags = _INTERPRETER_EXEC_FLAGS[family]
    skip_value = False
    for token in args:
        if skip_value:
            skip_value = False
            continue
        if token == "--":
            break
        if family != "powershell" and not token.startswith("-"):
            break
        option, attached = _split_option(token)
        comparable = option.lower() if family == "powershell" else option
        if comparable in flags:
            return comparable
        with_arg = _INTERPRETER_WITH_ARG[family]
        # `-Wonce` and `ruby -rjson` attach an option value; they are not
        # short-option bundles containing an execution flag. PowerShell's
        # normal long options also use one dash, so bundle parsing never
        # applies to that family.
        has_attached_option_value = any(
            option.startswith(short) and len(option) > len(short)
            for short in with_arg
            if short.startswith("-") and not short.startswith("--")
        )
        if (
            family != "powershell"
            and not option.startswith("--")
            and len(option) > 2
            and not has_attached_option_value
        ):
            for char in option[1:]:
                short = f"-{char}"
                if short in flags:
                    return short
        if comparable in with_arg and attached is None:
            skip_value = True
    return None


_BASH_OPTIONS_WITH_ARG = {"-O", "+O", "-o", "+o", "--init-file", "--rcfile"}
_BASH_SHORT_OPTION_LETTERS = frozenset("ilrsDcabefhkmnptuvxBCEHPTOo")


def _bash_exec_payload(args: list[str]) -> tuple[bool, str | None]:
    """Return whether Bash ``-c`` occurs and the command string it owns.

    Bash's O/o invocation options consume the following argument even when
    they precede a later ``-c`` or occur in the same short-option bundle.
    Likewise, the two startup-file long options own their next token. Parsing
    those operands first prevents both missed payloads and false ``-c`` hits.
    """
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--" or not token.startswith(("-", "+")):
            break
        if token in _BASH_OPTIONS_WITH_ARG:
            index += 2
            continue
        if token.startswith("--"):
            index += 1
            continue

        chars = token[1:]
        # Bash option letters are case-sensitive. Restricting this to its
        # documented alphabet preserves invalid controls such as `-Wc`.
        if not set(chars) <= _BASH_SHORT_OPTION_LETTERS:
            index += 1
            continue
        consumed_option_arg = "O" in chars or "o" in chars
        if "c" not in chars:
            index += 1 + int(consumed_option_arg)
            continue
        payload_index = index + 1 + int(consumed_option_arg)
        payload = args[payload_index] if payload_index < len(args) else None
        return True, payload
    return False, None


def _read_tool_exec_flag(tool: str, args: list[str]) -> tuple[str, str] | None:
    """Return (option, program) for a read-only tool's program-running flag."""
    flags = _READ_TOOL_EXEC_FLAGS[tool]
    index = 0
    while index < len(args):
        token = args[index]
        if token == "--":
            break
        option, payload = _split_option(token)
        matched = option if option in flags else None
        if tool == "man" and token.startswith(("-P", "-H")) and len(token) > 2:
            matched, payload = token[:2], token[2:]
        if matched:
            if payload is None and index + 1 < len(args):
                payload = args[index + 1]
            # This option owns its program argument regardless of spelling.
            # The real binaries execute a payload beginning with '-' rather
            # than reparsing it as one of the tool's later options.
            if payload:
                return matched, payload
            index += 2 if payload is not None and "=" not in token else 1
            continue

        if option in _READ_TOOL_LONG_OPTIONS_WITH_ARG[tool] and payload is None:
            index += 2
            continue

        # In a short bundle, the first argument-taking option owns the rest of
        # the token, or the following token when it occurs last.
        if token.startswith("-") and not token.startswith("--") and len(token) > 1:
            for short_index, char in enumerate(token[1:], start=1):
                if char in _READ_TOOL_SHORT_OPTIONS_WITH_ARG[tool]:
                    index += 2 if short_index == len(token) - 1 else 1
                    break
            else:
                index += 1
            continue
        index += 1
    return None


def _execution_flag_findings(command: str):
    """Yield scoped execution mechanisms and any executable payloads."""
    for segment in _iter_top_level_shell_segments(command):
        for start, _, word in _iter_shell_command_word_spans(segment):
            executable = _deobfuscate_shell_word_for_detection(word)
            tokens = _shell_segment_tokens(segment, start)
            executable_name = os.path.basename(executable).lower()
            family = _interpreter_family(executable)
            is_program_bearing = (
                family is not None or executable_name in _READ_TOOL_EXEC_FLAGS
            )
            if tokens is None:
                if is_program_bearing:
                    yield (_MALFORMED_EXEC_DESCRIPTION, None)
                continue
            if not tokens:
                continue
            if family:
                flag = _interpreter_exec_flag(family, tokens[1:])
                if flag:
                    yield ("script execution via -e/-c flag", None)
                    continue
                if any(token.startswith("<<") for token in tokens[1:]):
                    yield ("script execution via heredoc", None)
                    continue
            if executable_name in {"bash", "sh", "zsh", "ksh"}:
                found, payload = _bash_exec_payload(tokens[1:])
                if found:
                    yield ("shell command via -c/-lc flag", payload)
            tool = executable_name
            if tool in _READ_TOOL_EXEC_FLAGS:
                finding = _read_tool_exec_flag(tool, tokens[1:])
                if finding:
                    option, payload = finding
                    yield (f"arbitrary program execution via {tool} {option}", payload)


def _skip_shell_whitespace(command: str, pos: int) -> int:
    while pos < len(command) and command[pos].isspace():
        pos += 1
    return pos


def _scan_dollar_paren_end(command: str, start: int) -> int | None:
    """Return the offset after a balanced ``$(...)`` command substitution."""
    depth = 1
    quote: str | None = None
    i = start + 2
    while i < len(command):
        ch = command[i]
        if quote:
            if ch == "\\" and quote == '"' and i + 1 < len(command):
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            i += 1
            continue
        if ch == "\\" and i + 1 < len(command):
            i += 2
            continue
        if command.startswith("$(", i):
            depth += 1
            i += 2
            continue
        if ch == ")":
            depth -= 1
            i += 1
            if depth == 0:
                return i
            continue
        i += 1
    return None


def _scan_backtick_end(command: str, start: int) -> int | None:
    i = start + 1
    while i < len(command):
        if command[i] == "\\" and i + 1 < len(command):
            i += 2
            continue
        if command[i] == "`":
            return i + 1
        i += 1
    return None


def _read_shell_word(command: str, pos: int) -> tuple[int, int, str]:
    """Read one shell word without executing expansions."""
    start = _skip_shell_whitespace(command, pos)
    i = start
    quote: str | None = None
    while i < len(command):
        ch = command[i]
        if quote:
            if ch == "\\" and quote == '"' and i + 1 < len(command):
                i += 2
                continue
            if ch == quote:
                quote = None
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            i += 1
            continue
        if ch == "\\" and i + 1 < len(command):
            i += 2
            continue
        if command.startswith("$(", i):
            end = _scan_dollar_paren_end(command, i)
            if end is None:
                i += 2
            else:
                i = end
            continue
        if command.startswith("${", i):
            end = command.find("}", i + 2)
            if end == -1:
                i += 2
            else:
                i = end + 1
            continue
        if ch == "`":
            end = _scan_backtick_end(command, i)
            if end is None:
                i += 1
            else:
                i = end
            continue
        if ch.isspace() or ch in ";&|":
            break
        i += 1
    return (start, i, command[start:i])


def _strip_optional_shell_quotes(word: str) -> str:
    if len(word) >= 2 and word[0] == word[-1] and word[0] in ("'", '"'):
        return word[1:-1]
    return word


def _is_simple_shell_literal(value: str) -> bool:
    return bool(value and _SIMPLE_SHELL_LITERAL_RE.fullmatch(value))


def _literal_command_substitution_output(script: str) -> str | None:
    """Resolve tiny literal command substitutions without executing a shell."""
    try:
        tokens = shlex.split(script, posix=True)
    except ValueError:
        return None
    if not tokens:
        return None

    command = tokens[0].lower()
    args = tokens[1:]
    if command == "echo":
        while args and re.fullmatch(r"-[nEe]+", args[0]):
            args = args[1:]
        if len(args) == 1 and _is_simple_shell_literal(args[0]):
            return args[0]
        return None

    if command == "printf":
        if len(args) == 1 and _is_simple_shell_literal(args[0]):
            return args[0]
        if (
            len(args) == 2
            and args[0] == "%s"
            and _is_simple_shell_literal(args[1])
        ):
            return args[1]
    return None


def _replace_simple_command_substitutions(word: str) -> str:
    chars: list[str] = []
    i = 0
    while i < len(word):
        if word.startswith("$(", i):
            end = _scan_dollar_paren_end(word, i)
            if end is not None:
                replacement = _literal_command_substitution_output(word[i + 2:end - 1])
                if replacement is not None:
                    chars.append(replacement)
                    i = end
                    continue
        if word[i] == "`":
            end = _scan_backtick_end(word, i)
            if end is not None:
                replacement = _literal_command_substitution_output(word[i + 1:end - 1])
                if replacement is not None:
                    chars.append(replacement)
                    i = end
                    continue
        chars.append(word[i])
        i += 1
    return "".join(chars)


def _replace_simple_shell_expansions(word: str) -> str:
    word = _replace_simple_command_substitutions(word)
    word = _PARAM_REPLACEMENT_RE.sub(lambda match: match.group("replacement"), word)
    return _PARAM_DEFAULT_RE.sub(lambda match: match.group("default"), word)


def _strip_shell_word_syntax(word: str) -> str:
    chars: list[str] = []
    quote: str | None = None
    i = 0
    while i < len(word):
        ch = word[i]
        if quote:
            if ch == "\\" and quote == '"' and i + 1 < len(word):
                chars.append(word[i + 1])
                i += 2
                continue
            if ch == quote:
                quote = None
                i += 1
                continue
            chars.append(ch)
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
            i += 1
            continue
        if ch == "\\" and i + 1 < len(word):
            chars.append(word[i + 1])
            i += 2
            continue
        chars.append(ch)
        i += 1
    return "".join(chars)


def _deobfuscate_shell_word_for_detection(word: str) -> str:
    """Approximate how shell syntax can spell a command word.

    This is intentionally narrow and non-executing: it only collapses shell
    quoting/escaping plus simple literal command substitutions that appear in
    the command word itself.
    """
    deobfuscated = word
    for _ in range(2):
        previous = deobfuscated
        deobfuscated = _replace_simple_shell_expansions(deobfuscated)
        deobfuscated = _strip_shell_word_syntax(deobfuscated)
        if deobfuscated == previous:
            break
    return deobfuscated


def _iter_shell_command_starts(command: str):
    starts = [0]

    def scan(start: int, end: int) -> None:
        quote: str | None = None
        i = start
        while i < end:
            ch = command[i]
            if quote == "'":
                if ch == "'":
                    quote = None
                i += 1
                continue
            if quote == '"':
                if ch == "\\" and i + 1 < end:
                    i += 2
                    continue
                if ch == '"':
                    quote = None
                    i += 1
                    continue
                if command.startswith("$(", i):
                    nested_end = _scan_dollar_paren_end(command, i)
                    starts.append(i + 2)
                    scan(i + 2, nested_end - 1 if nested_end is not None else end)
                    i = nested_end if nested_end is not None else end
                    continue
                if ch == "`":
                    nested_end = _scan_backtick_end(command, i)
                    starts.append(i + 1)
                    scan(i + 1, nested_end - 1 if nested_end is not None else end)
                    i = nested_end if nested_end is not None else end
                    continue
                i += 1
                continue
            if ch in ("'", '"'):
                quote = ch
                i += 1
                continue
            if ch == "\\" and i + 1 < end:
                i += 2
                continue
            if command.startswith("$(", i):
                nested_end = _scan_dollar_paren_end(command, i)
                starts.append(i + 2)
                scan(i + 2, nested_end - 1 if nested_end is not None else end)
                i = nested_end if nested_end is not None else end
                continue
            if ch == "`":
                nested_end = _scan_backtick_end(command, i)
                starts.append(i + 1)
                scan(i + 1, nested_end - 1 if nested_end is not None else end)
                i = nested_end if nested_end is not None else end
                continue
            if ch in ("(", "{"):
                starts.append(i + 1)
            elif ch in ";\n":
                starts.append(i + 1)
            elif ch in "&|":
                repeated = i + 1 < end and command[i + 1] == ch
                starts.append(i + 2 if repeated else i + 1)
                if repeated:
                    i += 1
            i += 1

    scan(0, len(command))

    seen: set[int] = set()
    for start in starts:
        start = _skip_shell_whitespace(command, start)
        if start < len(command) and start not in seen:
            seen.add(start)
            yield start


def _mark_command_starts(command: str) -> str:
    """Insert a newline before each real (quote-aware) command start.

    ``\\n`` is already a ``_CMDPOS`` separator, so this rewrites subshell
    ``(cmd)`` and brace-group ``{ cmd; }`` openers — which the flat pattern
    class deliberately omits — into a form the anchored hardline/dangerous
    patterns recognize, WITHOUT the quoted-prose false positives that adding
    ``(`` / ``{`` to ``_CMDPOS`` would cause. Starts inside quotes are never
    produced by ``_iter_shell_command_starts``, so quoted arguments such as
    ``--title "block (reboot)"`` are left exactly as-is.
    """
    # Collect the (whitespace-skipped) start offsets, drop 0 (already anchored
    # by ``^``), and splice a newline in front of each — right-to-left so the
    # earlier offsets stay valid as we mutate.
    offsets = sorted(o for o in _iter_shell_command_starts(command) if o > 0)
    if not offsets:
        return command
    # Build once instead of repeatedly slicing and copying the full command for
    # every segment (quadratic at 10k+ compound-command segments).
    parts: list[str] = []
    previous = 0
    for offset in offsets:
        parts.extend((command[previous:offset], "\n"))
        previous = offset
    parts.append(command[previous:])
    return "".join(parts)


def _mask_quoted_newlines(command: str) -> str:
    """Replace raw newlines inside single/double quotes with a space.

    Detection-only rewrite. A newline inside a quoted string is DATA to the
    shell — part of the argument, not a command separator — yet the flat
    ``_CMDPOS`` start-position class treats every raw ``\\n`` as a command
    start. That made any multi-line quoted argument (``hermes send`` message
    bodies, ``git commit -m`` messages, heredoc text) trip the hardline
    blocklist when a data line began with e.g. ``sudo reboot``.

    Quote tracking mirrors ``_iter_shell_command_starts``: single quotes are
    literal until the closing quote; inside double quotes a backslash escapes
    the next character. Real command boundaries are unaffected: unquoted
    newlines pass through untouched, ``$(``/backtick remain ``_CMDPOS``
    anchors independent of newlines, and ``_mark_command_starts`` still
    re-inserts newlines at every genuine quote-aware command start. An
    unclosed quote absorbs following newlines exactly as the shell would
    (the quoted word continues across the line break), so masking them
    cannot hide a runnable command.
    """
    if "\n" not in command:
        return command
    out: list[str] = []
    quote: str | None = None
    i = 0
    while i < len(command):
        ch = command[i]
        if quote:
            if ch == "\\" and quote == '"' and i + 1 < len(command):
                out.append(command[i:i + 2])
                i += 2
                continue
            if ch == quote:
                quote = None
            out.append(" " if ch == "\n" else ch)
            i += 1
            continue
        if ch in ("'", '"'):
            quote = ch
        elif ch == "\\" and i + 1 < len(command):
            out.append(command[i:i + 2])
            i += 2
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _iter_shell_command_word_spans(command: str):
    """Yield command-position words that may be executable names."""
    for command_start in _iter_shell_command_starts(command):
        pos = command_start
        prefix_words = 0
        skip_wrapper_options = False
        skip_next_wrapper_arg = False
        while prefix_words < 12:
            word_start, word_end, word = _read_shell_word(command, pos)
            if word_start == word_end:
                break
            deobfuscated = _deobfuscate_shell_word_for_detection(word)
            lower_word = deobfuscated.lower()
            if skip_next_wrapper_arg:
                skip_next_wrapper_arg = False
                pos = word_end
                prefix_words += 1
                continue
            if skip_wrapper_options and lower_word.startswith("-"):
                option_name = lower_word.split("=", 1)[0]
                skip_next_wrapper_arg = (
                    "=" not in lower_word
                    and option_name in _SUDO_OPTIONS_WITH_ARG
                )
                pos = word_end
                prefix_words += 1
                continue

            yield (word_start, word_end, word)
            prefix_words += 1

            if lower_word in _COMMAND_WRAPPER_WORDS:
                skip_wrapper_options = lower_word in {"sudo", "env"}
                pos = word_end
                continue
            if _ENV_ASSIGNMENT_RE.fullmatch(deobfuscated):
                skip_wrapper_options = False
                pos = word_end
                continue
            break


def _command_detection_variants(command: str):
    # Mask quoted newlines BEFORE normalization: normalization strips
    # backslash-escapes (\" -> ") and empty-string pairs (""), which would
    # corrupt quote tracking — e.g. `echo "a\""` normalizes to `echo "a` (an
    # unterminated quote), so masking the normalized text could swallow a
    # REAL unquoted newline separator that follows. The raw command carries
    # faithful shell quote state.
    normalized = _normalize_command_for_detection(_mask_quoted_newlines(command))
    # Quote-aware grep parsing hides only structurally identified pattern
    # operands. Malformed/ambiguous input remains byte-for-byte intact.
    grep_safe, _ = _grep_safe_detection_variant(normalized)
    seen = {grep_safe}
    yield grep_safe
    # Windows-path variant (#69472): normalization treats backslashes as
    # shell escapes and strips them, so `del C:\Users\me\.ssh\id_rsa`
    # reaches the patterns as `del C:Usersme.sshid_rsa` — no path rule can
    # ever match a backslash Windows path. When the RAW command contains a
    # drive-letter or UNC backslash path, also yield a variant with
    # backslashes flattened to forward slashes BEFORE normalization eats
    # them. Gated on a real path shape (letter, colon, backslash — or
    # double backslash UNC) so POSIX escape semantics (`echo a\"b`) are
    # untouched on every other command.
    if re.search(r"(?:[A-Za-z]:|\\\\)[\\\\]", command) or re.search(r"[A-Za-z]:\\", command):
        win_variant = _normalize_command_for_detection(
            _mask_quoted_newlines(command.replace("\\", "/"))
        )
        if win_variant not in seen:
            seen.add(win_variant)
            yield win_variant
    # Program-bearing options are parsed in their owning command's context.
    # Surfacing only their payload lets the hardline floor inspect the command
    # that will actually run without promoting similar flags or quoted prose.
    pending = [normalized]
    while pending:
        variant = pending.pop()
        for _, payload in _execution_flag_findings(variant):
            if payload and payload not in seen:
                seen.add(payload)
                yield payload
                # A payload can begin with an option-looking program and then
                # invoke a hardline command after a separator. Mark its real
                # command starts just as we do for the outer command.
                marked_payload = _mark_command_starts(payload)
                if marked_payload != payload and marked_payload not in seen:
                    seen.add(marked_payload)
                    yield marked_payload
                pending.append(payload)
    # Subshell `(cmd)` and brace-group `{ cmd; }` openers put `cmd` at a real
    # command position, but the flat `_CMDPOS`-anchored patterns can't see it:
    # their start-position class deliberately omits `(`/`{` because a bare
    # regex cannot tell `(reboot)` (real subshell) from `--title "(reboot)"`
    # (quoted prose) — adding them there regresses ordinary quoted arguments.
    # Instead, reconstruct the command with a newline (already a `_CMDPOS`
    # separator) inserted at each command start the QUOTE-AWARE tokenizer
    # found. Openers inside quotes never yield a start, so quoted prose is
    # untouched, while `(reboot)` / `{ shutdown -h now; }` now anchor. This
    # covers every `_CMDPOS` rule (shutdown/reboot/init/systemctl/telinit and
    # the rm root/home/system floor) in one place.
    marked = _mark_command_starts(grep_safe)
    if marked != grep_safe and marked not in seen:
        seen.add(marked)
        yield marked
    # Shell quoting/escaping can spell a dangerous executable name in pieces
    # (for example r\m or r''m). Keep that deobfuscation scoped to command
    # words so similarly shaped arguments do not become false positives.
    for word_start, word_end, word in _iter_shell_command_word_spans(normalized):
        deobfuscated = _deobfuscate_shell_word_for_detection(word)
        if not deobfuscated or deobfuscated == word:
            continue
        variant = normalized[:word_start] + deobfuscated + normalized[word_end:]
        if variant in seen:
            continue
        seen.add(variant)
        yield variant


def _is_verification_artifact_cleanup(command: str) -> bool:
    """Return whether *command* only removes one Hermes ad-hoc temp script."""
    try:
        argv = shlex.split(command, posix=True)
    except ValueError:
        return False
    if len(argv) != 3 or argv[0] != "rm" or argv[1] != "-f":
        return False

    operand = argv[2]
    temp_dir = os.path.realpath(tempfile.gettempdir())
    basename = os.path.basename(operand)
    if operand != os.path.join(temp_dir, basename):
        return False

    target = os.path.realpath(operand)
    if os.path.dirname(target) != temp_dir:
        return False
    return re.fullmatch(r"hermes-(?:verify|ad-hoc)-[A-Za-z0-9_.-]+", basename) is not None


_GATEWAY_LIFECYCLE_SPLICE_DESCRIPTION = (
    "stop/restart hermes gateway via shell-spliced verb (kills running agents)"
)


def _is_shell_token_spliced_gateway_lifecycle(command: str) -> bool:
    """Catch gateway-lifecycle verbs spelled with quote/backslash splicing.

    ``_normalize_command_for_detection`` strips backslash escapes, so
    ``kick\\start`` already reaches the launchctl pattern above. Quote
    splicing does not: ``_deobfuscate_shell_word_for_detection`` is
    deliberately scoped to command-position words (widening it would let
    quoted prose like ``git commit -m "rm -rf /"`` match the destructive
    patterns), and the spliced verb sits in an ARGUMENT position. So
    ``launchctl kick"start" -k gui/501/ai.hermes.gateway`` auto-approved
    while executing exactly as the gated ``kickstart`` form (#80269).

    Delegate to ``cron.lifecycle_guard``, which tokenizes with shlex and is
    anchored on a hermes-gateway identifier — reusing its prose
    false-positive coverage instead of loosening the generic pattern
    engine. This runs last, so an ordinary pattern match still wins and
    keeps its more specific reason string. Unlike the guard's use inside
    ``terminal_tool``, this layer only raises an approval prompt; the
    non-bypassable block still lives in ``cron.lifecycle_guard``.
    """
    try:
        from cron.lifecycle_guard import contains_gateway_lifecycle_command
    except Exception:
        return False
    return contains_gateway_lifecycle_command(command)


def detect_dangerous_command(command: str) -> tuple:
    """Check if a command matches any dangerous patterns.

    Returns:
        (is_dangerous, pattern_key, description) or (False, None, None)
    """
    if _command_parser_limit_exceeded(command):
        return (True, _PARSER_LIMIT_DESCRIPTION, _PARSER_LIMIT_DESCRIPTION)
    if _is_verification_artifact_cleanup(command):
        return (False, None, None)

    for command_variant in _command_detection_variants(command):
        command_lower = command_variant.lower()
        for pattern_re, description in DANGEROUS_PATTERNS_COMPILED:
            if pattern_re.search(command_lower):
                pattern_key = description
                return (True, pattern_key, description)
    normalized = _normalize_command_for_detection(command)
    for description, _ in _execution_flag_findings(normalized):
        return (True, description, description)
    if _is_shell_token_spliced_gateway_lifecycle(command):
        return (
            True,
            _GATEWAY_LIFECYCLE_SPLICE_DESCRIPTION,
            _GATEWAY_LIFECYCLE_SPLICE_DESCRIPTION,
        )
    return (False, None, None)
