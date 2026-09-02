#!/usr/bin/env python3
"""
File Operations Module

Provides file manipulation capabilities (read, write, patch, search) that work
across all terminal backends (local, docker, ssh, singularity, modal, daytona, vercel_sandbox).

The key insight is that all file operations can be expressed as shell commands,
so we wrap the terminal backend's execute() interface to provide a unified file API.

Usage:
    from tools.file_operations import ShellFileOperations
    from tools.terminal_tool import _active_environments
    
    # Get file operations for a terminal environment
    file_ops = ShellFileOperations(terminal_env)
    
    # Read a file
    result = file_ops.read_file("/path/to/file.py")
    
    # Write a file
    result = file_ops.write_file("/path/to/new.py", "print('hello')")
    
    # Search for content
    result = file_ops.search("TODO", path=".", file_glob="*.py")
"""

import base64
import binascii
import os
import re
import sys
import difflib
import hashlib
import json
import unicodedata
from abc import ABC, abstractmethod

from typing import Optional, Dict
from pathlib import Path
from tools.binary_extensions import BINARY_EXTENSIONS

from agent.file_safety import (
    build_write_denied_paths,
    build_write_denied_prefixes,
    get_write_denied_error,
    is_write_denied as _shared_is_write_denied,
)
from tools.file_operations_common import (  # noqa: F401  (re-exported)
    DEFAULT_READ_LIMIT,
    DEFAULT_READ_OFFSET,
    DEFAULT_SEARCH_LIMIT,
    DEFAULT_SEARCH_OFFSET,
    ExecuteResult,
    LintResult,
    PatchResult,
    ReadResult,
    SearchMatch,
    SearchResult,
    WriteResult,
    _FENCE_MARKER_RE,
    _OSC_SEQUENCE_RE,
    _UTF8_BOM,
    _coerce_int,
    _detect_line_ending,
    _has_bom,
    _normalize_line_endings,
    _strip_bom,
    _strip_terminal_fence_leaks,
    normalize_read_pagination,
    normalize_search_pagination,
)
from tools.file_operations_lint import (  # noqa: F401  (re-exported)
    LINTERS,
    LintMixin,
    _FAIL_CLOSED_INPROC_EXTS,
    _LINTER_UNUSABLE_PATTERNS,
    _SHELL_LINTER_LSP_REDUNDANT,
    _lint_json_inproc,
    _lint_python_inproc,
    _lint_toml_inproc,
    _lint_yaml_inproc,
    _looks_like_linter_unusable,
)
from tools.file_operations_search import (  # noqa: F401  (re-exported)
    SearchMixin,
    _MACOS_TCC_PROTECTED_HOME_DIRS,
    _REGEX_NEWLINE_ESCAPE_RE,
    _SEARCH_OUTPUT_RE,
    _SEARCH_TIMEOUT_MARKER_RE,
    _is_line_oriented_newline_error,
    _macos_protected_search_exclusions,
    _maybe_warn_line_oriented_newline_pattern,
    _parse_search_context_line,
    _pattern_has_regex_newline,
    _search_stdout_and_limit,
    _split_tool_diagnostics,
)


# ---------------------------------------------------------------------------
# Write-path deny list — blocks writes to sensitive system/credential files
# ---------------------------------------------------------------------------

_HOME = str(Path.home())


WRITE_DENIED_PATHS = build_write_denied_paths(_HOME)

WRITE_DENIED_PREFIXES = build_write_denied_prefixes(_HOME)


def _is_write_denied(path: str) -> bool:
    """Return True if path is on the write deny list."""
    return _shared_is_write_denied(path)


# =============================================================================
# Result Data Classes
# =============================================================================


# =============================================================================
# Abstract Interface
# =============================================================================

_MAGIC_SIGNATURES: tuple = (
    # (prefix bytes, human name) — ordered, first match wins. Longest
    # prefixes for a shared first byte come first.
    (b"\x89PNG\r\n\x1a\n", "PNG image data"),
    (b"\xff\xd8\xff", "JPEG image data"),
    (b"GIF87a", "GIF image data"),
    (b"GIF89a", "GIF image data"),
    (b"RIFF", "RIFF container (WAV/AVI/WebP family)"),
    (b"%PDF-", "PDF document"),
    (b"PK\x03\x04", "ZIP archive (also docx/xlsx/jar/apk)"),
    (b"PK\x05\x06", "ZIP archive (empty)"),
    (b"\x1f\x8b", "gzip compressed data"),
    (b"BZh", "bzip2 compressed data"),
    (b"\xfd7zXZ\x00", "xz compressed data"),
    (b"7z\xbc\xaf\x27\x1c", "7-Zip archive"),
    (b"\x7fELF", "ELF executable"),
    (b"MZ", "Windows PE executable"),
    (b"\xcf\xfa\xed\xfe", "Mach-O executable (64-bit)"),
    (b"\xca\xfe\xba\xbe", "Mach-O universal binary / Java class"),
    (b"SQLite format 3\x00", "SQLite database"),
    (b"OggS", "Ogg container"),
    (b"fLaC", "FLAC audio"),
    (b"ID3", "MP3 audio (ID3 tag)"),
    (b"\x00\x00\x00", "ISO media container (MP4/MOV family)"),  # ftyp at +4
    (b"BM", "BMP image data"),
    (b"II*\x00", "TIFF image data (little-endian)"),
    (b"MM\x00*", "TIFF image data (big-endian)"),
)


def identify_binary_bytes(sample: bytes) -> str:
    """Best-effort human name for binary content from its magic bytes.

    Returns e.g. ``"PNG image data"`` or ``"unknown binary"``. Never raises.
    The ISO-media entry additionally checks for ``ftyp`` at offset 4, since
    the leading size field alone (three NULs) is too weak a signature.
    """
    if not sample:
        return "unknown binary"
    for prefix, name in _MAGIC_SIGNATURES:
        if sample.startswith(prefix):
            if name.startswith("ISO media") and sample[4:8] != b"ftyp":
                continue
            return name
    return "unknown binary"


def describe_binary_file(sample: Optional[bytes], file_size: int) -> str:
    """One-line answer for the binary-file refusal.

    Naming the dead end: "Binary file" alone sends the model hunting for
    'appropriate tools' that may not exist in its toolset. Naming the TYPE
    ("PNG image data, 4.1 KB") answers what-is-this in a single read.
    """
    kind = identify_binary_bytes(sample or b"")
    if file_size >= 1024 * 1024:
        size = f"{file_size / (1024 * 1024):.1f} MB"
    elif file_size >= 1024:
        size = f"{file_size / 1024:.1f} KB"
    else:
        size = f"{file_size} bytes"
    return f"Binary file ({kind}, {size}) — cannot display as text."


class FileOperations(ABC):
    """Abstract interface for file operations across terminal backends."""
    
    @abstractmethod
    def read_file(self, path: str, offset: int = 1, limit: int = 2000) -> ReadResult:
        """Read a file with pagination support."""
        ...

    @abstractmethod
    def read_file_raw(self, path: str) -> ReadResult:
        """Read the complete file content as a plain string.

        No pagination, no line-number prefixes, no per-line truncation.
        Returns ReadResult with .content = full file text, .error set on
        failure. Always reads to EOF regardless of file size.
        """
        ...

    def read_file_bytes(self, path: str, max_bytes: Optional[int] = None) -> ReadResult:
        """Read complete binary content as base64 across the backend boundary."""
        return ReadResult(error="Binary reads are not implemented for this backend")

    @abstractmethod
    def write_file(self, path: str, content: str,
                   pre_content: Optional[str] = None) -> WriteResult:
        """Write content to a file, creating directories as needed."""
        ...

    @abstractmethod
    def patch_replace(self, path: str, old_string: str, new_string: str,
                      replace_all: bool = False) -> PatchResult:
        """Replace text in a file using fuzzy matching."""
        ...

    @abstractmethod
    def patch_v4a(self, patch_content: str) -> PatchResult:
        """Apply a V4A format patch."""
        ...

    @abstractmethod
    def delete_file(self, path: str) -> WriteResult:
        """Delete a file. Returns WriteResult with .error set on failure."""
        ...

    def delete_path(self, path: str, recursive: bool = False) -> WriteResult:
        """Cross-platform delete that handles files and (with recursive=True)
        directory trees. Default implementation delegates to ``delete_file``
        for the non-recursive case; backends with native recursive support
        should override.
        """
        if recursive:
            return WriteResult(error="Recursive delete not implemented for this backend")
        return self.delete_file(path)

    @abstractmethod
    def move_file(self, src: str, dst: str) -> WriteResult:
        """Move/rename a file from src to dst. Returns WriteResult with .error set on failure."""
        ...

    @abstractmethod
    def search(self, pattern: str, path: str = ".", target: str = "content",
               file_glob: Optional[str] = None, limit: int = 50, offset: int = 0,
               output_mode: str = "content", context: int = 0) -> SearchResult:
        """Search for content or files."""
        ...


# =============================================================================
# Shell-based Implementation
# =============================================================================

# Image extensions (subset of binary that we can return as base64)
IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.gif', '.webp', '.bmp', '.ico'}


# In-process linters by file extension.  Preferred over shell linters when
# present — no subprocess overhead, microseconds per call.  Each callable
# takes file content (str) and returns (ok: bool, error: str).  An error
# string of ``"__SKIP__"`` signals the linter isn't available (missing
# dependency) and should be treated as "no linter".
LINTERS_INPROC = {
    '.py': _lint_python_inproc,
    '.json': _lint_json_inproc,
    '.yaml': _lint_yaml_inproc,
    '.yml': _lint_yaml_inproc,
    '.toml': _lint_toml_inproc,
}


# Max limits for read operations
MAX_LINES = 2000
MAX_LINE_LENGTH = 2000
MAX_FILE_SIZE = 50 * 1024  # 50KB

# Echoed by the size probe when the path exists but is not a regular file.
# `wc -c` prints only digits, so this can never collide with a real size.
NOT_REGULAR_SENTINEL = "__hermes_not_regular__"


class ShellFileOperations(LintMixin, SearchMixin, FileOperations):
    """
    File operations implemented via shell commands.
    
    Works with ANY terminal backend that has execute(command, cwd) method.
    This includes local, docker, singularity, ssh, modal, and daytona environments.
    """
    
    def __init__(self, terminal_env, cwd: str = None):
        """
        Initialize file operations with a terminal environment.

        Args:
            terminal_env: Any object with execute(command, cwd) method.
                         Returns {"output": str, "returncode": int}
            cwd: Optional explicit fallback cwd when the terminal env has
                 no cwd attribute (rare — most backends track cwd live).

        Note:
            Every _exec() call prefers the LIVE ``terminal_env.cwd`` over
            ``self.cwd`` so ``cd`` commands run via the terminal tool are
            picked up immediately.  ``self.cwd`` is only used as a fallback
            when the env has no cwd at all — it is NOT the authoritative
            cwd, despite being settable at init time.

            Historical bug (fixed): prior versions of this class used the
            init-time cwd for every _exec() call, which caused relative
            paths passed to patch/read/write to target the wrong directory
            after the user ran ``cd`` in the terminal.  Patches would
            claim success and return a plausible diff but land in the
            original directory, producing apparent silent failures.
        """
        self.env = terminal_env
        # Determine cwd from various possible sources.
        # IMPORTANT: do NOT fall back to os.getcwd() -- that's the HOST's local
        # path which doesn't exist inside container/cloud backends (modal, docker).
        # If nothing provides a cwd, use "/" as a safe universal default.
        self.cwd = cwd or getattr(terminal_env, 'cwd', None) or \
                   getattr(getattr(terminal_env, 'config', None), 'cwd', None) or "/"

        # Cache for command availability checks
        self._command_cache: Dict[str, bool] = {}
    
    def _exec(self, command: str, cwd: str = None, timeout: int = None,
              stdin_data: str = None) -> ExecuteResult:
        """Execute command via terminal backend.

        Args:
            stdin_data: If provided, piped to the process's stdin instead of
                        embedding in the command string. Bypasses ARG_MAX.

        Cwd resolution order (critical — see class docstring):
          1. Explicit ``cwd`` arg (if provided)
          2. Live ``self.env.cwd`` (tracks ``cd`` commands run via terminal)
          3. Init-time ``self.cwd`` (fallback when env has no cwd attribute)

        This ordering ensures relative paths in file operations follow the
        terminal's current directory — not the directory this file_ops was
        originally created in.  See test_file_ops_cwd_tracking.py.
        """
        kwargs = {}
        if timeout:
            kwargs['timeout'] = timeout
        if stdin_data is not None:
            kwargs['stdin_data'] = stdin_data

        # Resolve cwd from the live env so `cd` commands are picked up.
        # Fall through to init-time self.cwd only if the env doesn't track cwd.
        effective_cwd = cwd or getattr(self.env, 'cwd', None) or self.cwd
        result = self.env.execute(command, cwd=effective_cwd, **kwargs)
        exit_code = result.get("returncode", 0)
        # A stdin write failure with an otherwise-clean child exit is still
        # a failure: the child never received the intended input. write_file
        # rejects such content up front (Task 3); this mapping is
        # defense-in-depth for any other stdin caller.
        if result.get("stdin_error") and exit_code == 0:
            exit_code = 1
        return ExecuteResult(
            stdout=result.get("output", ""),
            exit_code=exit_code
        )
    
    def _has_command(self, cmd: str) -> bool:
        """Check if a command exists in the environment (cached)."""
        if cmd not in self._command_cache:
            result = self._exec(f"command -v {cmd} >/dev/null 2>&1 && echo 'yes'")
            self._command_cache[cmd] = result.stdout.strip() == 'yes'
        return self._command_cache[cmd]
    
    def _sample_file_bytes(self, path: str, length: int = 1000):
        """Fetch the first ``length`` raw bytes of a file through the terminal.

        File operations run through a terminal backend (possibly remote), so
        raw bytes cannot cross the transport directly — the terminal decodes
        stdout with ``errors="replace"`` and manufactures U+FFFD at every
        byte it cannot decode, including a multibyte character cut in half by
        ``head -c``. Wrapping the sample in base64 lets the original bytes
        survive the transport, so binary detection can happen at the byte
        layer where it is well-defined (#80308 and friends).

        Returns the sample bytes, or ``None`` when the transport could not
        produce clean base64 (exotic shells without ``base64``); callers fall
        back to the legacy text-sample heuristic in that case.
        """
        result = self._exec(
            f"head -c {length} {self._escape_shell_arg(path)} 2>/dev/null | base64"
        )
        if result.exit_code != 0:
            return None
        encoded = _strip_terminal_fence_leaks(result.stdout)
        encoded = "".join(encoded.split())
        if not encoded:
            return b""
        if not re.fullmatch(r"[A-Za-z0-9+/]+={0,2}", encoded):
            return None
        try:
            return base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            return None

    @staticmethod
    def _is_likely_binary_bytes(sample: bytes) -> bool:
        """Byte-layer binary detection (the boundary for the #80308 class).

        Contract: a file is text when its sample is valid UTF-8, allowing one
        incomplete multibyte sequence at the very end (an artifact of cutting
        the sample at a byte boundary, not a property of the file). Anything
        else — NUL bytes, mid-stream invalid UTF-8 such as latin-1 or true
        binaries — stays read-only, preserving the anti-mojibake guarantee
        the old U+FFFD check existed for: a read→edit→write round-trip must
        never rewrite undecodable bytes with replacement characters.

        A file that legitimately *contains* U+FFFD (EF BF BD — e.g. logs of
        lossy output) is valid UTF-8 and reads as text; the old text-layer
        check misclassified it because it could not tell a stored replacement
        character from a transport-manufactured one.
        """
        if not sample:
            return False
        if b"\x00" in sample:
            return True
        try:
            sample.decode("utf-8")
            return False
        except UnicodeDecodeError as exc:
            # UTF-8 sequences are at most 4 bytes: an error starting in the
            # last 3 bytes with a clean prefix is a boundary cut, not binary.
            if exc.start >= len(sample) - 3:
                try:
                    sample[: exc.start].decode("utf-8")
                    return False
                except UnicodeDecodeError:
                    pass
            return True

    def _is_likely_binary(self, path: str, content_sample: str = None) -> bool:
        """
        Check if a file is likely binary.
        
        Uses extension check (fast) + content analysis (fallback).
        """
        ext = os.path.splitext(path)[1].lower()
        if ext in BINARY_EXTENSIONS:
            return True
        
        # Content analysis: >30% non-printable chars = binary
        if content_sample:
            # Undecodable bytes: the terminal env decodes stdout with
            # errors="replace", so any non-UTF-8 byte arrives here already
            # turned into U+FFFD. That char is "printable" (ord 65533), so the
            # non-printable ratio below never catches it — and returning the
            # lossy text would let a read→edit→write round-trip silently
            # overwrite the original bytes with mojibake. Treat a file whose
            # sample carries the replacement char as binary (read-only) so the
            # agent can't corrupt it. Legitimate UTF-8 text effectively never
            # contains U+FFFD.
            if "\ufffd" in content_sample[:1000]:
                return True
            non_printable = sum(1 for c in content_sample[:1000]
                               if ord(c) < 32 and c not in '\n\r\t')
            return non_printable / min(len(content_sample), 1000) > 0.30
        
        return False
    
    def _is_image(self, path: str) -> bool:
        """Check if file is an image we can return as base64."""
        ext = os.path.splitext(path)[1].lower()
        return ext in IMAGE_EXTENSIONS
    
    def _add_line_numbers(self, content: str, start_line: int = 1) -> str:
        """Add line numbers to content in ``LINE_NUM|CONTENT`` format.

        The gutter uses a compact ``<n>|`` prefix (e.g. ``34|foo``) rather
        than a fixed-width zero/space-padded one (``    34|foo``). The
        padding was pure token overhead: on dense source the padded gutter
        cost ~48% more tokens than the bare content and ~16% more than the
        compact form, because the leading spaces + zero-padding tokenize
        into extra tokens on every single line. An A/B (Sonnet 4.6, 2
        passes) showed the compact gutter matches the padded gutter on
        line-reference / patch / value-lookup / structure tasks (4/4 both),
        while dropping line numbers entirely regressed line-referencing
        (the model hand-counted and was off-by-one, 3/4) — so we keep the
        numbers, just not the padding.
        """
        from tools.tool_output_limits import get_max_line_length
        max_line_length = get_max_line_length()
        lines = content.split('\n')
        numbered = []
        for i, line in enumerate(lines, start=start_line):
            # Truncate long lines
            if len(line) > max_line_length:
                line = line[:max_line_length] + "... [truncated]"
            numbered.append(f"{i}|{line}")
        return '\n'.join(numbered)
    
    def _expand_path(self, path: str) -> str:
        """
        Expand shell-style paths like ~ and ~user to absolute paths.
        
        This must be done BEFORE shell escaping, since ~ doesn't expand
        inside single quotes.
        """
        if not path:
            return path
        
        # Handle ~ and ~user
        if path.startswith('~'):
            # Get home directory via the terminal environment
            result = self._exec("echo $HOME")
            if result.exit_code == 0 and result.stdout.strip():
                home = result.stdout.strip()
                if path == '~':
                    return home
                elif path.startswith('~/'):
                    return home + path[1:]  # Replace ~ with home
                # ~username format - extract and validate username before
                # letting shell expand it (prevent shell injection via
                # paths like "~; rm -rf /").
                rest = path[1:]  # strip leading ~
                slash_idx = rest.find('/')
                username = rest[:slash_idx] if slash_idx >= 0 else rest
                if username and re.fullmatch(r'[a-zA-Z0-9._-]+', username):
                    # Only expand ~username (not the full path) to avoid shell
                    # injection via path suffixes like "~user/$(malicious)".
                    expand_result = self._exec(f"echo ~{username}")
                    if expand_result.exit_code == 0 and expand_result.stdout.strip():
                        user_home = expand_result.stdout.strip()
                        suffix = path[1 + len(username):]  # e.g. "/rest/of/path"
                        return user_home + suffix
        
        return path
    
    def _escape_shell_arg(self, arg: str) -> str:
        """Escape a string for safe use in shell commands.

        On Windows native drive paths (``C:\\Users\\x`` / ``C:/Users/x``)
        and mixed MSYS leftovers (``/c/Users\\x``) are rewritten to the
        Git Bash ``/c/Users/x`` form via ``_bash_safe_path``: bash eats
        backslashes and MSYS otherwise mangles drive paths into the
        ``Directory \\drivers\\etc does not exist`` failure class. Reuses
        the env-layer translator so shell file ops and the terminal ``cd``
        agree on the path form. No-op off Windows and for plain POSIX paths.
        """
        from tools.environments.local import _bash_safe_path

        arg = _bash_safe_path(arg)
        # Use single quotes and escape any single quotes in the string
        return "'" + arg.replace("'", "'\"'\"'") + "'"

    def _escape_native_tool_arg(self, arg: str) -> str:
        """Escape a path argument destined for a NATIVE Windows binary.

        ``_escape_shell_arg`` rewrites Windows paths to the Git Bash MSYS
        form (``/c/Users/x``) so bash builtins resolve them. But native
        Windows binaries invoked from that bash (ripgrep installed via
        winget/cargo/choco, native git, etc.) do not understand ``/c/...``
        paths — and Hermes disables MSYS argument conversion for its bash
        subprocesses (``MSYS_NO_PATHCONV=1`` / ``MSYS2_ARG_CONV_EXCL=*``,
        see ``_apply_windows_msys_bash_env_defaults``), so nothing ever
        translates the MSYS form back. The native tool then fails with
        ``The system cannot find the path specified. (os error 3)``.

        The forward-slash native form (``C:/Users/x``) is the one spelling
        every layer accepts: bash passes it through untouched (it is not an
        absolute POSIX path, so no conversion applies even without the
        opt-outs), and Windows APIs treat ``/`` and ``\\`` as equivalent
        separators. MSYS builds of the same tools accept it too, so this is
        safe regardless of which flavor of the binary is installed.

        On non-Windows hosts this is exactly ``_escape_shell_arg``.
        """
        from tools.environments.local import _IS_WINDOWS, _msys_to_windows_path

        if _IS_WINDOWS and arg:
            arg = _msys_to_windows_path(arg).replace("\\", "/")
        return "'" + arg.replace("'", "'\"'\"'") + "'"

    def _atomic_write(self, path: str, content: str) -> "ExecuteResult":
        """Write ``content`` to ``path`` atomically via temp-file + rename.

        Streams ``content`` over stdin into a temp file in the SAME
        directory as ``path`` (so the final ``mv`` is a real rename on the
        same filesystem, not a non-atomic cross-device copy), preserves the
        existing file's mode if it exists, then renames over the target.
        On any failure the temp file is removed so we never leak a partial
        ``.hermes-tmp`` file next to the user's data, and the original file
        is left untouched. Content rides stdin so there is no ARG_MAX limit.

        ``mkdir -p`` for the parent directory is folded into this script
        (one fewer subprocess vs. a separate ``mkdir -p`` call).

        Returns an :class:`ExecuteResult`; ``exit_code == 0`` means the file
        was swapped into place atomically. A non-zero exit means nothing was
        renamed and the original (if any) is intact.
        """
        q_path = self._escape_shell_arg(path)
        parent = os.path.dirname(path) or "."
        q_parent = self._escape_shell_arg(parent)
        # template basename: hidden so it doesn't show up in casual `ls`,
        # carries a marker so an orphaned temp (only possible on a hard
        # crash *between* cat and mv) is identifiable.
        tmpl = self._escape_shell_arg(".hermes-tmp.XXXXXX")

        # One shell script, fully quoted. Notes:
        #  - `mkdir -p "$d"` is folded in here so the parent directory is
        #    created in the same subprocess that writes the temp file —
        #    saves one entire subprocess spawn vs. a separate mkdir call.
        #  - `mktemp` lands the temp in the target's own dir (-p) so `mv` is
        #    same-FS atomic; we fall back to a PID-stamped name if the
        #    backend lacks mktemp (rare; busybox/macOS/Linux all ship it).
        #  - `chmod --reference` is GNU-only, so we read the octal mode with
        #    `stat` (GNU `-c%a` or BSD `-f%Lp`) and `chmod` it explicitly;
        #    silent best-effort — a perms-copy failure must not abort the
        #    write (the file then lands at mktemp's 0600, same as pre-fix).
        #  - brand-new targets get `chmod "=rw"` — the POSIX who-less
        #    symbolic form, which sets rw minus the process umask (e.g.
        #    0644 under umask 022) instead of mktemp's hardcoded 0600
        #    (#70856).  Deliberately NOT shell arithmetic on `$(umask)`:
        #    zsh (reachable via _find_bash's $SHELL fallback) parses
        #    leading-zero constants as decimal and silently computes a
        #    garbage mode, while `chmod "=rw"` is spec-identical in
        #    bash/dash/ash/zsh and degrades to 0600 (pre-fix behavior)
        #    if an exotic chmod rejects it.
        #  - `trap ... EXIT` guarantees the temp is removed on every error
        #    path (cat failure, mv failure, signal) but NOT after a
        #    successful mv (the temp no longer exists by then).
        #  - we `cat >` the temp, then `mv -f` it over the target.
        script = (
            "set -e; "
            f"d={q_parent}; t={q_path}; "
            # Follow a symlink target so we edit the file the link points at,
            # rather than replacing the symlink itself with a plain file (which
            # orphans the real target and destroys the link). Recompute the
            # temp dir from the RESOLVED target so `mv` stays same-filesystem
            # atomic. Best-effort: a broken link or missing readlink/realpath
            # falls back to the original path (pre-fix behavior, no regression).
            'if [ -L "$t" ]; then '
            'rt="$(readlink -f "$t" 2>/dev/null || realpath "$t" 2>/dev/null || true)"; '
            '[ -n "$rt" ] && { t="$rt"; d="$(dirname "$t")"; }; '
            "fi; "
            # Create the parent dir in the SAME subprocess that writes the
            # temp file (one fewer exec vs. a separate mkdir call). Runs
            # AFTER symlink resolution so a resolved target's directory is
            # the one created/confirmed.
            'mkdir -p "$d"; '
            'tmp="$(mktemp -p "$d" ' + tmpl + ' 2>/dev/null '
            '|| mktemp "$d/.hermes-tmp.$$.XXXXXX" 2>/dev/null '
            '|| { tmp="$d/.hermes-tmp.$$"; : > "$tmp" && echo "$tmp"; })"; '
            '[ -n "$tmp" ] || { echo "atomic write: could not create temp file" >&2; exit 1; }; '
            "trap 'rm -f \\\"$tmp\\\"' EXIT; "
            # preserve mode of an existing target (best-effort, never fatal)
            'if [ -e "$t" ]; then '
            'm="$(stat -c%a "$t" 2>/dev/null || stat -f%Lp "$t" 2>/dev/null || true)"; '
            '[ -n "$m" ] && chmod "$m" "$tmp" 2>/dev/null || true; '
            "fi; "
            'cat > "$tmp"; '
            # new file: umask-default perms instead of mktemp's 0600 (#70856).
            # Runs AFTER cat so a write-masking umask can't EACCES the stream;
            # quoted "=rw" so zsh doesn't =word-expand it.
            'if [ ! -e "$t" ]; then chmod "=rw" "$tmp" 2>/dev/null || true; fi; '
            'mv -f "$tmp" "$t"; '
            "trap - EXIT"
        )
        return self._exec(script, stdin_data=content)

    def _detect_file_line_ending(self, path: str, pre_content: Optional[str] = None) -> Optional[str]:
        """Detect the dominant line ending of a file on disk.

        If ``pre_content`` is already available (we just read the file
        for lint/LSP purposes), inspect that — zero extra exec calls.
        Otherwise issue a tiny ``head -c 4096`` to sample the first 4KB.

        Returns ``"\\r\\n"`` for CRLF (Windows), ``"\\n"`` for LF (Unix),
        or ``None`` if undetermined (new file, empty file, single-line
        file with no line break in the first chunk).
        """
        if pre_content:
            return _detect_line_ending(pre_content)
        # File may not exist (new write) — `head` exits 0 with empty
        # stdout in that case which yields None below.  Cheap probe.
        head_cmd = f"head -c 4096 {self._escape_shell_arg(path)} 2>/dev/null"
        head_result = self._exec(head_cmd)
        if head_result.exit_code != 0 or not head_result.stdout:
            return None
        return _detect_line_ending(head_result.stdout)

    def _file_has_bom(self, path: str, pre_content: Optional[str] = None) -> bool:
        """Whether the file on disk starts with a UTF-8 BOM.

        Always probes the first 3 bytes on disk — do NOT trust
        ``pre_content`` for BOM detection because the most common
        provider (``read_file_raw``) deliberately strips BOMs so the
        agent never sees U+FEFF glyphs.  Passing BOM-stripped content
        through ``pre_content`` would cause a false-negative and
        silently remove the marker on rewrite.

        A missing/empty file returns False (new writes get no BOM
        unless the caller explicitly includes one).
        """
        head_cmd = f"head -c 3 {self._escape_shell_arg(path)} 2>/dev/null"
        head_result = self._exec(head_cmd)
        if head_result.exit_code != 0 or not head_result.stdout:
            return False
        return _has_bom(head_result.stdout)


    def _unified_diff(self, old_content: str, new_content: str, filename: str) -> str:
        """Generate unified diff between old and new content."""
        old_lines = old_content.splitlines(keepends=True)
        new_lines = new_content.splitlines(keepends=True)
        diff = difflib.unified_diff(
            old_lines, new_lines,
            fromfile=f"a/{filename}",
            tofile=f"b/{filename}"
        )
        return ''.join(diff)
    
    # =========================================================================
    # READ Implementation
    # =========================================================================
    
    def _size_probe_cmd(self, path: str) -> str:
        """Byte size of a regular file, without opening one that never ends.

        ``wc -c < path`` opens the path. On a FIFO with no writer, a socket,
        or a character device like /dev/zero that never reaches EOF, that
        read blocks forever — and the read helpers pass no timeout to
        :meth:`_exec`, so the turn wedges until the process is killed. The
        device blocklist in ``tools/file_tools.py`` cannot cover this: it
        matches literal ``/dev/*`` names, while a FIFO is a file *type* and
        can sit at any path.

        ``[ -f ]`` is a stat, not an open — it answers exactly the question
        the size probe needs (regular file, symlinks followed) without
        touching the contents. Non-regular paths that exist report the
        sentinel so callers can say so instead of claiming the file is
        missing; a genuinely absent path still exits non-zero.
        """
        arg = self._escape_shell_arg(path)
        return (
            f"if [ -f {arg} ]; then wc -c < {arg} 2>/dev/null; "
            f"elif [ -e {arg} ]; then echo {NOT_REGULAR_SENTINEL}; "
            f"else exit 1; fi"
        )

    @staticmethod
    def _not_regular_error(path: str) -> ReadResult:
        """Error for a path that exists but would block if read."""
        return ReadResult(
            error=(
                f"Cannot read '{path}': not a regular file (directory, FIFO, "
                "socket, or device). Reading it could block indefinitely."
            )
        )

    # UTF-16 rescue constants (ported from MoonshotAI/kimi-code#2647,
    # detection derived from VS Code's encoding sniffer): sample the leading
    # bytes; trust a BOM first, then a zero-byte parity heuristic — zeros
    # clustering at odd indices mean UTF-16 LE (`0xAA 0x00`), at even indices
    # UTF-16 BE (`0x00 0xAA`). Only the *placement* of zeros is checked, not
    # density, so mixed Latin/CJK content (whose CJK units carry no zero
    # byte) still detects. Zeros at both parities, or a single isolated
    # zero, mean real binary. Legacy 8-bit encodings (GBK, Big5, ...) are
    # never guessed — a wrong silent guess is worse than a clear refusal.
    _UTF16_MAX_BYTES = 10 * 1024 * 1024
    _UTF16_SAMPLE_BYTES = 512

    def _try_read_utf16(self, path: str, offset: int, limit: int,
                        file_size: int) -> "Optional[ReadResult]":
        """Attempt to read ``path`` as UTF-16 text, transcoded to UTF-8.

        Returns a populated ``ReadResult`` when the file is UTF-16 (BOM or
        zero-byte parity heuristic), else ``None`` so the caller falls back
        to the binary-file error. Files over 10 MiB are not rescued.
        ``path`` must already be expanded (caller ran ``_expand_path``).
        """
        # Extensions that are definitively binary (images, archives, ...)
        # never contain UTF-16 text worth rescuing — skip the subprocess.
        ext = os.path.splitext(path)[1].lower()
        if ext in BINARY_EXTENSIONS:
            return None
        if file_size > self._UTF16_MAX_BYTES:
            return None

        snippet = (
            "import sys, json, os\n"
            f"p = {path!r}\n"
            f"offset = {int(offset)}\n"
            f"limit = {int(limit)}\n"
            f"MAX = {self._UTF16_MAX_BYTES}\n"
            f"SAMPLE = {self._UTF16_SAMPLE_BYTES}\n"
            "try:\n"
            "    size = os.path.getsize(p)\n"
            "    if size > MAX:\n"
            "        print('HERMES_UTF16:NO'); sys.exit(0)\n"
            "    with open(p, 'rb') as f:\n"
            "        data = f.read()\n"
            "    sample = data[:SAMPLE]\n"
            "    enc = None\n"
            "    if sample[:2] == b'\\xfe\\xff':\n"
            "        enc = 'utf-16-be'\n"
            "    elif sample[:2] == b'\\xff\\xfe':\n"
            "        enc = 'utf-16-le'\n"
            "    else:\n"
            "        odd = sum(1 for i in range(1, len(sample), 2) if sample[i] == 0)\n"
            "        even = sum(1 for i in range(0, len(sample), 2) if sample[i] == 0)\n"
            "        if even == 0 and odd >= 2:\n"
            "            enc = 'utf-16-le'\n"
            "        elif odd == 0 and even >= 2:\n"
            "            enc = 'utf-16-be'\n"
            "    if enc is None:\n"
            "        print('HERMES_UTF16:NO'); sys.exit(0)\n"
            "    text = data.decode(enc, 'replace')\n"
            "    if text[:1] == '\\ufeff':\n"
            "        text = text[1:]\n"
            "    text = text.replace('\\r\\n', '\\n')\n"
            "    lines = text.split('\\n')\n"
            "    total = len(lines)\n"
            "    sel = lines[offset - 1: offset - 1 + limit]\n"
            "    out = {'total_lines': total, 'encoding': enc,\n"
            "           'content': '\\n'.join(sel)}\n"
            "    print('HERMES_UTF16:OK')\n"
            "    print(json.dumps(out, ensure_ascii=True))\n"
            "except Exception:\n"
            "    print('HERMES_UTF16:NO'); sys.exit(0)\n"
        )

        result = self._exec(f"python3 -c {self._escape_shell_arg(snippet)}")
        if result.exit_code != 0 and "python3" in (result.stdout or ""):
            result = self._exec(f"python -c {self._escape_shell_arg(snippet)}")

        stdout = _strip_terminal_fence_leaks(result.stdout or "")
        marker = stdout.find("HERMES_UTF16:OK")
        if result.exit_code != 0 or marker < 0:
            return None
        payload = stdout[marker + len("HERMES_UTF16:OK"):].strip()
        try:
            data = json.loads(payload.split("\n", 1)[0] if "\n" in payload else payload)
            content = data["content"]
            total_lines = int(data["total_lines"])
            encoding = str(data.get("encoding", "utf-16"))
        except (ValueError, KeyError, TypeError):
            return None

        end_line = offset + limit - 1
        truncated = total_lines > end_line
        hint_parts = [f"Transcoded from {encoding.upper()} to UTF-8 for display. "
                      "Text edits via patch/write_file would re-encode as UTF-8."]
        if truncated:
            hint_parts.append(
                f"Use offset={end_line + 1} to continue reading "
                f"(showing {offset}-{end_line} of {total_lines} lines)"
            )
        return ReadResult(
            content=self._add_line_numbers(content, offset),
            total_lines=total_lines,
            file_size=file_size,
            truncated=truncated,
            hint=" ".join(hint_parts),
        )

    def read_file(self, path: str, offset: int = 1, limit: int = 2000) -> ReadResult:
        """
        Read a file with pagination, binary detection, and line numbers.
        
        Args:
            path: File path (absolute or relative to cwd)
            offset: Line number to start from (1-indexed, default 1)
            limit: Maximum lines to return (default 500, max 2000)
        
        Returns:
            ReadResult with content, metadata, or error info
        """
        # Expand ~ and other shell paths
        path = self._expand_path(path)
        
        offset, limit = normalize_read_pagination(offset, limit)
        
        # Check if file exists and get size (POSIX, works on Linux + macOS)
        stat_result = self._exec(self._size_probe_cmd(path))

        if stat_result.exit_code != 0:
            # File not found. Before failing, try unicode-equivalent
            # spellings — NFC/NFD, narrow no-break space, curly quotes
            # render identically in a terminal, so the model retyping a
            # visually-correct path can never discover the byte mismatch
            # on its own (retrying is the tool's job, not the model's).
            variant = self._unicode_variant_match(path)
            if variant is not None:
                result = self.read_file(variant, offset=offset, limit=limit)
                note = (
                    f"Note: '{path}' not found byte-for-byte; resolved to "
                    f"the unicode-equivalent file '{variant}' (invisible "
                    "encoding difference: NFC/NFD or special space/quote "
                    "characters)."
                )
                result.hint = f"{note} {result.hint}" if result.hint else note
                return result
            # No equivalent spelling — suggest similar files
            return self._suggest_similar_files(path)

        stat_output = _strip_terminal_fence_leaks(stat_result.stdout)
        if stat_output.strip() == NOT_REGULAR_SENTINEL:
            return self._not_regular_error(path)
        try:
            file_size = int(stat_output.strip())
        except ValueError:
            file_size = 0
        
        # Check if file is too large
        if file_size > MAX_FILE_SIZE:
            # Still try to read, but warn
            pass
        
        # Images are never inlined — redirect to the vision tool
        if self._is_image(path):
            return ReadResult(
                is_image=True,
                is_binary=True,
                file_size=file_size,
                hint=(
                    "Image file detected. Automatically redirected to vision_analyze tool. "
                    "Use vision_analyze with this file path to inspect the image contents."
                ),
            )
        
        # Read a sample to check for binary content — at the byte layer when
        # the transport allows, falling back to the legacy text heuristic.
        sample_bytes = self._sample_file_bytes(path)
        if sample_bytes is not None:
            ext_binary = os.path.splitext(path)[1].lower() in BINARY_EXTENSIONS
            is_binary = ext_binary or self._is_likely_binary_bytes(sample_bytes)
        else:
            sample_cmd = f"head -c 1000 {self._escape_shell_arg(path)} 2>/dev/null"
            sample_result = self._exec(sample_cmd)
            sample_output = _strip_terminal_fence_leaks(sample_result.stdout)
            is_binary = self._is_likely_binary(path, sample_output)

        if is_binary:
            # UTF-16 rescue (ported from MoonshotAI/kimi-code#2647): the
            # terminal env decodes stdout as UTF-8 with errors="replace", so
            # a UTF-16 text file (Windows Notepad .txt, PowerShell `>`
            # redirects) arrives mangled with U+FFFD and trips the binary
            # guard. Probe the raw bytes via the backend's Python and
            # transcode to UTF-8 when a BOM or the zero-byte parity
            # heuristic identifies UTF-16.
            utf16_result = self._try_read_utf16(path, offset, limit, file_size)
            if utf16_result is not None:
                return utf16_result
            return ReadResult(
                is_binary=True,
                file_size=file_size,
                error=describe_binary_file(sample_bytes, file_size),
            )
        
        # Read with pagination using sed, clamping each line to a byte
        # budget IN THE SHELL so a pathological single-line file (e.g. one
        # 400MB minified line) never crosses the exec transport. The Python
        # clamp in _add_line_numbers still runs afterwards; the shell clamp
        # only bounds what reaches it.
        #
        # Why 4*max_line_length + 1 bytes (not max_line_length + 1):
        # ``cut -c`` on GNU coreutils is byte-based despite its name, and a
        # byte clamp can split a multibyte UTF-8 codepoint at the boundary.
        # The transport decodes with errors="replace", so a split codepoint
        # becomes U+FFFD rather than an exception — but a clamp of
        # max_line_length+1 BYTES yields far fewer CHARS than
        # max_line_length for multibyte text, so the Python clamp would
        # never fire and truncation would be silent (no "... [truncated]"
        # suffix). UTF-8 codepoints are at most 4 bytes, so any line whose
        # first max_line_length chars survive occupies at most
        # 4*max_line_length bytes; keeping one byte more guarantees that
        # every line longer than max_line_length chars still decodes to
        # more than max_line_length chars, which triggers the existing
        # Python-side clamp (len(line) > max_line_length) and its
        # "... [truncated]" suffix. Any U+FFFD from a boundary split lands
        # beyond char max_line_length and is always removed by that clamp,
        # so mojibake is never visible. ``cut -b`` is used explicitly to
        # document the byte semantics.
        from tools.tool_output_limits import get_max_line_length
        line_clamp_bytes = 4 * get_max_line_length() + 1
        end_line = offset + limit - 1
        read_cmd = (
            f"sed -n '{offset},{end_line}p' {self._escape_shell_arg(path)}"
            f" | cut -b1-{line_clamp_bytes}"
        )
        read_result = self._exec(read_cmd)
        
        if read_result.exit_code != 0:
            return ReadResult(error=f"Failed to read file: {read_result.stdout}")
        read_output = _strip_terminal_fence_leaks(read_result.stdout)
        # Strip a leading UTF-8 BOM so the model never sees a phantom U+FEFF
        # before the first real character. Only meaningful on the first
        # chunk (the marker lives at byte 0); later pages can't carry it.
        if offset == 1:
            read_output, _ = _strip_bom(read_output)
        
        # Get total line count
        wc_cmd = f"wc -l < {self._escape_shell_arg(path)}"
        wc_result = self._exec(wc_cmd)
        wc_output = _strip_terminal_fence_leaks(wc_result.stdout)
        try:
            total_lines = int(wc_output.strip())
        except ValueError:
            total_lines = 0
        
        # Check if truncated
        truncated = total_lines > end_line
        hint = None
        if truncated:
            hint = f"Use offset={end_line + 1} to continue reading (showing {offset}-{end_line} of {total_lines} lines)"

        # ``cut`` (unlike sed -n p) always newline-terminates its output,
        # so a file whose final line has no trailing newline would grow a
        # phantom empty last line. Only possible when this page reaches the
        # file's final line; probe the last byte and strip the artifact.
        if not truncated and read_output.endswith('\n'):
            tail_cmd = f"tail -c 1 {self._escape_shell_arg(path)} | wc -l"
            tail_result = self._exec(tail_cmd)
            tail_output = _strip_terminal_fence_leaks(tail_result.stdout)
            if tail_result.exit_code == 0 and tail_output.strip() == "0":
                read_output = read_output[:-1]

        # Ambiguous-silence guards: an empty content string is
        # indistinguishable, from inside the model, from a broken tool —
        # it re-reads, widens the window, tries another path. Name the
        # dead end and its recovery instead.
        if file_size == 0:
            return ReadResult(
                content="",
                total_lines=0,
                file_size=0,
                hint="File is empty (0 bytes).",
            )
        if offset > total_lines > 0:
            return ReadResult(
                content="",
                total_lines=total_lines,
                file_size=file_size,
                hint=(
                    f"Note: offset {offset} is beyond the end of the file "
                    f"({total_lines} lines total). Retry with offset <= "
                    f"{total_lines}."
                ),
            )

        return ReadResult(
            content=self._add_line_numbers(read_output, offset),
            total_lines=total_lines,
            file_size=file_size,
            truncated=truncated,
            hint=hint
        )
    
    def _unicode_variant_match(self, path: str) -> Optional[str]:
        """Find an existing file whose name is unicode-equivalent to ``path``.

        macOS names screenshots with a NARROW NO-BREAK SPACE (U+202F) before
        AM/PM, stores names NFD-decomposed, and Finder renames turn ' into
        \u2019 — all invisible in rendered text. Compare directory entries
        under a normalization that erases exactly those differences and
        return the on-disk spelling when exactly one entry matches.
        """
        dir_path = os.path.dirname(path) or "."
        filename = os.path.basename(path)
        if not filename:
            return None

        def _canon(name: str) -> str:
            # NFC first so composed/decomposed collapse together, then the
            # confusable space/quote characters seen in real filenames.
            out = unicodedata.normalize("NFC", name)
            for src, dst in (
                ("\u202f", " "),  # narrow no-break space
                ("\u00a0", " "),  # no-break space
                ("\u2019", "'"),  # right single quotation mark
                ("\u2018", "'"),  # left single quotation mark
            ):
                out = out.replace(src, dst)
            return out

        target = _canon(filename)
        ls_cmd = f"ls -1 {self._escape_shell_arg(dir_path)} 2>/dev/null"
        ls_result = self._exec(ls_cmd)
        if ls_result.exit_code != 0 or not ls_result.stdout.strip():
            return None
        candidates = [
            entry
            for entry in _strip_terminal_fence_leaks(ls_result.stdout).splitlines()
            if entry and entry != filename and _canon(entry) == target
        ]
        # Exactly one equivalent spelling = unambiguous repair. Zero or
        # several = fall through to suggestions; guessing among homoglyph
        # collisions would silently read the wrong file.
        if len(candidates) == 1:
            return os.path.join(dir_path, candidates[0]) if dir_path != "." or "/" in path else candidates[0]
        return None

    def _suggest_similar_files(self, path: str) -> ReadResult:
        """Suggest similar files when the requested file is not found."""
        dir_path = os.path.dirname(path) or "."
        filename = os.path.basename(path)
        basename_no_ext = os.path.splitext(filename)[0]
        ext = os.path.splitext(filename)[1].lower()
        lower_name = filename.lower()

        # List files in the target directory
        ls_cmd = f"ls -1 {self._escape_shell_arg(dir_path)} 2>/dev/null | head -50"
        ls_result = self._exec(ls_cmd)

        scored: list = []  # (score, filepath) — higher is better
        if ls_result.exit_code == 0 and ls_result.stdout.strip():
            for f in ls_result.stdout.strip().split('\n'):
                if not f:
                    continue
                lf = f.lower()
                score = 0

                # Exact match (shouldn't happen, but guard)
                if lf == lower_name:
                    score = 100
                # Same base name, different extension (e.g. config.yml vs config.yaml)
                elif os.path.splitext(f)[0].lower() == basename_no_ext.lower():
                    score = 90
                # Target is prefix of candidate or vice-versa
                elif lf.startswith(lower_name) or lower_name.startswith(lf):
                    score = 70
                # Substring match (candidate contains query)
                elif lower_name in lf:
                    score = 60
                # Reverse substring (query contains candidate name)
                elif lf in lower_name and len(lf) > 2:
                    score = 40
                # Same extension with some overlap
                elif ext and os.path.splitext(f)[1].lower() == ext:
                    common = set(lower_name) & set(lf)
                    if len(common) >= max(len(lower_name), len(lf)) * 0.4:
                        score = 30
                # Near-miss spelling (AGENT.md -> AGENTS.md): substring
                # checks above find nothing, but a high sequence ratio
                # catches 1-2 edit typos without a homegrown levenshtein.
                if score == 0 and difflib.SequenceMatcher(
                    None, lower_name, lf
                ).ratio() >= 0.8:
                    score = 50

                if score > 0:
                    scored.append((score, os.path.join(dir_path, f)))

        scored.sort(key=lambda x: -x[0])
        similar = [fp for _, fp in scored[:5]]

        return ReadResult(
            error=f"File not found: {path}",
            similar_files=similar
        )
    
    def read_file_raw(self, path: str) -> ReadResult:
        """Read the complete file content as a plain string.

        No pagination, no line-number prefixes, no per-line truncation.
        Uses cat so the full file is returned regardless of size.
        """
        path = self._expand_path(path)
        stat_result = self._exec(self._size_probe_cmd(path))
        if stat_result.exit_code != 0:
            return self._suggest_similar_files(path)
        stat_output = _strip_terminal_fence_leaks(stat_result.stdout)
        if stat_output.strip() == NOT_REGULAR_SENTINEL:
            return self._not_regular_error(path)
        try:
            file_size = int(stat_output.strip())
        except ValueError:
            file_size = 0
        if self._is_image(path):
            return ReadResult(is_image=True, is_binary=True, file_size=file_size)
        sample_bytes = self._sample_file_bytes(path)
        if sample_bytes is not None:
            ext_binary = os.path.splitext(path)[1].lower() in BINARY_EXTENSIONS
            is_binary = ext_binary or self._is_likely_binary_bytes(sample_bytes)
        else:
            sample_result = self._exec(f"head -c 1000 {self._escape_shell_arg(path)} 2>/dev/null")
            sample_output = _strip_terminal_fence_leaks(sample_result.stdout)
            is_binary = self._is_likely_binary(path, sample_output)
        if is_binary:
            return ReadResult(
                is_binary=True, file_size=file_size,
                error=describe_binary_file(sample_bytes, file_size),
            )
        cat_result = self._exec(f"cat {self._escape_shell_arg(path)}")
        if cat_result.exit_code != 0:
            return ReadResult(error=f"Failed to read file: {cat_result.stdout}")
        # Strip a leading UTF-8 BOM so patch's fuzzy matcher operates on
        # clean content (a phantom U+FEFF before line 1 would defeat an
        # exact first-line match). write_file restores the BOM on the way
        # back out — it re-probes the on-disk file, which still has the
        # marker — so the round-trip preserves it.
        raw_content, _ = _strip_bom(_strip_terminal_fence_leaks(cat_result.stdout))
        return ReadResult(
            content=raw_content,
            file_size=file_size,
        )

    def read_file_bytes(self, path: str, max_bytes: Optional[int] = None) -> ReadResult:
        """Read binary-safe bytes from any shell-backed environment."""
        path = self._expand_path(path)
        stat_result = self._exec(self._size_probe_cmd(path))
        if stat_result.exit_code != 0:
            return ReadResult(error=f"File not found: {path}")
        stat_output = _strip_terminal_fence_leaks(stat_result.stdout)
        if stat_output.strip() == NOT_REGULAR_SENTINEL:
            return self._not_regular_error(path)
        try:
            file_size = int(stat_output.strip())
        except ValueError:
            return ReadResult(error=f"Could not determine file size: {path}")
        if max_bytes is not None and file_size > max_bytes:
            return ReadResult(
                file_size=file_size,
                error=f"File is too large ({file_size:,} bytes, limit is {max_bytes:,})",
            )

        encoded = self._exec(f"base64 < {self._escape_shell_arg(path)}")
        if encoded.exit_code != 0:
            return ReadResult(error=f"Failed to read binary file: {encoded.stdout}")
        compact = "".join(_strip_terminal_fence_leaks(encoded.stdout).split())
        try:
            base64.b64decode(compact, validate=True)
        except (ValueError, base64.binascii.Error):
            return ReadResult(error=f"Backend returned invalid binary data for: {path}")
        return ReadResult(
            base64_content=compact,
            file_size=file_size,
            is_binary=True,
        )

    def delete_file(self, path: str) -> WriteResult:
        """Delete a single file.

        Cross-platform: runs via ``python -c`` against the terminal env's
        Python so it works on Windows shells (``cmd.exe``/PowerShell) that
        don't ship ``rm``. Directories are rejected here — use
        ``delete_path(recursive=True)`` for trees.
        """
        return self._python_delete(path, recursive=False)

    def delete_path(self, path: str, recursive: bool = False) -> WriteResult:
        """Cross-platform delete that handles files and (with recursive=True)
        directory trees. Always preferred over emitting ``rm -rf`` /
        ``Remove-Item -Recurse`` directly so the same tool call works on
        every backend (local / docker / ssh / Windows).
        """
        return self._python_delete(path, recursive=recursive)

    def _python_delete(self, path: str, recursive: bool) -> WriteResult:
        path = self._expand_path(path)
        denied = get_write_denied_error(path, verb="Delete")
        if denied:
            return WriteResult(error=denied)

        # We can't shell out to ``rm`` here — it doesn't exist on Windows
        # ``cmd.exe`` or PowerShell, so this code path is what's left when
        # the backend's terminal is a Windows shell. Path is baked into the
        # snippet via ``repr()`` so quoting is correct on every shell.
        snippet = (
            "import shutil, pathlib, sys\n"
            f"p = pathlib.Path({path!r})\n"
            f"recursive = {bool(recursive)!r}\n"
            "try:\n"
            "    if p.is_dir() and not p.is_symlink():\n"
            "        if recursive:\n"
            "            shutil.rmtree(p)\n"
            "        else:\n"
            "            print('is a directory: ' + str(p), file=sys.stderr); sys.exit(2)\n"
            "    else:\n"
            # NOTE: avoid ``unlink(missing_ok=True)`` — that kwarg lands in
            # Python 3.8 and the remote interpreter (docker/ssh) may still
            # be 3.7 on older distros. The FileNotFoundError handler below
            # covers the same case and works back to 3.4.
            "        p.unlink()\n"
            "except FileNotFoundError:\n"
            "    pass\n"
            "except Exception as exc:\n"
            "    print(str(exc), file=sys.stderr); sys.exit(1)\n"
        )

        result = self._exec(f"python3 -c {self._escape_shell_arg(snippet)}")

        # Fall back to ``python`` (Windows / older systems where there's no
        # ``python3`` symlink but a ``python`` binary is on PATH).
        if result.exit_code != 0 and "python3" in (result.stdout or ""):
            result = self._exec(f"python -c {self._escape_shell_arg(snippet)}")

        if result.exit_code != 0:
            return WriteResult(error=f"Failed to delete {path}: {(result.stdout or '').strip() or 'unknown error'}")

        return WriteResult()

    def move_file(self, src: str, dst: str) -> WriteResult:
        """Move a file via mv."""
        src = self._expand_path(src)
        dst = self._expand_path(dst)
        for p in (src, dst):
            denied = get_write_denied_error(p, verb="Move")
            if denied:
                return WriteResult(error=denied)
        result = self._exec(
            f"mv {self._escape_shell_arg(src)} {self._escape_shell_arg(dst)}"
        )
        if result.exit_code != 0:
            return WriteResult(error=f"Failed to move {src} -> {dst}: {result.stdout}")
        return WriteResult()

    # =========================================================================
    # WRITE Implementation
    # =========================================================================

    def write_file(self, path: str, content: str,
                   pre_content: Optional[str] = None) -> WriteResult:
        """
        Write content to a file, creating parent directories as needed.

        Pipes content through stdin to avoid OS ARG_MAX limits on large
        files. The content never appears in the shell command string —
        only the file path does.

        Before anything touches disk, a fail-closed syntax gate runs
        against the CANDIDATE content: if ``path``'s extension is in
        ``_FAIL_CLOSED_INPROC_EXTS`` (JSON/YAML/TOML — structured data
        formats where a parse failure always means corruption) and the
        candidate content doesn't parse, the write is refused outright.
        No temp file, no rename, nothing on disk changes.

        After a write that clears the gate, runs a post-first / pre-lazy
        lint check via ``_check_lint_delta()``.  If the new content is
        clean, the lint call is O(one parse).  If the new content has
        errors the gate didn't already catch (i.e. errors from a linter
        outside ``_FAIL_CLOSED_INPROC_EXTS``, such as Python), the
        pre-write content is linted too and only errors newly introduced
        by this write are surfaced — pre-existing problems are filtered
        out so the agent isn't distracted chasing them.

        Args:
            path: File path to write
            content: Content to write
            pre_content: Pre-edit file content if the caller already has it
                (e.g. patch_replace read the file for fuzzy matching).
                When provided, skips a redundant ``cat`` subprocess to
                re-read the file for lint baseline / line-ending
                detection. BOM detection always probes disk (the most
                common provider — ``read_file_raw`` — strips BOMs, so
                trusting ``pre_content`` for BOM would cause false
                negatives and silent marker loss on rewrite). When
                None, reads from disk as before.

        Returns:
            WriteResult with bytes written, lint summary, or error.
        """
        # Expand ~ and other shell paths
        path = self._expand_path(path)

        # Block writes to sensitive paths
        denied = get_write_denied_error(path)
        if denied:
            return WriteResult(error=denied)

        # Reject lone surrogates up front with a regex scan (no encode, no
        # subprocess). surrogateescape-decoded content (U+DC80–U+DCFF)
        # round-trips through the pipe fine, but surrogates outside that
        # range cannot be encoded at all — and letting them reach the pipe
        # would spawn a child that then hangs, or truncates the target via
        # empty-stdin `cat`. Refuse synchronously before any subprocess.
        m = re.search(r"[\ud800-\udc7f\udd00-\udfff]", content)
        if m:
            return WriteResult(
                error=(
                    f"Refusing to write '{path}': content contains a lone "
                    f"surrogate character ({m.group(0)!r}) that cannot be "
                    "encoded as UTF-8. The file was NOT created or modified."
                )
            )

        # ── Fail-closed pre-write syntax gate ───────────────────────────
        # Validate the CANDIDATE content BEFORE any bytes touch disk —
        # previously this only ran as a post-write lint *report* that the
        # caller could ignore (or that ``files_modified`` gating wouldn't
        # catch, since a lint failure never set the top-level ``error``
        # key). A structured-format write that doesn't even parse (mashed
        # quotes, truncated generation, wrong indentation dialect) is a
        # corrupt write, not a style nit — refuse it outright instead of
        # writing first and reporting the damage afterward.
        #
        # Scope: only extensions in ``_FAIL_CLOSED_INPROC_EXTS`` (JSON/
        # YAML/TOML). ``.py`` deliberately keeps its pre-existing,
        # non-blocking lint-delta *report* instead of a hard refusal — see
        # ``_FAIL_CLOSED_INPROC_EXTS``'s docstring above for why. Extensions
        # with no in-process linter at all (including ones only covered by
        # a shell linter) are completely unaffected — this gate never runs
        # for them, so behavior there is unchanged.
        #
        # Checked against the raw ``content`` argument, before the
        # BOM/CRLF preservation shims below run. Those shims exist purely
        # to match the on-disk file's existing conventions; linting
        # post-shim would false-positive a JSONDecodeError on a
        # legitimately BOM-marked JSON file purely because this method
        # re-adds the marker the read layer strips — see
        # ``_file_has_bom``/``_UTF8_BOM`` below.
        ext = os.path.splitext(path)[1].lower()
        inproc_linter = LINTERS_INPROC.get(ext) if ext in _FAIL_CLOSED_INPROC_EXTS else None
        if inproc_linter is not None:
            _ok, _lint_err = inproc_linter(content)
            if not _ok and _lint_err != "__SKIP__":
                return WriteResult(
                    error=(
                        f"Refusing to write '{path}': candidate content fails "
                        f"{ext} syntax validation ({_lint_err}). The file was "
                        "NOT created or modified. Fix the content and retry."
                    )
                )

        # Capture pre-write content.  Two consumers want it:
        #
        #   1. The lint-delta layer (for in-process linters like ast.parse
        #      and json.loads) needs the previous content to compute the
        #      set of NEW lint errors introduced by this write.
        #   2. The LSP layer needs pre/post content to build a line-shift
        #      map — pre-existing diagnostics below the edit point shift
        #      when lines are added/removed, and the shift map remaps
        #      baseline diagnostics into post-edit coordinates so the
        #      strict (range-aware) delta key matches.
        #
        # The set of extensions we capture pre_content for is therefore
        # the UNION of in-process lint coverage and LSP coverage.  For
        # extensions outside both sets (binaries, opaque formats),
        # skipping the read keeps the hot path fast.
        want_pre = ext in LINTERS_INPROC or self._lsp_handles_extension(ext)
        if want_pre:
            if pre_content is not None:
                # Caller already has file content (e.g. patch_replace read it
                # for fuzzy matching) — reuse directly, skip redundant cat.
                pass
            else:
                # Best-effort read; failure (file missing, permission) leaves
                # pre_content as None which makes both downstream consumers
                # degrade gracefully (lint reports all errors; LSP skips the
                # shift map).
                read_cmd = f"cat {self._escape_shell_arg(path)} 2>/dev/null"
                read_result = self._exec(read_cmd)
                if read_result.exit_code == 0 and read_result.stdout:
                    pre_content = read_result.stdout

        # ── Line-ending preservation (Roo Code pattern) ──────────────
        # If the file existed with CRLF endings and the agent's content
        # has bare LFs, convert to CRLF before writing.  Otherwise the
        # write silently normalizes a Windows-line-ending file (and patch
        # produces mixed endings when only a substituted region changes).
        # Detect from a small head sample to avoid reading the full file
        # for line-ending purposes alone.
        original_ending = self._detect_file_line_ending(path, pre_content)
        if original_ending == "\r\n":
            content = _normalize_line_endings(content, "\r\n")

        # ── BOM preservation ──────────────────────────────────────────
        # If the file on disk started with a UTF-8 BOM, keep it. read_file
        # strips the BOM so the agent never sees it, which means the
        # content it hands back to write_file / patch has no BOM either —
        # without restoring it here a round-trip would silently strip the
        # marker and change the file's byte signature (some Windows
        # toolchains key on it). Only prepend when the original had a BOM
        # and the new content doesn't already carry one (guards against
        # double-BOM if a caller passed raw bytes).
        if self._file_has_bom(path, pre_content) and not _has_bom(content):
            content = _UTF8_BOM + content

        # Snapshot LSP diagnostics for this file (best-effort) so the
        # post-write LSP layer can return only diagnostics introduced
        # by this specific edit.  Mirrors claude-code's
        # ``beforeFileEdited`` pattern but wired to the local LSP
        # rather than an external IDE.
        self._snapshot_lsp_baseline(path)

        # Write atomically.  ``mkdir -p`` is folded into _atomic_write
        # (one fewer subprocess vs. a separate mkdir call).
        # ``dirs_created`` has always meant "parent dirs ensured" —
        # ``mkdir -p`` exits 0 even when the dirs pre-exist, so the old
        # separate-mkdir code reported True in exactly the same cases.
        # A mkdir failure now surfaces as the atomic-write error return
        # below, before this field is ever emitted.
        parent = os.path.dirname(path)
        dirs_created = bool(parent)

        # Write atomically: stream into a temp file in the SAME directory,
        # then ``mv`` it over the target. The rename is atomic on POSIX
        # (and on every backend FS we run on), so a crash / power loss /
        # truncated pipe mid-write leaves the original file intact instead
        # of a half-written corrupt file. Same-directory is load-bearing —
        # ``mv`` across filesystems degrades to copy+unlink, which is NOT
        # atomic; keeping the temp beside the target guarantees a real
        # rename. Content still rides stdin so there's no ARG_MAX limit.
        #
        # The temp file is created with ``mktemp`` (collision-safe) when the
        # backend has it, falling back to a PID-stamped name otherwise. We
        # then chmod the temp to match the existing file's mode (if any) so
        # the atomic swap doesn't silently widen or narrow permissions, and
        # clean the temp up on any failure so we never leak a ``.hermes-tmp``
        # turd next to the user's file.
        # Encode once for byte count + sha256. surrogateescape is the exact
        # inverse of the decode that may have produced this content, so these
        # are the bytes the pipe transmits and the bytes on disk. The early
        # rejection above guarantees this cannot raise; the try/except is
        # defense for future callers that bypass it.
        try:
            content_bytes = content.encode("utf-8", "surrogateescape")
        except UnicodeEncodeError as exc:
            return WriteResult(
                error=(
                    f"Refusing to write '{path}': content contains a lone "
                    f"surrogate character ({exc}) that cannot be encoded as "
                    "UTF-8. The file was NOT created or modified."
                )
            )
        write_result = self._atomic_write(path, content)

        if write_result.exit_code != 0:
            return WriteResult(error=f"Failed to write file: {write_result.stdout}")

        # Bytes written — computed from the exact bytes we just wrote (len
        # matches wc -c) instead of spawning a ``wc -c`` subprocess. The
        # encode happened up front with surrogateescape — the inverse of the
        # decode that produces surrogate content — so content_bytes == what
        # rode stdin == what is on disk, and the sha256 below compares like
        # with like.
        bytes_written = len(content_bytes)

        # Post-write content verification (cheap, one shell call): compare
        # the on-disk sha256 to the intended content's hash. Production
        # mining shows models re-reading files right after writing them to
        # confirm persistence (154 verify-reads in a 400k-msg window) —
        # an explicit verified flag makes that turn unnecessary, and a
        # mismatch is surfaced as a hard error instead of silent corruption
        # (mirrors patch_replace's post-write verification).
        content_verified: Optional[bool] = None
        try:
            hash_cmd = f"sha256sum {self._escape_shell_arg(path)} 2>/dev/null"
            hash_result = self._exec(hash_cmd)
            if hash_result.exit_code == 0 and hash_result.stdout.strip():
                disk_sha = hash_result.stdout.strip().split()[0]
                expected_sha = hashlib.sha256(content_bytes).hexdigest()
                content_verified = disk_sha == expected_sha
                if not content_verified:
                    return WriteResult(
                        error=(
                            f"Post-write verification failed for {path}: on-disk "
                            "content hash differs from the intended write. The "
                            "write did not persist correctly — re-read the file "
                            "and retry."
                        )
                    )
        except Exception:
            content_verified = None

        # Post-write lint with delta refinement.
        lint_result = self._check_lint_delta(path, pre_content=pre_content, post_content=content)

        # Semantic diagnostics from the LSP layer — separate channel.
        # Only fired when the syntax tier reported clean (no point asking
        # an LSP for a file that won't even parse).  Pass pre/post
        # content so the LSP layer can build a line-shift map and
        # remap baseline diagnostics into post-edit coordinates.
        # Best-effort: ``""`` is returned for any failure path.
        lsp_diagnostics: Optional[str] = None
        if lint_result.success or lint_result.skipped:
            block = self._maybe_lsp_diagnostics(
                path, pre_content=pre_content, post_content=content
            )
            if block:
                lsp_diagnostics = block

        return WriteResult(
            bytes_written=bytes_written,
            dirs_created=dirs_created,
            verified=content_verified,
            lint=lint_result.to_dict() if lint_result else None,
            lsp_diagnostics=lsp_diagnostics,
        )
    
    # =========================================================================
    # PATCH Implementation (Replace Mode)
    # =========================================================================
    
    def patch_replace(self, path: str, old_string: str, new_string: str,
                      replace_all: bool = False) -> PatchResult:
        """
        Replace text in a file using fuzzy matching.

        Args:
            path: File path to modify
            old_string: Text to find (must be unique unless replace_all=True)
            new_string: Replacement text
            replace_all: If True, replace all occurrences

        Returns:
            PatchResult with diff and lint results
        """
        # Expand ~ and other shell paths
        path = self._expand_path(path)

        # Block writes to sensitive paths
        denied = get_write_denied_error(path)
        if denied:
            return PatchResult(error=denied)

        # Read current content
        read_cmd = f"cat {self._escape_shell_arg(path)} 2>/dev/null"
        read_result = self._exec(read_cmd)
        
        if read_result.exit_code != 0:
            return PatchResult(error=f"Failed to read file: {path}")
        
        content = read_result.stdout
        # Preserve raw content (including BOM) for write_file's pre_content
        # so write_file can detect/restore BOM correctly.
        raw_content = content
        # Strip a leading UTF-8 BOM before matching so the fuzzy matcher and
        # the diff operate on clean content (a phantom U+FEFF before line 1
        # defeats an exact first-line match). write_file restores the BOM on
        # the way back out by re-probing the on-disk file, so the round-trip
        # preserves the marker.
        content, _ = _strip_bom(content)

        # Import and use fuzzy matching
        from tools.fuzzy_match import fuzzy_find_and_replace
        
        new_content, match_count, _strategy, error = fuzzy_find_and_replace(
            content, old_string, new_string, replace_all
        )
        
        if error or match_count == 0:
            # Already-applied detection: the most common patch failure in
            # production is a re-send of an edit that has already landed
            # (identical old/new strings, or old_string gone while
            # new_string is present verbatim). Surface that as an explicit
            # success-shaped no-op so the model moves on instead of
            # burning turns on re-reads and re-patches.
            from tools.fuzzy_match import is_already_applied
            if is_already_applied(content, old_string, new_string):
                return PatchResult(
                    success=True,
                    no_change=True,
                    note=(
                        f"File already contains the target text — the edit "
                        f"appears to be already applied to {path}. No write "
                        "performed; do not re-send this patch."
                    ),
                )
            err_msg = error or f"Could not find match for old_string in {path}"
            try:
                from tools.fuzzy_match import format_no_match_hint
                err_msg += format_no_match_hint(err_msg, match_count, old_string, content)
            except Exception:
                pass
            return PatchResult(error=err_msg)

        # ── Line-ending preservation ──────────────────────────────────
        # Models nearly always send old_string/new_string with bare LF
        # in tool args (JSON-encoded), but the file may have CRLF on
        # disk.  After fuzzy_find_and_replace, ``new_content`` is a
        # mixed-ending string: the substituted region is LF, surrounding
        # text keeps the file's CRLF.  Normalize the whole thing to the
        # file's detected line ending so the on-disk file is consistent
        # and the unified diff below reflects the actual change.
        file_ending = _detect_line_ending(content)
        if file_ending:
            new_content = _normalize_line_endings(new_content, file_ending)

        # Write back — pass pre_content (original read, with BOM) to avoid
        # a redundant cat subprocess inside write_file.  Must be the raw
        # content (before _strip_bom) so write_file can detect/restore BOM.
        write_result = self.write_file(path, new_content,
                                       pre_content=raw_content)
        if write_result.error:
            return PatchResult(error=f"Failed to write changes: {write_result.error}")

        # Post-write verification — re-read the file and confirm the bytes we
        # intended to write actually landed. Catches silent persistence
        # failures (backend FS oddities, race with another task, truncated
        # pipe, etc.) that would otherwise return success-with-diff while the
        # file is unchanged on disk.
        verify_cmd = f"cat {self._escape_shell_arg(path)} 2>/dev/null"
        verify_result = self._exec(verify_cmd)
        if verify_result.exit_code != 0:
            return PatchResult(error=f"Post-write verification failed: could not re-read {path}")
        # Normalize line endings before comparing.  On Windows, Python's
        # default text-mode ``open()`` translates ``\n`` → ``\r\n`` on
        # write, so the file on disk legitimately holds CRLFs while our
        # ``new_content`` string has bare LFs.  Without this normalization
        # every patch on Windows returns a bogus "wrote 39, read 42"
        # false-negative even though the edit landed correctly.  POSIX
        # backends don't translate, so this is a no-op there.  We also
        # strip a leading BOM from the re-read: write_file restored the
        # marker on disk but ``new_content`` is the BOM-less string we
        # matched against, so the comparison must drop it to stay
        # apples-to-apples.
        _verify_bomless, _ = _strip_bom(verify_result.stdout)
        _verify_stdout_normalized = _verify_bomless.replace("\r\n", "\n").replace("\r", "\n")
        _new_content_normalized = new_content.replace("\r\n", "\n").replace("\r", "\n")
        if _verify_stdout_normalized != _new_content_normalized:
            return PatchResult(error=(
                f"Post-write verification failed for {path}: on-disk content "
                f"differs from intended write "
                f"(wrote {len(_new_content_normalized)} chars, read back "
                f"{len(_verify_stdout_normalized)} chars after normalizing line endings). "
                "The patch did not persist. Re-read the file and try again."
            ))

        # Generate diff
        diff = self._unified_diff(content, new_content, path)

        # Auto-lint with delta refinement: only surface errors introduced
        # by this patch, filtering out pre-existing lint failures so the
        # agent isn't distracted by problems that were already there.
        lint_result = self._check_lint_delta(path, pre_content=content, post_content=new_content)

        return PatchResult(
            success=True,
            diff=diff,
            files_modified=[path],
            lint=lint_result.to_dict() if lint_result else None,
            # Propagate the LSP diagnostics already captured by the
            # internal ``write_file`` call.  Its baseline was the
            # pre-patch content (taken at the start of write_file via
            # ``_snapshot_lsp_baseline``) so the delta is correct for
            # the patch as a whole.  Keep the field separate from the
            # syntax-check ``lint`` so the agent can read both signals.
            lsp_diagnostics=write_result.lsp_diagnostics,
        )
    
    def patch_v4a(self, patch_content: str) -> PatchResult:
        """
        Apply a V4A format patch.
        
        V4A format:
            *** Begin Patch
            *** Update File: path/to/file.py
            @@ context hint @@
             context line
            -removed line
            +added line
            *** End Patch
        
        Args:
            patch_content: V4A format patch string
        
        Returns:
            PatchResult with changes made
        """
        # Import patch parser
        from tools.patch_parser import parse_v4a_patch, apply_v4a_operations
        
        operations, parse_error = parse_v4a_patch(patch_content)
        if parse_error:
            return PatchResult(error=f"Failed to parse patch: {parse_error}")
        
        # Apply operations
        result = apply_v4a_operations(operations, self)
        return result
    


    
    # =========================================================================
    # SEARCH Implementation
    # =========================================================================
    
    def search(self, pattern: str, path: str = ".", target: str = "content",
               file_glob: Optional[str] = None, limit: int = 50, offset: int = 0,
               output_mode: str = "content", context: int = 0) -> SearchResult:
        """
        Search for content or files.
        
        Args:
            pattern: Regex (for content) or glob pattern (for files)
            path: Directory/file to search (default: cwd)
            target: "content" (grep) or "files" (glob)
            file_glob: File pattern filter for content search (e.g., "*.py")
            limit: Max results (default 50)
            offset: Skip first N results
            output_mode: "content", "files_only", or "count"
            context: Lines of context around matches
        
        Returns:
            SearchResult with matches or file list
        """
        offset, limit = normalize_search_pagination(offset, limit)

        # Expand ~ and other shell paths
        path = self._expand_path(path)
        
        # Validate that the path exists before searching
        if "not_found" in self._path_exists_probe(path):
            # Multi-path recovery: models frequently pass several paths in
            # one string ("dir1 dir2 dir3" or comma-separated). Instead of
            # failing the whole call, split, search every path that exists,
            # merge the results, and report the skipped parts.
            multi = self._try_multi_path_search(
                pattern, path, target, file_glob, limit, offset, output_mode, context
            )
            if multi is not None:
                return multi
            return self._path_not_found_result(path)

        result = self._dispatch_search(pattern, path, target, file_glob, limit, offset,
                                       output_mode, context)

        exclusions = self._macos_search_exclusions(path)
        if exclusions and not result.error:
            skipped = ", ".join(item.split("/")[-1] for item in exclusions)
            result.warning = (
                "Skipped macOS protected folders during broad search to avoid "
                f"an unattended privacy prompt: {skipped}. Search a protected "
                "folder directly when access is intentional."
            )
        return result

    


    
    
    


