#!/usr/bin/env python3
"""File operations (read, write, patch, search) over any terminal backend.

Every operation is expressed as a shell command run through the backend's
``execute()``, so one implementation serves local, docker, ssh, singularity,
modal, daytona and vercel_sandbox. Companion modules (names re-exported here):
``file_operations_common`` (result dataclasses, line-ending/BOM helpers),
``file_operations_lint`` (LintMixin: syntax lint + LSP), ``file_operations_search``
(SearchMixin: rg/grep/find backends).

    file_ops = ShellFileOperations(terminal_env)
    file_ops.read_file("/path/to/file.py")
    file_ops.write_file("/path/to/new.py", "print('hello')")
    file_ops.search("TODO", path=".", file_glob="*.py")
"""

import base64
import binascii
import os
import re
import sys  # noqa: F401  (tests monkeypatch tools.file_operations.sys.platform)
import difflib
import hashlib
import json
import unicodedata
from abc import ABC, abstractmethod

from typing import Optional, Dict
from pathlib import Path
from tools.binary_extensions import BINARY_EXTENSIONS

from agent.file_safety import (
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
    LINTERS_INPROC,
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


def _is_write_denied(path: str) -> bool:
    """Return True if path is on the write deny list."""
    return _shared_is_write_denied(path)


# =============================================================================
# Binary-content identification
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

# Echoed by the size probe when the path exists but is not a regular file.
# `wc -c` prints only digits, so this can never collide with a real size.
NOT_REGULAR_SENTINEL = "__hermes_not_regular__"


class ShellFileOperations(LintMixin, SearchMixin, FileOperations):
    """File operations over any terminal backend exposing ``execute(command, cwd)``
    returning ``{"output": str, "returncode": int}``.

    cwd rule: every ``_exec`` prefers the LIVE ``env.cwd`` so ``cd`` run via the
    terminal tool is picked up immediately; the init-time ``self.cwd`` is only
    a fallback for envs that don't track cwd. (Using the init-time cwd for
    every call once made patches "succeed" with a plausible diff while landing
    in the wrong directory.)
    """

    def __init__(self, terminal_env, cwd: str = None):
        self.env = terminal_env
        # Never fall back to os.getcwd(): that is the HOST path, which doesn't
        # exist inside container/cloud backends. "/" is the universal default.
        self.cwd = cwd or getattr(terminal_env, 'cwd', None) or \
                   getattr(getattr(terminal_env, 'config', None), 'cwd', None) or "/"
        self._command_cache: Dict[str, bool] = {}

    def _exec(self, command: str, cwd: str = None, timeout: int = None,
              stdin_data: str = None) -> ExecuteResult:
        """Run ``command`` on the backend. cwd: explicit arg → live ``env.cwd`` →
        init-time ``self.cwd``. ``stdin_data`` is piped (bypasses ARG_MAX)."""
        kwargs = {}
        if timeout:
            kwargs['timeout'] = timeout
        if stdin_data is not None:
            kwargs['stdin_data'] = stdin_data

        effective_cwd = cwd or getattr(self.env, 'cwd', None) or self.cwd
        result = self.env.execute(command, cwd=effective_cwd, **kwargs)
        exit_code = result.get("returncode", 0)
        # A stdin write failure with a clean child exit is still a failure: the
        # child never received the input. Defense-in-depth for stdin callers
        # other than write_file (which rejects unencodable content up front).
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
        """First ``length`` raw bytes of a file, base64-wrapped so they survive the
        terminal transport (which decodes stdout with ``errors="replace"`` and
        manufactures U+FFFD for every undecodable byte, including a multibyte
        char cut in half by ``head -c``). Returns None when the shell produced
        no clean base64 (no ``base64`` binary); callers fall back to the text heuristic.
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
        """Byte-layer binary detection.

        Text iff the sample is valid UTF-8, allowing one incomplete multibyte
        sequence at the very end (an artifact of the byte-boundary cut, not of
        the file). NUL bytes or mid-stream invalid UTF-8 (latin-1, true binaries)
        stay read-only so a read→edit→write round-trip never rewrites
        undecodable bytes as U+FFFD. A file that legitimately CONTAINS U+FFFD
        is valid UTF-8 and reads as text (the text-layer check couldn't tell a
        stored replacement char from a transport-manufactured one).
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
        """Legacy text-layer binary check: extension, else >30% non-printable chars."""
        ext = os.path.splitext(path)[1].lower()
        if ext in BINARY_EXTENSIONS:
            return True
        if content_sample:
            # The terminal decodes stdout with errors="replace", so undecodable
            # bytes arrive as U+FFFD — "printable", so the ratio below misses
            # them. Treat the sample as binary (read-only) so a read→edit→write
            # round-trip can't overwrite the original bytes with mojibake.
            if "\ufffd" in content_sample[:1000]:
                return True
            non_printable = sum(1 for c in content_sample[:1000]
                               if ord(c) < 32 and c not in '\n\r\t')
            return non_printable / min(len(content_sample), 1000) > 0.30
        return False

    def _is_image(self, path: str) -> bool:
        return os.path.splitext(path)[1].lower() in IMAGE_EXTENSIONS

    def _add_line_numbers(self, content: str, start_line: int = 1) -> str:
        """Prefix each line with a compact ``<n>|`` gutter, clamping long lines.

        Compact (not fixed-width padded): padding cost ~16% more tokens per
        line for no accuracy gain in A/B, while dropping numbers entirely
        regressed line-referencing (models hand-count off-by-one).
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
        """Expand ``~`` / ``~user`` via the backend's shell (its HOME, not the
        host's). Must run BEFORE shell escaping — ~ doesn't expand in quotes."""
        if not path:
            return path

        if path.startswith('~'):
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
        """Write ``content`` to ``path`` atomically: stdin → temp file in the SAME
        directory → ``mv -f`` over the target (same-FS rename; a cross-device
        ``mv`` degrades to copy+unlink and is NOT atomic). ``mkdir -p`` is folded
        in (one subprocess). ``exit_code == 0`` means the swap happened; non-zero
        means nothing was renamed and the original (if any) is intact.

        Script notes:
        - Symlink targets are resolved first so we edit the file the link points
          at (replacing the link with a plain file orphans the target); the
          temp dir is recomputed from the RESOLVED target. Best-effort.
        - ``mktemp -p`` with a hidden, marked template (an orphan is only possible
          on a hard crash between cat and mv); PID-stamped fallback without mktemp.
        - Existing target: copy its mode via ``stat`` (GNU ``-c%a`` / BSD
          ``-f%Lp``) + explicit ``chmod`` — ``chmod --reference`` is GNU-only.
          Best-effort; a failure leaves mktemp's 0600.
        - New target: ``chmod "=rw"`` AFTER cat gives umask-default perms (0644
          under 022) instead of 0600. Deliberately not ``$(umask)`` arithmetic:
          zsh parses leading-zero constants as decimal and computes garbage;
          quoted so zsh doesn't =word-expand it.
        - ``trap ... EXIT`` removes the temp on every failure path; cleared
          after a successful mv.
        """
        q_path = self._escape_shell_arg(path)
        parent = os.path.dirname(path) or "."
        q_parent = self._escape_shell_arg(parent)
        tmpl = self._escape_shell_arg(".hermes-tmp.XXXXXX")

        script = (
            "set -e; "
            f"d={q_parent}; t={q_path}; "
            'if [ -L "$t" ]; then '
            'rt="$(readlink -f "$t" 2>/dev/null || realpath "$t" 2>/dev/null || true)"; '
            '[ -n "$rt" ] && { t="$rt"; d="$(dirname "$t")"; }; '
            "fi; "
            'mkdir -p "$d"; '
            'tmp="$(mktemp -p "$d" ' + tmpl + ' 2>/dev/null '
            '|| mktemp "$d/.hermes-tmp.$$.XXXXXX" 2>/dev/null '
            '|| { tmp="$d/.hermes-tmp.$$"; : > "$tmp" && echo "$tmp"; })"; '
            '[ -n "$tmp" ] || { echo "atomic write: could not create temp file" >&2; exit 1; }; '
            "trap 'rm -f \\\"$tmp\\\"' EXIT; "
            'if [ -e "$t" ]; then '
            'm="$(stat -c%a "$t" 2>/dev/null || stat -f%Lp "$t" 2>/dev/null || true)"; '
            '[ -n "$m" ] && chmod "$m" "$tmp" 2>/dev/null || true; '
            "fi; "
            'cat > "$tmp"; '
            'if [ ! -e "$t" ]; then chmod "=rw" "$tmp" 2>/dev/null || true; fi; '
            'mv -f "$tmp" "$t"; '
            "trap - EXIT"
        )
        return self._exec(script, stdin_data=content)

    def _detect_file_line_ending(self, path: str, pre_content: Optional[str] = None) -> Optional[str]:
        """Dominant line ending on disk (``"\\r\\n"``/``"\\n"``), or None when
        undeterminable (new/empty/single-line file). Uses ``pre_content`` when
        given, else a 4KB ``head`` sample (exits 0 with no output for a new file)."""
        if pre_content:
            return _detect_line_ending(pre_content)
        head_result = self._exec(f"head -c 4096 {self._escape_shell_arg(path)} 2>/dev/null")
        if head_result.exit_code != 0 or not head_result.stdout:
            return None
        return _detect_line_ending(head_result.stdout)

    def _file_has_bom(self, path: str, pre_content: Optional[str] = None) -> bool:
        """Whether the file on disk starts with a UTF-8 BOM. ALWAYS probes disk:
        ``pre_content`` usually comes from ``read_file_raw``, which strips BOMs,
        so trusting it would silently drop the marker on rewrite. Missing/empty
        file → False (new writes get no BOM unless the content carries one)."""
        head_result = self._exec(f"head -c 3 {self._escape_shell_arg(path)} 2>/dev/null")
        if head_result.exit_code != 0 or not head_result.stdout:
            return False
        return _has_bom(head_result.stdout)

    def _unified_diff(self, old_content: str, new_content: str, filename: str) -> str:
        return ''.join(difflib.unified_diff(
            old_content.splitlines(keepends=True), new_content.splitlines(keepends=True),
            fromfile=f"a/{filename}", tofile=f"b/{filename}",
        ))

    # =========================================================================
    # READ Implementation
    # =========================================================================
    
    def _size_probe_cmd(self, path: str) -> str:
        """Byte size of a REGULAR file without opening one that never ends.

        ``wc -c <`` on a writer-less FIFO, socket or /dev/zero blocks forever
        (read helpers pass no timeout). The name-based device blocklist in
        file_tools can't cover a FIFO — it's a file TYPE at any path. ``[ -f ]``
        is a stat (symlinks followed), so it answers without touching content;
        existing non-regular paths echo the sentinel, absent paths exit non-zero.
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

    def _probe_regular_file(self, path: str) -> tuple[int, str]:
        """Run the size probe. Returns ``(file_size, status)`` with status one of
        ``"ok"``, ``"missing"`` (path absent), ``"not_regular"`` (FIFO/socket/
        device/directory) or ``"bad_size"`` (unparseable ``wc`` output; size 0)."""
        stat_result = self._exec(self._size_probe_cmd(path))
        if stat_result.exit_code != 0:
            return 0, "missing"
        stat_output = _strip_terminal_fence_leaks(stat_result.stdout).strip()
        if stat_output == NOT_REGULAR_SENTINEL:
            return 0, "not_regular"
        try:
            return int(stat_output), "ok"
        except ValueError:
            return 0, "bad_size"

    def _detect_binary(self, path: str) -> tuple[bool, Optional[bytes]]:
        """``(is_binary, sample_bytes)`` — byte-layer detection when the transport
        allows (base64 sample), else the legacy text heuristic (sample is None)."""
        sample_bytes = self._sample_file_bytes(path)
        if sample_bytes is not None:
            ext_binary = os.path.splitext(path)[1].lower() in BINARY_EXTENSIONS
            return ext_binary or self._is_likely_binary_bytes(sample_bytes), sample_bytes
        sample_result = self._exec(f"head -c 1000 {self._escape_shell_arg(path)} 2>/dev/null")
        sample_output = _strip_terminal_fence_leaks(sample_result.stdout)
        return self._is_likely_binary(path, sample_output), None

    # UTF-16 rescue: trust a BOM first, then zero-byte PARITY (not density, so
    # mixed Latin/CJK still detects): zeros at odd indices → UTF-16 LE, at even
    # → BE; both parities or a single zero → real binary. Legacy 8-bit
    # encodings (GBK, Big5) are never guessed — a wrong silent guess is worse
    # than a clear refusal.
    _UTF16_MAX_BYTES = 10 * 1024 * 1024
    _UTF16_SAMPLE_BYTES = 512

    def _try_read_utf16(self, path: str, offset: int, limit: int,
                        file_size: int) -> "Optional[ReadResult]":
        """Read ``path`` as UTF-16 transcoded to UTF-8, or None (caller falls back
        to the binary-file error). Skips known-binary extensions and files over
        10 MiB. ``path`` must already be expanded."""
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
        """Read a file with pagination, binary detection, and line numbers.

        ``offset`` is 1-indexed; ``limit`` is clamped by ``normalize_read_pagination``.
        """
        path = self._expand_path(path)  # before shell escaping: ~ doesn't expand in quotes
        offset, limit = normalize_read_pagination(offset, limit)

        file_size, status = self._probe_regular_file(path)
        if status == "missing":
            # Before failing, try unicode-equivalent spellings — NFC/NFD, narrow
            # no-break space, curly quotes render identically in a terminal, so
            # the model retyping a visually-correct path can never discover the
            # byte mismatch on its own (retrying is the tool's job, not the model's).
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
            return self._suggest_similar_files(path)
        if status == "not_regular":
            return self._not_regular_error(path)

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

        is_binary, sample_bytes = self._detect_binary(path)
        if is_binary:
            # UTF-16 rescue: the terminal env decodes stdout as UTF-8 with
            # errors="replace", so a UTF-16 text file (Windows Notepad .txt,
            # PowerShell `>` redirects) arrives mangled with U+FFFD and trips
            # the binary guard. Probe the raw bytes via the backend's Python
            # and transcode when a BOM or zero-byte parity identifies UTF-16.
            utf16_result = self._try_read_utf16(path, offset, limit, file_size)
            if utf16_result is not None:
                return utf16_result
            return ReadResult(
                is_binary=True,
                file_size=file_size,
                error=describe_binary_file(sample_bytes, file_size),
            )

        # Read with pagination using sed, clamping each line to a byte budget IN
        # THE SHELL so a pathological single-line file (one 400MB minified line)
        # never crosses the exec transport; the Python clamp in
        # _add_line_numbers still runs afterwards.
        #
        # Why 4*max_line_length + 1 bytes: ``cut -c`` is byte-based on GNU
        # coreutils, and a byte clamp can split a multibyte UTF-8 codepoint (the
        # transport decodes with errors="replace", so that becomes U+FFFD). A
        # clamp of max_line_length+1 BYTES yields far fewer CHARS than
        # max_line_length for multibyte text, so the Python clamp would never
        # fire and truncation would be silent (no "... [truncated]" suffix).
        # UTF-8 codepoints are at most 4 bytes, so keeping 4*max+1 bytes
        # guarantees every over-long line still decodes to more than
        # max_line_length chars and trips the Python clamp, which also removes
        # any boundary-split U+FFFD (it lands beyond char max_line_length).
        # ``cut -b`` documents the byte semantics explicitly.
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
        # Strip a leading UTF-8 BOM so the model never sees a phantom U+FEFF.
        # Only the first chunk can carry it (the marker lives at byte 0).
        if offset == 1:
            read_output, _ = _strip_bom(read_output)

        wc_result = self._exec(f"wc -l < {self._escape_shell_arg(path)}")
        try:
            total_lines = int(_strip_terminal_fence_leaks(wc_result.stdout).strip())
        except ValueError:
            total_lines = 0

        truncated = total_lines > end_line
        hint = None
        if truncated:
            hint = f"Use offset={end_line + 1} to continue reading (showing {offset}-{end_line} of {total_lines} lines)"

        # ``cut`` (unlike sed -n p) always newline-terminates its output, so a
        # file whose final line has no trailing newline would grow a phantom
        # empty last line. Only possible when this page reaches the file's
        # final line; probe the last byte and strip the artifact.
        if not truncated and read_output.endswith('\n'):
            tail_result = self._exec(f"tail -c 1 {self._escape_shell_arg(path)} | wc -l")
            tail_output = _strip_terminal_fence_leaks(tail_result.stdout)
            if tail_result.exit_code == 0 and tail_output.strip() == "0":
                read_output = read_output[:-1]

        # Ambiguous-silence guards: an empty content string is indistinguishable,
        # from inside the model, from a broken tool — it re-reads, widens the
        # window, tries another path. Name the dead end and its recovery instead.
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
        """On-disk spelling of a file whose name is unicode-equivalent to ``path``.

        macOS puts a NARROW NO-BREAK SPACE (U+202F) in screenshot names, stores
        NFD, and Finder turns ' into \u2019 — all invisible when rendered.
        Returns the entry only when EXACTLY one matches under normalization.
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
        # Several candidates = homoglyph collision; guessing would read the wrong file.
        if len(candidates) == 1:
            return os.path.join(dir_path, candidates[0]) if dir_path != "." or "/" in path else candidates[0]
        return None

    def _suggest_similar_files(self, path: str) -> ReadResult:
        """"File not found" result listing up to 5 similar names from the same directory."""
        dir_path = os.path.dirname(path) or "."
        filename = os.path.basename(path)
        basename_no_ext = os.path.splitext(filename)[0].lower()
        ext = os.path.splitext(filename)[1].lower()
        lower_name = filename.lower()

        ls_result = self._exec(f"ls -1 {self._escape_shell_arg(dir_path)} 2>/dev/null | head -50")

        scored: list = []  # (score, filepath) — higher is better
        if ls_result.exit_code == 0 and ls_result.stdout.strip():
            for f in ls_result.stdout.strip().split('\n'):
                if not f:
                    continue
                lf = f.lower()
                score = 0
                if lf == lower_name:
                    score = 100
                elif os.path.splitext(f)[0].lower() == basename_no_ext:  # config.yml vs config.yaml
                    score = 90
                elif lf.startswith(lower_name) or lower_name.startswith(lf):
                    score = 70
                elif lower_name in lf:
                    score = 60
                elif lf in lower_name and len(lf) > 2:
                    score = 40
                elif ext and os.path.splitext(f)[1].lower() == ext:
                    common = set(lower_name) & set(lf)
                    if len(common) >= max(len(lower_name), len(lf)) * 0.4:
                        score = 30
                # Near-miss spelling (AGENT.md -> AGENTS.md): a high sequence ratio
                # catches 1-2 edit typos the substring checks miss.
                if score == 0 and difflib.SequenceMatcher(None, lower_name, lf).ratio() >= 0.8:
                    score = 50
                if score > 0:
                    scored.append((score, os.path.join(dir_path, f)))

        scored.sort(key=lambda x: -x[0])
        return ReadResult(
            error=f"File not found: {path}",
            similar_files=[fp for _, fp in scored[:5]],
        )
    
    def read_file_raw(self, path: str) -> ReadResult:
        """Whole file as a plain string (no pagination/line numbers/clamping)."""
        path = self._expand_path(path)
        file_size, status = self._probe_regular_file(path)
        if status == "missing":
            return self._suggest_similar_files(path)
        if status == "not_regular":
            return self._not_regular_error(path)
        if self._is_image(path):
            return ReadResult(is_image=True, is_binary=True, file_size=file_size)
        is_binary, sample_bytes = self._detect_binary(path)
        if is_binary:
            return ReadResult(
                is_binary=True, file_size=file_size,
                error=describe_binary_file(sample_bytes, file_size),
            )
        cat_result = self._exec(f"cat {self._escape_shell_arg(path)}")
        if cat_result.exit_code != 0:
            return ReadResult(error=f"Failed to read file: {cat_result.stdout}")
        # Strip a leading BOM so patch's fuzzy matcher sees clean content (a
        # phantom U+FEFF defeats an exact first-line match); write_file
        # re-probes disk and restores it, so the round-trip preserves it.
        raw_content, _ = _strip_bom(_strip_terminal_fence_leaks(cat_result.stdout))
        return ReadResult(
            content=raw_content,
            file_size=file_size,
        )

    def read_file_bytes(self, path: str, max_bytes: Optional[int] = None) -> ReadResult:
        """Read binary-safe bytes from any shell-backed environment."""
        path = self._expand_path(path)
        file_size, status = self._probe_regular_file(path)
        if status == "missing":
            return ReadResult(error=f"File not found: {path}")
        if status == "not_regular":
            return self._not_regular_error(path)
        if status == "bad_size":
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
        """Delete a single file (directories rejected; see ``delete_path``)."""
        return self._python_delete(path, recursive=False)

    def delete_path(self, path: str, recursive: bool = False) -> WriteResult:
        """Delete a file or (``recursive=True``) a directory tree."""
        return self._python_delete(path, recursive=recursive)

    def _python_delete(self, path: str, recursive: bool) -> WriteResult:
        """Delete via the backend's ``python -c`` so one code path works on
        local/docker/ssh AND Windows shells (no ``rm`` / ``Remove-Item``)."""
        path = self._expand_path(path)
        denied = get_write_denied_error(path, verb="Delete")
        if denied:
            return WriteResult(error=denied)

        # No ``rm``: it doesn't exist on Windows cmd.exe/PowerShell backends.
        # Path is baked in via ``repr()`` so quoting is correct on every shell.
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
            # Not ``unlink(missing_ok=True)``: a 3.7 remote interpreter lacks it;
            # the FileNotFoundError handler covers the same case.
            "        p.unlink()\n"
            "except FileNotFoundError:\n"
            "    pass\n"
            "except Exception as exc:\n"
            "    print(str(exc), file=sys.stderr); sys.exit(1)\n"
        )

        result = self._exec(f"python3 -c {self._escape_shell_arg(snippet)}")

        # Windows / older systems: no ``python3`` symlink, only ``python``.
        if result.exit_code != 0 and "python3" in (result.stdout or ""):
            result = self._exec(f"python -c {self._escape_shell_arg(snippet)}")

        if result.exit_code != 0:
            return WriteResult(error=f"Failed to delete {path}: {(result.stdout or '').strip() or 'unknown error'}")

        return WriteResult()

    def move_file(self, src: str, dst: str) -> WriteResult:
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

    _LONE_SURROGATE_RE = re.compile(r"[\ud800-\udc7f\udd00-\udfff]")

    def _reject_unencodable(self, path: str, content: str) -> Optional[WriteResult]:
        """Refuse content with a lone surrogate BEFORE any subprocess.

        surrogateescape-decoded content (U+DC80-U+DCFF) round-trips through the
        pipe; surrogates outside that range cannot be encoded at all, and
        letting them reach the pipe spawns a child that hangs or truncates the
        target via empty-stdin ``cat``. A regex scan needs no encode.
        """
        m = self._LONE_SURROGATE_RE.search(content)
        if m:
            return WriteResult(
                error=(
                    f"Refusing to write '{path}': content contains a lone "
                    f"surrogate character ({m.group(0)!r}) that cannot be "
                    "encoded as UTF-8. The file was NOT created or modified."
                )
            )
        return None

    @staticmethod
    def _fail_closed_syntax_error(path: str, ext: str, content: str) -> Optional[WriteResult]:
        """Fail-closed pre-write gate for ``_FAIL_CLOSED_INPROC_EXTS`` (JSON/YAML/TOML).

        A structured-format write that doesn't parse (mashed quotes, truncated
        generation) is a corrupt write, not a style nit: refuse it before any
        bytes touch disk instead of reporting damage afterwards. ``.py`` keeps
        its non-blocking lint-delta report (see ``_FAIL_CLOSED_INPROC_EXTS``);
        extensions without an in-process linter are untouched.

        Checked against the RAW content, before the BOM/CRLF shims: linting
        post-shim would false-positive a JSONDecodeError on a legitimately
        BOM-marked file purely because write_file re-adds the marker.
        """
        linter = LINTERS_INPROC.get(ext) if ext in _FAIL_CLOSED_INPROC_EXTS else None
        if linter is None:
            return None
        ok, err = linter(content)
        if ok or err == "__SKIP__":
            return None
        return WriteResult(
            error=(
                f"Refusing to write '{path}': candidate content fails "
                f"{ext} syntax validation ({err}). The file was "
                "NOT created or modified. Fix the content and retry."
            )
        )

    def _capture_pre_content(self, path: str, ext: str,
                             pre_content: Optional[str]) -> Optional[str]:
        """Pre-write content for the lint-delta and LSP line-shift consumers.

        Captured only for extensions in the UNION of in-process lint coverage
        and LSP coverage — for anything else (binaries, opaque formats) skipping
        the read keeps the hot path fast. A caller-supplied ``pre_content`` is
        reused as-is; otherwise a best-effort ``cat`` whose failure (missing
        file, permissions) leaves None so both consumers degrade gracefully
        (lint reports all errors; LSP skips the shift map).
        """
        if pre_content is not None:
            return pre_content
        if ext in LINTERS_INPROC or self._lsp_handles_extension(ext):
            read_result = self._exec(f"cat {self._escape_shell_arg(path)} 2>/dev/null")
            if read_result.exit_code == 0 and read_result.stdout:
                return read_result.stdout
        return None

    def _match_on_disk_conventions(self, path: str, content: str,
                                   pre_content: Optional[str]) -> str:
        """Re-apply the on-disk file's CRLF endings and UTF-8 BOM to ``content``.

        read_file strips the BOM and models send bare-LF text, so a round-trip
        would otherwise silently normalize a CRLF file (patch would leave mixed
        endings) and drop a byte signature some Windows toolchains key on. The
        BOM is only prepended when the original had one and ``content`` doesn't
        already (guards double-BOM from callers passing raw bytes).
        """
        if self._detect_file_line_ending(path, pre_content) == "\r\n":
            content = _normalize_line_endings(content, "\r\n")
        if self._file_has_bom(path, pre_content) and not _has_bom(content):
            content = _UTF8_BOM + content
        return content

    def _verify_written_hash(self, path: str, content_bytes: bytes) -> tuple[Optional[bool], Optional[WriteResult]]:
        """Compare the on-disk sha256 to the intended bytes (one shell call).

        Production mining shows models re-reading files right after writing to
        confirm persistence; an explicit ``verified`` flag makes that turn
        unnecessary, and a mismatch is a hard error instead of silent
        corruption. Returns ``(verified, error_result)``; ``verified`` is None
        when the hash could not be taken.
        """
        try:
            hash_result = self._exec(f"sha256sum {self._escape_shell_arg(path)} 2>/dev/null")
            if hash_result.exit_code == 0 and hash_result.stdout.strip():
                disk_sha = hash_result.stdout.strip().split()[0]
                if disk_sha != hashlib.sha256(content_bytes).hexdigest():
                    return False, WriteResult(
                        error=(
                            f"Post-write verification failed for {path}: on-disk "
                            "content hash differs from the intended write. The "
                            "write did not persist correctly — re-read the file "
                            "and retry."
                        )
                    )
                return True, None
        except Exception:
            pass
        return None, None

    def write_file(self, path: str, content: str,
                   pre_content: Optional[str] = None) -> WriteResult:
        """Write content to a file atomically, creating parent directories as needed.

        Order: deny list → lone-surrogate refusal → fail-closed syntax gate on
        the CANDIDATE content (JSON/YAML/TOML; nothing touches disk on failure)
        → pre-content capture → CRLF/BOM preservation → LSP baseline snapshot →
        atomic write (content rides stdin, so no ARG_MAX limit and the content
        never appears in the command string) → sha256 verification → lint delta
        (only errors THIS write introduced) → LSP diagnostics when syntax is clean.

        ``pre_content``: pre-edit content the caller already has (patch_replace
        read it for fuzzy matching); saves a ``cat``. BOM detection always
        probes disk regardless — ``read_file_raw`` strips BOMs, so trusting
        ``pre_content`` would silently drop the marker on rewrite.
        """
        path = self._expand_path(path)

        denied = get_write_denied_error(path)
        if denied:
            return WriteResult(error=denied)
        refused = self._reject_unencodable(path, content)
        if refused is not None:
            return refused

        ext = os.path.splitext(path)[1].lower()
        refused = self._fail_closed_syntax_error(path, ext, content)
        if refused is not None:
            return refused

        pre_content = self._capture_pre_content(path, ext, pre_content)
        content = self._match_on_disk_conventions(path, content, pre_content)

        # Snapshot LSP diagnostics (best-effort) so the post-write LSP layer
        # returns only diagnostics introduced by this edit.
        self._snapshot_lsp_baseline(path)

        # ``dirs_created`` has always meant "parent dirs ensured": mkdir -p is
        # folded into _atomic_write and exits 0 even when they pre-exist; a
        # mkdir failure surfaces as the atomic-write error below.
        dirs_created = bool(os.path.dirname(path))

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

        content_verified, verify_error = self._verify_written_hash(path, content_bytes)
        if verify_error is not None:
            return verify_error

        lint_result = self._check_lint_delta(path, pre_content=pre_content, post_content=content)

        # Semantic (LSP) diagnostics are a separate channel, fired only when the
        # syntax tier is clean (no point asking an LSP about a file that won't
        # parse). Best-effort: "" on any failure path.
        lsp_diagnostics: Optional[str] = None
        if lint_result.success or lint_result.skipped:
            lsp_diagnostics = self._maybe_lsp_diagnostics(
                path, pre_content=pre_content, post_content=content
            ) or None

        return WriteResult(
            bytes_written=len(content_bytes),
            dirs_created=dirs_created,
            verified=content_verified,
            lint=lint_result.to_dict() if lint_result else None,
            lsp_diagnostics=lsp_diagnostics,
        )

    # =========================================================================
    # PATCH Implementation (Replace Mode)
    # =========================================================================

    def _no_match_result(self, path: str, content: str, old_string: str,
                         new_string: str, match_count: int, error: Optional[str]) -> PatchResult:
        """PatchResult for a failed fuzzy match.

        Already-applied detection first: the most common production patch
        failure is a re-send of an edit that already landed (identical
        old/new, or old_string gone while new_string is present verbatim).
        That becomes a success-shaped no-op so the model moves on instead of
        burning turns on re-reads. Otherwise attach a best-effort
        "Did you mean?" snippet to the error.
        """
        from tools.fuzzy_match import format_no_match_hint, is_already_applied

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
            err_msg += format_no_match_hint(err_msg, match_count, old_string, content)
        except Exception:
            pass
        return PatchResult(error=err_msg)

    def _verify_patch_persisted(self, path: str, new_content: str) -> Optional[PatchResult]:
        """Re-read ``path`` and confirm the intended bytes landed; error result or None.

        Catches silent persistence failures (backend FS oddities, a race with
        another task, truncated pipe) that would otherwise return
        success-with-diff while the file is unchanged. Line endings are
        normalized before comparing: on Windows text-mode ``open()`` writes
        ``\\n`` as ``\\r\\n``, so the disk legitimately holds CRLF while
        ``new_content`` has LF (POSIX is a no-op). The re-read's leading BOM is
        stripped too — write_file restored it on disk but ``new_content`` is
        the BOM-less string we matched against.
        """
        verify_result = self._exec(f"cat {self._escape_shell_arg(path)} 2>/dev/null")
        if verify_result.exit_code != 0:
            return PatchResult(error=f"Post-write verification failed: could not re-read {path}")
        bomless, _ = _strip_bom(verify_result.stdout)
        on_disk = bomless.replace("\r\n", "\n").replace("\r", "\n")
        intended = new_content.replace("\r\n", "\n").replace("\r", "\n")
        if on_disk != intended:
            return PatchResult(error=(
                f"Post-write verification failed for {path}: on-disk content "
                f"differs from intended write "
                f"(wrote {len(intended)} chars, read back "
                f"{len(on_disk)} chars after normalizing line endings). "
                "The patch did not persist. Re-read the file and try again."
            ))
        return None

    def patch_replace(self, path: str, old_string: str, new_string: str,
                      replace_all: bool = False) -> PatchResult:
        """Replace text in a file using fuzzy matching (``old_string`` must be
        unique unless ``replace_all``). Returns a PatchResult with diff + lint."""
        path = self._expand_path(path)

        denied = get_write_denied_error(path)
        if denied:
            return PatchResult(error=denied)

        read_result = self._exec(f"cat {self._escape_shell_arg(path)} 2>/dev/null")
        if read_result.exit_code != 0:
            return PatchResult(error=f"Failed to read file: {path}")

        # Keep the raw read (with BOM) as write_file's pre_content so it can
        # detect/restore the BOM; match and diff on BOM-stripped content (a
        # phantom U+FEFF before line 1 defeats an exact first-line match).
        raw_content = read_result.stdout
        content, _ = _strip_bom(raw_content)

        from tools.fuzzy_match import fuzzy_find_and_replace

        new_content, match_count, _strategy, error = fuzzy_find_and_replace(
            content, old_string, new_string, replace_all
        )
        if error or match_count == 0:
            return self._no_match_result(path, content, old_string, new_string, match_count, error)

        # Models send bare-LF old/new strings, so after replacement the
        # substituted region is LF while the rest keeps the file's CRLF.
        # Normalize to the file's detected ending so the file stays consistent
        # and the diff reflects the real change.
        file_ending = _detect_line_ending(content)
        if file_ending:
            new_content = _normalize_line_endings(new_content, file_ending)

        # pre_content must be the RAW read (before _strip_bom) for BOM detection;
        # passing it also saves write_file a redundant cat.
        write_result = self.write_file(path, new_content, pre_content=raw_content)
        if write_result.error:
            return PatchResult(error=f"Failed to write changes: {write_result.error}")

        verify_error = self._verify_patch_persisted(path, new_content)
        if verify_error is not None:
            return verify_error

        # Lint delta: only surface errors introduced by this patch.
        lint_result = self._check_lint_delta(path, pre_content=content, post_content=new_content)

        return PatchResult(
            success=True,
            diff=self._unified_diff(content, new_content, path),
            files_modified=[path],
            lint=lint_result.to_dict() if lint_result else None,
            # LSP diagnostics already captured by the internal write_file call;
            # its baseline was the pre-patch content, so the delta is correct for
            # the whole patch. Kept separate from ``lint`` so both signals are readable.
            lsp_diagnostics=write_result.lsp_diagnostics,
        )

    def patch_v4a(self, patch_content: str) -> PatchResult:
        """Apply a V4A format patch (``*** Begin Patch`` / ``*** Update File:`` /
        ``@@ hint @@`` hunks / ``*** End Patch``)."""
        from tools.patch_parser import parse_v4a_patch, apply_v4a_operations

        operations, parse_error = parse_v4a_patch(patch_content)
        if parse_error:
            return PatchResult(error=f"Failed to parse patch: {parse_error}")
        return apply_v4a_operations(operations, self)

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

    


    
    
    


