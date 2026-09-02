"""Content/file search tier for ``tools.file_operations``.

Extracted from ``ShellFileOperations`` as ``SearchMixin``; the class inherits it
so every ``self._search_*`` call resolves unchanged via the MRO. Module-level
helpers are re-imported into ``tools.file_operations`` for back-compat.
"""

import os
import re
import sys
from pathlib import Path
from typing import List, Optional

from tools.file_operations_common import ExecuteResult, SearchMatch, SearchResult

_MACOS_TCC_PROTECTED_HOME_DIRS = (
    "Desktop", "Documents", "Downloads", "Library", "Movies", "Music", "Pictures",
)


def _macos_protected_search_exclusions(
    path: str,
    *,
    cwd: Optional[str] = None,
    home: Optional[str] = None,
    platform: Optional[str] = None,
) -> List[str]:
    """Protected home dirs (relative to ``path``) below a broad macOS search root.

    Only an ANCESTOR search (``$HOME``, ``/Users``) gets exclusions, so recursive
    tools never trigger unattended TCC prompts; a search rooted inside a
    protected dir stays allowed.
    """
    if (platform or sys.platform) != "darwin":
        return []

    home_path = Path(home or Path.home()).expanduser()
    root = Path(path).expanduser()
    if not root.is_absolute():
        root = Path(cwd or os.getcwd()) / root
    root = Path(os.path.normpath(str(root)))
    home_path = Path(os.path.normpath(str(home_path)))

    exclusions: List[str] = []
    for dirname in _MACOS_TCC_PROTECTED_HOME_DIRS:
        try:
            relative = (home_path / dirname).relative_to(root)
        except ValueError:
            continue
        if relative.parts:
            exclusions.append(relative.as_posix())
    return exclusions


_SEARCH_TIMEOUT_MARKER_RE = re.compile(r"\n?\[Command timed out after \d+s\]\s*$")


def _search_stdout_and_limit(result: ExecuteResult) -> tuple[str, Optional[str]]:
    """Return stdout cleaned for parsing and a limit reason for search timeouts."""
    if result.exit_code == 124:
        return _SEARCH_TIMEOUT_MARKER_RE.sub("", result.stdout), "search_timeout"
    return result.stdout, None


# A real rg/grep output line is a whitespace-free path token followed by ``:``
# (match/count), ``-`` (context), or nothing (files_only). Tool diagnostics
# ("rg: ...", "error: ...", indented carets) never match: the leading token
# forbids whitespace and a tool prefix is followed by ": " (space).
_SEARCH_OUTPUT_RE = re.compile(r'^([A-Za-z]:)?[^\s:][^\n]*?[:\-]\d|^[^\s:][^\s]*$')


def _split_tool_diagnostics(output: str) -> tuple[str, str]:
    """Separate rg/grep diagnostic lines from real match output.

    ``_exec`` merges stderr into stdout, so tool errors interleave with matches.
    Returns ``(diagnostics, payload)``. Classifying by SHAPE (not error prefix)
    lets the exit-2 guard tell a pure failure (no payload → surface the error)
    from a partial one (one unreadable file, others matched → keep matches), and
    guarantees error text is never parsed as a match.
    """
    diagnostics: list[str] = []
    payload: list[str] = []
    for line in output.split('\n'):
        if not line.strip():
            continue
        # Prefix check first: a real match path can contain "-<digit>" (e.g.
        # ".../pytest-686/..."), which the shape regex would accept as a match.
        stripped = line.lstrip()
        if stripped.startswith("rg: ") or stripped.startswith("grep: "):
            diagnostics.append(line)
            continue
        if line == "--" or _SEARCH_OUTPUT_RE.match(line):
            payload.append(line)
        else:
            diagnostics.append(line)
    return '\n'.join(diagnostics), '\n'.join(payload)


def _parse_search_context_line(line: str) -> tuple[str, int, str] | None:
    """Parse a ``path-line-content`` context line.

    Filenames may contain ``-<digits>-`` segments, so use the RIGHTMOST numeric
    separator: ``dir/file-12-name.py-8-context`` → (``dir/file-12-name.py``, 8).
    """
    if not line or line == "--":
        return None
    match = None
    for candidate in re.finditer(r'-(\d+)-', line):
        match = candidate
    if match is None:
        return None
    path = line[:match.start()]
    if not path:
        return None
    return path, int(match.group(1)), line[match.end():]


_REGEX_NEWLINE_ESCAPE_RE = re.compile(r"(?<!\\)(?:\\\\)*\\n")


def _pattern_has_regex_newline(pattern: str) -> bool:
    """True when a content regex wants to match a newline: a literal newline or a
    ``\\n`` escape with an ODD number of backslashes (``\\\\n`` is a literal
    backslash+n and must not count)."""
    return "\n" in pattern or bool(_REGEX_NEWLINE_ESCAPE_RE.search(pattern))


def _is_line_oriented_newline_error(error: Optional[str]) -> bool:
    """Return True for rg's hard error when multiline mode is required."""
    if not error:
        return False
    return "literal \"\\n\" is not allowed" in error and "--multiline" in error


def _maybe_warn_line_oriented_newline_pattern(result: SearchResult, pattern: str) -> SearchResult:
    """Attach a newline-regex warning only when search found no usable results."""
    if result.total_count != 0 or not _pattern_has_regex_newline(pattern):
        return result
    if result.error and not _is_line_oriented_newline_error(result.error):
        return result
    result.error = None
    result.warning = (
        "0 results found. Note: search_files content search is line-oriented "
        "and does not run ripgrep with -U/--multiline, so `\\n` in the regex "
        "does not match line breaks. Use context=N to inspect neighboring "
        "lines, or escape as `\\\\n` when searching for a literal backslash+n."
    )
    return result


# Match lines are "file:lineno:content". Windows paths carry a drive letter
# ("C:\path"), so a naive split(":") breaks — the regex handles both.
_MATCH_LINE_RE = re.compile(r'^([A-Za-z]:)?(.*?):(\d+):(.*)$')

# Output-mode → engine flag (identical for rg and grep).
_OUTPUT_MODE_FLAGS = {"files_only": "-l", "count": "-c"}


def _parse_search_output(result, output_mode: str, limit: int, offset: int,
                         context: int, warning: Optional[str] = None) -> SearchResult:
    """Parse rg/grep ``| head`` output into a SearchResult (shared by both engines).

    Exit codes: 0=matches, 1=none, 2=error — but both tools return 2 on PARTIAL
    errors (one unreadable file in a tree that otherwise matched), so an error is
    surfaced only when exit==2 AND no usable payload remains.
    ``warning`` is attached to files_only/content results (rg's multiline note).
    """
    stdout, limit_reason = _search_stdout_and_limit(result)
    diagnostics, payload = _split_tool_diagnostics(stdout)
    if result.exit_code == 2 and not payload.strip():
        error_msg = diagnostics.strip() or result.stdout.strip() or "Search error"
        return SearchResult(error=f"Search failed: {error_msg}", total_count=0)

    lines = [ln for ln in payload.strip().split('\n') if ln]
    if output_mode == "files_only":
        return SearchResult(
            files=lines[offset:offset + limit],
            total_count=len(lines),
            truncated=bool(limit_reason),
            limit_reason=limit_reason,
            warning=warning,
        )

    if output_mode == "count":
        counts = {}
        for line in lines:
            if ':' in line:
                path, n = line.rsplit(':', 1)
                try:
                    counts[path] = int(n)
                except ValueError:
                    pass
        return SearchResult(
            counts=counts,
            total_count=sum(counts.values()),
            truncated=bool(limit_reason),
            limit_reason=limit_reason,
        )

    matches = []
    for line in lines:
        if line == "--":
            continue
        m = _MATCH_LINE_RE.match(line)
        if m:
            matches.append(SearchMatch(
                path=(m.group(1) or '') + m.group(2),
                line_number=int(m.group(3)),
                content=m.group(4)[:500],
            ))
            continue
        # Context lines ("file-line-content") only when context was requested,
        # to avoid false positives on dash-heavy paths.
        if context > 0:
            parsed = _parse_search_context_line(line)
            if parsed:
                matches.append(SearchMatch(
                    path=parsed[0], line_number=parsed[1], content=parsed[2][:500],
                ))
    total = len(matches)
    return SearchResult(
        matches=matches[offset:offset + limit],
        total_count=total,
        truncated=total > offset + limit or bool(limit_reason),
        limit_reason=limit_reason,
        warning=warning,
    )


class SearchMixin:
    """File-name and content search via rg with find/grep fallbacks. Requires
    ``_exec``, ``_has_command``, ``_expand_path``, ``_escape_shell_arg``,
    ``_escape_native_tool_arg``, ``env`` and ``cwd`` from the host class."""

    def _macos_search_exclusions(self, path: str) -> List[str]:
        """Protected descendants to prune for this search root, if any.

        Gated on ``env.is_local``: ``sys.platform``/``_HOME`` describe the
        CONTROLLER, but the search runs on ``env``'s host — a macOS controller
        driving a Linux container must not prune the remote's Downloads. Envs
        without the flag (fakes, plugins) default to local semantics; pruning is
        a warning-carrying skip, never data loss.
        """
        env = getattr(self, "env", None)
        if env is not None and getattr(env, "is_local", True) is False:
            return []
        from tools import file_operations as _fo  # lazy: _HOME is monkeypatched there
        cwd = getattr(self.env, "cwd", None) or self.cwd
        return _macos_protected_search_exclusions(
            path, cwd=cwd, home=_fo._HOME, platform=sys.platform
        )

    def _protected_prune_paths(self, path: str) -> List[str]:
        """Absolute-ish protected paths for find's ``-path ... -prune``."""
        return [
            os.path.normpath(os.path.join(path, item))
            for item in self._macos_search_exclusions(path)
        ]

    def _path_exists_probe(self, path: str) -> str:
        """Stdout of the existence probe: contains "exists" or "not_found"."""
        return self._exec(
            f"test -e {self._escape_shell_arg(path)} && echo exists || echo not_found"
        ).stdout

    def _dispatch_search(self, pattern: str, path: str, target: str,
                         file_glob: Optional[str], limit: int, offset: int,
                         output_mode: str, context: int) -> SearchResult:
        if target == "files":
            return self._search_files(pattern, path, limit, offset)
        return self._search_content(pattern, path, file_glob, limit, offset,
                                    output_mode, context)

    def _path_not_found_result(self, path: str) -> SearchResult:
        """Error result for a missing search root, with nearby-entry suggestions."""
        parent = os.path.dirname(path) or "."
        basename_query = os.path.basename(path)
        hint_parts = [f"Path not found: {path}"]
        parent_check = self._exec(
            f"test -d {self._escape_shell_arg(parent)} && echo yes || echo no"
        )
        if "yes" in parent_check.stdout and basename_query:
            ls_result = self._exec(
                f"ls -1 {self._escape_shell_arg(parent)} 2>/dev/null | head -20"
            )
            if ls_result.exit_code == 0 and ls_result.stdout.strip():
                lower_q = basename_query.lower()
                candidates = []
                for entry in ls_result.stdout.strip().split('\n'):
                    if not entry:
                        continue
                    le = entry.lower()
                    if lower_q in le or le in lower_q or le.startswith(lower_q[:3]):
                        candidates.append(os.path.join(parent, entry))
                if candidates:
                    hint_parts.append("Similar paths: " + ", ".join(candidates[:5]))
        return SearchResult(error=". ".join(hint_parts), total_count=0)

    def _try_multi_path_search(self, pattern: str, path: str, target: str,
                               file_glob: Optional[str], limit: int, offset: int,
                               output_mode: str, context: int) -> Optional[SearchResult]:
        """Recover a not-found ``path`` that is really several paths in one string
        ("dir1 dir2" or comma-separated): search every existing part, merge, and
        note skipped parts. None when it doesn't look like a multi-path string."""
        parts = [p for chunk in path.split(",") for p in chunk.split() if p.strip()]
        if len(parts) < 2:
            return None
        existing, missing = [], []
        for p in parts:
            expanded = self._expand_path(p)
            (existing if self._path_exists(expanded) else missing).append(expanded)
        if not existing:
            return None

        merged = SearchResult()
        for p in existing:
            sub = self._dispatch_search(pattern, p, target, file_glob, limit, offset,
                                        output_mode, context)
            if sub.error:
                continue
            merged.matches.extend(sub.matches)
            merged.files.extend(sub.files)
            merged.counts.update(sub.counts)
            merged.total_count += sub.total_count
            merged.truncated = merged.truncated or sub.truncated
        merged.matches = merged.matches[:limit]
        merged.files = merged.files[:limit]
        note = f"path contained {len(parts)} entries; searched {len(existing)} that exist"
        if missing:
            note += "; skipped missing: " + ", ".join(missing[:3])
            if len(missing) > 3:
                note += f" (+{len(missing) - 3} more)"
        merged.warning = note
        return merged

    # (rg flags, message template) probes for a 0-match content search, in order.
    # The fixed-string probe only runs when the pattern has regex metacharacters.
    _ZERO_MATCH_PROBES = (
        ("-i", "0 exact matches, but {total} case-insensitive match(es) in {n} file(s): "
               "{paths} — the pattern's casing may be wrong."),
        # rg skips dotdirs and .gitignore'd files by default; say so instead of a bare zero.
        ("--hidden --no-ignore", "0 matches in visible files, but {total} match(es) in {n} "
                                 "hidden or gitignored file(s): {paths} — these are excluded by default."),
        ("-F", "0 regex matches, but {total} literal match(es) in {n} file(s): {paths} — the "
               "pattern contains regex metacharacters that likely need escaping "
               "(or pass a simpler substring)."),
    )

    def _zero_match_probe(self, pattern: str, path: str,
                          file_glob: Optional[str]) -> Optional[str]:
        """Steering hint for a 0-match content search, or None.

        A bare zero gives the model nothing to act on, so run cheap count-only rg
        probes (case-insensitive, hidden/ignored, fixed-string) and report the
        first that hits. Bounded to three rg invocations.
        """
        if not self._has_command('rg'):
            return None
        has_meta = bool(re.search(r"[.\[\](){}?*+^$\\|]", pattern))
        glob_expr = f" --glob {self._escape_shell_arg(file_glob)}" if file_glob else ""
        for flags, template in self._ZERO_MATCH_PROBES:
            if flags == "-F" and not has_meta:
                continue
            probe = self._exec(
                f"rg {flags} --count-matches{glob_expr} "
                f"{self._escape_shell_arg(pattern)} {self._escape_native_tool_arg(path)} "
                f"2>/dev/null | head -50",
                timeout=30,
            )
            total, per_file = 0, []
            for line in (probe.stdout or "").strip().splitlines():
                p, _sep, n = line.rpartition(":")
                if n.isdigit():
                    total += int(n)
                    per_file.append(p)
            if total > 0:
                extra = len(per_file) - 5
                paths = ", ".join(per_file[:5]) + (f" (+{extra} more)" if extra > 0 else "")
                return template.format(total=total, n=len(per_file), paths=paths)
        return None

    def _search_files(self, pattern: str, path: str, limit: int, offset: int) -> SearchResult:
        """Search for files by name (glob-like): rg --files, else find."""
        search_pattern = pattern if (not pattern.startswith('**/') and '/' not in pattern) \
            else pattern.split('/')[-1]

        search_root = Path(path)
        has_hidden_path_ancestor = any(
            part not in {".", ".."} and part.startswith(".")
            for part in search_root.parts
        )

        # rg respects .gitignore, skips hidden dirs, and walks in parallel (~200x find).
        if self._has_command('rg'):
            return self._search_files_rg(search_pattern, path, limit, offset)
        if not self._has_command('find'):
            return SearchResult(
                error="File search requires 'rg' (ripgrep) or 'find'. "
                      "Install ripgrep for best results: "
                      "https://github.com/BurntSushi/ripgrep#installation"
            )

        # Hidden roots: find's path filter would exclude everything under the root,
        # so gather full output and filter descendants in Python (pagination too).
        hidden_filter_expr = "" if has_hidden_path_ancestor else " -not -path '*/.*'"
        pagination_expr = "" if has_hidden_path_ancestor else f" | tail -n +{offset + 1} | head -n {limit}"

        # Prune protected dirs BEFORE traversal so macOS never sees an access attempt.
        protected_paths = self._protected_prune_paths(path)
        prune_expr = ""
        if protected_paths:
            prune_terms = " -o ".join(
                f"-path {self._escape_shell_arg(item)}" for item in protected_paths
            )
            prune_expr = f" \\( {prune_terms} \\) -prune -o"

        base = (f"find {self._escape_shell_arg(path)}{prune_expr}{hidden_filter_expr} "
                f"-type f -name {self._escape_shell_arg(search_pattern)} ")
        result = self._exec(f"{base}-printf '%T@ %p\\n' 2>/dev/null | sort -rn{pagination_expr}", timeout=60)
        stdout, limit_reason = _search_stdout_and_limit(result)
        if not stdout.strip() and not limit_reason:
            # BSD find (macOS) has no -printf.
            result = self._exec(f"{base}2>/dev/null | sort -rn{pagination_expr}", timeout=60)
            stdout, limit_reason = _search_stdout_and_limit(result)

        files = []
        for line in stdout.strip().split('\n'):
            if not line:
                continue
            parts = line.split(' ', 1)
            files.append(parts[1] if len(parts) == 2 and parts[0].replace('.', '').isdigit() else line)

        if has_hidden_path_ancestor:
            normalized_root = search_root.resolve()
            filtered_files = []
            for file_path in files:
                try:
                    rel_parts = Path(file_path).resolve().relative_to(normalized_root).parts
                except ValueError:
                    rel_parts = Path(file_path).parts
                if any(part not in {".", ".."} and part.startswith(".") for part in rel_parts):
                    continue
                filtered_files.append(file_path)
            files = filtered_files[offset:offset + limit]

        return SearchResult(
            files=files,
            total_count=len(files),
            truncated=bool(limit_reason),
            limit_reason=limit_reason,
        )

    def _search_files_rg(self, pattern: str, path: str, limit: int, offset: int) -> SearchResult:
        """File-name search via ``rg --files``, mtime-sorted when rg >= 13 supports --sortr."""
        # Wrap bare names so -g matches at any depth (equivalent to find -name).
        glob_pattern = f"*{pattern}" if ('/' not in pattern and not pattern.startswith('*')) else pattern

        fetch_limit = limit + offset
        exclusion_globs = " ".join(
            f"--glob {self._escape_shell_arg(f'!{item}/**')}"
            for item in self._macos_search_exclusions(path)
        )
        exclusion_args = f" {exclusion_globs}" if exclusion_globs else ""
        tail = (f"-g {self._escape_shell_arg(glob_pattern)}{exclusion_args} "
                f"{self._escape_native_tool_arg(path)} 2>/dev/null | head -n {fetch_limit}")
        result = self._exec(f"rg --files --sortr=modified {tail}", timeout=60)
        stdout, limit_reason = _search_stdout_and_limit(result)
        all_files = [f for f in stdout.strip().split('\n') if f]

        if not all_files and not limit_reason:
            # --sortr may have failed on older rg; retry without it.
            result = self._exec(f"rg --files {tail}", timeout=60)
            stdout, limit_reason = _search_stdout_and_limit(result)
            all_files = [f for f in stdout.strip().split('\n') if f]

        return SearchResult(
            files=all_files[offset:offset + limit],
            total_count=len(all_files),
            truncated=len(all_files) >= fetch_limit or bool(limit_reason),
            limit_reason=limit_reason,
        )

    def _search_content(self, pattern: str, path: str, file_glob: Optional[str],
                        limit: int, offset: int, output_mode: str, context: int) -> SearchResult:
        """Content search: rg, else grep; attaches zero-match steering hints."""
        used_rg = False
        if self._has_command('rg'):
            used_rg = True
            result = self._search_with_rg(pattern, path, file_glob, limit, offset,
                                          output_mode, context)
        elif self._has_command('grep'):
            result = self._search_with_grep(pattern, path, file_glob, limit, offset,
                                            output_mode, context)
        else:
            return SearchResult(
                error="Content search requires ripgrep (rg) or grep. "
                      "Install ripgrep: https://github.com/BurntSushi/ripgrep#installation"
            )

        if (not result.error and result.total_count == 0
                and not result.matches and not result.files and not result.counts):
            try:
                hint = self._zero_match_probe(pattern, path, file_glob)
            except Exception:
                hint = None
            if hint:
                result.warning = hint if not result.warning else f"{result.warning} {hint}"

        # rg auto-enables --multiline for \n patterns, so the line-oriented
        # explanation only applies to the grep fallback.
        if used_rg:
            return result
        return _maybe_warn_line_oriented_newline_pattern(result, pattern)

    def _search_with_rg(self, pattern: str, path: str, file_glob: Optional[str],
                        limit: int, offset: int, output_mode: str, context: int) -> SearchResult:
        """Search using ripgrep."""
        cmd_parts = ["rg", "--line-number", "--no-heading", "--with-filename"]

        # A regex \n can't match in line-oriented mode (rg hard-errors); enable -U
        # up front when the pattern clearly wants to cross lines, and say so.
        multiline = _pattern_has_regex_newline(pattern)
        if multiline:
            cmd_parts.append("--multiline")
        if context > 0:
            cmd_parts.extend(["-C", str(context)])
        for item in self._macos_search_exclusions(path):
            cmd_parts.extend(["--glob", self._escape_shell_arg(f"!{item}/**")])
        if file_glob:
            cmd_parts.extend(["--glob", self._escape_shell_arg(file_glob)])
        if output_mode in _OUTPUT_MODE_FLAGS:
            cmd_parts.append(_OUTPUT_MODE_FLAGS[output_mode])
        cmd_parts.append(self._escape_shell_arg(pattern))
        # rg is a native Windows binary (winget/cargo/choco): needs C:/... not MSYS /c/...
        cmd_parts.append(self._escape_native_tool_arg(path))

        # Fetch extra rows to report the true total; context mode also emits "--"
        # separators, so grab generously and filter in Python.
        fetch_limit = limit + offset + 200 if context > 0 else limit + offset
        cmd_parts.extend(["|", "head", "-n", str(fetch_limit)])

        # pipefail so rg's exit 2 survives `| head` (else head's 0 masks it). rg
        # exits 0 on SIGPIPE from a truncating head, so no false errors.
        cmd = "set -o pipefail; " + " ".join(cmd_parts)
        result = self._exec(cmd, timeout=60)
        ml_note = (
            "Pattern contains \\n — multiline mode (-U) was enabled automatically "
            "so the regex can match across line boundaries."
        ) if multiline else None
        return _parse_search_output(result, output_mode, limit, offset, context, warning=ml_note)

    def _search_with_grep(self, pattern: str, path: str, file_glob: Optional[str],
                          limit: int, offset: int, output_mode: str, context: int) -> SearchResult:
        """Fallback search using grep."""
        # -H forces filenames; -E matches rg regex behavior; --exclude-dir='.*'
        # mirrors rg's hidden-dir default (.git/, .hub/index-cache/, ...).
        cmd_parts = ["grep", "-rnHE", "--exclude-dir='.*'"]

        # grep's --exclude-dir matches BASENAMES anywhere in the tree, so it can't
        # express "only the home-level Downloads"; route protected-dir pruning
        # through find's path-scoped -prune instead.
        protected_paths = self._protected_prune_paths(path)
        if protected_paths:
            return self._search_with_grep_pruned(
                pattern, path, file_glob, limit, offset, output_mode, context,
                protected_paths,
            )

        if context > 0:
            cmd_parts.extend(["-C", str(context)])
        if file_glob:
            cmd_parts.extend(["--include", self._escape_shell_arg(file_glob)])
        if output_mode in _OUTPUT_MODE_FLAGS:
            cmd_parts.append(_OUTPUT_MODE_FLAGS[output_mode])
        cmd_parts.append(self._escape_shell_arg(pattern))

        # grep applies --exclude-dir to the search root too, so a relative root
        # "." would be excluded by '.*'. Anchor relative paths at the shell's
        # live $PWD (quoted separately so user paths stay escaped).
        is_absolute = path.startswith(("/", "\\\\")) or bool(
            re.match(r"^[A-Za-z]:[\\/]", path)
        )
        if is_absolute:
            search_root = self._escape_shell_arg(path)
        else:
            relative_path = path[2:] if path.startswith("./") else path
            search_root = '"$PWD"'
            if relative_path not in {"", "."}:
                search_root += f"/{self._escape_shell_arg(relative_path)}"
        cmd_parts.append(search_root)

        fetch_limit = limit + offset + (200 if context > 0 else 0)
        cmd_parts.extend(["|", "head", "-n", str(fetch_limit)])

        # pipefail so grep's exit 2 survives `| head`; a truncating head makes
        # grep exit 141 (SIGPIPE), which the strict ==2 guard ignores.
        cmd = "set -o pipefail; " + " ".join(cmd_parts)
        result = self._exec(cmd, timeout=60)
        return _parse_search_output(result, output_mode, limit, offset, context)

    def _search_with_grep_pruned(self, pattern: str, path: str, file_glob: Optional[str],
                                 limit: int, offset: int, output_mode: str, context: int,
                                 protected_paths: List[str]) -> SearchResult:
        """grep fallback with PATH-scoped protected-dir pruning.

        ``find ... -prune`` enumerates files (traversal never enters protected
        dirs, so macOS never sees an access attempt) and hands them to grep via
        ``-exec {} +``; hidden dirs are pruned to mirror ``--exclude-dir='.*'``.
        Trade-off: find folds grep's exit code into its own generic non-zero, so
        a hard grep error surfaces as an empty result rather than exit 2 —
        acceptable for this darwin-local-broad-search-only branch.
        """
        grep_parts = ["grep", "-nHE"]
        if context > 0:
            grep_parts.extend(["-C", str(context)])
        if output_mode in _OUTPUT_MODE_FLAGS:
            grep_parts.append(_OUTPUT_MODE_FLAGS[output_mode])
        grep_parts.append(self._escape_shell_arg(pattern))

        prune_terms = " -o ".join(
            f"-path {self._escape_shell_arg(item)}" for item in protected_paths
        )
        find_parts = [
            "find", self._escape_shell_arg(path or "."),
            f"\\( {prune_terms} \\) -prune", "-o",
            "\\( -type d -name '.*' \\) -prune", "-o",
            "-type f",
        ]
        if file_glob:
            find_parts.extend(["-name", self._escape_shell_arg(file_glob)])
        find_parts.extend(["-exec", *grep_parts, "{}", "+"])
        fetch_limit = limit + offset + (200 if context > 0 else 0)
        cmd = (
            "set -o pipefail; " + " ".join(find_parts)
            + f" 2>/dev/null | head -n {fetch_limit}"
        )
        result = self._exec(cmd, timeout=60)
        return _parse_search_output(result, output_mode, limit, offset, context)
