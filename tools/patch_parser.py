#!/usr/bin/env python3
"""V4A patch format parser and applier (format used by codex, cline, etc.).

    *** Begin Patch
    *** Update File: path/to/file.py
    @@ optional context hint @@
     context line (space prefix)
    -removed line
    +added line
    *** Add File: path/to/new.py
    +new file content
    *** Delete File: path/to/old.py
    *** Move File: old/path.py -> new/path.py
    *** End Patch

    operations, error = parse_v4a_patch(patch_content)
    result = apply_v4a_operations(operations, file_ops)
"""

import difflib
import inspect
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple


class OperationType(Enum):
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    MOVE = "move"


@dataclass
class HunkLine:
    """A single line in a patch hunk."""
    prefix: str  # ' ', '-', or '+'
    content: str


@dataclass
class Hunk:
    """A group of changes within a file."""
    context_hint: Optional[str] = None
    lines: List[HunkLine] = field(default_factory=list)


@dataclass
class PatchOperation:
    """A single operation in a V4A patch."""
    operation: OperationType
    file_path: str
    new_path: Optional[str] = None  # For move operations
    hunks: List[Hunk] = field(default_factory=list)
    content: Optional[str] = None  # For add file operations


# Markers must occupy the whole line at column 0 so content lines that merely
# mention the format ("+*** End Patch") can't truncate or reset the patch.
_BEGIN_MARKER = re.compile(r'^\*\*\*\s*Begin\s+Patch\s*$')
_END_MARKER = re.compile(r'^\*\*\*\s*End\s+Patch\s*$')
_OP_MARKERS: List[Tuple[OperationType, re.Pattern]] = [
    (OperationType.UPDATE, re.compile(r'\*\*\*\s*Update\s+File:\s*(.+)')),
    (OperationType.ADD, re.compile(r'\*\*\*\s*Add\s+File:\s*(.+)')),
    (OperationType.DELETE, re.compile(r'\*\*\*\s*Delete\s+File:\s*(.+)')),
    (OperationType.MOVE, re.compile(r'\*\*\*\s*Move\s+File:\s*(.+?)\s*->\s*(.+)')),
]
_HINT_RE = re.compile(r'@@\s*(.+?)\s*@@')


def parse_v4a_patch(patch_content: str) -> Tuple[List[PatchOperation], Optional[str]]:
    """Parse a V4A patch into operations.

    Returns ``(operations, None)`` — ``[]`` for an empty patch is not an
    error — or ``([], "Parse error: ...")`` for malformed operations.
    """
    # Tolerate CRLF bodies: a stray ``\r`` would otherwise end up in every
    # HunkLine.content and defeat the anchored Begin/End markers.
    lines = [ln[:-1] if ln.endswith('\r') else ln for ln in patch_content.split('\n')]

    start_idx = -1  # parse from the top when no Begin marker is present
    end_idx = len(lines)
    for i, line in enumerate(lines):
        if _BEGIN_MARKER.match(line):
            start_idx = i
        elif _END_MARKER.match(line):
            end_idx = i
            break

    operations: List[PatchOperation] = []
    current_op: Optional[PatchOperation] = None
    current_hunk: Optional[Hunk] = None

    def _flush() -> None:
        if current_op:
            if current_hunk and current_hunk.lines:
                current_op.hunks.append(current_hunk)
            operations.append(current_op)

    for line in lines[start_idx + 1:end_idx]:
        op_match = next(
            ((kind, m) for kind, rx in _OP_MARKERS if (m := rx.match(line))), None,
        )
        if op_match:
            kind, m = op_match
            _flush()
            current_op = PatchOperation(
                operation=kind,
                file_path=m.group(1).strip(),
                new_path=m.group(2).strip() if kind is OperationType.MOVE else None,
            )
            # UPDATE hunks start lazily (at '@@' or the first hunk line); ADD
            # collects all '+' lines into one hunk; DELETE/MOVE are complete.
            current_hunk = Hunk() if kind is OperationType.ADD else None
            if kind in (OperationType.DELETE, OperationType.MOVE):
                operations.append(current_op)
                current_op = None
        elif line.startswith('@@'):
            if current_op:
                if current_hunk and current_hunk.lines:
                    current_op.hunks.append(current_hunk)
                hint_match = _HINT_RE.match(line)
                current_hunk = Hunk(context_hint=hint_match.group(1) if hint_match else None)
        elif current_op and line:
            if current_hunk is None:
                current_hunk = Hunk()
            if line[0] in '+- ':
                current_hunk.lines.append(HunkLine(line[0], line[1:]))
            elif line[0] != '\\':  # "\ No newline at end of file" marker is skipped
                current_hunk.lines.append(HunkLine(' ', line))  # implicit context line
    _flush()

    parse_errors: List[str] = []
    for op in operations:
        if not op.file_path:
            parse_errors.append("Operation with empty file path")
        if op.operation == OperationType.UPDATE and not op.hunks:
            parse_errors.append(f"UPDATE {op.file_path!r}: no hunks found")
        if op.operation == OperationType.MOVE and not op.new_path:
            parse_errors.append(f"MOVE {op.file_path!r}: missing destination path (expected 'src -> dst')")
    if parse_errors:
        return [], "Parse error: " + "; ".join(parse_errors)
    return operations, None


def _count_occurrences(text: str, pattern: str) -> int:
    """Count occurrences of *pattern* in *text*, advancing one char per hit (overlaps count)."""
    count = 0
    start = 0
    while True:
        pos = text.find(pattern, start)
        if pos == -1:
            break
        count += 1
        start = pos + 1
    return count


def _split_hunk(hunk: Hunk) -> Tuple[List[str], List[str]]:
    """``(search_lines, replace_lines)``: context+removed vs context+added."""
    search = [l.content for l in hunk.lines if l.prefix in {' ', '-'}]
    replace = [l.content for l in hunk.lines if l.prefix in {' ', '+'}]
    return search, replace


def _no_match_hint(error: Optional[str], search_pattern: str, content: str) -> str:
    """Best-effort 'Did you mean...' suffix; never lets a hint failure mask the real error."""
    try:
        from tools.fuzzy_match import format_no_match_hint
        return format_no_match_hint(error, 0, search_pattern, content)
    except Exception:
        return ""


def _validate_operations(
    operations: List[PatchOperation],
    file_ops: Any,
) -> List[str]:
    """Dry-run every operation; return error strings (empty list = safe to apply).

    UPDATE hunks are simulated in order so later hunks validate against
    post-earlier-hunk content, exactly as the apply phase will see it.
    """
    from tools.fuzzy_match import fuzzy_find_and_replace, is_already_applied

    errors: List[str] = []
    real_change_count = 0

    # Virtual overlay so inter-op state validates (e.g. a MOVE creating the
    # destination a later UPDATE targets). UPDATE/MOVE reads consult it first.
    pending_content: dict = {}   # path -> content produced by an earlier op
    removed_paths: set = set()   # paths a MOVE/DELETE has taken away

    def _read(path: str):
        if path in removed_paths and path not in pending_content:
            return None, "file not found"
        if path in pending_content:
            return pending_content[path], None
        r = file_ops.read_file_raw(path)
        if r.error:
            return None, r.error
        return r.content, None

    for op in operations:
        if op.operation != OperationType.UPDATE:
            real_change_count += 1
        if op.operation == OperationType.UPDATE:
            content, read_err = _read(op.file_path)
            if read_err:
                errors.append(f"{op.file_path}: {read_err}")
                continue

            simulated = content
            for hunk_index, hunk in enumerate(op.hunks, start=1):
                search_lines, replace_lines = _split_hunk(hunk)
                if not any(l.prefix in '-+' for l in hunk.lines):
                    # Inert anchor hunk (context only) — models emit these
                    # between real changes; ignore without failing the patch.
                    continue
                real_change_count += 1
                if not search_lines:
                    # Addition-only hunk: the context hint must be unique.
                    if hunk.context_hint:
                        occurrences = _count_occurrences(simulated, hunk.context_hint)
                        if occurrences == 0:
                            errors.append(
                                f"{op.file_path}: addition-only hunk context hint "
                                f"'{hunk.context_hint}' not found"
                            )
                        elif occurrences > 1:
                            errors.append(
                                f"{op.file_path}: addition-only hunk context hint "
                                f"'{hunk.context_hint}' is ambiguous "
                                f"({occurrences} occurrences)"
                            )
                    continue

                search_pattern = '\n'.join(search_lines)
                replacement = '\n'.join(replace_lines)
                if search_lines == replace_lines:
                    # Identical -/+ lines: apply skips it as a no-op, so
                    # validation must not reject it with the identical-strings error.
                    continue

                new_simulated, count, _strategy, match_error = fuzzy_find_and_replace(
                    simulated, search_pattern, replacement, replace_all=False
                )
                if count == 0:
                    # Already-applied hunk (edit landed in a prior call): treat
                    # as a no-op so multi-hunk patches don't fail wholesale.
                    # The apply phase performs the same skip.
                    if is_already_applied(simulated or "", search_pattern, replacement):
                        continue
                    label = f"'{hunk.context_hint}'" if hunk.context_hint else "(no hint)"
                    errors.append(
                        f"{op.file_path}: hunk {hunk_index} {label} not found"
                        + (f" — {match_error}" if match_error else "")
                        + _no_match_hint(match_error, search_pattern, simulated)
                    )
                else:
                    simulated = new_simulated
            pending_content[op.file_path] = simulated

        elif op.operation == OperationType.DELETE:
            _content, read_err = _read(op.file_path)
            if read_err:
                errors.append(f"{op.file_path}: file not found for deletion")
            else:
                removed_paths.add(op.file_path)
                pending_content.pop(op.file_path, None)

        elif op.operation == OperationType.MOVE:
            if not op.new_path:
                errors.append(f"{op.file_path}: MOVE operation missing destination path")
                continue
            src_content, src_err = _read(op.file_path)
            if src_err:
                errors.append(f"{op.file_path}: source file not found for move")
            _dst, dst_err = _read(op.new_path)
            if not dst_err:
                errors.append(
                    f"{op.new_path}: destination already exists — move would overwrite"
                )
            # Only a cleanly-validated move updates the overlay.
            if not src_err and dst_err:
                pending_content[op.new_path] = src_content if src_content is not None else ""
                pending_content.pop(op.file_path, None)
                removed_paths.add(op.file_path)

        # ADD: parent directory creation handled by write_file; no pre-check needed.

    if not errors and real_change_count == 0:
        errors.append("Patch contains no changes (only context lines were provided)")
    return errors


# Every _apply_* returns (success, diff_or_error, lsp_diagnostics, lint_result).
ApplyResult = Tuple[bool, str, Optional[str], Optional[dict]]


def apply_v4a_operations(operations: List[PatchOperation],
                         file_ops: Any) -> 'PatchResult':
    """Validate all operations, then apply them (two-phase, atomic on validation failure).

    A phase-2 failure (e.g. a race between validation and apply) is reported
    with a note to run ``git diff`` since state may be inconsistent.
    ``file_ops`` needs ``read_file_raw``, ``write_file``, ``delete_file``, ``move_file``.
    """
    from tools.file_operations import PatchResult  # avoid circular import

    validation_errors = _validate_operations(operations, file_ops)
    if validation_errors:
        return PatchResult(
            success=False,
            error="Patch validation failed (no files were modified):\n"
                  + "\n".join(f"  • {e}" for e in validation_errors),
        )

    files_modified: List[str] = []
    files_created: List[str] = []
    files_deleted: List[str] = []
    all_diffs: List[str] = []
    # V4A bypasses the WriteResult/PatchResult plumbing that write_file uses,
    # so LSP diagnostics and lint must be propagated explicitly per file.
    lsp_blocks: List[str] = []
    errors: List[str] = []
    lint_results: Dict[str, dict] = {}

    dispatch: Dict[OperationType, Tuple[Callable[[PatchOperation, Any], ApplyResult], List[str], str]] = {
        OperationType.ADD: (_apply_add, files_created, "add"),
        OperationType.DELETE: (_apply_delete, files_deleted, "delete"),
        OperationType.MOVE: (_apply_move, files_modified, "move"),
        OperationType.UPDATE: (_apply_update, files_modified, "update"),
    }

    for op in operations:
        try:
            handler, bucket, verb = dispatch[op.operation]
            ok, payload, lsp, lint = handler(op, file_ops)
            if not ok:
                errors.append(f"Failed to {verb} {op.file_path}: {payload}")
                continue
            label = op.file_path
            if op.operation is OperationType.MOVE:
                label = f"{op.file_path} -> {op.new_path}"
            bucket.append(label)
            all_diffs.append(payload)
            if lsp:
                lsp_blocks.append(lsp)
            if lint:
                lint_results[op.file_path] = lint
        except Exception as e:
            errors.append(f"Error processing {op.file_path}: {str(e)}")

    # Each LSP block carries its own <diagnostics file="..."> header, so plain
    # concatenation keeps per-file attribution.
    result_kwargs = dict(
        diff='\n'.join(all_diffs),
        files_modified=files_modified,
        files_created=files_created,
        files_deleted=files_deleted,
        lint=lint_results if lint_results else None,
        lsp_diagnostics="\n\n".join(lsp_blocks) if lsp_blocks else None,
    )
    if errors:
        return PatchResult(
            success=False,
            error="Apply phase failed (state may be inconsistent — run `git diff` to assess):\n"
                  + "\n".join(f"  • {e}" for e in errors),
            **result_kwargs,
        )
    return PatchResult(success=True, **result_kwargs)


def _write_file_accepts_pre_content(file_ops: Any) -> bool:
    """True when ``file_ops.write_file`` accepts a ``pre_content`` kwarg.

    Decided from the signature rather than catching TypeError around the call,
    so a TypeError raised *inside* a capable write_file propagates instead of
    triggering a second, duplicate write. Unintrospectable callables get the
    basic two-argument form.
    """
    try:
        params = inspect.signature(file_ops.write_file).parameters
    except (TypeError, ValueError):
        return False
    return "pre_content" in params or any(
        p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


def _apply_add(op: PatchOperation, file_ops: Any) -> ApplyResult:
    """Create a file from the hunks' '+' lines."""
    content_lines = [line.content for hunk in op.hunks for line in hunk.lines if line.prefix == '+']
    result = file_ops.write_file(op.file_path, '\n'.join(content_lines))
    if result.error:
        return False, result.error, None, None
    diff = f"--- /dev/null\n+++ b/{op.file_path}\n" + '\n'.join(f"+{line}" for line in content_lines)
    return True, diff, getattr(result, "lsp_diagnostics", None), getattr(result, "lint", None)


def _apply_delete(op: PatchOperation, file_ops: Any) -> ApplyResult:
    """Delete a file, producing a real unified diff of the removed content."""
    # Validation already confirmed existence; the re-read guards against races.
    read_result = file_ops.read_file_raw(op.file_path)
    if read_result.error:
        return False, f"Cannot delete {op.file_path}: file not found", None, None
    result = file_ops.delete_file(op.file_path)
    if result.error:
        return False, result.error, None, None
    diff = ''.join(difflib.unified_diff(
        read_result.content.splitlines(keepends=True), [],
        fromfile=f"a/{op.file_path}", tofile="/dev/null",
    ))
    return True, diff or f"# Deleted: {op.file_path}", None, None


def _apply_move(op: PatchOperation, file_ops: Any) -> ApplyResult:
    result = file_ops.move_file(op.file_path, op.new_path)
    if result.error:
        return False, result.error, None, None
    return True, f"# Moved: {op.file_path} -> {op.new_path}", None, None


def _insert_addition_only(new_content: str, hunk: Hunk, insert_text: str) -> Tuple[Optional[str], Optional[str]]:
    """Place an addition-only hunk after its context hint (or at EOF). Returns (content, error)."""
    if hunk.context_hint:
        occurrences = _count_occurrences(new_content, hunk.context_hint)
        if occurrences > 1:
            return None, (
                f"Addition-only hunk: context hint '{hunk.context_hint}' is ambiguous "
                f"({occurrences} occurrences) — provide a more unique hint"
            )
        if occurrences == 1:
            hint_pos = new_content.find(hunk.context_hint)
            eol = new_content.find('\n', hint_pos)
            if eol != -1:
                return new_content[:eol + 1] + insert_text + '\n' + new_content[eol + 1:], None
            return new_content + '\n' + insert_text, None
        # Hint not found — append at end as a safe fallback.
    return new_content.rstrip('\n') + '\n' + insert_text + '\n', None


def _apply_update(op: PatchOperation, file_ops: Any) -> ApplyResult:
    """Apply each hunk via fuzzy replace, then write once."""
    from tools.fuzzy_match import fuzzy_find_and_replace, is_already_applied

    # Raw read: no line-number prefixes or per-line truncation.
    read_result = file_ops.read_file_raw(op.file_path)
    if read_result.error:
        return False, f"Cannot read file: {read_result.error}", None, None
    current_content = read_result.content
    new_content = current_content

    for hunk in op.hunks:
        search_lines, replace_lines = _split_hunk(hunk)
        if search_lines and search_lines == replace_lines:
            continue
        if not search_lines:
            new_content, err = _insert_addition_only(new_content, hunk, '\n'.join(replace_lines))
            if err:
                return False, err, None, None
            continue

        search_pattern = '\n'.join(search_lines)
        replacement = '\n'.join(replace_lines)
        new_content, count, _strategy, error = fuzzy_find_and_replace(
            new_content, search_pattern, replacement, replace_all=False
        )
        if not (error and count == 0):
            continue

        # Retry inside a window around the context hint, if any.
        if hunk.context_hint:
            hint_pos = new_content.find(hunk.context_hint)
            if hint_pos != -1:
                window_start = max(0, hint_pos - 500)
                window_end = min(len(new_content), hint_pos + 2000)
                window_new, count, _strategy, error = fuzzy_find_and_replace(
                    new_content[window_start:window_end], search_pattern, replacement, replace_all=False
                )
                if count > 0:
                    new_content = new_content[:window_start] + window_new + new_content[window_end:]
                    error = None
        if error:
            # Mirror the validation-phase already-applied skip, or the two
            # phases disagree and the whole patch fails here.
            if is_already_applied(new_content, search_pattern, replacement):
                continue
            return False, f"Could not apply hunk: {error}" + _no_match_hint(error, search_pattern, new_content), None, None

    # Pass pre_content to skip a redundant re-read inside write_file when supported.
    if _write_file_accepts_pre_content(file_ops):
        write_result = file_ops.write_file(op.file_path, new_content, pre_content=current_content)
    else:
        write_result = file_ops.write_file(op.file_path, new_content)
    if write_result.error:
        return False, write_result.error, None, None

    diff = ''.join(difflib.unified_diff(
        current_content.splitlines(keepends=True),
        new_content.splitlines(keepends=True),
        fromfile=f"a/{op.file_path}",
        tofile=f"b/{op.file_path}",
    ))
    return True, diff, getattr(write_result, "lsp_diagnostics", None), getattr(write_result, "lint", None)
