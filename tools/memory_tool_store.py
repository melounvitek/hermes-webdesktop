"""MemoryStore — bounded, file-backed curated memory (MEMORY.md / USER.md).
Entries are joined by ``ENTRY_DELIMITER``; budgets are in chars (model-
independent). Module state that tests monkeypatch (``get_memory_dir``,
``fcntl``/``msvcrt``) stays in ``tools.memory_tool`` and is read lazily."""

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils import atomic_write_text
from tools.threat_patterns import first_threat_message as _first_threat_message

logger = logging.getLogger("tools.memory_tool")

# System-prompt block header prefixes rendered by MemoryStore._render_block.
# agent/conversation_compression.py uses them to detect a leftover block for a
# target whose entries have since been emptied — keep in lockstep.
MEMORY_BLOCK_HEADERS = {
    "memory": "MEMORY (your personal notes)",
    "user": "USER PROFILE (who the user is)",
}

ENTRY_DELIMITER = "\n§\n"

# Sentinel from ``_reload_target``: the file EXISTS but could not be read.
# Distinct from a drift-backup path (str) and a clean reload (None); the caller
# must abort rather than persist over an unreadable file.
_READ_FAILED = object()


def _memory_dir() -> Path:
    from tools import memory_tool

    return memory_tool.get_memory_dir()


def _scan_memory_content(content: str) -> Optional[str]:
    """Scan memory content for injection/exfil patterns ("strict" scope: memory
    enters the system prompt as a frozen snapshot, so a poisoned entry persists
    across sessions). Returns the error string if blocked."""
    return _first_threat_message(content, scope="strict")


def _drift_error(path: "Path", bak_path: str) -> Dict[str, Any]:
    """Error dict for external drift: the on-disk file wouldn't round-trip
    through the parser/serializer, so flushing would discard content added by a
    patch tool, shell append, manual edit, or sister session."""
    return {
        "success": False,
        "error": (
            f"Refusing to write {path.name}: file on disk has content that wouldn't round-trip "
            f"through the memory tool (likely added by the patch tool, a shell append, a manual edit, "
            f"or a concurrent session). A snapshot was saved to {bak_path}. Resolve the drift first — "
            f"either rewrite the file as a clean §-delimited list of entries, or move the extra "
            f"content out — then retry. This guard exists to prevent silent data loss (issue #26045)."
        ),
        "drift_backup": bak_path,
        "remediation": (
            "Open the .bak file, integrate the missing entries into the memory tool one at a time via "
            "memory(action=add, content=...), then remove or rewrite the original file to a clean state."
        ),
    }


def _read_failed_error(path: "Path") -> Dict[str, Any]:
    """Error dict for an unreadable (but existing) memory file. Treating it as
    empty and saving would rewrite the whole file from ``[]`` — wiping memory."""
    return {"success": False, "error": (
        f"Refusing to write {path.name}: the file exists on disk but could not be read right now "
        f"(temporarily locked by another program, a permission change, invalid/corrupt text encoding, "
        f"or a filesystem error). Treating an unreadable file as empty and saving would wipe existing "
        f"memory, so the write is refused. Nothing was changed — retry in a moment."
    )}


def _find_unique_match(entries: List[str], old_text: str) -> Tuple[Optional[int], bool]:
    """``(index, ambiguous)`` for entries containing *old_text*. Exact-duplicate
    matches are safe (first wins); distinct matches → ``(None, True)``."""
    matches = [i for i, e in enumerate(entries) if old_text in e]
    if not matches:
        return None, False
    if len({entries[i] for i in matches}) > 1:
        return None, True
    return matches[0], False


class MemoryStore:
    """Bounded curated memory with file persistence; one instance per AIAgent.
    ``_system_prompt_snapshot`` is frozen at load time (prefix-cache stable);
    ``memory_entries`` / ``user_entries`` are live state persisted to disk."""

    # After this many failed consolidation attempts (overflow / zero-match) in
    # ONE turn, return a terminal "save skipped" result instead of "retry in
    # this turn", so a fragile replace/add can't loop the turn to budget
    # exhaustion and suppress the user's reply.
    _MAX_CONSOLIDATION_FAILURES_PER_TURN = 3

    def __init__(self, memory_char_limit: int = 2200, user_char_limit: int = 1375, *,
                 memory_enabled: bool = True, user_profile_enabled: bool = True):
        self.memory_entries: List[str] = []
        self.user_entries: List[str] = []
        self.memory_char_limit = memory_char_limit
        self.user_char_limit = user_char_limit
        self.memory_enabled = memory_enabled
        self.user_profile_enabled = user_profile_enabled
        self._system_prompt_snapshot: Dict[str, str] = {"memory": "", "user": ""}
        # Per-turn counter of failed at-capacity consolidation attempts; reset
        # at each turn boundary by reset_consolidation_failures().
        self._consolidation_failures = 0

    def target_enabled(self, target: str) -> bool:
        """Return whether this session's selected built-in store is writable."""
        return self.user_profile_enabled if target == "user" else self.memory_enabled

    def reset_consolidation_failures(self) -> None:
        """Reset the per-turn consolidation-failure counter (call at turn start)."""
        self._consolidation_failures = 0

    def _consolidation_failure(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Count an at-capacity consolidation failure. Under the per-turn cap
        return ``response`` unchanged (it says how to retry); past it return a
        TERMINAL result so the model stops looping — a failed memory side effect
        must never block the turn's reply."""
        self._consolidation_failures += 1
        if self._consolidation_failures <= self._MAX_CONSOLIDATION_FAILURES_PER_TURN:
            return response
        return {"success": False, "done": True, "error": (
            f"Memory consolidation failed {self._consolidation_failures} times this turn. Stop retrying "
            "memory calls — leave memory unchanged for now and continue with your reply to the user. "
            "The fact can be saved in a later turn."
        )}

    def load_from_disk(self):
        """Load MEMORY.md / USER.md and capture the frozen system-prompt snapshot.
        Threat hits are replaced in the SNAPSHOT only by a ``[BLOCKED: …]``
        placeholder; live lists keep the raw text so the user can see and remove
        poisoned entries (dropping them silently would hide the attack).
        Scanning is deterministic from disk bytes, so the snapshot stays stable."""
        mem_dir = _memory_dir()
        mem_dir.mkdir(parents=True, exist_ok=True)
        # Deduplicate (order-preserving, first occurrence wins).
        self.memory_entries = list(dict.fromkeys(self._read_file(mem_dir / "MEMORY.md")))
        self.user_entries = list(dict.fromkeys(self._read_file(mem_dir / "USER.md")))
        self._system_prompt_snapshot = {
            target: self._render_block(target, self._sanitize_entries_for_snapshot(entries, filename))
            for target, entries, filename in (
                ("memory", self.memory_entries, "MEMORY.md"),
                ("user", self.user_entries, "USER.md"),
            )
        }

    @staticmethod
    def _sanitize_entries_for_snapshot(entries: List[str], filename: str) -> List[str]:
        """Return *entries* with any threat-matching entry replaced by a
        ``[BLOCKED: …]`` placeholder (strict scope, same as writes). Empty or
        already-blocked entries pass through unchanged."""
        from tools.threat_patterns import scan_for_threats

        sanitized: List[str] = []
        for entry in entries:
            findings = scan_for_threats(entry, scope="strict") if entry and not entry.startswith("[BLOCKED:") else None
            if not findings:
                sanitized.append(entry)
                continue
            logger.warning("Memory entry from %s blocked at load time: %s", filename, ", ".join(findings))
            sanitized.append(f"[BLOCKED: {filename} entry contained threat pattern(s): {', '.join(findings)}. "
                             f"Removed from system prompt; use memory(action=remove) to delete the original.]")
        return sanitized

    @staticmethod
    @contextmanager
    def _file_lock(path: Path):
        """Exclusive lock for read-modify-write safety, on a separate .lock file
        so the memory file itself can still be atomically replaced."""
        from tools import memory_tool as _mt  # fcntl/msvcrt live (and are patched) there

        fcntl, msvcrt = _mt.fcntl, _mt.msvcrt
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        if fcntl is None and msvcrt is None:
            yield
            return
        fd = open(lock_path, "a+", encoding="utf-8")
        try:
            if fcntl:
                fcntl.flock(fd, fcntl.LOCK_EX)
            else:
                fd.seek(0)
                msvcrt.locking(fd.fileno(), msvcrt.LK_LOCK, 1)
            yield
        finally:
            try:
                if fcntl:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                else:
                    fd.seek(0)
                    msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            fd.close()

    @staticmethod
    def _path_for(target: str) -> Path:
        return _memory_dir() / ("USER.md" if target == "user" else "MEMORY.md")

    def _reload_target(self, target: str, *, skip_drift: bool = False):
        """Re-read entries from disk (under file lock) before mutating.

        Returns ``None`` on a clean reload; the backup path (str) on external
        drift (caller must abort — flushing would discard un-roundtrippable
        content); or ``_READ_FAILED`` when the file exists but could not be
        read (caller MUST abort — rewriting from an assumed-empty view would
        wipe it; even append-only ``add`` rewrites the whole file). Drift check
        and parse use the SAME raw snapshot — a failed second read used to count
        as "no drift". *skip_drift* skips the round-trip check (``add``).
        """
        raw, read_ok = self._read_raw_checked(self._path_for(target))
        if not read_ok:
            return _READ_FAILED
        bak = None if skip_drift else self._detect_external_drift(target, raw)
        self._set_entries(target, list(dict.fromkeys(self._parse_entries(raw))))
        return bak

    def _reload_or_error(self, target: str, *, skip_drift: bool = False) -> Optional[Dict[str, Any]]:
        """Reload under lock; return the abort error dict, or None to proceed."""
        bak = self._reload_target(target, skip_drift=skip_drift)
        if bak is _READ_FAILED:
            return _read_failed_error(self._path_for(target))
        if bak:
            return _drift_error(self._path_for(target), bak)
        return None

    def save_to_disk(self, target: str):
        """Persist entries to the appropriate file. Called after every mutation."""
        _memory_dir().mkdir(parents=True, exist_ok=True)
        self._write_file(self._path_for(target), self._entries_for(target))

    def _entries_for(self, target: str) -> List[str]:
        return self.user_entries if target == "user" else self.memory_entries

    def _set_entries(self, target: str, entries: List[str]):
        if target == "user":
            self.user_entries = entries
        else:
            self.memory_entries = entries

    def _char_count(self, target: str) -> int:
        return len(ENTRY_DELIMITER.join(self._entries_for(target)))

    def _char_limit(self, target: str) -> int:
        return self.user_char_limit if target == "user" else self.memory_char_limit

    def _usage(self, target: str) -> str:
        return f"{self._char_count(target):,}/{self._char_limit(target):,}"

    def _failure_with_entries(self, target: str, message: str) -> Dict[str, Any]:
        """Consolidation failure that shows the live entries so the model can
        decide what to consolidate."""
        return self._consolidation_failure({"success": False, "error": message,
                                            "current_entries": self._entries_for(target), "usage": self._usage(target)})

    def _locate(self, target: str, old_text: str, verb: str):
        """Resolve *old_text* to a unique entry index, or an error dict."""
        entries = self._entries_for(target)
        idx, ambiguous = _find_unique_match(entries, old_text)
        if ambiguous:
            return None, {"success": False, "error": f"Multiple entries matched '{old_text}'. Be more specific.",
                          "matches": self._previews([e for e in entries if old_text in e])}
        if idx is None:
            return None, self._consolidation_failure({
                "success": False,
                "error": f"No entry matched '{old_text}'. Check current_entries below and retry with the exact text of the entry you want to {verb}.",
                "current_entries": entries,
            })
        return idx, None

    def _commit(self, target: str, entries: List[str], message: str) -> Dict[str, Any]:
        self._set_entries(target, entries)
        self.save_to_disk(target)
        return self._success_response(target, message)

    def add(self, target: str, content: str) -> Dict[str, Any]:
        """Append a new entry. Returns error if it would exceed the char limit."""
        content = content.strip()
        if not content:
            return {"success": False, "error": "Content cannot be empty."}
        scan_error = _scan_memory_content(content)
        if scan_error:
            return {"success": False, "error": scan_error}
        with self._file_lock(self._path_for(target)):
            # Append-only: skip the drift guard (appending never clobbers
            # un-roundtrippable content), but still refuse on a failed read —
            # add rewrites the WHOLE file from the parsed entries.
            err = self._reload_or_error(target, skip_drift=True)
            if err:
                return err
            entries = self._entries_for(target)
            limit = self._char_limit(target)
            if content in entries:
                return self._success_response(target, "Entry already exists (no duplicate added).")
            if len(ENTRY_DELIMITER.join(entries + [content])) > limit:
                return self._failure_with_entries(target, (
                    f"Memory at {self._char_count(target):,}/{limit:,} chars. Adding this entry "
                    f"({len(content)} chars) would exceed the limit. Consolidate now: use 'replace' to merge "
                    f"overlapping entries into shorter ones or 'remove' stale or less important entries (see "
                    f"current_entries below), then retry this add — all in this turn."
                ))
            entries.append(content)
            return self._commit(target, entries, "Entry added.")

    def replace(self, target: str, old_text: str, new_content: str) -> Dict[str, Any]:
        """Find entry containing old_text substring, replace it with new_content."""
        old_text = old_text.strip()
        new_content = new_content.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        if not new_content:
            return {"success": False, "error": "new_content cannot be empty. Use 'remove' to delete entries."}
        scan_error = _scan_memory_content(new_content)
        if scan_error:
            return {"success": False, "error": scan_error}
        with self._file_lock(self._path_for(target)):
            err = self._reload_or_error(target)
            if err:
                return err
            idx, err = self._locate(target, old_text, "replace")
            if err:
                return err
            entries = self._entries_for(target)
            limit = self._char_limit(target)
            new_total = len(ENTRY_DELIMITER.join(entries[:idx] + [new_content] + entries[idx + 1:]))
            if new_total > limit:
                return self._failure_with_entries(target, (
                    f"Replacement would put memory at {new_total:,}/{limit:,} chars. Shorten the new content, "
                    f"or 'remove' other stale or less important entries to make room (see current_entries "
                    f"below), then retry — all in this turn."
                ))
            entries[idx] = new_content
            return self._commit(target, entries, "Entry replaced.")

    def remove(self, target: str, old_text: str) -> Dict[str, Any]:
        """Remove the entry containing old_text substring."""
        old_text = old_text.strip()
        if not old_text:
            return {"success": False, "error": "old_text cannot be empty."}
        with self._file_lock(self._path_for(target)):
            err = self._reload_or_error(target)
            if err:
                return err
            idx, err = self._locate(target, old_text, "remove")
            if err:
                return err
            entries = self._entries_for(target)
            entries.pop(idx)
            return self._commit(target, entries, "Entry removed.")

    # -- Batch --

    @staticmethod
    def _apply_batch_op(working: List[str], act: str, content: str, old_text: str, pos: str) -> Optional[str]:
        """Apply one batch op to *working* in place; return an error message or None."""
        if act == "add":
            if not content:
                return f"{pos}: content is required."
            if content not in working:  # idempotent -- skip duplicate, don't fail the batch
                working.append(content)
            return None
        if act not in ("replace", "remove"):
            return f"{pos}: unknown action. Use add, replace, or remove."
        if not old_text:
            return f"{pos}: old_text is required."
        if act == "replace" and not content:
            return f"{pos}: content is required (use action='remove' to delete)."
        idx, ambiguous = _find_unique_match(working, old_text)
        if ambiguous:
            return f"{pos}: '{old_text}' matched multiple distinct entries -- be more specific."
        if idx is None:
            return f"{pos}: no entry matched '{old_text}'."
        if act == "replace":
            working[idx] = content
        else:
            working.pop(idx)
        return None

    def apply_batch(self, target: str, operations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Apply a sequence of add/replace/remove ops to one target atomically.

        Ops are validated and applied against the FINAL budget only — so a single
        call can free space (remove/replace) and add new entries without the
        multi-turn consolidate-then-retry dance. All-or-nothing: if any op is
        malformed, doesn't match, or the net result exceeds the char limit,
        NOTHING is written and the first failure plus live state is returned.
        """
        if not operations:
            return {"success": False, "error": "operations list is empty."}

        # Scan every add/replace content BEFORE touching disk -- one poisoned
        # op rejects the whole batch.
        for i, op in enumerate(operations):
            op = op or {}
            scan_error = op.get("action") in {"add", "replace"} and op.get("content") and _scan_memory_content(op["content"])
            if scan_error:
                return {"success": False, "error": f"Operation {i + 1}: {scan_error}"}

        with self._file_lock(self._path_for(target)):
            err = self._reload_or_error(target)
            if err:
                return err
            # Work on a copy; only commit if the whole batch validates.
            working: List[str] = list(self._entries_for(target))
            limit = self._char_limit(target)
            for i, op in enumerate(operations):
                op = op or {}
                act = op.get("action")
                content = (op.get("content") or op.get("new_text") or "").strip()
                old_text = (op.get("old_text") or "").strip()
                pos = f"Operation {i + 1} ({act or 'unknown'})"
                msg = self._apply_batch_op(working, act, content, old_text, pos)
                if msg:
                    return self._batch_error(target, msg)
            # Budget check against the FINAL state only.
            new_total = len(ENTRY_DELIMITER.join(working))
            if new_total > limit:
                return self._failure_with_entries(target, (
                    f"After applying all {len(operations)} operations, memory would be at "
                    f"{new_total:,}/{limit:,} chars -- over the limit. Remove or shorten more "
                    f"entries in the same batch (see current_entries below), then retry."
                ))
            return self._commit(target, working, f"Applied {len(operations)} operation(s).")

    def _batch_error(self, target: str, message: str) -> Dict[str, Any]:
        """Build a batch-abort error that reports live (uncommitted) state."""
        return self._failure_with_entries(
            target, message + " No operations were applied (batch is all-or-nothing)."
        )

    def format_for_system_prompt(self, target: str) -> Optional[str]:
        """Return the frozen load-time snapshot for system-prompt injection (NOT
        live state — mid-session writes don't affect it, preserving the prefix
        cache). None if the snapshot is empty."""
        return self._system_prompt_snapshot.get(target, "") or None

    # -- Internal helpers --

    @staticmethod
    def _previews(entries: List[str], width: int = 80) -> List[str]:
        """Truncated one-line previews of entries for error feedback."""
        return [e[:width] + ("..." if len(e) > width else "") for e in entries]

    def _usage_pct(self, target: str, current: int) -> str:
        """``"<pct>% — <current>/<limit> chars"`` for the given target."""
        limit = self._char_limit(target)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0
        return f"{pct}% — {current:,}/{limit:,} chars"

    def _success_response(self, target: str, message: str = None) -> Dict[str, Any]:
        # A successful write means the consolidation loop made progress, so the
        # per-turn failure budget resets (the cap counts consecutive failures).
        self._consolidation_failures = 0
        # Intentionally TERMINAL and without the entries list: echoing entries
        # invites the model to "find more to fix" and re-issue the same ops.
        # Entries are only shown on error/over-budget paths.
        resp = {"success": True, "done": True, "target": target,
                "usage": self._usage_pct(target, self._char_count(target)),
                "entry_count": len(self._entries_for(target))}
        if message:
            resp["message"] = message
        resp["note"] = "Write saved. This update is complete — do not repeat it."
        return resp

    def _render_block(self, target: str, entries: List[str]) -> str:
        """Render a system prompt block with header and usage indicator."""
        if not entries:
            return ""
        content = ENTRY_DELIMITER.join(entries)
        title = MEMORY_BLOCK_HEADERS["user" if target == "user" else "memory"]
        separator = "═" * 46
        return f"{separator}\n{title} [{self._usage_pct(target, len(content))}]\n{separator}\n{content}"

    @staticmethod
    def _read_raw_checked(path: Path) -> Tuple[str, bool]:
        """Read raw text as ``(raw, read_ok)``. ``read_ok`` is False ONLY when the
        file EXISTS but can't be read (absent file → ``("", True)``). Invalid
        UTF-8 counts as unreadable; decoding stays STRICT because
        ``errors="replace"`` would hand callers a lossy view that a save then
        persists over the real bytes. ``utf-8-sig`` strips a Notepad BOM that
        otherwise glues U+FEFF onto the first entry forever."""
        if not path.exists():
            return "", True
        try:
            return path.read_text(encoding="utf-8-sig"), True
        except (OSError, UnicodeDecodeError):
            return "", False

    @staticmethod
    def _parse_entries(raw: str) -> List[str]:
        """Split raw memory-file text into stripped, non-empty entries. Splits on
        the full ENTRY_DELIMITER so a bare "§" inside an entry is preserved."""
        return [e for e in (x.strip() for x in raw.split(ENTRY_DELIMITER)) if e]

    @staticmethod
    def _read_entries_checked(path: Path) -> Tuple[List[str], bool]:
        """Read + parse as ``(entries, read_ok)`` — see ``_read_raw_checked``."""
        raw, read_ok = MemoryStore._read_raw_checked(path)
        return MemoryStore._parse_entries(raw), read_ok

    @staticmethod
    def _read_file(path: Path) -> List[str]:
        """Read a memory file into entries (empty list on any error). Only for
        read-only callers (``load_from_disk``, learning_mutations); mutation
        paths must use ``_read_raw_checked`` so they can refuse to overwrite an
        unreadable file."""
        return MemoryStore._read_entries_checked(path)[0]

    def _detect_external_drift(self, target: str, raw: str) -> Optional[str]:
        """Backup-path string if *raw* (the caller's checked-read snapshot) shows
        external drift, else None. Signals: round-trip mismatch, or one parsed
        entry exceeding the whole-file char limit (no tool-written entry can —
        an external writer appended free-form content a flush would truncate).
        The file is snapshotted to ``.bak.<ts>`` so the operator can recover it."""
        if not raw.strip():
            return None
        parsed = self._parse_entries(raw)
        if raw.strip() == ENTRY_DELIMITER.join(parsed) and max(map(len, parsed), default=0) <= self._char_limit(target):
            return None
        path = self._path_for(target)
        bak_path = path.with_suffix(path.suffix + f".bak.{int(time.time())}")
        try:
            bak_path.write_text(raw, encoding="utf-8")
        except OSError:
            return str(bak_path) + " (BACKUP FAILED — file unchanged on disk)"
        return str(bak_path)

    @staticmethod
    def _write_file(path: Path, entries: List[str]):
        """Atomic temp-file + rename: readers see the old or the new complete
        file, never a truncated one."""
        try:
            atomic_write_text(path, ENTRY_DELIMITER.join(entries), tmp_prefix=".mem_")
        except OSError as e:
            raise RuntimeError(f"Failed to write memory file {path}: {e}")
