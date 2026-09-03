"""MemoryStore — bounded, file-backed curated memory (MEMORY.md / USER.md).
Entries are joined by ``ENTRY_DELIMITER``; budgets are in chars (model-independent).
Module state that tests monkeypatch (``get_memory_dir``, ``fcntl``/``msvcrt``) stays
in ``tools.memory_tool`` and is read lazily."""

import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from utils import atomic_write_text
from tools.threat_patterns import first_threat_message as _first_threat_message

logger = logging.getLogger("tools.memory_tool")

# Block header prefixes rendered by _render_block; agent/conversation_compression.py
# matches them to detect a leftover block for an emptied target — keep in lockstep.
MEMORY_BLOCK_HEADERS = {
    "memory": "MEMORY (your personal notes)", "user": "USER PROFILE (who the user is)"}

ENTRY_DELIMITER = "\n§\n"

def _memory_dir() -> Path:
    from tools import memory_tool

    return memory_tool.get_memory_dir()


def _scan_memory_content(content: str) -> Optional[str]:
    """Error string if *content* matches injection/exfil patterns. Strict scope:
    memory enters the system prompt, so a poisoned entry persists across sessions."""
    return _first_threat_message(content, scope="strict")


def _drift_error(path: "Path", bak_path: str) -> Dict[str, Any]:
    """External drift: the file wouldn't round-trip, so flushing would discard content."""
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
        )}


def _read_failed_error(path: "Path") -> Dict[str, Any]:
    """Existing-but-unreadable file: saving from an assumed-empty view would wipe it."""
    return {"success": False, "error": (
        f"Refusing to write {path.name}: the file exists on disk but could not be read right now "
        f"(temporarily locked by another program, a permission change, invalid/corrupt text encoding, "
        f"or a filesystem error). Treating an unreadable file as empty and saving would wipe existing "
        f"memory, so the write is refused. Nothing was changed — retry in a moment.")}


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

    # Failed consolidation attempts (overflow / zero-match) allowed per turn before
    # a TERMINAL "save skipped" result, so a fragile replace/add can't loop the turn
    # to budget exhaustion and suppress the user's reply.
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
        self._consolidation_failures = 0  # per turn; reset by reset_consolidation_failures()

    def target_enabled(self, target: str) -> bool:
        """Return whether this session's selected built-in store is writable."""
        return self.user_profile_enabled if target == "user" else self.memory_enabled

    def reset_consolidation_failures(self) -> None:
        """Reset the per-turn consolidation-failure counter (call at turn start)."""
        self._consolidation_failures = 0

    def _consolidation_failure(self, response: Dict[str, Any]) -> Dict[str, Any]:
        """Count a consolidation failure: under the per-turn cap return ``response``
        (it says how to retry); past it a TERMINAL result so the model stops looping."""
        self._consolidation_failures += 1
        if self._consolidation_failures <= self._MAX_CONSOLIDATION_FAILURES_PER_TURN:
            return response
        return {"success": False, "done": True, "error": (
            f"Memory consolidation failed {self._consolidation_failures} times this turn. Stop retrying "
            "memory calls — leave memory unchanged for now and continue with your reply to the user. "
            "The fact can be saved in a later turn.")}

    def load_from_disk(self):
        """Load MEMORY.md / USER.md and capture the frozen system-prompt snapshot.
        Threat hits are replaced by a ``[BLOCKED: …]`` placeholder in the SNAPSHOT only;
        live lists keep the raw text so the user can see and remove poisoned entries
        (dropping them silently would hide the attack)."""
        mem_dir = _memory_dir()
        mem_dir.mkdir(parents=True, exist_ok=True)
        # Deduplicate (order-preserving, first occurrence wins).
        self.memory_entries = list(dict.fromkeys(self._read_file(mem_dir / "MEMORY.md")))
        self.user_entries = list(dict.fromkeys(self._read_file(mem_dir / "USER.md")))
        self._system_prompt_snapshot = {
            target: self._render_block(target, self._sanitize_entries_for_snapshot(entries, filename))
            for target, entries, filename in (
                ("memory", self.memory_entries, "MEMORY.md"), ("user", self.user_entries, "USER.md"))
        }

    @staticmethod
    def _sanitize_entries_for_snapshot(entries: List[str], filename: str) -> List[str]:
        """*entries* with threat matches replaced by a ``[BLOCKED: …]`` placeholder
        (strict scope, same as writes); empty / already-blocked entries pass through."""
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
        """Exclusive lock on a separate .lock file so the memory file itself can
        still be atomically replaced."""
        from tools import memory_tool as _mt  # fcntl/msvcrt live (and are patched) there

        fcntl, msvcrt = _mt.fcntl, _mt.msvcrt
        lock_path = path.with_suffix(path.suffix + ".lock")
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        if fcntl is None and msvcrt is None:
            yield
            return
        fd = open(lock_path, "a+", encoding="utf-8")

        def _flock(unlock: bool):
            if fcntl:
                fcntl.flock(fd, fcntl.LOCK_UN if unlock else fcntl.LOCK_EX)
            else:
                fd.seek(0)
                msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK if unlock else msvcrt.LK_LOCK, 1)

        try:
            _flock(False)
            yield
        finally:
            try:
                _flock(True)
            except OSError:
                pass
            fd.close()

    @staticmethod
    def _path_for(target: str) -> Path:
        return _memory_dir() / ("USER.md" if target == "user" else "MEMORY.md")

    def _reload_or_error(self, target: str, *, skip_drift: bool = False) -> Optional[Dict[str, Any]]:
        """Re-read entries from disk (under lock) before mutating; return the abort
        error dict or None. Aborts on external drift (flushing would discard
        un-roundtrippable content) and on an existing-but-unreadable file (even
        append-only ``add`` rewrites the whole file). Drift check and parse use the
        SAME raw snapshot — a failed second read used to count as "no drift"."""
        path = self._path_for(target)
        raw, read_ok = self._read_raw_checked(path)
        if not read_ok:
            return _read_failed_error(path)
        bak = None if skip_drift else self._detect_external_drift(target, raw)
        self._set_entries(target, list(dict.fromkeys(self._parse_entries(raw))))
        return _drift_error(path, bak) if bak else None

    def save_to_disk(self, target: str):
        """Persist entries to the appropriate file. Called after every mutation."""
        _memory_dir().mkdir(parents=True, exist_ok=True)
        self._write_file(self._path_for(target), self._entries_for(target))

    def _entries_for(self, target: str) -> List[str]:
        return self.user_entries if target == "user" else self.memory_entries

    def _set_entries(self, target: str, entries: List[str]):
        setattr(self, "user_entries" if target == "user" else "memory_entries", entries)

    def _char_count(self, target: str) -> int:
        return len(ENTRY_DELIMITER.join(self._entries_for(target)))

    def _char_limit(self, target: str) -> int:
        return self.user_char_limit if target == "user" else self.memory_char_limit

    def _usage(self, target: str) -> str:
        return f"{self._char_count(target):,}/{self._char_limit(target):,}"

    def _usage_pct(self, target: str, current: int) -> str:
        """``"<pct>% — <current>/<limit> chars"`` for the given target."""
        limit = self._char_limit(target)
        pct = min(100, int((current / limit) * 100)) if limit > 0 else 0
        return f"{pct}% — {current:,}/{limit:,} chars"

    def _failure_with_entries(self, target: str, message: str) -> Dict[str, Any]:
        """Consolidation failure carrying the live entries so the model can consolidate."""
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
                "current_entries": entries})
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
            # Append-only: skip the drift guard (appending never clobbers foreign
            # content) but still refuse a failed read — add rewrites the WHOLE file.
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
                    f"current_entries below), then retry this add — all in this turn."))
            entries.append(content)
            return self._commit(target, entries, "Entry added.")

    def replace(self, target: str, old_text: str, new_content: str) -> Dict[str, Any]:
        """Find entry containing old_text substring, replace it with new_content."""
        new_content = new_content.strip()
        if not old_text.strip():
            return {"success": False, "error": "old_text cannot be empty."}
        if not new_content:
            return {"success": False, "error": "new_content cannot be empty. Use 'remove' to delete entries."}
        scan_error = _scan_memory_content(new_content)
        if scan_error:
            return {"success": False, "error": scan_error}
        return self._edit(target, old_text.strip(), new_content)

    def remove(self, target: str, old_text: str) -> Dict[str, Any]:
        """Remove the entry containing old_text substring."""
        if not old_text.strip():
            return {"success": False, "error": "old_text cannot be empty."}
        return self._edit(target, old_text.strip(), None)

    def _edit(self, target: str, old_text: str, new_content: Optional[str]) -> Dict[str, Any]:
        """Locked replace (``new_content`` set) or remove (None) of the unique entry matching *old_text*."""
        with self._file_lock(self._path_for(target)):
            err = self._reload_or_error(target)
            if err:
                return err
            idx, err = self._locate(target, old_text, "replace" if new_content else "remove")
            if err:
                return err
            entries = self._entries_for(target)
            if new_content is None:
                entries.pop(idx)
                return self._commit(target, entries, "Entry removed.")
            limit = self._char_limit(target)
            new_total = len(ENTRY_DELIMITER.join(entries[:idx] + [new_content] + entries[idx + 1:]))
            if new_total > limit:
                return self._failure_with_entries(target, (
                    f"Replacement would put memory at {new_total:,}/{limit:,} chars. Shorten the new content, "
                    f"or 'remove' other stale or less important entries to make room (see current_entries "
                    f"below), then retry — all in this turn."))
            entries[idx] = new_content
            return self._commit(target, entries, "Entry replaced.")

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
        """Apply add/replace/remove ops to one target atomically against the FINAL
        budget, so one call can free space and add entries. All-or-nothing: any
        malformed / unmatched op or an over-limit result writes NOTHING and returns
        the first failure plus live state."""
        if not operations:
            return {"success": False, "error": "operations list is empty."}

        # Scan every add/replace content BEFORE touching disk -- one poisoned op rejects the batch.
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
                    return self._failure_with_entries(
                        target, msg + " No operations were applied (batch is all-or-nothing).")
            # Budget check against the FINAL state only.
            new_total = len(ENTRY_DELIMITER.join(working))
            if new_total > limit:
                return self._failure_with_entries(target, (
                    f"After applying all {len(operations)} operations, memory would be at "
                    f"{new_total:,}/{limit:,} chars -- over the limit. Remove or shorten more "
                    f"entries in the same batch (see current_entries below), then retry."))
            return self._commit(target, working, f"Applied {len(operations)} operation(s).")

    def format_for_system_prompt(self, target: str) -> Optional[str]:
        """Frozen load-time snapshot for the system prompt (NOT live state — mid-session
        writes don't touch it, preserving the prefix cache); None if empty."""
        return self._system_prompt_snapshot.get(target, "") or None

    @staticmethod
    def _previews(entries: List[str], width: int = 80) -> List[str]:
        """Truncated one-line previews of entries for error feedback."""
        return [e[:width] + ("..." if len(e) > width else "") for e in entries]

    def _success_response(self, target: str, message: str = None) -> Dict[str, Any]:
        # A successful write is progress: reset the per-turn (consecutive) failure budget.
        self._consolidation_failures = 0
        # TERMINAL and WITHOUT the entries list: echoing entries invites the model to
        # "find more to fix" and re-issue the same ops. Entries only appear on errors.
        return {"success": True, "done": True, "target": target,
                "usage": self._usage_pct(target, self._char_count(target)),
                "entry_count": len(self._entries_for(target)), **({"message": message} if message else {}),
                "note": "Write saved. This update is complete — do not repeat it."}

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
        """``(raw, read_ok)``; ``read_ok`` is False ONLY when the file EXISTS but can't
        be read (absent → ``("", True)``). Decoding stays STRICT: ``errors="replace"``
        would hand callers a lossy view that a save then persists. ``utf-8-sig`` strips
        a Notepad BOM that otherwise glues U+FEFF onto the first entry forever."""
        if not path.exists():
            return "", True
        try:
            return path.read_text(encoding="utf-8-sig"), True
        except (OSError, UnicodeDecodeError):
            return "", False

    @staticmethod
    def _parse_entries(raw: str) -> List[str]:
        """Stripped, non-empty entries; splits on the FULL delimiter so a bare "§" survives."""
        return [e for e in (x.strip() for x in raw.split(ENTRY_DELIMITER)) if e]

    @staticmethod
    def _read_file(path: Path) -> List[str]:
        """Entries of a memory file ([] on any error). Read-only callers only
        (``load_from_disk``, learning_mutations); mutation paths must use
        ``_read_raw_checked`` so they can refuse to overwrite an unreadable file."""
        return MemoryStore._parse_entries(MemoryStore._read_raw_checked(path)[0])

    def _detect_external_drift(self, target: str, raw: str) -> Optional[str]:
        """Backup path if *raw* shows external drift, else None. Signals: round-trip
        mismatch, or one entry over the whole-file limit (no tool-written entry can —
        an external writer appended free-form text). Snapshots to ``.bak.<ts>``."""
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
        """Atomic temp-file + rename: readers never see a truncated file."""
        try:
            atomic_write_text(path, ENTRY_DELIMITER.join(entries), tmp_prefix=".mem_")
        except OSError as e:
            raise RuntimeError(f"Failed to write memory file {path}: {e}")
