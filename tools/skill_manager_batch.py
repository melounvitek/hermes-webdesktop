"""Atomic multi-op batch path for ``skill_manage`` (extracted from skill_manager_tool).

``skill_manage``/``_find_skill``/``_skill_gate_bypass`` are reached lazily
through ``tools.skill_manager_tool`` so the origin module owns all state.
"""

import json
import logging
import posixpath
import shutil
import tempfile
from pathlib import Path

logger = logging.getLogger("tools.skill_manager_tool")

_BATCH_OP_ACTIONS = {"create", "patch", "write_file", "remove_file"}
_BATCH_MAX_OPS = 20


def _norm_target(op) -> str:
    fp = (op.get("file_path") or "").strip()
    if not fp:
        return "SKILL.md"
    return posixpath.normpath(fp.lstrip("/"))


def _validate_batch_ops(operations, default_name, tool_error):
    """Shape checks with no side effects. Returns (names, None) or (None, error_json)."""
    from tools.skill_manager_guards import _background_review_preflight

    names = []
    for i, op in enumerate(operations):
        if not isinstance(op, dict) or not op.get("action"):
            return None, tool_error(f"operations[{i}] needs an 'action'.", success=False)
        act = op["action"]
        if act not in _BATCH_OP_ACTIONS:
            return None, tool_error(
                f"operations[{i}]: unknown action '{act}'. "
                f"Batchable: {', '.join(sorted(_BATCH_OP_ACTIONS))}; "
                "delete must be sole.",
                success=False,
            )
        nm = op.get("name") or default_name
        if not nm:
            return None, tool_error(f"operations[{i}] needs a 'name' (the skill it targets).", success=False)
        names.append(nm)
        if act == "create" and nm in names[:-1]:
            return None, tool_error(
                f"operations[{i}]: create for '{nm}' must precede that "
                "skill's other ops.",
                success=False,
            )
        preflight = _background_review_preflight(act, nm)
        if preflight is not None:
            return None, json.dumps(preflight, ensure_ascii=False)

    # Intra-batch clobber guard: sequential last-wins would SILENTLY discard an
    # earlier op's work. A DESTRUCTIVE op (create/write_file/remove_file/full
    # SKILL.md rewrite) on a file an earlier op touched is rejected; additive
    # patches are always legal, so patch chains and write-then-patch stay
    # allowed. Paths are normalized so spelling variants can't slip past.
    touched_files = set()
    for i, op in enumerate(operations):
        act = op["action"]
        nm = names[i]
        # create and full-rewrite patch (content) always hit SKILL.md.
        full_rewrite = act == "patch" and bool(op.get("content"))
        target = "SKILL.md" if (act == "create" or full_rewrite) else _norm_target(op)
        key = (nm, target)
        destructive = act in ("create", "write_file", "remove_file") or full_rewrite
        if destructive and key in touched_files:
            return None, tool_error(
                f"operations[{i}]: {act} on '{target}' of skill '{nm}' — an "
                "earlier op in this batch already touched that file, and this "
                "op would silently discard its work. One destructive op "
                "(write_file/remove_file/full rewrite) per file per batch; "
                "put it first, or fold the change in. Patch chains are fine.",
                success=False,
            )
        touched_files.add(key)
    return names, None


def _stage_batch_if_gated(operations, names):
    """Approval gate for the WHOLE batch as one pending write."""
    from tools.skill_manager_tool import _run_write_gate

    def _staging(wa):
        acts = ", ".join(op["action"] for op in operations)
        skills = ", ".join(sorted(set(names)))
        gist = f"batch({len(operations)} ops: {acts}) on {skills}"
        return {"action": "batch", "operations": operations}, gist

    return _run_write_gate(_staging)


def _snapshot_skills(names, snap_root, find_skill):
    """Copy every touched skill aside. Returns (snapshots, None) or (None, error_text)."""
    snapshots = {}  # skill name -> (pre_dir or None, snapshot_dir or None)
    for nm in dict.fromkeys(names):  # ordered unique
        pre = find_skill(nm)
        pre_dir = Path(pre["path"]) if pre else None
        snap = None
        if pre_dir is not None and pre_dir.is_dir():
            snap = snap_root / nm
            try:
                shutil.copytree(pre_dir, snap)
            except Exception as exc:  # noqa: BLE001 — no snapshot, no atomicity
                return None, f"Could not snapshot '{nm}' for atomic batch: {exc}"
        snapshots[nm] = (pre_dir, snap)
    return snapshots, None


def _restore_snapshot(pre_dir, snap, post_dir) -> None:
    if snap is not None:
        if post_dir is not None and post_dir.is_dir():
            # Never destroy the only other copy before the restore lands: move
            # the broken state aside and delete it only after the snapshot is
            # back, so a failed copytree (disk full, locked file) can't turn
            # into total skill loss.
            aside = post_dir.with_name(post_dir.name + ".rollback-broken")
            shutil.rmtree(aside, ignore_errors=True)
            post_dir.rename(aside)
            try:
                shutil.copytree(snap, pre_dir)
            except Exception:
                # Restore failed: put the broken (half-applied) state back
                # rather than leaving nothing.
                shutil.rmtree(pre_dir, ignore_errors=True)
                aside.rename(pre_dir)
                raise
            shutil.rmtree(aside, ignore_errors=True)
        else:
            shutil.copytree(snap, pre_dir)
    elif post_dir is not None and post_dir.is_dir():
        # Batch created this skill: remove the partial result.
        shutil.rmtree(post_dir)


def _rollback(snapshots, find_skill):
    """Restore every snapshot. Returns (note, failed)."""
    notes = []
    for nm, (pre_dir, snap) in snapshots.items():
        try:
            post = find_skill(nm)
            _restore_snapshot(pre_dir, snap, Path(post["path"]) if post else None)
        except Exception as exc:  # noqa: BLE001
            notes.append(
                f"ROLLBACK FAILED for '{nm}' ({exc}); snapshot preserved at '{snap}'"
                if snap is not None
                else f"ROLLBACK FAILED for '{nm}' ({exc})"
            )
    return ("; ".join(notes) if notes else "all touched skills rolled back"), bool(notes)


def _skill_manage_batch(
    operations,
    default_name: str = None,
    task_id: str = None,
    session_id: str = None,
) -> str:
    """Apply a sequence of operations atomically (memory-tool pattern).

    Each op carries its own ``name`` and ``action``. Every touched skill is
    snapshotted before any op runs; any failure rolls ALL touched skills back
    (skills the batch created are removed).

    Rules: ``delete`` only as the SOLE op (its recoverable-archive path doesn't
    compose with rollback) and is routed to the single-op handler, preserving
    absorbed_into/archive semantics; ``create`` must precede that skill's other
    ops; the same-file clobber guard rejects silently-lost work.
    ``default_name`` is the legacy top-level ``name`` fallback (staged replay).
    """
    from tools import skill_manager_tool as _smt
    from tools.registry import tool_error

    if not isinstance(operations, list) or not operations:
        return tool_error("operations must be a non-empty array.", success=False)
    if len(operations) > _BATCH_MAX_OPS:
        return tool_error(f"operations is capped at {_BATCH_MAX_OPS} ops per call.", success=False)
    if any(isinstance(op, dict) and op.get("action") == "delete" for op in operations):
        if len(operations) != 1:
            return tool_error(
                "delete must be the SOLE op in its call — it doesn't "
                "compose with other ops' rollback.",
                success=False,
            )
        op = operations[0]
        nm = op.get("name") or default_name
        if not nm:
            return tool_error("operations[0] (delete) needs a 'name'.", success=False)
        return _smt.skill_manage(
            action="delete",
            name=nm,
            absorbed_into=op.get("absorbed_into"),
            task_id=task_id,
            session_id=session_id,
        )

    names, err = _validate_batch_ops(operations, default_name, tool_error)
    if err is not None:
        return err

    if not _smt._skill_gate_bypass.get():
        staged = _stage_batch_if_gated(operations, names)
        if staged is not None:
            return staged

    snap_root = Path(tempfile.mkdtemp(prefix="skill_batch_"))
    snapshots, snap_err = _snapshot_skills(names, snap_root, _smt._find_skill)
    if snap_err is not None:
        shutil.rmtree(snap_root, ignore_errors=True)
        return tool_error(snap_err, success=False)

    # Execute through the single-op path with the gate bypassed (the batch
    # already cleared/staged it); ledger + telemetry fire per-op.
    results = []
    rollback_failed = False
    token = _smt._skill_gate_bypass.set(True)
    try:
        for i, op in enumerate(operations):
            raw = _smt._skill_manage_from(
                {**op, "name": names[i]}, task_id=task_id, session_id=session_id,
            )
            try:
                parsed = json.loads(raw)
            except Exception:  # noqa: BLE001
                parsed = {"success": False, "error": "unparseable op result"}
            if not parsed.get("success"):
                note, rollback_failed = _rollback(snapshots, _smt._find_skill)
                fail = {
                    "success": False,
                    "error": (
                        f"operations[{i}] ({op['action']} on '{names[i]}') failed: "
                        f"{parsed.get('error', 'unknown error')} — batch aborted, {note}."
                    ),
                    "failed_index": i,
                    "completed_before_failure": i,
                }
                # Carry the failing op's teaching payload (patch's file_preview /
                # fuzzy-match hints) through — without it the model recovers blind.
                for k, v in parsed.items():
                    if k not in ("success", "error") and v is not None:
                        fail.setdefault(k, v)
                return json.dumps(fail, ensure_ascii=False)
            results.append({"name": names[i], "action": op["action"],
                            "file_path": op.get("file_path"),
                            "success": True})
    finally:
        _smt._skill_gate_bypass.reset(token)
        if rollback_failed:
            # Keep the snapshots so the operator can still recover by hand.
            logger.warning(
                "skill_manage batch rollback failed, snapshots kept at %s",
                snap_root,
            )
        else:
            shutil.rmtree(snap_root, ignore_errors=True)

    return json.dumps(
        {"success": True, "operations_applied": len(results),
         "results": results},
        ensure_ascii=False,
    )
