"""CLI for the Hermes Kanban board — ``hermes kanban …`` subcommand.

All DB work is delegated to ``kanban_db``; this module adds argparse
construction (``build_parser``), dispatch (``kanban_command``), text/``--json``
output, and ``run_slash`` for ``/kanban …`` from the CLI and gateway.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import shlex
import sys
import time
from pathlib import Path
from typing import Optional

from hermes_cli import kanban_db as kb
from hermes_cli import kanban_swarm as ks
from hermes_cli.kanban_output import (
    _ATTACHMENT_FIELDS, _RUNS_RUN_FIELDS, _SHOW_RUN_FIELDS, _bulk_apply, _err,
    _fmt_counts, _fmt_task_line, _fmt_ts, _json_out, _obj_dict, _print_json,
    _task_to_dict,
)
from hermes_cli.kanban_boards import _dispatch_boards
from hermes_cli.kanban_parser import build_parser  # noqa: F401  (re-exported: hermes_cli.main, run_slash)


# ---------------------------------------------------------------------------
# Flag parsing helpers
# ---------------------------------------------------------------------------

def _none_profile(value: str) -> Optional[str]:
    """``none`` / ``-`` / ``null`` mean "unassign"."""
    return None if value.lower() in {"none", "-", "null"} else value


def _parse_metadata_flag(raw: Optional[str]) -> tuple[Optional[dict], int]:
    """Parse ``--metadata`` JSON; returns ``(dict|None, rc)`` with rc=2 on error."""
    if not raw:
        return None, 0
    try:
        metadata = json.loads(raw)
        if not isinstance(metadata, dict):
            raise ValueError("must be a JSON object")
    except (ValueError, json.JSONDecodeError) as exc:
        return None, _err(f"kanban: --metadata: {exc}", 2)
    return metadata, 0


def _run_state_kwargs(args: argparse.Namespace, cmd: str) -> tuple[Optional[dict[str, str]], int]:
    """``--state-type``/``--state-name`` must be given together: ``(kwargs, 0)`` or ``(None, 2)``."""
    st = getattr(args, "state_type", None)
    sn = getattr(args, "state_name", None)
    if (st is None) != (sn is None):
        return None, _err(f"kanban {cmd}: pass both --state-type and --state-name, or omit both", 2)
    return ({} if st is None else {"state_type": st, "state_name": sn}), 0


def _parse_workspace_flag(value: str) -> tuple[str, Optional[str]]:
    """Parse ``--workspace`` into ``(kind, path|None)``.

    Accepts: ``scratch``, ``worktree``, ``worktree:<path>``, ``dir:<path>``.
    """
    if not value:
        return ("scratch", None)
    v = value.strip()
    if v in {"scratch", "worktree"}:
        return (v, None)
    for prefix, kind in (("dir:", "dir"), ("worktree:", "worktree")):
        if not v.startswith(prefix):
            continue
        path = v[len(prefix):].strip()
        if not path:
            raise argparse.ArgumentTypeError(
                f"--workspace {prefix} requires a path after the colon"
            )
        return (kind, os.path.expanduser(path))
    raise argparse.ArgumentTypeError(
        f"unknown --workspace value {value!r}: use scratch, worktree, "
        "worktree:<path>, or dir:<path>"
    )


def _parse_branch_flag(value: Optional[str]) -> Optional[str]:
    """Normalize an optional branch name from ``kanban create --branch``."""
    if value is None:
        return None
    branch = value.strip()
    if not branch:
        raise argparse.ArgumentTypeError("--branch requires a non-empty name")
    if branch.startswith("-"):
        raise argparse.ArgumentTypeError("--branch must not start with '-'")
    if any(ch.isspace() for ch in branch):
        raise argparse.ArgumentTypeError("--branch must not contain whitespace")
    return branch


def _check_dispatcher_presence(
    hermes_home: Optional[Path] = None,
) -> tuple[bool, str]:
    """Return ``(running, message)`` for the "will anything dispatch this?" warning.

    ``running=True`` when a gateway is alive for this HERMES_HOME with
    ``kanban.dispatch_in_gateway`` on (message is a status line); otherwise
    ``False`` with human guidance. Fails OPEN — import/probe/config errors
    return ``(True, "")`` — since a missed warning beats crying wolf.

    ``hermes_home`` scopes the probe to a profile's directory: the dashboard
    backend may run under a different HERMES_HOME than the profile it serves,
    which otherwise misreports a healthy gateway as absent. CLI callers pass
    ``None``.
    """
    try:
        from gateway.status import resolve_gateway_liveness  # type: ignore
    except Exception:
        return (True, "")  # can't probe — silent
    try:
        # Same ladder the dashboard status endpoints use, so PID-file-less or
        # cross-container gateways aren't misreported. use_cache=False: this
        # one-shot probe must see the gateway's state right now.
        liveness = resolve_gateway_liveness(
            profile_dir=hermes_home, use_cache=False
        )
    except Exception:
        return (True, "")  # probe errored — silent
    if liveness.probe_error:
        # The resolver swallows per-rung failures; "can't tell" != "no gateway".
        return (True, "")
    pid = liveness.pid

    # Even if the gateway is up, dispatch_in_gateway may be off (can't tell -> assume default).
    dispatch_on = bool(_kanban_config().get("dispatch_in_gateway", True))

    if pid and dispatch_on:
        return (True, f"gateway pid={pid}, dispatch enabled")
    if pid and not dispatch_on:
        return (
            False,
            "Gateway is running but kanban.dispatch_in_gateway=false in "
            "config.yaml — the task will sit in 'ready' until you flip it "
            "back on and restart the gateway, OR run the legacy "
            "standalone daemon (`hermes kanban daemon --force`)."
        )
    return (
        False,
        "No gateway is running — the task will sit in 'ready' until you "
        "start it. Run:\n"
        "    hermes gateway start\n"
        "The gateway hosts an embedded dispatcher (tick interval 60s by "
        "default); your task will be picked up on the next tick after "
        "the gateway comes up."
    )


# ---------------------------------------------------------------------------
# Command dispatch
# ---------------------------------------------------------------------------

def kanban_command(args: argparse.Namespace) -> int:
    """Entry point from ``hermes kanban …``; returns a shell-style exit code."""
    action = getattr(args, "kanban_action", None)
    if not action:
        # No subaction given: print help via the stored parser reference.
        parser = getattr(args, "_kanban_parser", None)
        if parser is not None:
            parser.print_help()
        else:
            print(
                "usage: hermes kanban <action> [options]\n"
                "Run 'hermes kanban --help' for the full list of actions.",
                file=sys.stderr,
            )
        return 0

    # Fast-fail for clearer CLI UX only. The durable trust boundary is lower in
    # hermes_cli.kanban_db, because children can import DB mutators directly.
    if _is_delegated_child_cli_mutation(args):
        return _err("kanban: delegate_task child contexts cannot mutate Kanban tasks via the CLI")

    # Board-management commands operate on board metadata and the persisted
    # current-board pointer itself, so they must ignore the shared `--board`
    # task-routing override (else `--board beta boards show` reports beta).
    if action == "boards":
        return _dispatch_boards(args)

    # `--board <slug>` applies to every subcommand below via an env-var pin
    # (HERMES_KANBAN_BOARD) for the duration of this call, so it inherits the
    # exact resolution the dispatcher uses for workers.
    board_override = getattr(args, "board", None)
    board_scope = contextlib.nullcontext()
    if board_override:
        try:
            normed = kb._normalize_board_slug(board_override)
        except ValueError as exc:
            return _err(f"kanban: {exc}", 2)
        if not normed:
            return _err("kanban: --board requires a slug", 2)
        # Boards other than 'default' must already exist — typoed slugs
        # would otherwise silently create an empty board.
        if normed != kb.DEFAULT_BOARD and not kb.board_exists(normed):
            return _err(
                f"kanban: board {normed!r} does not exist. "
                f"Create it with `hermes kanban boards create {normed}`."
            )
        board_scope = kb.scoped_current_board(normed)

    with board_scope:
        # `repair` must dispatch BEFORE the auto-init: on a corrupt DB init_db()
        # itself raises KanbanDbCorruptError, which would turn every
        # `hermes kanban repair` into "could not initialize database".
        if action == "repair":
            return _cmd_repair(args)
        # Auto-initialize the DB before any subcommand. init_db is idempotent
        # (one SELECT against sqlite_master when tables exist) and prevents
        # "no such table: tasks" on first use from a fresh HERMES_HOME.
        try:
            kb.init_db()
        except Exception as exc:
            return _err(f"kanban: could not initialize database: {exc}")

        handler = _HANDLERS.get(action)
        if not handler:
            return _err(f"kanban: unknown action {action!r}", 2)
        try:
            return int(handler(args) or 0)
        except (ValueError, RuntimeError) as exc:
            return _err(f"kanban: {exc}")


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------

def _kanban_config() -> dict:
    """``config.yaml`` ``kanban:`` section, or ``{}`` when config can't be loaded."""
    try:
        from hermes_cli.config import load_config
        cfg = load_config()
        return (cfg.get("kanban", {}) if isinstance(cfg, dict) else {}) or {}
    except Exception:
        return {}


def _profile_author() -> str:
    """Best-effort author name for an interactive CLI call."""
    for env in ("HERMES_PROFILE_NAME", "HERMES_PROFILE"):
        v = os.environ.get(env)
        if v:
            return v
    try:
        from hermes_cli.profiles import get_active_profile_name
        return get_active_profile_name() or "user"
    except Exception:
        return "user"


_DELEGATED_CHILD_DENIED_ACTIONS: frozenset[str] = frozenset({
    "init", "create", "swarm", "assign", "reclaim", "reassign", "link", "unlink",
    "claim", "comment", "attach", "attach-rm", "complete", "edit", "block",
    "schedule", "unblock", "promote", "archive", "dispatch", "daemon", "repair",
    "heartbeat", "notify-subscribe", "notify-unsubscribe", "specify", "decompose",
    "gc",
})

_DELEGATED_CHILD_DENIED_BOARD_ACTIONS: frozenset[str] = frozenset({
    "create", "new", "rm", "remove", "delete", "switch", "use", "rename",
    "set-default-workdir",
})


def _is_delegated_child_cli_mutation(args: argparse.Namespace) -> bool:
    action = getattr(args, "kanban_action", None)
    if action == "boards":
        boards_action = getattr(args, "boards_action", None) or "list"
        if boards_action not in _DELEGATED_CHILD_DENIED_BOARD_ACTIONS:
            return False
    elif action not in _DELEGATED_CHILD_DENIED_ACTIONS:
        return False
    try:
        from agent.delegation_context import is_delegated_child_process_context

        return is_delegated_child_process_context()
    except Exception:
        return bool(os.environ.get("HERMES_DELEGATED_CHILD_CONTEXT"))


# ---------------------------------------------------------------------------


def _joined_words(words) -> Optional[str]:
    """Free-text positional ``nargs="*"`` words -> stripped string, or None when absent."""
    return " ".join(words).strip() if words else None


def _stripped_or_none(value: Optional[str]) -> Optional[str]:
    """``None`` stays ``None``; otherwise strip, and treat the empty string as ``None``."""
    return None if value is None else (value.strip() or None)


def _bulk_ids(args: argparse.Namespace) -> list[str]:
    """Positional ``task_id`` plus ``--ids`` extras (bulk verbs)."""
    return [args.task_id] + list(getattr(args, "ids", None) or [])


def _require_ids(args: argparse.Namespace) -> tuple[list[str], int]:
    """``args.task_ids`` -> ``(ids, 0)`` or ``([], 1)`` after printing the standard error."""
    ids = list(args.task_ids or [])
    if not ids:
        return ids, _err("at least one task_id is required")
    return ids, 0


def _parse_duration(val) -> Optional[int]:
    """``30s`` / ``5m`` / ``2h`` / ``1d`` or a raw integer → seconds; None for
    empty input; ValueError on malformed input."""
    if val is None or val == "":
        return None
    s = str(val).strip().lower()
    # Bare integer → seconds.
    try:
        return int(s)
    except ValueError:
        pass
    # Suffixed form.
    units = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    if s and s[-1] in units:
        try:
            n = float(s[:-1])
        except ValueError as exc:
            raise ValueError(f"malformed duration {val!r}") from exc
        return int(n * units[s[-1]])
    raise ValueError(f"malformed duration {val!r} (expected 30s, 5m, 2h, 1d, or a number)")


def _cmd_init(args: argparse.Namespace) -> int:
    path = kb.init_db()
    print(f"Kanban DB initialized at {path}")

    print()
    # Enumerate profiles on disk so the user knows what assignees are
    # already addressable.
    try:
        profiles = kb.list_profiles_on_disk()
    except Exception:
        profiles = []
    if profiles:
        print(f"Discovered {len(profiles)} profile(s) on disk; any of these can "
              f"be an --assignee:")
        for name in profiles:
            print(f"  {name}")
    else:
        print("No profiles found under ~/.hermes/profiles/.")
        print("Create one with `hermes -p <name> setup` before assigning tasks.")
    print()
    print("Next step: start the gateway so ready tasks actually get picked up.")
    print("  hermes gateway start")
    print()
    print(
        "The gateway hosts an embedded dispatcher that ticks every 60 seconds\n"
        "by default (config: kanban.dispatch_interval_seconds). Without a\n"
        "running gateway, tasks stay in 'ready' forever."
    )
    return 0


def _cmd_heartbeat(args: argparse.Namespace) -> int:
    with kb.connect_closing() as conn:
        ok = kb.heartbeat_worker(
            conn,
            args.task_id,
            note=getattr(args, "note", None),
            expected_run_id=_worker_run_id_for(args.task_id),
        )
    if not ok:
        return _err(f"cannot heartbeat {args.task_id} (not running?)")
    print(f"Heartbeat recorded for {args.task_id}")
    return 0


def _cmd_assignees(args: argparse.Namespace) -> int:
    with kb.connect_closing() as conn:
        data = kb.known_assignees(conn)
    if _json_out(args, data):
        return 0
    if not data:
        print("(no assignees — create a profile with `hermes -p <name> setup`)")
        return 0
    print(f"{'NAME':20s}  {'ON DISK':8s}  COUNTS")
    for entry in data:
        on_disk = "yes" if entry["on_disk"] else "no"
        print(f"{entry['name']:20s}  {on_disk:8s}  {_fmt_counts(entry['counts'] or {}, '(idle)')}")
    return 0


def _cmd_create(args: argparse.Namespace) -> int:
    try:
        ws_kind, ws_path = _parse_workspace_flag(args.workspace)
        branch_name = _parse_branch_flag(getattr(args, "branch", None))
    except argparse.ArgumentTypeError as exc:
        return _err(f"kanban: {exc}", 2)
    if branch_name and ws_kind != "worktree":
        return _err("kanban: --branch is only valid with --workspace worktree", 2)
    try:
        max_runtime = _parse_duration(getattr(args, "max_runtime", None))
    except ValueError as exc:
        return _err(f"kanban: --max-runtime: {exc}", 2)
    max_retries = getattr(args, "max_retries", None)
    if max_retries is not None and max_retries < 1:
        return _err(
            f"kanban: --max-retries must be >= 1 (got {max_retries}); "
            "use 1 to trip on the first failure.",
            2,
        )
    with kb.connect_closing() as conn:
        task_id = kb.create_task(
            conn,
            title=args.title,
            body=args.body,
            assignee=args.assignee,
            created_by=args.created_by or _profile_author(),
            workspace_kind=ws_kind,
            workspace_path=ws_path,
            branch_name=branch_name,
            project_id=getattr(args, "project", None),
            tenant=args.tenant,
            priority=args.priority,
            parents=tuple(args.parent or ()),
            triage=bool(getattr(args, "triage", False)),
            idempotency_key=getattr(args, "idempotency_key", None),
            max_runtime_seconds=max_runtime,
            skills=getattr(args, "skills", None) or None,
            max_retries=max_retries,
            model_override=getattr(args, "model_override", None),
            provider_override=getattr(args, "provider_override", None),
            goal_mode=bool(getattr(args, "goal_mode", False)),
            goal_max_turns=getattr(args, "goal_max_turns", None),
            initial_status=getattr(args, "initial_status", "running"),
        )
        task = kb.get_task(conn, task_id)
    if getattr(args, "json", False):
        _print_json(_task_to_dict(task))
    else:
        print(f"Created {task_id}  ({task.status}, assignee={task.assignee or '-'})")

        # Warn when the task would sit in `ready` because no dispatcher is
        # present. Only ready+assigned tasks — triage/todo idle by design,
        # unassigned can't dispatch. Skipped in --json so stdout stays
        # machine-parseable.
        if task.status == "ready" and task.assignee:
            running, message = _check_dispatcher_presence()
            if not running and message:
                print(f"\n⚠  {message}", file=sys.stderr)
    return 0


def _cmd_swarm(args: argparse.Namespace) -> int:
    try:
        workers = [ks.parse_worker_arg(raw) for raw in (args.worker or [])]
    except ValueError as exc:
        return _err(f"kanban swarm: {exc}", 2)
    if not workers:
        return _err("kanban swarm: at least one --worker is required", 2)
    with kb.connect_closing() as conn:
        created = ks.create_swarm(
            conn,
            goal=args.goal,
            workers=workers,
            verifier_assignee=args.verifier,
            synthesizer_assignee=args.synthesizer,
            tenant=args.tenant,
            created_by=args.created_by or _profile_author(),
            priority=args.priority,
            idempotency_key=getattr(args, "idempotency_key", None),
        )
    if getattr(args, "json", False):
        _print_json(created.as_dict())
    else:
        print(f"Swarm root: {created.root_id}")
        print("Workers: " + ", ".join(created.worker_ids))
        print(f"Verifier: {created.verifier_id}")
        print(f"Synthesizer: {created.synthesizer_id}")
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    assignee = args.assignee
    if args.mine and not assignee:
        assignee = _profile_author()
    with kb.connect_closing() as conn:
        # Cheap "mini-dispatch": recompute ready so list output reflects
        # dependencies that may have cleared since the last dispatcher tick.
        kb.recompute_ready(conn)
        tasks = kb.list_tasks(
            conn,
            assignee=assignee,
            status=args.status,
            tenant=args.tenant,
            session_id=args.session,
            include_archived=args.archived,
            order_by=getattr(args, "sort", None),
            workflow_template_id=args.workflow_template_id,
            current_step_key=args.current_step_key,
        )
    if _json_out(args, [_task_to_dict(t) for t in tasks]):
        return 0
    # Passive discoverability: only multi-board users see which board this is.
    try:
        all_boards = kb.list_boards(include_archived=False)
    except Exception:
        all_boards = []
    if len(all_boards) > 1:
        current = kb.get_current_board()
        other_count = len(all_boards) - 1
        print(
            f"Board: {current} "
            f"({other_count} other board{'s' if other_count != 1 else ''} — "
            f"`hermes kanban boards list`)\n"
        )
    if not tasks:
        print("(no matching tasks)")
        return 0
    for t in tasks:
        print(_fmt_task_line(t))
    return 0


def _print_diagnostics(diags, indent: str, *, with_kind: bool) -> None:
    """Shared human rendering for ``show`` and ``diagnostics`` (suggested actions only)."""
    sev_marker = {"warning": "⚠", "error": "!!", "critical": "!!!"}
    for d in diags:
        head = f"{d.kind}: {d.title}" if with_kind else d.title
        print(f"{indent}{sev_marker.get(d.severity, '?')} [{d.severity}] {head}")
        if d.data:
            bits = [
                f"{k}={','.join(str(x) for x in v)}" if isinstance(v, list) else f"{k}={v}"
                for k, v in d.data.items()
            ]
            if bits:
                print(f"{indent}   data: {' | '.join(bits)}")
        for a in d.actions:
            if a.suggested:
                print(f"{indent}   → {a.label}")


def _cmd_show(args: argparse.Namespace) -> int:
    rsk, rc = _run_state_kwargs(args, "show")
    if rc:
        return rc
    graph = None
    with kb.connect_closing() as conn:
        task = kb.get_task(conn, args.task_id)
        if not task:
            return _err(f"no such task: {args.task_id}")
        comments = kb.list_comments(conn, args.task_id)
        events = kb.list_events(conn, args.task_id)
        parents = kb.parent_ids(conn, args.task_id)
        children = kb.child_ids(conn, args.task_id)
        runs = kb.list_runs(conn, args.task_id, **rsk)
        # Workers hand off via task_runs.summary; tasks.result stays NULL unless
        # explicitly set, so surface the latest summary here.
        latest_summary = kb.latest_summary(conn, args.task_id)
        if not getattr(args, "json", False):
            graph = kb.task_graph_context(conn, task.id)

    if getattr(args, "json", False):
        _print_json({
            "task": _task_to_dict(task),
            "latest_summary": latest_summary,
            "parents": parents,
            "children": children,
            "comments": [_obj_dict(c, ("author", "body", "created_at")) for c in comments],
            "events": [_obj_dict(e, ("kind", "payload", "created_at", "run_id")) for e in events],
            "runs": [_obj_dict(r, _SHOW_RUN_FIELDS) for r in runs],
        })
        return 0

    print(f"Task {task.id}: {task.title}")
    print(f"  status:    {task.status}")
    print(f"  assignee:  {task.assignee or '-'}")
    if task.tenant:
        print(f"  tenant:    {task.tenant}")
    print(f"  workspace: {task.workspace_kind}" +
          (f" @ {task.workspace_path}" if task.workspace_path else ""))
    if task.branch_name:
        print(f"  branch:    {task.branch_name}")
    if task.skills:
        print(f"  skills:    {', '.join(task.skills)}")
    if task.model_override:
        _prov = f" (provider: {task.provider_override})" if task.provider_override else ""
        print(f"  model:     {task.model_override}{_prov}")
    # Effective retry threshold: per-task override, else config, else default —
    # so operators can see why a task auto-blocked when it did.
    if task.max_retries is not None:
        print(f"  max-retries: {task.max_retries} (task)")
    else:
        cfg_val = _kanban_config().get("failure_limit")
        if cfg_val is not None and int(cfg_val) != kb.DEFAULT_FAILURE_LIMIT:
            print(f"  max-retries: {int(cfg_val)} (config kanban.failure_limit)")
        else:
            print(f"  max-retries: {kb.DEFAULT_FAILURE_LIMIT} (default)")
    print(f"  created:   {_fmt_ts(task.created_at)} by {task.created_by or '-'}")

    # Diagnostics up top so CLI users see distress signals before scrolling.
    from hermes_cli import kanban_diagnostics as kd
    diags = kd.compute_task_diagnostics(task, events, runs, graph=graph)
    if diags:
        print(f"\n  Diagnostics ({len(diags)}):")
        _print_diagnostics(diags, "    ", with_kind=False)
    if task.started_at:
        print(f"  started:   {_fmt_ts(task.started_at)}")
    if task.completed_at:
        print(f"  completed: {_fmt_ts(task.completed_at)}")
    if parents:
        print(f"  parents:   {', '.join(parents)}")
    if children:
        print(f"  children:  {', '.join(children)}")
    if task.body:
        print()
        print("Body:")
        print(task.body)
    if task.result:
        print()
        print("Result:")
        print(task.result)
    elif latest_summary:
        print()
        print("Latest summary:")
        print(latest_summary)
    if comments:
        print()
        print(f"Comments ({len(comments)}):")
        for c in comments:
            print(f"  [{_fmt_ts(c.created_at)}] {c.author}: {c.body}")
    if events:
        print()
        print(f"Events ({len(events)}):")
        for e in events[-20:]:
            pl = f" {e.payload}" if e.payload else ""
            run_tag = f" [run {e.run_id}]" if e.run_id else ""
            print(f"  [{_fmt_ts(e.created_at)}]{run_tag} {e.kind}{pl}")
    if runs:
        print()
        print(f"Runs ({len(runs)}):")
        for r in runs:
            # Clamp to 0 so NTP backward-jumps don't print negative seconds.
            elapsed = (max(0, r.ended_at - r.started_at)
                       if r.ended_at else None)
            el = f"{elapsed}s" if elapsed is not None else "active"
            outcome = r.outcome or r.status or "active"
            print(f"  #{r.id:<3} {outcome:<12} @{r.profile or '-'}  {el}  "
                  f"{_fmt_ts(r.started_at)}")
            if r.summary:
                print(f"        → {r.summary.splitlines()[0][:160]}")
            if r.error:
                print(f"        ! {r.error.splitlines()[0][:160]}")
    return 0


def _cmd_assign(args: argparse.Namespace) -> int:
    profile = _none_profile(args.profile)
    with kb.connect_closing() as conn:
        ok = kb.assign_task(conn, args.task_id, profile)
    if not ok:
        return _err(f"no such task: {args.task_id}")
    print(f"Assigned {args.task_id} to {profile or '(unassigned)'}")
    return 0


def _cmd_set_model(args: argparse.Namespace) -> int:
    model = args.model
    if model is not None and model.lower() in {"none", "-", "null", ""}:
        model = None
    provider = getattr(args, "provider", None)
    try:
        with kb.connect_closing() as conn:
            ok = kb.set_model_override(conn, args.task_id, model, provider=provider)
    except (ValueError, RuntimeError) as exc:
        return _err(f"kanban: {exc}", 2)
    if not ok:
        return _err(f"no such task: {args.task_id}")
    if model:
        label = f"{provider}:{model}" if provider else model
        print(f"Set model override on {args.task_id}: {label} "
              "(applies on next dispatch)")
    else:
        print(f"Cleared model override on {args.task_id} "
              "(worker uses its profile default)")
    return 0


def _cmd_reclaim(args: argparse.Namespace) -> int:
    with kb.connect_closing() as conn:
        ok = kb.reclaim_task(
            conn, args.task_id,
            reason=getattr(args, "reason", None),
        )
    if not ok:
        return _err(f"cannot reclaim {args.task_id} (not running or unknown id)")
    print(f"Reclaimed {args.task_id}")
    return 0


def _cmd_reassign(args: argparse.Namespace) -> int:
    profile = _none_profile(args.profile)
    with kb.connect_closing() as conn:
        ok = kb.reassign_task(
            conn, args.task_id, profile,
            reclaim_first=bool(getattr(args, "reclaim", False)),
            reason=getattr(args, "reason", None),
        )
    if not ok:
        return _err(
            f"cannot reassign {args.task_id} "
            f"(unknown id, or still running — pass --reclaim to release first)"
        )
    print(
        f"Reassigned {args.task_id} to "
        f"{profile or '(unassigned)'}"
        + (" (claim reclaimed)" if getattr(args, "reclaim", False) else "")
    )
    return 0


def _rows_by_task(conn, table: str, ids: list[str]) -> dict[str, list]:
    """``{task_id: [rows ordered by id]}`` for every id (empty list when none)."""
    by = {i: [] for i in ids}
    placeholders = ",".join(["?"] * len(ids))
    for row in conn.execute(
        f"SELECT * FROM {table} WHERE task_id IN ({placeholders}) ORDER BY id", tuple(ids),
    ):
        by.setdefault(row["task_id"], []).append(row)
    return by


def _cmd_diagnostics(args: argparse.Namespace) -> int:
    """List active diagnostics on the board via the same rule engine the dashboard uses."""
    from hermes_cli import kanban_diagnostics as kd
    from hermes_cli.config import load_config

    diag_config = kd.config_from_runtime_config(load_config())

    with kb.connect_closing() as conn:
        # Either one-task mode or fleet mode.
        if getattr(args, "task", None):
            task = kb.get_task(conn, args.task)
            if task is None:
                return _err(f"no such task: {args.task}")
            diags_by_task = {
                args.task: kd.compute_task_diagnostics(
                    task,
                    kb.list_events(conn, args.task),
                    kb.list_runs(conn, args.task),
                    graph=kb.task_graph_context(conn, args.task),
                    config=diag_config,
                )
            }
        else:
            # Fleet mode: pull all non-archived tasks + their events/runs.
            rows = list(conn.execute(
                "SELECT * FROM tasks WHERE status != 'archived'"
            ).fetchall())
            ids = [r["id"] for r in rows]
            diags_by_task = {}
            if ids:
                ev_by = _rows_by_task(conn, "task_events", ids)
                run_by = _rows_by_task(conn, "task_runs", ids)
                graph_by = kb.task_graph_contexts(conn, ids)
                for r in rows:
                    tid = r["id"]
                    dl = kd.compute_task_diagnostics(
                        r,
                        ev_by.get(tid, []),
                        run_by.get(tid, []),
                        graph=graph_by.get(tid),
                        config=diag_config,
                    )
                    if dl:
                        diags_by_task[tid] = dl

        # Severity filter.
        sev = getattr(args, "severity", None)
        if sev:
            floor = kd.SEVERITY_ORDER.index(sev)
            diags_by_task = {
                tid: kept
                for tid, dl in diags_by_task.items()
                if (kept := [d for d in dl if kd.SEVERITY_ORDER.index(d.severity) >= floor])
            }

        # Map task_id → title/status/assignee for the table output.
        meta: dict[str, dict] = {}
        if diags_by_task:
            placeholders = ",".join(["?"] * len(diags_by_task))
            for r in conn.execute(
                f"SELECT id, title, status, assignee FROM tasks WHERE id IN ({placeholders})",
                tuple(diags_by_task.keys()),
            ):
                meta[r["id"]] = {k: r[k] for k in ("title", "status", "assignee")}

    if getattr(args, "json", False):
        _print_json([
            {
                "task_id": tid,
                **meta.get(tid, {}),
                "diagnostics": [d.to_dict() for d in dl],
            }
            for tid, dl in diags_by_task.items()
        ])
        return 0

    if not diags_by_task:
        print("No active diagnostics on this board.")
        return 0

    total = sum(len(dl) for dl in diags_by_task.values())
    print(
        f"{total} active diagnostic(s) across "
        f"{len(diags_by_task)} task(s):\n"
    )
    for tid, dl in diags_by_task.items():
        m = meta.get(tid, {})
        title = m.get("title") or "(untitled)"
        status = m.get("status") or "?"
        assignee = m.get("assignee") or "(unassigned)"
        print(f"  {tid}  {status:8s}  @{assignee:18s}  {title}")
        _print_diagnostics(dl, "    ", with_kind=True)
        print()
    return 0


def _cmd_link(args: argparse.Namespace) -> int:
    with kb.connect_closing() as conn:
        kb.link_tasks(conn, args.parent_id, args.child_id)
    print(f"Linked {args.parent_id} -> {args.child_id}")
    return 0


def _cmd_unlink(args: argparse.Namespace) -> int:
    with kb.connect_closing() as conn:
        ok = kb.unlink_tasks(conn, args.parent_id, args.child_id)
    if not ok:
        return _err(f"No such link: {args.parent_id} -> {args.child_id}")
    print(f"Unlinked {args.parent_id} -> {args.child_id}")
    return 0


def _cmd_claim(args: argparse.Namespace) -> int:
    with kb.connect_closing() as conn:
        task = kb.claim_task(conn, args.task_id, ttl_seconds=args.ttl)
        if task is None:
            existing = kb.get_task(conn, args.task_id)
            if existing is None:
                return _err(f"no such task: {args.task_id}")
            return _err(
                f"cannot claim {args.task_id}: status={existing.status} "
                f"lock={existing.claim_lock or '(none)'}"
            )
        workspace = kb.resolve_workspace(task)
        kb.set_workspace_path(conn, task.id, str(workspace))
    print(f"Claimed {task.id}")
    print(f"Workspace: {workspace}")
    return 0


def _cmd_comment(args: argparse.Namespace) -> int:
    body = " ".join(args.text).strip()
    if args.max_len is not None:
        if args.max_len < 1:
            return _err("kanban: --max-len must be positive", 2)
        if len(body) > args.max_len:
            suffix = f"\n\n[trimmed to {args.max_len} chars by --max-len]"
            body = body[: max(0, args.max_len - len(suffix))].rstrip() + suffix
    author = args.author or _profile_author()
    with kb.connect_closing() as conn:
        kb.add_comment(conn, args.task_id, author, body)
    print(f"Comment added to {args.task_id}")
    return 0


def _cmd_attach(args: argparse.Namespace) -> int:
    """Attach a local file via the shared ``store_attachment_bytes`` path (same
    25 MB cap and name sanitisation as the dashboard upload and agent tool)."""
    import mimetypes

    src = Path(args.path).expanduser()
    if not src.is_file():
        return _err(f"kanban: no such file: {src}")
    data = src.read_bytes()
    name = args.name or src.name
    content_type = args.content_type or mimetypes.guess_type(name)[0]
    uploaded_by = args.author or _profile_author()
    try:
        with kb.connect_closing() as conn:
            att_id = kb.store_attachment_bytes(
                conn,
                args.task_id,
                name,
                data,
                content_type=content_type,
                uploaded_by=uploaded_by,
            )
    except kb.AttachmentTooLarge as exc:
        return _err(f"kanban: {exc}")
    print(f"Attached {name} to {args.task_id} (attachment {att_id}, {len(data)} bytes)")
    return 0


def _cmd_attachments(args: argparse.Namespace) -> int:
    with kb.connect_closing() as conn:
        if kb.get_task(conn, args.task_id) is None:
            return _err(f"no such task: {args.task_id}")
        atts = kb.list_attachments(conn, args.task_id)
    if _json_out(args, [_obj_dict(a, _ATTACHMENT_FIELDS) for a in atts], ascii=True):
        return 0
    if not atts:
        print(f"No attachments on {args.task_id}")
        return 0
    print(f"Attachments on {args.task_id}:")
    for a in atts:
        ct = a.content_type or "-"
        print(f"  [{a.id}] {a.filename}  ({a.size} bytes, {ct}, by {a.uploaded_by or '-'})")
        print(f"        {a.stored_path}")
    return 0


def _cmd_attach_rm(args: argparse.Namespace) -> int:
    with kb.connect_closing() as conn:
        removed = kb.delete_attachment(conn, args.attachment_id)
    if removed is None:
        return _err(f"no such attachment: {args.attachment_id}")
    print(f"Deleted attachment {args.attachment_id} ({removed.filename}) from {removed.task_id}")
    return 0


def _worker_run_id_for(task_id: str) -> Optional[int]:
    if os.environ.get("HERMES_KANBAN_TASK") != task_id:
        return None
    raw = os.environ.get("HERMES_KANBAN_RUN_ID")
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        return None


def _goal_mode_handoff_rejection(task: Optional[kb.Task], evidence: str):
    """Apply the goal judge to every terminal worker handoff, including review.

    Returns ``(verdict, reason_or_None)`` — ``"done"`` allows the handoff;
    ``"blocked"`` means the judge ruled the goal unachievable (#100954);
    ``"continue"``/``"wait"`` reject with the judge's reason.
    """
    if task is None or not task.goal_mode:
        return ("done", None)
    try:
        from agent.auxiliary_client import get_text_auxiliary_client

        client, model = get_text_auxiliary_client("goal_judge")
    except Exception:
        return ("done", None)
    if client is None or not model:
        return ("done", None)

    from hermes_cli.goals import judge_goal

    verdict = "done"
    reason = ""
    try:
        verdict, reason, _, _, _ = judge_goal(
            goal=f"{task.title}\n\n{task.body or ''}".strip(),
            last_response=evidence.strip(),
        )
    except Exception as judge_exc:
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "goal judge check failed, allowing lifecycle handoff: %s",
            judge_exc,
            exc_info=True,
        )
    return (verdict, None if verdict == "done" else reason)


def _goal_gate_error(conn, tid: str, evidence: str, handoff: str, blocked_hint: str,
                     continue_hint: str) -> Optional[str]:
    """Goal-mode judge gate shared by ``complete`` / ``request-review``
    (mirrors tools/kanban_tools.py); applied to every terminal handoff so
    request-review can't bypass it. Returns the error line, or None to allow."""
    verdict, rejection = _goal_mode_handoff_rejection(kb.get_task(conn, tid), evidence)
    if verdict == "blocked":
        return (f"kanban: goal {handoff} of {tid} rejected: judge ruled "
                f"the goal unachievable — {rejection}. {blocked_hint}")
    if rejection is not None:
        return f"kanban: goal {handoff} of {tid} rejected by judge: {rejection}. {continue_hint}"
    return None


def _cmd_complete(args: argparse.Namespace) -> int:
    """Mark one or more tasks done. Supports a single id or a list."""
    ids, rc = _require_ids(args)
    if rc:
        return rc
    summary = getattr(args, "summary", None)
    raw_meta = getattr(args, "metadata", None)
    # Structured handoff fields are per-run; copying them across N runs is
    # almost always a footgun, so refuse rather than silently do it.
    if len(ids) > 1 and (summary or raw_meta):
        return _err(
            "kanban: --summary / --metadata are per-task and can't be used "
            "with multiple ids (would apply the same handoff to every task). "
            "Complete tasks one at a time, or drop the flags for the bulk close.",
            2,
        )
    metadata, rc = _parse_metadata_flag(raw_meta)
    if rc:
        return rc
    fail_msg: dict[str, str] = {}
    with kb.connect_closing() as conn:
        def op(tid):
            gate_err = _goal_gate_error(
                conn, tid, (summary or args.result or "").strip(), "completion",
                "Re-scope with kanban edit, or record the block with kanban block "
                "instead of completing.",
                "Provide evidence matching the task's acceptance criteria.",
            )
            if gate_err:
                fail_msg[tid] = gate_err
                return False
            fail_msg[tid] = f"cannot complete {tid} (unknown id or terminal state)"
            return kb.complete_task(
                conn, tid,
                result=args.result,
                summary=summary,
                metadata=metadata,
                expected_run_id=_worker_run_id_for(tid),
            )

        return _bulk_apply(ids, op, lambda tid: f"Completed {tid}", fail_msg.__getitem__)


def _cmd_edit(args: argparse.Namespace) -> int:
    metadata, rc = _parse_metadata_flag(getattr(args, "metadata", None))
    if rc:
        return rc
    with kb.connect_closing() as conn:
        if not kb.edit_completed_task_result(
            conn,
            args.task_id,
            result=args.result,
            summary=getattr(args, "summary", None),
            metadata=metadata,
        ):
            return _err(f"cannot edit {args.task_id} (unknown id or task is not done)")
    print(f"Edited {args.task_id}")
    return 0


def _cmd_block(args: argparse.Namespace) -> int:
    reason = _joined_words(args.reason)
    kind = getattr(args, "kind", None)
    author = _profile_author()
    ids = _bulk_ids(args)
    suffix = f": {reason}" if reason else ""
    with kb.connect_closing() as conn:
        def op(tid):
            if reason:
                kb.add_comment(conn, tid, author, f"BLOCKED: {reason}")
            return kb.block_task(
                conn, tid, reason=reason, kind=kind,
                expected_run_id=_worker_run_id_for(tid),
            )

        def ok_msg(tid):
            # Report where the task actually landed — dependency blocks go
            # to todo, and a tripped unblock-loop breaker routes to triage.
            landed = kb.get_task(conn, tid)
            where = landed.status if landed else "blocked"
            if where == "todo":
                return f"{tid} → todo (dependency wait){suffix}"
            if where == "triage":
                return (f"{tid} → triage (unblock loop detected — needs a "
                        f"human decision){suffix}")
            return f"Blocked {tid}{suffix}"

        return _bulk_apply(ids, op, ok_msg, lambda tid: f"cannot block {tid}")


def _cmd_schedule(args: argparse.Namespace) -> int:
    reason = _joined_words(args.reason)
    author = _profile_author()
    ids = _bulk_ids(args)
    suffix = f": {reason}" if reason else ""
    with kb.connect_closing() as conn:
        def op(tid):
            if reason:
                kb.add_comment(conn, tid, author, f"SCHEDULED: {reason}")
            return kb.schedule_task(
                conn, tid, reason=reason, expected_run_id=_worker_run_id_for(tid),
            )

        return _bulk_apply(
            ids, op, lambda tid: f"Scheduled {tid}{suffix}", lambda tid: f"cannot schedule {tid}",
        )


def _cmd_unblock(args: argparse.Namespace) -> int:
    ids, rc = _require_ids(args)
    if rc:
        return rc
    reason = _stripped_or_none(getattr(args, "reason", None))
    author = _profile_author() if reason else None
    suffix = f": {reason}" if reason else ""
    with kb.connect_closing() as conn:
        def op(tid):
            if reason:
                kb.add_comment(conn, tid, author, f"UNBLOCK: {reason}")
            return kb.unblock_task(conn, tid)

        return _bulk_apply(
            ids, op, lambda tid: f"Unblocked {tid}{suffix}",
            lambda tid: f"cannot unblock {tid} (not blocked/scheduled?)",
        )


def _cmd_request_review(args: argparse.Namespace) -> int:
    tid = args.task_id
    summary = _stripped_or_none(getattr(args, "summary", None))
    metadata, rc = _parse_metadata_flag(getattr(args, "metadata", None))
    if rc:
        return rc
    reviewer = getattr(args, "reviewer", None)
    with kb.connect_closing() as conn:
        gate_err = _goal_gate_error(
            conn, tid, summary or "", "review handoff",
            "Record the block with kanban block instead of requesting review.",
            "Provide acceptance evidence matching the task.",
        )
        if gate_err:
            return _err(gate_err)
        ok, reason = kb.request_review(
            conn,
            tid,
            summary=summary,
            metadata=metadata,
            reviewer=reviewer,
            expected_run_id=_worker_run_id_for(tid),
            force=bool(getattr(args, "force", False)),
            with_reason=True,
        )
        if not ok:
            return _err(f"cannot request review for {tid}: {reason or 'not running/ready?'}")
        persisted_run = kb.latest_run(conn, tid)
        display_summary = persisted_run.summary if persisted_run else None
        print(
            f"Requested review for {tid}"
            + (f": {display_summary}" if display_summary else "")
        )
    return 0


def _cmd_request_changes(args: argparse.Namespace) -> int:
    tid = args.task_id
    reason = " ".join(args.reason).strip()
    with kb.connect_closing() as conn:
        ok, detail = kb.request_changes(
            conn,
            tid,
            reason=reason,
            expected_run_id=_worker_run_id_for(tid),
        )
        if not ok:
            return _err(f"cannot request changes for {tid}: {detail or 'invalid review state'}")
        print(
            f"Requested changes for {tid}"
            + (f"; routed to {detail}" if detail else "")
        )
    return 0


def _cmd_reopen_review(args: argparse.Namespace) -> int:
    ids, rc = _require_ids(args)
    if rc:
        return rc
    reason = getattr(args, "reason", None)
    if reason is not None:
        reason = str(kb.redact_review_value(reason.strip())).strip() or None
    author = _profile_author() if reason else None
    suffix = f": {reason}" if reason else ""
    with kb.connect_closing() as conn:
        def op(tid):
            if not kb.reopen_review_task(conn, tid):
                return False
            if reason:
                kb.add_comment(conn, tid, author or "operator", f"CHANGES REQUESTED: {reason}")
            return True

        return _bulk_apply(
            ids, op, lambda tid: f"Reopened {tid}{suffix}",
            lambda tid: f"cannot reopen {tid} (not in review?)",
        )


def _cmd_promote(args: argparse.Namespace) -> int:
    reason = _joined_words(args.reason)
    author = _profile_author()
    # Dedupe while preserving order; positional task_id always first.
    ids = list(dict.fromkeys(_bulk_ids(args)))

    results: list[dict[str, object]] = []
    with kb.connect_closing() as conn:
        for tid in ids:
            ok, err = kb.promote_task(
                conn,
                tid,
                actor=author,
                reason=reason,
                force=bool(args.force),
                dry_run=bool(args.dry_run),
            )
            results.append({
                "task_id": tid,
                "promoted": ok,
                "dry_run": bool(args.dry_run),
                "forced": bool(args.force),
                "reason": reason,
                "error": err,
            })

    failed = [r for r in results if not r["promoted"]]
    if getattr(args, "json", False):
        # Single-id stays a flat object for back-compat; bulk emits a list.
        _print_json(results[0] if len(results) == 1 else results)
        return 0 if not failed else 1

    tag = " (dry)" if args.dry_run else ""
    label = "Would promote" if args.dry_run else "Promoted"
    for r in results:
        if r["promoted"]:
            suffix = f": {reason}" if reason else ""
            print(f"{label} {r['task_id']} -> ready{tag}{suffix}")
        else:
            print(f"cannot promote {r['task_id']}: {r['error']}", file=sys.stderr)
    return 0 if not failed else 1


def _cmd_archive(args: argparse.Namespace) -> int:
    ids = list(args.task_ids or [])
    purge_ids = list(getattr(args, "purge_ids", None) or [])
    if ids and purge_ids:
        return _err("choose either task_ids to archive or --rm archived task_ids")
    if not ids and not purge_ids:
        return _err("at least one task_id is required")
    with kb.connect_closing() as conn:
        if purge_ids:
            return _bulk_apply(
                purge_ids, lambda tid: kb.delete_archived_task(conn, tid),
                lambda tid: f"Deleted {tid}",
                lambda tid: f"cannot delete {tid} (must already be archived)",
            )
        return _bulk_apply(
            ids, lambda tid: kb.archive_task(conn, tid),
            lambda tid: f"Archived {tid}", lambda tid: f"cannot archive {tid}",
        )


def _cmd_tail(args: argparse.Namespace) -> int:
    last_id = 0
    print(f"Tailing events for {args.task_id}. Ctrl-C to stop.")
    try:
        while True:
            with kb.connect_closing() as conn:
                events = kb.list_events(conn, args.task_id)
            for e in events:
                if e.id > last_id:
                    pl = f" {e.payload}" if e.payload else ""
                    print(f"[{_fmt_ts(e.created_at)}] {e.kind}{pl}", flush=True)
                    last_id = e.id
            time.sleep(max(0.1, args.interval))
    except KeyboardInterrupt:
        print("\n(stopped)")
        return 0


def _coerce_positive_int(value):
    if value is None:
        return None
    try:
        ival = int(value)
    except (TypeError, ValueError):
        return None
    return ival if ival >= 1 else None


def _cmd_dispatch(args: argparse.Namespace) -> int:
    # Honour kanban.default_assignee, kanban.max_in_progress,
    # kanban.max_in_progress_per_profile and kanban.max_spawn with the same
    # semantics as the gateway dispatch path.
    try:
        from hermes_cli.config import load_config
        _cfg = load_config()
        _kanban_cfg = _cfg.get("kanban", {}) if isinstance(_cfg, dict) else {}
        default_assignee = (_kanban_cfg.get("default_assignee") or "").strip() or None
        max_in_progress_per_profile = _coerce_positive_int(
            _kanban_cfg.get("max_in_progress_per_profile")
        )
        # Memory-derived default when unset — same fallback the gateway applies.
        max_in_progress = kb.resolve_max_in_progress(
            _coerce_positive_int(_kanban_cfg.get("max_in_progress"))
        )
        # CLI --max is the more explicit signal, so it wins over kanban.max_spawn.
        cli_max = getattr(args, "max", None)
        max_spawn = cli_max if cli_max is not None else _coerce_positive_int(
            _kanban_cfg.get("max_spawn")
        )
    except Exception:
        default_assignee = None
        max_in_progress_per_profile = None
        max_in_progress = None
        max_spawn = getattr(args, "max", None)
    with kb.connect_closing() as conn:
        res = kb.dispatch_once(
            conn,
            dry_run=args.dry_run,
            max_spawn=max_spawn,
            max_in_progress=max_in_progress,
            failure_limit=getattr(args, "failure_limit", kb.DEFAULT_SPAWN_FAILURE_LIMIT),
            default_assignee=default_assignee,
            max_in_progress_per_profile=max_in_progress_per_profile,
        )
    if getattr(args, "json", False):
        _print_json({
            "reclaimed": res.reclaimed,
            "crashed": res.crashed,
            "timed_out": res.timed_out,
            "stale": res.stale,
            "auto_blocked": res.auto_blocked,
            "promoted": res.promoted,
            "spawned": [
                {"task_id": tid, "assignee": who, "workspace": ws}
                for (tid, who, ws) in res.spawned
            ],
            "skipped_unassigned": res.skipped_unassigned,
            "skipped_nonspawnable": res.skipped_nonspawnable,
            "skipped_per_profile_capped": [
                {"task_id": tid, "assignee": who, "current": current}
                for (tid, who, current) in res.skipped_per_profile_capped
            ],
            "auto_assigned_default": res.auto_assigned_default,
        }, ascii=True)
        return 0
    print(f"Reclaimed:    {res.reclaimed}")
    for label, items in (
        ("Crashed:     ", res.crashed),
        ("Timed out:   ", res.timed_out),
        ("Stale:       ", res.stale),
        ("Auto-blocked:", res.auto_blocked),
    ):
        print(f"{label} {len(items)}")
        if items:
            print(f"  {', '.join(items)}")
    print(f"Promoted:     {res.promoted}")
    print(f"Spawned:      {len(res.spawned)}")
    tag = " (dry)" if args.dry_run else ""
    for tid, who, ws in res.spawned:
        print(f"  - {tid}  ->  {who}  @ {ws or '-'}{tag}")
    if res.auto_assigned_default:
        print(
            f"Auto-assigned to kanban.default_assignee={default_assignee!r}: "
            f"{', '.join(res.auto_assigned_default)}"
        )
    if res.skipped_unassigned:
        print(f"Skipped (unassigned): {', '.join(res.skipped_unassigned)}")
    for tid, who, current in res.skipped_per_profile_capped:
        print(f"Deferred ({who} at per-profile cap, {current} running): {tid}")
    if res.skipped_nonspawnable:
        print(
            f"Skipped (non-spawnable assignee — terminal lane, OK): "
            f"{', '.join(res.skipped_nonspawnable)}"
        )
    return 0


def _cmd_daemon(args: argparse.Namespace) -> int:
    """Deprecated — the dispatcher now runs inside the gateway.

    Kept as a stub so old scripts/systemd units get a clear migration message.
    ``--force`` (hidden from --help) keeps the standalone loop for hosts that
    truly cannot run the gateway; the default path exits 2 so nobody
    accidentally runs two dispatchers against the same kanban.db.
    """
    if not getattr(args, "force", False):
        return _err(
            "hermes kanban daemon: DEPRECATED — the dispatcher now runs\n"
            "inside the gateway. To use kanban:\n"
            "\n"
            "    hermes gateway start       # starts the gateway + embedded dispatcher\n"
            "\n"
            "Ready tasks will be picked up on the next dispatcher tick\n"
            "(default: every 60 seconds). Configure via config.yaml:\n"
            "\n"
            "    kanban:\n"
            "      dispatch_in_gateway: true      # default\n"
            "      dispatch_interval_seconds: 60\n"
            "      failure_limit: 2              # consecutive non-success attempts before auto-block\n"
            "\n"
            "Running both the gateway AND this standalone daemon will\n"
            "race for claims. If you truly need the old standalone\n"
            "daemon (no gateway available), rerun with --force.",
            2,
        )

    # Init before printing "started" so the DB path is right and init errors
    # surface immediately.
    kb.init_db()

    pidfile = getattr(args, "pidfile", None)
    if pidfile:
        try:
            Path(pidfile).parent.mkdir(parents=True, exist_ok=True)
            Path(pidfile).write_text(str(os.getpid()), encoding="utf-8")
        except OSError as exc:
            print(f"warning: could not write pidfile {pidfile}: {exc}", file=sys.stderr)

    verbose = bool(getattr(args, "verbose", False))
    print(
        f"Kanban dispatcher running STANDALONE via --force "
        f"(interval={args.interval}s, pid={os.getpid()}). "
        f"Ctrl-C to stop. NOTE: if a gateway is also running with "
        f"dispatch_in_gateway=true (default), you have two dispatchers "
        f"racing for claims.",
        file=sys.stderr,
    )

    # Health telemetry: warn when every tick finds ready work but spawns
    # nothing (broken profile, PATH drift, missing venv, credential loss) —
    # the per-task breaker auto-blocks quietly, so the operator needs a signal.
    HEALTH_WINDOW = 6  # ticks (default 30s at interval=5)
    health_state = {"bad_ticks": 0, "last_warn_at": 0}

    def _on_tick(res):
        ready_pending = bool(res.skipped_unassigned) or _ready_queue_nonempty()
        spawned_any = bool(res.spawned)
        if ready_pending and not spawned_any:
            health_state["bad_ticks"] += 1
        else:
            health_state["bad_ticks"] = 0
        # Warn once per HEALTH_WINDOW bad ticks, at most every 5 minutes.
        if health_state["bad_ticks"] >= HEALTH_WINDOW:
            now = int(time.time())
            if now - health_state["last_warn_at"] >= 300:
                print(
                    f"[{_fmt_ts(now)}] WARN dispatcher stuck: "
                    f"ready queue non-empty for {health_state['bad_ticks']} "
                    f"consecutive ticks but 0 workers spawned successfully. "
                    f"Check profile health (venv, PATH, credentials) and "
                    f"`hermes kanban list --status ready` / "
                    f"`hermes kanban list --status blocked` for recent "
                    f"spawn_failed tasks.",
                    file=sys.stderr, flush=True,
                )
                health_state["last_warn_at"] = now
        if not verbose:
            return
        did_work = (
            res.reclaimed or res.crashed or res.timed_out or res.promoted
            or res.spawned or res.auto_blocked or res.stale
        )
        if did_work:
            print(
                f"[{_fmt_ts(int(time.time()))}] "
                f"reclaimed={res.reclaimed} crashed={len(res.crashed)} "
                f"timed_out={len(res.timed_out)} stale={len(res.stale)} "
                f"promoted={res.promoted} spawned={len(res.spawned)} "
                f"auto_blocked={len(res.auto_blocked)}",
                flush=True,
            )

    def _ready_queue_nonempty() -> bool:
        """Is there a ready+assigned+unclaimed task the dispatcher would spawn for?
        Control-plane lanes pulled via ``claim_task`` are correctly idle, not stuck."""
        try:
            with kb.connect_closing() as conn:
                return kb.has_spawnable_ready(conn)
        except Exception:
            return False

    try:
        kb.run_daemon(
            interval=args.interval,
            max_spawn=args.max,
            failure_limit=getattr(args, "failure_limit", kb.DEFAULT_SPAWN_FAILURE_LIMIT),
            on_tick=_on_tick,
        )
    finally:
        if pidfile:
            try:
                Path(pidfile).unlink()
            except OSError:
                pass
    print("(dispatcher stopped)")
    return 0


def _cmd_watch(args: argparse.Namespace) -> int:
    """Live-stream task_events to the terminal."""
    kinds = (
        {k.strip() for k in args.kinds.split(",") if k.strip()}
        if args.kinds else None
    )
    print("Watching kanban events. Ctrl-C to stop.", flush=True)
    # Seed cursor at the latest id so we don't replay history.
    with kb.connect_closing() as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(id), 0) AS m FROM task_events"
        ).fetchone()
        cursor = int(row["m"])

    try:
        while True:
            with kb.connect_closing() as conn:
                rows = conn.execute(
                    "SELECT e.id, e.task_id, e.kind, e.payload, e.created_at, "
                    "       t.assignee, t.tenant "
                    "FROM task_events e LEFT JOIN tasks t ON t.id = e.task_id "
                    "WHERE e.id > ? ORDER BY e.id ASC LIMIT 200",
                    (cursor,),
                ).fetchall()
            for r in rows:
                cursor = max(cursor, int(r["id"]))
                if kinds and r["kind"] not in kinds:
                    continue
                if args.assignee and r["assignee"] != args.assignee:
                    continue
                if args.tenant and r["tenant"] != args.tenant:
                    continue
                try:
                    payload = json.loads(r["payload"]) if r["payload"] else None
                except Exception:
                    payload = None
                pl = f" {payload}" if payload else ""
                print(
                    f"[{_fmt_ts(r['created_at'])}] {r['task_id']:10s} "
                    f"{r['kind']:18s} (@{r['assignee'] or '-'}){pl}",
                    flush=True,
                )
            time.sleep(max(0.1, args.interval))
    except KeyboardInterrupt:
        print("\n(stopped)")
        return 0


def _cmd_stats(args: argparse.Namespace) -> int:
    with kb.connect_closing() as conn:
        stats = kb.board_stats(conn)
    if _json_out(args, stats):
        return 0
    print("By status:")
    for k in ("triage", "todo", "scheduled", "ready", "running", "blocked", "done"):
        print(f"  {k:8s}  {stats['by_status'].get(k, 0)}")
    if stats["by_assignee"]:
        print("\nBy assignee:")
        for who, counts in sorted(stats["by_assignee"].items()):
            print(f"  {who:20s}  {_fmt_counts(counts)}")
    age = stats["oldest_ready_age_seconds"]
    if age is not None:
        print(f"\nOldest ready task age: {int(age)}s")
    return 0


def _cmd_notify_subscribe(args: argparse.Namespace) -> int:
    with kb.connect_closing() as conn:
        if kb.get_task(conn, args.task_id) is None:
            return _err(f"no such task: {args.task_id}")
        kb.add_notify_sub(
            conn, task_id=args.task_id,
            platform=args.platform, chat_id=args.chat_id,
            chat_type=args.chat_type,
            thread_id=args.thread_id, user_id=args.user_id,
            user_id_alt=getattr(args, "user_id_alt", None),
            notifier_profile=args.notifier_profile or _profile_author(),
            delivery_mode=getattr(args, "delivery_mode", None),
        )
    print(f"Subscribed {args.platform}:{args.chat_id}"
          + (f":{args.thread_id}" if args.thread_id else "")
          + f" to {args.task_id}")
    return 0


def _cmd_notify_list(args: argparse.Namespace) -> int:
    with kb.connect_closing() as conn:
        subs = kb.list_notify_subs(conn, args.task_id)
    if _json_out(args, subs):
        return 0
    if not subs:
        print("(no subscriptions)")
        return 0
    for s in subs:
        thr = f":{s['thread_id']}" if s.get("thread_id") else ""
        owner = f"  owner={s['notifier_profile']}" if s.get("notifier_profile") else ""
        dmode = s.get("delivery_mode") or "notify"
        mode = "" if dmode == "notify" else f"  mode={dmode}"
        ctype = s.get("chat_type") or "dm"
        ct = "" if ctype == "dm" else f"  chat_type={ctype}"
        uid_alt = f"  user_id_alt={s['user_id_alt']}" if s.get("user_id_alt") else ""
        print(f"  {s['task_id']:10s}  {s['platform']}:{s['chat_id']}{thr}"
              f"  (since event {s['last_event_id']}){owner}{ct}{uid_alt}{mode}")
    return 0


def _cmd_notify_unsubscribe(args: argparse.Namespace) -> int:
    with kb.connect_closing() as conn:
        ok = kb.remove_notify_sub(
            conn, task_id=args.task_id,
            platform=args.platform, chat_id=args.chat_id,
            thread_id=args.thread_id,
        )
    if not ok:
        return _err("(no such subscription)")
    print(f"Unsubscribed from {args.task_id}")
    return 0


def _cmd_log(args: argparse.Namespace) -> int:
    content = kb.read_worker_log(args.task_id, tail_bytes=args.tail)
    if content is None:
        return _err(f"(no log for {args.task_id} — task may not have spawned yet)")
    sys.stdout.write(content)
    if not content.endswith("\n"):
        sys.stdout.write("\n")
    return 0


def _cmd_runs(args: argparse.Namespace) -> int:
    """Show attempt history for a task."""
    rsk, rc = _run_state_kwargs(args, "runs")
    if rc:
        return rc
    with kb.connect_closing() as conn:
        runs = kb.list_runs(conn, args.task_id, **rsk)
    if _json_out(args, [_obj_dict(r, _RUNS_RUN_FIELDS) for r in runs]):
        return 0
    if not runs:
        print(f"(no runs yet for {args.task_id})")
        return 0
    print(f"{'#':3s}  {'OUTCOME':12s}  {'PROFILE':16s}  {'ELAPSED':>8s}  STARTED")
    for i, r in enumerate(runs, 1):
        end = r.ended_at or int(time.time())
        # Clamp to 0 so NTP backward-jumps don't print negative durations.
        elapsed = max(0, end - r.started_at)
        if elapsed < 60:
            el = f"{elapsed}s"
        elif elapsed < 3600:
            el = f"{elapsed // 60}m"
        else:
            el = f"{elapsed / 3600:.1f}h"
        outcome = r.outcome or ("(running)" if not r.ended_at else r.status)
        print(f"{i:3d}  {outcome:12s}  {(r.profile or '-'):16s}  {el:>8s}  {_fmt_ts(r.started_at)}")
        if r.summary:
            print(f"     → {r.summary.splitlines()[0][:100]}")
        if r.error:
            print(f"     ✖ {r.error[:100]}")
    return 0


def _cmd_context(args: argparse.Namespace) -> int:
    with kb.connect_closing() as conn:
        text = kb.build_worker_context(conn, args.task_id)
    print(text)
    return 0


def _triage_sweep_ids(args: argparse.Namespace, verb: str, list_triage_ids, json_key: str):
    """Shared arg validation for ``specify`` / ``decompose``: ``(ids|None, rc)``.

    ``ids is None`` with ``rc == 0`` means "nothing to do, already reported".
    """
    all_flag = bool(getattr(args, "all_triage", False))
    tenant = getattr(args, "tenant", None)
    if args.task_id and all_flag:
        return None, _err("kanban: pass either a task id OR --all, not both", 2)
    if all_flag:
        ids = list_triage_ids(tenant=tenant)
        if not ids:
            if getattr(args, "json", False):
                print(json.dumps({json_key: 0, "total": 0}))
            else:
                print("No triage tasks" + (f" for tenant {tenant!r}" if tenant else "") + ".")
            return None, 0
        return ids, 0
    if args.task_id:
        return [args.task_id], 0
    return None, _err(f"kanban: {verb} requires a task id or --all", 2)


def _run_triage_sweep(args: argparse.Namespace, verb: str, mod, run_one, json_key: str,
                      json_fields: tuple[str, ...], human_ok) -> int:
    """Shared driver for ``specify`` / ``decompose``: validate ids, run
    ``run_one(tid, author=...)`` per id, print JSON or human lines, exit code."""
    all_flag = bool(getattr(args, "all_triage", False))
    author = getattr(args, "author", None) or _profile_author()
    want_json = bool(getattr(args, "json", False))
    ids, rc = _triage_sweep_ids(args, verb, mod.list_triage_ids, json_key)
    if ids is None:
        return rc

    ok_count = 0
    for tid in ids:
        outcome = run_one(tid, author=author)
        if outcome.ok:
            ok_count += 1
        if want_json:
            print(json.dumps(_obj_dict(outcome, json_fields)))
        elif outcome.ok:
            print(human_ok(outcome))
        else:
            print(f"kanban: {verb} {outcome.task_id}: {outcome.reason}", file=sys.stderr)
    if not all_flag:
        return 0 if ok_count == 1 else 1
    # --all: succeed if at least one promotion landed; exit 1 only when
    # every candidate failed (honest signal for scripts).
    return 0 if (ok_count > 0 or not ids) else 1


def _retitled_suffix(outcome) -> str:
    return f" — retitled: {outcome.new_title!r}" if outcome.new_title else ""


def _cmd_specify(args: argparse.Namespace) -> int:
    """Flesh out a triage task (or all of them) via auxiliary LLM, then
    promote to todo. Thin wrapper over ``kanban_specify``."""
    from hermes_cli import kanban_specify as spec

    return _run_triage_sweep(
        args, "specify", spec, spec.specify_task, "specified",
        ("task_id", "ok", "reason", "new_title"),
        lambda o: f"Specified {o.task_id} → todo{_retitled_suffix(o)}",
    )


def _decompose_ok_line(o) -> str:
    if o.fanout and o.child_ids:
        return (f"Decomposed {o.task_id} → {len(o.child_ids)} "
                f"children ({', '.join(o.child_ids)}); root promoted to todo")
    return f"Specified {o.task_id} → todo (no fanout){_retitled_suffix(o)}"


def _cmd_decompose(args: argparse.Namespace) -> int:
    """Fan a triage task (or all of them) out into a graph of child tasks via
    the auxiliary LLM. Thin wrapper over ``kanban_decompose``."""
    from hermes_cli import kanban_decompose as decomp

    return _run_triage_sweep(
        args, "decompose", decomp, decomp.decompose_task, "decomposed",
        ("task_id", "ok", "reason", "fanout", "child_ids", "new_title"),
        _decompose_ok_line,
    )


def _cmd_gc(args: argparse.Namespace) -> int:
    """Remove archived tasks' scratch workspaces, old events, and old worker logs."""
    import shutil
    scratch_root = kb.workspaces_root()
    removed_ws = 0
    with kb.connect_closing() as conn:
        rows = conn.execute(
            "SELECT id, workspace_kind, workspace_path, branch_name FROM tasks "
            "WHERE status = 'archived'"
        ).fetchall()
    for row in rows:
        if row["workspace_kind"] == "worktree":
            # Backstop for worktrees that escaped the completion/archive hook.
            # Same safety predicate: only clean, fully-pushed worktrees go.
            wt_path = row["workspace_path"]
            if wt_path and Path(wt_path).is_dir():
                kb._cleanup_worktree_workspace(row["id"], wt_path, row["branch_name"])
                if not Path(wt_path).is_dir():
                    removed_ws += 1
            continue
        if row["workspace_kind"] != "scratch":
            continue
        path = Path(row["workspace_path"] or (scratch_root / row["id"]))
        try:
            path = path.resolve()
        except OSError:
            continue
        try:
            path.relative_to(scratch_root.resolve())
        except ValueError:
            # Safety: never delete outside the scratch root.
            continue
        if path.exists() and path.is_dir():
            shutil.rmtree(path, ignore_errors=True)
            removed_ws += 1

    event_days = getattr(args, "event_retention_days", 30)
    log_days = getattr(args, "log_retention_days", 30)
    with kb.connect_closing() as conn:
        removed_events = kb.gc_events(
            conn, older_than_seconds=event_days * 24 * 3600,
        )
    removed_logs = kb.gc_worker_logs(
        older_than_seconds=log_days * 24 * 3600,
    )
    print(f"GC complete: {removed_ws} workspace(s), "
          f"{removed_events} event row(s), {removed_logs} log file(s) removed")
    return 0


def _cmd_repair(args: argparse.Namespace) -> int:
    """Integrity check + narrow index-REINDEX auto-repair. Dispatched BEFORE
    the auto ``kb.init_db()`` (init refuses corrupt DBs). Exit 0 = healthy /
    repaired / no DB file, 1 = still corrupt."""
    try:
        report = kb.repair_db()
    except Exception as exc:  # locked/busy probe, unexpected I/O
        return _err(f"kanban repair: {exc}")

    if getattr(args, "json", False):
        _print_json({
            "status": report.status,
            "db_path": str(report.db_path),
            "messages": report.messages,
            "post_repair_messages": report.post_repair_messages,
            "backup_path": (
                str(report.backup_path) if report.backup_path else None
            ),
            "reindexed": report.reindexed,
        }, ascii=True)
        return 0 if report.status in {"ok", "repaired", "missing"} else 1

    if report.status == "missing":
        print(f"No kanban DB at {report.db_path} — nothing to repair.")
        return 0
    if report.status == "ok":
        print(f"{report.db_path}: integrity_check ok — no repair needed.")
        return 0
    if report.status == "repaired":
        print(f"{report.db_path}: repaired.")
        print(f"  reindexed: {', '.join(report.reindexed)}")
        if report.backup_path:
            print(f"  pre-repair backup: {report.backup_path}")
        print("  integrity_check now ok.")
        return 0
    # still corrupt
    print(f"{report.db_path}: CORRUPT.", file=sys.stderr)
    for line in (report.messages or [])[:10]:
        print(f"  {line}", file=sys.stderr)
    if report.reindexed:
        print(
            f"  REINDEX ({', '.join(report.reindexed)}) attempted but "
            f"integrity_check is still failing:",
            file=sys.stderr,
        )
        for line in (report.post_repair_messages or [])[:10]:
            print(f"    {line}", file=sys.stderr)
    else:
        print(
            "  Not an index-only failure — automatic REINDEX repair does "
            "not apply (fail-closed).",
            file=sys.stderr,
        )
    if report.backup_path:
        print(f"  corrupt copy quarantined at: {report.backup_path}",
              file=sys.stderr)
    print(
        "  Recover manually (e.g. `sqlite3 kanban.db \".recover\"` into a "
        "fresh file) or move the file aside to start a new board.",
        file=sys.stderr,
    )
    return 1


_HANDLERS = {
    "init": _cmd_init, "create": _cmd_create, "swarm": _cmd_swarm,
    "list": _cmd_list, "ls": _cmd_list, "show": _cmd_show,
    "assign": _cmd_assign, "set-model": _cmd_set_model,
    "reclaim": _cmd_reclaim, "reassign": _cmd_reassign,
    "diagnostics": _cmd_diagnostics, "diag": _cmd_diagnostics,
    "link": _cmd_link, "unlink": _cmd_unlink, "claim": _cmd_claim,
    "comment": _cmd_comment, "attach": _cmd_attach,
    "attachments": _cmd_attachments, "attach-rm": _cmd_attach_rm,
    "complete": _cmd_complete, "edit": _cmd_edit, "block": _cmd_block,
    "schedule": _cmd_schedule, "unblock": _cmd_unblock,
    "request-review": _cmd_request_review, "request-changes": _cmd_request_changes,
    "reopen-review": _cmd_reopen_review, "promote": _cmd_promote,
    "archive": _cmd_archive, "tail": _cmd_tail, "dispatch": _cmd_dispatch,
    "daemon": _cmd_daemon, "watch": _cmd_watch, "stats": _cmd_stats,
    "log": _cmd_log, "runs": _cmd_runs, "heartbeat": _cmd_heartbeat,
    "assignees": _cmd_assignees, "notify-subscribe": _cmd_notify_subscribe,
    "notify-list": _cmd_notify_list, "notify-unsubscribe": _cmd_notify_unsubscribe,
    "context": _cmd_context, "specify": _cmd_specify, "decompose": _cmd_decompose,
    "gc": _cmd_gc,
}


# ---------------------------------------------------------------------------
# Slash-command entry point (used by /kanban from CLI and gateway)
# ---------------------------------------------------------------------------

_SLASH_KANBAN_HELP = """\
**/kanban** — manage the shared task board.

Common subcommands:
  `list` (alias `ls`)   List tasks on the current board
  `show <id>`           Task details + comments + events
  `stats`               Per-status / per-assignee counts
  `create <title>…`     Create a task (auto-subscribes you to events)
  `comment <id> <msg>`  Append a comment
  `attach <id> <path>`  Attach a local file; `attachments <id>` to list
  `complete <id>…`      Mark task(s) done
  `request-review <id>` Enter first-class review; `request-changes <id> <reason>` returns an active review to its implementer
  `block <id> [reason]` Mark blocked; `schedule <id> [reason]` parks time-delay work; `unblock <id>` to revive
  `assign <id> <profile>`  Reassign
  `boards list`         Show all boards
  `assignees`           Known profiles + counts
  `context <id>`        Full worker-context dump
  `runs <id>`           Attempt history
  `log <id>`            Worker log

Run `/kanban <subcommand> -h` for arguments. \
Read-only commands are safe while an agent is running.\
"""


def run_slash(rest: str) -> str:
    """Execute a ``/kanban …`` string and return captured stdout/stderr.

    ``rest`` is everything after ``/kanban``. Shared by the interactive CLI
    and the gateway so formatting is identical.
    """
    import io

    tokens = shlex.split(rest) if rest and rest.strip() else []

    # Bare ``/kanban`` / ``help`` / ``-h``: the curated short block, not
    # argparse's full usage tree (garbage in a chat bubble). Per-subcommand
    # help still works via ``/kanban foo -h``.
    if not tokens or tokens[0] in {"help", "--help", "-h", "?"}:
        return _SLASH_KANBAN_HELP

    # build_parser() needs a subparsers action to attach to, so build a
    # throwaway one and pull kanban_parser back out; drive it directly so
    # usage/error text reads as ``/kanban`` (not ``/kanban-wrap kanban``).
    _wrap = argparse.ArgumentParser(prog="/kanban-wrap", add_help=False)
    _wrap.exit_on_error = False  # type: ignore[attr-defined]
    _top_sub = _wrap.add_subparsers(dest="_top")
    kanban_parser = build_parser(_top_sub)
    kanban_parser.prog = "/kanban"
    kanban_parser.exit_on_error = False  # type: ignore[attr-defined]
    for _action in kanban_parser._actions:
        if isinstance(_action, argparse._SubParsersAction):
            for _name, _choice in _action.choices.items():
                _choice.prog = f"/kanban {_name}"
                _choice.exit_on_error = False  # type: ignore[attr-defined]

    def _usage_for_error() -> str:
        if tokens:
            for _action in kanban_parser._actions:
                if isinstance(_action, argparse._SubParsersAction):
                    subparser = _action.choices.get(tokens[0])
                    if subparser is not None:
                        return subparser.format_usage().rstrip()
        return kanban_parser.format_usage().rstrip()

    buf_out = io.StringIO()
    buf_err = io.StringIO()
    # ``-h`` prints to stdout and SystemExit(0); capture both streams.
    try:
        with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
            args = kanban_parser.parse_args(tokens)
    except SystemExit as exc:
        out = buf_out.getvalue().rstrip()
        err = buf_err.getvalue().rstrip()
        # Help dump (exit 0) → return the captured help text directly.
        if exc.code in {0, None} and out:
            return out
        body = err or out
        return f"⚠ /kanban usage error\n{body}" if body else "⚠ /kanban usage error"
    except argparse.ArgumentError as exc:
        return f"⚠ /kanban usage error\n{_usage_for_error()}\n{exc}"

    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        try:
            kanban_command(args)
        except SystemExit:
            pass
        except Exception as exc:
            print(f"error: {exc}", file=sys.stderr)

    out = buf_out.getvalue().rstrip()
    err = buf_err.getvalue().rstrip()
    if err and out:
        return f"{out}\n{err}"
    return err if err else (out or "(no output)")
