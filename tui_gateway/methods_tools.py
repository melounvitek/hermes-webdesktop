"""Tools & system / slash / insights / rollback / plugins / cron / skills / MCP JSON-RPC handlers.

Rebound onto server.py's globals at install time (``method_ctx.bind_module``), so
bodies reference server globals bare (``_ok``, ``_err``, ``_sessions``, ...).
Helper names must not collide with server.py's own (``_cmd_`` / ``_toolset_`` / ``_mcp_`` prefixes).
"""

import sys

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped


# ─── Shared helpers ──────────────────────────────────────────────────────────


def _profile_scoped_rpc(fail_code: int, *, required=(), catch_resolve: bool = True, prefix: str = "", scoped: bool = True):
    """Wrap a handler body with the optional ``profile`` HERMES_HOME scope.

    Order: ``required`` params checked first (4063 ``<key> required``), then the
    profile resolved (4064 when its dir is missing), then the body; body exceptions
    become ``fail_code`` (message prefixed with ``prefix``). ``catch_resolve`` also maps
    resolve-time exceptions to ``fail_code`` (cron/skills/catalog); mcp.servers.* let
    them propagate to dispatch(). The override is always reset afterwards.
    ``scoped=False`` (see ``_guarded``) ignores ``profile`` entirely.
    """

    def deco(body):
        def handler(rid, params: dict) -> dict:
            for key, present in required:
                if not present(params.get(key)):
                    return _err(rid, 4063, f"{key} required")
            profile = str(params.get("profile") or "").strip() if scoped else ""
            token = None
            if profile:
                try:
                    from hermes_cli.profiles import get_profile_dir
                    from hermes_constants import set_hermes_home_override

                    profile_dir = get_profile_dir(profile)
                    if not profile_dir or not profile_dir.is_dir():
                        return _err(rid, 4064, f"profile '{profile}' not found")
                    token = set_hermes_home_override(str(profile_dir))
                except Exception as e:
                    if not catch_resolve:
                        raise
                    return _err(rid, fail_code, str(e))
            try:
                return body(rid, params)
            except Exception as e:
                return _err(rid, fail_code, f"{prefix}{e}")
            finally:
                _mcp_reset_profile(token)

        handler.__doc__ = body.__doc__
        return handler

    return deco


def _guarded(fail_code: int, prefix: str = ""):
    """Handler body exceptions → ``_err(rid, fail_code, prefix + str(e))``."""
    return _profile_scoped_rpc(fail_code, prefix=prefix, scoped=False)


def _stripped(v) -> bool:
    return bool(str(v or "").strip())


def _nonempty(v) -> bool:
    return not (v is None or str(v) == "")


_NAME = (("name", _stripped),)
_NAME_SESSION = (("name", _stripped), ("session_id", _stripped))


def _mcp_server_scoped(body):
    """mcp.servers.* contract: ``name`` required, profile scope, body errors → 5024."""
    return _profile_scoped_rpc(5024, required=_NAME, catch_resolve=False)(body)


def _mcp_named_server(rid, params):
    """(name, servers, None) for a configured server, else (name, servers, 4064 error)."""
    from hermes_cli.mcp_config import _get_mcp_servers

    name = str(params.get("name") or "").strip()
    servers = _get_mcp_servers()
    err = None if name in servers else _err(rid, 4064, f"server '{name}' not found")
    return name, servers, err


def _busy_error(rid, session, cmd: str):
    if session.get("running"):
        return _err(rid, 4009, f"session busy — /interrupt the current turn before /{cmd}")
    return None


def _session_key_or_err(rid, session):
    """(session_key, None) or (None, 4001 error) for the /goal and /loop managers."""
    if not session:
        return None, _err(rid, 4001, "no active session")
    sid_key = session.get("session_key") or ""
    if not sid_key:
        return None, _err(rid, 4001, "no session key")
    return sid_key, None


def _user_turn_indices(session):
    """(history, indices of user-originated turns) minus ephemeral scaffolding. Call under history_lock."""
    from agent.context_compressor import user_originated_turn_view

    history = _history_without_ephemeral_scaffolding(session.get("history", []))
    return history, [i for i, m in enumerate(history) if user_originated_turn_view(m) is not None]


def _clip(text: str, n: int = 120) -> str:
    return text[:n] + ("…" if len(text) > n else "")


def _capture_run_kwargs(timeout: int) -> dict:
    """subprocess.run kwargs shared by cli.exec / shell.exec / quick commands: captured
    text, UTF-8 + lossy decode (non-UTF-8 child output must not crash the gateway thread
    on locale-mismatched Windows), no stdin, no console flash under the desktop parent."""
    from hermes_cli._subprocess_compat import windows_hide_flags

    return dict(
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
        stdin=subprocess.DEVNULL,
        creationflags=windows_hide_flags(),
    )


def _toolset_rows(params: dict, *, with_tools: bool) -> list[dict]:
    from toolsets import get_all_toolsets, get_toolset_info

    session = _sessions.get(params.get("session_id", ""))
    enabled = (
        set(getattr(session["agent"], "enabled_toolsets", []) or []) if session else set(_load_enabled_toolsets() or [])
    )
    items = []
    for name in sorted(get_all_toolsets().keys()):
        info = get_toolset_info(name)
        if not info:
            continue
        row = {
            "name": name,
            "description": info["description"],
            "tool_count": info["tool_count"],
            "enabled": name in enabled if enabled else True,
        }
        if with_tools:
            row["tools"] = info["resolved_tools"]
        items.append(row)
    return items


# ─── System / process ────────────────────────────────────────────────────────


@method("system.battery")
def _(rid, params: dict) -> dict:
    """Host battery for the status bar. Always resolves; ``available: false`` = no battery or read failed."""
    try:
        from agent.battery import battery_category, read_battery

        batt = read_battery()
        return _ok(
            rid,
            {
                "available": batt.available,
                "percent": batt.percent,
                "plugged": batt.plugged,
                "category": battery_category(batt),
            },
        )
    except Exception:
        return _ok(rid, {"available": False, "percent": None, "plugged": None, "category": "dim"})


@method("process.stop")
@_guarded(5010)
def _(rid, params: dict) -> dict:
    from tools.process_registry import process_registry

    return _ok(rid, {"killed": process_registry.kill_all()})


@method("process.list")
def _(rid, params: dict) -> dict:
    """Session-scoped view of the background process registry (desktop status stack)."""
    session, err = _sess(params, rid)
    if err:
        return err
    try:
        return _ok(rid, {"processes": _session_processes(session)})
    except Exception as e:
        return _err(rid, 5010, str(e))


@method("process.kill")
def _(rid, params: dict) -> dict:
    """Kill ONE background process, scoped to the caller's session (unlike process.stop's kill_all)."""
    session, err = _sess(params, rid)
    if err:
        return err
    proc_id = str(params.get("process_id") or "")
    if not proc_id:
        return _err(rid, 4012, "process_id required")
    try:
        from tools.process_registry import process_registry

        proc = process_registry.get(proc_id)
        if proc is None or str(getattr(proc, "session_key", "") or "") != str(session.get("session_key") or ""):
            return _err(rid, 4044, f"no such process: {proc_id}")
        return _ok(rid, process_registry.kill_process(proc_id))
    except Exception as e:
        return _err(rid, 5010, str(e))


def _mcp_reload_confirm_required() -> bool:
    """``approvals.mcp_reload_confirm`` from disk config; True (safe) on any failure."""
    try:
        from hermes_cli.config import load_config

        cfg = load_config()
        approvals = cfg.get("approvals") if isinstance(cfg, dict) else None
        return bool(approvals.get("mcp_reload_confirm", True)) if isinstance(approvals, dict) else True
    except Exception:
        return True


@method("reload.mcp")
@_guarded(5015)
def _(rid, params: dict) -> dict:
    session = _sessions.get(params.get("session_id", ""))
    # /reload-mcp invalidates the prompt cache: without confirm=true, honour
    # ``approvals.mcp_reload_confirm`` (default true) by returning confirm_required;
    # Ink prints ``message`` and re-invokes with confirm=true (or flips the config).
    if not bool(params.get("confirm", False)) and _mcp_reload_confirm_required():
        message = (
            "⚠️  /reload-mcp invalidates the prompt cache (next message re-sends full input tokens). "
            "Reply `/reload-mcp now` to proceed, or `/reload-mcp always` to proceed and "
            "silence this prompt permanently."
        )
        return _ok(rid, {"status": "confirm_required", "message": message})

    if session and _session_uses_compute_host(session):
        try:
            ack = _get_compute_host_supervisor().reload_mcp(
                str(params.get("session_id") or ""), request_id=f"reload-mcp-{rid}"
            )
        except Exception as exc:
            return _err(rid, 5019, f"compute-host reload_mcp failed: {exc}")
        return _ok(rid, {"status": "reloaded", "turn_isolation": True, "host_ack": ack})

    from tools.mcp_tool import shutdown_mcp_servers, discover_mcp_tools, reprobe_tool_availability

    def _refresh_session_agent() -> None:
        """Rebuild THIS session's cached tool snapshot from the live registry and push
        session.info (the agent never re-reads the registry itself; mirrors
        gateway/run.py::_execute_mcp_reload). Runs under _mcp_reload_lock so a
        concurrent reload can't tear the registry down mid-refresh."""
        if not session:
            return
        agent = session["agent"]
        try:
            from tools.mcp_tool import refresh_agent_mcp_tools

            # enabled_override re-resolves toolsets so a server enabled in config this session is picked up.
            refresh_agent_mcp_tools(agent, enabled_override=_load_enabled_toolsets(), quiet_mode=True)
        except Exception as _exc:
            logger.warning("Failed to refresh cached agent tools after /reload-mcp: %s", _exc)
        _emit("session.info", params.get("session_id", ""), _session_info(agent, session))

    global _mcp_reload_gen, _mcp_reload_loaded_rev

    # Revision the CALLER wants loaded (the mcp_rev its poll observed); empty on
    # legacy clients / manual /reload-mcp, which coalesce on generation alone.
    req_rev = str(params.get("rev") or "")

    def _do_full_reload() -> None:
        """shutdown+discover+refresh under the lock, then mark a completed generation.
        The lock spans the refresh too, else a second reload could tear the registry
        down mid-rebuild. Config can change WHILE discover connects: re-hash after
        discovery and repeat until stable so the marked generation matches what loaded."""
        global _mcp_reload_gen, _mcp_reload_loaded_rev

        loaded = _compute_mcp_rev()
        for _ in range(_MCP_RELOAD_MAX_PASSES):
            shutdown_mcp_servers()
            reprobe_tool_availability()
            discover_mcp_tools()
            after = _compute_mcp_rev()
            if after == loaded:
                break
            loaded = after

        _refresh_session_agent()
        _mcp_reload_loaded_rev = loaded
        _mcp_reload_gen += 1

    # LEADER (won the non-blocking acquire) runs the full reload. FOLLOWER snapshots
    # the generation, waits, then — still holding the lock — coalesces only if a
    # reload COMPLETED meanwhile (generation advanced ⇒ leader didn't throw) AND it
    # loaded the requested revision; otherwise it re-runs the full reload so a
    # failed/stale leader never leaves a follower acking an unloaded revision.
    if _mcp_reload_lock.acquire(blocking=False):
        try:
            _do_full_reload()
        finally:
            _mcp_reload_lock.release()

        return _finish_reload(rid, params, coalesced=False)

    gen_before = _mcp_reload_gen

    with _mcp_reload_lock:
        leader_completed = _mcp_reload_gen > gen_before
        rev_satisfied = not req_rev or req_rev == _mcp_reload_loaded_rev

        if leader_completed and rev_satisfied:
            _refresh_session_agent()
            coalesced = True
        else:
            _do_full_reload()
            coalesced = False

    return _finish_reload(rid, params, coalesced=coalesced)


@method("reload.env")
@_guarded(5015)
def _(rid, params: dict) -> dict:
    """Re-read ``~/.hermes/.env`` (classic CLI ``/reload`` parity). Already-built agents
    keep their credential pool / provider routing; ``/new`` resolves fresh."""
    from hermes_cli.config import reload_env

    return _ok(rid, {"updated": int(reload_env())})


# ─── Command catalog / dispatch ──────────────────────────────────────────────


@method("commands.catalog")
@_guarded(5020)
def _(rid, params: dict) -> dict:
    """Registry-backed slash metadata for the TUI — categorized, no aliases."""
    from hermes_cli.commands import COMMAND_REGISTRY, SUBCOMMANDS, _build_description, command_desktop_meta

    all_pairs: list[list[str]] = []
    canon: dict[str, str] = {}
    commands: dict[str, dict[str, str | None]] = {}
    cat_map: dict[str, list[list[str]]] = {}
    cat_order: list[str] = []

    def bucket(cat: str) -> list[list[str]]:
        if cat not in cat_map:
            cat_map[cat] = []
            cat_order.append(cat)
        return cat_map[cat]

    def add(key: str, desc: str, rows: list[list[str]]) -> None:
        canon[key.lower()] = key
        all_pairs.append([key, desc])
        rows.append([key, desc])

    for cmd in COMMAND_REGISTRY:
        meta = command_desktop_meta(cmd)
        commands[f"/{cmd.name}"] = dict(meta)
        for alias in cmd.aliases:
            commands[f"/{alias}"] = dict(meta)
        if cmd.name in _TUI_HIDDEN or cmd.gateway_only:
            continue
        c = f"/{cmd.name}"
        add(c, _build_description(cmd), bucket(cmd.category))
        for a in cmd.aliases:
            canon[f"/{a}".lower()] = c

    for name, desc, cat in _TUI_EXTRA:
        # Registry command/alias wins over a colliding TUI extra (e.g. /compact, /sessions).
        if name.lower() not in canon:
            add(name, desc, bucket(cat))

    warning = ""
    try:
        qcmds = _load_cfg().get("quick_commands", {}) or {}
        if isinstance(qcmds, dict) and qcmds:
            rows = bucket("User commands")
            for qname, qc in sorted(qcmds.items()):
                if not isinstance(qc, dict):
                    continue
                qtype = qc.get("type", "")
                default_desc = {
                    "exec": f"exec: {qc.get('command', '')}",
                    "alias": f"alias → {qc.get('target', '')}",
                }.get(qtype, qtype or "quick command")
                add(f"/{qname}", _clip(str(qc.get("description") or default_desc)), rows)
    except Exception as e:
        warning = f"quick_commands discovery unavailable: {e}"

    try:
        from hermes_cli.plugins import get_plugin_commands

        plugin_cmds = get_plugin_commands() or {}
        if plugin_cmds:
            rows = bucket("Plugin commands")
            for pname, info in sorted(plugin_cmds.items()):
                if not isinstance(info, dict):
                    continue
                key = f"/{pname}"
                if key.lower() in canon:
                    continue
                add(key, _clip(str(info.get("description") or "Plugin command")), rows)
                hint = str(info.get("args_hint") or "").strip()
                mode = info.get("argument_mode")
                if mode not in {"options", "text", "mixed"}:
                    mode = "text" if hint else None
                commands[key] = {"argument_mode": mode, "desktop": None}
    except Exception as e:
        if not warning:
            warning = f"plugin command discovery unavailable: {e}"

    skill_count = 0
    skills: dict[str, dict] = {}
    try:
        from agent.skill_commands import scan_skill_commands

        # Usage + origin ride along (not a second RPC): every catalog consumer also ranks it.
        usage, origin_of = _skill_usage_lookup()

        for k, info in sorted(scan_skill_commands().items()):
            all_pairs.append([k, _clip(str(info.get("description", "Skill")))])
            name = str(info.get("name") or k.lstrip("/"))
            skills[k] = {"usage": usage(name), "origin": origin_of(name)}
            skill_count += 1
    except Exception as e:
        warning = f"skill discovery unavailable: {e}"

    payload = {
        "pairs": all_pairs,
        "sub": {k: v[:] for k, v in SUBCOMMANDS.items()},
        "canon": canon,
        "commands": commands,
        "categories": [{"name": cat, "pairs": cat_map[cat]} for cat in cat_order],
        "skills": skills,
        "skill_count": skill_count,
        "warning": warning,
    }
    return _ok(rid, payload)


@method("cli.exec")
def _(rid, params: dict) -> dict:
    """Run `python -m hermes_cli.main` with argv; capture stdout/stderr (non-interactive only)."""
    argv = params.get("argv", [])
    if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
        return _err(rid, 4003, "argv must be list[str]")
    hint = _cli_exec_blocked(argv)
    if hint:
        return _ok(rid, {"blocked": True, "hint": hint, "code": -1, "output": ""})
    try:
        r = subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", *argv],
            cwd=os.getcwd(),
            # Can drive the agent → needs provider credentials; tier-1 secrets still stripped.
            env=hermes_subprocess_env(inherit_credentials=True),
            **_capture_run_kwargs(min(int(params.get("timeout", 240)), 600)),
        )
        parts = [r.stdout or "", r.stderr or ""]
        out = "\n".join(p for p in parts if p).strip() or "(no output)"
        return _ok(rid, {"blocked": False, "code": r.returncode, "output": out[:48_000]})
    except subprocess.TimeoutExpired:
        return _err(rid, 5016, "cli.exec: timeout")
    except Exception as e:
        return _err(rid, 5017, str(e))


@method("command.resolve")
@_guarded(5012)
def _(rid, params: dict) -> dict:
    from hermes_cli.commands import resolve_command

    r = resolve_command(params.get("name", ""))
    if r:
        return _ok(rid, {"canonical": r.name, "description": r.description, "category": r.category})
    return _err(rid, 4011, f"unknown command: {params.get('name')}")


# command.dispatch stages. Each takes (rid, params, session, name, arg) and
# returns a JSON-RPC envelope, or None to fall through to the next stage.


def _dispatch_quick(rid, params, session, name, arg):
    qcmds = _load_cfg().get("quick_commands", {})
    if name not in qcmds:
        return None
    qc = qcmds[name]
    if qc.get("type") == "exec":
        # Sanitized env: the TUI server process holds every API key in os.environ.
        from tools.environments.local import build_subprocess_env

        sanitized_env = build_subprocess_env()
        r = subprocess.run(qc.get("command", ""), shell=True, env=sanitized_env, **_capture_run_kwargs(30))
        output = ((r.stdout or "") + ("\n" if r.stdout and r.stderr else "") + (r.stderr or "")).strip()[:4000]
        if output:
            from agent.redact import redact_sensitive_text

            output = redact_sensitive_text(output)
        if r.returncode != 0:
            return _err(rid, 4018, output or f"quick command failed with exit code {r.returncode}")
        return _ok(rid, {"type": "exec", "output": output})
    if qc.get("type") == "alias":
        return _ok(rid, {"type": "alias", "target": qc.get("target", "")})
    return None


def _plugin_command_handler(name: str):
    try:
        from hermes_cli.plugins import get_plugin_command_handler

        return get_plugin_command_handler(name)
    except Exception:
        return None


def _is_profile_skill_command(session: dict, base: str) -> bool:
    """True when ``/base`` is a skill command of the session's profile. HERMES_HOME is bound
    to that profile so get_skill_commands() sees its skills.external_dirs: dispatch() runs on
    the pool and nothing upstream binds the override. False on any failure."""
    try:
        from agent.skill_commands import get_skill_commands
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        profile_home = session.get("profile_home")
        token = set_hermes_home_override(profile_home) if profile_home else None
        try:
            return f"/{base}" in get_skill_commands()
        finally:
            if token is not None:
                reset_hermes_home_override(token)
    except Exception:
        return False


def _dispatch_plugin(rid, params, session, name, arg):
    handler = _plugin_command_handler(name)
    if handler:
        try:
            from hermes_cli.plugins import resolve_plugin_command_result

            result = resolve_plugin_command_result(handler(arg))
            return _ok(rid, {"type": "plugin", "output": str(result or "")})
        except Exception:
            pass
    return None


def _bundle_key_for(name: str):
    """Skill-bundle key for ``name`` when it is NOT a registry command; None otherwise / on failure."""
    try:
        from agent.skill_bundles import resolve_bundle_command_key
        from hermes_cli.commands import resolve_command

        return resolve_bundle_command_key(name) if resolve_command(name) is None else None
    except Exception:
        return None


def _dispatch_bundle(rid, params, session, name, arg):
    bundle_key = _bundle_key_for(name)
    if bundle_key is None:
        return None
    from agent.skill_bundles import build_bundle_invocation_message, get_skill_bundles

    try:
        bundle_result = build_bundle_invocation_message(
            bundle_key,
            arg,
            task_id=session.get("session_key", "") if session else "",
            platform=_resolve_session_platform(),
        )
    except Exception as exc:
        return _err(rid, 4018, f"bundle dispatch failed: {exc}")
    if not bundle_result:
        return _err(rid, 4018, f"failed to load bundle: {bundle_key}")

    msg, loaded_names, missing = bundle_result
    bundle_name = get_skill_bundles().get(bundle_key, {}).get("name", bundle_key.lstrip("/"))
    notice = f"⚡ Loading bundle: {bundle_name} ({len(loaded_names)} skills)"
    if missing:
        notice += f"\nSkipped missing skills: {', '.join(missing)}"
    # UIs render `display`, never `message`: the expanded body is model-facing scaffolding.
    return _ok(rid, {"type": "send", "message": msg, "notice": notice, "display": _skill_scaffold_projection(msg)})


def _dispatch_skill(rid, params, session, name, arg):
    try:
        from agent.skill_commands import scan_skill_commands, build_skill_invocation_message

        cmds = scan_skill_commands()
        key = f"/{name}"
        if key in cmds:
            msg = build_skill_invocation_message(key, arg, task_id=session.get("session_key", "") if session else "")
            if msg:
                # UIs render `display`, never `message`.
                display = _skill_scaffold_projection(msg)
                return _ok(rid, {"type": "skill", "message": msg, "name": cmds[key].get("name", name), "display": display})
    except Exception:
        pass
    return None


# Built-ins that queue onto _pending_input in the CLI; the TUI slash worker has no
# reader for that queue, so they are handled here and return a structured payload.


def _cmd_queue(rid, params, session, name, arg):
    if not arg:
        return _err(rid, 4004, "usage: /queue <prompt>")
    return _ok(rid, {"type": "send", "message": arg})


def _cmd_learn(rid, params, session, name, arg):
    # Submitted as a normal turn; the live agent gathers sources and authors the skill via skill_manage.
    from agent.learn_prompt import build_learn_prompt

    return _ok(rid, {"type": "send", "message": build_learn_prompt(arg)})


def _cmd_plan(rid, params, session, name, arg):
    # Normal turn (as /learn); the agent saves the plan under .hermes/plans/ via write_file.
    from agent.plan_prompt import build_plan_prompt

    return _ok(rid, {"type": "send", "message": build_plan_prompt(arg)})


def _cmd_init(rid, params, session, name, arg):
    # Generate-or-update AGENTS.md as a normal turn (as /learn).
    from hermes_cli.init_command import build_init_prompt_for_cwd

    return _ok(rid, {"type": "send", "message": build_init_prompt_for_cwd(extra=arg)})


def _cmd_moa(rid, params, session, name, arg):
    # One prompt through the default MoA preset, then restore the prior model. Whole-session
    # switching goes through the model picker (MoA presets = virtual "Mixture of Agents" provider).
    try:
        from hermes_cli.moa_config import moa_usage, normalize_moa_config

        if not arg:
            return _err(rid, 4004, moa_usage())
        if not session:
            return _err(rid, 4001, "no active session")
        sid = params.get("session_id", "")
        preset = normalize_moa_config(_load_cfg().get("moa") or {})["default_preset"]
        # Record the live identity for post-turn restore, then swap the agent's client in
        # place: session["model_override"] alone never switches an already-built agent.
        agent = session.get("agent")
        session["moa_one_shot_restore"] = {
            "override": session.get("model_override"),
            "model": getattr(agent, "model", None) if agent else None,
            "provider": getattr(agent, "provider", None) if agent else None,
        }
        if agent is not None:
            try:
                _apply_model_switch(
                    sid,
                    session,
                    f"{preset} --provider moa",
                    confirm_expensive_model=False,
                    pin_session_override=True,
                    persist_override=False,  # turn-scoped: never persist the MoA provider to config.yaml
                )
            except Exception as exc:
                session.pop("moa_one_shot_restore", None)
                return _err(rid, 5030, f"moa unavailable: {exc}")
        else:
            # Lazy/fresh session: the override is consumed by the first build.
            session["model_override"] = {
                "provider": "moa",
                "model": preset,
                "base_url": "moa://local",
                "api_key": "moa-virtual-provider",
                "api_mode": "chat_completions",
            }
        notice = f"MoA one-shot queued with preset {preset}; previous model will be restored after this turn."
        return _ok(rid, {"type": "send", "notice": notice, "message": arg})
    except Exception as exc:
        return _err(rid, 5030, f"moa unavailable: {exc}")


def _cmd_focus(rid, params, session, name, arg):
    # Display-only; routed through the config.set branch Ink uses so both surfaces share one state machine.
    from hermes_cli.focus_view import format_focus_status, format_focus_toggle_message, resolve_focus_arg

    _display_focus = _load_cfg().get("display")
    _d_focus: dict = _display_focus if isinstance(_display_focus, dict) else {}
    _cur_focus = bool(_d_focus.get("focus_view", False))
    _action, _target = resolve_focus_arg(arg, _cur_focus)
    if _action == "usage":
        return _err(rid, 4004, "usage: /focus [on|off|status]")
    if _action == "status":
        _saved = _d_focus.get("focus_saved_tool_progress") or _load_tool_progress_mode()
        return _ok(rid, {"type": "exec", "output": format_focus_status(_cur_focus, _saved)})
    _res = _methods["config.set"](
        rid, {"key": "focus", "value": "on" if _target else "off", "session_id": params.get("session_id", "")}
    )
    if "error" in _res:
        return _res
    _payload = _res.get("result") or {}
    output = format_focus_toggle_message(bool(_target), _payload.get("tool_progress") or "all")
    return _ok(rid, {"type": "exec", "output": output})


def _cmd_retry(rid, params, session, name, arg):
    if not session:
        return _err(rid, 4001, "no active session to retry")
    if busy := _busy_error(rid, session, "retry"):
        return busy
    from agent.context_compressor import history_before_user_originated_turn, retryable_user_text

    with session["history_lock"]:
        if busy := _busy_error(rid, session, "retry"):
            return busy
        if session.get("attached_images"):
            return _err(rid, 4018, "retry cannot safely reconstruct or combine attached media")
        history, user_indices = _user_turn_indices(session)
        if not user_indices:
            return _err(rid, 4018, "no previous user message to retry")
        _prefix, live_view = history_before_user_originated_turn(history, user_indices[-1])
        try:
            content = retryable_user_text(live_view.get("content"))
        except ValueError as exc:
            return _err(rid, 4018, str(exc))
        try:
            _active, durable_live_view, _rewound_count = _rewind_active_session_history(
                session, len(user_indices) - 1, require_retryable=True
            )
        except ValueError as exc:
            return _err(rid, 4018, str(exc))
        except Exception as exc:
            return _err(rid, 5008, f"retry: failed to persist history: {exc}")
        content = retryable_user_text(durable_live_view.get("content"))
    return _ok(rid, {"type": "send", "message": content})


def _cmd_steer(rid, params, session, name, arg):
    if not arg:
        return _err(rid, 4004, "usage: /steer <prompt>")
    agent = session.get("agent") if session else None
    if agent and hasattr(agent, "steer"):
        try:
            if agent.steer(arg):
                shown = f"{arg[:80]}{'...' if len(arg) > 80 else ''}"
                return _ok(rid, {"type": "exec", "output": f"⏩ Steer queued — arrives after the next tool call: {shown}"})
        except Exception:
            pass
    # No active run: treat as next-turn message.
    return _ok(rid, {"type": "send", "message": arg})


def _cmd_goal(rid, params, session, name, arg):
    if not session:
        return _err(rid, 4001, "no active session")
    try:
        from hermes_cli.goals import GoalManager
    except Exception as exc:
        return _err(rid, 5030, f"goals unavailable: {exc}")
    sid_key, err = _session_key_or_err(rid, session)
    if err:
        return err
    try:
        max_turns = int((_load_cfg().get("goals") or {}).get("max_turns", 20) or 20)
    except Exception:
        max_turns = 20
    mgr = GoalManager(session_id=sid_key, default_max_turns=max_turns)

    lower = arg.strip().lower()
    if not arg.strip() or lower == "status":
        return _ok(rid, {"type": "exec", "output": mgr.status_line()})
    if lower == "pause":
        state = mgr.pause(reason="user-paused")
        out = "No goal set." if state is None else f"⏸ Goal paused: {state.goal}"
        return _ok(rid, {"type": "exec", "output": out})
    if lower == "resume":
        state = mgr.resume()
        if state is None:
            return _ok(rid, {"type": "exec", "output": "No goal to resume."})
        # Resume must restart work: `exec` is display-only, so return a `send` with the
        # continuation prompt; `display` keeps model-facing scaffolding out of the transcript.
        prompt = mgr.next_continuation_prompt()
        if not prompt:
            return _ok(rid, {"type": "exec", "output": f"▶ Goal resumed: {state.goal}"})
        notice = f"▶ Goal resumed: {state.goal}\nContinuing now — taking the next step."
        return _ok(rid, {"type": "send", "notice": notice, "message": prompt, "display": "/goal resume"})
    if lower in {"clear", "stop", "done"}:
        had = mgr.has_goal()
        mgr.clear()
        return _ok(rid, {"type": "exec", "output": "✓ Goal cleared." if had else "No active goal."})

    # Remaining text = new goal. Client renders `notice`, submits `message`; the post-turn judge takes over.
    try:
        state = mgr.set(arg)
    except ValueError as exc:
        return _err(rid, 4004, f"invalid goal: {exc}")
    notice = (
        f"⊙ Goal set ({state.max_turns}-turn budget): {state.goal}\n"
        "I'll keep working until the goal is done, you pause/clear it, or the budget is exhausted.\n"
        "Controls: /goal status · /goal pause · /goal resume · /goal clear"
    )
    return _ok(rid, {"type": "send", "notice": notice, "message": state.goal})


def _cmd_loop(rid, params, session, name, arg):
    # Recurring in-session wakeups; the notification poller fires due ones while the session is idle.
    if not session:
        return _err(rid, 4001, "no active session")
    try:
        from hermes_cli.loops import LoopManager, dispatch_loop_command
    except Exception as exc:
        return _err(rid, 5030, f"loops unavailable: {exc}")
    sid_key, err = _session_key_or_err(rid, session)
    if err:
        return err
    result = dispatch_loop_command(LoopManager(session_id=sid_key), arg)
    output = result.get("output") or ""
    if result.get("created"):
        with contextlib.suppress(Exception):
            from hermes_cli.loops import goal_blocks_loop_tick

            if goal_blocks_loop_tick(sid_key):
                output += (
                    "\nNote: an active /goal is driving this session — loop "
                    "wakeups defer until the goal finishes, pauses, or parks."
                )
    return _ok(rid, {"type": "exec", "output": output})


def _cmd_undo(rid, params, session, name, arg):
    # /undo [N]: back up N user turns, soft-delete truncated rows on disk, prefill the composer.
    if not session:
        return _err(rid, 4001, "no active session to undo")
    if busy := _busy_error(rid, session, "undo"):
        return busy
    session_key = session.get("session_key", "")
    if not session_key:
        return _err(rid, 4001, "no session key for undo")
    n = 1
    arg_str = (arg or "").strip()
    if arg_str:
        try:
            n = int(arg_str.split()[0])
        except (ValueError, IndexError):
            return _err(rid, 4004, f"undo: invalid count {arg_str!r} — use /undo or /undo N")
    n = max(n, 1)
    from agent.message_content import flatten_message_text

    with session["history_lock"]:
        if busy := _busy_error(rid, session, "undo"):
            return busy
        _history, user_indices = _user_turn_indices(session)
        if not user_indices:
            return _err(rid, 4018, "no user messages to undo")
        turns_undone = min(n, len(user_indices))
        try:
            active, live_view, rewound_count = _rewind_active_session_history(session, len(user_indices) - turns_undone)
        except ValueError as exc:
            return _err(rid, 4004, f"undo: {exc}")
        except Exception as exc:
            return _err(rid, 5008, f"undo: {exc}")
        target_text = flatten_message_text(live_view.get("content"))
    # Notify memory providers (same hook /branch fires) with rewound=True so cached per-turn state invalidates.
    agent = session.get("agent")
    if agent is not None:
        mm = getattr(agent, "_memory_manager", None)
        if mm is not None:
            with contextlib.suppress(Exception):
                mm.on_session_switch(session_key, parent_session_id="", reset=False, rewound=True)
        if hasattr(agent, "_invalidate_system_prompt"):
            with contextlib.suppress(Exception):
                agent._invalidate_system_prompt()
        if hasattr(agent, "_last_flushed_db_idx"):
            with contextlib.suppress(Exception):
                agent._last_flushed_db_idx = len(active)
    turn_word = "turn" if turns_undone == 1 else "turns"
    notice = f"↶ Undid {turns_undone} {turn_word} ({rewound_count} message(s)). Edit and resubmit, or send a new message."
    return _ok(rid, {"type": "prefill", "message": target_text, "notice": notice})


def _cmd_snapshot(rid, params, session, name, arg):
    subcommand = arg.split(maxsplit=1)[0].lower() if arg else ""
    if subcommand not in {"restore", "rewind"}:
        return None
    output = (
        "/snapshot restore is blocked in the TUI because it changes config/state on disk "
        "while the live agent has cached settings. Run it in the classic CLI, then restart the TUI."
    )
    return _ok(rid, {"type": "exec", "output": output})


def _cmd_compress(rid, params, session, name, arg):
    if not session:
        return _err(rid, 4001, "no active session to compress")
    if busy := _busy_error(rid, session, "compress"):
        return busy
    from agent.conversation_compression import finalize_context_engine_compression_notification

    sid = params.get("session_id", "")
    if _session_uses_compute_host(session):
        status, text = _compute_host_slash(sid, session, "compress", f"/{name}" + (f" {arg}" if arg else ""))
        if status in {"failed", "rejected"}:
            return _err(rid, 5019 if status == "failed" else 4009, text)
        payload = {"type": "exec", "status": "pending", "output": text} if status == "pending" else {"type": "exec", "output": text}
        return _ok(rid, payload)
    try:
        summary = _compress_live_with_feedback(sid, session, session["agent"], arg, snapshot_kwargs=True)
        output = "\n".join(filter(None, [summary["headline"], summary["token_line"], summary.get("note")]))
        return _ok(rid, {"type": "exec", "output": output})
    except CompressionLockHeld as e:
        # Clean no-op (parity with the slash mirror / session.compress), never "compress failed";
        # _compress_session_history already discarded the deferred context-engine notification.
        from agent.manual_compression_feedback import describe_compression_lock_skip

        return _ok(rid, {"type": "exec", "output": describe_compression_lock_skip(e.holder)})
    except Exception as exc:
        finalize_context_engine_compression_notification(session["agent"], committed=False)
        return _err(rid, 5009, f"compress failed: {exc}")


# name → built-in handler (values are rebound onto server globals by bind_module).
_SLASH_BUILTINS = {
    "queue": _cmd_queue, "q": _cmd_queue, "learn": _cmd_learn, "plan": _cmd_plan, "init": _cmd_init,
    "moa": _cmd_moa, "focus": _cmd_focus, "retry": _cmd_retry, "steer": _cmd_steer, "goal": _cmd_goal,
    "loop": _cmd_loop, "undo": _cmd_undo, "snapshot": _cmd_snapshot, "snap": _cmd_snapshot,
    "compress": _cmd_compress, "compact": _cmd_compress,
}


@method("command.dispatch")
def _(rid, params: dict) -> dict:
    name, arg = params.get("name", "").lstrip("/"), params.get("arg", "")
    name = _resolve_name(name)
    session = _sessions.get(params.get("session_id", ""))

    # Stage order is load-bearing: quick > plugin > bundle > skill > built-in.
    for stage in (_dispatch_quick, _dispatch_plugin, _dispatch_bundle, _dispatch_skill):
        res = stage(rid, params, session, name, arg)
        if res is not None:
            return res
    builtin = _SLASH_BUILTINS.get(name)
    if builtin is not None:
        res = builtin(rid, params, session, name, arg)
        if res is not None:
            return res
    return _err(rid, 4018, f"not a quick/plugin/bundle/skill command: {name}")


@method("slash.exec")
def _(rid, params: dict) -> dict:
    session, err = _sess_nowait(params, rid)
    if err:
        return err

    cmd = params.get("command", "").strip()
    if not cmd:
        return _err(rid, 4004, "empty command")

    # Skill/bundle and _PENDING_INPUT_COMMANDS must NOT reach the slash worker. Plugin
    # commands also bypass it but return normal slash.exec output (TUI keeps the pager path).
    _cmd_text = cmd.lstrip("/") if cmd.startswith("/") else cmd
    _cmd_parts = _cmd_text.split(maxsplit=1)
    _cmd_base = (_cmd_parts[0] if _cmd_parts else "").lower()
    _cmd_arg = _cmd_parts[1] if len(_cmd_parts) > 1 else ""
    sid = params.get("session_id", "")

    live_output = _live_slash_command_output(sid, session, _cmd_base, _cmd_arg)
    if live_output is not None:
        return _ok(rid, {"output": live_output or "(no output)"})

    if _cmd_base in _PENDING_INPUT_COMMANDS:
        # Route straight to command.dispatch: some clients fail the error-then-retry fallback ("empty command").
        return _methods["command.dispatch"](rid, {"name": _cmd_base, "arg": _cmd_arg, "session_id": sid})

    if _cmd_base in _WORKER_BLOCKED_COMMANDS:
        subcommand = _cmd_arg.split(maxsplit=1)[0].lower() if _cmd_arg else ""
        if subcommand in {"restore", "rewind"}:
            return _err(rid, 4018, "snapshot restore mutates live config/state; use command.dispatch for /snapshot restore")

    _bundle_key = _bundle_key_for(_cmd_base)
    if _bundle_key is not None:
        return _methods["command.dispatch"](rid, {"name": _bundle_key.lstrip("/"), "arg": _cmd_arg, "session_id": sid})

    if _is_profile_skill_command(session, _cmd_base):
        return _err(rid, 4018, f"skill command: use command.dispatch for /{_cmd_base}")

    plugin_handler = _plugin_command_handler(_cmd_base) if _cmd_base else None
    if plugin_handler:
        try:
            from hermes_cli.plugins import resolve_plugin_command_result

            result = resolve_plugin_command_result(plugin_handler(_cmd_arg))
            return _ok(rid, {"output": str(result or "(no output)")})
        except Exception as e:
            return _ok(rid, {"output": f"Plugin command error: {e}"})

    worker = session.get("slash_worker")
    if not worker:
        # slash.exec runs on the RPC pool: two concurrent commands could both see
        # slash_worker=None and each fork a full MCP-fleet worker (the _attach_worker
        # loser leaks). Serialize first-use spawn per session.
        with _sessions_lock:
            spawn_lock = session.setdefault("_slash_spawn_lock", threading.Lock())
        with spawn_lock:
            worker = session.get("slash_worker")
            if not worker:
                try:
                    worker = _SlashWorker(
                        session["session_key"],
                        getattr(session.get("agent"), "model", _resolve_model()),
                        profile_home=session.get("profile_home"),
                    )
                    _attach_worker(sid, session, worker)
                except Exception as e:
                    return _err(rid, 5030, f"slash worker start failed: {e}")

    try:
        output = worker.run(cmd)
        warning = _mirror_slash_side_effects(sid, session, cmd)
        payload = {"output": output or "(no output)"}
        if warning:
            payload["warning"] = warning
        return _ok(rid, payload)
    except Exception as e:
        with contextlib.suppress(Exception):
            worker.close()
        session["slash_worker"] = None
        return _err(rid, 5030, str(e))


# ─── Insights / rollback / browser / config ──────────────────────────────────


@method("insights.get")
def _(rid, params: dict) -> dict:
    days = params.get("days", 30)
    db = _get_db()
    if db is None:
        return _db_unavailable_error(rid, code=5017)
    try:
        cutoff = time.time() - days * 86400
        rows = [s for s in db.list_sessions_rich(limit=500, compact_rows=True) if (s.get("started_at") or 0) >= cutoff]
        return _ok(rid, {"days": days, "sessions": len(rows), "messages": sum(s.get("message_count", 0) for s in rows)})
    except Exception as e:
        return _err(rid, 5017, str(e))


@method("rollback.list")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    try:

        def go(mgr, cwd):
            if not mgr.enabled:
                return _ok(rid, {"enabled": False, "checkpoints": []})
            rows = [
                {"hash": c.get("hash", ""), "timestamp": c.get("timestamp", ""), "message": c.get("message", "")}
                for c in mgr.list_checkpoints(cwd)
            ]
            return _ok(rid, {"enabled": True, "checkpoints": rows})

        return _with_checkpoints(session, go)
    except Exception as e:
        return _err(rid, 5020, str(e))


@method("rollback.restore")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    target = params.get("hash", "")
    file_path = params.get("file_path", "")
    if not target:
        return _err(rid, 4014, "hash required")
    # Full-history rollback mutates session history → rejected mid-turn (prompt.submit
    # would drop the agent's output or clobber it). File-scoped only touches disk.
    if not file_path and session.get("running"):
        return _err(rid, 4009, "session busy — /interrupt the current turn before full rollback.restore")
    try:

        def go(mgr, cwd):
            resolved = _resolve_checkpoint_hash(mgr, cwd, target)
            result = mgr.restore(cwd, resolved, file_path=file_path or None)
            if result.get("success") and not file_path:
                removed = 0
                with session["history_lock"]:
                    _history, user_indices = _user_turn_indices(session)
                    if user_indices:
                        try:
                            _active, _live_view, removed = _rewind_active_session_history(session, len(user_indices) - 1)
                        except Exception as exc:
                            raise RuntimeError(f"checkpoint restored, but session history rewind failed: {exc}") from exc
                result["history_removed"] = removed
            return result

        return _ok(rid, _with_checkpoints(session, go))
    except Exception as e:
        return _err(rid, 5021, str(e))


@method("rollback.diff")
def _(rid, params: dict) -> dict:
    session, err = _sess(params, rid)
    if err:
        return err
    target = params.get("hash", "")
    if not target:
        return _err(rid, 4014, "hash required")
    try:
        r = _with_checkpoints(session, lambda mgr, cwd: mgr.diff(cwd, _resolve_checkpoint_hash(mgr, cwd, target)))
        raw = r.get("diff", "")[:4000]
        payload = {"stat": r.get("stat", ""), "diff": raw}
        rendered = render_diff(raw, session.get("cols", 80))
        if rendered:
            payload["rendered"] = rendered
        return _ok(rid, payload)
    except Exception as e:
        return _err(rid, 5022, str(e))


@method("browser.manage")
def _(rid, params: dict) -> dict:
    action = params.get("action", "status")
    if action == "status":
        url = _resolve_browser_cdp_url()
        return _ok(rid, {"connected": bool(url), "url": url})
    if action == "disconnect":
        return _browser_disconnect(rid)
    if action == "connect":
        return _browser_connect(rid, params)
    return _err(rid, 4015, f"unknown action: {action}")


@method("plugins.list")
@_guarded(5032)
def _(rid, params: dict) -> dict:
    from hermes_cli.plugins import get_plugin_manager

    rows = [
        {"name": n, "version": getattr(i, "version", "?"), "enabled": getattr(i, "enabled", True)}
        for n, i in get_plugin_manager()._plugins.items()
    ]
    return _ok(rid, {"plugins": rows})


@method("config.show")
@_guarded(5030)
def _(rid, params: dict) -> dict:
    cfg = _load_cfg()
    model = _resolve_model()
    from agent.secret_scope import get_secret

    api_key = get_secret("HERMES_API_KEY", "") or cfg.get("api_key", "")
    masked = f"****{api_key[-4:]}" if len(api_key) > 4 else "(not set)"
    base_url = os.environ.get("HERMES_BASE_URL", "") or cfg.get("base_url", "")
    agent_rows = [
        ["Max Turns", str(_cfg_max_turns(cfg, 500))],
        ["Toolsets", ", ".join(cfg.get("enabled_toolsets", [])) or "all"],
        ["Verbose", str(cfg.get("verbose", False))],
    ]
    sections = [
        {"title": "Model", "rows": [["Model", model], ["Base URL", base_url or "(default)"], ["API Key", masked]]},
        {"title": "Agent", "rows": agent_rows},
        {"title": "Environment", "rows": [["Working Dir", os.getcwd()], ["Config File", str(_hermes_home / "config.yaml")]]},
    ]
    return _ok(rid, {"sections": sections})


# ─── Tools / toolsets / agents ───────────────────────────────────────────────


@method("tools.list")
@_guarded(5031)
def _(rid, params: dict) -> dict:
    return _ok(rid, {"toolsets": _toolset_rows(params, with_tools=True)})


@method("toolsets.list")
@_guarded(5032)
def _(rid, params: dict) -> dict:
    return _ok(rid, {"toolsets": _toolset_rows(params, with_tools=False)})


@method("tools.show")
@_guarded(5034)
def _(rid, params: dict) -> dict:
    from model_tools import get_toolset_for_tool, get_tool_definitions

    session = _sessions.get(params.get("session_id", ""))
    enabled = getattr(session["agent"], "enabled_toolsets", None) if session else _load_enabled_toolsets()
    # Pre-assembly list: /tools must also show tools deferred behind the tool_search bridge (as the CLI).
    tools = get_tool_definitions(enabled_toolsets=enabled, quiet_mode=True, skip_tool_search_assembly=True)
    sections = {}
    for tool in sorted(tools, key=lambda t: t["function"]["name"]):
        name = tool["function"]["name"]
        desc = str(tool["function"].get("description", "") or "").split("\n")[0]
        if ". " in desc:
            desc = desc[: desc.index(". ") + 1]
        sections.setdefault(get_toolset_for_tool(name) or "unknown", []).append({"name": name, "description": desc})
    sections_out = [{"name": name, "tools": rows} for name, rows in sorted(sections.items())]
    return _ok(rid, {"sections": sections_out, "total": len(tools)})


@method("tools.configure")
def _(rid, params: dict) -> dict:
    action = str(params.get("action", "") or "").strip().lower()
    targets = [str(name).strip() for name in params.get("names", []) or [] if str(name).strip()]
    if action not in {"disable", "enable"}:
        return _err(rid, 4017, f"unknown tools action: {action}")
    if not targets:
        return _err(rid, 4018, "names required")

    try:
        from hermes_cli.config import load_config, save_config
        from hermes_cli.tools_config import (
            CONFIGURABLE_TOOLSETS,
            _apply_mcp_change,
            _apply_toolset_change,
            _get_platform_tools,
            _get_plugin_toolset_keys,
        )

        cfg = load_config()
        valid_toolsets = {ts_key for ts_key, _, _ in CONFIGURABLE_TOOLSETS} | _get_plugin_toolset_keys()
        toolset_targets = [name for name in targets if ":" not in name]
        mcp_targets = [name for name in targets if ":" in name]
        unknown = [name for name in toolset_targets if name not in valid_toolsets]
        toolset_targets = [name for name in toolset_targets if name in valid_toolsets]

        if toolset_targets:
            _apply_toolset_change(cfg, "cli", toolset_targets, action)

        missing_servers = _apply_mcp_change(cfg, mcp_targets, action) if mcp_targets else set()
        save_config(cfg)

        sid = params.get("session_id", "")
        session = _sessions.get(sid)
        info = _reset_session_agent(sid, session) if session else None
        enabled = sorted(_get_platform_tools(load_config(), "cli", include_default_mcp_servers=False))
        changed = [
            name
            for name in targets
            if name not in unknown and (":" not in name or name.split(":", 1)[0] not in missing_servers)
        ]

        return _ok(
            rid,
            {
                "changed": changed,
                "enabled_toolsets": enabled,
                "info": info,
                "missing_servers": sorted(missing_servers),
                "reset": bool(session),
                "unknown": unknown,
            },
        )
    except Exception as e:
        return _err(rid, 5035, str(e))


@method("agents.list")
@_guarded(5033)
def _(rid, params: dict) -> dict:
    from tools.process_registry import process_registry

    rows = [
        {"session_id": p["session_id"], "command": p["command"][:80], "status": p["status"], "uptime": p["uptime_seconds"]}
        for p in process_registry.list_sessions()
    ]
    return _ok(rid, {"processes": rows})


# ─── Cron / learning / skills ────────────────────────────────────────────────


@method("cron.manage")
@_profile_scoped_rpc(5023)
def _(rid, params: dict) -> dict:
    """cronjob() keys off HERMES_HOME, so the optional ``profile`` scope reaches a
    per-profile cron store even when that profile runs its own gateway."""
    from tools.cronjob_tools import cronjob

    action, jid = params.get("action", "list"), params.get("name", "")
    if action == "list":
        # Paused jobs are excluded by default (reads as deletion in a toggle UI) — forward the flag.
        result = json.loads(
            cronjob(action="list", include_disabled=is_truthy_value(params.get("include_disabled", False)))
        )
        # ``scoped`` proves the profile scope was honored: new clients treat every job as that
        # profile's; older gateways omit it and clients keep the safe [bot:<name>] filter.
        profile = str(params.get("profile") or "").strip()
        if profile:
            result["scoped"] = profile
        return _ok(rid, result)
    if action == "add":
        # Optional repeat / continuity / deliver ('bot-chat[:name]'): None keeps each cronjob() default.
        raw = cronjob(
            action="create",
            name=jid,
            schedule=params.get("schedule", ""),
            prompt=params.get("prompt", ""),
            repeat=int(params["repeat"]) if str(params.get("repeat", "")).strip().isdigit() else None,
            continuity=is_truthy_value(params.get("continuity")) if params.get("continuity") is not None else None,
            deliver=str(params.get("deliver") or "").strip() or None,
        )
        return _ok(rid, json.loads(raw))
    if action in {"remove", "pause", "resume"}:
        return _ok(rid, json.loads(cronjob(action=action, job_id=jid)))
    return _err(rid, 4016, f"unknown cron action: {action}")


@method("learning.frames")
@_guarded(5000, "learning.frames failed: ")
def _(rid, params: dict) -> dict:
    """Pre-render the ``/journey`` timeline: ``frames`` (reveal 0→1) plus legend/summary/
    bucket metadata so Ink walks the tree locally. Shares its renderer with ``hermes journey``."""
    try:
        cols = int(params.get("cols", 80) or 80)
        rows = int(params.get("rows", 24) or 24)
        frames = int(params.get("frames", 48) or 48)
    except (TypeError, ValueError):
        cols, rows, frames = 80, 24, 48
    from agent.learning_graph import build_learning_graph
    from agent.learning_graph_render import render_frames

    return _ok(rid, render_frames(build_learning_graph(), cols=max(20, cols), rows=max(10, rows), frames=frames))


@method("learning.detail")
@_guarded(5000, "learning.detail failed: ")
def _(rid, params: dict) -> dict:
    """Current content of a journey node, for an edit prefill."""
    from agent.learning_mutations import node_detail

    return _ok(rid, node_detail(str(params.get("id", ""))))


@method("learning.delete")
@_guarded(5000, "learning.delete failed: ")
def _(rid, params: dict) -> dict:
    """Delete a journey node — skills are archived (restorable), memories removed."""
    from agent.learning_mutations import delete_node

    return _ok(rid, delete_node(str(params.get("id", ""))))


@method("learning.edit")
@_guarded(5000, "learning.edit failed: ")
def _(rid, params: dict) -> dict:
    """Rewrite a journey node's content (SKILL.md or memory chunk)."""
    from agent.learning_mutations import edit_node

    return _ok(rid, edit_node(str(params.get("id", "")), str(params.get("content", ""))))


def _skills_list(rid, params, query):
    from hermes_cli.banner import get_available_skills

    return _ok(rid, {"skills": get_available_skills()})


def _skills_search(rid, params, query):
    from tools.skills_hub import GitHubAuth, create_source_router, unified_search

    raw = unified_search(query, create_source_router(GitHubAuth()), source_filter="all", limit=20) or []
    return _ok(rid, {"results": [{"name": r.name, "description": r.description} for r in raw]})


def _skills_install(rid, params, query):
    from hermes_cli.skills_hub import do_install

    class _Q:
        def print(self, *a, **k):
            pass

    do_install(query, skip_confirm=True, console=_Q())
    return _ok(rid, {"installed": True, "name": query})


def _skills_browse(rid, params, query):
    from hermes_cli.skills_hub import browse_skills

    pg = int(params.get("page", 0) or 0) or (int(query) if query.isdigit() else 1)
    return _ok(rid, browse_skills(page=pg, page_size=int(params.get("page_size", 20))))


def _skills_inspect(rid, params, query):
    from hermes_cli.skills_hub import inspect_skill

    return _ok(rid, {"info": inspect_skill(query) or {}})


@method("skills.manage")
@_profile_scoped_rpc(5024)
def _(rid, params: dict) -> dict:
    """list/install use the scoped profile's skills dir; search/browse/inspect hit the shared hub."""
    action, query = params.get("action", "list"), params.get("query", "")
    handler = {
        "list": _skills_list,
        "search": _skills_search,
        "install": _skills_install,
        "browse": _skills_browse,
        "inspect": _skills_inspect,
    }.get(action)
    if handler is None:
        return _err(rid, 4017, f"unknown skills action: {action}")
    return handler(rid, params, query)


@method("skills.reload")
@_guarded(5025)
def _(rid, params: dict) -> dict:
    from agent.skill_commands import reload_skills

    result = reload_skills()
    added = result.get("added") or []
    removed = result.get("removed") or []
    lines = ["Reloading skills..."]
    if not added and not removed:
        lines.append("No new skills detected.")
    for label, items in (("Added skills:", added), ("Removed skills:", removed)):
        if items:
            lines.append(label)
            lines.extend(f"  - {item.get('name', '')}" for item in items)
    lines.append(f"{int(result.get('total') or 0)} skill(s) available")
    return _ok(rid, {"output": "\n".join(lines), "result": result})


# ─── MCP catalog + per-profile server lifecycle (mcp.servers.*) ─────────────
# Gateway mirrors of the dashboard REST surface (hermes_cli/web_routers/mcp.py) so a
# desktop plugin can manage MCP servers for ANY profile. Persistence: hermes_cli/mcp_config.py.


@method("mcp.catalog")
@_profile_scoped_rpc(5024)
def _(rid, params: dict) -> dict:
    """``{servers: [{name, description, installed, enabled, requires: [env keys], transport}]}``
    — the `hermes mcp` menu with per-profile state, so UIs know which entries need setup."""
    from hermes_cli import mcp_catalog

    out = []
    for entry in mcp_catalog.list_catalog():
        try:
            requires = [str(k) for k in (getattr(entry, "env_keys", None) or [])]
        except Exception:
            requires = []
        transport = getattr(entry, "transport", None)  # TransportSpec → its kind string
        out.append(
            {
                "name": entry.name,
                "description": getattr(entry, "description", "") or "",
                "installed": bool(mcp_catalog.is_installed(entry.name)),
                "enabled": bool(mcp_catalog.is_enabled(entry.name)),
                "requires": requires,
                "transport": str(getattr(transport, "kind", "") or transport or "stdio"),
            }
        )
    return _ok(rid, {"servers": out})


@method("mcp.servers.list")
@_profile_scoped_rpc(5024, catch_resolve=False)
def _(rid, params: dict) -> dict:
    """``{servers: [{name, transport, url, command, args, env (key names only),
    auth, oauth_tokens_present, enabled, tools}]}`` for the scoped profile."""
    from hermes_cli.mcp_config import _get_mcp_servers

    servers = _get_mcp_servers()
    return _ok(rid, {"servers": [_mcp_summarize_server(name, cfg) for name, cfg in sorted(servers.items())]})


@method("mcp.servers.add")
@_mcp_server_scoped
def _(rid, params: dict) -> dict:
    """Add ``name`` with EITHER ``preset`` (catalog id) or ``config`` (url/command/args/env/
    headers/auth/tools). ``bearer_token`` goes to the profile's .env; only the
    ``Authorization`` header template is persisted. Duplicate names → 4090."""
    from hermes_cli.mcp_config import _apply_mcp_preset, _get_mcp_servers, _save_bearer_auth_token, _save_mcp_server

    name = str(params.get("name") or "").strip()
    if name in _get_mcp_servers():
        return _err(rid, 4090, f"server '{name}' already exists")
    preset = str(params.get("preset") or "").strip()
    raw_cfg = params.get("config")
    server_config: dict = dict(raw_cfg) if isinstance(raw_cfg, dict) else {}
    if preset:  # fills url/command/args when omitted; mutates server_config in place
        _apply_mcp_preset(
            name,
            preset_name=preset,
            url=server_config.get("url"),
            command=server_config.get("command"),
            cmd_args=list(server_config.get("args") or []),
            server_config=server_config,
        )

    if not server_config.get("url") and not server_config.get("command"):
        return _err(rid, 4063, "config must specify a 'url' (http) or 'command' (stdio), or a valid 'preset'")
    bearer_token = params.get("bearer_token")
    if bearer_token:
        server_config["headers"] = _save_bearer_auth_token(name, str(bearer_token))
    if not _save_mcp_server(name, server_config):
        return _err(rid, 4001, f"server '{name}' rejected: suspicious command/args configuration")
    saved = _get_mcp_servers().get(name, server_config)
    return _ok(rid, {"ok": True, "name": name, "server": _mcp_summarize_server(name, saved)})


@method("mcp.servers.set_api_key")
@_profile_scoped_rpc(5024, required=(("name", _stripped), ("value", _nonempty)), catch_resolve=False)
def _(rid, params: dict) -> dict:
    """Secret → profile .env under ``env_var`` (default ``MCP_<NAME>_API_KEY``); config.yaml
    gets a reference: ``Authorization: Bearer ${ENV}`` header (http) or ``env: {VAR: "${ENV}"}``
    (stdio), matching ``cmd_mcp_configure`` / ``_save_bearer_auth_token``."""
    from hermes_cli.config import load_config, save_config, save_env_value
    from hermes_cli.mcp_config import _bearer_auth_headers, _env_key_for_server, _strip_bearer_prefix

    name, servers, err = _mcp_named_server(rid, params)
    if err:
        return err
    value = params.get("value")
    env_var = str(params.get("env_var") or "").strip() or _env_key_for_server(name)
    entry = servers[name]
    if not isinstance(entry, dict):
        return _err(rid, 4001, "malformed server config")

    if entry.get("url"):
        normalized = _strip_bearer_prefix(str(value))
        if not normalized or normalized.lower() == "bearer":
            return _err(rid, 4063, "value is not a valid credential")
        save_env_value(env_var, normalized)
        if env_var == _env_key_for_server(name):
            entry["headers"] = _bearer_auth_headers(name)
        else:
            entry["headers"] = {"Authorization": f"Bearer ${{{env_var}}}"}
    else:
        save_env_value(env_var, str(value))
        env_block = entry.get("env")
        if not isinstance(env_block, dict):
            env_block = {}
        env_block[env_var] = f"${{{env_var}}}"
        entry["env"] = env_block

    cfg = load_config()
    cfg.setdefault("mcp_servers", {})[name] = entry
    save_config(cfg)
    return _ok(rid, {"ok": True, "name": name, "env_var": env_var, "server": _mcp_summarize_server(name, entry)})


@method("mcp.servers.test")
@_mcp_server_scoped
def _(rid, params: dict) -> dict:
    """Connect, list tools, disconnect. Success: ``{ok, tools, prompts, resources, oauth_needed,
    oauth_tokens_present}``; failure: ``{ok: false, error, tools: [], oauth_needed, ...}``.
    Runs on the RPC pool (_LONG_HANDLERS): a cold stdio `npx` spawn can block for seconds."""
    from hermes_cli.mcp_config import _oauth_tokens_present, _probe_single_server

    name, servers, err = _mcp_named_server(rid, params)
    if err:
        return err
    cfg = servers[name]
    # An `auth: oauth` server serving tools/list anonymously would probe OK with no
    # token — a false green. Require a token on disk for it.
    needs_oauth_token = cfg.get("auth") == "oauth"
    details: dict = {}

    def failure(error: str, oauth_needed: bool, tokens_present) -> dict:
        payload = {"ok": False, "error": error, "tools": [], "oauth_needed": oauth_needed}
        return _ok(rid, {**payload, "oauth_tokens_present": tokens_present})

    try:
        tools = _probe_single_server(name, cfg, details=details)
        token_present = _oauth_tokens_present(name) if needs_oauth_token else True
    except Exception as exc:
        return failure(str(exc), needs_oauth_token, _oauth_tokens_present(name) if needs_oauth_token else None)
    if not token_present:
        return failure("OAuth authentication required — no token found.", True, False)
    payload = {
        "ok": True,
        "tools": [{"name": t, "description": d} for t, d in tools],
        "prompts": details.get("prompts", 0),
        "resources": details.get("resources", 0),
        "oauth_needed": needs_oauth_token,
        "oauth_tokens_present": True if needs_oauth_token else None,
    }
    return _ok(rid, payload)


@method("mcp.servers.remove")
@_mcp_server_scoped
def _(rid, params: dict) -> dict:
    """Remove a server from the profile's config.yaml → ``{ok: true, removed: true}``."""
    from hermes_cli.mcp_config import _remove_mcp_server

    name = str(params.get("name") or "").strip()
    if not _remove_mcp_server(name):
        return _err(rid, 4064, f"server '{name}' not found")
    return _ok(rid, {"ok": True, "removed": True})


@method("mcp.servers.oauth.start")
@_mcp_server_scoped
def _(rid, params: dict) -> dict:
    """Begin a session-backed OAuth flow → ``{ok, session_id, auth_url, flow: "pkce"}``.

    The client opens ``auth_url`` and polls ``mcp.servers.oauth.poll`` until ``approved``.
    A background worker drives the ``hermes mcp login`` machinery with a loopback
    listener. With ``client_redirect_uri`` the CLIENT hosts the loopback and relays the
    code via ``mcp.servers.oauth.callback`` — the only flow that works when desktop and
    gateway are on different machines. Runs on the RPC pool (_LONG_HANDLERS)."""
    client_redirect_uri = str(params.get("client_redirect_uri") or "").strip() or None
    try:
        from hermes_constants import get_hermes_home
        from tui_gateway import mcp_oauth_sessions

        name, servers, err = _mcp_named_server(rid, params)
        if err:
            return err
        cfg = dict(servers[name])
        if not cfg.get("url"):
            return _err(rid, 4001, "stdio servers authenticate via env keys, not OAuth")
        if cfg.get("headers") and cfg.get("auth") != "oauth":
            return _err(rid, 4001, "this server uses header/API-key auth, not OAuth")
        cfg["auth"] = "oauth"

        hermes_home = str(get_hermes_home().expanduser().resolve(strict=False))
        result = mcp_oauth_sessions.start_flow(hermes_home, name, cfg, client_redirect_uri=client_redirect_uri)
    except ValueError as e:
        return _err(rid, 4001, str(e))
    return _ok(rid, {"ok": True, "session_id": result["session_id"], "auth_url": result["auth_url"], "flow": result["flow"]})


@method("mcp.servers.oauth.poll")
@_profile_scoped_rpc(5024, required=_NAME_SESSION, catch_resolve=False)
def _(rid, params: dict) -> dict:
    """Poll a flow → ``{ok, status: pending|approved|error, error_message?, auth_url?, tools?}``.
    On ``approved`` tokens persist for that server/profile (profile scope applies here too)."""
    from tui_gateway import mcp_oauth_sessions

    name = str(params.get("name") or "").strip()
    session_id = str(params.get("session_id") or "").strip()
    result = mcp_oauth_sessions.poll_flow(session_id, name)
    return _ok(rid, {"ok": True, **result})


@method("mcp.servers.oauth.callback")
@_profile_scoped_rpc(5024, required=_NAME_SESSION, catch_resolve=False)
def _(rid, params: dict) -> dict:
    """Relay a client-captured redirect (``code``/``state``/``error``) into a flow started with
    ``client_redirect_uri``. ``{ok: true}`` once accepted (state verified), else ``{ok: false, error_message}``."""
    from tui_gateway import mcp_oauth_sessions

    name = str(params.get("name") or "").strip()
    session_id = str(params.get("session_id") or "").strip()
    result = mcp_oauth_sessions.deliver_callback_flow(
        session_id,
        name,
        code=str(params.get("code") or "") or None,
        state=str(params.get("state") or "") or None,
        error=str(params.get("error") or "") or None,
    )
    return _ok(rid, result)


# ─── Plugins ─────────────────────────────────────────────────────────────────


def _plugin_rows() -> list[dict]:
    from hermes_cli.plugins_cmd import (
        _bundled_default_on,
        _discover_all_plugins,
        _get_disabled_set,
        _get_enabled_set,
        _is_portable_plugin_dir,
        _plugin_status,
    )

    enabled = _get_enabled_set()
    disabled = _get_disabled_set()
    out = []
    for name, version, desc, source, _dir, key in sorted(_discover_all_plugins()):
        status = _plugin_status(name, enabled, disabled, key=key)
        # Bundled backends/platforms/providers run without an explicit enable: report the
        # truthful default instead of "not enabled" (reads as OFF).
        if status == "not enabled" and source == "bundled" and _bundled_default_on(_dir):
            status = "enabled"
        out.append(
            {
                "name": name,
                "key": key,  # canonical registry key (``image_gen/fal``): names collide across category dirs
                "version": str(version or ""),
                "description": desc or "",
                "source": source,
                "status": status,
                "portable": _is_portable_plugin_dir(_dir),  # Agent Plugins v1 package vs native Hermes plugin
            }
        )
    return out


def _plugins_list(rid, params):
    rows = _plugin_rows()
    user_count = sum(1 for r in rows if r["source"] != "bundled")
    return _ok(rid, {"plugins": rows, "user_count": user_count, "bundled_count": len(rows) - user_count})


def _plugins_toggle(rid, params):
    from hermes_cli.plugins_cmd import dashboard_set_agent_plugin_enabled

    # Prefer the canonical key — bare names are ambiguous across categories.
    ident = (params.get("key") or params.get("name") or "").strip()
    if not ident:
        return _err(rid, 4019, "plugins.toggle requires a 'key' or 'name'")
    result = dashboard_set_agent_plugin_enabled(ident, enabled=bool(params.get("enable")))
    if not result.get("ok"):
        return _err(rid, 5026, result.get("error") or "toggle failed")
    row = next((r for r in _plugin_rows() if ident in (r["key"], r["name"])), None)
    return _ok(rid, {"ok": True, "unchanged": bool(result.get("unchanged")), "name": ident, "plugin": row})


def _plugins_install(rid, params):
    from hermes_cli.plugins_cmd import dashboard_install_plugin

    ident = (params.get("identifier") or params.get("repo") or "").strip()
    if not ident:
        return _err(rid, 4019, "plugins.install requires 'identifier' or 'repo'")
    result = dashboard_install_plugin(ident, force=bool(params.get("force")), enable=params.get("enable", True))
    if not result.get("ok"):
        return _err(rid, 5026, result.get("error") or "install failed")
    return _ok(rid, result)


@method("plugins.manage")
@_profile_scoped_rpc(5026, catch_resolve=False)
def _(rid, params: dict) -> dict:
    """TUI Plugins Hub backend (shares primitives with ``hermes plugins`` / the dashboard).
      - ``list``    → {plugins: [{name, key, version, description, source, status, portable}], user_count, bundled_count}
      - ``toggle``  → flip ``key`` (or ``name``) per ``enable``; returns the row + {ok, unchanged}
      - ``install`` → git-clone ``identifier``/``repo`` into ~/.hermes/plugins/ (``force``, ``enable`` default True)
    Optional ``profile`` scopes HERMES_HOME (mcp.servers.* contract)."""
    action = params.get("action", "list")
    handler = {"list": _plugins_list, "toggle": _plugins_toggle, "install": _plugins_install}.get(action)
    if handler is None:
        return _err(rid, 4017, f"unknown plugins action: {action}")
    return handler(rid, params)


@method("shell.exec")
def _(rid, params: dict) -> dict:
    cmd = params.get("command", "")
    if not cmd:
        return _err(rid, 4004, "empty command")
    try:
        from tools.approval import detect_dangerous_command, detect_hardline_command

        is_hardline, hardline_desc = detect_hardline_command(cmd)
        if is_hardline:
            return _err(rid, 4005, f"blocked (hardline): {hardline_desc}. Use the agent for dangerous commands.")
        is_dangerous, _, desc = detect_dangerous_command(cmd)
        if is_dangerous:
            return _err(rid, 4005, f"blocked: {desc}. Use the agent for dangerous commands.")
    except ImportError:
        return _err(rid, 5001, "shell.exec unavailable: approval safety module not importable")
    try:
        r = subprocess.run(cmd, shell=True, cwd=os.getcwd(), **_capture_run_kwargs(30))
        return _ok(rid, {"stdout": r.stdout[-4000:], "stderr": r.stderr[-2000:], "code": r.returncode})
    except subprocess.TimeoutExpired:
        return _err(rid, 5002, "command timed out (30s)")
    except Exception as e:
        return _err(rid, 5003, str(e))


def register(server) -> None:
    """Rebind this module's helpers + handlers onto ``server`` and register the handlers."""
    bind_module(globals(), server, skip=("_",))
