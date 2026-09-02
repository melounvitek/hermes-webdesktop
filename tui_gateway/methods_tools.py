"""Tools & system / slash / insights / rollback / plugins / cron / skills / MCP JSON-RPC handlers.

Everything defined here is rebound onto server.py's globals at install time
(``method_ctx.bind_module``), so handler bodies AND module-level helpers may
reference server globals bare (``_ok``, ``_err``, ``_sessions``, ...).  Names
must not collide with server.py's own; helpers here use a ``_cmd_`` /
``_slash_`` / ``_toolset_`` / ``_mcp_`` prefix.
"""

import sys

from .method_ctx import HandlerRegistry, bind_module

_registry = HandlerRegistry()
method = _registry.method
_profile_scoped = _registry.profile_scoped


# ─── Shared helpers ──────────────────────────────────────────────────────────


def _profile_scoped_rpc(fail_code: int, *, required=(), catch_resolve: bool = True):
    """Wrap a handler body with the optional ``profile`` HERMES_HOME scope.

    Order preserved from the original handlers: ``required`` params are checked
    first (4063 ``<key> required``), then the profile is resolved (4064 when the
    profile dir is missing), then the body runs; any body exception becomes
    ``fail_code``.  ``catch_resolve`` also maps resolve-time exceptions to
    ``fail_code`` (cron/skills/catalog); the mcp.servers.* handlers let them
    propagate to dispatch().  The override is always reset afterwards.
    """

    def deco(body):
        def handler(rid, params: dict) -> dict:
            for key, present in required:
                if not present(params.get(key)):
                    return _err(rid, 4063, f"{key} required")
            profile = str(params.get("profile") or "").strip()
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
                return _err(rid, fail_code, str(e))
            finally:
                _mcp_reset_profile(token)

        handler.__doc__ = body.__doc__
        return handler

    return deco


def _stripped(v) -> bool:
    return bool(str(v or "").strip())


def _nonempty(v) -> bool:
    return not (v is None or str(v) == "")


_NAME = (("name", _stripped),)
_NAME_SESSION = (("name", _stripped), ("session_id", _stripped))


def _mcp_server_scoped(body):
    """mcp.servers.* contract: ``name`` required, profile scope, body errors → 5024."""
    return _profile_scoped_rpc(5024, required=_NAME, catch_resolve=False)(body)


def _busy_error(rid, session, cmd: str):
    if session.get("running"):
        return _err(rid, 4009, f"session busy — /interrupt the current turn before /{cmd}")
    return None


def _user_turn_indices(session):
    """(history, indices of user-originated turns) minus ephemeral scaffolding. Call under history_lock."""
    from agent.context_compressor import user_originated_turn_view

    history = _history_without_ephemeral_scaffolding(session.get("history", []))
    return history, [i for i, m in enumerate(history) if user_originated_turn_view(m) is not None]


def _clip(text: str, n: int = 120) -> str:
    return text[:n] + ("…" if len(text) > n else "")


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
def _(rid, params: dict) -> dict:
    try:
        from tools.process_registry import process_registry

        return _ok(rid, {"killed": process_registry.kill_all()})
    except Exception as e:
        return _err(rid, 5010, str(e))


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


@method("reload.mcp")
def _(rid, params: dict) -> dict:
    session = _sessions.get(params.get("session_id", ""))
    try:
        # /reload-mcp invalidates the prompt cache. Unless the caller passed
        # confirm=true, honour ``approvals.mcp_reload_confirm`` (default true) by
        # returning a confirm_required payload instead of reloading; Ink prints
        # ``message`` and re-invokes with confirm=true (or flips the config).
        if not bool(params.get("confirm", False)):
            try:
                from hermes_cli.config import load_config as _load_config

                _cfg = _load_config()
                _approvals = _cfg.get("approvals") if isinstance(_cfg, dict) else None
                _confirm_required = True
                if isinstance(_approvals, dict):
                    _confirm_required = bool(_approvals.get("mcp_reload_confirm", True))
            except Exception:
                _confirm_required = True
            if _confirm_required:
                return _ok(
                    rid,
                    {
                        "status": "confirm_required",
                        "message": (
                            "⚠️  /reload-mcp invalidates the prompt cache (next "
                            "message re-sends full input tokens). Reply `/reload-mcp "
                            "now` to proceed, or `/reload-mcp always` to proceed and "
                            "silence this prompt permanently."
                        ),
                    },
                )

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
            """Rebuild THIS session's cached tool snapshot from the live registry and
            push session.info (the agent never re-reads the registry on its own;
            mirrors gateway/run.py::_execute_mcp_reload). Runs under _mcp_reload_lock
            so a concurrent reload can't tear the registry down mid-refresh."""
            if not session:
                return
            agent = session["agent"]
            try:
                from tools.mcp_tool import refresh_agent_mcp_tools

                # Re-resolve enabled toolsets so a server enabled in config this
                # session is picked up.
                refresh_agent_mcp_tools(agent, enabled_override=_load_enabled_toolsets(), quiet_mode=True)
            except Exception as _exc:
                logger.warning("Failed to refresh cached agent tools after /reload-mcp: %s", _exc)
            _emit("session.info", params.get("session_id", ""), _session_info(agent, session))

        global _mcp_reload_gen, _mcp_reload_loaded_rev

        # Revision the CALLER wants loaded (the mcp_rev its poll observed). Empty
        # on legacy clients / manual /reload-mcp — those coalesce on generation alone.
        req_rev = str(params.get("rev") or "")

        def _do_full_reload() -> None:
            """shutdown+discover+refresh under the lock, then mark a completed generation.
            The lock spans the refresh too, else a second reload could tear the registry
            down while this one is still rebuilding the session snapshot. Config can
            change WHILE discover connects: re-hash after discovery and repeat until
            stable, so the marked generation reflects the config actually loaded."""
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

        # Serialize reloads. LEADER (won the non-blocking acquire) runs the full
        # reload. FOLLOWER snapshots the generation, waits, then — still holding the
        # lock — coalesces only if a reload COMPLETED meanwhile (generation advanced,
        # so the leader didn't throw) AND it loaded the revision this request asked
        # for; otherwise it re-runs the full reload so a failed/stale leader can
        # never leave a follower acking a revision that was never loaded.
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
    except Exception as e:
        return _err(rid, 5015, str(e))


@method("reload.env")
def _(rid, params: dict) -> dict:
    """Re-read ``~/.hermes/.env`` into the gateway (classic CLI ``/reload`` parity).

    Already-constructed agents keep their credential pool / provider routing —
    same as classic CLI; ``/new`` gets a fresh credential resolution.
    """
    try:
        from hermes_cli.config import reload_env

        count = reload_env()
        return _ok(rid, {"updated": int(count)})
    except Exception as e:
        return _err(rid, 5015, str(e))


# ─── Command catalog / dispatch ──────────────────────────────────────────────


@method("commands.catalog")
def _(rid, params: dict) -> dict:
    """Registry-backed slash metadata for the TUI — categorized, no aliases."""
    try:
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

        for cmd in COMMAND_REGISTRY:
            meta = command_desktop_meta(cmd)
            commands[f"/{cmd.name}"] = dict(meta)
            for alias in cmd.aliases:
                commands[f"/{alias}"] = dict(meta)

            if cmd.name in _TUI_HIDDEN or cmd.gateway_only:
                continue

            c = f"/{cmd.name}"
            canon[c.lower()] = c
            for a in cmd.aliases:
                canon[f"/{a}".lower()] = c

            desc = _build_description(cmd)
            all_pairs.append([c, desc])
            bucket(cmd.category).append([c, desc])

        for name, desc, cat in _TUI_EXTRA:
            # A TUI extra colliding with a registry command/alias (e.g. /compact,
            # /sessions) is skipped: the registry entry is canonical.
            if name.lower() in canon:
                continue
            canon[name.lower()] = name
            all_pairs.append([name, desc])
            bucket(cat).append([name, desc])

        warning = ""
        try:
            qcmds = _load_cfg().get("quick_commands", {}) or {}
            if isinstance(qcmds, dict) and qcmds:
                rows = bucket("User commands")
                for qname, qc in sorted(qcmds.items()):
                    if not isinstance(qc, dict):
                        continue
                    key = f"/{qname}"
                    canon[key.lower()] = key
                    qtype = qc.get("type", "")
                    if qtype == "exec":
                        default_desc = f"exec: {qc.get('command', '')}"
                    elif qtype == "alias":
                        default_desc = f"alias → {qc.get('target', '')}"
                    else:
                        default_desc = qtype or "quick command"
                    qdesc = _clip(str(qc.get("description") or default_desc))
                    all_pairs.append([key, qdesc])
                    rows.append([key, qdesc])
        except Exception as e:
            if not warning:
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
                    canon[key.lower()] = key
                    pdesc = _clip(str(info.get("description") or "Plugin command"))
                    all_pairs.append([key, pdesc])
                    rows.append([key, pdesc])
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

            # Usage + origin ride along here (not a second RPC): every catalog
            # consumer also ranks it, and both sidecars are already loaded.
            usage, origin_of = _skill_usage_lookup()

            for k, info in sorted(scan_skill_commands().items()):
                all_pairs.append([k, _clip(str(info.get("description", "Skill")))])
                name = str(info.get("name") or k.lstrip("/"))
                skills[k] = {"usage": usage(name), "origin": origin_of(name)}
                skill_count += 1
        except Exception as e:
            warning = f"skill discovery unavailable: {e}"

        return _ok(
            rid,
            {
                "pairs": all_pairs,
                "sub": {k: v[:] for k, v in SUBCOMMANDS.items()},
                "canon": canon,
                "commands": commands,
                "categories": [{"name": cat, "pairs": cat_map[cat]} for cat in cat_order],
                "skills": skills,
                "skill_count": skill_count,
                "warning": warning,
            },
        )
    except Exception as e:
        return _err(rid, 5020, str(e))


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
        # CREATE_NO_WINDOW on Windows: under the windowless desktop parent this
        # spawn otherwise flashes a console.
        from hermes_cli._subprocess_compat import windows_hide_flags

        r = subprocess.run(
            [sys.executable, "-m", "hermes_cli.main", *argv],
            capture_output=True,
            text=True,
            # UTF-8 + lossy decode: non-UTF-8 child output must not crash the
            # gateway thread on locale-mismatched Windows.
            encoding="utf-8",
            errors="replace",
            timeout=min(int(params.get("timeout", 240)), 600),
            cwd=os.getcwd(),
            # Can drive the agent → needs provider credentials; tier-1 secrets still stripped.
            env=hermes_subprocess_env(inherit_credentials=True),
            stdin=subprocess.DEVNULL,
            creationflags=windows_hide_flags(),
        )
        parts = [r.stdout or "", r.stderr or ""]
        out = "\n".join(p for p in parts if p).strip() or "(no output)"
        return _ok(rid, {"blocked": False, "code": r.returncode, "output": out[:48_000]})
    except subprocess.TimeoutExpired:
        return _err(rid, 5016, "cli.exec: timeout")
    except Exception as e:
        return _err(rid, 5017, str(e))


@method("command.resolve")
def _(rid, params: dict) -> dict:
    try:
        from hermes_cli.commands import resolve_command

        r = resolve_command(params.get("name", ""))
        if r:
            return _ok(rid, {"canonical": r.name, "description": r.description, "category": r.category})
        return _err(rid, 4011, f"unknown command: {params.get('name')}")
    except Exception as e:
        return _err(rid, 5012, str(e))


# command.dispatch stages. Each takes (rid, params, session, name, arg) and
# returns a JSON-RPC envelope, or None to fall through to the next stage.


def _dispatch_quick(rid, params, session, name, arg):
    qcmds = _load_cfg().get("quick_commands", {})
    if name not in qcmds:
        return None
    qc = qcmds[name]
    if qc.get("type") == "exec":
        # Sanitized env: quick commands run in the TUI server process, which
        # holds every API key in os.environ.
        from tools.environments.local import build_subprocess_env

        sanitized_env = build_subprocess_env()
        from hermes_cli._subprocess_compat import windows_hide_flags

        r = subprocess.run(
            qc.get("command", ""),
            shell=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",  # lossy decode: see cli.exec
            timeout=30,
            stdin=subprocess.DEVNULL,
            env=sanitized_env,
            creationflags=windows_hide_flags(),
        )
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


def _dispatch_plugin(rid, params, session, name, arg):
    try:
        from hermes_cli.plugins import get_plugin_command_handler, resolve_plugin_command_result

        handler = get_plugin_command_handler(name)
        if handler:
            result = resolve_plugin_command_result(handler(arg))
            return _ok(rid, {"type": "plugin", "output": str(result or "")})
    except Exception:
        pass
    return None


def _dispatch_bundle(rid, params, session, name, arg):
    try:
        from agent.skill_bundles import build_bundle_invocation_message, get_skill_bundles, resolve_bundle_command_key
        from hermes_cli.commands import resolve_command

        bundle_key = resolve_bundle_command_key(name) if resolve_command(name) is None else None
    except Exception:
        bundle_key = None
    if bundle_key is None:
        return None
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
    return _ok(
        rid,
        {
            "type": "send",
            "message": msg,
            "notice": notice,
            # UIs render `display`, never `message`: the expanded body is model-facing scaffolding.
            "display": _skill_scaffold_projection(msg),
        },
    )


def _dispatch_skill(rid, params, session, name, arg):
    try:
        from agent.skill_commands import scan_skill_commands, build_skill_invocation_message

        cmds = scan_skill_commands()
        key = f"/{name}"
        if key in cmds:
            msg = build_skill_invocation_message(key, arg, task_id=session.get("session_key", "") if session else "")
            if msg:
                return _ok(
                    rid,
                    {
                        "type": "skill",
                        "message": msg,
                        "name": cmds[key].get("name", name),
                        "display": _skill_scaffold_projection(msg),  # UIs render this, never `message`
                    },
                )
    except Exception:
        pass
    return None


# Built-in commands that queue messages onto _pending_input in the CLI. The TUI
# slash worker has no reader for that queue, so they are handled here and return
# a structured payload.


def _cmd_queue(rid, params, session, name, arg):
    if not arg:
        return _err(rid, 4004, "usage: /queue <prompt>")
    return _ok(rid, {"type": "send", "message": arg})


def _cmd_learn(rid, params, session, name, arg):
    # Standards-guided prompt submitted as a normal turn; the live agent gathers
    # sources with its own tools and authors the skill via skill_manage.
    from agent.learn_prompt import build_learn_prompt

    return _ok(rid, {"type": "send", "message": build_learn_prompt(arg)})


def _cmd_plan(rid, params, session, name, arg):
    # Plan-mode prompt as a normal turn (same pattern as /learn); the agent saves
    # the plan under .hermes/plans/ via write_file.
    from agent.plan_prompt import build_plan_prompt

    return _ok(rid, {"type": "send", "message": build_plan_prompt(arg)})


def _cmd_init(rid, params, session, name, arg):
    # Generate-or-update AGENTS.md as a normal turn (same pattern as /learn).
    from hermes_cli.init_command import build_init_prompt_for_cwd

    return _ok(rid, {"type": "send", "message": build_init_prompt_for_cwd(extra=arg)})


def _cmd_moa(rid, params, session, name, arg):
    # One-shot sugar: run ONE prompt through the default MoA preset, then restore
    # the prior model. Switching for the whole session goes through the model
    # picker (MoA presets surface as a virtual "Mixture of Agents" provider).
    try:
        from hermes_cli.moa_config import moa_usage, normalize_moa_config

        if not arg:
            return _err(rid, 4004, moa_usage())
        if not session:
            return _err(rid, 4001, "no active session")
        sid = params.get("session_id", "")
        preset = normalize_moa_config(_load_cfg().get("moa") or {})["default_preset"]
        # Record the live model identity for post-turn restore, then swap the
        # agent's client in place: setting session["model_override"] alone never
        # switches an already-built agent.
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
        return _ok(
            rid,
            {
                "type": "send",
                "notice": f"MoA one-shot queued with preset {preset}; previous model will be restored after this turn.",
                "message": arg,
            },
        )
    except Exception as exc:
        return _err(rid, 5030, f"moa unavailable: {exc}")


def _cmd_focus(rid, params, session, name, arg):
    # Display-only. Routed through the same config.set branch the Ink slash
    # command uses so both surfaces share one state machine and persistence path.
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
    return _ok(
        rid,
        {"type": "exec", "output": format_focus_toggle_message(bool(_target), _payload.get("tool_progress") or "all")},
    )


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
                return _ok(
                    rid,
                    {
                        "type": "exec",
                        "output": f"⏩ Steer queued — arrives after the next tool call: {arg[:80]}{'...' if len(arg) > 80 else ''}",
                    },
                )
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

    sid_key = session.get("session_key") or ""
    if not sid_key:
        return _err(rid, 4001, "no session key")

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
        # Resume must restart work, not just flip persisted state: an `exec`
        # result is display-only, so return a `send` carrying the continuation
        # prompt; `display` keeps the transcript free of model-facing scaffolding.
        prompt = mgr.next_continuation_prompt()
        if not prompt:
            return _ok(rid, {"type": "exec", "output": f"▶ Goal resumed: {state.goal}"})
        return _ok(
            rid,
            {
                "type": "send",
                "notice": f"▶ Goal resumed: {state.goal}\nContinuing now — taking the next step.",
                "message": prompt,
                "display": "/goal resume",
            },
        )
    if lower in {"clear", "stop", "done"}:
        had = mgr.has_goal()
        mgr.clear()
        return _ok(rid, {"type": "exec", "output": "✓ Goal cleared." if had else "No active goal."})

    # Remaining text = the new goal. The client renders `notice` as a sys line
    # then submits `message`; the post-turn judge in _run_prompt_submit takes over.
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
    # Recurring in-session wakeups; the notification poller fires due wakeups
    # into this session while it's idle.
    if not session:
        return _err(rid, 4001, "no active session")
    try:
        from hermes_cli.loops import LoopManager, dispatch_loop_command
    except Exception as exc:
        return _err(rid, 5030, f"loops unavailable: {exc}")

    sid_key = session.get("session_key") or ""
    if not sid_key:
        return _err(rid, 4001, "no session key")

    result = dispatch_loop_command(LoopManager(session_id=sid_key), arg)
    output = result.get("output") or ""
    if result.get("created"):
        try:
            from hermes_cli.loops import goal_blocks_loop_tick

            if goal_blocks_loop_tick(sid_key):
                output += (
                    "\nNote: an active /goal is driving this session — loop "
                    "wakeups defer until the goal finishes, pauses, or parks."
                )
        except Exception:
            pass
    return _ok(rid, {"type": "exec", "output": output})


def _cmd_undo(rid, params, session, name, arg):
    # /undo [N]: back up N user turns (default 1), soft-delete the truncated rows
    # on disk, and prefill the composer with the backed-up user text.
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
    if n < 1:
        n = 1
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
    # Notify memory providers (same hook /branch fires) with rewound=True so
    # providers caching per-turn document state invalidate.
    agent = session.get("agent")
    if agent is not None:
        mm = getattr(agent, "_memory_manager", None)
        if mm is not None:
            try:
                mm.on_session_switch(session_key, parent_session_id="", reset=False, rewound=True)
            except Exception:
                pass
        if hasattr(agent, "_invalidate_system_prompt"):
            try:
                agent._invalidate_system_prompt()
            except Exception:
                pass
        if hasattr(agent, "_last_flushed_db_idx"):
            try:
                agent._last_flushed_db_idx = len(active)
            except Exception:
                pass
    turn_word = "turn" if turns_undone == 1 else "turns"
    notice = (
        f"↶ Undid {turns_undone} {turn_word} ({rewound_count} message(s)). Edit and resubmit, or send a new message."
    )
    return _ok(rid, {"type": "prefill", "message": target_text, "notice": notice})


def _cmd_snapshot(rid, params, session, name, arg):
    subcommand = arg.split(maxsplit=1)[0].lower() if arg else ""
    if subcommand not in {"restore", "rewind"}:
        return None
    return _ok(
        rid,
        {
            "type": "exec",
            "output": (
                "/snapshot restore is blocked in the TUI because it changes "
                "config/state on disk while the live agent has cached settings. "
                "Run it in the classic CLI, then restart the TUI."
            ),
        },
    )


def _cmd_compress(rid, params, session, name, arg):
    if not session:
        return _err(rid, 4001, "no active session to compress")
    if busy := _busy_error(rid, session, "compress"):
        return busy
    from agent.conversation_compression import finalize_context_engine_compression_notification

    sid = params.get("session_id", "")
    if _session_uses_compute_host(session):
        command = f"/{name}" + (f" {arg}" if arg else "")
        _late_session = session

        def _on_late_ack(late: dict, _sid=sid) -> None:
            _adopt_late_compute_host_compress_ack(_sid, _late_session, late, route_name="slash.compress")

        try:
            ack = _send_compute_host_control(
                sid,
                route_name="slash.compress",
                command=command,
                wait=True,
                timeout=_compute_host_compress_wait_seconds(),
                on_late_ack=_on_late_ack,
            )
        except queue.Empty:
            return _ok(
                rid,
                {
                    "type": "exec",
                    "status": "pending",
                    "output": "compression still running in the background; the transcript will refresh when it finishes",
                },
            )
        except Exception as exc:
            return _err(rid, 5019, f"compute-host slash.compress failed: {exc}")
        if ack.get("type") in {"control.error", "error"}:
            return _err(rid, 4009, str(ack.get("message") or "compute-host slash.compress failed"))
        _apply_compute_host_metadata_mirror(session, ack)
        return _ok(rid, {"type": "exec", "output": str(ack.get("output") or "")})
    try:
        from agent.manual_compression_feedback import summarize_manual_compression
        from agent.model_metadata import estimate_request_tokens_rough

        with session["history_lock"]:
            before_messages = list(session.get("history", []))
            history_version = int(session.get("history_version", 0))
        _agent = session["agent"]
        _sys_prompt = getattr(_agent, "_cached_system_prompt", "") or ""
        _tools = getattr(_agent, "tools", None) or None
        before_tokens = (
            estimate_request_tokens_rough(before_messages, system_prompt=_sys_prompt, tools=_tools)
            if before_messages
            else 0
        )
        removed, usage = _compress_session_history(
            session,
            arg.strip() or None,
            approx_tokens=before_tokens,
            before_messages=before_messages,
            history_version=history_version,
        )
        with session["history_lock"]:
            after_messages = list(session.get("history", []))
        after_tokens = (
            estimate_request_tokens_rough(
                after_messages,
                system_prompt=getattr(_agent, "_cached_system_prompt", "") or _sys_prompt,
                tools=getattr(_agent, "tools", None) or _tools,
            )
            if after_messages
            else 0
        )
        _sync_session_key_after_compress(sid, session)
        summary = summarize_manual_compression(
            before_messages,
            after_messages,
            before_tokens,
            after_tokens,
            compression_state=getattr(_agent, "context_compressor", None),
        )
        _emit("session.info", sid, _session_info(session.get("agent"), session))
        finalize_context_engine_compression_notification(_agent, committed=True)
        return _ok(
            rid,
            {
                "type": "exec",
                "output": "\n".join(filter(None, [summary["headline"], summary["token_line"], summary.get("note")])),
            },
        )
    except CompressionLockHeld as e:
        # Lock-skip is a clean no-op (matches the slash mirror and session.compress
        # RPC), never a "compress failed" error. _compress_session_history already
        # discarded the deferred context-engine notification before raising.
        from agent.manual_compression_feedback import describe_compression_lock_skip

        return _ok(rid, {"type": "exec", "output": describe_compression_lock_skip(e.holder)})
    except Exception as exc:
        finalize_context_engine_compression_notification(session["agent"], committed=False)
        return _err(rid, 5009, f"compress failed: {exc}")


def _slash_builtin_table() -> dict:
    """name → handler. Built per call so the entries resolve to the rebound helpers."""
    return {
        "queue": _cmd_queue,
        "q": _cmd_queue,
        "learn": _cmd_learn,
        "plan": _cmd_plan,
        "init": _cmd_init,
        "moa": _cmd_moa,
        "focus": _cmd_focus,
        "retry": _cmd_retry,
        "steer": _cmd_steer,
        "goal": _cmd_goal,
        "loop": _cmd_loop,
        "undo": _cmd_undo,
        "snapshot": _cmd_snapshot,
        "snap": _cmd_snapshot,
        "compress": _cmd_compress,
        "compact": _cmd_compress,
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
    builtin = _slash_builtin_table().get(name)
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

    # Skill/bundle and _pending_input commands must NOT reach the slash worker
    # (see _PENDING_INPUT_COMMANDS). Plugin commands also bypass the worker but
    # still return normal slash.exec output so the TUI keeps the pager path.
    _cmd_text = cmd.lstrip("/") if cmd.startswith("/") else cmd
    _cmd_parts = _cmd_text.split(maxsplit=1)
    _cmd_base = (_cmd_parts[0] if _cmd_parts else "").lower()
    _cmd_arg = _cmd_parts[1] if len(_cmd_parts) > 1 else ""
    sid = params.get("session_id", "")

    live_output = _live_slash_command_output(sid, session, _cmd_base, _cmd_arg)
    if live_output is not None:
        return _ok(rid, {"output": live_output or "(no output)"})

    if _cmd_base in _PENDING_INPUT_COMMANDS:
        # Route straight to command.dispatch rather than erroring and relying on a
        # client-side retry (some clients fail the fallback → "empty command").
        return _methods["command.dispatch"](rid, {"name": _cmd_base, "arg": _cmd_arg, "session_id": sid})

    if _cmd_base in _WORKER_BLOCKED_COMMANDS:
        subcommand = _cmd_arg.split(maxsplit=1)[0].lower() if _cmd_arg else ""
        if subcommand in {"restore", "rewind"}:
            return _err(
                rid, 4018, "snapshot restore mutates live config/state; use command.dispatch for /snapshot restore"
            )

    try:
        from agent.skill_bundles import resolve_bundle_command_key
        from hermes_cli.commands import resolve_command

        _bundle_key = resolve_bundle_command_key(_cmd_base) if resolve_command(_cmd_base) is None else None
        if _bundle_key is not None:
            return _methods["command.dispatch"](
                rid, {"name": _bundle_key.lstrip("/"), "arg": _cmd_arg, "session_id": sid}
            )
    except Exception:
        pass

    try:
        from agent.skill_commands import get_skill_commands
        from hermes_constants import reset_hermes_home_override, set_hermes_home_override

        # Bind HERMES_HOME to the session's profile so get_skill_commands() sees
        # that profile's skills.external_dirs: dispatch() runs this on the pool
        # with a copied context and nothing upstream binds the override here.
        _profile_home = session.get("profile_home")
        _home_token = set_hermes_home_override(_profile_home) if _profile_home else None
        try:
            _cmd_key = f"/{_cmd_base}"
            if _cmd_key in get_skill_commands():
                return _err(rid, 4018, f"skill command: use command.dispatch for {_cmd_key}")
        finally:
            if _home_token is not None:
                reset_hermes_home_override(_home_token)
    except Exception:
        pass

    plugin_handler = None
    resolve_plugin_command_result = None
    if _cmd_base:
        try:
            from hermes_cli.plugins import get_plugin_command_handler, resolve_plugin_command_result

            plugin_handler = get_plugin_command_handler(_cmd_base)
        except Exception:
            plugin_handler = None
            resolve_plugin_command_result = None

    if plugin_handler and resolve_plugin_command_result:
        try:
            result = resolve_plugin_command_result(plugin_handler(_cmd_arg))
            return _ok(rid, {"output": str(result or "(no output)")})
        except Exception as e:
            return _ok(rid, {"output": f"Plugin command error: {e}"})

    worker = session.get("slash_worker")
    if not worker:
        # On-demand spawn is the ONLY spawn path, and slash.exec runs on the RPC
        # pool: two concurrent commands could both see slash_worker=None and each
        # fork a full MCP-fleet worker (the _attach_worker loser leaks). Serialize
        # first-use spawn per session.
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
        try:
            worker.close()
        except Exception:
            pass
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
            return _ok(
                rid,
                {
                    "enabled": True,
                    "checkpoints": [
                        {
                            "hash": c.get("hash", ""),
                            "timestamp": c.get("timestamp", ""),
                            "message": c.get("message", ""),
                        }
                        for c in mgr.list_checkpoints(cwd)
                    ],
                },
            )

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
    # A full-history rollback mutates session history, so it is rejected during
    # an in-flight turn (prompt.submit would drop the agent's output or clobber
    # the rollback). A file-scoped rollback only touches disk and is allowed.
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
                            _active, _live_view, removed = _rewind_active_session_history(
                                session, len(user_indices) - 1
                            )
                        except Exception as exc:
                            raise RuntimeError(
                                f"checkpoint restored, but session history rewind failed: {exc}"
                            ) from exc
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
    if action != "connect":
        return _err(rid, 4015, f"unknown action: {action}")
    return _browser_connect(rid, params)


@method("plugins.list")
def _(rid, params: dict) -> dict:
    try:
        from hermes_cli.plugins import get_plugin_manager

        return _ok(
            rid,
            {
                "plugins": [
                    {"name": n, "version": getattr(i, "version", "?"), "enabled": getattr(i, "enabled", True)}
                    for n, i in get_plugin_manager()._plugins.items()
                ]
            },
        )
    except Exception as e:
        return _err(rid, 5032, str(e))


@method("config.show")
def _(rid, params: dict) -> dict:
    try:
        cfg = _load_cfg()
        model = _resolve_model()
        from agent.secret_scope import get_secret

        api_key = get_secret("HERMES_API_KEY", "") or cfg.get("api_key", "")
        masked = f"****{api_key[-4:]}" if len(api_key) > 4 else "(not set)"
        base_url = os.environ.get("HERMES_BASE_URL", "") or cfg.get("base_url", "")

        sections = [
            {"title": "Model", "rows": [["Model", model], ["Base URL", base_url or "(default)"], ["API Key", masked]]},
            {
                "title": "Agent",
                "rows": [
                    ["Max Turns", str(_cfg_max_turns(cfg, 500))],
                    ["Toolsets", ", ".join(cfg.get("enabled_toolsets", [])) or "all"],
                    ["Verbose", str(cfg.get("verbose", False))],
                ],
            },
            {
                "title": "Environment",
                "rows": [["Working Dir", os.getcwd()], ["Config File", str(_hermes_home / "config.yaml")]],
            },
        ]
        return _ok(rid, {"sections": sections})
    except Exception as e:
        return _err(rid, 5030, str(e))


# ─── Tools / toolsets / agents ───────────────────────────────────────────────


@method("tools.list")
def _(rid, params: dict) -> dict:
    try:
        return _ok(rid, {"toolsets": _toolset_rows(params, with_tools=True)})
    except Exception as e:
        return _err(rid, 5031, str(e))


@method("toolsets.list")
def _(rid, params: dict) -> dict:
    try:
        return _ok(rid, {"toolsets": _toolset_rows(params, with_tools=False)})
    except Exception as e:
        return _err(rid, 5032, str(e))


@method("tools.show")
def _(rid, params: dict) -> dict:
    try:
        from model_tools import get_toolset_for_tool, get_tool_definitions

        session = _sessions.get(params.get("session_id", ""))
        enabled = getattr(session["agent"], "enabled_toolsets", None) if session else _load_enabled_toolsets()
        # Pre-assembly list: /tools is a discovery surface and must show tools
        # deferred behind the tool_search bridge (same as the CLI).
        tools = get_tool_definitions(enabled_toolsets=enabled, quiet_mode=True, skip_tool_search_assembly=True)
        sections = {}

        for tool in sorted(tools, key=lambda t: t["function"]["name"]):
            name = tool["function"]["name"]
            desc = str(tool["function"].get("description", "") or "").split("\n")[0]
            if ". " in desc:
                desc = desc[: desc.index(". ") + 1]
            sections.setdefault(get_toolset_for_tool(name) or "unknown", []).append({"name": name, "description": desc})

        return _ok(
            rid,
            {
                "sections": [{"name": name, "tools": rows} for name, rows in sorted(sections.items())],
                "total": len(tools),
            },
        )
    except Exception as e:
        return _err(rid, 5034, str(e))


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
def _(rid, params: dict) -> dict:
    try:
        from tools.process_registry import process_registry

        return _ok(
            rid,
            {
                "processes": [
                    {
                        "session_id": p["session_id"],
                        "command": p["command"][:80],
                        "status": p["status"],
                        "uptime": p["uptime_seconds"],
                    }
                    for p in process_registry.list_sessions()
                ]
            },
        )
    except Exception as e:
        return _err(rid, 5033, str(e))


# ─── Cron / learning / skills ────────────────────────────────────────────────


@method("cron.manage")
@_profile_scoped_rpc(5023)
def _(rid, params: dict) -> dict:
    """cronjob() keys off HERMES_HOME, so the optional ``profile`` scope lets a
    per-profile cron store be listed/mutated even when that profile runs its own
    gateway (mirrors skills.manage / mcp.catalog)."""
    from tools.cronjob_tools import cronjob

    action, jid = params.get("action", "list"), params.get("name", "")
    if action == "list":
        # Paused jobs are excluded by default, which reads as deletion in any UI
        # with an enable/disable toggle — forward the flag.
        result = json.loads(
            cronjob(action="list", include_disabled=is_truthy_value(params.get("include_disabled", False)))
        )
        # ``scoped`` proves the gateway honored the profile scope: new clients may
        # treat every job as owned by that profile; older gateways omit it and
        # keep the safe [bot:<name>] compatibility filter.
        profile = str(params.get("profile") or "").strip()
        if profile:
            result["scoped"] = profile
        return _ok(rid, result)
    if action == "add":
        return _ok(
            rid,
            json.loads(
                cronjob(
                    action="create",
                    name=jid,
                    schedule=params.get("schedule", ""),
                    prompt=params.get("prompt", ""),
                    # Optional repeat cap; None keeps the schedule-kind default.
                    repeat=int(params["repeat"]) if str(params.get("repeat", "")).strip().isdigit() else None,
                    # Optional continuity toggle: previous output injected into each run.
                    continuity=(
                        is_truthy_value(params.get("continuity")) if params.get("continuity") is not None else None
                    ),
                    # Optional delivery target, e.g. 'bot-chat[:name]'; empty keeps the cronjob() default.
                    deliver=(str(params.get("deliver") or "").strip() or None),
                )
            ),
        )
    if action in {"remove", "pause", "resume"}:
        return _ok(rid, json.loads(cronjob(action=action, job_id=jid)))
    return _err(rid, 4016, f"unknown cron action: {action}")


@method("learning.frames")
def _(rid, params: dict) -> dict:
    """Pre-render the learning timeline for the TUI ``/journey`` overlay: ``frames``
    (reveal 0→1) plus legend/summary/bucket metadata so Ink walks the tree locally.
    Shares its renderer with ``hermes journey``."""
    try:
        cols = int(params.get("cols", 80) or 80)
        rows = int(params.get("rows", 24) or 24)
        frames = int(params.get("frames", 48) or 48)
    except (TypeError, ValueError):
        cols, rows, frames = 80, 24, 48
    try:
        from agent.learning_graph import build_learning_graph
        from agent.learning_graph_render import render_frames

        payload = build_learning_graph()
        return _ok(rid, render_frames(payload, cols=max(20, cols), rows=max(10, rows), frames=frames))
    except Exception as exc:  # noqa: BLE001
        return _err(rid, 5000, f"learning.frames failed: {exc}")


@method("learning.detail")
def _(rid, params: dict) -> dict:
    """Current content of a journey node, for an edit prefill."""
    try:
        from agent.learning_mutations import node_detail

        return _ok(rid, node_detail(str(params.get("id", ""))))
    except Exception as exc:  # noqa: BLE001
        return _err(rid, 5000, f"learning.detail failed: {exc}")


@method("learning.delete")
def _(rid, params: dict) -> dict:
    """Delete a journey node — skills are archived (restorable), memories removed."""
    try:
        from agent.learning_mutations import delete_node

        return _ok(rid, delete_node(str(params.get("id", ""))))
    except Exception as exc:  # noqa: BLE001
        return _err(rid, 5000, f"learning.delete failed: {exc}")


@method("learning.edit")
def _(rid, params: dict) -> dict:
    """Rewrite a journey node's content (SKILL.md or memory chunk)."""
    try:
        from agent.learning_mutations import edit_node

        return _ok(rid, edit_node(str(params.get("id", "")), str(params.get("content", ""))))
    except Exception as exc:  # noqa: BLE001
        return _err(rid, 5000, f"learning.edit failed: {exc}")


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
    """list/install operate on the scoped profile's skills dir; search/browse/
    inspect hit the shared hub catalog (the override is harmless there)."""
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
def _(rid, params: dict) -> dict:
    try:
        from agent.skill_commands import reload_skills

        result = reload_skills()
        added = result.get("added") or []
        removed = result.get("removed") or []
        total = int(result.get("total") or 0)

        lines = ["Reloading skills..."]
        if not added and not removed:
            lines.append("No new skills detected.")
        if added:
            lines.append("Added skills:")
            lines.extend(f"  - {item.get('name', '')}" for item in added)
        if removed:
            lines.append("Removed skills:")
            lines.extend(f"  - {item.get('name', '')}" for item in removed)
        lines.append(f"{total} skill(s) available")
        return _ok(rid, {"output": "\n".join(lines), "result": result})
    except Exception as e:
        return _err(rid, 5025, str(e))


# ─── MCP catalog + per-profile server lifecycle (mcp.servers.*) ─────────────
#
# Gateway mirrors of the dashboard REST surface (hermes_cli/web_routers/mcp.py) so
# a desktop plugin can manage MCP servers for ANY profile. Persistence reuses
# hermes_cli/mcp_config.py; summaries come from tui_gateway.mcp_rpc_helpers.


@method("mcp.catalog")
@_profile_scoped_rpc(5024)
def _(rid, params: dict) -> dict:
    """Bundled MCP catalog with per-profile install/enable state: ``{servers:
    [{name, description, installed, enabled, requires: [env keys], transport}]}``
    — the same menu `hermes mcp` offers, so UIs know which entries need setup."""
    from hermes_cli import mcp_catalog

    out = []
    for entry in mcp_catalog.list_catalog():
        try:
            requires = [str(k) for k in (getattr(entry, "env_keys", None) or [])]
        except Exception:
            requires = []
        out.append(
            {
                "name": entry.name,
                "description": getattr(entry, "description", "") or "",
                "installed": bool(mcp_catalog.is_installed(entry.name)),
                "enabled": bool(mcp_catalog.is_enabled(entry.name)),
                "requires": requires,
                # TransportSpec object — reduce to its kind string.
                "transport": str(
                    getattr(getattr(entry, "transport", None), "kind", "") or getattr(entry, "transport", "") or "stdio"
                ),
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
    """Add a server to the profile's config.yaml. ``name`` plus EITHER ``preset``
    (catalog id, via ``_apply_mcp_preset``) or ``config`` (url/command/args/env/
    headers/auth/tools). ``bearer_token`` goes to the profile's .env; only the
    ``Authorization`` header template is persisted. Duplicate names → 4090."""
    from hermes_cli.mcp_config import _apply_mcp_preset, _get_mcp_servers, _save_bearer_auth_token, _save_mcp_server

    name = str(params.get("name") or "").strip()
    if name in _get_mcp_servers():
        return _err(rid, 4090, f"server '{name}' already exists")

    preset = str(params.get("preset") or "").strip()
    raw_cfg = params.get("config")
    server_config: dict = dict(raw_cfg) if isinstance(raw_cfg, dict) else {}

    if preset:
        # Fills url/command/args from the preset when omitted; mutates server_config in place.
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
    """Store a credential for a server: the secret goes to the profile's .env under
    ``env_var`` (default ``MCP_<NAME>_API_KEY``); config.yaml gets a reference —
    ``Authorization: Bearer ${ENV}`` header for http, ``env: {VAR: "${ENV}"}`` for
    stdio — matching ``cmd_mcp_configure`` / ``_save_bearer_auth_token``."""
    from hermes_cli.config import load_config, save_config, save_env_value
    from hermes_cli.mcp_config import _bearer_auth_headers, _env_key_for_server, _get_mcp_servers, _strip_bearer_prefix

    name = str(params.get("name") or "").strip()
    value = params.get("value")
    servers = _get_mcp_servers()
    if name not in servers:
        return _err(rid, 4064, f"server '{name}' not found")

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
            headers = _bearer_auth_headers(name)
        else:
            headers = {"Authorization": f"Bearer ${{{env_var}}}"}
        entry["headers"] = headers
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
    """Connect, list tools, disconnect (``_probe_single_server``). Success:
    ``{ok, tools, prompts, resources, oauth_needed, oauth_tokens_present}``;
    failure: ``{ok: false, error, tools: [], oauth_needed}``. Runs on the RPC
    pool (_LONG_HANDLERS): a cold stdio `npx` spawn can block for seconds."""
    from hermes_cli.mcp_config import _get_mcp_servers, _oauth_tokens_present, _probe_single_server

    name = str(params.get("name") or "").strip()
    servers = _get_mcp_servers()
    if name not in servers:
        return _err(rid, 4064, f"server '{name}' not found")

    cfg = servers[name]
    # An `auth: oauth` server that serves tools/list anonymously would probe OK
    # with no token — a false green. Require a token on disk for it.
    needs_oauth_token = cfg.get("auth") == "oauth"
    details: dict = {}
    try:
        tools = _probe_single_server(name, cfg, details=details)
        token_present = _oauth_tokens_present(name) if needs_oauth_token else True
    except Exception as exc:
        return _ok(
            rid,
            {
                "ok": False,
                "error": str(exc),
                "tools": [],
                "oauth_needed": needs_oauth_token,
                "oauth_tokens_present": _oauth_tokens_present(name) if needs_oauth_token else None,
            },
        )
    if not token_present:
        return _ok(
            rid,
            {
                "ok": False,
                "error": "OAuth authentication required — no token found.",
                "tools": [],
                "oauth_needed": True,
                "oauth_tokens_present": False,
            },
        )
    return _ok(
        rid,
        {
            "ok": True,
            "tools": [{"name": t, "description": d} for t, d in tools],
            "prompts": details.get("prompts", 0),
            "resources": details.get("resources", 0),
            "oauth_needed": needs_oauth_token,
            "oauth_tokens_present": True if needs_oauth_token else None,
        },
    )


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

    The client opens ``auth_url`` in the native browser and polls
    ``mcp.servers.oauth.poll`` until ``status == "approved"``. A background
    worker drives the same interactive machinery as ``hermes mcp login``
    (``_probe_single_server`` under ``force_interactive_oauth``) with a loopback
    listener for the redirect. ``client_redirect_uri`` (remote backends): the
    CLIENT hosts the loopback and relays the code via
    ``mcp.servers.oauth.callback`` — the only flow that works when desktop and
    gateway are on different machines. Runs on the RPC pool (_LONG_HANDLERS)."""
    name = str(params.get("name") or "").strip()
    client_redirect_uri = str(params.get("client_redirect_uri") or "").strip() or None
    try:
        from hermes_cli.mcp_config import _get_mcp_servers
        from hermes_constants import get_hermes_home
        from tui_gateway import mcp_oauth_sessions

        servers = _get_mcp_servers()
        if name not in servers:
            return _err(rid, 4064, f"server '{name}' not found")
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
    return _ok(
        rid, {"ok": True, "session_id": result["session_id"], "auth_url": result["auth_url"], "flow": result["flow"]}
    )


@method("mcp.servers.oauth.poll")
@_profile_scoped_rpc(5024, required=_NAME_SESSION, catch_resolve=False)
def _(rid, params: dict) -> dict:
    """Poll a flow → ``{ok, status: pending|approved|error, error_message?, auth_url?,
    tools?}``. On ``approved`` the tokens are persisted for that server/profile;
    the profile scope applies here too so a same-profile token read resolves."""
    from tui_gateway import mcp_oauth_sessions

    name = str(params.get("name") or "").strip()
    session_id = str(params.get("session_id") or "").strip()
    result = mcp_oauth_sessions.poll_flow(session_id, name)
    return _ok(rid, {"ok": True, **result})


@method("mcp.servers.oauth.callback")
@_profile_scoped_rpc(5024, required=_NAME_SESSION, catch_resolve=False)
def _(rid, params: dict) -> dict:
    """Relay a client-captured OAuth redirect (``code``/``state``/``error``) into a
    running flow started with ``client_redirect_uri``. ``{ok: true}`` once
    accepted (state verified in the flow bridge), else ``{ok: false, error_message}``."""
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
        # Bundled backends/platforms/providers run without an explicit enable;
        # report the truthful default instead of "not enabled" (reads as OFF).
        if status == "not enabled" and source == "bundled" and _bundled_default_on(_dir):
            status = "enabled"
        out.append(
            {
                "name": name,
                # Canonical registry key (``image_gen/fal``): names collide across
                # category dirs, so toggles must address the key.
                "key": key,
                "version": str(version or ""),
                "description": desc or "",
                "source": source,
                "status": status,
                # Agent Plugins v1 package (plugin.json) vs a native Hermes plugin.
                "portable": _is_portable_plugin_dir(_dir),
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
    """TUI Plugins Hub backend, sharing discovery + enable/disable primitives with
    ``hermes plugins`` and the dashboard.
      - ``list``    → {plugins: [{name, key, version, description, source, status,
                       portable}], user_count, bundled_count}
      - ``toggle``  → flip ``key`` (or ``name``) per ``enable``; returns the row + {ok, unchanged}
      - ``install`` → git-clone ``identifier``/``repo`` into ~/.hermes/plugins/
                       (``force``, ``enable`` default True); returns the dashboard dict.
    Optional ``profile`` scopes to that profile's HERMES_HOME (mcp.servers.* contract)."""
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
        from hermes_cli._subprocess_compat import windows_hide_flags

        r = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30,
            cwd=os.getcwd(),
            encoding="utf-8",
            errors="replace",  # lossy decode: see cli.exec
            stdin=subprocess.DEVNULL,
            creationflags=windows_hide_flags(),
        )
        return _ok(rid, {"stdout": r.stdout[-4000:], "stderr": r.stderr[-2000:], "code": r.returncode})
    except subprocess.TimeoutExpired:
        return _err(rid, 5002, "command timed out (30s)")
    except Exception as e:
        return _err(rid, 5003, str(e))


def register(server) -> None:
    """Rebind this module's helpers + handlers onto ``server`` and register the handlers."""
    bind_module(globals(), server, skip=("_",))
