"""Agent-facing subcommand handlers: memory, acp, tools, insights, monitoring, skills (+trust).

Split out of ``hermes_cli/main.py``; every moved name is re-imported there, so
``hermes_cli.main.<name>`` keeps resolving (and monkeypatching) as before.
Names that stay in main are imported lazily inside the functions that use them
(call-time resolution keeps ``hermes_cli.main.<name>`` patches effective and
avoids an import cycle).
"""

import sys


def cmd_memory(args):
    sub = getattr(args, "memory_command", None)
    if sub == "off":
        from hermes_cli.config import load_config, save_config

        config = load_config()
        if not isinstance(config.get("memory"), dict):
            config["memory"] = {}
        config["memory"]["provider"] = ""
        save_config(config)
        print("\n  ✓ Memory provider: built-in only")
        print("  Saved to config.yaml\n")
    elif sub == "reset":
        from hermes_constants import get_hermes_home, display_hermes_home

        mem_dir = get_hermes_home() / "memories"
        target = getattr(args, "target", "all")
        files_to_reset = []
        if target in {"all", "memory"}:
            files_to_reset.append(("MEMORY.md", "agent notes"))
        if target in {"all", "user"}:
            files_to_reset.append(("USER.md", "user profile"))

        # Check what exists
        existing = [
            (f, desc) for f, desc in files_to_reset if (mem_dir / f).exists()
        ]
        if not existing:
            print(
                f"\n  Nothing to reset — no memory files found in {display_hermes_home()}/memories/\n"
            )
            return

        print("\n  This will permanently erase the following memory files:")
        for f, desc in existing:
            path = mem_dir / f
            size = path.stat().st_size
            print(f"    ◆ {f} ({desc}) — {size:,} bytes")

        if not getattr(args, "yes", False):
            try:
                answer = input("\n  Type 'yes' to confirm: ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                print("\n  Cancelled.\n")
                return
            if answer != "yes":
                print("  Cancelled.\n")
                return

        for f, desc in existing:
            (mem_dir / f).unlink()
            print(f"  ✓ Deleted {f} ({desc})")

        print(
            "\n  Memory reset complete. New sessions will start with a blank slate."
        )
        print(f"  Files were in: {display_hermes_home()}/memories/\n")
    else:
        from hermes_cli.memory_setup import memory_command

        memory_command(args)


def cmd_acp(args):
    """Launch Hermes Agent as an ACP server."""
    try:
        from acp_adapter.entry import main as acp_main

        acp_argv = []
        if getattr(args, "acp_version", False):
            acp_argv.append("--version")
        if getattr(args, "check", False):
            acp_argv.append("--check")
        if getattr(args, "setup", False):
            acp_argv.append("--setup")
        if getattr(args, "setup_browser", False):
            acp_argv.append("--setup-browser")
        if getattr(args, "assume_yes", False):
            acp_argv.append("--yes")
        acp_main(acp_argv)
    except ImportError:
        print("ACP dependencies not installed.", file=sys.stderr)
        print("Install them with:  pip install -e '.[acp]'", file=sys.stderr)
        sys.exit(1)


def cmd_tools(args):
    from hermes_cli.main import _require_tty
    action = getattr(args, "tools_action", None)
    if action in {"list", "disable", "enable"}:
        from hermes_cli.tools_config import tools_disable_enable_command

        tools_disable_enable_command(args)
    elif action == "post-setup":
        from hermes_cli.tools_config import run_post_setup_command

        sys.exit(run_post_setup_command(args))
    else:
        _require_tty("tools")
        from hermes_cli.tools_config import tools_command

        tools_command(args)


def cmd_insights(args):
    db = None
    try:
        from hermes_state import SessionDB
        from agent.insights import InsightsEngine

        db = SessionDB()
        engine = InsightsEngine(db)
        report = engine.generate(days=args.days, source=args.source)
        print(engine.format_terminal(report))
    except Exception as e:
        print(f"Error generating insights: {e}")
    finally:
        if db is not None:
            try:
                db.close()
            except Exception:
                pass


def cmd_monitoring(args):
    """Gateway monitoring status: health & diagnostics export posture."""
    from hermes_cli.config import load_config

    action = getattr(args, "monitoring_action", None) or "status"
    config = load_config()
    mon_raw = config.get("monitoring")
    mon: dict = mon_raw if isinstance(mon_raw, dict) else {}

    if action == "status":
        from agent.monitoring import otlp_exporter

        gh_raw = mon.get("gateway_health_export")
        gh: dict = gh_raw if isinstance(gh_raw, dict) else {}
        export_raw = mon.get("export")
        export_cfg: dict = export_raw if isinstance(export_raw, dict) else {}
        otlp_raw = export_cfg.get("otlp")
        otlp: dict = otlp_raw if isinstance(otlp_raw, dict) else {}

        print("Gateway monitoring")
        print(f"  Health export:  {'enabled' if gh.get('enabled') else 'disabled'} "
              f"(monitoring.gateway_health_export.enabled)")
        if gh.get("enabled"):
            print(f"    Metrics:            {'on' if gh.get('metrics_enabled', True) else 'off'} "
                  f"(interval {gh.get('export_interval_seconds', 60)}s)")
            print(f"    Diagnostic events:  {'on' if gh.get('diagnostic_events_enabled', True) else 'off'}")
            print(f"    Warning/error logs: {'on' if gh.get('warning_error_events_enabled', True) else 'off'} "
                  f"(interval {gh.get('logs_export_interval_seconds', 5)}s)")
            print("    Content safety:     always on "
                  "(rendered messages are never exported; not configurable)")
        endpoint = otlp.get("endpoint") or ""
        if otlp.get("enabled") and endpoint:
            print(f"  OTLP endpoint:  {endpoint}")
        else:
            print("  OTLP endpoint:  not configured (monitoring.export.otlp)")
        print(f"  OTel SDK:       {'installed' if otlp_exporter.is_available() else 'not installed'} "
              f"(optional extra: hermes-agent[otlp])")
        print("\n  Scope: gateway service health + redacted diagnostics only.")
        print("  No prompts, messages, tool args/results, usage analytics, or traces.")
        return

    print(f"Unknown monitoring action: {action}", file=sys.stderr)
    sys.exit(2)


def cmd_skills(args):
    # Route 'config' action to skills_config module
    from hermes_cli.main import _require_tty
    if getattr(args, "skills_action", None) == "config":
        _require_tty("skills config")
        from hermes_cli.skills_config import skills_command as skills_config_command

        skills_config_command(args)
    elif getattr(args, "skills_action", None) in ("trust", "untrust"):
        _cmd_skills_trust(args)
    else:
        from hermes_cli.skills_hub import skills_command

        skills_command(args)


def _cmd_skills_trust(args):
    """``hermes skills trust [path]`` / ``hermes skills untrust [path]``.

    Manages ``skills.trusted_project_dirs`` in config.yaml. With no path,
    operates on the project root enclosing the current directory (nearest
    ancestor with ``.git``).
    """
    from pathlib import Path

    from agent.skill_utils import (
        PROJECT_SKILLS_SUBDIRS,
        _candidate_project_skills_dirs,
        find_project_root,
        iter_skill_index_files,
    )
    from hermes_cli.config import load_config, save_config

    action = args.skills_action
    raw_path = getattr(args, "path", None)
    if raw_path:
        root = Path(raw_path).expanduser().resolve()
        if not root.is_dir():
            print(f"Not a directory: {root}")
            return
    else:
        root = find_project_root()
        if root is None:
            print(
                "Not inside a git checkout. Run from a project directory or "
                "pass the project root path explicitly."
            )
            return

    config = load_config()
    skills_cfg = config.setdefault("skills", {})
    trusted = skills_cfg.get("trusted_project_dirs") or []
    if not isinstance(trusted, list):
        trusted = [trusted]
    trusted = [str(t) for t in trusted]
    root_str = str(root)

    if action == "untrust":
        kept = [t for t in trusted if str(Path(t).expanduser().resolve()) != root_str]
        if len(kept) == len(trusted):
            print(f"{root} was not trusted.")
            return
        skills_cfg["trusted_project_dirs"] = kept
        save_config(config)
        print(f"Untrusted: {root}")
        print("Project skills from this repo will no longer load.")
        return

    # trust
    if any(str(Path(t).expanduser().resolve()) == root_str for t in trusted):
        print(f"Already trusted: {root}")
    else:
        trusted.append(root_str)
        skills_cfg["trusted_project_dirs"] = trusted
        save_config(config)
        print(f"Trusted: {root}")

    # Show what this unlocks
    count = 0
    for d in _candidate_project_skills_dirs(root):
        count += sum(1 for _ in iter_skill_index_files(d, "SKILL.md"))
    if count:
        print(
            f"{count} project skill(s) will load in sessions started inside "
            "this repo (they take precedence over same-named profile skills)."
        )
    else:
        subdirs = " or ".join(PROJECT_SKILLS_SUBDIRS)
        print(f"No project skills found yet — add them under {subdirs}.")
