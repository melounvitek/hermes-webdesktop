"""Streamlined setup flows extracted from hermes_cli/setup.py: the Nous Portal
one-shot (`hermes portal`), first-time quick setup, Blank Slate setup and the
`--quick` missing-items pass. Names from setup.py are imported lazily per
function so test patches on ``hermes_cli.setup`` take effect."""

import logging
import os
import sys
from pathlib import Path

logger = logging.getLogger("hermes_cli.setup")

# (env-var name substring, platform label, emoji) — order matters: first match wins.
_MESSAGING_PLATFORMS = (("TELEGRAM", "Telegram", "📱"), ("DISCORD", "Discord", "💬"), ("SLACK", "Slack", "💼"))


def _reload_config_into(config: dict) -> None:
    """Re-sync the in-memory config dict from disk after a sub-flow that saved
    via its own load/save cycle, so a later save_config(config) can't clobber it."""
    from hermes_cli.setup import load_config
    _refreshed = load_config()
    config.clear()
    config.update(_refreshed)


def _prompt_and_save_env_var(var: dict, saved_msg: str, skipped_msg: str) -> None:
    """Prompt for one env-var value (masked when secret); persist and confirm, or report the skip."""
    from hermes_cli.setup import print_success, print_warning, prompt, save_env_value
    value = prompt(f"  {var.get('prompt', var['name'])}", password=bool(var.get("password")))
    if value:
        save_env_value(var["name"], value)
        print_success(saved_msg)
    else:
        print_warning(skipped_msg)


def _run_portal_one_shot(config: dict) -> None:
    """One-shot Nous Portal setup (``hermes setup --portal`` / ``hermes portal``).

    Login, model pick, provider switch and Tool Gateway opt-in are all delegated to
    ``_model_flow_nous`` — the same flow quick setup and ``hermes model`` use for Nous — so
    there is one source of truth and ``hermes portal`` always offers a picker.
    """
    from hermes_cli.setup import Colors, color, _info, load_config, print_error, print_info, print_success
    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.MAGENTA))
    print(color("│     ⚕ Hermes Setup — Nous Portal (one-shot)             │", Colors.MAGENTA))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.MAGENTA))
    _info(None, "  One subscription, 300+ models, plus the Tool Gateway:",
          "    web search, image generation, TTS, browser automation",
          "    — all routed through your Nous Portal sub.", None,
          "  Sign up: https://portal.nousresearch.com/manage-subscription", None)

    try:
        from hermes_cli.main import _model_flow_nous
        _model_flow_nous(config)
    except (KeyboardInterrupt, EOFError, SystemExit):
        # _login_nous raises SystemExit(130)/(1) on cancel/failure; the expired-session re-login
        # path inside _model_flow_nous only catches Exception, so SystemExit would kill the CLI.
        _info(None, "  Setup cancelled.", "  You can retry later with `hermes portal`.")
        return
    except Exception as exc:
        logger.debug("_model_flow_nous error during `hermes portal`: %s", exc)
        print()
        print_error(f"  Nous Portal setup encountered an error: {exc}")
        print_info("  You can retry later with `hermes portal`.")
        return

    # Re-sync from disk so a caller's later save_config(config) can't clobber the login save.
    try:
        _refreshed = load_config()
        if isinstance(_refreshed, dict):
            config.clear()
            config.update(_refreshed)
    except Exception:
        pass

    print()
    print_success("Portal setup complete.")
    _info("  Run `hermes portal info` to inspect routing.", "  Run `hermes` to start chatting.")


def _run_first_time_quick_setup(config: dict, hermes_home, is_existing: bool):
    """Streamlined first-time setup via Nous Portal: OAuth, model, terminal & messaging.

    Everything else gets sensible defaults; customize later via ``hermes setup
    <section>`` or switch providers with ``hermes model``.
    """
    from hermes_cli.setup import (
        _apply_default_agent_settings, _info, print_header, print_info, _print_macos_fda_tip,
        _print_setup_summary, print_success, print_warning, prompt_choice, save_config, setup_gateway,
        setup_terminal_backend,
    )
    # Step 1: Nous Portal — OAuth login + model selection (provider set to "nous" by the save).
    print()
    print_header("Nous Portal")
    _info("One subscription, 300+ models, plus the Tool Gateway:",
          "  web search, image generation, TTS, browser automation.",
          "Sign up: https://portal.nousresearch.com/manage-subscription", None)
    try:
        from hermes_cli.main import _model_flow_nous
        _model_flow_nous(config)
    except (KeyboardInterrupt, EOFError):
        _info(None, "Nous Portal setup cancelled.")
    except Exception as exc:
        logger.debug("_model_flow_nous error during quick setup: %s", exc)
        print_warning(f"Nous Portal setup encountered an error: {exc}")
        print_info("You can try again later with: hermes model")

    # The wizard's later save_config(config) must not clobber the login/model save (#4172).
    _reload_config_into(config)

    # Step 2: Terminal Backend; Step 3: defaults for everything else.
    setup_terminal_backend(config)
    _apply_default_agent_settings(config)
    save_config(config)

    # Step 4: Offer messaging gateway setup
    print()
    gateway_choice = prompt_choice("Connect a messaging platform? (Telegram, Discord, etc.)", [
        "Set up messaging now (recommended)", "Skip — set up later with 'hermes setup gateway'",
    ], 0)

    if gateway_choice == 0:
        setup_gateway(config)
        save_config(config)
    else:
        # Messaging skipped — still install/start the gateway service so cron
        # jobs run and platforms come alive as soon as tokens are added later
        # (e.g. via `hermes import` from another machine).
        from hermes_cli.gateway import ensure_gateway_service
        ensure_gateway_service(context="setup")

    print()
    print_success("Setup complete! You're ready to go.")
    _info(None, "  Configure all settings:    hermes setup")
    if gateway_choice != 0:
        print_info("  Connect Telegram/Discord:  hermes setup gateway")
    _print_macos_fda_tip()
    print()

    _print_setup_summary(config, hermes_home)


def _print_macos_fda_tip() -> None:
    """One-time macOS onboarding tip: a single Full Disk Access grant kills
    every per-folder permission prompt, permanently (issue #52010 follow-up).

    Uses the same prompt-free probe as doctor's check_macos_full_disk_access
    (the TCC db dir is FDA-gated but probing it never triggers a dialog).
    Silent on non-macOS and when FDA is already granted or indeterminate.
    """
    from hermes_cli.setup import _info
    if sys.platform != "darwin":
        return
    tcc_dir = Path.home() / "Library" / "Application Support" / "com.apple.TCC"
    try:
        os.listdir(tcc_dir)
        return  # already granted — nothing to teach
    except PermissionError:
        pass
    except OSError:
        return  # indeterminate — don't nag
    _info(None, "  macOS tip: silence ALL folder permission prompts with one switch —",
          "  System Settings → Privacy & Security → Full Disk Access → enable",
          "  your terminal (and Hermes.app if you use Desktop), or run:",
          "    open \"x-apple.systempreferences:com.apple.preference" ".security?Privacy_AllFiles\"",
          "  The grant is permanent — it survives every Hermes update.")


def _blank_slate_minimal_toolsets(config: dict):
    """Write the minimal toolset state for a Blank Slate install.

    Only ``file``, ``terminal``, ``vision`` and ``skills`` stay on: ``read_file``
    can't read images (it points at ``vision_analyze``), and the always-seeded
    ``hermes-agent`` skill needs ``skill_view``. Two layers enforce the selection:
    1. ``platform_toolsets["cli"]`` — an explicit configurable-key list the resolver
       treats as authoritative (``has_explicit_config``), so defaults aren't re-expanded.
    2. ``agent.disabled_toolsets`` — a global hard-suppression list applied last in
       ``_get_platform_tools``, overriding even the non-configurable platform-toolset
       recovery that would re-add e.g. ``kanban``. Users re-enable via ``hermes tools``
       (rewrites ``platform_toolsets``) or by editing ``agent.disabled_toolsets``.
    """
    keep = {"file", "terminal", "vision", "skills"}
    config.setdefault("platform_toolsets", {})["cli"] = sorted(keep)

    try:
        from toolsets import TOOLSETS
        from hermes_cli.tools_config import CONFIGURABLE_TOOLSETS, _get_plugin_toolset_keys

        all_keys = {k for k, _, _ in CONFIGURABLE_TOOLSETS}
        all_keys.update(_get_plugin_toolset_keys())
        # Plain TOOLSETS entries catch recovered toolsets like ``kanban``. Skip
        # "hermes-*" platform composites, "includes" groupings, and posture toolsets
        # (session-level picks by agent/coding_context.py — disabling them would make
        # model_tools subtract terminal/read_file from the minimal surface, #57315).
        for k, tdef in TOOLSETS.items():
            if k.startswith("hermes-"):
                continue
            if isinstance(tdef, dict) and (tdef.get("includes") or tdef.get("posture")):
                continue
            all_keys.add(k)

        disabled = sorted(all_keys - keep)
        if disabled:
            config.setdefault("agent", {})["disabled_toolsets"] = disabled
    except Exception as exc:
        logger.debug("blank-slate disabled_toolsets computation skipped: %s", exc)


def _blank_slate_minimize_config(config: dict):
    """Turn OFF the optional config features (compression, memory/profile capture,
    checkpoints, smart routing, auto session reset; quiet display). All opt-in
    afterwards via ``hermes setup agent`` / ``hermes config set``."""
    config.setdefault("agent", {})["max_turns"] = 90
    config.setdefault("compression", {})["enabled"] = False
    mem = config.setdefault("memory", {})
    mem["memory_enabled"] = False
    mem["user_profile_enabled"] = False
    config.setdefault("checkpoints", {})["enabled"] = False
    config.setdefault("smart_model_routing", {})["enabled"] = False
    config.setdefault("session_reset", {})["mode"] = "none"
    config.setdefault("display", {})["tool_progress"] = "all"


def _run_blank_slate_setup(config: dict, hermes_home, is_existing: bool):
    """Blank Slate setup — essentials only (provider/model, file + terminal), everything
    else OFF; then either finish now or walk through opting capabilities back in.
    Nothing is enabled that the user did not explicitly choose."""
    from hermes_cli.setup import (
        _blank_slate_minimal_toolsets, _blank_slate_minimize_config, _blank_slate_walkthrough, _info,
        print_header, print_info, _print_setup_summary, print_success, prompt_choice, save_config,
        setup_model_provider, setup_terminal_backend,
    )
    print()
    print_header("Blank Slate Setup")
    _info("Everything starts OFF. First we force-enable only what's required",
          "to run an agent, then you choose whether to stop there or walk",
          "through enabling more — opting in to exactly what you want.", "",
          "Forced on: Provider & Model, File Operations, Terminal, Vision, Skills.",
          "Everything else (web, browser, code exec, memory,",
          "delegation, cron, plugins, MCP, …) starts disabled. The",
          "essential `hermes-agent` skill is always kept so the agent",
          "can help you drive and configure Hermes itself.", None)

    # Step 1: Provider & Model (REQUIRED — the agent cannot run without it)
    print_header("Step 1 — Provider & Model (required)")
    setup_model_provider(config)
    save_config(config)

    # Step 2: Terminal backend (where commands run — a core decision)
    print_header("Step 2 — Terminal Backend")
    setup_terminal_backend(config)

    # Step 3: Lock in the minimal toolset + minimized config knobs
    _blank_slate_minimal_toolsets(config)
    _blank_slate_minimize_config(config)
    save_config(config)
    print()
    print_success("Minimal baseline applied:")
    print_info("  Toolsets: file, terminal, vision, skills (everything else off)")
    print_info("  Compression, memory, checkpoints, smart routing: off")

    # The fork: stop here, or walk through enabling things
    print()
    print_header("How far do you want to go?")
    path = prompt_choice("Your minimal agent is ready. What next?", [
        "Start with everything disabled — finish now (most minimal)",
        "Walk through all configurations — opt in to tools, skills, plugins, MCP",
    ], 0)

    if path == 0:
        save_config(config)
        # Blank Slate means no bundled skills; record the opt-out so future
        # `hermes update` runs don't re-inject them. Essential skills (the
        # `hermes-agent` operating manual) are still seeded by the sync.
        try:
            from tools.skills_sync import set_bundled_skills_opt_out, sync_skills
            set_bundled_skills_opt_out(True)
            sync_skills(quiet=True)
        except Exception as exc:
            logger.debug("blank-slate skill opt-out error: %s", exc)
        print()
        print_success("Blank Slate setup complete — minimal agent ready.")
        _info("Enable anything later, on demand:", "  Enable tools:        hermes tools",
              "  Seed skills:         hermes skills opt-in --sync", "  Add MCP servers:     hermes mcp add",
              "  Enable plugins:      hermes plugins", "  Tune agent settings: hermes setup agent", None)
        _print_setup_summary(config, hermes_home)
        return

    _blank_slate_walkthrough(config, hermes_home)


def _blank_slate_walkthrough(config: dict, hermes_home):
    """Opt-in walkthrough for Blank Slate: skills, tools, plugins, MCP, gateway."""
    from hermes_cli.setup import (
        _info, print_header, print_info, _print_setup_summary, print_success, print_warning, prompt_yes_no,
        save_config, setup_gateway,
    )
    # Bundled skills — default to NONE, offer to seed all
    print()
    print_header("Bundled Skills")
    print_info("Blank Slate ships with NO bundled skills by default.")
    seed_skills = prompt_yes_no(
        "Seed the full bundled skill catalog? (No = start with zero skills)", default=False
    )
    try:
        from tools.skills_sync import set_bundled_skills_opt_out, sync_skills
        if seed_skills:
            # Make sure no stale opt-out marker blocks the seed, then sync.
            set_bundled_skills_opt_out(False)
            result = sync_skills(quiet=True)
            copied = len(result.get("copied", [])) if isinstance(result, dict) else 0
            print_success(f"Seeded {copied} bundled skills.")
        else:
            set_bundled_skills_opt_out(True)
            # Essential skills (`hermes-agent`) are still seeded for an opted-out profile.
            sync_skills(quiet=True)
            _info("No skills seeded (except the essential `hermes-agent`",
                  "skill). A .no-bundled-skills marker keeps future",
                  "`hermes update` runs from re-injecting them. Opt back in any",
                  "time with `hermes skills opt-in --sync`.")
    except Exception as exc:
        logger.debug("blank-slate skill handling error: %s", exc)
        print_warning(f"Skill setup step encountered an error: {exc}")

    # Walk through enabling additional tools
    print()
    print_header("Tools")
    _info("Pick exactly which additional toolsets to turn on.",
          "(file and terminal are already on; leave the rest off if you want", " the most minimal agent.)")
    if prompt_yes_no("Open the tool selector to enable more tools?", default=False):
        try:
            from hermes_cli.tools_config import tools_command
            tools_command(first_install=False, config=config)
            _reload_config_into(config)  # tools_command saves via its own load/save cycle
        except Exception as exc:
            logger.debug("blank-slate tools_command error: %s", exc)
            print_warning(f"Tool selector encountered an error: {exc}")
    else:
        print_info("Keeping the minimal toolset. Add tools later with `hermes tools`.")

    # Built-in plugins (off unless chosen)
    print()
    print_header("Plugins")
    if prompt_yes_no("Review and enable built-in plugins now?", default=False):
        print_info("Manage plugins with `hermes plugins list` / `hermes plugins install`.")
    else:
        print_info("No plugins enabled. Add later with `hermes plugins`.")

    # MCP servers (off unless chosen)
    print()
    print_header("MCP Servers")
    if prompt_yes_no("Add an MCP server now?", default=False):
        print_info("Add servers with `hermes mcp add <name> --url ... | --command ...`.")
    else:
        print_info("No MCP servers configured. Add later with `hermes mcp add`.")

    # Optional messaging gateway
    print()
    if prompt_yes_no("Connect a messaging platform (Telegram, Discord, …)?", default=False):
        setup_gateway(config)

    save_config(config)

    print()
    print_success("Blank Slate setup complete — minimal agent ready.")
    _info("  Enable more tools:   hermes tools", "  Seed skills:         hermes skills opt-in --sync",
          "  Add MCP servers:     hermes mcp add", "  Tune agent settings: hermes setup agent", None)

    _print_setup_summary(config, hermes_home)


def _run_quick_setup(config: dict, hermes_home):
    """Quick setup — only configure items that are missing."""
    from hermes_cli.setup import (
        Colors, color, _info, print_header, print_info, _print_setup_summary, print_success, _prompt_api_key, prompt_checklist, save_config,
    )
    from hermes_cli.config import (get_missing_env_vars, get_missing_config_fields, check_config_version)
    print()
    print_header("Quick Setup — Missing Items Only")

    # Check what's missing
    missing_required = [v for v in get_missing_env_vars(required_only=False) if v.get("is_required")]
    missing_optional = [v for v in get_missing_env_vars(required_only=False) if not v.get("is_required")]
    missing_config = get_missing_config_fields()
    current_ver, latest_ver = check_config_version()

    if not (missing_required or missing_optional or missing_config or current_ver < latest_ver):
        print_success("Everything is configured! Nothing to do.")
        _info(None, "Run 'hermes setup' and choose 'Full Setup' to reconfigure,",
              "or pick a specific section from the menu.")
        return

    if missing_required:
        _info(None, f"{len(missing_required)} required setting(s) missing:")
        for var in missing_required:
            print(f"     • {var['name']}")
        print()
        for var in missing_required:
            print()
            print(color(f"  {var['name']}", Colors.CYAN))
            print_info(f"  {var.get('description', '')}")
            if var.get("url"):
                print_info(f"  Get key at: {var['url']}")
            _prompt_and_save_env_var(var, f"  Saved {var['name']}", f"  Skipped {var['name']}")

    missing_tools = [v for v in missing_optional if v.get("category") == "tool"]
    missing_messaging = [
        v for v in missing_optional if v.get("category") == "messaging" and not v.get("advanced")
    ]

    if missing_tools:  # checklist, then the API-key screen for each pick
        print()
        print_header("Tool API Keys")
        labels = []
        for var in missing_tools:
            tools = var.get("tools", [])
            tools_str = f" → {', '.join(tools[:2])}" if tools else ""
            labels.append(f"{var.get('description', var['name'])}{tools_str}")
        for idx in prompt_checklist("Which tools would you like to configure?", labels):
            _prompt_api_key(missing_tools[idx])

    if missing_messaging:  # checklist, then prompt for each selected platform's vars
        print()
        print_header("Messaging Platforms")
        _info("Connect Hermes to messaging apps to chat from anywhere.",
              "You can configure these later with 'hermes setup gateway'.")
        # Group by platform in first-seen order; vars matching no platform are dropped.
        grouped: dict[str, list] = {}
        emojis = {}
        for var in missing_messaging:
            for needle, plat, emoji in _MESSAGING_PLATFORMS:
                if needle in var["name"]:
                    grouped.setdefault(plat, []).append(var)
                    emojis[plat] = emoji
                    break
        platform_order = list(grouped)
        labels = [f"{emojis[p]} {p}" for p in platform_order]
        for idx in prompt_checklist("Which platforms would you like to set up?", labels):
            plat = platform_order[idx]
            print()
            print(color(f"  ─── {emojis[plat]} {plat} ───", Colors.CYAN))
            print()
            for var in grouped[plat]:
                print_info(f"  {var.get('description', '')}")
                if var.get("url"):
                    print_info(f"  {var['url']}")
                _prompt_and_save_env_var(var, "  ✓ Saved", "  Skipped")
                print()

    # Handle missing config fields
    if missing_config:
        _info(None, f"Adding {len(missing_config)} new config option(s) with defaults...")
        for field in missing_config:
            print_success(f"  Added {field['key']} = {field['default']}")

        config["_config_version"] = latest_ver
        save_config(config)

    _print_setup_summary(config, hermes_home)
