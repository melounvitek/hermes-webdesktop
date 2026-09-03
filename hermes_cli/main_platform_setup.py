"""Interactive messaging-platform setup wizards: WhatsApp (bridge + Cloud API), Slack manifest, Skill Sync.

Split out of ``hermes_cli/main.py``; every moved name is re-imported there, so
``hermes_cli.main.<name>`` keeps resolving (and monkeypatching) as before.
Names that stay in main are imported lazily inside the functions that use them
(call-time resolution keeps ``hermes_cli.main.<name>`` patches effective and
avoids an import cycle).
"""

import shutil
import subprocess
import sys

from hermes_cli.cli_output import line_input


def cmd_whatsapp(args):
    """Set up WhatsApp: choose mode, configure, install bridge, pair via QR."""
    from hermes_cli.main import _require_tty, get_hermes_home
    _require_tty("whatsapp")
    from hermes_cli.config import get_env_value, save_env_value
    from hermes_constants import find_node_executable, with_hermes_node_path

    print()
    print("⚕ WhatsApp Setup")
    print("=" * 50)

    # ── Step 1: Choose mode ──────────────────────────────────────────────
    current_mode = get_env_value("WHATSAPP_MODE") or ""
    if not current_mode:
        print()
        print("How will you use WhatsApp with Hermes?")
        print()
        print("  1. Separate bot number (recommended)")
        print("     People message the bot's number directly — cleanest experience.")
        print(
            "     Requires a second phone number with WhatsApp installed on a device.")
        print()
        print("  2. Personal number (self-chat)")
        print("     You message yourself to talk to the agent.")
        print("     Quick to set up, but the UX is less intuitive.")
        print()
        try:
            choice = input("  Choose [1/2]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nSetup cancelled.")
            return

        if choice == "1":
            save_env_value("WHATSAPP_MODE", "bot")
            wa_mode = "bot"
            print("  ✓ Mode: separate bot number")
            print()
            print("  ┌─────────────────────────────────────────────────┐")
            print("  │  Getting a second number for the bot:           │")
            print("  │                                                 │")
            print("  │  Easiest: Install WhatsApp Business (free app)  │")
            print("  │  on your phone with a second number:            │")
            print("  │    • Dual-SIM: use your 2nd SIM slot            │")
            print("  │    • Google Voice: free US number (voice.google) │")
            print("  │    • Prepaid SIM: $3-10, verify once            │")
            print("  │                                                 │")
            print("  │  WhatsApp Business runs alongside your personal │")
            print("  │  WhatsApp — no second phone needed.             │")
            print("  └─────────────────────────────────────────────────┘")
        else:
            save_env_value("WHATSAPP_MODE", "self-chat")
            wa_mode = "self-chat"
            print("  ✓ Mode: personal number (self-chat)")
    else:
        wa_mode = current_mode
        mode_label = (
            "separate bot number" if wa_mode == "bot" else "personal number (self-chat)")
        print(f"\n✓ Mode: {mode_label}")

    # ── Step 2: Mode is selected, will enable WhatsApp only after pairing ──
    # We intentionally don't write WHATSAPP_ENABLED=true here.  If the user
    # aborts the wizard later (Ctrl+C, failed npm install, missed QR scan),
    # we'd otherwise leave .env claiming WhatsApp is ready when the bridge
    # has no creds.json.  Every subsequent `hermes gateway` then paid a 30s
    # bridge-bootstrap timeout and queued WhatsApp for indefinite retries.
    # Now: aborted setup leaves WHATSAPP_ENABLED unset → gateway skips it.
    # Re-runs that already have WHATSAPP_ENABLED=true (from a prior
    # successful pairing) stay enabled — we just don't write it pre-emptively.
    print()
    if (get_env_value("WHATSAPP_ENABLED") or "").lower() == "true":
        print("✓ WhatsApp is already enabled")

    # ── Step 3: Allowed users ────────────────────────────────────────────
    current_users = get_env_value("WHATSAPP_ALLOWED_USERS") or ""
    if current_users:
        print(f"✓ Allowed users: {current_users}")
        try:
            response = input("\n  Update allowed users? [y/N] ").strip()
        except (EOFError, KeyboardInterrupt):
            response = "n"
        if response.lower() in {"y", "yes"}:
            if wa_mode == "bot":
                phone = line_input(
                    "  Phone numbers that can message the bot (comma-separated): ").strip()
            else:
                phone = line_input("  Your phone number (e.g. 15551234567): ").strip()
            if phone:
                save_env_value("WHATSAPP_ALLOWED_USERS", phone.replace(" ", ""))
                print(f"  ✓ Updated to: {phone}")
    else:
        print()
        if wa_mode == "bot":
            print("  Who should be allowed to message the bot?")
            phone = line_input(
                "  Phone numbers (comma-separated, or * for anyone): ").strip()
        else:
            phone = line_input("  Your phone number (e.g. 15551234567): ").strip()
        if phone:
            save_env_value("WHATSAPP_ALLOWED_USERS", phone.replace(" ", ""))
            print(f"  ✓ Allowed users set: {phone}")
        else:
            print("  ⚠ No allowlist — the agent will respond to ALL incoming messages")

    # ── Step 4: Install bridge dependencies ──────────────────────────────
    from gateway.platforms.whatsapp_common import resolve_whatsapp_bridge_dir
    bridge_dir = resolve_whatsapp_bridge_dir()
    bridge_script = bridge_dir / "bridge.js"

    if not bridge_script.exists():
        print(f"\n✗ Bridge script not found at {bridge_script}")
        return

    if not (bridge_dir / "node_modules").exists():
        print(
            "\n→ Installing WhatsApp bridge dependencies (this can take a few minutes)...")
        npm = find_node_executable("npm")
        if not npm:
            print("  ✗ npm not found on PATH — install Node.js first")
            return
        try:
            result = subprocess.run(
                [npm, "install", "--no-fund", "--no-audit", "--progress=false"],
                cwd=str(bridge_dir),
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=with_hermes_node_path())
        except KeyboardInterrupt:
            print("\n  ✗ Install cancelled")
            return
        if result.returncode != 0:
            err = (result.stderr or "").strip()
            preview = "\n".join(err.splitlines()[-30:]) if err else "(no output)"
            print("  ✗ npm install failed:")
            print(preview)
            return
        print("  ✓ Dependencies installed")
    else:
        print("✓ Bridge dependencies already installed")

    # ── Step 5: Check for existing session ───────────────────────────────
    session_dir = get_hermes_home() / "whatsapp" / "session"
    session_dir.mkdir(parents=True, exist_ok=True)

    if (session_dir / "creds.json").exists():
        print("✓ Existing WhatsApp session found")
        try:
            response = input(
                "\n  Re-pair? This will clear the existing session. [y/N] ").strip()
        except (EOFError, KeyboardInterrupt):
            response = "n"
        if response.lower() in {"y", "yes"}:
            shutil.rmtree(session_dir, ignore_errors=True)
            session_dir.mkdir(parents=True, exist_ok=True)
            print("  ✓ Session cleared")
        else:
            # Existing pairing — ensure WHATSAPP_ENABLED reflects that.
            # (Older installs may have lost the env var; covers re-runs
            # where the user picked "no, keep my session" but the var
            # was never set or got removed.)
            if (get_env_value("WHATSAPP_ENABLED") or "").lower() != "true":
                save_env_value("WHATSAPP_ENABLED", "true")
            print("\n✓ WhatsApp is configured and paired!")
            print("  Start the gateway with: hermes gateway")
            return

    # ── Step 6: QR code pairing ──────────────────────────────────────────
    print()
    print("─" * 50)
    if wa_mode == "bot":
        print("📱 Open WhatsApp (or WhatsApp Business) on the")
        print("   phone with the BOT's number, then scan:")
    else:
        print("📱 Open WhatsApp on your phone, then scan:")
    print()
    print("   Settings → Linked Devices → Link a Device")
    print("─" * 50)
    print()

    try:
        subprocess.run(
            [
                find_node_executable("node") or "node",
                str(bridge_script),
                "--pair-only",
                "--session",
                str(session_dir)],
            cwd=str(bridge_dir),
            env=with_hermes_node_path())
    except KeyboardInterrupt:
        pass

    # ── Step 7: Post-pairing ─────────────────────────────────────────────
    print()
    if (session_dir / "creds.json").exists():
        # Only enable WhatsApp now that pairing actually succeeded.  If the
        # user Ctrl+C'd at any earlier step, WHATSAPP_ENABLED stays unset
        # and `hermes gateway` skips it cleanly instead of paying a 30s
        # bridge timeout + queueing the platform for indefinite retries.
        save_env_value("WHATSAPP_ENABLED", "true")
        print("✓ WhatsApp paired successfully!")
        print()
        if wa_mode == "bot":
            print("  Next steps:")
            print("    1. Start the gateway:  hermes gateway")
            print("    2. Send a message to the bot's WhatsApp number")
            print("    3. The agent will reply automatically")
            print()
            print("  Tip: Agent responses are prefixed with '⚕ Hermes Agent'")
        else:
            print("  Next steps:")
            print("    1. Start the gateway:  hermes gateway")
            print("    2. Open WhatsApp → Message Yourself")
            print("    3. Type a message — the agent will reply")
            print()
            print("  Tip: Agent responses are prefixed with '⚕ Hermes Agent'")
            print("  so you can tell them apart from your own messages.")
        print()
        print("  Or install as a service: hermes gateway install")
    else:
        print("⚠ Pairing may not have completed. Run 'hermes whatsapp' to try again.")


def cmd_whatsapp_cloud(args):
    """Set up WhatsApp Business Cloud API (official Meta integration).

    Walks the user through the Meta-side credentials (Phone Number ID,
    Access Token, App Secret, optional App/WABA IDs) plus webhook
    configuration. Includes field-shape validators that catch the most
    common setup mistakes (e.g. pasting a phone number into the Phone
    Number ID field).

    Distinct from ``hermes whatsapp`` (the Baileys bridge wizard) — the
    two adapters are complementary, not alternatives. See
    ``hermes_cli/setup_whatsapp_cloud.py``.
    """
    from hermes_cli.main import _require_tty
    _require_tty("whatsapp-cloud")
    from hermes_cli.setup_whatsapp_cloud import run_whatsapp_cloud_setup

    return run_whatsapp_cloud_setup()


def cmd_sync(args):
    """Skill Sync — personal sync across devices, plus sharing with your org."""
    import json as _json

    sub = getattr(args, "sync_command", None)

    if sub in {None, ""}:
        print(
            "usage: hermes sync "
            "<status|pull|push|now|enable|disable|device|propose>\n"
            "\n"
            "Your skills, across your devices:\n"
            "  status            Show what is synced, and from where\n"
            "  pull              Pull your synced skills\n"
            "  push              Push your opted-in skills\n"
            "  now               Reconcile now: pull then push\n"
            "  enable <skill>    Include a skill in your sync\n"
            "  disable <skill>   Exclude a skill from your sync\n"
            "  device [--name N] Show or set this device's label\n"
            "\n"
            "Shared with your team:\n"
            "  propose <skill>   Share a skill with your organisation",
            file=sys.stderr)
        return 1

    if sub == "device":
        from tools import skills_sync_client as ssc

        name = getattr(args, "device_name", None)
        if name is not None:
            try:
                stored = ssc.set_device_name(name)
            except ValueError as e:
                print(f"error: {e}", file=sys.stderr)
                return 1
            print(f"device label set to '{stored}'.")
            print(
                "New commits from this device will use this label; existing "
                "commits keep their previous one.",
                file=sys.stderr)
            return 0
        # No --name: print the current (creating a default on first use).
        print(ssc.stable_device_id())
        return 0

    if sub == "propose":
        from tools import skills_sync_client as ssc

        name = args.name
        try:
            result = ssc.propose_skill(name, message=args.message)
        except ssc.SyncInertError as e:
            print(f"cannot share this skill: {e}", file=sys.stderr)
            return 1
        except ssc.SyncError as e:
            print(f"could not share '{name}': {e}", file=sys.stderr)
            return 1
        if result.get("proposal_pending"):
            print(
                f"Shared '{name}' with your organisation — an admin needs to "
                f"approve it (proposal #{result.get('proposal_id')}). It is "
                f"not live for the team until then.")
        else:
            print(f"Added '{name}' to your organisation's shared skills.")
        return 0

    if sub in {"enable", "disable"}:
        from tools.skill_usage import set_sync, is_curation_eligible

        skill = args.skill
        if not is_curation_eligible(skill):
            print(
                f"'{skill}' is not sync-eligible (bundled, hub-installed, "
                f"external, or not found). Only agent-created / user-authored "
                f"skills under ~/.hermes/skills/ can sync.",
                file=sys.stderr)
            return 1
        set_sync(skill, sub == "enable")
        print(f"sync {'enabled' if sub == 'enable' else 'disabled'} for '{skill}'.")
        return 0

    from tools import skills_sync_client as ssc

    if sub == "status":
        status = ssc.sync_status()
        print(_json.dumps(status, indent=2, ensure_ascii=False))
        if status.get("org_available"):
            n = len(status.get("org_skills") or [])
            modified = status.get("org_skills_modified") or []
            print(
                f"\nOrg skills: {n} shared skill(s) from your organisation "
                f"(your role: {status.get('org_role')}). They load alongside "
                f"your own, labeled by origin, and you can edit them.",
                file=sys.stderr)
            if modified:
                print(
                    f"  {len(modified)} with local edits not yet shared: "
                    f"{', '.join(modified)}\n"
                    f"  Share them back with `hermes sync propose <skill>`. "
                    f"Org updates will not overwrite them.",
                    file=sys.stderr)
        elif status.get("logged_in"):
            print(
                "\nOrg skills: not applicable — this account isn't a member "
                "of a shared organisation.",
                file=sys.stderr)
        if not status.get("logged_in"):
            print("\nNot logged into Nous Portal — sync is inert.", file=sys.stderr)
        elif not status.get("nous_admin"):
            print(
                "\nSync is not enabled for your account yet.", file=sys.stderr)
        elif not status.get("feature_enabled"):
            print(
                "\nSync feature is off for this instance (set HERMES_SYNC_ENABLED=1 "
                "or config.yaml sync.enabled: true). Sync is inert.",
                file=sys.stderr)
        elif not status.get("base_url"):
            print(
                "\nNo sync base URL configured (config.yaml sync.base_url or "
                "HERMES_SYNC_BASE_URL). Sync is inert.",
                file=sys.stderr)
        return 0

    # pull / push / now — enforce the gate up front with a clear message.
    try:
        identity = ssc.resolve_identity()
    except ssc.SyncInertError as e:
        print(f"sync inert: {e}", file=sys.stderr)
        return 1
    if not identity.get("nous_admin"):
        print(
            "sync unavailable: not enabled for your account yet.", file=sys.stderr)
        return 1
    if not ssc.resolve_sync_base_url():
        print(
            "sync inert: no sync base URL configured (config.yaml sync.base_url "
            "or HERMES_SYNC_BASE_URL).",
            file=sys.stderr)
        return 1

    try:
        if sub == "pull":
            result = ssc.pull_skills(identity=identity)
            # Refresh the org mirror too when this account belongs to an
            # organisation (no-op otherwise), so one pull covers both.
            org_result = ssc.maybe_pull_org_skills()
            if org_result:
                n = len(org_result.get("updated") or [])
                print(
                    f"org: refreshed {n} shared skill(s) from your "
                    f"organisation.",
                    file=sys.stderr)
                clashes = org_result.get("conflicted") or []
                if clashes:
                    print(
                        f"org: {len(clashes)} skill(s) have BOTH local edits "
                        f"and org updates, so they were left as-is: "
                        f"{', '.join(clashes)}\n"
                        f"     Your local version is intact. Review it, then "
                        f"either propose it or delete the local copy and pull "
                        f"again to take the org version.",
                        file=sys.stderr)
        elif sub == "push":
            result = ssc.push_skills(identity=identity, message="hermes sync push")
        elif sub == "now":
            pull_res = ssc.pull_skills(identity=identity)
            push_res = ssc.push_skills(identity=identity, message="hermes sync now")
            result = {"pull": pull_res, "push": push_res}
        else:
            print(f"Unknown sync subcommand: {sub}", file=sys.stderr)
            return 1
    except ssc.SyncError as e:
        print(f"sync failed: {e}", file=sys.stderr)
        return 1

    print(_json.dumps(result, indent=2, ensure_ascii=False))
    return 0


def cmd_slack(args):
    """Slack integration helpers.

    Dispatches ``hermes slack <subcommand>``. Currently supports:
      manifest — print or write a Slack app manifest with every gateway
                 command registered as a first-class slash.
    """
    sub = getattr(args, "slack_command", None)
    if sub in {None, ""}:
        # No subcommand — print usage hint.
        print(
            "usage: hermes slack <subcommand>\n"
            "\n"
            "subcommands:\n"
            "  manifest   Generate a Slack app manifest with every gateway\n"
            "             command registered as a native slash\n"
            "\n"
            "Run `hermes slack manifest -h` for details.",
            file=sys.stderr)
        return 1

    if sub == "manifest":
        from hermes_cli.slack_cli import slack_manifest_command

        status = slack_manifest_command(args)
        if status:
            raise SystemExit(status)
        return status

    print(f"Unknown slack subcommand: {sub}", file=sys.stderr)
    return 1
