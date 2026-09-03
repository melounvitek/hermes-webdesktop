"""``hermes photon ...`` CLI subcommands (registered via ``ctx.register_cli_command()``):

    setup              full first-time setup (device login + project + user + sidecar)
    status             show login + project + sidecar dep state
    install-sidecar    npm install inside plugins/platforms/photon/sidecar/
    telemetry          show or toggle Spectrum SDK telemetry (on/off)

Device login is the first step of ``setup`` (no standalone ``login`` verb); inbound is
the spectrum-ts gRPC stream, so there are no webhook subcommands.
"""
from __future__ import annotations

import argparse
import getpass
import os
import shutil
import subprocess
import sys
from typing import Optional

from hermes_cli.colors import Colors, color

from . import auth as photon_auth
from .adapter import sidecar_deps_installed
from .sidecar_paths import _NPM_ERROR_LOG_MAX_CHARS, _npm_error_log, _sidecar_dir
import contextlib


# ---------------------------------------------------------------------------
# argparse wiring

def register_cli(parser: argparse.ArgumentParser) -> None:
    """Wire up `hermes photon ...` subcommands."""
    subs = parser.add_subparsers(dest="photon_command", required=False)
    p_setup = subs.add_parser("setup", help="First-time setup (device login + project + user + sidecar)")
    p_setup.add_argument("--project-name", default=None, help="Project name (default: 'Hermes Agent')")
    p_setup.add_argument("--phone", default=None, help="Your E.164 phone number (e.g. +15551234567)")
    p_setup.add_argument("--first-name", default=None)
    p_setup.add_argument("--last-name", default=None)
    p_setup.add_argument("--email", default=None)
    p_setup.add_argument("--no-browser", action="store_true",
                         help="Don't try to open a browser for device login; print the URL only")
    p_setup.add_argument("--skip-sidecar-install", action="store_true",
                         help="Skip `npm install` inside the sidecar directory")
    subs.add_parser("status", help="Show login + project + sidecar dep state")
    subs.add_parser("install-sidecar", help="Run npm install inside the sidecar directory")
    p_telemetry = subs.add_parser("telemetry", help="Show or toggle Spectrum SDK telemetry (on/off)")
    p_telemetry.add_argument(
        "state", nargs="?", choices=("on", "off"),
        help="Turn telemetry on or off (omit to show the current state)")
    parser.set_defaults(func=dispatch)


# ---------------------------------------------------------------------------
# Dispatch

def dispatch(args: argparse.Namespace) -> int:
    sub = getattr(args, "photon_command", None)
    if sub is None:
        return _cmd_status(args)  # no subcommand — show status by default
    handler = _COMMANDS.get(sub)
    if handler is None:
        print(f"unknown subcommand: {sub}", file=sys.stderr)
        return 2
    return handler(args)


# ---------------------------------------------------------------------------
# Subcommand handlers

def _run_device_login(args: argparse.Namespace) -> int:
    """Run the RFC 8628 device-code login flow and persist the token (first step of ``setup``)."""
    def _print_code(code):
        target = code.verification_uri_complete or code.verification_uri
        print()
        print("┌─ Photon device login ────────────────────────────────────────")
        print(f"│  Open this URL:  {target}")
        print(f"│  Enter the code: {code.user_code}")
        print("│  (waiting for approval — Ctrl-C to cancel)")
        print("└──────────────────────────────────────────────────────────────")
        print()
    try:
        photon_auth.login_device_flow(open_browser=not args.no_browser, on_user_code=_print_code)
    except Exception as e:
        print(f"login failed: {e}", file=sys.stderr)
        return 1
    # Never print any portion of the token (shoulder-surfing / screen recordings).
    print(f"✓ logged in — token saved to {photon_auth._auth_json_path()}")
    return 0


def _setup_token(args: argparse.Namespace) -> Optional[str]:
    """[1/5] Reuse a valid dashboard token or run device login; None on failure."""
    token = photon_auth.load_photon_token()
    if token:
        # The dashboard token has a short TTL (~3-4 days); a stale one makes every
        # management call 401, so validate upfront and fall back to a fresh login.
        print("[1/5] Checking existing Photon token...")
        if photon_auth.check_photon_token_valid(token):
            print("  ✓ token is valid")
        else:
            print("  ✗ token is stale (dashboard rejected it) — re-authenticating")
            photon_auth.clear_photon_token()
            token = None
    if not token:
        print("[1/5] No valid Photon token found — running device login...")
        if _run_device_login(args) != 0:
            return None
        token = photon_auth.load_photon_token()
        if not token:
            print("login completed but token was not stored", file=sys.stderr)
            return None
    else:
        print("[1/5] Reusing existing Photon token")
    return token


def _setup_project(token: str, name: str) -> Optional[str]:
    """[2/5] Find or create the project; returns its id or None on failure."""
    dashboard_id = photon_auth.load_dashboard_project_id()
    try:
        if dashboard_id:
            print("[2/5] Reusing configured Photon project")
        else:
            existing = photon_auth.find_project_by_name(token, name)
            if existing and existing.get("id"):
                dashboard_id = existing["id"]
                print(f"[2/5] Found existing project '{name}'")
            else:
                print(f"[2/5] Creating Photon project '{name}'...")
                created = photon_auth.create_project(token, name=name)
                dashboard_id = created.get("id")
                print("  ✓ project created")
    except Exception as e:
        print(f"project setup failed: {e}", file=sys.stderr)
        return None
    if not dashboard_id:
        print("could not resolve a Photon project id", file=sys.stderr)
        return None
    return dashboard_id


def _setup_credentials(token: str, dashboard_id: str, name: str) -> Optional[str]:
    """[3/5] Provision Spectrum credentials (runtime -> .env, ids -> auth.json); the
    dashboard id *is* the Spectrum id. A valid existing secret is reused: regenerating
    breaks a running sidecar's sends until restart. Returns the secret or None."""
    try:
        print("[3/5] Provisioning Spectrum credentials...")
        existing_id, existing_secret = photon_auth.load_project_credentials()
        secret: str = ""
        reused = False
        if existing_id and existing_secret:
            try:
                photon_auth.list_users(existing_id, existing_secret)  # lightweight validation
                secret = existing_secret
                reused = True
            except Exception:
                secret = ""  # fall through to regeneration
        if not secret:
            secret = photon_auth.regenerate_project_secret(token, dashboard_id)
        photon_auth.store_project_credentials(
            spectrum_project_id=dashboard_id, project_secret=secret, dashboard_project_id=dashboard_id, name=name,
        )
        if reused:
            print(f"  ✓ Spectrum ready (project id {dashboard_id}) — existing credentials valid")
        else:
            print(f"  ✓ Spectrum ready (project id {dashboard_id}) — new secret saved")
            print(
                "  ⚠ Project secret was regenerated. If the gateway is running, "
                "restart it so the sidecar picks up the new secret:\n"
                "      hermes gateway restart")
    except Exception as e:
        print(f"spectrum provisioning failed: {e}", file=sys.stderr)
        return None
    return secret


def _cmd_setup(args: argparse.Namespace) -> int:
    token = _setup_token(args)
    if not token:
        return 1
    name = args.project_name or photon_auth.DEFAULT_PROJECT_NAME
    dashboard_id = _setup_project(token, name)
    if not dashboard_id:
        return 1
    secret = _setup_credentials(token, dashboard_id, name)
    if not secret:
        return 1
    # 4. Register the operator's phone number as a Spectrum user (idempotent).
    phone = args.phone or _prompt(
        color("[4/5] Your iMessage phone number (E.164, e.g. +15551234567): ", Colors.CYAN))
    agent_number = registered_phone = registered_user_id = None
    if not phone:
        print("      Skipped user registration (no phone given). Re-run with --phone later.")
    else:
        # Name/email are optional and never prompted for (--first-name / --email).
        try:
            user, created = photon_auth.register_user_if_absent(
                dashboard_id, secret, phone_number=phone, first_name=args.first_name,
                last_name=args.last_name, email=args.email)
        except ValueError as e:
            print(f"      invalid phone number: {e}", file=sys.stderr)
            return 1
        except Exception as e:
            print(f"      user registration failed: {e}", file=sys.stderr)
            return 1
        print("  ✓ phone registered" if created else "  ✓ phone already registered")
        registered_phone = phone
        registered_user_id = user.get("id")
        # The number to text the agent is the user's assigned line ("TEXTS ON");
        # shared-number plans have no dedicated /lines entry.
        agent_number = photon_auth.user_assigned_line(user)
        # Otherwise the gateway denies the operator's own inbound ("Unauthorized user")
        # and has no default space for cron delivery.
        _autoconfigure_access(phone)
    # 5. Surface the agent's iMessage number.
    if not agent_number:
        try:
            line = photon_auth.get_imessage_line(token, dashboard_id)
            if line:
                agent_number = line.get("phoneNumber")
        except Exception as e:
            print(f"      (could not fetch the assigned line: {e})", file=sys.stderr)
    if agent_number:
        print()
        print(color("┌─ Your agent's iMessage number ───────────────────────────────", Colors.GREEN))
        print(color("│  📱 ", Colors.GREEN) + color(str(agent_number), Colors.GREEN, Colors.BOLD))
        print(color("│  Text this number from your phone to talk to your agent.", Colors.GREEN))
        print(color("└──────────────────────────────────────────────────────────────", Colors.GREEN))
    else:
        print("      No iMessage line assigned yet — check the Photon dashboard.")
    if registered_phone:
        try:
            photon_auth.store_user_numbers(
                phone_number=registered_phone, assigned_phone_number=agent_number,
                user_id=str(registered_user_id) if registered_user_id else None, dashboard_project_id=dashboard_id,
            )
        except Exception as e:
            print(f"      (could not save Photon status metadata: {e})", file=sys.stderr)
    # 6. Sidecar deps (spectrum-ts).
    if args.skip_sidecar_install:
        print("[5/5] Skipping sidecar npm install (--skip-sidecar-install)")
    else:
        print("[5/5] Installing Node sidecar deps (spectrum-ts)...")
        rc = _install_sidecar()
        if rc != 0:
            return rc
    # 7. Enable the platform in config.yaml, or the channel silently stays offline.
    try:
        from hermes_cli.config import write_platform_config_field
        write_platform_config_field("photon", "enabled", True, raw=True)
        print("  ✓ photon platform enabled in config.yaml")
    except Exception as e:
        print(f"      (could not enable Photon in config: {e})", file=sys.stderr)
    print()
    print("✓ Photon setup complete.")
    print("  Start the gateway:  hermes gateway start")
    return 0


def _autoconfigure_access(phone: str) -> None:
    """Set PHOTON_ALLOWED_USERS and PHOTON_HOME_CHANNEL to the operator's number,
    each only when unset so a hand-tuned value is never clobbered on re-run."""
    try:
        from hermes_cli.config import get_env_value, save_env_value
    except ImportError:
        return
    for key, label in (
        ("PHOTON_ALLOWED_USERS", "allowlisted your number"),
        ("PHOTON_HOME_CHANNEL", "set your DM as the cron home channel")):
        try:
            if get_env_value(key):
                print(f"      {key} already set — leaving it as-is.")
                continue
            save_env_value(key, phone)
            print(f"  ✓ {label} ({key})")
        except Exception as e:
            print(f"      could not set {key}: {e}", file=sys.stderr)


def _cmd_status(_args: argparse.Namespace) -> int:
    _refresh_status_numbers()
    # auth.print_credential_summary's emit callback is the only sink that sees
    # credential-derived strings (keeps cli.py taint-free for CodeQL).
    photon_auth.print_credential_summary(print)
    node_bin = os.getenv("PHOTON_NODE_BIN") or shutil.which("node")
    sidecar_installed = sidecar_deps_installed()
    print(f"  node binary         : {node_bin or '✗ missing (install Node 18+)'}")
    print(f"  sidecar deps        : {'✓ installed' if sidecar_installed else '✗ run `hermes photon install-sidecar`'}")
    print(f"  telemetry           : {'on' if _telemetry_enabled() else 'off'} (`hermes photon telemetry on|off`)")
    return 0


def _refresh_status_numbers() -> None:
    phone, assigned = photon_auth.load_user_numbers()
    if phone and assigned:
        return
    spectrum_id, project_secret = photon_auth.load_project_credentials()
    if not spectrum_id or not project_secret:
        return
    try:
        photon_auth.refresh_user_numbers(spectrum_id, project_secret)
    except Exception as e:
        print(f"      (could not refresh Photon user numbers: {e})", file=sys.stderr)


def _telemetry_enabled() -> bool:
    """PHOTON_TELEMETRY from env / ~/.hermes/.env; truthy set mirrors the sidecar's."""
    try:
        from hermes_cli.config import get_env_value
        raw = get_env_value("PHOTON_TELEMETRY")
    except ImportError:
        raw = os.getenv("PHOTON_TELEMETRY")
    return (raw or "").strip().lower() in ("1", "true", "yes", "on")


def _cmd_telemetry(args: argparse.Namespace) -> int:
    state = getattr(args, "state", None)
    if state is None:
        print(f"Photon telemetry: {'on' if _telemetry_enabled() else 'off'}")
        print("  Toggle with `hermes photon telemetry on` / `hermes photon telemetry off`.")
        return 0
    try:
        from hermes_cli.config import save_env_value
        save_env_value("PHOTON_TELEMETRY", "true" if state == "on" else "false")
    except Exception as e:
        print(f"could not save PHOTON_TELEMETRY: {e}", file=sys.stderr)
        return 1
    print(f"✓ Spectrum telemetry turned {state} (PHOTON_TELEMETRY in ~/.hermes/.env)")
    print("  Restart the gateway for the sidecar to pick it up:  hermes gateway restart")
    return 0


def _install_sidecar() -> int:
    npm = shutil.which("npm") or "npm"
    if not shutil.which(npm):
        print("npm is not on PATH. Install Node.js 18+ (https://nodejs.org/) and re-run.", file=sys.stderr)
        return 1
    # spectrum-ts is pinned exactly (the SDK ships breaking majors); upgrades are
    # deliberate — never `@latest` (see README "Upgrading spectrum-ts"). `npm ci`
    # installs the lockfile verbatim; `npm install` is the fallback for a
    # missing/drifted lockfile.
    print(f"  $ cd {_sidecar_dir()} && {npm} ci")

    def _run(verb: str) -> subprocess.CompletedProcess:
        # stdout streams to the terminal; stderr is captured so the failure reason
        # can be persisted for check_requirements() to surface later.
        proc = subprocess.run(  # noqa: S603
            [npm, verb], cwd=str(_sidecar_dir()), check=False, stderr=subprocess.PIPE, text=True)
        if proc.stderr:
            print(proc.stderr, end="", file=sys.stderr)
        return proc
    proc = _run("ci")
    if proc.returncode != 0:
        print(f"  npm ci failed — falling back to:  {npm} install")
        proc = _run("install")
    if proc.returncode != 0:
        print("npm install failed", file=sys.stderr)
        # Bounded to what check_requirements() will ever surface.
        error = (proc.stderr or "").strip()[:_NPM_ERROR_LOG_MAX_CHARS]
        if error:
            with contextlib.suppress(OSError):
                _npm_error_log().write_text(error, encoding="utf-8")
    else:
        with contextlib.suppress(OSError):
            _npm_error_log().unlink()
    return proc.returncode


_COMMANDS = {
    "setup": _cmd_setup,
    "status": _cmd_status,
    "install-sidecar": lambda _args: _install_sidecar(),
    "telemetry": _cmd_telemetry}


# ---------------------------------------------------------------------------
# Gateway-setup entry point

def gateway_setup() -> None:
    """Run Photon first-time setup from the unified `hermes gateway setup` wizard
    (same flow as ``hermes photon setup``; phone is prompted when stdin is a TTY)."""
    _cmd_setup(argparse.Namespace(
        photon_command="setup", project_name=None, phone=None, first_name=None, last_name=None,
        email=None, no_browser=False, skip_sidecar_install=False))


# ---------------------------------------------------------------------------
# Small interactive helpers

def _prompt(prompt: str, *, secret: bool = False) -> str:
    if not sys.stdin.isatty():
        return ""
    try:
        if secret:
            return getpass.getpass(prompt).strip()
        return input(prompt).strip()
    except (KeyboardInterrupt, EOFError):
        print()
        return ""
