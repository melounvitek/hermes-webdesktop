"""CLI handlers for ``hermes secrets bitwarden ...``."""

from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# NOTE: the Bitwarden backend (``agent.secret_sources.bitwarden``) pulls in
# ``cryptography`` at module-import time.  On Windows the resulting
# ``cryptography._rust.pyd`` is mapped into the running process — and when
# that process is ``hermes update``, the self-lock preflight detects the
# loaded native module and defers (#86781).  Keep the backend import lazy:
# this module is registered parse-time from ``hermes_cli.main`` and must not
# touch ``bw`` until a handler actually runs.
#
# ``_BWS_VERSION`` is duplicated here (as a plain string) so ``register_cli``
# can render the ``install --help`` text without importing the backend.
# ``agent.secret_sources.bitwarden._BWS_VERSION`` is the source of truth;
# bump both together when pinning a new bws release.
_BWS_VERSION = "2.0.0"

from hermes_cli._secrets_common import (
    arg,
    cfg_str,
    cli_version,
    disable_secret_source,
    flag,
    print_status_panel,
    print_table,
    prompt_index,
    register_subcommands,
    require_enabled,
    rotate_token,
    secret_cli_env,
    section_cfg,
    yn,
)
from hermes_cli.config import (
    get_env_path,
    load_config,
    save_config,
    save_env_value,
)
from hermes_cli.secret_prompt import masked_secret_prompt

# Old names kept bound: tests monkeypatch ``secrets_cli._bws_version``.
_bws_version = cli_version
_yn = yn


_NOT_BSM_TOKEN_WARNING = (
    "[yellow]Warning: token doesn't start with '0.' — usually that means "
    "you pasted something other than a BSM access token.[/yellow]"
)
_NOT_BSM_TOKEN_WARNING_CONTINUING = (
    "  [yellow]Warning: token doesn't start with '0.' — usually that means "
    "you pasted something other than a BSM access token.  Continuing anyway.[/yellow]"
)


def _load_bw():
    """Import ``agent.secret_sources.bitwarden`` on first use (crypto payload)."""
    from agent.secret_sources import bitwarden as _bw

    return _bw


def __getattr__(name: str):
    """PEP 562 module-level lazy resolver.

    Existing callers (and upstream tests) monkeypatch attributes on ``hermes_cli.secrets_cli.bw``
    directly. Resolving that attribute at module-import time would re-import ``cryptography``
    eagerly — the very self-lock we are preventing (#86781).
    """
    if name == "bw":
        return _load_bw()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


# ---------------------------------------------------------------------------
# Argparse wiring — called from hermes_cli.main
# ---------------------------------------------------------------------------


def register_cli(parent_parser: argparse.ArgumentParser) -> None:
    """Attach the ``bitwarden`` subcommand tree to a parent parser."""
    register_subcommands(parent_parser, "secrets_bw_command", (
        ("setup", "Interactive wizard: install bws, store access token, pick project", cmd_setup, (
            arg("--project-id", "Pre-select a project UUID instead of prompting"),
            arg("--access-token", "Provide the access token non-interactively (will be stored in .env)"),
            arg("--server-url", (
                "Bitwarden region / self-hosted endpoint. Examples: "
                "https://vault.bitwarden.com (US, default), "
                "https://vault.bitwarden.eu (EU), or your self-hosted URL. "
                "Skips the interactive region prompt."
            )),
        )),
        ("status", "Show config + binary + token validation status", cmd_status, ()),
        ("token", "Rotate the access token: validate a new one and store it in .env", cmd_token, (
            arg("--access-token", "Provide the new token non-interactively (default: masked prompt)"),
            flag("--no-verify", "Store without probing Bitwarden first (not recommended)"),
        )),
        ("sync", "Fetch secrets now and report what changed", cmd_sync, (
            flag("--apply", "Actually export the secrets into the current shell's env (default: dry-run)"),
        )),
        ("disable", "Turn off the Bitwarden integration", cmd_disable, ()),
        ("install", f"Download and verify the pinned bws binary (v{_BWS_VERSION})", cmd_install, (
            flag("--force", "Re-download even if a managed copy already exists"),
        )),
    ))


# ---------------------------------------------------------------------------
# Handlers
# ---------------------------------------------------------------------------


def cmd_setup(args: argparse.Namespace) -> int:
    bw = _load_bw()
    console = Console()
    console.print(
        Panel.fit(
            "[bold]Bitwarden Secrets Manager setup[/bold]\n\n"
            "Need an access token? In the Bitwarden web app:\n"
            "  Secrets Manager → Machine accounts → [your account] →\n"
            "  Access tokens → Create access token\n\n"
            "Copy the token (starts with [cyan]0.[/cyan]…) — it cannot be retrieved later.",
            border_style="cyan",
        )
    )

    # ------------------------------------------------------------------ binary
    console.print()
    console.print("[bold]Step 1[/bold]  Install the bws CLI")
    try:
        binary = bw.find_bws(install_if_missing=False)
        if binary is None:
            console.print("  No bws on PATH — downloading…")
            binary = bw.install_bws()
        version = _bws_version(binary)
        console.print(f"  [green]✓[/green] {binary}  ({version})")
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [red]✗ Could not install bws: {exc}[/red]")
        console.print(
            "  Manual install: "
            "https://github.com/bitwarden/sdk-sm/releases"
        )
        return 1

    # -- non-interactive guard --
    if not sys.stdin.isatty():
        # BWS_SERVER_URL env var is accepted as a non-interactive substitute for --server-url.
        provided = {
            "--access-token": args.access_token,
            "--server-url": (args.server_url or "").strip() or os.environ.get("BWS_SERVER_URL", ""),
            "--project-id": args.project_id,
        }
        missing = [flag for flag, value in provided.items() if not (value and value.strip())]
        if missing:
            console.print(
                f"  [red]Non-interactive mode (no TTY) requires all setup flags.[/red]\n"
                f"  Missing: {', '.join(missing)}\n\n"
                "  Usage:\n"
                "    hermes secrets bitwarden setup \\\n"
                "      --access-token '0.xxx' \\\n"
                "      --server-url 'https://vault.bitwarden.com' \\\n"
                "      --project-id 'xxxxxxxx-xxxx-xxxx-xxxx-xxxxxxxxxxxx'"
            )
            return 1

    # ------------------------------------------------------------------- token
    console.print()
    console.print("[bold]Step 2[/bold]  Provide your access token")
    cfg = load_config()
    secrets_cfg = cfg.setdefault("secrets", {}).setdefault("bitwarden", {})
    token_env = secrets_cfg.get("access_token_env", "BWS_ACCESS_TOKEN")

    token = (args.access_token or "").strip()
    if not token:
        token = masked_secret_prompt(f"  Paste access token ({token_env}): ").strip()
    if not token:
        console.print("  [red]Empty token, aborting.[/red]")
        return 1
    if not token.startswith("0."):
        console.print(_NOT_BSM_TOKEN_WARNING_CONTINUING)

    save_env_value(token_env, token)
    os.environ[token_env] = token  # so the test fetch below sees it
    console.print(f"  [green]✓[/green] stored in {get_env_path()} as {token_env}")

    # ------------------------------------------------------------------ region
    console.print()
    console.print("[bold]Step 3[/bold]  Pick a Bitwarden region")
    server_url = _resolve_server_url(args, secrets_cfg, console)
    if server_url is None:
        return 1
    if server_url:
        console.print(f"  [green]✓[/green] using {server_url}")
    else:
        console.print(
            "  [green]✓[/green] using bws default "
            "(US Cloud, https://vault.bitwarden.com)"
        )

    # ------------------------------------------------------------------- project
    project_given = bool(args.project_id and args.project_id.strip())
    if project_given:
        project_id = args.project_id.strip()
    else:
        console.print()
        console.print("[bold]Step 4[/bold]  Pick a project")
        projects = _list_projects(binary, token, console, server_url=server_url)
        if projects is None:
            return 1
        if not projects:
            console.print("  [yellow]No projects visible to this machine account.[/yellow]")
            console.print(
                "  In the Bitwarden web app, open the machine account → Projects tab "
                "and grant it access to at least one project."
            )
            return 1

        print_table(console, (("#", {"style": "cyan", "width": 4}), "Name", ("ID", {"style": "dim"})),
                    ((str(i), p.get("name", "?"), p.get("id", "?")) for i, p in enumerate(projects, 1)))

        idx = prompt_index(console, f"  Select project [1-{len(projects)}]: ", len(projects))
        project_id = projects[idx - 1]["id"]

    # ------------------------------------------------------------------- test
    console.print()
    console.print(f"[bold]Step {4 if project_given else 5}[/bold]  Test fetch")
    try:
        secrets, warnings = bw.fetch_bitwarden_secrets(
            access_token=token,
            project_id=project_id,
            binary=binary,
            use_cache=False,
            server_url=server_url,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"  [red]✗ Fetch failed: {exc}[/red]")
        return 1

    if not secrets:
        console.print("  [yellow]Fetch succeeded but the project has no secrets.[/yellow]")
    else:
        print_table(console, (("Name", {"style": "cyan"}), "Status"),
                    ((key, _fetch_status(key, token_env)) for key in sorted(secrets)))
    for w in warnings:
        console.print(f"  [yellow]warning:[/yellow] {w}")

    # ------------------------------------------------------------------- save
    secrets_cfg["enabled"] = True
    secrets_cfg["project_id"] = project_id
    secrets_cfg["server_url"] = server_url
    secrets_cfg.setdefault("access_token_env", token_env)
    secrets_cfg.setdefault("cache_ttl_seconds", 300)
    secrets_cfg.setdefault("override_existing", True)
    secrets_cfg.setdefault("auto_install", True)
    save_config(cfg)

    console.print()
    console.print(
        "[green]✓ Bitwarden Secrets Manager is enabled.[/green]  "
        "Secrets will be pulled at the start of every Hermes process."
    )
    console.print(
        "  Status:  [cyan]hermes secrets bitwarden status[/cyan]\n"
        "  Refresh: [cyan]hermes secrets bitwarden sync[/cyan]\n"
        "  Disable: [cyan]hermes secrets bitwarden disable[/cyan]"
    )
    return 0


def _bw_cfg(cfg: dict) -> dict:
    return section_cfg(cfg, "bitwarden")


def _fetch_status(key: str, token_env: str) -> str:
    if key == token_env:
        return "[dim]bootstrap token — never overrides itself[/dim]"
    if os.environ.get(key):
        return "[yellow]already set in env (will be overwritten)[/yellow]"
    return "[green]new[/green]"


def cmd_status(args: argparse.Namespace) -> int:
    bw = _load_bw()
    console = Console()
    bw_cfg = _bw_cfg(load_config())

    enabled = bool(bw_cfg.get("enabled"))
    token_env = bw_cfg.get("access_token_env", "BWS_ACCESS_TOKEN")
    project_id = bw_cfg.get("project_id", "")
    server_url = cfg_str(bw_cfg, "server_url")
    token = os.environ.get(token_env, "").strip()
    token_set = bool(token)
    binary = bw.find_bws(install_if_missing=False)
    token_validation, validation_messages = _token_validation_status(
        enabled=enabled,
        binary=binary,
        token=token,
        server_url=server_url,
    )

    print_status_panel(console, "Bitwarden Secrets Manager", (
        ("Enabled", _yn(enabled)),
        ("Token env var", token_env),
        ("Token in env", _yn(token_set)),
        ("Token validation", token_validation),
        ("Project ID", project_id or "[dim](unset)[/dim]"),
        ("Server URL", server_url or "[dim]default (US Cloud, https://vault.bitwarden.com)[/dim]"),
        ("Override existing", _yn(bool(bw_cfg.get("override_existing", False)))),
        ("Cache TTL (s)", str(bw_cfg.get("cache_ttl_seconds", 300))),
        ("Auto-install", _yn(bool(bw_cfg.get("auto_install", True)))),
        ("bws binary",
         f"{binary} ({_bws_version(binary)})" if binary else "[yellow]not installed[/yellow]"),
    ))
    for message in validation_messages:
        console.print(message)

    if not enabled:
        console.print("\n  Run [cyan]hermes secrets bitwarden setup[/cyan] to enable.")
        return 0
    if not token_set:
        console.print(
            f"\n  [yellow]Enabled but {token_env} is not set — Hermes will skip BSM "
            "and warn on next startup.[/yellow]"
        )
    if not project_id:
        console.print(
            "\n  [yellow]Enabled but no project_id — nothing to fetch.[/yellow]"
        )
    return 0


def cmd_token(args: argparse.Namespace) -> int:
    """Rotate the BSM access token without re-running the whole setup wizard.

    Prompts for (or accepts via ``--access-token``) a new machine-account token, probes Bitwarden
    with it (unless ``--no-verify``), and only then persists it to .env — so a bad paste never
    bricks the working token.
    """
    bw = _load_bw()
    console = Console()
    bw_cfg = _bw_cfg(load_config())
    token_env = bw_cfg.get("access_token_env", "BWS_ACCESS_TOKEN")
    server_url = cfg_str(bw_cfg, "server_url")

    def verify(token: str) -> bool:
        if not token.startswith("0."):
            console.print(_NOT_BSM_TOKEN_WARNING)
        if args.no_verify:
            return True
        binary = bw.find_bws(install_if_missing=True)
        if binary is None:
            console.print(
                "[red]bws binary not available — cannot verify.  "
                "Re-run with --no-verify to store anyway.[/red]"
            )
            return False
        console.print("Verifying against Bitwarden…")
        projects = _list_projects(binary, token, console, server_url=server_url)
        if projects is None:
            console.print(
                "[red]✗ New token was rejected — nothing was changed.[/red]"
            )
            return False
        console.print(
            f"[green]✓ Token accepted[/green] "
            f"({len(projects)} project{'s' if len(projects) != 1 else ''} visible)."
        )
        project_id = str(bw_cfg.get("project_id", "") or "")
        if project_id and projects and project_id not in {p["id"] for p in projects}:
            console.print(
                f"[yellow]Warning: configured project {project_id} is not visible "
                "to this machine account.  Grant it access in the Bitwarden web "
                "app or re-run `hermes secrets bitwarden setup` to pick a "
                "different project.[/yellow]"
            )
        return True

    return rotate_token(
        console, args.access_token, token_env,
        flag="--access-token",
        intro=(
            "Create a new token in the Bitwarden web app:\n"
            "  Secrets Manager → Machine accounts → [your account] → "
            "Access tokens → Create access token\n"
        ),
        prompt=f"Paste new access token ({token_env}): ",
        verify=verify,
        save=save_env_value, env_path=get_env_path, clear_caches=bw.clear_caches,
        disabled_note=None if bw_cfg.get("enabled") else (
            "[yellow]Note: the Bitwarden integration is currently disabled — "
            "run `hermes secrets bitwarden setup` (or set "
            "secrets.bitwarden.enabled: true) to turn it on.[/yellow]"
        ),
    )


def cmd_sync(args: argparse.Namespace) -> int:
    bw = _load_bw()
    console = Console()
    bw_cfg = _bw_cfg(load_config())
    if not require_enabled(console, bw_cfg, "Bitwarden", "bitwarden"):
        return 1

    token_env = bw_cfg.get("access_token_env", "BWS_ACCESS_TOKEN")
    token = os.environ.get(token_env, "").strip()
    if not token:
        console.print(f"[red]{token_env} is not set.[/red]")
        return 1

    project_id = bw_cfg.get("project_id", "")
    if not project_id:
        console.print("[red]No project_id configured.[/red]")
        return 1

    server_url = cfg_str(bw_cfg, "server_url")

    try:
        secrets, warnings = bw.fetch_bitwarden_secrets(
            access_token=token,
            project_id=project_id,
            use_cache=False,
            server_url=server_url,
        )
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Fetch failed: {exc}[/red]")
        return 1

    if not secrets:
        console.print("[yellow]No secrets in project.[/yellow]")
        return 0

    override = bool(bw_cfg.get("override_existing", False)) or args.apply
    rows = []
    applied = 0
    for key in sorted(secrets):
        if key == token_env:
            rows.append((key, "[dim]skip (bootstrap token)[/dim]"))
            continue
        already = bool(os.environ.get(key))
        if already and not override:
            rows.append((key, "[dim]skip (already set)[/dim]"))
            continue
        if args.apply:
            os.environ[key] = secrets[key]
            applied += 1
            rows.append((key, "[green]exported[/green]" + (" (overrode)" if already else "")))
        else:
            rows.append((key, "[green]would export[/green]" + (" (overrides)" if already else "")))

    print_table(console, (("Name", {"style": "cyan"}), "Action"), rows, warnings)

    if not args.apply:
        console.print(
            "\n  This was a dry-run — secrets are picked up automatically on the "
            "next [cyan]hermes[/cyan] invocation.  Re-run with [cyan]--apply[/cyan] "
            "to export into the current shell instead."
        )
    else:
        console.print(f"\n  [green]Exported {applied} secret(s) into current process.[/green]")
    return 0


def cmd_disable(args: argparse.Namespace) -> int:
    return disable_secret_source(
        "bitwarden",
        "[green]Disabled.[/green]  Bitwarden secrets will NOT be pulled on the next "
        "Hermes invocation.\n"
        "  Your access token is left in .env — remove it manually if you also want "
        "to revoke the credential.",
    )


def cmd_install(args: argparse.Namespace) -> int:
    bw = _load_bw()
    console = Console()
    try:
        path = bw.install_bws(force=bool(args.force))
        console.print(f"[green]✓[/green] {path}  ({_bws_version(path)})")
        return 0
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]Install failed: {exc}[/red]")
        return 1


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _token_validation_status(
    *,
    enabled: bool,
    binary: Optional[Path],
    token: str,
    server_url: str = "",
) -> tuple[str, list[str]]:
    if not enabled:
        return "[dim]not checked[/dim] (integration disabled)", []
    if not token:
        return "[dim]not checked[/dim] (token missing)", []
    if binary is None:
        return "[dim]not checked[/dim] (bws not installed)", []

    messages: list[str] = []
    if not token.startswith("0."):
        messages.append(_NOT_BSM_TOKEN_WARNING_CONTINUING)

    capture = io.StringIO()
    probe_console = Console(file=capture, record=True, width=200)
    projects = _list_projects(binary, token, probe_console, server_url=server_url)
    if projects is None:
        details = probe_console.export_text(styles=False).strip()
        if details:
            messages.extend(line.rstrip() for line in details.splitlines())
        return "[red]failed[/red]", messages
    return "[green]passed[/green]", messages


def _list_projects(
    binary: Path, token: str, console: Console, *, server_url: str = ""
) -> Optional[List[dict]]:
    """Call ``bws project list`` and return the parsed list, or None on failure."""
    env = secret_cli_env()
    env["BWS_ACCESS_TOKEN"] = token
    if server_url:
        env["BWS_SERVER_URL"] = server_url
    try:
        res = subprocess.run(
            [str(binary), "project", "list", "--output", "json"],
            env=env,
            capture_output=True,
            text=True, encoding='utf-8', errors='replace',
            timeout=15,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        console.print(f"  [red]Couldn't list projects: {exc}[/red]")
        return None

    if res.returncode != 0:
        err = (res.stderr or res.stdout).strip()[:300]
        console.print(f"  [red]bws project list failed: {err}[/red]")
        lowered = err.lower()
        if "invalid_client" in lowered or "400 bad request" in lowered:
            console.print(
                "  [yellow]'invalid_client' from the US identity endpoint usually "
                "means the token is for a different Bitwarden region.  Re-run "
                "[cyan]hermes secrets bitwarden setup[/cyan] and pick EU or "
                "self-hosted at the region prompt, or set [cyan]secrets.bitwarden."
                "server_url[/cyan] in config.yaml.[/yellow]"
            )
        elif "authorization" in lowered or "invalid" in lowered:
            console.print(
                "  [yellow]This usually means the access token is wrong or revoked. "
                "Double-check it in the Bitwarden web app.[/yellow]"
            )
        return None

    try:
        data = json.loads(res.stdout or "[]")
    except json.JSONDecodeError as exc:
        console.print(f"  [red]bws returned non-JSON: {exc}[/red]")
        return None
    if not isinstance(data, list):
        return []
    return [p for p in data if isinstance(p, dict) and p.get("id")]


# Canonical Bitwarden region endpoints.  Keep in sync with what Bitwarden
# publishes — these are stable but if a third region appears, add it here
# and to the prompt below.
_REGION_PRESETS = [
    ("US Cloud  (https://vault.bitwarden.com — bws default)", ""),
    ("EU Cloud  (https://vault.bitwarden.eu)", "https://vault.bitwarden.eu"),
]


def _resolve_server_url(
    args: argparse.Namespace,
    secrets_cfg: dict,
    console: Console,
) -> Optional[str]:
    """Pick a Bitwarden server URL for setup.

    Resolution order: 1. ``--server-url`` CLI flag (non-interactive) 2. ``BWS_SERVER_URL`` env var
    (so users running with that already set in their shell don't have to re-enter it) 3. Existing
    ``secrets.bitwarden.server_url`` value (for re-runs) 4. Interactive menu: US / EU / self-hosted
    """
    if args.server_url and args.server_url.strip():
        return args.server_url.strip()

    env_url = os.environ.get("BWS_SERVER_URL", "").strip()
    if env_url:
        console.print(
            f"  Detected [cyan]BWS_SERVER_URL[/cyan]={env_url} in your shell — using it."
        )
        return env_url

    existing = cfg_str(secrets_cfg, "server_url")
    if existing:
        console.print(
            f"  Existing config: [cyan]{existing}[/cyan]. "
            "Press Enter to keep, or pick a different option below."
        )

    table = Table(show_header=True, header_style="bold", box=None, padding=(0, 2))
    table.add_column("#", style="cyan", width=4)
    table.add_column("Region / endpoint")
    for i, (label, _url) in enumerate(_REGION_PRESETS, 1):
        table.add_row(str(i), label)
    table.add_row(str(len(_REGION_PRESETS) + 1), "Self-hosted / custom URL")
    console.print(table)

    custom_idx = len(_REGION_PRESETS) + 1
    prompt = f"  Select region [1-{custom_idx}]"
    if existing:
        prompt += " (Enter to keep current)"
    idx = prompt_index(
        console,
        prompt + ": ",
        custom_idx,
        allow_empty=bool(existing),
        empty_message="  [red]Enter a number.[/red]",
    )
    if idx == 0:
        return existing
    if idx <= len(_REGION_PRESETS):
        return _REGION_PRESETS[idx - 1][1]
    custom = console.input(
        "  Enter your Bitwarden server URL "
        "(e.g. https://vault.example.com): "
    ).strip()
    if not custom:
        console.print("  [red]Empty URL, aborting.[/red]")
        return None
    if not custom.startswith(("http://", "https://")):
        console.print(
            "  [yellow]Warning: URL doesn't start with http:// or "
            "https:// — bws may reject it.[/yellow]"
        )
    return custom
