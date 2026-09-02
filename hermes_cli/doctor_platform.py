"""Host-platform checks for hermes doctor: interpreter, SQLite, certificates, macOS TCC, gateway supervision, command install.

Split out of ``hermes_cli/doctor.py``; every moved name is re-imported there, so
``hermes_cli.doctor.<name>`` keeps resolving (and monkeypatching) as before.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path
from hermes_cli.colors import Colors, color
from hermes_cli.config import is_nix_install_method, recommended_update_command_for_method
from hermes_cli.doctor_report import (
    Finding,
    _fail_and_issue,
    _section,
    check_fail,
    check_info,
    check_ok,
    check_warn,
)
from hermes_constants import is_termux as _is_termux


def _python_install_cmd() -> str:
    return "python -m pip install" if _is_termux() else "uv pip install"


def _system_package_install_cmd(pkg: str) -> str:
    if _is_termux():
        return f"pkg install {pkg}"
    if sys.platform == "darwin":
        return f"brew install {pkg}"
    return f"sudo apt install {pkg}"


def _sqlite_upgrade_hint(install_method: str | None = None) -> str:
    """Return an actionable SQLite upgrade hint for this install layout."""
    from hermes_cli.doctor import PROJECT_ROOT, detect_install_method
    method = install_method or detect_install_method(PROJECT_ROOT)
    if method == "docker":
        command = recommended_update_command_for_method(method)
        action = f"run `{command}`, then recreate all Hermes containers"
    elif is_nix_install_method(method):
        # The Nix helper is prose guidance, not a literal shell command.
        action = recommended_update_command_for_method(method)
    elif method == "apt":
        action = f"run `{recommended_update_command_for_method(method)}`"
    else:
        action = "run `hermes update`"
    return (
        f"({action}; fixed versions: 3.51.3+ / 3.50.7 / 3.44.6 — "
        "see https://sqlite.org/wal.html#walresetbug)"
    )


def _hermes_database_paths(hermes_home: Path) -> list[tuple[str, Path]]:
    """Return (display name, path) pairs for Hermes-managed SQLite databases."""
    # backup.py owns the canonical list of per-profile stores; reuse it.
    from hermes_cli.backup import _QUICK_STATE_FILES

    entries = [
        (name, hermes_home / name)
        for name in _QUICK_STATE_FILES
        if name.endswith(".db")
    ]
    # Non-default kanban boards each keep their own kanban.db.
    for board_db in sorted((hermes_home / "kanban" / "boards").glob("*/kanban.db")):
        entries.append((str(board_db.relative_to(hermes_home)), board_db))
    return entries


_SQLITE_HEADER_MAGIC = b"SQLite format 3\x00"


def _unreadable_reason(db_path: Path) -> str:
    """Explain why a database file could not be read, without opening it.

    ``read_header_bytes_preopen`` collapses every ``OSError`` into ``None``,
    but doctor's job is to say *which* problem it hit. ``stat()`` and
    ``access()`` answer that from directory metadata alone — neither takes a
    file descriptor, so neither can cancel the file's POSIX advisory locks.
    """
    try:
        db_path.stat()
    except OSError as exc:
        return str(exc)
    if not os.access(db_path, os.R_OK):
        return f"permission denied: {db_path}"
    return "file could not be read"


def _read_journal_mode(db_path: Path) -> tuple[str | None, str | None]:
    """Return (journal mode, error) from the file header without opening the database.

    Header byte 18 is 2 for WAL and 1 for a rollback journal. Opening the
    database through the SQLite engine — even read-only — creates -wal/-shm
    sidecar files, which a diagnostic must not do.

    The byte read is routed through ``read_header_bytes_preopen`` rather than
    a bare ``open()``: closing *any* descriptor for a database file cancels
    this process's POSIX advisory locks on it, so a raw read would drop the
    locks a live connection is holding (see ``hermes_cli.sqlite_safe_read``).
    ``run_doctor`` is also called in-process by the dashboard console, which
    holds live ``SessionDB`` connections. The helper refuses in that case and
    the mode is reported as unreadable instead.
    """
    from hermes_cli.sqlite_safe_read import (
        has_live_connection,
        read_header_bytes_preopen,
    )

    header = read_header_bytes_preopen(db_path, length=20)
    if header is None:
        if has_live_connection(db_path):
            return None, "database is open in this process"
        return None, _unreadable_reason(db_path)
    if len(header) == 0:
        return None, "file is empty"
    if len(header) < 20 or not header.startswith(_SQLITE_HEADER_MAGIC):
        return None, "file is not a database"
    if header[18] == 2:
        return "wal", None
    if header[18] == 1:
        return "rollback", None
    return None, f"unrecognized file-format version {header[18]}"


def _format_db_size(db_path: Path) -> str:
    # backup.py owns human-readable size formatting; reuse it (as with
    # _QUICK_STATE_FILES above) and keep only the stat-failure wrap here.
    from hermes_cli.backup import _format_size

    try:
        nbytes = db_path.stat().st_size
    except OSError:
        return "size unknown"
    return _format_size(nbytes)


def _report_database_journal_modes(
    hermes_home: Path | None = None,
    version_info: tuple[int, ...] | None = None,
) -> None:
    """List each database's journal mode; warn on WAL under a vulnerable SQLite."""
    from hermes_cli.doctor import HERMES_HOME
    from hermes_state import _wal_reset_repair_hint, is_sqlite_wal_reset_vulnerable

    vulnerable = is_sqlite_wal_reset_vulnerable(version_info)
    home = hermes_home if hermes_home is not None else HERMES_HOME
    try:
        databases = _hermes_database_paths(home)
    except Exception as exc:
        check_warn(f"Could not list Hermes databases: {exc}")
        return
    exposed = []
    for name, path in databases:
        if not path.is_file():
            continue
        mode, error = _read_journal_mode(path)
        size = _format_db_size(path)
        if error is not None:
            if vulnerable:
                check_warn(
                    f"{name}: journal mode could not be read",
                    f"({error}; cannot rule out WAL exposure)",
                )
            else:
                check_info(f"{name}: journal mode could not be read ({error})")
        elif mode == "wal":
            if vulnerable:
                exposed.append(name)
                check_warn(
                    f"{name} is in WAL mode ({size})",
                    "(exposed to the WAL-reset bug until SQLite is upgraded)",
                )
            else:
                check_info(f"{name}: WAL journal mode ({size})")
        elif vulnerable:
            check_info(f"{name}: rollback journal mode ({size}, not exposed)")
        else:
            check_info(f"{name}: rollback journal mode ({size})")
    if exposed:
        check_info(f"To clear the exposure: {_wal_reset_repair_hint()}")


def _read_pyproject_version() -> str | None:
    """Read the ``version = "..."`` from ``pyproject.toml`` at the project root.

    Returns None when running from an installed wheel (no pyproject.toml ships
    with the package) or when the file can't be parsed. Reads only the
    ``[project]`` version, ignoring any version strings that appear in other
    tables.
    """
    from hermes_cli.doctor import PROJECT_ROOT
    pyproject = PROJECT_ROOT / "pyproject.toml"
    try:
        text = pyproject.read_text(encoding="utf-8")
    except OSError:
        return None
    in_project = False
    for raw in text.splitlines():
        line = raw.strip()
        if line.startswith("[") and line.endswith("]"):
            in_project = line == "[project]"
            continue
        if in_project and line.startswith("version") and "=" in line:
            value = line.split("=", 1)[1]
            value = value.split("#", 1)[0].strip().strip("\"'")
            return value or None
    return None


def _check_version_consistency(issues: list[str]) -> None:
    """Verify pyproject.toml version matches hermes_cli.__version__.

    A git conflict resolution (reset/merge) can revert one file without the
    other, leaving ``hermes --version`` reporting a stale version while
    ``pyproject.toml`` is current. Detect that drift so users can re-sync.
    Silent no-op for installed wheels where pyproject.toml isn't present.
    """
    try:
        from hermes_cli import __version__ as init_version
    except Exception:
        return
    pyproject_version = _read_pyproject_version()
    if pyproject_version is None:
        # Installed wheel or unreadable pyproject — nothing to cross-check.
        return
    if pyproject_version == init_version:
        check_ok("Version files consistent", f"({init_version})")
    else:
        _fail_and_issue(
            "Version mismatch between source files",
            f"(pyproject.toml {pyproject_version} != hermes_cli/__init__.py {init_version})",
            "Re-sync version files (e.g. run 'hermes update', or set "
            "hermes_cli/__init__.py __version__ to match pyproject.toml)",
            issues,
        )


def _check_s6_supervision(issues: list[str]) -> None:
    """Inside a container under our s6 /init, surface what s6 sees.

    Runs as a counterpart to :func:`_check_gateway_service_linger` for
    the systemd-on-host case. No-op everywhere except in the s6
    container so host runs aren't cluttered with irrelevant output.

    Reports:
      - Whether the main-hermes and dashboard static services are up
      - How many per-profile gateway slots are registered (via
        ``S6ServiceManager.list_profile_gateways()``) and how many are
        currently supervised as ``up``
    """
    try:
        from hermes_cli.service_manager import (
            S6ServiceManager,
            detect_service_manager,
        )
    except Exception:
        return

    if detect_service_manager() != "s6":
        return

    _section("s6 Supervision")

    mgr = S6ServiceManager()

    # Static services. They live under /run/service/ via s6-rc symlinks,
    # so the same s6-svstat probe works.
    for static in ("main-hermes", "dashboard"):
        if mgr.is_running(static):
            check_ok(f"{static}: up")
        else:
            check_info(f"{static}: down (expected if not enabled via env)")

    profiles = mgr.list_profile_gateways()
    if not profiles:
        check_info("No per-profile gateways registered yet — create one with `hermes profile create <name>`")
        return

    up_count = sum(1 for p in profiles if mgr.is_running(f"gateway-{p}"))
    check_ok(
        f"Per-profile gateways: {up_count}/{len(profiles)} supervised up"
        + (f" ({', '.join(sorted(profiles))})" if len(profiles) <= 8 else "")
    )


def check_certificates(should_fix: bool = False, issues: "list | None" = None) -> None:
    """Verify the certifi CA bundle is loadable.

    Surfaces the SSLConfigurationError user-friendly path before they hit
    a wall of tracebacks on the first outbound HTTPS call.

    With ``--fix``, a broken bundle (missing/corrupt ``cacert.pem`` — e.g.
    after a brew Python upgrade rebuilt the venv, #29866) is repaired by
    force-reinstalling certifi into THIS interpreter's environment and
    re-verifying.
    """
    try:
        from agent.ssl_guard import verify_ca_bundle_with_fallback
        from agent.errors import SSLConfigurationError
    except Exception as e:
        check_warn("SSL certificate check skipped", str(e))
        return

    try:
        verify_ca_bundle_with_fallback()
        check_ok("SSL CA certificate bundle is valid")
        return
    except SSLConfigurationError as e:
        first_error = str(e)
    except Exception as e:
        check_warn("SSL certificate check skipped", str(e))
        return

    if not should_fix:
        check_fail("SSL CA certificate bundle is broken", first_error)
        if issues is not None:
            issues.append(
                "Repair the CA bundle: run `hermes doctor --fix`, or "
                f"`{sys.executable} -m pip install --force-reinstall certifi`"
            )
        return

    # --fix: force-reinstall certifi into the running interpreter's env and
    # re-verify. importlib caches are invalidated so certifi.where() resolves
    # the fresh install without a process restart.
    check_fail("SSL CA certificate bundle is broken", first_error)
    print("    → Repairing: force-reinstalling certifi...")
    try:
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", "--force-reinstall", "certifi"],
            capture_output=True,
            text=True,
            timeout=300,
        )
    except Exception as exc:
        check_fail("certifi repair could not run pip", str(exc))
        if issues is not None:
            issues.append(
                f"Reinstall certifi manually: {sys.executable} -m pip install "
                "--force-reinstall certifi"
            )
        return
    if result.returncode != 0:
        tail = (result.stderr or result.stdout or "")[-500:]
        check_fail("certifi reinstall failed", tail)
        if issues is not None:
            issues.append(
                f"Reinstall certifi manually: {sys.executable} -m pip install "
                "--force-reinstall certifi"
            )
        return

    # Drop any cached certifi module so where() re-resolves the new bundle.
    import importlib
    for mod_name in [m for m in sys.modules if m == "certifi" or m.startswith("certifi.")]:
        sys.modules.pop(mod_name, None)
    importlib.invalidate_caches()

    try:
        verify_ca_bundle_with_fallback()
        check_ok("SSL CA certificate bundle repaired (certifi reinstalled)")
    except SSLConfigurationError as e:
        check_fail("SSL CA certificate bundle still broken after reinstall", str(e))
        if issues is not None:
            issues.append(
                "certifi reinstall did not restore the CA bundle — check for a "
                "custom CA env var (SSL_CERT_FILE/REQUESTS_CA_BUNDLE) pointing "
                "at a missing file, or recreate the venv."
            )


def _check_gateway_service_linger(issues: list[str]) -> None:
    """Warn when a systemd user gateway service will stop after logout.

    Skipped inside a container running under s6 — the linger concept
    (user-systemd surviving SSH logout) doesn't apply there, and the
    s6 supervision state is surfaced separately by
    ``_check_s6_supervision``.
    """
    try:
        from hermes_cli.gateway import (
            get_systemd_linger_status,
            get_systemd_unit_path,
            is_linux,
        )
        from hermes_cli.service_manager import detect_service_manager
    except Exception as e:
        check_warn("Gateway service linger", f"(could not import gateway helpers: {e})")
        return

    if not is_linux():
        return

    # Inside a container under our s6 /init, _check_s6_supervision
    # reports the live supervision state; the linger warning would be
    # confusing here (no systemd, no logout, no "lingering" concept).
    if detect_service_manager() == "s6":
        return

    unit_path = get_systemd_unit_path()
    if not unit_path.exists():
        return

    _section("Gateway Service")
    linger_enabled, linger_detail = get_systemd_linger_status()
    if linger_enabled is True:
        check_ok("Systemd linger enabled", "(gateway service survives logout)")
    elif linger_enabled is False:
        check_warn("Systemd linger disabled", "(gateway may stop after logout)")
        check_info("Run: sudo loginctl enable-linger $USER")
        issues.append("Enable linger for the gateway user service: sudo loginctl enable-linger $USER")
    else:
        check_warn("Could not verify systemd linger", f"({linger_detail})")


def check_macos_tcc_grants() -> None:
    """Check macOS TCC grant persistence for a locally-built desktop bundle.

    TCC keys permission grants (Screen Recording, Full Disk Access,
    Accessibility, ...) to the app's code-signing requirement. A bundle
    signed with the pre-#73681 cdhash-pinned ad-hoc identity gets a new DR on
    every rebuild, so all grants silently stop matching — and the stale row
    keeps the System Settings toggle ON while macOS re-prompts on every
    capture (issue #86385).

    Post-#73681 builds pin ``designated => identifier "com.nousresearch.hermes"``
    (no cdhash), so new grants survive rebuilds — but grants made to older
    binaries remain stale until re-granted once. The stale state is not
    directly readable (TCC.db needs Full Disk Access), so this check reports
    the DR class and, when the DR is stable, prints the exact one-time repair.
    Silent on non-macOS and when no desktop bundle is installed.
    """
    from hermes_cli.doctor import _desktop_app_bundle, _macos_desktop_dr
    if sys.platform != "darwin":
        return
    app = _desktop_app_bundle()
    if app is None:
        return
    dr = _macos_desktop_dr(app)
    if not dr:
        check_warn(
            "macOS TCC grant check",
            "(could not read code-signing requirement of the desktop bundle)",
        )
        return
    # The DR string is the only readable signal — TCC.db itself needs Full
    # Disk Access. A cdhash anchor marks the pre-#73681 ad-hoc identity
    # (rebuild ⇒ new cdhash ⇒ stale grants); its absence marks identifier-
    # pinned. Treat the match as a proxy for the signing class, not a
    # contract on DR wording.
    if "cdhash" in dr.lower():
        check_warn(
            "macOS TCC grants will reset after every update",
            "the desktop bundle's designated requirement is cdhash-pinned "
            "(pre-#73681 build) — rebuilds invalidate all permission grants. "
            "Run `hermes update` to get the stable identifier-pinned signing "
            "identity, then re-grant permissions once.",
        )
        return
    if "certificate" in dr.lower():
        # Certificate-anchored DR (hermes desktop --setup-tcc-identity, or a
        # notarized release build): the strongest anchor TCC can key on.
        check_ok(
            "macOS TCC signing identity is stable",
            "(certificate-anchored DR; grants survive rebuilds)",
        )
    else:
        check_ok(
            "macOS TCC signing identity is stable",
            "(identifier-pinned DR; grants survive rebuilds — for the strongest "
            "anchor, see `hermes desktop --setup-tcc-identity`)",
        )
    check_info(
        "If macOS still re-prompts for permissions (toggle shows ON): the stored "
        "grant is stale — run `tccutil reset ScreenCapture com.nousresearch.hermes` "
        "(repeat per affected service), toggle it ON in System Settings, then "
        "fully quit & relaunch Hermes once."
    )


def _desktop_app_bundle() -> Path | None:
    """Locate the locally-built desktop app bundle, if any.

    Mirrors the install layout the self-updater produces
    (``apps/desktop/release/mac-<arch>/Hermes.app``) — the only layout whose
    ad-hoc re-signed bundle can invalidate TCC grants. When multiple arch
    trees coexist (stale cross-build), the newest wins, matching
    ``_desktop_packaged_executable``'s selection. ``/Applications/Hermes.app``
    is deliberately not probed: it is the separately-signed Hermes-Setup
    launcher (``com.nousresearch.hermes.setup``, certificate-anchored), whose
    grants are stable by construction and unaffected by rebuilds.
    """
    root = Path(__file__).resolve().parents[1]
    release_dir = root / "apps" / "desktop" / "release"
    candidates = [p for p in release_dir.glob("mac*/Hermes.app") if p.is_dir()]
    if not candidates:
        return None
    return max(candidates, key=lambda p: p.stat().st_mtime)


def _macos_desktop_dr(app: Path) -> str | None:
    """Return the bundle's designated requirement string, or None on failure."""
    codesign = shutil.which("codesign")
    if not codesign:
        return None
    try:
        proc = subprocess.run(
            [codesign, "-d", "--requirements", "-", str(app)],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired):
        # Never let a hanging codesign abort the whole doctor run — the
        # caller falls through to its "could not read" warning.
        return None
    if proc.returncode != 0:
        return None
    return (proc.stdout or "") + (proc.stderr or "")


def check_macos_tcc_anchor(should_fix: bool = False) -> None:
    """Report (and optionally install) the dylib-complete TCC anchor (#95596).

    Silent on non-macOS and for interpreters that are not uv-managed.  Never
    raises — a failed check must not crash doctor.  Install is gated by the
    module's pre-install boot probe, so ``--fix`` cannot brick the CLI.
    """
    try:
        from hermes_cli import macos_tcc_anchor as tcc

        status, detail = tcc.tcc_anchor_state()
        if status == "skip":
            return
        if status == "active":
            check_ok("macOS TCC anchor active", f"({detail})")
            return
        if should_fix:
            anchored = tcc.ensure_tcc_anchor()
            if anchored is not None:
                check_ok("macOS TCC anchor installed", f"({anchored})")
                return
        check_warn(
            "macOS TCC anchor missing" if status == "missing" else "macOS TCC anchor stale",
            f"({detail})",
        )
    except Exception as e:  # diagnostics must never crash
        check_warn("macOS TCC anchor check failed", f"({e})")


def check_macos_full_disk_access() -> None:
    """One-grant guidance: Full Disk Access silences every folder prompt.

    macOS TCC prompts per-category (Desktop, then Downloads, then Documents,
    ...), so first-run agents drip-feed permission dialogs as they touch each
    folder. ONE Full Disk Access grant covers all of them, permanently — and
    with the stable signing identities now in place (#73681/#95091/#95131),
    it survives updates too. This check probes whether the terminal context
    already has FDA and, when it doesn't, prints the exact one-switch setup
    with the System Settings deep link.

    Probe: readability of ``~/Library/Application Support/com.apple.TCC`` —
    the TCC database directory itself is FDA-gated, readable ONLY with the
    grant, and (critically) probing it with os.access/listdir does NOT
    trigger a prompt: TCC prompts fire for protected-CATEGORY paths (Desktop
    etc.), while the TCC dir simply returns EPERM without one. Silent on
    non-macOS.
    """
    if sys.platform != "darwin":
        return
    tcc_dir = Path.home() / "Library" / "Application Support" / "com.apple.TCC"
    try:
        os.listdir(tcc_dir)
        has_fda = True
    except PermissionError:
        has_fda = False
    except OSError:
        # Missing dir / other error: can't tell — stay silent rather than
        # nag on an indeterminate probe.
        return
    if has_fda:
        check_ok(
            "macOS Full Disk Access granted",
            "(no per-folder permission prompts will occur)",
        )
        return
    check_info(
        "One switch silences all macOS folder prompts: grant your terminal "
        "app Full Disk Access and Hermes will never trip per-folder dialogs "
        "(Desktop/Downloads/Documents/...) again. Open: System Settings → "
        "Privacy & Security → Full Disk Access — or run:\n"
        "      open \"x-apple.systempreferences:com.apple.preference"
        ".security?Privacy_AllFiles\"\n"
        "    then enable your terminal (and Hermes.app if you use Desktop), "
        "and restart them once. With Hermes' stable signing identities the "
        "grant survives every update."
    )


def _check_security_advisories(should_fix: bool) -> Finding:
    """Compromised-package advisories; funnels remediation into manual issues."""
    f = Finding()
    manual_issues = f.manual_issues
    try:
        from hermes_cli.security_advisories import (
            detect_compromised,
            filter_unacked,
            full_remediation_text,
            get_acked_ids,
        )
        all_hits = detect_compromised()
        fresh_hits = filter_unacked(all_hits)
        if fresh_hits:
            for hit in fresh_hits:
                check_fail(
                    f"{hit.advisory.title}",
                    f"({hit.package}=={hit.installed_version})",
                )
                # Print the full remediation block, indented under the
                # check_fail header so it reads as a single section.
                for line in full_remediation_text(hit):
                    if line:
                        print(f"    {color(line, Colors.YELLOW)}")
                    else:
                        print()
                # Funnel into the action list so the summary block surfaces it
                # for users who scroll past the section.
                manual_issues.append(
                    f"Resolve security advisory {hit.advisory.id}: "
                    f"uninstall {hit.package}=={hit.installed_version} and "
                    f"rotate credentials, then run "
                    f"`hermes doctor --ack {hit.advisory.id}`."
                )
            # Acked-but-still-installed: show as informational so the user
            # knows the package is still on disk after the ack.
            acked_ids = get_acked_ids()
            for h in all_hits:
                if h.advisory.id in acked_ids:
                    check_warn(
                        f"{h.package}=={h.installed_version} still installed "
                        f"(advisory {h.advisory.id} acknowledged)",
                    )
        else:
            check_ok("No active security advisories")
    except Exception as e:
        # Never let a bug in the advisory check block the rest of doctor.
        check_warn(f"Security advisory check failed: {e}")
    return f


def _check_python_environment(should_fix: bool) -> Finding:
    """Interpreter, linked SQLite, venv, macOS TCC anchors, version-file drift."""
    f = Finding()
    issues = f.issues
    py_version = sys.version_info
    if py_version >= (3, 11):
        check_ok(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}")
    elif py_version >= (3, 10):
        check_ok(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}")
        check_warn("Python 3.11+ recommended for RL Training tools (tinker requires >= 3.11)")
    elif py_version >= (3, 8):
        check_warn(f"Python {py_version.major}.{py_version.minor}.{py_version.micro}", "(3.10+ recommended)")
    else:
        _fail_and_issue(
            f"Python {py_version.major}.{py_version.minor}.{py_version.micro}",
            "(3.10+ required)",
            "Upgrade Python to 3.10+",
            issues,
        )

    # Linked SQLite library (issue #69784): version + source id matter independently
    # of the Python minor — uv's python-build-standalone can keep a vulnerable
    # SQLite across Python upgrades.
    try:
        import sqlite3
        from hermes_state import is_sqlite_wal_reset_vulnerable, sqlite_source_id

        _sqlite_ver = sqlite3.sqlite_version
        _sqlite_src = sqlite_source_id()
        _sqlite_src_short = (
            (_sqlite_src[:48] + "…") if len(_sqlite_src) > 48 else _sqlite_src
        )
        if is_sqlite_wal_reset_vulnerable():
            # Warn-only: Hermes already refuses to enable WAL on fresh DBs.
            # Do not append to ``issues`` because runtime repair remains
            # best-effort and unsupported installs may need manual action.
            check_warn(
                f"SQLite {_sqlite_ver} (WAL-reset bug)",
                _sqlite_upgrade_hint(),
            )
        else:
            check_ok(f"SQLite {_sqlite_ver}")
        if _sqlite_src_short:
            check_info(f"SQLite source id: {_sqlite_src_short}")
        _report_database_journal_modes()
    except Exception as e:
        check_warn(f"SQLite version probe failed: {e}")
    # Check if in virtual environment
    in_venv = sys.prefix != sys.base_prefix
    if in_venv:
        check_ok("Virtual environment active")
    else:
        check_warn("Not in virtual environment", "(recommended)")

    # macOS TCC interpreter anchor (#95596): dylib-complete re-land of the
    # mechanism reverted in #95563. Silent on non-macOS.
    check_macos_tcc_anchor(should_fix=should_fix)

    # macOS Full Disk Access (issue #52010 follow-up): one grant silences
    # every per-folder prompt permanently. Silent on non-macOS.
    check_macos_full_disk_access()

    # Detect drift between pyproject.toml and hermes_cli/__init__.py versions
    # (a git conflict resolution can silently revert one but not the other).
    _check_version_consistency(issues)

    # macOS TCC grant persistence (issue #86385): a locally-built desktop
    # bundle whose DR is cdhash-pinned loses every permission grant on each
    # rebuild; a post-#73681 identifier-pinned DR survives, but grants made
    # to older binaries stay stale (toggle shows ON while macOS re-prompts).
    check_macos_tcc_grants()
    return f


def _check_certificates(should_fix: bool) -> Finding:
    f = Finding()
    manual_issues = f.manual_issues
    check_certificates(should_fix=should_fix, issues=manual_issues)
    return f


def _check_required_packages(should_fix: bool) -> Finding:
    f = Finding()
    issues = f.issues
    required_packages = [
        ("openai", "OpenAI SDK"),
        ("rich", "Rich (terminal UI)"),
        ("dotenv", "python-dotenv"),
        ("yaml", "PyYAML"),
        ("httpx", "HTTPX"),
    ]
    
    optional_packages = [
        ("croniter", "Croniter (cron expressions)"),
        ("telegram", "python-telegram-bot"),
        ("discord", "discord.py"),
    ]
    
    for module, name in required_packages:
        try:
            __import__(module)
            check_ok(name)
        except ImportError:
            _fail_and_issue(name, "(missing)", f"Install {name}: {_python_install_cmd()} {module}", issues)
    
    for module, name in optional_packages:
        try:
            __import__(module)
            check_ok(name, "(optional)")
        except ImportError:
            check_warn(name, "(optional, not installed)")
    return f


def _check_gateway_supervision(should_fix: bool) -> Finding:
    f = Finding()
    issues = f.issues
    _check_gateway_service_linger(issues)
    _check_s6_supervision(issues)
    return f


def _check_command_installation(should_fix: bool) -> Finding:
    """Venv entry point and the ~/.local/bin (or $PREFIX/bin) symlink; skipped on Windows."""
    from hermes_cli.doctor import PROJECT_ROOT
    f = Finding()
    issues, manual_issues = f.issues, f.manual_issues
    if sys.platform != "win32":
        _section("Command Installation")
        # Determine the venv entry point location
        _venv_bin = None
        for _venv_name in ("venv", ".venv"):
            _candidate = PROJECT_ROOT / _venv_name / "bin" / "hermes"
            if _candidate.exists():
                _venv_bin = _candidate
                break

        # Determine the expected command link directory (mirrors install.sh logic)
        _prefix = os.environ.get("PREFIX", "")
        _is_termux_env = bool(os.environ.get("TERMUX_VERSION")) or "com.termux/files/usr" in _prefix
        if _is_termux_env and _prefix:
            _cmd_link_dir = Path(_prefix) / "bin"
            _cmd_link_display = "$PREFIX/bin"
        else:
            _cmd_link_dir = Path.home() / ".local" / "bin"
            _cmd_link_display = "~/.local/bin"
        _cmd_link = _cmd_link_dir / "hermes"

        if _venv_bin is None:
            check_warn(
                "Venv entry point not found",
                "(hermes not in venv/bin/ or .venv/bin/ — reinstall with pip install -e '.[all]')"
            )
            manual_issues.append(
                f"Reinstall entry point: cd {PROJECT_ROOT} && source venv/bin/activate && pip install -e '.[all]'"
            )
        else:
            check_ok(f"Venv entry point exists ({_venv_bin.relative_to(PROJECT_ROOT)})")

            # Check the symlink at the command link location
            if _cmd_link.is_symlink():
                _target = _cmd_link.resolve()
                _expected = _venv_bin.resolve()
                if _target == _expected:
                    check_ok(f"{_cmd_link_display}/hermes → correct target")
                else:
                    check_warn(
                        f"{_cmd_link_display}/hermes points to wrong target",
                        f"(→ {_target}, expected → {_expected})"
                    )
                    if should_fix:
                        _cmd_link.unlink()
                        _cmd_link.symlink_to(_venv_bin)
                        check_ok(f"Fixed symlink: {_cmd_link_display}/hermes → {_venv_bin}")
                        f.fixed += 1
                    else:
                        issues.append(f"Broken symlink at {_cmd_link_display}/hermes — run 'hermes doctor --fix'")
            elif _cmd_link.exists():
                # It's a regular file, not a symlink — possibly a wrapper script
                check_ok(f"{_cmd_link_display}/hermes exists (non-symlink)")
            else:
                check_fail(
                    f"{_cmd_link_display}/hermes not found",
                    "(hermes command may not work outside the venv)"
                )
                if should_fix:
                    _cmd_link_dir.mkdir(parents=True, exist_ok=True)
                    _cmd_link.symlink_to(_venv_bin)
                    check_ok(f"Created symlink: {_cmd_link_display}/hermes → {_venv_bin}")
                    f.fixed += 1

                    # Check if the link dir is on PATH
                    _path_dirs = os.environ.get("PATH", "").split(os.pathsep)
                    if str(_cmd_link_dir) not in _path_dirs:
                        check_warn(
                            f"{_cmd_link_display} is not on your PATH",
                            "(add it to your shell config: export PATH=\"$HOME/.local/bin:$PATH\")"
                        )
                        manual_issues.append(f"Add {_cmd_link_display} to your PATH")
                else:
                    issues.append(f"Missing {_cmd_link_display}/hermes symlink — run 'hermes doctor --fix'")
    return f
