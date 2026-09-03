"""Tirith pre-exec security scanning wrapper.

Runs the tirith binary as a subprocess to scan commands for content-level threats
(homograph URLs, pipe-to-interpreter, terminal injection, ...). The exit code is the
verdict source of truth (0 allow, 1 block, 2 warn); JSON stdout only enriches findings.
Operational failures (spawn error, timeout, unknown exit) respect ``fail_open``;
programming errors propagate. Auto-install: if tirith is not on PATH / the configured
path it is downloaded from GitHub releases to $HERMES_HOME/bin/tirith in a background
thread. SHA-256 is always verified; cosign provenance when cosign is on PATH.
"""

import hashlib
import json
import logging
import os
import platform
import shutil
import stat
import subprocess
import tarfile
import tempfile
import threading
import time
import urllib.request

from hermes_constants import get_hermes_home

logger = logging.getLogger(__name__)

_REPO = "sheeki03/tirith"
# Cosign provenance pinned to the release workflow, not the whole repo.
_COSIGN_IDENTITY_REGEXP = f"^https://github.com/{_REPO}/\\.github/workflows/release\\.yml@refs/tags/v"
_COSIGN_ISSUER = "https://token.actions.githubusercontent.com"

# --- Config helpers ---

def _env_bool(key: str, default: bool) -> bool:
    val = os.getenv(key)
    return default if val is None else val.lower() in {"1", "true", "yes"}


def _env_int(key: str, default: int) -> int:
    val = os.getenv(key)
    try:
        return default if val is None else int(val)
    except ValueError:
        return default


def _load_security_config() -> dict:
    """Security settings from config.yaml, with env var overrides."""
    try:
        from hermes_cli.config import load_config_readonly
        cfg = load_config_readonly().get("security", {}) or {}
    except Exception:
        cfg = {}
    return {
        "tirith_enabled": _env_bool("TIRITH_ENABLED", cfg.get("tirith_enabled", True)),
        "tirith_path": os.getenv("TIRITH_BIN", cfg.get("tirith_path", "tirith")),
        "tirith_timeout": _env_int("TIRITH_TIMEOUT", cfg.get("tirith_timeout", 5)),
        "tirith_fail_open": _env_bool("TIRITH_FAIL_OPEN", cfg.get("tirith_fail_open", True)),
    }


# --- Module state ---

# Cached path after first resolution. _INSTALL_FAILED means "tried and failed" (distinct
# from None = "not yet tried") so a failed install is not retried per command.
_resolved_path: str | None | bool = None
_INSTALL_FAILED = False
_install_failure_reason: str = ""  # reason tag when _resolved_path is _INSTALL_FAILED

# Circuit breaker: after _CRASH_LIMIT consecutive spawn/execution failures tirith is disabled
# for the rest of the process so a broken binary can't turn every tool call into a fail-open
# retry loop. Reset on success. Lock-free on purpose: a racing double-increment only opens the
# breaker one call early; no corruption or security bypass is possible.
_CRASH_LIMIT = 3
_crash_count: int = 0
_circuit_open: bool = False

_install_lock = threading.Lock()
_install_thread: threading.Thread | None = None

# Warn-once: spawn/path warnings sit in the hot path and would otherwise repeat once per
# terminal command while tirith is unavailable (e.g. install thread still running).
_warned_messages: set[str] = set()
_warned_lock = threading.Lock()

_MARKER_TTL = 86400  # disk failure marker validity (24h) -- avoids retry across restarts


def _record_tirith_crash() -> None:
    global _crash_count, _circuit_open
    _crash_count += 1
    if _crash_count >= _CRASH_LIMIT:
        _circuit_open = True
        logger.warning(
            "tirith circuit breaker opened after %d consecutive failures; "
            "disabling for the rest of the process",
            _crash_count,
        )


def _warn_once(key: str, message: str, *args) -> None:
    """``logger.warning`` at most once per ``key`` for the process lifetime."""
    with _warned_lock:
        if key in _warned_messages:
            return
        _warned_messages.add(key)
    logger.warning(message, *args)


def _reset_spawn_warning_state() -> None:
    """Clear the warn-once set so a failure after a fresh (re)install surfaces again."""
    with _warned_lock:
        _warned_messages.clear()


def _cached_path() -> str | None:
    """The path resolved on a previous call, or None if unresolved/failed."""
    if _resolved_path is None or _resolved_path is _INSTALL_FAILED:
        return None
    return _resolved_path


def _set_resolved(path: str) -> None:
    global _resolved_path, _install_failure_reason
    _resolved_path, _install_failure_reason = path, ""


def _set_failed(reason: str) -> None:
    global _resolved_path, _install_failure_reason
    _resolved_path, _install_failure_reason = _INSTALL_FAILED, reason


# --- Disk failure marker ---

def _failure_marker_path() -> str:
    return os.path.join(str(get_hermes_home()), ".tirith-install-failed")


def _read_failure_reason() -> str | None:
    """The marker's reason, or None if absent or older than _MARKER_TTL."""
    try:
        p = _failure_marker_path()
        if (time.time() - os.path.getmtime(p)) >= _MARKER_TTL:
            return None
        with open(p, "r", encoding="utf-8") as f:
            return f.read().strip()
    except OSError:
        return None


def _is_install_failed_on_disk() -> bool:
    """True if a recent install failure was persisted and is still non-retryable.
    A 'cosign_missing' marker is auto-cleared once cosign appears on PATH."""
    reason = _read_failure_reason()
    if reason is None:
        return False
    if reason == "cosign_missing" and shutil.which("cosign"):
        _clear_install_failed()
        return False
    return True


def _mark_install_failed(reason: str = ""):
    """Persist install failure to disk; ``reason`` is a short retryability tag."""
    try:
        p = _failure_marker_path()
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            f.write(reason)
    except OSError:
        pass


def _clear_install_failed():
    """Remove the failure marker and reset warn-once state after a successful install."""
    _reset_spawn_warning_state()
    try:
        os.unlink(_failure_marker_path())
    except OSError:
        pass


def _disk_marker_blocks_install() -> bool:
    """Apply a still-valid disk marker to module state; True if install must be skipped.
    Keeps the marker's real reason so in-process retry can detect cosign_missing."""
    disk_reason = _read_failure_reason()
    if disk_reason is None or not _is_install_failed_on_disk():
        return False
    _set_failed(disk_reason)
    return True


# --- Auto-install ---

def _hermes_bin_dir() -> str:
    """$HERMES_HOME/bin, created if needed."""
    d = os.path.join(str(get_hermes_home()), "bin")
    os.makedirs(d, exist_ok=True)
    return d


# Rust target triple components. Android (Termux) is ABI-compatible with Linux. Windows is
# absent on purpose (no tirith build): None = "never available here", pattern guards still run.
_TARGET_PLATFORMS = {"Darwin": "apple-darwin", "Linux": "unknown-linux-gnu", "Android": "unknown-linux-gnu"}
_TARGET_ARCHES = {"x86_64": "x86_64", "amd64": "x86_64", "aarch64": "aarch64", "arm64": "aarch64"}


def _detect_target() -> str | None:
    """Rust target triple for this platform, or None if tirith has no build for it."""
    plat = _TARGET_PLATFORMS.get(platform.system())
    arch = _TARGET_ARCHES.get(platform.machine().lower())
    return f"{arch}-{plat}" if plat and arch else None


def is_platform_supported() -> bool:
    """True when tirith ships a prebuilt binary for this OS+arch (CLI banner uses this)."""
    return _detect_target() is not None


def _download_file(url: str, dest: str, timeout: int = 10):
    req = urllib.request.Request(url)
    from agent.secret_scope import get_secret
    token = get_secret("GITHUB_TOKEN")
    if token:
        req.add_header("Authorization", f"token {token}")
    with urllib.request.urlopen(req, timeout=timeout) as resp, open(dest, "wb") as f:
        shutil.copyfileobj(resp, f)


def _verify_cosign(checksums_path: str, sig_path: str, cert_path: str) -> bool | None:
    """Verify cosign provenance on checksums.txt: True verified, False rejected,
    None when cosign is not on PATH / failed to execute."""
    cosign = shutil.which("cosign")
    if not cosign:
        logger.info("cosign not found on PATH")
        return None
    try:
        result = subprocess.run(
            [cosign, "verify-blob", "--certificate", cert_path, "--signature", sig_path,
             "--certificate-identity-regexp", _COSIGN_IDENTITY_REGEXP,
             "--certificate-oidc-issuer", _COSIGN_ISSUER, checksums_path],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
            timeout=15, stdin=subprocess.DEVNULL,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        logger.warning("cosign execution failed: %s", exc)
        return None
    if result.returncode == 0:
        logger.info("cosign provenance verification passed")
        return True
    logger.warning("cosign verification failed (exit %d): %s", result.returncode, result.stderr.strip())
    return False


def _verify_release_provenance(base_url: str, tmpdir: str, checksums_path: str, log) -> tuple[bool, str]:
    """Cosign step of the install -> ``(cosign_verified, failure_reason)``. Only an explicit
    cosign rejection aborts; missing/broken cosign or artifacts fall back to SHA-256 only."""
    if not shutil.which("cosign"):
        logger.info("cosign not on PATH — installing tirith with SHA-256 verification only "
                    "(install cosign for full supply chain verification)")
        return False, ""
    sig_path = os.path.join(tmpdir, "checksums.txt.sig")
    cert_path = os.path.join(tmpdir, "checksums.txt.pem")
    try:
        _download_file(f"{base_url}/checksums.txt.sig", sig_path)
        _download_file(f"{base_url}/checksums.txt.pem", cert_path)
    except Exception as exc:
        logger.info("cosign artifacts unavailable (%s), proceeding with SHA-256 only", exc)
        return False, ""
    verified = _verify_cosign(checksums_path, sig_path, cert_path)
    if verified is False:
        log("tirith install aborted: cosign provenance verification failed")
        return False, "cosign_verification_failed"
    if verified is None:
        logger.info("cosign execution failed, proceeding with SHA-256 only")
    return verified is True, ""


def _verify_checksum(archive_path: str, checksums_path: str, archive_name: str) -> bool:
    """Verify SHA-256 of the archive against checksums.txt ("<hash>  <filename>" lines)."""
    with open(checksums_path, encoding="utf-8") as f:
        parsed = (line.strip().split("  ", 1) for line in f)
        expected = next((h for h, *n in parsed if n == [archive_name]), None)
    if not expected:
        logger.warning("No checksum entry for %s", archive_name)
        return False
    sha = hashlib.sha256()
    with open(archive_path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    actual = sha.hexdigest()
    if actual != expected:
        logger.warning("Checksum mismatch: expected %s, got %s", expected, actual)
        return False
    return True


def _extract_tirith_binary(tar: tarfile.TarFile, dest_dir: str, log) -> tuple[str | None, str]:
    """Extract the tirith binary from a release archive into dest_dir -> ``(path, reason)``."""
    for member in tar.getmembers():
        is_tirith = member.name == "tirith" or member.name.endswith("/tirith")
        if not is_tirith or ".." in member.name:
            continue
        if not member.isfile():
            log("tirith archive member is not a regular file: %s", member.name)
            return None, "binary_not_regular_file"
        src_file = tar.extractfile(member)
        if src_file is None:
            log("tirith binary could not be read from archive")
            return None, "binary_extract_failed"
        dest_path = os.path.join(dest_dir, "tirith")
        with src_file, open(dest_path, "wb") as out:
            shutil.copyfileobj(src_file, out)
        return dest_path, ""
    log("tirith binary not found in archive")
    return None, "binary_not_in_archive"


def _install_tirith(*, log_failures: bool = True) -> tuple[str | None, str]:
    """Download and install tirith to $HERMES_HOME/bin/tirith -> ``(installed_path,
    failure_reason)``; the reason ("" on success) is the disk marker's retryability tag."""
    log = logger.warning if log_failures else logger.debug
    target = _detect_target()
    if not target:
        logger.info("tirith auto-install: unsupported platform %s/%s", platform.system(), platform.machine())
        return None, "unsupported_platform"
    archive_name = f"tirith-{target}.tar.gz"
    base_url = f"https://github.com/{_REPO}/releases/latest/download"
    try:
        tmpdir = tempfile.mkdtemp(prefix="tirith-install-")
    except OSError as exc:
        log("tirith install failed: cannot create temp dir: %s", exc)
        return None, "no_space"
    try:
        archive_path = os.path.join(tmpdir, archive_name)
        checksums_path = os.path.join(tmpdir, "checksums.txt")
        logger.info("tirith not found — downloading latest release for %s...", target)
        try:
            _download_file(f"{base_url}/{archive_name}", archive_path)
            _download_file(f"{base_url}/checksums.txt", checksums_path)
        except Exception as exc:
            log("tirith download failed: %s", exc)
            return None, "download_failed"
        cosign_verified, reason = _verify_release_provenance(base_url, tmpdir, checksums_path, log)
        if reason:
            return None, reason
        if not _verify_checksum(archive_path, checksums_path, archive_name):
            return None, "checksum_failed"
        with tarfile.open(archive_path, "r:gz") as tar:
            src, reason = _extract_tirith_binary(tar, tmpdir, log)
            if src is None:
                return None, reason
        dest = os.path.join(_hermes_bin_dir(), "tirith")
        try:
            shutil.move(src, dest)
        except OSError:
            # Cross-device move (Docker, NFS): copy2's metadata step can raise PermissionError,
            # so fall back to plain copy + chmod; a partial dest is removed to avoid a
            # non-executable retry loop.
            try:
                shutil.copy(src, dest)
            except OSError:
                try:
                    os.unlink(dest)
                except OSError:
                    pass
                return None, "cross_device_copy_failed"
        os.chmod(dest, os.stat(dest).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
        logger.info("tirith installed to %s (%s)", dest, "cosign + SHA-256" if cosign_verified else "SHA-256 only")
        return dest, ""
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


# --- Path resolution ---

def _is_executable(path: str) -> bool:
    return os.path.isfile(path) and os.access(path, os.X_OK)


def _find_local_tirith() -> str | None:
    """Cheap local lookup for the default "tirith": PATH, then $HERMES_HOME/bin."""
    found = shutil.which("tirith")
    if found:
        return found
    hermes_bin = os.path.join(_hermes_bin_dir(), "tirith")
    return hermes_bin if _is_executable(hermes_bin) else None


def _resolve_locally(configured_path: str, *, warn_missing: bool) -> tuple[str | None, bool]:
    """Network-free resolution shared by _resolve_tirith_path and ensure_installed ->
    ``(path, may_install)``. ``path`` set = resolved (module state updated). Otherwise
    ``may_install`` is False when the miss is terminal (explicit path missing, cached
    non-retryable failure) and True when the disk marker / install step may proceed."""
    global _resolved_path, _install_failure_reason
    expanded = os.path.expanduser(configured_path)
    # An explicit (non-"tirith") path is authoritative: never auto-download a replacement.
    if configured_path != "tirith":
        found = expanded if _is_executable(expanded) else shutil.which(expanded)
        if found:
            _resolved_path = found
            return found, False
        if warn_missing:
            logger.warning("Configured tirith path %r not found; scanning disabled", configured_path)
        _set_failed("explicit_path_missing")
        return None, False
    # Always re-run the cheap local checks so a manual install is picked up even after a
    # previous network failure (a long-lived gateway recovers without restart).
    found = _find_local_tirith()
    if found:
        _set_resolved(found)
        _clear_install_failed()
        return found, False
    # Previous install failed: skip the network retry unless the retryable cosign_missing
    # cause has been resolved in-process.
    if _resolved_path is _INSTALL_FAILED:
        if _install_failure_reason != "cosign_missing" or not shutil.which("cosign"):
            return None, False
        _resolved_path, _install_failure_reason = None, ""
        _clear_install_failed()
    return None, True


def _record_install_result(installed: str | None, reason: str) -> None:
    if installed:
        _set_resolved(installed)
        _clear_install_failed()
    else:
        _set_failed(reason)
        _mark_install_failed(reason)


def _resolve_tirith_path(configured_path: str) -> str:
    """Resolve the tirith binary path, auto-installing synchronously if necessary.
    Default "tirith": PATH → $HERMES_HOME/bin/tirith → auto-install; failed installs are
    cached for the process (and on disk for 24h). On any miss the expanded configured path
    is returned so the spawn fails open via the dedupe'd OSError handler."""
    cached = _cached_path()
    if cached:
        return cached
    expanded = os.path.expanduser(configured_path)
    # No tirith build for this platform: cache the verdict; the spawn fails open once, then
    # the fast path above short-circuits.
    if configured_path == "tirith" and not is_platform_supported():
        _set_failed("unsupported_platform")
        return expanded
    found, may_install = _resolve_locally(configured_path, warn_missing=True)
    if found or not may_install:
        return found or expanded
    # A background install is running: don't start a parallel one; fail-open until it finishes.
    if (_install_thread is not None and _install_thread.is_alive()) or _disk_marker_blocks_install():
        return expanded
    installed, reason = _install_tirith()
    _record_install_result(installed, reason)
    return installed or expanded


def _background_install(*, log_failures: bool = True):
    """Background thread target: download and install tirith."""
    with _install_lock:
        if _resolved_path is not None:  # another thread resolved meanwhile
            return
        found = _find_local_tirith()  # may have been installed by another process
        if found:
            _set_resolved(found)
            return
        _record_install_result(*_install_tirith(log_failures=log_failures))


def ensure_installed(*, log_failures: bool = True):
    """Ensure tirith is available, downloading in a daemon thread if needed. Local checks
    are synchronous; the download never blocks startup. Returns the resolved path if
    available now, else None. Safe to call repeatedly."""
    global _install_thread
    cfg = _load_security_config()
    if not cfg["tirith_enabled"]:
        return None
    cached = _cached_path()
    if cached:
        return cached if _is_executable(cached) else None
    # No tirith build here (e.g. Windows): stay silent -- no PATH probe, no download thread,
    # no disk marker. Pattern-matching guards still run.
    if not is_platform_supported():
        _set_failed("unsupported_platform")
        return None
    found, may_install = _resolve_locally(cfg["tirith_path"], warn_missing=False)
    if found or not may_install or _disk_marker_blocks_install():
        return found
    if _install_thread is None or not _install_thread.is_alive():
        _install_thread = threading.Thread(
            target=_background_install, kwargs={"log_failures": log_failures}, daemon=True
        )
        _install_thread.start()
    return None  # not available yet; commands fail-open until ready


# --- Main API ---

_MAX_FINDINGS = 50
_MAX_SUMMARY_LEN = 500
_EXIT_ACTIONS = {0: "allow", 1: "block", 2: "warn"}
# Summary when tirith's JSON is unparseable and only the exit code is known.
_NO_DETAILS_SUMMARY = {
    "block": "security issue detected (details unavailable)",
    "warn": "security warning detected (details unavailable)",
}


def _verdict(action: str, summary: str = "", findings: list | None = None) -> dict:
    return {"action": action, "findings": [] if findings is None else findings, "summary": summary}


def _fail(fail_open: bool, open_summary: str, closed_summary: str) -> dict:
    return _verdict("allow", open_summary) if fail_open else _verdict("block", closed_summary)


def check_command_security(command: str) -> dict:
    """Run the tirith scan on a command -> ``{"action": allow|warn|block, "findings", "summary"}``.
    Exit code determines the action; JSON enriches. Spawn failures/timeouts respect fail_open."""
    global _crash_count
    cfg = _load_security_config()
    if not cfg["tirith_enabled"]:
        return _verdict("allow")
    if _circuit_open:
        return _verdict("allow", "tirith disabled (circuit breaker)")
    # No binary for this platform, ever: skip the resolver so we never spawn.
    if not is_platform_supported():
        return _verdict("allow")
    tirith_path = _resolve_tirith_path(cfg["tirith_path"])
    timeout = cfg["tirith_timeout"]
    fail_open = cfg["tirith_fail_open"]
    if tirith_path is None:
        _warn_once("tirith_path_none", "tirith path resolved to None; scanning disabled")
        return _fail(fail_open, "tirith path unavailable", "tirith path unavailable (fail-closed)")
    try:
        result = subprocess.run(
            [tirith_path, "check", "--json", "--non-interactive", "--shell", "posix", "--", command],
            capture_output=True, text=True, encoding='utf-8', errors='replace',
            timeout=timeout, stdin=subprocess.DEVNULL,
        )
    except OSError as exc:
        # FileNotFoundError / PermissionError / exec format error: dedupe by (class, errno)
        # so each failure mode surfaces once, not per command.
        _warn_once(f"tirith_spawn_failed:{type(exc).__name__}:{getattr(exc, 'errno', '')}", "tirith spawn failed: %s", exc)
        _record_tirith_crash()
        return _fail(fail_open, f"tirith unavailable: {exc}", f"tirith spawn failed (fail-closed): {exc}")
    except subprocess.TimeoutExpired:
        _warn_once(f"tirith_timeout:{timeout}", "tirith timed out after %ds", timeout)
        _record_tirith_crash()
        return _fail(fail_open, f"tirith timed out ({timeout}s)", "tirith timed out (fail-closed)")

    exit_code = result.returncode
    action = _EXIT_ACTIONS.get(exit_code)
    if action is None:
        # Unknown exit code (includes signal-killed, e.g. -11): respect fail_open.
        logger.warning("tirith returned unexpected exit code %d", exit_code)
        _record_tirith_crash()
        return _fail(fail_open, f"tirith exit code {exit_code} (fail-open)", f"tirith exit code {exit_code} (fail-closed)")
    if action == "allow":
        _crash_count = 0  # successful execution resets the circuit breaker
    # JSON enriches findings/summary; a parse failure never changes the verdict.
    findings = []
    summary = ""
    try:
        data = json.loads(result.stdout) if result.stdout.strip() else {}
        findings = data.get("findings", [])[:_MAX_FINDINGS]
        summary = (data.get("summary", "") or "")[:_MAX_SUMMARY_LEN]
    except (json.JSONDecodeError, AttributeError):
        logger.debug("tirith JSON parse failed, using exit code only")
        summary = _NO_DETAILS_SUMMARY.get(action, "")
    # .app is a legitimate gTLD: a warn consisting solely of lookalike_tld findings for .app is a
    # known false positive and is downgraded to allow. Any other finding keeps the warn.
    if action == "warn" and findings and all(_is_app_tld_finding(f) for f in findings):
        return _verdict("allow")
    return _verdict(action, summary, findings)


def _is_app_tld_finding(finding: dict) -> bool:
    """True if this finding is a lookalike_tld warning for the .app TLD only."""
    if not isinstance(finding, dict) or finding.get("rule_id") != "lookalike_tld":
        return False
    return any(
        val is not None and ".app" in str(val).lower()
        for val in (finding.get(k) for k in ("value", "tld", "detail", "description", "message"))
    )
