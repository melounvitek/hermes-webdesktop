"""`hermes computer-use doctor` — thin client for cua-driver's `health_report` MCP tool.

cua-driver owns the health model; this module drives the stdio JSON-RPC handshake,
calls `health_report`, and renders the response. The only contract is the stable
`schema_version="1"` payload shape. cua-driver 0.10.x marks `health_report`
risk-unclassified (isError=true, structuredContent ``{"exit_code": 1}``) — we detect
that and synthesize a composite report from working probes (check_permissions,
list_apps, CLI --version).

Exit codes: 0 overall=="ok"; 1 degraded/failed; 2 binary missing / protocol error.
"""

from __future__ import annotations

import json
import os
import platform as _platform_mod
import re
import subprocess
import sys
from contextlib import contextmanager
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from hermes_cli._subprocess_compat import windows_hide_flags


# Match the ALLOWED_STATUS_VALUES + ALLOWED_OVERALL_VALUES the cua-driver
# integration test pins. If health_report widens its vocabulary, add here.
_STATUS_GLYPH = {"pass": "✅", "fail": "❌", "skip": "⏭️"}
_OVERALL_GLYPH = {"ok": "✅", "degraded": "⚠️", "failed": "❌"}
_SUPPORTED_PLATFORMS = ("darwin", "linux", "windows")
_TCC_HINT = "Grant {} to CuaDriver in System Settings → Privacy & Security."


class HealthReportUnavailable(RuntimeError):
    """health_report denied or non-schema payload — ``run_doctor`` falls back to probes."""


def _sanitized_cua_env() -> Dict[str, str]:
    """cua-driver child env (telemetry policy + provider secrets stripped);
    degrades to ``os.environ`` on import error so doctor keeps working."""
    try:
        from tools.computer_use.cua_backend import sanitized_cua_driver_env

        return sanitized_cua_driver_env()
    except Exception:
        return dict(os.environ)


def _is_valid_health_report(payload: Any) -> bool:
    """True when *payload* looks like a schema_version=1 health_report."""
    return (
        isinstance(payload, dict)
        and "schema_version" in payload
        and "overall" in payload
        and isinstance(payload.get("checks"), list)
    )


def _run_cli(binary: str, *args: str, timeout: float) -> subprocess.CompletedProcess:
    """Run ``<binary> args`` with UTF-8 capture + sanitized env (raises on failure)."""
    return subprocess.run(
        [binary, *args], capture_output=True, text=True, encoding="utf-8",
        errors="replace", timeout=timeout, env=_sanitized_cua_env(),
    )


def _first_text(result: Dict[str, Any], default: str) -> str:
    """First non-empty text content item of an MCP tools/call result, else *default*."""
    for item in result.get("content") or []:
        if isinstance(item, dict) and item.get("type") == "text":
            text = (item.get("text") or "").strip()
            if text:
                return text
    return default


def _read_cli_version(binary: str, *, timeout: float = 5.0) -> Optional[str]:
    """Return ``cua-driver --version`` first line (stripped), or None on failure.

    health_report's ``driver_version`` can disagree with the actual binary
    (observed on Windows); doctor surfaces both so operators are not misled.
    """
    try:
        completed = _run_cli(binary, "--version", timeout=timeout)
    except (OSError, subprocess.TimeoutExpired, ValueError, TypeError):
        return None
    text = (completed.stdout or completed.stderr or "").strip()
    return text.splitlines()[0].strip() if text else None


def _normalize_version_token(text: str) -> str:
    """Pull a dotted version-ish token out of a free-form version string."""
    if not text:
        return ""
    m = re.search(r"(\d+\.\d+(?:\.\d+)?(?:[-+][\w.]+)?)", text)
    return m.group(1) if m else text.strip().lower()


def _build_identity(binary: str, report: Dict[str, Any]) -> Dict[str, Any]:
    """Hermes-side identity block comparing resolved binary vs health_report."""
    cli = _read_cli_version(binary) or ""
    report_v = str(report.get("driver_version") or "")
    cli_tok = _normalize_version_token(cli)
    report_tok = _normalize_version_token(report_v)
    return {
        "resolved_binary": binary,
        "cli_version": cli or None,
        "health_report_driver_version": report_v or None,
        "version_mismatch": bool(cli_tok and report_tok and cli_tok != report_tok),
    }


def _extract_health_report_from_result(result: Dict[str, Any]) -> Dict[str, Any]:
    """Pull a schema_version=1 report out of an MCP tools/call result.

    Raises ``HealthReportUnavailable`` when the tool denied the call (isError)
    or the payload is not a real health report (0.10's ``{"exit_code": 1}``);
    ``RuntimeError`` when the response carries no content at all.
    """
    if result.get("isError") is True:
        raise HealthReportUnavailable(_first_text(result, "health_report returned isError=true"))

    sc = result.get("structuredContent")
    if _is_valid_health_report(sc):
        return sc  # type: ignore[return-value]

    # Older builds: JSON text block with schema_version.
    for item in result.get("content") or []:
        if not isinstance(item, dict) or item.get("type") != "text":
            continue
        try:
            parsed = json.loads(item.get("text", ""))
        except (ValueError, TypeError):
            continue
        if _is_valid_health_report(parsed):
            return parsed

    # structuredContent present but not a real report — unavailable, not fatal protocol.
    if isinstance(sc, dict):
        raise HealthReportUnavailable(
            "health_report structuredContent lacks schema_version/overall/checks "
            f"(keys={sorted(sc.keys())})"
        )
    raise RuntimeError(
        "health_report response carried neither structuredContent nor a parseable "
        f"JSON text block. Result keys: {list(result.keys())}"
    )


def _open_mcp(binary: str) -> subprocess.Popen:
    """Spawn ``<binary> mcp`` with UTF-8 + sanitized env.

    cua-driver emits UTF-8 (emoji, arbitrary paths); the locale default
    (`cp1252` on Windows) would raise UnicodeDecodeError, so pin the codec.
    """
    return subprocess.Popen(
        [binary, "mcp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, encoding="utf-8", errors="replace", bufsize=1,
        creationflags=windows_hide_flags(), env=_sanitized_cua_env(),
    )


def _mcp_rpc(proc: subprocess.Popen, msg_id: int, method: str, params: Any = None) -> Dict[str, Any]:
    """Write one JSON-RPC request and read one response line."""
    assert proc.stdin is not None and proc.stdout is not None
    payload: Dict[str, Any] = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        payload["params"] = params
    proc.stdin.write(json.dumps(payload) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        stderr_tail: List[str] = []
        if proc.stderr is not None:
            try:
                raw_err = proc.stderr.read() or ""
                stderr_tail = [str(x) for x in raw_err.strip().splitlines()[-3:]]
            except Exception:
                pass
        raise RuntimeError(
            f"cua-driver mcp produced no response for {method!r}. "
            f"stderr tail: {stderr_tail or '(empty)'}"
        )
    try:
        resp = json.loads(line)
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"{method} response was not valid JSON: {e}\nraw: {line[:200]}")
    if "error" in resp:
        raise RuntimeError(f"{method} JSON-RPC error: {resp['error']}")
    return resp


def _call_tool(proc: subprocess.Popen, msg_id: int, name: str, arguments: Any = None) -> Any:
    """tools/call *name* and return the raw ``result`` value (``{}`` when absent)."""
    resp = _mcp_rpc(proc, msg_id, "tools/call", {"name": name, "arguments": arguments or {}})
    return resp.get("result") or {}


@contextmanager
def _mcp_session(binary: str, timeout: float) -> Iterator[subprocess.Popen]:
    """Spawn ``<binary> mcp`` and always close stdin / wait / kill it on exit."""
    proc = _open_mcp(binary)
    try:
        yield proc
    finally:
        try:
            if proc.stdin is not None:
                proc.stdin.close()
        except Exception:
            pass
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def _drive_health_report(binary: str, *, include: Sequence[str] = (), skip: Sequence[str] = (),
                         timeout: float = 12.0) -> Dict[str, Any]:
    """Spawn `<binary> mcp`, handshake, call `health_report`, return the parsed report.

    Raises HealthReportUnavailable (denied / non-schema — caller falls back) or
    RuntimeError (protocol-level failure).
    """
    args = {k: list(v) for k, v in (("include", include), ("skip", skip)) if v}
    with _mcp_session(binary, timeout) as proc:
        _mcp_rpc(proc, 1, "initialize", {})
        result = _call_tool(proc, 2, "health_report", args)
    if not isinstance(result, dict):
        raise RuntimeError(f"health_report result was not an object: {type(result).__name__}")
    return _extract_health_report_from_result(result)


def _cli_driver_version(binary: str, timeout: float = 5.0) -> Tuple[str, Optional[str]]:
    """Return (status, version_or_message) from ``cua-driver --version``."""
    try:
        completed = _run_cli(binary, "--version", timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return "fail", f"--version failed: {e}"

    text = ((completed.stdout or "") + (completed.stderr or "")).strip()
    if completed.returncode != 0 and not text:
        return "fail", f"--version exited {completed.returncode}"

    # Typical: "cua-driver 0.10.0"
    m = re.search(r"(\d+\.\d+\.\d+(?:[-+][\w.]+)?)", text)
    version = m.group(1) if m else (text.splitlines()[0] if text else "unknown")
    return ("fail" if completed.returncode != 0 else "pass"), version


def _cli_doctor_snippet(binary: str, timeout: float = 8.0) -> Optional[str]:
    """Optional one-shot ``cua-driver doctor`` text (best-effort, never fatal)."""
    try:
        completed = _run_cli(binary, "doctor", timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return ((completed.stdout or "") + (completed.stderr or "")).strip() or None


def _drive_fallback_probes(binary: str, *, timeout: float = 12.0) -> Dict[str, Any]:
    """Call working MCP tools (check_permissions, list_apps) in one session.

    Returns init_version (initialize serverInfo), permissions (structuredContent
    dict | None), permissions_error, list_apps_ok, list_apps_error, list_apps_count.
    """
    out: Dict[str, Any] = dict.fromkeys((
        "init_version", "permissions", "permissions_error",
        "list_apps_ok", "list_apps_error", "list_apps_count",
    ))
    with _mcp_session(binary, timeout) as proc:
        init_resp = _mcp_rpc(proc, 1, "initialize", {})
        server_info = ((init_resp.get("result") or {}).get("serverInfo") or {})
        if isinstance(server_info, dict):
            out["init_version"] = server_info.get("version")

        # check_permissions — primary TCC signal on 0.10
        try:
            perm_result = _call_tool(proc, 2, "check_permissions")
            if perm_result.get("isError") is True:
                out["permissions_error"] = _first_text(perm_result, "check_permissions isError")
            else:
                sc = perm_result.get("structuredContent")
                out["permissions"] = sc if isinstance(sc, dict) else {}
        except RuntimeError as e:
            out["permissions_error"] = str(e)

        # list_apps — light AX capability probe; text-only success still counts as AX working
        try:
            apps_result = _call_tool(proc, 3, "list_apps")
            if apps_result.get("isError") is True:
                out["list_apps_ok"] = False
                out["list_apps_error"] = _first_text(apps_result, "list_apps isError")
            else:
                sc = apps_result.get("structuredContent") or {}
                apps = sc.get("apps") if isinstance(sc, dict) else None
                out["list_apps_ok"] = True
                out["list_apps_count"] = len(apps) if isinstance(apps, list) else None
        except RuntimeError as e:
            out["list_apps_ok"] = False
            out["list_apps_error"] = str(e)
    return out


def _platform_name() -> str:
    sysname = (_platform_mod.system() or "").lower()
    return sysname if sysname in _SUPPORTED_PLATFORMS else (sysname or "unknown")


def _check(name: str, status: str, message: str, **extra: Any) -> Dict[str, Any]:
    """Build one health check dict (``hint`` / ``data`` only when given)."""
    return {"name": name, "status": status, "message": message, **extra}


def _tcc_checks(perms: Optional[Dict[str, Any]], perm_err: Optional[str], plat: str) -> List[Dict[str, Any]]:
    """tcc_accessibility + tcc_screen_recording checks from check_permissions output."""
    if perms is None:
        status = "fail" if perm_err else "skip"
        msg = perm_err or "check_permissions unavailable"
        return [_check("tcc_accessibility", status, msg), _check("tcc_screen_recording", status, msg)]

    # Only real booleans select a branch; anything else (missing/odd) is the "absent" row.
    ax, scr, capturable = (perms.get(k) for k in ("accessibility", "screen_recording", "screen_recording_capturable"))
    ax = ax if isinstance(ax, bool) else None
    scr = scr if isinstance(scr, bool) else None

    ax_rows = {
        True: ("pass", "Accessibility is granted.", {"data": {"accessibility": True}}),
        False: ("fail", "Accessibility is not granted.",
                {"hint": _TCC_HINT.format("Accessibility"), "data": {"accessibility": False}}),
        None: ("skip", "accessibility field absent from check_permissions", {}),
    }
    scr_rows = {
        # (scr, capturable is False) — the granted-but-not-capturable row wins first.
        (True, True): ("fail", "Screen Recording granted but not capturable.",
                       {"hint": "Screen Recording permission may need a restart of CuaDriver "
                                "or a re-grant in System Settings.",
                        "data": {"screen_recording": True, "screen_recording_capturable": False}}),
        (True, False): ("pass", "Screen Recording is granted.",
                        {"data": {"screen_recording": True, "screen_recording_capturable": capturable}}),
        (False, False): ("fail", "Screen Recording is not granted.",
                         {"hint": _TCC_HINT.format("Screen Recording"), "data": {"screen_recording": False}}),
        (None, False): ("skip", "screen_recording field absent from check_permissions" if plat == "darwin"
                        else f"not applicable on {plat}", {}),
    }
    ax_status, ax_msg, ax_extra = ax_rows[ax]
    scr_status, scr_msg, scr_extra = scr_rows[(scr, capturable is False and scr is True)]
    return [
        _check("tcc_accessibility", ax_status, ax_msg, **ax_extra),
        _check("tcc_screen_recording", scr_status, scr_msg, **scr_extra),
    ]


def _ax_capability_check(probes: Dict[str, Any], ax_granted: bool) -> Dict[str, Any]:
    """ax_capability — inferred from list_apps success or the accessibility grant."""
    list_ok = probes.get("list_apps_ok")
    list_count = probes.get("list_apps_count")
    if list_ok is True:
        count_msg = f" ({list_count} apps)" if isinstance(list_count, int) else ""
        return _check("ax_capability", "pass", f"list_apps succeeded{count_msg}")
    if list_ok is False:
        default = "list_apps failed despite accessibility grant" if ax_granted else "list_apps failed"
        return _check("ax_capability", "fail", probes.get("list_apps_error") or default)
    if ax_granted:
        return _check("ax_capability", "pass", "inferred from accessibility grant (list_apps not probed)")
    return _check("ax_capability", "skip", "not probed")


def _compose_fallback_report(binary: str, *, reason: str = "", timeout: float = 12.0) -> Dict[str, Any]:
    """Build a schema_version=1 report from CLI + working MCP probes.

    Used when ``health_report`` is denied (unclassified risk on 0.10) or
    returns a non-schema payload. Compatible with ``_print_text_report``.
    """
    plat = _platform_name()

    ver_status, ver_value = _cli_driver_version(binary)
    driver_version = ver_value if ver_status == "pass" else (ver_value or "?")
    # Prefer MCP initialize version when CLI parse is messy
    probes = _drive_fallback_probes(binary, timeout=timeout)
    if probes.get("init_version"):
        driver_version = str(probes["init_version"])
        ver_status = "pass"
        ver_msg = f"cua-driver {driver_version}"
    else:
        ver_msg = f"cua-driver {ver_value}" if ver_status == "pass" else (ver_value or "version unknown")

    supported = plat in _SUPPORTED_PLATFORMS
    checks: List[Dict[str, Any]] = [
        _check("binary_version", ver_status, ver_msg),
        _check("platform_supported", "pass" if supported else "fail",
               f"platform={plat}" + ("" if supported else " (unsupported)")),
        # doctor does not start a session, so session_active is never probed
        _check("session_active", "skip", "not probed (doctor does not open a cua session)"),
    ]

    perms = probes.get("permissions") if isinstance(probes.get("permissions"), dict) else None
    checks += _tcc_checks(perms, probes.get("permissions_error"), plat)
    checks.append(_ax_capability_check(probes, bool(perms and perms.get("accessibility") is True)))

    # Annotate that we used the fallback path
    reason_short = (reason or "health_report unavailable").strip()
    if len(reason_short) > 160:
        reason_short = reason_short[:157] + "..."
    checks.append(_check(
        "health_report_path", "skip",
        f"fallback composite (cua-driver 0.10 unclassified health_report); cause: {reason_short}",
    ))

    # Optional CLI doctor text (best-effort)
    doctor_txt = _cli_doctor_snippet(binary)
    if doctor_txt:
        cli_ok = "[ok" in doctor_txt.lower() or "ok  ]" in doctor_txt
        checks.append(_check("cli_doctor", "pass" if cli_ok else "skip",
                             doctor_txt.splitlines()[0].strip(), data={"snippet": doctor_txt[:2000]}))

    # Normalize any accidental non-vocab status values
    for c in checks:
        if c.get("status") not in ("pass", "fail", "skip"):
            c["status"] = "fail"

    # overall: failed if binary missing/bad; ok if accessibility fine and nothing
    # failed; otherwise degraded (screen recording or accessibility problems).
    status_by_name = {c.get("name"): c.get("status") for c in checks}
    fail_count = sum(1 for c in checks if c.get("status") == "fail")
    if status_by_name.get("binary_version") != "pass":
        overall = "failed"
    elif status_by_name.get("tcc_accessibility") in ("pass", "skip", None) and fail_count == 0:
        overall = "ok"
    else:
        overall = "degraded"

    return {
        "schema_version": "1",
        "platform": plat,
        "driver_version": str(driver_version),
        "overall": overall,
        "checks": checks,
        "fallback": True,
        "fallback_reason": reason or "health_report unavailable",
    }


def _drive_health_report_or_fallback(binary: str, *, include: Sequence[str] = (), skip: Sequence[str] = (),
                                     timeout: float = 12.0) -> Dict[str, Any]:
    """Prefer real health_report; on denial/non-schema, synthesize via probes."""
    try:
        report = _drive_health_report(binary, include=include, skip=skip, timeout=timeout)
    except HealthReportUnavailable as e:
        report = _compose_fallback_report(binary, reason=str(e), timeout=timeout)
    return _apply_display_count_guard(report)


def _apply_display_count_guard(report: Dict[str, Any]) -> Dict[str, Any]:
    """Downgrade an 'ok' report whose screen capture has zero displays.

    macOS ScreenCaptureKit reports ``display_count=0`` on headless Macs and
    when the built-in panel is asleep — TCC grants are fine, health_report
    can still say pass/ok, but every capture will come back 0x0. Marking the
    check failed turns a silent failure into an actionable one. Applied at
    the report seam so both the real and the composed fallback path get it.
    """
    checks = report.get("checks")
    if not isinstance(checks, list):
        return report
    for check in checks:
        if not isinstance(check, dict) or check.get("name") != "screen_capture_capability":
            continue
        data = check.get("data")
        count = data.get("display_count") if isinstance(data, dict) else None
        if count == 0 and check.get("status") == "pass":
            check["status"] = "fail"
            check["message"] = "ScreenCaptureKit reachable but 0 shareable display(s) — every capture will return 0x0."
            check["hint"] = (
                "Wake the built-in display, connect a monitor or HDMI dummy dongle (e.g. Headless Ghost), "
                "or enable a virtual display (Screen Sharing/VNC, BetterDisplay). "
                "Verify with `system_profiler SPDisplaysDataType`."
            )
            if report.get("overall") == "ok":
                report["overall"] = "degraded"
    return report


def _print_text_report(report: Dict[str, Any], color: bool, *, identity: Optional[Dict[str, Any]] = None) -> None:
    """Render the report like `cua-driver call health_report` (one line per check).

    With *identity* (resolved binary + ``--version``) the header prefers the CLI
    version over health_report's ``driver_version`` and prints an identity block.
    """
    platform = report.get("platform", "?")
    report_v = report.get("driver_version", "?")
    overall = report.get("overall", "?")
    identity = identity or {}
    cli_v = identity.get("cli_version") or ""
    mismatch = bool(identity.get("version_mismatch"))
    header_v = cli_v or report_v  # binary's own --version wins when health_report is stale

    # No external color library — keep ANSI inline so doctor stays self-contained.
    # Colors only apply when overall is a known vocabulary value.
    ansi = ("\033[31m", "\033[33m", "\033[32m", "\033[0m", "\033[2m")
    red, yellow, green, reset, dim = ansi if color and overall in _OVERALL_GLYPH else ("",) * 5
    col_for = {"failed": red, "degraded": yellow, "ok": green}.get(overall, "")
    status_cols = {"pass": green, "fail": red, "skip": dim}

    print(f"{_OVERALL_GLYPH.get(overall, '•')} cua-driver {header_v} on {platform} — {col_for}{overall}{reset}")
    if identity.get("resolved_binary"):
        print(f"  {dim}binary: {identity['resolved_binary']}{reset}")
    if cli_v and report_v and str(report_v) not in str(cli_v) and str(cli_v) not in str(report_v):
        # Only annotate when the free-form strings clearly differ.
        print(f"  {dim}--version: {cli_v}{reset}")
        print(f"  {dim}health_report.driver_version: {report_v}{reset}")
    if mismatch:
        print(f"  {yellow}⚠️ version mismatch: health_report says {report_v!r} but binary --version is {cli_v!r}{reset}")
        print(f"  {dim}→ trust --version / packages/current for debugging; health_report's binary_version check can lag on Windows{reset}")

    for check in report.get("checks", []):
        name = check.get("name", "?")
        status = check.get("status", "?")
        glyph = _STATUS_GLYPH.get(status, "•")
        message = check.get("message") or ""
        print(f"  {glyph} {status_cols.get(status, '')}{name}{reset}: {message}")
        hint = check.get("hint")
        if hint:
            print(f"      → {dim}{hint}{reset}")
        # `data` is the structured payload some checks attach (bundle id, AX
        # state, version triple) — users / support staff frequently need it.
        data = check.get("data")
        if isinstance(data, dict) and data:
            for key, value in data.items():
                rendered = json.dumps(value) if isinstance(value, (dict, list)) else value
                print(f"      {dim}{key}={rendered}{reset}")


def run_doctor(driver_cmd: Optional[str] = None, *, include: Sequence[str] = (), skip: Sequence[str] = (),
               json_output: bool = False, color: Optional[bool] = None) -> int:
    """Resolve the cua-driver binary, call `health_report`, render the result.

    Honors `HERMES_CUA_DRIVER_CMD` via the shared runtime resolver, so doctor
    diagnoses what `computer_use` will actually invoke. On 0.10.x (health_report
    denied) it synthesizes a report from check_permissions / list_apps / CLI probes.
    """
    # Windows' locale codec (cp1252, cp936, ...) cannot encode the ✅ ❌ ⚠️ ⏭️ glyphs — force UTF-8.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
        except (AttributeError, OSError):
            pass
    from tools.computer_use.cua_backend import resolve_cua_driver_cmd

    binary = resolve_cua_driver_cmd(driver_cmd)
    if not binary:
        looked_for = driver_cmd or "cua-driver (PATH and canonical install paths)"
        print(f"cua-driver: not installed (looked for {looked_for!r}).")
        print("  Run: hermes computer-use install")
        return 2

    try:
        report = _drive_health_report_or_fallback(binary, include=include, skip=skip)
    except RuntimeError as e:
        print(f"cua-driver health_report failed: {e}", file=sys.stderr)
        return 2

    identity = _build_identity(binary, report)

    if json_output:
        # Additive envelope: upstream health_report keys preserved, Hermes identity
        # under hermes_identity so parsers that only read overall/checks keep working.
        payload = dict(report)
        payload["hermes_identity"] = identity
        json.dump(payload, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        if color is None:
            color = sys.stdout.isatty()
        _print_text_report(report, color=bool(color), identity=identity)

    # Unknown / missing overall after fallback must not look like success.
    return 0 if report.get("overall") == "ok" else 1
