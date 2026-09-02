"""`hermes computer-use doctor` — thin client for cua-driver's `health_report` MCP tool.

cua-driver owns the health model; we drive the stdio JSON-RPC handshake, call
`health_report` and render the stable ``schema_version="1"`` payload. cua-driver 0.10.x
marks `health_report` risk-unclassified (isError=true, structuredContent ``{"exit_code": 1}``)
— we detect that and synthesize a composite report from working probes
(check_permissions, list_apps, CLI --version).

Exit codes: 0 overall=="ok"; 1 degraded/failed; 2 binary missing / protocol error.
"""

from __future__ import annotations

import json
import platform as _platform_mod
import re
import subprocess
import sys
from contextlib import contextmanager, suppress
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

from hermes_cli._subprocess_compat import windows_hide_flags
from tools.computer_use.permissions import _child_env as _sanitized_cua_env

# Match the ALLOWED_STATUS_VALUES + ALLOWED_OVERALL_VALUES the cua-driver
# integration test pins. If health_report widens its vocabulary, add here.
_STATUS_GLYPH = {"pass": "✅", "fail": "❌", "skip": "⏭️"}
_OVERALL_GLYPH = {"ok": "✅", "degraded": "⚠️", "failed": "❌"}
_SUPPORTED_PLATFORMS = ("darwin", "linux", "windows")
_TCC_HINT = "Grant {} to CuaDriver in System Settings → Privacy & Security."
_ZERO_DISPLAY_MSG = "ScreenCaptureKit reachable but 0 shareable display(s) — every capture will return 0x0."
_ZERO_DISPLAY_HINT = ("Wake the built-in display, connect a monitor or HDMI dummy dongle (e.g. Headless Ghost), "
                      "or enable a virtual display (Screen Sharing/VNC, BetterDisplay). "
                      "Verify with `system_profiler SPDisplaysDataType`.")
Report = Dict[str, Any]


class HealthReportUnavailable(RuntimeError):
    """health_report denied or non-schema payload — ``run_doctor`` falls back to probes."""


# ── CLI probes ───────────────────────────────────────────────────────────────

def _run_cli(binary: str, *args: str, timeout: float) -> subprocess.CompletedProcess:
    """Run ``<binary> args`` with UTF-8 capture + sanitized env (raises on failure)."""
    return subprocess.run([binary, *args], capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=timeout, env=_sanitized_cua_env())

def _read_cli_version(binary: str, *, timeout: float = 5.0) -> Optional[str]:
    """First line of ``cua-driver --version`` or None. health_report's ``driver_version``
    can disagree with the real binary (seen on Windows); doctor surfaces both."""
    try:
        completed = _run_cli(binary, "--version", timeout=timeout)
    except (OSError, subprocess.TimeoutExpired, ValueError, TypeError):
        return None
    text = (completed.stdout or completed.stderr or "").strip()
    return text.splitlines()[0].strip() if text else None

def _cli_driver_version(binary: str, timeout: float = 5.0) -> Tuple[str, Optional[str]]:
    """Return (status, version_or_message) from ``cua-driver --version``."""
    try:
        completed = _run_cli(binary, "--version", timeout=timeout)
    except (OSError, subprocess.TimeoutExpired) as e:
        return "fail", f"--version failed: {e}"
    text = ((completed.stdout or "") + (completed.stderr or "")).strip()
    if completed.returncode != 0 and not text:
        return "fail", f"--version exited {completed.returncode}"
    m = re.search(r"(\d+\.\d+\.\d+(?:[-+][\w.]+)?)", text)  # typical: "cua-driver 0.10.0"
    version = m.group(1) if m else (text.splitlines()[0] if text else "unknown")
    return ("fail" if completed.returncode != 0 else "pass"), version

def _cli_doctor_snippet(binary: str, timeout: float = 8.0) -> Optional[str]:
    """Optional one-shot ``cua-driver doctor`` text (best-effort, never fatal)."""
    try:
        completed = _run_cli(binary, "doctor", timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return ((completed.stdout or "") + (completed.stderr or "")).strip() or None

def _build_identity(binary: str, report: Report) -> Report:
    """Hermes-side identity block comparing resolved binary vs health_report."""
    def token(text: str) -> str:  # dotted version-ish token out of a free-form string
        m = text and re.search(r"(\d+\.\d+(?:\.\d+)?(?:[-+][\w.]+)?)", text)
        return m.group(1) if m else text.strip().lower()

    cli = _read_cli_version(binary) or ""
    report_v = str(report.get("driver_version") or "")
    cli_tok, report_tok = token(cli), token(report_v)
    return {"resolved_binary": binary, "cli_version": cli or None, "health_report_driver_version": report_v or None,
            "version_mismatch": bool(cli_tok and report_tok and cli_tok != report_tok)}


# ── MCP transport ────────────────────────────────────────────────────────────

def _is_valid_health_report(payload: Any) -> bool:
    """True when *payload* looks like a schema_version=1 health_report."""
    return (isinstance(payload, dict) and "schema_version" in payload
            and "overall" in payload and isinstance(payload.get("checks"), list))

def _text_items(result: Report) -> Iterator[str]:
    """Text of every ``{"type": "text"}`` content item of an MCP tools/call result."""
    for item in result.get("content") or []:
        if isinstance(item, dict) and item.get("type") == "text":
            yield item.get("text") or ""

def _first_text(result: Report, default: str) -> str:
    """First non-empty text content item, else *default*."""
    return next((t.strip() for t in _text_items(result) if t.strip()), default)

def _extract_health_report_from_result(result: Report) -> Report:
    """Pull a schema_version=1 report out of an MCP tools/call result.

    Raises ``HealthReportUnavailable`` when the tool denied the call (isError) or the
    payload is not a real report (0.10's ``{"exit_code": 1}``); ``RuntimeError`` when
    the response carries no content at all.
    """
    if result.get("isError") is True:
        raise HealthReportUnavailable(_first_text(result, "health_report returned isError=true"))
    sc = result.get("structuredContent")
    if _is_valid_health_report(sc):
        return sc  # type: ignore[return-value]
    for text in _text_items(result):  # older builds: JSON text block with schema_version
        with suppress(ValueError, TypeError):
            parsed = json.loads(text)
            if _is_valid_health_report(parsed):
                return parsed
    if isinstance(sc, dict):  # present but not a real report — unavailable, not fatal
        raise HealthReportUnavailable("health_report structuredContent lacks schema_version/overall/checks "
                                      f"(keys={sorted(sc.keys())})")
    raise RuntimeError("health_report response carried neither structuredContent nor a parseable "
                       f"JSON text block. Result keys: {list(result.keys())}")

def _open_mcp(binary: str) -> subprocess.Popen:
    """Spawn ``<binary> mcp``. cua-driver emits UTF-8 (emoji, arbitrary paths); the
    locale default (`cp1252` on Windows) would raise UnicodeDecodeError, so pin the codec."""
    return subprocess.Popen([binary, "mcp"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, encoding="utf-8", errors="replace",
                            bufsize=1, creationflags=windows_hide_flags(), env=_sanitized_cua_env())

def _stderr_tail(proc: subprocess.Popen) -> List[str]:
    """Last 3 stderr lines of *proc* (best-effort, ``[]`` when unreadable)."""
    with suppress(Exception):
        if proc.stderr is not None:
            return [str(x) for x in (proc.stderr.read() or "").strip().splitlines()[-3:]]
    return []

def _mcp_rpc(proc: subprocess.Popen, msg_id: int, method: str, params: Any = None) -> Report:
    """Write one JSON-RPC request and read one response line."""
    assert proc.stdin is not None and proc.stdout is not None
    payload: Report = {"jsonrpc": "2.0", "id": msg_id, "method": method}
    if params is not None:
        payload["params"] = params
    proc.stdin.write(json.dumps(payload) + "\n")
    proc.stdin.flush()
    line = proc.stdout.readline()
    if not line:
        raise RuntimeError(f"cua-driver mcp produced no response for {method!r}. "
                           f"stderr tail: {_stderr_tail(proc) or '(empty)'}")
    try:
        resp = json.loads(line)
    except (ValueError, TypeError) as e:
        raise RuntimeError(f"{method} response was not valid JSON: {e}\nraw: {line[:200]}")
    if "error" in resp:
        raise RuntimeError(f"{method} JSON-RPC error: {resp['error']}")
    return resp

def _call_tool(proc: subprocess.Popen, msg_id: int, name: str, arguments: Any = None) -> Any:
    """tools/call *name* and return the raw ``result`` value (``{}`` when absent)."""
    return _mcp_rpc(proc, msg_id, "tools/call", {"name": name, "arguments": arguments or {}}).get("result") or {}

@contextmanager
def _mcp_session(binary: str, timeout: float) -> Iterator[subprocess.Popen]:
    """Spawn ``<binary> mcp`` and always close stdin / wait / kill it on exit."""
    proc = _open_mcp(binary)
    try:
        yield proc
    finally:
        with suppress(Exception):
            if proc.stdin is not None:
                proc.stdin.close()
        try:
            proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()

def _drive_health_report(binary: str, *, include: Sequence[str] = (), skip: Sequence[str] = (),
                         timeout: float = 12.0) -> Report:
    """Handshake + `health_report` → parsed report. Raises HealthReportUnavailable
    (denied / non-schema — caller falls back) or RuntimeError (protocol failure)."""
    args = {k: list(v) for k, v in (("include", include), ("skip", skip)) if v}
    with _mcp_session(binary, timeout) as proc:
        _mcp_rpc(proc, 1, "initialize", {})
        result = _call_tool(proc, 2, "health_report", args)
    if not isinstance(result, dict):
        raise RuntimeError(f"health_report result was not an object: {type(result).__name__}")
    return _extract_health_report_from_result(result)


# ── 0.10 fallback: compose a report from working probes ──────────────────────

def _probe_tool(proc: subprocess.Popen, msg_id: int, name: str) -> Tuple[Optional[Report], Optional[str]]:
    """``(result, None)`` on success; ``(None, error_text)`` on isError or RPC failure."""
    try:
        result = _call_tool(proc, msg_id, name)
    except RuntimeError as e:
        return None, str(e)
    if result.get("isError") is True:
        return None, _first_text(result, f"{name} isError")
    return result, None

def _drive_fallback_probes(binary: str, *, timeout: float = 12.0) -> Report:
    """Call working MCP tools (check_permissions, list_apps) in one session.

    Returns init_version (initialize serverInfo), permissions (structuredContent
    dict | None), permissions_error, list_apps_ok, list_apps_error, list_apps_count.
    """
    out: Report = dict.fromkeys(("init_version", "permissions", "permissions_error",
                                 "list_apps_ok", "list_apps_error", "list_apps_count"))
    with _mcp_session(binary, timeout) as proc:
        init_resp = _mcp_rpc(proc, 1, "initialize", {})
        server_info = ((init_resp.get("result") or {}).get("serverInfo") or {})
        if isinstance(server_info, dict):
            out["init_version"] = server_info.get("version")
        perms, err = _probe_tool(proc, 2, "check_permissions")  # primary TCC signal on 0.10
        if perms is None:
            out["permissions_error"] = err
        else:
            sc = perms.get("structuredContent")
            out["permissions"] = sc if isinstance(sc, dict) else {}
        # list_apps — light AX capability probe; text-only success still counts as AX working
        apps, err = _probe_tool(proc, 3, "list_apps")
        out["list_apps_ok"] = apps is not None
        if apps is None:
            out["list_apps_error"] = err
        else:
            sc = apps.get("structuredContent") or {}
            app_list = sc.get("apps") if isinstance(sc, dict) else None
            out["list_apps_count"] = len(app_list) if isinstance(app_list, list) else None
    return out

def _platform_name() -> str:
    sysname = (_platform_mod.system() or "").lower()
    return sysname if sysname in _SUPPORTED_PLATFORMS else (sysname or "unknown")

def _check(name: str, status: str, message: str, **extra: Any) -> Report:
    """Build one health check dict (``hint`` / ``data`` only when given)."""
    return {"name": name, "status": status, "message": message, **extra}

def _tcc_checks(perms: Optional[Report], perm_err: Optional[str], plat: str) -> List[Report]:
    """tcc_accessibility + tcc_screen_recording checks from check_permissions output."""
    if perms is None:
        status, msg = ("fail" if perm_err else "skip"), perm_err or "check_permissions unavailable"
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
    scr_rows = {  # (scr, capturable is False) — the granted-but-not-capturable row wins first.
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
    return [_check("tcc_accessibility", ax_status, ax_msg, **ax_extra),
            _check("tcc_screen_recording", scr_status, scr_msg, **scr_extra)]

def _ax_capability_check(probes: Report, ax_granted: bool) -> Report:
    """ax_capability — inferred from list_apps success or the accessibility grant."""
    list_ok, list_count = probes.get("list_apps_ok"), probes.get("list_apps_count")
    if list_ok is True:
        count_msg = f" ({list_count} apps)" if isinstance(list_count, int) else ""
        return _check("ax_capability", "pass", f"list_apps succeeded{count_msg}")
    if list_ok is False:
        default = "list_apps failed despite accessibility grant" if ax_granted else "list_apps failed"
        return _check("ax_capability", "fail", probes.get("list_apps_error") or default)
    if ax_granted:
        return _check("ax_capability", "pass", "inferred from accessibility grant (list_apps not probed)")
    return _check("ax_capability", "skip", "not probed")

def _overall_from(checks: List[Report]) -> str:
    """failed if binary missing/bad; ok if accessibility fine and nothing failed;
    otherwise degraded (screen recording or accessibility problems)."""
    by_name = {c.get("name"): c.get("status") for c in checks}
    if by_name.get("binary_version") != "pass":
        return "failed"
    ax_ok = by_name.get("tcc_accessibility") in ("pass", "skip", None)
    return "ok" if ax_ok and not any(c.get("status") == "fail" for c in checks) else "degraded"

def _compose_fallback_report(binary: str, *, reason: str = "", timeout: float = 12.0) -> Report:
    """Build a schema_version=1 report from CLI + working MCP probes when
    ``health_report`` is denied (0.10) or non-schema. Renders via ``_print_text_report``."""
    plat = _platform_name()
    ver_status, ver_value = _cli_driver_version(binary)
    probes = _drive_fallback_probes(binary, timeout=timeout)
    if probes.get("init_version"):  # MCP initialize version beats a messy CLI parse
        ver_status, driver_version = "pass", str(probes["init_version"])
        ver_msg = f"cua-driver {driver_version}"
    else:
        driver_version = ver_value if ver_status == "pass" else (ver_value or "?")
        ver_msg = f"cua-driver {ver_value}" if ver_status == "pass" else (ver_value or "version unknown")
    supported = plat in _SUPPORTED_PLATFORMS
    perms = probes.get("permissions") if isinstance(probes.get("permissions"), dict) else None
    reason_short = (reason or "health_report unavailable").strip()
    if len(reason_short) > 160:
        reason_short = reason_short[:157] + "..."
    checks: List[Report] = [
        _check("binary_version", ver_status, ver_msg),
        _check("platform_supported", "pass" if supported else "fail",
               f"platform={plat}" + ("" if supported else " (unsupported)")),
        # doctor does not start a session, so session_active is never probed
        _check("session_active", "skip", "not probed (doctor does not open a cua session)"),
        *_tcc_checks(perms, probes.get("permissions_error"), plat),
        _ax_capability_check(probes, bool(perms and perms.get("accessibility") is True)),
        _check("health_report_path", "skip",
               f"fallback composite (cua-driver 0.10 unclassified health_report); cause: {reason_short}"),
    ]
    doctor_txt = _cli_doctor_snippet(binary)  # optional CLI doctor text (best-effort)
    if doctor_txt:
        cli_ok = "[ok" in doctor_txt.lower() or "ok  ]" in doctor_txt
        checks.append(_check("cli_doctor", "pass" if cli_ok else "skip",
                             doctor_txt.splitlines()[0].strip(), data={"snippet": doctor_txt[:2000]}))
    for c in checks:  # normalize any accidental non-vocab status values
        if c.get("status") not in ("pass", "fail", "skip"):
            c["status"] = "fail"
    return {"schema_version": "1", "platform": plat, "driver_version": str(driver_version),
            "overall": _overall_from(checks), "checks": checks,
            "fallback": True, "fallback_reason": reason or "health_report unavailable"}

def _drive_health_report_or_fallback(binary: str, *, include: Sequence[str] = (), skip: Sequence[str] = (),
                                     timeout: float = 12.0) -> Report:
    """Prefer real health_report; on denial/non-schema, synthesize via probes."""
    try:
        report = _drive_health_report(binary, include=include, skip=skip, timeout=timeout)
    except HealthReportUnavailable as e:
        report = _compose_fallback_report(binary, reason=str(e), timeout=timeout)
    return _apply_display_count_guard(report)

def _apply_display_count_guard(report: Report) -> Report:
    """Downgrade an 'ok' report whose screen capture has zero displays.

    macOS ScreenCaptureKit reports ``display_count=0`` on headless Macs and when the
    built-in panel is asleep — TCC grants are fine, health_report can still say
    pass/ok, but every capture comes back 0x0. Failing the check turns a silent
    failure into an actionable one. Applied at the report seam so both the real and
    the composed fallback path get it.
    """
    checks = report.get("checks")
    for check in checks if isinstance(checks, list) else ():
        if not isinstance(check, dict) or check.get("name") != "screen_capture_capability":
            continue
        data = check.get("data")
        if (data.get("display_count") if isinstance(data, dict) else None) == 0 and check.get("status") == "pass":
            check.update(status="fail", message=_ZERO_DISPLAY_MSG, hint=_ZERO_DISPLAY_HINT)
            if report.get("overall") == "ok":
                report["overall"] = "degraded"
    return report


# ── Rendering ────────────────────────────────────────────────────────────────

def _check_lines(check: Report, status_cols: Dict[str, str], reset: str, dim: str) -> List[str]:
    """One line per check, plus indented hint and ``data`` rows (structured payload
    some checks attach — bundle id, AX state, version triple — support staff need it)."""
    status = check.get("status", "?")
    lines = [f"  {_STATUS_GLYPH.get(status, '•')} {status_cols.get(status, '')}{check.get('name', '?')}{reset}: "
             f"{check.get('message') or ''}"]
    if check.get("hint"):
        lines.append(f"      → {dim}{check['hint']}{reset}")
    data = check.get("data")
    for key, value in (data.items() if isinstance(data, dict) else ()):
        lines.append(f"      {dim}{key}={json.dumps(value) if isinstance(value, (dict, list)) else value}{reset}")
    return lines

def _print_text_report(report: Report, color: bool, *, identity: Optional[Report] = None) -> None:
    """Render the report like `cua-driver call health_report` (one line per check).
    With *identity* (resolved binary + ``--version``) the header prefers the CLI
    version over health_report's ``driver_version`` and prints an identity block."""
    platform, report_v, overall = (report.get(k, "?") for k in ("platform", "driver_version", "overall"))
    identity = identity or {}
    cli_v = identity.get("cli_version") or ""
    header_v = cli_v or report_v  # binary's own --version wins when health_report is stale
    # No external color library — inline ANSI keeps doctor self-contained.
    # Colors only apply when overall is a known vocabulary value.
    ansi = ("\033[31m", "\033[33m", "\033[32m", "\033[0m", "\033[2m")
    red, yellow, green, reset, dim = ansi if color and overall in _OVERALL_GLYPH else ("",) * 5
    col_for = {"failed": red, "degraded": yellow, "ok": green}.get(overall, "")
    status_cols = {"pass": green, "fail": red, "skip": dim}

    lines = [f"{_OVERALL_GLYPH.get(overall, '•')} cua-driver {header_v} on {platform} — {col_for}{overall}{reset}"]
    if identity.get("resolved_binary"):
        lines.append(f"  {dim}binary: {identity['resolved_binary']}{reset}")
    if cli_v and report_v and str(report_v) not in str(cli_v) and str(cli_v) not in str(report_v):
        # Only annotate when the free-form strings clearly differ.
        lines += [f"  {dim}--version: {cli_v}{reset}", f"  {dim}health_report.driver_version: {report_v}{reset}"]
    if identity.get("version_mismatch"):
        lines += [f"  {yellow}⚠️ version mismatch: health_report says {report_v!r} but binary --version is {cli_v!r}{reset}",
                  f"  {dim}→ trust --version / packages/current for debugging; health_report's binary_version check can lag on Windows{reset}"]
    for check in report.get("checks", []):
        lines += _check_lines(check, status_cols, reset, dim)
    print("\n".join(lines))

def run_doctor(driver_cmd: Optional[str] = None, *, include: Sequence[str] = (), skip: Sequence[str] = (),
               json_output: bool = False, color: Optional[bool] = None) -> int:
    """Resolve the cua-driver binary, call `health_report`, render the result.

    Honors `HERMES_CUA_DRIVER_CMD` via the shared runtime resolver, so doctor diagnoses
    what `computer_use` will actually invoke. On 0.10.x (health_report denied) it
    synthesizes a report from check_permissions / list_apps / CLI probes.
    """
    # Windows' locale codec (cp1252, cp936, ...) cannot encode the ✅ ❌ ⚠️ ⏭️ glyphs — force UTF-8.
    for stream in (sys.stdout, sys.stderr):
        with suppress(AttributeError, OSError):
            stream.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[union-attr]
    from tools.computer_use.cua_backend import resolve_cua_driver_cmd

    binary = resolve_cua_driver_cmd(driver_cmd)
    if not binary:
        print(f"cua-driver: not installed (looked for {driver_cmd or 'cua-driver (PATH and canonical install paths)'!r}).")
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
        json.dump({**report, "hermes_identity": identity}, sys.stdout, indent=2, sort_keys=True)
        sys.stdout.write("\n")
    else:
        _print_text_report(report, color=sys.stdout.isatty() if color is None else bool(color), identity=identity)
    return 0 if report.get("overall") == "ok" else 1  # unknown/missing overall must not look like success
