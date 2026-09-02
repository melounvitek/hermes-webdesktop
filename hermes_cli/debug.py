"""``hermes debug`` debug tools for Hermes Agent."""

import datetime
import gzip
import io
import json
import logging
import re
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from hermes_constants import get_hermes_home
from utils import atomic_replace

logger = logging.getLogger(__name__)

# Banner prepended to upload-bound log content when redaction is enabled.
# Visible in the public paste so reviewers know the content was sanitized.
# Kept short; the trailing newline guarantees the banner sits on its own line.
_REDACTION_BANNER = (
    "[hermes debug share: log content redacted at upload time. "
    "run with --no-redact to disable]\n"
)

_EMAIL_ADDRESS_RE = re.compile(
    r"(?<![A-Za-z0-9._%+-])"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    r"(?![A-Za-z0-9._%+-])"
)


# ---------------------------------------------------------------------------
# Paste services — try paste.rs first, dpaste.com as fallback.
# ---------------------------------------------------------------------------

_PASTE_RS_URL = "https://paste.rs/"
_DPASTE_COM_URL = "https://dpaste.com/api/"

# Maximum bytes to read from a single log file for upload.
# paste.rs caps at ~1 MB; we stay under that with headroom.
_MAX_LOG_BYTES = 512_000

# Auto-delete pastes after this many seconds (6 hours).
_AUTO_DELETE_SECONDS = 21600


# ---------------------------------------------------------------------------
# Pending-deletion tracking (replaces the old fork-and-sleep subprocess).
# ---------------------------------------------------------------------------

def _pending_file() -> Path:
    """Path to ``~/.hermes/pastes/pending.json``.

    Deletion is now driven by the gateway's cron ticker (``gateway/run.py::_start_cron_ticker``)
    which calls ``_sweep_expired_pastes`` once per hour. ``hermes debug share`` also runs an
    opportunistic sweep on entry as a fallback for CLI-only users who never start the gateway.
    """
    return get_hermes_home() / "pastes" / "pending.json"


def _load_pending() -> list[dict]:
    path = _pending_file()
    if not path.exists():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    if not isinstance(data, list):
        return []
    # Filter to well-formed entries only
    return [e for e in data if isinstance(e, dict) and "url" in e and "expire_at" in e]


def _save_pending(entries: list[dict]) -> None:
    path = _pending_file()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(entries, indent=2), encoding="utf-8")
        atomic_replace(tmp, path)
    except OSError:
        # Non-fatal — worst case the user has to run ``hermes debug delete``
        # manually.
        pass


def _sweep_expired_pastes(now: Optional[float] = None) -> tuple[int, int]:
    """Synchronously DELETE any pending pastes whose ``expire_at`` has passed.

    Returns ``(deleted, remaining)``. Best-effort: failed deletes stay in the pending file for
    the next sweep. Silent, since it runs on every ``hermes debug`` invocation.
    """
    entries = _load_pending()
    if not entries:
        return (0, 0)

    current = time.time() if now is None else now
    deleted = 0
    remaining: list[dict] = []

    for entry in entries:
        try:
            expire_at = float(entry.get("expire_at", 0))
        except (TypeError, ValueError):
            continue  # drop malformed entries
        if expire_at > current:
            remaining.append(entry)
            continue

        url = entry.get("url", "")
        try:
            if delete_paste(url):
                deleted += 1
                continue
        except Exception:
            # Network hiccup, 404 (already gone), etc. — drop the entry
            # after a grace period; don't retry forever.
            pass

        # Retain failed deletes for up to 24h past expiration, then give up.
        if expire_at + 86400 > current:
            remaining.append(entry)
        else:
            deleted += 1  # count as reaped (paste.rs will GC eventually)

    if deleted:
        _save_pending(remaining)

    return (deleted, len(remaining))


def _best_effort_sweep_expired_pastes() -> None:
    """Attempt pending-paste cleanup without letting /debug fail offline."""
    try:
        _sweep_expired_pastes()
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Privacy / delete helpers
# ---------------------------------------------------------------------------

_PRIVACY_NOTICE = """\
⚠️  This will upload system info + logs to a PUBLIC paste service.

Cryptographic secrets (API keys, tokens, passwords) are redacted before
upload, but the following personal data is NOT redacted and will be public:
  • Your display name and persistent platform user ID
  • Verbatim content of your recent messages (prompts, responses, tool output)
  • Local filesystem paths
  • Any other PII present in the logs

The resulting URL is public to anyone who has the link. Pastes auto-delete
after 6 hours, but may be archived by third parties in the meantime.

Use --local to view the report without uploading.
"""

_GATEWAY_PRIVACY_NOTICE = (
    "⚠️ **Privacy notice:** This uploads system info + recent log tails "
    "(may contain conversation fragments) to a public paste service. "
    "Full logs are NOT included from the gateway — use `hermes debug share` "
    "from the CLI for full log uploads.\n"
    "Pastes auto-delete after 6 hours."
)


def _extract_paste_id(url: str) -> Optional[str]:
    """Extract the paste ID from a paste.rs or dpaste.com URL."""
    url = url.strip().rstrip("/")
    for prefix in ("https://paste.rs/", "http://paste.rs/"):
        if url.startswith(prefix):
            return url[len(prefix):]
    return None


def delete_paste(url: str) -> bool:
    """Delete a paste from paste.rs. Returns True on success.

    Only paste.rs supports unauthenticated DELETE. dpaste.com pastes expire automatically but cannot
    be deleted via API.
    """
    paste_id = _extract_paste_id(url)
    if not paste_id:
        raise ValueError(
            f"Cannot delete: only paste.rs URLs are supported.  Got: {url}"
        )

    target = f"{_PASTE_RS_URL}{paste_id}"
    req = urllib.request.Request(
        target, method="DELETE",
        headers={"User-Agent": "hermes-agent/debug-share"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        return 200 <= resp.status < 300


def _schedule_auto_delete(urls: list[str], delay_seconds: int = _AUTO_DELETE_SECONDS):
    """Record *urls* for deletion ``delay_seconds`` from now.

    Only paste.rs URLs are recorded (dpaste.com auto-expires); entries are merged into any existing
    pending.json. The old fork-and-sleep subprocesses leaked ~20 MB of resident interpreter per
    ``hermes debug share``. This replacement is stateless: the gateway's cron ticker sweeps expired
    entries once per hour, ``hermes debug share`` runs an opportunistic sweep as a fallback for
    CLI-only users, and paste.rs's own retention policy is the last resort.
    """
    paste_rs_urls = [u for u in urls if _extract_paste_id(u)]
    if not paste_rs_urls:
        return

    # Dedupe by URL: keep the later expire_at if same URL appears twice
    by_url: dict[str, float] = {e["url"]: float(e["expire_at"]) for e in _load_pending()}
    expire_at = time.time() + delay_seconds
    for u in paste_rs_urls:
        by_url[u] = max(expire_at, by_url.get(u, 0.0))
    _save_pending([{"url": u, "expire_at": ts} for u, ts in by_url.items()])


def _post_paste(service: str, endpoint: str, body: bytes, content_type: str) -> str:
    """POST *body* to a paste service and return the paste URL it echoes back."""
    req = urllib.request.Request(
        endpoint, data=body, method="POST",
        headers={"Content-Type": content_type, "User-Agent": "hermes-agent/debug-share"},
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        url = resp.read().decode("utf-8").strip()
    if not url.startswith("http"):
        raise ValueError(f"Unexpected response from {service}: {url[:200]}")
    return url


def _upload_paste_rs(content: str) -> str:
    """Upload to paste.rs. Returns the paste URL."""
    return _post_paste("paste.rs", _PASTE_RS_URL, content.encode("utf-8"), "text/plain; charset=utf-8")


def _upload_dpaste_com(content: str, expiry_days: int = 7) -> str:
    """Upload to dpaste.com. Returns the paste URL."""
    boundary = "----HermesDebugBoundary9f3c"

    def _field(name: str, value: str) -> str:
        return (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{name}"\r\n'
            f"\r\n"
            f"{value}\r\n"
        )

    body = (
        _field("content", content)
        + _field("syntax", "text")
        + _field("expiry_days", str(expiry_days))
        + f"--{boundary}--\r\n"
    ).encode("utf-8")
    return _post_paste("dpaste.com", _DPASTE_COM_URL, body, f"multipart/form-data; boundary={boundary}")


def upload_to_pastebin(content: str, expiry_days: int = 7) -> str:
    """Upload *content* to a paste service, trying paste.rs then dpaste.com."""
    errors: list[str] = []

    # Try paste.rs first (simple, fast)
    try:
        return _upload_paste_rs(content)
    except Exception as exc:
        errors.append(f"paste.rs: {exc}")

    # Fallback: dpaste.com (supports expiry)
    try:
        return _upload_dpaste_com(content, expiry_days=expiry_days)
    except Exception as exc:
        errors.append(f"dpaste.com: {exc}")

    raise RuntimeError(
        "Failed to upload to any paste service:\n  " + "\n  ".join(errors)
    )


# ---------------------------------------------------------------------------
# Log file reading
# ---------------------------------------------------------------------------


@dataclass
class LogSnapshot:
    """Single-read snapshot of a log file used by debug-share."""

    path: Optional[Path]
    tail_text: str
    full_text: Optional[str]


def _primary_log_path(log_name: str) -> Optional[Path]:
    """Where *log_name* would live if present. Doesn't check existence."""
    from hermes_cli.logs import LOG_FILES

    filename = LOG_FILES.get(log_name)
    return (get_hermes_home() / "logs" / filename) if filename else None


# Logs written by a client process rather than by this backend. When the
# desktop app talks to a remote/docker/SSH backend, `hermes debug share` runs
# on the *backend* and can never see them — a bare "(file not found)" then
# reads as "the app logged nothing" and sends triage down a dead end, which is
# exactly the wrong answer when the client is the thing being debugged.
_CLIENT_SIDE_LOGS = {
    "desktop": (
        "written by Hermes Desktop on the machine running the app, not by this "
        "backend. If the desktop connects to a remote/docker/SSH backend, collect "
        "it on that client machine"
    ),
}


def _missing_log_note(log_name: str) -> str:
    """Explain a missing log instead of stating a bare absence."""
    reason = _CLIENT_SIDE_LOGS.get(log_name)
    if reason is None:
        return "(file not found)"

    primary = _primary_log_path(log_name)
    where = f" — expected at {primary}" if primary else ""
    return f"(not on this host: {reason}{where})"


def _resolve_log_path(log_name: str) -> Optional[Path]:
    """Find the log file for *log_name*, falling back to the .1 rotation.

    Returns the first non-empty candidate (primary, then .1), or None. Callers distinguish
    'empty primary' from 'truly missing' via :func:`_primary_log_path`.
    """
    primary = _primary_log_path(log_name)
    if primary is None:
        return None

    if primary.exists() and primary.stat().st_size > 0:
        return primary

    rotated = primary.parent / f"{primary.name}.1"
    if rotated.exists() and rotated.stat().st_size > 0:
        return rotated

    return None


def _redact_log_text(text: str) -> str:
    """Run ``redact_sensitive_text`` with ``force=True`` over upload-bound text.

    Uses ``force=True`` so redaction fires regardless of the operator's ``security.redact_secrets``
    setting. The local on-disk log file is not modified; only the in-memory copy headed for the
    public paste service is sanitized. Returns the redacted text (or the original when empty / non-
    string).
    """
    if not text:
        return text
    from agent.redact import redact_sensitive_text

    text = redact_sensitive_text(text, force=True)
    return _EMAIL_ADDRESS_RE.sub("[REDACTED_EMAIL]", text)


def _capture_log_snapshot(
    log_name: str,
    *,
    tail_lines: int,
    max_bytes: int = _MAX_LOG_BYTES,
    redact: bool = True,
) -> LogSnapshot:
    """Capture a log once and derive summary/full-log views from it.

    The report tail and standalone log upload must come from the same file snapshot. Otherwise a
    rotation/truncate between reads can make the report look newer than the uploaded ``agent.log``
    paste.

    When ``redact`` is True (the default), both ``tail_text`` and ``full_text`` are run through
    ``_redact_log_text`` so the snapshot returned is upload-safe. The on-disk log file is never
    modified. Pass ``redact=False`` to capture original log content (used by ``hermes debug share
    --no-redact``).
    """
    log_path = _resolve_log_path(log_name)
    if log_path is None:
        primary = _primary_log_path(log_name)
        tail = "(file empty)" if primary and primary.exists() else _missing_log_note(log_name)
        return LogSnapshot(path=None, tail_text=tail, full_text=None)

    try:
        size = log_path.stat().st_size
        if size == 0:
            # race: file was truncated between _resolve_log_path and stat
            return LogSnapshot(path=log_path, tail_text="(file empty)", full_text=None)

        with open(log_path, "rb") as f:
            if size <= max_bytes:
                raw = f.read()
                truncated = False
            else:
                # Read from the end until we have enough bytes for the
                # standalone upload and enough newline context to render the
                # summary tail from the same snapshot.
                chunk_size = 8192
                pos = size
                chunks: list[bytes] = []
                total = 0
                newline_count = 0

                while pos > 0 and (total < max_bytes or newline_count <= tail_lines + 1) and total < max_bytes * 2:
                    read_size = min(chunk_size, pos)
                    pos -= read_size
                    f.seek(pos)
                    chunk = f.read(read_size)
                    chunks.insert(0, chunk)
                    total += len(chunk)
                    newline_count += chunk.count(b"\n")
                    chunk_size = min(chunk_size * 2, 65536)

                raw = b"".join(chunks)
                truncated = pos > 0

        full_raw = raw
        if truncated and len(full_raw) > max_bytes:
            cut = len(full_raw) - max_bytes
            # Check whether the cut lands exactly on a line boundary.  If the
            # byte just before the cut position is a newline the first retained
            # byte starts a complete line and we should keep it.  Only drop a
            # partial first line when we're genuinely mid-line.
            on_boundary = cut > 0 and full_raw[cut - 1 : cut] == b"\n"
            full_raw = full_raw[cut:]
            if not on_boundary and b"\n" in full_raw:
                full_raw = full_raw.split(b"\n", 1)[1]

        all_text = raw.decode("utf-8", errors="replace")
        tail_text = "".join(all_text.splitlines(keepends=True)[-tail_lines:]).rstrip("\n")

        full_text = full_raw.decode("utf-8", errors="replace")
        if truncated:
            full_text = f"[... truncated — showing last ~{max_bytes // 1024}KB ...]\n{full_text}"

        if redact:
            tail_text = _redact_log_text(tail_text)
            full_text = _redact_log_text(full_text)

        return LogSnapshot(path=log_path, tail_text=tail_text, full_text=full_text)
    except Exception as exc:
        return LogSnapshot(path=log_path, tail_text=f"(error reading: {exc})", full_text=None)


# Logs the debug report tails, in output order. ``agent`` gets the full ``--lines`` budget;
# the rest are capped at 100 lines. Every log but ``errors`` is also uploaded in full.
_REPORT_LOGS = ("agent", "errors", "gateway", "gui", "desktop")
_FULL_LOGS = ("agent", "gateway", "gui", "desktop")


def _capture_default_log_snapshots(
    log_lines: int, *, redact: bool = True
) -> dict[str, LogSnapshot]:
    """Capture all logs used by debug-share exactly once."""
    errors_lines = min(log_lines, 100)
    return {
        name: _capture_log_snapshot(
            name, tail_lines=log_lines if name == "agent" else errors_lines, redact=redact
        )
        for name in _REPORT_LOGS
    }


# ---------------------------------------------------------------------------
# Debug report collection
# ---------------------------------------------------------------------------

def _capture_dump() -> str:
    """Run ``hermes dump`` and return its stdout as a string."""
    from hermes_cli.dump import run_dump

    class _FakeArgs:
        show_keys = False

    old_stdout = sys.stdout
    sys.stdout = capture = io.StringIO()
    try:
        run_dump(_FakeArgs())
    except SystemExit:
        pass
    finally:
        sys.stdout = old_stdout

    return capture.getvalue()


def collect_debug_report(
    *,
    log_lines: int = 200,
    dump_text: str = "",
    log_snapshots: Optional[dict[str, LogSnapshot]] = None,
) -> str:
    """Build the summary debug report: system dump + log tails.

    ``dump_text`` is pre-captured dump output; when empty, ``hermes dump`` is run internally.
    Returns plain text ready for upload.
    """
    buf = io.StringIO()

    if not dump_text:
        dump_text = _capture_dump()
    buf.write(dump_text)

    if log_snapshots is None:
        log_snapshots = _capture_default_log_snapshots(log_lines)

    # ── Sanitiser heal counters (#96870) ─────────────────────────────────
    # In-process, in-memory counters: populated when this report is built
    # inside a process that ran agent turns (gateway /debug share); empty
    # from a fresh CLI process, where the errors.log tail below carries the
    # same escalation lines instead.
    try:
        from agent.agent_runtime_helpers import get_sanitizer_heal_stats

        heal_stats = get_sanitizer_heal_stats()
        if heal_stats:
            buf.write("\n\n--- transcript sanitiser heal counters ---\n")
            for sess, st in sorted(heal_stats.items()):
                buf.write(
                    f"session {sess}: {st['heal_events']} heal events, "
                    f"{st['messages_healed']} messages healed, "
                    f"escalated={st['escalated']}\n"
                )
    except Exception:
        pass

    # ── Recent log tails (summary only) ──────────────────────────────────
    errors_lines = min(log_lines, 100)
    buf.write("\n")
    for name in _REPORT_LOGS:
        lines = log_lines if name == "agent" else errors_lines
        buf.write(f"\n--- {name}.log (last {lines} lines) ---\n")
        buf.write(log_snapshots[name].tail_text)
        buf.write("\n")

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Shared bundle collection (used by both the paste.rs and Nous-S3 paths)
# ---------------------------------------------------------------------------

# Bundle format identifier embedded in the Nous-S3 JSON envelope. The
# discord-support viewer keys off this string to parse the bundle.
_NOUS_BUNDLE_FORMAT = "hermes-debug-share/1"


def collect_share_bundle(
    log_lines: int = 200,
    redact: bool = True,
) -> dict[str, str]:
    """Collect the debug report + full logs as a label→text mapping.

    The dump header is prepended to each full log (mirroring the historical paste behaviour) so
    every file is self-contained, and the redaction banner is prepended when ``redact`` is True.
    """
    dump_text = _capture_dump()
    log_snapshots = _capture_default_log_snapshots(log_lines, redact=redact)

    report = collect_debug_report(
        log_lines=log_lines,
        dump_text=dump_text,
        log_snapshots=log_snapshots,
    )
    # Visible banner so reviewers know redaction was applied at upload time.
    banner = _REDACTION_BANNER if redact else ""
    bundle: dict[str, str] = {"report": banner + report}
    # Prepend dump header to each full log so every file is self-contained.
    for name in _FULL_LOGS:
        full = log_snapshots[name].full_text
        if full:
            bundle[f"{name}.log"] = banner + dump_text + f"\n\n--- full {name}.log ---\n" + full
    return bundle


def build_nous_bundle(bundle: dict[str, str], redact: bool = True) -> bytes:
    """Gzip-compress a :func:`collect_share_bundle` mapping into the Nous envelope.

    The JSON shape (``format: hermes-debug-share/1``, ``redacted``, ``created``, ``files``) is
    what the discord-support viewer parses — keep it stable.
    """
    created = datetime.datetime.now(datetime.timezone.utc).isoformat()
    envelope = {
        "format": _NOUS_BUNDLE_FORMAT,
        "redacted": bool(redact),
        "created": created,
        "files": bundle,
    }
    return gzip.compress(json.dumps(envelope).encode("utf-8"))


# ---------------------------------------------------------------------------
# CLI entry points
# ---------------------------------------------------------------------------

@dataclass
class DebugShareResult:
    """Structured outcome of a ``debug share`` upload.

    Returned by :func:`build_debug_share` so non-CLI callers (dashboard web server, gateway) can
    render the uploaded paste URLs as real links instead of scraping printed text.
    """

    urls: dict  # label -> paste URL (e.g. {"Report": "...", "agent.log": "..."})
    failures: list  # human-readable "label: error" strings for optional uploads
    redacted: bool  # whether force-mode redaction was applied before upload
    auto_delete_seconds: int  # how long until the pastes auto-delete
    report: str = ""  # the summary report text (kept for local fallback)


def build_debug_share(
    *,
    log_lines: int = 200,
    expiry: int = 7,
    redact: bool = True,
) -> DebugShareResult:
    """Collect the debug report + full logs, upload each, return the URLs.

    This is the shared core behind ``hermes debug share`` (CLI) and the dashboard ``POST
    /api/ops/debug-share`` endpoint. It performs blocking network I/O (paste uploads) — callers
    inside an event loop must run it in a worker thread.
    """
    _best_effort_sweep_expired_pastes()

    # Collect the report + full logs (force-redacted when redact=True) via the
    # shared collector so the paste.rs and Nous-S3 paths build identical,
    # identically-redacted bundles. The dump header + redaction banner are
    # applied inside collect_share_bundle.
    bundle = collect_share_bundle(log_lines=log_lines, redact=redact)

    if redact:
        logger.info(
            "hermes debug share: applied force-mode redaction to log snapshots before upload"
        )

    report = bundle["report"]

    urls: dict[str, str] = {}
    failures: list[str] = []

    # 1. Summary report (required — raises on failure so callers can fall back)
    urls["Report"] = upload_to_pastebin(report, expiry_days=expiry)

    # 2-5. Full logs (optional — failures are collected, not raised)
    for name in _FULL_LOGS:
        label = f"{name}.log"
        content = bundle.get(label)
        if not content:
            continue
        try:
            urls[label] = upload_to_pastebin(content, expiry_days=expiry)
        except Exception as exc:
            failures.append(f"{label}: {exc}")

    # Schedule auto-deletion after 6 hours.
    _schedule_auto_delete(list(urls.values()))

    return DebugShareResult(
        urls=urls,
        failures=failures,
        redacted=redact,
        auto_delete_seconds=_AUTO_DELETE_SECONDS,
        report=report,
    )


def _confirm_upload(args) -> bool:
    """Require explicit consent before any debug-share upload.

    The privacy notice is printed by the caller. This gates the actual upload: with ``--yes`` (or
    ``-y``) we proceed unprompted; otherwise we ask an interactive ``[y/N]`` question.
    """
    if bool(getattr(args, "yes", False)):
        return True
    if not sys.stdin.isatty():
        print(
            "ERROR: Non-interactive mode requires --yes to confirm upload.\n"
            "       This prevents accidental exposure of personal data.\n"
            "       Use --local to view the report without uploading.",
            file=sys.stderr,
        )
        sys.exit(1)
    try:
        answer = input("Upload debug report? [y/N] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        answer = ""
    if answer not in ("y", "yes"):
        print("Aborted.")
        return False
    return True


def run_debug_share(args):
    """Collect debug report + full logs, upload each, print URLs."""
    log_lines = getattr(args, "lines", 200)
    expiry = getattr(args, "expire", 7)
    local_only = getattr(args, "local", False)
    nous = getattr(args, "nous", False)
    redact = not getattr(args, "no_redact", False)

    if local_only:
        # Local-only path never uploads — render the report to stdout and bail
        # before any network I/O. Reuses the shared collector so the rendered
        # output matches exactly what would be uploaded.
        _best_effort_sweep_expired_pastes()
        print("Collecting debug report...")
        bundle = collect_share_bundle(log_lines=log_lines, redact=redact)
        print(bundle["report"])
        for name in _FULL_LOGS:
            body = bundle.get(f"{name}.log")
            if body:
                print(f"\n\n{'=' * 60}\nFULL {name}.log\n{'=' * 60}\n")
                print(body)
        return

    if nous:
        _run_debug_share_nous(args, log_lines=log_lines, redact=redact)
        return

    print(_PRIVACY_NOTICE)
    if not _confirm_upload(args):
        return
    print("Collecting debug report...")
    print("Uploading...")

    try:
        result = build_debug_share(
            log_lines=log_lines,
            expiry=expiry,
            redact=redact,
        )
    except RuntimeError as exc:
        print(f"\nUpload failed: {exc}", file=sys.stderr)
        print("\nRun `hermes debug share --local` to print the report instead.\n")
        sys.exit(1)

    # Print results
    label_width = max(len(k) for k in result.urls)
    print("\nDebug report uploaded:")
    for label, url in result.urls.items():
        print(f"  {label:<{label_width}}  {url}")

    if result.failures:
        print(f"\n  (failed to upload: {', '.join(result.failures)})")

    hours = result.auto_delete_seconds // 3600
    print(f"\n⏱  Pastes will auto-delete in {hours} hours.")

    # Manual delete fallback
    print("To delete now:  hermes debug delete <url>")

    print("\nShare these links with the Hermes team for support.")


_NOUS_PRIVACY_NOTICE = """\
⚠️  --nous: This uploads your debug bundle to Nous-INTERNAL storage (AWS S3),
    NOT a public paste service. The following is included:
  • System info (OS, Python/Hermes version, provider, which API keys are
    configured — NOT the actual keys)
  • Full agent.log, gateway.log, and desktop.log (up to 512 KB each — likely
    contains conversation content, tool outputs, and file paths)

  • The bundle is viewable only by Nous staff (and allowlisted Discord mods)
    via a Google-login-gated viewer.
  • It is NOT a public paste — there is no public URL to the contents.
  • It auto-deletes after 14 days.
"""


def _run_debug_share_nous(args, *, log_lines: int, redact: bool) -> None:
    """Handle ``hermes debug share --nous``: upload the bundle to Nous-S3.

    Collects the same force-redacted bundle as the paste path, gzips it into the Nous envelope,
    requests a signed URL from NAS, uploads, and prints the private viewer link. On any failure
    falls back to a clear error that suggests ``--local``.
    """
    from hermes_cli.diagnostics_upload import share_to_nous

    print(_NOUS_PRIVACY_NOTICE)
    if not _confirm_upload(args):
        return
    if not redact:
        print(
            "⚠️  --no-redact is set: secrets in your logs will NOT be redacted "
            "before upload.\n"
        )
    print("Collecting debug report...")
    _best_effort_sweep_expired_pastes()

    bundle = collect_share_bundle(log_lines=log_lines, redact=redact)
    if redact:
        logger.info(
            "hermes debug share --nous: applied force-mode redaction before upload"
        )
    blob = build_nous_bundle(bundle, redact=redact)

    print("Uploading to Nous diagnostics storage...")
    try:
        res = share_to_nous(blob)
    except Exception as exc:
        print(
            f"\nNous upload failed: {exc}\n"
            "\nThe Nous diagnostics service may be unavailable or not yet "
            "provisioned.\n"
            "Run `hermes debug share --local` to print the report instead, "
            "or `hermes debug share` to upload to a public paste service.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    view_url = res.get("viewUrl") or res.get("view_url")
    print("\nDebug bundle uploaded to Nous (private):")
    if view_url:
        print(f"  View URL  {view_url}")
    else:
        print(f"  (no view URL returned; upload id: {res.get('id', '?')})")

    expires_at = res.get("expiresAt") or res.get("expires_at")
    if expires_at:
        print(f"\n⏱  Auto-deletes at {expires_at} (14-day retention).")
    else:
        print("\n⏱  Auto-deletes after 14 days.")

    print(
        "\nShare this private link with the Nous team — only Nous staff "
        "(via Google login) can open it."
    )
    print(
        "\nPick up the discussion in:\n"
        "  GitHub Issues        https://github.com/NousResearch/hermes-agent/issues\n"
        "  Nous Portal Support  https://portal.nousresearch.com/help\n"
        "  Discord              https://discord.gg/NousResearch"
    )


def run_debug_delete(args):
    """Delete one or more paste URLs uploaded by /debug."""
    urls = getattr(args, "urls", [])
    if not urls:
        print("Usage: hermes debug delete <url> [<url> ...]")
        print("  Deletes paste.rs pastes uploaded by 'hermes debug share'.")
        return

    for url in urls:
        try:
            if delete_paste(url):
                print(f"  ✓ Deleted: {url}")
            else:
                print(f"  ✗ Failed to delete: {url} (unexpected response)")
        except ValueError as exc:
            print(f"  ✗ {exc}")
        except Exception as exc:
            print(f"  ✗ Could not delete {url}: {exc}")


def run_debug(args):
    """Route debug subcommands."""
    # Opportunistic sweep of expired pastes on every ``hermes debug`` call.
    # Replaces the old per-paste sleeping subprocess that used to leak as
    # one orphaned Python interpreter per scheduled deletion.
    _best_effort_sweep_expired_pastes()

    subcmd = getattr(args, "debug_command", None)
    if subcmd == "share":
        run_debug_share(args)
    elif subcmd == "delete":
        run_debug_delete(args)
    else:
        print(_DEBUG_USAGE)


_DEBUG_USAGE = """\
Usage: hermes debug <command>

Commands:
  share    Upload debug report to a paste service and print URL
  delete   Delete a previously uploaded paste

Options (share):
  --lines N    Number of log lines to include (default: 200)
  --expire N   Paste expiry in days (default: 7)
  --local      Print report locally instead of uploading
  --nous       Upload to Nous-internal storage (private, staff-only,
               auto-deletes in 14 days) instead of a public paste
  --no-redact  Disable upload-time secret redaction (default: redact)

Options (delete):
  <url> ...    One or more paste URLs to delete"""
