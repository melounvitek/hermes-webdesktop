"""Shared utility functions for hermes-agent."""

import errno
import json
import logging
import os
import shutil
import stat
import tempfile
import time
from pathlib import Path
from typing import Any, Union
from urllib.parse import urlparse

import yaml

logger = logging.getLogger(__name__)


TRUTHY_STRINGS = frozenset({"1", "true", "yes", "on"})


def is_truthy_value(value: Any, default: bool = False) -> bool:
    """Coerce bool-ish values using the project's shared truthy string set."""
    if value is None:
        return default
    if isinstance(value, str):
        return value.strip().lower() in TRUTHY_STRINGS
    return bool(value)


def env_var_enabled(name: str, default: str = "") -> bool:
    """Return True when an environment variable is set to a truthy value."""
    return is_truthy_value(os.getenv(name, default), default=False)


def _preserve_file_mode(path: Path) -> "int | None":
    """Capture the permission bits of *path* if it exists, else ``None``."""
    try:
        return stat.S_IMODE(path.stat().st_mode) if path.exists() else None
    except OSError:
        return None


def _preserve_file_owner(path: Path) -> "tuple[int, int] | None":
    """Capture the owning uid/gid of *path* if the platform supports it."""
    try:
        st = path.stat() if os.name == "posix" else None
    except OSError:
        return None
    return (st.st_uid, st.st_gid) if st else None


def _restore_file_metadata(path: Path, owner: "tuple[int, int] | None", mode: "int | None") -> None:
    """Best-effort re-apply of uid/gid and permission bits after an atomic replace.

    Docker/NAS installs often run some commands as root while the volume is owned by the runtime
    user; ``os.replace`` swaps in the temp file's owner, leaving ``config.yaml`` root-owned, so
    privileged callers chown it back (harmless otherwise). ``tempfile.mkstemp`` creates files 0o600;
    without re-applying *mode* the target would inherit that and break volume mounts relying on
    broader permissions.
    """
    if owner is not None and hasattr(os, "chown"):
        try:
            os.chown(path, owner[0], owner[1])
        except OSError:
            pass
    if mode is not None:
        try:
            os.chmod(path, mode)
        except OSError:
            pass


def _restore_file_owner(path: Path, owner: "tuple[int, int] | None") -> None:
    _restore_file_metadata(path, owner, None)


def _restore_file_mode(path: Path, mode: "int | None") -> None:
    _restore_file_metadata(path, None, mode)


_IS_WINDOWS = os.name == "nt"

# Windows rename failures that can be caused by another handle on the target
# rather than by a permission problem.  ``os.replace`` onto a file that any
# other handle has open is denied because CPython opens files without
# ``FILE_SHARE_DELETE``:
#
#   5  ERROR_ACCESS_DENIED    — what a held *target* handle actually reports
#   32 ERROR_SHARING_VIOLATION — reported when the *source* temp file is held
#   33 ERROR_LOCK_VIOLATION    — byte-range lock on the target
#
# Measured on Windows 11 (build 26200, CPython 3.11): a plain reader on the
# target — in-process or cross-process — yields winerror 5, NOT 32.  Keying
# recovery on 32 alone therefore misses every real occurrence of this bug.
# These codes are ambiguous (a genuine ACL denial is also 5), which is why
# recovery is bounded and any still-failing write is re-raised unchanged
# rather than being classified up front.
_WINDOWS_CONTENDED_REPLACE_ERRORS = frozenset({5, 32, 33})

# Retry budget for the atomic rename.  A rename that wins here keeps the write
# fully atomic, so the budget is sized to cover a realistic contended hold: an
# observed desktop auth-init holds auth.json past 100 ms, while an ordinary
# status read is ~0.05 ms.  Measured on Windows 11 build 26200, this recovers
# holds up to ~200 ms atomically (~310 ms worst case).
#
# The cap matters as much as the attempt count.  gateway_state.json is
# rewritten at every turn boundary, so a permanently-held target pays the full
# budget on every write: a longer 6 x 20..400 ms budget cost ~1.3 s per write
# under a persistent reader, versus ~0.3 s here for the same atomic coverage.
# Jittered so concurrent writers don't retry in lockstep.
_REPLACE_RETRY_ATTEMPTS = 4
_REPLACE_RETRY_BASE_DELAY_S = 0.02
_REPLACE_RETRY_MAX_DELAY_S = 0.1


def _is_contended_windows_replace_error(exc: OSError) -> bool:
    """Return True for Windows rename failures a retry might clear.

    Only a *candidate* classification: ``ERROR_ACCESS_DENIED`` covers both a concurrent handle and a
    real ACL denial, and the two are not reliably distinguishable up front.
    """
    return _IS_WINDOWS and getattr(exc, "winerror", None) in _WINDOWS_CONTENDED_REPLACE_ERRORS


def _rewrite_in_place(tmp_str: str, real_path: str) -> None:
    """Overwrite *real_path* with the contents of *tmp_str*, in place.

    Last-resort path for a target whose handle is still held after the retry budget: writing through
    the existing file works where renaming onto it does not.

    This is still not atomic — it is a strictly smaller window than a copy, not the absence of one —
    so it runs only after the rename has genuinely failed. Writing through the target also preserves
    its ACL, which ``os.replace`` does not (the temp file's inherited ACL wins there).
    """
    with open(tmp_str, "rb") as src:
        data = src.read()
    flags = os.O_WRONLY | getattr(os, "O_BINARY", 0)
    fd = os.open(real_path, flags)
    try:
        written = 0
        while written < len(data):
            written += os.write(fd, data[written:])
        os.ftruncate(fd, len(data))
        try:
            os.fsync(fd)
        except OSError:
            pass
    finally:
        os.close(fd)
    os.unlink(tmp_str)


def _copy_fallback(tmp_str: str, real_path: str) -> None:
    """Copy/fsync/unlink fallback for cross-device and bind-mount renames."""
    shutil.copyfile(tmp_str, real_path)
    try:
        shutil.copystat(tmp_str, real_path)
    except OSError:
        pass
    try:
        with open(real_path, "rb") as f:
            os.fsync(f.fileno())
    except OSError:
        pass
    os.unlink(tmp_str)


def atomic_replace(tmp_path: Union[str, Path], target: Union[str, Path]) -> str:
    """Atomically move *tmp_path* onto *target*, preserving symlinks.

    This helper resolves the symlink first so ``os.replace`` writes to the real file in-place while
    the symlink survives. For non-symlink and non-existent paths the behavior is identical to a
    plain ``os.replace`` call unless the rename fails with:

    * ``EXDEV`` / ``EBUSY`` (any platform) — cross-device, bind-mount, and busy-file deployments
    fall back to copy/fsync/unlink immediately. These never clear on retry. * A Windows rename
    contended by another open handle (winerror 5/32/33).
    """
    target_str = str(target)
    real_path = os.path.realpath(target_str) if os.path.islink(target_str) else target_str
    tmp_str = str(tmp_path)
    try:
        os.replace(tmp_str, real_path)
        return real_path
    except OSError as exc:
        contended = _is_contended_windows_replace_error(exc)
        if exc.errno not in (errno.EXDEV, errno.EBUSY) and not contended:
            raise
        if contended:
            # Lazy import: keeps ``utils`` free of a package-level dependency
            # on ``agent`` for every consumer that never hits this path.
            from agent.retry_utils import jittered_backoff

            for attempt in range(1, _REPLACE_RETRY_ATTEMPTS + 1):
                time.sleep(jittered_backoff(
                    attempt, base_delay=_REPLACE_RETRY_BASE_DELAY_S, max_delay=_REPLACE_RETRY_MAX_DELAY_S
                ))
                try:
                    os.replace(tmp_str, real_path)
                    return real_path
                except OSError as retry_exc:
                    if retry_exc.errno in (errno.EXDEV, errno.EBUSY):
                        # Not contention after all — stop burning the budget.
                        exc = retry_exc
                        contended = False
                        break
                    if not _is_contended_windows_replace_error(retry_exc):
                        raise
                    exc = retry_exc
        logger.debug(
            "atomic_replace: %s -> %s failed with %s; falling back to %s",
            tmp_str, real_path,
            getattr(exc, "winerror", None) or errno.errorcode.get(exc.errno or 0, exc.errno),
            "in-place rewrite" if contended else "copy",
        )
        if contended:
            # Re-raises the rewrite's own error (not the rename's) when the
            # target is genuinely unwritable — an ACL denial stays an ACL
            # denial rather than being reported as contention.
            _rewrite_in_place(tmp_str, real_path)
        else:
            _copy_fallback(tmp_str, real_path)
    return real_path


def _atomic_write(
    path: Path,
    write,
    *,
    prefix: str,
    encoding: str = "utf-8",
    mode: "int | None" = None,
    preserve_owner: bool = True,
) -> None:
    """Temp file + fsync + :func:`atomic_replace`, then re-apply owner/mode.

    *write(f)* emits the payload into the open text handle. *mode* (when not ``None``) is fchmod'd
    onto the temp fd BEFORE the replace so the target never transits through mkstemp's 0600 (fchmod
    is Unix-only; the post-replace chmod is the sole path on Windows and harmless elsewhere). The
    temp file is removed on any failure — ``BaseException`` on purpose, so KeyboardInterrupt /
    SystemExit still clean up before re-raising.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    original_owner = _preserve_file_owner(path) if preserve_owner else None
    fd, tmp_path = tempfile.mkstemp(dir=str(path.parent), prefix=prefix, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding=encoding) as f:
            if mode is not None and hasattr(os, "fchmod"):
                os.fchmod(f.fileno(), mode)
            write(f)
            f.flush()
            os.fsync(f.fileno())
        # Preserve symlinks — swap in-place on the real file (GitHub #16743).
        _restore_file_metadata(Path(atomic_replace(tmp_path, path)), original_owner, mode)
    except BaseException:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def _mode_for_write(
    path: Path, create_mode: "int | None", preserve: bool = True
) -> "int | None":
    """Existing permission bits of *path* (when *preserve*), else *create_mode* for a new file."""
    mode = _preserve_file_mode(path) if preserve else None
    if mode is None and create_mode is not None and not path.exists():
        mode = create_mode
    return mode


def atomic_write_text(
    path: Union[str, Path],
    content: str,
    *,
    encoding: str = "utf-8",
    tmp_prefix: str = ".tmp_",
    preserve_mode: bool = False,
    create_mode: "int | None" = None,
) -> None:
    """Write *content* to *path* via temp file + fsync + atomic rename.

    Ensures the target file is never left in a partially-written state if the process crashes or is
    interrupted. ``atomic_replace`` preserves symlinks and handles cross-device / busy-file
    fallbacks.

    Used by the memory store, skill manager, and agent importer so that every destructive file
    rewrite in the codebase shares one implementation.
    """
    path = Path(path)
    _atomic_write(
        path,
        lambda f: f.write(content),
        prefix=tmp_prefix,
        encoding=encoding,
        mode=_mode_for_write(path, create_mode, preserve=preserve_mode),
        preserve_owner=preserve_mode,
    )


def atomic_json_write(
    path: Union[str, Path],
    data: Any,
    *,
    indent: int = 2,
    mode: int | None = None,
    **dump_kwargs: Any,
) -> None:
    """Write JSON data to a file atomically.

    Uses temp file + fsync + os.replace to ensure the target file is never left in a partially-
    written state. If the process crashes mid-write, the previous version of the file remains
    intact.
    """
    path = Path(path)
    _atomic_write(
        path,
        lambda f: json.dump(data, f, indent=indent, ensure_ascii=False, **dump_kwargs),
        prefix=f".{path.stem}_",
        mode=mode if mode is not None else _preserve_file_mode(path),
    )


def warn_if_credential_file_broadly_readable(
    path: Union[str, Path],
    *,
    label: str = "",
    log: logging.Logger | None = None,
) -> bool:
    """Warn (once per call) when a credential file is group/world-readable.

    Secret-bearing files that users create by hand (or that older Hermes versions wrote without an
    explicit mode) commonly end up 0o644 under the default umask. This helper is the shared read-
    time check for that class: call it before loading any token/credential file so the owner gets a
    remediation hint in the logs.

    Returns True when a warning was emitted. No-ops (returns False) on platforms without POSIX
    permission bits semantics (best effort), when the file is missing, or when permissions are
    already tight.
    """
    p = Path(path)
    try:
        file_mode = p.stat().st_mode
    except OSError:
        return False
    # Windows ACLs don't map onto POSIX group/other bits; st_mode there is synthesized.
    if os.name != "posix" or not (file_mode & (stat.S_IRGRP | stat.S_IROTH)):
        return False
    (log or logger).warning(
        "%s%s is group/world-readable (mode 0%o) and contains secrets. "
        "Run: chmod 600 %s",
        f"{label} " if label else "",
        p.name,
        stat.S_IMODE(file_mode),
        p,
    )
    return True


class IndentDumper(yaml.SafeDumper):
    """PyYAML dumper that indents list items under mapping keys (2-space).

    Default PyYAML emits "indentless" sequences while ``ruamel.yaml`` (used by
    :func:`atomic_roundtrip_yaml_update`) indents them; mixing both in one ``config.yaml`` makes
    stricter parsers like ``js-yaml`` reject it. Forcing ``indentless=False`` keeps every write
    path byte-identical.
    """

    def increase_indent(self, flow=False, indentless=False):  # noqa: ARG002
        return super().increase_indent(flow, False)


def atomic_yaml_write(
    path: Union[str, Path],
    data: Any,
    *,
    default_flow_style: bool = False,
    sort_keys: bool = False,
    extra_content: str | None = None,
    create_mode: "int | None" = None,
) -> None:
    """Write YAML data to a file atomically.

    Uses temp file + fsync + os.replace to ensure the target file is never left in a partially-
    written state. If the process crashes mid-write, the previous version of the file remains
    intact.
    """
    path = Path(path)

    def _write(f) -> None:
        # allow_unicode=True writes emoji/kaomoji (e.g. personalities, skin
        # cursors) as real UTF-8 instead of fragile escape sequences. Without
        # it, PyYAML emits astral-plane chars as `\UXXXXXXXX` (8-digit) escapes
        # inside multi-line double-quoted strings wrapped with `\`
        # continuations — a structure that stricter/non-PyYAML parsers and
        # hand-edits routinely break into unclosed quotes, corrupting the whole
        # config (GitHub #51356).
        yaml.dump(
            data,
            f,
            Dumper=IndentDumper,
            default_flow_style=default_flow_style,
            sort_keys=sort_keys,
            allow_unicode=True,
        )
        if extra_content:
            f.write(extra_content)

    _atomic_write(
        path, _write, prefix=f".{path.stem}_", mode=_mode_for_write(path, create_mode)
    )


def _roundtrip_yaml():
    """ruamel round-trip ``YAML`` configured to keep quotes/Unicode with 2-space indents."""
    from ruamel.yaml import YAML

    yaml_rt = YAML(typ="rt")
    yaml_rt.preserve_quotes = True
    yaml_rt.allow_unicode = True
    yaml_rt.default_flow_style = False
    yaml_rt.indent(mapping=2, sequence=4, offset=2)
    return yaml_rt


def _load_commented_map(yaml_rt, path: Path):
    """Load *path* with *yaml_rt* as a ``CommentedMap`` (empty when missing/blank)."""
    from ruamel.yaml.comments import CommentedMap

    data = None
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            data = yaml_rt.load(f)
    return data if isinstance(data, CommentedMap) else CommentedMap(data or {})


def atomic_roundtrip_yaml_update(
    path: Union[str, Path],
    key_path: str,
    value: Any,
) -> None:
    """Update one dotted YAML key while preserving comments and readable text.

    Narrower than :func:`atomic_yaml_write` on purpose: for user-edited config files where
    comments, ordering, quoting and Unicode must survive a single setting mutation. Still writes
    via temp file + fsync + atomic replace.
    """
    from ruamel.yaml.comments import CommentedMap

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    yaml_rt = _roundtrip_yaml()
    config = _load_commented_map(yaml_rt, path)

    current = config
    # Honor escaped dots and prefer existing literal dotted keys (e.g. model
    # IDs like ``glm-5.3``) over blind splitting — same navigation as
    # ``hermes config set``'s ``_set_nested`` (#91607: /model + TUI
    # persistence route through here and used to write ``glm-5: {'3': ...}``
    # phantom siblings while the runtime kept reading the literal key).
    from hermes_cli.config import _greedy_literal_match, _split_key_path

    keys = _split_key_path(key_path)
    i = 0
    while True:
        remaining = keys[i:]
        seg, consumed = remaining[0], 1
        match = _greedy_literal_match(dict(current), remaining)
        if match is not None:
            seg, consumed = match
        if i + consumed == len(keys):
            current[seg] = value
            break
        next_value = current.get(seg)
        if not isinstance(next_value, CommentedMap):
            next_value = CommentedMap()
            current[seg] = next_value
        current = next_value
        i += consumed

    _atomic_write(
        path,
        lambda f: yaml_rt.dump(config, f),
        prefix=f".{path.stem}_",
        mode=_preserve_file_mode(path),
    )


def atomic_roundtrip_yaml_save(
    path: Union[str, Path],
    new_state: dict,
) -> None:
    """Persist a full config-state dict while preserving comments and ordering.

    Behaves like ``atomic_yaml_write`` (writes the whole file in one shot from ``new_state``), but
    routes through ruamel.yaml round-trip mode so existing comments, key order, quotes, and readable
    Unicode survive.

    This is the comment-safe replacement for ``yaml.safe_dump(cfg, f)`` in callers that mutate a
    deep-loaded config dict and want to persist the whole thing.
    """
    from ruamel.yaml.comments import CommentedMap
    from ruamel.yaml.scalarstring import DoubleQuotedScalarString

    from hermes_cli.config import require_readable_config_before_write

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require_readable_config_before_write(path)
    yaml_rt = _roundtrip_yaml()
    existing = _load_commented_map(yaml_rt, path)

    # ruamel's round-trip dumper resolves plain scalars against the YAML 1.2
    # core schema, where only true/false/null are reserved words — so a plain
    # python str like "off" or "yes" is emitted unquoted. Every other config
    # reader in this codebase (atomic_config_write's PyYAML path, yaml.safe_load
    # call sites, etc.) parses under YAML 1.1 rules, where on/off/yes/no are
    # boolean synonyms. Without forcing quotes here, a freshly written
    # `approvals.mode: off` silently round-trips back as `False` under
    # yaml.safe_load. Force-quote any new string value that YAML 1.1 would
    # otherwise misparse as bool/null.
    _YAML11_AMBIGUOUS_WORDS = {"y", "n", "yes", "no", "true", "false", "on", "off", "null", "~"}

    def _quote_if_yaml11_ambiguous(value):
        if isinstance(value, str) and value.lower() in _YAML11_AMBIGUOUS_WORDS:
            return DoubleQuotedScalarString(value)
        return value

    def _merge(dst: CommentedMap, src: dict) -> None:
        # Update / recurse into keys present in src.
        for key, value in src.items():
            if isinstance(value, dict):
                current = dst.get(key)
                if not isinstance(current, CommentedMap):
                    current = CommentedMap()
                    dst[key] = current
                _merge(current, value)
            else:
                dst[key] = _quote_if_yaml11_ambiguous(value)
        # Delete keys missing from src — preserves "explicit absence" semantics
        # of the old _save_cfg(cfg) pattern (e.g. cfg.pop("custom_prompt", None)
        # then _save_cfg must actually remove the key from disk).
        for key in [k for k in dst if k not in src]:
            del dst[key]

    _merge(existing, new_state)

    _atomic_write(
        path,
        lambda f: yaml_rt.dump(existing, f),
        prefix=f".{path.stem}_",
        mode=_preserve_file_mode(path),
    )


# ─── JSON Helpers ─────────────────────────────────────────────────────────────


def safe_json_loads(text: str, default: Any = None) -> Any:
    """Parse JSON, returning *default* on any parse error."""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return default


# ── Fast YAML loading ────────────────────────────────────────────────────
#
# PyYAML's pure-Python SafeLoader is ~8x slower than the libyaml-backed
# ``CSafeLoader`` C extension. Startup parses config.yaml and every plugin
# manifest with the slow path, costing ~0.9s of cold-start time. The C loader
# is a true drop-in for ``safe_load`` (same restricted tag set), so prefer it
# and fall back to the pure-Python loader only when libyaml isn't compiled in.
_fast_yaml_loader = getattr(yaml, "CSafeLoader", None) or yaml.SafeLoader


def fast_safe_load(stream: Any) -> Any:
    """``yaml.safe_load`` using the libyaml C loader when available.

    Accepts the same inputs as ``yaml.safe_load`` (a ``str``/``bytes`` document or a readable file
    object) and returns the same parsed structure. Falls back to PyYAML's pure-Python ``SafeLoader``
    when ``CSafeLoader`` isn't available, so behavior is identical everywhere — only the speed
    differs.
    """
    return yaml.load(stream, Loader=_fast_yaml_loader)


# ─── Environment Variable Helpers ─────────────────────────────────────────────


def _env_number(key: str, default, cast):
    raw = os.getenv(key, "").strip()
    if not raw:
        return default
    try:
        return cast(raw)
    except (ValueError, TypeError):
        return default


def env_int(key: str, default: int = 0) -> int:
    """Read an environment variable as an integer, with fallback."""
    return _env_number(key, default, int)


def env_float(key: str, default: float = 0.0) -> float:
    """Read an environment variable as a float, with fallback."""
    return _env_number(key, default, float)


def env_bool(key: str, default: bool = False) -> bool:
    """Read an environment variable as a boolean."""
    return is_truthy_value(os.getenv(key, ""), default=default)


# ─── Proxy Helpers ────────────────────────────────────────────────────────────


_PROXY_ENV_KEYS = (
    "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY",
    "https_proxy", "http_proxy", "all_proxy",
)


def normalize_proxy_url(proxy_url: str | None) -> str | None:
    """Normalize proxy URLs for httpx/aiohttp compatibility.

    WSL/Clash-style environments export SOCKS proxies as ``socks://host:port``; httpx rejects
    that alias and needs the explicit ``socks5://`` scheme.
    """
    candidate = str(proxy_url or "").strip()
    if candidate.lower().startswith("socks://"):
        return f"socks5://{candidate[len('socks://'):]}"
    return candidate or None


def normalize_proxy_env_vars() -> None:
    """Rewrite supported proxy env vars to canonical URL forms in-place."""
    for key in _PROXY_ENV_KEYS:
        value = os.getenv(key, "")
        normalized = normalize_proxy_url(value)
        if normalized and normalized != value:
            os.environ[key] = normalized


# ─── URL Parsing Helpers ──────────────────────────────────────────────────────


def _parse_base_url(base_url: str):
    """``urlparse`` that tolerates a bare ``host[:port][/path]`` (no scheme)."""
    raw = (base_url or "").strip()
    if not raw:
        return None
    return urlparse(raw if "://" in raw else f"//{raw}")


def base_url_hostname(base_url: str) -> str:
    """Return the lowercased hostname for a base URL, or ``""`` if absent.

    Compare exact hostnames against known provider hosts instead of substring-matching the raw
    URL: substring checks treat ``https://api.openai.com.example/v1`` or
    ``https://proxy.test/api.openai.com/v1`` as native endpoints, mis-routing api_mode and auth.
    """
    parsed = _parse_base_url(base_url)
    return (parsed.hostname or "").lower().rstrip(".") if parsed else ""


# ─── Model Capability Detection ──────────────────────────────────────────────


def model_forces_max_completion_tokens(model: str) -> bool:
    """Return True for model families that require ``max_completion_tokens``.

    OpenAI's newer families reject ``max_tokens`` on /v1/chat/completions with HTTP 400
    ``unsupported_parameter`` — the caller must send ``max_completion_tokens`` instead. This covers:
    """
    m = (model or "").strip().lower().rsplit("/", 1)[-1]
    return m.startswith(("gpt-4o", "gpt-4.1", "gpt-5", "o1", "o3", "o4"))


def base_url_origin(base_url: str) -> tuple[str, str, int]:
    """Return ``(scheme, hostname, effective_port)`` for a base URL.

    Origin, not just host: ``https://h`` vs ``http://h`` and two ports on one host are different
    trust boundaries, so any decision to hand a bearer secret to a new URL must compare all
    three — hostname alone would authorise an HTTPS→HTTP downgrade. The port defaults to 443/80
    when absent so ``https://h`` equals ``https://h:443``. Returns ``("", "", 0)`` on no
    hostname or a bad port.
    """
    parsed = _parse_base_url(base_url)
    hostname = (parsed.hostname or "").lower().rstrip(".") if parsed else ""
    if not hostname:
        return ("", "", 0)
    scheme = (parsed.scheme or "").lower()
    try:
        port = parsed.port
    except ValueError:
        # Out-of-range or non-numeric port — not a usable origin.
        return ("", "", 0)
    if port is None:
        port = {"https": 443, "http": 80}.get(scheme, 0)
    return (scheme, hostname, port)


def base_url_host_matches(base_url: str, domain: str) -> bool:
    """Return True when the base URL's hostname is ``domain`` or a subdomain.

    Safer counterpart to ``domain in base_url``, which has the substring false-positive class
    noted on ``base_url_hostname`` (``evil.com/moonshot.ai`` or ``moonshot.ai.evil`` must not
    match). Accepts bare hosts, full URLs, and URLs with paths.
    """
    hostname = base_url_hostname(base_url)
    domain = (domain or "").strip().lower().rstrip(".")
    return bool(hostname and domain) and (hostname == domain or hostname.endswith("." + domain))
