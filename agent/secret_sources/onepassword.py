"""1Password (`op` CLI) secret source.

Users map env-var names to official ``op://vault/item/field`` references in
``secrets.onepassword.env``; after ``.env`` loads each reference is resolved
with one ``op read -- <reference>`` call.  Authentication is whatever the
user's ``op`` CLI already uses (``OP_SERVICE_ACCOUNT_TOKEN`` for headless
boxes, ``OP_SESSION_*`` for interactive sessions) — Hermes never authenticates
on the user's behalf.  Failures NEVER block startup.

Successful, complete pulls are cached in-process and under
``<hermes_home>/cache/op_cache.json`` (values only; auth material is
fingerprinted, never stored) so back-to-back ``hermes`` invocations don't
re-shell ``op`` for every reference.
"""

from __future__ import annotations

import hashlib
import logging
import os
import shutil
import subprocess  # noqa: F401 — tests monkeypatch ``op.subprocess.run``
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from agent.secret_sources._cache import CachedFetch, SecretCache
from agent.secret_sources.base import (
    ErrorKind,
    FetchResult,
    SecretSource,
    classify_cli_error,
    coerce_float,
    get_source_environment,
    is_valid_env_name,
    run_cli,
)

logger = logging.getLogger(__name__)

_OP_RUN_TIMEOUT = 30

# `op` itself reads OP_SERVICE_ACCOUNT_TOKEN; `service_account_token_env` lets
# the user source it from another name, and _op_child_env normalizes it back.
_DEFAULT_TOKEN_ENV = "OP_SERVICE_ACCOUNT_TOKEN"

# Minimal allowlisted child env (never the full post-dotenv os.environ, which
# holds every provider credential).  OP_SESSION_* and the token are added
# dynamically in _op_child_env().
_OP_ENV_ALLOWLIST = (
    "PATH", "HOME", "USERPROFILE", "APPDATA", "LOCALAPPDATA", "SystemRoot",
    "TMPDIR", "TMP", "TEMP", "XDG_CONFIG_HOME", "XDG_RUNTIME_DIR",
    "OP_ACCOUNT", "OP_CONNECT_HOST", "OP_CONNECT_TOKEN",
    # Lets a user skip op's desktop-app integration probe (which can hang with
    # no timeout on a wedged desktop container) and go straight to token auth.
    "OP_LOAD_DESKTOP_APP_SETTINGS",
)


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

# L1 key folds in str(home_path) so a HERMES_HOME switch inside one long-lived
# process (the gateway) can't return another profile's secrets.  The disk key
# omits home because the file already lives under <home>/cache/.
_CacheKey = Tuple[str, str, str, str]  # (auth_fp, account, home, refs_fp)
_DISK_CACHE_BASENAME = "op_cache.json"


def _disk_key_str(cache_key: _CacheKey) -> str:
    auth_fp, account, _home, refs_fp = cache_key
    return f"{auth_fp}|{account}|{refs_fp}"


_STORE: SecretCache[_CacheKey] = SecretCache(_DISK_CACHE_BASENAME, key_serializer=_disk_key_str)
_CACHE = _STORE.memory  # tests flush L1 directly


# ---------------------------------------------------------------------------
# Reference validation + fingerprinting
# ---------------------------------------------------------------------------


def _validate_references(
    references: Optional[Dict[str, str]],
) -> Tuple[Dict[str, str], List[str]]:
    """``(valid_refs, warnings)``: keep valid env names bound to stripped ``op://`` strings."""
    valid: Dict[str, str] = {}
    warnings: List[str] = []
    for name, ref in (references or {}).items():
        if not is_valid_env_name(name):
            warnings.append(f"Skipping {name!r}: not a valid env-var name")
            continue
        if not isinstance(ref, str):
            warnings.append(f"Skipping {name!r}: reference is not a string")
            continue
        cleaned = ref.strip()
        if not cleaned.startswith("op://"):
            warnings.append(
                f"Skipping {name!r}: {ref!r} is not an op:// secret reference"
            )
            continue
        valid[name] = cleaned
    return valid, warnings


def _fingerprint(material: str) -> str:
    return hashlib.sha256(material.encode("utf-8")).hexdigest()[:16]


def _auth_fingerprint(token_env: str) -> str:
    """SHA-256 prefix over everything `op` would authenticate with.

    Folds in the service-account token, OP_ACCOUNT, Connect host/token and all
    ``OP_SESSION_*`` vars, so signing into a different identity changes the
    cache key and a value cached under the old identity is never served.
    """
    source_env = get_source_environment()
    parts: List[str] = [
        f"token={source_env.get(token_env, '')}",
        f"account={source_env.get('OP_ACCOUNT', '')}",
        f"connect_host={source_env.get('OP_CONNECT_HOST', '')}",
        f"connect_token={source_env.get('OP_CONNECT_TOKEN', '')}",
    ]
    for key in sorted(source_env):
        if key.startswith("OP_SESSION_"):
            parts.append(f"{key}={source_env[key]}")
    return _fingerprint("\n".join(parts))


def _refs_fingerprint(references: Dict[str, str]) -> str:
    return _fingerprint("\n".join(f"{name}={references[name]}" for name in sorted(references)))


# ---------------------------------------------------------------------------
# Binary discovery + `op read`
# ---------------------------------------------------------------------------


def find_op(binary_path: str = "") -> Optional[Path]:
    """Resolve a usable ``op`` binary, or None.

    A pinned ``binary_path`` is used verbatim (PATH is NOT consulted) and a
    pinned-but-missing path returns None rather than silently falling back.
    """
    if binary_path:
        pinned = Path(binary_path)
        if pinned.exists() and os.access(pinned, os.X_OK):
            return pinned
        return None
    found = shutil.which("op")
    return Path(found) if found else None


def _scrub(text: str) -> str:
    """Full ECMA-48 ANSI strip (so a control sequence can't hide text after a redaction marker) + trim."""
    from tools.ansi_strip import strip_ansi

    return strip_ansi(text).replace("\x1b", "").strip()


def _op_child_env(token_value: str) -> Dict[str, str]:
    source_env = get_source_environment()
    env = {k: source_env[k] for k in _OP_ENV_ALLOWLIST if k in source_env}
    env.update((k, v) for k, v in source_env.items() if k.startswith("OP_SESSION_"))
    if token_value:
        env["OP_SERVICE_ACCOUNT_TOKEN"] = token_value
    env["NO_COLOR"] = "1"
    return env


def _run_op_read(
    op: Path,
    reference: str,
    *,
    account: str = "",
    token_value: str = "",
) -> str:
    """Resolve one ``op://`` reference; raises ``RuntimeError`` on any failure.

    An exit-0 empty/whitespace-only value is a failure too — applying it would
    silently clobber a good .env/shell credential with ``""``.
    """
    cmd: List[str] = [str(op), "read"]
    if account:
        cmd += ["--account", account]
    cmd += ["--", reference]  # `--` so a reference can never parse as an op flag

    proc = run_cli(
        cmd, env=_op_child_env(token_value), timeout=_OP_RUN_TIMEOUT, label="op",
        timeout_message=f"op read timed out after {_OP_RUN_TIMEOUT}s for {reference!r}",
        stdin=None,
    )

    if proc.returncode != 0:
        err = _scrub(proc.stderr or "")[:200]
        if err:
            raise RuntimeError(f"op read failed for {reference!r}: {err}")
        raise RuntimeError(
            f"op read exited {proc.returncode} for {reference!r}"
        )

    # Strip only op's trailing newline so intentional edge spaces survive.
    value = (proc.stdout or "").rstrip("\r\n")
    if not value.strip():
        raise RuntimeError(f"op read returned an empty value for {reference!r}")
    return value


# ---------------------------------------------------------------------------
# Fetch
# ---------------------------------------------------------------------------


def fetch_onepassword_secrets(
    *,
    references: Dict[str, str],
    account: str = "",
    token_env: str = _DEFAULT_TOKEN_ENV,
    binary: Optional[Path] = None,
    binary_path: str = "",
    use_cache: bool = True,
    cache_ttl_seconds: float = 300,
    home_path: Optional[Path] = None,
) -> Tuple[Dict[str, str], List[str]]:
    """Resolve ``references`` (name → ``op://…``) to ``(secrets, warnings)``.

    Raises ``RuntimeError`` only when no ``op`` binary is available.  Per-ref
    failures become warnings and the ref is dropped, so one bad entry never
    sinks the rest.  Only a complete, error-free pull is cached, so a transient
    auth failure isn't frozen in for the whole TTL window.
    """
    valid, warnings = _validate_references(references)
    if not valid:
        return {}, warnings

    token_value = get_source_environment().get(token_env, "").strip()
    cache_key: _CacheKey = (
        _auth_fingerprint(token_env),
        account or "",
        str(home_path) if home_path is not None else "",
        _refs_fingerprint(valid),
    )

    if use_cache:
        cached = _STORE.lookup(cache_key, cache_ttl_seconds, home_path)
        if cached is not None:
            return dict(cached.secrets), warnings

    op = binary or find_op(binary_path)
    if op is None:
        raise RuntimeError(
            "op CLI not found.  Install the 1Password CLI "
            "(https://developer.1password.com/docs/cli/get-started/) or set "
            "secrets.onepassword.binary_path to its absolute location."
        )

    secrets: Dict[str, str] = {}
    read_errors = 0
    for name in sorted(valid):
        try:
            secrets[name] = _run_op_read(
                op, valid[name], account=account, token_value=token_value
            )
        except RuntimeError as exc:
            warnings.append(str(exc))
            read_errors += 1

    if use_cache and not read_errors and secrets:
        entry = CachedFetch(secrets=dict(secrets), fetched_at=time.time())
        _STORE.store(cache_key, entry, cache_ttl_seconds, home_path)

    return secrets, warnings


def _missing_binary_error(binary_path: str) -> str:
    if binary_path:
        return (
            f"secrets.onepassword.binary_path ({binary_path!r}) is not an "
            "executable op binary."
        )
    return (
        "secrets.onepassword.enabled is true but the op CLI was not "
        "found on PATH.  Install it "
        "(https://developer.1password.com/docs/cli/get-started/) or set "
        "secrets.onepassword.binary_path."
    )


# ---------------------------------------------------------------------------
# Public entry point — used by `hermes secrets onepassword sync --apply`
# ---------------------------------------------------------------------------


def apply_onepassword_secrets(
    *,
    enabled: bool,
    env: Optional[Dict[str, str]] = None,
    account: str = "",
    service_account_token_env: str = _DEFAULT_TOKEN_ENV,
    binary_path: str = "",
    override_existing: bool = True,
    cache_ttl_seconds: float = 300,
    home_path: Optional[Path] = None,
) -> FetchResult:
    """Resolve configured ``op://`` references and set them on ``os.environ``.

    Never raises.  References already satisfied by the environment (when
    ``override_existing`` is false) and the token var itself are skipped
    *before* fetching, so ``op`` is never invoked for a value that would be
    discarded.
    """
    result = FetchResult()

    if not enabled:
        return result

    valid, warnings = _validate_references(env)
    result.warnings.extend(warnings)

    def _guarded(name: str) -> bool:
        """True when ``name`` must not be applied (token var or env already set)."""
        return name == service_account_token_env or (
            not override_existing and bool(os.environ.get(name))
        )

    refs_to_fetch: Dict[str, str] = {}
    for name, ref in valid.items():
        if _guarded(name):
            result.skipped.append(name)
        else:
            refs_to_fetch[name] = ref

    if not refs_to_fetch:
        return result

    binary = find_op(binary_path)
    result.binary_path = binary
    if binary is None:
        result.error = _missing_binary_error(binary_path)
        return result

    try:
        secrets, fetch_warnings = fetch_onepassword_secrets(
            references=refs_to_fetch,
            account=account,
            token_env=service_account_token_env,
            binary=binary,
            cache_ttl_seconds=cache_ttl_seconds,
            home_path=home_path,
        )
    except RuntimeError as exc:
        result.error = str(exc)
        return result

    result.secrets = secrets
    result.warnings.extend(fetch_warnings)

    for name, value in secrets.items():
        # Defensive re-check: keys should already be ⊆ refs_to_fetch.
        if _guarded(name):
            if name not in result.skipped:
                result.skipped.append(name)
            continue
        os.environ[name] = value
        result.applied.append(name)

    return result


# ---------------------------------------------------------------------------
# SecretSource adapter — the registry-facing wrapper around this module.
# ---------------------------------------------------------------------------


class OnePasswordSource(SecretSource):
    """1Password as a registered **mapped** source.

    ``fetch()`` only fetches — precedence, overrides and the ``os.environ``
    writes are the orchestrator's.  Mapped: the user explicitly binds each env
    var to an ``op://`` ref, so its claims outrank bulk sources on contested vars.
    """

    name = "onepassword"
    label = "1Password"
    shape = "mapped"
    scheme = "op"
    token_env_key = "service_account_token_env"
    default_token_env = _DEFAULT_TOKEN_ENV
    # override_existing defaults True: an explicit VAR→op:// binding is the
    # strongest user intent; a stale .env line must not silently defeat it.
    override_existing_default = True

    def config_schema(self) -> dict:
        return {
            "enabled": {"description": "Master switch", "default": False},
            "env": {"description": "Map of ENV_VAR -> op://vault/item/field reference", "default": {}},
            "account": {"description": "op --account shorthand (empty = default account)", "default": ""},
            "service_account_token_env": {
                "description": "Env var holding the service-account token "
                               "(unset = desktop/interactive session)",
                "default": _DEFAULT_TOKEN_ENV,
            },
            "binary_path": {"description": "Pin the op binary (empty = resolve via PATH)", "default": ""},
            "cache_ttl_seconds": {"description": "Disk+memory cache TTL; 0 disables", "default": 300},
            "override_existing": {"description": "Resolved values overwrite .env/shell values", "default": True},
        }

    def fetch(self, cfg: dict, home_path: Path) -> FetchResult:
        cfg = cfg if isinstance(cfg, dict) else {}
        result = FetchResult()

        env_map = cfg.get("env")
        valid, warnings = _validate_references(
            env_map if isinstance(env_map, dict) else None
        )
        result.warnings.extend(warnings)
        if not valid:
            if not warnings:
                result.fail(
                    "secrets.onepassword.enabled is true but the env: map is "
                    "empty.  Add ENV_VAR: op://vault/item/field entries.",
                    ErrorKind.NOT_CONFIGURED,
                )
            return result

        binary_path = str(cfg.get("binary_path") or "")
        binary = find_op(binary_path)
        result.binary_path = binary
        if binary is None:
            return result.fail(_missing_binary_error(binary_path), ErrorKind.BINARY_MISSING)

        try:
            secrets, fetch_warnings = fetch_onepassword_secrets(
                references=valid,
                account=str(cfg.get("account") or ""),
                token_env=self.token_env(cfg),
                binary=binary,
                cache_ttl_seconds=coerce_float(cfg.get("cache_ttl_seconds", 300), 300.0),
                home_path=home_path,
            )
        except RuntimeError as exc:
            return result.fail(str(exc), _classify_op_error(str(exc)))

        result.secrets = secrets
        result.warnings.extend(fetch_warnings)
        return result

    def remediation(self, kind, cfg: dict) -> str:
        if kind in (ErrorKind.AUTH_FAILED, ErrorKind.AUTH_EXPIRED):
            return (
                "Run `hermes secrets onepassword token` to paste a fresh "
                f"service-account token ({self.token_env(cfg)}), or `op signin` for an "
                "interactive session."
            )
        if kind == ErrorKind.BINARY_MISSING:
            return (
                "Install the 1Password CLI "
                "(https://developer.1password.com/docs/cli/get-started/) or "
                "set secrets.onepassword.binary_path."
            )
        return super().remediation(kind, cfg)


_OP_ERROR_RULES = (
    (ErrorKind.TIMEOUT, ("timed out",)),
    (ErrorKind.BINARY_MISSING, ("not found on path", "not an executable", "failed to invoke")),
    (ErrorKind.AUTH_FAILED, ("unauthorized", "not signed in", "session expired",
                             "authentication", "401", "403")),
    (ErrorKind.EMPTY_VALUE, ("empty value",)),
    (ErrorKind.NETWORK, ("network", "connection", "resolve host", "dns")),
)


def _classify_op_error(message: str) -> ErrorKind:
    return classify_cli_error(message, _OP_ERROR_RULES)


def clear_caches(home_path: Optional[Path] = None) -> None:
    """Drop in-process AND disk caches (after a token rotation, so the next
    startup resolves fresh instead of serving values cached under the old token)."""
    _STORE.clear(home_path)


_reset_cache_for_tests = clear_caches
