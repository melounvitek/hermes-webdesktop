"""Shared file safety rules used by both tools and ACP shims.

Every guard here is defense-in-depth, NOT a security boundary: the terminal
tool runs as the same OS user and can read/write anything. The value is a
clear denial for models that respect tool errors plus a visible audit trail.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


def _constants_path(getter_name: str) -> Path:
    """Call ``hermes_constants.<getter_name>()`` (local import avoids cycles); ``~/.hermes`` on any failure."""
    try:
        import hermes_constants

        return getattr(hermes_constants, getter_name)()
    except Exception:
        return Path(os.path.expanduser("~/.hermes"))


def _hermes_home_path() -> Path:
    """Active HERMES_HOME (profile-aware). Tests monkeypatch this name."""
    return _constants_path("get_hermes_home")


def _hermes_root_path() -> Path:
    """Hermes root dir (parent of any profile, never per-profile)."""
    return _constants_path("get_default_hermes_root")


def _hermes_dirs() -> list[Path]:
    """Resolved active HERMES_HOME and global root, deduplicated.

    Both are checked so credential stores at <root>/... stay guarded when
    running under a profile (HERMES_HOME = <root>/profiles/<name>).
    """
    dirs: list[Path] = []
    for base in (_hermes_home_path(), _hermes_root_path()):
        try:
            real = base.resolve()
        except Exception:
            continue
        if real not in dirs:
            dirs.append(real)
    return dirs


def _is_under(resolved: str, prefix: str) -> bool:
    return resolved == prefix or resolved.startswith(prefix + os.sep)


def _under_any(resolved: Path, base: Path) -> bool:
    try:
        resolved.relative_to(base)
        return True
    except ValueError:
        return False


def _home_and_resolved(path: str) -> tuple[str, str]:
    """``(realpath(~), realpath(expanduser(path)))`` — the write-guard coordinate pair."""
    return os.path.realpath(os.path.expanduser("~")), os.path.realpath(os.path.expanduser(str(path)))


def build_write_denied_paths(home: str) -> set[str]:
    """Return exact sensitive paths that must never be written."""
    hermes_home = _hermes_home_path()
    hermes_root = _hermes_root_path()
    return {
        os.path.realpath(p)
        for p in [
            os.path.join(home, ".ssh", "authorized_keys"),
            os.path.join(home, ".ssh", "id_rsa"),
            os.path.join(home, ".ssh", "id_ed25519"),
            # ``~/.ssh/config`` is deliberately NOT hard-denied: no key bytes, and
            # editing it is routine. It can carry ProxyCommand / Match exec, so it
            # goes through the approval gate instead (build_write_approval_paths).
            # Both the active-profile and top-level .env: overwriting the root .env
            # leaks credentials across every profile that inherits from it.
            str(hermes_home / ".env"),
            str(hermes_root / ".env"),
            # Anthropic PKCE credential stores; the root copy is still read by
            # default/non-profile sessions when a profile is active.
            str(hermes_home / ".anthropic_oauth.json"),
            str(hermes_root / ".anthropic_oauth.json"),
            # Bitwarden Secrets Manager encrypted disk cache.
            str(hermes_home / "cache" / "bws_cache.enc.json"),
            str(hermes_root / "cache" / "bws_cache.enc.json"),
            os.path.join(home, ".netrc"),
            os.path.join(home, ".pgpass"),
            os.path.join(home, ".npmrc"),
            os.path.join(home, ".pypirc"),
            os.path.join(home, ".git-credentials"),
            "/etc/sudoers",
            "/etc/passwd",
            "/etc/shadow",
        ]
    }


def build_write_denied_prefixes(home: str) -> list[str]:
    """Return sensitive directory prefixes that must never be written."""
    return [
        os.path.realpath(p) + os.sep
        for p in [
            os.path.join(home, ".ssh"),
            os.path.join(home, ".aws"),
            os.path.join(home, ".gnupg"),
            os.path.join(home, ".kube"),
            "/etc/sudoers.d",
            "/etc/systemd",
            os.path.join(home, ".docker"),
            os.path.join(home, ".azure"),
            os.path.join(home, ".config", "gh"),
            os.path.join(home, ".config", "gcloud"),
        ]
    ]


def get_safe_write_roots() -> set[str]:
    """Resolved HERMES_WRITE_SAFE_ROOT paths (``os.pathsep``-separated list)."""
    env = os.getenv("HERMES_WRITE_SAFE_ROOT", "")
    if not env:
        return set()
    roots: set[str] = set()
    for path in env.split(os.pathsep):
        if path:
            try:
                roots.add(os.path.realpath(os.path.expanduser(path)))
            except (OSError, ValueError):
                continue
    return roots


def build_write_approval_paths(home: str) -> set[str]:
    """Paths that need human APPROVAL to write but are not hard-denied credentials.

    ``~/.ssh/config`` is routine to edit and holds no key bytes, but can carry
    ``ProxyCommand`` / ``Match exec``. Interactive file tools prompt
    (approve-once/session/always, like the terminal tool's ``~/.ssh`` gate);
    non-interactive callers (ACP shims, background jobs) fail closed.
    """
    return {os.path.realpath(os.path.join(home, ".ssh", "config"))}


# HERMES_HOME / root subpaths that the agent's generic file tools must not
# rewrite. Session transcripts (state.db, sessions/) are application-owned
# state whose rewrite can falsify history and break resume/compression;
# mcp-tokens/ and pairing/ hold credential material.
_HERMES_PROTECTED_SUBPATHS = ("state.db", "sessions", "mcp-tokens", "pairing")


def _classify_write_denial(path: str) -> Optional[str]:
    """Return ``'credential'``, ``'safe_root'``, or ``None`` if writes are allowed."""
    home, resolved = _home_and_resolved(path)

    # Approval-gated paths are allowed at this layer so interactive tools can
    # prompt; checked first so the ``.ssh/`` prefix deny doesn't swallow them.
    if resolved in build_write_approval_paths(home):
        return None

    if resolved in build_write_denied_paths(home) or any(
        resolved.startswith(prefix) for prefix in build_write_denied_prefixes(home)
    ):
        return "credential"

    for base in _hermes_dirs():
        for sub in _HERMES_PROTECTED_SUBPATHS:
            try:
                if _is_under(resolved, os.path.realpath(os.path.join(str(base), sub))):
                    return "credential"
            except Exception:
                pass

    safe_roots = get_safe_write_roots()
    if safe_roots and not any(_is_under(resolved, root) for root in safe_roots):
        return "safe_root"

    return None


def is_write_denied(path: str) -> bool:
    """Return True if path is blocked by the write denylist or safe root."""
    return _classify_write_denial(path) is not None


def get_write_denied_error(path: str, *, verb: str = "Write") -> Optional[str]:
    """Return a user/model-facing error when writes to ``path`` are blocked."""
    denial = _classify_write_denial(path)
    if denial is None:
        return None
    if denial == "safe_root":
        roots_display = os.pathsep.join(sorted(get_safe_write_roots()))
        return (
            f"{verb} denied: '{path}' is outside HERMES_WRITE_SAFE_ROOT "
            f"({roots_display}). Unset the variable or add this path's directory prefix."
        )
    return f"{verb} denied: '{path}' is a protected system/credential file."


def is_write_approval_required(path: str) -> bool:
    """True if ``path`` is approval-gated (``~/.ssh/config``): interactive callers
    prompt, callers without a channel treat it as a block (fail closed)."""
    home, resolved = _home_and_resolved(path)
    return resolved in build_write_approval_paths(home)


# Secret-bearing project-local env file basenames, blocked anywhere on disk.
_BLOCKED_PROJECT_ENV_BASENAMES: set[str] = {
    ".env",
    ".env.local",
    ".env.development",
    ".env.production",
    ".env.test",
    ".env.staging",
    ".envrc",
}

_DID_SUFFIX = (
    " (Defense-in-depth — not a security boundary; the terminal tool can still bypass.)"
)

# Exact-file credential stores under HERMES_HOME / <root>. The agent never
# needs these directly — provider tools consume them through internal channels.
_CREDENTIAL_FILE_NAMES = (
    "auth.json",
    "auth.lock",
    ".anthropic_oauth.json",
    ".env",
    "webhook_subscriptions.json",
    os.path.join("auth", "google_oauth.json"),
    # Bitwarden Secrets Manager disk cache: plaintext secret values.
    os.path.join("cache", "bws_cache.json"),
)

# Directory-prefix read denies under HERMES_HOME / <root>: (subdir, message for
# the directory itself, message for a file inside). browser-profile/ is a copy
# of the user's Cookies / Login Data — the same credential class as auth.json.
_READ_DENIED_DIRS = (
    (
        "mcp-tokens",
        "is the Hermes MCP token directory and cannot be read directly.",
        "is a Hermes MCP token file and cannot be read directly.",
    ),
    (
        "browser-profile",
        "is the Hermes real-profile browser snapshot directory (copied "
        "cookies/logins) and cannot be read directly.",
        "is inside the Hermes real-profile browser snapshot (copied "
        "cookies/logins) and cannot be read directly.",
    ),
)


def get_read_block_error(path: str) -> Optional[str]:
    """Return an error message when a read targets a denied Hermes path.

    Blocked: internal skill-hub caches (prompt-injection carriers), credential
    stores under HERMES_HOME and the global root (exact files, plus anything
    under ``mcp-tokens/`` and ``browser-profile/``), and project-local ``.env``
    files anywhere on disk (``.env.example`` is the documented-shape substitute).

    Callers that resolve relative paths against a non-process cwd (e.g.
    ``TERMINAL_CWD``) MUST pass an absolute path: ``resolve()`` here anchors at
    the process cwd, so a relative ``"auth.json"`` would miss the denylist.
    """
    resolved = Path(path).expanduser().resolve()
    hermes_dirs = _hermes_dirs()

    for hd in hermes_dirs:
        if _under_any(resolved, hd / "skills" / ".hub"):
            return (
                f"Access denied: {path} is an internal Hermes cache file "
                "and cannot be read directly to prevent prompt injection. "
                "Use the skills_list or skill_view tools instead."
            )

    for hd in hermes_dirs:
        for name in _CREDENTIAL_FILE_NAMES:
            try:
                blocked = (hd / name).resolve()
            except Exception:
                continue
            if resolved == blocked:
                return (
                    f"Access denied: {path} is a Hermes credential store "
                    "and cannot be read directly. Provider tools consume "
                    "these credentials through internal channels." + _DID_SUFFIX
                )

    for subdir, dir_msg, file_msg in _READ_DENIED_DIRS:
        for hd in hermes_dirs:
            try:
                blocked_dir = (hd / subdir).resolve()
            except Exception:
                continue
            if resolved == blocked_dir:
                return f"Access denied: {path} {dir_msg}{_DID_SUFFIX}"
            if _under_any(resolved, blocked_dir):
                return f"Access denied: {path} {file_msg}{_DID_SUFFIX}"

    if resolved.name.lower() in _BLOCKED_PROJECT_ENV_BASENAMES:
        return (
            f"Access denied: {path} is a secret-bearing environment file "
            "and cannot be read to prevent credential leakage. "
            "If you need to check the file structure, read .env.example instead." + _DID_SUFFIX
        )

    return None


def raise_if_read_blocked(path: str) -> None:
    """Raise ``ValueError`` if ``path`` is a denied Hermes read (see ``get_read_block_error``).

    Shared chokepoint for provider input-loading sites (e.g. image-gen local
    paths). Best-effort: unexpected internal errors no-op rather than break
    local-file loading; a real block still propagates.
    """
    try:
        blocked = get_read_block_error(path)
    except Exception:  # noqa: BLE001 - guard must never break local-file loading
        return
    if blocked:
        raise ValueError(blocked)


def _resolve_active_profile_name() -> str:
    """Active profile name from HERMES_HOME: ``~/.hermes`` -> ``"default"``,
    ``~/.hermes/profiles/X`` -> ``"X"``; ``"default"`` on any resolution failure."""
    try:
        home_real = _hermes_home_path().resolve()
        root_real = _hermes_root_path().resolve()
    except (OSError, RuntimeError):
        return "default"
    try:
        parts = home_real.relative_to(root_real / "profiles").parts
        if parts:
            return parts[0]
    except ValueError:
        pass
    return "default"


def get_cross_profile_warning(path: str) -> Optional[str]:
    """RETIRED: always ``None``. Profiles were never isolated (same OS user), so
    the guard was ceremony that taught a bypass arg. Stub kept so external
    callers/plugins fail soft; the system prompt's active-profile hint remains."""
    return None


# --- Sandbox-mirror write guard ---
# Non-local terminal backends bind a sandbox-local dir to the container's $HOME:
#   <HERMES_HOME>/profiles/<name>/sandboxes/<backend>/<task>/home/.hermes/...
# A host-side write there lands on a mirror the host never reads: silent
# success, divergent copies. Path-shape-only detection, independent of the
# active profile. Does NOT cover the inner-container case where the bind mount
# strips the prefix — that is classify_container_mirror_target below.

_SANDBOX_MIRROR_WARNING = (
    "Sandbox-mirror write blocked by soft guard: {target_path} "
    "sits under {mirror_root!r}, which is {body} "
    "Use the host-side tool for authoritative state (e.g. ``memory`` for memories), "
    "or address the host path directly. To bypass {bypass} with ``cross_profile=True``. "
    "(Defense-in-depth — not a security boundary; the terminal tool can still bypass.)"
)


def classify_sandbox_mirror_target(path: str) -> Optional[dict]:
    """Classify a write target as a sandbox-mirror of authoritative Hermes state.

    Returns ``None`` for non-mirror paths, else ``target_path`` (resolved),
    ``mirror_root`` (the ``…/home/.hermes`` prefix) and ``inner_path`` (what
    the agent likely meant to address on the host).
    """
    try:
        target = Path(os.path.expanduser(str(path))).resolve()
    except (OSError, RuntimeError):
        return None

    parts = target.parts
    # Need at least: sandboxes / <backend> / <task> / home / .hermes / <thing>; inner_idx = the .hermes part.
    for i, part in enumerate(parts):
        if part == "sandboxes" and i + 5 < len(parts) and parts[i + 3] == "home" and parts[i + 4] == ".hermes":
            inner_idx = i + 4
            break
    else:
        return None
    return {
        "target_path": str(target),
        "mirror_root": str(Path(*parts[: inner_idx + 1])),
        "inner_path": str(Path(*parts[inner_idx + 1 :])) if inner_idx + 1 < len(parts) else "",
    }


def _mirror_warning(info: Optional[dict], body: str, bypass: str) -> Optional[str]:
    """Render ``_SANDBOX_MIRROR_WARNING`` for a classify_* result (``body`` may use ``{inner_path}``)."""
    if info is None:
        return None
    return _SANDBOX_MIRROR_WARNING.format(
        target_path=info["target_path"],
        mirror_root=info["mirror_root"],
        body=body.format(inner_path=info["inner_path"]),
        bypass=bypass,
    )


def get_sandbox_mirror_warning(path: str) -> Optional[str]:
    """Model-facing soft-guard warning when ``path`` lands in a sandbox mirror, else ``None``.

    Caller surfaces it as a tool-result error; ``cross_profile=True`` bypasses.
    """
    return _mirror_warning(
        classify_sandbox_mirror_target(path),
        "a per-task mirror created by a non-local terminal backend (docker/daytona/etc.). "
        "Writes here land on a copy that the host Hermes process never reads — the "
        "authoritative file is likely {inner_path!r} under the real HERMES_HOME.",
        "this guard after explicit user direction, retry the call",
    )


def classify_container_mirror_target(
    path: str,
    mirror_prefix: str | None = None,
) -> Optional[dict]:
    """Classify a write target as a container-side sandbox mirror.

    Inside the container the bind mount strips the ``sandboxes/`` prefix (the
    agent sees plain ``/root/.hermes/…``), so the caller must supply
    ``mirror_prefix`` once it knows file tools run in a docker sandbox.
    Returns ``None`` without a prefix or when the path is outside it, else
    ``target_path``, ``mirror_root`` and ``inner_path``.
    """
    if not mirror_prefix:
        return None
    try:
        target = Path(os.path.expanduser(str(path))).resolve()
        mirror = Path(os.path.expanduser(mirror_prefix)).resolve()
        inner = target.relative_to(mirror)
    except (OSError, RuntimeError, ValueError):
        return None
    return {
        "target_path": str(target),
        "mirror_root": str(mirror),
        "inner_path": inner.as_posix(),
    }


def get_container_mirror_warning(
    path: str,
    mirror_prefix: str | None = None,
) -> Optional[str]:
    """Model-facing soft-guard warning when ``path`` lands in the container's mirror, else ``None``."""
    return _mirror_warning(
        classify_container_mirror_target(path, mirror_prefix),
        "the container's bind-mounted home — a per-task mirror that the host Hermes "
        "process never reads. The authoritative file is {inner_path!r} under "
        "the real HERMES_HOME.",
        "after explicit user direction, retry",
    )
