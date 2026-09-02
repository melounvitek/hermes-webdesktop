"""Per-platform slash command access control.

A second axis beside ``allow_from``: of the users allowed to talk to the
gateway, which ones may run which slash commands. Two lists per scope (DM vs
group, mirroring ``allow_from`` / ``group_allow_from``):

  - ``allow_admin_from``      — user IDs that get every registered slash
                                command (built-in + plugin-registered).
  - ``user_allowed_commands`` — command names non-admins may run. Empty /
                                unset → non-admins get no slash commands
                                (beyond the ``_ALWAYS_ALLOWED_FOR_USERS`` floor).

Backward compatibility: if ``allow_admin_from`` is not set for a scope,
gating is disabled for that scope and every allowed user can run every
command, so existing installs are unaffected until an operator lists an admin.

The gate is applied at the dispatch site in ``gateway/run.py`` so it covers
both built-in and plugin commands via the live registry. It never affects
plain chat — non-admins still talk to the agent normally.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, FrozenSet, Iterable, Optional, Tuple


# Floor of read-only commands every allowed user keeps even under gating, so a
# non-admin can still discover what they can do. ``user_allowed_commands`` only
# adds to this set, never restricts it.
_ALWAYS_ALLOWED_FOR_USERS: FrozenSet[str] = frozenset({
    "help",
    "whoami",
})


@dataclass(frozen=True)
class SlashAccessPolicy:
    """Resolved access policy for a single (platform, scope) pair.

    ``scope`` is ``"dm"`` for direct messages and ``"group"`` for every other
    multi-user context; ``policy_for_source`` maps chat_type → scope.
    """

    enabled: bool                      # gating active for this scope?
    admin_user_ids: FrozenSet[str]
    user_allowed_commands: FrozenSet[str]

    def is_admin(self, user_id: Optional[str]) -> bool:
        # Gating disabled → everyone is admin so callers can use is_admin/can_run uniformly.
        if not self.enabled:
            return True
        if not user_id:
            return False
        return str(user_id) in self.admin_user_ids

    def can_run(self, user_id: Optional[str], canonical_cmd: str) -> bool:
        if self.is_admin(user_id):
            return True
        if not canonical_cmd:
            return False
        return canonical_cmd in _ALWAYS_ALLOWED_FOR_USERS or canonical_cmd in self.user_allowed_commands


_DM_CHAT_TYPES = frozenset({"dm", "direct", "private", ""})


def _coerce_list(raw: Any, normalize: Callable[[str], str]) -> FrozenSet[str]:
    """Normalize a YAML-loaded value (None, list/tuple/set, comma string, or scalar)
    into a frozenset of stripped, non-empty strings, applying ``normalize`` to each."""
    if raw is None:
        return frozenset()
    if isinstance(raw, (list, tuple, set, frozenset)):
        items: Iterable[Any] = raw
    elif isinstance(raw, str):
        items = (s for s in raw.split(",") if s.strip())
    else:
        items = (raw,)  # single scalar (int user id, etc.)
    return frozenset(s for s in (normalize(str(it).strip()) for it in items) if s)


def _coerce_id_list(raw: Any) -> FrozenSet[str]:
    """Normalize an admin/user id list into a frozenset of strings."""
    return _coerce_list(raw, lambda s: s)


def _coerce_command_list(raw: Any) -> FrozenSet[str]:
    """Normalize a command allowlist: strip leading slashes (accepts ``/help`` or
    ``help``) and lowercase to match how ``resolve_command()`` stores names."""
    return _coerce_list(raw, lambda s: s.lstrip("/").lower())


def _scope_for_chat_type(chat_type: Optional[str]) -> str:
    if chat_type and chat_type.lower() in _DM_CHAT_TYPES:
        return "dm"
    return "group"


def _platform_extra(platform_config: Any) -> dict:
    """Return the ``extra`` dict from a PlatformConfig-like object (or a bare
    dict, as some test harnesses pass); {} for None/unknown shapes."""
    if platform_config is None:
        return {}
    extra = getattr(platform_config, "extra", None)
    if isinstance(extra, dict):
        return extra
    if isinstance(platform_config, dict):
        return platform_config
    return {}


def _keys_for_scope(scope: str) -> Tuple[str, str]:
    """Return (admin_key, user_cmd_key) names for a scope."""
    if scope == "group":
        return ("group_allow_admin_from", "group_user_allowed_commands")
    return ("allow_admin_from", "user_allowed_commands")


def policy_from_extra(extra: dict, scope: str) -> SlashAccessPolicy:
    """Build a policy from a platform's ``extra`` dict for one scope.

    DM scope falls back to ``group_user_allowed_commands`` ONLY for the command
    list, and only when DM didn't set its own, so operators list a shared set
    once. Admin lists are NOT cross-scope: a DM admin is not a group admin.
    """
    admin_key, cmd_key = _keys_for_scope(scope)
    admin_ids = _coerce_id_list(extra.get(admin_key))
    cmds = _coerce_command_list(extra.get(cmd_key))

    if scope == "dm" and not cmds:
        cmds = _coerce_command_list(extra.get("group_user_allowed_commands"))

    return SlashAccessPolicy(
        enabled=bool(admin_ids),
        admin_user_ids=admin_ids,
        user_allowed_commands=cmds,
    )


def policy_for_source(gateway_config: Any, source: Any) -> SlashAccessPolicy:
    """Resolve the slash-gating policy for a SessionSource.

    Returns a disabled (allow-everything) policy when gateway_config/source is
    None, the platform has no PlatformConfig, or no admin list is set for the
    scope. Gates slash commands only, never plain chat.
    """
    if gateway_config is None or source is None:
        return SlashAccessPolicy(
            enabled=False,
            admin_user_ids=frozenset(),
            user_allowed_commands=frozenset(),
        )
    platforms = getattr(gateway_config, "platforms", None)
    platform_config = None
    if platforms is not None:
        try:
            platform_config = platforms.get(source.platform)
        except Exception:
            platform_config = None
    extra = _platform_extra(platform_config)
    scope = _scope_for_chat_type(getattr(source, "chat_type", None))
    return policy_from_extra(extra, scope)


__all__ = [
    "SlashAccessPolicy",
    "policy_from_extra",
    "policy_for_source",
]
