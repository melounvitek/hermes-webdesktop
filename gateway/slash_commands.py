"""Gateway slash-command handlers for GatewayRunner.

The in-session slash commands dispatched from ``_handle_message``, lifted out of ``gateway/run.py``
into a mixin so every ``self._handle_*_command`` reference keeps working via the MRO. Cohesive
clusters live in sibling mixins this class inherits: ``slash_commands_model`` (/model, /reasoning,
/fast, ...), ``slash_commands_session`` (/new, /resume, /branch, /compress, ...),
``slash_commands_status`` (/status, /context, /usage, ...) and ``slash_commands_goals`` (/goal,
/loop, /heartbeat, ...). This module keeps the shared helpers plus the remaining one-off commands.
run.py helpers (``_hermes_home``, ``_load_gateway_config``, ...) are imported lazily inside handler
bodies — a deferred ``from gateway.run import ...`` avoids the import cycle.
"""

from __future__ import annotations

import asyncio
import dataclasses
import inspect
import logging
import os
import re
import shlex
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

from agent.i18n import t
from gateway.config import HomeChannel, Platform, PlatformConfig, persist_home_channel
from gateway.platforms.base import EphemeralReply, MessageEvent
from gateway.session import AsyncSessionStore
from gateway.slash_commands_goals import GatewayGoalCommandsMixin
from gateway.slash_commands_model import (  # noqa: F401 — _model_switch_skew_guard re-exported for tests
    GatewayModelCommandsMixin,
    _model_switch_skew_guard,
)
from gateway.slash_commands_session import GatewaySessionCommandsMixin
from gateway.slash_commands_status import GatewayStatusCommandsMixin
from hermes_cli.config import atomic_config_write, cfg_get
from utils import atomic_json_write, is_truthy_value

logger = logging.getLogger("gateway.run")


# /rollback result keys -> i18n line for files the safe restore left alone.
_ROLLBACK_SKIP_LINES = (
    ("skipped_user_edits", "gateway.rollback.kept_user_edits"),
    ("skipped_oversize", "gateway.rollback.kept_oversize"),
    ("failed_deletes", "gateway.rollback.failed_deletes"),
)

# /busy input modes -> (status-card behavior, set-confirmation behavior).
_BUSY_MODE_BEHAVIOR = {
    "queue": ("queues for next turn", "Messages will be queued for the next turn while Hermes is busy."),
    "steer": (
        "steers into current run (after next tool call)",
        "Messages will be steered into the current run (after the next tool call).",
    ),
    "interrupt": ("interrupts current run", "Messages will interrupt the current run while Hermes is busy."),
}

# /diff argument -> diff mode (unknown args leave the mode unchanged).
_DIFF_MODE_BY_ARG = {
    "staged": "staged", "--staged": "staged", "cached": "staged", "--cached": "staged",
    "all": "all", "--all": "all", "head": "all",
    "session": "session",
}

# /voice subcommand -> stored mode (None = auto-TTS disabled), confirmation i18n key.
_VOICE_MODE_BY_ARG = {
    "on": ("voice_only", "gateway.voice.enabled_voice_only"),
    "enable": ("voice_only", "gateway.voice.enabled_voice_only"),
    "off": ("off", "gateway.voice.disabled_text"),
    "disable": ("off", "gateway.voice.disabled_text"),
    "tts": ("all", "gateway.voice.tts_enabled"),
}


def _nested_dict(root: dict, *keys: str) -> dict:
    """Walk/create ``root[k1][k2]...`` as dicts, replacing any non-dict value on the path."""
    current = root
    for k in keys:
        if not isinstance(current.get(k), dict):
            current[k] = {}
        current = current[k]
    return current


def _restart_notify_payload(event: MessageEvent) -> dict:
    """Requester routing info so the new gateway process can notify them once back online."""
    source = event.source
    notify_data = {
        "platform": source.platform.value if source.platform else None,
        "chat_id": source.chat_id,
        "chat_type": source.chat_type,
    }
    if source.delivered_via_upstream_relay is True:
        notify_data["delivered_via_upstream_relay"] = True
        if source.user_id:
            notify_data["user_id"] = source.user_id
        if source.scope_id:
            notify_data["scope_id"] = source.scope_id
    if source.thread_id:
        notify_data["thread_id"] = source.thread_id
    if event.message_id:
        notify_data["message_id"] = event.message_id
    return notify_data


_WINDOWS_UPDATE_HELPER = """
import os, subprocess, sys
output_path = sys.argv[1]
exit_code_path = sys.argv[2]
cmd = sys.argv[3:]
env = dict(os.environ)
env["PYTHONUNBUFFERED"] = "1"
with open(output_path, "wb") as f:
    proc = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, env=env)
    rc = proc.wait(timeout=3600)
with open(exit_code_path, "w", encoding="utf-8") as f:
    f.write(str(rc))
""".strip()


def _spawn_detached_update(hermes_cmd, output_path, exit_code_path) -> None:
    """Spawn ``hermes update --gateway`` detached so it survives the gateway restart it may trigger.

    setsid is portable (works where ``systemd-run --user`` lacks a D-Bus session); ``--gateway``
    enables file-based IPC for interactive prompts so the gateway forwards them instead of skipping;
    PYTHONUNBUFFERED lets the gateway stream output in near-real-time. Windows has no setsid chain:
    an inline Python helper runs the command, redirects both outputs to the same file and writes the
    exit code. It invokes the updater as a module under this interpreter rather than through
    hermes_cmd (venv\Scripts\hermes.exe): the shim launcher holds its own file open for the whole
    run, and the update has to replace it.
    """
    import shutil
    import subprocess

    if sys.platform == "win32":
        from hermes_cli._subprocess_compat import windows_detach_popen_kwargs

        subprocess.Popen(
            [
                sys.executable, "-c", _WINDOWS_UPDATE_HELPER,
                str(output_path), str(exit_code_path),
                sys.executable, "-m", "hermes_cli.main",
                "update", "--gateway",
            ],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **windows_detach_popen_kwargs(),
        )
        return
    hermes_cmd_str = " ".join(shlex.quote(part) for part in hermes_cmd)
    update_cmd = (
        f"PYTHONUNBUFFERED=1 {hermes_cmd_str} update --gateway"
        f" > {shlex.quote(str(output_path))} 2>&1; "
        # Avoid `status=$?`: `status` is read-only in zsh and this template is reused in
        # macOS/zsh operator wrappers, so keep it zsh-safe even though bash runs it here.
        f"rc=$?; printf '%s' \"$rc\" > {shlex.quote(str(exit_code_path))}"
    )
    setsid_bin = shutil.which("setsid")
    # Preferred: setsid creates a new session, fully detached; fallback start_new_session=True
    # calls os.setsid() in the child.
    argv = [setsid_bin, "bash", "-c", update_cmd] if setsid_bin else ["bash", "-c", update_cmd]
    subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)


def _home_thread_from_source(source) -> Optional[str]:
    """The thread id /sethome should persist on the home target, or None.

    Slack thread-per-message keying stamps a top-level message's own id as ``source.thread_id`` (a
    session key, not a location); persisting it would pin HOME to that ephemeral thread. A thread
    id equal to the message's own id is synthetic and dropped; a real thread (id = parent's) is kept.
    """
    thread_id = getattr(source, "thread_id", None)
    if not thread_id:
        return None
    if (
        getattr(source, "platform", None) == Platform.SLACK
        and getattr(source, "message_id", None)
        and str(thread_id) == str(source.message_id)
    ):
        return None
    return str(thread_id)


class GatewaySlashCommandsMixin(
    GatewayModelCommandsMixin,
    GatewaySessionCommandsMixin,
    GatewayStatusCommandsMixin,
    GatewayGoalCommandsMixin,
):
    """In-session slash-command handlers for GatewayRunner (plus the helpers the sibling mixins share)."""

    async_session_store: AsyncSessionStore

    # ------------------------------------------------------------------ shared helpers
    def _cached_agent_for(self, session_key: str):
        """Peek the cached AIAgent for *session_key* without evicting it, or None.

        Cache entries are ``(agent, signature, ...)`` tuples; bare agents (test doubles) are
        accepted too. Lock/cache may be absent on fixtures that skip ``__init__``.
        """
        cache = getattr(self, "_agent_cache", None)
        if cache is None:
            return None
        lock = getattr(self, "_agent_cache_lock", None)
        try:
            if lock is not None:
                with lock:
                    entry = cache.get(session_key)
            else:
                entry = cache.get(session_key)
        except Exception:
            return None
        if isinstance(entry, (tuple, list)):
            return entry[0] if entry else None
        return entry or None

    def _resident_agent_for(self, session_key: str):
        """The live running agent for *session_key*, else the cached one, else None.

        The pending sentinel (a run that is starting) never counts as a usable agent.
        """
        from gateway.run import _AGENT_PENDING_SENTINEL

        agent = self._running_agents.get(session_key)
        if agent is not None and agent is not _AGENT_PENDING_SENTINEL:
            return agent
        return self._cached_agent_for(session_key)

    @staticmethod
    def _session_db_unavailable_reply() -> str:
        from hermes_state import format_session_db_unavailable
        return format_session_db_unavailable(prefix=t("gateway.shared.session_db_unavailable_prefix"))

    def _reply_metadata(self, event: MessageEvent):
        """Thread/reply metadata for an outbound send anchored on *event*."""
        return self._thread_metadata_for_source(event.source, self._reply_anchor_for_event(event))

    def _adapter_and_key_for(self, event: MessageEvent):
        """``(adapter, session_key)`` for the event's source, either None when no source."""
        if not event.source:
            return None, None
        return self.adapters.get(event.source.platform), self._session_key_for_source(event.source)

    def _telegramized_command_reply(self, event: MessageEvent, text: str) -> str:
        from gateway.run import _telegramize_command_mentions
        return _telegramize_command_mentions(
            text, getattr(getattr(event, "source", None), "platform", None)
        )

    def _checkpoint_manager(self):
        """A CheckpointManager from gateway config, or None when checkpoints are disabled."""
        from gateway.run import _checkpoint_agent_kwargs, _load_gateway_config
        from tools.checkpoint_manager import CheckpointManager

        cp_kwargs = _checkpoint_agent_kwargs(_load_gateway_config())
        if not cp_kwargs["checkpoints_enabled"]:
            return None
        return CheckpointManager(
            enabled=True,
            max_snapshots=cp_kwargs["checkpoint_max_snapshots"],
            max_total_size_mb=cp_kwargs["checkpoint_max_total_size_mb"],
            max_file_size_mb=cp_kwargs["checkpoint_max_file_size_mb"],
        )

    def _write_approval_setter(self, section: str, session_key: str):
        """``set_mode_fn`` for /memory and /skills: persist ``<section>.write_approval``.

        Write-back round-trip: raw read is correct (merged defaults must not be persisted back to
        the user's file). The new setting must take effect next message, so the cached agent is dropped.
        """
        from gateway.run import _gateway_config_home
        from hermes_cli.config import read_user_config_raw
        config_path = _gateway_config_home() / "config.yaml"

        def _set_approval(enabled: bool):
            user_config = read_user_config_raw(config_path)
            user_config.setdefault(section, {})["write_approval"] = bool(enabled)
            atomic_config_write(config_path, user_config)
            self._evict_cached_agent(session_key)

        return _set_approval

    async def _deliver_approval_confirmation(self, event: MessageEvent, confirmation_text: str, verb: str):
        """Return *confirmation_text* for normal delivery, or push it on native-streaming adapters.

        Native-streaming adapters (WeCom msgtype:"stream") need the confirmation sent directly with
        control-lane metadata (reliable proactive send, not the finalized reply stream). Everyone
        else returns text for normal delivery. (``is not True``: mocks auto-create attrs.)
        """
        source = event.source
        adapter = self.adapters.get(source.platform)
        if adapter:
            adapter.resume_typing_for_chat(source.chat_id)  # agent is about to continue
        if getattr(adapter, "SUPPORTS_NATIVE_STREAMING", False) is not True:
            return confirmation_text
        if adapter:
            try:
                await adapter.send(
                    source.chat_id,
                    confirmation_text,
                    reply_to=event.message_id,
                    metadata={"is_approval_prompt": True, "force_proactive_send": True},
                )
            except Exception as exc:
                logger.warning(
                    "Failed to send /%s confirmation to %s: %s", verb, source.chat_id, exc, exc_info=True,
                )
        return None

    def _typed_command_prefix_for(self, platform) -> str:
        """Return the prefix users can always type to reach Hermes commands.

        Adapter ``typed_command_prefix`` capability (default "/"). Slack and Matrix use "!" because
        typed "/" is blocked/reserved there; their adapters rewrite "!command" to "/command".
        """
        adapter = self.adapters.get(platform) if getattr(self, "adapters", None) else None
        return getattr(adapter, "typed_command_prefix", "/") if adapter is not None else "/"

    async def _handle_profile_command(self, event: MessageEvent) -> str:
        """Handle /profile — show the profile serving this source and its home.

        On a multiplexed gateway the process-level active profile is the multiplexer's own, so it
        would read "default" in every chat. With ``multiplex_profiles`` on, report ``source.profile``
        and resolve home under that profile's runtime scope (like the scoped /reset banner); when
        off the stamp is ignored, mirroring ``_run_agent``.
        """
        from hermes_constants import display_hermes_home
        from hermes_cli.slash_exec import CommandContext, execute_command

        multiplexed = getattr(
            getattr(self, "config", None), "multiplex_profiles", False
        )
        source = getattr(event, "source", None)

        profile_name = ""
        display = ""
        if multiplexed:
            profile_name = (getattr(source, "profile", "") or "").strip()
            try:
                from gateway.run import _profile_runtime_scope

                profile_home = self._resolve_profile_home_for_source(source)
                with _profile_runtime_scope(profile_home):
                    display = display_hermes_home()
            except Exception:
                display = display_hermes_home()

        # Shared executor resolves process-level fallbacks; the multiplexed
        # per-source overrides (when any) ride in via options.
        reply = execute_command(
            "profile",
            CommandContext(
                surface="gateway",
                options={"profile_name": profile_name, "home_display": display},
            ),
        )

        lines = [
            t("gateway.profile.header", profile=reply.data["profile"]),
            t("gateway.profile.home", home=reply.data["home"]),
        ]

        return "\n".join(lines)

    async def _handle_whoami_command(self, event: MessageEvent) -> str:
        """Handle /whoami — show the user's slash command access on this scope.

        Always allowed (slash_access floor). Reports platform, DM-vs-group scope, tier, and the
        commands the user can actually run here.
        """
        from gateway.slash_access import policy_for_source as _policy_for_source

        source = event.source
        policy = _policy_for_source(self.config, source)
        platform = source.platform.value if source and source.platform else "?"
        chat_type = (source.chat_type if source else "") or "dm"
        scope = "DM" if chat_type.lower() in {"dm", "direct", "private", ""} else "group/channel"
        user_id = (source.user_id if source else None) or "?"

        head = f"**You** — {platform} ({scope})\nUser ID: `{user_id}`\n"
        if not policy.enabled:
            return head + "Tier: unrestricted (no admin list configured for this scope)\nSlash commands: all available"
        if policy.is_admin(user_id):
            return head + "Tier: **admin**\nSlash commands: all available"
        # Non-admin user: show what's actually reachable. Floor first (mirrors
        # slash_access._ALWAYS_ALLOWED_FOR_USERS), then operator additions, deduped in order.
        runnable = list(dict.fromkeys(["help", "whoami"] + sorted(policy.user_allowed_commands)))
        runnable_str = ", ".join(f"/{c}" for c in runnable) if runnable else "(none)"
        return head + f"Tier: user\nSlash commands you can run: {runnable_str}"

    async def _handle_kanban_command(self, event: MessageEvent) -> str:
        """Handle /kanban — delegate to the shared kanban CLI.

        DB work runs in a thread pool to keep the event loop responsive. Reads and mutations are
        allowed while an agent runs: the board is profile-agnostic and never touches agent state.
        """
        from hermes_cli.kanban import run_slash

        text = (event.text or "").strip()
        # Strip the leading "/kanban" (with or without slash), leaving args.
        if text.startswith("/"):
            text = text.lstrip("/")
        if text.startswith("kanban"):
            text = text[len("kanban"):].lstrip()

        tokens = shlex.split(text) if text else []
        requested_board = None
        action = None
        i = 0
        while i < len(tokens):
            tok = tokens[i]
            if tok == "--board":
                if i + 1 >= len(tokens):
                    break
                requested_board = tokens[i + 1]
                i += 2
                continue
            if tok.startswith("--board="):
                requested_board = tok.split("=", 1)[1]
                i += 1
                continue
            action = tok
            break

        is_create = action == "create"

        try:
            output = await asyncio.to_thread(run_slash, text)
        except Exception as exc:  # pragma: no cover - defensive
            return t("gateway.kanban.error_prefix", error=exc)

        # Auto-subscribe on create. Parse the task id from the CLI's standard success line ("Created
        # t_abcd  (ready, assignee=...)"). If the user passed --json we don't subscribe; they're
        # clearly scripting and can call /kanban notify-subscribe explicitly.
        if is_create and output:
            m = re.search(r"Created\s+(t_[0-9a-f]+)\b", output)
            if m:
                task_id = m.group(1)
                try:
                    if await self._kanban_auto_subscribe(event, task_id, requested_board):
                        output = (
                            output.rstrip()
                            + "\n"
                            + t("gateway.kanban.subscribed_suffix", task_id=task_id)
                        )
                except Exception as exc:
                    logger.warning("kanban create auto-subscribe failed: %s", exc)

        # Gateway messages have practical length caps; truncate long
        # listings to keep the UX reasonable.
        if len(output) > 3800:
            output = output[:3800] + "\n" + t("gateway.kanban.truncated_suffix")
        return output or t("gateway.kanban.no_output")

    async def _kanban_auto_subscribe(self, event: MessageEvent, task_id: str, requested_board) -> bool:
        """Subscribe the event's chat to *task_id* notifications (notify+wake). False when the
        source has no platform/chat to route back to."""
        source = event.source
        platform = getattr(source, "platform", None)
        platform_str = (platform.value if hasattr(platform, "value") else str(platform or "")).lower()
        chat_id = str(getattr(source, "chat_id", "") or "")
        chat_type = str(getattr(source, "chat_type", "") or "") or None
        thread_id = str(getattr(source, "thread_id", "") or "")
        user_id = str(getattr(source, "user_id", "") or "") or None
        # Also persist the stable alt id (Signal UUID, Feishu union_id): build_session_key keys the
        # participant on ``user_id_alt or user_id``, so a replayed wake rebuilds the same session
        # key only when the alt id survives the round-trip.
        user_id_alt = str(getattr(source, "user_id_alt", "") or "") or None
        delivery_metadata = self._reply_metadata(event) or None
        if isinstance(delivery_metadata, dict):
            chat_type = str(getattr(source, "chat_type", "") or "")
            if chat_type:
                delivery_metadata.setdefault("chat_type", chat_type)
        if not (platform_str and chat_id):
            return False

        def _sub():
            from hermes_cli import kanban_db as _kb
            conn = _kb.connect(board=requested_board)
            try:
                _kb.add_notify_sub(
                    conn, task_id=task_id,
                    platform=platform_str, chat_id=chat_id,
                    chat_type=chat_type,
                    thread_id=thread_id or None,
                    user_id=user_id,
                    user_id_alt=user_id_alt,
                    notifier_profile=getattr(self, "_kanban_notifier_profile", None) or self._active_profile_name(),
                    # Subscribing from chat: deliver the passive message and wake the destination agent.
                    delivery_mode="notify+wake",
                    delivery_metadata=delivery_metadata,
                )
            finally:
                conn.close()
        await asyncio.to_thread(_sub)
        return True

    async def _handle_stop_command(self, event: MessageEvent) -> Union[str, EphemeralReply]:
        """Handle /stop command - interrupt a running agent.

        When an agent is truly hung (blocked thread that never checks _interrupt_requested), the
        early intercept in _handle_message() handles /stop before this method is reached; this
        handler fires only via normal dispatch (no running agent) or as a fallback, and force-cleans
        the session lock in all cases. The session is preserved so the user can continue.
        """
        from gateway.run import _AGENT_PENDING_SENTINEL, _INTERRUPT_REASON_STOP
        source = event.source
        session_entry = await self.async_session_store.get_or_create_session(source)
        session_key = session_entry.session_key

        async def _stop(key: str, invalidation_reason: str) -> None:
            await self._interrupt_and_clear_session(
                key, source, interrupt_reason=_INTERRUPT_REASON_STOP, invalidation_reason=invalidation_reason,
            )

        agent = self._running_agents.get(session_key)
        if agent is _AGENT_PENDING_SENTINEL:
            # Force-clean the sentinel so the session is unlocked.
            await _stop(session_key, "stop_command_pending")
            logger.info("STOP (pending) for session %s — sentinel cleared", session_key)
            return EphemeralReply(t("gateway.stop.stopped_pending"))
        if agent:
            # Force-clean the session lock so a truly hung agent doesn't keep it locked forever.
            await _stop(session_key, "stop_command_handler")
            return EphemeralReply(t("gateway.stop.stopped"))

        # No run under the caller's own key. In a per-user thread (thread_sessions_per_user=True) a
        # run another user started lives under a different key, yet authorized users must still be
        # able to /stop it: fall back to sibling runs in this thread, gated on authorization.
        sibling_keys = self._sibling_thread_run_keys(source, session_key)
        if sibling_keys and self._is_user_authorized(source):
            for sibling_key in sibling_keys:
                await _stop(sibling_key, "stop_command_thread_sibling")
            logger.info(
                "STOP (thread sibling) by %s — interrupted %d run(s) in thread: %s",
                session_key,
                len(sibling_keys),
                ", ".join(sibling_keys),
            )
            return EphemeralReply(t("gateway.stop.stopped"))

        # No running agent anywhere for this scope. A platform status indicator can still be stuck —
        # e.g. Slack's persistent assistant.threads.setStatus survives a gateway restart or a turn
        # that died without a final send.
        adapter = getattr(self, "adapters", {}).get(source.platform)
        if adapter and hasattr(adapter, "_stop_typing_with_metadata"):
            try:
                await adapter._stop_typing_with_metadata(source.chat_id, self._reply_metadata(event))
            except Exception:
                logger.debug(
                    "Failed to clear typing on /stop with no active agent",
                    exc_info=True,
                )

        return t("gateway.stop.no_active")

    async def _handle_platform_command(self, event: MessageEvent) -> str:
        """Handle ``/platform list|pause|resume [name]`` — inspect and manually control failed/paused
        gateway adapters (pause stops the reconnect watcher; resume re-queues for retry).
        """
        text = (getattr(event, "content", "") or "").strip()
        # Strip the leading "/platform" (or "/PLATFORM") token if present
        parts = text.split(maxsplit=2)
        if parts and parts[0].lower().lstrip("/").startswith("platform"):
            parts = parts[1:]
        action = (parts[0] if parts else "list").lower()
        target = parts[1].lower() if len(parts) > 1 else ""

        failed = getattr(self, "_failed_platforms", {}) or {}
        if action == "list":
            connected = sorted(p.value for p in self.adapters)
            lines = ["**Gateway platforms**", "Connected: " + (", ".join(connected) if connected else "(none)")]
            for p, info in failed.items():
                if info.get("paused"):
                    reason = info.get("pause_reason") or "paused"
                    lines.append(f"  · {p.value} — PAUSED ({reason}). Resume with `/platform resume {p.value}`.")
                else:
                    lines.append(f"  · {p.value} — retrying (attempt {info.get('attempts', 0)})")
            if not failed:
                lines.append("Failed/paused: (none)")
            return "\n".join(lines)

        if action in {"pause", "resume"}:
            if not target:
                return f"Usage: /platform {action} <name>"
            # Resolve platform name (case-insensitive, value match)
            platform = next((p for p in Platform.__members__.values() if p.value.lower() == target), None)
            if platform is None:
                return f"Unknown platform: {target}"
            if action == "pause":
                if platform not in failed:
                    return (
                        f"{platform.value} is not in the retry queue "
                        f"(it's either connected or not enabled)."
                    )
                if failed[platform].get("paused"):
                    return f"{platform.value} is already paused."
                self._pause_failed_platform(platform, reason="paused via /platform pause")
                return (
                    f"✓ {platform.value} paused. "
                    f"Resume with `/platform resume {platform.value}` or "
                    f"`hermes gateway restart` to reset."
                )
            # action == "resume"
            if platform not in failed:
                return (
                    f"{platform.value} is not in the retry queue — "
                    f"nothing to resume."
                )
            if not failed[platform].get("paused"):
                return (
                    f"{platform.value} is already retrying — "
                    f"no resume needed."
                )
            self._resume_paused_platform(platform)
            return f"✓ {platform.value} resumed — retrying on next watcher tick."

        return (
            "Usage: /platform <list|pause|resume> [name]\n"
            "  /platform list — show platform status\n"
            "  /platform pause <name> — stop retrying a failing platform\n"
            "  /platform resume <name> — re-queue a paused platform"
        )

    async def _handle_restart_command(self, event: MessageEvent) -> Union[str, EphemeralReply]:
        """Handle /restart command - drain active work, then restart the gateway."""
        from gateway.run import _hermes_home
        # Idempotency check: if the previous gateway process recorded this same /restart (platform +
        # update_id) and we see it *again*, it's a redelivery from PTB's graceful-shutdown get_updates
        # ACK failing on the way out. Ignoring it prevents a loop where every fresh gateway re-restarts.
        if self._is_stale_restart_redelivery(event):
            logger.info(
                "Ignoring redelivered /restart (platform=%s, update_id=%s) — "
                "already processed by a previous gateway instance.",
                event.source.platform.value if event.source and event.source.platform else "?",
                event.platform_update_id,
            )
            return ""

        if self._restart_requested or self._draining:
            count = self._running_agent_count()
            if count:
                return t("gateway.draining", count=count)
            return EphemeralReply(t("gateway.restart.in_progress"))

        # Save the requester's routing info so the new gateway process can
        # notify them once it comes back online.
        try:
            notify_data = _restart_notify_payload(event)
            try:
                self._restart_command_source = dataclasses.replace(
                    event.source,
                    message_id=str(event.message_id)
                    if event.message_id is not None
                    else event.source.message_id,
                )
            except Exception:
                self._restart_command_source = event.source
            await asyncio.to_thread(
                atomic_json_write,
                _hermes_home / ".restart_notify.json",
                notify_data,
                indent=None,
            )
        except Exception as e:
            logger.debug("Failed to write restart notify file: %s", e)

        # Record the triggering platform + update_id in a dedicated dedup marker. Unlike
        # .restart_notify.json (unlinked once the new gateway sends its notification) this persists
        # so a delayed Telegram redelivery is still detectable. Overwritten on every /restart.
        try:
            dedup_data = {
                "platform": event.source.platform.value if event.source.platform else None,
                "requested_at": time.time(),
            }
            if event.platform_update_id is not None:
                dedup_data["update_id"] = event.platform_update_id
            await asyncio.to_thread(
                atomic_json_write,
                _hermes_home / ".restart_last_processed.json",
                dedup_data,
                indent=None,
            )
        except Exception as e:
            logger.debug("Failed to write restart dedup marker: %s", e)

        active_agents = self._running_agent_count()
        # Under a service manager (systemd/launchd) or Docker/Podman, exit 75 so the supervisor /
        # restart policy restarts us — detached setsid+bash fails there (systemd KillMode=mixed kills
        # the cgroup; tini exits with the gateway). The explicit marker covers ``sudo env -i`` wrappers.
        from gateway.restart import (
            is_container_restart_context,
            is_gateway_supervisor_process,
        )

        _under_service = is_gateway_supervisor_process()
        _in_container = is_container_restart_context()
        if _under_service or _in_container:
            self.request_restart(detached=False, via_service=True)
        else:
            self.request_restart(detached=True, via_service=False)
        if active_agents:
            return t("gateway.draining", count=active_agents)
        return EphemeralReply(t("gateway.restart.restarting"))

    async def _handle_version_command(self, event: MessageEvent) -> str:
        """Handle /version — show the running Hermes Agent version."""
        from hermes_cli.slash_exec import CommandContext, execute_command

        return execute_command("version", CommandContext(surface="gateway")).text

    async def _handle_help_command(self, event: MessageEvent) -> str:
        """Handle /help command - list available commands."""
        from hermes_cli.slash_exec import CommandContext, execute_command

        reply = execute_command("help", CommandContext(surface="gateway"))
        return self._telegramized_command_reply(event, reply.text)

    async def _handle_commands_command(self, event: MessageEvent) -> str:
        from hermes_cli.slash_exec import CommandContext, execute_command

        # Page size is a surface parameter (Telegram messages are shorter).
        page_size = 15 if event.source.platform == Platform.TELEGRAM else 20
        reply = execute_command(
            "commands",
            CommandContext(
                surface="gateway",
                args=event.get_command_args(),
                options={"page_size": page_size},
            ),
        )
        return self._telegramized_command_reply(event, reply.text)

    async def _handle_set_home_command(self, event: MessageEvent) -> str:
        """Handle /sethome command -- set the current chat as the platform's home channel."""
        from gateway.run import _home_target_env_var, _home_thread_env_var
        source = event.source
        platform_name = source.platform.value if source.platform else "unknown"
        chat_id = source.chat_id
        chat_name = source.chat_name or chat_id
        if source.platform is None:
            return t("gateway.set_home.save_failed", error="Missing logical platform")

        via_relay = getattr(source, "delivered_via_upstream_relay", False) is True
        if via_relay:
            adapter_for_source = getattr(self, "_adapter_for_source", None)
            relay_adapter = adapter_for_source(source) if callable(adapter_for_source) else None
            fronts_platform = getattr(relay_adapter, "fronts_platform", None)
            if (
                source.platform in {None, Platform.LOCAL, Platform.RELAY}
                or not getattr(source, "user_id", None)
                or not callable(fronts_platform)
                or not fronts_platform(source.platform)
            ):
                return t(
                    "gateway.set_home.save_failed",
                    error="Relay does not authenticate this logical home target",
                )

        thread_id = _home_thread_from_source(source)
        home = HomeChannel(
            platform=source.platform,
            chat_id=str(chat_id),
            name=chat_name,
            thread_id=str(thread_id) if thread_id else None,
            user_id=(
                str(source.user_id)
                if getattr(source, "user_id", None)
                else None
            ),
            scope_id=(
                str(source.scope_id)
                if getattr(source, "scope_id", None)
                else None
            ),
        )

        # config.yaml is canonical because it can persist the authenticated
        # logical-target provenance required by Relay after a restart.
        try:
            persist_home_channel(home, enabled_if_new=not via_relay)
        except Exception as e:
            return t("gateway.set_home.save_failed", error=e)

        # Preserve legacy home env vars for existing cron/setup consumers.
        env_key = _home_target_env_var(platform_name)
        thread_env_key = _home_thread_env_var(platform_name)
        try:
            from hermes_cli.config import save_env_value
            save_env_value(env_key, str(chat_id))
            save_env_value(thread_env_key, str(thread_id or ""))
        except Exception as e:
            logger.warning("Home config saved but legacy env persistence failed: %s", e)

        # Keep the running gateway config in sync too. The pre-restart
        # notification path reads self.config before the process reloads config.
        platform_config = self.config.platforms.setdefault(
            source.platform,
            PlatformConfig(enabled=not via_relay),
        )
        platform_config.home_channel = home

        return t("gateway.set_home.success", name=chat_name, chat_id=chat_id)

    async def _handle_voice_command(self, event: MessageEvent) -> str:
        """Handle /voice [on|off|tts|channel|leave|status] command."""
        args = event.get_command_args().strip().lower()
        chat_id = event.source.chat_id
        # Voice state belongs to the (bot, chat) pair: resolve the adapter that
        # received the command and key the mode by its owning profile so two
        # multiplexed bots in one chat keep independent /voice state (#75198).
        voice_key = self._voice_key_for_source(event.source)

        adapter = self._adapter_for_source(event.source)

        def _set_mode(mode: str) -> None:
            self._voice_mode[voice_key] = mode
            self._save_voice_modes()
            if adapter:
                if mode == "off":
                    self._set_adapter_auto_tts_disabled(adapter, chat_id, disabled=True)
                else:
                    self._set_adapter_auto_tts_enabled(adapter, chat_id, enabled=True)

        if args in _VOICE_MODE_BY_ARG:
            mode, reply_key = _VOICE_MODE_BY_ARG[args]
            _set_mode(mode)
            return t(reply_key)
        elif args in {"channel", "join"}:
            return await self._handle_voice_channel_join(event)
        elif args == "leave":
            return await self._handle_voice_channel_leave(event)
        elif args == "status":
            mode = self._voice_mode.get(voice_key, "off")
            labels = {
                "off": t("gateway.voice.label_off"),
                "voice_only": t("gateway.voice.label_voice_only"),
                "all": t("gateway.voice.label_all"),
            }
            # Append voice channel info if connected
            guild_id = self._get_guild_id(event)
            if guild_id and hasattr(adapter, "get_voice_channel_info"):
                info = adapter.get_voice_channel_info(guild_id)
                if info:
                    lines = [
                        t("gateway.voice.status_mode", label=labels.get(mode, mode)),
                        t("gateway.voice.status_channel", channel=info['channel_name']),
                        t("gateway.voice.status_participants", count=info['member_count']),
                    ]
                    for m in info["members"]:
                        status = t("gateway.voice.speaking") if m.get("is_speaking") else ""
                        lines.append(t("gateway.voice.status_member", name=m['display_name'], status=status))
                    return "\n".join(lines)
            return t("gateway.voice.status_mode", label=labels.get(mode, mode))
        else:
            # Toggle: off → on, on/all → off
            if self._voice_mode.get(voice_key, "off") == "off":
                _set_mode("voice_only")
                toggle_line = t("gateway.voice.enabled_short")
            else:
                _set_mode("off")
                toggle_line = t("gateway.voice.disabled_short")
            # Bare /voice still toggles, but append an explainer so users discover the
            # on/off/tts/status subcommands (and, on Discord, live voice-channel join/leave). The
            # toggle result is shown first via the {toggle} placeholder.
            supports_voice_channels = adapter is not None and hasattr(
                adapter, "join_voice_channel"
            )
            channels = (
                t("gateway.voice.help_channels") if supports_voice_channels else ""
            )
            return t("gateway.voice.help", toggle=toggle_line, channels=channels)

    async def _handle_rollback_command(self, event: MessageEvent) -> str:
        """Handle /rollback command — list or restore filesystem checkpoints."""
        from tools.checkpoint_manager import format_checkpoint_list

        mgr = self._checkpoint_manager()
        if mgr is None:
            return t("gateway.rollback.not_enabled")

        from tools.terminal_scope import terminal_env as _tenv

        cwd = _tenv("TERMINAL_CWD", str(Path.home()))
        arg = event.get_command_args().strip()

        # --all / --force: classic full restore, overwriting user edits too.
        restore_all = False
        arg_parts = []
        for tok in arg.split():
            if tok.lower() in ("--all", "--force"):
                restore_all = True
            else:
                arg_parts.append(tok)
        arg = " ".join(arg_parts)

        if not arg:
            checkpoints = mgr.list_checkpoints(cwd)
            return format_checkpoint_list(checkpoints, cwd)

        # Restore by number or hash
        checkpoints = mgr.list_checkpoints(cwd)
        if not checkpoints:
            return t("gateway.rollback.none_found", cwd=cwd)

        target_hash = None
        try:
            idx = int(arg) - 1
            if 0 <= idx < len(checkpoints):
                target_hash = checkpoints[idx]["hash"]
            else:
                return t("gateway.rollback.invalid_number", max=len(checkpoints))
        except ValueError:
            target_hash = arg

        result = mgr.restore(cwd, target_hash, safe=not restore_all)
        if result["success"]:
            msg = t(
                "gateway.rollback.restored",
                hash=result["restored_to"],
                reason=result["reason"],
            )
            for result_key, i18n_key in _ROLLBACK_SKIP_LINES:
                files = result.get(result_key) or []
                if files:
                    shown = ", ".join(files[:5])
                    more = f" (+{len(files) - 5})" if len(files) > 5 else ""
                    msg += "\n" + t(i18n_key, files=shown + more)
            return msg
        return t("gateway.rollback.restore_failed", error=result["error"])

    async def _handle_diff_command(self, event: MessageEvent) -> str:
        """Handle /diff — show git changes in the working directory.

        Diff body is truncated hard here (chat is not a pager); platform senders clamp further.
        """
        args = event.get_command_args().strip()

        stat_only = False
        mode = "working"
        for arg in args.split():
            low = arg.lower()
            if low in ("--stat", "stat"):
                stat_only = True
            else:
                mode = _DIFF_MODE_BY_ARG.get(low, mode)

        from tools.terminal_scope import terminal_env as _tenv

        cwd = _tenv("TERMINAL_CWD", str(Path.home()))

        if mode == "session":
            return await self._gateway_session_diff(cwd, stat_only)

        from tools.working_diff import collect_working_diff

        result = await asyncio.to_thread(collect_working_diff, cwd, mode)
        if not result.get("success"):
            return t("gateway.diff.failed",
                     error=result.get("error", "Could not generate diff"))

        return self._render_diff_result(result, stat_only)

    async def _gateway_session_diff(self, cwd: str, stat_only: bool) -> str:
        """Cumulative checkpoint-baseline diff for /diff session (gateway)."""
        mgr = self._checkpoint_manager()
        if mgr is None:
            return t("gateway.diff.not_enabled")

        result = await asyncio.to_thread(mgr.session_diff, cwd)
        if not result.get("success"):
            return t("gateway.diff.failed",
                     error=result.get("error", "Could not generate diff"))
        return self._render_diff_result(result, stat_only)

    def _render_diff_result(self, result: dict, stat_only: bool) -> str:
        """Render a working/session diff result: stat block, untracked list, fenced (truncated) diff."""
        stat = result.get("stat", "")
        diff = result.get("diff", "")
        untracked = result.get("untracked", [])
        if result.get("empty") or (not stat and not diff and not untracked):
            return t("gateway.diff.no_changes")
        out: list[str] = []
        if stat:
            out.append(f"```\n{stat}\n```")
        if untracked:
            shown = "\n".join(f"+ {rel}" for rel in untracked[:15])
            more = f"\n... and {len(untracked) - 15} more" if len(untracked) > 15 else ""
            out.append(f"**Untracked:**\n```\n{shown}{more}\n```")
        if not stat_only and diff:
            out.append(self._fenced_truncated_diff(diff))
        return "\n\n".join(out)

    @staticmethod
    def _fenced_truncated_diff(diff: str, max_lines: int = 60,
                               max_chars: int = 3000) -> str:
        """Fence a diff body, truncating to messaging-friendly size."""
        diff_lines = diff.splitlines()
        truncated = False
        if len(diff_lines) > max_lines:
            diff = "\n".join(diff_lines[:max_lines])
            truncated = True
        if len(diff) > max_chars:
            diff = diff[:max_chars]
            truncated = True
        note = ""
        if truncated:
            note = (
                f"\n... (truncated — {len(diff_lines)} lines total; "
                "use /diff --stat for a summary)"
            )
        return f"```diff\n{diff}{note}\n```"

    def _track_background_task(self, coro) -> None:
        """Fire-and-forget *coro*, keeping a strong ref in ``_background_tasks`` until it finishes."""
        task = asyncio.create_task(coro)
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _handle_background_command(self, event: MessageEvent) -> str:
        """Handle /bg <prompt> — run a prompt in a background thread with its own session; the
        result is sent to the same chat without touching the active session's history.
        """
        prompt = event.get_command_args().strip()
        if not prompt:
            return t("gateway.background.usage")

        source = event.source
        task_id = f"bg_{datetime.now().strftime('%H%M%S')}_{os.urandom(3).hex()}"

        event_message_id = self._reply_anchor_for_event(event)

        # Forward image/audio attachments so the background agent can see them.
        media_urls = list(event.media_urls) if event.media_urls else []
        media_types = list(event.media_types) if event.media_types else []

        # Fire-and-forget the background task
        self._track_background_task(
            self._run_background_task(
                prompt,
                source,
                task_id,
                event_message_id=event_message_id,
                media_urls=media_urls,
                media_types=media_types,
            )
        )

        preview = prompt[:60] + ("..." if len(prompt) > 60 else "")
        return t("gateway.background.started", preview=preview, task_id=task_id)

    async def _handle_btw_command(self, event: MessageEvent) -> str:
        """Handle /btw <question> — answer a side question via a one-shot auxiliary LLM call on a
        transcript snapshot; live history is never touched (alternation + prompt cache intact,
        current turn keeps running). Unlike /bg, which spawns a fresh contextless session.
        """
        question = event.get_command_args().strip()
        if not question:
            return t("gateway.btw.usage")

        source = event.source
        session_entry = await self.async_session_store.get_or_create_session(source)
        history = await self.async_session_store.load_transcript(session_entry.session_id)
        if not history:
            return t("gateway.btw.no_history")

        try:
            model, runtime_kwargs = self._resolve_session_agent_runtime(
                source=source,
            )
        except Exception:
            model, runtime_kwargs = None, {}
        if not runtime_kwargs.get("api_key"):
            return t("gateway.btw.no_provider")

        main_runtime = {
            "model": model,
            "provider": runtime_kwargs.get("provider"),
            "base_url": runtime_kwargs.get("base_url"),
            "api_key": runtime_kwargs.get("api_key"),
            "api_mode": runtime_kwargs.get("api_mode"),
        }
        history_snapshot = list(history)
        # Prefer the cache-parity fork when a live cached AIAgent exists: it replays the snapshot
        # against the warm provider prefix cache, giving FULL context at cache-read prices. With no
        # cached agent the cache is cold anyway — answer_side_question's digest fallback handles it.
        try:
            parent_agent = self._cached_agent_for(self._session_key_for_source(source))
        except Exception:
            parent_agent = None
        _thread_metadata = self._reply_metadata(event)
        adapter = self._adapter_for_source(source)
        preview = question[:60] + ("..." if len(question) > 60 else "")

        async def _run_side_question() -> None:
            from agent.side_question import answer_side_question
            try:
                answer = await asyncio.to_thread(
                    answer_side_question,
                    question,
                    history_snapshot,
                    parent_agent=parent_agent,
                    main_runtime=main_runtime,
                )
            except Exception as e:
                logger.warning("/btw side question failed: %s", e)
                if adapter is not None:
                    await adapter.send(
                        source.chat_id,
                        t("gateway.btw.failed", preview=preview, error=str(e)),
                        metadata=_thread_metadata,
                    )
                return
            if adapter is not None:
                await adapter.send(
                    source.chat_id,
                    t("gateway.btw.answer", preview=preview, answer=answer or ""),
                    metadata=_thread_metadata,
                )

        self._track_background_task(_run_side_question())

        return t("gateway.btw.started", preview=preview)

    async def _handle_memory_command(self, event: MessageEvent) -> str:
        """Handle /memory — review pending memory writes + toggle the approval gate.

        Entries are small enough to review inline, so the full flow works on every platform.
        """
        from hermes_cli.write_approval_commands import handle_pending_subcommand
        from tools import write_approval as wa
        from tools.memory_tool import load_on_disk_store

        raw_args = event.get_command_args().strip()
        args = raw_args.split() if raw_args else []
        _set_approval = self._write_approval_setter("memory", self._session_key_for_source(event.source))

        # Apply approved writes against a fresh on-disk store (the gateway has
        # no long-lived agent; the store persists to the same MEMORY/USER.md).
        # load_on_disk_store() honors the user's configured char limits.
        store = load_on_disk_store()

        out = handle_pending_subcommand(
            wa.MEMORY, args, memory_store=store, set_mode_fn=_set_approval,
        )
        if out is None:
            out = ("Unknown /memory subcommand. Use: pending, approve <id>, "
                   "reject <id>, approval <on|off>.")
        return out

    async def _handle_skills_command(self, event: MessageEvent) -> str:
        """Handle /skills on the gateway — pending skill-write review only (hub stays CLI-only).

        Gated by ``skills.write_approval`` but still answers when staged writes exist after the
        gate is off (never stranded). ``diff`` is truncated for chat.
        """
        from hermes_cli.write_approval_commands import handle_pending_subcommand
        from tools import write_approval as wa

        raw_args = event.get_command_args().strip()
        args = raw_args.split() if raw_args else []

        gate_on = wa.write_approval_enabled(wa.SKILLS)
        wants_toggle = bool(args) and args[0].lower() in {"approval", "mode"}
        if not gate_on and not wants_toggle and wa.pending_count(wa.SKILLS) == 0:
            return ("Skill write approval is off (skills.write_approval). "
                    "Enable it with /skills approval on, then review staged "
                    "writes here with /skills pending.")

        out = handle_pending_subcommand(
            wa.SKILLS, args,
            set_mode_fn=self._write_approval_setter("skills", self._session_key_for_source(event.source)),
        )
        if out is None:
            return ("Unknown /skills subcommand on this platform. Use: pending, "
                    "approve <id>, reject <id>, diff <id>, approval <on|off>. "
                    "(Search/install are CLI-only.)")

        # Chat bubbles can't hold a full skill diff — truncate and point at the pending JSON file
        # (NOT `hermes skills diff <name>`, which diffs a bundled skill against its stock version).
        if args and args[0].lower() == "diff" and len(out) > 3000:
            pending_id = args[1] if len(args) > 1 else "<id>"
            out = (out[:3000]
                   + "\n… (truncated — full diff in "
                     f"~/.hermes/pending/skills/{pending_id}.json)")
        return out

    async def _handle_approvals_command(self, event: MessageEvent) -> str:
        """Show or persist the profile-wide dangerous-command approval mode."""
        from gateway.slash_access import policy_for_source
        from hermes_cli.approval_mode import run_approval_mode_command

        requested = event.get_command_args().strip() or None
        # This mutates profile-wide security policy. The central slash gate can
        # allow selected commands to non-admin users, so enforce admin again at
        # this side-effect boundary. Unconfigured policies remain unrestricted.
        policy = policy_for_source(self.config, event.source)
        if requested and not policy.is_admin(event.source.user_id):
            return "Only gateway admins can change the persistent approval mode."
        result = run_approval_mode_command(requested)
        # Approval checks load config dynamically; do not evict the cached agent
        # or alter its system prompt/tool schema (prompt-cache prefix is sacred).
        return result.message

    async def _handle_yolo_command(self, event: MessageEvent) -> Union[str, EphemeralReply]:
        """Handle /yolo — toggle dangerous command approval bypass for this session only."""
        from tools.approval import (
            disable_session_yolo,
            enable_session_yolo,
            is_session_yolo_enabled,
        )

        session_key = self._session_key_for_source(event.source)
        current = is_session_yolo_enabled(session_key)
        if current:
            disable_session_yolo(session_key)
            return EphemeralReply(t("gateway.yolo.disabled"))
        else:
            enable_session_yolo(session_key)
            return EphemeralReply(t("gateway.yolo.enabled"))

    async def _handle_verbose_command(self, event: MessageEvent) -> str:
        """Handle /verbose command — cycle tool progress display mode.

        Gated by ``display.tool_progress_command`` (default off). Cycles off → new → all → verbose
        per *current platform*, saved to ``display.platforms.<platform>.tool_progress``.
        """
        from gateway.run import _gateway_config_home, _load_gateway_config, _platform_config_key

        config_path = _gateway_config_home() / "config.yaml"
        platform_key = _platform_config_key(event.source.platform)

        try:
            user_config = _load_gateway_config()
            gate_enabled = is_truthy_value(
                cfg_get(user_config, "display", "tool_progress_command"),
                default=False,
            )
        except Exception:
            gate_enabled = False

        if not gate_enabled:
            return t("gateway.verbose.not_enabled")

        # Cycle mode (per-platform).
        cycle = ["off", "new", "all", "verbose", "log"]
        # Read current effective mode for this platform via the resolver
        from gateway.display_config import resolve_display_setting
        current = resolve_display_setting(user_config, platform_key, "tool_progress", "all")
        if current not in cycle:
            current = "all"
        new_mode = cycle[(cycle.index(current) + 1) % len(cycle)]
        description = t(f"gateway.verbose.mode_{new_mode}")

        try:
            _nested_dict(user_config, "display", "platforms", platform_key)["tool_progress"] = new_mode
            atomic_config_write(config_path, user_config)
            return f"{description}\n" + t("gateway.verbose.saved_suffix", platform=platform_key)
        except Exception as e:
            logger.warning("Failed to save tool_progress mode: %s", e)
            return f"{description}\n" + t("gateway.verbose.save_failed", error=e)

    async def _handle_busy_command(self, event: MessageEvent) -> Union[str, EphemeralReply]:
        """Handle /busy — control what happens when messaging while Hermes is working."""
        arg = event.get_command_args().strip().lower()
        if not arg or arg == "status":
            mode = self._effective_busy_input_mode(event.source)
            behavior = _BUSY_MODE_BEHAVIOR.get(mode, _BUSY_MODE_BEHAVIOR["interrupt"])[0]
            return EphemeralReply(
                f"**Busy input mode: `{mode}`" + "\n"
                f"Messages while busy: _{behavior}_" + "\n"
                f"Change with `/busy queue`, `/busy steer`, or `/busy interrupt`."
            )

        if arg not in _BUSY_MODE_BEHAVIOR:
            return EphemeralReply(
                f"Unknown mode `{arg}`. Use `/busy queue`, `/busy steer`, or `/busy interrupt`."
            )

        # Persist before mutate
        from cli import save_config_value
        if save_config_value("display.busy_input_mode", arg):
            profile_name = self._busy_profile_name_for_source(event.source)
            if profile_name:
                from gateway.run import _load_gateway_runtime_config

                self._snapshot_profile_busy_modes(
                    profile_name,
                    _load_gateway_runtime_config(),
                )
            else:
                self._busy_input_mode = arg
                # busy_input_mode is also the source of truth for the text mode — re-derive it so the
                # adapter refresh below doesn't keep a stale value and keep interrupting.
                self._busy_text_mode = self._load_busy_text_mode()

            adapter = self._adapter_for_source(event.source)
            if adapter is not None:
                adapter._busy_text_mode = self._effective_busy_text_mode(event.source)

            behavior = _BUSY_MODE_BEHAVIOR[arg][1]
            return EphemeralReply(
                f"Busy input mode set to **`{arg}`** (saved)." + "\n"
                f"_{behavior}_"
            )
        return EphemeralReply(
            f"Busy input mode could not be saved to config. Mode unchanged."
        )

    async def _handle_footer_command(self, event: MessageEvent) -> str:
        """Handle /footer command — toggle the runtime-metadata footer."""
        from gateway.run import _gateway_config_home, _load_gateway_config, _platform_config_key, _resolve_gateway_model
        from gateway.runtime_footer import resolve_footer_config

        config_path = _gateway_config_home() / "config.yaml"
        platform_key = _platform_config_key(event.source.platform)

        arg = ""
        try:
            text = (getattr(event, "message", None) or "").strip()
            if text.startswith("/"):
                parts = text.split(None, 1)
                if len(parts) > 1:
                    arg = parts[1].strip().lower()
        except Exception:
            arg = ""

        try:
            user_config: dict = _load_gateway_config()
        except Exception as e:
            return t("gateway.config_read_failed", error=e)

        effective = resolve_footer_config(user_config, platform_key)

        if arg in {"status", "?"}:
            state = t("gateway.footer.state_on") if effective["enabled"] else t("gateway.footer.state_off")
            fields = ", ".join(effective.get("fields") or [])
            return t(
                "gateway.footer.status",
                state=state,
                fields=fields,
                platform=platform_key,
            )

        if arg in {"on", "enable", "true", "1"}:
            new_state = True
        elif arg in {"off", "disable", "false", "0"}:
            new_state = False
        elif arg == "":
            new_state = not effective["enabled"]
        else:
            return t("gateway.footer.usage")

        try:
            _nested_dict(user_config, "display", "runtime_footer")["enabled"] = new_state
            atomic_config_write(config_path, user_config)
        except Exception as e:
            logger.warning("Failed to save runtime_footer.enabled: %s", e)
            return t("gateway.config_save_failed", error=e)

        state = t("gateway.footer.state_on") if new_state else t("gateway.footer.state_off")
        example = ""
        if new_state:
            # Show a preview using current agent state if available.
            from gateway.runtime_footer import format_runtime_footer
            preview = format_runtime_footer(
                model=_resolve_gateway_model(user_config) or None,
                context_tokens=0,
                context_length=None,
                fields=effective.get("fields") or ["model", "context_pct", "cwd"],
            )
            if preview:
                example = t("gateway.footer.example_line", preview=preview)
        return t("gateway.footer.saved", state=state, example=example)

    async def _handle_reload_mcp_command(self, event: MessageEvent) -> Optional[str]:
        """Handle /reload-mcp — reconnect MCP servers and rebuild the cached agent.

        Reloading invalidates the provider prompt cache (tool schemas live in the system prompt),
        so it routes through slash-confirm; "Always Approve" persists
        ``approvals.mcp_reload_confirm: false``.
        """
        source = event.source
        session_key = self._session_key_for_source(source)

        # Read the gate fresh from disk so a prior "always" click takes
        # effect on the next invocation without restarting the gateway.
        user_config = self._read_user_config()
        approvals = user_config.get("approvals") if isinstance(user_config, dict) else None
        confirm_required = True
        if isinstance(approvals, dict):
            confirm_required = bool(approvals.get("mcp_reload_confirm", True))

        if not confirm_required:
            return await self._execute_mcp_reload(event)

        # Route through slash-confirm. The primitive sends the prompt and stores the resume handler;
        # the button/text response triggers ``_resolve_slash_confirm`` which invokes the handler
        # with the chosen outcome.
        async def _on_confirm(choice: str) -> Optional[str]:
            if choice == "cancel":
                return t("gateway.reload_mcp.cancelled")
            if choice == "always":
                # Persist the opt-out and run the reload.
                try:
                    from cli import save_config_value
                    save_config_value("approvals.mcp_reload_confirm", False)
                    logger.info(
                        "User opted out of /reload-mcp confirmation (session=%s)",
                        session_key,
                    )
                except Exception as exc:
                    logger.warning("Failed to persist mcp_reload_confirm=false: %s", exc)
            # once / always → run the reload
            result = await self._execute_mcp_reload(event)
            if choice == "always":
                return f"{result}\n\n" + t("gateway.reload_mcp.always_followup")
            return result

        prompt_message = t("gateway.reload_mcp.confirm_prompt")
        return await self._request_slash_confirm(
            event=event,
            command="reload-mcp",
            title="/reload-mcp",
            message=prompt_message,
            handler=_on_confirm,
        )

    async def _handle_reload_skills_command(self, event: MessageEvent) -> str:
        """Handle /reload-skills — rescan skills dir, queue a note for next turn.

        Skills are invoked at runtime, not baked into the system prompt, so this does NOT clear the
        prompt cache. Added/removed skills go into ``_pending_skills_reload_notes[session_key]``,
        prepended to the NEXT user message — nothing out-of-band, so alternation is preserved.
        """
        try:
            from agent.skill_commands import reload_skills

            # _run_in_executor_with_context, not a bare hop: the rescan walks
            # get_hermes_home()/skills, a contextvar override under multiplex.
            result = await self._run_in_executor_with_context(reload_skills)
            added = result.get("added", [])      # [{"name", "description"}, ...]
            removed = result.get("removed", [])  # [{"name", "description"}, ...]
            total = result.get("total", 0)

            # Let adapters refresh platform-side state that cached the skill list at startup (today:
            # Discord /skill autocomplete — otherwise new skills stay invisible and deleted ones
            # error). Adapters without refresh_skill_group are skipped; the in-process reload suffices.
            for adapter in list(self.adapters.values()):
                refresh = getattr(adapter, "refresh_skill_group", None)
                if not callable(refresh):
                    continue
                try:
                    maybe = refresh()
                    if inspect.isawaitable(maybe):
                        await maybe
                except Exception as exc:
                    logger.warning(
                        "Adapter %s refresh_skill_group raised: %s",
                        getattr(adapter, "name", adapter), exc,
                    )

            lines = [t("gateway.reload_skills.header")]
            if not added and not removed:
                lines.append(t("gateway.reload_skills.no_new"))
                lines.append(t("gateway.reload_skills.total", count=total))
                return "\n".join(lines)

            def _fmt_line(item: dict) -> str:
                nm = item.get("name", "")
                desc = item.get("description", "")
                if desc:
                    return t("gateway.reload_skills.item_with_desc", name=nm, desc=desc)
                return t("gateway.reload_skills.item_no_desc", name=nm)

            # Queue a one-shot note for the next user turn in this session too. Format matches how
            # the system prompt renders pre-existing skills (``    - name: description``) so the
            # model reads the diff in the same shape as its original skill catalog.
            sections = ["[USER INITIATED SKILLS RELOAD:"]
            for i18n_key, note_header, items in (
                ("gateway.reload_skills.added_header", "Added Skills:", added),
                ("gateway.reload_skills.removed_header", "Removed Skills:", removed),
            ):
                if items:
                    lines.append(t(i18n_key))
                    lines.extend(_fmt_line(item) for item in items)
                    sections.extend(["", note_header])
                    sections.extend(_fmt_line(item) for item in items)
            lines.append(t("gateway.reload_skills.total", count=total))
            sections.append("")
            sections.append("Use skills_list to see the updated catalog.]")
            note = "\n".join(sections)

            session_key = self._session_key_for_source(event.source)
            if not hasattr(self, "_pending_skills_reload_notes"):
                self._pending_skills_reload_notes = {}
            if session_key:
                self._pending_skills_reload_notes[session_key] = note

            return "\n".join(lines)

        except Exception as e:
            logger.warning("Skills reload failed: %s", e)
            return t("gateway.reload_skills.failed", error=e)

    async def _handle_bundles_command(self, event: MessageEvent) -> str:
        """Handle /bundles — list installed skill bundles (mirrors the CLI handler).

        Bundles are loaded by invoking their own ``/<slug>`` command, not by this one.
        """
        from hermes_cli.slash_exec import CommandContext, execute_command

        reply = execute_command("bundles", CommandContext(surface="gateway"))
        if "error" in reply.data:
            logger.warning("Bundles command unavailable: %s", reply.data["error"])
            return reply.text

        bundles = reply.data["bundles"]
        if not bundles:
            return (
                "No skill bundles installed.\n"
                "Create one on the host with:\n"
                "  `hermes bundles create <name> --skill <s1> --skill <s2>`\n"
                f"Directory: `{reply.data['dir']}`"
            )

        lines = [f"**Skill Bundles** ({len(bundles)} installed):", ""]
        for info in bundles:
            skill_count = len(info.get("skills", []))
            desc = info.get("description") or f"Load {skill_count} skills"
            lines.append(
                f"• `/{info['slug']}` — {desc} _({skill_count} skills)_"
            )
            for s in info.get("skills", []):
                lines.append(f"    · {s}")
        lines.append("")
        lines.append("Invoke a bundle with `/<slug>` to load all its skills.")
        return "\n".join(lines)

    def _blocking_approval_or_stale(self, event: MessageEvent, stale_key: str, none_key: str):
        """``(session_key, None)`` when an agent thread is blocked on approval, else the reply to send.

        A pending-approvals entry with no blocked thread is a stale prompt: drop it and say so.
        """
        from tools.approval import has_blocking_approval

        session_key = self._session_key_for_source(event.source)
        if has_blocking_approval(session_key):
            return session_key, None
        if session_key in self._pending_approvals:
            self._pending_approvals.pop(session_key)
            return session_key, t(stale_key)
        return session_key, t(none_key)

    async def _handle_approve_command(self, event: MessageEvent) -> Optional[str]:
        """Handle /approve command — unblock waiting agent thread(s).

        Agent threads block inside tools/approval.py; signalling the event resumes them so the
        command executes inline — same flow as the CLI's synchronous approval.
        """
        from tools.approval import resolve_gateway_approval

        session_key, stale = self._blocking_approval_or_stale(event, "gateway.approval_expired", "gateway.approve.no_pending")
        if stale:
            return stale

        # Parse args: support "all", "all session", "all always", "session", "always"
        args = event.get_command_args().strip().lower().split()
        resolve_all = "all" in args
        remaining = [a for a in args if a != "all"]

        if any(a in {"always", "permanent", "permanently"} for a in remaining):
            choice = "always"
        elif any(a in {"session", "ses"} for a in remaining):
            choice = "session"
        else:
            choice = "once"

        count = resolve_gateway_approval(session_key, choice, resolve_all=resolve_all)
        if not count:
            return t("gateway.approve.no_pending")

        plural = "plural" if count > 1 else "singular"
        confirmation_text = t(f"gateway.approve.{choice}_{plural}", count=count)
        logger.info("User approved %d dangerous command(s) via /approve (%s)", count, choice)
        return await self._deliver_approval_confirmation(event, confirmation_text, "approve")

    async def _handle_deny_command(self, event: MessageEvent) -> str:
        """Handle /deny command — reject pending dangerous command(s).

        Signals blocked thread(s) with a 'deny' result so they get a definitive BLOCKED message,
        as in the CLI. ``/deny`` denies the oldest; ``/deny all`` denies everything.
        """
        from tools.approval import resolve_gateway_approval

        session_key, stale = self._blocking_approval_or_stale(event, "gateway.deny.stale", "gateway.deny.no_pending")
        if stale:
            return stale

        # Parse args: a leading "all" token denies every pending command;
        # anything after it (or the whole arg string when "all" is absent) is
        # captured verbatim as the optional deny reason relayed to the agent.
        raw_args = event.get_command_args().strip()
        tokens = raw_args.split()
        resolve_all = bool(tokens) and tokens[0].lower() == "all"
        reason = raw_args[len(tokens[0]):].strip() if resolve_all else raw_args
        # Cap to a sane one-liner; the agent only needs a short hint.
        if reason:
            reason = reason[:280].strip()

        count = resolve_gateway_approval(
            session_key, "deny", resolve_all=resolve_all,
            reason=reason or None,
        )
        if not count:
            return t("gateway.deny.no_pending")

        logger.info(
            "User denied %d dangerous command(s) via /deny%s",
            count, " (with reason)" if reason else "",
        )
        if reason:
            if count > 1:
                confirmation_text = t("gateway.deny.denied_reason_plural", count=count, reason=reason)
            else:
                confirmation_text = t("gateway.deny.denied_reason_singular", reason=reason)
        elif count > 1:
            confirmation_text = t("gateway.deny.denied_plural", count=count)
        else:
            confirmation_text = t("gateway.deny.denied_singular")
        return await self._deliver_approval_confirmation(event, confirmation_text, "deny")

    async def _handle_debug_command(self, event: MessageEvent) -> str:
        """Handle /debug — upload debug report (summary only) and return paste URLs.

        Uploads ONLY the summary (system info + log tails), never full logs, to protect privacy;
        use ``hermes debug share`` from the CLI for full uploads.
        """
        from hermes_cli.debug import (
            _capture_dump, collect_debug_report,
            upload_to_pastebin, _schedule_auto_delete,
            _GATEWAY_PRIVACY_NOTICE, _best_effort_sweep_expired_pastes,
        )

        # Run blocking I/O (dump capture, log reads, uploads) in a thread.
        def _collect_and_upload():
            _best_effort_sweep_expired_pastes()
            dump_text = _capture_dump()
            report = collect_debug_report(log_lines=200, dump_text=dump_text)

            urls = {}
            try:
                urls["Report"] = upload_to_pastebin(report)
            except Exception as exc:
                return t("gateway.debug.upload_failed", error=exc)

            # Schedule auto-deletion after 6 hours
            _schedule_auto_delete(list(urls.values()))

            lines = [_GATEWAY_PRIVACY_NOTICE, "", t("gateway.debug.header"), ""]
            label_width = max(len(k) for k in urls)
            for label, url in urls.items():
                lines.append(f"`{label:<{label_width}}`  {url}")

            lines.append("")
            lines.append(t("gateway.debug.auto_delete"))
            lines.append(t("gateway.debug.full_logs_hint"))
            lines.append(t("gateway.debug.share_hint"))
            return "\n".join(lines)

        # _run_in_executor_with_context, not a bare hop: this collects the profile's logs/config off
        # ``get_hermes_home()`` and uploads them to a public paste. Losing the contextvar override
        # would publish the DEFAULT profile's diagnostics from another profile's chat.
        return await self._run_in_executor_with_context(_collect_and_upload)

    async def _handle_update_command(self, event: MessageEvent) -> str:
        """Handle /update command — update Hermes Agent to the latest version.

        Spawns ``hermes update`` detached (``setsid``) so it survives the gateway restart it may
        trigger; marker files let this or the next gateway process notify the user on completion.
        """
        from gateway.run import _hermes_home, _resolve_hermes_bin
        import json
        from hermes_cli.config import is_managed, format_managed_message

        # Block non-messaging platforms (API server, webhooks, ACP)
        platform = event.source.platform
        _allowed = self._UPDATE_ALLOWED_PLATFORMS
        # Plugin platforms with allow_update_command=True are also allowed
        if platform not in _allowed:
            try:
                from gateway.platform_registry import platform_registry
                entry = platform_registry.get(platform.value)
                if not entry or not entry.allow_update_command:
                    return t("gateway.update.platform_not_messaging")
            except Exception:
                return t("gateway.update.platform_not_messaging")

        if is_managed():
            return f"✗ {format_managed_message('update Hermes Agent')}"

        project_root = Path(__file__).parent.parent.resolve()
        git_dir = project_root / '.git'

        if not git_dir.exists():
            return t("gateway.update.not_git_repo")

        hermes_cmd = _resolve_hermes_bin()
        if not hermes_cmd:
            return t("gateway.update.hermes_cmd_not_found")

        pending_path = _hermes_home / ".update_pending.json"
        output_path = _hermes_home / ".update_output.txt"
        exit_code_path = _hermes_home / ".update_exit_code"
        session_key = self._session_key_for_source(event.source)
        pending = {
            "platform": event.source.platform.value,
            "chat_id": event.source.chat_id,
            "chat_type": event.source.chat_type,
            "user_id": event.source.user_id,
            "session_key": session_key,
            "timestamp": datetime.now().isoformat(),
        }
        if event.source.thread_id:
            pending["thread_id"] = event.source.thread_id
        if event.message_id:
            pending["message_id"] = event.message_id
        _tmp_pending = pending_path.with_suffix(".tmp")
        _tmp_pending.write_text(json.dumps(pending), encoding="utf-8")
        _tmp_pending.replace(pending_path)
        exit_code_path.unlink(missing_ok=True)

        try:
            _spawn_detached_update(hermes_cmd, output_path, exit_code_path)
        except Exception as e:
            pending_path.unlink(missing_ok=True)
            exit_code_path.unlink(missing_ok=True)
            return t("gateway.update.start_failed", error=e)

        self._schedule_update_notification_watch()
        return t("gateway.update.starting")
