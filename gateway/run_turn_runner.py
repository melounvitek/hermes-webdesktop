"""Per-turn callback runner (progress/status/voice/run_sync) for the gateway agent turn.

Split out of ``gateway/run.py``; ``TurnRunner`` owns the per-turn callbacks/closures ``GatewayRunner._run_agent_inner`` binds.
``gateway.run`` internals are imported lazily inside method bodies (import cycle),
so ``patch("gateway.run.X")`` keeps intercepting them at call time.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
import asyncio
import dataclasses
import inspect
import json
import queue
import re
import threading
import time
from agent.replay_cleanup import strip_stale_dangerous_confirmations
from contextlib import suppress
from datetime import datetime
from gateway.config import Platform
from gateway.media_repair import repair_explicit_computer_use_media_paths
from gateway.platforms.base import BasePlatformAdapter
from gateway.turn_context import TurnContext
from hermes_cli.config import cfg_get
from typing import Any, Dict, List, Optional
from utils import is_truthy_value

if TYPE_CHECKING:  # string annotations only; never imported at runtime (cycle)
    from gateway.run import GatewayRunner, TurnRunner  # noqa: F401

# Log-record parity with the origin module.
logger = logging.getLogger("gateway.run")


class TurnRunner:
    """Per-turn collaborator carrying ``GatewayRunner._run_agent_inner``'s tool-progress callbacks.

    Module-global references (logger, cfg_get, BasePlatformAdapter, ...) resolve in this module.
    """

    def __init__(self, runner: "GatewayRunner", ctx: TurnContext) -> None:
        self._runner = runner
        self._ctx = ctx

    def progress_callback(self, event_type: str, tool_name: str = None, preview: str = None, args: dict = None, **kwargs):
        """Callback invoked by agent on tool lifecycle events."""
        from gateway.run import _hermes_home, _load_gateway_config, safe_schedule_threadsafe
        ctx = self._ctx
        # Failed subagent → one clean user-facing notice, handled FIRST, before every progress-queue
        # gate: platforms with tool_progress off must still hear about a dead delegation. Only
        # terminal failure statuses render (same notice rail as credit warnings); success/interrupt
        # stay quiet.
        if event_type == "subagent.complete":
            _sub_status = kwargs.get("status")
            try:
                from tools.delegate_tool import (
                    SUBAGENT_FAILURE_STATUSES,
                    format_subagent_failure_line,
                )
                if _sub_status in SUBAGENT_FAILURE_STATUSES and ctx._run_still_current():
                    _line = format_subagent_failure_line(
                        kwargs.get("goal"),
                        _sub_status,
                        error=kwargs.get("summary") or preview,
                        duration_seconds=kwargs.get("duration_seconds"),
                    )
                    safe_schedule_threadsafe(
                        self._runner._deliver_platform_notice(ctx.source, _line),
                        ctx._loop_for_step,
                        logger=logger,
                        log_message="subagent failure notice scheduling error",
                    )
            except Exception:
                logger.debug("subagent failure notice failed", exc_info=True)
            return
        # Live status line (Slack assistant status): stash the tool phrase on the adapter; the
        # _keep_typing refresh renders it. Plain dict write, safe from the sync worker thread.
        if (
            ctx._live_status_adapter is not None
            and ctx._live_status_mode != "off"
            and tool_name != "_thinking"
        ):
            try:
                if event_type == "tool.started" and tool_name and ctx._run_still_current():
                    from agent.display import build_status_phrase
                    _phrase = build_status_phrase(
                        tool_name,
                        args if ctx._live_status_mode == "full" else None,
                    )
                    ctx._live_status_adapter.set_status_text(ctx.source.chat_id, _phrase)
                elif event_type == "tool.completed":
                    # Between tools the model is genuinely "thinking"
                    # again — revert to the static default.
                    ctx._live_status_adapter.set_status_text(ctx.source.chat_id, None)
            except Exception as _ls_err:
                logger.debug("live status update failed: %s", _ls_err)
        # "log" mode: append tool.started lines to the log queue, silent in chat. Handled before
        # the progress_queue guard because log mode runs without a chat progress queue.
        if ctx.log_queue is not None:
            if event_type == "tool.started" and tool_name and tool_name != "_thinking":
                ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                preview_str = f' "{preview}"' if preview else ""
                ctx.log_queue.put(f"{ts}  {tool_name}:{preview_str}".rstrip())
            if not ctx.progress_queue:
                return
        if not ctx.progress_queue or not ctx._run_still_current():
            return

        # First-touch onboarding: the first time a tool exceeds _LONG_TOOL_THRESHOLD_S while
        # streaming every tool (progress_mode == "all"), append a one-time /verbose hint.
        if event_type == "tool.completed" and not ctx.long_tool_hint_fired[0]:
            try:
                duration = kwargs.get("duration") or 0
                if duration >= ctx._LONG_TOOL_THRESHOLD_S and ctx.progress_mode == "all":
                    from agent.onboarding import (
                        TOOL_PROGRESS_FLAG,
                        is_seen,
                        mark_seen,
                        tool_progress_hint_gateway,
                    )
                    _cfg = _load_gateway_config()
                    gate_on = is_truthy_value(
                        cfg_get(_cfg, "display", "tool_progress_command"),
                        default=False,
                    )
                    if gate_on and not is_seen(_cfg, TOOL_PROGRESS_FLAG):
                        ctx.long_tool_hint_fired[0] = True
                        ctx.progress_queue.put(tool_progress_hint_gateway())
                        mark_seen(_hermes_home / "config.yaml", TOOL_PROGRESS_FLAG)
            except Exception as _hint_err:
                logger.debug("tool-progress onboarding hint failed: %s", _hint_err)
            return

        # "_thinking" is assistant scratch text between tool calls. It is never ordinary tool
        # progress: only relay it when the platform explicitly opted into thinking_progress.
        if event_type == "_thinking" or tool_name == "_thinking":
            if not ctx._thinking_enabled:
                return
            thinking_text = preview if tool_name == "_thinking" else tool_name
            msg = f"💬 {thinking_text}" if thinking_text else None
            if msg:
                ctx.progress_queue.put(msg)
            return

        # Native task cards consume the ID-bearing tool_start/tool_complete callbacks instead;
        # name-correlated text events would duplicate cards and mispair concurrent same-tool calls.
        if ctx._native_slack_task_cards and event_type in {
            "tool.started",
            "tool.completed",
        }:
            return

        # If tool_progress is off, only _thinking passes through (above).
        # Regular tool calls are suppressed.
        if not ctx.tool_progress_enabled:
            return

        # Only act on tool.started events (ignore tool.completed, reasoning.available, etc.)
        if event_type not in {"tool.started",}:
            return

        # Never render a progress bubble for clarify: send_clarify IS the user-facing rendering, so
        # a bubble is duplication, and verbose mode would dump the raw tool-call args JSON, which
        # (progress queue drains on a background task) lands right under the rendered prompt.
        if tool_name == "clarify":
            return

        # Suppress tool-progress bubbles once the user sent `stop`: N parallel tool calls fire N
        # "tool.started" events before the interrupt check, so a late `stop` would still render
        # all N bubbles. (agent_holder[0] is the shared agent handle across nested scopes.)
        try:
            _agent_for_interrupt = ctx.agent_holder[0] if ctx.agent_holder else None
            if _agent_for_interrupt is not None and getattr(
                _agent_for_interrupt, "is_interrupted", False
            ):
                return
        except Exception:
            pass

        # "new" mode: only report when tool changes
        if ctx.progress_mode == "new" and tool_name == ctx.last_tool[0]:
            return
        ctx.last_tool[0] = tool_name

        # Build progress message with primary argument preview
        from agent.display import get_tool_emoji
        emoji = get_tool_emoji(tool_name, default="⚙️")

        # Markdown platforms (``supports_code_blocks``) fence terminal commands; plain-text ones
        # keep the compact `terminal: "cmd…"` line. No language tag: Slack mrkdwn renders it as a
        # literal first code line. Verbose shows the FULL command; "all"/"new" fence but truncate to
        # one line capped at ``tool_preview_length`` (default 40), the non-terminal preview budget.
        _code_block_full = None
        _code_block_short = None
        try:
            _progress_adapter = self._runner._adapter_for_source(ctx.source)
        except Exception:
            _progress_adapter = None
        if (
            getattr(_progress_adapter, "supports_code_blocks", False)
            and tool_name == "terminal"
            and isinstance(args, dict)
            and isinstance(args.get("command"), str)
            and args["command"].strip()
        ):
            from agent.display import get_tool_preview_max_len
            _cmd_full = args["command"].rstrip()
            # Consecutive terminal calls drop the repeated "💻 terminal" header so back-to-back
            # commands render as adjacent code blocks under one header.
            _block_header = (
                "" if ctx.last_was_terminal_block[0] else f"{emoji} {tool_name}\n"
            )
            _code_block_full = f"{_block_header}```\n{_cmd_full}\n```"
            # Single-line, capped preview for non-verbose modes.
            _pl = get_tool_preview_max_len()
            _cap = _pl if _pl > 0 else 40
            _lines = _cmd_full.splitlines()
            _cmd_short = _lines[0] if _lines else _cmd_full
            _multiline = len(_lines) > 1
            if len(_cmd_short) > _cap:
                _cmd_short = _cmd_short[:_cap - 3] + "..."
            elif _multiline:
                _cmd_short = _cmd_short + " ..."
            _code_block_short = f"{_block_header}```\n{_cmd_short}\n```"

        # Verbose mode: show detailed arguments, respects tool_preview_length
        if ctx.progress_mode == "verbose":
            if _code_block_full is not None:
                ctx.last_was_terminal_block[0] = True
                ctx.progress_queue.put(_code_block_full)
                return
            ctx.last_was_terminal_block[0] = False
            if args:
                from agent.display import get_tool_preview_max_len
                _pl = get_tool_preview_max_len()
                args_str = json.dumps(args, ensure_ascii=False, default=str)
                # tool_preview_length 0 (default) = no truncation in verbose mode; the user asked
                # for full detail and platform message-length limits handle the rest.
                if _pl > 0 and len(args_str) > _pl:
                    args_str = args_str[:_pl - 3] + "..."
                msg = f"{emoji} {tool_name}({list(args.keys())})\n{args_str}"
            elif preview:
                msg = f"{emoji} {tool_name}: \"{preview}\""
            else:
                msg = f"{emoji} {tool_name}..."
            ctx.progress_queue.put(msg)
            return

        # "all" / "new" modes: short preview capped by tool_preview_length (default 40; gateway
        # messages persist, unlike CLI spinners). Markdown terminal commands use the fence above.
        if _code_block_short is not None:
            msg = _code_block_short
            ctx.last_was_terminal_block[0] = True
        elif preview:
            from agent.display import (
                get_tool_preview_max_len,
                get_tool_verb,
                prepare_tool_preview,
                tool_verb_connector,
                verb_drops_preview,
            )
            _pl = get_tool_preview_max_len()
            _cap = _pl if _pl > 0 else 40
            _prepared_preview = prepare_tool_preview(
                tool_name,
                args,
                fallback=preview,
                max_len=_cap,
            )
            if _progress_adapter is not None:
                preview = _progress_adapter.format_tool_preview(_prepared_preview)
            else:
                preview = _prepared_preview.text
            # Friendly labels: human-phrased line for built-in tools ("🔍 Searching the web for ...")
            # by prefixing the verb onto the computed preview, so the command/url/query is kept.
            _verb = get_tool_verb(tool_name)
            if _verb:
                if verb_drops_preview(tool_name):
                    msg = f"{emoji} {_verb}"
                else:
                    msg = f"{emoji} {_verb}{tool_verb_connector(tool_name)}{preview}"
            else:
                msg = f"{emoji} {tool_name}: \"{preview}\""
            ctx.last_was_terminal_block[0] = False
        else:
            msg = f"{emoji} {tool_name}..."
            ctx.last_was_terminal_block[0] = False

        # Dedup consecutive identical progress messages (common with execute_code: same
        # boilerplate imports → identical previews).
        if msg == ctx.last_progress_msg[0]:
            ctx.repeat_count[0] += 1
            # Native-stream-progress routing: dedup updates the last line
            # in the overlay rather than sending a queue signal.
            _sc = ctx.stream_consumer_holder[0] if ctx.stream_consumer_holder else None
            if _sc is not None and getattr(_sc, "accepts_tool_progress", False):
                # Replace the last progress line with the dedup version
                _sc.on_tool_progress(f"{msg} (×{ctx.repeat_count[0] + 1})")
                return
            # Update the last line in progress_lines with a counter
            # via a special "dedup" queue message.
            ctx.progress_queue.put(("__dedup__", msg, ctx.repeat_count[0]))
            return
        ctx.last_progress_msg[0] = msg
        ctx.repeat_count[0] = 0

        # If the stream consumer is active with native streaming, inject progress into the stream
        # bubble instead of the separate progress queue.
        _sc = ctx.stream_consumer_holder[0] if ctx.stream_consumer_holder else None
        if _sc is not None and getattr(_sc, "accepts_tool_progress", False):
            _sc.on_tool_progress(msg)
            return

        ctx.progress_queue.put(msg)

    async def _send_native_task_card_progress(self, adapter) -> None:
        """Drain the progress queue into Slack-native plan/task cards.

        On any native failure, fall back to an editable in-thread message so progress stays live.
        """
        ctx = self._ctx
        tasks: Dict[str, Dict[str, str]] = {}
        task_order: List[str] = []
        fallback_msg_id: Optional[str] = None
        native_failed = False
        anonymous_seq = 0

        def _compact(value: Any, limit: int = 120) -> str:
            text = re.sub(r"\s+", " ", str(value or "")).strip()
            if len(text) <= limit:
                return text
            return text[: limit - 3].rstrip() + "..."

        def _visible_tasks() -> List[Dict[str, str]]:
            return [tasks[task_id] for task_id in task_order[-8:]]

        def _fallback_text() -> str:
            labels = {
                "in_progress": "running",
                "complete": "complete",
                "error": "error",
            }
            lines = [
                f"- {task['title']} - {labels.get(task['status'], task['status'])}"
                for task in _visible_tasks()
            ]
            return "Hermes is working\n" + "\n".join(lines)

        def _apply_native_event(raw: Any) -> bool:
            nonlocal anonymous_seq
            if not isinstance(raw, dict):
                return False
            event_type = raw.get("type")
            if event_type not in {"tool.started", "tool.completed"}:
                return False
            call_id = str(raw.get("tool_call_id") or "")
            if not call_id:
                anonymous_seq += 1
                call_id = f"anonymous_{anonymous_seq}"
            tool_name = str(raw.get("tool_name") or "tool")

            if event_type == "tool.started":
                title = tool_name
                preview = _compact(raw.get("preview"), 64)
                if preview:
                    title = f"{tool_name} - {preview}"
                if call_id not in tasks:
                    task_order.append(call_id)
                tasks[call_id] = {
                    "id": call_id,
                    "title": _compact(title),
                    "status": "in_progress",
                }
                return True

            task = tasks.get(call_id)
            if task is None:
                # Completion-only events are rare but valid on some runtimes; keep their real ID
                # instead of guessing a same-name pending call.
                task = {
                    "id": call_id,
                    "title": _compact(tool_name),
                    "status": "in_progress",
                }
                tasks[call_id] = task
                task_order.append(call_id)
            task["status"] = "error" if raw.get("is_error") else "complete"
            return True

        async def _send_or_edit_fallback() -> None:
            nonlocal fallback_msg_id
            text = _fallback_text()
            if fallback_msg_id:
                result = await adapter.edit_message(
                    chat_id=ctx.source.chat_id,
                    message_id=fallback_msg_id,
                    content=text,
                    metadata=ctx._progress_metadata,
                )
                if getattr(result, "success", False):
                    return
            result = await adapter.send(
                chat_id=ctx.source.chat_id,
                content=text,
                reply_to=ctx._progress_reply_to,
                metadata=ctx._progress_metadata,
            )
            if getattr(result, "success", False) and getattr(
                result, "message_id", None
            ):
                fallback_msg_id = str(result.message_id)
                if ctx._cleanup_progress:
                    ctx._cleanup_msg_ids.append(fallback_msg_id)

        async def _publish_native_progress() -> None:
            nonlocal native_failed
            if not tasks:
                return
            if not native_failed:
                result = await adapter.send_native_task_card_progress(
                    chat_id=ctx.source.chat_id,
                    tasks=_visible_tasks(),
                    title="Hermes is working",
                    reply_to=ctx._progress_reply_to,
                    metadata=ctx._progress_metadata,
                    fallback_text=_fallback_text(),
                )
                if getattr(result, "success", False):
                    return
                native_failed = True
                logger.warning(
                    "Slack native task-card progress failed; falling back "
                    "to an editable text update: %s",
                    getattr(result, "error", "unknown error"),
                )
            # Once the native rail fails, every later lifecycle event
            # edits the same fallback message so progress remains live.
            await _send_or_edit_fallback()

        def _drain_native_queue() -> bool:
            changed = False
            while True:
                try:
                    changed = _apply_native_event(
                        ctx.progress_queue.get_nowait()
                    ) or changed
                except queue.Empty:
                    return changed
                except Exception:
                    logger.debug(
                        "Slack native progress queue drain failed",
                        exc_info=True,
                    )
                    return changed

        def _agent_interrupted() -> bool:
            try:
                _agent = ctx.agent_holder[0] if ctx.agent_holder else None
                return bool(
                    _agent is not None and getattr(_agent, "is_interrupted", False)
                )
            except Exception:
                return False

        try:
            while True:
                if not ctx._run_still_current():
                    return
                try:
                    raw = ctx.progress_queue.get_nowait()
                except queue.Empty:
                    await asyncio.sleep(0.1)
                    continue

                if _agent_interrupted():
                    continue

                if _apply_native_event(raw):
                    await _publish_native_progress()
        except asyncio.CancelledError:
            if _drain_native_queue() and ctx._run_still_current():
                if not _agent_interrupted():
                    await _publish_native_progress()
            return
        finally:
            if hasattr(adapter, "stop_native_task_card_progress"):
                # Best-effort on the turn-cleanup path: an escaping transport exception would skip
                # final-delivery logic (cleanup awaits catch only CancelledError).
                try:
                    await adapter.stop_native_task_card_progress(
                        ctx.source.chat_id,
                        reply_to=ctx._progress_reply_to,
                        metadata=ctx._progress_metadata,
                    )
                except asyncio.CancelledError:
                    raise
                except Exception:
                    logger.debug(
                        "task-card stop failed during turn cleanup",
                        exc_info=True,
                    )

    @dataclasses.dataclass
    class _ProgressEditState:
        """Mutable editable-bubble state shared by ``send_progress_messages`` and its helpers."""
        adapter: Any
        progress_lines: list
        progress_msg_id: Any
        can_edit: bool
        _progress_len_fn: Any
        _PROGRESS_TEXT_LIMIT: int
        _edit_accepts_metadata: bool

    def _progress_edit_state(self, adapter) -> "TurnRunner._ProgressEditState":
        ctx = self._ctx
        progress_lines = []      # Accumulated tool lines for the CURRENT editable bubble
        progress_msg_id = None   # ID of the current progress message to edit
        can_edit = ctx.progress_grouping != "separate"  # "separate" = one message per tool (pre-v0.9 behavior)

        _progress_len_fn = (
            adapter.message_len_fn
            if isinstance(adapter, BasePlatformAdapter)
            else len
        )
        try:
            _raw_progress_limit = int(getattr(adapter, "MAX_MESSAGE_LENGTH", 4000) or 4000)
        except Exception:
            _raw_progress_limit = 4000
        # Per-chat resolution (relay adapter fronting N platforms): cap and length unit follow the
        # chat's underlying platform; native adapters return their scalar/property unchanged.
        if isinstance(adapter, BasePlatformAdapter):
            try:
                _raw_progress_limit = int(
                    adapter.max_message_length_for_chat(ctx.source.chat_id) or 4000
                )
                _progress_len_fn = adapter.message_len_fn_for_chat(ctx.source.chat_id)
            except Exception:
                pass
        # Leave a little room for platform quirks / formatting.  For tiny
        # test adapters keep the limit usable instead of clamping to 500+.
        _PROGRESS_TEXT_LIMIT = max(
            1,
            _raw_progress_limit - (64 if _raw_progress_limit > 128 else 0),
        )

        # Detect whether the adapter's edit_message accepts metadata so
        # overflow edits preserve Telegram topic/thread routing (#27487).
        _edit_accepts_metadata = False
        if ctx._progress_metadata:
            try:
                _edit_params = inspect.signature(adapter.edit_message).parameters
                _edit_accepts_metadata = (
                    "metadata" in _edit_params
                    or any(
                        param.kind is inspect.Parameter.VAR_KEYWORD
                        for param in _edit_params.values()
                    )
                )
            except (TypeError, ValueError):
                _edit_accepts_metadata = False
        return self._ProgressEditState(
            adapter=adapter,
            progress_lines=progress_lines,
            progress_msg_id=progress_msg_id,
            can_edit=can_edit,
            _progress_len_fn=_progress_len_fn,
            _PROGRESS_TEXT_LIMIT=_PROGRESS_TEXT_LIMIT,
            _edit_accepts_metadata=_edit_accepts_metadata,
        )

    async def _edit_progress_message(self, st, message_id: str, content: str):
        ctx = self._ctx
        kwargs = {
            "chat_id": ctx.source.chat_id,
            "message_id": message_id,
            "content": content,
        }
        if getattr(st.adapter, "REQUIRES_EDIT_FINALIZE", False):
            kwargs["finalize"] = True
        if st._edit_accepts_metadata:
            kwargs["metadata"] = ctx._progress_metadata
        return await st.adapter.edit_message(**kwargs)

    def _progress_text(self, lines: list) -> str:
        return "\n".join(str(line) for line in lines)

    def _split_progress_groups(self, st, lines: list) -> list[list]:
        """Partition progress lines into platform-sized editable bubbles."""
        groups: list[list] = []
        current: list = []
        for line in lines:
            candidate = current + [line]
            if current and st._progress_len_fn(self._progress_text(candidate)) > st._PROGRESS_TEXT_LIMIT:
                groups.append(current)
                current = [line]
            else:
                current = candidate
        if current:
            groups.append(current)
        return groups

    def _track_progress_result(self, result) -> None:
        ctx = self._ctx
        if (
            ctx._cleanup_progress
            and getattr(result, "success", False)
            and getattr(result, "message_id", None)
        ):
            ctx._cleanup_msg_ids.append(str(result.message_id))

    async def _send_progress_text(self, st, text: str):
        ctx = self._ctx
        result = await st.adapter.send(
            chat_id=ctx.source.chat_id,
            content=text,
            reply_to=ctx._progress_reply_to,
            metadata=ctx._progress_metadata,
        )
        self._track_progress_result(result)
        return result

    async def _roll_progress_overflow_if_needed(self, st) -> bool:
        """Start fresh editable progress bubbles before a bubble exceeds limit.

                Returns True when it delivered/split the buffer or a transient edit failure left it
                intact for retry — either way the caller skips the normal send/edit path this tick.
                """
        if not st.progress_lines or not st.can_edit:
            return False
        groups = self._split_progress_groups(st, st.progress_lines)
        if len(groups) <= 1:
            return False

        first_text = self._progress_text(groups[0])
        if st.progress_msg_id is not None:
            result = await self._edit_progress_message(st, st.progress_msg_id, first_text)
            if not result.success:
                if getattr(result, "retryable", False):
                    logger.debug(
                        "[%s] Transient overflow edit failure — keeping can_edit=True",
                        st.adapter.name,
                    )
                    return True
                st.can_edit = False
                # Fall back to the existing non-edit behavior below.
                return False
        else:
            result = await self._send_progress_text(st, first_text)
            if result.success and result.message_id:
                st.progress_msg_id = result.message_id

        for group in groups[1:]:
            result = await self._send_progress_text(st, self._progress_text(group))
            if result.success and result.message_id:
                st.progress_msg_id = result.message_id

        # The newest continuation is the only mutable bubble: keep just its lines so later
        # edits update it instead of replaying the full transcript into new messages.
        st.progress_lines = groups[-1]
        return True

    async def _drain_progress_on_cancel(self, st) -> None:
        ctx = self._ctx
        # Drain remaining queued messages
        while not ctx.progress_queue.empty():
            try:
                raw = ctx.progress_queue.get_nowait()
                if isinstance(raw, tuple) and len(raw) == 3 and raw[0] == "__dedup__":
                    _, base_msg, count = raw
                    if st.progress_lines:
                        st.progress_lines[-1] = f"{base_msg} (×{count + 1})"
                        await self._roll_progress_overflow_if_needed(st)
                elif isinstance(raw, tuple) and len(raw) >= 1 and raw[0] == "__reset__":
                    # Content-bubble marker during drain: close the current progress bubble
                    # and start a fresh one for tool lines that arrived after.
                    await self._roll_progress_overflow_if_needed(st)
                    if st.can_edit and st.progress_lines and st.progress_msg_id:
                        _pending_text = self._progress_text(st.progress_lines)
                        with suppress(Exception):
                            await self._edit_progress_message(st, st.progress_msg_id, _pending_text)
                    st.progress_msg_id = None
                    st.progress_lines = []
                    ctx.last_progress_msg[0] = None
                    ctx.repeat_count[0] = 0
                else:
                    st.progress_lines.append(raw)
                    await self._roll_progress_overflow_if_needed(st)
            except Exception:
                break
        # Final edit with all remaining tools (only if editing works)
        if st.can_edit and st.progress_lines and st.progress_msg_id:
            await self._roll_progress_overflow_if_needed(st)
        if st.can_edit and st.progress_lines and st.progress_msg_id:
            full_text = self._progress_text(st.progress_lines)
            with suppress(Exception):
                await self._edit_progress_message(st, st.progress_msg_id, full_text)

    async def send_progress_messages(self):
        ctx = self._ctx
        if not ctx.progress_queue:
            return

        adapter = self._runner._adapter_for_source(ctx.source)
        if not adapter:
            return

        if ctx._native_slack_task_cards and hasattr(
            adapter, "send_native_task_card_progress"
        ):
            await self._send_native_task_card_progress(adapter)
            return

        # Skip tool progress for platforms that can't edit messages (e.g. iMessage/BlueBubbles):
        # each update would be a separate bubble. getattr, not attribute access: duck-typed
        # adapters (test fakes, minimal plugins) may lack edit_message — treated as "can't edit".
        _adapter_edit = getattr(type(adapter), "edit_message", None)
        if _adapter_edit is None or _adapter_edit is BasePlatformAdapter.edit_message:
            while not ctx.progress_queue.empty():
                try:
                    ctx.progress_queue.get_nowait()
                except Exception:
                    break
            return

        st = self._progress_edit_state(adapter)
        _last_edit_ts = 0.0      # Throttle edits to avoid Telegram flood control
        _PROGRESS_EDIT_INTERVAL = 1.5  # Minimum seconds between edits

        while True:
            try:
                if not ctx._run_still_current():
                    while not ctx.progress_queue.empty():
                        try:
                            ctx.progress_queue.get_nowait()
                        except Exception:
                            break
                    return

                raw = ctx.progress_queue.get_nowait()

                # Drain silently when interrupted: events queued in the window between tool parse
                # and interrupt processing should not render as bubbles.
                try:
                    _agent_for_interrupt = ctx.agent_holder[0] if ctx.agent_holder else None
                    if _agent_for_interrupt is not None and getattr(
                        _agent_for_interrupt, "is_interrupted", False
                    ):
                        # Drop this event and continue draining.
                        await asyncio.sleep(0)
                        continue
                except Exception:
                    pass

                # Handle dedup messages: update last line with repeat counter
                if isinstance(raw, tuple) and len(raw) == 3 and raw[0] == "__dedup__":
                    _, base_msg, count = raw
                    if st.progress_lines:
                        st.progress_lines[-1] = f"{base_msg} (×{count + 1})"
                    msg = st.progress_lines[-1] if st.progress_lines else base_msg
                elif isinstance(raw, tuple) and len(raw) >= 1 and raw[0] == "__reset__":
                    # Content bubble landed — close the tool-progress bubble so the next tool starts
                    # fresh below it; else tool edits hit the ORIGINAL message above (out of order).
                    st.progress_msg_id = None
                    st.progress_lines = []
                    ctx.last_progress_msg[0] = None
                    ctx.repeat_count[0] = 0
                    continue
                else:
                    msg = raw
                    st.progress_lines.append(msg)

                if await self._roll_progress_overflow_if_needed(st):
                    _last_edit_ts = time.monotonic()
                    await asyncio.sleep(0.3)
                    if ctx._run_still_current():
                        await st.adapter.send_typing(ctx.source.chat_id, metadata=ctx._progress_metadata)
                    continue

                # Throttle edits: batch rapid tool updates into fewer API calls to avoid Telegram
                # flood control (grammY pattern: proactively rate-limit rather than react to 429s).
                _now = time.monotonic()
                _remaining = _PROGRESS_EDIT_INTERVAL - (_now - _last_edit_ts)
                if _remaining > 0:
                    # Wait out the throttle interval, then loop back to drain any further queued
                    # messages before sending a single batched edit.
                    await asyncio.sleep(_remaining)
                    continue

                if not ctx._run_still_current():
                    return

                if st.can_edit and st.progress_msg_id is not None:
                    # Try to edit the existing progress message
                    full_text = "\n".join(st.progress_lines)
                    result = await self._edit_progress_message(st, st.progress_msg_id, full_text)
                    if not result.success:
                        _err = (getattr(result, "error", "") or "").lower()
                        # Transient network errors (ConnectError, timeouts) must not disable editing;
                        # only permanent failures (flood, not found, permissions) set can_edit = False.
                        if getattr(result, "retryable", False):
                            logger.debug(
                                "[%s] Transient edit failure — keeping can_edit=True",
                                st.adapter.name,
                            )
                            continue
                        if "flood" in _err or "retry after" in _err:
                            # Flood control hit — backoff but keep editing.
                            # Only disable edits for non-recoverable errors.
                            logger.info(
                                "[%s] Progress edit flood control, backing off",
                                st.adapter.name,
                            )
                            _last_edit_ts = time.monotonic()
                        else:
                            st.can_edit = False
                        _flood_result = await st.adapter.send(
                            chat_id=ctx.source.chat_id,
                            content=msg,
                            reply_to=ctx._progress_reply_to,
                            metadata=ctx._progress_metadata,
                        )
                        if (
                            ctx._cleanup_progress
                            and getattr(_flood_result, "success", False)
                            and getattr(_flood_result, "message_id", None)
                        ):
                            ctx._cleanup_msg_ids.append(str(_flood_result.message_id))
                else:
                    if st.can_edit:
                        # First tool: send all accumulated text as new message
                        full_text = "\n".join(st.progress_lines)
                        result = await st.adapter.send(
                            chat_id=ctx.source.chat_id,
                            content=full_text,
                            reply_to=ctx._progress_reply_to,
                            metadata=ctx._progress_metadata,
                        )
                    else:
                        # Editing unsupported: send just this line
                        result = await st.adapter.send(
                            chat_id=ctx.source.chat_id,
                            content=msg,
                            reply_to=ctx._progress_reply_to,
                            metadata=ctx._progress_metadata,
                        )
                    if result.success and result.message_id:
                        st.progress_msg_id = result.message_id
                        if ctx._cleanup_progress:
                            ctx._cleanup_msg_ids.append(str(result.message_id))

                _last_edit_ts = time.monotonic()

                # Restore typing indicator
                await asyncio.sleep(0.3)
                if ctx._run_still_current():
                    await st.adapter.send_typing(ctx.source.chat_id, metadata=ctx._progress_metadata)

            except queue.Empty:
                await asyncio.sleep(0.3)
            except asyncio.CancelledError:
                await self._drain_progress_on_cancel(st)
                return
            except Exception as e:
                logger.error("Progress message error: %s", e)
                await asyncio.sleep(1)

    def voice_ack_callback(self, call_id, tool_name, args):
        """tool_start_callback: speak a one-time ack in the voice channel."""
        from gateway.run import safe_schedule_threadsafe
        ctx = self._ctx
        if ctx._voice_ack_fired[0] or ctx._voice_ack_guild[0] is None:
            return
        if not ctx._run_still_current():
            return
        ctx._voice_ack_fired[0] = True
        _adapter = self._runner.adapters.get(Platform.DISCORD)
        if _adapter is None or not hasattr(_adapter, "play_ack_in_voice"):
            return
        try:
            safe_schedule_threadsafe(
                _adapter.play_ack_in_voice(ctx._voice_ack_guild[0]),
                ctx._voice_ack_loop,
                logger=logger,
                log_message="voice ack scheduling error",
            )
        except Exception as _ack_err:
            logger.debug("voice ack schedule failed: %s", _ack_err)

    # ── Slack-native task cards: ID-bearing lifecycle callbacks ── ride agent.tool_start_callback /
    # agent.tool_complete_callback so start/completion correlate by the REAL tool-call id; the
    # name-correlated progress_callback text events would duplicate cards and mispair concurrent calls.

    def native_tool_start_callback(self, call_id, tool_name, args):
        """Queue an ID-correlated native progress start from the agent thread."""
        ctx = self._ctx
        if not ctx.progress_queue or not ctx._run_still_current():
            return
        try:
            _agent = ctx.agent_holder[0] if ctx.agent_holder else None
            if _agent is not None and getattr(_agent, "is_interrupted", False):
                return
        except Exception:
            pass
        from agent.display import build_tool_preview

        ctx.progress_queue.put(
            {
                "type": "tool.started",
                "tool_call_id": str(call_id or ""),
                "tool_name": str(tool_name or "tool"),
                "preview": build_tool_preview(
                    str(tool_name or "tool"), args or {}, max_len=64
                )
                or "",
            }
        )

    def native_tool_complete_callback(self, call_id, tool_name, args, result):
        """Queue the matching native completion using the real tool-call ID."""
        ctx = self._ctx
        if not ctx.progress_queue or not ctx._run_still_current():
            return
        try:
            _agent = ctx.agent_holder[0] if ctx.agent_holder else None
            if _agent is not None and getattr(_agent, "is_interrupted", False):
                return
        except Exception:
            pass
        from agent.display import _detect_tool_failure

        is_error, _ = _detect_tool_failure(str(tool_name or "tool"), result)
        ctx.progress_queue.put(
            {
                "type": "tool.completed",
                "tool_call_id": str(call_id or ""),
                "tool_name": str(tool_name or "tool"),
                "is_error": bool(is_error),
            }
        )

    def combined_tool_start_callback(self, call_id, tool_name, args):
        """Compose the voice ack + native task-card start consumers."""
        ctx = self._ctx
        if ctx._voice_ack_guild[0] is not None:
            self.voice_ack_callback(call_id, tool_name, args)
        if ctx._native_slack_task_cards:
            self.native_tool_start_callback(call_id, tool_name, args)

    def _step_callback_sync(self, iteration: int, prev_tools: list) -> None:
        from gateway.run import safe_schedule_threadsafe
        ctx = self._ctx
        if not ctx._run_still_current():
            return
        # prev_tools may be list[str] or list[dict] with "name"/"result" keys. Normalise so
        # "tool_names" stays backward-compatible for user hooks that do ', '.join(tool_names).
        _names: list[str] = []
        for _t in (prev_tools or []):
            if isinstance(_t, dict):
                _names.append(_t.get("name") or "")
            else:
                _names.append(str(_t))
        safe_schedule_threadsafe(
            ctx._hooks_ref.emit("agent:step", {
                "platform": ctx.source.platform.value if ctx.source.platform else "",
                "user_id": ctx.source.user_id,
                "session_id": ctx.session_id,
                "iteration": iteration,
                "tool_names": _names,
                "tools": prev_tools,
            }),
            ctx._loop_for_step,
            logger=logger,
            log_message="agent:step hook scheduling error",
        )

    def _event_callback_sync(self, event_type: str, context: dict) -> None:
        ctx = self._ctx
        try:
            asyncio.run_coroutine_threadsafe(
                ctx._hooks_ref.emit(event_type, context),
                ctx._loop_for_step,
            )
        except Exception as _e:
            logger.debug("event_callback hook error: %s", _e)

    def _attach_session_title_callback(self, agent, ctx) -> None:
        """Wire the platform thread-rename lane onto the agent as `_on_session_title`.

        The titler runs in the turn prologue, so attach before the run, not after it.
        """
        try:
            # Gateway auto-title failures are not user-actionable, so never surface them as messages;
            # overriding the failure sink keeps CLI on _emit_auxiliary_failure while gateway logs debug.
            def _title_failure_cb(task: str, exc: BaseException) -> None:
                logger.debug(
                    "Gateway auto-title failure suppressed (not user-visible): %s: %s",
                    task, exc,
                )

            agent._title_failure_callback = _title_failure_cb

            session_id = getattr(agent, "session_id", None)
            source = ctx.source

            # Both lanes spend a rate-limited platform call per title, so they use the model's title
            # only (TitleCallback); renaming twice burns Discord's 2-per-10-min budget on a throwaway.
            if self._runner._is_telegram_topic_lane(source):
                agent._on_session_title = lambda title, title_source: (
                    title_source == "llm"
                    and self._runner._schedule_telegram_topic_title_rename(
                        source, session_id, title,
                    )
                )
            elif self._runner._is_discord_auto_thread_lane(source) or (
                self._runner._is_relay_discord_channel_lane(source)
            ):
                # Relay note: the second predicate is shape-only (relay Discord channel event).
                # Whether the connector auto-threaded our reply is only knowable AFTER delivery, so
                # the callback must be registered eagerly and the rename lane does the cache lookup
                # at fire time — gating registration on the cache read meant it never registered.
                agent._on_session_title = lambda title, title_source: (
                    title_source == "llm"
                    and self._runner._schedule_discord_semantic_thread_rename(
                        source, session_id, title,
                    )
                )
        except Exception:
            logger.debug("Failed to attach session title callback", exc_info=True)

    def _status_callback_sync(self, event_type: str, message: str) -> None:
        from gateway.run import (
            _prepare_gateway_status_message,
            _redact_gateway_user_facing_secrets,
            _send_or_update_status_coro,
            safe_schedule_threadsafe,
        )
        ctx = self._ctx
        if not ctx._status_adapter or not ctx._run_still_current():
            return
        prepared_message = _prepare_gateway_status_message(
            ctx.source.platform,
            event_type,
            message,
        )
        if prepared_message is None:
            logger.debug(
                "status_callback suppressed for %s/%s: %s",
                ctx.source.platform.value if ctx.source.platform else "unknown",
                event_type,
                _redact_gateway_user_facing_secrets(str(message or ""))[:160],
            )
            return
        _fut = safe_schedule_threadsafe(
            _send_or_update_status_coro(ctx._status_adapter, ctx._status_chat_id, event_type, prepared_message, ctx._status_thread_metadata),
            ctx._loop_for_step,
            logger=logger,
            log_message=f"status_callback ({event_type}) scheduling error",
        )
        if _fut is None:
            return
        if ctx._cleanup_progress:
            def _track_status_id(fut) -> None:
                try:
                    res = fut.result()
                except Exception:
                    return
                mid = getattr(res, "message_id", None)
                if getattr(res, "success", False) and mid:
                    ctx._cleanup_msg_ids.append(str(mid))
            _fut.add_done_callback(_track_status_id)

    def _setup_stream_consumer(self, platform_key):
        from gateway.run import safe_schedule_threadsafe
        ctx = self._ctx
        # Set up stream consumer for token streaming or interim commentary.
        _stream_consumer = None
        _stream_delta_cb = None
        # streaming TTS consumer is created on the outer event-loop thread before run_sync launches.
        # run_sync only reads it via ``streaming_tts_consumer_holder[0]`` for delta callback wiring.
        _stts_consumer_ref = ctx.streaming_tts_consumer_holder[0]
        _scfg = getattr(getattr(self._runner, 'config', None), 'streaming', None)
        if _scfg is None:
            from gateway.config import StreamingConfig
            _scfg = StreamingConfig()

        # Per-platform streaming gate: display.platforms.<plat>.streaming can disable streaming
        # for specific platforms even when the global streaming config is enabled.
        _plat_streaming = ctx.resolve_display_setting(
            ctx.user_config, platform_key, "streaming"
        )
        # None = no per-platform override → follow global config
        _streaming_enabled = (
            _scfg.enabled and _scfg.transport != "off"
            if _plat_streaming is None
            else bool(_plat_streaming)
        )
        _want_stream_deltas = _streaming_enabled
        _want_interim_messages = ctx.interim_assistant_messages_enabled
        _want_interim_consumer = _want_interim_messages
        if _want_stream_deltas or _want_interim_consumer:
            try:
                from gateway.stream_consumer import GatewayStreamConsumer
                _adapter = self._runner._adapter_for_source(ctx.source)
                if _adapter:
                    _consumer_cfg, _pause_typing_before_finalize = (
                        self._runner._build_stream_consumer_config(
                            ctx.source, _scfg, _adapter,
                            on_missing_cursor="raise",
                        )
                    )
                    _stream_consumer = GatewayStreamConsumer(
                        adapter=_adapter,
                        chat_id=ctx.source.chat_id,
                        config=_consumer_cfg,
                        metadata=ctx._status_thread_metadata,
                        on_new_message=(
                            (lambda: ctx.progress_queue.put(("__reset__",)))
                            if ctx.progress_queue is not None
                            else None
                        ),
                        on_before_finalize=_pause_typing_before_finalize,
                        initial_reply_to_id=ctx.event_message_id,
                        run_still_current=ctx._run_still_current,
                    )
                    if _want_stream_deltas:
                        def _stream_delta_cb(text: str) -> None:
                            if ctx._run_still_current():
                                _stream_consumer.on_delta(text)
                                # Tee to the streaming-TTS consumer (#60671).
                                if _stts_consumer_ref is not None:
                                    _stts_consumer_ref.on_delta(text)
                    ctx.stream_consumer_holder[0] = _stream_consumer
            except Exception as _sc_err:
                logger.debug("Could not set up stream consumer: %s", _sc_err)

        # Text streaming off but streaming TTS active: install a TTS-only delta callback so the
        # consumer still receives LLM deltas for audio synthesis.
        if _stream_delta_cb is None and _stts_consumer_ref is not None:
            def _stream_delta_cb(text: str) -> None:
                if ctx._run_still_current():
                    _stts_consumer_ref.on_delta(text)

        def _interim_assistant_cb(text: str, *, already_streamed: bool = False) -> None:
            if not ctx._run_still_current():
                return
            display_text = text
            if _stream_consumer is not None:
                if already_streamed:
                    _stream_consumer.on_segment_break()
                else:
                    _stream_consumer.on_commentary(display_text)
                return
            if already_streamed or not ctx._status_adapter or not str(display_text or "").strip():
                return
            safe_schedule_threadsafe(
                ctx._status_adapter.send(
                    ctx._status_chat_id,
                    display_text,
                    metadata=ctx._status_thread_metadata,
                ),
                ctx._loop_for_step,
                logger=logger,
                log_message="interim_assistant_callback scheduling error",
            )
        return _stream_consumer, _stream_delta_cb, _interim_assistant_cb, _want_interim_messages

    def _resolve_turn_agent(
        self, turn_route, platform_key, combined_ephemeral, max_iterations, reasoning_config, pr,
    ):
        from gateway.run import _AGENT_PENDING_SENTINEL, _checkpoint_agent_kwargs
        ctx = self._ctx
        # Per-platform skip_context_files — messaging platforms can opt out of filesystem-heavy
        # context-file discovery (SOUL.md, AGENTS.md, .cursorrules) to cut AIAgent build latency.
        _platforms_gw_cfg = (ctx.user_config.get("gateway") or {}).get("platforms") or {}
        # ``hermes gateway setup`` writes ``gateway.platforms`` as a LIST of enabled platform names,
        # not a dict; treat any non-dict shape as "no per-platform overrides" rather than crashing.
        if not isinstance(_platforms_gw_cfg, dict):
            _platforms_gw_cfg = {}
        _plat_gw_cfg = _platforms_gw_cfg.get(platform_key) or {}
        _skip_context = _plat_gw_cfg.get("skip_context_files")
        skip_context_files = bool(_skip_context) if _skip_context is not None else False

        # Agent cache: reuse this session's previous AIAgent to preserve the frozen system prompt
        # and tool schemas for prompt cache hits.
        _sig = self._runner._agent_config_signature(
            turn_route["model"],
            turn_route["runtime"],
            ctx.enabled_toolsets,
            combined_ephemeral,
            cache_keys=self._runner._extract_cache_busting_config(ctx.user_config),
            user_id=getattr(ctx.source, "user_id", None),
            user_id_alt=getattr(ctx.source, "user_id_alt", None),
            skip_context_files=skip_context_files,
        )
        agent = None
        reused_cached_agent = False
        _cache_lock = getattr(self._runner, "_agent_cache_lock", None)
        _cache = getattr(self._runner, "_agent_cache", None)

        # Peek at the cached entry's snapshot session_id so we can check, OUTSIDE the cache lock,
        # whether it is a DEAD session in state.db. "cached sid != current sid" normally means an
        # intentional switch (reuse the agent), but the routing-key self-heal yields the same shape
        # with an agent bound to a DEAD session; reusing it re-binds the dead sid and loops.
        _peek_cached_sid = None
        if _cache_lock and _cache is not None:
            with _cache_lock:
                _peek_entry = _cache.get(ctx.session_key)
            if _peek_entry and len(_peek_entry) > 3:
                _peek_cached_sid = _peek_entry[3]
        _cached_sid_is_dead = False
        if (
            _peek_cached_sid is not None
            and ctx.session_id is not None
            and _peek_cached_sid != ctx.session_id
        ):
            try:
                _cached_sid_is_dead = self._runner.session_store._is_session_ended_in_db(
                    _peek_cached_sid
                )
            except Exception:
                _cached_sid_is_dead = False

        # Cross-process write guard: another process (e.g. hermes dashboard) appending to the same
        # SessionDB session makes the cached agent's transcript stale. On message_count mismatch vs
        # the count recorded at cache time, invalidate so a fresh agent re-reads from disk.
        _current_msg_count = None
        if self._runner._session_db is not None and ctx.session_id:
            try:
                # run_sync is off-loop (executor); sync DB is fine.
                _sess_row = self._runner._session_db._db.get_session(ctx.session_id)
                if _sess_row:
                    _current_msg_count = _sess_row.get("message_count", 0)
            except Exception:
                pass

        _xproc_evicted_agent = None
        if _cache_lock and _cache is not None:
            with _cache_lock:
                cached = _cache.get(ctx.session_key)
                if cached and cached[1] == _sig:
                    # cached[2] is the message_count at cache time; stale when a second process
                    # appended rows. cached[3] (when present) is the session_id the snapshot was
                    # taken for — used to skip the guard when the active session_id differs.
                    _cached_mc = cached[2] if len(cached) > 2 else None
                    _cached_sid = cached[3] if len(cached) > 3 else None
                    # Snapshot from a different session_id (same session_key, other conversation): the
                    # counts track DIFFERENT DB rows, so the comparison is meaningless. REUSE the cached
                    # agent rather than rebuild and bust the prompt cache on every session switch.
                    _session_id_mismatch = (
                        _cached_sid is not None
                        and ctx.session_id is not None
                        and _cached_sid != ctx.session_id
                    )
                    # Re-validate the OUTSIDE-lock dead-session peek against the tuple read under THIS
                    # lock: the entry may have been replaced between peek and acquisition, and a stale
                    # "dead" verdict must never be applied to a different (possibly live) cached agent.
                    _stale_dead_sid_reuse = (
                        _session_id_mismatch
                        and _cached_sid_is_dead
                        and _cached_sid == _peek_cached_sid
                    )
                    if _stale_dead_sid_reuse:
                        # The routing key was just self-healed away from a session state.db marked
                        # ended, but this cached AIAgent still belongs to that DEAD session_id.
                        # Reusing it would re-bind the dead sid and undo the self-heal; rebuild fresh.
                        logger.info(
                            "Agent cache invalidated for session %s: "
                            "cached agent's session_id %s is ended in "
                            "state.db (stale self-heal artifact, "
                            "#54878 x #54947) — discarding instead of "
                            "reusing across the routing recovery",
                            ctx.session_key, _cached_sid,
                        )
                        evicted = self._runner._agent_cache.pop(ctx.session_key, None)
                        _ev_agent = evicted[0] if isinstance(evicted, tuple) and evicted else None
                        if _ev_agent and _ev_agent is not _AGENT_PENDING_SENTINEL:
                            # Same deferred-cleanup rationale as the cross-process branch below: don't
                            # block the event loop / cache lock on memory-provider or socket teardown.
                            _xproc_evicted_agent = _ev_agent
                    elif (
                        not _session_id_mismatch
                        and _cached_mc is not None
                        and _current_msg_count is not None
                        and _current_msg_count != _cached_mc
                    ):
                        # Cross-process write detected — discard stale
                        # agent so it rebuilds from fresh DB transcript.
                        logger.info(
                            "Agent cache invalidated for session %s: "
                            "message_count changed (%s -> %s), "
                            "possible cross-process write",
                            ctx.session_key, _cached_mc, _current_msg_count,
                        )
                        evicted = self._runner._agent_cache.pop(ctx.session_key, None)
                        _ev_agent = evicted[0] if isinstance(evicted, tuple) and evicted else None
                        if _ev_agent and _ev_agent is not _AGENT_PENDING_SENTINEL:
                            # Defer cleanup until AFTER the lock is released: release_clients can
                            # block on memory-provider/socket teardown, stalling the event loop while
                            # the idle sweeper waits on this lock (blocking Discord heartbeats). The
                            # session rebuilds a fresh agent below, so use the SOFT release that keeps
                            # its terminal sandbox / browser / bg processes for the new agent to
                            # inherit — mirrors _evict_cached_agent / idle-sweep.
                            _xproc_evicted_agent = _ev_agent
                    else:
                        agent = cached[0]
                        # Refresh LRU order so the cap enforcement evicts
                        # truly-oldest entries, not the one we just used.
                        if hasattr(_cache, "move_to_end"):
                            with suppress(KeyError):
                                _cache.move_to_end(ctx.session_key)
                        self._runner._init_cached_agent_for_turn(agent, ctx._interrupt_depth)
                        # Refresh agent max_iterations from current config
                        # (cached agent may have been created with old config)
                        agent.max_iterations = max_iterations
                        logger.debug("Reusing cached agent for session %s", ctx.session_key)
                        reused_cached_agent = True

        # Lock released — refresh the reused agent's fallback chain from disk OUTSIDE the cache lock
        # (disk I/O under the lock stalls the idle-sweep watcher and Discord heartbeats). A chain
        # configured after caching must reach the next turn; per-session serialization keeps it safe.
        if reused_cached_agent and agent is not None:
            self._runner._apply_fallback_chain_to_agent(
                agent, self._runner._refresh_fallback_model(),
            )

        # Lock released — schedule cleanup of any cross-process-evicted agent on a daemon thread so
        # memory-provider/socket teardown never blocks the gateway loop or the expiry watcher's lock.
        if _xproc_evicted_agent is not None:
            try:
                threading.Thread(
                    target=self._runner._release_evicted_agent_soft,
                    args=(_xproc_evicted_agent,),
                    daemon=True,
                    name=f"agent-xproc-evict-{str(ctx.session_key)[:24]}",
                ).start()
            except Exception:
                # Interpreter shutdown or thread-spawn failure — release
                # inline as a best-effort fallback.
                with suppress(Exception):
                    self._runner._release_evicted_agent_soft(_xproc_evicted_agent)

        if agent is None:
            # Config changed or first message — create fresh agent
            agent = ctx.AIAgent(
                model=turn_route["model"],
                **turn_route["runtime"],
                **_checkpoint_agent_kwargs(ctx.user_config),
                max_iterations=max_iterations,
                quiet_mode=True,
                verbose_logging=False,
                enabled_toolsets=ctx.enabled_toolsets,
                disabled_toolsets=ctx.disabled_toolsets,
                ephemeral_system_prompt=combined_ephemeral or None,
                prefill_messages=self._runner._prefill_messages or None,
                reasoning_config=reasoning_config,
                service_tier=self._runner._service_tier,
                request_overrides=turn_route.get("request_overrides"),
                providers_allowed=pr.get("only"),
                providers_ignored=pr.get("ignore"),
                providers_order=pr.get("order"),
                provider_sort=pr.get("sort"),
                provider_require_parameters=pr.get("require_parameters", False),
                provider_data_collection=pr.get("data_collection"),
                session_id=ctx.session_id,
                platform=platform_key,
                user_id=ctx.source.user_id,
                user_id_alt=ctx.source.user_id_alt,
                user_name=ctx.source.user_name,
                chat_id=ctx.source.chat_id,
                chat_name=ctx.source.chat_name,
                chat_type=ctx.source.chat_type,
                thread_id=ctx.source.thread_id,
                gateway_session_key=ctx.session_key,
                session_db=getattr(self._runner._session_db, "_db", self._runner._session_db),
                # Reload from disk — do not reuse the startup snapshot (#60955).
                fallback_model=self._runner._refresh_fallback_model(),
                skip_context_files=skip_context_files,
                # Keep the persona even with minimal context: soul identity is
                # a single small file, not part of the expensive walk.
                load_soul_identity=True,
            )
            if _cache_lock and _cache is not None:
                with _cache_lock:
                    # Record the snapshot's session_id with message_count so the cross-process guard can
                    # skip the meaningless count comparison if the active session_id later switches.
                    _cache[ctx.session_key] = (
                        agent, _sig, _current_msg_count, ctx.session_id,
                    )
                    self._runner._enforce_agent_cache_cap()
            logger.debug("Created new agent for session %s (sig=%s)", ctx.session_key, _sig)
        return agent, reused_cached_agent

    def _wire_turn_agent_callbacks(
        self, agent, turn_route, reasoning_config,
        _stream_delta_cb, _interim_assistant_cb, _want_interim_messages,
    ):
        from gateway.run import (
            _interim_metadata,
            _non_conversational_metadata,
            render_notice_line,
            safe_schedule_threadsafe,
        )
        ctx = self._ctx
        # Per-message state — callbacks and reasoning config change every turn, so they aren't baked
        # into the cached agent. The progress callback is ALWAYS attached (never gated to None): its
        # body gates each event class, and subagent-failure notices must fire even with
        # tool_progress/thinking off — a None gate made dead subagents vanish silently.
        agent.tool_progress_callback = ctx.progress_callback
        # Compose ID-bearing lifecycle consumers: Discord's one-time voice ack and Slack's task cards
        # both ride the authoritative start callback, so neither infers identity from tool names.
        _combined_start_cb = ctx.native_tool_start_callback or ctx.voice_ack_callback
        agent.tool_start_callback = (
            _combined_start_cb
            if (
                ctx._voice_ack_guild[0] is not None
                or ctx._native_slack_task_cards
            )
            else None
        )
        agent.tool_complete_callback = (
            ctx.native_tool_complete_callback
            if ctx._native_slack_task_cards
            and ctx.native_tool_complete_callback is not None
            else None
        )
        agent.step_callback = ctx._step_callback_sync if ctx._hooks_ref.loaded_hooks else None
        agent.stream_delta_callback = _stream_delta_cb
        agent.interim_assistant_callback = _interim_assistant_cb if _want_interim_messages else None
        agent.status_callback = ctx._status_callback_sync
        # Credits / out-of-band notices (usage bands, depletion, restored) fire from the agent's sync
        # worker thread, so hop onto the gateway loop via safe_schedule_threadsafe. Fired-once latch
        # lives on the cached agent (no per-turn re-nag); clear is a no-op — sends can't be retracted.
        def _notice_callback_sync(notice) -> None:
            if not ctx._status_adapter or not ctx._run_still_current():
                return
            try:
                line = render_notice_line(notice)
            except Exception:
                logger.debug("render_notice_line failed", exc_info=True)
                return
            if not line:
                return
            safe_schedule_threadsafe(
                self._runner._deliver_platform_notice(ctx.source, line),
                ctx._loop_for_step,
                logger=logger,
                log_message="notice_callback delivery scheduling error",
            )

        agent.notice_callback = _notice_callback_sync
        agent.notice_clear_callback = None
        agent.event_callback = ctx._event_callback_sync
        agent.reasoning_config = reasoning_config
        agent.service_tier = self._runner._service_tier
        # Merge, never overwrite: init-time request overrides (e.g. a custom provider's extra_body
        # merged at agent construction) must survive every reused-agent turn. Drop only the PREVIOUS
        # turn's routing overrides before layering this turn's, so stale per-turn values never linger.
        request_overrides = dict(getattr(agent, "request_overrides", {}) or {})
        previous_turn_overrides = dict(
            getattr(agent, "_gateway_turn_request_overrides", {}) or {}
        )
        for key, value in previous_turn_overrides.items():
            if request_overrides.get(key) == value:
                request_overrides.pop(key, None)
        turn_request_overrides = dict(turn_route.get("request_overrides") or {})
        request_overrides.update(turn_request_overrides)
        agent.request_overrides = request_overrides
        agent._gateway_turn_request_overrides = turn_request_overrides
        # Must-deliver notes for THIS turn ride the current user message (api_content sidecar), never
        # the system prompt: staged by _handle_message_with_agent (auto-reset, first-contact intro,
        # voice-channel change). Assigned unconditionally so a reused agent never replays a stale note.
        agent._gateway_turn_context_notes = "\n\n".join(
            self._runner._consume_pending_turn_sidecar_notes(ctx.session_key)
        )

        _bg_review_release = threading.Event()
        _bg_review_pending: list[str] = []
        _bg_review_pending_lock = threading.Lock()

        def _deliver_bg_review_message(message: str) -> None:
            if not ctx._status_adapter or not ctx._run_still_current():
                return
            safe_schedule_threadsafe(
                ctx._status_adapter.send(
                    ctx._status_chat_id,
                    message,
                    metadata=_interim_metadata(_non_conversational_metadata(ctx._status_thread_metadata, platform=ctx.source.platform)),
                ),
                ctx._loop_for_step,
                logger=logger,
                log_message="background_review_callback scheduling error",
            )

        def _release_bg_review_messages() -> None:
            _bg_review_release.set()
            with _bg_review_pending_lock:
                pending = list(_bg_review_pending)
                _bg_review_pending.clear()
            for queued in pending:
                _deliver_bg_review_message(queued)

        # Background review delivery — send "💾 Memory updated" etc. to user
        def _bg_review_send(message: str) -> None:
            if not ctx._status_adapter or not ctx._run_still_current():
                return
            if not _bg_review_release.is_set():
                with _bg_review_pending_lock:
                    if not _bg_review_release.is_set():
                        _bg_review_pending.append(message)
                        return
            _deliver_bg_review_message(message)

        agent.background_review_callback = _bg_review_send
        # Register the release hook on the adapter so base.py's finally
        # block can fire it after delivering the main response.
        if ctx._status_adapter and ctx.session_key:
            if getattr(type(ctx._status_adapter), "register_post_delivery_callback", None) is not None:
                ctx._status_adapter.register_post_delivery_callback(
                    ctx.session_key,
                    _release_bg_review_messages,
                    generation=ctx.run_generation,
                )
            else:
                _pdc = getattr(ctx._status_adapter, "_post_delivery_callbacks", None)
                if _pdc is not None:
                    _pdc[ctx.session_key] = _release_bg_review_messages
        # Memory update notifications in chat.  Config: display.memory_notifications
        #   off     — no chat notification (still logged to stdout)
        #   on      — generic "💾 Memory updated" (default)
        #   verbose — content preview: "💾 Memory ➕ Hermes Repo..."
        _mem_notif = ctx.user_config.get("display", {}).get("memory_notifications")
        if isinstance(_mem_notif, bool):
            _mem_notif = "on" if _mem_notif else "off"
        agent.memory_notifications = str(_mem_notif).lower() if _mem_notif else "on"

        agent.clarify_callback = self._clarify_callback_sync

        # Show assistant thinking between tool calls — independent of tool_progress mode. Mattermost
        # needs an explicit per-platform opt-in so global scratch-text doesn't leak into threads.
        agent.thinking_progress = ctx._thinking_enabled
        # Store agent reference for interrupt support
        ctx.agent_holder[0] = agent
        # Wire the platform thread-rename lane onto the agent: the titler fires from the turn prologue,
        # not after the response, so titles are pushed the moment they land.
        self._attach_session_title_callback(agent, ctx)
        # Publish turn ownership for explicit /stop, /new, disconnect, and shutdown interrupts.
        # Older session processes are outside this baseline and remain alive.
        agent._gateway_turn_process_task_id = ctx.process_task_id
        agent._gateway_turn_process_baseline = ctx.process_baseline
        # Capture the full tool definitions for transcript logging
        ctx.tools_holder[0] = agent.tools if hasattr(agent, 'tools') else None

    # ------------------------------------------------------------------
    # Shared native-stream boundary close: for native-streaming platforms (e.g. WeCom), an
    # interrupting interaction (approval or clarify prompt) must finalize the current stream
    # and disable native streaming first, or post-interaction output keeps updating the OLD
    # bubble above the prompt. Runs on the agent thread; the consumer serializes via its queue.
    def _close_native_stream_boundary(
        self, _reason: str, _placeholder: str | None = None, _reopen: bool = False,
    ) -> bool:
        ctx = self._ctx
        _sc = ctx.stream_consumer_holder[0] if ctx.stream_consumer_holder else None
        if not (_sc and getattr(_sc, "_use_native_streaming", False)):
            return True
        _cancelled_flag = None
        try:
            _boundary_result = _sc.close_for_approval_prompt(
                _placeholder, reason=_reason, reopen=_reopen,
            )
            # Returns (future, cancelled_flag) or just a future.
            if isinstance(_boundary_result, tuple):
                _boundary_future, _cancelled_flag = _boundary_result
            else:
                _boundary_future = _boundary_result
            if hasattr(_boundary_future, "result"):
                _ok = _boundary_future.result(timeout=10)
                if not _ok:
                    logger.warning(
                        "%s boundary failed to close stream properly — "
                        "prompt may still appear in typing bubble", _reason,
                    )
                return bool(_ok)
            return True
        except (TimeoutError, Exception) as _boundary_err:
            if _cancelled_flag is not None:
                _cancelled_flag["cancelled"] = True
            logger.warning(
                "%s boundary timed out or failed: %s", _reason, _boundary_err,
            )
            return False

    # ------------------------------------------------------------------
    # Clarify callback: present a clarify prompt and block on a response. Runs on the agent's
    # worker thread (clarify_tool's synchronous contract): schedules the adapter's send_clarify
    # on the gateway loop, then blocks on the primitive's threading.Event with a timeout.
    # Returns the response string, or a sentinel explaining no response arrived.
    # ------------------------------------------------------------------
    def _clarify_callback_sync(self, question: str, choices, multi_select: bool = False) -> str:
        from gateway.run import _clarify_send_then_wait, safe_schedule_threadsafe
        ctx = self._ctx
        from tools import clarify_gateway as _clarify_mod
        import uuid as _uuid

        if not ctx._status_adapter:
            return ""

        clarify_id = _uuid.uuid4().hex[:10]
        _clarify_mod.register(
            clarify_id=clarify_id,
            session_key=ctx.session_key or "",
            question=question,
            choices=list(choices) if choices else None,
            multi_select=bool(multi_select),
        )

        # WeCom native streaming: finalize the current stream before the clarify prompt so the
        # post-answer output opens a fresh bubble below the question ("气泡割裂" otherwise). Unlike
        # approval, clarify passes reopen=True so the continuation re-opens a native stream; if
        # the re-seed fails the consumer degrades to send() automatically.
        self._close_native_stream_boundary(
            "Clarify", "💬 等待你的选择...", _reopen=True,
        )

        # Pause typing — as with approval, a "thinking..." status must not obscure the prompt or
        # block an "Other" reply on platforms that disable input while typing (Slack Assistant).
        with suppress(Exception):
            ctx._status_adapter.pause_typing_for_chat(ctx._status_chat_id)

        # Ordering barrier: flush buffered assistant prose to the platform BEFORE sending the
        # poll, which goes out on a separate agent-thread-blocking path and would otherwise
        # render ABOVE its own explanation. Best-effort + short timeout so the agent thread
        # never hangs if the consumer task isn't running.
        try:
            _sc = ctx.stream_consumer_holder[0] if ctx.stream_consumer_holder else None
            _flush = getattr(_sc, "flush_pending_sync", None)
            if callable(_flush):
                _flush(timeout=3.0)
        except Exception:
            logger.debug(
                "Stream-consumer flush before clarify prompt failed",
                exc_info=True,
            )

        fut = safe_schedule_threadsafe(
            ctx._status_adapter.send_clarify(
                chat_id=ctx._status_chat_id,
                question=question,
                choices=list(choices) if choices else None,
                clarify_id=clarify_id,
                session_key=ctx.session_key or "",
                metadata=ctx._status_thread_metadata,
            ),
            ctx._loop_for_step,
            logger=logger,
            log_message="Clarify send failed to schedule",
        )
        # Boundary rule (see _approval_send_outcome): a send timeout is AMBIGUOUS — the card may
        # have posted with a late ack. Only a definitive failure tears down the registration;
        # ambiguous falls through to the bounded wait so a late reply resolves.
        _clarify_response = _clarify_send_then_wait(
            fut,
            clarify_id=clarify_id,
            session_key=ctx.session_key or "",
            clarify_mod=_clarify_mod,
        )
        # Only re-arm typing when the user actually answered — the undeliverable sentinel and the
        # timeout/cancellation strings start with '[' and must pass through untouched.
        if not (
            isinstance(_clarify_response, str)
            and _clarify_response.startswith("[")
        ):
            # User answered: reopen typing IMMEDIATELY, not on the LLM's first post-answer token
            # (native streaming otherwise re-seeds lazily on the first delta: ~48s of dead air).
            # request_reopen_seed is a no-op outside the reopen-pending native state; always safe.
            _sc_reopen = ctx.stream_consumer_holder[0] if ctx.stream_consumer_holder else None
            if _sc_reopen is not None:
                try:
                    _sc_reopen.request_reopen_seed()
                except Exception:
                    logger.debug(
                        "request_reopen_seed after clarify answer failed",
                        exc_info=True,
                    )
            try:
                ctx._status_adapter.resume_typing_for_chat(ctx._status_chat_id)
            except Exception:
                logger.debug(
                    "resume_typing_for_chat after clarify answer failed",
                    exc_info=True,
                )
        return _clarify_response

    def _load_turn_history(self, agent, reused_cached_agent):
        from gateway.run import (
            _build_gateway_agent_history,
            _collect_history_media_paths,
            _message_timestamps_enabled,
            _select_cached_agent_history,
        )
        ctx = self._ctx
        # Convert history to agent format. Transcript path: {role, content, timestamp} dicts — strip
        # timestamps. Interrupt path (agent result["messages"]): full agent messages with
        # tool_calls/tool_call_id/reasoning — pass through intact so the API sees valid assistant→tool
        # sequences (dropping tool_calls causes 500s). Telegram observed group context: observed=True
        # rows are withheld from replayable history and attached to the current addressed message as
        # API-only context, so persisted history stores only the real addressed user turn.
        agent_history, observed_group_context = _build_gateway_agent_history(
            ctx.history,
            channel_prompt=ctx.channel_prompt,
            inject_timestamps=_message_timestamps_enabled(ctx.user_config),
        )

        # FTS write-corruption guard: if persistence failed silently via corrupt FTS triggers, the
        # reloaded transcript is stale/empty while the SAME cached agent still holds the full live
        # conversation in `_session_messages`; replacing it causes same-session amnesia. Only for
        # a reused agent bound to this exact session_id.
        if reused_cached_agent and getattr(agent, "session_id", None) == ctx.session_id:
            _selected = _select_cached_agent_history(
                agent_history, getattr(agent, "_session_messages", None)
            )
            if _selected is not agent_history:
                logger.warning(
                    "Persisted transcript lagged live cached history for "
                    "session %s (disk=%d, memory=%d); preserving live "
                    "conversation context (possible FTS write corruption)",
                    ctx.session_key, len(agent_history), len(_selected),
                )
                # The live in-memory history bypassed the _build_gateway_agent_history cleanup above —
                # re-apply the stale-confirmation expiry so a dangerous confirmation can't slip through.
                agent_history = strip_stale_dangerous_confirmations(
                    _selected, now=time.time()
                )

        # Collect MEDIA paths already in history to exclude them from this turn's extraction.
        # Compression-safe: even if the message list shrinks, we know which paths are old.
        _history_media_paths: set = _collect_history_media_paths(agent_history)
        return agent_history, observed_group_context, _history_media_paths

    def _approval_notify_sync(self, approval_data: dict) -> None:
        """Send the approval request to the user from the agent thread.

            Uses the adapter's interactive button approvals (e.g. ``send_exec_approval``) when
            available, else a plain text message with ``/approve`` instructions.
            """
        from gateway.run import (
            _approval_send_outcome,
            _format_exec_approval_fallback,
            _interim_metadata,
            _redact_approval_command,
            safe_schedule_threadsafe,
        )
        ctx = self._ctx
        # Pause typing while awaiting approval: Slack's assistant_threads_setStatus disables the
        # compose box, so the user can't type /approve while "is thinking..." shows. The approval
        # send auto-clears it; pausing stops _keep_typing re-setting it. Resumed in approve/deny.
        ctx._status_adapter.pause_typing_for_chat(ctx._status_chat_id)

        # WeCom native streaming: ask the stream consumer to close the current stream before the
        # approval prompt — via the consumer's queue, so it serializes with pending deltas.
        self._close_native_stream_boundary("Approval")

        cmd = approval_data.get("command", "")
        desc = approval_data.get("description", "dangerous command")

        # Redact credentials from the command before display — Tirith's findings are already
        # redacted, but the raw command string still leaks secrets to the chat platform. Done
        # here so BOTH the button-based and plain-text fallback paths use the redacted value.
        cmd = _redact_approval_command(cmd)

        # Prefer button-based approval when the adapter supports it. Check the *class*, not the
        # instance — avoids false positives from MagicMock auto-attribute creation in tests.
        if getattr(type(ctx._status_adapter), "send_exec_approval", None) is not None:
            try:
                _approval_fut = safe_schedule_threadsafe(
                    ctx._status_adapter.send_exec_approval(
                        chat_id=ctx._status_chat_id,
                        command=cmd,
                        session_key=ctx.session_key or "",
                        description=desc,
                        metadata=ctx._status_thread_metadata,
                        allow_permanent=approval_data.get("allow_permanent", True),
                        allow_session=approval_data.get("allow_session", True),
                        smart_denied=approval_data.get("smart_denied", False),
                    ),
                    ctx._loop_for_step,
                    logger=logger,
                    log_message="send_exec_approval scheduling error",
                )
                if _approval_fut is None:
                    raise RuntimeError("send_exec_approval: loop unavailable")
                _outcome = _approval_send_outcome(_approval_fut, timeout=15)
                if _outcome == "sent":
                    return
                if _outcome == "ambiguous":
                    # Timeout ≠ failure: the card may have posted with a late ack (slow API or
                    # backpressure). The prompt registration stays alive so a tap still resolves;
                    # re-sending made duplicate cards + orphaned "/approve: nothing pending". Skip.
                    logger.warning(
                        "Button-based approval send timed out — treating "
                        "as possibly-delivered (no re-send; the prompt "
                        "stays armed for a late tap)"
                    )
                    return
                logger.warning(
                    "Button-based approval failed (send returned error), falling back to text"
                )
            except Exception as _e:
                logger.warning(
                    "Button-based approval failed, falling back to text: %s", _e
                )

        # Fallback: plain-text approval prompt with the adapter's typed prefix (e.g. `!approve`) —
        # typed "/" is blocked in Slack threads and reserved by Matrix clients.
        _p = getattr(ctx._status_adapter, "typed_command_prefix", "/")
        msg = _format_exec_approval_fallback(
            cmd,
            desc,
            _p,
            allow_permanent=approval_data.get("allow_permanent", True),
            allow_session=approval_data.get("allow_session", True),
            smart_denied=approval_data.get("smart_denied", False),
        )
        try:
            # Mark as approval prompt so WeCom routes through control lane
            _approval_metadata = dict(ctx._status_thread_metadata or {})
            _approval_metadata["is_approval_prompt"] = True

            _approval_send_fut = safe_schedule_threadsafe(
                ctx._status_adapter.send(
                    ctx._status_chat_id,
                    msg,
                    metadata=_interim_metadata(_approval_metadata),
                ),
                ctx._loop_for_step,
                logger=logger,
                log_message="Approval text-send scheduling error",
            )
            if _approval_send_fut is not None:
                _approval_send_fut.result(timeout=15)
        except Exception as _e:
            logger.error("Failed to send approval request: %s", _e)

    def _prepare_turn_message(self, agent_history):
        from gateway.run import (
            _auto_continue_freshness_window,
            _is_fresh_gateway_interruption,
            _last_transcript_timestamp,
            _prepare_resume_pending_message,
            build_resume_recovery_note,
        )
        ctx = self._ctx
        # Keep real user text separate from API-only recovery guidance: if an auto-continue note is
        # prepended below, persist the original so stale guidance never replays as user text.
        _persist_user_message_override: Optional[Any] = ctx.persist_user_message
        _persist_user_timestamp_override: Optional[float] = ctx.persist_user_timestamp

        # Prepend pending model switch note so the model knows about the switch
        _pending_notes = getattr(self._runner, '_pending_model_notes', {})
        _msn = _pending_notes.pop(ctx.session_key, None) if ctx.session_key else None
        if _msn:
            ctx.message = _msn + "\n\n" + ctx.message

        # Auto-continue: history ending with a tool result means the previous turn was cut off
        # (restart, crash, SIGTERM) — prepend a system note so the model finishes the pending tool
        # results first. Session-level resume_pending (drain-timeout shutdown) uses stronger
        # reason-aware wording that subsumes this case. Both gate on the age of ``history[-1]`` (not
        # agent_history, which stripped ``timestamp`` off tool rows); rows without one are fresh.
        _freshness_window = _auto_continue_freshness_window()
        _interruption_is_fresh = _is_fresh_gateway_interruption(
            _last_transcript_timestamp(ctx.history),
            window_secs=_freshness_window,
        )

        _resume_entry = None
        if ctx.session_key:
            try:
                _resume_entry = self._runner.session_store._entries.get(ctx.session_key)
            except Exception:
                _resume_entry = None

        # resume_pending freshness also uses the restart watchdog's ``last_resume_marked_at`` (the
        # true interruption stamp): the transcript clock (_interruption_is_fresh) can be hours older
        # for an active thread, so gating on it alone drops the recovery note — and the startup
        # auto-resume turn has empty text, so the model gets a blank user message. Fresh if EITHER is.
        _resume_mark_is_fresh = False
        if _resume_entry is not None and getattr(_resume_entry, "resume_pending", False):
            _resume_mark_is_fresh = _is_fresh_gateway_interruption(
                getattr(_resume_entry, "last_resume_marked_at", None),
                window_secs=_freshness_window,
            )
        _is_resume_pending = bool(
            _resume_entry is not None
            and getattr(_resume_entry, "resume_pending", False)
            and (_interruption_is_fresh or _resume_mark_is_fresh)
        )
        _has_fresh_tool_tail = bool(
            agent_history
            and agent_history[-1].get("role") == "tool"
            and _interruption_is_fresh
        )

        if _is_resume_pending:
            _reason = getattr(_resume_entry, "resume_reason", None) or "restart_timeout"
            # Empty message = the startup auto-resume turn from _schedule_resume_pending_sessions;
            # there is no NEW user message. Interactive platforms report the restore and ask what
            # next; event platforms (webhook, API server) continue the work — nobody is present to
            # answer, and an acknowledgement would silently abandon the task.
            _resume_adapter = self._runner._adapter_for_source(ctx.source)
            _interactive_resume = bool(
                getattr(_resume_adapter, "interactive_resume", True)
            )
            ctx.message, _persist_user_message_override = _prepare_resume_pending_message(
                _reason, ctx.message, interactive=_interactive_resume,
            )
        elif _has_fresh_tool_tail:
            _persist_user_message_override = ctx.message
            ctx.message = (
                "[System note: A new message has arrived. The conversation "
                "history contains pending tool outputs from an interrupted turn. "
                "IGNORE those pending results. Address the user's NEW message "
                "below FIRST. Do NOT re-execute old tool calls from the history.]\n\n"
                + ctx.message
            )

        # Consume one-shot /reload-skills note (same queue pattern as CLI): prepend to the NEXT user
        # message, then clear. Nothing hit the transcript out-of-band, so alternation stays intact.
        _pending_notes = getattr(self._runner, "_pending_skills_reload_notes", None)
        if _pending_notes and ctx.session_key and ctx.session_key in _pending_notes:
            _srn = _pending_notes.pop(ctx.session_key, None)
            if _srn:
                ctx.message = _srn + "\n\n" + ctx.message

        # Safety net: a startup auto-resume event carries empty text and relies on the resume_pending
        # branch above for the recovery note. If it did not fire (freshness signals disagreed, marker
        # cleared before dispatch) we must NOT hand the model a blank user turn. Restricted to
        # resume_pending sessions so legitimately empty turns (caption-less image) are untouched.
        if (
            isinstance(ctx.message, str)
            and not ctx.message.strip()
            and _resume_entry is not None
            and getattr(_resume_entry, "resume_pending", False)
        ):
            _sn_reason = (
                getattr(_resume_entry, "resume_reason", None) or "restart_timeout"
            )
            _sn_adapter = self._runner._adapter_for_source(ctx.source)
            ctx.message = build_resume_recovery_note(
                _sn_reason,
                "",
                interactive=bool(
                    getattr(_sn_adapter, "interactive_resume", True)
                ),
            )
        return _persist_user_message_override, _persist_user_timestamp_override

    def _run_conversation_with_approval(
        self, agent, agent_history, observed_group_context,
        _persist_user_message_override, _persist_user_timestamp_override,
    ):
        from gateway.run import _wrap_current_message_with_observed_context
        ctx = self._ctx
        # Per-session gateway approval callback: dangerous-command approval blocks the agent thread
        # (mirrors CLI input()); the callback bridges sync→async to send the request immediately.
        from tools.approval import (
            register_gateway_notify,
            reset_current_session_key,
            set_current_session_key,
            unregister_gateway_notify,
        )

        _approval_session_key = ctx.session_key or ""
        _approval_session_token = set_current_session_key(_approval_session_key)
        register_gateway_notify(_approval_session_key, self._approval_notify_sync)
        try:
            # If _prepare_inbound_message_text buffered image paths for native attachment, wrap the
            # user turn as an OpenAI-style multimodal content list. Consume-and-clear so subsequent
            # turns on the same runner instance don't re-attach stale images.
            _native_imgs = self._runner._consume_pending_native_image_paths(ctx.session_key)
            if _native_imgs:
                try:
                    from agent.image_routing import build_native_content_parts
                    _parts, _skipped = build_native_content_parts(
                        ctx.message,
                        _native_imgs,
                    )
                    if _skipped:
                        logger.warning(
                            "Native image attachment: skipped %d unreadable path(s): %s",
                            len(_skipped), _skipped,
                        )
                    if any(p.get("type") == "image_url" for p in _parts):
                        _run_message: Any = _parts
                    else:
                        # All images failed to read — fall back to plain text.
                        _run_message = ctx.message
                except Exception as _img_exc:
                    logger.warning(
                        "Native image attachment failed, falling back to text: %s",
                        _img_exc,
                    )
                    _run_message = ctx.message
            else:
                _run_message = ctx.message

            _api_run_message = _wrap_current_message_with_observed_context(
                _run_message,
                observed_group_context,
            )
            _conversation_kwargs = {
                "conversation_history": agent_history,
                "task_id": ctx.session_id,
            }
            if _persist_user_message_override is not None:
                _conversation_kwargs["persist_user_message"] = _persist_user_message_override
            elif observed_group_context:
                _conversation_kwargs["persist_user_message"] = ctx.message
            if ctx.persist_user_display_kind:
                # Internal self-injected turn: type the persisted user row at turn start so UIs
                # render it as a timeline notice, not a user bubble. Role/content are untouched and
                # the key is stripped from provider-bound payloads in conversation_loop.
                _conversation_kwargs["persist_user_display_kind"] = (
                    ctx.persist_user_display_kind
                )
            if ctx.moa_config is not None:
                _conversation_kwargs["moa_config"] = ctx.moa_config
            if _persist_user_timestamp_override is not None:
                _conversation_kwargs["persist_user_timestamp"] = _persist_user_timestamp_override
            # Thread the platform-side inbound message id onto the persisted user turn so a turn
            # interrupted by a restart is recorded WITH its id — drain-window recovery dedups on
            # has_platform_message_id. Uses the raw inbound id, NOT event_message_id (reply anchor).
            if ctx.inbound_message_id is not None:
                _conversation_kwargs["persist_user_platform_id"] = str(ctx.inbound_message_id)
            result = agent.run_conversation(_api_run_message, **_conversation_kwargs)
        finally:
            unregister_gateway_notify(_approval_session_key)
            # Cancel any pending clarify entries so blocked agent threads don't hang past the end of
            # the run (interrupt, completion, gateway shutdown). Idempotent.
            try:
                from tools.clarify_gateway import clear_session as _clear_clarify_session
                _clear_clarify_session(_approval_session_key)
            except Exception:
                pass
            reset_current_session_key(_approval_session_token)
        return result

    def _finish_stream_consumer(self, result, agent_history, _stream_consumer):
        ctx = self._ctx
        # Canonicalize a model-emitted computer-use screenshot path at the common result boundary: the
        # streaming finalizer below and the non-streaming delivery path must see the same response;
        # repairing only in later media scanning leaves streaming a mangled path + rejected attachment.
        if isinstance(result, dict):
            _result_final = result.get("final_response")
            if isinstance(_result_final, str):
                result["final_response"] = repair_explicit_computer_use_media_paths(
                    _result_final,
                    result.get("messages", []),
                    history_offset=len(agent_history),
                )

        ctx.result_holder[0] = result

        # Signal the stream consumer that the agent is done, passing final_response as the
        # authoritative finalize payload: it includes post-stream augmentation (verifier footer,
        # explainer) the accumulator never saw, so the seal delivers the TRUE final with no
        # corrective send. Failed turns pass nothing — error text goes via the normal path.
        if _stream_consumer is not None:
            _final_for_stream = None
            # Adopt ONLY a genuinely completed final: interrupt paths return {interrupted: True,
            # completed: False} with a DIAGNOSTIC final_response and no failed key — adopting it
            # would seal the streamed partial answer over with the diagnostic AND make
            # delivered_final_matches reconcile, suppressing the gateway's own error delivery.
            if (
                isinstance(result, dict)
                and not result.get("failed")
                and not result.get("interrupted")
                and result.get("completed") is not False
            ):
                _fr = result.get("final_response")
                if isinstance(_fr, str) and _fr.strip() and _fr != "(empty)":
                    _final_for_stream = _fr
            if _final_for_stream is not None:
                # Duck-type safe: test doubles / older consumers may expose a zero-arg finish(). The
                # payload is an optimization, not a requirement — fall back to the bare signal.
                try:
                    _stream_consumer.finish(_final_for_stream)
                except TypeError:
                    _stream_consumer.finish()
            else:
                _stream_consumer.finish()

    def _sync_session_after_run(self, agent_history):
        ctx = self._ctx
        # Sync session_id right after run_conversation(): compression can rotate before a follow-up
        # model call fails, and the failure return below must still point at the compressed child.
        agent = ctx.agent_holder[0]
        _session_was_split = False
        # In-place compaction (compression.in_place) compacts the transcript WITHOUT rotating the id,
        # so the id-change diff below can't see it. compress_context() sets this flag on the agent; the
        # gateway re-baselines (history_offset=0 + JSONL rewrite) as for a split despite unchanged id.
        _compacted_in_place = bool(getattr(agent, "_last_compaction_in_place", False)) if agent else False
        agent_session_id = getattr(agent, 'session_id', ctx.session_id) if agent else ctx.session_id
        if agent and ctx.session_key and agent_session_id != ctx.session_id:
            _session_was_split = True
            logger.info(
                "Session split detected: %s → %s (compression)",
                ctx.session_id, agent_session_id,
            )
            entry = self._runner.session_store._entries.get(ctx.session_key)
            _session_split_entry_persisted = False
            if entry:
                entry_session_id = getattr(entry, "session_id", None)
                if not ctx._run_still_current():
                    logger.info(
                        "Skipping session split sync for stale run %s — "
                        "generation %s is no longer current",
                        ctx.session_key or "?",
                        ctx.run_generation,
                    )
                elif entry_session_id == agent_session_id:
                    _session_split_entry_persisted = True
                elif entry_session_id != ctx.session_id:
                    logger.info(
                        "Skipping session split sync for %s because the "
                        "session binding moved from %s to %s before "
                        "compression finished",
                        ctx.session_key or "?",
                        ctx.session_id,
                        entry_session_id,
                    )
                else:
                    entry.session_id = agent_session_id
                    self._runner.session_store._save()
                    self._runner.session_store._record_gateway_session_peer(
                        agent_session_id,
                        ctx.session_key,
                        ctx.source,
                    )
                    _session_split_entry_persisted = True

            # Telegram DM whose source.thread_id was lost in the session split (synthetic/recovered
            # event): restore it from the binding so _thread_metadata_for_source yields the right
            # message_thread_id instead of the General thread (non-fatal). Only after this run
            # published its split — a stale /stop→/new predecessor must not mutate routing state.
            if _session_split_entry_persisted and (
                getattr(ctx.source, "platform", None) == Platform.TELEGRAM
                and getattr(ctx.source, "chat_type", None) == "dm"
                and getattr(ctx.source, "thread_id", None) is None
                and self._runner._session_db is not None
            ):
                try:
                    # run_sync is off-loop (executor); sync DB is fine.
                    _binding = self._runner._session_db._db.get_telegram_topic_binding_by_session(
                        session_id=agent_session_id,
                    )
                    if _binding and _binding.get("thread_id"):
                        ctx.source.thread_id = str(_binding["thread_id"])
                        logger.debug(
                            "Restored source.thread_id=%s from binding after session split %s → %s",
                            ctx.source.thread_id,
                            ctx.session_id,
                            agent_session_id,
                        )
                except Exception:
                    logger.debug(
                        "Failed to restore thread_id from binding after session split",
                        exc_info=True,
                    )
            if _session_split_entry_persisted:
                self._runner._sync_telegram_topic_binding(
                    ctx.source, entry, reason="agent-run-compression",
                )

        effective_session_id = agent_session_id
        self._runner._sync_session_model_from_agent(effective_session_id, agent)
        # history_offset=0 whenever the agent's message list lost the original history prefix: rotation
        # (split) OR in-place compaction. Either way the returned `messages` is the compacted set, so
        # persist all of it; slicing past the pre-compaction length would drop everything.
        _effective_history_offset = (
            0 if (_session_was_split or _compacted_in_place) else len(agent_history)
        )
        return _compacted_in_place, effective_session_id, _effective_history_offset

    def run_sync(self):
        from gateway.run import (
            _collect_auto_append_media_tags,
            _current_max_iterations,
            _normalize_empty_agent_response,
            _sanitize_gateway_final_response,
        )
        ctx = self._ctx
        # As a method the turn message lives on the shared TurnContext: every rebind writes
        # `ctx.message`, so the outer `_run_agent_inner` body sees the update as via the closure cell.

        # session_key propagates via contextvars (_set_session_env / set_current_session_key):
        # concurrency-safe and inherited by tool worker threads. Deliberately do NOT write
        # os.environ["HERMES_SESSION_KEY"]: it is process-global, so concurrent sessions would clobber
        # each other and a tool thread with an unset contextvar would read the wrong key, misrouting
        # approvals. Only the TUI slash-worker subprocess exports the env var (from its own argv).

        # Map platform enum to the platform hint key the agent understands.
        # Platform.LOCAL ("local") maps to "cli"; others pass through as-is.
        platform_key = "cli" if ctx.source.platform == Platform.LOCAL else ctx.source.platform.value

        # Combine platform context, YAML channel_prompts hint for this chat, channel_overrides
        # system_prompt (or global ephemeral), and the gateway ephemeral prompt.
        combined_ephemeral = ctx.context_prompt or ""
        event_channel_prompt = (ctx.channel_prompt or "").strip()
        if event_channel_prompt:
            combined_ephemeral = (combined_ephemeral + "\n\n" + event_channel_prompt).strip()
        cfg_channel_prompt = self._runner._get_system_prompt_for_channel(
            ctx.source.platform,
            ctx.source.chat_id or "",
            thread_id=getattr(ctx.source, "thread_id", None),
            parent_id=getattr(ctx.source, "parent_chat_id", None),
        )
        if cfg_channel_prompt:
            combined_ephemeral = (combined_ephemeral + "\n\n" + cfg_channel_prompt).strip()

        max_iterations = _current_max_iterations()

        try:
            model, runtime_kwargs = self._runner._resolve_session_agent_runtime(
                source=ctx.source,
                session_key=ctx.session_key,
                user_config=ctx.user_config,
            )
            logger.debug(
                "run_agent resolved: model=%s provider=%s session=%s",
                model, runtime_kwargs.get("provider"), ctx.session_key or "",
            )
        except Exception as exc:
            return {
                "final_response": f"⚠️ Provider authentication failed: {exc}",
                "messages": [],
                "api_calls": 0,
                "tools": [],
            }

        pr = self._runner._provider_routing
        reasoning_config = self._runner._resolve_session_reasoning_config(
            source=ctx.source,
            session_key=ctx.session_key,
            model=model,
        )
        self._runner._reasoning_config = reasoning_config
        self._runner._service_tier = self._runner._resolve_session_service_tier(
            source=ctx.source, session_key=ctx.session_key
        )
        (
            _stream_consumer,
            _stream_delta_cb,
            _interim_assistant_cb,
            _want_interim_messages,
        ) = self._setup_stream_consumer(platform_key)

        turn_route = self._runner._resolve_turn_agent_config(ctx.message, model, runtime_kwargs)
        agent, reused_cached_agent = self._resolve_turn_agent(
            turn_route, platform_key, combined_ephemeral, max_iterations, reasoning_config, pr,
        )
        self._wire_turn_agent_callbacks(
            agent, turn_route, reasoning_config,
            _stream_delta_cb, _interim_assistant_cb, _want_interim_messages,
        )
        agent_history, observed_group_context, _history_media_paths = (
            self._load_turn_history(agent, reused_cached_agent)
        )
        _persist_user_message_override, _persist_user_timestamp_override = (
            self._prepare_turn_message(agent_history)
        )
        result = self._run_conversation_with_approval(
            agent, agent_history, observed_group_context,
            _persist_user_message_override, _persist_user_timestamp_override,
        )
        self._finish_stream_consumer(result, agent_history, _stream_consumer)

        # Signal the streaming-TTS consumer that the agent is done. finish() runs on the outer
        # event-loop thread after the executor returns, so early run_sync returns are also finalised.

        # Return final response, or a message if something went wrong
        final_response = result.get("final_response")

        # Extract actual token counts from the agent instance used for this run
        _last_prompt_toks = 0
        _input_toks = 0
        _output_toks = 0
        _context_length = 0
        _agent = ctx.agent_holder[0]
        if _agent and hasattr(_agent, "context_compressor"):
            _last_prompt_toks = getattr(_agent.context_compressor, "last_prompt_tokens", 0)
            _input_toks = getattr(_agent, "session_prompt_tokens", 0)
            _output_toks = getattr(_agent, "session_completion_tokens", 0)
            _context_length = getattr(_agent.context_compressor, "context_length", 0) or 0
        _resolved_model = getattr(_agent, "model", None) if _agent else None

        _compacted_in_place, effective_session_id, _effective_history_offset = (
            self._sync_session_after_run(agent_history)
        )

        if not final_response:
            final_response = _normalize_empty_agent_response(
                result, final_response or "", history_len=len(agent_history),
            )
            final_response = _sanitize_gateway_final_response(ctx.source.platform, final_response)
            if not final_response:
                final_response = f"⚠️ {result['error']}" if result.get("error") else ""
            return {
                "final_response": final_response,
                "messages": result.get("messages", []),
                "api_calls": result.get("api_calls", 0),
                "failed": result.get("failed", False),
                # Sibling of the non-empty-response return below: the classifier's failure_reason
                # must survive the empty-response path too, or downstream consumers (TUI billing,
                # transient-failure persistence) lose the structured reason when no text was produced.
                "failure_reason": result.get("failure_reason"),
                "partial": result.get("partial", False),
                "completed": result.get("completed"),
                "interrupted": result.get("interrupted", False),
                "interrupt_message": result.get("interrupt_message"),
                "error": result.get("error"),
                "compression_exhausted": result.get("compression_exhausted", False),
                "compression_deferred": result.get("compression_deferred", False),
                "tools": ctx.tools_holder[0] or [],
                "history_offset": _effective_history_offset,
                "compacted_in_place": _compacted_in_place,
                "session_id": effective_session_id,
                "last_prompt_tokens": _last_prompt_toks,
                "input_tokens": _input_toks,
                "output_tokens": _output_toks,
                "model": _resolved_model,
                "context_length": _context_length,
            }

        # Append MEDIA:<path> tags from tool results (e.g. TTS) that the model's final text omits, so
        # extract_media() delivers each file once. Scope to THIS turn (slice at ``len(agent_history)``)
        # so a stale MEDIA: path from an earlier turn doesn't ride a later text-only reply; dedup
        # against _history_media_paths is the secondary guard — and the sole one on the fallback
        # branch when mid-run compression shrank the list below the history length.
        if "MEDIA:" not in final_response:
            media_tags, has_voice_directive = _collect_auto_append_media_tags(
                result.get("messages", []),
                history_offset=len(agent_history),
                history_media_paths=_history_media_paths,
            )

            if media_tags:
                seen = set()
                unique_tags = []
                for tag in media_tags:
                    if tag not in seen:
                        seen.add(tag)
                        unique_tags.append(tag)
                if has_voice_directive:
                    unique_tags.insert(0, "[[audio_as_voice]]")
                final_response = final_response + "\n" + "\n".join(unique_tags)

        # Auto-titling runs at TURN START (agent/turn_context.py) from the user's message alone, so a
        # failed/interrupted turn is still titled. Thread-rename callbacks are attached as
        # `_on_session_title` before the run because the titler fires from the turn prologue.

        return {
            "final_response": final_response,
            "last_reasoning": result.get("last_reasoning"),
            "messages": ctx.result_holder[0].get("messages", []) if ctx.result_holder[0] else [],
            "api_calls": ctx.result_holder[0].get("api_calls", 0) if ctx.result_holder[0] else 0,
            "failed": ctx.result_holder[0].get("failed", False) if ctx.result_holder[0] else False,
            "failure_reason": (
                ctx.result_holder[0].get("failure_reason") if ctx.result_holder[0] else None
            ),
            "completed": ctx.result_holder[0].get("completed") if ctx.result_holder[0] else None,
            "interrupted": ctx.result_holder[0].get("interrupted", False) if ctx.result_holder[0] else False,
            "partial": ctx.result_holder[0].get("partial", False) if ctx.result_holder[0] else False,
            "error": ctx.result_holder[0].get("error") if ctx.result_holder[0] else None,
            "interrupt_message": ctx.result_holder[0].get("interrupt_message") if ctx.result_holder[0] else None,
            "compression_exhausted": (
                ctx.result_holder[0].get("compression_exhausted", False)
                if ctx.result_holder[0] else False
            ),
            # Soft lock-contention defer: distinct from compression_exhausted so the gateway never
            # auto-resets a session that a concurrent compressor is about to shrink.
            "compression_deferred": (
                ctx.result_holder[0].get("compression_deferred", False)
                if ctx.result_holder[0] else False
            ),
            "tools": ctx.tools_holder[0] or [],
            "history_offset": _effective_history_offset,
            "compacted_in_place": _compacted_in_place,
            "last_prompt_tokens": _last_prompt_toks,
            "input_tokens": _input_toks,
            "output_tokens": _output_toks,
            "model": _resolved_model,
            "context_length": _context_length,
            "session_id": effective_session_id,
            "response_previewed": result.get("response_previewed", False),
            "response_transformed": result.get("response_transformed", False),
            # Pass through agent_persisted so the persistence block above can tell whether the codex
            # app-server path self-persisted (it didn't — see codex_runtime.py); default True keeps the
            # skip-db behaviour for the standard runtime.
            "agent_persisted": (ctx.result_holder[0].get("agent_persisted", True) if ctx.result_holder[0] else True),
        }
