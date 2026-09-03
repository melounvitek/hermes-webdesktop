"""Auto-generate short session titles from the user's opening message.

Two stages, both off the critical path: an **instant** deterministic title derived from the first
user message (written before the model is called, cannot fail), then an **upgrade** from one
small-model call (cheap tier, thinking off, JSON-constrained). Provenance ``derived < llm < user``
is enforced by the storage layer, so stage 2 only replaces stage 1 and neither replaces a name the
user typed.
"""

import json
import logging
import re
import threading
from contextlib import suppress
from typing import Any, Callable, Optional

from agent.auxiliary_client import call_llm
from agent.context_compressor import LEGACY_SUMMARY_PREFIX
from agent.message_content import flatten_message_text

logger = logging.getLogger(__name__)

# (task_name, exception) -> None; surfaces auxiliary failures so silent drops don't pile up as NULL titles.
FailureCallback = Callable[[str, BaseException], None]

# (title, source) -> None; source is the persisted provenance (``derived`` / ``llm``). Consumers paying a
# rate-limited remote rename per title (Discord thread, Telegram topic) should act on ``llm`` only.
TitleCallback = Callable[[str, str], None]

# () -> bool, called right before the LLM request; False skips (e.g. the user switched models and
# the request would reload one the runtime already evicted).
RuntimeValidator = Callable[[], bool]

# Text budget handed to the model (Claude Code / OpenClaw converged on 1000).
MAX_TITLE_INPUT_CHARS = 1000
# Cap on the instant derived title; a raw fragment reads worse the longer it runs.
MAX_DERIVED_TITLE_CHARS = 48
# Answer-shaped guard: a tiny model sometimes answers instead of titling; longer is rejected, not truncated.
_MAX_TITLE_WORDS = 12

_TITLE_PROMPT_TEMPLATE = (
    "You name chat sessions. Given the user's opening message, write a title "
    "that lets them find this conversation again in a list.\n\n"
    "Rules:\n"
    "- 3 to 7 words, sentence case (capitalize only the first word and proper nouns).\n"
    "- Name what the user wants DONE, not that they asked a question.\n"
    "- Keep technical terms, filenames, numbers, and error codes exact.\n"
    "- Drop filler words: the, this, my, a, an.\n"
    "- No trailing punctuation, no quotes, no tool names, no 'Title:' prefix.\n"
    "- Never answer the message. Name it.\n"
    "- Always produce something, even for a bare greeting.\n"
    "__LANGUAGE_RULE__\n"
    'Good: {"title": "Fix login button on mobile"}\n'
    'Good: {"title": "Postgres connection pool exhaustion"}\n'
    'Good: {"title": "Friendly greeting"}\n'
    'Too vague: {"title": "Code changes"}\n'
    'Too long: {"title": "Investigate and fix the issue where the login button '
    'does not respond on mobile devices"}\n\n'
    'Reply with JSON only: {"title": "..."}'
)

_LANGUAGE_RULE_MATCH_USER = "- Write the title in the same language as the user's message."
_LANGUAGE_RULE_PINNED = "- Write the title in {language}."

# Constrains the response to a single title field ("model answered instead of titling" failure class).
_TITLE_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "session_title",
        "strict": True,
        "schema": {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
            "additionalProperties": False,
        },
    },
}

# Control-tag wrappers around machine-authored content inside a nominal "user" message (Codex CLI's
# RECOGNIZED_CONTROL_WRAPPERS): stripped, titling continues on what remains.
_CONTROL_WRAPPERS = tuple(
    (f"<{tag}>", f"</{tag}>")
    for tag in (
        "command-message", "command-name", "command-args", "local-command-caveat",
        "local-command-stderr", "local-command-stdout", "task-notification",
        "system-reminder", "ide_opened_file", "ide_selection",
    )
)

# Hermes' own machine-authored openers: a compaction handoff or resumed session must not be titled after them.
_MACHINE_PREFIXES = (
    "[CONTEXT COMPACTION",
    LEGACY_SUMMARY_PREFIX,
    "[Runtime note:",
    "[System note:",
    "[SYSTEM]",
    # tui_gateway.server._MODEL_SWITCH_MARKER_PREFIX (keep in sync); persisted as role="user" because
    # strict providers reject a non-first system message.
    "[System: The active model for this chat has changed to ",
)


def _title_config() -> dict:
    """``auxiliary.title_generation`` from config (lazy read-only import: no hermes_cli cycle, no migration writes)."""
    from hermes_cli.config import load_config_readonly

    return ((load_config_readonly() or {}).get("auxiliary") or {}).get("title_generation") or {}


def _title_language() -> str:
    """Configured title language, or "" to match the user."""
    try:
        return str(_title_config().get("language", "")).strip()
    except Exception:
        return ""


def _auto_title_enabled() -> bool:
    try:
        from utils import is_truthy_value

        return is_truthy_value(_title_config().get("enabled"), default=True)
    except Exception:
        logger.debug("Failed to read title_generation.enabled", exc_info=True)
        return True


def strip_control_wrappers(text: str) -> str:
    """Remove leading control wrappers (nested too) so a slash-command turn reduces to the prose the user typed."""
    current = (text or "").strip()
    # Bounded: each pass must remove at least one wrapper or we stop.
    for _ in range(len(_CONTROL_WRAPPERS) * 2):
        stripped = _strip_one_wrapper(current)
        if stripped == current:
            break
        current = stripped
    return current


def _strip_one_wrapper(text: str) -> str:
    lowered = text.lower()
    for open_tag, close_tag in _CONTROL_WRAPPERS:
        if not lowered.startswith(open_tag):
            continue
        end = lowered.find(close_tag)
        if end == -1:  # unterminated wrapper: drop the opening tag and keep the body
            return text[len(open_tag):].strip()
        # Prefer trailing prose; otherwise the wrapper body is all we have.
        return (text[end + len(close_tag):].strip() or text[len(open_tag):end].strip()).strip()
    return text


def _summarize_user_message(user_message: str) -> str:
    """Text worth titling: a ``/skill`` invocation embeds the whole skill body, so describe it first, then strip wrappers."""
    if not user_message:
        return ""
    described = None
    try:
        from agent.skill_commands import describe_skill_invocation

        described = describe_skill_invocation(user_message)
    except Exception:
        logger.debug("Skill-scaffolding summary failed; titling raw", exc_info=True)
    return strip_control_wrappers(user_message if described is None else described)


def is_titleable_user_message(user_message: str) -> bool:
    """False for machine-authored openers and turns that reduce to nothing once scaffolding is stripped."""
    if not isinstance(user_message, str) or not user_message.strip() or user_message.lstrip().startswith(_MACHINE_PREFIXES):
        return False
    return bool(_summarize_user_message(user_message).strip())


def derive_title(user_message: str) -> Optional[str]:
    """Instant title: first meaningful line trimmed to a word boundary. No model, never fails."""
    line = " ".join(_first_line(_summarize_user_message(user_message)).split())
    if not line:
        return None
    if len(line) > MAX_DERIVED_TITLE_CHARS:
        cut = line[:MAX_DERIVED_TITLE_CHARS]
        space = cut.rfind(" ")
        line = (cut[:space] if space > MAX_DERIVED_TITLE_CHARS // 2 else cut).rstrip(" ,.;:—-") + "…"
    return line or None


def _strip_title_prefix(text: str) -> str:
    return text[6:].strip() if text.lower().startswith("title:") else text


def _first_line(text: str) -> str:
    return next((ln.strip() for ln in text.splitlines() if ln.strip()), "")


def _extract_title_text(content: str) -> str:
    """Title from a model response: strict JSON, then a loose JSON scan, then first-line prose
    (a provider ignoring ``response_format`` still titles)."""
    if not content:
        return ""
    raw = content.strip()
    fenced = re.match(r"^```(?:json)?\s*(.*?)\s*```$", raw, re.DOTALL)
    if fenced:
        raw = fenced.group(1).strip()
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and isinstance(parsed.get("title"), str):
            return parsed["title"].strip()
    except (ValueError, TypeError):
        pass
    match = re.search(r'"title\"\s*:\s*"((?:[^"\\]|\\.)*)"', raw)
    if match:
        try:
            return json.loads(f'"{match.group(1)}"').strip()
        except ValueError:
            return match.group(1).strip()
    # Prose fallback: scrub <think> blocks so reasoning can't leak into a title.
    try:
        from agent.agent_runtime_helpers import strip_think_blocks

        raw = strip_think_blocks(None, raw).strip()
    except Exception:
        logger.debug("strip_think_blocks unavailable for title output", exc_info=True)
    return _strip_title_prefix(_first_line(raw)).strip("\"'").strip()


def _clean_title(text: str) -> Optional[str]:
    """Normalize a model-produced title, or None when nothing usable remains."""
    title = _strip_title_prefix(" ".join((text or "").split()).strip("\"'").strip()).rstrip(".!,;:")
    if not title:
        return None
    if len(title) > 80:
        title = title[:77].rstrip() + "..."
    return title


def _safe_callback(callback: Optional[Callable], args: tuple, log_fmt: str, label: str) -> None:
    """Invoke an optional consumer callback, never raising."""
    if callback is None:
        return
    try:
        callback(*args)
    except Exception:
        logger.debug(log_fmt, label, exc_info=True)


def _report_failure(failure_callback: Optional[FailureCallback], exc: BaseException, label: str) -> None:
    _safe_callback(failure_callback, ("title generation", exc), "%s failure_callback raised", label)


def _notify_title(title_callback: Optional[TitleCallback], title: str, source: str, label: str) -> None:
    _safe_callback(title_callback, (title, source), "%s callback failed", label)


def generate_title(
    user_message: str,
    timeout: Optional[float] = None,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
    runtime_validator: Optional[RuntimeValidator] = None,
) -> Optional[str]:
    """Title from the user's opening message alone (waiting for the assistant made this slow and
    bought nothing). ``runtime_validator`` runs right before the request; False skips silently."""
    if not _auto_title_enabled():
        logger.debug("Auto-title skipped: auxiliary.title_generation.enabled=false")
        return None

    if runtime_validator is not None:
        try:
            if not runtime_validator():
                logger.debug("Title generation skipped: runtime validator returned False")
                return None
        except Exception:
            # Fail open: a broken validator must not disable titling.
            logger.debug("Title runtime validator raised; proceeding", exc_info=True)

    user_snippet = _summarize_user_message(user_message)[:MAX_TITLE_INPUT_CHARS]
    if not user_snippet.strip():
        return None

    language = _title_language()
    language_rule = _LANGUAGE_RULE_PINNED.format(language=language) if language else _LANGUAGE_RULE_MATCH_USER
    # str.replace, not str.format: the prompt embeds literal JSON braces.
    prompt = _TITLE_PROMPT_TEMPLATE.replace("__LANGUAGE_RULE__", language_rule)

    try:
        response = call_llm(
            task="title_generation",
            messages=[{"role": "system", "content": prompt}, {"role": "user", "content": user_snippet}],
            # A title is a handful of tokens; a larger ceiling let chatty models burn seconds.
            max_tokens=64, temperature=0.3, timeout=timeout, main_runtime=main_runtime,
            extra_body={"response_format": _TITLE_RESPONSE_FORMAT},
        )
        title = _clean_title(_extract_title_text(response.choices[0].message.content or ""))
        # Answer-shaped output: reject (not truncate) so the caller retries next exchange.
        if title is not None and len(title.split()) > _MAX_TITLE_WORDS:
            logger.debug("Rejecting answer-shaped title output (%d words > %d)", len(title.split()), _MAX_TITLE_WORDS)
            return None
        return title
    except Exception as e:
        # WARNING so it shows in agent.log without debug mode; stack at debug.
        logger.warning("Title generation failed: %s", e)
        logger.debug("Title generation traceback", exc_info=True)
        _report_failure(failure_callback, e, "Title generation")
        return None


def _has_upgraded_title(session_db, session_id: str) -> bool:
    """True when the session already carries an ``llm``/``user`` title (or the check fails)."""
    try:
        source_fn = getattr(session_db, "get_session_title_source", None)
        if source_fn is not None:
            return source_fn(session_id) not in (None, "derived")
        return bool(session_db.get_session_title(session_id))
    except Exception:
        return True


def _persist_session_title(session_db, session_id, title, *, source, dedupe=True):
    """Persist a title at *source* authority via ``set_auto_title`` (precedence check + write in
    one transaction, so a manual ``/title`` is never overwritten); None when a higher-authority
    title held the row.

    ``ValueError`` = unique-title index rejected the name → append ``#N`` via
    ``get_next_title_in_lineage``. ``dedupe=False`` re-raises instead: the derived title is on the
    critical path, collides constantly ("hi"), and the model replaces it a second later anyway.
    """
    auto_fn = getattr(session_db, "set_auto_title", None)

    def _set(candidate):
        if auto_fn is not None:
            if auto_fn(session_id, candidate, source=source):
                return candidate
            logger.debug("Skipping %s title: a higher-authority title already holds session %s", source, session_id)
            return None
        # Older store without provenance support.
        legacy_fn = getattr(session_db, "set_auto_title_if_empty", None)
        if legacy_fn is not None:
            return candidate if legacy_fn(session_id, candidate) else None
        if session_db.set_session_title(session_id, candidate) is False:
            raise RuntimeError(f"session {session_id} not found when storing title")
        return candidate

    try:
        return _set(title)
    except ValueError:
        next_title_fn = getattr(session_db, "get_next_title_in_lineage", None)
        if not dedupe or next_title_fn is None:
            raise
        deduped = next_title_fn(title)
        if not deduped or deduped == title:
            raise
        return _set(deduped)


def apply_instant_title(
    session_db,
    session_id: str,
    user_message: str,
    title_callback: Optional[TitleCallback] = None,
) -> Optional[str]:
    """Write the derived title synchronously (cheap enough to run inline). Returns the title
    written, or None (no usable text, or a title of at least ``derived`` authority exists). Never raises."""
    if not session_db or not session_id:
        return None
    try:
        if not is_titleable_user_message(user_message):
            return None
        title = derive_title(user_message)
        if not title:
            return None
        persisted = _persist_session_title(session_db, session_id, title, source="derived", dedupe=False)
        if persisted:
            _notify_title(title_callback, persisted, "derived", "Instant-title")
        return persisted
    except Exception:
        logger.debug("Instant title failed", exc_info=True)
        return None


def auto_title_session(
    session_db,
    session_id: str,
    user_message: str,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
    title_callback: Optional[TitleCallback] = None,
    runtime_validator: Optional[RuntimeValidator] = None,
) -> None:
    """Generate and store the model title (daemon-thread target); skips sessions already carrying
    an ``llm``/``user`` title. Never lets an exception escape (the threading excepthook would spray
    a traceback into the terminal); the canonical trigger is the post-``hermes update``
    stale-module window, where lazy imports read NEW source against OLD cached modules."""
    try:
        # A derived title is expected here — upgrading it is the point.
        if not session_db or not session_id or _has_upgraded_title(session_db, session_id):
            return

        # This daemon thread starts AFTER the turn's ambient conversation context was reset;
        # republish it so the title call carries the same Portal ``conversation=`` tag
        # (root-of-lineage) and bills usage to this session.
        from agent.aux_accounting import set_accounting_context
        from agent.portal_tags import set_conversation_context

        conversation_id = session_id
        with suppress(Exception):
            conversation_id = session_db.get_conversation_root(session_id) or session_id
        set_conversation_context(conversation_id)
        set_accounting_context(session_db, session_id)

        title = generate_title(
            user_message, failure_callback=failure_callback,
            main_runtime=main_runtime, runtime_validator=runtime_validator,
        )
        source = "llm"
        if not title:
            # The inline attempt declines collisions rather than scan the lineage on the critical
            # path; off that path the scan is affordable.
            title, source = derive_title(user_message), "derived"
            if not title:
                return

        try:
            persisted = _persist_session_title(session_db, session_id, title, source=source)
            if persisted is None:
                return
            logger.debug("Auto-generated session title: %s", persisted)
            _notify_title(title_callback, persisted, source, "Auto-title")
        except Exception as e:
            logger.debug("Failed to set auto-generated title: %s", e)
    except Exception as e:
        # WARNING so operators see it in agent.log; names the likely cause.
        logger.warning("Auto-title failed (harmless; if this started after an update, restart the running Hermes process): %s", e)
        logger.debug("Auto-title traceback", exc_info=True)
        _report_failure(failure_callback, e, "Auto-title")


def _is_real_user_turn(message: Any) -> bool:
    """A question a person actually asked (Hermes persists machinery under ``role="user"``)."""
    if not isinstance(message, dict) or message.get("role") != "user":
        return False
    content = message.get("content")
    return is_titleable_user_message(content if isinstance(content, str) else flatten_message_text(content))


def _session_is_untitled(session_db, session_id: str) -> bool:
    """No title of any provenance. False when it can't tell — no model call per turn for an unreadable title."""
    getter = getattr(session_db, "get_session_title", None)
    if not callable(getter):
        return False
    try:
        return not str(getter(session_id) or "").strip()
    except Exception:
        logger.debug("Untitled check failed for %s", session_id, exc_info=True)
        return False


def maybe_auto_title(
    session_db,
    session_id: str,
    user_message: str,
    conversation_history: Optional[list] = None,
    failure_callback: Optional[FailureCallback] = None,
    main_runtime: dict = None,
    title_callback: Optional[TitleCallback] = None,
    runtime_validator: Optional[RuntimeValidator] = None,
) -> None:
    """Title a session from its opening message: instant inline, then upgraded on a daemon thread.
    Call at the START of a turn, before the model is invoked."""
    if not session_db or not session_id or not user_message:
        return

    # History may be pre- or post-message. Skip only when BOTH past the opening turn AND already named:
    # count alone left a machinery-opened session nameless; title alone never titles on an old store.
    user_msg_count = sum(1 for m in (conversation_history or []) if _is_real_user_turn(m))
    if user_msg_count > 1 and not _session_is_untitled(session_db, session_id):
        return

    if not is_titleable_user_message(user_message):
        return

    # Config read after the cheap guards so the file isn't touched every turn.
    if not _auto_title_enabled():
        logger.debug("Auto-title skipped: auxiliary.title_generation.enabled=false")
        return

    apply_instant_title(session_db, session_id, user_message, title_callback)

    threading.Thread(
        target=auto_title_session,
        args=(session_db, session_id, user_message),
        kwargs=dict(
            failure_callback=failure_callback, main_runtime=main_runtime,
            title_callback=title_callback, runtime_validator=runtime_validator,
        ),
        daemon=True,
        name="auto-title",
    ).start()
