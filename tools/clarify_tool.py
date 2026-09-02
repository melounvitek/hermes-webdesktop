#!/usr/bin/env python3
"""Clarify tool: structured multiple-choice / open-ended questions to the user.

Schema, validation and a thin dispatcher; the UI lives in a platform-provided
callback (cli.py, gateway/run.py, tui_gateway).
"""

import inspect
import json
from typing import Dict, List, Optional, Callable


MAX_CHOICES = 4  # the UI always appends an "Other (type your answer)" row
MAX_QUESTIONS = 5  # independent questions per batch call

# Canonical timeout sentinel. The CLI returns this exact text; the batch loop
# treats it (like ``None``) as "the user walked away" and aborts remaining questions.
TIMEOUT_RESPONSE = (
    "The user did not provide a response within the time limit. "
    "Use your best judgement to make the choice and proceed."
)

# Applied to the first choice here (not per-surface) so every adapter renders it identically.
RECOMMENDED_LABEL = "(Recommended)"


def _flatten_choice(c) -> str:
    """Coerce one choice to display text.

    LLMs sometimes emit dict-shaped choices; ``str(c)`` would leak the dict repr
    onto every surface and back as the answer, so normalise once here. Unwrap
    order ``label`` > ``description`` > ``text`` > ``title``; ``name``/``value``
    are excluded (component fields carrying raw enums, not labels). A dict with
    none of these becomes "" and is dropped — no choice beats a garbage label.
    """
    if c is None:
        return ""
    if isinstance(c, str):
        return c.strip()
    if isinstance(c, dict):
        for key in ("label", "description", "text", "title"):
            v = c.get(key)
            if isinstance(v, str) and v.strip():
                return v.strip()
        return ""
    if isinstance(c, (list, tuple)):
        return " ".join(_flatten_choice(x) for x in c).strip()
    return str(c).strip()


def mark_recommended(choices: List[str]) -> List[str]:
    """Suffix the first choice (the schema says best-first) with RECOMMENDED_LABEL.

    Idempotent, and a lone choice is left untouched (nothing to prefer it over).
    """
    if len(choices) < 2:
        return choices
    first = str(choices[0]).strip()
    if first != strip_recommended(first):
        return choices
    return [f"{first} {RECOMMENDED_LABEL}"] + list(choices[1:])


def strip_recommended(text: str) -> str:
    """Remove the recommendation label so presentation never leaks into ``user_response``."""
    stripped = str(text).strip()
    if stripped.casefold().endswith(RECOMMENDED_LABEL.casefold()):
        return stripped[: -len(RECOMMENDED_LABEL)].strip()
    return stripped


def _accepts_kwarg(callback, name: str) -> bool:
    """Signature-inspect (never a TypeError retry, which could re-prompt the user)
    whether ``callback`` takes ``name`` or ``**kwargs``. Non-introspectable
    callables are conservatively treated as legacy."""
    try:
        params = inspect.signature(callback).parameters
    except (TypeError, ValueError):
        return False
    return name in params or any(
        p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()
    )


def _invoke_callback(callback, question, choices, multi_select):
    """Invoke the platform callback, passing multi_select if supported."""
    if _accepts_kwarg(callback, "multi_select"):
        return callback(question, choices, multi_select=multi_select)
    return callback(question, choices)


def _parse_multi_select_response(raw_response) -> List[str]:
    """Parse a list / JSON array / comma-separated reply into stripped non-empty strings."""
    if isinstance(raw_response, list):
        return [str(r).strip() for r in raw_response if str(r).strip()]

    raw = str(raw_response).strip()

    if raw.startswith("["):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(p).strip() for p in parsed if str(p).strip()]
        except json.JSONDecodeError:
            pass
    return [s.strip() for s in raw.split(",") if s.strip()]


def _clean_choices(choices: list) -> Optional[List[str]]:
    """Flatten, drop empties, cap at MAX_CHOICES; None when nothing survives (open-ended)."""
    cleaned = [s for s in (_flatten_choice(c) for c in choices) if s]
    return cleaned[:MAX_CHOICES] or None


def _is_timeout(raw) -> bool:
    return raw is None or (isinstance(raw, str) and raw.strip() == TIMEOUT_RESPONSE)


# --- batch (multi-question) support -----------------------------------------

def _normalize_questions(questions) -> tuple:
    """Validate the ``questions`` batch param -> ``(normalized, error)``.

    An empty list returns ``(None, None)`` (fall back to the single-question
    path). Each entry carries ``qid`` (stable wire id ``q<index>`` surfaces key
    answers by; the model's ``id`` is unvalidated text so it is only echoed in
    results), ``question``, decorated ``choices``, bare ``choices_offered``,
    and ``multi_select`` (honored only with choices).
    """
    if not isinstance(questions, list):
        return None, "questions must be an array of question objects."
    if not questions:
        return None, None
    if len(questions) > MAX_QUESTIONS:
        return None, f"questions supports at most {MAX_QUESTIONS} items."

    normalized = []
    for index, item in enumerate(questions):
        if isinstance(item, str):
            # Tolerate bare-string items: LLMs sometimes send ["Q1?", "Q2?"].
            item = {"question": item}
        if not isinstance(item, dict):
            return None, f"questions[{index}] must be an object with a 'question'."

        text = str(item.get("question") or "").strip()
        if not text:
            return None, f"questions[{index}].question must be non-empty text."

        choices = item.get("choices")
        if choices is not None:
            if not isinstance(choices, list):
                return None, f"questions[{index}].choices must be a list."
            choices = _clean_choices(choices)

        model_id = str(item.get("id") or "").strip() or None

        normalized.append({
            "qid": f"q{index}",
            "id": model_id,
            "question": text,
            "choices": mark_recommended(list(choices)) if choices else None,
            "choices_offered": list(choices) if choices else None,
            "multi_select": bool(item.get("multi_select")) and bool(choices),
        })

    return normalized, None


def _clean_batch_answer(entry: dict, raw) -> object:
    """Strip presentation from one locked answer (label, multi-select JSON)."""
    if entry["multi_select"]:
        return [strip_recommended(r) for r in _parse_multi_select_response(raw)]
    return strip_recommended(raw)


def _batch_result(normalized: List[dict], answers: dict, timed_out: bool) -> str:
    """Batch result JSON; unanswered -> "" — the top-level ``timed_out`` flag
    (present only when true) tells the agent whether blanks are deliberate
    skips or the user walking away."""
    responses = []
    for entry in normalized:
        row = {}
        if entry["id"]:
            row["id"] = entry["id"]
        row["question"] = entry["question"]
        row["choices_offered"] = entry["choices_offered"]
        raw = answers.get(entry["qid"])
        row["user_response"] = _clean_batch_answer(entry, raw) if raw else ""
        responses.append(row)

    result: Dict[str, object] = {"responses": responses}
    if timed_out:
        result["timed_out"] = True
    return json.dumps(result, ensure_ascii=False)


def _run_batch(normalized: List[dict], callback, question: str) -> str:
    """Dispatch a validated batch to the platform callback.

    Batch-capable callbacks (``questions`` kwarg) get the whole list once and
    reply ``{"answers": {qid: raw}, "timed_out"?}`` as a dict or JSON string
    (the tui_gateway bridge only carries strings). Legacy callbacks are looped
    per question; an empty answer is a skip, a timeout (``None`` or the
    sentinel) means the user walked away so the loop aborts instead of pestering
    them — answers collected before the abort are kept either way.
    """
    answers: dict = {}
    timed_out = False
    if _accepts_kwarg(callback, "questions"):
        raw = callback(question, None, questions=normalized)
        if _is_timeout(raw):
            timed_out = True
        elif isinstance(raw, dict):
            answers = dict(raw.get("answers") or {})
            timed_out = bool(raw.get("timed_out"))
        elif isinstance(raw, str) and raw.strip():
            try:
                parsed = json.loads(raw)
            except json.JSONDecodeError:
                parsed = None
            if isinstance(parsed, dict):
                answers = dict(parsed.get("answers") or {})
                timed_out = bool(parsed.get("timed_out"))
        # Any other falsy/unparseable reply is a cancel-all (mirrors the single-question skip).
        return _batch_result(normalized, answers, timed_out)

    for entry in normalized:
        raw = _invoke_callback(
            callback, entry["question"], entry["choices"], entry["multi_select"],
        )
        if _is_timeout(raw):
            timed_out = True
            break
        answers[entry["qid"]] = raw
    return _batch_result(normalized, answers, timed_out)


def clarify_tool(
    question: str,
    choices: Optional[List[str]] = None,
    multi_select: bool = False,
    questions: Optional[List[dict]] = None,
    callback: Optional[Callable] = None,
) -> str:
    """Ask the user one question (``question``/``choices``/``multi_select``) or a
    batch (``questions``, which takes precedence when non-empty).

    ``callback(question, choices, multi_select=False) -> str`` is platform
    injected; batch-capable callbacks also accept ``questions=``. Returns the
    result JSON (``{"responses": [...]}`` for batches).
    """
    if questions is not None:
        normalized, error = _normalize_questions(questions)
        if error:
            return tool_error(error)
        if normalized:
            if callback is None:
                return tool_error(
                    "Clarify tool is not available in this execution context."
                )
            try:
                return _run_batch(normalized, callback, str(question or "").strip())
            except Exception as exc:
                return tool_error(f"Failed to get user input: {exc}")
        # Empty questions array → fall through to the single-question path.

    if not question or not question.strip():
        return tool_error(
            "No question provided. Pass questions=[{question: '...', "
            "choices?: [...], multi_select?: bool}, ...] — a single question "
            "is a one-entry array."
        )

    question = question.strip()

    if choices is not None:
        if not isinstance(choices, list):
            return tool_error("choices must be a list of strings.")
        choices = _clean_choices(choices)

    if callback is None:
        return tool_error("Clarify tool is not available in this execution context.")

    # The bare list goes back to the agent; the "(Recommended)" label is presentation only.
    offered = choices
    if choices is not None:
        choices = mark_recommended(choices)

    try:
        raw_response = _invoke_callback(callback, question, choices, multi_select)
    except Exception as exc:
        return tool_error(f"Failed to get user input: {exc}")

    if multi_select and choices is not None:
        user_response = [strip_recommended(r) for r in _parse_multi_select_response(raw_response)]
    else:
        user_response = strip_recommended(raw_response)

    return json.dumps({
        "question": question,
        "choices_offered": offered,
        "user_response": user_response,
    }, ensure_ascii=False)


def check_clarify_requirements() -> bool:
    """Clarify tool has no external requirements -- always available."""
    return True


CLARIFY_SCHEMA = {
    "name": "clarify",
    "description": (
        "Ask the user one or more questions when you need a decision, "
        "clarification, or feedback before proceeding. Pass every question "
        f"in `questions` (1-{MAX_QUESTIONS} entries) — a single question is a "
        "one-entry array, and several INDEPENDENT questions belong in ONE "
        "call (one form beats a chain of clarify calls; if one answer would "
        "change another question, ask separately). Per question: "
        f"single-select (up to {MAX_CHOICES} choices — put your recommended "
        "option FIRST, the UI marks it '(Recommended)' and auto-appends an "
        "'Other' free-text row), multi-select (multi_select=true), or "
        "open-ended (omit choices). Options go ONLY in `choices`, never "
        "enumerated inside the question text (choices render as pickable "
        "rows; options written into the question are dead prose the user "
        "can't click). Result: {responses: [...]} in question order (plus "
        "timed_out=true if the user stopped part-way). Prefer deciding "
        "low-stakes questions yourself; don't use this for dangerous-command "
        "confirmation (the terminal tool handles that)."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "minItems": 1,
                "maxItems": MAX_QUESTIONS,
                "description": (
                    "The question(s). Each: question text (options excluded), "
                    "optional choices (recommended first; omit for free-text), "
                    "optional multi_select. Responses come back in question "
                    "order with the question text echoed."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {"type": "string"},
                        "choices": {
                            "type": "array",
                            "items": {"type": "string"},
                            "maxItems": MAX_CHOICES,
                        },
                        "multi_select": {"type": "boolean"},
                    },
                    "required": ["question"],
                },
            },
            # NOTE: the handler also accepts (unadvertised): a per-question
            # `id` (echoed in the matching response — redundant since rows
            # carry the question text and preserve order), and the legacy
            # single-question shape (`question` + `choices` + `multi_select`
            # at top level; a top-level `question` beside `questions` is the
            # batch form's title). One documented way to call.
        },
        "required": ["questions"],
    },
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="clarify",
    toolset="clarify",
    schema=CLARIFY_SCHEMA,
    handler=lambda args, **kw: clarify_tool(
        question=args.get("question", ""),
        choices=args.get("choices"),
        multi_select=args.get("multi_select", False),
        questions=args.get("questions"),
        callback=kw.get("callback")),
    check_fn=check_clarify_requirements,
    emoji="❓",
)
