#!/usr/bin/env python3
"""Todo tool: in-memory, revisioned task list for multi-step work.

State lives on the AIAgent (one per session), is re-injected after context
compression, and every write bumps a monotonic revision so UI clients can
reject stale updates. One ``todo_list`` tool: pass ``todos`` to write, omit to
read; every call returns the full list. No system-prompt mutation.
"""

import json
from typing import Any, Dict, List, Optional


VALID_STATUSES = {"pending", "in_progress", "completed", "cancelled"}

# The list is re-read after every compression (format_for_injection), so
# unbounded content/count would defeat the compression it rides through. Caps
# apply equally to model-authored items and caller-replayed API history.
MAX_TODO_CONTENT_CHARS = 4000
MAX_TODO_ITEMS = 256
# Max single todo tool-result payload accepted during history hydration, so a
# forged oversized result is dropped before parsing (AIAgent._hydrate_todo_store).
MAX_TODO_RESULT_CHARS = 512_000
_TRUNCATION_MARKER = "… [truncated]"
# Persisted as ordinary message content; ContextCompressor keys on this stable
# header to tell the synthetic post-compaction row from a real user message.
TODO_INJECTION_HEADER = (
    "[Your active task list was preserved across context compression]"
)


class TodoStore:
    """In-memory todo list, one per AIAgent. List position is priority.

    Items: ``{id, content, status, parent?}`` — ``parent`` nests a subtask.
    """

    def __init__(self):
        self._items: List[Dict[str, str]] = []
        self._revision = 0

    def _fresh_items(self, todos: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Validate, dedupe and order a whole new list (replace / restore)."""
        return self._normalize_order([self._validate(t) for t in self._dedupe_by_id(todos)])

    def write(self, todos: List[Dict[str, Any]], merge: bool = False) -> List[Dict[str, str]]:
        """Replace the list (default) or merge by id; returns the full list after writing."""
        before = self.read()
        if not merge:
            self._items = self._fresh_items(todos)
        else:
            self._merge(todos)
        # Keep the highest-priority head so a replayed list can't grow re-injection unbounded.
        if len(self._items) > MAX_TODO_ITEMS:
            self._items = self._items[:MAX_TODO_ITEMS]
        self._sanitize_parents(self._items)
        if self._items != before:
            self._revision += 1
        return self.read()

    def _merge(self, todos: List[Dict[str, Any]]) -> None:
        """Update existing items only in the fields provided; append new ones (validated)."""
        existing = {item["id"]: item for item in self._items}
        for t in self._dedupe_by_id(todos):
            item_id = str(t.get("id", "")).strip()
            if not item_id:
                continue  # Can't merge without an id
            cur = existing.get(item_id)
            if cur is None:
                validated = self._validate(t)
                existing[validated["id"]] = validated
                self._items.append(validated)
                continue
            if t.get("content"):
                cur["content"] = self._cap_content(str(t["content"]).strip())
            if t.get("status"):
                status = str(t["status"]).strip().lower()
                if status in VALID_STATUSES:
                    cur["status"] = status
            if "parent" in t:
                parent = str(t["parent"] or "").strip()
                if parent:
                    cur["parent"] = parent
                else:
                    cur.pop("parent", None)
        # Rebuild preserving original order for existing items.
        seen = set()
        rebuilt = []
        for item in self._items:
            current = existing.get(item["id"], item)
            if current["id"] not in seen:
                rebuilt.append(current)
                seen.add(current["id"])
        self._items = self._normalize_order(rebuilt)

    def read(self) -> List[Dict[str, str]]:
        """Return a copy of the current list."""
        return [item.copy() for item in self._items]

    def has_items(self) -> bool:
        return bool(self._items)

    def snapshot(self) -> Dict[str, Any]:
        """Return the full state clients can reconcile atomically."""
        return {"todos": self.read(), "revision": self._revision}

    def restore(
        self,
        todos: List[Dict[str, Any]],
        *,
        revision: Any = 0,
    ) -> List[Dict[str, str]]:
        """Restore a trusted snapshot without manufacturing a new revision."""
        self._items = self._fresh_items(todos)[:MAX_TODO_ITEMS]
        try:
            self._revision = max(0, int(revision or 0))
        except (TypeError, ValueError):
            self._revision = 0
        return self.read()

    def format_for_injection(self) -> Optional[str]:
        """Render the list for post-compression injection, or None if nothing active.

        Only pending/in_progress items are injected — finished ones make the
        model re-do work after compression. A parent is kept (with its real
        status marker) when any descendant is active so subtasks keep context.
        """
        if not self._items:
            return None

        markers = {"completed": "[x]", "in_progress": "[>]", "pending": "[ ]", "cancelled": "[~]"}
        active = {"pending", "in_progress"}
        children: Dict[str, List[Dict[str, str]]] = {}
        roots: List[Dict[str, str]] = []
        for item in self._items:
            parent = item.get("parent")
            if parent:
                children.setdefault(parent, []).append(item)
            else:
                roots.append(item)

        def render(item: Dict[str, str], depth: int, out: List[str]) -> bool:
            kid_lines: List[str] = []
            has_active_kid = False
            for kid in children.get(item["id"], []):
                has_active_kid |= render(kid, depth + 1, kid_lines)
            keep = item["status"] in active or has_active_kid
            if keep:
                marker = markers.get(item["status"], "[?]")
                out.append(
                    f"{'  ' * depth}- {marker} {item['id']}. "
                    f"{item['content']} ({item['status']})"
                )
                out.extend(kid_lines)
            return keep

        lines = [TODO_INJECTION_HEADER]
        for item in roots:
            render(item, 0, lines)
        if len(lines) == 1:
            return None

        return "\n".join(lines)

    @staticmethod
    def _cap_content(content: str) -> str:
        """Truncate to MAX_TODO_CONTENT_CHARS keeping the head (the actionable part) + marker."""
        if len(content) > MAX_TODO_CONTENT_CHARS:
            keep = MAX_TODO_CONTENT_CHARS - len(_TRUNCATION_MARKER)
            return content[:keep] + _TRUNCATION_MARKER
        return content

    @staticmethod
    def _validate(item: Dict[str, Any]) -> Dict[str, str]:
        """Normalize one item to ``{id, content, status, parent?}`` with placeholders for missing fields."""
        if not isinstance(item, dict):
            return {"id": "?", "content": "(invalid item)", "status": "pending"}

        item_id = str(item.get("id", "")).strip() or "?"
        content = str(item.get("content", "")).strip()
        content = TodoStore._cap_content(content) if content else "(no description)"
        status = str(item.get("status", "pending")).strip().lower()
        if status not in VALID_STATUSES:
            status = "pending"

        result = {"id": item_id, "content": content, "status": status}
        parent = str(item.get("parent") or "").strip()
        if parent and parent != item_id:
            result["parent"] = parent
        return result

    @staticmethod
    def _sanitize_parents(items: List[Dict[str, str]]) -> None:
        """Drop dangling parent refs and break cycles in place (such items become roots)."""
        by_id = {item["id"]: item for item in items}
        for item in items:
            if item.get("parent") and item["parent"] not in by_id:
                item.pop("parent", None)
        for item in items:
            seen = {item["id"]}
            node = item
            while node.get("parent"):
                if node["parent"] in seen:
                    item.pop("parent", None)
                    break
                seen.add(node["parent"])
                node = by_id[node["parent"]]

    @staticmethod
    def _dedupe_by_id(todos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Collapse duplicate ids, keeping the last occurrence in its position."""
        last_index: Dict[str, int] = {}
        for i, item in enumerate(todos):
            if not isinstance(item, dict):
                # Non-dict items get a synthetic key so _validate can handle them
                last_index[f"__invalid_{i}"] = i
                continue
            item_id = str(item.get("id", "")).strip() or "?"
            last_index[item_id] = i
        return [todos[i] for i in sorted(last_index.values())]

    @staticmethod
    def _normalize_order(items: List[Dict[str, str]]) -> List[Dict[str, str]]:
        """Lift the in_progress step ahead of any earlier pending placeholder.

        Nested lists keep authored order — reordering would tear a subtask from its siblings.
        """
        if any(item.get("parent") for item in items):
            return items
        statuses = [item["status"] for item in items]
        if "in_progress" not in statuses:
            return items
        active_index = statuses.index("in_progress")
        if "pending" not in statuses[:active_index]:
            return items
        pending_index = statuses.index("pending")

        normalized = items.copy()
        active_item = normalized.pop(active_index)
        normalized.insert(pending_index, active_item)
        return normalized


def todo_tool(
    todos: Optional[List[Dict[str, Any]]] = None,
    merge: bool = False,
    store: Optional[TodoStore] = None,
) -> str:
    """Write ``todos`` (replace or ``merge`` by id) or read when None; returns list + summary JSON."""
    if store is None:
        return tool_error("TodoStore not initialized")

    if todos is not None:
        if isinstance(todos, str):  # LLMs sometimes send a JSON string instead of a list
            try:
                todos = json.loads(todos)
            except (json.JSONDecodeError, TypeError):
                return tool_error("todos must be a list of objects, got unparseable string")
        if not isinstance(todos, list):
            return tool_error(
                f"todos must be a list, got {type(todos).__name__}"
            )
        items = store.write(todos, merge)
    else:
        items = store.read()

    summary = {"total": len(items)}
    for status in ("pending", "in_progress", "completed", "cancelled"):
        summary[status] = sum(1 for i in items if i["status"] == status)

    return json.dumps({
        "todos": items,
        "revision": store.snapshot()["revision"],
        "summary": summary,
    }, ensure_ascii=False)


def check_todo_requirements() -> bool:
    """Todo tool has no external requirements -- always available."""
    return True


# Behavioral guidance is baked into the (static, cached) description; item
# shape and merge semantics live ONLY in the parameter schema.
TODO_SCHEMA = {
    "name": "todo_list",
    "description": (
        "Track a task list for multi-step work (3+ steps). Use for complex tasks "
        "with 3+ steps or when the user provides multiple tasks. "
        "For 'all N items' tasks, enumerate every instance as its own checklist "
        "item so none are silently dropped. "
        "Call with no parameters to read the current list.\n"
        "List order is priority. Only ONE item in_progress at a time. "
        "Break large phases into subtasks via parent. "
        "Mark an item completed only after the work is verified done, never "
        "based on intent. If something fails, cancel it and add a revised "
        "item. Always returns the full current list."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "todos": {
                "type": "array",
                "description": "Task items to write.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {
                            "type": "string"
                        },
                        "content": {
                            "type": "string",
                            "description": "Task description"
                        },
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed", "cancelled"]
                        },
                        "parent": {
                            "type": "string",
                            "description": "Optional id of another item, making this a nested subtask. Omit for top-level."
                        }
                    },
                    "required": ["id", "content", "status"]
                }
            },
            "merge": {
                "type": "boolean",
                "description": (
                    "true: update existing items by id, add new ones. "
                    "false (default): replace the entire list with a fresh plan."
                ),
                "default": False
            }
        },
        "required": []
    }
}


# --- Registry ---
from tools.registry import registry, tool_error

registry.register(
    name="todo_list",
    toolset="todo",
    schema=TODO_SCHEMA,
    handler=lambda args, **kw: todo_tool(
        todos=args.get("todos"), merge=args.get("merge", False), store=kw.get("store")),
    check_fn=check_todo_requirements,
    emoji="📋",
)
