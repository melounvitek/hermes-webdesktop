"""DDGS search child-process entrypoint, run as ``python plugins/web/ddgs/_search_worker.py``.

Reads one JSON request ``{"query": str, "safe_limit": int}`` from stdin, writes one
envelope ``{"ok": true, "results": [...]}`` / ``{"ok": false, "error": str}`` to
stdout, exits. Test hooks (``"test_hook": "sleep"|"gil"|"success"|"error"|"empty"``)
are honored only when ``HERMES_DDGS_ALLOW_TEST_HOOKS=1``.
"""

from __future__ import annotations

import json
import os
import sys
import time


def _hold_gil(secs: int) -> None:
    """Block in a foreign call that keeps the GIL — mirrors native ``primp``.
    ``ctypes.PyDLL`` (unlike ``CDLL``/``WinDLL``) does not release the GIL."""
    import ctypes

    if sys.platform == "win32":
        lib = ctypes.PyDLL("kernel32")
        lib.Sleep.argtypes = [ctypes.c_uint]
        lib.Sleep(int(secs * 1000))
        return

    lib = ctypes.PyDLL(None)
    try:
        sleep = lib.sleep
    except AttributeError:  # pragma: no cover — macOS libSystem fallback
        sleep = ctypes.PyDLL("/usr/lib/libSystem.B.dylib").sleep
    sleep.argtypes = [ctypes.c_uint]
    sleep(int(secs))


def _hook_sleep() -> dict:
    time.sleep(30)
    return {"ok": False, "error": "sleep hook returned unexpectedly"}


def _hook_gil() -> dict:
    _hold_gil(30)
    return {"ok": False, "error": "gil hook returned unexpectedly"}


_TEST_HOOKS = {
    "sleep": _hook_sleep,
    "gil": _hook_gil,
    "success": lambda: {
        "ok": True,
        "results": [{"title": "Hit", "url": "https://example.com", "description": "body", "position": 1}],
    },
    "empty": lambda: {"ok": True, "results": []},
    "error": lambda: {"ok": False, "error": "RuntimeError: boom"},
}


def _write_envelope(envelope: dict) -> None:
    json.dump(envelope, sys.stdout)
    sys.stdout.flush()


def main() -> int:
    try:
        request = json.load(sys.stdin)
    except Exception as exc:  # noqa: BLE001
        _write_envelope({"ok": False, "error": f"invalid request: {exc}"})
        return 2

    hook = request.get("test_hook")
    if hook:
        if os.environ.get("HERMES_DDGS_ALLOW_TEST_HOOKS") != "1":
            _write_envelope({"ok": False, "error": "test_hook refused (hooks not enabled)"})
            return 3
        fn = _TEST_HOOKS.get(str(hook))
        envelope = fn() if fn else {"ok": False, "error": f"unknown test_hook: {hook!r}"}
        _write_envelope(envelope)
        return 0 if envelope.get("ok") else 1

    query = str(request.get("query") or "")
    safe_limit = max(1, int(request.get("safe_limit") or 1))
    try:
        from plugins.web.ddgs.provider import _run_ddgs_search  # lazy: light startup, patchable

        results = _run_ddgs_search(query, safe_limit)
        _write_envelope({"ok": True, "results": results})
        return 0
    except Exception as exc:  # noqa: BLE001
        _write_envelope({"ok": False, "error": f"{type(exc).__name__}: {exc}"})
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
