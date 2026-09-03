"""Live-DB test-isolation guard and the per-process "last init error" record.
Every SessionDB construction resolves its path through _ensure_test_isolation
so a pytest-context process (env OR ancestry) can never open a production
state.db; env-based so subprocess children are protected too."""

import os
import sys
import threading
from pathlib import Path
from typing import Any, Optional

try:  # Hard dependency, but tolerate scaffold-phase imports before pip install.
    import psutil
except ImportError:  # pragma: no cover - stripped/scaffold installs only
    psutil = None  # type: ignore[assignment]

# Field evidence: pytest fixture rows landed in the production state.db and a
# pytest-spawned child flipped the journal mode under the live WAL writer,
# destroying committed transcripts; any HERMES_HOME escape (fixture ordering, a
# child spawned without it, a shell exporting the real home) fell through silently.

#: Env twin of ``_STATE_DB_GUARD_BYPASS`` for child processes (a module global
#: cannot cross a process boundary, and ancestry arms the guard there).
_STATE_DB_GUARD_BYPASS_ENV = "HERMES_STATE_DB_GUARD_BYPASS"


def _real_platform_state_root() -> Optional[Path]:
    """The REAL platform-default Hermes root. Avoids ``Path.home()`` /
    ``hermes_constants``: tests monkeypatch Path.home to a tempdir while this
    module is imported lazily, which would misidentify the hermetic home as
    production or miss the real one. ``expanduser`` reads HOME/passwd, which the
    conftest never rewrites."""
    try:
        if sys.platform == "win32":
            base = os.environ.get("LOCALAPPDATA", "").strip()
            root = (
                Path(base) / "hermes"
                if base
                else Path(os.path.expanduser("~")) / "AppData" / "Local" / "hermes"
            )
        else:
            root = Path(os.path.expanduser("~")) / ".hermes"
        return root.resolve()
    except Exception:
        return None


#: Exported by the hermetic conftest alongside the HERMES_HOME redirect (value:
#: the isolation root). Unlike PYTEST_* (scrubbed by tests that rebuild a child
#: env) it is OURS and inherits by default, so a child carrying it that resolves
#: a production DB is by definition an isolation escape.
_TEST_ISOLATION_MARKER_ENV = "HERMES_TEST_ISOLATION"


def _running_under_pytest() -> bool:
    """True when this process (or a parent test process) is a pytest run."""
    return bool(
        os.environ.get("PYTEST_CURRENT_TEST")
        or os.environ.get("PYTEST_VERSION")
        or os.environ.get(_TEST_ISOLATION_MARKER_ENV)
    )


#: pytest launcher names, matched against each argv token's *basename* so
#: ``/tmp/pytest-of-dev/...`` paths cannot false-positive.
_PYTEST_LAUNCHER_NAMES = frozenset({"pytest", "py.test", "pytest.exe", "py.test.exe"})

#: Memoised ancestry answer: the tree above us doesn't change; keep the hot path free.
_PYTEST_ANCESTOR: Optional[bool] = None


def _process_looks_like_pytest(proc: Any) -> bool:
    """True when *proc*'s command line is a pytest invocation (``pytest ...`` or
    ``python -m pytest``). Unreadable cmdline => not pytest: guessing the other
    way would refuse production opens for unrelated reasons."""
    try:
        cmdline = proc.cmdline() or []
    except Exception:
        return False
    for arg in cmdline:
        try:
            # Split on both separators on every host: os.path.basename is
            # POSIX-only under Linux and would leave a Windows-style path
            # intact, making the matcher's answer depend on the platform.
            name = str(arg).strip('"').strip("'").replace("\\", "/").rsplit("/", 1)[-1].lower()
        except Exception:
            continue
        if name in _PYTEST_LAUNCHER_NAMES:
            return True
    return False


def _has_pytest_ancestor() -> bool:
    """True when an ancestor process is a pytest run. A child spawned with a
    rebuilt env loses PYTEST_* and the HERMES_HOME redirect together — aiming at
    production AND disarming the guard in one step; ancestry survives that.
    Fails open without psutil / on walk errors (never block real user runs)."""
    global _PYTEST_ANCESTOR
    if _PYTEST_ANCESTOR is not None:
        return _PYTEST_ANCESTOR
    found = False
    if psutil is not None:
        try:
            found = any(_process_looks_like_pytest(p) for p in psutil.Process().parents())
        except Exception:
            found = False
    _PYTEST_ANCESTOR = found
    return found


def _in_test_context() -> bool:
    """Test run by environment or ancestry. Env first (two dict lookups); the
    memoised ancestry walk runs at most once per real ``hermes`` invocation."""
    return _running_under_pytest() or _has_pytest_ancestor()


def _is_production_state_db(resolved: Path, root: Path) -> bool:
    """*resolved* is ``<root>/state.db`` or ``<root>/profiles/<name>/state.db``.
    Deeper scratch paths (repo worktrees under ~/.hermes/hermes-agent/...) are
    deliberately NOT matched so hermetic tests cannot false-positive."""
    if resolved.parent == root:
        return True
    try:
        parts = resolved.relative_to(root).parts
    except ValueError:
        return False
    return len(parts) == 3 and parts[0] == "profiles"


# Last SessionDB() init error, per-process; surfaced by /resume-style slash
# commands so users know WHY. Only SessionDB.__init__ writes it (kanban_db
# failures are reported via their own callers, by design).
_last_init_error: Optional[str] = None
_last_init_error_lock = threading.Lock()


def _set_last_init_error(msg: Optional[str]) -> None:
    """Record (or clear with None) the most recent state.db init failure.
    __init__ only SETs on failure and never clears on success: a concurrent
    successful open would erase the cause another thread's /resume is about to format."""
    global _last_init_error
    with _last_init_error_lock:
        _last_init_error = msg


def get_last_init_error() -> Optional[str]:
    """Most recent state.db init failure (None if none/never attempted)."""
    return _last_init_error
