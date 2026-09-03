"""Doctor output primitives shared by every ``hermes_cli.doctor_*`` module."""

from __future__ import annotations

import functools
from dataclasses import dataclass, field

from hermes_cli.colors import Colors, color


def _mark(glyph: str, col: str):
    def check(text: str, detail: str = ""):
        print(f"  {color(glyph, col)} {text}" + (f" {color(detail, Colors.DIM)}" if detail else ""))
    return check


check_ok, check_warn, check_fail = _mark("✓", Colors.GREEN), _mark("⚠", Colors.YELLOW), _mark("✗", Colors.RED)


def check_info(text: str):
    print(f"    {color('→', Colors.CYAN)} {text}")


def check_bool(cond, ok, bad, *, fail: bool = False):
    """``check_ok(*ok)`` when *cond* else ``check_warn(*bad)`` (``check_fail`` with fail=True); returns bool(cond).
    *ok* / *bad* are a text string or a ``(text, detail)`` tuple."""
    args = ok if cond else bad
    (check_ok if cond else (check_fail if fail else check_warn))(*((args,) if isinstance(args, str) else args))
    return bool(cond)


def _section(title: str) -> None:
    """Print a doctor section banner: blank line + bold cyan ◆ title."""
    print()
    print(color(f"◆ {title}", Colors.CYAN, Colors.BOLD))


def _fail_and_issue(text: str, detail: str, fix: str, issues: list[str]) -> None:
    """Emit a check_fail and append the corresponding fix instruction."""
    check_fail(text, detail)
    issues.append(fix)


@dataclass
class Finding:
    """What one doctor check contributed: auto-fixable issues, manual-only issues, fixes applied."""

    issues: list = field(default_factory=list)
    manual_issues: list = field(default_factory=list)
    fixed: int = 0

    def merge(self, other: "Finding") -> None:
        self.issues.extend(other.issues)
        self.manual_issues.extend(other.manual_issues)
        self.fixed += other.fixed


def doctor_check(on_error: str | None = None, detail: str = ""):
    """Turn ``fn(should_fix, f: Finding)`` into a ``(should_fix) -> Finding`` doctor check.
    If *fn* raises, prints ``check_warn(on_error.format(e=e), detail.format(e=e))`` (nothing when *on_error*
    is None); the partial Finding is still returned, so issues recorded before the crash survive."""
    def deco(fn):
        @functools.wraps(fn)
        def check(should_fix: bool) -> Finding:
            f = Finding()
            try:
                fn(should_fix, f)
            except Exception as e:
                if on_error is not None:
                    check_warn(on_error.format(e=e), detail.format(e=e))
            return f
        return check
    return deco


def ensure_dir(f: Finding, should_fix: bool, path, exists_msg: str, created_msg: str, missing_msg: str) -> None:
    """ok when *path* exists; with --fix create it (counts as fixed); else warn "(will be created on first use)"."""
    if path.exists():
        check_ok(exists_msg)
    elif should_fix:
        path.mkdir(parents=True, exist_ok=True)
        check_ok(created_msg)
        f.fixed += 1
    else:
        check_warn(missing_msg, "(will be created on first use)")
