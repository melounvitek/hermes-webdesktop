"""Doctor output primitives shared by every ``hermes_cli.doctor_*`` module."""

from __future__ import annotations

from dataclasses import dataclass, field

from hermes_cli.colors import Colors, color


def check_ok(text: str, detail: str = ""):
    print(f"  {color('✓', Colors.GREEN)} {text}" + (f" {color(detail, Colors.DIM)}" if detail else ""))


def check_warn(text: str, detail: str = ""):
    print(f"  {color('⚠', Colors.YELLOW)} {text}" + (f" {color(detail, Colors.DIM)}" if detail else ""))


def check_fail(text: str, detail: str = ""):
    print(f"  {color('✗', Colors.RED)} {text}" + (f" {color(detail, Colors.DIM)}" if detail else ""))


def check_info(text: str):
    print(f"    {color('→', Colors.CYAN)} {text}")


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
