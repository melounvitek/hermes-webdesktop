"""Safe Hermes Console command engine."""

from __future__ import annotations

import argparse
import contextlib
import difflib
import functools
import importlib
import io
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Literal, NoReturn, Sequence

from tools.ansi_strip import strip_ansi as _strip_ansi


ConsoleStatus = Literal["ok", "error", "confirm_required", "exit", "clear"]


class ConsoleCommandError(RuntimeError):
    """User-facing console command failure."""


@dataclass(frozen=True)
class ConsoleResult:
    status: ConsoleStatus
    output: str = ""
    command: str = ""
    confirmation_message: str = ""


@dataclass(frozen=True)
class ConsoleCommand:
    path: tuple[str, ...]
    usage: str
    summary: str
    handler: Callable[["HermesConsoleEngine", list[str]], str]
    mutating: bool = False
    confirmation: str = ""


class _ArgumentParser(argparse.ArgumentParser):
    def error(self, message: str) -> NoReturn:  # pragma: no cover - argparse hook
        raise ConsoleCommandError(f"{self.prog}: {message}")


def _capture_output(fn: Callable[[], object]) -> str:
    stdout = io.StringIO()
    stderr = io.StringIO()
    code = 0
    message = ""
    with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
        try:
            result = fn()
            if isinstance(result, int) and result:
                raise SystemExit(result)
        except SystemExit as exc:
            # sys.exit("msg") / raise SystemExit("msg") is the standard non-zero-exit idiom:
            # exc.code is the message string, not an int. int() would raise ValueError here,
            # which escapes execute()'s ConsoleCommandError handler and crashes the REPL.
            if isinstance(exc.code, str):
                message = exc.code
                code = 1
            else:
                code = int(exc.code or 0)
        except ConsoleCommandError:
            raise
        except RuntimeError as exc:
            # Fail-closed config write guards raise RuntimeError (e.g.
            # require_readable_config_before_write refusing an unparseable
            # config.yaml). Convert to a console error instead of letting it
            # escape execute() and kill the REPL / websocket session.
            message = str(exc)
            code = 1
    text = stdout.getvalue() + stderr.getvalue()
    if code:
        raise ConsoleCommandError(message.strip() or text.strip() or f"Command exited with status {code}")
    return text.rstrip()


def _is_status_footer_rule(line: str) -> bool:
    stripped = _strip_ansi(line).strip()
    if len(stripped) < 8:
        return False
    normalized = stripped.replace("\u2500", "-")
    return set(normalized) <= {"-"}


def _strip_console_status_footer(text: str) -> str:
    lines = text.splitlines()
    while lines and not _strip_ansi(lines[-1]).strip():
        lines.pop()
    if len(lines) < 2:
        return text.rstrip()

    last = _strip_ansi(lines[-1]).strip()
    prev = _strip_ansi(lines[-2]).strip()
    if not (
        prev.startswith("Run 'hermes doctor'")
        and last.startswith("Run 'hermes setup'")
    ):
        return text.rstrip()

    lines = lines[:-2]
    while lines and not _strip_ansi(lines[-1]).strip():
        lines.pop()
    if lines and _is_status_footer_rule(lines[-1]):
        lines.pop()
    return "\n".join(lines).rstrip()


def _table_summary(summary: str, *, limit: int = 76) -> str:
    summary = " ".join(summary.split())
    if len(summary) <= limit:
        return summary
    return f"{summary[: limit - 3].rstrip()}..."


def _split_line(line: str) -> list[str]:
    try:
        # Windows-safe splitter: plain shlex posix=True eats backslashes, so
        # `sessions export C:\Users\me\out.jsonl` silently became a mangled
        # relative filename in the cwd (#83934).
        from hermes_cli._subprocess_compat import split_command_line

        return split_command_line(line)
    except ValueError as exc:
        raise ConsoleCommandError(f"Could not parse command: {exc}") from exc


def _contains_shell_syntax(line: str, tokens: Sequence[str]) -> bool:
    if "$(" in line or "`" in line:
        return True
    shell_tokens = {"|", "||", "&", "&&", ";", ">", ">>", "<", "<<", "2>", "2>>"}
    if any(token in shell_tokens for token in tokens):
        return True
    return any(ch in line for ch in "|<>;")


def _format_sessions(sessions: Sequence[dict]) -> str:
    if not sessions:
        return "No sessions found."
    lines = [f"{'ID':<32} {'Source':<12} {'Msgs':>5}  Title / Preview"]
    lines.append("-" * 82)
    for session in sessions:
        sid = str(session.get("id") or "")[:32]
        source = str(session.get("source") or "-")[:12]
        messages = session.get("message_count") or 0
        title = session.get("title") or session.get("preview") or ""
        title = str(title).replace("\n", " ")[:60]
        lines.append(f"{sid:<32} {source:<12} {messages:>5}  {title}")
    return "\n".join(lines)


def _format_job(job: dict, action: str) -> str:
    from cron.jobs import effective_job_state

    job_id = job.get("id") or job.get("job_id") or "?"
    name = job.get("name") or "(unnamed)"
    state = effective_job_state(job)
    return f"{action} job: {name} ({job_id}) [{state}]"


def _parser_root() -> tuple[_ArgumentParser, argparse._SubParsersAction]:
    parser = _ArgumentParser(prog="hermes", add_help=False)
    subparsers = parser.add_subparsers(dest="_console_command")
    return parser, subparsers


def _subparser_actions(parser: argparse.ArgumentParser) -> list[argparse._SubParsersAction]:
    return [
        action
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    ]


def _choice_help(action: argparse._SubParsersAction, name: str) -> str:
    for choice in action._choices_actions:
        if getattr(choice, "dest", None) == name or getattr(choice, "metavar", None) == name:
            help_text = getattr(choice, "help", None)
            if help_text and help_text is not argparse.SUPPRESS:
                return str(help_text)
    return ""


def _clean_summary(text: str | None) -> str:
    if not text or text is argparse.SUPPRESS:
        return ""
    summary = " ".join(str(text).split())
    if not summary or summary.startswith("Run `hermes "):
        return ""
    return summary


def _summaries_from_parser(parser: argparse.ArgumentParser) -> dict[tuple[str, ...], str]:
    summaries: dict[tuple[str, ...], str] = {}

    def walk(current: argparse.ArgumentParser, path: tuple[str, ...]) -> None:
        for action in _subparser_actions(current):
            for name, child in action.choices.items():
                child_path = (*path, name)
                summary = _clean_summary(_choice_help(action, name)) or _clean_summary(
                    child.description
                )
                if summary:
                    summaries.setdefault(child_path, summary)
                walk(child, child_path)

    walk(parser, ())
    return summaries


def _noop_console_command(_args: argparse.Namespace) -> None:
    return None


@dataclass(frozen=True)
class _CliSurface:
    """How a CLI subcommand module hangs its argparse tree off a root subparsers action.

    ``kind`` selects the wiring convention:
      * ``extracted``  — ``builder(subparsers, <handler>=fn)`` (hermes_cli.subcommands.*; fn from hermes_cli.main)
      * ``registered`` — ``register(subparsers.add_parser(root))``; optional module-level ``handler`` as func
      * ``builder``    — ``top = builder(subparsers)``; func = ``handler`` from hermes_cli.main
      * ``adder``      — ``add(subparsers)`` wires its own func
    """

    kind: Literal["extracted", "registered", "builder", "adder"]
    module: str
    builder: str
    handler: str | None = None

    def build(self, root: str, *, live: bool) -> _ArgumentParser:
        """Build a throwaway parser. ``live=False`` wires no-op handlers (summary extraction only)."""
        parser, subparsers = _parser_root()
        module = importlib.import_module(self.module)
        entry = getattr(module, self.builder)
        if self.kind == "extracted":
            fn = (
                getattr(importlib.import_module("hermes_cli.main"), self.handler)
                if live
                else _noop_console_command
            )
            entry(subparsers, **{self.handler: fn})
        elif self.kind == "registered":
            top_parser = subparsers.add_parser(root)
            entry(top_parser)
            if live and self.handler:
                top_parser.set_defaults(func=getattr(module, self.handler))
        elif self.kind == "builder":
            main_module = importlib.import_module("hermes_cli.main") if live else None
            top_parser = entry(subparsers)
            if live:
                top_parser.set_defaults(func=getattr(main_module, self.handler))
        else:
            entry(subparsers)
        return parser


# The CLI surface these helpers reflect is process-static: they import a
# subcommand module and build a throwaway argparse tree purely to extract help
# summaries. Nothing about the result changes across engine instances, but the
# dashboard opens a fresh HermesConsoleEngine per /api/console connection, so
# without memoization every reconnect re-imports + re-parses the whole surface.
@functools.lru_cache(maxsize=None)
def _surface_summaries(surface: _CliSurface, root: str) -> dict[tuple[str, ...], str]:
    try:
        return _summaries_from_parser(surface.build(root, live=False))
    except Exception:
        return {}


def _invoke_namespace(args: argparse.Namespace) -> object:
    func = getattr(args, "func", None)
    if not callable(func):
        raise ConsoleCommandError("No handler is available for that console command.")
    return func(args)


def _dispatch(
    surface: _CliSurface,
    root: str,
    fixed: Sequence[str],
    args: Sequence[str],
    namespace_update: Callable[[argparse.Namespace], None] | None = None,
) -> str:
    parser = surface.build(root, live=True)
    namespace = parser.parse_args([root, *fixed, *args])
    if namespace_update:
        namespace_update(namespace)
    return _capture_output(lambda: _invoke_namespace(namespace))


def _paths(spec: str) -> list[tuple[str, ...]]:
    """``"list, snapshot export"`` -> ``[("list",), ("snapshot", "export")]``; ``"."`` is the bare root."""
    return [() if item.strip() == "." else tuple(item.split()) for item in spec.split(",") if item.strip()]


def _sub(module: str, builder: str, handler: str) -> _CliSurface:
    return _CliSurface("extracted", f"hermes_cli.subcommands.{module}", builder, handler)


def _reg(module: str, handler: str | None = None) -> _CliSurface:
    return _CliSurface("registered", f"hermes_cli.{module}", "register_cli", handler)


# root -> (surface, paths, mutating paths). Registered in this order.
_CLI_FAMILIES: dict[str, tuple[_CliSurface, str, str]] = {
    "dump": (_sub("dump", "build_dump_parser", "cmd_dump"), ".", ""),
    "debug": (_sub("debug", "build_debug_parser", "cmd_debug"), "share, delete", "share, delete"),
    "prompt-size": (_sub("prompt_size", "build_prompt_size_parser", "cmd_prompt_size"), ".", ""),
    "insights": (_sub("insights", "build_insights_parser", "cmd_insights"), ".", ""),
    "security": (_sub("security", "build_security_parser", "cmd_security"), "audit", ""),
    "backup": (_sub("backup", "build_backup_parser", "cmd_backup"), ".", "."),
    "import": (_sub("import_cmd", "build_import_cmd_parser", "cmd_import"), ".", "."),
    "config": (_sub("config", "build_config_parser", "cmd_config"), "env-path, check", ""),
    "tools": (
        _sub("tools", "build_tools_parser", "cmd_tools"),
        "list, enable, disable, post-setup",
        "enable, disable, post-setup",
    ),
    "plugins": (
        _sub("plugins", "build_plugins_parser", "cmd_plugins"),
        "list, enable, disable, install, update, remove",
        "enable, disable, install, update, remove",
    ),
    "skills": (
        _sub("skills", "build_skills_parser", "cmd_skills"),
        "browse, search, inspect, list, check, list-modified, diff, install, update, audit, "
        "uninstall, reset, opt-in, opt-out, repair-official, snapshot export, snapshot import, "
        "tap list, tap add, tap remove",
        "install, update, audit, uninstall, reset, opt-in, opt-out, repair-official, "
        "snapshot export, snapshot import, tap add, tap remove",
    ),
    "mcp": (
        _sub("mcp", "build_mcp_parser", "cmd_mcp"),
        "list, catalog, test, add, remove, install, login, reauth, configure, picker",
        "add, remove, install, login, reauth, configure, picker",
    ),
    "memory": (_sub("memory", "build_memory_parser", "cmd_memory"), "status, off, reset", "off, reset"),
    "auth": (
        _sub("auth", "build_auth_parser", "cmd_auth"),
        "list, status, reset, add, remove, logout, spotify status, spotify login, spotify logout",
        "reset, add, remove, logout, spotify login, spotify logout",
    ),
    "pairing": (
        _sub("pairing", "build_pairing_parser", "cmd_pairing"),
        "list, approve, revoke, clear-pending",
        "approve, revoke, clear-pending",
    ),
    "webhook": (
        _sub("webhook", "build_webhook_parser", "cmd_webhook"),
        "list, subscribe, remove, test",
        "subscribe, remove",
    ),
    "hooks": (
        _sub("hooks", "build_hooks_parser", "cmd_hooks"),
        "list, test, doctor, revoke",
        "test, doctor, revoke",
    ),
    "slack": (_sub("slack", "build_slack_parser", "cmd_slack"), "manifest", ""),
    "profile": (
        _sub("profile", "build_profile_parser", "cmd_profile"),
        "list, show, info, create, use, describe, rename, delete, export, import, install, update",
        "create, use, describe, rename, delete, export, import, install, update",
    ),
    "cron": (
        _sub("cron", "build_cron_parser", "cmd_cron"),
        "create, edit, remove, tick",
        "create, edit, remove, tick",
    ),
    "portal": (_CliSurface("adder", "hermes_cli.portal_cli", "add_parser"), "info, tools", ""),
    "project": (
        _CliSurface("builder", "hermes_cli.projects_cmd", "build_parser", "cmd_project"),
        "list, show, create, add-folder, remove-folder, rename, set-primary, use, archive, "
        "restore, bind-board",
        "create, add-folder, remove-folder, rename, set-primary, use, archive, restore, bind-board",
    ),
    "kanban": (
        _CliSurface("builder", "hermes_cli.kanban", "build_parser", "cmd_kanban"),
        "init, boards list, boards create, boards rm, boards switch, boards current, "
        "boards rename, boards set-workdir, create, list, show, assign, reclaim, reassign, "
        "diagnose, link, unlink, claim, comment, complete, edit, block, schedule, unblock, "
        "promote, archive, stats, runs, heartbeat, assignments, context",
        "init, boards create, boards rm, boards switch, boards rename, boards set-workdir, "
        "create, assign, reclaim, reassign, link, unlink, claim, comment, complete, edit, "
        "block, schedule, unblock, promote, archive",
    ),
    "bundles": (
        _reg("bundles", "bundles_command"),
        "list, show, create, delete, reload",
        "create, delete, reload",
    ),
    "checkpoints": (
        _reg("checkpoints"),
        "status, list, prune, clear, clear-legacy",
        "prune, clear, clear-legacy",
    ),
    "curator": (
        _reg("curator"),
        "status, run, pause, resume, pin, unpin, restore, list-archived, archive, prune, "
        "backup, rollback",
        "run, pause, resume, pin, unpin, restore, archive, prune, backup, rollback",
    ),
    "pets": (
        _reg("pets"),
        "list, install, select, show, off, scale, remove, doctor",
        "install, select, off, scale, remove",
    ),
}

# Only extracted/registered families skip nested prompts after console confirmation
# (builder/adder families never did).
_CONFIRMED_KINDS = {"extracted", "registered"}

_SEND_SURFACE = _CliSurface("adder", "hermes_cli.send_cmd", "register_send_subparser")


def _register_command_family(
    engine: "HermesConsoleEngine",
    root: str,
    surface: _CliSurface,
    paths: str,
    mutating: str,
) -> None:
    summaries = _surface_summaries(surface, root)
    mutating_paths = set(_paths(mutating))
    namespace_update = _apply_confirmed_defaults if surface.kind in _CONFIRMED_KINDS else None
    for child_path in _paths(paths):
        full_path = (root, *child_path)
        usage = " ".join(full_path)

        def handler(_engine: HermesConsoleEngine, args: list[str], fixed=child_path) -> str:
            return _dispatch(surface, root, fixed, args, namespace_update)

        engine.register(
            full_path,
            usage,
            summaries.get(full_path) or f"Run `hermes {usage}`.",
            handler,
            mutating=child_path in mutating_paths,
            confirmation=f"Run `hermes {usage}`?",
        )


_BLOCKED_TOP = frozenset(
    "acp chat claw completion dashboard desktop fallback gateway gui login logout model moa "
    "oneshot proxy serve setup uninstall update whatsapp whatsapp-cloud".split()
)

_BLOCKED_PAIRS = {
    ("config", "edit"): "`config edit` opens an editor and is not available in Hermes Console.",
    ("mcp", "serve"): "`mcp serve` starts a server and is not available in Hermes Console.",
    ("profile", "alias"): "`profile alias` creates shell wrappers and is not available in Hermes Console.",
    ("skills", "config"): "`skills config` is interactive and is not available in Hermes Console.",
    ("skills", "publish"): "`skills publish` is not available in Hermes Console.",
    ("portal", "login"): "`portal login` is interactive and is not available in Hermes Console.",
    ("portal", "open"): "`portal open` opens a browser and is not available in Hermes Console.",
    ("kanban", "tail"): "`kanban tail` streams output and is not available in Hermes Console.",
    ("kanban", "watch"): "`kanban watch` streams output and is not available in Hermes Console.",
    ("kanban", "daemon"): "`kanban daemon` starts a service and is not available in Hermes Console.",
    ("kanban", "dispatcher"): "`kanban dispatcher` starts a worker and is not available in Hermes Console.",
    ("kanban", "swarm"): "`kanban swarm` starts agent work and is not available in Hermes Console.",
    ("kanban", "decompose"): "`kanban decompose` starts agent work and is not available in Hermes Console.",
    ("kanban", "specify"): "`kanban specify` starts agent work and is not available in Hermes Console.",
    ("kanban", "gc"): "`kanban gc` is not available in Hermes Console.",
    ("sessions", "delete"): "`sessions delete` and `sessions prune` are not available in Hermes Console.",
    ("sessions", "prune"): "`sessions delete` and `sessions prune` are not available in Hermes Console.",
}


class HermesConsoleEngine:
    """Curated line-command executor for Hermes Console."""

    def __init__(self, *, output_limit: int = 20000):
        self.output_limit = output_limit
        self.history: list[str] = []
        self.commands: dict[tuple[str, ...], ConsoleCommand] = {}
        self._register_defaults()

    def execute(self, line: str, *, confirmed: bool = False) -> ConsoleResult:
        raw_line = line.strip()
        if not raw_line:
            return ConsoleResult("ok")

        try:
            tokens = _split_line(raw_line)
            if tokens and tokens[0] == "hermes":
                tokens = tokens[1:]
            if not tokens:
                return ConsoleResult("ok", output=self.help_text())

            if _contains_shell_syntax(raw_line, tokens):
                raise ConsoleCommandError(
                    "Hermes Console does not run shell syntax. Use one supported "
                    "Hermes command at a time."
                )

            builtin = self._execute_builtin(tokens)
            if builtin is not None:
                if raw_line not in {"history", "clear"}:
                    self.history.append(raw_line)
                return builtin

            command, args = self._resolve_command(tokens)
            if command.mutating and not confirmed:
                return ConsoleResult(
                    "confirm_required",
                    command=raw_line,
                    confirmation_message=command.confirmation
                    or f"Run `{command.usage}`?",
                )

            output = command.handler(self, args).rstrip()
            output = self._cap_output(output)
            self.history.append(raw_line)
            return ConsoleResult("ok", output=output, command=raw_line)
        except ConsoleCommandError as exc:
            return ConsoleResult("error", output=str(exc).strip(), command=raw_line)

    def help_text(self, subject: str | None = None) -> str:
        if subject:
            tokens = subject.split()
            command, _args = self._resolve_command(tokens)
            return f"{command.usage}\n{command.summary}"

        lines = [
            "Hermes Console",
            "",
            "Supported commands:",
        ]
        for command in sorted(self.commands.values(), key=lambda c: c.usage):
            marker = " *" if command.mutating else "  "
            lines.append(f"{marker} {command.usage:<32} {_table_summary(command.summary)}")
        lines.extend(
            [
                "",
                "* requires confirmation",
                "Built-ins: help, help <command>, history, clear, exit, quit",
            ]
        )
        return "\n".join(lines)

    def _register_defaults(self) -> None:
        for path, usage, summary, handler in _READONLY_COMMANDS:
            self.register(path, usage, summary, handler)
        for path, usage, summary, handler, confirmation in _MUTATING_COMMANDS:
            self.register(path, usage, summary, handler, mutating=True, confirmation=confirmation)
        for root, (surface, paths, mutating) in _CLI_FAMILIES.items():
            _register_command_family(self, root, surface, paths, mutating)
        self.register(
            ("send",),
            "send --to <target> <message>",
            "Send a message to a configured platform.",
            lambda _engine, args: _dispatch(_SEND_SURFACE, "send", (), args),
            mutating=True,
            confirmation="Send this message?",
        )

    def register(
        self,
        path: Iterable[str],
        usage: str,
        summary: str,
        handler: Callable[["HermesConsoleEngine", list[str]], str],
        *,
        mutating: bool = False,
        confirmation: str = "",
    ) -> None:
        key = tuple(path)
        self.commands[key] = ConsoleCommand(
            path=key,
            usage=usage,
            summary=summary,
            handler=handler,
            mutating=mutating,
            confirmation=confirmation,
        )

    def _execute_builtin(self, tokens: list[str]) -> ConsoleResult | None:
        head = tokens[0]
        if head == "help":
            subject = " ".join(tokens[1:]).strip() or None
            try:
                return ConsoleResult("ok", output=self.help_text(subject))
            except ConsoleCommandError as exc:
                return ConsoleResult("error", output=str(exc))
        if head == "history":
            output = "\n".join(f"{idx + 1}: {cmd}" for idx, cmd in enumerate(self.history))
            return ConsoleResult("ok", output=output or "No history yet.")
        if head == "clear":
            return ConsoleResult("clear", output="\033[2J\033[H")
        if head in {"exit", "quit"}:
            return ConsoleResult("exit")
        return None

    def _resolve_command(self, tokens: Sequence[str]) -> tuple[ConsoleCommand, list[str]]:
        rejected = self._rejection_for(tokens)
        if rejected:
            raise ConsoleCommandError(rejected)

        for size in range(min(len(tokens), 3), 0, -1):
            key = tuple(tokens[:size])
            command = self.commands.get(key)
            if command:
                return command, list(tokens[size:])

        available = [" ".join(path) for path in self.commands]
        probe = " ".join(tokens[:2]) if len(tokens) > 1 else tokens[0]
        suggestions = difflib.get_close_matches(probe, available, n=3, cutoff=0.45)
        suffix = f" Did you mean: {', '.join(suggestions)}?" if suggestions else ""
        raise ConsoleCommandError(f"Unsupported Hermes Console command: {probe}.{suffix}")

    def _rejection_for(self, tokens: Sequence[str]) -> str:
        first = tokens[0]
        if first.startswith("-"):
            return f"{first} is not available in Hermes Console."
        if first in _BLOCKED_TOP:
            return f"`hermes {first}` is not available in Hermes Console."
        return _BLOCKED_PAIRS.get(tuple(tokens[:2]), "")

    def _cap_output(self, output: str) -> str:
        if len(output) <= self.output_limit:
            return output
        omitted = len(output) - self.output_limit
        return f"{output[:self.output_limit]}\n... output truncated ({omitted} bytes omitted)"


def _expect_no_args(args: Sequence[str], usage: str) -> None:
    if args:
        raise ConsoleCommandError(f"Usage: {usage}")


def _apply_confirmed_defaults(args: argparse.Namespace) -> None:
    """Skip nested prompts after the console-level confirmation has happened."""

    for attr in ("yes",):
        if hasattr(args, attr):
            setattr(args, attr, True)
    if getattr(args, "_console_command", None) == "import":
        setattr(args, "force", True)
    # Every checkpoints subcommand the console registers as mutating gates its
    # own confirmation on --force, so all three belong here. `prune` reaches
    # _confirm() for its orphan preview, and the console never redirects stdin.
    if getattr(args, "checkpoints_command", None) in {"prune", "clear", "clear-legacy"}:
        setattr(args, "force", True)
    if (
        getattr(args, "plugins_action", None) == "install"
        and not getattr(args, "enable", False)
        and not getattr(args, "no_enable", False)
    ):
        setattr(args, "no_enable", True)
    if getattr(args, "auth_action", None) == "add":
        auth_type = getattr(args, "auth_type", None)
        if auth_type in {"api-key", "api_key"} and not getattr(args, "api_key", None):
            raise ConsoleCommandError("auth add --type api-key requires --api-key in Hermes Console.")
    if getattr(args, "import_name", None) is not None:
        # profile import has no prompt flag; leave it alone.
        return
    if getattr(args, "skills_action", None) in {
        "install",
        "reset",
        "opt-out",
        "repair-official",
    }:
        setattr(args, "yes", True)
    if getattr(args, "memory_command", None) == "reset":
        setattr(args, "yes", True)


def _version(_engine: HermesConsoleEngine, args: list[str]) -> str:
    _expect_no_args(args, "version")
    from hermes_cli._startup_fast import print_fast_version_info

    return _capture_output(lambda: print_fast_version_info(check_updates=True))


def _status(_engine: HermesConsoleEngine, args: list[str]) -> str:
    _expect_no_args(args, "status")
    from types import SimpleNamespace

    from hermes_cli.status import show_status

    output = _capture_output(lambda: show_status(SimpleNamespace(all=False, deep=False)))
    return _strip_console_status_footer(output)


def _doctor(_engine: HermesConsoleEngine, args: list[str]) -> str:
    _expect_no_args(args, "doctor")
    from types import SimpleNamespace

    from hermes_cli.doctor import run_doctor

    return _capture_output(lambda: run_doctor(SimpleNamespace(fix=False, ack=None)))


def _logs(_engine: HermesConsoleEngine, args: list[str]) -> str:
    if "-f" in args or "--follow" in args:
        raise ConsoleCommandError("`logs -f` is not available in Hermes Console.")
    parser = _ArgumentParser(prog="logs", add_help=False)
    parser.add_argument("log_name", nargs="?", default="agent")
    parser.add_argument("-n", "--lines", type=int, default=50)
    parser.add_argument("--level")
    parser.add_argument("--session")
    parser.add_argument("--since")
    parser.add_argument("--component")
    ns = parser.parse_args(args)
    if ns.lines < 1 or ns.lines > 500:
        raise ConsoleCommandError("logs --lines must be between 1 and 500")

    from hermes_cli.logs import list_logs, tail_log

    if ns.log_name == "list":
        return _capture_output(list_logs)
    return _capture_output(
        lambda: tail_log(
            ns.log_name,
            num_lines=ns.lines,
            follow=False,
            level=ns.level,
            session=ns.session,
            since=ns.since,
            component=ns.component,
        )
    )


def _sessions_list(_engine: HermesConsoleEngine, args: list[str]) -> str:
    parser = _ArgumentParser(prog="sessions list", add_help=False)
    parser.add_argument("--limit", type=int, default=20)
    ns = parser.parse_args(args)
    if ns.limit < 1 or ns.limit > 200:
        raise ConsoleCommandError("sessions list --limit must be between 1 and 200")

    from hermes_state import SessionDB

    db = SessionDB()
    try:
        sessions = db.list_sessions_rich(
            exclude_sources=["kanban", "tool"],
            limit=ns.limit,
            order_by_last_active=True,
        )
    finally:
        db.close()
    return _format_sessions(sessions)


def _sessions_stats(_engine: HermesConsoleEngine, args: list[str]) -> str:
    _expect_no_args(args, "sessions stats")
    from hermes_state import SessionDB

    db = SessionDB()
    try:
        total = db.session_count()
        listable = db.session_count(exclude_children=True, exclude_sources=["kanban", "tool"])
        messages = db.message_count()
        lines = [
            f"Total sessions: {total}",
            f"Listable sessions: {listable}",
            f"Total messages: {messages}",
        ]
        for source in ["cli", "tui", "telegram", "discord", "slack", "cron"]:
            count = db.session_count(source=source)
            if count:
                lines.append(f"  {source}: {count}")
        return "\n".join(lines)
    finally:
        db.close()


def _config_show(_engine: HermesConsoleEngine, args: list[str]) -> str:
    _expect_no_args(args, "config show")
    from hermes_cli.config import show_config

    return _capture_output(show_config)


def _config_path(_engine: HermesConsoleEngine, args: list[str]) -> str:
    _expect_no_args(args, "config path")
    from hermes_cli.config import get_config_path

    return str(get_config_path())


def _config_set(_engine: HermesConsoleEngine, args: list[str]) -> str:
    if len(args) < 2:
        raise ConsoleCommandError("Usage: config set <key> <value>")
    key = args[0]
    value = " ".join(args[1:])
    from hermes_cli.config import set_config_value

    return _capture_output(lambda: set_config_value(key, value))


def _config_migrate(_engine: HermesConsoleEngine, args: list[str]) -> str:
    _expect_no_args(args, "config migrate")

    def _run() -> None:
        from hermes_cli.config import migrate_config

        results = migrate_config(interactive=False, quiet=False)
        if results.get("env_added") or results.get("config_added"):
            print("Configuration updated.")
        else:
            print("Configuration is up to date.")
        warnings = results.get("warnings") or []
        for warning in warnings:
            print(f"Warning: {warning}")

    return _capture_output(_run)


def _sessions_export(_engine: HermesConsoleEngine, args: list[str]) -> str:
    parser = _ArgumentParser(prog="sessions export", add_help=False)
    parser.add_argument("output")
    parser.add_argument("--source")
    parser.add_argument("--session-id")
    ns = parser.parse_args(args)

    def _run() -> None:
        from hermes_state import (
            SessionDB,
            SessionExportTooLargeError,
            resolved_max_export_messages,
        )

        db = SessionDB()
        try:
            def _guard_exports(session_ids: list[str]) -> None:
                # Per-session budget: each session is checked independently
                # against the configured limit, so a full-DB backup of many
                # small sessions never trips the guard — only an individual
                # runaway transcript does. 0 disables the guard.
                limit = resolved_max_export_messages()
                if limit <= 0:
                    return
                try:
                    for session_id in session_ids:
                        db.assert_export_safe(session_id, max_messages=limit)
                except SessionExportTooLargeError as exc:
                    raise ConsoleCommandError(
                        f"Session '{exc.session_id}' has more than {limit:,} active "
                        "messages; in-memory export is capped per session. "
                        "Use the Sessions page's streaming Export action, or set "
                        "sessions.max_export_messages: 0 in config.yaml to disable "
                        "the guard."
                    ) from exc

            if ns.session_id:
                resolved_session_id = db.resolve_session_id(ns.session_id)
                if not resolved_session_id:
                    raise ConsoleCommandError(f"Session '{ns.session_id}' not found.")
                _guard_exports([resolved_session_id])
                data = db.export_session(resolved_session_id)
                if not data:
                    raise ConsoleCommandError(f"Session '{ns.session_id}' not found.")
                rows = [data]
            else:
                session_ids = [
                    session["id"]
                    for session in db.search_sessions(source=ns.source, limit=100000)
                ]
                _guard_exports(session_ids)
                rows = db.export_all(source=ns.source)

            lines = [json.dumps(row, ensure_ascii=False) for row in rows]
            text = "\n".join(lines)
            if text:
                text += "\n"
            if ns.output == "-":
                sys.stdout.write(text)
            else:
                Path(ns.output).expanduser().write_text(text, encoding="utf-8")
                print(f"Exported {len(rows)} session(s) to {ns.output}")
        finally:
            db.close()

    return _capture_output(_run)


def _sessions_rename(_engine: HermesConsoleEngine, args: list[str]) -> str:
    parser = _ArgumentParser(prog="sessions rename", add_help=False)
    parser.add_argument("session_id")
    parser.add_argument("title", nargs="+")
    ns = parser.parse_args(args)

    def _run() -> None:
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            resolved_session_id = db.resolve_session_id(ns.session_id)
            if not resolved_session_id:
                raise ConsoleCommandError(f"Session '{ns.session_id}' not found.")
            title = " ".join(ns.title)
            if not db.set_session_title(resolved_session_id, title):
                raise ConsoleCommandError(f"Session '{ns.session_id}' not found.")
            print(f"Session '{resolved_session_id}' renamed to: {title}")
        finally:
            db.close()

    return _capture_output(_run)


def _sessions_optimize(_engine: HermesConsoleEngine, args: list[str]) -> str:
    _expect_no_args(args, "sessions optimize")

    def _run() -> None:
        from hermes_state import SessionDB

        db = SessionDB()
        try:
            count = db.vacuum()
            print(f"Optimized {count} FTS index(es).")
        finally:
            db.close()

    return _capture_output(_run)


def _sessions_repair(_engine: HermesConsoleEngine, args: list[str]) -> str:
    parser = _ArgumentParser(prog="sessions repair", add_help=False)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--no-backup", action="store_true")
    ns = parser.parse_args(args)

    def _run() -> None:
        from hermes_state import DEFAULT_DB_PATH, _db_opens_cleanly, repair_state_db_schema

        db_path = DEFAULT_DB_PATH
        if not db_path.exists():
            print(f"No session database at {db_path} (nothing to repair).")
            return
        reason = _db_opens_cleanly(db_path)
        if reason is None:
            print(f"{db_path} opens cleanly; no repair needed.")
            return
        print(f"{db_path} does not open cleanly: {reason}")
        if ns.check_only:
            return
        report = repair_state_db_schema(db_path, backup=not ns.no_backup)
        if report.get("repaired"):
            if report.get("backup_path"):
                print(f"backup: {report['backup_path']}")
            print(f"strategy: {report.get('strategy')}")
            print("Repaired session database.")
            return
        raise ConsoleCommandError(f"Repair failed: {report.get('error')}")

    return _capture_output(_run)


def _profile_status(_engine: HermesConsoleEngine, args: list[str]) -> str:
    _expect_no_args(args, "profile")
    return _dispatch(_CLI_FAMILIES["profile"][0], "profile", (), ())


def _cron_list(_engine: HermesConsoleEngine, args: list[str]) -> str:
    parser = _ArgumentParser(prog="cron list", add_help=False)
    parser.add_argument("--all", action="store_true")
    ns = parser.parse_args(args)
    from hermes_cli.cron import cron_list

    return _capture_output(lambda: cron_list(show_all=ns.all))


def _cron_status(_engine: HermesConsoleEngine, args: list[str]) -> str:
    _expect_no_args(args, "cron status")
    from hermes_cli.cron import cron_status

    return _capture_output(cron_status)


def _cron_job_action(args: list[str], usage: str, action: str, run) -> str:
    """Shared body for single-job cron commands: ``run(job_ref) -> job | None``."""
    if len(args) != 1:
        raise ConsoleCommandError(f"Usage: {usage}")
    from cron.jobs import AmbiguousJobReference

    try:
        job = run(args[0])
    except AmbiguousJobReference as exc:
        raise ConsoleCommandError(str(exc)) from exc
    if not job:
        raise ConsoleCommandError(f"Job not found: {args[0]}")
    return _format_job(job, action)


def _cron_pause(_engine: HermesConsoleEngine, args: list[str]) -> str:
    from cron.jobs import pause_job

    return _cron_job_action(
        args, "cron pause <job>", "Paused",
        lambda ref: pause_job(ref, reason="paused from hermes console"),
    )


def _cron_resume(_engine: HermesConsoleEngine, args: list[str]) -> str:
    parser = _ArgumentParser(prog="cron resume", add_help=False)
    parser.add_argument("job")
    parser.add_argument("--at")
    parser.add_argument("--run-now", action="store_true")
    ns = parser.parse_args(args)
    if ns.at and ns.run_now:
        raise ConsoleCommandError("Use exactly one of --at or --run-now.")
    from cron.jobs import AmbiguousJobReference, _hermes_now, rearm_oneshot, resume_job

    try:
        if ns.at or ns.run_now:
            job = rearm_oneshot(ns.job, _hermes_now().isoformat() if ns.run_now else ns.at)
        else:
            job = resume_job(ns.job)
    except (AmbiguousJobReference, ValueError) as exc:
        raise ConsoleCommandError(str(exc)) from exc
    if not job:
        raise ConsoleCommandError(f"Job not found: {ns.job}")
    return _format_job(job, "Resumed")


def _cron_run(_engine: HermesConsoleEngine, args: list[str]) -> str:
    from cron.jobs import trigger_job

    return _cron_job_action(args, "cron run <job>", "Triggered", trigger_job)


_READONLY_COMMANDS = (
    (("status",), "status", "Show Hermes component status.", _status),
    (("version",), "version", "Show Hermes version information.", _version),
    (("doctor",), "doctor", "Run diagnostics without auto-fix.", _doctor),
    (("logs",), "logs [name] [-n N]", "Show recent Hermes logs.", _logs),
    (("sessions", "list"), "sessions list [--limit N]", "List recent sessions.", _sessions_list),
    (("sessions", "stats"), "sessions stats", "Show session store statistics.", _sessions_stats),
    (("config", "show"), "config show", "Show current configuration.", _config_show),
    (("config", "path"), "config path", "Print config.yaml path.", _config_path),
    (("cron", "list"), "cron list [--all]", "List scheduled jobs.", _cron_list),
    (("cron", "status"), "cron status", "Show cron scheduler status.", _cron_status),
    (("profile",), "profile", "Show active profile status.", _profile_status),
)

# (path, usage, summary, handler, confirmation prompt)
_MUTATING_COMMANDS = (
    (("config", "set"), "config set <key> <value>", "Set a configuration value.", _config_set,
     "Update Hermes configuration?"),
    (("cron", "pause"), "cron pause <job>", "Pause a scheduled job.", _cron_pause, "Pause this cron job?"),
    (("cron", "resume"), "cron resume <job>", "Resume a paused cron job.", _cron_resume, "Resume this cron job?"),
    (("cron", "run"), "cron run <job>", "Run a job on the next scheduler tick.", _cron_run,
     "Trigger this cron job?"),
    (("config", "migrate"), "config migrate", "Update config with new options.", _config_migrate,
     "Update Hermes configuration with missing defaults?"),
    (("sessions", "export"), "sessions export <output> [--source SOURCE] [--session-id ID]",
     "Export sessions to JSONL.", _sessions_export, "Export session data?"),
    (("sessions", "rename"), "sessions rename <session> <title>", "Rename a session.", _sessions_rename,
     "Rename this session?"),
    (("sessions", "optimize"), "sessions optimize", "Optimize the session store.", _sessions_optimize,
     "Optimize the session database?"),
    (("sessions", "repair"), "sessions repair [--check-only] [--no-backup]",
     "Repair a malformed session database schema.", _sessions_repair, "Repair the session database?"),
)


def run_console_repl(
    *,
    stdin=None,
    stdout=None,
    stderr=None,
    interactive: bool | None = None,
) -> int:
    """Run the local ``hermes console`` REPL."""

    stdin = stdin or sys.stdin
    stdout = stdout or sys.stdout
    stderr = stderr or sys.stderr
    if interactive is None:
        interactive = bool(getattr(stdin, "isatty", lambda: False)())

    engine = HermesConsoleEngine()
    if interactive:
        print("Hermes Console. Type `help` for commands, `exit` to quit.", file=stdout)

    while True:
        if interactive:
            print("hermes> ", end="", file=stdout, flush=True)
        line = stdin.readline()
        if line == "":
            if interactive:
                print(file=stdout)
            return 0

        result = engine.execute(line)
        if result.status == "confirm_required":
            if not interactive:
                print(
                    f"Confirmation required: {result.confirmation_message}",
                    file=stderr,
                )
                return 1
            print(f"{result.confirmation_message} [y/N] ", end="", file=stdout, flush=True)
            answer = stdin.readline()
            if answer.strip().lower() not in {"y", "yes"}:
                print("Cancelled.", file=stdout)
                continue
            result = engine.execute(result.command, confirmed=True)

        if result.output:
            stream = stderr if result.status == "error" else stdout
            print(result.output, file=stream)
        if result.status == "exit":
            return 0
