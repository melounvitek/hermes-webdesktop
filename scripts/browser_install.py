#!/usr/bin/env python3
"""Small user-facing installer; the offline engine owns installation semantics."""

import argparse
from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import shlex
import stat
import subprocess
import sys
import tarfile
import tempfile
import http.client

# Isolated Python deliberately omits the script directory from sys.path.
sys.path.insert(0, str(Path(__file__).absolute().parent))
from browser_setup import E, detect, preflight  # noqa: E402
from browser_download import current, packaged_source  # noqa: E402

OWNER = "hermes-browser-setup-v1"
CONTROL_FILES = (
    "hermes-browser.py",
    "browser_setup.py",
    "browser_download.py",
    "browser_install.py",
    "source.json",
)


def locations():
    base = Path.home() / ".local/lib/hermes-browser"
    command = Path.home() / ".local/bin/hermes-browser"
    for path in (base, command):
        E.absolute_path(str(path))
        E.no_links(path)
        E.require(
            not any(ord(c) < 32 or ord(c) == 127 or c == "\\" for c in str(path)),
            "Unsafe convenience-command path",
        )
        for parent in (path, *path.parents):
            if parent.exists() and parent.is_relative_to(Path.home()):
                meta = parent.stat()
                E.require(
                    meta.st_uid == os.getuid() and not meta.st_mode & 0o022,
                    f"Unsafe owned path: {parent}",
                )
    return base, command


def namespace(base):
    E.no_links(base)
    if not os.path.lexists(base):
        return
    E.require(base.is_dir(), "Foreign browser namespace")
    allowed = {
        "control",
        "command.lock",
        "installation",
        "installation.run",
        "installation.history",
    }
    E.require(
        {p.name for p in base.iterdir()} <= allowed,
        "Unknown browser namespace entries; inspect manually, do not delete them automatically",
    )
    E.require(
        E.load_json(E.read_regular(base / "command.lock", 4096))
        == {"owner": OWNER, "root": str(base)},
        "Foreign/incomplete browser namespace; inspect manually",
    )


def facade(base, selection, hashes, owner_bytes):
    checks = {**hashes, "owner.json": E.digest(owner_bytes)}
    lines = "\n".join(
        sha + "  " + str(base / "control" / name)
        for name, sha in sorted(checks.items())
    )
    return (
        "#!/bin/sh\nset -eu\n"
        + "printf '%s\\n' "
        + shlex.quote(lines)
        + " | sha256sum --check --status || { echo 'Browser control files modified; refusing execution.' >&2; exit 1; }\n"
        + "exec "
        + shlex.quote(selection["python"])
        + " -I -S -B "
        + shlex.quote(str(base / "control/browser_install.py"))
        + ' "$@"\n'
    ).encode()


def owned_control(base, command):
    control = base / "control"
    E.no_links(control)
    meta = control.stat()
    E.require(
        meta.st_uid == os.getuid() and stat.S_IMODE(meta.st_mode) == 0o700,
        "Unsafe controller directory ownership/mode",
    )
    E.require(
        E.tree_files(control, strict_dirs=True) == {*CONTROL_FILES, "owner.json"},
        "Foreign/incomplete controller files",
    )
    for name in (*CONTROL_FILES, "owner.json"):
        meta = (control / name).lstat()
        E.require(
            meta.st_uid == os.getuid()
            and meta.st_nlink == 1
            and stat.S_IMODE(meta.st_mode) == 0o600,
            f"Unsafe controller ownership/mode: {name}",
        )
    raw = E.read_regular(control / "owner.json")
    owner = E.load_json(raw)
    E.keys(owner, "owner selection files")
    E.require(owner["owner"] == OWNER, "Foreign controller")
    E.keys(owner["selection"], " ".join(E.SELECTION))
    E.keys(owner["files"], " ".join(CONTROL_FILES))
    for name, sha in owner["files"].items():
        E.require(
            E.digest(E.read_regular(control / name)) == E.hex_value(sha),
            f"Modified controller: {name}",
        )
    expected = facade(base, owner["selection"], owner["files"], raw)
    if os.path.lexists(command):
        E.no_links(command)
        meta = command.lstat()
        E.require(
            meta.st_uid == os.getuid()
            and meta.st_nlink == 1
            and stat.S_IMODE(meta.st_mode) == 0o700,
            "Unsafe convenience command ownership/mode",
        )
        E.require(
            E.read_regular(command) == expected, "Foreign/modified convenience command"
        )
    return owner, expected


@contextmanager
def command_lock(base, exclusive=True):
    with (base / "command.lock").open("r+b") as stream:
        meta = os.fstat(stream.fileno())
        E.require(
            stat.S_ISREG(meta.st_mode)
            and meta.st_uid == os.getuid()
            and meta.st_nlink == 1
            and stat.S_IMODE(meta.st_mode) == 0o600,
            "Unsafe command lock",
        )
        try:
            fcntl.flock(
                stream, (fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH) | fcntl.LOCK_NB
            )
        except BlockingIOError:
            raise ValueError(
                "Browser command is active; stop it before maintenance"
            ) from None
        E.require(
            os.stat(base / "command.lock").st_ino == meta.st_ino, "Command lock changed"
        )
        yield stream


def confirm(preview):
    print(preview, flush=True)
    try:
        with (
            open("/dev/tty", "w", buffering=1, encoding="utf-8") as output,
            open("/dev/tty", encoding="utf-8") as input,
        ):
            output.write("Type yes to continue [no]: ")
            output.flush()
            return input.readline().rstrip("\r\n") == "yes"
    except OSError as error:
        raise ValueError(
            "An interactive terminal is required. Run the same command in a terminal; piped input cannot confirm installation."
        ) from error


def invoke(base, selection, command, *extra):
    result = subprocess.run(
        [
            selection["python"],
            "-I",
            "-S",
            "-B",
            str(base / "control/hermes-browser.py"),
            command,
            "--install-root",
            str(base / "installation"),
            *map(str, extra),
        ],
        input="yes\n",
        text=True,
        capture_output=True,
    )
    E.require(result.returncode == 0, result.stderr or result.stdout)
    return result.stdout


def publish_file(path, data, mode):
    E.no_links(path)
    with tempfile.TemporaryDirectory(
        prefix=".browser-publish-", dir=path.parent
    ) as temp:
        stage = Path(temp) / "file"
        stage.write_bytes(data)
        stage.chmod(mode)
        with stage.open("rb") as stream:
            os.fsync(stream.fileno())
        E.atomic_rename(stage, path)
        E.sync_directory(path.parent)


def setup(args):
    base, command = locations()
    namespace(base)
    if (base / "control").exists():
        with command_lock(base, exclusive=False):
            owner, _ = owned_control(base, command)
            selection = owner["selection"]
            # Ambient active_profile/HERMES_HOME may have changed. Repeat setup
            # verifies the original choice; only explicit options may challenge it.
            requested = argparse.Namespace(**{
                "hermes_home" if key == "hermes_root" else key: getattr(
                    args, "hermes_home" if key == "hermes_root" else key
                )
                or value
                for key, value in selection.items()
            })
            E.require(
                detect(requested) == selection,
                "Existing selection differs; uninstall explicitly before changing it",
            )
            receipt, manifest = E.installed(base / "installation")
            E.require(receipt["selection"] == selection, "Existing selection differs")
            preflight(selection, manifest)
            E.require(
                command.exists(),
                "Command publication incomplete; inspect owned paths manually before retrying",
            )
            print(
                "Already installed; verified without bundle downloads or changes. Use hermes-browser update explicitly."
            )
        return
    E.require(
        not os.path.lexists(command),
        "Foreign convenience command; choose no overwrite, inspect it manually",
    )
    E.require(
        not os.path.lexists(base / "installation"),
        "Installation without owned controller; inspect manually",
    )
    selection = detect(args)
    source = packaged_source()
    files = {
        name: E.read_regular(Path(__file__).with_name(name)) for name in CONTROL_FILES
    }
    hashes = {name: E.digest(raw) for name, raw in files.items()}
    owner = {"owner": OWNER, "selection": selection, "files": hashes}
    with tempfile.TemporaryDirectory(prefix="hermes-browser-download-") as temp:
        descriptor, manifest = current(source, Path(temp))
        runtime = preflight(selection, manifest)
        for destination in (base, command):
            E.safe_destination(destination, selection, parent_required=False)
        if not confirm(
            f"Install browser {manifest['release']} for existing Hermes\n  Backend: {selection['backend_root']}\n  Python: {selection['python']}\n  Profile: {runtime['profile_home']}\n  Files: {base}\n  Command: {command}\nNo Hermes changes. Nothing starts automatically.\nDownloads trust {source}; same-origin hashes check integrity, not independent publisher authentication."
        ):
            print("Cancelled; no installation writes.")
            return
        E.require(
            preflight(selection, manifest) == runtime,
            "Runtime changed during confirmation",
        )
        locations()
        namespace(base)
        for destination in (base, command):
            E.safe_destination(destination, selection, parent_required=False)
        if not base.exists():
            base.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            base.mkdir(mode=0o700)
            publish_file(
                base / "command.lock",
                E.json_bytes({"owner": OWNER, "root": str(base)}),
                0o600,
            )
        with command_lock(base):
            E.require(
                not os.path.lexists(command),
                "Convenience command appeared during confirmation; inspect manually",
            )
            if not (base / "control").exists():
                with tempfile.TemporaryDirectory(
                    prefix=".browser-control-", dir=base.parent
                ) as stage_dir:
                    stage = Path(stage_dir) / "control"
                    stage.mkdir(mode=0o700)
                    for name, raw in {
                        **files,
                        "owner.json": E.json_bytes(owner),
                    }.items():
                        (stage / name).write_bytes(raw)
                        (stage / name).chmod(0o600)
                    E.sync_installation(stage)
                    E.atomic_rename(stage, base / "control")
                    E.sync_directory(base)
            actual, command_bytes = owned_control(base, command)
            E.require(actual == owner, "Controller changed during confirmation")
            options = [
                "--archive",
                str(Path(temp) / "archive"),
                "--sha256",
                descriptor["archive"]["sha256"],
                "--launcher",
                str(Path(temp) / "launcher"),
            ]
            for name, value in selection.items():
                options.extend(["--" + name.replace("_", "-"), value])
            invoke(base, selection, "install", *options)
            command.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            if not os.path.lexists(command):
                publish_file(command, command_bytes, 0o700)
            owned_control(base, command)
        print(
            f"Installed. Nothing started. Run {command} start\nIf ~/.local/bin is on PATH: hermes-browser start. No shell startup files were changed."
        )


def manage(args):
    base, command = locations()
    namespace(base)
    E.require(
        (base / "control").exists(),
        "Browser is not installed; run the setup command first",
    )
    with command_lock(
        base, exclusive=args.command in ("update", "rollback", "uninstall")
    ):
        owner, _ = owned_control(base, command)
        selection = owner["selection"]
        root = base / "installation"
        args.install_root = str(root)
        if args.command == "start":
            E.require(
                E.installed(root)[0]["selection"] == selection,
                "Installation selection differs from controller",
            )
            print(
                "Starting in the foreground on loopback. Wait for 'Browser ready'; Ctrl-C stops it.",
                flush=True,
            )
            return E.lifecycle(args)
        if args.command in ("status", "stop"):
            extra = ["--timeout", args.timeout] if args.command == "stop" else []
            result = E.load_json(invoke(base, selection, args.command, *extra))
            print(
                "Browser "
                + result["state"]
                + (": " + result["url"] if result.get("state") == "ready" else "")
            )
            if result.get("detail"):
                print(result["detail"])
            return 0
        if args.command == "inspect":
            print(invoke(base, selection, "inspect"), end="")
            return 0
        receipt, manifest = E.installed(root)
        E.require(
            receipt["selection"] == selection,
            "Installation selection differs from controller",
        )

        def confirmation(preview):
            E.require(
                preview["current"] == receipt["archive_sha256"]
                and preview["selection"] == selection,
                "Installation changed before preview; inspect and retry",
            )
            action = (
                "Remove the browser, all retained versions, and the convenience command"
                if args.command == "uninstall"
                else f"{args.command.capitalize()} browser {manifest['release']} to {target['release']}"
            )
            accepted = confirm(
                f"{action}?\nProfile: {selection['profile']}\nHermes, data and plugins stay untouched. Nothing starts or stops automatically.\nStable ownership locks remain after uninstall."
            )
            locations()
            namespace(base)
            E.require(
                owned_control(base, command)[0] == owner,
                "Controller changed during confirmation",
            )
            return accepted

        target = None
        if args.command == "update":
            # Reject running/unknown ownership before downloading. Maintenance
            # reacquires this authoritative lock for its preview and publication.
            with E.stopped_control(root):
                pass
            with tempfile.TemporaryDirectory(prefix="hermes-browser-update-") as temp:
                descriptor, target = current(packaged_source(), Path(temp))
                preflight(selection, target)
                args.archive = str(Path(temp) / "archive")
                args.launcher = str(Path(temp) / "launcher")
                args.sha256 = descriptor["archive"]["sha256"]
                E.maintenance(args, confirm=confirmation)
            return 0
        if args.command == "rollback":
            versions = E.retained(root, selection)
            choices = {
                sha: item
                for sha, item in versions.items()
                if sha != receipt["archive_sha256"]
                and (args.to is None or args.to in (sha, item[1]["release"]))
            }
            E.require(
                len(choices) == 1,
                "Choose a retained version with rollback --to RELEASE (or its full ID); inspect lists retained versions",
            )
            args.to, (_, target) = next(iter(choices.items()))
        E.maintenance(args, confirm=confirmation)
        if args.command == "uninstall" and not root.exists():
            owned_control(base, command)
            if command.exists():
                command.unlink()
                E.sync_directory(command.parent)
            for name in (*CONTROL_FILES, "owner.json"):
                (base / "control" / name).unlink()
            (base / "control").rmdir()
            E.sync_directory(base)
            print(
                "Removed the owned command and controller. Hermes and stable lock files remain."
            )
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__, prog="hermes-browser")
    sub = parser.add_subparsers(dest="command", required=True)
    setup_parser = sub.add_parser(
        "setup", help="Detect existing Hermes and install the browser only"
    )
    for name in ("backend-root", "python", "hermes-home", "profile"):
        setup_parser.add_argument("--" + name)
    for name in (
        "start",
        "status",
        "stop",
        "inspect",
        "update",
        "rollback",
        "uninstall",
    ):
        child = sub.add_parser(name)
        if name == "start":
            child.add_argument("--port", type=int, default=9119)
        if name in ("start", "stop"):
            child.add_argument(
                "--timeout", type=float, default=60 if name == "start" else 10
            )
        if name == "rollback":
            child.add_argument(
                "--to",
                help="Retained release name; only needed if several choices exist",
            )
    argv = sys.argv[1:]
    if not argv or argv[0].startswith("--") and argv[0] != "--help":
        argv = ["setup", *argv]
    args = parser.parse_args(argv)
    try:
        os.umask(0o077)
        E.require(sys.platform == "linux", "Only Linux is supported")
        if args.command == "setup":
            setup(args)
        else:
            return manage(args)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        EOFError,
        tarfile.TarError,
        http.client.HTTPException,
        subprocess.SubprocessError,
    ) as error:
        print(f"hermes-browser: {error}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print(
            "Interrupted; inspect owned paths before retrying if publication had begun.",
            file=sys.stderr,
        )
        return 130
    return 0


if __name__ == "__main__":
    sys.exit(main())
