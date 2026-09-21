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
            return input.readline().strip() == "yes"
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
    selection = detect(args)
    source = packaged_source()
    files = {
        name: E.read_regular(Path(__file__).with_name(name)) for name in CONTROL_FILES
    }
    hashes = {name: E.digest(raw) for name, raw in files.items()}
    owner = {"owner": OWNER, "selection": selection, "files": hashes}
    existing = None
    if (base / "control").exists():
        existing, _ = owned_control(base, command)
        E.require(
            existing == owner,
            "Existing selection/controller differs; repeat setup never replaces it. Use the installed command or uninstall explicitly first.",
        )
    else:
        E.require(
            not os.path.lexists(command),
            "Foreign convenience command; choose no overwrite, inspect it manually",
        )
    if (base / "installation").exists():
        E.require(
            existing is not None,
            "Installation without owned controller; inspect manually",
        )
        receipt, manifest = E.installed(base / "installation")
        E.require(receipt["selection"] == selection, "Existing selection differs")
        preflight(selection, manifest)
        E.require(
            command.exists(),
            "Installation complete but command publication interrupted; inspect owned paths manually before retrying",
        )
        print(
            "Already installed; verified without downloads or changes. Use hermes-browser update explicitly."
        )
        return
    with tempfile.TemporaryDirectory(prefix="hermes-browser-download-") as temp:
        descriptor, manifest = current(source, Path(temp))
        runtime = preflight(selection, manifest)
        # safe_destination needs an existing parent; validate overlaps against a
        # hypothetical leaf under the nearest existing parent before mkdir too.
        protected = [Path(selection[k]) for k in ("backend_root", "hermes_root")]
        protected.append(Path(selection["python"]).parent.parent)
        E.require(
            all(
                not (base.is_relative_to(p) or p.is_relative_to(base))
                for p in protected
            ),
            "Browser control namespace overlaps Hermes/runtime",
        )
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
        if not base.exists():
            base.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            base.mkdir(mode=0o700)
            publish_file(
                base / "command.lock",
                E.json_bytes({"owner": OWNER, "root": str(base)}),
                0o600,
            )
        with command_lock(base):
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command", nargs="?", choices=("setup", "inspect"), default="setup"
    )
    for name in ("backend-root", "python", "hermes-home", "profile"):
        parser.add_argument("--" + name)
    args = parser.parse_args()
    try:
        os.umask(0o077)
        E.require(sys.platform == "linux", "Only Linux is supported")
        if args.command == "setup":
            setup(args)
        else:
            base, command = locations()
            namespace(base)
            with command_lock(base, exclusive=False):
                owner, _ = owned_control(base, command)
                print(invoke(base, owner["selection"], "inspect"), end="")
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
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
