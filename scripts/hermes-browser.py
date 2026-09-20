#!/usr/bin/env python3
"""Offline browser bundles for Linux; Python 3.10+, standard library only.

Trust this launcher BEFORE executing it. Obtain --sha256 independently of the
archive. A matching checksum is integrity, not publisher authentication.
Run as your normal user with a trusted Python and stable, trusted parent
directories; this is not a sandbox against a hostile runtime or parent replacement.

Maintainers: pack consumes a verification receipt, never builds or tests the UI.
The receipt supplies schema=1, release, ui_commit, upstream_commit, verification
(sha256 of the test receipt, scope), tested_backend (revision, reference_files:
path -> sha256), and files (web-relative path -> {size, sha256}). Only package
inventories recorded by successful verification; do not generate a receipt from
an arbitrary build and describe it as tested.

Users: inspect/install require explicit existing Python, backend and Hermes root,
profile, and a new install-root whose parent already exists. No backend imports,
configuration reads, start, repair, update, service or network operations occur.
Reference-file matches do not certify dependencies or running backend identity.
"""

import argparse
import ctypes
import gzip
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tarfile
import tempfile


MAX_ARCHIVE = 256 * 1024 * 1024
MAX_EXPANDED = 512 * 1024 * 1024
MAX_FILE = 128 * 1024 * 1024
MAX_FILES = 10000
MAX_JSON = 8 * 1024 * 1024
OWNER = "hermes-browser-offline-v1"
SELECTION = ("python", "backend_root", "hermes_root", "profile")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def hex_value(value, length=64):
    require(
        isinstance(value, str) and re.fullmatch(r"[0-9a-f]{%d}" % length, value),
        "Invalid digest/revision",
    )
    return value


def relative_name(name):
    require(isinstance(name, str) and 0 < len(name) <= 1024, "Invalid file path")
    require(
        not any(ord(c) < 32 or ord(c) == 127 or c == "\\" for c in name),
        "Unsafe file path",
    )
    require(
        not name.startswith("/")
        and all(p not in ("", ".", "..") for p in name.split("/")),
        "Unsafe file path",
    )
    return name


def absolute_path(value):
    path = Path(value)
    require(
        path.is_absolute()
        and not str(path).startswith("//")
        and ".." not in path.parts,
        "Paths must be absolute without '..' or double-leading slashes",
    )
    return path


def no_links(path):
    for part in (path, *path.parents):
        require(not part.is_symlink(), f"Symlink path refused: {part}")


def read_regular(path, limit=MAX_FILE):
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(fd, "rb") as stream:
        require(
            stat.S_ISREG(os.fstat(stream.fileno()).st_mode),
            f"Not a regular file: {path}",
        )
        data = stream.read(limit + 1)
    require(len(data) <= limit, f"File exceeds size limit: {path}")
    return data


def json_bytes(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def load_json(data):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            require(key not in result, f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    require(len(data) <= MAX_JSON, "JSON exceeds size limit")
    return json.loads(data, object_pairs_hook=unique)


def keys(value, expected):
    require(
        isinstance(value, dict) and set(value) == set(expected.split()),
        "Unsupported metadata fields",
    )


def validate_manifest(value, packed=True):
    keys(
        value,
        "schema release ui_commit upstream_commit verification tested_backend files"
        + (" launcher_sha256" if packed else ""),
    )
    require(type(value["schema"]) is int and value["schema"] == 1, "Unsupported schema")
    require(
        isinstance(value["release"], str)
        and re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", value["release"]),
        "Invalid release",
    )
    hex_value(value["ui_commit"], 40)
    hex_value(value["upstream_commit"], 40)
    keys(value["verification"], "sha256 scope")
    hex_value(value["verification"]["sha256"])
    require(
        isinstance(value["verification"]["scope"], str)
        and 0 < len(value["verification"]["scope"]) <= 4000,
        "Missing verification scope",
    )
    keys(value["tested_backend"], "revision reference_files")
    hex_value(value["tested_backend"]["revision"], 40)
    references = value["tested_backend"]["reference_files"]
    require(
        isinstance(references, dict) and 0 < len(references) <= MAX_FILES,
        "Missing backend references",
    )
    for name, sha in references.items():
        relative_name(name)
        hex_value(sha)
    files = value["files"]
    require(
        isinstance(files, dict)
        and 0 < len(files) <= MAX_FILES
        and "index.html" in files,
        "Missing browser inventory/index.html",
    )
    total = 0
    for name, info in files.items():
        relative_name(name)
        keys(info, "size sha256")
        hex_value(info["sha256"])
        require(
            type(info["size"]) is int and 0 <= info["size"] <= MAX_FILE,
            "Invalid file size",
        )
        require(
            not any(
                str(parent) in files
                for parent in PurePosixPath(name).parents
                if str(parent) != "."
            ),
            "Conflicting file paths",
        )
        total += info["size"]
    require(total <= MAX_EXPANDED, "Bundle exceeds size limit")
    if packed:
        hex_value(value["launcher_sha256"])
    return value


def tree_files(root):
    no_links(root)
    require(root.is_dir(), f"Missing directory: {root}")
    files = set()
    for parent, dirs, names in os.walk(root, followlinks=False):
        for name in dirs + names:
            path = Path(parent) / name
            require(not path.is_symlink(), f"Symlink refused: {path}")
        for name in names:
            path = Path(parent) / name
            require(stat.S_ISREG(path.lstat().st_mode), f"Special file refused: {path}")
            files.add(path.relative_to(root).as_posix())
        require(len(files) <= MAX_FILES + 4, "Too many files")
    return files


def verified_web(root, manifest):
    require(
        tree_files(root) == set(manifest["files"]),
        "Browser inventory differs from verification receipt",
    )
    for name, info in manifest["files"].items():
        data = read_regular(root / name)
        require(
            len(data) == info["size"] and digest(data) == info["sha256"],
            f"Browser hash/size mismatch: {name}",
        )
        yield name, data


def pack(args):
    manifest = validate_manifest(
        load_json(read_regular(args.receipt, MAX_JSON)), packed=False
    )
    files = dict(verified_web(absolute_path(args.web_dir), manifest))
    manifest["launcher_sha256"] = digest(read_regular(Path(__file__)))
    manifest_bytes = json_bytes(manifest)
    require(len(manifest_bytes) <= MAX_JSON, "JSON exceeds size limit")
    # Capture verified bytes once. Do not re-open a changing build while archiving.
    payload = {
        "manifest.json": manifest_bytes,
        **{"web/" + name: data for name, data in files.items()},
    }
    output = absolute_path(args.output)
    no_links(output)
    require(not os.path.lexists(output), "Output already exists")
    with tempfile.TemporaryDirectory(
        prefix=".hermes-browser-pack-", dir=output.parent
    ) as temp:
        staged = Path(temp) / "bundle.tar.gz"
        with staged.open("xb") as stream:
            with gzip.GzipFile(
                fileobj=stream, mode="wb", filename="", mtime=0
            ) as compressed:
                with tarfile.open(
                    fileobj=compressed, mode="w", format=tarfile.PAX_FORMAT
                ) as tar:
                    for name, data in sorted(payload.items()):
                        member = tarfile.TarInfo(name)
                        member.size = len(data)
                        member.mode = 0o600
                        tar.addfile(member, io.BytesIO(data))
                require(
                    compressed.tell() <= MAX_EXPANDED,
                    "Expanded archive exceeds size limit",
                )
        sha = digest(read_regular(staged, MAX_ARCHIVE))
        publish_new(staged, output)
    print(
        json.dumps(
            {
                "archive": str(output),
                "sha256": sha,
                "launcher_sha256": manifest["launcher_sha256"],
            },
            indent=2,
        )
    )


def archive_payload(path, expected):
    hex_value(expected)
    raw = read_regular(absolute_path(path), MAX_ARCHIVE)
    require(digest(raw) == expected, "Archive SHA-256 mismatch")
    # Bound decompression before tarfile processes extended headers as well as data.
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:
        expanded = stream.read(MAX_EXPANDED + 1)
    require(len(expanded) <= MAX_EXPANDED, "Expanded archive exceeds size limit")
    payload = {}
    with tarfile.open(fileobj=io.BytesIO(expanded), mode="r:") as tar:
        for member in tar:
            name = relative_name(member.name)
            require(
                member.type in (tarfile.REGTYPE, tarfile.AREGTYPE)
                and member.sparse is None
                and not member.linkname,
                "Only ordinary files allowed in archive",
            )
            require(
                name not in payload and len(payload) <= MAX_FILES,
                "Duplicate/too many archive files",
            )
            require(0 <= member.size <= MAX_FILE, "Archive member exceeds size limit")
            payload[name] = tar.extractfile(member).read()
    require("manifest.json" in payload, "Missing manifest")
    manifest = validate_manifest(load_json(payload["manifest.json"]))
    require(
        set(payload)
        == {"manifest.json", *("web/" + name for name in manifest["files"])},
        "Incomplete or extra archive files",
    )
    for name, info in manifest["files"].items():
        data = payload["web/" + name]
        require(
            len(data) == info["size"] and digest(data) == info["sha256"],
            f"Archive file mismatch: {name}",
        )
    require(
        manifest["launcher_sha256"] == digest(read_regular(Path(__file__))),
        "Archive requires a different trusted launcher",
    )
    return manifest, payload


def inspect_runtime(selection, manifest):
    python = absolute_path(selection["python"])
    require(
        python.is_file() and os.access(python, os.X_OK),
        "Missing executable Python runtime",
    )
    backend = absolute_path(selection["backend_root"])
    root = absolute_path(selection["hermes_root"])
    no_links(backend)
    no_links(root)
    require(
        root.is_dir() and root.parent.name != "profiles",
        "Expected an existing Hermes data root, not a profile home",
    )
    read_regular(backend / "hermes_cli/main.py")
    profile = selection["profile"]
    require(
        isinstance(profile, str) and re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,63}", profile),
        "Invalid profile name",
    )
    home = root if profile == "default" else root / "profiles" / profile
    no_links(home)
    require(home.is_dir(), "Missing selected profile")
    if profile != "default":
        require(
            not os.path.lexists(root / "profiles/.deleted" / profile),
            "Selected profile is deleted",
        )
        require(
            any(
                (home / name).is_file() or (home / name).is_symlink()
                for name in (
                    "config.yaml",
                    ".env",
                    "SOUL.md",
                    "profile.yaml",
                    "auth.json",
                    "state.db",
                )
            ),
            "Profile has no identity marker",
        )
    # -S suppresses .pth/sitecustomize; keep the lexical venv executable, not its
    # resolved base interpreter. Never import Hermes or execute its CLI for metadata.
    result = subprocess.run(
        [str(python), "-I", "-S", "-B", "--version"],
        cwd="/",
        env={
            "HOME": "/dev/null",
            "HERMES_HOME": "/dev/null",
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8",
        },
        capture_output=True,
        text=True,
        timeout=10,
        check=True,
    )
    version = result.stdout.strip()
    require(
        re.fullmatch(r"Python [0-9]+\.[0-9]+\.[0-9]+[a-z0-9.+-]*", version),
        "Unrecognized Python version output",
    )
    differences = []
    for name, sha in manifest["tested_backend"]["reference_files"].items():
        try:
            path = backend / name
            no_links(path)
            match = digest(read_regular(path)) == sha
        except (OSError, ValueError):
            match = False
        if not match:
            differences.append(name)
    return {
        **selection,
        "python_version": version.removeprefix("Python "),
        "profile_home": str(home),
        "compatibility": "untested" if differences else "reference-match",
        "reference_differences": differences,
        "tested_revision": manifest["tested_backend"]["revision"],
        "limitations": "Reference files only; extra source files, dependencies, import resolution and runtime functionality not exercised. Running services not inspected.",
    }


def safe_destination(root, selection):
    no_links(root)
    require(root.parent.is_dir(), "Install parent must already exist")
    protected = [
        absolute_path(selection["backend_root"]),
        absolute_path(selection["hermes_root"]),
        Path.home() / ".local/share/hermes-browser",
        Path.home() / ".local/state/hermes-browser",
    ]
    python = absolute_path(selection["python"])
    protected.append(python.parent)
    if (python.parent.parent / "pyvenv.cfg").is_file():
        protected.append(python.parent.parent)
    if os.environ.get("XDG_DATA_HOME"):
        protected.append(absolute_path(os.environ["XDG_DATA_HOME"]) / "hermes-browser")
    for other in protected:
        other = other.resolve()
        require(
            not (root.is_relative_to(other) or other.is_relative_to(root)),
            f"Install destination overlaps protected path: {other}",
        )


def installed(root):
    no_links(root)
    receipt = load_json(read_regular(root / "installation.json", MAX_JSON))
    keys(receipt, "owner archive_sha256 manifest_sha256 selection")
    require(receipt["owner"] == OWNER, "Foreign installation")
    sha = hex_value(receipt["archive_sha256"])
    hex_value(receipt["manifest_sha256"])
    keys(receipt["selection"], " ".join(SELECTION))
    prefix = "versions/" + sha + "/"
    manifest_bytes = read_regular(root / prefix / "manifest.json", MAX_JSON)
    require(
        digest(manifest_bytes) == receipt["manifest_sha256"],
        "Installed manifest modified",
    )
    manifest = validate_manifest(load_json(manifest_bytes))
    expected = {
        "installation.json",
        "hermes-browser.py",
        prefix + "manifest.json",
        *(prefix + "web/" + name for name in manifest["files"]),
    }
    require(tree_files(root) == expected, "Foreign/incomplete installation files")
    require(
        digest(read_regular(root / "hermes-browser.py")) == manifest["launcher_sha256"],
        "Installed launcher modified",
    )
    # Exhaust validation without retaining a second copy of every installed asset.
    for _ in verified_web(root / prefix / "web", manifest):
        pass
    return receipt, manifest


def publish_new(stage, root):
    # Linux RENAME_NOREPLACE closes the check/rename race even for an empty foreign
    # directory. Ordinary os.rename would silently replace that directory.
    libc = ctypes.CDLL(None, use_errno=True)
    rename = libc.renameat2
    rename.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_uint,
    ]
    rename.restype = ctypes.c_int
    if rename(-100, os.fsencode(stage), -100, os.fsencode(root), 1) != 0:
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(root))


def install_or_inspect(args):
    root = absolute_path(args.install_root)
    if not args.archive:
        require(
            args.command == "inspect"
            and all(getattr(args, name) is None for name in (*SELECTION, "sha256")),
            "Supply archive and all selection options",
        )
        receipt, manifest = installed(root)
        safe_destination(root, receipt["selection"])
        print(
            json.dumps(
                {
                    "installation": str(root),
                    "release": manifest["release"],
                    "runtime": inspect_runtime(receipt["selection"], manifest),
                },
                indent=2,
            )
        )
        return
    require(
        all(getattr(args, name) for name in (*SELECTION, "sha256")),
        "Archive inspection/install requires SHA-256, Python, backend root, Hermes root and profile",
    )
    selection = {name: getattr(args, name) for name in SELECTION}
    # Normalize redundant slashes, but do not resolve the venv interpreter symlink.
    for name in SELECTION[:-1]:
        selection[name] = str(absolute_path(selection[name]))
    safe_destination(root, selection)
    manifest, payload = archive_payload(args.archive, args.sha256)
    runtime = inspect_runtime(selection, manifest)
    receipt = {
        "owner": OWNER,
        "archive_sha256": args.sha256,
        "manifest_sha256": digest(payload["manifest.json"]),
        "selection": selection,
    }
    prefix = "versions/" + args.sha256 + "/"
    files = {
        "installation.json": json_bytes(receipt),
        "hermes-browser.py": read_regular(Path(__file__)),
        **{prefix + name: data for name, data in payload.items()},
    }
    exists = root.exists()
    if exists:
        previous, _ = installed(root)
        require(
            previous == receipt,
            "Different release or runtime/profile selection; maintenance is not supported yet",
        )
    print(
        json.dumps(
            {
                "release": manifest["release"],
                "installation": str(root),
                "runtime": runtime,
                "verification": manifest["verification"],
                "writes": []
                if exists
                else [str(root / name) for name in sorted(files)],
                "temporary_work": "Private sibling staging on confirmed install; runtime probe writes nothing",
                "activation": "None. No backend or configuration changes.",
            },
            indent=2,
        ),
        flush=True,
    )
    if args.command == "inspect":
        return
    if exists:
        print("Already installed; verified without changes.")
        return
    try:
        answer = input("Install these files only? Type yes to confirm [no]: ")
    except EOFError:
        answer = ""
    if answer.strip() != "yes":
        print("Cancelled; no installation writes.")
        return
    safe_destination(root, selection)
    require(
        not os.path.lexists(root),
        "Destination appeared during confirmation; inspect again",
    )
    require(
        inspect_runtime(selection, manifest) == runtime,
        "Runtime changed during confirmation; inspect again",
    )
    stage = Path(tempfile.mkdtemp(prefix=".hermes-browser-stage-", dir=root.parent))
    try:
        for name, data in files.items():
            path = stage / name
            path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            with path.open("xb") as stream:
                stream.write(data)
            path.chmod(0o600)
        installed(stage)
        publish_new(stage, root)
    finally:
        if stage.exists():
            shutil.rmtree(stage)
    print(f"Installed at {root}. Nothing started.")


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)
    pack_parser = sub.add_parser(
        "pack", help="Maintainer: package already-verified UI bytes and a receipt"
    )
    for option in ("web-dir", "receipt", "output"):
        pack_parser.add_argument("--" + option, required=True)
    for command in ("inspect", "install"):
        command_parser = sub.add_parser(command)
        command_parser.add_argument("--install-root", required=True)
        for option in (
            "archive",
            "sha256",
            "python",
            "backend-root",
            "hermes-root",
            "profile",
        ):
            command_parser.add_argument("--" + option)
    args = parser.parse_args()
    try:
        require(sys.platform == "linux", "This increment supports Linux only")
        os.umask(0o077)
        pack(args) if args.command == "pack" else install_or_inspect(args)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        AttributeError,
        EOFError,
        tarfile.TarError,
        subprocess.SubprocessError,
    ) as error:
        print(f"hermes-browser: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
