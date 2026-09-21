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
configuration reads, repair, update, service or network operations occur in those
commands. Start runs the stock dashboard in the foreground on loopback; it may
write profile state and contact configured services. Stop interrupts the owned
child gracefully; no draining, force-kill or descendant cleanup is promised.
After a controller crash ownership is unknown; automatic recovery is refused.
Reference-file matches do not certify dependencies or running backend identity.

Update/rollback/uninstall require stopped, known ownership and confirmation. They
never stop or start Hermes. Previous complete installations are retained in the
sibling .history directory; the sibling .run lock survives uninstall. Update may
accept a separately trusted --launcher paired with its archive; it is copied,
not executed. Rollback --to takes a full retained archive SHA-256 (listed by
inspect). Keep using this current trusted tool after rollback: older launchers
cannot coordinate with maintenance and are fenced from starting. Concurrent
legacy installer commands are unsupported. Interrupted cleanup can leave private
staging directories or partial removal; inspect manually, never delete unknown
remnants automatically. Runtime, data and pre-existing plugins are not removed.
"""

import argparse
from contextlib import contextmanager
import ctypes
import fcntl
import gzip
import hashlib
import http.client
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import selectors
import shutil
import signal
import socket
import stat
import struct
import subprocess
import sys
import tarfile
import tempfile
import time
from urllib.parse import quote


MAX_ARCHIVE = 256 * 1024 * 1024
MAX_EXPANDED = 512 * 1024 * 1024
MAX_FILE = 128 * 1024 * 1024
MAX_FILES = 10000
MAX_JSON = 8 * 1024 * 1024
OWNER = "hermes-browser-offline-v1"
CONTROL_OWNER = "hermes-browser-control-v2"
HISTORY_OWNER = "hermes-browser-history-v1"
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


def tree_files(root, strict_dirs=False):
    no_links(root)
    require(root.is_dir(), f"Missing directory: {root}")
    files = set()
    directories = set()
    for parent, dirs, names in os.walk(root, followlinks=False):
        directories.update(
            (Path(parent) / name).relative_to(root).as_posix() for name in dirs
        )
        for name in dirs + names:
            path = Path(parent) / name
            require(not path.is_symlink(), f"Symlink refused: {path}")
        for name in names:
            path = Path(parent) / name
            require(stat.S_ISREG(path.lstat().st_mode), f"Special file refused: {path}")
            files.add(path.relative_to(root).as_posix())
        require(len(files) <= MAX_FILES + 4, "Too many files")
    if strict_dirs:
        expected = {
            str(parent)
            for name in files
            for parent in PurePosixPath(name).parents
            if str(parent) != "."
        }
        require(directories == expected, "Foreign installation directories")
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
        atomic_rename(staged, output)
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


def archive_payload(path, expected, launcher):
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
        manifest["launcher_sha256"] == digest(launcher),
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


def safe_destination(root, selection, *, parent_required=True):
    no_links(root)
    if parent_required:
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
    for owned in (
        root,
        root.with_name(root.name + ".run"),
        root.with_name(root.name + ".history"),
    ):
        no_links(owned)
        for other in protected:
            other = other.resolve()
            require(
                not (owned.is_relative_to(other) or other.is_relative_to(owned)),
                f"Browser namespace overlaps protected path: {other}",
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
    require(
        tree_files(root, strict_dirs=True) == expected,
        "Foreign/incomplete installation files",
    )
    require(
        digest(read_regular(root / "hermes-browser.py")) == manifest["launcher_sha256"],
        "Installed launcher modified",
    )
    # Exhaust validation without retaining a second copy of every installed asset.
    for _ in verified_web(root / prefix / "web", manifest):
        pass
    return receipt, manifest


def atomic_rename(stage, root, exchange=False):
    # Linux NOREPLACE refuses even empty foreign destinations. EXCHANGE switches
    # two complete installations without an absent-root or mixed-file window.
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
    if (
        rename(-100, os.fsencode(stage), -100, os.fsencode(root), 2 if exchange else 1)
        != 0
    ):
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error), str(root))


def install_or_inspect(args):
    root = absolute_path(args.install_root)
    if not args.archive:
        require(
            args.command == "inspect"
            and all(
                getattr(args, name) is None
                for name in (*SELECTION, "sha256", "launcher")
            ),
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
                    "retained_versions": {
                        sha: item[1]["release"]
                        for sha, item in retained(root, receipt["selection"]).items()
                    },
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
    launcher = read_regular(
        absolute_path(args.launcher) if args.launcher else Path(__file__)
    )
    manifest, payload = archive_payload(args.archive, args.sha256, launcher)
    runtime = inspect_runtime(selection, manifest)
    receipt, files = installation_files(args.sha256, selection, payload, launcher)
    exists = root.exists()
    if exists:
        previous, _ = installed(root)
        require(
            previous == receipt,
            "Different release or runtime/profile selection; use explicit stopped-only maintenance",
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
    with stopped_control(root) as control:
        safe_destination(root, selection)
        require(
            not os.path.lexists(root),
            "Destination appeared during confirmation; inspect again",
        )
        require(
            inspect_runtime(selection, manifest) == runtime,
            "Runtime changed during confirmation; inspect again",
        )
        # A leftover history from interrupted removal must not be silently adopted.
        require(
            not os.path.lexists(root.with_name(root.name + ".history")),
            "Existing history requires manual inspection before reinstall",
        )
        fence_control(*control)
        stage = Path(tempfile.mkdtemp(prefix=".hermes-browser-stage-", dir=root.parent))
        try:
            write_installation(stage, files)
            atomic_rename(stage, root)
            sync_directory(root.parent)
        finally:
            if stage.exists():
                shutil.rmtree(stage)
    print(f"Installed at {root}. Nothing started.")


def control_address(root):
    # Linux abstract sockets avoid pathname length limits and stale socket files.
    # Both ends check SO_PEERCRED; only the controller signals its unreaped child.
    return "\0hermes-browser-" + str(os.getuid()) + "-" + digest(os.fsencode(root))


def same_user(connection):
    _, uid, _ = struct.unpack(
        "3i", connection.getsockopt(socket.SOL_SOCKET, socket.SO_PEERCRED, 12)
    )
    require(uid == os.getuid(), "Foreign lifecycle controller/client")


def control_request(root, command):
    with socket.socket(socket.AF_UNIX) as connection:
        connection.settimeout(2)
        connection.connect(control_address(root))
        same_user(connection)
        connection.sendall(command.encode() + b"\n")
        with connection.makefile("rb") as stream:
            result = load_json(stream.readline(MAX_JSON + 1))
        require(result["installation"] == str(root), "Controller installation mismatch")
        return result


def write_control(stream, record):
    stream.seek(0)
    stream.write(json_bytes(record))
    stream.truncate()
    stream.flush()
    os.fsync(stream.fileno())


def read_control(stream, root):
    stream.seek(0)
    record = load_json(stream.read(MAX_JSON + 1))
    keys(record, "owner installation state generation pid")
    require(
        record["owner"] in (OWNER, CONTROL_OWNER)
        and record["installation"] == str(root)
        and record["state"] in ("stopped", "unknown"),
        "Foreign lifecycle state",
    )
    generation, pid = record["generation"], record["pid"]
    if generation is not None:
        hex_value(generation, 32)
    require(
        pid is None or (type(pid) is int and pid > 1 and generation is not None),
        "Invalid process diagnostic",
    )
    require(
        record["state"] == "stopped" or generation is not None,
        "Missing process generation",
    )
    return record


def open_control(root, create, owner=CONTROL_OWNER):
    path = root.with_name(root.name + ".run")
    try:
        return os.fdopen(
            os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK), "r+b"
        )
    except FileNotFoundError:
        if not create:
            return None
    # Publish an initialized, already-locked inode, never a visible empty file.
    # Keep this inode stable for its entire lifetime; contenders can hold it open.
    with tempfile.TemporaryDirectory(
        prefix=".hermes-browser-state-", dir=root.parent
    ) as temp:
        staged = Path(temp) / "state"
        stream = staged.open("x+b")
        try:
            staged.chmod(0o600)
            write_control(
                stream,
                {
                    "owner": owner,
                    "installation": str(root),
                    "state": "stopped",
                    "generation": None,
                    "pid": None,
                },
            )
            fcntl.flock(stream, fcntl.LOCK_EX)
            atomic_rename(staged, path)
            sync_directory(root.parent)
            return stream
        except FileExistsError:
            stream.close()
            return os.fdopen(
                os.open(path, os.O_RDWR | os.O_NOFOLLOW | os.O_NONBLOCK), "r+b"
            )
        except BaseException:
            stream.close()
            raise


def startup_configuration(selection, runtime):
    require(
        not os.environ.get("HERMES_MANAGED_DIR") and not Path("/etc/hermes").exists(),
        "Managed configuration is unsupported; nothing started",
    )
    home = Path(runtime["profile_home"])
    require(
        not os.path.lexists(home / ".container-mode"),
        "Container routing is unsupported; nothing started",
    )
    for path in (
        home / ".env",
        home / ".op.env",
        Path(selection["backend_root"]) / ".env",
    ):
        no_links(path)
        if os.path.lexists(path):
            text = read_regular(path, MAX_JSON).decode("utf-8-sig")
            # Stock sanitizes assignments before loading them. Refuse formats
            # needing a rewrite instead of letting startup edit configuration.
            require(
                not text
                or (
                    text.endswith("\n")
                    and "\r" not in text
                    and all(
                        not line.strip()
                        or line.lstrip().startswith("#")
                        or line == line.strip()
                        for line in text.split("\n")[:-1]
                    )
                ),
                "Dotenv normalization required; nothing started",
            )
            # Deliberately conservative, including comments/values. Do not
            # reimplement dotenv quoting, interpolation or override precedence.
            require(
                "\0" not in text
                and not re.search(
                    r"HERMES_(?:HOME|WEB_DIST|DISABLE_LAZY_INSTALLS|LAZY_INSTALL_TARGET|SERVE_HEADLESS|MANAGED_DIR|DESKTOP|PARENT_)|PYTHON",
                    text,
                ),
                "Configuration contains reserved launch controls; nothing started",
            )
    path = home / "config.yaml"
    no_links(path)
    config = read_regular(path, MAX_JSON) if os.path.lexists(path) else b"{}"
    # Use the runtime's existing YAML parser, not Hermes loaders (which sanitize
    # dotenv files and fetch secrets). -S prevents .pth/sitecustomize execution;
    # add package directories explicitly, keeping the lexical venv prefix.
    code = """import sys, site
try:
    sys.path.extend(site.getsitepackages([sys.argv[1]]) + site.getsitepackages())
    import yaml
    value = yaml.safe_load(sys.stdin.buffer.read())
    if value is not None and not isinstance(value, dict):
        sys.exit(2)
    sys.exit(3 if (value or {}).get("secrets") else 0)
except Exception:
    sys.exit(2)
"""
    result = subprocess.run(
        [
            selection["python"],
            "-I",
            "-S",
            "-B",
            "-c",
            code,
            str(Path(selection["python"]).parent.parent),
        ],
        input=config,
        cwd="/",
        env={
            "HOME": "/dev/null",
            "HERMES_HOME": "/dev/null",
            "PATH": "/usr/bin:/bin",
            "LANG": "C.UTF-8",
        },
        capture_output=True,
        timeout=10,
    )
    require(
        result.returncode == 0,
        "External secret sources or unreadable configuration/parser are unsupported; nothing started",
    )


def current_disk(root):
    try:
        receipt, manifest = installed(root)
        return inspect_runtime(receipt["selection"], manifest)
    except (
        OSError,
        ValueError,
        KeyError,
        TypeError,
        subprocess.SubprocessError,
    ) as error:
        return {"compatibility": "unavailable", "error": str(error)}


def check_ready(port, asset, info):
    # Direct loopback, no proxy environment, redirects, credentials or auth bypass.
    for path, limit in (("/api/health", MAX_JSON), ("/" + quote(asset), info["size"])):
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
        try:
            connection.request("GET", path)
            response = connection.getresponse()
            data = response.read(limit + 1)
            require(
                response.status == 200 and len(data) <= limit,
                "Readiness HTTP check failed",
            )
            if path == "/api/health":
                require(
                    load_json(data).get("ok") is True, "Health response is not ready"
                )
            else:
                require(
                    len(data) == info["size"] and digest(data) == info["sha256"],
                    "Served browser asset mismatch",
                )
        finally:
            connection.close()


def run_foreground(args, root, stream, record, receipt, manifest, runtime):
    selection = receipt["selection"]
    backend = Path(selection["backend_root"])
    for name in (".update-incomplete", ".lazy-refresh-incomplete"):
        require(
            not os.path.lexists(backend / name),
            "Pending backend repair; refusing startup",
        )
    require(
        runtime["compatibility"] == "reference-match",
        "Untested backend references; startup unsupported",
    )
    startup_configuration(selection, runtime)
    asset = next(
        (
            name
            for name in sorted(manifest["files"])
            if name.startswith("assets/") and name.endswith(".js")
        ),
        None,
    )
    require(asset is not None, "No browser JavaScript asset available for readiness")
    with socket.socket() as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", args.port))
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("HERMES_DESKTOP", "HERMES_PARENT_", "PYTHON"))
        and key not in ("HERMES_SERVE_HEADLESS", "HERMES_LAZY_INSTALL_TARGET")
    }
    env.update(
        HERMES_HOME=selection["hermes_root"],
        HERMES_WEB_DIST=str(root / "versions" / receipt["archive_sha256"] / "web"),
        HERMES_DISABLE_LAZY_INSTALLS="1",
    )
    command = [
        selection["python"],
        "-E",
        "-s",
        "-B",
        "-u",
        # Match the stock console entry point. `-m hermes_cli.main` also loads
        # it as __main__; sibling imports then execute profile selection twice.
        "-c",
        "from hermes_cli.main import main; main()",
        "-p",
        selection["profile"],
        "dashboard",
        "--host",
        "127.0.0.1",
        "--port",
        str(args.port),
        "--isolated",
        "--skip-build",
        "--no-open",
    ]
    info = {
        "installation": str(root),
        "state": "starting",
        "generation": os.urandom(16).hex(),
        "pid": None,
        "url": f"http://127.0.0.1:{args.port}/",
        "startup": runtime,
    }
    child = None
    requested = False
    failed = False

    def request_stop(_signum, _frame):
        nonlocal requested
        requested = True

    previous = {
        sig: signal.signal(sig, request_stop)
        for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    }
    try:
        with (
            socket.socket(socket.AF_UNIX) as server,
            selectors.DefaultSelector() as selector,
        ):
            server.bind(control_address(root))
            server.listen(8)
            selector.register(server, selectors.EVENT_READ)
            # Persist uncertainty BEFORE spawn. A crash in any subsequent window
            # can never turn a leftover child into permission to start or kill one.
            record.update(state="unknown", generation=info["generation"], pid=None)
            write_control(stream, record)
            child = subprocess.Popen(
                command,
                cwd=backend,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                start_new_session=True,
            )
            info["pid"] = record["pid"] = child.pid
            write_control(stream, record)
            os.set_blocking(child.stdout.fileno(), False)
            selector.register(child.stdout, selectors.EVENT_READ)
            deadline = time.monotonic() + args.timeout
            pending = b""
            print(json.dumps(info), flush=True)
            while child.poll() is None:
                if info["state"] == "starting" and time.monotonic() >= deadline:
                    print(
                        "Readiness deadline expired; requesting graceful stop",
                        file=sys.stderr,
                    )
                    requested = failed = True
                if requested and info["state"] != "stopping":
                    child.terminate()
                    info["state"] = "stopping"
                for key, _ in selector.select(0.1):
                    if key.fileobj is server:
                        connection, _ = server.accept()
                        with connection:
                            connection.settimeout(0.5)
                            try:
                                same_user(connection)
                                with connection.makefile("rb") as request:
                                    action = request.readline(32)
                                require(
                                    action in (b"status\n", b"stop\n"),
                                    "Invalid control request",
                                )
                                if action == b"stop\n":
                                    requested = True
                                connection.sendall(json.dumps(info).encode() + b"\n")
                            except (OSError, ValueError):
                                # A disconnected/malformed client must not stop the server.
                                continue
                    else:
                        data = os.read(child.stdout.fileno(), 65536)
                        if not data:
                            selector.unregister(child.stdout)
                            continue
                        sys.stderr.buffer.write(data)
                        sys.stderr.buffer.flush()
                        pending += data
                        lines = pending.split(b"\n")
                        pending = lines.pop()[-4096:]
                        if (
                            info["state"] == "starting"
                            and f"HERMES_DASHBOARD_READY port={args.port}".encode()
                            in lines
                        ):
                            try:
                                check_ready(args.port, asset, manifest["files"][asset])
                                require(
                                    child.poll() is None,
                                    "Dashboard exited during readiness",
                                )
                                info["state"] = "ready"
                                print(json.dumps(info), flush=True)
                                print(
                                    f"Browser ready: {info['url']} (foreground; Ctrl-C to stop)",
                                    file=sys.stderr,
                                    flush=True,
                                )
                            except (
                                OSError,
                                ValueError,
                                http.client.HTTPException,
                            ) as error:
                                print(f"Readiness failed: {error}", file=sys.stderr)
                                requested = failed = True
            failed = failed or (
                not requested and (info["state"] != "ready" or child.returncode != 0)
            )
    finally:
        if child is not None:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    print(
                        "Child exit unconfirmed; ownership remains unknown. No force-kill.",
                        file=sys.stderr,
                    )
            child.stdout.close()
        if child is None or child.returncode is not None:
            record["state"] = "stopped"
            write_control(stream, record)
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 1 if failed else 0


def acquire_control(stream):
    meta = os.fstat(stream.fileno())
    require(
        stat.S_ISREG(meta.st_mode)
        and meta.st_uid == os.getuid()
        and meta.st_nlink == 1
        and meta.st_mode & 0o777 == 0o600,
        "Unsafe lifecycle state file",
    )
    try:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def fence_control(stream, record):
    require(
        record["state"] == "stopped",
        "Ownership unknown after controller loss; automatic recovery refused",
    )
    if record["owner"] != CONTROL_OWNER:
        # Same inode, upgraded under its lock. Legacy start validates the old
        # owner after locking, so it cannot launch a pre-maintenance selection.
        record["owner"] = CONTROL_OWNER
        write_control(stream, record)


@contextmanager
def stopped_control(root):
    # A declined/invalid operation must leave legacy startup usable. Install and
    # maintenance fence this same inode only after confirmation and revalidation.
    with open_control(root, True, owner=OWNER) as stream:
        require(
            acquire_control(stream),
            "Controller or maintenance is active; refusing mutation",
        )
        record = read_control(stream, root)
        require(record["state"] == "stopped", "Ownership unknown; refusing mutation")
        yield stream, record


def installation_files(sha, selection, payload, launcher):
    receipt = {
        "owner": OWNER,
        "archive_sha256": sha,
        "manifest_sha256": digest(payload["manifest.json"]),
        "selection": selection,
    }
    prefix = "versions/" + sha + "/"
    return receipt, {
        "installation.json": json_bytes(receipt),
        "hermes-browser.py": launcher,
        **{prefix + name: data for name, data in payload.items()},
    }


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def sync_installation(root):
    for name in tree_files(root, strict_dirs=True):
        with (root / name).open("rb") as stream:
            os.fsync(stream.fileno())
    for path in sorted(
        (p for p in root.rglob("*") if p.is_dir()),
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        sync_directory(path)
    sync_directory(root)


def write_installation(root, files):
    root.mkdir(mode=0o700, exist_ok=True)
    for name, data in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        with path.open("xb") as stream:
            stream.write(data)
        path.chmod(0o600)
    installed(root)
    sync_installation(root)


def retained(root, selection):
    history = root.with_name(root.name + ".history")
    no_links(history)
    if not os.path.lexists(history):
        return {}
    require(history.is_dir(), "Foreign history path")
    marker = load_json(read_regular(history / "owner.json", MAX_JSON))
    require(
        marker == {"owner": HISTORY_OWNER, "installation": str(root)},
        "Foreign history owner",
    )
    entries = {}
    for path in history.iterdir():
        if path.name == "owner.json":
            continue
        sha = hex_value(path.name)
        receipt, manifest = installed(path)
        require(
            receipt["archive_sha256"] == sha and receipt["selection"] == selection,
            "Foreign history selection",
        )
        entries[sha] = receipt, manifest
    return entries


def retain_current(root, receipt):
    history = root.with_name(root.name + ".history")
    entries = retained(root, receipt["selection"])
    sha = receipt["archive_sha256"]
    if sha in entries:
        require(entries[sha][0] == receipt, "Retained installation differs")
        return
    if not history.exists():
        with tempfile.TemporaryDirectory(
            prefix=".hermes-browser-history-", dir=root.parent
        ) as temp:
            staged = Path(temp) / "history"
            staged.mkdir(mode=0o700)
            marker = staged / "owner.json"
            with marker.open("xb") as stream:
                stream.write(
                    json_bytes({"owner": HISTORY_OWNER, "installation": str(root)})
                )
                stream.flush()
                os.fsync(stream.fileno())
            sync_directory(staged)
            atomic_rename(staged, history)
            sync_directory(root.parent)
    with tempfile.TemporaryDirectory(
        prefix=".hermes-browser-retain-", dir=root.parent
    ) as temp:
        staged = Path(temp) / "installation"
        shutil.copytree(root, staged, symlinks=True)
        require(installed(staged)[0] == receipt, "Installation changed while retaining")
        sync_installation(staged)
        atomic_rename(staged, history / sha)
        sync_directory(history)


def remove_installation(root):
    installed(root)
    files = tree_files(root, strict_dirs=True)
    directories = {
        root / parent
        for name in files
        for parent in PurePosixPath(name).parents
        if str(parent) != "."
    }
    # Delete only validated inventory, not a recursive namespace. A concurrent
    # foreign addition makes rmdir fail, preserving the foreign entry.
    for name in sorted(files):
        (root / name).unlink()
    for path in sorted(directories, key=lambda p: len(p.parts), reverse=True):
        path.rmdir()
    root.rmdir()
    sync_directory(root.parent)


def maintenance(args, confirm=None):
    root = absolute_path(args.install_root)
    initial, _ = installed(root)
    safe_destination(root, initial["selection"])
    with stopped_control(root) as control:
        current, _ = installed(root)
        selection = current["selection"]
        safe_destination(root, selection)
        versions = retained(root, selection)
        history = root.with_name(root.name + ".history")
        files = target = target_manifest = runtime = None
        if args.command == "update":
            launcher = read_regular(
                absolute_path(args.launcher) if args.launcher else Path(__file__)
            )
            target_manifest, payload = archive_payload(
                args.archive, args.sha256, launcher
            )
            target, files = installation_files(
                args.sha256, selection, payload, launcher
            )
        elif args.command == "rollback":
            sha = hex_value(args.to)
            if sha == current["archive_sha256"]:
                print("Already selected; verified without installation changes.")
                return
            require(sha in versions, "Requested version is not retained")
            target, target_manifest = versions[sha]
        if target is not None:
            runtime = inspect_runtime(selection, target_manifest)
            require(
                runtime["compatibility"] == "reference-match",
                "Candidate backend references do not match",
            )
            if target == current:
                print("Already selected; verified without installation changes.")
                return
        preview = {
            "command": args.command,
            "installation": str(root),
            "current": current["archive_sha256"],
            "target": target["archive_sha256"] if target else None,
            "selection": selection,
            "retained_versions": sorted(versions),
            "action": "Remove verified installation and all retained snapshots"
            if args.command == "uninstall"
            else "Retain current installation, then atomically switch complete directories",
            "preserved": [
                str(root.with_name(root.name + ".run")),
                selection["backend_root"],
                selection["hermes_root"],
            ],
            "activation": "None; no processes started or stopped",
        }
        if confirm is None:
            print(json.dumps(preview, indent=2), flush=True)
            try:
                answer = input(
                    "Proceed with these owned files only? Type yes to confirm [no]: "
                )
            except EOFError:
                answer = ""
            accepted = answer == "yes"
        else:
            # The friendly entry point prompts on /dev/tty under this same lock.
            accepted = confirm(preview)
        if not accepted:
            print("Cancelled; no installation changes.")
            return
        safe_destination(root, selection)
        require(
            installed(root)[0] == current and retained(root, selection) == versions,
            "Installation/history changed during confirmation",
        )
        if args.command == "uninstall":
            fence_control(*control)
            for sha in sorted(versions):
                remove_installation(history / sha)
            if history.exists():
                (history / "owner.json").unlink()
                history.rmdir()
                sync_directory(root.parent)
            remove_installation(root)
            print(
                "Uninstalled verified browser files; lifecycle lock, Hermes and data preserved."
            )
            return
        require(
            inspect_runtime(selection, target_manifest) == runtime,
            "Runtime changed during confirmation",
        )
        fence_control(*control)
        private = Path(
            tempfile.mkdtemp(prefix=".hermes-browser-switch-", dir=root.parent)
        )
        staged = private / "installation"
        complete = False
        try:
            if files is not None:
                write_installation(staged, files)
            else:
                shutil.copytree(
                    history / target["archive_sha256"], staged, symlinks=True
                )
                sync_installation(staged)
            require(installed(staged)[0] == target, "Candidate changed during staging")
            complete = True
            retain_current(root, current)
            # Revalidate the active bytes after copying them into history.
            require(
                installed(root)[0] == current, "Installation changed before switching"
            )
            atomic_rename(staged, root, exchange=True)
            sync_directory(root.parent)
            sync_directory(private)
        finally:
            if complete:
                # After exchange this is the displaced installation. Preserve it
                # if validation fails; never recursively delete unknown contents.
                remove_installation(staged)
                private.rmdir()
                sync_directory(root.parent)
            else:
                shutil.rmtree(private)
        print(f"Selected {target['archive_sha256']}. Nothing started.")


def lifecycle(args):
    root = absolute_path(args.install_root)
    no_links(root)
    receipt = manifest = runtime = None
    if args.command == "start":
        require(
            1 <= args.port <= 65535 and 1 <= args.timeout <= 300,
            "Invalid port/startup timeout",
        )
        receipt, manifest = installed(root)
        safe_destination(root, receipt["selection"])
    if args.command == "stop":
        require(1 <= args.timeout <= 300, "Invalid stop timeout")
    stream = open_control(root, args.command == "start")
    if stream is None:
        installed(root)
        print(json.dumps({"installation": str(root), "state": "stopped"}))
        return 0
    with stream:
        fd = stream.fileno()
        if acquire_control(stream):
            record = read_control(stream, root)
            if args.command == "start":
                fence_control(stream, record)
                # The pre-lock check only establishes ownership of the namespace.
                # Maintenance may have switched it while this command was waiting.
                receipt, manifest = installed(root)
                safe_destination(root, receipt["selection"])
                runtime = inspect_runtime(receipt["selection"], manifest)
                return run_foreground(
                    args, root, stream, record, receipt, manifest, runtime
                )
            print(json.dumps(record))
            return int(args.command == "stop" and record["state"] != "stopped")
        require(
            args.command != "start", "Installation already has a foreground controller"
        )
        try:
            result = control_request(root, args.command)
            if args.command == "stop":
                result["state"] = "stopping"
                deadline = time.monotonic() + args.timeout
                while time.monotonic() < deadline:
                    try:
                        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                        result = read_control(stream, root)
                        break
                    except BlockingIOError:
                        time.sleep(0.1)
            elif "startup" in result:
                result["current_disk"] = current_disk(root)
        except (OSError, ValueError):
            result = {
                "installation": str(root),
                "state": "unknown",
                "detail": "Controller unavailable; no PID signaling or recovery attempted",
            }
            # The controller may have exited between the lock check and connect.
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
                result = read_control(stream, root)
            except (OSError, ValueError):
                pass
        print(json.dumps(result))
        return int(args.command == "stop" and result["state"] != "stopped")


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
            "launcher",
            "python",
            "backend-root",
            "hermes-root",
            "profile",
        ):
            command_parser.add_argument("--" + option)
    for command in ("start", "status", "stop"):
        command_parser = sub.add_parser(command)
        command_parser.add_argument("--install-root", required=True)
        if command == "start":
            command_parser.add_argument("--port", type=int, default=9119)
        if command != "status":
            command_parser.add_argument(
                "--timeout", type=float, default=60 if command == "start" else 10
            )
    for command in ("update", "rollback", "uninstall"):
        command_parser = sub.add_parser(command)
        command_parser.add_argument("--install-root", required=True)
        if command == "update":
            command_parser.add_argument("--archive", required=True)
            command_parser.add_argument("--sha256", required=True)
            command_parser.add_argument(
                "--launcher",
                help="Separately trusted launcher paired with candidate archive; never executed",
            )
        if command == "rollback":
            command_parser.add_argument(
                "--to", required=True, help="Full SHA-256 of a retained archive"
            )
    args = parser.parse_args()
    try:
        require(sys.platform == "linux", "This increment supports Linux only")
        os.umask(0o077)
        if args.command in ("start", "status", "stop"):
            return lifecycle(args)
        if args.command in ("update", "rollback", "uninstall"):
            maintenance(args)
        elif args.command == "pack":
            pack(args)
        else:
            install_or_inspect(args)
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
