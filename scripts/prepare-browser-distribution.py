#!/usr/bin/env python3
"""Prepare local distribution files from an existing verified archive/launcher pair.

No UI build or verification receipt is generated. --source selects the one issuer
at packaging time; nothing is uploaded. The default .invalid URL is deliberately
unusable until a public home is approved. Serve all output files at that URL's
parent. Trust the HTTPS issuer before running its bootstrap; hashes from that
same issuer provide integrity, not independent authentication.
"""

import argparse
from pathlib import Path
import shlex
import sys
import tempfile
import zipfile

sys.path.insert(0, str(Path(__file__).absolute().parent))
from browser_setup import E  # noqa: E402
from browser_download import source_url  # noqa: E402
from browser_install import CONTROL_FILES  # noqa: E402

DEFAULT_SOURCE = "https://hermes-browser.example.invalid/CURRENT.json"


def prepare(args):
    source = source_url(args.source)
    launcher = E.read_regular(E.absolute_path(args.launcher), 1024 * 1024)
    archive = E.read_regular(E.absolute_path(args.archive), E.MAX_ARCHIVE)
    E.archive_payload(args.archive, args.sha256, launcher)
    # The trusted controller is kept outside snapshots and must be this version.
    E.require(
        launcher == E.read_regular(Path(__file__).with_name("hermes-browser.py")),
        "Package with the current trusted engine; repack a verified UI receipt if engine bytes changed",
    )
    files = {
        name: E.read_regular(Path(__file__).with_name(name))
        for name in CONTROL_FILES
        if name != "source.json"
    }
    files["source.json"] = E.json_bytes({"source": source})
    main = """import os, pathlib, subprocess, sys, tempfile, zipfile
os.umask(0o077)
with tempfile.TemporaryDirectory(prefix="hermes-browser-installer-") as directory:
    with zipfile.ZipFile(sys.argv[0]) as archive:
        for name in NAMES:
            pathlib.Path(directory, name).write_bytes(archive.read(name))
    result = subprocess.run([sys.executable, "-I", "-S", "-B", str(pathlib.Path(directory, "browser_install.py")), *sys.argv[1:]])
    sys.exit(result.returncode)
""".replace("NAMES", repr(list(files)))
    output = E.absolute_path(args.output)
    E.no_links(output)
    E.require(not output.exists(), "Output already exists")
    with tempfile.TemporaryDirectory(
        prefix=".browser-distribution-", dir=output.parent
    ) as temporary:
        stage = Path(temporary) / "distribution"
        stage.mkdir(mode=0o700)
        installer = stage / "installer.pyz"
        with zipfile.ZipFile(installer, "w", compression=zipfile.ZIP_DEFLATED) as zip:
            for name, raw in {**files, "__main__.py": main.encode()}.items():
                zip.writestr(name, raw)
        raw = E.read_regular(installer, 2 * 1024 * 1024)
        url = source.rsplit("/", 1)[0] + "/installer.pyz"
        bootstrap = """#!/bin/sh
set -eu
umask 077
command -v curl >/dev/null && command -v sha256sum >/dev/null || { echo 'Existing curl and sha256sum are required; nothing installed.' >&2; exit 1; }
p=''
home=${HERMES_HOME:-"$HOME/.hermes"}
case "$home" in */profiles/*) home=${home%/profiles/*};; esac
for candidate in "$home/hermes-agent/venv/bin/python" "$home/hermes-agent/.venv/bin/python" "$HOME/.hermes/hermes-agent/venv/bin/python" "$HOME/.hermes/hermes-agent/.venv/bin/python"; do
    if [ -x "$candidate" ]; then p=$candidate; break; fi
done
if [ -z "$p" ]; then p=$(command -v python3) || { echo 'Existing Python 3.10+ is required; nothing installed.' >&2; exit 1; }; fi
"$p" -I -S -B -c 'import sys; sys.exit(sys.version_info < (3,10))' || { echo 'Existing Python 3.10+ is required; nothing installed.' >&2; exit 1; }
t=$(mktemp -d)
trap 'rm -rf "$t"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
status=$(curl --fail --silent --show-error --proto '=https' --max-time 120 --max-filesize 2097152 --write-out '%{http_code}' URL -o "$t/installer.pyz")
[ "$status" = 200 ] || { echo 'Expected complete HTTPS 200 download; not executed.' >&2; exit 1; }
printf '%s  %s\\n' DIGEST "$t/installer.pyz" | sha256sum --check --status || { echo 'Installer checksum mismatch; not executed.' >&2; exit 1; }
"$p" -I -S -B "$t/installer.pyz" "$@"
""".replace("URL", shlex.quote(url)).replace("DIGEST", shlex.quote(E.digest(raw)))
        (stage / "install.sh").write_text(bootstrap)
        descriptor = {"schema": 1}
        for key, name, data in (
            ("archive", "hermes-browser.tar.gz", archive),
            ("launcher", "hermes-browser.py", launcher),
        ):
            (stage / name).write_bytes(data)
            descriptor[key] = dict(name=name, size=len(data), sha256=E.digest(data))
        (stage / "CURRENT.json").write_bytes(E.json_bytes(descriptor))
        E.sync_installation(stage)
        E.atomic_rename(stage, output)
        E.sync_directory(output.parent)
    entry = source.rsplit("/", 1)[0] + "/install.sh"
    print("Prepared locally; not published. Trust this HTTPS issuer before execution:")
    print(
        't=$(mktemp) && (trap \'rm -f "$t"\' EXIT; status=$(curl --fail --silent --show-error --proto "=https" --max-time 60 --max-filesize 65536 --write-out "%{http_code}" '
        + shlex.quote(entry)
        + ' -o "$t") && [ "$status" = 200 ] && sh "$t")'
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("archive", "sha256", "launcher", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--source", default=DEFAULT_SOURCE)
    args = parser.parse_args()
    try:
        prepare(args)
    except (OSError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
