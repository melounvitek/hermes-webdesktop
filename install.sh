#!/bin/sh
set -eu
umask 077
command -v curl >/dev/null && command -v sha256sum >/dev/null || { echo 'Existing curl and sha256sum are required; nothing installed.' >&2; exit 1; }
p=''
home=${HERMES_HOME:-"$HOME/.hermes"}
case "$home" in */profiles/*) home=${home%/profiles/*};; esac
custom=${HERMES_INSTALL_DIR:-"$home/hermes-agent"}
for candidate in "$custom/venv/bin/python" "$custom/.venv/bin/python" "$home/hermes-agent/venv/bin/python" "$home/hermes-agent/.venv/bin/python" "$HOME/.hermes/hermes-agent/venv/bin/python" "$HOME/.hermes/hermes-agent/.venv/bin/python"; do
    if [ -x "$candidate" ]; then p=$candidate; break; fi
done
if [ -z "$p" ]; then p=$(command -v python3) || { echo 'Existing Python 3.10+ is required; nothing installed.' >&2; exit 1; }; fi
"$p" -I -S -B -c 'import sys; sys.exit(sys.version_info < (3,10))' || { echo 'Existing Python 3.10+ is required; nothing installed.' >&2; exit 1; }
t=$(mktemp -d)
trap 'rm -rf "$t"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
status=$(curl -q --fail --silent --show-error --proto '=https' --max-time 120 --max-filesize 2097152 --write-out '%{http_code}' https://raw.githubusercontent.com/melounvitek/hermes-webdesktop/main/installer.pyz -o "$t/installer.pyz")
[ "$status" = 200 ] || { echo 'Expected complete HTTPS 200 download; not executed.' >&2; exit 1; }
printf '%s  %s\n' fdf804533ca1f06d41f4b352ac712a9a36e681589adea90f1a6291d462340e08 "$t/installer.pyz" | sha256sum --check --status || { echo 'Installer checksum mismatch; not executed.' >&2; exit 1; }
"$p" -I -S -B "$t/installer.pyz" "$@"
