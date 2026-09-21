# Hermes Webdesktop: official desktop app, just served as webapp

An unofficial browser-based wrapper for [Hermes Agent](https://github.com/NousResearch/hermes-agent) desktop app.Prebuilt web interface and a small installer, no Electron or frontend build tools needed.

## Installation
Use Linux with an existing, configured Hermes installation, `curl`, `sha256sum` and Python 3.10+ (the installer can reuse Hermes’s Python). Hermes is not pinned to a commit or file hashes. Startup and session-list browser checks passed on stock Hermes [`30de041`](https://github.com/NousResearch/hermes-agent/commit/30de041b011aa3d3830a7ffa05815e2cb2f063be) and [`2ed6387`](https://github.com/NousResearch/hermes-agent/commit/2ed6387d87b4db091af2f05db32faab6e0dbb9a2); these are test references, not an allowlist. Setup checks prerequisites; startup checks dashboard readiness and the served browser assets. This does not guarantee every API or model workflow, and some incompatibilities can only be detected at startup. Run the command below in a terminal. It detects Hermes and your profile, verifies the download, and asks before installing.

```sh
t=$(mktemp) && (trap 'rm -f "$t"' EXIT; status=$(curl -q --fail --silent --show-error --proto "=https" --max-time 60 --max-filesize 65536 --write-out "%{http_code}" https://raw.githubusercontent.com/melounvitek/hermes-webdesktop/main/install.sh -o "$t") && [ "$status" = 200 ] && sh "$t")
```

After installation, run `hermes-browser start` (or `~/.local/bin/hermes-browser start` if that directory is not on your PATH). Wait for the ready message, then open `http://127.0.0.1:9119/`. It runs in the foreground; Ctrl-C stops it. Installation does not start a service or configure remote access.

If you installed the earlier hash-pinned installer, stop any running browser, run `hermes-browser uninstall`, then rerun the installation command above. A normal `update` or repeated setup does not replace the installed controller. This one-time reinstall removes retained browser rollback versions, but preserves Hermes, its data and plugins.

## About
The web interface is derived from the official Hermes dashboard, but this is an independently maintained distribution, not an official release or a fork of the backend. You install and maintain official Hermes separately; this installer never installs, upgrades or patches it. The upstream MIT license and copyright notice are retained in [LICENSE](LICENSE).
