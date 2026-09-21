# Hermes Webdesktop

An unofficial browser-based desktop for [Hermes Agent](https://github.com/NousResearch/hermes-agent). This repository ships a prebuilt web interface and a small installer; no Electron or frontend build tools are needed.

To install, use Linux with an existing, configured Hermes installation, `curl`, `sha256sum` and Python 3.10+ (the installer can reuse Hermes’s Python). The current bundle was verified against stock Hermes [`30de041`](https://github.com/NousResearch/hermes-agent/commit/30de041b011aa3d3830a7ffa05815e2cb2f063be); mismatched backend references are refused, not repaired. Run the command below in a terminal. It detects Hermes and your profile, verifies the download, and asks before installing. **The download command requires this repository to be public.**

```sh
t=$(mktemp) && (trap 'rm -f "$t"' EXIT; status=$(curl -q --fail --silent --show-error --proto "=https" --max-time 60 --max-filesize 65536 --write-out "%{http_code}" https://raw.githubusercontent.com/melounvitek/hermes-webdesktop/main/install.sh -o "$t") && [ "$status" = 200 ] && sh "$t")
```

After installation, run `hermes-browser start` (or `~/.local/bin/hermes-browser start` if that directory is not on your PATH). Wait for the ready message, then open `http://127.0.0.1:9119/`. It runs in the foreground; Ctrl-C stops it. Installation does not start a service or configure remote access.

The web interface is derived from the official Hermes dashboard, but this is an independently maintained distribution, not an official release or a fork of the backend. You install and maintain official Hermes separately; this installer never installs, upgrades or patches it. The upstream MIT license and copyright notice are retained in [LICENSE](LICENSE).
