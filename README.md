# Hermes Webdesktop

The [Hermes Agent](https://github.com/NousResearch/hermes-agent) desktop interface, adapted for the browser. No Electron or frontend build required.

This is an unofficial project. It uses your existing Hermes installation without patching the backend; Hermes itself is installed and updated separately.

## Install

Requires Linux, a configured Hermes installation, `curl` and `sha256sum`.

```sh
t=$(mktemp) && (trap 'rm -f "$t"' EXIT; status=$(curl -q --fail --silent --show-error --proto "=https" --max-time 60 --max-filesize 65536 --write-out "%{http_code}" https://raw.githubusercontent.com/melounvitek/hermes-webdesktop/main/install.sh -o "$t") && [ "$status" = 200 ] && sh "$t")
```

Run `hermes-browser start`, then open **http://127.0.0.1:9119/**. Press Ctrl-C to stop.
