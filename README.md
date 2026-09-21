# Hermes Webdesktop

The [Hermes Desktop](https://hermes-agent.nousresearch.com/desktop), just running in your browser. No Electron or frontend build required. Unofficial project, obviously :-).

## Install

Requires Linux, a configured Hermes installation, and `curl`:

```sh
t=$(mktemp) && (trap 'rm -f "$t"' EXIT; status=$(curl -q --fail --silent --show-error --proto "=https" --max-time 60 --max-filesize 65536 --write-out "%{http_code}" https://raw.githubusercontent.com/melounvitek/hermes-webdesktop/main/install.sh -o "$t") && [ "$status" = 200 ] && sh "$t")
```

Run `hermes-browser start`, then open **http://127.0.0.1:9119/**. Press Ctrl-C to stop.

## Using Cloudflare Tunnel

You can point Cloudflare Tunnel at `http://127.0.0.1:<port>`, where the browser launcher runs Hermes. No separate `hermes serve` is needed.

**Set up authentication before exposing it.** HTTPS encrypts traffic; it doesn’t restrict who can use your agent.

In your Hermes profile’s `config.yaml`, merge these settings into the existing `dashboard` section:

```yaml
dashboard:
  public_url: "https://hermes.your-domain.com"
  basic_auth:
    username: "admin"
    password_hash: "<generated hash>"
    secret: "<generated secret>"
```

Use your actual Cloudflare hostname. Generate the hash and secret from your Hermes installation directory, with its Python environment activated:

```bash
python -c 'from getpass import getpass; from plugins.dashboard_auth.basic import hash_password; print(hash_password(getpass("Password: ")))'
python -c 'import secrets; print(secrets.token_urlsafe(32))'
```

Paste the outputs into the matching fields, then restart the browser launcher when no work is running.

**Don’t skip `public_url`:** on loopback, setting a password alone doesn’t require visitors to log in.

Open the public URL in a private browser window and confirm it requires login before showing your chats.

## Screenshot

![Hermes Webdesktop in Chrome with a demo conversation](assets/screenshot.png)
