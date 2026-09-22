# Hermes Webdesktop

The [Hermes Desktop](https://hermes-agent.nousresearch.com/desktop), just running in your browser. No Electron or frontend build required. Unofficial project, obviously :-).

## Install

Requires Linux, a configured Hermes installation, and `curl`:

```sh
curl -qfsS --proto '=https' --max-time 60 --max-filesize 65536 https://raw.githubusercontent.com/melounvitek/hermes-webdesktop/main/install.sh -o hermes-browser-install.sh && sh ./hermes-browser-install.sh
```

Run this in a trusted, writable directory: it replaces `hermes-browser-install.sh`. The script runs only after curl succeeds; you must trust this repository. You can remove the downloaded file afterward.

Run `hermes-browser start`, then open **http://127.0.0.1:9119/**. Press Ctrl-C to stop.

## Remote access

**Tailscale Serve is the recommended option:** it keeps access within your tailnet rather than exposing Hermes publicly. Use Serve, not Funnel.

Cloudflare Tunnel is an alternative if you need a public URL. Both can forward to `http://127.0.0.1:9119`; no separate `hermes serve` is needed.

We recommend a Hermes password with either option. Don’t expose a public endpoint without authentication.

In your Hermes profile’s `config.yaml`, merge these settings into the existing `dashboard` section:

```yaml
dashboard:
  public_url: "https://hermes.your-domain.com"
  basic_auth:
    username: "admin"
    password_hash: "<generated hash>"
    secret: "<generated secret>"
```

Use your actual Tailscale or Cloudflare HTTPS URL. Generate the hash and secret from your Hermes installation directory, with its Python environment activated:

```bash
python -c 'from getpass import getpass; from plugins.dashboard_auth.basic import hash_password; print(hash_password(getpass("Password: ")))'
python -c 'import secrets; print(secrets.token_urlsafe(32))'
```

Paste the outputs into the matching fields, then restart the browser launcher when no work is running.

**Don’t skip `public_url`:** on loopback, setting a password alone doesn’t require visitors to log in.

Open the public URL in a private browser window and confirm it requires login before showing your chats.

## Screenshot

![Hermes Webdesktop in Chrome with a demo conversation](assets/screenshot.png)
