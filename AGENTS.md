# Hermes Webdesktop

## Layout

- This repository contains both the browser source and its distribution.
- `source/` is an unsquashed, sanitized subtree of Hermes Agent. Develop here;
  no separate source checkout or submodule is needed. See `SOURCE_PROVENANCE.json`.
- `source/apps/desktop/src/` is the React UI; `src/browser/` adapts it for browsers.
  `source/apps/shared/` contains the shared frontend code. `source/web/` is a
  different dashboard, not this UI.
- Root Python scripts, `install.sh`, `installer.pyz`, `hermes-browser.tar.gz`,
  `CURRENT.json` and `SHA256SUMS` are the published installer and release files.
  `source.json` selects their download URL; it is not source-code provenance.

## Boundaries

Never patch, monkey-patch, upgrade or deploy a replacement Hermes backend.
The preserved backend code is not the server installation. Use stock APIs;
leave unsupported features unsupported. Keep the approved terminal plugin
optional. These rules take precedence over upstream guidance under `source/`.

Do not rebuild or replace release files during source-only work. For an approved
installer change, edit `source/scripts/` first and keep the root copies in sync.
Never run tests against real profiles or start/stop the user's server.
Publication and deployment each need explicit approval.

## Development

Use the setup and test commands in the root README. Run npm from `source/`, not
from this repository's root. `build:browser` produces
`source/apps/desktop/dist-browser/`; `dev` and `build` are Electron commands.
For Python tests, use `source/scripts/run_tests.sh` with a local test venv and a
scratch HOME. The installer suite includes the committed archive's HTTPS install
and uninstall test. Browser API checks need a separate clean stock checkout.

Read the relevant nested `AGENTS.md` before changing source. Keep the root
README's install command working and its checksum in `SHA256SUMS` current.
Do not blindly pull upstream into the subtree: review frontend changes, preserve
our browser adaptations, and do not change backend files. Do not reintroduce
private plans or removed history while syncing upstream.
