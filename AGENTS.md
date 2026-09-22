# Hermes Webdesktop

## Layout

- This repository contains both the browser source and its distribution.
- `source/` contains the browser frontend and its development tools. It was
  selected from an unsquashed, sanitized Hermes import; history is unchanged.
  No backend source checkout is needed to build. See `SOURCE_PROVENANCE.json`.
- `source/apps/desktop/src/` is the React UI; `src/browser/` adapts it for browsers.
  `source/apps/shared/` contains shared frontend code. Other Hermes applications
  and the backend are not included in the current source tree.
- Root Python scripts, `install.sh`, `installer.pyz`, `hermes-browser.tar.gz`,
  `CURRENT.json` and `SHA256SUMS` are the published installer and release files.
  `source.json` selects their download URL; it is not source-code provenance.

## Boundaries

Never patch, monkey-patch, upgrade or deploy a replacement Hermes backend.
Use stock APIs; leave unsupported features unsupported. Integration checks need
an explicit, separate stock checkout. Keep the approved terminal plugin
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
scratch HOME. Its default suite covers the installer, including the committed
archive's HTTPS install/uninstall. Plugin integration is a separate opt-in suite.

Read the relevant nested `AGENTS.md` before changing source. Keep the root
README's install command working and its checksum in `SHA256SUMS` current.
Do not blindly pull upstream into this selected tree: review frontend changes
and preserve our browser adaptations. Do not reintroduce backend packages,
private plans or removed history while syncing upstream.
