# Browser development source

Follow `../AGENTS.md` and the development commands in `../README.md`.
This is a selected frontend tree, not a Hermes backend checkout.

- Keep `apps/desktop/` and `apps/shared/` at their existing paths. Read
  `apps/desktop/AGENTS.md` and `apps/desktop/src/AGENTS.md` for UI conventions;
  their upstream backend references are context, not files to restore here.
- Retain Electron code and the optional terminal plugin. Extracting them is a
  separate task. Native application packaging is not this repository's release.
- Keep frontend-generated gateway contracts and the slash registry committed.
  Compare them with a separate stock checkout when updating API compatibility;
  ordinary builds must not import or run backend generators.
- Shared UI state belongs in small nanostores. Keep routes thin, use existing
  components, and test behavior rather than source text.
- Python tests use `scripts/run_tests.sh`, a test venv and a scratch HOME. The
  default suite needs no backend. Plugin integration requires an explicit stock
  checkout and its separate Python environment; never use a live profile.
- Never patch backend code, load real credentials, send live model requests,
  or change services while testing. Preserve the optional plugin unchanged
  during source-only cleanup.
