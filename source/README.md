# Browser development source

Setup, build, test and release instructions are in the [repository README](../README.md#development).

`apps/desktop/` contains the browser-adapted UI and retained Electron code;
`apps/shared/` contains frontend utilities and generated API contracts.
`scripts/` and `tests/` contain browser installer tooling, fixtures and tests.
`tests-js/` retains frontend dependency/security checks and the mock server.
The optional terminal plugin remains in `apps/desktop/browser-terminal-plugin/`.

The Hermes backend, other applications and unrelated tests are not included in
this working tree. Integration tests use a separate stock Hermes checkout.
Historical commits still contain the original sanitized source import; see
[`SOURCE_PROVENANCE.json`](../SOURCE_PROVENANCE.json).
