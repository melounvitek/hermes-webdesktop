"""Computer use toolset — universal (any-model) desktop control via cua-driver.

Drives apps through cua-driver's background primitive (focus-without-raise + pid-scoped
event posting): it does NOT steal the user's cursor, keyboard focus, or Space. Plain
OpenAI function-calling schema; vision models get SOM captures (numbered overlays + AX
tree) and click by index, non-vision models use the AX tree alone. Model-facing guidance
lives in the schema description and each action result's `verdict`.

Modules: `tool.py` (handler, approval gate, response shaping), `backend.py` (abstract
`ComputerUseBackend` + result dataclasses), `cua_backend.py` (default MCP-over-stdio
backend + `cua_backend_parse`/`_session`/`_daemon` siblings), `schema.py` (byte-frozen).
"""

from __future__ import annotations

from tools.computer_use.tool import (  # noqa: F401  (public re-exports)
    handle_computer_use,
    release_computer_use_session,
    set_approval_callback,
    check_computer_use_requirements,
    get_computer_use_schema,
)
