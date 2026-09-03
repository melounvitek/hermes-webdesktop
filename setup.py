"""setup.py — wheel/sdist build guard.

pip/PyPI and Homebrew are not supported distribution methods: a wheel would ship without the
bundled assets (locales, skills, optional-mcps, web_dist, tui_dist, plugin manifests) that the nix
wrapper / source checkout resolve at runtime. The ``sdist`` and ``bdist_wheel`` commands raise
outside a Nix build; ``setuptools.build_meta`` calls them for every PEP 517 path (``uv build``,
``pip wheel``, ``python -m build``). uv2nix builds inside the Nix sandbox with
``HERMES_NIX_BUILD=1`` (set by ``nix/python.nix``). Editable installs (``uv sync``,
``pip install -e .``) use ``build_editable`` → ``build_ext``, so development is unaffected.
"""

import os

from setuptools import setup
from setuptools.command.sdist import sdist

_IN_NIX_BUILD = os.environ.get("HERMES_NIX_BUILD") == "1"

_BLOCK_MESSAGE = (
    "Building wheels or sdists for hermes-agent is not supported.\n"
    "Hermes is distributed via the shell installer, Docker image, or Nix.\n"
    "See: https://hermes-agent.nousresearch.com/docs/getting-started/installation\n"
    "\n"
    "If you are developing, use an editable install instead:\n"
    "  uv sync          # or: uv pip install -e .\n"
    "\n"
    "If you are building with Nix (uv2nix), this error should not fire —\n"
    "the Hermes Nix derivation sets HERMES_NIX_BUILD=1. If it does, file a bug."
)


def _guarded(base):
    class Guarded(base):
        def run(self, *args, **kwargs):
            if not _IN_NIX_BUILD:
                raise RuntimeError(_BLOCK_MESSAGE)
            return super().run(*args, **kwargs)

    return Guarded


cmdclass = {"sdist": _guarded(sdist)}

# bdist_wheel exists only when `wheel` is installed; a None base class would raise TypeError at
# class-definition time, before the guard could run.
try:
    from setuptools.command.bdist_wheel import bdist_wheel

    cmdclass["bdist_wheel"] = _guarded(bdist_wheel)
except ImportError:
    pass

setup(cmdclass=cmdclass)
