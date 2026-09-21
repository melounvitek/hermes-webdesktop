#!/usr/bin/env python3
"""Read-only discovery for the existing-Hermes browser installer."""

import importlib.util
import os
from pathlib import Path
import sys


def engine():
    spec = importlib.util.spec_from_file_location(
        "browser_engine", Path(__file__).with_name("hermes-browser.py")
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


E = engine()


def choose(candidates, option):
    candidates = list(dict.fromkeys(candidates))
    E.require(
        candidates,
        f"Missing existing Hermes {option}; supply --{option}. Nothing will be installed or repaired.",
    )
    E.require(
        len(candidates) == 1,
        f"Ambiguous {option}; select with --{option}: "
        + ", ".join(map(str, candidates)),
    )
    return candidates[0]


def detect(args):
    # Mirror stock's profile-shaped HERMES_HOME without importing Hermes or .env.
    home = E.absolute_path(
        args.hermes_home
        or os.environ.get("HERMES_HOME", "").strip()
        or str(Path.home() / ".hermes")
    )
    E.no_links(home)
    root = home.parent.parent if home.parent.name == "profiles" else home
    E.require(
        root.is_dir(),
        "Missing existing Hermes data home; supply --hermes-home. Nothing will be repaired.",
    )
    profile = args.profile
    if profile is None:
        if home != root:
            profile = home.name
        else:
            active = root / "active_profile"
            profile = (
                E.read_regular(active, 1024).decode().strip()
                if os.path.lexists(active)
                else "default"
            )
    if args.backend_root:
        backend = E.absolute_path(args.backend_root)
    else:
        candidates = [root / "hermes-agent", Path.home() / ".hermes/hermes-agent"]
        # Only a symlink to a recognized checkout CLI is evidence. Never execute
        # or parse an arbitrary PATH wrapper (including shell/dotenv contents).
        for directory in os.environ.get("PATH", "").split(os.pathsep):
            path = Path(directory) / "hermes"
            if path.is_absolute() and path.is_symlink():
                target = path.resolve()
                if target.name == "main.py" and target.parent.name == "hermes_cli":
                    candidates.append(target.parent.parent)
        backend = choose(
            [p for p in candidates if (p / "hermes_cli/main.py").is_file()],
            "backend-root",
        )
    E.no_links(backend)
    E.read_regular(backend / "hermes_cli/main.py")
    python = (
        E.absolute_path(args.python)
        if args.python
        else choose(
            [
                backend / name / "bin/python"
                for name in ("venv", ".venv")
                if (backend / name / "bin/python").is_file()
            ],
            "python",
        )
    )
    return dict(
        python=str(python),
        backend_root=str(backend),
        hermes_root=str(root),
        profile=profile,
    )


def preflight(selection, manifest):
    E.require(sys.platform == "linux", "Only Linux is supported")
    runtime = E.inspect_runtime(selection, manifest)
    E.require(
        tuple(map(int, runtime["python_version"].split(".")[:2])) >= (3, 10),
        "Existing Python 3.10+ is required; no runtime will be installed",
    )
    E.require(
        runtime["compatibility"] == "reference-match",
        "Unsupported Hermes backend: bundle reference files do not match. No backend changes will be made.",
    )
    return runtime
