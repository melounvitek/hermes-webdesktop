"""Setup wizard for Mem0 plugin — interactive and flag-based modes."""

from __future__ import annotations

import getpass
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from hermes_constants import get_hermes_home  # noqa: F401 — patched by tests

from . import _read_mem0_json
from ._oss_providers import EMBEDDER_PROVIDERS, KNOWN_DIMS, LLM_PROVIDERS, SECTION_REGISTRIES, VECTOR_PROVIDERS, validate_oss_config


def _curses_select(title: str, items: list[tuple[str, str]], default: int = 0) -> int:
    """Interactive single-select with arrow keys."""
    from hermes_cli.curses_ui import curses_radiolist
    return curses_radiolist(title, [f"{label}  {desc}" if desc else label for label, desc in items], selected=default, cancel_returns=default)


def _prompt(label: str, default: str | None = None, secret: bool = False) -> str:
    """Prompt for a value with optional default and secret masking."""
    sys.stdout.write(f"  {label}{f' [{default}]' if default else ''}: ")
    sys.stdout.flush()
    val = getpass.getpass(prompt="") if secret and sys.stdin.isatty() else sys.stdin.readline().strip()
    return val or (default or "")


def _input(label: str, default: str) -> str:
    """input() with a bracketed default shown and applied on blank."""
    return input(f"  {label} [{default}]: ").strip() or default


def _masked(secret: str) -> str:
    return f"...{secret[-4:]}" if len(secret) > 4 else "set"


def _http_get(url: str, path: str, timeout: int):
    return urllib.request.urlopen(urllib.request.Request(f"{url.rstrip('/')}{path}", method="GET"), timeout=timeout)


def _prompt_api_key(label: str, env_var: str, hermes_home: str) -> str:
    """Prompt for API key, showing masked existing value if found."""
    existing = os.environ.get(env_var, "")
    env_path = Path(hermes_home) / ".env"
    if not existing and env_path.exists():
        # utf-8-sig: a Notepad BOM on line 1 would otherwise defeat the key match.
        for line in env_path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            if line.startswith(f"{env_var}="):
                existing = line.split("=", 1)[1].strip()
                break
    hint = f" (current: {_masked(existing)}, blank to keep)" if existing else ""
    return getpass.getpass(f"  {label} API key{hint}: ").strip()


def _api_key_writes(flags: dict, label: str, *, url: str | None = None, fresh_label: str | None = None) -> dict[str, str]:
    """MEM0_API_KEY for .env: from --api-key, else prompt (masking any key already in the environment)."""
    if flags.get("api_key"):
        return {"MEM0_API_KEY": flags["api_key"]}
    existing = os.environ.get("MEM0_API_KEY", "")
    if existing:
        val = _prompt(f"{label} (current: {_masked(existing)}, blank to keep)", secret=True)
    else:
        if url:
            print(f"  Get yours at {url}")
        val = _prompt(fresh_label or label, secret=True)
    return {"MEM0_API_KEY": val} if val else {}


def _print_dry_run(summary: str, env_writes: dict, check=None) -> None:
    print(f"\n  [dry-run] Would save config: {summary}")
    if env_writes:
        print("  [dry-run] Would write API key to .env")
    if check:
        check()
    print("  [dry-run] No files written.\n")


def _print_saved(label: str, env_writes: dict, key_line: str, server: str | None = None) -> None:
    print(f"\n  Memory provider: {label}")
    if server:
        print(f"  Server: {server}")
    print("  Activation saved to config.yaml")
    print("  Provider config saved")
    if env_writes:
        print(f"  {key_line}")
    print("\n  Start a new session to activate.\n")


_FLAG_KEYS = (
    "mode", "api_key", "host",
    "oss_llm", "oss_llm_key", "oss_llm_model", "oss_llm_url",
    "oss_embedder", "oss_embedder_key", "oss_embedder_model", "oss_embedder_url",
    "oss_vector", "oss_vector_path", "oss_vector_url", "oss_vector_host",
    "oss_vector_port", "oss_vector_user", "oss_vector_password", "oss_vector_dbname",
    "user_id",
)
_FLAG_DEFAULTS = {"oss_llm": "openai", "oss_embedder": "openai", "oss_vector": "qdrant"}
# --oss-vector-<key> flags accepted per vector store (also the pgvector key order).
_VECTOR_FLAG_KEYS = {"qdrant": ("path", "url"), "pgvector": ("host", "port", "user", "password", "dbname")}


def parse_flags(argv: list[str] | None = None) -> dict[str, str]:
    """Parse CLI flags from argv. Returns dict of flag values."""
    args = argv if argv is not None else sys.argv[1:]
    flags: dict[str, Any] = {k: _FLAG_DEFAULTS.get(k, "") for k in _FLAG_KEYS}
    flags["dry_run"] = False
    flag_map = {"--" + k.replace("_", "-"): k for k in _FLAG_KEYS}
    i = 0
    while i < len(args):
        if args[i] == "--dry-run":
            flags["dry_run"] = True
            i += 1
        elif args[i] in flag_map and i + 1 < len(args):
            flags[flag_map[args[i]]] = args[i + 1]
            i += 2
        else:
            i += 1
    return flags


def _model_block(flags: dict, registry: dict, prefix: str) -> tuple[str, dict, dict[str, Any]]:
    """Resolve (provider_id, provider_def, config) for an LLM/embedder section from flags."""
    pid = flags.get(prefix, "openai")
    pdef = registry[pid]
    model = flags.get(f"{prefix}_model") or pdef["default_model"]
    cfg: dict[str, Any] = {"model": model}
    url = flags.get(f"{prefix}_url") or pdef.get("default_url")
    if url and pdef.get("base_url_key"):
        cfg[pdef["base_url_key"]] = url
    return pid, pdef, cfg


def build_oss_config(flags: dict[str, str]) -> tuple[dict, dict[str, str]]:
    """Build (oss_config for mem0.json, env_writes of secrets for .env) from parsed flags."""
    llm_id, llm_def, llm_config = _model_block(flags, LLM_PROVIDERS, "oss_llm")
    if llm_id == "openai" and llm_config["model"] == "gpt-5-mini":
        llm_config["is_reasoning_model"] = True

    embedder_id, embedder_def, embedder_config = _model_block(flags, EMBEDDER_PROVIDERS, "oss_embedder")
    dims = KNOWN_DIMS.get(embedder_config["model"])
    if dims:
        embedder_config["embedding_dims"] = dims

    vector_id = flags.get("oss_vector", "qdrant")
    vector_config = dict(VECTOR_PROVIDERS[vector_id]["default_config"])
    for key in _VECTOR_FLAG_KEYS.get(vector_id, ()):
        val = flags.get(f"oss_vector_{key}")
        if val:
            vector_config[key] = int(val) if key == "port" else val
    if "url" in vector_config:
        vector_config.pop("path", None)  # a remote Qdrant URL replaces local storage

    oss_config = {
        "llm": {"provider": llm_id, "config": llm_config},
        "embedder": {"provider": embedder_id, "config": embedder_config},
        "vector_store": {"provider": vector_id, "config": vector_config},
    }

    env_writes: dict[str, str] = {}
    if llm_def.get("needs_key") and flags.get("oss_llm_key"):
        env_writes[llm_def["env_var"]] = flags["oss_llm_key"]
    if embedder_def.get("needs_key"):
        # An embedder sharing the LLM's provider reuses the LLM key when no embedder key was given.
        key = flags.get("oss_embedder_key") or (flags.get("oss_llm_key") if embedder_id == llm_id else "")
        if key:
            env_writes[embedder_def["env_var"]] = key
    return oss_config, env_writes


def _write_env(env_path: Path, env_writes: dict[str, str]) -> None:
    """Append or update env vars in .env file."""
    env_path.parent.mkdir(parents=True, exist_ok=True)
    # utf-8-sig like the canonical .env readers: locale decoding (cp1252/GBK) mangles
    # non-ASCII values, and a BOM'd first line would miss the key match and get duplicated.
    existing_lines = env_path.read_text(encoding="utf-8-sig").splitlines() if env_path.exists() else []
    updated_keys: set[str] = set()
    new_lines: list[str] = []
    for line in existing_lines:
        key = line.split("=", 1)[0].strip() if "=" in line and not line.startswith("#") else None
        if key and key in env_writes:
            updated_keys.add(key)
            line = f"{key}={env_writes[key]}"
        new_lines.append(line)
    new_lines += [f"{k}={v}" for k, v in env_writes.items() if k not in updated_keys]
    env_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")


def _save_mem0_json(hermes_home: str, data: dict) -> None:
    """Merge-write to mem0.json."""
    config_path = Path(hermes_home) / "mem0.json"
    existing = _read_mem0_json(config_path)
    existing.update(data)
    config_path.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")


def _activate_provider(config: dict) -> None:
    """Point config.yaml's memory.provider at mem0."""
    from hermes_cli.config import save_config
    config["memory"]["provider"] = "mem0"
    save_config(config)


def _persist_provider_config(hermes_home: str, config: dict, provider_config: dict, env_writes: dict[str, str]) -> None:
    """Shared platform/self-hosted tail: activate, write mem0.json (0600), then .env."""
    _activate_provider(config)
    from plugins.memory.mem0 import Mem0MemoryProvider
    Mem0MemoryProvider().save_config(provider_config, hermes_home)
    if env_writes:
        _write_env(Path(hermes_home) / ".env", env_writes)


def _setup_platform(hermes_home: str, config: dict, flags: dict[str, str]) -> None:
    """Platform mode setup — prompts for API key (secret -> .env), user/agent ids and rerank (-> mem0.json)."""
    provider_config = _read_mem0_json(Path(hermes_home) / "mem0.json")

    print("\n  Configuring mem0:\n")

    env_writes = _api_key_writes(flags, "Mem0 Platform API key", url="https://app.mem0.ai")
    for key, desc, default in (("user_id", "User identifier", "hermes-user"), ("agent_id", "Agent identifier", "hermes")):
        val = _prompt(desc, default=str(provider_config.get(key) or default))
        if val:
            provider_config[key] = val
    choices = ["true", "false"]
    current = provider_config.get("rerank", "false")
    current_idx = choices.index(str(current).lower()) if current and str(current).lower() in choices else 0
    sel = _curses_select("  Enable reranking for recall", [(c, "") for c in choices], default=current_idx)
    provider_config["rerank"] = choices[sel]

    if flags.get("dry_run"):
        _print_dry_run(str(provider_config), env_writes)
        return

    provider_config["mode"] = "platform"
    # Routing checks ``host`` before platform (_create_backend), so a stale
    # self-hosted host must be cleared. Set "" rather than pop(): save_config
    # merges into the existing mem0.json, so a popped key would survive.
    provider_config["host"] = ""
    # _load_config() also seeds ``host`` from MEM0_HOST (docs tell self-hosted
    # users to put it in .env); the file clear can't help there, so warn.
    if os.environ.get("MEM0_HOST", "").strip():
        print(
            "\n  ⚠ MEM0_HOST is set in your environment "
            f"({os.environ['MEM0_HOST']}). It overrides platform mode — "
            "remove it from ~/.hermes/.env (or unset it) or Hermes will keep "
            "routing to the self-hosted server."
        )

    _persist_provider_config(hermes_home, config, provider_config, env_writes)
    _print_saved("mem0", env_writes, "API keys saved to .env")


def _check_selfhosted_server(host: str) -> None:
    """Best-effort reachability check for a self-hosted Mem0 server (non-fatal)."""
    try:
        _http_get(host, "/docs", 5)
        print(f"  ✓ Mem0 server reachable at {host}")
    except urllib.error.HTTPError:
        # Any HTTP response (401/403/404) still means something is listening.
        print(f"  ✓ Mem0 server responding at {host}")
    except Exception:
        print(f"  ⚠ Could not reach {host} — check the URL and that the server is running.")


def _setup_selfhosted(hermes_home: str, config: dict, flags: dict[str, str]) -> None:
    """Self-hosted mode — point at an existing Mem0 server: URL -> mem0.json, key -> .env (MEM0_API_KEY)."""
    provider_config = _read_mem0_json(Path(hermes_home) / "mem0.json")

    print("\n  Configuring mem0 (self-hosted server):\n")

    host = flags.get("host") or _prompt("Mem0 server URL (e.g. http://localhost:8888)", default=provider_config.get("host") or None)
    if not host:
        print("  Error: a server URL is required for self-hosted mode.", file=sys.stderr)
        return
    host = host.rstrip("/")

    env_writes = _api_key_writes(flags, "Server API key", fresh_label="Server API key (blank if AUTH_DISABLED)")
    user_id = flags.get("user_id") or _prompt("User identifier", default=provider_config.get("user_id") or "hermes-user")
    agent_id = _prompt("Agent identifier", default=provider_config.get("agent_id") or "hermes")

    if flags.get("dry_run"):
        _print_dry_run(f"host={host}, user_id={user_id}, agent_id={agent_id}", env_writes, lambda: _check_selfhosted_server(host))
        return

    provider_config.update(mode="platform", host=host, user_id=user_id, agent_id=agent_id)  # routing: oss > host > platform
    _persist_provider_config(hermes_home, config, provider_config, env_writes)

    _check_selfhosted_server(host)
    _print_saved("mem0 (self-hosted)", env_writes, "API key saved to .env", server=host)


def _print_oss_summary(oss_config: dict, env_writes: dict, dry_run: bool = False) -> None:
    llm, emb = oss_config["llm"], oss_config["embedder"]
    w = 0 if dry_run else 9  # final summary column-aligns the labels
    print("\n  [dry-run] OSS config would be:" if dry_run else "\n  ✓ Mem0 configured (OSS mode)")
    print(f"    {'LLM:':<{w}} {llm['provider']} ({llm['config'].get('model', '')})")
    print(f"    {'Embedder:':<{w}} {emb['provider']} ({emb['config'].get('model', '')})")
    print(f"    {'Vector:':<{w}} {oss_config['vector_store']['provider']}")
    if dry_run:
        if env_writes:
            print(f"    Env vars: {', '.join(env_writes.keys())}")
        return
    if env_writes:
        print("    API keys saved to .env")
    print("    Config saved to mem0.json")
    print("    Provider set in config.yaml")
    print("\n  Start a new session to activate.\n")


def _finish_oss(
    hermes_home: str, config: dict, oss_config: dict, env_writes: dict[str, str],
    user_id: str, agent_id: str, pgvector_config: dict | None = None,
) -> None:
    """Shared OSS tail: write secrets + mem0.json, install deps, activate, check, summarize."""
    if env_writes:
        _write_env(Path(hermes_home) / ".env", env_writes)
    _save_mem0_json(hermes_home, {"mode": "oss", "user_id": user_id, "agent_id": agent_id, "oss": oss_config})
    _install_provider_deps(oss_config["llm"]["provider"], oss_config["embedder"]["provider"], oss_config["vector_store"]["provider"])
    if pgvector_config:
        _ensure_pgvector_extension(pgvector_config)
    _activate_provider(config)
    _run_connectivity_checks(oss_config)
    _print_oss_summary(oss_config, env_writes)


def _setup_oss(hermes_home: str, config: dict, flags: dict[str, str]) -> None:
    """OSS mode — non-interactive when --mode was given, otherwise curses pickers."""
    if not flags.get("_mode_from_flag"):
        _setup_oss_interactive(hermes_home, config)
        return
    oss_config, env_writes = build_oss_config(flags)
    errors = validate_oss_config(oss_config)
    if errors:
        for e in errors:
            print(f"  Error: {e}", file=sys.stderr)
        sys.exit(1)
    if flags.get("dry_run"):
        _print_oss_summary(oss_config, env_writes, dry_run=True)
        _run_connectivity_checks(oss_config)
        print("  [dry-run] No files written.\n")
        return

    _finish_oss(hermes_home, config, oss_config, env_writes, flags.get("user_id") or os.getenv("USER", "hermes-user"), "hermes")


_PGVECTOR_CONTAINER = "hermes-pgvector"
_PGVECTOR_IMAGE = "pgvector/pgvector:pg17"
_PGVECTOR_PASSWORD = "hermes"


def _docker(*args: str, timeout: int, **kwargs) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, timeout=timeout, stdin=subprocess.DEVNULL, **kwargs)


def _ensure_pgvector(host: str = "localhost", port: int = 5432) -> dict | None:
    """Ensure pgvector is reachable; offer Docker setup if not. Returns the Docker
    container's vector_config if one was started, None otherwise."""
    if _check_pgvector(host, port)[0]:
        print(f"  ✓ PostgreSQL reachable at {host}:{port}")
        return None

    print(f"  PostgreSQL not reachable at {host}:{port}")
    if not shutil.which("docker"):
        print("  Docker not found. Install Docker to auto-start pgvector,\n  or run PostgreSQL with pgvector manually.")
        return None

    # Restart our own container if it exists but is stopped.
    try:
        result = _docker("inspect", _PGVECTOR_CONTAINER, "--format", "{{.State.Status}}",
                         timeout=10, text=True, encoding='utf-8', errors='replace')
        if result.returncode == 0 and "exited" in result.stdout:
            print(f"  Found stopped container '{_PGVECTOR_CONTAINER}', restarting...")
            _docker("start", _PGVECTOR_CONTAINER, timeout=15)
            _wait_for_port(host, port, timeout=15)
            if _check_pgvector(host, port)[0]:
                print("  ✓ PostgreSQL container restarted")
                return None
    except Exception:
        pass

    if input("  Start pgvector via Docker? [Y/n]: ").strip().lower() in ("", "y", "yes"):
        return _start_pgvector_docker(host, port)
    print("  Skipping Docker setup. Make sure PostgreSQL with pgvector is running.")
    return None


def _start_pgvector_docker(host: str, port: int) -> dict | None:
    """Pull and start pgvector Docker container."""
    try:
        print(f"  Pulling {_PGVECTOR_IMAGE}...")
        _docker("pull", _PGVECTOR_IMAGE, timeout=120)
        _docker("rm", "-f", _PGVECTOR_CONTAINER, timeout=10)  # remove existing container if present
        print(f"  Starting container '{_PGVECTOR_CONTAINER}' on port {port}...")
        _docker(
            "run", "-d", "--name", _PGVECTOR_CONTAINER,
            "-e", f"POSTGRES_PASSWORD={_PGVECTOR_PASSWORD}",
            "-p", f"{port}:5432", _PGVECTOR_IMAGE,
            timeout=30, check=True,
        )
        _wait_for_port(host, port, timeout=20)
        if _check_pgvector(host, port)[0]:
            print(f"  ✓ pgvector running on {host}:{port}")
        else:
            print("  Warning: Container started but PostgreSQL not yet accepting connections.\n"
                  "  It may need a few more seconds. Config will be saved; retry later.")
        return {"host": host, "port": port, "user": "postgres", "password": _PGVECTOR_PASSWORD, "dbname": "postgres"}
    except subprocess.CalledProcessError as e:
        print(f"  Failed to start Docker container: {e}")
    except Exception as e:
        print(f"  Docker error: {e}")
    return None


def _ensure_ollama(models: list[str]) -> bool:
    """Ensure Ollama is running and required models are pulled. Returns False when
    the user must handle it manually."""
    url = "http://localhost:11434"
    ollama_bin = shutil.which("ollama")
    ok = _check_ollama(url)[0]
    if not ok:
        if not ollama_bin:
            print("  Ollama not found. Install it:\n    curl -fsSL https://ollama.com/install.sh | sh\n"
                  "  Or on macOS: brew install ollama")
            return False
        print("  Ollama installed but not running. Starting...")
        try:
            subprocess.Popen(
                [ollama_bin, "serve"], stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            _wait_for_port("localhost", 11434, timeout=10)
            ok = _check_ollama(url)[0]
            if ok:
                print("  ✓ Ollama started")
        except Exception as e:
            print(f"  Could not start Ollama: {e}")
    if not ok:
        print("  Warning: Ollama not reachable. Models cannot be pulled.")
        return False

    for model in models:
        if _ollama_has_model(url, model):
            print(f"  ✓ Model '{model}' available")
            continue
        print(f"  Pulling '{model}'... (this may take a few minutes)")
        try:
            subprocess.run([ollama_bin or "ollama", "pull", model], timeout=600, stdin=subprocess.DEVNULL)
            print(f"  ✓ Model '{model}' pulled")
        except Exception as e:
            print(f"  Warning: Could not pull '{model}': {e}\n  Run manually: ollama pull {model}")
    return True


def _ollama_has_model(url: str, model: str) -> bool:
    """Check if Ollama already has a model pulled."""
    try:
        names = [m.get("name", "") for m in json.loads(_http_get(url, "/api/tags", 5).read()).get("models", [])]
        base_model = model.split(":")[0]
        return any(model in n or base_model in n for n in names)
    except Exception:
        return False


def _ensure_pgvector_extension(pg_config: dict) -> None:
    """Create the pgvector extension if it doesn't exist."""
    try:
        import psycopg2
    except ImportError:
        return
    conn_params = {k: pg_config.get(k, d) for k, d in (("host", "localhost"), ("port", 5432), ("user", "postgres"), ("dbname", "postgres"))}
    if pg_config.get("password"):
        conn_params["password"] = pg_config["password"]
    try:
        conn = psycopg2.connect(**conn_params)
        conn.autocommit = True
        conn.cursor().execute("CREATE EXTENSION IF NOT EXISTS vector")
        conn.close()
        print("  ✓ pgvector extension enabled")
    except Exception as e:
        print(f"  Warning: Could not enable pgvector extension: {e}")


def _wait_for_port(host: str, port: int, timeout: int = 15) -> None:
    """Wait until a TCP port is accepting connections."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            socket.create_connection((host, port), timeout=1).close()
            return
        except OSError:
            time.sleep(0.5)


def _provider_description(v: dict) -> str:
    """Description for LLM/embedder picker: model + URL if applicable."""
    model, url = v.get("default_model", ""), v.get("default_url")
    return f"{model} ({url})" if url else model


def _vector_description(pid: str, v: dict) -> str:
    cfg = v.get("default_config", {})
    if pid == "qdrant":
        return cfg.get("path", "local storage")
    return f"{cfg.get('host', 'localhost')}:{cfg.get('port', 5432)}" if pid == "pgvector" else pid


def _configure_model_provider(
    kind: str, registry: dict, hermes_home: str, env_writes: dict[str, str], llm: tuple[str, dict] | None = None,
) -> tuple[str, dict, str, str | None]:
    """Pick an LLM/embedder provider, collect its key, and (for Ollama) model + URL.
    Returns (id, definition, model, url). For the embedder (``llm`` given), a provider
    shared with the LLM reuses the LLM key instead of prompting again."""
    items = [(v["label"], _provider_description(v)) for v in registry.values()]
    pid = list(registry)[_curses_select(f"{kind} Provider", items, 0)]
    pdef = registry[pid]
    model, url = pdef["default_model"], pdef.get("default_url")
    if pdef["needs_key"]:
        if llm is None or pid != llm[0]:
            label = pdef["label"] if llm is None else f"{pdef['label']} embedder"
            key = _prompt_api_key(label, pdef["env_var"], hermes_home)
            if key:
                env_writes[pdef["env_var"]] = key
        elif llm[1].get("env_var") in env_writes:
            env_writes[pdef["env_var"]] = env_writes[llm[1]["env_var"]]
    if pid == "ollama":
        model = _input(f"{kind} model", pdef["default_model"])
        url = _input("Ollama URL", pdef["default_url"])
    return pid, pdef, model, url


def _setup_oss_interactive(hermes_home: str, config: dict) -> None:
    """Interactive OSS setup using curses pickers."""
    env_writes: dict[str, str] = {}
    llm_id, llm_def, llm_model, llm_url = _configure_model_provider("LLM", LLM_PROVIDERS, hermes_home, env_writes)
    embedder_id, _, embedder_model, embedder_url = _configure_model_provider(
        "Embedder", EMBEDDER_PROVIDERS, hermes_home, env_writes, llm=(llm_id, llm_def),
    )

    vector_items = [(v["label"], _vector_description(pid, v)) for pid, v in VECTOR_PROVIDERS.items()]
    vector_id = list(VECTOR_PROVIDERS)[_curses_select("Vector Store", vector_items, 0)]

    # Auto-setup: ensure Ollama is running and models are pulled
    ollama_models = [m for pid, m in ((llm_id, llm_model), (embedder_id, embedder_model)) if pid == "ollama"]
    if ollama_models:
        _ensure_ollama(ollama_models)

    # Auto-setup: ensure pgvector is reachable (offer Docker if not)
    pgvector_config = None
    if vector_id == "pgvector":
        pgvector_config = _ensure_pgvector()
        if not pgvector_config:
            # Native PostgreSQL — prompt for connection details (user first, matching the historical order)
            pg = {k: _input(f"PostgreSQL {label}", d) for k, label, d in (
                ("user", "user", os.getenv("USER", "postgres")), ("host", "host", "localhost"),
                ("port", "port", "5432"), ("dbname", "database", "postgres"))}
            pg_password = getpass.getpass("  PostgreSQL password (blank if none): ").strip()
            pgvector_config = {"host": pg["host"], "port": int(pg["port"]), "user": pg["user"], "dbname": pg["dbname"]}
            if pg_password:
                pgvector_config["password"] = pg_password

    user_id = _input("User ID", os.getenv("USER", "hermes-user"))
    agent_id = _input("Agent ID", "hermes")

    flags = {
        "oss_llm": llm_id, "oss_llm_model": llm_model, "oss_llm_url": llm_url or "",
        "oss_llm_key": env_writes.get(llm_def["env_var"], "") if llm_def.get("env_var") else "",
        "oss_embedder": embedder_id, "oss_embedder_model": embedder_model, "oss_embedder_url": embedder_url or "",
        "oss_vector": vector_id, "user_id": user_id,
    }
    if pgvector_config:
        for key in ("host", "port", "user", "password", "dbname"):
            if pgvector_config.get(key):
                flags[f"oss_vector_{key}"] = str(pgvector_config[key])

    oss_config, _ = build_oss_config(flags)
    _finish_oss(hermes_home, config, oss_config, env_writes, user_id, agent_id, pgvector_config)


def _install_provider_deps(llm_id: str, embedder_id: str, vector_id: str) -> None:
    """Install all optional pip deps for selected providers."""
    deps = {
        registry[pid]["pip_dep"]
        for (_, registry), pid in zip(SECTION_REGISTRIES, (llm_id, embedder_id, vector_id))
        if registry.get(pid, {}).get("pip_dep")
    }
    for dep in sorted(deps):
        try:
            print(f"  Installing {dep}...")
            # Environment-aware install: sealed hosted venvs redirect to the
            # durable data-volume target instead of /opt/hermes.
            from tools.lazy_deps import install_specs
            outcome = install_specs([dep], timeout=60)
            if outcome.ok:
                print(f"  ✓ Installed {dep}")
            elif outcome.blocked:
                print(f"  Warning: cannot install {dep}: {outcome.reason}")
            else:
                print(f"  Warning: Could not install {dep}. Install manually: uv pip install {dep}")
        except Exception:
            print(f"  Warning: Could not install {dep}. Install manually: uv pip install {dep}")
    if deps:
        import importlib
        importlib.invalidate_caches()


def _check_qdrant_path(path: str) -> tuple[bool, str]:
    """Check that qdrant local storage parent dir is writable."""
    parent = Path(path).expanduser().parent
    try:
        parent.mkdir(parents=True, exist_ok=True)
        return True, f"Directory writable: {parent}"
    except OSError as e:
        return False, f"Cannot write to {parent}: {e}"


def _check_ollama(url: str) -> tuple[bool, str]:
    """Check Ollama is reachable via /api/tags."""
    try:
        _http_get(url, "/api/tags", 3)
        return True, "Ollama reachable"
    except Exception as e:
        return False, f"Ollama not reachable at {url}: {e}"


def _check_pgvector(host: str, port: int) -> tuple[bool, str]:
    """Check PGVector via TCP socket."""
    try:
        socket.create_connection((host, port), timeout=3).close()
        return True, f"PGVector reachable at {host}:{port}"
    except Exception as e:
        return False, f"PGVector not reachable at {host}:{port}: {e}"


def _warn_unless(check: tuple[bool, str]) -> None:
    ok, msg = check
    if not ok:
        print(f"  Warning: {msg}")


def _run_connectivity_checks(oss_config: dict) -> None:
    """Run connectivity checks and print warnings."""
    vs = oss_config.get("vector_store", {})
    cfg = vs.get("config", {})
    if vs.get("provider") == "qdrant":
        path, url = cfg.get("path"), cfg.get("url")
        if path:
            _warn_unless(_check_qdrant_path(path))
        elif url:
            try:
                _http_get(url, "/healthz", 3)
            except Exception as e:
                print(f"  Warning: Qdrant not reachable at {url}: {e}")
    elif vs.get("provider") == "pgvector":
        _warn_unless(_check_pgvector(cfg.get("host", "localhost"), cfg.get("port", 5432)))

    llm = oss_config.get("llm", {})
    if llm.get("provider") == "ollama":
        _warn_unless(_check_ollama(llm.get("config", {}).get("ollama_base_url", "http://localhost:11434")))


def _check_min_dep_version() -> None:
    """Ensure mem0ai meets the minimum version from plugin.yaml."""
    try:
        import mem0
        installed_ver = getattr(mem0, "__version__", None)
        if installed_ver and tuple(int(x) for x in installed_ver.split(".")[:3]) < (2, 0, 7):
            print(f"\n  ⚠ mem0ai {installed_ver} installed but >=2.0.7 required.\n"
                  f"  Run: uv pip install --python {sys.executable} 'mem0ai>=2.0.7'")
    except Exception:
        pass


_MODE_HANDLERS = {
    "oss": _setup_oss,
    "selfhosted": _setup_selfhosted,
    "self-hosted": _setup_selfhosted,
    "platform": _setup_platform,
}
# Interactive picker order: Platform, Self-hosted server, Open Source.
_MODE_ITEMS = [
    ("Platform", "Mem0 Cloud API (lightweight, just needs an API key)"),
    ("Self-hosted server", "Connect to an existing self-hosted Mem0 server (Docker/FastAPI)"),
    ("Open Source", "Run Mem0 locally (self-hosted LLM + vector store)"),
]
_MODE_PICKER = (_setup_platform, _setup_selfhosted, _setup_oss)


def post_setup(hermes_home: str, config: dict) -> None:
    """Entry point called by hermes memory setup framework. Routes on --mode
    (platform / selfhosted / oss); with no flag shows a picker. OSS is
    non-interactive only when the mode came from the flag."""
    _check_min_dep_version()
    flags = parse_flags(sys.argv[1:])
    handler = _MODE_HANDLERS.get(flags["mode"])
    flags["_mode_from_flag"] = handler is not None
    if handler is None:
        handler = _MODE_PICKER[_curses_select("  Select mode", _MODE_ITEMS, 0)]
    handler(hermes_home, config, flags)
