"""Server registry — per-language LSP server definitions.

Each :class:`ServerDef` matches files (by extension or basename for
extensionless files like ``Dockerfile``), resolves a project root, and
assembles the spawn command.  Auto-installation lives in
:mod:`agent.lsp.install`; nothing here probes binaries until a file in
that language is actually edited.
"""
from __future__ import annotations

import logging
import os
import shutil
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from agent.lsp.workspace import nearest_root

logger = logging.getLogger("agent.lsp.servers")

# LSP languageId for ``textDocument/didOpen``.  A few servers
# (typescript-language-server, vue-language-server) refuse wrong IDs.
LANGUAGE_BY_EXT: Dict[str, str] = {
    ".py": "python",
    ".pyi": "python",
    ".ts": "typescript",
    ".tsx": "typescriptreact",
    ".js": "javascript",
    ".jsx": "javascriptreact",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".vue": "vue",
    ".svelte": "svelte",
    ".astro": "astro",
    ".go": "go",
    ".rs": "rust",
    ".rb": "ruby",
    ".rake": "ruby",
    ".gemspec": "ruby",
    ".ru": "ruby",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hh": "cpp",
    ".hpp": "cpp",
    ".hxx": "cpp",
    ".cs": "csharp",
    ".csx": "csharp",
    ".fs": "fsharp",
    ".fsi": "fsharp",
    ".fsx": "fsharp",
    ".swift": "swift",
    ".java": "java",
    ".kt": "kotlin",
    ".kts": "kotlin",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".jsonc": "jsonc",
    ".lua": "lua",
    ".php": "php",
    ".prisma": "prisma",
    ".dart": "dart",
    ".ml": "ocaml",
    ".mli": "ocaml",
    ".sh": "shellscript",
    ".bash": "shellscript",
    ".zsh": "shellscript",
    ".tf": "terraform",
    ".tfvars": "terraform",
    ".tex": "latex",
    ".bib": "bibtex",
    ".gleam": "gleam",
    ".clj": "clojure",
    ".cljs": "clojurescript",
    ".cljc": "clojure",
    ".edn": "clojure",
    ".nix": "nix",
    ".typ": "typst",
    ".typc": "typst",
    ".hs": "haskell",
    ".lhs": "haskell",
    ".jl": "julia",
    ".ex": "elixir",
    ".exs": "elixir",
    ".zig": "zig",
    ".zon": "zig",
    ".dockerfile": "dockerfile",
    ".ps1": "powershell",
    ".psm1": "powershell",
    ".psd1": "powershell",
}


@dataclass
class SpawnSpec:
    """Result of resolving a server for a file (``None`` means skip)."""

    command: List[str]
    workspace_root: str
    cwd: str
    env: Dict[str, str] = field(default_factory=dict)
    initialization_options: Dict[str, Any] = field(default_factory=dict)
    seed_diagnostics_on_first_push: bool = False


@dataclass
class ServerDef:
    """Definition of one language server.

    ``resolve_root(file_path, workspace_root)`` returns the per-server
    project root or ``None`` to skip; ``build_spawn(root, ctx)`` returns
    a :class:`SpawnSpec` or ``None`` when the binary can't be found.
    """

    server_id: str
    extensions: Tuple[str, ...]
    resolve_root: Callable[[str, str], Optional[str]]
    build_spawn: Callable[[str, "ServerContext"], Optional[SpawnSpec]]
    seed_first_push: bool = False
    description: str = ""

    def matches(self, file_path: str) -> bool:
        """Return True iff this server handles ``file_path``."""
        return _file_ext_or_basename(file_path) in self.extensions


@dataclass
class ServerContext:
    """User policy passed into :meth:`ServerDef.build_spawn` (install strategy, overrides)."""

    workspace_root: str
    install_strategy: str = "auto"  # "auto" | "manual" | "off"
    binary_overrides: Dict[str, List[str]] = field(default_factory=dict)
    env_overrides: Dict[str, Dict[str, str]] = field(default_factory=dict)
    init_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _file_ext_or_basename(path: str) -> str:
    """Lower-cased extension, or the full basename for extensionless files (``Dockerfile``)."""
    base = os.path.basename(path)
    _root, ext = os.path.splitext(base)
    return ext.lower() if ext else base


def _which(*names: str) -> Optional[str]:
    """Return the full path of the first command found on PATH."""
    for n in names:
        path = shutil.which(n)
        if path:
            return path
    return None


def _root_or_workspace(file_path: str, workspace: str, markers: Sequence[str], excludes: Sequence[str] = ()) -> Optional[str]:
    """``nearest_root`` with workspace fallback; ``None`` iff an exclude marker hit."""
    ceiling = os.path.dirname(workspace) if workspace else None
    found = nearest_root(file_path, markers, excludes=excludes, ceiling=ceiling)
    if found is None and excludes:
        # None is ambiguous with excludes configured: re-check without them —
        # a hit now means the exclude fired (gated off), else fall back.
        if nearest_root(file_path, markers, ceiling=ceiling) is not None:
            return None
        return workspace
    return found or workspace


def _resolve_override(ctx: ServerContext, server_id: str) -> Optional[str]:
    """User can pin a binary path in config."""
    override = ctx.binary_overrides.get(server_id)
    if override and override[0] and os.path.exists(override[0]):
        return override[0]
    return None


def _find_binary(ctx: ServerContext, server_id: str, which: Sequence[str], install_pkg: Optional[str]) -> Optional[str]:
    """Override → PATH → (optional) auto-install; ``None`` when nothing resolves."""
    bin_path = _resolve_override(ctx, server_id) or _which(*which)
    if bin_path is None and install_pkg is not None:
        from agent.lsp.install import try_install
        bin_path = try_install(install_pkg, ctx.install_strategy)
    return bin_path


def _make_spec(root: str, ctx: ServerContext, server_id: str, command: List[str],
               base_init: Optional[Dict[str, Any]] = None, seed: bool = False) -> SpawnSpec:
    if base_init is None:
        init = ctx.init_overrides.get(server_id, {})
    else:
        init = dict(base_init)
        init.update(ctx.init_overrides.get(server_id, {}))
    return SpawnSpec(
        command=command,
        workspace_root=root,
        cwd=root,
        env=ctx.env_overrides.get(server_id, {}),
        initialization_options=init,
        seed_diagnostics_on_first_push=seed,
    )


def _simple_spawn(server_id: str, which: Sequence[str], args: Sequence[str] = (),
                  install_pkg: Optional[str] = None, base_init: Optional[Dict[str, Any]] = None,
                  seed: bool = False) -> Callable[[str, ServerContext], Optional[SpawnSpec]]:
    """Build a spawn function for the common single-binary server shape."""
    def build(root: str, ctx: ServerContext) -> Optional[SpawnSpec]:
        bin_path = _find_binary(ctx, server_id, which, install_pkg)
        if bin_path is None:
            return None
        return _make_spec(root, ctx, server_id, [bin_path, *args], base_init, seed)
    return build


def _markers_root(markers: Optional[Sequence[str]], excludes: Sequence[str] = ()) -> Callable[[str, str], Optional[str]]:
    """Root resolver over marker files; ``None`` markers means "always the workspace root"."""
    if markers is None:
        return lambda fp, ws: ws
    return lambda fp, ws: _root_or_workspace(fp, ws, markers, excludes=excludes)


# ---------------------------------------------------------------------------
# bespoke spawn builders
# ---------------------------------------------------------------------------


def _spawn_pyright(root: str, ctx: ServerContext) -> Optional[SpawnSpec]:
    bin_path = _find_binary(ctx, "pyright", ("pyright-langserver", "pyright"), "pyright")
    if bin_path is None:
        return None
    # If we got the cli ``pyright``, the langserver is its sibling.
    if os.path.basename(bin_path) in {"pyright", "pyright.exe"}:
        sibling = os.path.join(os.path.dirname(bin_path), "pyright-langserver")
        if os.path.exists(sibling):
            bin_path = sibling
    init: Dict[str, Any] = {}
    # Point pyright at the project venv; its default "python on PATH" rarely is.
    py = _detect_python(root)
    if py:
        init["python"] = {"pythonPath": py}
    return _make_spec(root, ctx, "pyright", [bin_path, "--stdio"], init)


def _detect_python(root: str) -> Optional[str]:
    candidates = []
    if os.environ.get("VIRTUAL_ENV"):
        candidates.append(os.environ["VIRTUAL_ENV"])
    candidates.extend([os.path.join(root, ".venv"), os.path.join(root, "venv")])
    for v in candidates:
        for sub in ("bin/python", "bin/python3", "Scripts/python.exe"):
            p = os.path.join(v, sub)
            if os.path.exists(p):
                return p
    return None


_BASH_SHELLCHECK_WARNED = False


def _spawn_bash_ls(root: str, ctx: ServerContext) -> Optional[SpawnSpec]:
    bin_path = _find_binary(ctx, "bash-language-server", ("bash-language-server",), "bash-language-server")
    if bin_path is None:
        return None
    # bash-language-server delegates diagnostics to shellcheck; without it the
    # server runs but never reports anything.  Warn once so the gap is visible.
    global _BASH_SHELLCHECK_WARNED
    if not _BASH_SHELLCHECK_WARNED and _which("shellcheck") is None:
        _BASH_SHELLCHECK_WARNED = True
        logger.warning(
            "bash-language-server: shellcheck not found on PATH — "
            "diagnostics will be empty until shellcheck is installed "
            "(apt: shellcheck, brew: shellcheck, scoop: shellcheck)."
        )
    return _make_spec(root, ctx, "bash-language-server", [bin_path, "start"])


_PSES_BUNDLE_WARNED = False


def _find_pses_bundle(ctx: ServerContext) -> Optional[str]:
    """Locate the PowerShellEditorServices bundle dir (release zip, manual install).

    Resolution order: ``lsp.servers.powershell.command[0]`` when a directory,
    ``init_overrides["powershell"]["bundlePath"]``, ``PSES_BUNDLE_PATH`` env,
    then ``<HERMES_HOME>/lsp/PowerShellEditorServices``.
    """
    candidates: List[str] = []
    override = ctx.binary_overrides.get("powershell")
    if override and override[0]:
        candidates.append(override[0])
    init = ctx.init_overrides.get("powershell", {})
    if isinstance(init, dict) and init.get("bundlePath"):
        candidates.append(str(init["bundlePath"]))
    env_path = os.environ.get("PSES_BUNDLE_PATH")
    if env_path:
        candidates.append(env_path)
    from hermes_constants import get_hermes_home

    candidates.append(os.path.join(str(get_hermes_home()), "lsp", "PowerShellEditorServices"))

    for cand in candidates:
        if not cand:
            continue
        # Accept either the bundle root or the inner module dir.
        if os.path.isfile(os.path.join(cand, "PowerShellEditorServices", "Start-EditorServices.ps1")):
            return cand
        if os.path.isfile(os.path.join(cand, "Start-EditorServices.ps1")):
            return os.path.dirname(cand)
    return None


def _spawn_powershell_es(root: str, ctx: ServerContext) -> Optional[SpawnSpec]:
    """Spawn PowerShellEditorServices: needs a ``pwsh``/``powershell`` host plus the module bundle."""
    pwsh = _which("pwsh", "powershell")
    if pwsh is None:
        return None
    bundle = _find_pses_bundle(ctx)
    if bundle is None:
        global _PSES_BUNDLE_WARNED
        if not _PSES_BUNDLE_WARNED:
            _PSES_BUNDLE_WARNED = True
            logger.warning(
                "powershell: pwsh found but the PowerShellEditorServices "
                "bundle is missing. Download the release zip from "
                "https://github.com/PowerShell/PowerShellEditorServices/releases, "
                "extract it, and either set lsp.servers.powershell.command "
                "to the bundle path or unzip it to "
                "<HERMES_HOME>/lsp/PowerShellEditorServices."
            )
        return None
    start_script = os.path.join(bundle, "PowerShellEditorServices", "Start-EditorServices.ps1")
    # PSES writes connection info to the session details file on startup.
    session_path = os.path.join(hermes_lsp_session_dir(), f"pses-session-{os.getpid()}.json")
    log_path = os.path.join(hermes_lsp_session_dir(), "pses.log")
    inner = (
        f"& '{start_script}' "
        f"-BundledModulesPath '{bundle}' "
        f"-LogPath '{log_path}' "
        f"-SessionDetailsPath '{session_path}' "
        f"-FeatureFlags @() -AdditionalModules @() "
        f"-HostName Hermes -HostProfileId hermes -HostVersion 1.0.0 "
        f"-Stdio -LogLevel Normal"
    )
    return SpawnSpec(
        command=[pwsh, "-NoLogo", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", inner],
        workspace_root=root,
        cwd=root,
        env=ctx.env_overrides.get("powershell", {}),
        initialization_options={
            k: v
            for k, v in ctx.init_overrides.get("powershell", {}).items()
            if k != "bundlePath"
        },
    )


def hermes_lsp_session_dir() -> str:
    """Return (and create) the dir for PSES session/log scratch files."""
    from hermes_constants import get_hermes_home

    d = os.path.join(str(get_hermes_home()), "lsp", "pses")
    os.makedirs(d, exist_ok=True)
    return d


# ---------------------------------------------------------------------------
# the registry
# ---------------------------------------------------------------------------

_JS_MARKERS = ["package-lock.json", "bun.lockb", "bun.lock", "pnpm-lock.yaml", "yarn.lock", "package.json", "tsconfig.json"]
_DENO_EXCLUDES = ["deno.json", "deno.jsonc"]
_root_typescript = _markers_root(_JS_MARKERS, _DENO_EXCLUDES)


def _server(server_id: str, extensions: Tuple[str, ...], description: str, *,
            markers: Optional[Sequence[str]] = None, excludes: Sequence[str] = (),
            resolve_root: Optional[Callable[[str, str], Optional[str]]] = None,
            build_spawn: Optional[Callable[[str, ServerContext], Optional[SpawnSpec]]] = None,
            which: Sequence[str] = (), args: Sequence[str] = (), install_pkg: Optional[str] = None,
            base_init: Optional[Dict[str, Any]] = None, seed: bool = False) -> ServerDef:
    return ServerDef(
        server_id=server_id,
        extensions=extensions,
        resolve_root=resolve_root or _markers_root(markers, excludes),
        build_spawn=build_spawn or _simple_spawn(server_id, which or (server_id,), args, install_pkg, base_init, seed),
        seed_first_push=seed,
        description=description,
    )


SERVERS: List[ServerDef] = [
    _server("pyright", (".py", ".pyi"), "Python — Microsoft pyright",
            markers=["pyproject.toml", "setup.py", "setup.cfg", "requirements.txt", "Pipfile", "pyrightconfig.json"],
            build_spawn=_spawn_pyright),
    _server("typescript", (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".mts", ".cts"),
            "JavaScript/TypeScript — typescript-language-server", resolve_root=_root_typescript,
            which=("typescript-language-server",), args=("--stdio",), install_pkg="typescript-language-server", seed=True),
    _server("vue-language-server", (".vue",), "Vue.js — @vue/language-server", resolve_root=_root_typescript,
            args=("--stdio",), install_pkg="@vue/language-server"),
    _server("svelte-language-server", (".svelte",), "Svelte — svelte-language-server", resolve_root=_root_typescript,
            which=("svelteserver", "svelte-language-server"), args=("--stdio",), install_pkg="svelte-language-server"),
    _server("astro-language-server", (".astro",), "Astro — @astrojs/language-server", resolve_root=_root_typescript,
            which=("astro-ls", "astro-language-server"), args=("--stdio",), install_pkg="@astrojs/language-server"),
    _server("gopls", (".go",), "Go — gopls", markers=["go.work", "go.mod", "go.sum"], install_pkg="gopls"),
    _server("rust-analyzer", (".rs",), "Rust — rust-analyzer", markers=["Cargo.toml", "Cargo.lock"], install_pkg="rust-analyzer"),
    _server("clangd", (".c", ".cpp", ".cc", ".cxx", ".h", ".hh", ".hpp", ".hxx"), "C/C++ — clangd",
            markers=["compile_commands.json", "compile_flags.txt", ".clangd"],
            args=("--background-index", "--clang-tidy"), install_pkg="clangd"),
    _server("bash-language-server", (".sh", ".bash", ".zsh", ".ksh"), "Bash — bash-language-server", build_spawn=_spawn_bash_ls),
    _server("yaml-language-server", (".yaml", ".yml"), "YAML — yaml-language-server",
            args=("--stdio",), install_pkg="yaml-language-server"),
    _server("lua-language-server", (".lua",), "Lua — lua-language-server",
            markers=[".luarc.json", ".luarc.jsonc", ".luacheckrc", ".stylua.toml", "stylua.toml", "selene.toml", "selene.yml"],
            install_pkg="lua-language-server"),
    _server("intelephense", (".php",), "PHP — intelephense", markers=["composer.json", "composer.lock", ".php-version"],
            args=("--stdio",), install_pkg="intelephense", base_init={"telemetry": {"enabled": False}}),
    _server("ocaml-lsp", (".ml", ".mli"), "OCaml — ocaml-lsp", markers=["dune-project", "dune-workspace", ".merlin", "opam"],
            which=("ocamllsp",)),
    _server("dockerfile-ls", (".dockerfile", "Dockerfile"), "Dockerfile — dockerfile-language-server-nodejs",
            which=("docker-langserver",), args=("--stdio",), install_pkg="dockerfile-language-server-nodejs"),
    # terraform-ls is heavy to auto-install; require the user to provide it.
    _server("terraform-ls", (".tf", ".tfvars"), "Terraform — terraform-ls", markers=[".terraform.lock.hcl", "terraform.tfstate"],
            args=("serve",), base_init={"experimentalFeatures": {"prefillRequiredFields": True, "validateOnSave": True}}),
    _server("dart", (".dart",), "Dart — built-in language server", markers=["pubspec.yaml", "analysis_options.yaml"],
            args=("language-server", "--lsp")),
    _server("haskell-language-server", (".hs", ".lhs"), "Haskell — haskell-language-server",
            markers=["stack.yaml", "cabal.project", "hie.yaml"],
            which=("haskell-language-server-wrapper", "haskell-language-server"), args=("--lsp",)),
    _server("julia", (".jl",), "Julia — LanguageServer.jl", markers=["Project.toml", "Manifest.toml"],
            args=("--startup-file=no", "--history-file=no", "-e", "using LanguageServer; runserver()")),
    _server("clojure-lsp", (".clj", ".cljs", ".cljc", ".edn"), "Clojure — clojure-lsp",
            markers=["deps.edn", "project.clj", "shadow-cljs.edn", "bb.edn", "build.boot"], args=("listen",)),
    _server("nixd", (".nix",), "Nix — nixd", resolve_root=lambda fp, ws: nearest_root(fp, ["flake.nix"]) or ws),
    _server("zls", (".zig", ".zon"), "Zig — zls", markers=["build.zig"]),
    _server("gleam", (".gleam",), "Gleam — built-in language server", markers=["gleam.toml"], args=("lsp",)),
    _server("elixir-ls", (".ex", ".exs"), "Elixir — elixir-ls", markers=["mix.exs", "mix.lock"],
            which=("elixir-ls", "language_server.sh")),
    _server("prisma", (".prisma",), "Prisma — built-in language server", markers=["schema.prisma", "prisma/schema.prisma"],
            args=("language-server",)),
    _server("kotlin-language-server", (".kt", ".kts"), "Kotlin — kotlin-language-server",
            markers=["settings.gradle", "settings.gradle.kts", "build.gradle", "build.gradle.kts", "pom.xml"]),
    # jdtls has a complex install flow; we look for the wrapper script a manual install produces.
    _server("jdtls", (".java",), "Java — Eclipse JDT Language Server",
            markers=["pom.xml", "build.gradle", "build.gradle.kts", ".project", ".classpath", "settings.gradle"]),
    # No universal PowerShell root marker; nearest_root is exact-name only (no globs).
    _server("powershell", (".ps1", ".psm1", ".psd1"), "PowerShell — PowerShellEditorServices (manual bundle)",
            markers=["PSScriptAnalyzerSettings.psd1"], build_spawn=_spawn_powershell_es),
]


def find_server_for_file(file_path: str) -> Optional[ServerDef]:
    """Return the registry entry that handles ``file_path``, or None."""
    for srv in SERVERS:
        if srv.matches(file_path):
            return srv
    return None


def language_id_for(path: str) -> str:
    """Return the LSP languageId to send in didOpen for ``path``."""
    return LANGUAGE_BY_EXT.get(_file_ext_or_basename(path), "plaintext")


__all__ = [
    "ServerDef",
    "ServerContext",
    "SpawnSpec",
    "SERVERS",
    "find_server_for_file",
    "language_id_for",
    "LANGUAGE_BY_EXT",
]
