"""Welcome banner, ASCII art, skills summary, and update check for the CLI."""
import json
import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from urllib.parse import urlparse
from hermes_constants import get_hermes_home
from typing import TYPE_CHECKING, Any, Dict, List, Optional

# rich and prompt_toolkit are imported lazily (inside the functions that use
# them) rather than at module level.  Importing this module is on the TUI
# gateway's critical startup path purely to reach the lightweight update-check
# helpers (``prefetch_update_check``); pulling rich.console + prompt_toolkit
# eagerly added ~50ms of wasted imports before ``gateway.ready`` could fire.
# Keep the type-only reference available to checkers without the runtime cost.
if TYPE_CHECKING:
    from rich.console import Console

logger = logging.getLogger(__name__)


# =========================================================================
# ANSI building blocks for conversation display
# =========================================================================

_GOLD = "\033[1;38;2;255;215;0m"  # True-color #FFD700 bold
_BOLD = "\033[1m"
_DIM = "\033[2m"
_RST = "\033[0m"


def cprint(text: str):
    """Print ANSI-colored text through prompt_toolkit's renderer."""
    from prompt_toolkit import print_formatted_text as _pt_print
    from prompt_toolkit.formatted_text import ANSI as _PT_ANSI
    # prompt_toolkit needs a real console. On Windows, a redirected or absent stdout (pythonw.exe,
    # CI, `hermes ... > file`) raises NoConsoleScreenBufferError from its Win32Output — display
    # helpers must never crash the caller over that, so degrade to plain print.
    if _quiet(lambda: _pt_print(_PT_ANSI(text)) or True) is None:
        print(text)


# =========================================================================
# Skin-aware color helpers
# =========================================================================

def _quiet(fn, default=None):
    """``fn()``, or ``default`` on any exception — for best-effort display inputs."""
    try:
        return fn()
    except Exception:
        return default


def _active_skin():
    """The active skin object, or None when the skin engine is unavailable."""
    from hermes_cli.skin_engine import get_active_skin
    return get_active_skin()


def _skin_color(key: str, fallback: str) -> str:
    """Get a color from the active skin, or return fallback."""
    return _quiet(lambda: _active_skin().get_color(key, fallback), fallback)
# =========================================================================
# ASCII Art & Branding
# =========================================================================

from hermes_cli import __version__ as VERSION, __release_date__ as RELEASE_DATE

HERMES_AGENT_LOGO = """[bold #FFD700]██╗  ██╗███████╗██████╗ ███╗   ███╗███████╗███████╗       █████╗  ██████╗ ███████╗███╗   ██╗████████╗[/]
[bold #FFD700]██║  ██║██╔════╝██╔══██╗████╗ ████║██╔════╝██╔════╝      ██╔══██╗██╔════╝ ██╔════╝████╗  ██║╚══██╔══╝[/]
[#FFBF00]███████║█████╗  ██████╔╝██╔████╔██║█████╗  ███████╗█████╗███████║██║  ███╗█████╗  ██╔██╗ ██║   ██║[/]
[#FFBF00]██╔══██║██╔══╝  ██╔══██╗██║╚██╔╝██║██╔══╝  ╚════██║╚════╝██╔══██║██║   ██║██╔══╝  ██║╚██╗██║   ██║[/]
[#CD7F32]██║  ██║███████╗██║  ██║██║ ╚═╝ ██║███████╗███████║      ██║  ██║╚██████╔╝███████╗██║ ╚████║   ██║[/]
[#CD7F32]╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚══════╝╚══════╝      ╚═╝  ╚═╝ ╚═════╝ ╚══════╝╚═╝  ╚═══╝   ╚═╝[/]"""

HERMES_CADUCEUS = """[#CD7F32]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⡀⠀⣀⣀⠀⢀⣀⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#CD7F32]⠀⠀⠀⠀⠀⠀⢀⣠⣴⣾⣿⣿⣇⠸⣿⣿⠇⣸⣿⣿⣷⣦⣄⡀⠀⠀⠀⠀⠀⠀[/]
[#FFBF00]⠀⢀⣠⣴⣶⠿⠋⣩⡿⣿⡿⠻⣿⡇⢠⡄⢸⣿⠟⢿⣿⢿⣍⠙⠿⣶⣦⣄⡀⠀[/]
[#FFBF00]⠀⠀⠉⠉⠁⠶⠟⠋⠀⠉⠀⢀⣈⣁⡈⢁⣈⣁⡀⠀⠉⠀⠙⠻⠶⠈⠉⠉⠀⠀[/]
[#FFD700]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣴⣿⡿⠛⢁⡈⠛⢿⣿⣦⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#FFD700]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠿⣿⣦⣤⣈⠁⢠⣴⣿⠿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#FFBF00]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠉⠻⢿⣿⣦⡉⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#FFBF00]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⢷⣦⣈⠛⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#CD7F32]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣴⠦⠈⠙⠿⣦⡄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#CD7F32]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣿⣤⡈⠁⢤⣿⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠛⠷⠄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣀⠑⢶⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⠁⢰⡆⠈⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠳⠈⣡⠞⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]
[#B8860B]⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀[/]"""


# =========================================================================
# Skills scanning
# =========================================================================

# Per-process caches: ``None`` until computed, then a 1-tuple ``(value,)`` so a computed ``None``
# is distinguishable from "not yet computed". Reset by assigning ``None`` (tests, ``hermes skills``).
_available_skills_cache: Optional[tuple] = None
_git_banner_state_cache: Optional[tuple] = None
_latest_release_cache: Optional[tuple] = None


_UNCACHED = object()  # compute() result that must not be memoized


def _memo(cache_name: str, compute):
    """Return the cached value under module global ``cache_name``, computing (and storing) it once."""
    cached = globals()[cache_name]
    if cached is not None:
        return cached[0]
    value = compute()
    if value is not _UNCACHED:
        globals()[cache_name] = (value,)
    return value


def get_available_skills() -> Dict[str, List[str]]:
    """Return skills grouped by category, filtered by platform and disabled state.

    Cached per-process: this feeds only the startup banner, whose snapshot is taken once anyway, and
    the underlying skills-tree walk costs ~100ms. ``prefetch_banner_data()`` uses the cache to pay
    that walk off-thread. A failed scan yields ``{}`` and is not cached.
    """
    def _scan():
        from tools.skills_tool import _find_all_skills
        return _find_all_skills()  # already filtered

    def _compute():
        all_skills = _quiet(_scan)
        if all_skills is None:
            return _UNCACHED
        skills_by_category: Dict[str, List[str]] = {}
        for skill in all_skills:
            skills_by_category.setdefault(skill.get("category") or "general", []).append(skill["name"])
        return skills_by_category

    result = _memo("_available_skills_cache", _compute)
    return {} if result is _UNCACHED else result


# =========================================================================
# Update check
# =========================================================================

# Cache update check results for 6 hours to avoid repeated git fetches
_UPDATE_CHECK_CACHE_SECONDS = 6 * 3600

# Sentinel returned when we know an update exists but can't count commits
# (e.g. nix-built hermes — no local git history to count against).
UPDATE_AVAILABLE_NO_COUNT = -1

_UPSTREAM_REPO_URL = "https://github.com/NousResearch/hermes-agent.git"
_OFFICIAL_REPO_CANONICAL = "github.com/nousresearch/hermes-agent"


def _canonical_github_remote(url: str | None) -> str:
    """Return ``host/owner/repo`` for common GitHub remote URL forms."""
    if not url:
        return ""
    value = url.strip()
    for ssh_prefix in ("git@github.com:", "ssh://git@github.com/"):
        if value.startswith(ssh_prefix):
            value = "github.com/" + value[len(ssh_prefix):]
            break
    else:
        parsed = urlparse(value)
        if parsed.netloc and parsed.path:
            value = f"{parsed.netloc}{parsed.path}"
    return value.strip().rstrip("/").removesuffix(".git").lower()


def _is_official_ssh_remote(url: str | None) -> bool:
    if not url or not url.strip().lower().startswith(("git@", "ssh://")):
        return False
    return _canonical_github_remote(url) == _OFFICIAL_REPO_CANONICAL


_GIT_TEXT_KW = {"text": True, "encoding": "utf-8", "errors": "replace"}


def _git_run(
    args: list[str], *, cwd: Optional[Path] = None, timeout: int = 5, text: bool = True, network: bool = False
):
    """Run ``git <args>`` with the shared subprocess boilerplate; None on any exception.

    git output is UTF-8; on Windows ``text=True`` defaults to the ANSI code page and bytes like 0x90
    (3rd byte of 🐛 in a commit subject) crash the stdlib reader thread (#52649), hence the explicit
    encoding. ``network=True`` (ls-remote/fetch) detaches stdin and disables git/GCM prompts so a
    passive update check can never hang on a ``Username for 'https://github.com':`` prompt.
    """
    kwargs: dict = {}
    if network:
        from hermes_cli._subprocess_compat import noninteractive_git_env

        kwargs = {"stdin": subprocess.DEVNULL, "env": noninteractive_git_env()}
    try:
        return subprocess.run(
            ["git", *args],
            capture_output=True,
            timeout=timeout,
            cwd=str(cwd) if cwd is not None else None,
            **(_GIT_TEXT_KW if text else {}),
            **kwargs,
        )
    except Exception:
        return None


def _git_stdout(args: list[str], *, cwd: Path, timeout: int = 5) -> Optional[str]:
    result = _git_run(args, cwd=cwd, timeout=timeout)
    if result is None or result.returncode != 0:
        return None
    return (result.stdout or "").strip()


def _git_count(args: list[str], *, cwd: Path) -> Optional[int]:
    """``int`` of a successful ``git rev-list --count``-style command, else None.

    Deliberately bypasses ``_git_stdout`` so tests can stub the two layers independently.
    """
    result = _git_run(args, cwd=cwd)
    try:
        if result is not None and result.returncode == 0:
            return int(result.stdout.strip())
    except Exception:
        pass
    return None


def _github_compare_behind(current_rev: str, target_rev: str) -> Optional[int]:
    """Exact behind-count via the GitHub compare API for uncountable graphs.

    Shallow installer clones and ls-remote-only probes know the two tip SHAs but have no local
    history to run ``rev-list --count`` across.
    """
    if not (_is_full_sha(current_rev) and _is_full_sha(target_rev)):
        return None
    url = (
        "https://api.github.com/repos/nousresearch/hermes-agent/"
        f"compare/{current_rev}...{target_rev}"
    )
    def _fetch():
        import urllib.request

        # api.github.com 403s requests without a User-Agent.
        req = urllib.request.Request(
            url, headers={"Accept": "application/vnd.github+json", "User-Agent": "hermes-cli-update-check"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            return json.loads(resp.read().decode("utf-8"))

    payload = _quiet(_fetch)
    ahead = payload.get("ahead_by") if isinstance(payload, dict) else None
    if isinstance(ahead, int) and not isinstance(ahead, bool) and ahead >= 0:
        return ahead
    return None


def _behind_count_or_sentinel(local_rev: str, upstream_rev: str) -> int:
    """Exact behind-count via the compare API, else the honest no-count sentinel.

    ``ahead_by == 0`` with differing tips means the remote tip is reachable from our HEAD — a
    local-ahead checkout, i.e. NOT behind. A local-only HEAD 404s on the API, which safely degrades
    to ``UPDATE_AVAILABLE_NO_COUNT`` — never a fabricated 1.
    """
    counted = _github_compare_behind(local_rev, upstream_rev)
    return counted if counted is not None else UPDATE_AVAILABLE_NO_COUNT


def _tips_behind(head_rev: Optional[str], target_rev: Optional[str], repo_dir: Optional[Path] = None) -> Optional[int]:
    """Behind-count from two tip SHAs: None if either is unknown, 0 when equal, else count/sentinel.

    With ``repo_dir``, a target that is already an ancestor of HEAD (local-ahead checkout) is 0 too.
    """
    if not head_rev or not target_rev:
        return None
    if head_rev == target_rev:
        return 0
    if repo_dir is not None:
        ancestor = _git_run(["merge-base", "--is-ancestor", target_rev, "HEAD"], cwd=repo_dir, text=False)
        if ancestor is not None and ancestor.returncode == 0:
            return 0
    return _behind_count_or_sentinel(head_rev, target_rev)


def _is_full_sha(value: Optional[str]) -> bool:
    return isinstance(value, str) and len(value) == 40 and all(c in "0123456789abcdefABCDEF" for c in value)


def _upstream_main_sha() -> Optional[str]:
    """Tip SHA of upstream main via HTTPS ls-remote (no auth, no prompts)."""
    result = _git_run(["ls-remote", _UPSTREAM_REPO_URL, "refs/heads/main"], timeout=10, network=True)
    if result is None or result.returncode != 0 or not result.stdout:
        return None
    return result.stdout.split()[0] or None


def _check_via_rev(local_rev: str) -> Optional[int]:
    """Compare an embedded git revision to upstream main via ls-remote.

    Returns 0 if up-to-date, the exact behind-count when the GitHub compare API can recover it,
    ``UPDATE_AVAILABLE_NO_COUNT`` if behind by an unknown amount, or ``None`` on failure.
    """
    return _tips_behind(local_rev, _upstream_main_sha())


def _check_via_local_git(repo_dir: Path) -> Optional[int]:
    """Count commits behind origin/main in a local checkout."""
    origin_url = _git_stdout(["remote", "get-url", "origin"], cwd=repo_dir)
    if _is_official_ssh_remote(origin_url):
        head_rev = _git_stdout(["rev-parse", "HEAD"], cwd=repo_dir)
        if not head_rev:
            return None
        # Passive probe via HTTPS ls-remote (never SSH — no hardware-key prompts). Tip SHAs alone
        # can't distinguish "behind" from a local carried commit sitting AHEAD of origin/main, and
        # misreporting an ahead checkout as behind nudges the user into `hermes update`, which can
        # wipe their carried work — hence the ancestor check, against the FRESH upstream SHA (not
        # the possibly stale origin/main tracking ref) so a stale ref can't fake an up-to-date report.
        return _tips_behind(head_rev, _upstream_main_sha(), repo_dir)

    # Installer checkouts are shallow (`git clone --depth 1`). On a shallow
    # clone the history stops at a single commit, so a plain `git fetch` would
    # unshallow the repo (dragging in the whole history) and
    # `rev-list --count HEAD..origin/main` would report a huge bogus "behind"
    # number (e.g. "12492 commits behind"). Detect shallow up front: fetch with
    # --depth 1 to preserve the boundary and compare tip SHAs instead of
    # counting. Full clones (developers, Docker dev images) keep the exact
    # count path unchanged. Mirrors the desktop fix in apps/desktop/electron/main.cjs.
    is_shallow = _git_stdout(["rev-parse", "--is-shallow-repository"], cwd=repo_dir) == "true"

    def _fetch() -> bool:
        # Self-heal abandoned git lock files before fetching. A stale .git/shallow.lock from a
        # crashed fetch makes the fetch fail, the exception is swallowed, and stale refs get
        # compared against HEAD — silently degrading the passive check until a human removes the
        # lock (git never self-heals these). The passive check is also the main tmp_pack GENERATOR
        # on flaky lines (several aborted fetches per day), so it must be the janitor too, or debris
        # accumulates unbounded between manual updates (#93732).
        from hermes_cli.gitlock import clear_stale_git_locks, clear_stale_tmp_packs

        clear_stale_git_locks(repo_dir)
        clear_stale_tmp_packs(repo_dir)

        # Scope the fetch to the one branch the behind-count compares against. An unscoped
        # ``git fetch origin`` transfers every remote head (~1,400 on this repo — measured 3.0 s vs
        # 0.55 s scoped) and can burn the full 10 s timeout on slow links. Modern git updates the
        # ``origin/main`` tracking ref on a scoped fetch, so the ``HEAD..origin/main`` count below
        # is unaffected; the shallow path compares against FETCH_HEAD, which a scoped fetch also
        # updates. ``--depth 1`` preserves the shallow boundary.
        fetch_args = ["fetch", "origin", "main", *(["--depth", "1"] if is_shallow else []), "--quiet"]
        fetch_proc = _git_run(fetch_args, cwd=repo_dir, timeout=10, text=False, network=True)
        return fetch_proc is not None and fetch_proc.returncode == 0

    fetch_ok = _quiet(_fetch, False)  # Offline or timeout — don't use stale refs

    # When the fetch fails, the local origin/main tracking ref is stale. It
    # cannot prove *currentness* (a 0 behind-count may just mean the stale ref
    # hasn't caught up), but if it already shows HEAD behind, that is sound
    # evidence an update exists — the ref was good at some point in the past.
    # Return the positive stale count; return None (inconclusive) otherwise so
    # the caller doesn't cache a false "up to date". (#82166, review #92578)
    if is_shallow:
        if not fetch_ok:
            return None
        # No history to count across the shallow boundary. `origin/main` may not be a tracking ref
        # in a `clone --depth 1`, so prefer FETCH_HEAD (just updated by the fetch above) and fall
        # back to origin/main. Tips differ but the shallow boundary hides the history between them.
        head_rev = _git_stdout(["rev-parse", "HEAD"], cwd=repo_dir)
        target_rev = (
            _git_stdout(["rev-parse", "FETCH_HEAD"], cwd=repo_dir)
            or _git_stdout(["rev-parse", "origin/main"], cwd=repo_dir)
        )
        return _tips_behind(head_rev, target_rev)

    behind = _git_count(["rev-list", "--count", "HEAD..origin/main"], cwd=repo_dir)
    if fetch_ok or (behind is not None and behind > 0):
        return behind
    return None


def _read_json(path: Path) -> Optional[dict]:
    """Parse ``path`` as a JSON object; None when missing, unreadable, or not a dict."""
    blob = _quiet(lambda: json.loads(path.read_text(encoding="utf-8")))
    return blob if isinstance(blob, dict) else None


def check_for_updates() -> Optional[int]:
    """Check whether a Hermes update is available.

    Two paths: if ``HERMES_REVISION`` is set (nix builds embed it), compare it to upstream main via
    ``git ls-remote``. Otherwise look for a local git checkout and count commits behind
    ``origin/main``.
    """
    hermes_home = get_hermes_home()
    cache_file = hermes_home / ".update_check"
    embedded_rev = os.environ.get("HERMES_REVISION") or None

    # Docker images have no working tree to count commits against — the
    # published image excludes `.git` (see .dockerignore) and sets no
    # HERMES_REVISION (that's nix-only). Returning None makes both the Rich
    # banner (build_welcome_banner) and the Ink badge (branding.tsx, guarded
    # on `typeof === 'number' && > 0`) show nothing. The dashboard's REST
    # `/api/hermes/update/check` endpoint short-circuits docker the same way
    # (web_server.py); mirror that here so the banner/TUI surfaces agree.
    def _install_method():
        from hermes_cli.config import detect_install_method, get_project_root
        return detect_install_method(get_project_root())

    if _quiet(_install_method) in {"docker", "apt"}:
        return None

    # Read cache — invalidate if the embedded rev OR installed version has
    # changed since the last check.
    now = time.time()
    cached = _read_json(cache_file)
    if (
        cached is not None
        and now - cached.get("ts", 0) < _UPDATE_CHECK_CACHE_SECONDS
        and cached.get("rev") == embedded_rev
        and cached.get("ver") == VERSION
    ):
        return cached.get("behind")

    if embedded_rev:
        behind = _check_via_rev(embedded_rev)
    else:
        # No git checkout and no embedded revision — can't determine update
        # status (the Docker path, already short-circuited above, or an
        # unsupported install without a source tree).
        repo_dir = _resolve_repo_dir()
        behind = _check_via_local_git(repo_dir) if repo_dir is not None else None

    # Don't cache inconclusive results (None). A None means the check could not run — typically a
    # failed git fetch. Caching None would suppress retries for the full 6-hour cache window,
    # leaving the user with a stale "up to date" or no information for hours after connectivity
    # is restored (#82166).
    if behind is not None:
        _quiet(lambda: cache_file.write_text(
            json.dumps({"ts": now, "behind": behind, "rev": embedded_rev, "ver": VERSION}),
            encoding="utf-8",
        ))

    return behind


def _resolve_repo_dir() -> Optional[Path]:
    """Return the active Hermes git checkout, or None if this isn't a git install.

    Prefers the running code's location over the profile-scoped path because ``$HERMES_HOME/hermes-
    agent/`` may be a stale copy carried over by ``--clone-all``.
    """
    repo_dir = Path(__file__).parent.parent.resolve()
    if not (repo_dir / ".git").exists():
        hermes_home = get_hermes_home()
        repo_dir = hermes_home / "hermes-agent"
    return repo_dir if (repo_dir / ".git").exists() else None


def get_git_banner_state(repo_dir: Optional[Path] = None) -> Optional[dict]:
    """Return upstream/local git hashes for the startup banner.

    Cached per-process (default ``repo_dir`` only): the state costs 2-3 git subprocesses (~100ms)
    and the checkout revision cannot change under a running CLI in a way the banner needs to observe
    live. The cache also lets ``prefetch_banner_data()`` pay this cost off-thread before the banner
    renders.
    """
    if repo_dir is not None:
        return _compute_git_banner_state(repo_dir)
    return _memo("_git_banner_state_cache", _compute_git_banner_state)


def _baked_banner_state() -> Optional[dict]:
    """Banner state from the baked build SHA (Docker image path), or None."""
    def _baked():
        from hermes_cli.build_info import get_build_sha
        return get_build_sha(short=8)

    baked = _quiet(_baked)
    return {"upstream": baked, "local": baked, "ahead": 0} if baked else None


def _compute_git_banner_state(repo_dir: Optional[Path] = None) -> Optional[dict]:
    repo_dir = repo_dir or _resolve_repo_dir()
    if repo_dir is None:
        return _baked_banner_state()

    upstream, local = (_git_stdout(["rev-parse", "--short=8", rev], cwd=repo_dir) for rev in ("origin/main", "HEAD"))
    if not upstream or not local:
        # Live-git lookup failed (e.g. shallow clone without origin/main).
        return _baked_banner_state()

    ahead = _git_count(["rev-list", "--count", "origin/main..HEAD"], cwd=repo_dir) or 0
    return {"upstream": upstream, "local": local, "ahead": max(ahead, 0)}


_RELEASE_URL_BASE = "https://github.com/NousResearch/hermes-agent/releases/tag"


def get_latest_release_tag(repo_dir: Optional[Path] = None) -> Optional[tuple]:
    """Return ``(tag, release_url)`` for the latest git tag, or None.

    Local-only — runs ``git describe --tags --abbrev=0`` against the Hermes checkout. Cached per-
    process (a miss is cached too). Release URL always points at the canonical
    NousResearch/hermes-agent repo (forks don't get a link).
    """
    def _compute():
        rd = repo_dir or _resolve_repo_dir()
        tag = _git_stdout(["describe", "--tags", "--abbrev=0"], cwd=rd, timeout=3) if rd else None
        return (tag, f"{_RELEASE_URL_BASE}/{tag}") if tag else None

    return _memo("_latest_release_cache", _compute)


def format_banner_version_label() -> str:
    """Return the version label shown in the startup banner title."""
    base = f"Hermes Agent v{VERSION} ({RELEASE_DATE})"
    state = get_git_banner_state()
    if not state:
        return base

    upstream, local = state["upstream"], state["local"]
    ahead = int(state.get("ahead") or 0)
    if ahead <= 0 or upstream == local:
        return f"{base} · upstream {upstream}"
    return f"{base} · upstream {upstream} · local {local} (+{ahead} carried {_plural(ahead, 'commit')})"


# =========================================================================
# Non-blocking update check
# =========================================================================

_update_result: Optional[int] = None
_update_check_done = threading.Event()


def _daemon(name: Optional[str], target) -> None:
    """Start a daemon thread running ``target`` with any exception swallowed."""
    threading.Thread(target=lambda: _quiet(target), name=name, daemon=True).start()


def prefetch_update_check():
    """Kick off update check in a background daemon thread."""
    def _run():
        global _update_result
        _update_result = check_for_updates()
        _update_check_done.set()
    _daemon(None, _run)


_banner_data_prefetch_started = False


def prefetch_banner_data():
    """Warm the banner's subprocess/I/O-heavy inputs in a daemon thread.

    Git state (~130ms of subprocesses) and the skills index (~110ms rglob) are cached per-process
    by their own modules, so warming them while the main thread pays the CPU-bound ``cli`` /
    prompt_toolkit imports overlaps GIL-releasing I/O with import work. Idempotent; failures
    don't matter because the banner recomputes anything missing.
    """
    global _banner_data_prefetch_started
    if _banner_data_prefetch_started:
        return
    _banner_data_prefetch_started = True

    def _run() -> None:
        for warm in (get_git_banner_state, get_latest_release_tag, get_available_skills):
            _quiet(warm)

    _daemon("banner-data-prefetch", _run)


def get_update_result(timeout: float = 0.5) -> Optional[int]:
    """Get result of prefetched check. Returns None if not ready."""
    _update_check_done.wait(timeout=timeout)
    return _update_result


def _format_update_notice(behind: int) -> str:
    """Render the update warning line for a non-zero ``behind`` result."""
    from hermes_cli.config import get_managed_update_command, recommended_update_command
    if behind > 0:
        return (
            f"[bold yellow]⚠ {behind} {_plural(behind, 'commit')} behind[/]"
            f"[dim yellow] — run [bold]{recommended_update_command()}[/bold] to update[/]"
        )
    # UPDATE_AVAILABLE_NO_COUNT: nix-built hermes; we know an update
    # exists but not by how much, and we don't know how the user
    # installed it (nix run, profile, system flake, home-manager).
    managed_cmd = get_managed_update_command()
    suffix = f"[dim yellow] — run [bold]{managed_cmd}[/bold][/]" if managed_cmd else ""
    return f"[bold yellow]⚠ update available[/]{suffix}"


_deferred_update_notice_started = False


def _defer_update_notice(console: "Console", max_wait: float = 30.0) -> None:
    """Print the update warning once the prefetched check completes.

    Used when the banner rendered before the update prefetch finished so startup never blocks on
    git/network. Prints at most once per process.
    """
    global _deferred_update_notice_started
    if _deferred_update_notice_started:
        return
    _deferred_update_notice_started = True

    def _wait_and_print() -> None:
        if _update_check_done.wait(timeout=max_wait) and _update_result:
            console.print(_format_update_notice(_update_result))

    _daemon("update-notice", _wait_and_print)  # never break the session over an update notice


# =========================================================================
# Welcome banner
# =========================================================================

def _plural(n: int, word: str) -> str:
    return word if n == 1 else f"{word}s"


def _format_context_length(tokens: int) -> str:
    """Format a token count for display (e.g. 128000 → '128K', 1048576 → '1M')."""
    for unit, div in (("M", 1_000_000), ("K", 1_000)):
        if tokens >= div:
            val = tokens / div
            rounded = round(val)
            return f"{rounded}{unit}" if abs(val - rounded) < 0.05 else f"{val:.1f}{unit}"
    return str(tokens)


def _display_toolset_name(toolset_name: str) -> str:
    """Normalize internal/legacy toolset identifiers for banner display."""
    if not toolset_name:
        return "unknown"
    return toolset_name[:-6] if toolset_name.endswith("_tools") else toolset_name


def _short_label(name: str) -> str:
    """Truncate a model/preset slug to fit the banner's left column."""
    return name[:25] + "..." if len(name) > 28 else name


# =========================================================================
# Banner snapshot — warm-launch fast path
# =========================================================================
# The banner's tool panel needs the full tool registry (get_tool_definitions:
# tools/*.py discovery + every check_fn), which costs ~0.5-0.9s cold and is
# the single largest chunk of CLI time-to-banner. The tool list shown in the
# banner is a pure function of (config.yaml, .env, code checkout, enabled
# toolsets), so we snapshot the rendered inputs to disk after each launch
# and replay them on the next one when the fingerprint matches. The agent's
# REAL tool list is still computed fresh at first message (agent init) —
# the snapshot only feeds the cosmetic startup panel, and a background
# refresh re-verifies it right after the banner renders (see
# cli.show_banner), so a stale panel self-heals within one launch.

_BANNER_SNAPSHOT_VERSION = 1


def _banner_snapshot_path() -> Path:
    return get_hermes_home() / "cache" / "banner_snapshot.json"


def banner_snapshot_fingerprint() -> Optional[str]:
    """Fingerprint the inputs the banner tool panel depends on."""
    import hashlib

    def _inputs():
        from hermes_cli.config import get_config_path
        return (get_config_path(), get_hermes_home() / ".env")

    paths = _quiet(_inputs)
    if paths is None:
        return None
    parts = [f"v{_BANNER_SNAPSHOT_VERSION}"]
    for p in paths:
        st = _quiet(p.stat)
        parts.append(f"{p.name}:{st.st_mtime_ns}:{st.st_size}" if st else f"{p.name}:absent")
    # Code checkout: version + git HEAD when available (post-update change).
    parts.append(str(VERSION))
    state = get_git_banner_state()
    if state:
        parts.append(str(state.get("local", "")))
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def load_banner_snapshot(enabled_toolsets: List[str] = None) -> Optional[Dict[str, Any]]:
    """Return the stored banner snapshot when its fingerprint is current."""
    blob = _read_json(_banner_snapshot_path())
    if blob is None:
        return None
    fp = banner_snapshot_fingerprint()
    if not fp or blob.get("fingerprint") != fp:
        return None
    if blob.get("enabled_toolsets") != sorted(enabled_toolsets or []):
        return None
    if not isinstance(blob.get("tools"), list) or not all(
        isinstance(blob.get(k), dict) for k in ("toolset_map", "availability", "skills_by_category")
    ):
        return None
    return blob


def save_banner_snapshot(
    tools: List[dict],
    enabled_toolsets: List[str],
    availability: Dict[str, Any],
    toolset_map: Dict[str, str],
) -> None:
    """Persist the banner tool panel inputs for next launch (best-effort)."""
    fp = banner_snapshot_fingerprint()
    if not fp:
        return
    payload = {
        "fingerprint": fp,
        "enabled_toolsets": sorted(enabled_toolsets or []),
        "tools": [
            {"function": {"name": t["function"]["name"]}}
            for t in tools
            if isinstance(t, dict) and t.get("function", {}).get("name")
        ],
        "toolset_map": toolset_map,
        "availability": {
            "unavailable_toolsets": availability.get("unavailable_toolsets", []),
            **{k: list(availability.get(k, [])) for k in ("lazy_tools", "disabled_tools")},
        },
        "skills_by_category": get_available_skills(),
    }
    def _write():
        import tempfile
        path = _banner_snapshot_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=str(path.parent), prefix=".banner_snap.")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(payload, fh)
        os.replace(tmp, path)

    _quiet(_write)


def compute_toolset_availability(enabled_toolsets: List[str] = None) -> Dict[str, Any]:
    """Compute the banner's toolset-availability payload.

    Returns ``{"unavailable_toolsets", "lazy_tools", "disabled_tools"}``. Split out so the result
    can be snapshotted and replayed on the next launch without importing ``model_tools``.
    """
    from model_tools import check_tool_availability, TOOLSET_REQUIREMENTS

    enabled_toolsets = enabled_toolsets or []
    _, unavailable_toolsets = check_tool_availability(quiet=True)
    # The availability check walks the GLOBAL toolset registry, so it includes
    # toolsets that aren't part of this agent's platform set at all (e.g.
    # `discord`, `feishu_doc` on a CLI session). Those must never surface in the
    # banner's "Available Tools" — they aren't exposed to the agent. Restrict to
    # toolsets actually enabled for this agent; a toolset that's enabled but
    # currently has unmet deps legitimately shows as disabled/lazy below.
    _enabled_ts = {str(t) for t in enabled_toolsets}
    if _enabled_ts:
        unavailable_toolsets = [
            item for item in unavailable_toolsets
            if str(item.get("id", item.get("name", ""))) in _enabled_ts
        ]
    # Tools whose toolset has a check_fn are lazy-initialized (e.g. honcho, homeassistant) — they
    # show as unavailable at banner time because the check hasn't run yet, but aren't misconfigured.
    lazy_tools, disabled_tools = set(), set()
    for item in unavailable_toolsets:
        is_lazy = TOOLSET_REQUIREMENTS.get(item.get("name", ""), {}).get("check_fn")
        (lazy_tools if is_lazy else disabled_tools).update(item.get("tools", []))
    return {
        "unavailable_toolsets": unavailable_toolsets,
        "lazy_tools": sorted(lazy_tools),
        "disabled_tools": sorted(disabled_tools),
    }


def _mcp_server_line(srv: dict, *, dim: str, text: str) -> str:
    """One banner line for an MCP server status entry."""
    name, transport = srv["name"], srv["transport"]
    if srv["connected"]:
        return f"[dim {dim}]{name}[/] [{text}]({transport})[/] [dim {dim}]—[/] [{text}]{srv['tools']} tool(s)[/]"
    status = "disabled" if srv.get("disabled") else srv.get("status")
    suffix = {
        "disabled": f"[dim {dim}]— disabled[/]",
        "connecting": "[yellow]— connecting[/]",
        "configured": f"[dim {dim}]— configured[/]",
    }.get(status)
    if suffix is not None:
        return f"[dim {dim}]{name}[/] [dim]({transport})[/] {suffix}"
    return f"[red]{name}[/] [dim]({transport})[/] [red]— failed[/]"


def _truncate_tool_names(tool_names: List[str]) -> List[Optional[str]]:
    """Cut a toolset's tool list to ~42 columns; ``None`` marks the elided tail."""
    if len(", ".join(tool_names)) <= 45:
        return list(tool_names)
    short_names: List[Optional[str]] = []
    length = 0
    for name in tool_names:
        if length + len(name) + 2 > 42:
            short_names.append(None)
            break
        short_names.append(name)
        length += len(name) + 2
    return short_names


def _pack_skill_names(skill_names: List[str], avail: int) -> str:
    """Join skill names into ``avail`` columns, ending with ``+N more`` when they don't all fit."""
    parts: List[str] = []
    length = 0
    for i, name in enumerate(skill_names):
        needed = (2 if parts else 0) + len(name)
        # Indicator size IF we were to add this skill then stop.
        after = len(skill_names) - (i + 1)
        ind_len = len(f", +{after} more") if after > 0 else 0
        if parts and length + needed + ind_len > avail:
            parts.append(f"+{len(skill_names) - len(parts)} more")
            break
        parts.append(name)
        length += needed
    return ", ".join(parts)


def _moa_aggregator_label(preset_name: str) -> str:
    """Short aggregator-model label for a MoA preset ("" when the preset has none)."""
    from hermes_cli.config import load_config
    from hermes_cli.moa_config import normalize_moa_config

    preset = normalize_moa_config(load_config().get("moa") or {}).get("presets", {}).get(preset_name)
    model = str(((preset or {}).get("aggregator") or {}).get("model") or "")
    return model.split("/")[-1]


def _mcp_configured() -> bool:
    """Cheap probe: does config.yaml or the persisted plugin key cache name any MCP server?

    The full ``get_mcp_status()`` path resolves portable plugin MCP servers, which JOINS the in-flight
    background plugin discovery (~100ms on the startup path), so skip it when nothing is configured.
    When either probe can't tell, take the full path.
    """
    def _native():
        from hermes_cli.config import load_config
        return bool((load_config() or {}).get("mcp_servers"))

    def _portable():
        from hermes_cli.plugins import get_portable_mcp_server_names_nowait
        return bool(get_portable_mcp_server_names_nowait())

    return _quiet(_native, True) or _quiet(_portable, True)


def _probe_mcp_status() -> list:
    from tools.mcp_tool import get_mcp_status
    return get_mcp_status()


def _codex_runtime_active() -> bool:
    """True when the codex_app_server runtime is active (tool counts then live inside codex)."""
    from hermes_cli.codex_runtime_switch import get_current_runtime
    from hermes_cli.config import load_config
    return get_current_runtime(load_config()) == "codex_app_server"


def _active_profile_name() -> Optional[str]:
    from hermes_cli.profiles import get_active_profile_name
    return get_active_profile_name()


def build_welcome_banner(console: "Console", model: str, cwd: str,
                         tools: List[dict] = None,
                         enabled_toolsets: List[str] = None,
                         session_id: str = None,
                         get_toolset_for_tool=None,
                         context_length: int = None,
                         provider: str = None,
                         availability: Dict[str, Any] = None,
                         skills_by_category: Dict[str, List[str]] = None):
    """Build and print a welcome banner with caduceus on left and info on right.

    When ``provider == "moa"``, ``model`` is a MoA preset name and the aggregator is rendered.
    Passing a precomputed ``availability`` together with ``get_toolset_for_tool`` avoids any
    ``model_tools`` import (banner snapshot replay).
    """
    from rich.panel import Panel
    from rich.table import Table
    if get_toolset_for_tool is None:
        from model_tools import get_toolset_for_tool

    tools = tools or []
    enabled_toolsets = enabled_toolsets or []

    if availability is None:
        availability = compute_toolset_availability(enabled_toolsets)
    unavailable_toolsets = availability.get("unavailable_toolsets", [])
    lazy_tools = set(availability.get("lazy_tools", []))
    disabled_tools = set(availability.get("disabled_tools", []))
    _enabled_ts = {str(t) for t in enabled_toolsets}

    layout_table = Table.grid(padding=(0, 2))
    layout_table.add_column("left", justify="center")
    layout_table.add_column("right", justify="left")

    # Resolve skin colors once for the entire banner
    accent = _skin_color("banner_accent", "#FFBF00")
    dim = _skin_color("banner_dim", "#B8860B")
    text = _skin_color("banner_text", "#FFF8DC")

    # Use skin's custom caduceus art if provided
    _bskin = _quiet(_active_skin)
    left_lines = ["", getattr(_bskin, "banner_hero", None) or HERMES_CADUCEUS, ""]

    def _dim_sep(label: str) -> str:
        return f" [dim {dim}]·[/] [dim {dim}]{label}[/]"

    ctx_str = _dim_sep(f"{_format_context_length(context_length)} context") if context_length else ""
    nous_str = _dim_sep("Nous Research")
    if (provider or "").strip().lower() == "moa":
        # MoA virtual provider: ``model`` is a preset name. Show the preset and
        # its aggregator so the banner is meaningful instead of a bare slug.
        agg_label = _quiet(lambda: _moa_aggregator_label(model), "")
        agg_str = _dim_sep(f"agg {agg_label}") if agg_label else ""
        left_lines.append(f"[{accent}]MoA: {_short_label(model)}[/]{agg_str}{ctx_str}{nous_str}")
    elif not (model or "").strip() or (model or "").strip().lower() == "unknown":
        # Unconfigured install: say so in red instead of a blank/"unknown"
        # slug — this is the single clearest place to tell the user what
        # is wrong and how to fix it.
        left_lines.append(
            f"[bold red]no model configured[/] "
            f"[dim {dim}]— run /model or hermes setup[/]"
        )
    else:
        model_short = model.split("/")[-1].removesuffix(".gguf")
        left_lines.append(f"[{accent}]{_short_label(model_short)}[/]{ctx_str}{nous_str}")

    if os.getenv("HERMES_YOLO_MODE"):
        left_lines.append(f"[bold red]⚠ YOLO mode[/] [dim {dim}]— all approval prompts bypassed[/]")
    left_lines.append(f"[dim {dim}]{cwd}[/]")
    if session_id:
        left_lines.append(f"[dim {_skin_color('session_border', '#8B8682')}]Session: {session_id}[/]")
    left_content = "\n".join(left_lines)

    right_lines = [f"[bold {accent}]Available Tools[/]"]
    toolsets_dict: Dict[str, list] = {}

    for tool in tools:
        tool_name = tool["function"]["name"]
        toolset = _display_toolset_name(get_toolset_for_tool(tool_name) or "other")
        toolsets_dict.setdefault(toolset, []).append(tool_name)

    for item in unavailable_toolsets:
        names = toolsets_dict.setdefault(_display_toolset_name(item.get("id", item.get("name", "unknown"))), [])
        for tool_name in item.get("tools", []):
            if tool_name not in names:
                names.append(tool_name)

    sorted_toolsets = sorted(toolsets_dict.keys())

    def _color_tool(name: Optional[str]) -> str:
        if name is None:  # truncation marker
            return "[dim]...[/]"
        if name in disabled_tools:
            return f"[red]{name}[/]"
        if name in lazy_tools:
            return f"[yellow]{name}[/]"
        return f"[{text}]{name}[/]"

    for toolset in sorted_toolsets[:8]:
        tool_names = _truncate_tool_names(sorted(toolsets_dict[toolset]))
        right_lines.append(f"[dim {dim}]{toolset}:[/] {', '.join(_color_tool(n) for n in tool_names)}")

    if len(sorted_toolsets) > 8:
        right_lines.append(f"[dim {dim}](and {len(sorted_toolsets) - 8} more toolsets...)[/]")

    # MCP Servers section (only if configured). Probe cheaply first: the
    # full get_mcp_status() path resolves portable plugin MCP servers,
    # which JOINS the in-flight background plugin discovery (~100ms on the
    # startup path). When neither config.yaml nor the persisted plugin
    # key cache mentions any MCP server, skip the section outright.
    mcp_status = _quiet(_probe_mcp_status, []) if _mcp_configured() else []

    if mcp_status:
        right_lines += ["", f"[bold {accent}]MCP Servers[/]"]
        right_lines.extend(_mcp_server_line(srv, dim=dim, text=text) for srv in mcp_status)

    right_lines += ["", f"[bold {accent}]Available Skills[/]"]
    # The skills catalog is only reachable when the `skills` toolset is enabled
    # (it exposes skill_view / skill_manage). When it's disabled — e.g. a Blank
    # Slate install — the agent literally cannot load any skill, so advertising
    # the on-disk catalog here is misleading. Reflect the real state instead.
    _skills_enabled = (not _enabled_ts) or ("skills" in _enabled_ts)
    if not _skills_enabled:
        skills_by_category = {}
    elif skills_by_category is None:
        skills_by_category = get_available_skills()
    total_skills = sum(len(s) for s in skills_by_category.values())

    # Dynamically size skills display based on terminal width.
    # Rich grid with 2 columns; right column gets roughly 60% of terminal.
    _term_cols = shutil.get_terminal_size().columns
    _right_col_width = max(int(_term_cols * 0.6) - 10, 30)

    if not _skills_enabled:
        right_lines.append(f"[dim {dim}]Skills toolset disabled[/]")
    elif skills_by_category:
        for category in sorted(skills_by_category.keys()):
            # Account for the "category: " prefix.
            skills_str = _pack_skill_names(sorted(skills_by_category[category]), max(_right_col_width - len(category) - 2, 20))
            right_lines.append(f"[dim {dim}]{category}:[/] [{text}]{skills_str}[/]")
    else:
        right_lines.append(f"[dim {dim}]No skills installed[/]")

    right_lines.append("")
    mcp_connected = sum(1 for s in mcp_status if s["connected"])
    summary_parts = [f"{len(tools)} tools", f"{total_skills} skills"]
    if mcp_connected:
        summary_parts.append(f"{mcp_connected} MCP servers")
    summary_parts.append("/help for commands")
    # Indicate when the codex_app_server runtime is active so users
    # understand why tool counts may not match what's actually reachable
    # (codex builds its own tool list inside the spawned subprocess).
    if _quiet(_codex_runtime_active, False):
        right_lines.append(
            f"[bold {accent}]Runtime:[/] [{text}]codex app-server[/] "
            f"[dim {dim}](terminal/file ops/MCP run inside codex)[/]"
        )
    # Show active profile name when not 'default'. Never break the banner over a profiles.py bug.
    _profile_name = _quiet(_active_profile_name)
    if _profile_name and _profile_name != "default":
        right_lines.append(f"[bold {accent}]Profile:[/] [{text}]{_profile_name}[/]")

    right_lines.append(f"[dim {dim}]{' · '.join(summary_parts)}[/]")

    # Update check — use prefetched result if available. NEVER block the
    # banner on it: the prefetch does git/network work that rarely finishes
    # before the banner renders, so a blocking wait here just adds its full
    # timeout to every startup (500ms of the banner path pre-fix). If the
    # result isn't ready yet, defer the warning line: a daemon thread waits
    # for the prefetch and prints the same notice above the prompt when it
    # lands (prompt_toolkit's patch_stdout renders late prints safely).
    def _update_line():
        behind = get_update_result(timeout=0.05)
        if behind is None and not _update_check_done.is_set():
            _defer_update_notice(console)
        elif behind is not None and behind != 0:
            right_lines.append(_format_update_notice(behind))

    _quiet(_update_line)  # Never break the banner over an update check

    layout_table.add_row(left_content, "\n".join(right_lines))

    version_label = format_banner_version_label()
    release_info = get_latest_release_tag()
    if release_info:
        version_label = f"[link={release_info[1]}]{version_label}[/link]"
    outer_panel = Panel(
        layout_table,
        title=f"[bold {_skin_color('banner_title', '#FFD700')}]{version_label}[/]",
        border_style=_skin_color("banner_border", "#CD7F32"),
        padding=(0, 2),
    )

    console.print()
    if shutil.get_terminal_size().columns >= 95:
        console.print(getattr(_bskin, "banner_logo", None) or HERMES_AGENT_LOGO)
        console.print()
    console.print(outer_panel)
