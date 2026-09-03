"""Single-use OAuth grant hygiene: strip cloned grants from profiles, heal forked grants.

Split out of ``hermes_cli/auth.py``; every moved name is re-imported there, so
``hermes_cli.auth.<name>`` keeps resolving (and monkeypatching) as before. Origin-internal
helpers are imported lazily inside each function (no import cycle; patches on
``hermes_cli.auth.<helper>`` still intercept).
"""

from __future__ import annotations

import logging
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from hermes_cli.auth_constants import _decode_jwt_claims

# Log-record parity with the origin module (caplog tests pin "hermes_cli.auth").
logger = logging.getLogger("hermes_cli.auth")


# Pool providers whose OAuth refresh tokens are SINGLE-USE: redeeming the
# refresh token rotates the pair and revokes the old one. A grant forked into
# two auth.json files is therefore not two credentials but one credential with
# two owners — the first owner to refresh strands the other with
# ``invalid_grant`` / ``refresh_token_reused`` (#100339; same class as the
# ``providers.<id>`` write-through hazard in #48415 / #43589). Profiles must
# never receive a copy of these grants: ONE grant lives at the global root and
# named profiles read it through the ``read_credential_pool`` root fallback.
SINGLE_USE_REFRESH_POOL_PROVIDERS = frozenset({
    "anthropic",
    "openai-codex",
    "xai-oauth",
})


# Singleton credential files that hold the same single-use grants outside
# ``auth.json``. Copying one into a profile re-seeds a forked pool row on the
# profile's next ``load_pool()``.
SINGLE_USE_OAUTH_SINGLETON_FILES = (".anthropic_oauth.json",)


def _is_oauth_pool_payload(entry: Any) -> bool:
    if not isinstance(entry, dict):
        return False
    auth_type = str(entry.get("auth_type") or "").strip().lower()
    if auth_type == "oauth":
        return True
    # Legacy rows predating ``auth_type``: an Anthropic OAuth access token or
    # any row carrying a refresh token is an OAuth grant.
    if str(entry.get("refresh_token") or "").strip():
        return True
    return str(entry.get("access_token") or "").startswith("sk-ant-oat")


def strip_cloned_single_use_oauth_grants(profile_dir: Path) -> Dict[str, Any]:
    """Remove forked single-use OAuth grants from a freshly cloned profile.

    Called after any code path that copies credential files from one profile into another (``hermes
    profile create --clone-all``, the dashboard/TUI ``mirror_credentials`` flow). API-key pool rows
    are kept — a static key is safe to duplicate.

    Returns a summary ``{"pool": [...provider ids], "providers": [...], "files": [...]}`` of what
    was stripped (empty lists when nothing was). Never raises: a clone must not fail because
    credential hygiene could not run — the caller logs the summary.
    """
    from hermes_cli.auth import _save_auth_store
    stripped: Dict[str, Any] = {"pool": [], "providers": [], "files": []}
    profile_dir = Path(profile_dir)
    for name in SINGLE_USE_OAUTH_SINGLETON_FILES:
        try:
            target = profile_dir / name
            if target.is_file() or target.is_symlink():
                target.unlink()
                stripped["files"].append(name)
        except OSError:
            logger.debug("Could not remove cloned %s from %s", name, profile_dir, exc_info=True)

    auth_path = profile_dir / "auth.json"
    if not auth_path.is_file():
        return stripped
    try:
        store = json.loads(auth_path.read_text(encoding="utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return stripped
    if not isinstance(store, dict):
        return stripped

    changed = False
    pool = store.get("credential_pool")
    if isinstance(pool, dict):
        for provider_id in list(pool):
            if provider_id not in SINGLE_USE_REFRESH_POOL_PROVIDERS:
                continue
            entries = pool.get(provider_id)
            if not isinstance(entries, list):
                continue
            kept = [e for e in entries if not _is_oauth_pool_payload(e)]
            if len(kept) != len(entries):
                changed = True
                stripped["pool"].append(provider_id)
                if kept:
                    pool[provider_id] = kept
                else:
                    # No local rows at all → read_credential_pool falls back
                    # to the root slice for this provider.
                    del pool[provider_id]
    providers = store.get("providers")
    if isinstance(providers, dict):
        # Device-code grants for these providers live under providers.<id>;
        # _load_provider_state has the same root fallback, so dropping the
        # copy keeps the profile working while removing the fork.
        for provider_id in ("openai-codex", "xai-oauth"):
            block = providers.get(provider_id)
            if isinstance(block, dict) and block:
                del providers[provider_id]
                stripped["providers"].append(provider_id)
                changed = True
    if not changed:
        return stripped
    try:
        _save_auth_store(store, target_path=auth_path)
    except Exception:
        logger.debug(
            "Failed to strip cloned single-use OAuth grants from %s",
            auth_path,
            exc_info=True,
        )
    return stripped


_OAUTH_TOKEN_FIELDS = (
    "access_token",
    "refresh_token",
    "expires_at",
    "expires_at_ms",
    "last_refresh",
)


_oauth_heal_notices: List[str] = []


# provider -> (profile auth.json path, auth.json mtime_ns, singleton mtime_ns)
# of the last store verified fork-free; lets load_pool() skip the locked scan.
_oauth_heal_clean_marks: Dict[str, Tuple[str, Optional[int], Optional[int]]] = {}


def consume_oauth_heal_notices() -> List[str]:
    """Return (and clear) human-readable notes about heals run in this process.

    ``hermes auth list`` / ``hermes auth status`` print them so the user sees that a forked grant
    was consolidated rather than only finding it in logs.
    """
    from hermes_cli.auth import _oauth_heal_notices
    notes = list(_oauth_heal_notices)
    _oauth_heal_notices.clear()
    return notes


def _oauth_identity(entry: Dict[str, Any]) -> Optional[str]:
    """Stable account identity for an OAuth row when the token carries one.

    Codex / xAI access tokens are JWTs with ``sub`` / ``email`` / ``chatgpt_account_id`` claims;
    Anthropic ``sk-ant-oat`` tokens carry none (returns None, so lineage rests on id / token
    material).
    """
    from hermes_cli.auth import _nonempty_str
    if not isinstance(entry, dict):
        return None
    for token in (entry.get("access_token"), entry.get("id_token")):
        claims = _decode_jwt_claims(token)
        if not claims:
            continue
        nested = claims.get("https://api.openai.com/auth")
        account = nested.get("chatgpt_account_id") if isinstance(nested, dict) else None
        for value in (account, claims.get("sub"), claims.get("email")):
            if _nonempty_str(value):
                return value.strip()
    return None


def _oauth_freshness(entry: Dict[str, Any]) -> float:
    """Best-effort 'how recently was this pair issued' score (epoch seconds).

    A rotation always issues a later-expiring access token, so ``expires_at`` ordering identifies
    the live copy; ``last_refresh`` and the JWT ``exp`` claim are fallbacks for rows that do not
    persist expiry.
    """
    from agent.credential_pool import _parse_absolute_timestamp

    best = 0.0
    for key in ("expires_at_ms", "expires_at", "last_refresh"):
        ts = _parse_absolute_timestamp(entry.get(key))
        if ts and ts > best:
            best = ts
    if best == 0.0:
        exp = _decode_jwt_claims(entry.get("access_token")).get("exp")
        ts = _parse_absolute_timestamp(exp)
        if ts:
            best = ts
    return best


def _find_root_counterpart(
    profile_row: Dict[str, Any], root_rows: List[Dict[str, Any]]
) -> Optional[int]:
    """Index of the root OAuth row that shares a grant lineage with *profile_row*.

    Fallback per the one-grant-at-root rule: same provider + same OAuth client — every Anthropic
    ``hermes_pkce`` grant uses one client id and carries no claims, so two Anthropic OAuth rows with
    no contrary identity are one lineage.
    """
    from hermes_cli.auth import _nonempty_str
    candidates = [i for i, r in enumerate(root_rows) if _is_oauth_pool_payload(r)]
    if not candidates:
        return None
    pid = profile_row.get("id")
    for i in candidates:
        if pid and root_rows[i].get("id") == pid:
            return i
    p_ident = _oauth_identity(profile_row)
    for i in candidates:
        r_ident = _oauth_identity(root_rows[i])
        if p_ident and r_ident and p_ident == r_ident:
            return i
    for key in ("refresh_token", "access_token"):
        p_val = profile_row.get(key)
        if not _nonempty_str(p_val):
            continue
        for i in candidates:
            if root_rows[i].get(key) == p_val:
                return i
    # Fallback: same provider + same client. Only a contradicting identity
    # (both sides carry claims and they differ from every root row) blocks it.
    if p_ident:
        for i in candidates:
            if not _oauth_identity(root_rows[i]):
                return i
        return None
    return candidates[0]


def _adopt_oauth_material(target: Dict[str, Any], winner: Dict[str, Any]) -> Dict[str, Any]:
    """Return *target* carrying *winner*'s token pair, status markers cleared."""
    from hermes_cli.auth import _POOL_STATUS_FIELDS
    merged = dict(target)
    for key in _OAUTH_TOKEN_FIELDS:
        if winner.get(key) is not None:
            merged[key] = winner[key]
        else:
            merged.pop(key, None)
    for status_field in _POOL_STATUS_FIELDS:
        merged[status_field] = None
    return merged


def _singleton_as_row(path: Path) -> Optional[Dict[str, Any]]:
    """Read a ``.anthropic_oauth.json`` as a pool-row-shaped dict, or None."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not str(data.get("accessToken") or "").strip():
        return None
    return {
        "access_token": data.get("accessToken"),
        "refresh_token": data.get("refreshToken"),
        "expires_at_ms": data.get("expiresAt"),
    }


# ── One-time heal for installs that ALREADY forked a single-use grant ────────
#
# Fleets created before the clone-strip (``strip_cloned_single_use_oauth_grants``) / root-write-through have
# profile-local copies of the root grant. Those copies are the same credential
# with several owners: whichever profile rotated last holds the only live
# refresh token and every other copy (root included) is spent. Upgrading alone
# does not fix that — the first load in each profile would keep using its own
# doomed copy. ``heal_forked_single_use_oauth_grants`` runs at profile
# ``load_pool()`` time: it finds the profile rows that share LINEAGE with a
# root row (same pool id — clone-all and the old borrowed-persist both kept
# it — or the same account identity / token material), keeps the copy most
# likely to still be live (freshest rotation), writes that copy into ROOT when
# root's is older, and strips the profile's copy so the profile borrows root
# from then on. Idempotent (a healed profile has no matched rows), never
# touches API-key rows, never deletes a row that has no root counterpart
# (an independent ``hermes -p <p> auth add`` grant, or the only surviving
# copy), and reads only the two auth.json files the existing root fallback
# already reads — no environ / secret-scope reads.


def heal_forked_single_use_oauth_grants(provider_id: str) -> Optional[Dict[str, Any]]:
    """Consolidate a profile's forked copy of a single-use OAuth grant into root.

    Runs only in profile mode for ``SINGLE_USE_REFRESH_POOL_PROVIDERS``. Returns a summary
    ``{"adopted": bool, "stripped_ids": [...], "files": [...], "providers_block": bool}`` when
    something was healed, else ``None``. Never raises.
    """
    if provider_id not in SINGLE_USE_REFRESH_POOL_PROVIDERS:
        return None
    try:
        return _heal_forked_single_use_oauth_grants(provider_id)
    except Exception:
        logger.debug("%s: forked-OAuth heal skipped", provider_id, exc_info=True)
        return None


def _heal_forked_provider_block(
    profile_store: Dict[str, Any], root_store: Dict[str, Any], provider_id: str,
) -> Optional[bool]:
    """Consolidate a forked ``providers.<id>`` device-code block into root.

    Returns None when nothing matched, False when the profile copy was dropped (root already
    newest), True when the profile copy was fresher and was adopted into root.
    """
    p_providers = profile_store.get("providers")
    r_providers = root_store.get("providers")
    if not (isinstance(p_providers, dict) and isinstance(r_providers, dict)):
        return None
    p_block = p_providers.get(provider_id)
    r_block = r_providers.get(provider_id)
    if not (isinstance(p_block, dict) and p_block and isinstance(r_block, dict) and r_block):
        return None

    def _flat(block: Dict[str, Any]) -> Dict[str, Any]:
        tokens = block.get("tokens") if isinstance(block.get("tokens"), dict) else {}
        return {**tokens, "last_refresh": block.get("last_refresh")}

    p_flat, r_flat = _flat(p_block), _flat(r_block)
    p_ident, r_ident = _oauth_identity(p_flat), _oauth_identity(r_flat)
    if p_ident and r_ident and p_ident != r_ident:
        return None
    adopted = _oauth_freshness(p_flat) > _oauth_freshness(r_flat)
    if adopted:
        r_providers[provider_id] = dict(p_block)
    del p_providers[provider_id]
    return adopted


def _heal_forked_single_use_oauth_grants(provider_id: str) -> Optional[Dict[str, Any]]:
    from hermes_cli.auth import _auth_file_path, _auth_store_lock, _global_auth_file_path, _load_auth_store, _oauth_heal_clean_marks, _oauth_heal_notices, _same_path, _save_auth_store
    root_path = _global_auth_file_path()
    if root_path is None:
        return None  # classic mode: nothing to consolidate into
    if os.environ.get("PYTEST_CURRENT_TEST"):
        # Same seat belt as the write-through paths: never touch the real
        # user's ~/.hermes/auth.json from a test that forgot to isolate HOME.
        real_home_env = os.environ.get("HOME", "")
        if real_home_env and _same_path(root_path, Path(real_home_env) / ".hermes" / "auth.json"):
            return None
    profile_path = _auth_file_path()
    profile_home = profile_path.parent
    root_home = root_path.parent
    profile_singleton = profile_home / ".anthropic_oauth.json" if provider_id == "anthropic" else None

    # Hot-path short-circuit: load_pool() runs per model call. Once this
    # profile's store was verified clean for *provider_id*, skip the locked
    # read-modify-write until the profile's own files change (mtime key).
    def _stamp(p: Optional[Path]) -> Optional[int]:
        try:
            return p.stat().st_mtime_ns if p is not None else None
        except OSError:
            return None

    fingerprint = (str(profile_path), _stamp(profile_path), _stamp(profile_singleton))
    if _oauth_heal_clean_marks.get(provider_id) == fingerprint:
        return None
    if fingerprint[1] is None and fingerprint[2] is None:
        _oauth_heal_clean_marks[provider_id] = fingerprint
        return None

    summary: Dict[str, Any] = {"adopted": False, "stripped_ids": [], "files": [], "providers_block": False}
    log_bits: List[str] = []

    # Lock order: active (profile) store first, then the root source store —
    # the same order ``_provider_state_transaction`` uses.
    with _auth_store_lock():
        profile_store = _load_auth_store(profile_path) if profile_path.exists() else {"providers": {}}
        with _auth_store_lock(target_path=root_path):
            root_store = _load_auth_store(root_path) if root_path.exists() else {"providers": {}}
            profile_changed = False
            root_changed = False

            p_pool = profile_store.get("credential_pool")
            p_rows = p_pool.get(provider_id) if isinstance(p_pool, dict) else None
            p_rows = p_rows if isinstance(p_rows, list) else []
            r_pool = root_store.get("credential_pool")
            r_rows = r_pool.get(provider_id) if isinstance(r_pool, dict) else None
            r_rows = r_rows if isinstance(r_rows, list) else []
            r_oauth = [r for r in r_rows if _is_oauth_pool_payload(r)]

            root_singleton = root_home / ".anthropic_oauth.json" if provider_id == "anthropic" else None
            root_singleton_row = (
                _singleton_as_row(root_singleton)
                if root_singleton is not None and root_singleton.exists() else None
            )

            # ── credential_pool rows ────────────────────────────────────
            kept_rows: List[Any] = []
            for row in p_rows:
                if not _is_oauth_pool_payload(row):
                    kept_rows.append(row)  # API keys are safe to duplicate
                    continue
                match_idx = _find_root_counterpart(row, r_rows)
                if match_idx is not None:
                    root_row = r_rows[match_idx]
                    if _oauth_freshness(row) > _oauth_freshness(root_row):
                        r_rows[match_idx] = _adopt_oauth_material(root_row, row)
                        root_changed = True
                        summary["adopted"] = True
                    summary["stripped_ids"].append(row.get("id"))
                    profile_changed = True
                    continue
                # No root pool counterpart. Root's grant may live only in its
                # .anthropic_oauth.json (the ``hermes auth`` PKCE shape); a
                # profile hermes_pkce-family row is that grant's copy.
                is_pkce = str(row.get("source") or "").endswith("hermes_pkce")
                if is_pkce and root_singleton_row is not None and not r_oauth:
                    if _oauth_freshness(row) > _oauth_freshness(root_singleton_row):
                        root_singleton_row = _adopt_oauth_material(root_singleton_row, row)
                        summary["adopted"] = True
                    summary["stripped_ids"].append(row.get("id"))
                    profile_changed = True
                    continue
                # Root holds no copy of this lineage (independent account, or
                # root never had the grant): the profile's row may be the
                # only surviving copy — leave it alone.
                kept_rows.append(row)
            if profile_changed and isinstance(p_pool, dict):
                if kept_rows:
                    p_pool[provider_id] = kept_rows
                else:
                    p_pool.pop(provider_id, None)

            # ── providers.<id> device-code blocks (Codex / xAI) ─────────
            if provider_id in ("openai-codex", "xai-oauth"):
                block_result = _heal_forked_provider_block(profile_store, root_store, provider_id)
                if block_result is not None:
                    profile_changed = True
                    summary["providers_block"] = True
                    if block_result:
                        root_changed = True
                        summary["adopted"] = True

            # ── profile-local .anthropic_oauth.json singleton ───────────
            if profile_singleton is not None and profile_singleton.exists():
                p_single = _singleton_as_row(profile_singleton)
                root_has_grant = bool(r_oauth) or root_singleton_row is not None
                if p_single is not None and root_has_grant:
                    if root_singleton_row is not None:
                        if _oauth_freshness(p_single) > _oauth_freshness(root_singleton_row):
                            root_singleton_row = _adopt_oauth_material(root_singleton_row, p_single)
                            summary["adopted"] = True
                    else:
                        # Root only has pool rows: fold the singleton's pair
                        # into the freshest-matching root pkce row, if any.
                        idx = next(
                            (i for i, r in enumerate(r_rows)
                             if _is_oauth_pool_payload(r)
                             and str(r.get("source") or "").endswith("hermes_pkce")),
                            None,
                        )
                        if idx is not None and _oauth_freshness(p_single) > _oauth_freshness(r_rows[idx]):
                            r_rows[idx] = _adopt_oauth_material(r_rows[idx], p_single)
                            root_changed = True
                            summary["adopted"] = True
                    try:
                        profile_singleton.unlink()
                        summary["files"].append(profile_singleton.name)
                    except OSError:
                        logger.debug("could not remove %s", profile_singleton, exc_info=True)
                # Otherwise root has NO grant for this provider (or the file
                # is not a grant): the profile's singleton may be the only
                # surviving copy — never delete it.

            if not (profile_changed or root_changed or summary["adopted"]):
                _oauth_heal_clean_marks[provider_id] = fingerprint
                return None

            if summary["adopted"] and root_singleton is not None and root_singleton_row is not None:
                # Keep root's singleton and its ``hermes_pkce``-seeded pool row
                # in step: root's next load_pool() re-seeds that row FROM the
                # singleton file, so a stale file would resurrect the spent
                # pair (and a stale row would be overwritten by a fresh file).
                pkce_idx = next(
                    (i for i, r in enumerate(r_rows)
                     if _is_oauth_pool_payload(r) and r.get("source") == "hermes_pkce"),
                    None,
                )
                if pkce_idx is not None:
                    pkce_row = r_rows[pkce_idx]
                    if _oauth_freshness(pkce_row) > _oauth_freshness(root_singleton_row):
                        root_singleton_row = _adopt_oauth_material(root_singleton_row, pkce_row)
                    elif _oauth_freshness(root_singleton_row) > _oauth_freshness(pkce_row):
                        r_rows[pkce_idx] = _adopt_oauth_material(pkce_row, root_singleton_row)
                        root_changed = True

            if root_changed:
                if isinstance(r_pool, dict):
                    r_pool[provider_id] = r_rows
                else:
                    root_store["credential_pool"] = {provider_id: r_rows}
                _save_auth_store(root_store, target_path=root_path)
            if summary["adopted"] and root_singleton is not None and root_singleton_row is not None:
                from agent.anthropic_credentials import _write_hermes_oauth_credentials
                _write_hermes_oauth_credentials(
                    root_singleton_row.get("access_token") or "",
                    root_singleton_row.get("refresh_token"),
                    root_singleton_row.get("expires_at_ms"),
                    target=root_singleton,
                )
            if profile_changed and profile_path.exists():
                _save_auth_store(profile_store, target_path=profile_path)

    if summary["stripped_ids"]:
        log_bits.append(f"pool rows {summary['stripped_ids']}")
    if summary["providers_block"]:
        log_bits.append(f"providers.{provider_id} block")
    if summary["files"]:
        log_bits.append(", ".join(summary["files"]))
    verdict = (
        "profile copy was the live pair; root updated"
        if summary["adopted"] else "root copy already newest; profile copy dropped"
    )
    message = (
        f"profile {profile_home.name}: consolidated forked {provider_id} OAuth grant "
        f"({'; '.join(log_bits) or 'no-op'}) into the root grant — {verdict}; "
        f"this profile now borrows the root grant (#100339)"
    )
    logger.info(message)
    _oauth_heal_notices.append(message)
    return summary
