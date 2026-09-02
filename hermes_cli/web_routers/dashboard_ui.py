"""Dashboard theme/font and dashboard-plugin (discovery, hub, install/enable, asset serving) routes.

Extracted from ``hermes_cli.web_server``; helpers/state that tests monkeypatch on
``web_server`` stay there and are imported lazily at call time (cycle-safe).
"""

import logging
import asyncio
from fastapi import APIRouter
from fastapi import HTTPException, Request
from fastapi.responses import FileResponse
from hermes_cli.web_models import ThemeSetBody, FontSetBody, _AgentPluginInstallBody, _PluginProvidersPutBody, _PluginVisibilityBody
from pathlib import Path

_log = logging.getLogger("hermes_cli.web_server")
router = APIRouter()


@router.get("/api/dashboard/themes")
async def get_dashboard_themes():
    """Return available themes and the currently active one.

    Built-in entries ship name/label/description only (the frontend owns
    their full definitions in `web/src/themes/presets.ts`).  User themes
    from `~/.hermes/dashboard-themes/*.yaml` ship with their full
    normalised definition under `definition`, so the client can apply
    them without a stub.
    """
    from hermes_cli.web_server import (
        _BUILTIN_DASHBOARD_THEMES,
        _discover_user_themes,
        cfg_get,
        load_config,
    )
    def _run():
        config = load_config()
        active = cfg_get(config, "dashboard", "theme", default="default")
        user_themes = _discover_user_themes()
        seen = set()
        themes = []
        for t in _BUILTIN_DASHBOARD_THEMES:
            seen.add(t["name"])
            themes.append(t)
        for t in user_themes:
            if t["name"] in seen:
                continue
            themes.append({
                "name": t["name"],
                "label": t["label"],
                "description": t["description"],
                "definition": t,
            })
            seen.add(t["name"])
        return {"themes": themes, "active": active}

    return await asyncio.to_thread(_run)


@router.put("/api/dashboard/theme")
async def set_dashboard_theme(body: ThemeSetBody):
    """Set the active dashboard theme (persists to config.yaml)."""
    from hermes_cli.web_server import _CONFIG_MUTATION_LOCK, load_config, save_config
    def _run():
        with _CONFIG_MUTATION_LOCK:
            config = load_config()
            if "dashboard" not in config:
                config["dashboard"] = {}
            config["dashboard"]["theme"] = body.name
            save_config(config)
        return {"ok": True, "theme": body.name}

    return await asyncio.to_thread(_run)


# Curated font-override ids. Kept in sync with FONT_CHOICES in
# web/src/themes/fonts.ts — the frontend owns the stacks + webfont URLs;
# the backend only needs the id allow-list so it can reject anything not
# in the vetted catalog (the font's webfont URL is injected as a <link>,
# so we never accept an arbitrary user-supplied id/URL here).
_FONT_DEFAULT_ID = "theme"


_FONT_CHOICES = frozenset({
    "system-sans", "system-serif", "system-mono",
    "inter", "ibm-plex-sans", "work-sans", "atkinson-hyperlegible", "dm-sans",
    "spectral", "fraunces", "source-serif",
    "jetbrains-mono", "ibm-plex-mono", "space-mono",
})


@router.get("/api/dashboard/font")
async def get_dashboard_font():
    """Return the active font override (``"theme"`` = use the theme's font)."""
    from hermes_cli.web_server import cfg_get, load_config
    def _run():
        config = load_config()
        font = cfg_get(config, "dashboard", "font", default=_FONT_DEFAULT_ID)
        if font not in _FONT_CHOICES:
            font = _FONT_DEFAULT_ID
        return {"font": font}

    return await asyncio.to_thread(_run)


@router.put("/api/dashboard/font")
async def set_dashboard_font(body: FontSetBody):
    """Set the dashboard font override (persists to config.yaml).

    Accepts any id in the curated catalog, or ``"theme"`` to clear the
    override and fall back to the active theme's own font. Unknown ids are
    coerced to ``"theme"`` rather than 400'd so a stale client can't wedge
    the picker.
    """
    from hermes_cli.web_server import _CONFIG_MUTATION_LOCK, load_config, save_config
    font = body.font if body.font in _FONT_CHOICES else _FONT_DEFAULT_ID

    def _run():
        with _CONFIG_MUTATION_LOCK:
            config = load_config()
            if "dashboard" not in config:
                config["dashboard"] = {}
            config["dashboard"]["font"] = font
            save_config(config)
        return {"ok": True, "font": font}

    return await asyncio.to_thread(_run)


@router.get("/api/dashboard/plugins")
async def get_dashboard_plugins():
    """Return discovered dashboard plugins (excludes user-hidden and non-enabled ones)."""
    from hermes_cli.web_server import _get_dashboard_plugins, cfg_get, load_config
    def _run():
        plugins = _get_dashboard_plugins()
        # Read user's hidden plugins list from config.
        config = load_config()
        hidden: list = cfg_get(config, "dashboard", "hidden_plugins", default=[]) or []
        # Gate: only serve user plugins that are in plugins.enabled and not
        # in plugins.disabled.  This prevents the frontend from loading JS/CSS
        # from plugins the user has not explicitly activated.  (#46435)
        try:
            from hermes_cli.plugins_cmd import _get_enabled_set, _get_disabled_set
            enabled_set = _get_enabled_set()
            disabled_set = _get_disabled_set()
        except Exception:
            enabled_set = set()
            disabled_set = set()
        return plugins, hidden, enabled_set, disabled_set

    plugins, hidden, enabled_set, disabled_set = await asyncio.to_thread(_run)

    def _is_active(p: dict) -> bool:
        name = p.get("name", "")
        if name in hidden:
            return False
        if p.get("source") == "user":
            if name in disabled_set:
                return False
            if name not in enabled_set:
                return False
        elif p.get("source") == "bundled":
            if name in disabled_set:
                return False
        return True

    # Strip internal fields before sending to frontend.
    return [
        {k: v for k, v in p.items() if not k.startswith("_")}
        for p in plugins
        if _is_active(p)
    ]


@router.get("/api/dashboard/plugins/rescan")
async def rescan_dashboard_plugins():
    """Force re-scan of dashboard plugins."""
    from hermes_cli.web_server import _get_dashboard_plugins
    plugins = _get_dashboard_plugins(force_rescan=True)
    return {"ok": True, "count": len(plugins)}


@router.get("/api/dashboard/plugins/hub")
async def get_plugins_hub(request: Request):
    """Unified agent plugins + dashboard extension metadata (session protected)."""
    from hermes_cli.web_server import _merged_plugins_hub, _require_token
    _require_token(request)
    try:
        return _merged_plugins_hub()
    except Exception as exc:
        _log.warning("plugins/hub failed: %s", exc)
        raise HTTPException(status_code=500, detail="Failed to build plugins hub.") from exc


@router.post("/api/dashboard/agent-plugins/install")
async def post_agent_plugin_install(request: Request, body: _AgentPluginInstallBody):
    from hermes_cli.web_server import (
        _get_dashboard_plugins,
        _invalidate_plugins_hub_cache,
        _require_token,
    )
    _require_token(request)
    from hermes_cli.plugins_cmd import dashboard_install_plugin

    result = dashboard_install_plugin(
        body.identifier.strip(),
        force=body.force,
        enable=body.enable,
    )
    if not result.get("ok"):
        raise HTTPException(
            status_code=400,
            detail=result.get("error") or "Install failed.",
        )
    _get_dashboard_plugins(force_rescan=True)
    _invalidate_plugins_hub_cache()
    # Strip internal paths from the response
    result.pop("after_install_path", None)
    return result


def _validate_plugin_name(name: str) -> str:
    """Reject path-traversal attempts in plugin name URL parameters."""
    name = name.strip("/")
    if not name or ".." in name or "\\" in name:
        raise HTTPException(status_code=400, detail="Invalid plugin name.")
    return name


@router.post("/api/dashboard/agent-plugins/{name:path}/enable")
async def post_agent_plugin_enable(request: Request, name: str):
    from hermes_cli.web_server import _invalidate_plugins_hub_cache, _require_token
    _require_token(request)
    name = _validate_plugin_name(name)
    from hermes_cli.plugins_cmd import dashboard_set_agent_plugin_enabled

    result = dashboard_set_agent_plugin_enabled(name, enabled=True)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error") or "Enable failed.")
    _invalidate_plugins_hub_cache()
    return result


@router.post("/api/dashboard/agent-plugins/{name:path}/disable")
async def post_agent_plugin_disable(request: Request, name: str):
    from hermes_cli.web_server import _invalidate_plugins_hub_cache, _require_token
    _require_token(request)
    name = _validate_plugin_name(name)
    from hermes_cli.plugins_cmd import dashboard_set_agent_plugin_enabled

    result = dashboard_set_agent_plugin_enabled(name, enabled=False)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error") or "Disable failed.")
    _invalidate_plugins_hub_cache()
    return result


@router.post("/api/dashboard/agent-plugins/{name:path}/update")
async def post_agent_plugin_update(request: Request, name: str):
    from hermes_cli.web_server import (
        _get_dashboard_plugins,
        _invalidate_plugins_hub_cache,
        _require_token,
    )
    _require_token(request)
    name = _validate_plugin_name(name)
    from hermes_cli.plugins_cmd import dashboard_update_user_plugin

    result = dashboard_update_user_plugin(name)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error") or "Update failed.")
    _get_dashboard_plugins(force_rescan=True)
    _invalidate_plugins_hub_cache()
    return result


@router.delete("/api/dashboard/agent-plugins/{name:path}")
async def delete_agent_plugin(request: Request, name: str):
    from hermes_cli.web_server import (
        _get_dashboard_plugins,
        _invalidate_plugins_hub_cache,
        _require_token,
    )
    _require_token(request)
    name = _validate_plugin_name(name)
    from hermes_cli.plugins_cmd import dashboard_remove_user_plugin

    result = dashboard_remove_user_plugin(name)
    if not result.get("ok"):
        raise HTTPException(status_code=400, detail=result.get("error") or "Remove failed.")
    _get_dashboard_plugins(force_rescan=True)
    _invalidate_plugins_hub_cache()
    return result


@router.put("/api/dashboard/plugin-providers")
async def put_plugin_providers(request: Request, body: _PluginProvidersPutBody):
    """Persist memory provider / context engine selection (writes config.yaml)."""
    from hermes_cli.web_server import (
        _CONFIG_MUTATION_LOCK,
        _invalidate_plugins_hub_cache,
        _normalize_memory_provider_name,
        _require_memory_provider_ready,
        _require_token,
    )
    _require_token(request)
    from hermes_cli.plugins_cmd import (
        _save_context_engine,
        _save_memory_provider,
    )

    def _run():
        with _CONFIG_MUTATION_LOCK:
            if body.memory_provider is not None:
                memory_provider = _normalize_memory_provider_name(body.memory_provider)
                _require_memory_provider_ready(memory_provider)
                _save_memory_provider(memory_provider)
            if body.context_engine is not None:
                _save_context_engine(body.context_engine)
        _invalidate_plugins_hub_cache()
        return {"ok": True}

    return await asyncio.to_thread(_run)


@router.post("/api/dashboard/plugins/{name:path}/visibility")
async def post_plugin_visibility(request: Request, name: str, body: _PluginVisibilityBody):
    """Toggle a plugin's sidebar visibility (persists to config.yaml dashboard.hidden_plugins)."""
    from hermes_cli.web_server import (
        _CONFIG_MUTATION_LOCK,
        _invalidate_plugins_hub_cache,
        _require_token,
        load_config,
        save_config,
    )
    _require_token(request)
    name = _validate_plugin_name(name)

    def _run():
        with _CONFIG_MUTATION_LOCK:
            config = load_config()
            if "dashboard" not in config or not isinstance(config.get("dashboard"), dict):
                config["dashboard"] = {}
            hidden_list: list = config["dashboard"].get("hidden_plugins") or []
            if not isinstance(hidden_list, list):
                hidden_list = []

            if body.hidden and name not in hidden_list:
                hidden_list.append(name)
            elif not body.hidden and name in hidden_list:
                hidden_list.remove(name)

            config["dashboard"]["hidden_plugins"] = hidden_list
            save_config(config)
        _invalidate_plugins_hub_cache()
        return {"ok": True, "name": name, "hidden": body.hidden}

    return await asyncio.to_thread(_run)


@router.get("/dashboard-plugins/{plugin_name}/{file_path:path}")
async def serve_plugin_asset(plugin_name: str, file_path: str):
    """Serve static assets from a dashboard plugin directory.

    Only serves files from the plugin's ``dashboard/`` subdirectory.
    Path traversal is blocked by checking ``resolve().is_relative_to()``.

    Restricted to a browser-fetchable suffix allowlist (JS/CSS/JSON/HTML/
    SVG/PNG/JPG/WOFF). The dashboard loads plugin JS via ``<script src>``
    and CSS via ``<link href>``, neither of which can attach a custom
    auth header — so this route stays unauthenticated to keep the SPA
    working. But user-installed plugins ship a ``plugin_api.py``
    backend module that the browser never fetches; it's only imported
    by :func:`_mount_plugin_api_routes` at startup. Without a suffix
    allowlist, anyone on the loopback port can curl the ``.py`` source
    of a private third-party plugin. Reject everything outside the
    browser-asset set.

    User plugins must be in plugins.enabled before their assets are
    served. (#46435, GHSA-mcfc-hp25-cjv7)
    """
    from hermes_cli.web_server import _get_dashboard_plugins
    plugins = _get_dashboard_plugins()
    plugin = next((p for p in plugins if p["name"] == plugin_name), None)
    if not plugin:
        raise HTTPException(status_code=404, detail="Plugin not found")

    # Gate: user plugins must be enabled to serve assets;
    # bundled plugins must not be explicitly disabled.
    try:
        from hermes_cli.plugins_cmd import _get_enabled_set, _get_disabled_set
        enabled_set = _get_enabled_set()
        disabled_set = _get_disabled_set()
    except Exception:
        enabled_set = set()
        disabled_set = set()
    if plugin.get("source") == "user":
        if plugin_name in disabled_set or plugin_name not in enabled_set:
            raise HTTPException(status_code=404, detail="Plugin not found")
    elif plugin.get("source") == "bundled":
        if plugin_name in disabled_set:
            raise HTTPException(status_code=404, detail="Plugin not found")

    base = Path(plugin["_dir"])
    target = (base / file_path).resolve()

    if not target.is_relative_to(base.resolve()):
        raise HTTPException(status_code=403, detail="Path traversal blocked")
    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    # Browser-asset suffix allowlist. Everything outside this set is
    # rejected with 404 so we don't leak ``.py`` backend sources, README
    # files, ``.env.example`` templates, etc. — none of which the SPA
    # actually fetches. Add to this set deliberately when a new asset
    # type comes up; do NOT change the default fallback.
    suffix = target.suffix.lower()
    content_types = {
        ".js": "application/javascript",
        ".mjs": "application/javascript",
        ".css": "text/css",
        ".json": "application/json",
        ".html": "text/html",
        ".svg": "image/svg+xml",
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".gif": "image/gif",
        ".webp": "image/webp",
        ".ico": "image/x-icon",
        ".woff2": "font/woff2",
        ".woff": "font/woff",
        ".ttf": "font/ttf",
        ".otf": "font/otf",
        ".map": "application/json",
    }
    if suffix not in content_types:
        raise HTTPException(
            status_code=404,
            detail="File not found",
        )
    media_type = content_types[suffix]
    return FileResponse(
        target,
        media_type=media_type,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )
