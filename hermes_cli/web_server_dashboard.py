"""Dashboard UI assets: SPA mount, theme normalisation/bootstrap CSS, dashboard-plugin discovery and the plugins-hub merge.

Split out of ``hermes_cli.web_server``; every externally used name is re-imported
there, so ``web_server.<name>`` keeps resolving (and monkeypatching) as before.
Helpers that tests patch on ``web_server`` are reached lazily through it.
"""

import logging
import importlib.util
import json
import os
import sys
import threading
import time
import yaml
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from typing import Any, Dict, List, Optional
from hermes_cli.config import cfg_get, get_process_hermes_home
from utils import env_var_enabled

# Same logger the code used before extraction (record parity).
_log = logging.getLogger("hermes_cli.web_server")


def _normalise_prefix(raw: Optional[str]) -> str:
    """Normalise an X-Forwarded-Prefix header value.

    Thin re-export of :func:`hermes_cli.dashboard_auth.prefix.normalise_prefix`
    — the single source of truth lives in the dashboard_auth package so
    the gate middleware, the OAuth routes, the cookie helpers, and the
    SPA mount all agree on validation rules.
    """
    from hermes_cli.dashboard_auth.prefix import normalise_prefix
    return normalise_prefix(raw)


def _render_active_theme_bootstrap_css() -> str:
    """Critical-CSS shim for the active user theme.

    Returns a ``<style>`` block with the ``:root`` CSS variables that
    ``ThemeProvider.applyTheme()`` installs once the
    ``/api/dashboard/themes`` round-trip completes.  The goal is to
    eliminate the green flash where the first paint shows the bundle's
    default Hermes Teal canvas before the SPA flips the configured user
    theme into place.

    Built-in themes return an empty string — their full definitions live
    in ``web/src/themes/presets.ts`` and are applied by the bundle
    before paint, so no shim is needed for them.
    """
    from hermes_cli.web_server import load_config
    try:
        config = load_config()
        active = cfg_get(config, "dashboard", "theme", default="default")
        if not active or not isinstance(active, str):
            return ""
        # Built-in: the bundle already owns the definition, no flash.
        if any(b["name"] == active for b in _BUILTIN_DASHBOARD_THEMES):
            return ""
        for theme in _discover_user_themes():
            if theme.get("name") != active:
                continue
            palette = theme.get("palette") or {}
            bg = palette.get("background") or {}
            mg = palette.get("midground") or {}
            bg_hex = bg.get("hex", "#0a0a0a") if isinstance(bg, dict) else "#0a0a0a"
            mg_hex = mg.get("hex", "#e5e5e5") if isinstance(mg, dict) else "#e5e5e5"
            typo = theme.get("typography") or {}
            font_sans = typo.get("fontSans") or _THEME_DEFAULT_TYPOGRAPHY["fontSans"]
            base_size = typo.get("baseSize") or _THEME_DEFAULT_TYPOGRAPHY["baseSize"]
            # Defensive ``</style>`` escape — current values are well-known
            # hex/font strings, but this keeps the helper safe if it is
            # later extended to ship user-authored CSS literals.
            def _esc(s: str) -> str:
                return str(s).replace("</", "<\\/")
            # Variable names MUST match what the bundle actually consumes:
            #   - ``--background-base`` / ``--midground-base`` come from
            #     ``layerVars()`` in ``web/src/themes/context.tsx``.
            #   - ``--theme-font-sans`` / ``--theme-base-size`` come from
            #     ``typographyVars()`` there, and ``index.css`` applies them
            #     via ``html{font-family:var(--theme-font-sans);
            #     font-size:var(--theme-base-size)}``.
            # The ``html,body`` canvas rule references the SAME variables
            # instead of literal values so runtime theme switches stay
            # live: ``applyTheme()`` writes these vars as inline styles on
            # ``documentElement``, which outrank this stylesheet block in
            # the cascade — the rule below re-resolves automatically and
            # never goes stale when the user picks a different theme.
            return (
                '<style id="hermes-theme-bootstrap">'
                ":root{"
                f"--background-base:{_esc(bg_hex)};"
                f"--midground-base:{_esc(mg_hex)};"
                f"--theme-font-sans:{_esc(font_sans)};"
                f"--theme-base-size:{_esc(base_size)};"
                "}"
                "html,body{background-color:var(--background-base);"
                "color:var(--midground-base);"
                "font-family:var(--theme-font-sans);"
                "font-size:var(--theme-base-size);}"
                "</style>"
            )
        return ""
    except Exception:
        _log.debug("theme bootstrap render failed", exc_info=True)
        return ""


# Hashed bundle assets (``/assets/<name>-<contenthash>.<ext>``) are immutable
# by construction: any content change produces a new filename, and the entry
# point (index.html) is served ``no-store`` so it always references the
# current hashes. A year-long immutable cache lets browsers skip even the
# revalidation round-trip on every dashboard load.
_IMMUTABLE_ASSET_CACHE_CONTROL = "public, max-age=31536000, immutable"


def mount_spa(application: FastAPI):
    """Mount the built SPA. Falls back to index.html for client-side routing.

    The session token is injected into index.html via a ``<script>`` tag so
    the SPA can authenticate against protected API endpoints without a
    separate (unauthenticated) token-dispensing endpoint.

    When served behind a path-prefix reverse proxy (e.g.
    ``mission-control.tilos.com/hermes/*`` -> local Caddy -> :9119), the
    proxy injects ``X-Forwarded-Prefix: /hermes`` on every request. We
    rewrite the served ``index.html`` so absolute asset URLs (``/assets/...``)
    and the SPA's runtime ``__HERMES_BASE_PATH__`` honour that prefix
    without rebuilding the bundle.
    """
    from hermes_cli.web_server import (
        WEB_DIST,
        _DASHBOARD_EMBEDDED_CHAT_ENABLED,
        _SESSION_TOKEN,
        app,
    )
    # `hermes serve` is the headless backend: it must NEVER serve the browser
    # SPA, even if a dist is lying around from a prior `dashboard`/build. Take
    # the no-frontend path so only the JSON-RPC/WS/API surface is reachable.
    _headless = os.environ.get("HERMES_SERVE_HEADLESS") == "1"
    if _headless:
        _msg = (
            "Headless backend (hermes serve): web UI disabled — use "
            "`hermes dashboard` for the browser UI."
        )

        @application.get("/{full_path:path}")
        async def no_frontend(full_path: str):
            # Desktop token handshake (#94227): the Electron shell boots by
            # fetching `/` and extracting ``window.__HERMES_SESSION_TOKEN__``
            # for /api/ws auth (apps/desktop/electron/dashboard-token.ts).
            # When headless serve 404'd every path, a renderer whose spawn
            # token no longer matched the backend's live token (e.g. after
            # `hermes update` replaced the backend) had no way to adopt the
            # served token — the WS handshake failed and the window
            # white-screened (#95575). Serve a minimal token-only page at the
            # exact root, but ONLY when the dashboard auth gate is off: on a
            # gated (non-loopback/remote) serve the token must never be
            # readable without auth, so the 404 JSON stays.
            gated = bool(getattr(application.state, "auth_required", False))
            if full_path == "" and not gated:
                token_js = json.dumps(_SESSION_TOKEN)
                return HTMLResponse(
                    "<!doctype html><html><head><script>"
                    f"window.__HERMES_SESSION_TOKEN__={token_js};"
                    "window.__HERMES_AUTH_REQUIRED__=false;"
                    "</script></head><body>"
                    "Headless backend (hermes serve): web UI disabled — use "
                    "`hermes dashboard` for the browser UI."
                    "</body></html>",
                    headers={
                        "Cache-Control": "no-store, no-cache, must-revalidate"
                    },
                )
            return JSONResponse({"error": _msg}, status_code=404)
        return

    # A missing WEB_DIST is deliberately NOT a mount-time terminal state
    # (#82614): a long-lived `hermes dashboard --skip-build` process that
    # survives a `git pull` (or starts before the first build) used to
    # install a permanent no_frontend catch-all here and could never
    # recover — every route answered 404 "Frontend not built" until the
    # process was restarted, even after `npm run build` completed. The SPA
    # routes below all cope with a missing dist per-request (`_serve_index`
    # returns the same 404 JSON when index.html is unreadable; the asset
    # mounts use check_dir=False and 404 on missing files), so mounting
    # them unconditionally makes the dashboard recover the moment a build
    # appears on disk — no restart needed.

    _index_path = WEB_DIST / "index.html"

    def _serve_index(prefix: str = ""):
        """Return index.html with the session token + base-path injected.

        ``prefix`` is the normalised ``X-Forwarded-Prefix`` (e.g. ``/hermes``)
        or empty string when served at root.

        When the OAuth auth gate is active (``app.state.auth_required``),
        the legacy ``_SESSION_TOKEN`` is NOT injected — the SPA reads
        identity from ``/api/auth/me`` over cookie auth instead.  The
        ``__HERMES_AUTH_REQUIRED__`` flag lets the SPA pick the right
        auth scheme for /api/pty and /api/ws (ticket vs token).
        """
        try:
            html = _index_path.read_text(encoding="utf-8")
        except OSError:
            # The dist dir existed at mount time but index.html is missing or
            # unreadable now (partial build, wiped dist, permissions). Without
            # this guard every request raises FileNotFoundError (500). Return
            # the same JSON 404 payload mount_spa uses for a fully-missing
            # dist so clients get a clear, consistent signal.
            return JSONResponse(
                {"error": "Frontend not built. Run: cd web && npm run build"},
                status_code=404,
            )
        chat_js = "true" if _DASHBOARD_EMBEDDED_CHAT_ENABLED else "false"
        gated = bool(getattr(app.state, "auth_required", False))
        gated_js = "true" if gated else "false"
        if gated:
            bootstrap_script = (
                f"<script>"
                f"window.__HERMES_DASHBOARD_EMBEDDED_CHAT__={chat_js};"
                f'window.__HERMES_BASE_PATH__="{prefix}";'
                f"window.__HERMES_AUTH_REQUIRED__={gated_js};"
                f"</script>"
            )
        else:
            bootstrap_script = (
                f'<script>window.__HERMES_SESSION_TOKEN__="{_SESSION_TOKEN}";'
                f"window.__HERMES_DASHBOARD_EMBEDDED_CHAT__={chat_js};"
                f'window.__HERMES_BASE_PATH__="{prefix}";'
                f"window.__HERMES_AUTH_REQUIRED__={gated_js};"
                f"</script>"
            )
        if prefix:
            # Rewrite absolute asset URLs baked into the Vite build so the
            # browser fetches them through the same proxy prefix.
            html = html.replace('href="/assets/', f'href="{prefix}/assets/')
            html = html.replace('src="/assets/', f'src="{prefix}/assets/')
            html = html.replace('href="/favicon.ico"', f'href="{prefix}/favicon.ico"')
            html = html.replace('href="/fonts/', f'href="{prefix}/fonts/')
            html = html.replace('href="/ds-assets/', f'href="{prefix}/ds-assets/')
            html = html.replace('src="/ds-assets/', f'src="{prefix}/ds-assets/')
        # Theme flash mitigation: when the active theme is a user theme
        # (``HERMES_HOME/dashboard-themes/<name>.yaml``), inject a minimal
        # critical-CSS block so the first paint uses the target palette.
        # Without this the SPA paints the default Hermes Teal canvas, then
        # ``ThemeProvider`` flips the CSS variables once
        # ``/api/dashboard/themes`` resolves.  Built-in themes are already
        # in the bundle's ``presets.ts`` so no shim is needed for them.
        theme_bootstrap = _render_active_theme_bootstrap_css()
        if theme_bootstrap:
            html = html.replace("</head>", f"{theme_bootstrap}</head>", 1)
        html = html.replace("</head>", f"{bootstrap_script}</head>", 1)
        return HTMLResponse(
            html,
            headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
        )

    # When served behind a path-prefix proxy, the built CSS contains
    # absolute ``url(/fonts/...)`` and ``url(/ds-assets/...)`` references.
    # Browsers resolve those against the document origin, which means
    # under ``/hermes`` they'd hit ``mission-control.tilos.com/fonts/...``
    # (the MC Pages app), not the Hermes backend. Intercept CSS asset
    # requests BEFORE the StaticFiles mount and rewrite the absolute paths
    # when a prefix is in play.
    @application.get("/assets/{filename}.css")
    async def serve_css(filename: str, request: Request):
        css_path = WEB_DIST / "assets" / f"{filename}.css"
        if not css_path.is_file() or not css_path.resolve().is_relative_to(
            WEB_DIST.resolve()
        ):
            return JSONResponse({"error": "not found"}, status_code=404)
        prefix = _normalise_prefix(request.headers.get("x-forwarded-prefix"))
        css = css_path.read_text(encoding="utf-8")
        if prefix:
            for asset_dir in ("/fonts/", "/fonts-terminal/", "/ds-assets/", "/assets/"):
                css = css.replace(f"url({asset_dir}", f"url({prefix}{asset_dir}")
                css = css.replace(f"url(\"{asset_dir}", f"url(\"{prefix}{asset_dir}")
                css = css.replace(f"url('{asset_dir}", f"url('{prefix}{asset_dir}")
        return Response(
            content=css,
            media_type="text/css",
            headers={"Cache-Control": _IMMUTABLE_ASSET_CACHE_CONTROL},
        )

    class _ImmutableAssetFiles(StaticFiles):
        """StaticFiles that marks hashed bundle assets immutable.

        Everything under ``/assets/`` carries a Vite content hash in its
        filename, so a given URL's bytes can never change — a rebuild
        produces a NEW filename referenced by a fresh (``no-store``)
        index.html. Without this header every dashboard load re-validated
        each chunk; with it the browser serves reloads straight from its
        HTTP cache.
        """

        async def get_response(self, path: str, scope):
            response = await super().get_response(path, scope)
            if response.status_code == 200:
                response.headers["Cache-Control"] = _IMMUTABLE_ASSET_CACHE_CONTROL
            return response

    application.mount(
        "/assets",
        # check_dir=False: the dist (and its assets/ dir) may not exist yet —
        # the whole point of the dynamic recheck (#82614). StaticFiles then
        # 404s per-request until a build appears instead of raising at mount.
        _ImmutableAssetFiles(directory=WEB_DIST / "assets", check_dir=False),
        name="assets",
    )

    @application.get("/{full_path:path}")
    async def serve_spa(full_path: str, request: Request):
        prefix = _normalise_prefix(request.headers.get("x-forwarded-prefix"))
        # An unmatched /api/* path is a missing/renamed endpoint, NOT a
        # client-side route. Falling through to index.html here returns
        # `<!doctype html>` with status 200, which makes JSON clients (the
        # desktop app's fetchJson, dashboard fetch wrappers) blow up with an
        # opaque `SyntaxError: Unexpected token '<'`. Return a real 404 JSON
        # so the caller sees a clear "no such endpoint" instead.
        if full_path == "api" or full_path.startswith("api/"):
            return JSONResponse(
                {"detail": f"No such API endpoint: /{full_path}"},
                status_code=404,
            )
        file_path = WEB_DIST / full_path
        # Prevent path traversal via url-encoded sequences (%2e%2e/)
        if (
            full_path
            and file_path.resolve().is_relative_to(WEB_DIST.resolve())
            and file_path.exists()
            and file_path.is_file()
        ):
            return FileResponse(file_path)
        return _serve_index(prefix)


# ---------------------------------------------------------------------------
# Dashboard theme endpoints
# ---------------------------------------------------------------------------

# Built-in dashboard themes — label + description only.  The actual color
# definitions live in the frontend (web/src/themes/presets.ts).
_BUILTIN_DASHBOARD_THEMES = [
    {"name": "default",       "label": "Hermes Teal",         "description": "Classic dark teal — the canonical Hermes look"},
    {"name": "default-large", "label": "Hermes Teal (Large)", "description": "Hermes Teal with bigger fonts and roomier spacing"},
    {"name": "nous-blue",     "label": "Nous Blue",           "description": "Light mode — vivid Nous-blue accents on cream canvas"},
    {"name": "midnight",      "label": "Midnight",            "description": "Deep blue-violet with cool accents"},
    {"name": "ember",     "label": "Ember",          "description": "Warm crimson and bronze — forge vibes"},
    {"name": "mono",      "label": "Mono",           "description": "Clean grayscale — minimal and focused"},
    {"name": "cyberpunk", "label": "Cyberpunk",      "description": "Neon green on black — matrix terminal"},
    {"name": "rose",      "label": "Rosé",           "description": "Soft pink and warm ivory — easy on the eyes"},
]


def _parse_theme_layer(value: Any, default_hex: str, default_alpha: float = 1.0) -> Optional[Dict[str, Any]]:
    """Normalise a theme layer spec from YAML into `{hex, alpha}` form.

    Accepts shorthand (a bare hex string) or full dict form.  Returns
    ``None`` on garbage input so the caller can fall back to a built-in
    default rather than blowing up.
    """
    if value is None:
        return {"hex": default_hex, "alpha": default_alpha}
    if isinstance(value, str):
        return {"hex": value, "alpha": default_alpha}
    if isinstance(value, dict):
        hex_val = value.get("hex", default_hex)
        alpha_val = value.get("alpha", default_alpha)
        if not isinstance(hex_val, str):
            return None
        try:
            alpha_f = float(alpha_val)
        except (TypeError, ValueError):
            alpha_f = default_alpha
        return {"hex": hex_val, "alpha": max(0.0, min(1.0, alpha_f))}
    return None


_THEME_DEFAULT_TYPOGRAPHY: Dict[str, str] = {
    "fontSans": 'system-ui, -apple-system, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif',
    "fontMono": 'ui-monospace, "SF Mono", "Cascadia Mono", Menlo, Consolas, monospace',
    "baseSize": "15px",
    "lineHeight": "1.55",
    "letterSpacing": "0",
}
_THEME_DEFAULT_LAYOUT: Dict[str, str] = {
    "radius": "0.5rem",
    "density": "comfortable",
}
_THEME_OVERRIDE_KEYS = {
    "card", "cardForeground", "popover", "popoverForeground",
    "primary", "primaryForeground", "secondary", "secondaryForeground",
    "muted", "mutedForeground", "accent", "accentForeground",
    "destructive", "destructiveForeground", "success", "warning",
    "border", "input", "ring",
}

# Well-known named asset slots themes can populate.  Any other keys under
# ``assets.custom`` are exposed as ``--theme-asset-custom-<key>`` CSS vars
# for plugin/shell use.
_THEME_NAMED_ASSET_KEYS = {"bg", "hero", "logo", "crest", "sidebar", "header"}

# Component-style buckets themes can override.  The value under each bucket
# is a mapping from camelCase property name to CSS string; each pair emits
# ``--component-<bucket>-<kebab-property>`` on :root.  The frontend's shell
# components (Card, App header, Backdrop, etc.) consume these vars so themes
# can restyle chrome (clip-path, border-image, segmented progress, etc.)
# without shipping their own CSS.
_THEME_COMPONENT_BUCKETS = {
    "card", "header", "footer", "sidebar", "tab",
    "progress", "badge", "backdrop", "page",
}
_THEME_LAYOUT_VARIANTS = {"standard", "cockpit", "tiled"}

# Cap on customCSS length so a malformed/oversized theme YAML can't blow up
# the response payload or the <style> tag.  32 KiB is plenty for every
# practical reskin (the Strike Freedom demo is ~2 KiB).
_THEME_CUSTOM_CSS_MAX = 32 * 1024


def _normalise_theme_definition(data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Normalise a user theme YAML into the wire format `ThemeProvider`
    expects.  Returns ``None`` if the theme is unusable.

    Accepts both the full schema (palette/typography/layout) and a loose
    form with bare hex strings, so hand-written YAMLs stay friendly.
    """
    if not isinstance(data, dict):
        return None
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        return None

    # Palette
    palette_src = data.get("palette", {}) if isinstance(data.get("palette"), dict) else {}
    # Allow top-level `colors.background` as a shorthand too.
    colors_src = data.get("colors", {}) if isinstance(data.get("colors"), dict) else {}

    def _layer(key: str, default_hex: str, default_alpha: float = 1.0) -> Dict[str, Any]:
        spec = palette_src.get(key, colors_src.get(key))
        parsed = _parse_theme_layer(spec, default_hex, default_alpha)
        return parsed if parsed is not None else {"hex": default_hex, "alpha": default_alpha}

    palette = {
        "background": _layer("background", "#041c1c", 1.0),
        "midground": _layer("midground", "#ffe6cb", 1.0),
        "foreground": _layer("foreground", "#ffffff", 0.0),
        "warmGlow": palette_src.get("warmGlow") or data.get("warmGlow") or "rgba(255, 189, 56, 0.35)",
        "noiseOpacity": 1.0,
    }
    raw_noise = palette_src.get("noiseOpacity", data.get("noiseOpacity"))
    try:
        palette["noiseOpacity"] = float(raw_noise) if raw_noise is not None else 1.0
    except (TypeError, ValueError):
        palette["noiseOpacity"] = 1.0

    # Typography
    typo_src = data.get("typography", {}) if isinstance(data.get("typography"), dict) else {}
    typography = dict(_THEME_DEFAULT_TYPOGRAPHY)
    for key in ("fontSans", "fontMono", "fontDisplay", "fontUrl", "baseSize", "lineHeight", "letterSpacing"):
        val = typo_src.get(key)
        if isinstance(val, str) and val.strip():
            typography[key] = val

    # Layout
    layout_src = data.get("layout", {}) if isinstance(data.get("layout"), dict) else {}
    layout = dict(_THEME_DEFAULT_LAYOUT)
    radius = layout_src.get("radius")
    if isinstance(radius, str) and radius.strip():
        layout["radius"] = radius
    density = layout_src.get("density")
    if isinstance(density, str) and density in {"compact", "comfortable", "spacious"}:
        layout["density"] = density

    # Color overrides — keep only valid keys with string values.
    overrides_src = data.get("colorOverrides", {})
    color_overrides: Dict[str, str] = {}
    if isinstance(overrides_src, dict):
        for key, val in overrides_src.items():
            if key in _THEME_OVERRIDE_KEYS and isinstance(val, str) and val.strip():
                color_overrides[key] = val

    # Assets — named slots + arbitrary user-defined keys.  Values must be
    # strings (URLs or CSS ``url(...)``/``linear-gradient(...)`` expressions).
    # We don't fetch remote assets here; the frontend just injects them as
    # CSS vars.  Empty values are dropped so a theme can explicitly clear a
    # slot by setting ``hero: ""``.
    assets_out: Dict[str, Any] = {}
    assets_src = data.get("assets", {}) if isinstance(data.get("assets"), dict) else {}
    for key in _THEME_NAMED_ASSET_KEYS:
        val = assets_src.get(key)
        if isinstance(val, str) and val.strip():
            assets_out[key] = val
    custom_assets_src = assets_src.get("custom")
    if isinstance(custom_assets_src, dict):
        custom_assets: Dict[str, str] = {}
        for key, val in custom_assets_src.items():
            if (
                isinstance(key, str)
                and key.replace("-", "").replace("_", "").isalnum()
                and isinstance(val, str)
                and val.strip()
            ):
                custom_assets[key] = val
        if custom_assets:
            assets_out["custom"] = custom_assets

    # Custom CSS — raw CSS text the frontend injects as a scoped <style>
    # tag on theme apply.  Clipped to _THEME_CUSTOM_CSS_MAX to keep the
    # payload bounded.  We intentionally do NOT parse/sanitise the CSS
    # here — the dashboard is localhost-only and themes are user-authored
    # YAML in ~/.hermes/, same trust level as the config file itself.
    custom_css_val = data.get("customCSS")
    custom_css: Optional[str] = None
    if isinstance(custom_css_val, str) and custom_css_val.strip():
        custom_css = custom_css_val[:_THEME_CUSTOM_CSS_MAX]

    # Component style overrides — per-bucket dicts of camelCase CSS
    # property -> CSS string.  The frontend converts these into CSS vars
    # that shell components (Card, App header, Backdrop) consume.
    component_styles_src = data.get("componentStyles", {})
    component_styles: Dict[str, Dict[str, str]] = {}
    if isinstance(component_styles_src, dict):
        for bucket, props in component_styles_src.items():
            if bucket not in _THEME_COMPONENT_BUCKETS or not isinstance(props, dict):
                continue
            clean: Dict[str, str] = {}
            for prop, value in props.items():
                if (
                    isinstance(prop, str)
                    and prop.replace("-", "").replace("_", "").isalnum()
                    and isinstance(value, (str, int, float))
                    and str(value).strip()
                ):
                    clean[prop] = str(value)
            if clean:
                component_styles[bucket] = clean

    layout_variant_src = data.get("layoutVariant")
    layout_variant = (
        layout_variant_src
        if isinstance(layout_variant_src, str) and layout_variant_src in _THEME_LAYOUT_VARIANTS
        else "standard"
    )

    result: Dict[str, Any] = {
        "name": name,
        "label": data.get("label") or name,
        "description": data.get("description", ""),
        "palette": palette,
        "typography": typography,
        "layout": layout,
        "layoutVariant": layout_variant,
    }
    if color_overrides:
        result["colorOverrides"] = color_overrides
    if assets_out:
        result["assets"] = assets_out
    if custom_css is not None:
        result["customCSS"] = custom_css
    if component_styles:
        result["componentStyles"] = component_styles
    return result


def _discover_user_themes() -> list:
    """Scan ~/.hermes/dashboard-themes/*.yaml for user-created themes.

    Returns a list of fully-normalised theme definitions ready to ship
    to the frontend, so the client can apply them without a secondary
    round-trip or a built-in stub.

    Uses the dashboard process launch home, not ``get_hermes_home()``, so a
    transient profile override from embedded chat does not hide themes that
    live under the server's own ``HERMES_HOME``.
    """
    themes_dir = get_process_hermes_home() / "dashboard-themes"
    if not themes_dir.is_dir():
        return []
    result = []
    for f in sorted(themes_dir.glob("*.yaml")):
        try:
            data = yaml.safe_load(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        normalised = _normalise_theme_definition(data)
        if normalised is not None:
            result.append(normalised)
    return result


# ---------------------------------------------------------------------------
# Dashboard plugin system
# ---------------------------------------------------------------------------

def _safe_plugin_api_relpath(api_field: Any, *, dashboard_dir: Path) -> Optional[str]:
    """Validate the manifest's ``api`` field for the plugin loader.

    The web server later imports this file as a Python module via
    ``importlib.util.spec_from_file_location`` (arbitrary code
    execution by design — that's how plugins extend the backend).
    Pre-#29156 the field was used as-is, which meant:

    * An absolute path swallowed the plugin's dashboard directory
      entirely — ``Path('safe/dashboard') / '/tmp/evil.py'`` resolves
      to ``/tmp/evil.py``, so any attacker-controlled manifest could
      point the import at any Python file on disk (GHSA-5qr3-c538-wm9j).
    * A ``../..`` traversal could climb out of the plugin into
      neighbouring directories on the search path.

    Return the original string when the resolved path stays under
    ``dashboard_dir``; return ``None`` (with a warning logged at the
    call site) otherwise so the plugin still loads its static JS/CSS
    but its backend ``api`` is rejected.
    """
    if not isinstance(api_field, str) or not api_field.strip():
        return None
    candidate = Path(api_field)
    if candidate.is_absolute():
        return None
    try:
        resolved = (dashboard_dir / candidate).resolve()
        base = dashboard_dir.resolve()
    except (OSError, RuntimeError):
        return None
    try:
        resolved.relative_to(base)
    except ValueError:
        return None
    return api_field


def _discover_dashboard_plugins() -> list:
    """Scan plugins/*/dashboard/manifest.json for dashboard extensions.

    Checks three plugin sources (same as hermes_cli.plugins):
    1. User plugins:    ~/.hermes/plugins/<name>/dashboard/manifest.json
    2. Bundled plugins: <repo>/plugins/<name>/dashboard/manifest.json  (memory/, etc.)
    3. Project plugins: ./.hermes/plugins/  (only if HERMES_ENABLE_PROJECT_PLUGINS)
    """
    plugins = []
    seen_names: set = set()

    from hermes_cli.plugins import get_bundled_plugins_dir
    bundled_root = get_bundled_plugins_dir()
    # User dashboard plugins are a dashboard-owned asset (same category as
    # theme YAML): resolve them from the process launch home so they don't
    # vanish when a request is scoped to another profile via a context-local
    # HERMES_HOME override (e.g. embedded /chat under --open-profile).
    #
    # #87197: when the process itself is profile-scoped (``--profile <name>``
    # sets ``HERMES_HOME=<root>/profiles/<name>``), the launch home is the
    # profile directory, which has no ``plugins/`` — user plugins are
    # installed in the hermes root (``~/.hermes/plugins``). Scan the default
    # root as well (``get_default_hermes_root()`` unwraps
    # ``<root>/profiles/<name>`` → ``<root>`` and returns a custom
    # ``HERMES_HOME`` unchanged when it *is* the root), mirroring how
    # ``hermes_cli.plugins`` resolves plugin install locations. The
    # ``seen_names`` dedupe below keeps profile-local plugins (if any)
    # authoritative over same-named root plugins.
    from hermes_constants import get_default_hermes_root

    user_plugin_roots = [get_process_hermes_home() / "plugins"]
    root_plugins = get_default_hermes_root() / "plugins"
    if root_plugins.resolve(strict=False) != user_plugin_roots[0].resolve(strict=False):
        user_plugin_roots.append(root_plugins)
    search_dirs = [(d, "user") for d in user_plugin_roots]
    search_dirs += [
        (bundled_root / "memory", "bundled"),
        (bundled_root, "bundled"),
    ]
    # GHSA-5qr3-c538-wm9j (#29156): the previous ``os.environ.get(...)``
    # check treated *any* non-empty string as truthy, so ``=0``, ``=false``,
    # and ``=no`` — all of which the agent loader and operators correctly
    # read as "disabled" — silently *enabled* the untrusted project source
    # in the web server.  Combined with the absolute-path RCE primitive on
    # the manifest's ``api`` field (now patched below), this turned the
    # opt-in into a sticky always-on switch.  Use the shared truthy
    # semantics (``1`` / ``true`` / ``yes`` / ``on``) so the gate matches
    # ``hermes_cli/plugins.py`` and the documented user contract.
    if env_var_enabled("HERMES_ENABLE_PROJECT_PLUGINS"):
        search_dirs.append((Path.cwd() / ".hermes" / "plugins", "project"))

    for plugins_root, source in search_dirs:
        if not plugins_root.is_dir():
            continue
        with os.scandir(plugins_root) as scan:
            children = sorted((Path(e.path) for e in scan), key=lambda p: p.name)
        for child in children:
            if not child.is_dir():
                continue
            manifest_file = child / "dashboard" / "manifest.json"
            if not manifest_file.exists():
                continue
            try:
                data = json.loads(manifest_file.read_text(encoding="utf-8"))
                name = data.get("name", child.name)
                if name in seen_names:
                    continue
                seen_names.add(name)
                # Tab options: ``path`` + ``position`` for a new tab, optional
                # ``override`` to replace a built-in route, and ``hidden`` to
                # register the plugin component/slots without adding a tab
                # (useful for slot-only plugins like a header-crest injector).
                raw_tab = data.get("tab", {}) if isinstance(data.get("tab"), dict) else {}
                tab_info = {
                    "path": raw_tab.get("path", f"/{name}"),
                    "position": raw_tab.get("position", "end"),
                }
                override_path = raw_tab.get("override")
                if isinstance(override_path, str) and override_path.startswith("/"):
                    tab_info["override"] = override_path
                if bool(raw_tab.get("hidden")):
                    tab_info["hidden"] = True
                # Slots: list of named slot locations this plugin populates.
                # The frontend exposes ``registerSlot(pluginName, slotName, Component)``
                # on window; plugins with non-empty slots call it from their JS bundle.
                slots_src = data.get("slots")
                slots: List[str] = []
                if isinstance(slots_src, list):
                    slots = [s for s in slots_src if isinstance(s, str) and s]
                # Validate ``api`` at discovery time so the value cached
                # on the plugin entry is already safe to feed into the
                # importer.  An attacker-controlled manifest can name
                # any absolute path or ``..`` traversal here — the
                # web server then imports that file as a Python module
                # (RCE, GHSA-5qr3-c538-wm9j).
                raw_api = data.get("api")
                dashboard_dir = child / "dashboard"
                safe_api = _safe_plugin_api_relpath(raw_api, dashboard_dir=dashboard_dir)
                if raw_api and safe_api is None:
                    _log.warning(
                        "Plugin %s: refusing unsafe api path %r (must be a "
                        "relative file inside the plugin's dashboard/ "
                        "directory); backend routes from this plugin will "
                        "not be mounted",
                        name, raw_api,
                    )
                plugins.append({
                    "name": name,
                    "label": data.get("label", name),
                    "description": data.get("description", ""),
                    "icon": data.get("icon", "Puzzle"),
                    "version": data.get("version", "0.0.0"),
                    "tab": tab_info,
                    "slots": slots,
                    "entry": data.get("entry", "dist/index.js"),
                    "css": data.get("css"),
                    "has_api": bool(safe_api),
                    "source": source,
                    "_dir": str(dashboard_dir),
                    "_api_file": safe_api,
                })
            except Exception as exc:
                _log.warning("Bad dashboard plugin manifest %s: %s", manifest_file, exc)
                continue
    return plugins


def _strip_dashboard_manifest(p: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in p.items() if not k.startswith("_")}


_PLUGINS_HUB_CACHE_TTL_SECONDS = 5.0
_plugins_hub_cache: Optional[Dict[str, Any]] = None
_plugins_hub_cache_expires_at = 0.0
_plugins_hub_cache_lock = threading.Lock()


def _invalidate_plugins_hub_cache() -> None:
    global _plugins_hub_cache, _plugins_hub_cache_expires_at
    with _plugins_hub_cache_lock:
        _plugins_hub_cache = None
        _plugins_hub_cache_expires_at = 0.0


_plugins_hub_probe_inflight: set = set()
_plugins_hub_probe_lock = threading.Lock()


def _schedule_check_fn_probe(fn) -> Optional[threading.Thread]:
    """Warm a cold ``check_fn`` verdict off the request path.

    The hub read path only consumes cached availability (never probes
    inline). But the only other warmer lives in the tool-schema build, which
    a dashboard-only session never runs — so a cold cache would report
    ``auth_required=False`` forever. Kick a daemon-thread probe on the miss;
    the short hub TTL picks up the verdict on the next fetch. Deduplicates
    concurrent probes per function. Returns the spawned thread (or ``None``
    when a probe for *fn* is already in flight).
    """
    with _plugins_hub_probe_lock:
        if fn in _plugins_hub_probe_inflight:
            return None
        _plugins_hub_probe_inflight.add(fn)

    def _probe():
        try:
            from tools.registry import _check_fn_cached

            _check_fn_cached(fn)
        except Exception:
            pass
        finally:
            with _plugins_hub_probe_lock:
                _plugins_hub_probe_inflight.discard(fn)

    thread = threading.Thread(
        target=_probe, name="plugins-hub-checkfn-probe", daemon=True
    )
    thread.start()
    return thread


def _merged_plugins_hub(force_refresh: bool = False) -> Dict[str, Any]:
    """Agent discovery + dashboard manifests + optional provider picker metadata.

    IMPORTANT: this powers a dashboard request path, so it must stay read-only
    and cheap. In particular, do not execute tool ``check_fn`` probes here —
    those can trigger imports, auth/network checks, and other synchronous work
    that starves the root event loop. We only consume last-known cached tool
    availability, and we memoize the assembled payload briefly to collapse the
    dashboard's bursty duplicate fetches.
    """
    from hermes_cli.web_server import (
        _discover_memory_provider_statuses,
        _get_dashboard_plugins,
        _normalize_memory_provider_name,
        _schedule_check_fn_probe,
        get_hermes_home,
        load_config,
    )
    global _plugins_hub_cache, _plugins_hub_cache_expires_at
    now = time.monotonic()
    if not force_refresh:
        with _plugins_hub_cache_lock:
            if _plugins_hub_cache is not None and now < _plugins_hub_cache_expires_at:
                return _plugins_hub_cache

    started_at = time.monotonic()
    from hermes_cli.plugins_cmd import (
        _discover_all_plugins,
        _get_current_context_engine,
        _get_current_memory_provider,
        _discover_context_engines,
        _get_disabled_set,
        _get_enabled_set,
        _read_manifest as _read_plugin_manifest_at,
    )

    dashboard_list = _get_dashboard_plugins()
    dash_by_name = {str(p["name"]): p for p in dashboard_list}

    disabled_set = _get_disabled_set()
    enabled_set = _get_enabled_set()

    # Read user-hidden plugins from config for the user_hidden field.
    config = load_config()
    hidden_plugins: list = cfg_get(config, "dashboard", "hidden_plugins", default=[]) or []

    plugins_root_resolved = (get_hermes_home() / "plugins").resolve()
    rows: List[Dict[str, Any]] = []

    for name, version, description, source, dir_str, key in _discover_all_plugins():
        # Both the path-derived key (nested category plugins) and the bare
        # manifest name count for enabled/disabled state, matching the runtime
        # loader's back-compat lookup.
        aliases = {name}
        if key:
            aliases.add(key)
        if aliases & disabled_set:
            runtime_status = "disabled"
        elif aliases & enabled_set:
            runtime_status = "enabled"
        else:
            runtime_status = "inactive"

        dir_path = Path(dir_str)
        dm = dash_by_name.get(name)
        has_dash_manifest = dm is not None or (dir_path / "dashboard" / "manifest.json").exists()

        under_user_tree = False
        try:
            dir_path.resolve().relative_to(plugins_root_resolved)
            under_user_tree = True
        except ValueError:
            pass

        can_remove_update = (
            source in {"user", "git"} and under_user_tree and Path(dir_str).is_dir()
        )

        # Read-only auth hint: consult only last-known cached tool availability.
        # A missing cache entry is treated as "unknown" rather than triggering a
        # live probe inside this request path.
        auth_required = False
        auth_command = ""
        manifest_data = _read_plugin_manifest_at(dir_path)
        provides_tools = manifest_data.get("provides_tools") or []
        if provides_tools:
            try:
                from tools.registry import get_cached_check_fn_result, registry
                for tname in provides_tools:
                    entry = registry.get_entry(tname)
                    if not entry or not entry.check_fn:
                        continue
                    cached_result = get_cached_check_fn_result(entry.check_fn)
                    if cached_result is None:
                        # Cold cache: nothing else warms check_fns on
                        # dashboard-only sessions, so kick a background
                        # probe; the short hub TTL surfaces the verdict on
                        # the next fetch instead of pinning auth_required
                        # to False forever.
                        _schedule_check_fn_probe(entry.check_fn)
                        continue
                    if cached_result is False:
                        auth_required = True
                        auth_command = f"hermes auth {name}"
                        break
            except Exception:
                pass

        rows.append({
            "name": name,
            "version": version or "",
            "description": description or "",
            "source": source,
            "runtime_status": runtime_status,
            "has_dashboard_manifest": has_dash_manifest,
            "dashboard_manifest": _strip_dashboard_manifest(dm) if dm else None,
            "path": dir_str,
            "can_remove": can_remove_update,
            "can_update_git": can_remove_update and (Path(dir_str) / ".git").exists(),
            "auth_required": auth_required,
            "auth_command": auth_command,
            "user_hidden": name in hidden_plugins,
        })

    agent_names = {r["name"] for r in rows}
    orphan_dashboard = [
        _strip_dashboard_manifest(p)
        for p in dashboard_list
        if str(p["name"]) not in agent_names
    ]

    memory_providers = _discover_memory_provider_statuses()

    context_engines: List[Dict[str, str]] = []
    try:
        for n, desc in _discover_context_engines():
            context_engines.append({"name": n, "description": desc})
    except Exception:
        context_engines = []

    payload = {
        "plugins": rows,
        "orphan_dashboard_plugins": orphan_dashboard,
        "providers": {
            "memory_provider": _normalize_memory_provider_name(_get_current_memory_provider()),
            "memory_options": memory_providers,
            "context_engine": _get_current_context_engine(),
            "context_options": context_engines,
        },
    }
    duration = time.monotonic() - started_at
    if duration >= 0.25:
        _log.info(
            "plugins/hub rebuilt in %.3fs (plugins=%d memory_options=%d)",
            duration,
            len(rows),
            len(memory_providers),
        )
    with _plugins_hub_cache_lock:
        _plugins_hub_cache = payload
        _plugins_hub_cache_expires_at = time.monotonic() + _PLUGINS_HUB_CACHE_TTL_SECONDS
    return payload


def _mount_plugin_api_routes():
    """Import and mount backend API routes from plugins that declare them.

    Each plugin's ``api`` field points to a Python file that must expose
    a ``router`` (FastAPI APIRouter).  Routes are mounted under
    ``/api/plugins/<name>/``.

    Backend import is restricted to ``bundled`` and ``user`` sources.
    Project plugins (``./.hermes/plugins/``) ship with the CWD and are
    therefore attacker-controlled in any threat model where the user
    opens a malicious repo; they can extend the dashboard UI via
    static JS/CSS but their Python ``api`` file is never auto-imported
    by the web server.  See GHSA-5qr3-c538-wm9j (#29156).

    Additionally, user plugins must be explicitly enabled via the
    ``plugins.enabled`` allow-list in config.yaml before their backend
    code is imported. Without this gate, an installed-but-not-enabled
    plugin's Python code would execute at dashboard startup — a code
    execution vector that bypasses the user's intent. (#46435,
    GHSA-mcfc-hp25-cjv7)
    """
    from hermes_cli.web_server import _get_dashboard_plugins, app
    # Load the enabled/disabled sets once for the loop.
    try:
        from hermes_cli.plugins_cmd import _get_enabled_set, _get_disabled_set
        enabled_set = _get_enabled_set()
        disabled_set = _get_disabled_set()
    except Exception:
        enabled_set = set()
        disabled_set = set()

    for plugin in _get_dashboard_plugins():
        api_file_name = plugin.get("_api_file")
        if not api_file_name:
            continue
        plugin_name = plugin.get("name", "")
        # Gate: user plugins must be in plugins.enabled and not in
        # plugins.disabled before we import their Python code.
        # Bundled plugins are trusted (they ship with the release) but
        # still respect an explicit disable.
        if plugin.get("source") == "user":
            if plugin_name in disabled_set:
                _log.debug(
                    "Plugin %s: skipping API mount (explicitly disabled)",
                    plugin_name,
                )
                continue
            if plugin_name not in enabled_set:
                _log.debug(
                    "Plugin %s: skipping API mount (not in plugins.enabled)",
                    plugin_name,
                )
                continue
        elif plugin.get("source") == "bundled":
            if plugin_name in disabled_set:
                _log.debug(
                    "Plugin %s: skipping API mount (explicitly disabled)",
                    plugin_name,
                )
                continue
        if plugin.get("source") == "project":
            _log.warning(
                "Plugin %s: ignoring backend api=%s (project plugins may "
                "not auto-import Python code; move the plugin to "
                "~/.hermes/plugins/ if you trust it)",
                plugin["name"], api_file_name,
            )
            continue
        dashboard_dir = Path(plugin["_dir"])
        api_path = dashboard_dir / api_file_name
        try:
            resolved_api = api_path.resolve()
            resolved_base = dashboard_dir.resolve()
            resolved_api.relative_to(resolved_base)
        except (OSError, RuntimeError, ValueError):
            # Discovery already filters this, but re-check here in case
            # ``_dir`` was tampered with after caching or a future caller
            # bypasses the validator.  Defence in depth keeps the import
            # primitive contained even if the upstream check regresses.
            _log.warning(
                "Plugin %s: refusing to import api file outside its "
                "dashboard directory (%s)", plugin["name"], api_path,
            )
            continue
        if not api_path.exists():
            _log.warning("Plugin %s declares api=%s but file not found", plugin["name"], api_file_name)
            continue
        try:
            module_name = f"hermes_dashboard_plugin_{plugin['name']}"
            spec = importlib.util.spec_from_file_location(module_name, api_path)
            if spec is None or spec.loader is None:
                continue
            mod = importlib.util.module_from_spec(spec)
            # Register in sys.modules BEFORE exec_module so pydantic/FastAPI
            # can resolve forward references (e.g. models defined in a file
            # that uses `from __future__ import annotations`). Without this,
            # TypeAdapter lazy-build fails at first request with
            # "is not fully defined" because the module namespace isn't
            # reachable by name for string-annotation resolution.
            sys.modules[module_name] = mod
            try:
                spec.loader.exec_module(mod)
            except Exception:
                sys.modules.pop(module_name, None)
                raise
            router = getattr(mod, "router", None)
            if router is None:
                _log.warning("Plugin %s api file has no 'router' attribute", plugin["name"])
                continue
            app.include_router(router, prefix=f"/api/plugins/{plugin['name']}")
            _log.info("Mounted plugin API routes: /api/plugins/%s/", plugin["name"])
        except Exception as exc:
            _log.warning("Failed to load plugin %s API routes: %s", plugin["name"], exc)
