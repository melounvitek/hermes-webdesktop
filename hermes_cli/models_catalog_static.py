"""Static provider/model catalog tables: curated per-provider model lists, canonical provider registry, display groups, alias maps.

Split out of ``hermes_cli.models``; every moved name is re-imported there, so
``hermes_cli.models.<name>`` keeps resolving (and monkeypatching) as before.
"""

from __future__ import annotations

from typing import NamedTuple


# Fallback OpenRouter snapshot used when the live catalog is unavailable.
# (model_id, display description shown in menus)
OPENROUTER_MODELS: list[tuple[str, str]] = [
    # Anthropic
    ("anthropic/claude-fable-5.1",             ""),
    ("anthropic/claude-fable-5",               ""),
    ("anthropic/claude-opus-5",                ""),
    ("anthropic/claude-opus-5-fast",           "2x price, higher output speed"),
    ("anthropic/claude-opus-4.8",              ""),
    ("anthropic/claude-opus-4.8-fast",         "2x price, higher output speed"),
    ("anthropic/claude-sonnet-5",              ""),
    ("anthropic/claude-haiku-4.5",             ""),
    # OpenAI
    ("openai/gpt-5.6-sol",                     ""),
    ("openai/gpt-5.6-sol-pro",                 ""),
    ("openai/gpt-5.6-terra",                   ""),
    ("openai/gpt-5.6-terra-pro",               ""),
    ("openai/gpt-5.6-luna",                    ""),
    ("openai/gpt-5.6-luna-pro",                ""),
    ("openai/gpt-5.5",                         ""),
    ("openai/gpt-5.5-pro",                     ""),
    ("openai/gpt-5.4-mini",                    ""),
    # Google
    ("google/gemini-3.1-pro-preview",          ""),
    ("google/gemini-3.8-flash",                ""),
    ("google/gemini-3.7-flash",                ""),
    # xAI
    ("x-ai/grok-4.6",                          ""),
    # DeepSeek
    ("deepseek/deepseek-v4-pro",               ""),
    ("deepseek/deepseek-v4-pro-0813",          "dated snapshot of v4-pro"),
    ("deepseek/deepseek-v4-flash",             ""),
    ("deepseek/deepseek-v4-flash-0731",        "dated snapshot of v4-flash"),
    # Qwen
    ("qwen/qwen3.8-max",                       ""),
    ("qwen/qwen3.8-flash",                     ""),
    # MoonshotAI
    ("moonshotai/kimi-k3",                     "recommended"),
    # MiniMax
    ("minimax/minimax-m3",                     ""),
    # Z-AI
    ("z-ai/glm-5.3",                           ""),
    ("z-ai/glm-5.3-flash",                     ""),
    ("z-ai/glm-5.2",                           "default"),
    # Xiaomi
    ("xiaomi/mimo-v2.5-pro",                   ""),
    # Tencent
    ("tencent/hy4-preview",                    ""),
    ("tencent/hy3",                            ""),
    # StepFun
    ("stepfun/step-3.7-flash",                 ""),
    # NVIDIA
    ("nvidia/nemotron-3-super-120b-a12b",      ""),
    # Meta
    ("meta/muse-spark-1.2",                    ""),
    # Sakana
    ("sakana/fugu-ultra",                      ""),
    # OpenRouter routers
    ("openrouter/pareto-code",                 "auto-routes to cheapest coder meeting openrouter.min_coding_score"),
    # Free tier
    ("thinkingmachines/inkling:free",          "free"),
    ("thinkingmachines/inkling-small:free",    "free"),
    ("minimax/minimax-m3:free",                "free"),
    ("z-ai/glm-5.2:free",                      "free"),
    ("poolside/laguna-s-2.1:free",             "free"),
    ("poolside/laguna-xs-2.1:free",            "free"),
    ("nvidia/nemotron-3-super-120b-a12b:free", "free"),
    ("nvidia/nemotron-3-ultra-550b-a55b:free", "free"),
    ("nvidia/nemotron-3.5-lightning:free",     "free"),
]


# Fallback Vercel AI Gateway snapshot used when the live catalog is unavailable.
# OSS / open-weight models prioritized first, then closed-source by family.
# Slugs match Vercel's actual /v1/models catalog (e.g. alibaba/ for Qwen,
# zai/ and xai/ without hyphens).
VERCEL_AI_GATEWAY_MODELS: list[tuple[str, str]] = [
    ("moonshotai/kimi-k2.6",                 "recommended"),
    ("alibaba/qwen3.6-plus",                 ""),
    ("zai/glm-5.1",                          ""),
    ("minimax/minimax-m2.7",                 ""),
    ("anthropic/claude-sonnet-4.6",          ""),
    ("anthropic/claude-opus-4.7",            ""),
    ("anthropic/claude-opus-4.6",            ""),
    ("anthropic/claude-haiku-4.5",           ""),
    ("openai/gpt-5.4",                       ""),
    ("openai/gpt-5.4-mini",                  ""),
    ("openai/gpt-5.3-codex",                 ""),
    ("google/gemini-3.1-pro-preview",        ""),
    ("google/gemini-3-flash",                ""),
    ("google/gemini-3.1-flash-lite-preview", ""),
    ("xai/grok-4.20-reasoning",              ""),
]


def _codex_curated_models() -> list[str]:
    """Derive the openai-codex curated list from codex_models.py.

    Single source of truth: DEFAULT_CODEX_MODELS + forward-compat synthesis. This keeps the gateway
    /model picker in sync with the CLI `hermes model` flow without maintaining a separate static
    list.
    """
    from hermes_cli.codex_models import DEFAULT_CODEX_MODELS, _finalize_codex_models
    return _finalize_codex_models(list(DEFAULT_CODEX_MODELS))


# Static fallback for xAI when the models.dev disk cache is empty (fresh
# install, offline first run, etc.). Mirrors the xAI-direct model IDs from
# $HERMES_HOME/models_dev_cache.json as of 2026-04-28. Whenever xAI renames
# or retires a model, the disk cache picks it up on the next refresh and the
# fallback here only matters until that refresh lands.
#
# Models retired by xAI on May 15, 2026 are excluded — see
# https://docs.x.ai/developers/migration/may-15-retirement
# (grok-4, grok-4-0709, grok-4-fast{,-reasoning,-non-reasoning},
#  grok-4-1-fast{,-reasoning,-non-reasoning}, grok-code-fast-1 → grok-4.3).
_XAI_STATIC_FALLBACK: list[str] = [
    "grok-4.6",
    "grok-build-0.1",
    "grok-4.5",
    "grok-4.3",
    "grok-4.20-0309-reasoning",
    "grok-4.20-0309-non-reasoning",
    "grok-4.20-multi-agent-0309",
]


# Callable via xAI OAuth but omitted from models.dev and /v1/models listings.
_XAI_CURATED_EXTRAS: list[str] = [
    "grok-4.6",  # GA 2026-08 — kept until the models.dev disk cache refreshes
    "grok-4.5",  # GA 2026-07 — kept until the models.dev disk cache refreshes
    "grok-composer-2.5-fast",
]


_XAI_TOP_MODEL = "grok-4.6"


def _xai_promote_top(ids: list[str]) -> list[str]:
    """Pin the headline xAI model to the top of the curated list."""
    if _XAI_TOP_MODEL in ids:
        return [_XAI_TOP_MODEL] + [m for m in ids if m != _XAI_TOP_MODEL]
    return ids


def _xai_merge_curated_extras(ids: list[str]) -> list[str]:
    """Append Hermes-curated xAI models that are missing from models.dev."""
    out = list(ids)
    for extra in _XAI_CURATED_EXTRAS:
        if extra in out:
            continue
        # Keep the headline model pinned; slot extras immediately after it.
        insert_at = 1 if out and out[0] == _XAI_TOP_MODEL else len(out)
        out.insert(insert_at, extra)
    return out


def _xai_finalize_catalog(ids: list[str]) -> list[str]:
    return _xai_promote_top(_xai_merge_curated_extras(ids))


def _xai_curated_models() -> list[str]:
    """Offline curated floor for xAI / xAI OAuth pickers.

    Reads $HERMES_HOME/models_dev_cache.json directly (no network). Falls back to
    ``_XAI_STATIC_FALLBACK`` when the cache is empty or unreadable.
    """
    try:
        from agent.models_dev import _load_disk_cache
        data = _load_disk_cache()
        xai = data.get("xai") if isinstance(data, dict) else None
        models = xai.get("models") if isinstance(xai, dict) else None
        if isinstance(models, dict) and models:
            ids = [mid for mid in models.keys() if isinstance(mid, str)]
            if ids:
                return _xai_finalize_catalog(sorted(ids))
    except Exception:
        # Any failure (missing file, malformed JSON, import error)
        # falls through to the static list.
        pass
    return _xai_finalize_catalog(list(_XAI_STATIC_FALLBACK))


_PROVIDER_MODELS: dict[str, list[str]] = {
    "moa": ["default"],
    "nous": [
        # Anthropic
        "anthropic/claude-fable-5.1",
        "anthropic/claude-fable-5",
        "anthropic/claude-opus-5",
        "anthropic/claude-opus-4.8",
        "anthropic/claude-sonnet-5",
        "anthropic/claude-haiku-4.5",
        # OpenAI
        "openai/gpt-5.6-sol",
        "openai/gpt-5.6-sol-pro",
        "openai/gpt-5.6-terra",
        "openai/gpt-5.6-terra-pro",
        "openai/gpt-5.6-luna",
        "openai/gpt-5.6-luna-pro",
        "openai/gpt-5.5",
        "openai/gpt-5.5-pro",
        "openai/gpt-5.4-mini",
        # Google
        "google/gemini-3.1-pro-preview",
        "google/gemini-3.8-flash",
        "google/gemini-3.7-flash",
        # xAI
        "x-ai/grok-4.6",
        # DeepSeek
        "deepseek/deepseek-v4-pro",
        "deepseek/deepseek-v4-pro-0813",
        "deepseek/deepseek-v4-flash",
        "deepseek/deepseek-v4-flash-0731",
        # Qwen
        "qwen/qwen3.8-max",
        "qwen/qwen3.8-flash",
        # MoonshotAI
        "moonshotai/kimi-k3",
        # MiniMax
        "minimax/minimax-m3",
        # Z-AI
        "z-ai/glm-5.3",
        "z-ai/glm-5.3-flash",
        "z-ai/glm-5.2",
        # Xiaomi
        "xiaomi/mimo-v2.5-pro",
        # Tencent
        "tencent/hy4-preview",
        "tencent/hy3",
        # StepFun
        "stepfun/step-3.7-flash",
        # NVIDIA
        "nvidia/nemotron-3-super-120b-a12b",
        # Sakana
        "sakana/fugu-ultra",
    ],
    # Native OpenAI Chat Completions (api.openai.com). Used by /model counts and
    # provider_model_ids fallback when /v1/models is unavailable.
    "openai": [
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5-mini",
        "gpt-5.3-codex",
        "gpt-5.2-codex",
        "gpt-4.1",
        "gpt-4o",
        "gpt-4o-mini",
    ],
    "openai-api": [
        "gpt-5.6-sol",
        "gpt-5.6-sol-pro",
        "gpt-5.6-terra",
        "gpt-5.6-terra-pro",
        "gpt-5.6-luna",
        "gpt-5.6-luna-pro",
        "gpt-5.5",
        "gpt-5.5-pro",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5-mini",
        "gpt-5.3-codex",
        "gpt-4.1",
        "gpt-4o",
        "gpt-4o-mini",
    ],
    "openai-codex": _codex_curated_models(),
    "xai-oauth": _xai_curated_models(),
    "copilot-acp": [
        "copilot-acp",
    ],
    "copilot": [
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5-mini",
        "gpt-5.3-codex",
        "gpt-5.2-codex",
        "gpt-4.1",
        "gpt-4o",
        "gpt-4o-mini",
        "claude-sonnet-4.6",
        "claude-sonnet-5",
        "claude-sonnet-4",
        "claude-sonnet-4.5",
        "claude-haiku-4.5",
        "gemini-3.1-pro-preview",
        "gemini-3-pro-preview",
        "gemini-3-flash-preview",
        "gemini-2.5-pro",
    ],
    "gemini": [
        "gemini-3.1-pro-preview",
        "gemini-3-pro-preview",
        "gemini-3.6-flash",
        "gemini-3.1-flash-lite-preview",
    ],
    "zai": [
        "glm-5.3",
        "glm-5.3-flash",
        "glm-5.2",
        "glm-5.1",
        "glm-5",
        "glm-5v-turbo",
        "glm-5-turbo",
        "glm-4.7",
        "glm-4.5",
        "glm-4.5-flash",
    ],
    "xai": _xai_curated_models(),
    "nvidia": [
        # NVIDIA flagship reasoning models
        "nvidia/nemotron-3-ultra-550b-a55b",
        "nvidia/nemotron-3-super-120b-a12b",
        "nvidia/nemotron-3.5-lightning-30b-a3b",
        "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning",
        # Third-party agentic models hosted on build.nvidia.com
        # (map to OpenRouter defaults — users get familiar picks on NIM)
        "z-ai/glm-5.3",
        "z-ai/glm-5.2",
        "moonshotai/kimi-k2.6",
        "minimaxai/minimax-m3",
    ],
    "kimi-coding": [
        "kimi-k3",
        "kimi-k2.7-code",
        "kimi-k2.6",
        "kimi-k2.5",
        "kimi-for-coding",
        "kimi-for-coding-highspeed",
        "kimi-k2-thinking",
        "kimi-k2-thinking-turbo",
        "kimi-k2-turbo-preview",
        "kimi-k2-0905-preview",
    ],
    "kimi-coding-cn": [
        "kimi-k3",
        "kimi-k2.7-code",
        "kimi-k2.7-code-highspeed",
        "kimi-k2.6",
        "kimi-k2.5",
        "kimi-k2-thinking",
        "kimi-k2-turbo-preview",
        "kimi-k2-0905-preview",
    ],
    "stepfun": [
        "step-3.5-flash",
        "step-3.5-flash-2603",
    ],
    "moonshot": [
        "kimi-k3",
        "kimi-k2.6",
        "kimi-k2.5",
        "kimi-k2-thinking",
        "kimi-k2-turbo-preview",
        "kimi-k2-0905-preview",
    ],
    "minimax": [
        "MiniMax-M3",
        "MiniMax-M2.7",
        "MiniMax-M2.5",
        "MiniMax-M2.1",
        "MiniMax-M2",
    ],
    "minimax-oauth": [
        "MiniMax-M3",
        "MiniMax-M2.7",
        "MiniMax-M2.7-highspeed",
    ],
    "minimax-cn": [
        "MiniMax-M3",
        "MiniMax-M2.7",
        "MiniMax-M2.5",
        "MiniMax-M2.1",
        "MiniMax-M2",
    ],
    "anthropic": [
        "claude-fable-5",
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-sonnet-4-6",
        "claude-opus-4-5-20251101",
        "claude-sonnet-4-5-20250929",
        "claude-opus-4-20250514",
        "claude-sonnet-4-20250514",
        "claude-haiku-4-5-20251001",
    ],
    "deepseek": [
        "deepseek-v4-pro",
        "deepseek-v4-flash",
    ],
    "xiaomi": [
        "mimo-v2.5-pro",
        "mimo-v2.5",
        "mimo-v2-pro",
        "mimo-v2-omni",
        "mimo-v2-flash",
    ],
    "tencent-tokenhub": [
        "hy4-preview",
        "hy3",
        "hy3-preview",
    ],
    "tencent-tokenplan": [
        "hy4-preview",
        "hy3",
        "hy3-preview",
    ],
    "arcee": [
        "trinity-large-thinking",
        "trinity-large-preview",
        "trinity-mini",
    ],
    "gmi": [
        "zai-org/GLM-5.1-FP8",
        "deepseek-ai/DeepSeek-V3.2",
        "moonshotai/Kimi-K2.5",
        "google/gemini-3.1-flash-lite-preview",
        "anthropic/claude-sonnet-5",
        "anthropic/claude-sonnet-4.6",
        "openai/gpt-5.4",
    ],
    # Synced against https://opencode.ai/docs/zen/ + live GET /zen/v1/models
    # (2026-08-20). Zen/Go are _LIVE_FIRST_PICKER_PROVIDERS, so this list is a
    # discovery floor — live entries lead in the picker and stale curated
    # names never pollute the top.
    "opencode-zen": [
        "x-preview-f-free",  # "Ox Alpha" stealth model — free, 1M ctx, ZDR
        "kimi-k3",
        "kimi-k2.5",
        "kimi-k2.6",
        "gpt-5.6-sol",
        "gpt-5.6-terra",
        "gpt-5.6-luna",
        "gpt-5.5",
        "gpt-5.5-pro",
        "gpt-5.4-pro",
        "gpt-5.4",
        "gpt-5.4-mini",
        "gpt-5.4-nano",
        "gpt-5.3-codex",
        "gpt-5.3-codex-spark",
        "gpt-5.2",
        "gpt-5.2-codex",
        "gpt-5.1",
        "gpt-5.1-codex",
        "gpt-5.1-codex-max",
        "gpt-5.1-codex-mini",
        "gpt-5",
        "gpt-5-codex",
        "gpt-5-nano",
        "claude-fable-5",
        "claude-opus-5",
        "claude-sonnet-5",
        "claude-opus-4-8",
        "claude-opus-4-7",
        "claude-opus-4-6",
        "claude-opus-4-5",
        "claude-sonnet-4-6",
        "claude-sonnet-4-5",
        "claude-sonnet-4",
        "claude-haiku-4-5",
        "gemini-3.7-flash",
        "gemini-3.6-flash",
        "gemini-3.5-flash",
        "gemini-3.5-flash-lite",
        "gemini-3.1-pro",
        "gemini-3-flash",
        "grok-4.6",
        "grok-4.5",
        "grok-build-0.1",
        "muse-spark-1.2",
        "minimax-m3",
        "minimax-m2.7",
        "minimax-m2.5",
        "glm-5.3",
        "glm-5.3-flash",
        "glm-5.2",
        "glm-5.1",
        "glm-5",
        "kimi-k2.7-code",
        "deepseek-v4-pro",
        "deepseek-v4-flash",
        "deepseek-v4-flash-free",
        "qwen3.6-plus",
        "qwen3.5-plus",
        "big-pickle",
        "mimo-v2.5-free",
        "hy3-free",
        "laguna-s-2.1-free",
        "nemotron-3-ultra-free",
        "nemotron-3.5-lightning-free",
        "muse-spark-1.2-contributor-free",
    ],
    # OpenCode free tier — keyless (no OpenCode account needed). This is the
    # OFFLINE FLOOR only: provider_model_ids("opencode-free") revalidates live
    # against GET /zen/v1/models (keyless) and filters to the anonymous free
    # tier, so a relay-delisted model stops appearing in the picker and a
    # newly-live one becomes selectable without a release. This floor keeps the
    # picker populated when the relay is unreachable. Note: this floor may lag
    # the live relay — that is intentional; the live revalidation is the
    # source of truth when reachable. Known-delisted models are REMOVED from
    # the floor (x-preview-f-free delisted 2026-08-26 — offline fallback must
    # not offer a model that 401s). deepseek-v4-flash-free and mimo-v2.5-free
    # are back on the live list.
    "opencode-free": [
        "deepseek-v4-flash-free",
        "hy3-free",
        "mimo-v2.5-free",
        "laguna-s-2.1-free",
        "nemotron-3-ultra-free",
        "nemotron-3.5-lightning-free",
        "muse-spark-1.2-contributor-free",
    ],
    # Synced against https://opencode.ai/docs/go/ + live GET /zen/go/v1/models
    # (2026-08-20).
    "opencode-go": [
        "kimi-k3",
        "kimi-k2.7-code",
        "kimi-k2.6",
        "kimi-k2.5",
        "gpt-5.6-luna",
        "grok-4.5",
        "glm-5.3",
        "glm-5.3-flash",
        "glm-5.2",
        "glm-5.1",
        "glm-5",
        "mimo-v2.5-pro",
        "mimo-v2.5",
        "mimo-v2-pro",
        "mimo-v2-omni",
        "minimax-m3",
        "minimax-m2.7",
        "minimax-m2.5",
        "deepseek-v4-pro",
        "deepseek-v4-flash",
        "qwen3.8-max",
        "qwen3.7-max",
        "qwen3.7-plus",
        "qwen3.6-plus",
        "qwen3.5-plus",
        "hy3",
        "hy3-preview",
        "muse-spark-1.2-contributor",
        # Go-subscription twin of the Zen keyless Ox Alpha (live go/v1
        # catalog 2026-08-21; NOT keyless — Go relay requires a Go key).
        "ox-alpha-free",
    ],
    "kilocode": [
        "anthropic/claude-opus-4.6",
        "anthropic/claude-sonnet-4.6",
        "openai/gpt-5.4",
        "google/gemini-3-pro-preview",
        "google/gemini-3-flash-preview",
    ],
    # Alibaba DashScope Coding platform (coding-intl) — default endpoint.
    # Supports Qwen models + third-party providers (GLM, Kimi, MiniMax).
    # Users with classic DashScope keys should override DASHSCOPE_BASE_URL
    # to https://dashscope-intl.aliyuncs.com/compatible-mode/v1 (OpenAI-compat)
    # or https://dashscope-intl.aliyuncs.com/apps/anthropic (Anthropic-compat).
    "alibaba": [
        # Qwen 千问系列 (DashScope / Qwen Cloud)
        "qwen3.8-max",
        "qwen3.7-max",
        "qwen3.7-plus",
        "qwen3.6-plus",
        "qwen3.6-flash",
        "kimi-k2.5",
        "qwen3.5-plus",
        "qwen3-coder-plus",
        "qwen3-coder-next",
        # Third-party models available on coding-intl / DashScope
        "glm-5.2",
        "glm-5",
        "glm-4.7",
        "deepseek-v4-pro",
        "deepseek-v4-flash-0731",
        "MiniMax-M2.5",
    ],
    # Alibaba DashScope (China) — same platform as alibaba, domestic endpoint
    # (dashscope.aliyuncs.com); same catalog as the international tier.
    "alibaba-cn": [
        "qwen3.8-max",
        "qwen3.7-max",
        "qwen3.7-plus",
        "qwen3.6-plus",
        "qwen3.6-flash",
        "kimi-k2.5",
        "qwen3.5-plus",
        "qwen3-coder-plus",
        "qwen3-coder-next",
        "glm-5.2",
        "glm-5",
        "glm-4.7",
        "deepseek-v4-pro",
        "deepseek-v4-flash-0731",
        "MiniMax-M2.5",
    ],
    # Alibaba Coding Plan — same platform as alibaba (DashScope coding-intl),
    # separate provider ID with its own base_url_env_var.
    "alibaba-coding-plan": [
        "qwen3.7-plus",
        "qwen3.6-plus",
        "qwen3.5-plus",
        "qwen3-max-2026-01-23",
        "qwen3-coder-plus",
        "qwen3-coder-next",
        "kimi-k2.5",
        "glm-5",
        "glm-4.7",
        "MiniMax-M2.5",
    ],
    # Alibaba Coding Plan (China) — domestic coding endpoint
    # (coding.dashscope.aliyuncs.com); same catalog as the international tier.
    "alibaba-coding-plan-cn": [
        "qwen3.7-plus",
        "qwen3.6-plus",
        "qwen3.5-plus",
        "qwen3-max-2026-01-23",
        "qwen3-coder-plus",
        "qwen3-coder-next",
        "kimi-k2.5",
        "glm-5",
        "glm-4.7",
        "MiniMax-M2.5",
    ],
    # Alibaba Token Plan (Personal Edition) — dedicated token-plan endpoint
    # (token-plan.ap-southeast-1.maas.aliyuncs.com), key tier `sk-sp-...`.
    # Catalog verified against a live Token Plan subscription (2026-08-03).
    "alibaba-token-plan": [
        "qwen3.8-max-preview",
        "qwen3.7-max",
        "qwen3.7-plus",
        "qwen3.6-plus",
        "qwen3.6-flash",
        "deepseek-v4-pro",
        "deepseek-v4-flash",
        "deepseek-v3.2",
        "kimi-k2.7-code",
        "kimi-k2.6",
        "kimi-k2.5",
        "glm-5.2",
        "glm-5.1",
        "glm-5",
    ],
    # Alibaba Token Plan (China) — domestic token-plan endpoint
    # (token-plan.cn-beijing.maas.aliyuncs.com); same catalog as intl.
    "alibaba-token-plan-cn": [
        "qwen3.8-max-preview",
        "qwen3.7-max",
        "qwen3.7-plus",
        "qwen3.6-plus",
        "qwen3.6-flash",
        "deepseek-v4-pro",
        "deepseek-v4-flash",
        "deepseek-v3.2",
        "kimi-k2.7-code",
        "kimi-k2.6",
        "kimi-k2.5",
        "glm-5.2",
        "glm-5.1",
        "glm-5",
    ],
    # Curated HF model list — only agentic models that map to OpenRouter defaults.
    "huggingface": [
        "moonshotai/Kimi-K2.5",
        "Qwen/Qwen3.5-397B-A17B",
        "Qwen/Qwen3.5-35B-A3B",
        "deepseek-ai/DeepSeek-V3.2",
        "MiniMaxAI/MiniMax-M2.5",
        "zai-org/GLM-5",
        "XiaomiMiMo/MiMo-V2-Flash",
        "moonshotai/Kimi-K2-Thinking",
        "moonshotai/Kimi-K2.6",
    ],
    # AWS Bedrock — static fallback list used when dynamic discovery is
    # unavailable (no boto3, no credentials, or API error).  The agent
    # prefers live discovery via ListFoundationModels + ListInferenceProfiles.
    # Use inference profile IDs (us.*) since most models require them.
    "bedrock": [
        "us.anthropic.claude-sonnet-5",
        "us.anthropic.claude-sonnet-4-6",
        "us.anthropic.claude-opus-4-6-v1",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "us.anthropic.claude-sonnet-4-5-20250929-v1:0",
        "openai.gpt-5.5",
        "openai.gpt-5.6-sol",
        "openai.gpt-5.6-terra",
        "openai.gpt-5.6-luna",
        "us.amazon.nova-pro-v1:0",
        "us.amazon.nova-lite-v1:0",
        "us.amazon.nova-micro-v1:0",
        "deepseek.v3.2",
        "us.meta.llama4-maverick-17b-instruct-v1:0",
        "us.meta.llama4-scout-17b-instruct-v1:0",
    ],
    # Azure Foundry: user-provided endpoint and model.
    # Empty list because models depend on the endpoint configuration.
    "azure-foundry": [],
    # Google Vertex AI — static curated list.  Vertex's OpenAI-compatible
    # endpoint has no /models listing route, so without this entry the
    # /model picker only ever shows the currently-configured model.
    # Model IDs use the "google/" publisher prefix Vertex's openapi
    # endpoint expects (see hermes_cli/model_setup_flows.py).
    # Entries validated live against a GCP project (global region,
    # HTTP 200) as of 2026-07-21 (PR #68767).
    "vertex": [
        "google/gemini-3.1-pro-preview",
        "google/gemini-3-pro-preview",
        "google/gemini-3.6-flash",
        "google/gemini-3.5-flash",
        "google/gemini-3.5-flash-lite",
        "google/gemini-3-flash-preview",
        "google/gemini-3.1-flash-lite-preview",
        "google/gemini-3.1-flash-lite",
    ],
    "novita": [
        "moonshotai/kimi-k2.5",
        "minimax/minimax-m2.7",
        "zai-org/glm-5",
        "deepseek/deepseek-v3-0324",
        "deepseek/deepseek-r1-0528",
        "qwen/qwen3-235b-a22b-fp8",
    ],
}


# Vercel AI Gateway: derive the bare-model-id catalog from the curated
# ``VERCEL_AI_GATEWAY_MODELS`` snapshot so both the picker (tuples with descriptions)
# and the static fallback catalog (bare ids) stay in sync from a single
# source of truth.
_PROVIDER_MODELS["ai-gateway"] = [mid for mid, _ in VERCEL_AI_GATEWAY_MODELS]


# ---------------------------------------------------------------------------
# Canonical provider list — single source of truth for provider identity.
# Every code path that lists, displays, or iterates providers derives from
# this list:  hermes model, /model, list_authenticated_providers.
#
# Fields:
#   slug        — internal provider ID (used in config.yaml, --provider flag)
#   label       — short display name
#   tui_desc    — longer description for the `hermes model` interactive picker
# ---------------------------------------------------------------------------

class ProviderEntry(NamedTuple):
    slug: str
    label: str
    tui_desc: str   # detailed description for `hermes model` TUI


CANONICAL_PROVIDERS: list[ProviderEntry] = [
    ProviderEntry("nous",           "Nous Portal",              "Nous Portal (Everything your agent needs, 300+ models with bundled tool use)"),
    ProviderEntry("fireworks",      "Fireworks AI",             "Fireworks AI (OpenAI-compatible direct model API)"),
    ProviderEntry("openrouter",     "OpenRouter",               "OpenRouter (Pay-per-use API aggregator)"),
    ProviderEntry("moa",            "Mixture of Agents",        "Mixture of Agents (named presets; aggregator acts after reference models)"),
    ProviderEntry("novita",         "NovitaAI",                 "NovitaAI (Cloud: Model API, Agent Sandbox, GPU Cloud)"),
    ProviderEntry("lmstudio",       "LM Studio",                "LM Studio (Local desktop app with built-in model server)"),
    ProviderEntry("anthropic",      "Anthropic",                "Anthropic (Claude models via API key or Claude Code)"),
    ProviderEntry("openai-codex",   "ChatGPT or Codex Subscription", "ChatGPT or Codex Subscription (Sign in with your ChatGPT account, uses Codex models)"),
    ProviderEntry("openai-api",     "OpenAI API",               "OpenAI API (api.openai.com, API key)"),
    ProviderEntry("alibaba",        "Qwen Cloud",               "Qwen Cloud / DashScope (Qwen + multi-provider)"),
    ProviderEntry("xai-oauth",      "xAI Grok OAuth (SuperGrok / Premium+)", "xAI Grok OAuth (SuperGrok / Premium+ subscription)"),
    ProviderEntry("xiaomi",         "Xiaomi MiMo",              "Xiaomi MiMo (MiMo-V2.5 and V2 models: pro, omni, flash)"),
    ProviderEntry("tencent-tokenhub", "Tencent TokenHub",       "Tencent TokenHub (Hy4 preview via tokenhub.tencentmaas.com)"),
    ProviderEntry("tencent-tokenplan", "Tencent TokenPlan",     "Tencent TokenPlan (Hy4 preview via api.lkeap.cloud.tencent.com, Anthropic Messages)"),
    ProviderEntry("nvidia",         "NVIDIA NIM",               "NVIDIA NIM (Nemotron models via build.nvidia.com or local NIM)"),
    ProviderEntry("copilot",        "GitHub Copilot",           "GitHub Copilot (Uses GITHUB_TOKEN or gh auth token)"),
    ProviderEntry("copilot-acp",    "GitHub Copilot ACP",       "GitHub Copilot ACP (Spawns copilot --acp --stdio)"),
    ProviderEntry("huggingface",    "Hugging Face",             "Hugging Face Inference Providers"),
    ProviderEntry("gemini",         "Google AI Studio",         "Google AI Studio (Native Gemini API)"),
    ProviderEntry("vertex",         "Google Vertex AI",         "Google Vertex AI (Gemini via GCP; OAuth2 service account or ADC, GCP billing/quotas)"),
    ProviderEntry("deepseek",       "DeepSeek",                 "DeepSeek (V3, R1, coder, direct API)"),
    ProviderEntry("xai",            "xAI",                      "xAI Grok (Direct API)"),
    ProviderEntry("zai",            "Z.AI / GLM",               "Z.AI / GLM (Zhipu direct API)"),
    ProviderEntry("kimi-coding",    "Kimi / Kimi Coding Plan",  "Kimi Coding Plan (api.kimi.com & Moonshot API)"),
    ProviderEntry("kimi-coding-cn", "Kimi / Moonshot (China)",  "Kimi / Moonshot China (Domestic direct API)"),
    ProviderEntry("stepfun",        "StepFun Step Plan",       "StepFun Step Plan (Agent / coding models via Step Plan API)"),
    ProviderEntry("minimax",        "MiniMax",                  "MiniMax (Global direct API)"),
    ProviderEntry("minimax-oauth",  "MiniMax (OAuth)",          "MiniMax via OAuth browser login (Coding Plan, minimax.io)"),
    ProviderEntry("minimax-cn",     "MiniMax (China)",          "MiniMax China (Domestic direct API)"),
    ProviderEntry("ollama-cloud",   "Ollama Cloud",             "Ollama Cloud (Cloud-hosted open models, ollama.com)"),
    ProviderEntry("arcee",          "Arcee AI",                 "Arcee AI (Trinity models, direct API)"),
    ProviderEntry("gmi",            "GMI Cloud",                "GMI Cloud (Multi-model direct API)"),
    ProviderEntry("kilocode",       "Kilo Code",                "Kilo Code (Kilo Gateway API)"),
    ProviderEntry("opencode-zen",   "OpenCode Zen",             "OpenCode Zen (Curated models, pay-as-you-go)"),
    ProviderEntry("opencode-go",    "OpenCode Go",              "OpenCode Go (Open models subscription)"),
    ProviderEntry("bedrock",        "AWS Bedrock",              "AWS Bedrock (Claude, Nova, Llama, DeepSeek; IAM or API key)"),
    ProviderEntry("azure-foundry",  "Azure Foundry",            "Azure Foundry (OpenAI-style or Anthropic-style endpoint, your Azure AI deployment)"),
    ProviderEntry("ai-gateway",     "Vercel AI Gateway",        "Vercel AI Gateway (Multi-model aggregator)"),
    ProviderEntry("qwen-oauth",     "Qwen OAuth (Portal)",      "Qwen OAuth (Reuses local Qwen CLI login)"),
]


# Auto-extend CANONICAL_PROVIDERS with any provider registered in providers/
# that is not already in the list above.  Adding plugins/model-providers/<name>/
# is sufficient to expose a new provider in the model picker, /model, and all
# downstream consumers — no edits to this file needed.
_canonical_slugs = {p.slug for p in CANONICAL_PROVIDERS}


try:
    from providers import list_providers as _list_providers_for_canonical
    for _pp in _list_providers_for_canonical():
        if _pp.name in _canonical_slugs:
            continue
        if _pp.auth_type in {"oauth_device_code", "oauth_external", "external_process", "aws_sdk", "copilot", "vertex"}:
            continue  # non-api-key flows need bespoke picker UX; skip auto-inject
        _label = _pp.display_name or _pp.name
        _desc = _pp.description or f"{_label} (direct API)"
        CANONICAL_PROVIDERS.append(ProviderEntry(_pp.name, _label, _desc))
        _canonical_slugs.add(_pp.name)
except Exception:
    pass


# Derived dicts — used throughout the codebase
_PROVIDER_LABELS = {p.slug: p.label for p in CANONICAL_PROVIDERS}
_PROVIDER_LABELS["custom"] = "Custom endpoint"  # special case: not a named provider


# ---------------------------------------------------------------------------
# Provider groups — DISPLAY ONLY
#
# Some vendors expose several Hermes provider slugs (one per endpoint /
# auth method: global API, China API, OAuth coding plan, ...). Listing every
# slug as a top-level row in the interactive `hermes model` / setup wizard /
# Telegram `/model` pickers makes that list long and noisy.
#
# These groups fold related slugs under one top-level row in INTERACTIVE
# PICKERS only. They do NOT change ``CANONICAL_PROVIDERS``, slug identity,
# the ``--provider`` flag, ``/model <provider:model>``, or any typed path —
# every member slug remains individually addressable. Grouping is a pure
# display affordance; ``group_providers()`` is the single fold used by all
# three picker surfaces so they stay consistent.
#
#   group_id -> (display_label, group_description, [member_slug, ...])
#
# ``group_description`` is a short blurb shown on the collapsed top-level group
# row in the interactive pickers (alongside the label). Member-specific detail
# lives in each member's ``tui_desc`` and shows in the drill-down sub-picker.
# Member order is the order shown inside the group submenu.
# ---------------------------------------------------------------------------
PROVIDER_GROUPS: dict[str, tuple[str, str, list[str]]] = {
    "kimi":     ("Kimi / Moonshot", "Coding Plan, Moonshot global & China endpoints", ["kimi-coding", "kimi-coding-cn"]),
    "minimax":  ("MiniMax",         "Global, OAuth Coding Plan & China endpoints",     ["minimax", "minimax-oauth", "minimax-cn"]),
    "xai":      ("xAI Grok",        "Direct API or SuperGrok / Premium+ OAuth",        ["xai", "xai-oauth"]),
    "google":   ("Google Gemini",   "Google AI Studio (API key)",                     ["gemini"]),
    "openai":   ("OpenAI",          "ChatGPT/Codex subscription or direct OpenAI API", ["openai-codex", "openai-api"]),
    "qwen":     ("Qwen",            "Qwen Cloud / DashScope, Coding Plan, Token Plan & Qwen CLI OAuth", ["alibaba", "alibaba-cn", "alibaba-coding-plan", "alibaba-coding-plan-cn", "alibaba-token-plan", "alibaba-token-plan-cn", "qwen-oauth"]),
    "opencode": ("OpenCode",        "Zen pay-as-you-go, Go subscription, or free tier", ["opencode-zen", "opencode-go", "opencode-free"]),
    "copilot":  ("GitHub Copilot",  "GitHub token API or copilot --acp process",       ["copilot", "copilot-acp"]),
    "tencent":  ("Tencent Hy",      "Hy4 / Hy3 via TokenHub & TokenPlan", ["tencent-tokenhub", "tencent-tokenplan"]),
}


# Reverse index: member slug -> group_id. Built once at import.
_SLUG_TO_GROUP: dict[str, str] = {
    slug: gid for gid, (_label, _desc, members) in PROVIDER_GROUPS.items() for slug in members
}


def provider_group_for_slug(slug: str) -> str:
    """Return the group_id a provider slug belongs to, or "" if ungrouped."""
    return _SLUG_TO_GROUP.get(str(slug or "").strip().lower(), "")


def group_providers(slugs):
    """Fold a flat ordered slug iterable into picker rows by provider group.

    DISPLAY ONLY. Used by every interactive picker (``hermes model``, the setup wizard, the Telegram
    ``/model`` keyboard) so grouping is identical across surfaces.

    Rules: * A group row appears at the position of its FIRST present member, in the input order.
    Subsequent members fold into that row (and are not emitted again). * Member order inside a group
    follows ``PROVIDER_GROUPS`` declaration, restricted to the members actually present in
    ``slugs``.
    """
    seen: set[str] = set()
    # Which present members each group has, in declaration order.
    group_members: dict[str, list[str]] = {}
    for gid, (_label, _desc, members) in PROVIDER_GROUPS.items():
        present = [m for m in members if m in set(slugs)]
        if present:
            group_members[gid] = present

    rows = []
    emitted_groups: set[str] = set()
    for slug in slugs:
        s = str(slug or "").strip().lower()
        if not s or s in seen:
            continue
        seen.add(s)
        gid = _SLUG_TO_GROUP.get(s, "")
        if not gid:
            rows.append({"kind": "single", "slug": s})
            continue
        if gid in emitted_groups:
            continue  # already folded at the first member's position
        emitted_groups.add(gid)
        members = group_members.get(gid, [s])
        if len(members) <= 1:
            rows.append({"kind": "single", "slug": members[0]})
        else:
            label, desc, _ = PROVIDER_GROUPS[gid]
            rows.append(
                {"kind": "group", "group_id": gid, "label": label,
                 "description": desc, "members": list(members)}
            )
    return rows


_PROVIDER_ALIASES = {
    "glm": "zai",
    "z-ai": "zai",
    "z.ai": "zai",
    "zhipu": "zai",
    "github": "copilot",
    "github-copilot": "copilot",
    "github-models": "copilot",
    "github-model": "copilot",
    "github-copilot-acp": "copilot-acp",
    "copilot-acp-agent": "copilot-acp",
    "google": "gemini",
    "google-gemini": "gemini",
    "google-ai-studio": "gemini",
    "google-vertex": "vertex",
    "vertex-ai": "vertex",
    "gcp-vertex": "vertex",
    "vertexai": "vertex",
    "kimi": "kimi-coding",
    "moonshot": "kimi-coding",
    "kimi-cn": "kimi-coding-cn",
    "moonshot-cn": "kimi-coding-cn",
    "step": "stepfun",
    "stepfun-coding-plan": "stepfun",
    "arcee-ai": "arcee",
    "arceeai": "arcee",
    "gmi-cloud": "gmi",
    "gmicloud": "gmi",
    "fireworks-ai": "fireworks",
    "fw": "fireworks",
    "actual-computer": "actual",
    "actualcomputer": "actual",
    "aci": "actual",
    "nebius": "nebius-token-factory",
    "nebius-tokenfactory": "nebius-token-factory",
    "nebius-tf": "nebius-token-factory",
    "token-factory": "nebius-token-factory",
    "tokenfactory": "nebius-token-factory",
    "minimax-china": "minimax-cn",
    "minimax_cn": "minimax-cn",
    "minimax-portal": "minimax-oauth",
    "minimax-global": "minimax-oauth",
    "minimax_oauth": "minimax-oauth",
    "claude": "anthropic",
    "claude-code": "anthropic",
    "deep-seek": "deepseek",
    "opencode": "opencode-zen",
    "zen": "opencode-zen",
    "go": "opencode-go",
    "opencode-go-sub": "opencode-go",
    "free": "opencode-free",
    "opencode_free": "opencode-free",
    "aigateway": "ai-gateway",
    "vercel": "ai-gateway",
    "vercel-ai-gateway": "ai-gateway",
    "kilo": "kilocode",
    "kilo-code": "kilocode",
    "kilo-gateway": "kilocode",
    "dashscope": "alibaba",
    "aliyun": "alibaba",
    "qwen": "alibaba",
    "alibaba-cloud": "alibaba",
    "qwen-portal": "qwen-oauth",
    "hf": "huggingface",
    "hugging-face": "huggingface",
    "huggingface-hub": "huggingface",
    "novita-ai": "novita",
    "novitaai": "novita",
    "mimo": "xiaomi",
    "xiaomi-mimo": "xiaomi",
    "tencent": "tencent-tokenhub",
    "tokenhub": "tencent-tokenhub",
    "tencent-cloud": "tencent-tokenhub",
    "tencentmaas": "tencent-tokenhub",
    "tokenplan": "tencent-tokenplan",
    "tencent-lkeap": "tencent-tokenplan",
    "aws": "bedrock",
    "aws-bedrock": "bedrock",
    "amazon-bedrock": "bedrock",
    "amazon": "bedrock",
    "grok": "xai",
    "grok-oauth": "xai-oauth",
    "xai-oauth": "xai-oauth",
    "x-ai-oauth": "xai-oauth",
    "xai-grok-oauth": "xai-oauth",
    "x-ai": "xai",
    "x.ai": "xai",
    "nim": "nvidia",
    "nvidia-nim": "nvidia",
    "build-nvidia": "nvidia",
    "nemotron": "nvidia",
    "lmstudio": "lmstudio",
    "lm-studio": "lmstudio",
    "lm_studio": "lmstudio",
    "ollama": "custom",  # bare "ollama" = local; use "ollama-cloud" for cloud
    "ollama_cloud": "ollama-cloud",
}


# In-repo fallback for the model Hermes silently lands on when the user never
# picked one (GUI onboarding confirm card, empty ``model.default``,
# provider-set-but-model-missing resolution). The AUTHORITATIVE source is the
# remote model catalog: the manifest labels exactly one entry per provider
# with ``"default": true`` (see get_default_model_from_cache in
# model_catalog.py), so maintainers can rotate the default without shipping a
# release. This constant is the offline/fresh-install fallback and MUST match
# the labeled entry in website/static/api/model-catalog.json. Deliberately a
# capable low-cost model rather than the curated lists' entry [0]: aggregator
# lists are ordered most-capable-first, so [0] is the priciest Anthropic
# flagship (claude-fable-5 / opus) — silently billing the most expensive model
# for traffic the user never opted into.
PREFERRED_SILENT_DEFAULT_MODEL = "z-ai/glm-5.2"


# Providers whose *silent* auto-default must go through the cost-safe
# catalog-labeled default (``get_preferred_silent_default_model``) instead of
# curated-list entry [0]. Metered aggregators (Nous Portal, OpenRouter) order
# their lists best-/most-capable-first — entry [0] is the priciest flagship
# (``anthropic/claude-fable-5``). Using that as the non-interactive fallback
# when a profile sets a provider with no model silently bills the most
# expensive model for traffic the user never opted into (a missing default
# escalated to Opus and billed 863 requests before the user noticed). The
# catalog manifest labels the default entry (``"default": true``) so it can
# rotate without a release; a missing model must never escalate to the
# flagship.
#
# This is deliberately a network-free lookup for the hot resolution path
# (cache-only catalog read). The *interactive* default (GUI onboarding /
# ``hermes model``) uses the richer free/paid-tier-aware resolver — see
# ``get_recommended_default_model`` in hermes_cli/web_server.py and
# ``partition_nous_models_by_tier`` — which can hit the Portal.
_SILENT_DEFAULT_PROVIDERS: frozenset[str] = frozenset({"nous", "openrouter"})


# Retired model IDs kept for /model auto-detect only — not shown in pickers.
# DeepSeek cut these off on 2026-07-24; model_normalize remaps them on the wire.
_PROVIDER_RETIRED_ALIASES: dict[str, tuple[str, ...]] = {
    "deepseek": ("deepseek-chat", "deepseek-reasoner"),
}


_AGGREGATOR_PROVIDERS = frozenset(
    {"nous", "openrouter", "ai-gateway", "copilot", "kilocode"}
)


# OpenRouter request-time routing variants (docs: guides/routing/model-variants).
# These suffixes are per-request routing modifiers valid on ANY model id —
# ":nitro" sorts the endpoint pool by throughput and admits priority-tier
# endpoints, ":floor" sorts by price and admits flex-tier endpoints, ":exacto"
# applies quality-first provider sorting, ":online" attaches the web plugin.
# They are never separate catalog entries: /models lists only the base id.
# NOT in this set: ":free", ":batch", ":thinking", ":extended" — those ARE
# distinct catalog SKUs that appear in /models when they exist, so absence
# from the listing is authoritative for them and the direct-membership check
# above handles the valid ones.
_OPENROUTER_VARIANT_SUFFIXES = frozenset({"nitro", "floor", "exacto", "online"})


# Subscription/OAuth providers whose catalogs RE-EXPOSE other vendors' models
# would be listed here (tried only as a last resort for bare short-alias
# resolution, after every native-vendor catalog, so they never hijack an alias
# away from the model's native vendor). None are currently defined.
_BORROWED_MODEL_PROVIDERS: frozenset[str] = frozenset()


# Providers whose live /v1/models endpoint is the authoritative catalog, so the
# curated list is a discovery-only fallback. For these, the picker merges
# live-first (live entries lead, curated-only entries append). Every OTHER
# provider keeps curated-first (commit 658ac1d86, #46309) so a deliberately
# surfaced newest model stays at the top even when the live API lags. OpenCode
# Zen / Go re-expose dozens of upstream vendors and rotate them frequently, so
# their stale curated entries must not pollute the top of the picker. (#49129)
_LIVE_FIRST_PICKER_PROVIDERS: frozenset[str] = frozenset(
    {"opencode-zen", "opencode-go"}
)


# Models that support OpenAI Priority Processing (service_tier="priority").
# See https://openai.com/api-priority-processing/ for the canonical list.
#
# Pattern-based matching — any OpenAI flagship model (gpt-*, o1*, o3*, o4*)
# is assumed to support Priority Processing. service_tier=priority is silently
# ignored by non-OpenAI endpoints (OpenRouter/Copilot/opencode-zen proxies
# strip the field), so false positives are harmless. Codex-series models
# (gpt-5-codex, gpt-5.3-codex, etc.) are excluded — they don't expose the
# service_tier parameter through the Codex Responses API.
_OPENAI_FAST_MODE_PREFIXES: tuple[str, ...] = (
    "gpt-",
    "o1",
    "o3",
    "o4",
)


# Providers where models.dev is treated as authoritative: curated static
# lists are kept only as an offline fallback and to capture custom additions
# the registry doesn't publish yet. Adding a provider here causes its
# curated list to be merged with fresh models.dev entries (fresh first, any
# curated-only names appended) for both the CLI and the gateway /model picker.
#
# DELIBERATELY EXCLUDED:
#   - "openrouter": curated list is already a hand-picked agentic subset of
#     OpenRouter's 400+ catalog. Blindly merging would dump everything.
#   - "nous": curated list and Portal /models endpoint are the source of
#     truth for the subscription tier.
# Also excluded: providers that already have dedicated live-endpoint
# branches below (copilot, anthropic, ai-gateway, ollama-cloud, custom,
# stepfun, openai-codex) — those paths handle freshness themselves.
_MODELS_DEV_PREFERRED: frozenset[str] = frozenset({
    "opencode-go",
    "opencode-zen",
    "deepseek",
    "kilocode",
    "fireworks",
    "mistral",
    "togetherai",
    "cohere",
    "perplexity",
    "groq",
    "nvidia",
    "huggingface",
    "zai",
    "gemini",
    "google",
    "xai",
    "xai-oauth",
})


# Providers whose catalog is served with NO credential and therefore gets a
# stable (constant) credential fingerprint in the disk cache. The opencode-free
# catalog is anonymous — its freshness comes from TTL revalidation, not from
# user-rotatable credentials — so folding in unrelated auth.json mtimes would
# only needlessly bust the SWR cache.
_KEYLESS_STABLE_CACHE_PROVIDERS = frozenset({"opencode-free"})


_COPILOT_MODEL_ALIASES = {
    "openai/gpt-5": "gpt-5-mini",
    "openai/gpt-5-chat": "gpt-5-mini",
    "openai/gpt-5-mini": "gpt-5-mini",
    "openai/gpt-5-nano": "gpt-5-mini",
    "openai/gpt-4.1": "gpt-4.1",
    "openai/gpt-4.1-mini": "gpt-4.1",
    "openai/gpt-4.1-nano": "gpt-4.1",
    "openai/gpt-4o": "gpt-4o",
    "openai/gpt-4o-mini": "gpt-4o-mini",
    "openai/o1": "gpt-5.2",
    "openai/o1-mini": "gpt-5-mini",
    "openai/o1-preview": "gpt-5.2",
    "openai/o3": "gpt-5.3-codex",
    "openai/o3-mini": "gpt-5-mini",
    "openai/o4-mini": "gpt-5-mini",
    "anthropic/claude-opus-4.6": "claude-opus-4.6",
    "anthropic/claude-sonnet-5": "claude-sonnet-5",
    "anthropic/claude-sonnet-4.6": "claude-sonnet-4.6",
    "anthropic/claude-sonnet-4": "claude-sonnet-4",
    "anthropic/claude-sonnet-4.5": "claude-sonnet-4.5",
    "anthropic/claude-haiku-4.5": "claude-haiku-4.5",
    # Dash-notation fallbacks: Hermes' default Claude IDs elsewhere use
    # hyphens (anthropic native format), but Copilot's API only accepts
    # dot-notation.  Accept both so users who configure copilot + a
    # default hyphenated Claude model don't hit HTTP 400
    # "model_not_supported".  See issue #6879.
    "claude-sonnet-5": "claude-sonnet-5",
    "claude-opus-4-6": "claude-opus-4.6",
    "claude-sonnet-4-6": "claude-sonnet-4.6",
    "claude-sonnet-4-0": "claude-sonnet-4",
    "claude-sonnet-4-5": "claude-sonnet-4.5",
    "claude-haiku-4-5": "claude-haiku-4.5",
    "anthropic/claude-opus-4-6": "claude-opus-4.6",
    "anthropic/claude-sonnet-5": "claude-sonnet-5",
    "anthropic/claude-sonnet-4-6": "claude-sonnet-4.6",
    "anthropic/claude-sonnet-4-0": "claude-sonnet-4",
    "anthropic/claude-sonnet-4-5": "claude-sonnet-4.5",
    "anthropic/claude-haiku-4-5": "claude-haiku-4.5",
}


# Azure Foundry model families that require the Responses API.  Azure
# rejects /chat/completions against these deployments with
# ``400 "The requested operation is unsupported."`` — the same payload Bob
# Dobolina hit in April 2026 on ``gpt-5.3-codex`` while ``gpt-4o-pure`` on
# the same endpoint worked fine.  Keep the patterns broad enough to cover
# vendor-renamed deployments (e.g. ``gpt-5.3-codex``, ``gpt-5-codex``,
# ``gpt-5.4``, ``o1-preview``) but tight enough to leave GPT-4 / 3.5 / Llama /
# Mistral / Grok deployments on chat completions.
_AZURE_FOUNDRY_RESPONSES_PREFIXES = (
    "codex",       # codex-*, codex-mini
    "gpt-5",       # gpt-5, gpt-5.x, gpt-5-codex, gpt-5.x-codex
    "o1",          # o1, o1-preview, o1-mini
    "o3",          # o3, o3-mini
    "o4",          # o4, o4-mini
)
