"""Shared base classes for the pluggable-backend provider ABCs.

Every tool-provider ABC (browser, TTS, image/video gen, transcription, web
search, terminal env) shares the same identity + ``hermes tools`` picker
surface; the previous per-ABC copies of these defaults were byte-identical.
Concrete ABCs subclass :class:`ProviderBase` (or :class:`CatalogProviderBase`
when the backend also exposes a model catalog and is available by default) and
add only their domain methods. Plugins keep subclassing the concrete ABC, so
``isinstance`` checks and abstract-method sets are unchanged.
"""

from __future__ import annotations

import abc
from typing import Any, Dict, List, Optional


class ProviderBase(abc.ABC):
    """Identity + picker metadata common to every provider ABC."""

    @property
    @abc.abstractmethod
    def name(self) -> str:
        """Stable short identifier used as the provider's config-key value.

        Lowercase, no spaces (hyphens allowed where they preserve an existing
        user-visible name). Registries key providers by this string.
        """

    @property
    def display_name(self) -> str:
        """Human-readable label shown in ``hermes tools``. Defaults to ``name``."""
        return self.name

    def get_setup_schema(self) -> Dict[str, Any]:
        """Provider row for the ``hermes tools`` picker.

        Shape: ``{"name", "badge", "tag", "env_vars": [{"key", "prompt", "url"}, ...]}``
        (browser providers may add ``"post_setup"``). Default: a minimal entry
        derived from ``display_name`` with no env vars — override to expose API
        key prompts and badges.
        """
        return {
            "name": self.display_name,
            "badge": "",
            "tag": "",
            "env_vars": [],
        }


class CatalogProviderBase(ProviderBase):
    """Provider with a model catalog; available by default, ``display_name`` is title-cased."""

    @property
    def display_name(self) -> str:
        """Human-readable label shown in ``hermes tools``. Defaults to ``name.title()``."""
        return self.name.title()

    def is_available(self) -> bool:
        """True when this provider can service calls (API key present, SDK importable).

        Default True. Must NOT raise and must NOT make network calls — the picker
        and ``hermes setup`` call it on every paint.
        """
        return True

    def list_models(self) -> List[Dict[str, Any]]:
        """Model catalog entries (``{"id": ..., "display": ...}`` plus optional
        provider-specific keys). Default: empty (no user-selectable models)."""
        return []

    def default_model(self) -> Optional[str]:
        """Id of the first catalog entry, or None when the catalog is empty."""
        models = self.list_models()
        if models:
            return models[0].get("id")
        return None
