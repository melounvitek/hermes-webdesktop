"""Pipeline-facing Teams outbound delivery (meeting-summary writer).

Lives inside the Teams platform plugin so the meeting pipeline reuses one Teams
integration surface. httpx is imported lazily: plugin discovery imports this
module on every CLI start, but only ``incoming_webhook`` delivery needs it.
"""

from __future__ import annotations

import html
import os
from typing import Any, Optional
from urllib.parse import quote

from gateway.config import PlatformConfig
from gateway.platforms._shared import get_scoped_secret as _get_scoped_secret


def _parse_bool(value: Any, *, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off"}:
            return False
    return default


_LIST_SECTIONS = (("Key decisions", "key_decisions"), ("Action items", "action_items"), ("Risks", "risks"))


class _StaticAccessTokenProvider:
    """Minimal token-provider shim so outbound Graph delivery can reuse the shared client."""

    def __init__(self, access_token: str):
        self._access_token = str(access_token or "").strip()

    async def get_access_token(self, *, force_refresh: bool = False) -> str:
        del force_refresh
        if not self._access_token:
            raise ValueError("TEAMS_GRAPH_ACCESS_TOKEN is required for graph delivery mode.")
        return self._access_token

    def clear_cache(self) -> None:
        return None


class TeamsSummaryWriter:
    """Deliver a meeting summary to Teams via incoming webhook or Graph."""

    def __init__(
        self, platform_config: PlatformConfig | None = None, *,
        graph_client: Any | None = None, transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._platform_config = platform_config
        self._graph_client = graph_client
        self._transport = transport

    async def write_summary(
        self, payload: Any, config: dict[str, Any] | None, existing_record: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        merged = self._resolve_delivery_config(config)
        if existing_record and not _parse_bool(merged.get("force_resend"), default=False):
            return dict(existing_record)
        mode = str(merged.get("delivery_mode") or merged.get("mode") or "").strip().lower()
        if not mode:
            if merged.get("incoming_webhook_url"):
                mode = "incoming_webhook"
            elif merged.get("chat_id") or (merged.get("team_id") and merged.get("channel_id")):
                mode = "graph"
        if mode == "incoming_webhook":
            return await self._write_summary_via_incoming_webhook(payload, merged)
        if mode == "graph":
            return await self._write_summary_via_graph(payload, merged)
        raise ValueError("Teams delivery_mode must be 'incoming_webhook' or 'graph'.")

    def _resolve_delivery_config(self, config: dict[str, Any] | None) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        platform_cfg = self._platform_config
        if platform_cfg is not None:
            merged.update(dict(platform_cfg.extra or {}))
            if platform_cfg.token and "access_token" not in merged:
                merged["access_token"] = platform_cfg.token
            if platform_cfg.home_channel:
                merged.setdefault("channel_id", platform_cfg.home_channel.chat_id)
        merged.update(dict(config or {}))
        env_defaults = {
            "delivery_mode": os.getenv("TEAMS_DELIVERY_MODE", ""),
            "incoming_webhook_url": os.getenv("TEAMS_INCOMING_WEBHOOK_URL", ""),
            "access_token": _get_scoped_secret("TEAMS_GRAPH_ACCESS_TOKEN", ""),
            "team_id": os.getenv("TEAMS_TEAM_ID", ""),
            "channel_id": os.getenv("TEAMS_CHANNEL_ID", ""),
            "chat_id": os.getenv("TEAMS_CHAT_ID", ""),
        }
        for key, value in env_defaults.items():
            if value and not merged.get(key):
                merged[key] = value
        return merged

    async def _write_summary_via_incoming_webhook(self, payload: Any, config: dict[str, Any]) -> dict[str, Any]:
        import httpx  # lazy — see module docstring

        webhook_url = str(config.get("incoming_webhook_url") or "").strip()
        if not webhook_url:
            raise ValueError("TEAMS_INCOMING_WEBHOOK_URL is required for incoming_webhook mode.")
        body = {"text": self._render_summary_markdown(payload)}
        async with httpx.AsyncClient(timeout=20.0, transport=self._transport) as client:
            response = await client.post(webhook_url, json=body)
            response.raise_for_status()
        return {
            "delivery_mode": "incoming_webhook", "webhook_url": webhook_url,
            "status_code": response.status_code, "delivered": True,
        }

    async def _write_summary_via_graph(self, payload: Any, config: dict[str, Any]) -> dict[str, Any]:
        graph_client = self._build_graph_client(config)
        chat_id = str(config.get("chat_id") or "").strip()
        if chat_id:
            path = f"/chats/{quote(chat_id, safe='')}/messages"
            target = {"target_type": "chat", "chat_id": chat_id}
        else:
            team_id = str(config.get("team_id") or "").strip()
            channel_id = str(config.get("channel_id") or "").strip()
            if not team_id or not channel_id:
                raise ValueError("Graph delivery mode requires chat_id, or both team_id and channel_id.")
            path = f"/teams/{quote(team_id, safe='')}/channels/{quote(channel_id, safe='')}/messages"
            target = {"target_type": "channel", "team_id": team_id, "channel_id": channel_id}
        response = await graph_client.post_json(
            path,
            json_body={"body": {"contentType": "html", "content": self._render_summary_html(payload)}},
        )
        return {
            "delivery_mode": "graph", **target,
            "message_id": (response or {}).get("id"), "web_url": (response or {}).get("webUrl"),
        }

    def _build_graph_client(self, config: dict[str, Any]) -> Any:
        if self._graph_client is not None:
            return self._graph_client
        from tools.microsoft_graph_auth import MicrosoftGraphTokenProvider
        from tools.microsoft_graph_client import MicrosoftGraphClient

        access_token = str(config.get("access_token") or "").strip()
        if access_token:
            return MicrosoftGraphClient(_StaticAccessTokenProvider(access_token), transport=self._transport)
        return MicrosoftGraphClient(MicrosoftGraphTokenProvider.from_env(), transport=self._transport)

    def _render_summary_markdown(self, payload: Any) -> str:
        lines = [
            f"**{self._title(payload)}**",
            "",
            f"Summary: {self._text(getattr(payload, 'summary', None), 'No summary available.')}",
        ]
        for heading, attr in _LIST_SECTIONS:
            lines += ["", f"{heading}:", *self._bullet_lines(getattr(payload, attr, None))]
        return "\n".join(lines)

    def _render_summary_html(self, payload: Any) -> str:
        sections = [
            ("Summary", [self._text(getattr(payload, "summary", None), "No summary available.")]),
            *((heading, list(getattr(payload, attr, None) or [])) for heading, attr in _LIST_SECTIONS),
        ]
        blocks = [f"<h2>{html.escape(self._title(payload))}</h2>"]
        for heading, items in sections:
            blocks.append(f"<h3>{html.escape(heading)}</h3>")
            if len(items) == 1 and heading == "Summary":
                blocks.append(f"<p>{html.escape(str(items[0]))}</p>")
                continue
            if items:
                rendered = "".join(f"<li>{html.escape(str(item))}</li>" for item in items if str(item).strip())
                blocks.append(rendered and f"<ul>{rendered}</ul>" or "<p>None</p>")
            else:
                blocks.append("<p>None</p>")
        return "".join(blocks)

    @staticmethod
    def _title(payload: Any) -> str:
        title = getattr(payload, "title", None)
        if title:
            return str(title)
        meeting_ref = getattr(payload, "meeting_ref", None)
        meeting_id = getattr(meeting_ref, "meeting_id", None) if meeting_ref else None
        return f"Meeting {meeting_id or 'summary'}"

    @staticmethod
    def _text(value: Any, default: str) -> str:
        text = str(value or "").strip()
        return text or default

    @classmethod
    def _bullet_lines(cls, values: Any) -> list[str]:
        items = [str(item).strip() for item in (values or []) if str(item).strip()]
        return [f"- {item}" for item in items] or ["- None"]
