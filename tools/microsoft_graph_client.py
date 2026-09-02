"""Reusable Microsoft Graph REST client helpers."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

import httpx

from agent.retry_utils import parse_retry_after_seconds
from tools.microsoft_graph_auth import (
    GraphCredentials,
    MicrosoftGraphTokenProvider,
    format_graph_error,
)


DEFAULT_GRAPH_BASE_URL = "https://graph.microsoft.com/v1.0"

Headers = dict[str, str] | None
Params = dict[str, Any] | None


class MicrosoftGraphClientError(RuntimeError):
    """Base class for Graph client failures."""


class MicrosoftGraphAPIError(MicrosoftGraphClientError):
    """Raised when a Graph API request fails."""

    def __init__(
        self, status_code: int, method: str, url: str, message: str, *,
        retry_after_seconds: float | None = None, payload: Any = None,
    ) -> None:
        self.status_code = status_code
        self.method = method
        self.url = url
        self.retry_after_seconds = retry_after_seconds
        self.payload = payload
        super().__init__(f"Microsoft Graph API error {status_code} for {method} {url}: {message}")


class MicrosoftGraphClient:
    """Minimal async Microsoft Graph client with retries and pagination.

    Retry policy (shared by JSON requests and streaming downloads): transport
    errors back off exponentially; 401 clears the token cache and refetches;
    429/5xx honor ``Retry-After``. Each attempt uses a fresh ``AsyncClient``.
    """

    def __init__(
        self, token_provider: MicrosoftGraphTokenProvider, *,
        base_url: str = DEFAULT_GRAPH_BASE_URL, timeout: float = 60.0, max_retries: int = 3,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] | None = None,
        user_agent: str = "Hermes-Agent/graph-client",
    ) -> None:
        self.token_provider = token_provider
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max(0, int(max_retries))
        self._transport = transport
        self._sleep = sleep or asyncio.sleep
        self.user_agent = user_agent

    @classmethod
    def from_env(cls, **kwargs: Any) -> "MicrosoftGraphClient":
        return cls(MicrosoftGraphTokenProvider(GraphCredentials.from_env()), **kwargs)

    async def get_json(self, path: str, *, params: Params = None, headers: Headers = None) -> Any:
        return self._decode_json(await self._request("GET", path, params=params, headers=headers))

    async def post_json(self, path: str, *, json_body: Any | None = None, headers: Headers = None) -> Any:
        return self._decode_json(await self._request("POST", path, json_body=json_body, headers=headers))

    async def patch_json(self, path: str, *, json_body: Any | None = None, headers: Headers = None) -> Any:
        response = await self._request("PATCH", path, json_body=json_body, headers=headers)
        if response.status_code == 204 or not response.content:
            return {}
        return self._decode_json(response)

    async def delete(self, path: str, *, headers: Headers = None) -> dict[str, Any]:
        response = await self._request("DELETE", path, headers=headers)
        if response.status_code == 204 or not response.content:
            return {"deleted": True, "status_code": response.status_code}
        return self._decode_json(response)

    async def iterate_pages(
        self, path: str, *, params: Params = None, headers: Headers = None
    ) -> AsyncIterator[dict[str, Any]]:
        # Query params go on the first request only; @odata.nextLink already embeds them.
        next_url: str | None = self._resolve_url(path)
        next_params = dict(params or {})
        while next_url:
            response = await self._request("GET", next_url, params=next_params or None, headers=headers)
            payload = self._decode_json(response)
            if not isinstance(payload, dict):
                raise MicrosoftGraphClientError(
                    f"Expected paginated Graph response dict, got {type(payload).__name__}."
                )
            yield payload
            next_url = payload.get("@odata.nextLink")
            next_params = {}

    async def collect_paginated(
        self, path: str, *, params: Params = None, headers: Headers = None
    ) -> list[Any]:
        items: list[Any] = []
        async for page in self.iterate_pages(path, params=params, headers=headers):
            value = page.get("value")
            if isinstance(value, list):
                items.extend(value)
        return items

    async def download_to_file(
        self, path: str, destination: str | Path, *, headers: Headers = None, chunk_size: int = 65536
    ) -> dict[str, Any]:
        """Download a Graph resource to disk, streaming the body chunk-by-chunk
        (recordings and other large artifacts never need to fit in memory).
        Written to a ``.part`` file and renamed into place only on success."""
        url = self._resolve_url(path)
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        tmp_target = target.with_suffix(target.suffix + ".part")

        async def perform(client: httpx.AsyncClient, request_headers: dict[str, str]):
            try:
                async with client.stream("GET", url, headers=request_headers) as response:
                    if response.status_code >= 400:
                        # Materialize the (small) error body so the message is meaningful.
                        await response.aread()
                        return response, None
                    with tmp_target.open("wb") as handle:
                        async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                            if chunk:
                                handle.write(chunk)
                    return response, response.headers.get("content-type")
            except httpx.HTTPError:
                tmp_target.unlink(missing_ok=True)
                raise

        content_type = await self._with_retries("GET", url, "*/*", None, headers, perform, "download")
        os.replace(tmp_target, target)
        return {"path": str(target), "size_bytes": target.stat().st_size, "content_type": content_type}

    async def _request(
        self, method: str, path_or_url: str, *,
        params: Params = None, json_body: Any | None = None, headers: Headers = None,
    ) -> httpx.Response:
        url = self._resolve_url(path_or_url)

        async def perform(client: httpx.AsyncClient, request_headers: dict[str, str]):
            response = await client.request(method, url, params=params, json=json_body, headers=request_headers)
            return response, response

        return await self._with_retries(method, url, "application/json", json_body, headers, perform, "request")

    async def _with_retries(
        self, method: str, url: str, accept: str, json_body: Any | None, headers: Headers,
        perform: Callable[[httpx.AsyncClient, dict[str, str]], Awaitable[tuple[httpx.Response, Any]]],
        kind: str,
    ) -> Any:
        """Run ``perform`` (returning ``(response, result)``) under the retry policy.

        ``kind`` ("request"/"download") only labels the transport-failure messages.
        A ``MicrosoftGraphAPIError`` for the failing status is raised once retries
        are exhausted or the status is not retryable; only a 401 forces a token refresh.
        """
        attempt = 0
        last_error: Exception | None = None

        while attempt <= self.max_retries:
            token = await self.token_provider.get_access_token(
                force_refresh=attempt > 0
                and isinstance(last_error, MicrosoftGraphAPIError)
                and last_error.status_code == 401
            )
            request_headers = {"Authorization": f"Bearer {token}", "Accept": accept, "User-Agent": self.user_agent}
            if json_body is not None:
                request_headers["Content-Type"] = "application/json"
            if headers:
                request_headers.update(headers)

            try:
                async with httpx.AsyncClient(timeout=httpx.Timeout(self.timeout), transport=self._transport) as client:
                    response, result = await perform(client, request_headers)
            except httpx.HTTPError as exc:
                last_error = exc
                if attempt >= self.max_retries:
                    raise MicrosoftGraphClientError(
                        f"Microsoft Graph {kind} failed for {method} {url}: {exc}"
                    ) from exc
                await self._sleep(self._retry_delay(None, attempt))
                attempt += 1
                continue

            if response.status_code < 400:
                return result

            api_error = last_error = self._build_api_error(method, url, response)
            status = response.status_code
            if attempt < self.max_retries and (status in (401, 429) or 500 <= status < 600):
                if status == 401:
                    self.token_provider.clear_cache()
                await self._sleep(self._retry_delay(response, attempt))
                attempt += 1
                continue
            raise api_error

        raise MicrosoftGraphClientError(f"Microsoft Graph {kind} exhausted retries for {method} {url}.")

    def _resolve_url(self, path_or_url: str) -> str:
        if path_or_url.startswith(("http://", "https://")):
            return path_or_url
        path = path_or_url if path_or_url.startswith("/") else f"/{path_or_url}"
        return f"{self.base_url}{path}"

    @staticmethod
    def _decode_json(response: httpx.Response) -> Any:
        try:
            return response.json()
        except ValueError as exc:
            raise MicrosoftGraphClientError(
                "Microsoft Graph response was not valid JSON for "
                f"{response.request.method} {response.request.url}"
            ) from exc

    @staticmethod
    def _retry_delay(response: httpx.Response | None, attempt: int) -> float:
        if response is not None:
            retry_after = parse_retry_after_seconds(response.headers)
            if retry_after is not None:
                return retry_after
        return min(8.0, 0.5 * (2 ** attempt))

    @staticmethod
    def _build_api_error(method: str, url: str, response: httpx.Response) -> MicrosoftGraphAPIError:
        message = response.text.strip() or "unknown error"
        try:
            payload: Any = response.json()
        except ValueError:
            payload = None
        if isinstance(payload, dict):
            detail = format_graph_error(payload.get("error"))
            if detail is not None:
                message = detail
        return MicrosoftGraphAPIError(
            response.status_code, method, url, message,
            retry_after_seconds=parse_retry_after_seconds(response.headers), payload=payload,
        )
