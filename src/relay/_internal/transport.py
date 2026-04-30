"""Shared HTTP transport.

One :class:`httpx.AsyncClient` per ``(provider, base_url)`` keyed pool — connection
reuse and HTTP/2 multiplexing matter a lot for streaming workloads (the research
report flagged this as a top latency lever).
"""

from __future__ import annotations

import asyncio
from typing import Any

import httpx

from relay.config._schema import GlobalDefaults

_PoolKey = tuple[str, str]


class HttpClientManager:
    """Process-wide manager of pooled httpx clients.

    Clients are created on first use and reused for the lifetime of the manager.
    A single :class:`Hub` instance owns one of these and closes it on shutdown.
    """

    def __init__(self, defaults: GlobalDefaults) -> None:
        self._defaults = defaults
        self._clients: dict[_PoolKey, httpx.AsyncClient] = {}
        self._lock = asyncio.Lock()
        self._closed = False

    async def get(
        self,
        *,
        provider: str,
        base_url: str,
        timeout: float | None = None,
        extra_headers: dict[str, str] | None = None,
    ) -> httpx.AsyncClient:
        if self._closed:
            raise RuntimeError("HttpClientManager has been closed")
        key = (provider, base_url)
        if key in self._clients:
            return self._clients[key]
        async with self._lock:
            if key in self._clients:
                return self._clients[key]
            client = httpx.AsyncClient(
                base_url=base_url,
                http2=self._defaults.http2,
                timeout=httpx.Timeout(timeout or self._defaults.timeout),
                limits=httpx.Limits(
                    max_keepalive_connections=self._defaults.pool_max_keepalive,
                    max_connections=self._defaults.pool_max_connections,
                    keepalive_expiry=self._defaults.keepalive_expiry,
                ),
                headers={
                    "user-agent": "relay-py/0.1",
                    **(extra_headers or {}),
                },
            )
            self._clients[key] = client
            return client

    async def aclose(self) -> None:
        self._closed = True
        async with self._lock:
            clients = list(self._clients.values())
            self._clients.clear()
        # Close outside the lock to avoid holding it during network I/O.
        await asyncio.gather(*(c.aclose() for c in clients), return_exceptions=True)


def parse_retry_after(headers: httpx.Headers) -> float | None:
    """Parse a ``Retry-After`` header (seconds or HTTP-date — we only handle seconds)."""
    raw = headers.get("retry-after")
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def detect_limit_type(body: dict[str, Any] | str | None, headers: httpx.Headers) -> str | None:
    """Heuristic — distinguish RPM, TPM, and concurrency limits from a 429.

    OpenAI sets ``x-ratelimit-*`` headers; Anthropic includes ``rate_limit_type`` in
    error JSON; many others put it in the message text.
    """
    if isinstance(body, dict):
        err = body.get("error") or {}
        if isinstance(err, dict):
            msg = (err.get("message") or "").lower()
            err_type = (err.get("type") or "").lower()
            if "concurren" in msg or "concurren" in err_type:
                return "concurrency"
            if "tokens" in msg or "tpm" in msg or "tokens_per_min" in err_type:
                return "tpm"
            if "request" in msg or "rpm" in msg or "requests_per_min" in err_type:
                return "rpm"
    # Header heuristics.
    if "x-ratelimit-remaining-tokens" in headers:
        rem = headers.get("x-ratelimit-remaining-tokens")
        if rem and rem.strip() in {"0", "0.0"}:
            return "tpm"
    if "x-ratelimit-remaining-requests" in headers:
        rem = headers.get("x-ratelimit-remaining-requests")
        if rem and rem.strip() in {"0", "0.0"}:
            return "rpm"
    return None
