"""Raw httpx client — the floor. No gateway in the loop."""

from __future__ import annotations

from typing import Any

import httpx


class RawHttpxClient:
    name = "raw-httpx"

    def __init__(self, base_url: str) -> None:
        self._client = httpx.AsyncClient(
            base_url=base_url,
            http2=True,
            limits=httpx.Limits(
                max_keepalive_connections=50, max_connections=200, keepalive_expiry=30.0
            ),
            timeout=30.0,
            headers={"authorization": "Bearer mock-key"},
        )

    async def chat(self, prompt: str) -> dict[str, Any]:
        resp = await self._client.post(
            "/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": prompt}],
            },
        )
        resp.raise_for_status()
        return resp.json()

    async def stream_first_token(self, prompt: str) -> tuple[float, int]:
        """Returns (ttft_seconds, total_chunks)."""
        import time

        async with self._client.stream(
            "POST",
            "/chat/completions",
            json={
                "model": "mock-model",
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            },
        ) as resp:
            start = time.perf_counter()
            chunks = 0
            ttft: float | None = None
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                if line[6:].strip() == "[DONE]":
                    break
                chunks += 1
                if ttft is None and "content" in line:
                    ttft = time.perf_counter() - start
            return (ttft or 0.0), chunks

    async def aclose(self) -> None:
        await self._client.aclose()
