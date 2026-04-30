"""Relay client wrapper for benchmarking."""

from __future__ import annotations

import os
from typing import Any

from relay import Hub
from relay.config import load_str
from relay.types import StreamEnd, TextDelta


def _yaml(base_url: str) -> str:
    os.environ.setdefault("BENCH_KEY", "mock-key")
    return f"""
        version: 1
        catalog:
          fetch_live_pricing: false
          offline: true
        models:
          mock:
            target: openai/mock-model
            base_url: {base_url}
            credential: $env.BENCH_KEY
    """


class RelayClient:
    name = "relay"

    def __init__(self, base_url: str) -> None:
        self._hub = Hub.from_config(load_str(_yaml(base_url)))

    async def chat(self, prompt: str) -> Any:
        return await self._hub.chat("mock", messages=[{"role": "user", "content": prompt}])

    async def stream_first_token(self, prompt: str) -> tuple[float, int]:
        import time

        start = time.perf_counter()
        chunks = 0
        ttft: float | None = None
        async for ev in self._hub.stream("mock", messages=[{"role": "user", "content": prompt}]):
            if isinstance(ev, TextDelta):
                chunks += 1
                if ttft is None:
                    ttft = time.perf_counter() - start
            elif isinstance(ev, StreamEnd):
                break
        return (ttft or 0.0), chunks

    async def aclose(self) -> None:
        await self._hub.aclose()
