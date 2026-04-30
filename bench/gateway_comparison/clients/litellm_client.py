"""LiteLLM client wrapper for benchmarking."""

from __future__ import annotations

import os
from typing import Any


class LiteLLMClient:
    name = "litellm"

    def __init__(self, base_url: str) -> None:
        # litellm reads OPENAI_API_KEY from env and OPENAI_BASE_URL via api_base.
        os.environ["OPENAI_API_KEY"] = "mock-key"
        self._base_url = base_url
        # Lazy-import to keep cold-start fair when the harness is initializing
        # multiple clients.
        import litellm  # type: ignore[import-untyped,import-not-found]

        # Disable LiteLLM's debug logging which adds significant overhead.
        litellm.suppress_debug_info = True
        self._litellm = litellm

    async def chat(self, prompt: str) -> Any:
        return await self._litellm.acompletion(
            model="openai/mock-model",
            messages=[{"role": "user", "content": prompt}],
            api_base=self._base_url,
        )

    async def stream_first_token(self, prompt: str) -> tuple[float, int]:
        import time

        start = time.perf_counter()
        chunks = 0
        ttft: float | None = None
        stream = await self._litellm.acompletion(
            model="openai/mock-model",
            messages=[{"role": "user", "content": prompt}],
            api_base=self._base_url,
            stream=True,
        )
        async for chunk in stream:
            content = ""
            try:
                content = chunk.choices[0].delta.content or ""
            except (AttributeError, IndexError):
                pass
            if content:
                chunks += 1
                if ttft is None:
                    ttft = time.perf_counter() - start
        return (ttft or 0.0), chunks

    async def aclose(self) -> None:
        return None
