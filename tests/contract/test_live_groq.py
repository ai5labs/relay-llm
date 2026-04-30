"""Live contract tests against Groq (OpenAI-compatible)."""

from __future__ import annotations

import pytest

from relay import Hub
from relay.config import load_str

from .conftest import skip_unless_keys


@pytest.fixture
def hub() -> Hub:
    yaml = """
        version: 1
        catalog:
          fetch_live_pricing: false
          offline: true
        models:
          fast:
            target: groq/llama-3.3-70b-versatile
            credential: $env.GROQ_API_KEY
            params:
              max_tokens: 64
    """
    return Hub.from_config(load_str(yaml))


@pytest.mark.live
@skip_unless_keys("GROQ_API_KEY")
@pytest.mark.asyncio
async def test_basic_chat(hub: Hub) -> None:
    try:
        resp = await hub.chat("fast", messages=[{"role": "user", "content": "Reply with 'pong'"}])
        assert resp.text
        assert resp.usage.input_tokens > 0
    finally:
        await hub.aclose()


@pytest.mark.live
@skip_unless_keys("GROQ_API_KEY")
@pytest.mark.asyncio
async def test_streaming_low_latency(hub: Hub) -> None:
    """Groq's whole pitch is low TTFB — verify streaming actually streams."""
    import time

    from relay.types import StreamEnd, TextDelta

    try:
        start = time.perf_counter()
        first_token_at: float | None = None
        async for ev in hub.stream(
            "fast",
            messages=[{"role": "user", "content": "Write a haiku about Python"}],
        ):
            if isinstance(ev, TextDelta) and first_token_at is None:
                first_token_at = time.perf_counter() - start
            if isinstance(ev, StreamEnd):
                break
        # Should be well under 5s on Groq.
        assert first_token_at is not None
        assert first_token_at < 5.0
    finally:
        await hub.aclose()
