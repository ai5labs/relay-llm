"""Live contract tests against the real Anthropic API."""

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
            target: anthropic/claude-haiku-4-5
            credential: $env.ANTHROPIC_API_KEY
            params:
              max_tokens: 64
    """
    return Hub.from_config(load_str(yaml))


@pytest.mark.live
@skip_unless_keys("ANTHROPIC_API_KEY")
@pytest.mark.asyncio
async def test_basic_chat(hub: Hub) -> None:
    try:
        resp = await hub.chat("fast", messages=[{"role": "user", "content": "Reply with 'pong'"}])
        assert resp.text
        assert resp.usage.input_tokens > 0
        assert resp.cost is not None
    finally:
        await hub.aclose()


@pytest.mark.live
@skip_unless_keys("ANTHROPIC_API_KEY")
@pytest.mark.asyncio
async def test_streaming_with_thinking(hub: Hub) -> None:
    """Anthropic extended thinking — verify ThinkingDelta events when enabled."""
    from relay.types import StreamEnd, TextDelta

    try:
        text_chunks: list[str] = []
        final = None
        async for ev in hub.stream(
            "fast",
            messages=[{"role": "user", "content": "What is 2+2? Reply briefly."}],
        ):
            if isinstance(ev, TextDelta):
                text_chunks.append(ev.text)
            elif isinstance(ev, StreamEnd):
                final = ev
        assert final is not None
        assert "".join(text_chunks) == final.response.text
    finally:
        await hub.aclose()


@pytest.mark.live
@skip_unless_keys("ANTHROPIC_API_KEY")
@pytest.mark.asyncio
async def test_tool_calling(hub: Hub) -> None:
    from relay.types import ToolDefinition

    weather = ToolDefinition(
        name="get_weather",
        description="Get current weather for a city",
        parameters={
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        },
    )
    try:
        resp = await hub.chat(
            "fast",
            messages=[{"role": "user", "content": "What's the weather in Paris? Use the tool."}],
            tools=[weather],
            tool_choice="required",
        )
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "get_weather"
    finally:
        await hub.aclose()
