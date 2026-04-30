"""Live contract tests against the real OpenAI API."""

from __future__ import annotations

import pytest
from pydantic import BaseModel

from relay import Hub
from relay.config import load_str
from relay.structured import request_structured

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
            target: openai/gpt-4o-mini
            credential: $env.OPENAI_API_KEY
            params:
              max_tokens: 64
    """
    return Hub.from_config(load_str(yaml))


@pytest.mark.live
@skip_unless_keys("OPENAI_API_KEY")
@pytest.mark.asyncio
async def test_basic_chat_returns_text(hub: Hub) -> None:
    try:
        resp = await hub.chat(
            "fast", messages=[{"role": "user", "content": "Reply with the word 'pong'"}]
        )
        assert resp.text  # non-empty
        assert resp.usage.input_tokens > 0
        assert resp.usage.output_tokens > 0
        assert resp.cost is not None
        assert resp.cost.total_usd is not None
    finally:
        await hub.aclose()


@pytest.mark.live
@skip_unless_keys("OPENAI_API_KEY")
@pytest.mark.asyncio
async def test_streaming_assembles(hub: Hub) -> None:
    from relay.types import StreamEnd, TextDelta

    try:
        deltas: list[str] = []
        final = None
        async for ev in hub.stream(
            "fast", messages=[{"role": "user", "content": "Count: 1, 2, 3"}]
        ):
            if isinstance(ev, TextDelta):
                deltas.append(ev.text)
            elif isinstance(ev, StreamEnd):
                final = ev
        assert final is not None
        assert "".join(deltas) == final.response.text
        assert final.response.usage.input_tokens > 0
    finally:
        await hub.aclose()


class Person(BaseModel):
    name: str
    age: int


@pytest.mark.live
@skip_unless_keys("OPENAI_API_KEY")
@pytest.mark.asyncio
async def test_structured_output(hub: Hub) -> None:
    try:
        out = await request_structured(
            hub=hub,
            alias="fast",
            schema=Person,
            messages=[
                {
                    "role": "user",
                    "content": "Make up a person. Return only the JSON object.",
                }
            ],
        )
        assert isinstance(out, Person)
        assert out.name
        assert out.age >= 0
    finally:
        await hub.aclose()


@pytest.mark.live
@skip_unless_keys("OPENAI_API_KEY")
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
        assert "city" in resp.tool_calls[0].arguments
    finally:
        await hub.aclose()
