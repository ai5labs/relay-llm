"""Structured output tests."""

from __future__ import annotations

import httpx
import pytest
import respx
from pydantic import BaseModel, Field

from relay import Hub
from relay.config import load_str
from relay.structured import (
    StructuredOutputError,
    build_request_overrides,
    parse_response,
    request_structured,
)
from relay.types import ChatResponse, Choice, Message, ToolCall, Usage


class Person(BaseModel):
    name: str
    age: int = Field(ge=0)


# ---------------------------------------------------------------------------
# build_request_overrides
# ---------------------------------------------------------------------------


def test_overrides_openai_uses_json_schema_strict() -> None:
    out = build_request_overrides(Person, provider="openai")
    assert out["response_format"]["type"] == "json_schema"
    assert out["response_format"]["json_schema"]["strict"] is True
    assert out["response_format"]["json_schema"]["name"] == "Person"


def test_overrides_anthropic_uses_single_tool_trick() -> None:
    out = build_request_overrides(Person, provider="anthropic")
    assert "tools" in out
    assert out["tools"][0].name == "Person"
    assert out["tool_choice"] == {"type": "tool", "name": "Person"}


def test_overrides_gemini_uses_response_schema() -> None:
    out = build_request_overrides(Person, provider="google")
    assert out["response_format"]["type"] == "json_schema"


def test_overrides_unknown_falls_back_to_json_object() -> None:
    out = build_request_overrides(Person, provider="cohere")
    assert out["response_format"] == {"type": "json_object"}


def test_overrides_accepts_dict_schema() -> None:
    schema = {"type": "object", "properties": {"x": {"type": "string"}}, "title": "Custom"}
    out = build_request_overrides(schema, provider="openai")
    assert out["response_format"]["json_schema"]["name"] == "Custom"


# ---------------------------------------------------------------------------
# parse_response
# ---------------------------------------------------------------------------


def _resp(text: str = "", tool_calls: list[ToolCall] | None = None) -> ChatResponse:
    return ChatResponse(
        id="x",
        model="alias",
        provider_model="m",
        provider="p",
        choices=[
            Choice(message=Message(role="assistant", content=text), tool_calls=tool_calls or [])
        ],
        usage=Usage(),
        created_at=0.0,
        latency_ms=0.0,
    )


def test_parse_from_text() -> None:
    resp = _resp(text='{"name":"alice","age":30}')
    out = parse_response(Person, resp)
    assert isinstance(out, Person)
    assert out.name == "alice"
    assert out.age == 30


def test_parse_from_text_strips_code_fence() -> None:
    resp = _resp(text='```json\n{"name":"bob","age":40}\n```')
    out = parse_response(Person, resp)
    assert isinstance(out, Person)
    assert out.name == "bob"


def test_parse_from_tool_call() -> None:
    resp = _resp(
        text="",
        tool_calls=[ToolCall(id="1", name="Person", arguments={"name": "carol", "age": 50})],
    )
    out = parse_response(Person, resp)
    assert isinstance(out, Person)
    assert out.name == "carol"


def test_parse_validation_failure_raises() -> None:
    resp = _resp(text='{"name":"x","age":-1}')
    with pytest.raises(StructuredOutputError):
        parse_response(Person, resp)


def test_parse_invalid_json_raises() -> None:
    resp = _resp(text="not json at all")
    with pytest.raises(StructuredOutputError):
        parse_response(Person, resp)


def test_parse_with_dict_schema_returns_dict() -> None:
    resp = _resp(text='{"name":"d","age":1}')
    out = parse_response({"type": "object", "title": "X"}, resp)
    assert out == {"name": "d", "age": 1}


# ---------------------------------------------------------------------------
# End-to-end: request_structured with a mocked OpenAI response
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_request_structured_end_to_end(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_KEY", "sk-fake")
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "x",
                "model": "gpt-4o-mini",
                "created": 1700000000,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": '{"name":"eve","age":25}'},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
            },
        )
    )
    hub = Hub.from_config(
        load_str("""
            version: 1
            catalog:
              fetch_live_pricing: false
              offline: true
            models:
              m:
                target: openai/gpt-4o-mini
                credential: $env.TEST_KEY
        """)
    )
    try:
        out = await request_structured(
            hub=hub,
            alias="m",
            schema=Person,
            messages=[{"role": "user", "content": "make a person"}],
        )
        assert isinstance(out, Person)
        assert out.name == "eve"
        assert out.age == 25
    finally:
        await hub.aclose()
