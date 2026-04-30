"""OpenAI Responses API tests."""

from __future__ import annotations

import httpx
import pytest
import respx

from relay import Hub
from relay.config import load_str


@pytest.fixture
def env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_KEY", "sk-fake")


def _yaml() -> str:
    return """
        version: 1
        catalog:
          fetch_live_pricing: false
          offline: true
        models:
          m:
            target: openai/gpt-4o
            credential: $env.OPENAI_KEY
            api_style: responses
        """


@pytest.mark.asyncio
@respx.mock
async def test_responses_basic_chat(env_key: None) -> None:
    respx.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "resp_1",
                "model": "gpt-4o-2024-11-20",
                "status": "completed",
                "output": [
                    {
                        "type": "message",
                        "content": [{"type": "output_text", "text": "hello"}],
                    }
                ],
                "usage": {
                    "input_tokens": 5,
                    "output_tokens": 1,
                    "total_tokens": 6,
                },
            },
        )
    )
    hub = Hub.from_config(load_str(_yaml()))
    try:
        resp = await hub.chat("m", messages=[{"role": "user", "content": "hi"}])
        assert resp.text == "hello"
        assert resp.id == "resp_1"
        assert resp.choices[0].finish_reason == "stop"
        assert resp.usage.input_tokens == 5
    finally:
        await hub.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_responses_function_call(env_key: None) -> None:
    respx.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "resp_2",
                "model": "gpt-4o",
                "status": "completed",
                "output": [
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_abc",
                        "name": "get_weather",
                        "arguments": '{"city":"NYC"}',
                    }
                ],
                "usage": {"input_tokens": 5, "output_tokens": 8, "total_tokens": 13},
            },
        )
    )
    hub = Hub.from_config(load_str(_yaml()))
    try:
        resp = await hub.chat("m", messages=[{"role": "user", "content": "weather"}])
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "get_weather"
        assert resp.tool_calls[0].arguments == {"city": "NYC"}
    finally:
        await hub.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_responses_uses_previous_response_id(env_key: None) -> None:
    route = respx.post("https://api.openai.com/v1/responses").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "resp_3",
                "model": "gpt-4o",
                "status": "completed",
                "output": [{"type": "message", "content": [{"type": "output_text", "text": "ok"}]}],
                "usage": {"input_tokens": 2, "output_tokens": 1},
            },
        )
    )
    hub = Hub.from_config(load_str(_yaml()))
    try:
        await hub.chat(
            "m",
            messages=[{"role": "user", "content": "follow up"}],
            metadata={"previous_response_id": "resp_prev"},
        )
        import orjson

        body = orjson.loads(route.calls[0].request.content)
        assert body["previous_response_id"] == "resp_prev"
    finally:
        await hub.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_responses_default_is_chat_completions(env_key: None) -> None:
    """Without api_style: responses, the request should hit /v1/chat/completions."""
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "x",
                "model": "gpt-4o",
                "created": 1,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    )
    yaml = """
        version: 1
        catalog:
          fetch_live_pricing: false
          offline: true
        models:
          m:
            target: openai/gpt-4o
            credential: $env.OPENAI_KEY
    """
    hub = Hub.from_config(load_str(yaml))
    try:
        resp = await hub.chat("m", messages=[{"role": "user", "content": "hi"}])
        assert resp.text == "ok"
    finally:
        await hub.aclose()
