"""Native Google Gemini adapter tests."""

from __future__ import annotations

import httpx
import pytest
import respx

from relay import Hub
from relay.config import load_str


@pytest.fixture
def env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_KEY", "g-fake")


def _yaml() -> str:
    return """
        version: 1
        catalog:
          fetch_live_pricing: false
          offline: true
        models:
          m:
            target: google/gemini-2.5-flash
            credential: $env.GEMINI_KEY
        """


@pytest.mark.asyncio
@respx.mock
async def test_gemini_basic_chat(env_key: None) -> None:
    respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [{"text": "hi from gemini"}],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {
                    "promptTokenCount": 5,
                    "candidatesTokenCount": 3,
                    "totalTokenCount": 8,
                },
                "modelVersion": "gemini-2.5-flash",
            },
        )
    )
    hub = Hub.from_config(load_str(_yaml()))
    try:
        resp = await hub.chat("m", messages=[{"role": "user", "content": "hi"}])
        assert resp.text == "hi from gemini"
        assert resp.usage.input_tokens == 5
        assert resp.usage.output_tokens == 3
        assert resp.choices[0].finish_reason == "stop"
    finally:
        await hub.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_gemini_uses_x_goog_api_key_header(env_key: None) -> None:
    """The API key is sent in the x-goog-api-key header, never the query string
    (query strings land in proxy/reverse-proxy access logs)."""
    route = respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "x"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
            },
        )
    )
    hub = Hub.from_config(load_str(_yaml()))
    try:
        await hub.chat("m", messages=[{"role": "user", "content": "hi"}])
        req = route.calls[0].request
        assert req.headers.get("x-goog-api-key") == "g-fake"
        assert "key=" not in str(req.url)
    finally:
        await hub.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_gemini_system_message_split(env_key: None) -> None:
    route = respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "x"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
            },
        )
    )
    hub = Hub.from_config(load_str(_yaml()))
    try:
        await hub.chat(
            "m",
            messages=[
                {"role": "system", "content": "You are a pirate."},
                {"role": "user", "content": "hi"},
            ],
        )
        body = route.calls[0].request.content
        import orjson

        parsed = orjson.loads(body)
        assert "systemInstruction" in parsed
        assert "pirate" in parsed["systemInstruction"]["parts"][0]["text"]
        # System message should NOT appear in contents
        assert all(c["role"] != "system" for c in parsed["contents"])
    finally:
        await hub.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_gemini_function_call(env_key: None) -> None:
    respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {
                            "role": "model",
                            "parts": [
                                {
                                    "functionCall": {
                                        "name": "get_weather",
                                        "args": {"city": "Paris"},
                                    }
                                }
                            ],
                        },
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 5, "candidatesTokenCount": 5},
            },
        )
    )
    hub = Hub.from_config(load_str(_yaml()))
    try:
        resp = await hub.chat("m", messages=[{"role": "user", "content": "weather"}])
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "get_weather"
        assert resp.tool_calls[0].arguments == {"city": "Paris"}
    finally:
        await hub.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_gemini_reasoning_budget_translated(env_key: None) -> None:
    route = respx.post(
        "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "ok"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 1, "candidatesTokenCount": 1},
            },
        )
    )
    hub = Hub.from_config(load_str(_yaml()))
    try:
        await hub.chat(
            "m",
            messages=[{"role": "user", "content": "think"}],
            reasoning="medium",
        )
        import orjson

        parsed = orjson.loads(route.calls[0].request.content)
        assert parsed["generationConfig"]["thinkingConfig"]["thinking_budget"] == 16384
    finally:
        await hub.aclose()
