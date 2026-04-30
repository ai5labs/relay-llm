"""Cohere adapter tests."""

from __future__ import annotations

import httpx
import pytest
import respx

from relay import Hub
from relay.config import load_str


@pytest.fixture
def env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COHERE_KEY", "co-fake")


def _yaml() -> str:
    return """
        version: 1
        catalog:
          fetch_live_pricing: false
          offline: true
        models:
          m:
            target: cohere/command-r-plus-08-2024
            credential: $env.COHERE_KEY
        """


@pytest.mark.asyncio
@respx.mock
async def test_cohere_basic_chat(env_key: None) -> None:
    respx.post("https://api.cohere.com/v2/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "co-1",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hello!"}],
                },
                "finish_reason": "COMPLETE",
                "usage": {"billed_units": {"input_tokens": 5, "output_tokens": 2}},
            },
        )
    )
    hub = Hub.from_config(load_str(_yaml()))
    try:
        resp = await hub.chat("m", messages=[{"role": "user", "content": "hi"}])
        assert resp.text == "hello!"
        assert resp.usage.input_tokens == 5
        assert resp.usage.output_tokens == 2
        assert resp.choices[0].finish_reason == "stop"
    finally:
        await hub.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_cohere_uses_bearer_auth(env_key: None) -> None:
    route = respx.post("https://api.cohere.com/v2/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "x",
                "message": {"role": "assistant", "content": [{"type": "text", "text": "ok"}]},
                "finish_reason": "COMPLETE",
                "usage": {"billed_units": {"input_tokens": 1, "output_tokens": 1}},
            },
        )
    )
    hub = Hub.from_config(load_str(_yaml()))
    try:
        await hub.chat("m", messages=[{"role": "user", "content": "hi"}])
        assert route.calls[0].request.headers.get("authorization") == "Bearer co-fake"
    finally:
        await hub.aclose()
