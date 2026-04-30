"""OpenAI-compatible adapter tests using respx for HTTP mocking."""

from __future__ import annotations

import httpx
import pytest
import respx

from relay import Hub
from relay.errors import (
    AuthenticationError,
    ContentPolicyError,
    ContextWindowError,
    RateLimitError,
)


def _yaml(provider: str = "openai", target: str | None = None) -> str:
    target = target or f"{provider}/test-model"
    return f"""
        version: 1
        catalog:
          fetch_live_pricing: false
          offline: true
        models:
          m:
            target: {target}
            credential: $env.TEST_KEY
        """


@pytest.fixture
def env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_KEY", "sk-fake")


@pytest.mark.asyncio
@respx.mock
async def test_basic_chat_round_trip(env_key: None) -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "chatcmpl-1",
                "object": "chat.completion",
                "model": "test-model",
                "created": 1700000000,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi there"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 5,
                    "completion_tokens": 2,
                    "total_tokens": 7,
                },
            },
        )
    )
    from relay.config import load_str

    hub = Hub.from_config(load_str(_yaml()))
    try:
        resp = await hub.chat("m", messages=[{"role": "user", "content": "hi"}])
        assert resp.text == "hi there"
        assert resp.usage.input_tokens == 5
        assert resp.usage.output_tokens == 2
        assert resp.choices[0].finish_reason == "stop"
    finally:
        await hub.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_401_maps_to_authentication_error(env_key: None) -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(401, json={"error": {"message": "bad key"}}),
    )
    from relay.config import load_str

    hub = Hub.from_config(load_str(_yaml()))
    try:
        with pytest.raises(AuthenticationError):
            await hub.chat("m", messages=[{"role": "user", "content": "hi"}])
    finally:
        await hub.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_429_maps_to_rate_limit(env_key: None) -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            429,
            headers={"retry-after": "2.5"},
            json={"error": {"message": "rate limited", "type": "tokens_per_min"}},
        ),
    )
    from relay.config import load_str

    cfg = load_str(_yaml())
    # Disable retries for this test so we observe the raw error.
    hub = Hub.from_config(
        cfg.model_copy(
            update={
                "defaults": cfg.defaults.model_copy(update={"max_retries": 0}),
            }
        )
    )
    try:
        with pytest.raises(RateLimitError) as exc:
            await hub.chat("m", messages=[{"role": "user", "content": "hi"}])
        assert exc.value.retry_after == 2.5
        assert exc.value.limit_type == "tpm"
    finally:
        await hub.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_400_context_window_classified(env_key: None) -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            400,
            json={
                "error": {
                    "message": "Request exceeds the model's context length window",
                    "type": "invalid_request_error",
                }
            },
        ),
    )
    from relay.config import load_str

    hub = Hub.from_config(load_str(_yaml()))
    try:
        with pytest.raises(ContextWindowError):
            await hub.chat("m", messages=[{"role": "user", "content": "hi"}])
    finally:
        await hub.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_400_content_policy_classified(env_key: None) -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            400,
            json={"error": {"message": "request blocked by content policy"}},
        ),
    )
    from relay.config import load_str

    hub = Hub.from_config(load_str(_yaml()))
    try:
        with pytest.raises(ContentPolicyError):
            await hub.chat("m", messages=[{"role": "user", "content": "hi"}])
    finally:
        await hub.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_groq_routes_to_groq_base_url(env_key: None) -> None:
    route = respx.post("https://api.groq.com/openai/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "x",
                "model": "llama-3.3-70b-versatile",
                "created": 1700000000,
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
    from relay.config import load_str

    hub = Hub.from_config(load_str(_yaml(provider="groq", target="groq/llama-3.3-70b-versatile")))
    try:
        await hub.chat("m", messages=[{"role": "user", "content": "hi"}])
        assert route.called
    finally:
        await hub.aclose()
