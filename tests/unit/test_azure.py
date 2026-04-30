"""Azure OpenAI adapter tests."""

from __future__ import annotations

import httpx
import pytest
import respx

from relay import Hub
from relay.config import load_str
from relay.errors import AuthenticationError


def _yaml(
    *, deployment: str = "gpt-4o-prod", base_url: str = "https://acme.openai.azure.com"
) -> str:
    return f"""
        version: 1
        catalog:
          fetch_live_pricing: false
          offline: true
        models:
          m:
            target: azure/gpt-4o
            base_url: {base_url}
            deployment: {deployment}
            api_version: "2024-08-01-preview"
            credential: $env.AZURE_KEY
        """


@pytest.fixture
def env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AZURE_KEY", "azure-fake-key")


@pytest.mark.asyncio
@respx.mock
async def test_azure_routes_to_deployment_url(env_key: None) -> None:
    route = respx.post(
        "https://acme.openai.azure.com/openai/deployments/gpt-4o-prod/chat/completions"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "azure-1",
                "model": "gpt-4o",
                "created": 1700000000,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "hi"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    )
    hub = Hub.from_config(load_str(_yaml()))
    try:
        resp = await hub.chat("m", messages=[{"role": "user", "content": "hi"}])
        assert resp.text == "hi"
        assert route.called
        # Verify api-version was sent as query param
        called = route.calls[0]
        assert "api-version=2024-08-01-preview" in str(called.request.url)
    finally:
        await hub.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_azure_uses_api_key_header_not_bearer(env_key: None) -> None:
    route = respx.post(
        "https://acme.openai.azure.com/openai/deployments/gpt-4o-prod/chat/completions"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "x",
                "model": "gpt-4o",
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
    hub = Hub.from_config(load_str(_yaml()))
    try:
        await hub.chat("m", messages=[{"role": "user", "content": "hi"}])
        called = route.calls[0]
        assert called.request.headers.get("api-key") == "azure-fake-key"
        assert "authorization" not in {k.lower() for k in called.request.headers}
    finally:
        await hub.aclose()


@pytest.mark.asyncio
async def test_azure_missing_base_url_raises(env_key: None) -> None:
    yaml = """
        version: 1
        catalog:
          fetch_live_pricing: false
          offline: true
        models:
          m:
            target: azure/gpt-4o
            deployment: foo
            credential: $env.AZURE_KEY
    """
    hub = Hub.from_config(load_str(yaml))
    try:
        with pytest.raises(AuthenticationError, match="base_url"):
            await hub.chat("m", messages=[{"role": "user", "content": "hi"}])
    finally:
        await hub.aclose()


@pytest.mark.asyncio
async def test_azure_missing_deployment_raises(env_key: None) -> None:
    yaml = """
        version: 1
        catalog:
          fetch_live_pricing: false
          offline: true
        models:
          m:
            target: azure/gpt-4o
            base_url: https://acme.openai.azure.com
            credential: $env.AZURE_KEY
    """
    hub = Hub.from_config(load_str(yaml))
    try:
        with pytest.raises(AuthenticationError, match="deployment"):
            await hub.chat("m", messages=[{"role": "user", "content": "hi"}])
    finally:
        await hub.aclose()
