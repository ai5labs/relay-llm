"""Vertex AI adapter tests.

We mock the google.auth token refresh and the Vertex HTTP endpoint.
"""

from __future__ import annotations

from unittest.mock import patch

import httpx
import pytest
import respx

from relay import Hub
from relay.config import load_str
from relay.errors import AuthenticationError


def _yaml(*, project: str = "acme-prod", location: str = "us-central1") -> str:
    return f"""
        version: 1
        catalog:
          fetch_live_pricing: false
          offline: true
        models:
          m:
            target: vertex/gemini-2.5-flash
            project: {project}
            location: {location}
        """


@pytest.fixture
def mocked_token() -> None:
    """Stub the Vertex provider's token method so we don't need real GCP creds."""
    from relay.providers.vertex import VertexProvider

    async def _fake_get_token(self: VertexProvider, project: str) -> str:
        return "fake-vertex-token"

    with patch.object(VertexProvider, "_get_token", _fake_get_token):
        yield


@pytest.mark.asyncio
@respx.mock
async def test_vertex_basic_chat(mocked_token: None) -> None:
    respx.post(
        "https://us-central1-aiplatform.googleapis.com/v1/projects/acme-prod/locations/"
        "us-central1/publishers/google/models/gemini-2.5-flash:generateContent"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "candidates": [
                    {
                        "content": {"role": "model", "parts": [{"text": "vertex hi"}]},
                        "finishReason": "STOP",
                    }
                ],
                "usageMetadata": {"promptTokenCount": 4, "candidatesTokenCount": 2},
                "modelVersion": "gemini-2.5-flash-001",
            },
        )
    )
    hub = Hub.from_config(load_str(_yaml()))
    try:
        resp = await hub.chat("m", messages=[{"role": "user", "content": "hi"}])
        assert resp.text == "vertex hi"
        assert resp.usage.input_tokens == 4
        assert resp.usage.output_tokens == 2
        assert resp.choices[0].finish_reason == "stop"
    finally:
        await hub.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_vertex_uses_bearer_token_from_adc(mocked_token: None) -> None:
    route = respx.post(
        "https://us-central1-aiplatform.googleapis.com/v1/projects/acme-prod/locations/"
        "us-central1/publishers/google/models/gemini-2.5-flash:generateContent"
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
        assert route.calls[0].request.headers.get("authorization") == "Bearer fake-vertex-token"
    finally:
        await hub.aclose()


@pytest.mark.asyncio
async def test_vertex_missing_project_raises() -> None:
    yaml = """
        version: 1
        catalog:
          fetch_live_pricing: false
          offline: true
        models:
          m:
            target: vertex/gemini-2.5-flash
    """
    hub = Hub.from_config(load_str(yaml))
    try:
        with pytest.raises(AuthenticationError, match="project"):
            await hub.chat("m", messages=[{"role": "user", "content": "hi"}])
    finally:
        await hub.aclose()
