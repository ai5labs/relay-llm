"""Cost computation + pricing tier resolution tests."""

from __future__ import annotations

import httpx
import pytest
import respx

from relay import Hub
from relay.config import load_str


@pytest.fixture
def env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_KEY", "sk-fake")


def _yaml(extra: str = "") -> str:
    return f"""
        version: 1
        catalog:
          fetch_live_pricing: false
          offline: true
        models:
          m:
            target: openai/gpt-4o-mini
            credential: $env.TEST_KEY
        {extra}
        """


@pytest.mark.asyncio
@respx.mock
async def test_cost_computed_from_snapshot(env_key: None) -> None:
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
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1_000_000,
                    "completion_tokens": 1_000_000,
                    "total_tokens": 2_000_000,
                },
            },
        )
    )
    hub = Hub.from_config(load_str(_yaml()))
    try:
        resp = await hub.chat("m", messages=[{"role": "user", "content": "hi"}])
        assert resp.cost is not None
        # GPT-4o-mini list price: $0.15/M in, $0.60/M out → $0.75 for 1M+1M.
        assert resp.cost.total_usd == pytest.approx(0.75, rel=0.01)
        assert resp.cost.source == "snapshot"
        assert resp.cost.confidence == "list_price"
    finally:
        await hub.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_user_override_beats_snapshot(env_key: None) -> None:
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
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1_000_000,
                    "completion_tokens": 0,
                    "total_tokens": 1_000_000,
                },
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
            target: openai/gpt-4o-mini
            credential: $env.TEST_KEY
            cost:
              input_per_1m: 5.0
              output_per_1m: 10.0
        """
    hub = Hub.from_config(load_str(yaml))
    try:
        resp = await hub.chat("m", messages=[{"role": "user", "content": "hi"}])
        assert resp.cost is not None
        # 1M input * $5 = $5; 0 output → $5.0
        assert resp.cost.total_usd == pytest.approx(5.0)
        assert resp.cost.source == "user_override"
        assert resp.cost.confidence == "exact"
    finally:
        await hub.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_pricing_profile_multiplier(env_key: None) -> None:
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
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {
                    "prompt_tokens": 1_000_000,
                    "completion_tokens": 0,
                    "total_tokens": 1_000_000,
                },
            },
        )
    )
    yaml = """
        version: 1
        catalog:
          fetch_live_pricing: false
          offline: true
        pricing_profiles:
          discount:
            input_multiplier: 0.5
            output_multiplier: 0.5
        models:
          m:
            target: openai/gpt-4o-mini
            credential: $env.TEST_KEY
            pricing_profile: discount
        """
    hub = Hub.from_config(load_str(yaml))
    try:
        resp = await hub.chat("m", messages=[{"role": "user", "content": "hi"}])
        # Snapshot $0.15/M input, halved by profile → $0.075/M.
        # 1M tokens → $0.075.
        assert resp.cost is not None
        assert resp.cost.total_usd == pytest.approx(0.075, rel=0.01)
    finally:
        await hub.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_unknown_model_returns_unknown_cost(env_key: None) -> None:
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "x",
                "model": "ft:custom",
                "created": 1700000000,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "ok"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10, "total_tokens": 110},
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
            target: openai/ft:custom-not-in-catalog
            credential: $env.TEST_KEY
        """
    hub = Hub.from_config(load_str(yaml))
    try:
        resp = await hub.chat("m", messages=[{"role": "user", "content": "hi"}])
        assert resp.cost is not None
        assert resp.cost.source == "unknown"
        assert resp.cost.confidence == "unknown"
    finally:
        await hub.aclose()
