"""Cache tests — exact-match Hub-level + Anthropic passthrough markers."""

from __future__ import annotations

import asyncio

import httpx
import pytest
import respx

from relay import CacheHint, Hub, MemoryCache, Message
from relay.cache import _to_anthropic_cache_control, cache_key
from relay.config import load_str
from relay.types import ChatRequest, TextBlock


@pytest.fixture
def env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_KEY", "sk-fake")


def _yaml() -> str:
    return """
        version: 1
        catalog:
          fetch_live_pricing: false
          offline: true
        models:
          m:
            target: openai/gpt-4o-mini
            credential: $env.TEST_KEY
        """


# ---------------------------------------------------------------------------
# Cache key hashing
# ---------------------------------------------------------------------------


def test_cache_key_stable_for_same_request() -> None:
    req1 = ChatRequest(messages=[Message(role="user", content="hi")], temperature=0.5)
    req2 = ChatRequest(messages=[Message(role="user", content="hi")], temperature=0.5)
    assert cache_key("alias", req1) == cache_key("alias", req2)


def test_cache_key_changes_with_content() -> None:
    req1 = ChatRequest(messages=[Message(role="user", content="hi")])
    req2 = ChatRequest(messages=[Message(role="user", content="bye")])
    assert cache_key("alias", req1) != cache_key("alias", req2)


def test_cache_key_changes_with_alias() -> None:
    req = ChatRequest(messages=[Message(role="user", content="hi")])
    assert cache_key("a", req) != cache_key("b", req)


def test_cache_hint_does_not_affect_cache_key() -> None:
    req1 = ChatRequest(messages=[Message(role="user", content=[TextBlock(text="hi")])])
    req2 = ChatRequest(
        messages=[Message(role="user", content=[TextBlock(text="hi"), CacheHint(ttl="1h")])]
    )
    assert cache_key("alias", req1) == cache_key("alias", req2)


# ---------------------------------------------------------------------------
# MemoryCache LRU + TTL
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_memory_cache_basic_set_get() -> None:
    cache = MemoryCache()
    from relay.types import ChatResponse, Choice, Usage

    resp = ChatResponse(
        id="x",
        model="m",
        provider_model="m",
        provider="p",
        choices=[Choice(message=Message(role="assistant", content="ok"))],
        usage=Usage(),
        created_at=0.0,
        latency_ms=0.0,
    )
    await cache.set("key1", resp)
    got = await cache.get("key1")
    assert got is not None
    assert got.text == "ok"
    assert cache.hits == 1


@pytest.mark.asyncio
async def test_memory_cache_miss() -> None:
    cache = MemoryCache()
    got = await cache.get("missing")
    assert got is None
    assert cache.misses == 1


@pytest.mark.asyncio
async def test_memory_cache_ttl_expiry() -> None:
    from relay.types import ChatResponse, Choice, Usage

    cache = MemoryCache(default_ttl_s=0.05)
    resp = ChatResponse(
        id="x",
        model="m",
        provider_model="m",
        provider="p",
        choices=[Choice(message=Message(role="assistant", content="ok"))],
        usage=Usage(),
        created_at=0.0,
        latency_ms=0.0,
    )
    await cache.set("k", resp)
    assert await cache.get("k") is not None
    await asyncio.sleep(0.1)
    assert await cache.get("k") is None


@pytest.mark.asyncio
async def test_memory_cache_evicts_lru() -> None:
    from relay.types import ChatResponse, Choice, Usage

    cache = MemoryCache(max_size=2)
    for i in range(3):
        resp = ChatResponse(
            id=str(i),
            model="m",
            provider_model="m",
            provider="p",
            choices=[Choice(message=Message(role="assistant", content=str(i)))],
            usage=Usage(),
            created_at=0.0,
            latency_ms=0.0,
        )
        await cache.set(f"k{i}", resp)
    assert cache.size == 2
    # Oldest should be evicted
    assert await cache.get("k0") is None
    assert await cache.get("k2") is not None


# ---------------------------------------------------------------------------
# CacheHint translation
# ---------------------------------------------------------------------------


def test_cache_hint_to_anthropic_default() -> None:
    hint = CacheHint()
    assert _to_anthropic_cache_control(hint) == {"type": "ephemeral"}


def test_cache_hint_to_anthropic_1h() -> None:
    hint = CacheHint(ttl="1h")
    assert _to_anthropic_cache_control(hint) == {"type": "ephemeral", "ttl": "1h"}


# ---------------------------------------------------------------------------
# End-to-end: Hub with cache short-circuits second call
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_hub_serves_from_cache_on_second_call(env_key: None) -> None:
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "x",
                "model": "gpt-4o-mini",
                "created": 1700000000,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "cached!"},
                        "finish_reason": "stop",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        )
    )
    cache = MemoryCache()
    hub = Hub.from_config(load_str(_yaml()), cache=cache)
    try:
        msgs = [{"role": "user", "content": "hi"}]
        r1 = await hub.chat("m", messages=msgs)
        r2 = await hub.chat("m", messages=msgs)
        assert r1.text == r2.text == "cached!"
        assert route.call_count == 1  # second call hit cache
        assert cache.hits == 1
        assert cache.misses == 1  # first call missed
    finally:
        await hub.aclose()
