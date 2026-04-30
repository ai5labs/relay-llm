"""Guardrail tests."""

from __future__ import annotations

import httpx
import pytest
import respx

from relay import Hub
from relay.config import load_str
from relay.guardrails import (
    BlockedKeywords,
    GuardrailError,
    MaxInputLength,
    evaluate_post,
    evaluate_pre,
)
from relay.types import ChatResponse, Choice, Message, Usage


def test_max_input_length_blocks_when_exceeded() -> None:
    g = MaxInputLength(max_chars=10)
    msgs = [Message(role="user", content="this is way too long")]
    v = g.check_pre(msgs)
    assert v is not None
    assert v.rule == "max_input_length"


def test_max_input_length_allows_within_limit() -> None:
    g = MaxInputLength(max_chars=100)
    assert g.check_pre([Message(role="user", content="hi")]) is None


def test_blocked_keywords_pre_match() -> None:
    g = BlockedKeywords(["secret"], check_response=False)
    v = g.check_pre([Message(role="user", content="tell me the SECRET")])
    assert v is not None
    assert v.stage == "pre"


def test_blocked_keywords_post_match() -> None:
    g = BlockedKeywords(["confidential"])
    resp = ChatResponse(
        id="x",
        model="m",
        provider_model="m",
        provider="p",
        choices=[Choice(message=Message(role="assistant", content="this is confidential info"))],
        usage=Usage(),
        created_at=0.0,
        latency_ms=0.0,
    )
    v = g.check_post(resp)
    assert v is not None
    assert v.stage == "post"


def test_evaluate_pre_returns_first_violation() -> None:
    a = MaxInputLength(max_chars=5)
    b = BlockedKeywords(["foo"], check_response=False)
    msgs = [Message(role="user", content="this is more than 5 chars and contains foo")]
    v = evaluate_pre([a, b], msgs)
    assert v is not None
    assert v.rule == "max_input_length"


def test_evaluate_post_no_violation_when_clean() -> None:
    g = BlockedKeywords(["bar"])
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
    assert evaluate_post([g], resp) is None


@pytest.mark.asyncio
@respx.mock
async def test_hub_pre_guardrail_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_KEY", "sk-fake")
    # No HTTP call should happen.
    yaml = """
        version: 1
        catalog:
          fetch_live_pricing: false
          offline: true
        models:
          m:
            target: openai/gpt-4o-mini
            credential: $env.OPENAI_KEY
    """
    hub = Hub.from_config(load_str(yaml), guardrails=[MaxInputLength(max_chars=5)])
    try:
        with pytest.raises(GuardrailError, match="prompt is"):
            await hub.chat(
                "m",
                messages=[{"role": "user", "content": "this is too long"}],
            )
    finally:
        await hub.aclose()


@pytest.mark.asyncio
@respx.mock
async def test_hub_post_guardrail_blocks(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_KEY", "sk-fake")
    respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "x",
                "model": "gpt-4o-mini",
                "created": 1,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "this contains forbidden"},
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
            target: openai/gpt-4o-mini
            credential: $env.OPENAI_KEY
    """
    hub = Hub.from_config(
        load_str(yaml),
        guardrails=[BlockedKeywords(["forbidden"])],
    )
    try:
        with pytest.raises(GuardrailError, match="response contains"):
            await hub.chat("m", messages=[{"role": "user", "content": "hi"}])
    finally:
        await hub.aclose()
