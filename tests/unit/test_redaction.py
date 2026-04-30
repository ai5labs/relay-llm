"""Redaction tests."""

from __future__ import annotations

import re

import httpx
import pytest
import respx

from relay import Hub
from relay.config import load_str
from relay.redaction import RegexRedactor
from relay.types import Message, TextBlock


def test_regex_redactor_redacts_email_in_string_content() -> None:
    r = RegexRedactor()
    msgs = [Message(role="user", content="contact me at alice@example.com please")]
    result = r.redact(msgs)
    assert "[REDACTED:email]" in result.messages[0].content
    assert "alice@example.com" not in result.messages[0].content
    assert result.redactions == 1
    assert "email" in result.matched_kinds


def test_regex_redactor_redacts_blocks() -> None:
    r = RegexRedactor()
    msgs = [
        Message(
            role="user",
            content=[
                TextBlock(text="ssn 123-45-6789"),
                TextBlock(text="card 4111 1111 1111 1111"),
            ],
        )
    ]
    result = r.redact(msgs)
    blocks = result.messages[0].content
    assert "[REDACTED:ssn]" in blocks[0].text
    assert "[REDACTED:credit_card]" in blocks[1].text
    assert {"ssn", "credit_card"} <= set(result.matched_kinds)


def test_regex_redactor_custom_pattern() -> None:
    r = RegexRedactor(
        patterns={"customer_id": re.compile(r"CUST-\d{6}")},
        inherit_defaults=False,
    )
    msgs = [Message(role="user", content="my id is CUST-123456 plus alice@x.com")]
    result = r.redact(msgs)
    # Custom pattern fired; default email pattern was disabled.
    assert "[REDACTED:customer_id]" in result.messages[0].content
    assert "alice@x.com" in result.messages[0].content


def test_regex_redactor_no_match_returns_original() -> None:
    r = RegexRedactor()
    msgs = [Message(role="user", content="hello world")]
    result = r.redact(msgs)
    assert result.messages[0].content == "hello world"
    assert result.redactions == 0


@pytest.mark.asyncio
@respx.mock
async def test_hub_redacts_messages_before_send(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_KEY", "sk-fake")
    route = respx.post("https://api.openai.com/v1/chat/completions").mock(
        return_value=httpx.Response(
            200,
            json={
                "id": "x",
                "model": "gpt-4o-mini",
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
            target: openai/gpt-4o-mini
            credential: $env.OPENAI_KEY
    """
    hub = Hub.from_config(load_str(yaml), redactor=RegexRedactor())
    try:
        await hub.chat(
            "m",
            messages=[{"role": "user", "content": "email is alice@example.com"}],
        )
        # Verify the outbound request had email redacted
        import orjson

        body = orjson.loads(route.calls[0].request.content)
        sent_text = body["messages"][0]["content"]
        assert "alice@example.com" not in sent_text
        assert "[REDACTED:email]" in sent_text
    finally:
        await hub.aclose()
