"""PR 8 audit follow-ups — fixes for issues the post-PR-7 re-audit flagged.

Specifically:
- Schema validator no longer waves through allOf / enum / additionalProperties.
- _stream_one's underlying provider generator is aclose()'d after deadline.
- RelayError scrubs ``message`` as well as ``raw``.
- Unknown tool names in a response raise ToolSchemaError.
- CallbackSink sync-warning catches callable classes / partial / lambdas.
- Additional secret patterns (GitHub PAT, JWT, PEM, Google AIza).
"""

from __future__ import annotations

import asyncio
import warnings
from collections.abc import AsyncIterator
from typing import Any

import httpx
import pytest
import respx

from relay import Hub
from relay.audit import AuditEvent, CallbackSink
from relay.config import load_str
from relay.errors import ProviderError, ToolSchemaError
from relay.types import StreamEvent, StreamStart, TextDelta, ToolDefinition


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


@pytest.fixture
def env_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_KEY", "sk-fake")


# ---------------------------------------------------------------------------
# schema validator no longer waves through non-trivial schemas
# ---------------------------------------------------------------------------


def test_schema_validator_runs_on_allof() -> None:
    """The pre-PR-8 fast-path skipped any schema without ``properties`` /
    ``type`` / ``required`` / ``anyOf`` / ``oneOf`` — meaning ``allOf`` schemas
    silently disabled validation. After PR8 the validator must enforce."""
    from relay._internal.schema_validate import validate_tool_arguments

    schema = {"allOf": [{"type": "object", "required": ["k"]}]}
    # Missing required key — must raise.
    with pytest.raises(ToolSchemaError):
        validate_tool_arguments("t", {"other": "x"}, schema)
    # Compliant — passes.
    validate_tool_arguments("t", {"k": 1}, schema)


def test_schema_validator_runs_on_additional_properties_false() -> None:
    """Old fast-path skipped a schema with only ``additionalProperties: false``.
    Now it enforces."""
    from relay._internal.schema_validate import validate_tool_arguments

    schema = {"additionalProperties": False}
    with pytest.raises(ToolSchemaError):
        validate_tool_arguments("t", {"surprise": "x"}, schema)


def test_schema_validator_skips_only_empty_schema() -> None:
    """A literally empty schema is the one no-op case."""
    from relay._internal.schema_validate import validate_tool_arguments

    validate_tool_arguments("t", {"anything": 1}, {})  # no raise


# ---------------------------------------------------------------------------
# unknown tool name in response is a hard error
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_unknown_tool_name_in_response_raises(env_key: None) -> None:
    """A model that 'invents' a tool name not in the declared tools list
    must surface as ToolSchemaError — silently skipping would let the
    hallucinated call reach the caller's dispatcher."""
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
                        "message": {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {
                                        "name": "os_system",
                                        "arguments": '{"cmd": "rm -rf /"}',
                                    },
                                }
                            ],
                        },
                        "finish_reason": "tool_calls",
                    }
                ],
                "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            },
        ),
    )
    tool = ToolDefinition(
        name="create_issue",
        description="",
        parameters={"type": "object", "properties": {"title": {"type": "string"}}},
    )
    hub = Hub.from_config(load_str(_yaml()))
    try:
        with pytest.raises(ToolSchemaError, match="undeclared tool 'os_system'"):
            await hub.chat("m", messages=[{"role": "user", "content": "hi"}], tools=[tool])
    finally:
        await hub.aclose()


# ---------------------------------------------------------------------------
# stream deadline aclose()s the inner generator
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_stream_deadline_closes_inner_generator(env_key: None) -> None:
    """When the wall-clock deadline fires, the inner provider generator
    must be aclose()'d so the HTTP socket is torn down promptly. Without
    this, a slow-loris provider keeps the connection open until GC."""
    from relay.providers._base import BaseProvider

    aclose_called = asyncio.Event()

    class _LeakySlowProvider(BaseProvider):
        name = "openai"

        async def chat(self, **kwargs: Any) -> Any:  # pragma: no cover - unused
            raise NotImplementedError

        async def stream(self, **kwargs: Any) -> AsyncIterator[StreamEvent]:  # type: ignore[override]
            try:
                yield StreamStart(id="x", model="m", provider="openai")
                while True:
                    await asyncio.sleep(0.2)
                    yield TextDelta(text="byte")
            finally:
                # Async generator cleanup — GeneratorExit -> finally runs
                # on aclose().
                aclose_called.set()

    yaml = """
        version: 1
        catalog:
          fetch_live_pricing: false
          offline: true
        defaults:
          stream_overall_timeout: 0.3
        models:
          m:
            target: openai/test
            credential: $env.TEST_KEY
    """
    hub = Hub.from_config(load_str(yaml))
    hub._providers["openai"] = _LeakySlowProvider()
    try:
        with pytest.raises(Exception):  # noqa: B017 - RelayTimeoutError, but kept loose
            async for _ev in hub.stream("m", messages=[{"role": "user", "content": "hi"}]):
                pass
        # The finally inside _stream_one must have invoked aclose() on the
        # inner generator within a short grace period.
        await asyncio.wait_for(aclose_called.wait(), timeout=1.0)
    finally:
        await hub.aclose()


# ---------------------------------------------------------------------------
# RelayError.message is also scrubbed
# ---------------------------------------------------------------------------


def test_relay_error_message_is_scrubbed() -> None:
    """Providers stuff body excerpts into the message. PR5 scrubbed .raw;
    PR8 also scrubs the message so str(err) doesn't leak."""
    err = ProviderError(
        "upstream said: Authorization: Bearer sk-leak-this-please-12345",
    )
    assert "sk-leak-this-please" not in str(err)
    assert "***" in str(err)


def test_relay_error_scrubs_github_pat() -> None:
    err = ProviderError("upstream echoed token=ghp_abcdefghijklmnopqrstuvwxyz0123456789")
    assert "ghp_abcdefghijklmnopqrstuvwxyz" not in str(err)


def test_relay_error_scrubs_jwt() -> None:
    jwt = "eyJhbGciOiJIUzI1NiIs.eyJzdWIiOiIxMjM0NTY3ODkwIn0.SflKxwRJSMeKKF2QT4fwpMeJf36POk6yJV_adQssw5c"
    err = ProviderError(f"bad token: {jwt}")
    assert jwt not in str(err)


def test_relay_error_scrubs_google_aiza_key() -> None:
    key = "AIzaSyABCDEFGHIJKLMNOPQRSTUVWXYZabcd012"  # 39 chars total
    err = ProviderError(f"echoed: {key}")
    assert key not in str(err)


def test_relay_error_scrubs_pem_private_key() -> None:
    pem = (
        "-----BEGIN PRIVATE KEY-----\n"
        "MIIEvQIBADANBgkqhkiG9w0BAQEFAASCBKcwggSjAgEAAoIBAQ\n"
        "-----END PRIVATE KEY-----"
    )
    err = ProviderError(f"echoed key:\n{pem}\nrest")
    assert "MIIEvQIBADAN" not in str(err)


# ---------------------------------------------------------------------------
# CallbackSink sync-warning catches more shapes
# ---------------------------------------------------------------------------


def test_callback_sink_warns_on_callable_class() -> None:
    """A callable class with a sync __call__ would silently block the loop
    under the old isfunction-only check."""

    class _Sink:
        def __call__(self, event: AuditEvent) -> None:
            return None

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        CallbackSink(_Sink())
    assert any(
        issubclass(w.category, UserWarning) and "sync callback" in str(w.message)
        for w in caught
    )


def test_callback_sink_warns_on_lambda() -> None:
    """Lambdas are sync — should warn."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        CallbackSink(lambda event: None)
    assert any(
        issubclass(w.category, UserWarning) and "sync callback" in str(w.message)
        for w in caught
    )
