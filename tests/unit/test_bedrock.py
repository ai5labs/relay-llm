"""Bedrock adapter tests.

We mock the boto3 client so tests don't touch the AWS network or require creds.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from relay import Hub
from relay.config import load_str


def _yaml() -> str:
    return """
        version: 1
        catalog:
          fetch_live_pricing: false
          offline: true
        models:
          m:
            target: bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0
            region: us-east-1
        """


@pytest.fixture
def mocked_boto() -> MagicMock:
    """Replace boto3.client with a mock returning canned converse() results."""
    fake_client = MagicMock()
    fake_client.converse.return_value = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [{"text": "hi from bedrock"}],
            }
        },
        "stopReason": "end_turn",
        "usage": {"inputTokens": 5, "outputTokens": 3},
        "ResponseMetadata": {"RequestId": "req-bedrock-1"},
    }
    with patch("boto3.client", return_value=fake_client):
        yield fake_client


@pytest.mark.asyncio
async def test_bedrock_basic_chat(mocked_boto: MagicMock) -> None:
    hub = Hub.from_config(load_str(_yaml()))
    try:
        resp = await hub.chat("m", messages=[{"role": "user", "content": "hi"}])
        assert resp.text == "hi from bedrock"
        assert resp.usage.input_tokens == 5
        assert resp.usage.output_tokens == 3
        assert resp.choices[0].finish_reason == "stop"
        # Verify boto3 was called with the right modelId
        called_args = mocked_boto.converse.call_args.kwargs
        assert called_args["modelId"] == "anthropic.claude-sonnet-4-5-20250929-v1:0"
    finally:
        await hub.aclose()


@pytest.mark.asyncio
async def test_bedrock_handles_tool_use_response(mocked_boto: MagicMock) -> None:
    mocked_boto.converse.return_value = {
        "output": {
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "toolUse": {
                            "toolUseId": "tu_1",
                            "name": "get_weather",
                            "input": {"city": "Paris"},
                        }
                    }
                ],
            }
        },
        "stopReason": "tool_use",
        "usage": {"inputTokens": 10, "outputTokens": 5},
    }
    hub = Hub.from_config(load_str(_yaml()))
    try:
        resp = await hub.chat("m", messages=[{"role": "user", "content": "weather?"}])
        assert len(resp.tool_calls) == 1
        assert resp.tool_calls[0].name == "get_weather"
        assert resp.tool_calls[0].arguments == {"city": "Paris"}
        assert resp.choices[0].finish_reason == "tool_calls"
    finally:
        await hub.aclose()


@pytest.mark.asyncio
async def test_bedrock_translates_throttling_to_rate_limit(mocked_boto: MagicMock) -> None:
    from botocore.exceptions import ClientError

    from relay.errors import RateLimitError

    mocked_boto.converse.side_effect = ClientError(
        error_response={"Error": {"Code": "ThrottlingException", "Message": "slow down"}},
        operation_name="Converse",
    )
    hub = Hub.from_config(load_str(_yaml()))
    try:
        # Disable retries so the error surfaces
        cfg = hub.config
        hub2 = Hub.from_config(
            cfg.model_copy(update={"defaults": cfg.defaults.model_copy(update={"max_retries": 0})})
        )
        try:
            with pytest.raises(RateLimitError):
                await hub2.chat("m", messages=[{"role": "user", "content": "hi"}])
        finally:
            await hub2.aclose()
    finally:
        await hub.aclose()


@pytest.mark.asyncio
async def test_bedrock_translates_access_denied_to_authentication(
    mocked_boto: MagicMock,
) -> None:
    from botocore.exceptions import ClientError

    from relay.errors import AuthenticationError

    mocked_boto.converse.side_effect = ClientError(
        error_response={"Error": {"Code": "AccessDeniedException", "Message": "no"}},
        operation_name="Converse",
    )
    hub = Hub.from_config(load_str(_yaml()))
    try:
        with pytest.raises(AuthenticationError):
            await hub.chat("m", messages=[{"role": "user", "content": "hi"}])
    finally:
        await hub.aclose()
