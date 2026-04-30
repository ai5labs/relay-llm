"""Live pricing fetcher tests.

We don't hit the real AWS / Azure / OpenRouter endpoints — we mock the HTTP
calls with respx and verify the parsing + indexing logic.
"""

from __future__ import annotations

import httpx
import pytest
import respx

from relay.catalog._pricing import (
    _fetch_azure_pricing,
    _fetch_bedrock_pricing,
    _fetch_openrouter_pricing,
    _reset_caches_for_test,
)
from relay.config._schema import ModelEntry


@pytest.fixture(autouse=True)
def _reset_pricing_caches() -> None:
    _reset_caches_for_test()


# ---------------------------------------------------------------------------
# OpenRouter
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_openrouter_returns_priced_resolution() -> None:
    respx.get("https://openrouter.ai/api/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "openai/gpt-4o",
                        "pricing": {"prompt": "0.0000025", "completion": "0.00001"},
                    },
                    {
                        "id": "anthropic/claude-sonnet-4-5",
                        "pricing": {"prompt": "0.000003", "completion": "0.000015"},
                    },
                ]
            },
        )
    )
    entry = ModelEntry(target="openai/gpt-4o")
    res = await _fetch_openrouter_pricing(entry)
    assert res is not None
    assert res.input_per_1m == pytest.approx(2.5)
    assert res.output_per_1m == pytest.approx(10.0)
    assert res.source == "live_api"


@pytest.mark.asyncio
@respx.mock
async def test_openrouter_unknown_model_returns_none() -> None:
    respx.get("https://openrouter.ai/api/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={"data": [{"id": "x/y", "pricing": {"prompt": "0.001", "completion": "0.001"}}]},
        )
    )
    entry = ModelEntry(target="openai/never-shipped")
    res = await _fetch_openrouter_pricing(entry)
    assert res is None


@pytest.mark.asyncio
@respx.mock
async def test_openrouter_500_returns_none() -> None:
    respx.get("https://openrouter.ai/api/v1/models").mock(return_value=httpx.Response(500))
    entry = ModelEntry(target="openai/gpt-4o")
    res = await _fetch_openrouter_pricing(entry)
    assert res is None


@pytest.mark.asyncio
@respx.mock
async def test_openrouter_vertex_falls_back_to_google_slug() -> None:
    respx.get("https://openrouter.ai/api/v1/models").mock(
        return_value=httpx.Response(
            200,
            json={
                "data": [
                    {
                        "id": "google/gemini-2.5-flash",
                        "pricing": {"prompt": "0.0000003", "completion": "0.0000025"},
                    }
                ]
            },
        )
    )
    entry = ModelEntry(target="vertex/gemini-2.5-flash", project="acme")
    res = await _fetch_openrouter_pricing(entry)
    assert res is not None
    assert res.input_per_1m == pytest.approx(0.3)


# ---------------------------------------------------------------------------
# Bedrock (AWS Pricing API)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_bedrock_pricing_extracts_input_output_per_1m() -> None:
    # Minimal AWS Pricing offer-file shape.
    respx.get(
        "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrock/current/us-east-1/index.json"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "products": {
                    "SKU_IN": {
                        "attributes": {
                            "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
                            "inferenceType": "On-Demand input-tokens",
                        }
                    },
                    "SKU_OUT": {
                        "attributes": {
                            "model": "anthropic.claude-sonnet-4-5-20250929-v1:0",
                            "inferenceType": "On-Demand output-tokens",
                        }
                    },
                },
                "terms": {
                    "OnDemand": {
                        "SKU_IN": {
                            "term1": {
                                "priceDimensions": {"rate1": {"pricePerUnit": {"USD": "0.003"}}}
                            }
                        },
                        "SKU_OUT": {
                            "term1": {
                                "priceDimensions": {"rate1": {"pricePerUnit": {"USD": "0.015"}}}
                            }
                        },
                    }
                },
            },
        )
    )
    entry = ModelEntry(
        target="bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0", region="us-east-1"
    )
    res = await _fetch_bedrock_pricing(entry)
    assert res is not None
    # 0.003 per 1k → 3.0 per 1M
    assert res.input_per_1m == pytest.approx(3.0)
    assert res.output_per_1m == pytest.approx(15.0)
    assert res.source == "live_api"


@pytest.mark.asyncio
@respx.mock
async def test_bedrock_pricing_skips_provisioned() -> None:
    respx.get(
        "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrock/current/us-east-1/index.json"
    ).mock(
        return_value=httpx.Response(
            200,
            json={
                "products": {
                    "SKU_PROV": {
                        "attributes": {
                            "model": "anthropic.foo",
                            "inferenceType": "Provisioned Throughput",
                        }
                    }
                },
                "terms": {"OnDemand": {}},
            },
        )
    )
    entry = ModelEntry(target="bedrock/anthropic.foo", region="us-east-1")
    res = await _fetch_bedrock_pricing(entry)
    # Only provisioned entries — should return None for on-demand pricing.
    assert res is None


@pytest.mark.asyncio
@respx.mock
async def test_bedrock_pricing_unknown_region_returns_none() -> None:
    respx.get(
        "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrock/current/eu-fake-1/index.json"
    ).mock(return_value=httpx.Response(404))
    entry = ModelEntry(target="bedrock/anthropic.foo", region="eu-fake-1")
    res = await _fetch_bedrock_pricing(entry)
    assert res is None


# ---------------------------------------------------------------------------
# Azure Retail Prices
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@respx.mock
async def test_azure_retail_extracts_input_output() -> None:
    respx.get("https://prices.azure.com/api/retail/prices").mock(
        return_value=httpx.Response(
            200,
            json={
                "Items": [
                    {"meterName": "gpt-4o input tokens", "retailPrice": 0.0025},
                    {"meterName": "gpt-4o output tokens", "retailPrice": 0.01},
                ]
            },
        )
    )
    entry = ModelEntry(target="azure/gpt-4o", deployment="gpt-4o", base_url="https://x")
    res = await _fetch_azure_pricing(entry)
    assert res is not None
    # 0.0025 per 1k → 2.5 per 1M
    assert res.input_per_1m == pytest.approx(2.5)
    assert res.output_per_1m == pytest.approx(10.0)
    assert res.source == "live_api"


@pytest.mark.asyncio
@respx.mock
async def test_azure_retail_500_returns_none() -> None:
    respx.get("https://prices.azure.com/api/retail/prices").mock(return_value=httpx.Response(500))
    entry = ModelEntry(target="azure/gpt-4o", deployment="gpt-4o", base_url="https://x")
    res = await _fetch_azure_pricing(entry)
    assert res is None
