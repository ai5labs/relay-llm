"""Tiered pricing resolver.

Order of precedence (first match wins):

1. **User override** — explicit ``cost:`` block on the model entry, or a
   ``pricing_profile`` referenced from the entry, possibly via ``fixed_overrides``.
2. **Live API** — refreshed in-process every ``catalog.refresh_interval_hours``.
   - AWS Pricing API for Bedrock (`pricing.us-east-1.amazonaws.com`).
   - Azure Retail Prices API for Azure OpenAI (`prices.azure.com`).
   - OpenRouter `/api/v1/models` for OpenAI/Anthropic/Google/Groq/etc. list prices.
3. **Snapshot** — the JSON shipped with this library.
4. **Estimated** — fallback heuristic. We currently mark these ``confidence="unknown"``
   and return ``None`` for the cost; a future version may apply tier-based estimates.

A successful resolution carries provenance metadata (``source``, ``fetched_at``,
``confidence``) that propagates onto every :class:`Cost` result.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Literal

import httpx
import orjson

from relay.catalog._loader import CatalogRow, lookup
from relay.config._schema import CatalogSettings, ModelEntry, PricingProfile

PricingSource = Literal["live_api", "snapshot", "user_override", "estimated", "unknown"]
PricingConfidence = Literal["exact", "list_price", "estimated", "unknown"]


@dataclass(frozen=True, slots=True)
class PricingResolution:
    """Resolved per-1M-token rates for one model.

    All values are USD per million tokens. ``None`` means "unknown — don't bill".
    """

    input_per_1m: float | None
    output_per_1m: float | None
    cached_input_per_1m: float | None = None
    source: PricingSource = "unknown"
    fetched_at: str | None = None
    confidence: PricingConfidence = "unknown"

    def is_complete(self) -> bool:
        return self.input_per_1m is not None and self.output_per_1m is not None


# ---------------------------------------------------------------------------
# Resolver
# ---------------------------------------------------------------------------


@dataclass
class _LiveCacheEntry:
    resolution: PricingResolution
    expires_at: float


class PricingResolver:
    """Resolves a per-call price against the configured tier order.

    The resolver is async because live tiers hit the network; for the snapshot-only
    path it returns synchronously fast. One resolver per :class:`Hub`.
    """

    def __init__(
        self,
        *,
        settings: CatalogSettings,
        pricing_profiles: dict[str, PricingProfile] | None = None,
    ) -> None:
        self._settings = settings
        self._profiles = pricing_profiles or {}
        self._live_cache: dict[str, _LiveCacheEntry] = {}
        self._live_lock = asyncio.Lock()

    # -- public API ---------------------------------------------------------

    async def resolve(self, entry: ModelEntry) -> PricingResolution:
        # Tier 1: explicit cost on the entry.
        if entry.cost:
            return self._from_user_cost(entry.cost)

        # Tier 1b: pricing profile fixed_override for this exact slug.
        profile = self._profiles.get(entry.pricing_profile or "") if entry.pricing_profile else None
        if profile and profile.fixed_overrides:
            override = profile.fixed_overrides.get(entry.target)
            if override:
                return self._from_user_cost(override)

        # Tier 2: live API (network).
        if self._settings.fetch_live_pricing and not self._settings.offline:
            live = await self._resolve_live(entry)
            if live is not None:
                return self._apply_profile_multipliers(live, profile)

        # Tier 3: snapshot.
        snapshot = self._resolve_snapshot(entry)
        if snapshot is not None:
            return self._apply_profile_multipliers(snapshot, profile)

        # Tier 4: unknown.
        return PricingResolution(
            input_per_1m=None,
            output_per_1m=None,
            source="unknown",
            confidence="unknown",
        )

    # -- tier implementations -----------------------------------------------

    @staticmethod
    def _from_user_cost(cost: dict[str, float]) -> PricingResolution:
        return PricingResolution(
            input_per_1m=cost.get("input_per_1m"),
            output_per_1m=cost.get("output_per_1m"),
            cached_input_per_1m=cost.get("cached_input_per_1m"),
            source="user_override",
            confidence="exact",
        )

    @staticmethod
    def _resolve_snapshot(entry: ModelEntry) -> PricingResolution | None:
        # If user used inherit_from, look that up; otherwise look up the target itself.
        slug = entry.inherit_from or entry.target
        provider, _, model_id = slug.partition("/")
        row: CatalogRow | None = lookup(provider, model_id)
        if row is None:
            return None
        if row.input_per_1m is None and row.output_per_1m is None:
            return None
        return PricingResolution(
            input_per_1m=row.input_per_1m,
            output_per_1m=row.output_per_1m,
            cached_input_per_1m=row.cached_input_per_1m,
            source="snapshot",
            confidence="list_price",
        )

    async def _resolve_live(self, entry: ModelEntry) -> PricingResolution | None:
        """Hit a provider-appropriate live pricing source. Cached per-process."""
        cache_key = entry.target + (f"|{entry.region or ''}" if entry.region else "")
        now = time.time()
        cached = self._live_cache.get(cache_key)
        if cached and cached.expires_at > now:
            return cached.resolution

        async with self._live_lock:
            cached = self._live_cache.get(cache_key)
            if cached and cached.expires_at > now:
                return cached.resolution

            resolution: PricingResolution | None = None
            try:
                if entry.provider == "bedrock":
                    resolution = await _fetch_bedrock_pricing(entry)
                elif entry.provider == "azure":
                    resolution = await _fetch_azure_pricing(entry)
                else:
                    # Most providers: try OpenRouter as the cross-provider live source.
                    resolution = await _fetch_openrouter_pricing(entry)
            except Exception:
                resolution = None

            if resolution is not None:
                ttl = self._settings.refresh_interval_hours * 3600
                self._live_cache[cache_key] = _LiveCacheEntry(
                    resolution=resolution,
                    expires_at=now + ttl,
                )
            return resolution

    @staticmethod
    def _apply_profile_multipliers(
        base: PricingResolution,
        profile: PricingProfile | None,
    ) -> PricingResolution:
        if profile is None:
            return base
        return PricingResolution(
            input_per_1m=(
                base.input_per_1m * profile.input_multiplier
                if base.input_per_1m is not None
                else None
            ),
            output_per_1m=(
                base.output_per_1m * profile.output_multiplier
                if base.output_per_1m is not None
                else None
            ),
            cached_input_per_1m=(
                base.cached_input_per_1m * profile.cached_input_multiplier
                if base.cached_input_per_1m is not None
                else None
            ),
            source=base.source,
            fetched_at=base.fetched_at,
            confidence="exact" if profile.fixed_overrides else base.confidence,
        )


# ---------------------------------------------------------------------------
# Live-pricing fetchers
#
# Stubbed for v0.1. Each is a future enhancement — the architecture is in place,
# the network code is intentionally narrow so it can be reviewed independently.
# All three return ``None`` until implemented; the snapshot tier covers v0.1.
# ---------------------------------------------------------------------------


# Singleton fetcher caches — built lazily so tests/users that opt out via
# ``catalog.fetch_live_pricing=false`` never make any network calls.
_OPENROUTER_INDEX: dict[str, dict[str, Any]] | None = None
_OPENROUTER_FETCHED_AT: float | None = None
_OPENROUTER_LOCK = asyncio.Lock()

_BEDROCK_INDEX: dict[str, dict[str, Any]] = {}
_BEDROCK_FETCHED_AT: dict[str, float] = {}  # keyed by region
_BEDROCK_LOCK = asyncio.Lock()

_AZURE_INDEX: dict[str, dict[str, Any]] | None = None
_AZURE_FETCHED_AT: float | None = None
_AZURE_LOCK = asyncio.Lock()

# Per-fetcher cache TTL: 6 hours.
_LIVE_TTL_S = 6 * 3600


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


async def _fetch_bedrock_pricing(entry: ModelEntry) -> PricingResolution | None:
    """AWS Pricing API for Bedrock — free, public, no auth needed.

    Endpoint: ``https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrock/current/{region}/index.json``
    Returns the per-region offer file. We cache the parsed model→price index for
    ``_LIVE_TTL_S`` seconds. The shape is dense; we only extract per-1M
    input/output USD pricing for the model id.
    """
    global _BEDROCK_FETCHED_AT
    region = entry.region or "us-east-1"
    now = time.time()
    fetched = _BEDROCK_FETCHED_AT.get(region, 0.0)
    if not _BEDROCK_INDEX.get(region) or (now - fetched) > _LIVE_TTL_S:
        async with _BEDROCK_LOCK:
            fetched = _BEDROCK_FETCHED_AT.get(region, 0.0)
            if not _BEDROCK_INDEX.get(region) or (now - fetched) > _LIVE_TTL_S:
                index = await _fetch_bedrock_offer(region)
                if index is not None:
                    _BEDROCK_INDEX[region] = index
                    _BEDROCK_FETCHED_AT[region] = now

    region_index = _BEDROCK_INDEX.get(region)
    if not region_index:
        return None
    row = region_index.get(entry.model_id)
    if not row:
        return None
    return PricingResolution(
        input_per_1m=row.get("input_per_1m"),
        output_per_1m=row.get("output_per_1m"),
        cached_input_per_1m=row.get("cached_input_per_1m"),
        source="live_api",
        fetched_at=_now_iso(),
        confidence="list_price",
    )


async def _fetch_bedrock_offer(region: str) -> dict[str, dict[str, Any]] | None:
    """Fetch + parse the Bedrock offer file for ``region``."""
    url = (
        f"https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/AmazonBedrock/"
        f"current/{region}/index.json"
    )
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get(url)
            if resp.status_code != 200:
                return None
            data = orjson.loads(resp.content)
    except Exception:
        return None

    # offer file shape: { products: { sku: {...} }, terms: { OnDemand: { sku: {...} } } }
    products = data.get("products") or {}
    terms = (data.get("terms") or {}).get("OnDemand") or {}
    index: dict[str, dict[str, Any]] = {}

    for sku, prod in products.items():
        attrs = prod.get("attributes") or {}
        model_id = attrs.get("model")
        if not model_id:
            continue
        usage_type = (attrs.get("inferenceType") or attrs.get("usagetype") or "").lower()
        # We want On-Demand, not provisioned throughput.
        if "provisioned" in usage_type:
            continue
        is_input = "input" in usage_type or "input-tokens" in usage_type
        is_output = "output" in usage_type or "output-tokens" in usage_type
        is_cache_read = "cache" in usage_type and "read" in usage_type
        if not (is_input or is_output or is_cache_read):
            continue

        sku_terms = terms.get(sku) or {}
        # Each sku has one or more rate codes; pick the first.
        for term in sku_terms.values():
            for dim in (term.get("priceDimensions") or {}).values():
                price_per_unit = (dim.get("pricePerUnit") or {}).get("USD")
                if price_per_unit is None:
                    continue
                # Bedrock prices are per-1k-tokens; convert to per-1M.
                try:
                    per_1k = float(price_per_unit)
                except ValueError:
                    continue
                per_1m = per_1k * 1000
                row = index.setdefault(model_id, {})
                if is_cache_read:
                    row["cached_input_per_1m"] = per_1m
                elif is_input:
                    row["input_per_1m"] = per_1m
                elif is_output:
                    row["output_per_1m"] = per_1m
                break
            break
    return index


async def _fetch_azure_pricing(entry: ModelEntry) -> PricingResolution | None:
    """Azure Retail Prices API — free, public, no auth.

    Endpoint: ``https://prices.azure.com/api/retail/prices`` with OData filtering.
    Cached process-wide for 6 hours. Indexed by Azure ``meterName`` which usually
    matches the underlying model id.
    """
    global _AZURE_INDEX, _AZURE_FETCHED_AT
    now = time.time()
    if _AZURE_INDEX is None or (
        _AZURE_FETCHED_AT is not None and (now - _AZURE_FETCHED_AT) > _LIVE_TTL_S
    ):
        async with _AZURE_LOCK:
            if _AZURE_INDEX is None or (
                _AZURE_FETCHED_AT is not None and (now - _AZURE_FETCHED_AT) > _LIVE_TTL_S
            ):
                index = await _fetch_azure_index()
                if index is not None:
                    _AZURE_INDEX = index
                    _AZURE_FETCHED_AT = now

    if not _AZURE_INDEX:
        return None
    # Try direct match on model id first; fall back to deployment.
    row = _AZURE_INDEX.get(entry.model_id) or _AZURE_INDEX.get(entry.deployment or "")
    if not row:
        return None
    return PricingResolution(
        input_per_1m=row.get("input_per_1m"),
        output_per_1m=row.get("output_per_1m"),
        source="live_api",
        fetched_at=_now_iso(),
        confidence="list_price",
    )


async def _fetch_azure_index() -> dict[str, dict[str, Any]] | None:
    """Fetch a paginated set of Azure OpenAI retail prices and index by meterName."""
    base_url = "https://prices.azure.com/api/retail/prices"
    params = {
        "$filter": "serviceName eq 'Cognitive Services' and productName eq 'Azure OpenAI'",
        "api-version": "2023-01-01-preview",
    }
    index: dict[str, dict[str, Any]] = {}
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            url: str | None = base_url
            page = 0
            while url and page < 20:  # safety cap
                resp = await client.get(url, params=params if page == 0 else None)
                if resp.status_code != 200:
                    return None
                data = orjson.loads(resp.content)
                for item in data.get("Items") or []:
                    meter = (item.get("meterName") or "").lower()
                    is_input = "input" in meter
                    is_output = "output" in meter
                    if not (is_input or is_output):
                        continue
                    # Strip suffixes: "gpt-4o-2024-11-20 inp regional" → model id
                    model_id = (
                        meter.replace(" input tokens", "")
                        .replace(" output tokens", "")
                        .replace(" inp", "")
                        .replace(" out", "")
                        .replace(" regional", "")
                        .strip()
                    )
                    price = item.get("retailPrice")
                    if price is None:
                        continue
                    # Azure quotes per-1k tokens.
                    per_1m = float(price) * 1000
                    row = index.setdefault(model_id, {})
                    if is_input:
                        row["input_per_1m"] = per_1m
                    elif is_output:
                        row["output_per_1m"] = per_1m
                url = data.get("NextPageLink")
                page += 1
    except Exception:
        return None
    return index


async def _fetch_openrouter_pricing(entry: ModelEntry) -> PricingResolution | None:
    """OpenRouter ``/api/v1/models`` — covers ~400 models at list price.

    Each model returns ``pricing.{prompt, completion}`` in USD per token (not
    per-1k or per-1M). We multiply by 1e6 for the per-1M figure.
    """
    global _OPENROUTER_INDEX, _OPENROUTER_FETCHED_AT
    now = time.time()
    if _OPENROUTER_INDEX is None or (
        _OPENROUTER_FETCHED_AT is not None and (now - _OPENROUTER_FETCHED_AT) > _LIVE_TTL_S
    ):
        async with _OPENROUTER_LOCK:
            if _OPENROUTER_INDEX is None or (
                _OPENROUTER_FETCHED_AT is not None and (now - _OPENROUTER_FETCHED_AT) > _LIVE_TTL_S
            ):
                index = await _fetch_openrouter_models()
                if index is not None:
                    _OPENROUTER_INDEX = index
                    _OPENROUTER_FETCHED_AT = now

    if not _OPENROUTER_INDEX:
        return None
    # OpenRouter's slug is "openai/gpt-4o" which matches Relay's target. Try
    # exact match on entry.target first; then on a few normalized variants.
    candidates = _openrouter_candidates(entry)
    for cand in candidates:
        row = _OPENROUTER_INDEX.get(cand)
        if row is not None:
            return PricingResolution(
                input_per_1m=row.get("input_per_1m"),
                output_per_1m=row.get("output_per_1m"),
                source="live_api",
                fetched_at=_now_iso(),
                confidence="list_price",
            )
    return None


def _openrouter_candidates(entry: ModelEntry) -> list[str]:
    """Possible OpenRouter slug spellings to look up.

    OpenRouter sometimes uses ``provider/model`` slugs that differ from ours
    (``google/`` vs ``vertex/`` for Gemini, etc.). We try a few permutations.
    """
    out = [entry.target]
    if entry.provider == "vertex":
        out.append(f"google/{entry.model_id}")
    if entry.provider == "azure":
        out.append(f"openai/{entry.model_id}")
    if entry.provider == "google":
        out.append(f"google/{entry.model_id}")
    return out


async def _fetch_openrouter_models() -> dict[str, dict[str, Any]] | None:
    try:
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.get("https://openrouter.ai/api/v1/models")
            if resp.status_code != 200:
                return None
            data = orjson.loads(resp.content)
    except Exception:
        return None

    index: dict[str, dict[str, Any]] = {}
    for m in data.get("data") or []:
        slug = m.get("id")
        pricing = m.get("pricing") or {}
        if not slug:
            continue
        try:
            prompt = float(pricing.get("prompt") or 0)
            completion = float(pricing.get("completion") or 0)
        except ValueError:
            continue
        if prompt == 0 and completion == 0:
            continue
        index[slug] = {
            "input_per_1m": prompt * 1_000_000,
            "output_per_1m": completion * 1_000_000,
        }
    return index


def _reset_caches_for_test() -> None:
    """Test helper — clear all live-pricing caches between tests."""
    global _OPENROUTER_INDEX, _OPENROUTER_FETCHED_AT
    global _AZURE_INDEX, _AZURE_FETCHED_AT
    _OPENROUTER_INDEX = None
    _OPENROUTER_FETCHED_AT = None
    _AZURE_INDEX = None
    _AZURE_FETCHED_AT = None
    _BEDROCK_INDEX.clear()
    _BEDROCK_FETCHED_AT.clear()
