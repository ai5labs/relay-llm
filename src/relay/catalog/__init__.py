"""Built-in model catalog with tiered pricing resolution.

The catalog is a curated list of ``(provider, model_id) -> CatalogRow`` shipped
with the library. It serves as the baseline for every model entry: when a user's
YAML specifies ``target: openai/gpt-4o-2024-11-20`` and nothing else, we look up
context window, max output, capabilities, and list pricing from the catalog.

Pricing resolution is tiered (highest priority first):

1. **User override** — explicit ``cost:`` block in YAML, or ``pricing_profiles``.
2. **Live API** — AWS Pricing API for Bedrock, Azure Retail Prices for Azure
   OpenAI, OpenRouter ``/api/v1/models`` for everyone else (in-process, refreshed
   every 6h by default).
3. **Snapshot** — the JSON shipped with this library, updated each release by an
   automated CI job.

The first tier that returns a value wins, and the resulting :class:`Cost` carries
``source=...`` so downstream callers can see provenance.
"""

from relay.catalog._loader import (
    BenchmarkScores,
    CatalogRow,
    get_catalog,
    lookup,
)
from relay.catalog._pricing import (
    PricingResolution,
    PricingResolver,
)

__all__ = [
    "BenchmarkScores",
    "CatalogRow",
    "PricingResolution",
    "PricingResolver",
    "get_catalog",
    "lookup",
]
