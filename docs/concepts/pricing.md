# Pricing &amp; cost tracking

Every chat response carries a `Cost` object with full provenance:

```python
resp.cost.total_usd        # 0.00234
resp.cost.input_usd
resp.cost.output_usd
resp.cost.cached_input_usd
resp.cost.reasoning_usd
resp.cost.source           # "live_api" | "snapshot" | "user_override" | "estimated" | "unknown"
resp.cost.confidence       # "exact" | "list_price" | "estimated" | "unknown"
resp.cost.fetched_at       # ISO 8601 timestamp
```

## Tier order

First match wins:

1. **User override** — explicit `cost:` block on the model entry, or a `pricing_profile` reference.
2. **Live API** (cached 6h in-process):
    - **AWS Pricing API** for Bedrock — `pricing.us-east-1.amazonaws.com`, free, no auth.
    - **Azure Retail Prices API** for Azure OpenAI — `prices.azure.com/api/retail/prices`, free, no auth.
    - **OpenRouter `/api/v1/models`** — covers ~400 models from OpenAI, Anthropic, Google, Groq, Mistral, etc. at list price.
3. **Snapshot** — JSON shipped with each release, regenerated weekly via CI from OpenRouter + curated overrides.
4. **Unknown** — `cost.total_usd = 0.0`, `cost.source = "unknown"`.

## Negotiated rates

No public API exposes enterprise discounts (AWS EDP, Azure committed-use, OpenAI custom tiers). Configure them yourself:

```yaml
pricing_profiles:
  acme-aws-prod:
    description: "15% EDP discount"
    input_multiplier: 0.85
    output_multiplier: 0.85

  openai-team-tier:
    fixed_overrides:
      openai/gpt-4o:
        input_per_1m: 1.25
        output_per_1m: 5.00

models:
  bedrock-sonnet:
    target: bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0
    credential: { type: aws_profile, profile: prod }
    pricing_profile: acme-aws-prod
```

Profiles compose with the catalog — `multiplier` scales the underlying tier, `fixed_overrides` replace it entirely for specific slugs.

## Catalog refresh

Run weekly via `.github/workflows/refresh_catalog.yml`. The job:

1. Fetches OpenRouter `/api/v1/models` (~400 models with prices).
2. Loads `scripts/curated.json` for capability flags + cached pricing tiers.
3. Merges with the existing `src/relay/catalog/data/models.json`.
4. Opens a PR with the diff for human review.

Run locally:

```bash
uv run python scripts/refresh_catalog.py --dry-run
uv run python scripts/refresh_catalog.py            # in-place
```
