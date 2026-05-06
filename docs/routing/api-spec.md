# Hosted router wire protocol

This is the contract between `relay.routing.SemanticRouter` and any compatible
routing service (the ai5labs-hosted router, or a self-hosted deployment).
Stable as of `ai5labs-relay` v0.2.0.

## Endpoint

```
POST {endpoint}/v1/route
Authorization: Bearer <api_key>
Content-Type: application/json
```

The default `endpoint` for `SemanticRouter` is
`https://router.relay.ai5labs.com`. Self-hosters override via the
constructor.

## Request body

```json
{
  "messages": [
    {"role": "user", "content": "..."}
  ],
  "candidates": ["smart", "fast"],
  "constraints": {
    "budget": "cheap",
    "needs": ["tools", "vision"],
    "exclude_models": [],
    "prefer_models": [],
    "max_cost_per_1m": null,
    "min_quality_index": null
  },
  "metadata": {"request_id": "..."}
}
```

Field semantics:

| field | type | notes |
| ----- | ---- | ----- |
| `messages` | array of `{role, content}` | Reuses the Relay `Message` schema. Routers may use this to classify intent. |
| `candidates` | array of strings | Aliases the caller is willing to dispatch to. Empty array means "use the full catalog" by convention. |
| `constraints.budget` | `"cheap" \| "balanced" \| "premium" \| null` | Coarse cost tier. `cheap` ≈ avg <$1/1M, `balanced` ≈ avg <$10/1M, `premium` = no upper bound. |
| `constraints.needs` | array of strings | Required capabilities — e.g. `tools`, `vision`, `thinking`, `json_mode`. All must be present. |
| `constraints.exclude_models` | array of strings | Aliases to drop. |
| `constraints.prefer_models` | array of strings | Aliases to bias toward (not a hard filter). |
| `constraints.max_cost_per_1m` | `float \| null` | Hard upper bound on average cost per 1M tokens, USD. |
| `constraints.min_quality_index` | `int \| null` | Minimum acceptable composite quality index (0-100). |
| `constraints` | object \| null | The whole object may be `null` for "no constraints". |
| `metadata` | object \| null | Free-form key/value strings forwarded for tracing/billing. |

## Success response — HTTP 200

```json
{
  "alias": "smart",
  "confidence": 0.87,
  "reasoning": "Code task; smart > fast on HumanEval at acceptable cost.",
  "alternates": [["fast", 0.62]],
  "classified_intent": "code",
  "source": "hosted",
  "ts": "2026-05-06T12:34:56Z"
}
```

| field | type | notes |
| ----- | ---- | ----- |
| `alias` | string | The chosen alias. Must be drawn from `candidates`. |
| `confidence` | float in `[0.0, 1.0]` | Heuristic confidence in the choice. |
| `reasoning` | string \| null | Optional human-readable explanation. |
| `alternates` | array of `[alias, score]` pairs | Runner-ups, descending by preference. Suitable for client-side fallback. |
| `classified_intent` | string \| null | Intent label assigned by the router (e.g. `code`, `reasoning`, `chat`, `math`, `vision`). |
| `source` | `"rule" \| "hosted" \| "self-hosted"` | Which kind of router emitted the decision. |
| `ts` | RFC 3339 timestamp string | Decision timestamp. |

## Error response — any non-200

```json
{
  "error_code": "INVALID_REQUEST",
  "message": "candidates must contain at least one alias"
}
```

| `error_code` | client maps to |
| ------------ | -------------- |
| `INVALID_REQUEST` | `RoutingError` |
| `AUTH_FAILED` (or HTTP 401/403) | `RouterAuthError` |
| `QUOTA_EXCEEDED` (or HTTP 429) | `RouterQuotaError` |
| `TIMEOUT` (or HTTP 408) | `RouterTimeoutError` |
| `INTERNAL` (or any other non-200) | `RoutingError` |

The client treats HTTP-status-based mapping and `error_code`-based mapping
as equivalent; either signal triggers the matching exception class.

## Compatibility

Servers may add new fields to either request or response — clients tolerate
unknown keys (Pydantic `extra="allow"`). New `error_code` values fall back
to the generic `RoutingError`. Removing or repurposing a field is a
breaking change that requires bumping the path (`/v2/route`).
