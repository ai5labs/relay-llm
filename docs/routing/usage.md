# Routing

Relay's `relay.routing` package is the public extension point for picking a
model per call. Two implementations ship with v0.2:

- `RuleBasedRouter` — deterministic, constraint-driven. Lifts the same
  scoring logic that powers `relay models recommend`. Free, OSS,
  in-process, no network dependency.
- `SemanticRouter` — HTTP client for the hosted semantic router (paid).
  Wire protocol documented at [`api-spec.md`](api-spec.md).

Both implement the `Router` Protocol; you can also write your own.

## Rule-based router

Pure constraint-driven choice — no LLM call, no message inspection,
deterministic for a given catalog + request.

```python
import asyncio
from relay.routing import RuleBasedRouter, RouteRequest, RouteConstraints
from relay.types import Message

async def main() -> None:
    router = RuleBasedRouter()
    decision = await router.route(
        RouteRequest(
            messages=[Message(role="user", content="write me a Python function")],
            candidates=[],  # empty list = pick from the full built-in catalog
            constraints=RouteConstraints(budget="cheap", needs=["tools"]),
        )
    )
    print(decision.alias, decision.confidence, decision.alternates)

asyncio.run(main())
```

Key fields on `RouteConstraints`:

- `budget`: `"cheap" | "balanced" | "premium" | None` — coarse cost tier.
- `needs`: required capabilities, e.g. `["tools", "vision", "thinking"]`.
- `exclude_models` / `prefer_models`: alias-level filters and biases.
- `max_cost_per_1m`: hard upper bound on average cost per 1M tokens, USD.
- `min_quality_index`: minimum composite quality (0-100).

## Semantic router (hosted)

`SemanticRouter` is a thin HTTP client. Point it at the ai5labs-hosted
service or any compatible self-hosted deployment.

```python
from relay.routing import SemanticRouter, RouteRequest, RouteConstraints

router = SemanticRouter(api_key="...")  # default endpoint
try:
    decision = await router.route(
        RouteRequest(
            messages=[...],
            candidates=["smart", "fast"],
            constraints=RouteConstraints(budget="balanced"),
        )
    )
finally:
    await router.aclose()
```

Errors map onto the `RoutingError` hierarchy: `RouterAuthError` (401/403),
`RouterQuotaError` (429), `RouterTimeoutError`, `NoCandidatesError`. The
default endpoint is `https://router.relay.ai5labs.com`; pass `endpoint=`
to override.

## Hub integration

Attach a router and use `chat_routed` instead of `chat`:

```python
from relay import Hub
from relay.routing import RuleBasedRouter, RouteConstraints

async with Hub.from_yaml("models.yaml") as hub:
    hub.attach_router(RuleBasedRouter())
    response = await hub.chat_routed(
        messages=[{"role": "user", "content": "summarize this"}],
        constraints=RouteConstraints(budget="cheap"),
    )
    print(response.text)
    print(response.metadata["routing"].alias)
```

When the chosen alias errors, `chat_routed` falls through the
`alternates` list automatically. The successful `RouteDecision` is
attached to `response.metadata["routing"]`.

## Custom routers

Anything satisfying the `Router` Protocol works:

```python
from datetime import datetime, timezone
from relay.routing import Router, RouteRequest, RouteDecision

class MyRouter:
    async def route(self, request: RouteRequest) -> RouteDecision:
        return RouteDecision(
            alias=request.candidates[0],
            confidence=1.0,
            source="self-hosted",
            ts=datetime.now(timezone.utc),
        )

    async def aclose(self) -> None:
        return None
```
