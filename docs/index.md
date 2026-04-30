# Relay

> One YAML, one interface, every model.

Relay is a Python library that gives you a unified interface to every major LLM provider. Models live in a YAML file you check into your repo; in code, you call them by alias.

```python
from relay import Hub

async with Hub.from_yaml("models.yaml") as hub:
    resp = await hub.chat("smart", messages=[
        {"role": "user", "content": "What is 2+2?"}
    ])
    print(resp.text, f"${resp.cost_usd:.4f}")
```

## What's different from LiteLLM

- **Real native adapters**, not OpenAI-compat shims, for Anthropic, Bedrock, Vertex, Azure, Gemini, Cohere — each preserves provider-specific blocks (`thinking`, `grounding`) instead of flattening them.
- **Tool-call streaming deltas keyed by `index`**, not `id` — fixes the [LiteLLM #20711 bug](https://github.com/BerriAI/litellm/issues/20711) that drops 90% of argument fragments.
- **MCP universal tool layer** — attach an MCP server (GitHub, Slack, Postgres) and use its tools against *any* provider, including ones without native MCP support.
- **Cross-provider tool-schema compiler** — compiles your JSON Schema to each provider's dialect, with the Mastra trick (instruction-injection) when constraints can't be expressed natively.
- **Pydantic structured output** — accept a Pydantic model, get a validated instance back. Compiles to OpenAI strict mode, Anthropic single-tool, or Gemini `responseSchema`.
- **Live pricing fetchers** — AWS Pricing API for Bedrock, Azure Retail Prices for Azure OpenAI, OpenRouter `/api/v1/models` for everyone else. Costs carry provenance.
- **Circuit breakers** — failing providers cool off automatically before retries thrash them.
- **OpenTelemetry GenAI semantic conventions** — first-class, opt-in. Datadog, Honeycomb, Langfuse, Arize all consume it.
- **Production-ready governance** — PII redaction, audit logging, pre/post guardrails as core OSS features.

## Get started

- [Quickstart](quickstart.md) — install + first call
- [YAML config](concepts/config.md) — define your model catalog
- [Providers](concepts/providers.md) — what's supported and how
- [API reference](api/relay.md)

## Status

**v0.2 (alpha)** — chat, streaming, tool calls, structured output, batch, MCP, caching, redaction, audit, guardrails, OTel. 162+ tests, mypy strict, ruff clean.

API surface under `relay.*` is stable. Anything in `_internal/` or under a `_*` name is not.
