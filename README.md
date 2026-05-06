# Relay

> The fastest, lightest BYOK relay for any and every LLM model — open source.

[![CI](https://github.com/ai5labs/relay-llm/actions/workflows/ci.yml/badge.svg)](https://github.com/ai5labs/relay-llm/actions/workflows/ci.yml)
[![Apache-2.0](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![Sponsor](https://img.shields.io/github/sponsors/ai5labs?label=sponsor&logo=GitHub&color=ea4aaa)](https://github.com/sponsors/ai5labs)

A Python library that gives you one interface to every major LLM — chat, streaming, tool calls, structured output, batch, MCP — defined in a YAML file you check into your repo. Production-grade, enterprise-ready, OSS.

**~5–19× faster cold start than LiteLLM**, **~20% faster streaming TTFT**, and tied at the median on chat overhead with more consistent tails ([reproducible benchmarks](BENCHMARKS.md)).

```bash
pip install ai5labs-relay
```

```python
from relay import Hub

async with Hub.from_yaml("models.yaml") as hub:
    resp = await hub.chat(
        "fast-cheap",
        messages=[{"role": "user", "content": "What is 2+2?"}],
    )
    print(resp.text)
    print(resp.cost_usd, resp.cost.source)
```

## Why Relay

| | LiteLLM | LangChain | **Relay** |
|---|---|---|---|
| YAML model catalog | ✓ | — | ✓ |
| Built-in pricing snapshot with provenance | partial | — | ✓ |
| Live pricing (Bedrock, Azure, OpenRouter) | — | — | **✓** |
| Tool-call streaming deltas keyed by `index` (not `id`) | bug ([#20711](https://github.com/BerriAI/litellm/issues/20711)) | n/a | **✓** |
| **MCP universal tool layer** (any MCP server → any provider) | — | — | **✓** |
| **Cross-provider tool-schema compiler** with Mastra-style fallback | — | — | **✓** |
| **Pydantic structured output** (compiles per-provider, not text-coerced) | — | partial | **✓** |
| **Hub-level cache + Anthropic prompt-cache passthrough** | partial | — | **✓** |
| **Circuit breakers** with cooldown + half-open probes | — | — | **✓** |
| **OpenTelemetry GenAI semantic conventions** (opt-in) | — | — | **✓** |
| **Reasoning budget unification** across OpenAI/Anthropic/Gemini | — | — | **✓** |
| **OpenAI Responses API** opt-in (alongside Chat Completions) | — | — | **✓** |
| **Batch API wrapper** (OpenAI Batch + Anthropic Message Batches, ~50% off) | — | — | **✓** |
| **Native Bedrock / Azure / Gemini / Vertex / Cohere adapters** | OpenAI-compat shims | partial | **✓** native |
| **PII redaction pipeline** (regex + Presidio hooks) | — | — | **✓** |
| **Audit logging** (OTel-aligned schema, pluggable sinks) | enterprise SKU | — | **✓** |
| **Pre/post guardrails** (max-input, blocked-keywords, plugin-able) | enterprise SKU | — | **✓** |
| Anthropic `thinking` blocks preserved | flattened | flattened | **✓** |
| Typed errors (rate-limit / context-window / content-policy distinct) | partial | — | **✓** |
| OpenTelemetry GenAI semantic conventions | — | — | ✓ (opt-in) |
| `mypy --strict` clean | — | — | ✓ |
| Apache-2.0 with explicit patent grant | MIT | MIT | ✓ |

## Quickstart

### 1. Define your models

Create `models.yaml`:

```yaml
# yaml-language-server: $schema=https://relay.ai5labs.com/schema/v1.json
version: 1

models:
  fast-cheap:
    target: groq/llama-3.3-70b-versatile
    credential: $env.GROQ_API_KEY

  smart:
    target: anthropic/claude-sonnet-4-5
    credential: $env.ANTHROPIC_API_KEY
    params:
      max_tokens: 4096

  cheap-vision:
    target: openai/gpt-4o-mini
    credential: $env.OPENAI_API_KEY

groups:
  default:
    strategy: fallback
    members: [smart, fast-cheap]    # try smart first, fall back to fast-cheap
```

Then point your editor at the schema URL on line 1 — the [Red Hat YAML extension](https://marketplace.visualstudio.com/items?itemName=redhat.vscode-yaml) for VS Code will give you autocomplete and inline validation while editing.

### 2. Use it

```python
from relay import Hub

async with Hub.from_yaml("models.yaml") as hub:
    # Single model
    resp = await hub.chat("fast-cheap", messages=[
        {"role": "user", "content": "Hello"}
    ])

    # Group with fallback
    resp = await hub.chat("default", messages=[...])

    # Streaming
    async for ev in hub.stream("smart", messages=[...]):
        if ev.type == "text_delta":
            print(ev.text, end="", flush=True)
        elif ev.type == "thinking_delta":     # Anthropic extended thinking
            ...
        elif ev.type == "end":
            print(f"\nDone in {ev.response.latency_ms:.0f}ms, "
                  f"${ev.response.cost_usd:.4f}")

    # Bound handle for hot loops
    model = hub.get("fast-cheap")
    for prompt in prompts:
        resp = await model.chat(messages=[{"role": "user", "content": prompt}])
```

### 3. CLI

```bash
relay schema --out relay.schema.json     # JSON Schema for editors / docs
relay validate models.yaml               # validate config
relay models list                        # list configured aliases
relay models inspect smart               # show one alias's full config + catalog row
relay models compare sonnet 4o flash     # side-by-side: price, speed, MMLU, GPQA, HumanEval...
relay models recommend --task code --budget cheap --needs tools  # which model for the job?
relay catalog list --provider anthropic  # browse the built-in catalog
relay providers                          # list all supported providers
```

## Supported providers

**OpenAI-compatible** (one adapter): OpenAI, Groq, Together, DeepSeek, xAI, Mistral, Fireworks, Perplexity, OpenRouter, Ollama, vLLM, LM Studio.

**Native** (proper, lossless adapters): Anthropic, Azure OpenAI, AWS Bedrock, Cohere, Google Gemini direct, Vertex AI.

## Routing

`relay.routing` is the public extension point for picking a model per call. Two implementations ship with v0.2:

- `RuleBasedRouter` — deterministic, constraint-driven, in-process. Same scoring logic as `relay models recommend`, free.
- `SemanticRouter` — HTTP client for the hosted semantic router (paid, optional). Wire protocol documented in [`docs/routing/api-spec.md`](docs/routing/api-spec.md).

Attach a router and call `chat_routed` instead of `chat` — Relay picks the alias, falls back through alternates on error, and stamps the decision onto `response.metadata["routing"]`. Custom routers satisfying the `Router` Protocol are accepted. See [`docs/routing/usage.md`](docs/routing/usage.md) for examples.

## Pricing & cost tracking

Every response carries a `Cost` object with full provenance:

```python
resp.cost.total_usd        # 0.00234
resp.cost.source           # "live_api" | "snapshot" | "user_override" | "estimated" | "unknown"
resp.cost.confidence       # "exact" | "list_price" | "estimated" | "unknown"
resp.cost.fetched_at       # ISO 8601 timestamp (when fetched live)
```

**Tier order** (first match wins):

1. **User override** — explicit `cost:` block on a model entry, or a `pricing_profile`.
2. **Live APIs** (cached 6h in-process):
   - AWS Pricing API for Bedrock
   - Azure Retail Prices API for Azure OpenAI
   - OpenRouter `/api/v1/models` for ~400 models from OpenAI, Anthropic, Google, Groq, etc. at list price
3. **Snapshot** — JSON shipped with each release, regenerated weekly via CI.
4. **Unknown** — `cost_usd = None`, never wrong-by-default.

### Negotiated rates

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
    region: us-east-1
    credential: { type: aws_profile, profile: prod }
    pricing_profile: acme-aws-prod
```

## Production-grade design

- **Connection pooling**: one `httpx.AsyncClient` per `(provider, base_url)`, HTTP/2 enabled, keep-alive tuned for streaming workloads.
- **Lazy SDK imports**: `boto3` and other heavy deps only load when their first call happens.
- **Streaming hot path** uses `orjson` and dicts — no Pydantic validation per-token. Pydantic only runs on the final assembled response.
- **Tool-call delta merging keyed by `index`**, not `id`. (LiteLLM keys by `id` and drops ~90% of argument deltas — issue [#20711](https://github.com/BerriAI/litellm/issues/20711).)
- **Provider-specific blocks preserved**: Anthropic `thinking`, Gemini `grounding`, citations — emitted as typed events, not flattened.
- **Classified errors**: `RateLimitError`, `ContextWindowError`, `ContentPolicyError`, `AuthenticationError` are distinct types — fall back vs retry vs fail-fast can be decided automatically.
- **OpenTelemetry GenAI semantic conventions** (opt-in): emits `gen_ai.*` spans + metrics that Datadog, Honeycomb, Langfuse, and Arize all consume.

## Security

- **Keys never inline in YAML** — credentials are reified objects (env var, AWS Secrets Manager, GCP Secret Manager, Vault).
- **Library, not a hosted proxy** by default. Your API keys stay in your process. (Compare: the LiteLLM proxy PyPI compromise of March 2026 leaked keys from every centralized deployment.)
- Releases will be **Sigstore-signed** via OIDC Trusted Publishing.
- See [SECURITY.md](SECURITY.md) for vulnerability reporting.

## Status

**v0.1 (alpha)** — chat + streaming + tool calls + cost tracking + the OpenAI-compatible adapter (12 providers) + native adapters for Anthropic, Azure OpenAI, AWS Bedrock, Cohere, Google Gemini direct, and Vertex AI.

API surface is stable; everything under `_internal/` and `_*` modules is not.

## Development

```bash
uv sync --all-groups
uv run pytest
uv run ruff check
uv run mypy
uv run pyright
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Please read [CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md) before opening a PR.

## Support &amp; sponsorship

Relay is free, Apache-2.0, and actively maintained by [ai5labs Research OPC Pvt Ltd](https://ai5labs.com). If your team uses it in production, please consider:

- ⭐ **[Star the repo](https://github.com/ai5labs/relay-llm)** — actually helps a lot at this stage
- 💖 **[Sponsor on GitHub](https://github.com/sponsors/ai5labs)** — pays for maintainer time, contract-test budgets, and CI compute
- 🤝 **[Become a design partner](SUPPORT.md#3-become-a-design-partner)** — direct line to maintainers, roadmap influence, free for the program duration
- 🏢 **Enterprise support** — SLAs, custom features, VPC deployment, SOC 2, BAA/DPA. Email **engineering@ai5labs.com**

See [SUPPORT.md](SUPPORT.md) for full details.

### Sponsors

*This space is for our sponsors — be the first.*

### Design partners

*This space is for our design partners — currently accepting our first cohort.*

## License

Apache-2.0. See [LICENSE](LICENSE). Copyright © 2026 ai5labs Research OPC Pvt Ltd.
