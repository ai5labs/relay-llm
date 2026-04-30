# Providers

Relay supports 18 providers. Native adapters preserve provider-specific blocks (`thinking`, `grounding`, citations) instead of flattening them.

## OpenAI-compatible (one adapter, 12 providers)

| Provider | Default base URL | Notes |
|---|---|---|
| OpenAI | `api.openai.com/v1` | Set `api_style: responses` to use `/v1/responses` |
| Groq | `api.groq.com/openai/v1` | Lowest TTFB |
| Together | `api.together.xyz/v1` | Open-weight zoo |
| DeepSeek | `api.deepseek.com/v1` | DeepSeek-V3 + Reasoner |
| xAI | `api.x.ai/v1` | Grok models |
| Mistral | `api.mistral.ai/v1` | |
| Fireworks | `api.fireworks.ai/inference/v1` | |
| Perplexity | `api.perplexity.ai` | Web-search grounded |
| OpenRouter | `openrouter.ai/api/v1` | Aggregator with `auto` routing |
| Ollama | `localhost:11434/v1` | Self-hosted, no auth |
| vLLM | `localhost:8000/v1` | Self-hosted, no auth |
| LM Studio | `localhost:1234/v1` | Self-hosted, no auth |

## Native adapters (proper, lossless)

### Anthropic

```yaml
models:
  smart:
    target: anthropic/claude-sonnet-4-5
    credential: $env.ANTHROPIC_API_KEY
```

Surfaces `thinking` blocks as first-class. `CacheHint` markers compile to `cache_control`.

### Azure OpenAI

```yaml
models:
  prod:
    target: azure/gpt-4o
    base_url: https://acme.openai.azure.com
    deployment: gpt-4o-prod-deployment
    api_version: "2024-08-01-preview"
    credential: $env.AZURE_KEY
```

Routes by deployment, not model id. Uses `api-key` header (not bearer).

### AWS Bedrock

```yaml
models:
  bedrock-sonnet:
    target: bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0
    region: us-east-1
    credential: { type: aws_profile, profile: prod }
```

Uses the Converse API; SigV4 signing handled by `boto3`. Streaming supported.

### Google Gemini direct

```yaml
models:
  gemini:
    target: google/gemini-2.5-flash
    credential: $env.GEMINI_API_KEY
```

API key via `?key=` query param. `thinking_config`, `responseSchema`, `functionDeclarations` all wired.

### Vertex AI

```yaml
models:
  vertex-pro:
    target: vertex/gemini-2.5-pro
    project: acme-prod
    location: us-central1
    credential: { type: gcp_adc }
```

Same wire format as Gemini direct; auth via GCP Application Default Credentials.

### Cohere

```yaml
models:
  cmdr:
    target: cohere/command-r-plus-08-2024
    credential: $env.COHERE_API_KEY
```

Uses Cohere v2 chat API.

## Adding a new provider

1. New file in `src/relay/providers/`.
2. New entry in `PROVIDER_REGISTRY` in `src/relay/providers/__init__.py`.
3. Catalog rows in `src/relay/catalog/data/models.json`.
4. Tests under `tests/unit/`.
5. (Optional) Curated capability flags in `scripts/curated.json`.
