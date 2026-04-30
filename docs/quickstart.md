# Quickstart

## Install

```bash
pip install relayllm
```

Optional extras:

```bash
pip install 'relayllm[otel]'    # OpenTelemetry instrumentation
pip install 'relayllm[aws]'     # AWS Bedrock + Secrets Manager
pip install 'relayllm[gcp]'     # GCP Vertex AI + Secret Manager
pip install 'relayllm[mcp]'     # Model Context Protocol
pip install 'relayllm[all]'     # everything
```

## Define your models

Create `models.yaml`:

```yaml
# yaml-language-server: $schema=https://relay.ai5labs.com/schema/v1.json
version: 1

models:
  fast:
    target: groq/llama-3.3-70b-versatile
    credential: $env.GROQ_API_KEY

  smart:
    target: anthropic/claude-sonnet-4-5
    credential: $env.ANTHROPIC_API_KEY

  vision:
    target: openai/gpt-4o-mini
    credential: $env.OPENAI_API_KEY

groups:
  default:
    strategy: fallback
    members: [smart, fast]
```

Point your editor at the `$schema` URL on line 1 — the [Red Hat YAML extension](https://marketplace.visualstudio.com/items?itemName=redhat.vscode-yaml) for VS Code will give you autocomplete + inline validation.

## First call

```python
import asyncio
from relay import Hub

async def main():
    async with Hub.from_yaml("models.yaml") as hub:
        resp = await hub.chat(
            "fast",
            messages=[{"role": "user", "content": "Say 'pong' in one word."}],
        )
        print(resp.text)
        print(f"Cost: ${resp.cost_usd:.6f} (source: {resp.cost.source})")

asyncio.run(main())
```

## Streaming

```python
async for ev in hub.stream("smart", messages=[...]):
    if ev.type == "text_delta":
        print(ev.text, end="", flush=True)
    elif ev.type == "thinking_delta":     # Anthropic extended thinking
        ...
    elif ev.type == "end":
        print(f"\n[{ev.response.latency_ms:.0f}ms, ${ev.response.cost_usd:.4f}]")
```

## Group fallback

```python
# Tries 'smart' first; on context-window or content-policy error, falls back to 'fast'.
resp = await hub.chat("default", messages=[...])
```

## Bound handles for hot loops

```python
model = hub.get("fast")
for prompt in prompts:
    resp = await model.chat(messages=[{"role": "user", "content": prompt}])
```

## CLI

```bash
relay schema --out relay.schema.json     # JSON Schema for editors
relay validate models.yaml               # validate config
relay models list                        # list configured aliases
relay models inspect smart               # one alias's full config + catalog row
relay catalog list --provider anthropic  # browse the built-in catalog
relay providers                          # list all supported providers
```
