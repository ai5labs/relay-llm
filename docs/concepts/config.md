# YAML config

Your model catalog lives in a YAML file. Everything Relay does — routing, pricing, capabilities — derives from this one source of truth.

## Anatomy

```yaml
# yaml-language-server: $schema=https://relay.ai5labs.com/schema/v1.json
version: 1

defaults:
  timeout: 60
  max_retries: 2

catalog:
  fetch_live_pricing: true       # AWS / Azure / OpenRouter
  refresh_interval_hours: 6
  offline: false                  # if true, skip all live calls

observability:
  otel_enabled: false             # opt in
  capture_messages: metadata_only # never | metadata_only | full

pricing_profiles:
  acme-aws-edp:
    description: "15% AWS EDP discount"
    input_multiplier: 0.85
    output_multiplier: 0.85

models:
  fast:
    target: groq/llama-3.3-70b-versatile
    credential: $env.GROQ_API_KEY
    params: { temperature: 0.2 }
    tags: [fast, cheap]

  smart:
    target: anthropic/claude-sonnet-4-5
    credential: $env.ANTHROPIC_API_KEY
    params: { max_tokens: 4096 }

  bedrock-sonnet:
    target: bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0
    region: us-east-1
    credential: { type: aws_profile, profile: prod }
    pricing_profile: acme-aws-edp

groups:
  default:
    strategy: fallback           # fallback | loadbalance | weighted | conditional
    members: [smart, fast]
```

## Targets

A model target is `provider/model_id`. The first `/` splits — anything after is provider-specific. Examples:

```
openai/gpt-4o-2024-11-20
anthropic/claude-sonnet-4-5
bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0
azure/gpt-4o                              # + base_url + deployment
vertex/gemini-2.5-flash                   # + project + location
groq/llama-3.3-70b-versatile
```

## Credentials

**Never inline keys in YAML.** Credentials are reified objects:

```yaml
# Env var (most common)
credential: $env.OPENAI_API_KEY

# AWS
credential: { type: aws_profile, profile: prod, region: us-east-1 }
credential: { type: aws_secrets, arn: "arn:aws:secretsmanager:..." }

# GCP
credential: { type: gcp_adc }                      # application default
credential: { type: gcp_secret_manager, name: "projects/.../secrets/..." }

# Vault (planned)
credential: { type: vault, path: secret/data/openai, field: api_key }
```

Missing or empty env vars fail loud at load time — they are never silently coerced to empty strings.

## Layered loading

```python
hub = Hub.from_yaml("base.yaml", "prod.yaml")
```

Later files deep-merge over earlier ones (lists are replaced, not concatenated). Useful for environment-specific tweaks.

## JSON Schema

Run `relay schema --out relay.schema.json` to emit the schema. Reference it from your YAML's first line:

```yaml
# yaml-language-server: $schema=./relay.schema.json
```

Most modern YAML editors (VS Code with the Red Hat YAML extension, IntelliJ, Helix) will then give you autocomplete and inline validation.
