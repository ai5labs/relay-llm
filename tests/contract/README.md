# Contract tests

These tests hit **real provider APIs**. They are gated by environment variables
so the unit suite stays hermetic.

## Running locally

```bash
export OPENAI_API_KEY=sk-...
export ANTHROPIC_API_KEY=sk-ant-...
export GROQ_API_KEY=gsk-...
uv run pytest tests/contract/ -m live
```

Each test costs only a few cents (cheapest models, max 64 output tokens).

## In CI

Run nightly via the `contract.yml` GitHub workflow with secrets, not on every
PR. Failures here are *signal*: usually a provider changed their wire format.
