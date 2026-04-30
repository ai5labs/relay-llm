# Reasoning budgets

OpenAI, Anthropic, and Gemini all support reasoning/thinking budgets — but with three different APIs. Relay unifies them.

```python
resp = await hub.chat(
    "thinker",
    messages=[...],
    reasoning="high",       # or "minimal" / "low" / "medium"
)

# Or specify exact tokens:
resp = await hub.chat("thinker", messages=[...], reasoning=8192)
```

## Mapping

| Relay value | OpenAI `reasoning.effort` | Anthropic `thinking.budget_tokens` | Gemini `thinkingConfig.thinking_budget` |
|---|---|---|---|
| `"minimal"` | `minimal` | 1024 | 1024 |
| `"low"` | `low` | 4096 | 4096 |
| `"medium"` | `medium` | 16384 | 16384 |
| `"high"` | `high` | 24576 | 24576 |
| Integer `N` | bucketed (<2048→minimal, <8192→low, <20000→medium, else high) | exactly `N` | exactly `N` |

OpenAI is string-only at the API level, so integer values get bucketed for OpenAI but pass through to Anthropic/Gemini exactly.

## Provider-specific overrides

If you need to set a provider-native field directly:

```python
# Anthropic-native
resp = await hub.chat("anthropic-m", messages=[...], thinking={"type": "enabled", "budget_tokens": 8192})
```

`thinking` (Anthropic) and per-provider native fields take precedence over `reasoning` if both are passed.
