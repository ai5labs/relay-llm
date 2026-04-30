# Streaming

```python
async for ev in hub.stream("smart", messages=[...]):
    if ev.type == "text_delta":
        print(ev.text, end="", flush=True)
    elif ev.type == "thinking_delta":     # Anthropic / OpenAI o-series
        ...
    elif ev.type == "tool_call_delta":    # function-call argument fragment
        ...
    elif ev.type == "usage":              # one or more partial usage frames
        ...
    elif ev.type == "end":
        # Fully assembled response with cost + final usage
        print(ev.response.text)
        print(ev.response.cost_usd)
```

`hub.stream()` returns an async iterator directly — no `await` needed.

## Event types

| Event | When |
|---|---|
| `StreamStart` | First. Carries `id`, `model`, `provider`. |
| `TextDelta` | A piece of visible response text. Concatenating these gives `response.text`. |
| `ThinkingDelta` | A piece of internal reasoning text — preserved as a distinct event so callers can render it differently. |
| `ToolCallDelta` | A fragment of a tool call. Argument fragments are merged by `index`. |
| `UsageDelta` | A usage frame (some providers send these throughout the stream). |
| `StreamEnd` | Last. Carries the fully-assembled `ChatResponse` including cost. |
| `StreamErrorEvent` | A non-fatal stream error. |

## Tool-call delta merging

The single most common bug in multi-provider clients is dropping tool-call argument fragments. LiteLLM ships this bug ([#20711](https://github.com/BerriAI/litellm/issues/20711)) — they key fragments by `id`, but `id` only appears on the *first* fragment. Subsequent fragments carry only `index`.

Relay merges by `index`. There are property-based tests (`tests/unit/test_sse_invariants.py`) that generate arbitrary chunk splits and verify reassembly.
