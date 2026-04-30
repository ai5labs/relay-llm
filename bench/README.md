# Benchmarks

Microbenchmarks for hot paths. Use to catch regressions and to make data-driven optimization decisions.

```bash
# Quick run, no comparison
uv run pytest bench/ --benchmark-only -q

# Save baseline for later comparison
uv run pytest bench/ --benchmark-only --benchmark-save=baseline

# After your change, compare
uv run pytest bench/ --benchmark-only --benchmark-compare=baseline
```

## What's measured

| Bench | What | Why |
|---|---|---|
| `bench_sse_parsing` | parse 1KB SSE chunk → emit `TextDelta` | streaming hot path |
| `bench_tool_call_merge` | merge 50 tool-call deltas → final `ChatResponse` | LiteLLM-bug regression target |
| `bench_cache_key` | hash a 10-message request | cache-lookup overhead per request |
| `bench_pydantic_serialization` | `ChatResponse.model_dump()` | response materialization cost |
| `bench_tool_compile` | compile a 5-tool request for OpenAI / Anthropic / Gemini | per-request adapter overhead |

These are microbenchmarks; **don't** infer end-to-end latency from them. The real numbers are dominated by network RTT to the provider.
