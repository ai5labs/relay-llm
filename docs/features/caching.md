# Caching

Two distinct caches, deliberately distinguished.

## Hub-level exact-match cache

Hashes the request and short-circuits on hit. Pluggable via the `Cache` Protocol — `MemoryCache` is the default.

```python
from relay import Hub, MemoryCache

cache = MemoryCache(max_size=10_000, default_ttl_s=600)
hub = Hub.from_yaml("models.yaml", cache=cache)

# First call hits the API; second call serves from cache.
r1 = await hub.chat("smart", messages=msgs)
r2 = await hub.chat("smart", messages=msgs)   # 0ms latency, $0 cost
```

The cache key includes alias, messages, sampling params, tools, tool_choice, response_format. It excludes streaming flag, timeout, metadata.

## Provider prompt-cache passthrough

Each provider has its own server-side cache (Anthropic prompt cache, OpenAI auto-cache, Gemini cachedContent). Mark cacheable spans of your prompt with `CacheHint`:

```python
from relay import CacheHint, Message
from relay.types import TextBlock

resp = await hub.chat("smart", messages=[
    Message(role="user", content=[
        TextBlock(text="Long stable prefix that rarely changes ..."),
        CacheHint(ttl="1h"),       # 1-hour Anthropic cache
        TextBlock(text="...the user's actual question varies."),
    ]),
])
```

| Provider | What `CacheHint` does |
|---|---|
| Anthropic | Emits `cache_control: {type: "ephemeral", ttl: "5m" or "1h"}` on the prior block. ~90% read discount, 25% write premium. |
| OpenAI | No-op (auto-cache happens implicitly on prefixes ≥ 1024 tokens) — placement of the hint helps users keep prompts prefix-stable. |
| Gemini / Vertex | (v0.3) gathers preceding content into a `cachedContent` resource. |

## Stacking

You can use both caches together: Hub-level cache for repeat exact prompts, provider cache for stable prefixes within varied prompts.
