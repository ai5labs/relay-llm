"""Caching: exact-match Hub-level cache + provider passthrough hints.

Two distinct things, deliberately distinguished:

1. **Hub-level exact-match cache** — Relay-side. Hashes (request body, model
   alias) and short-circuits on cache hit. Pluggable backends (memory, redis,
   sqlite) via the :class:`Cache` Protocol.
2. **Provider prompt-cache passthrough** — Anthropic ``cache_control``,
   OpenAI auto-cached prefix, Gemini ``cachedContent``. Users mark a piece of
   message content with :class:`CacheHint`; we translate per provider so the
   provider's cache (priced separately, often much cheaper) handles it.

Both can be used together: Hub-level cache for repeat exact prompts, provider
cache for stable prefixes within varied prompts.
"""

from __future__ import annotations

import hashlib
import time
from collections import OrderedDict
from typing import Any, Protocol, runtime_checkable

import orjson

from relay.types import CacheHintBlock, ChatRequest, ChatResponse, Message

# ---------------------------------------------------------------------------
# Cache hint markers (provider passthrough)
# ---------------------------------------------------------------------------


# Re-export under the friendlier name.
CacheHint = CacheHintBlock


def _to_anthropic_cache_control(hint: CacheHintBlock) -> dict[str, Any]:
    """Translate a CacheHint to Anthropic's ``cache_control`` shape."""
    if hint.ttl in ("1h", "1hr", "60m"):
        return {"type": "ephemeral", "ttl": "1h"}
    return {"type": "ephemeral"}


# ---------------------------------------------------------------------------
# Hub-level cache
# ---------------------------------------------------------------------------


@runtime_checkable
class Cache(Protocol):
    """Pluggable cache backend.

    Implementations must be safe for concurrent access. Async to allow Redis /
    SQLite / network backends without blocking the event loop.
    """

    async def get(self, key: str) -> ChatResponse | None: ...
    async def set(self, key: str, value: ChatResponse, ttl_s: float | None = None) -> None: ...
    async def aclose(self) -> None: ...


def cache_key(alias: str, request: ChatRequest) -> str:
    """Hash a chat request to a stable cache key.

    Includes: alias, messages, sampling params, tools, tool_choice, response_format.
    Excludes: ``stream``, timeout, metadata.
    """
    payload: dict[str, Any] = {
        "alias": alias,
        "messages": [_msg_to_hashable(m) for m in request.messages],
    }
    for k in (
        "max_tokens",
        "temperature",
        "top_p",
        "stop",
        "seed",
        "response_format",
        "tool_choice",
        "thinking",
    ):
        v = getattr(request, k, None)
        if v is not None:
            payload[k] = v
    if request.tools:
        payload["tools"] = [
            {"name": t.name, "parameters": t.parameters, "strict": t.strict} for t in request.tools
        ]
    raw = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS)
    return "relay:" + hashlib.blake2b(raw, digest_size=16).hexdigest()


def _msg_to_hashable(m: Message) -> dict[str, Any]:
    if isinstance(m.content, str):
        return {"role": m.role, "content": m.content}
    parts: list[Any] = []
    for block in m.content:
        if isinstance(block, CacheHintBlock):
            continue  # the hint is metadata; doesn't affect cache identity
        if hasattr(block, "model_dump"):
            parts.append(block.model_dump(exclude_none=True))
        else:
            parts.append(str(block))
    return {"role": m.role, "content": parts}


# ---------------------------------------------------------------------------
# In-memory implementation (default)
# ---------------------------------------------------------------------------


class MemoryCache:
    """Bounded LRU + TTL in-memory cache. Process-local.

    Good for development and single-process production. For multi-replica
    deployments use a Redis backend (see ``relay.cache.redis`` in v0.2).
    """

    def __init__(self, *, max_size: int = 10_000, default_ttl_s: float | None = 600.0) -> None:
        self._store: OrderedDict[str, tuple[ChatResponse, float]] = OrderedDict()
        self._max_size = max_size
        self._default_ttl = default_ttl_s
        self.hits = 0
        self.misses = 0

    async def get(self, key: str) -> ChatResponse | None:
        now = time.time()
        entry = self._store.get(key)
        if entry is None:
            self.misses += 1
            return None
        value, expires_at = entry
        if expires_at < now:
            del self._store[key]
            self.misses += 1
            return None
        self._store.move_to_end(key)
        self.hits += 1
        return value

    async def set(
        self,
        key: str,
        value: ChatResponse,
        ttl_s: float | None = None,
    ) -> None:
        ttl = ttl_s if ttl_s is not None else self._default_ttl
        expires = time.time() + ttl if ttl is not None else float("inf")
        self._store[key] = (value, expires)
        self._store.move_to_end(key)
        while len(self._store) > self._max_size:
            self._store.popitem(last=False)

    async def aclose(self) -> None:
        self._store.clear()

    @property
    def size(self) -> int:
        return len(self._store)


__all__ = [
    "Cache",
    "CacheHint",
    "MemoryCache",
    "cache_key",
]
