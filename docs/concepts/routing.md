# Routing &amp; failover

Groups bundle multiple model aliases under one logical name with a routing strategy.

## Strategies

```yaml
groups:
  smart-with-fallback:
    strategy: fallback
    members: [smart, smart-cheap, fast]

  load-balanced:
    strategy: loadbalance
    members: [openai-key-1, openai-key-2, openai-key-3]

  weighted:
    strategy: weighted
    members:
      - { name: smart, weight: 9 }
      - { name: backup, weight: 1 }
```

| Strategy | Behavior |
|---|---|
| `fallback` | Try members in declared order; on retryable failure or fall-back error move to the next. |
| `loadbalance` | Random shuffle of members per request. |
| `weighted` | Weighted random sample without replacement. |
| `conditional` | (v0.3) pick the first member whose `when` predicate matches. |

## Classified retry semantics

Errors are sorted into three buckets:

* **transient** (`RateLimitError`, `TimeoutError`, `ProviderError`, `CircuitOpenError`) — retry on the same member up to `max_retries`, then fall back.
* **fall_back** (`ContextWindowError`, `ContentPolicyError`) — never retry; immediately try the next member.
* **fatal** (`AuthenticationError`, `ConfigError`) — don't retry, don't fall back. The user has a config problem; surface it loudly.

## Circuit breakers

Each `(provider, model_id)` has its own three-state breaker:

* **Closed** — requests pass through. Failures in a sliding `window_s` count toward `failure_threshold`.
* **Open** — requests fail-fast (`CircuitOpenError`) for `cooldown_s`, so the router immediately tries other members.
* **Half-open** — one probe attempt is admitted; success closes, failure re-opens.

Defaults: 5 failures in 30s open the breaker for 60s. Tuneable via `defaults` in your YAML (in v0.3) or by constructing your own `CircuitBreaker` and passing it in.

`ContextWindowError` and `ContentPolicyError` do **not** count toward breaker failures — they're request-shape problems, not provider health.
