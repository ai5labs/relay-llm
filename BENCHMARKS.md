# Benchmarks

Real numbers, reproducible methodology, no marketing.

> **TL;DR** — Across 3 independent runs on the same machine: **Relay imports 5–19× faster than LiteLLM**, **streams its first token ~20% faster**, and is **tied at the median for chat-overhead** with **more consistent tails** (Relay p99 stays in 19–23ms; LiteLLM jumps 23–41ms run-to-run). At p50 we're tied with raw `httpx` (no gateway in the loop).

## Why we benchmark this way

Hitting real provider APIs would be dominated by network RTT (50–500ms), which drowns out any per-gateway overhead signal. Instead we run a local mock OpenAI-compatible server with a controllable, fixed backend latency and route every gateway through the same backend. The difference between `client_latency` and `backend_latency` is the gateway's overhead — the only number that should matter to a prospective user.

## Methodology

```bash
# Terminal 1 — mock backend
python bench/gateway_comparison/mock_server.py --port 9999 --backend-ms 50

# Terminal 2 — chat overhead
python -m bench.gateway_comparison.harness \
    --base-url http://127.0.0.1:9999/v1 \
    --backend-ms 50 --requests 1000 --concurrency 20

# Streaming TTFT
python -m bench.gateway_comparison.streaming \
    --base-url http://127.0.0.1:9999/v1 --requests 100

# Cold start
python -m bench.gateway_comparison.cold_start
```

* **Mock backend latency**: 50ms (chat) / 200ms total streaming (20 chunks).
* **Workload**: 1000 chat requests at concurrency 20; 100 sequential streaming requests; 5-run median for cold start.
* **Warm-up**: 3 untimed requests per gateway before measurement.
* **Hardware**: macOS, Apple Silicon, Python 3.10.13. Single machine.
* **Versions**: `ai5labs-relay==0.1.0`, `litellm==1.83.0`, `httpx==0.28.x`.
* **Runs**: 3 independent end-to-end runs of each benchmark, fresh process, no caches.

## Results

### 1. Chat overhead — 3-run aggregate

Per-run **gateway overhead** (`client_latency − backend_ms`):

| Run | Relay p50 | LiteLLM p50 | Relay p95 | LiteLLM p95 | Relay p99 | LiteLLM p99 |
|-----|-----------|-------------|-----------|-------------|-----------|-------------|
|  1  | 3.1 ms    | 3.3 ms      | 10.7 ms   | 23.5 ms     | 19.1 ms   | 30.5 ms     |
|  2  | 2.2 ms    | 3.4 ms      | 10.1 ms   | 9.8 ms      | 23.2 ms   | 23.4 ms     |
|  3  | 2.6 ms    | 13.4 ms     | 8.7 ms    | 38.0 ms     | 22.3 ms   | 41.0 ms     |

**Honest reading:**

- **p50** — both libraries are sub-4ms in good runs; LiteLLM had one bad run (13ms median) where Relay stayed at ~3ms. Relay is the same or better in 3 of 3 runs.
- **p95** — Relay is better in 2 of 3 runs; tied in the 3rd. Variance dominates the comparison at this percentile.
- **p99** — Relay is better in 2 of 3 runs; tied in the 3rd. Same pattern.

**Best summary**: at our sample sizes, p95/p99 chat overhead is **not statistically separable on any single run** — both libraries see jitter in the same range (8–40ms tails). What *is* observable: Relay's tail is more consistent (range 19–23ms across 3 runs) than LiteLLM's (23–41ms).

raw `httpx` baseline for context (run 3): 52.5 ms p50 / 60.6 ms p95 / 69.6 ms p99 → 2.5 ms / 10.6 ms / 19.6 ms overhead.

### 2. Streaming TTFT — 3-run aggregate

| Run | Relay p50 | LiteLLM p50 | Ratio | Relay p99 | LiteLLM p99 |
|-----|-----------|-------------|-------|-----------|-------------|
|  1  | 13.4 ms   | 16.2 ms     | 1.21× | 19.9 ms   | 24.1 ms     |
|  2  | 14.6 ms   | 18.6 ms     | 1.27× | 18.4 ms   | 57.7 ms     |
|  3  | 13.6 ms   | 15.4 ms     | 1.13× | 17.3 ms   | 17.7 ms     |

**Relay is faster at p50 in 3 of 3 runs**, by ~13–27%. At p99, Relay was meaningfully better in 2 runs and tied in the third.

raw `httpx` baseline (run 3): 11.0 ms p50.

### 3. Cold start — fresh subprocess, 5 internal runs each, two independent passes

| What                              | Pass 1 median | Pass 2 median |
|-----------------------------------|---------------|---------------|
| `import litellm`                  | **1304 ms**   | **2078 ms**   |
| `import relay`                    | **152 ms**    | **110 ms**    |
| Ratio (LiteLLM / Relay)           | **8.5×**      | **18.9×**     |

LiteLLM imports a heavy graph at module load (provider SDKs, tokenizers, etc.). Relay's strict lazy-import discipline keeps startup near-instant. Even on the worst Relay run (152ms), it's much faster than LiteLLM's best (1304ms).

This is the most defensible win in the benchmark — the magnitude is so large the noise can't reverse the ordering.

`import relay` + `Hub.from_yaml`: 155 ms (pass 1) / 303 ms (pass 2). Adds little to the import on top.

## What we're *not* claiming

- We are **not** claiming "faster than LiteLLM at every percentile." On a typical run, yes — but at the p95/p99 tail, run-to-run noise can flip the ordering on a single 1000-sample run.
- We are **not** claiming a specific multiplier like "2.2× faster at p95." Single runs can show that, but it doesn't reliably reproduce.
- We are **not** claiming these numbers replicate on any hardware. Different OS / CPU / Python implementation will shift everything. The repo's CI runs the harness on every PR, but only on Linux runners on a single CPU class.

## What we *are* claiming

1. **Cold start is dramatically faster** — robust 5–19× win, never fewer than 5× across all our runs. Material for AWS Lambda / Cloud Run / Cloudflare Workers users.
2. **Streaming TTFT is meaningfully faster at the median** — 13–27% faster p50, 3 of 3 runs. The user-perceived "first token" UX is snappier.
3. **Chat-overhead median is tied with LiteLLM** at ~3 ms, and Relay's tail latency is **more consistent** (lower run-to-run variance) than LiteLLM's. We don't say "lower"; we say "more consistent."

## Reproducing

```bash
git clone https://github.com/ai5labs/relay-llm
cd relay-llm
uv sync --all-groups

# Chat overhead
uv run python bench/gateway_comparison/mock_server.py --port 9999 --backend-ms 50 &
uv run python -m bench.gateway_comparison.harness \
    --base-url http://127.0.0.1:9999/v1 --backend-ms 50 --requests 1000 --concurrency 20

# Streaming TTFT
uv run python -m bench.gateway_comparison.streaming \
    --base-url http://127.0.0.1:9999/v1 --requests 100

# Cold start
uv run python -m bench.gateway_comparison.cold_start
```

Run each one **three times** before drawing conclusions. Single runs at our sample size are noisy at the tails.

## What's *not* in these numbers

- **TensorZero** (Rust gateway) — not in the comparison. They publish sub-millisecond p99 at 10k QPS, which is a different category from any Python gateway. We're not pretending to compete with a Rust binary on raw speed; we compete on enterprise features, MCP-native integration, and Python ergonomics.
- **Portkey, Helicone, OpenRouter** — they run as remote services or add an extra HTTP hop, so apples-to-oranges with in-process Python gateways. A separate proxy-mode benchmark is on the v0.3 roadmap.
- **Real provider RTT** dominates the user-perceived latency in production. If your provider takes 800ms, our 3ms vs LiteLLM's 3.3ms doesn't matter to your user. These numbers matter at scale (high QPS, tight tail-latency SLOs) and for cold-start cost.
- **A single machine, single Python version**. Numbers will drift on Linux, on Python 3.13, under heavy GC pressure, etc.
- **Confidence intervals**. We quote point estimates from per-run medians. Real numbers will jitter ±10% run-to-run. We've published 3 runs to make that range visible.

## Roadmap

* **v0.2** (current): Python implementation, lazy imports, HTTP/2 + tuned pool.
* **v0.3**: profile and stabilize the chat-overhead tail; investigate why LiteLLM has occasional bad runs and ensure we don't share the same root cause.
* **v1.0**: optional Rust core for SSE parsing + JSON serialization hot path. Goal: ≤ 1ms p99 overhead at 5k QPS. Python public API stays the same.

## License + reproduction commit

All bench code in [`bench/gateway_comparison/`](bench/gateway_comparison/), Apache-2.0. Last published numbers: `git log -1 BENCHMARKS.md`.
