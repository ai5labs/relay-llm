# Benchmarks

Real numbers, reproducible methodology, no marketing.

> **TL;DR** — Relay is faster than LiteLLM at every percentile and 8.5× faster
> on cold start. At p50 we're tied with raw `httpx` (no gateway in the loop).
> All results from running [`bench/gateway_comparison/`](bench/gateway_comparison)
> against a local mock OpenAI-compatible server on commodity hardware.

## Why we benchmark this way

Hitting real provider APIs would be dominated by network RTT (50–500ms),
which drowns out any per-gateway overhead signal. Instead we run a local
mock OpenAI-compatible server with a controllable, fixed backend latency
and route every gateway through the same backend. The difference between
`client_latency` and `backend_latency` is the gateway's overhead — which
is the only thing we control and the only number that should matter to a
prospective user.

## Methodology

```bash
# Terminal 1
python bench/gateway_comparison/mock_server.py --port 9999 --backend-ms 50

# Terminal 2
python -m bench.gateway_comparison.harness \
    --base-url http://127.0.0.1:9999/v1 \
    --backend-ms 50 \
    --requests 1000 \
    --concurrency 20
```

* **Mock backend latency**: 50ms (chat) / 200ms total streaming (20 chunks).
* **Workload**: 1000 chat requests, concurrency 20.
* **Streaming workload**: 100 streaming requests, sequential.
* **Warm-up**: 3 untimed requests per gateway before measurement.
* **Hardware**: macOS, Apple Silicon, Python 3.10.13.
* **Versions**: `relayllm==0.1.0`, `litellm==1.83.0`, `httpx==0.28.x`.

## Results

### 1. Chat overhead — 1000 requests at concurrency 20

| Gateway      | p50      | p95      | p99      | Overhead p50 | Overhead p95 | Overhead p99 | ops/s |
|--------------|----------|----------|----------|--------------|--------------|--------------|-------|
| `httpx` raw  | 52.16 ms | 65.98 ms | 98.25 ms | **2.2 ms**   | 16.0 ms      | 48.3 ms      | 363.5 |
| **Relay**    | 53.10 ms | 60.65 ms | 69.07 ms | **3.1 ms**   | **10.7 ms**  | **19.1 ms**  | **364.8** |
| LiteLLM      | 53.33 ms | 73.53 ms | 80.52 ms | 3.3 ms       | 23.5 ms      | 30.5 ms      | 353.3 |

**Relay's tail latency (p95, p99) is meaningfully better than both LiteLLM
*and* raw httpx.** That's not a typo — the connection-pooling tuning in
Relay's transport layer (HTTP/2, longer keep-alive) handles tail spikes
better than naïve httpx defaults.

### 2. Streaming TTFT — 100 streaming requests, time-to-first-token

| Gateway      | TTFT p50 | TTFT p95 | TTFT p99 | Mean    |
|--------------|----------|----------|----------|---------|
| `httpx` raw  | 11.0 ms  | 11.3 ms  | 27.7 ms  | 11.0 ms |
| **Relay**    | **13.4 ms** | **15.1 ms** | **19.9 ms** | **13.6 ms** |
| LiteLLM      | 16.2 ms  | 19.1 ms  | 24.1 ms  | 16.5 ms |

Relay is **~20% faster than LiteLLM at the median** and has **lower p99 than
both LiteLLM and raw httpx** (again, due to connection-pool tuning).

### 3. Cold start — fresh subprocess, 5 runs

| What                              | Median    |
|-----------------------------------|-----------|
| `import litellm`                  | **1304 ms** |
| `import relay` (full public API)  | **152 ms**  |
| `import relay` + `Hub.from_yaml`  | **155 ms**  |

**Relay imports 8.5× faster than LiteLLM** because of strict lazy-import
discipline (provider SDKs, OTel, MCP, AWS — none load unless you actually
call them). Constructing a Hub from YAML adds essentially zero cost on top
of the import.

This matters for serverless deployments: AWS Lambda, Cloud Run, Cloudflare
Workers, etc. all charge for cold-start time. A 1.3-second import is a
meaningful UX hit; 150ms is essentially nothing.

## Reproducing

```bash
git clone https://github.com/ai5labs/relay-llm
cd relay-llm
uv sync --all-groups

# Chat overhead
uv run python bench/gateway_comparison/mock_server.py --port 9999 --backend-ms 50 &
uv run python -m bench.gateway_comparison.harness \
    --base-url http://127.0.0.1:9999/v1 --backend-ms 50 \
    --requests 1000 --concurrency 20

# Streaming TTFT
uv run python -m bench.gateway_comparison.streaming \
    --base-url http://127.0.0.1:9999/v1 --requests 100

# Cold start
uv run python -m bench.gateway_comparison.cold_start
```

## What's *not* in these numbers

Honest disclosures:

* **TensorZero** (Rust gateway) is not in the comparison. They publish their
  own benchmarks claiming sub-millisecond p99 at 10k QPS, which is a
  different category from any Python gateway. We're not pretending to compete
  on raw speed with a Rust binary; we compete on enterprise features,
  MCP-native integration, and *Python ergonomics with credible performance*.
* **Portkey, Helicone, OpenRouter** are not in the comparison either. They
  run as remote services (or their OSS gateways add an extra HTTP hop), so
  comparing their numbers against in-process LiteLLM/Relay would be
  apples-to-oranges. A separate proxy-mode benchmark is on the v0.3 roadmap.
* **Real provider RTT** dominates the user-perceived latency in production.
  If your provider takes 800ms, our 3ms vs LiteLLM's 3.3ms doesn't matter to
  your user. These numbers matter at scale (high QPS, tight tail-latency
  SLOs), and for cold-start cost.
* **A single machine, single Python version**. Numbers will drift on
  Windows, on Python 3.13, under heavy GC pressure, etc. The repository's
  CI runs the bench harness on every PR so regressions are caught early.

## Roadmap

* **v0.2** (current): Python implementation, lazy imports, HTTP/2 + tuned pool.
* **v0.3**: profile and close the small p50 gap to raw httpx (likely
  Pydantic on the response path; investigate `model_construct` shortcuts).
* **v1.0**: optional Rust core for the SSE parsing + JSON serialization hot
  path. Goal: ≤ 1ms p99 overhead at 5k QPS. Python public API stays the same.

## License + reproduction commit

All bench code in [`bench/gateway_comparison/`](bench/gateway_comparison/),
Apache-2.0. Last published numbers: `git log -1 BENCHMARKS.md`.
