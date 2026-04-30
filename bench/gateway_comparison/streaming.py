"""Streaming TTFT (time-to-first-token) benchmark.

The user-perceived latency for streaming workloads. A gateway that takes 50ms
to flush the first byte to the caller is worse than a gateway with higher
total throughput but lower TTFT, because chat UIs feel snappy or sluggish on
the *first* token, not the last.

Methodology
-----------
Each gateway issues a streaming request to the same mock backend. The mock is
configured to start emitting SSE chunks immediately (no upfront sleep), so
TTFT measures pure gateway overhead between caller and the first text delta.
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from typing import Any


async def bench_one(client: Any, requests: int) -> dict[str, Any]:
    ttfts: list[float] = []
    chunk_counts: list[int] = []
    errors = 0
    overall_start = time.perf_counter()
    for i in range(requests):
        try:
            ttft, chunks = await client.stream_first_token(f"prompt {i}")
            ttfts.append(ttft * 1000)  # ms
            chunk_counts.append(chunks)
        except Exception:
            errors += 1
    overall_elapsed = time.perf_counter() - overall_start

    if not ttfts:
        return {"name": client.name, "errors": errors}
    ttfts.sort()
    return {
        "name": client.name,
        "n": len(ttfts),
        "ttft_p50_ms": statistics.median(ttfts),
        "ttft_p95_ms": ttfts[int(0.95 * len(ttfts))],
        "ttft_p99_ms": ttfts[int(0.99 * len(ttfts))] if len(ttfts) >= 100 else ttfts[-1],
        "ttft_mean_ms": statistics.mean(ttfts),
        "mean_chunks": statistics.mean(chunk_counts),
        "ops_per_sec": len(ttfts) / overall_elapsed if overall_elapsed > 0 else 0.0,
        "errors": errors,
    }


def print_table(rows: list[dict[str, Any]]) -> None:
    print()
    print(
        f"{'gateway':<12} {'n':>5} {'TTFT p50':>10} {'TTFT p95':>10} {'TTFT p99':>10} {'mean':>10} {'chunks':>8} {'ops/s':>8} {'err':>5}"
    )
    print("-" * 92)
    for r in rows:
        if "ttft_p50_ms" not in r:
            print(f"{r['name']:<12} (errors: {r.get('errors', 0)})")
            continue
        print(
            f"{r['name']:<12} {r['n']:>5} "
            f"{r['ttft_p50_ms']:>8.2f}ms {r['ttft_p95_ms']:>8.2f}ms {r['ttft_p99_ms']:>8.2f}ms "
            f"{r['ttft_mean_ms']:>8.2f}ms {r['mean_chunks']:>8.1f} "
            f"{r['ops_per_sec']:>6.1f} {r['errors']:>5}"
        )
    print()


async def main_async(
    *,
    base_url: str,
    requests: int,
    skip: set[str],
) -> None:
    from bench.gateway_comparison.clients.raw_httpx import RawHttpxClient
    from bench.gateway_comparison.clients.relay_client import RelayClient

    rows: list[dict[str, Any]] = []
    clients: list[Any] = []
    if "raw" not in skip:
        clients.append(RawHttpxClient(base_url))
    if "relay" not in skip:
        clients.append(RelayClient(base_url))
    if "litellm" not in skip:
        try:
            from bench.gateway_comparison.clients.litellm_client import LiteLLMClient

            clients.append(LiteLLMClient(base_url))
        except Exception as e:
            print(f"  (skipping litellm: {e})", file=sys.stderr)

    try:
        for c in clients:
            print(f"warming up {c.name} ...", file=sys.stderr)
            for _ in range(2):
                try:
                    await c.stream_first_token("warmup")
                except Exception:
                    pass
            print(f"benchmarking streaming {c.name} ...", file=sys.stderr)
            row = await bench_one(c, requests)
            rows.append(row)
    finally:
        for c in clients:
            try:
                await c.aclose()
            except Exception:
                pass

    print_table(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:9999/v1")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--skip", nargs="*", default=[])
    args = parser.parse_args()
    asyncio.run(main_async(base_url=args.base_url, requests=args.requests, skip=set(args.skip)))


if __name__ == "__main__":
    main()
