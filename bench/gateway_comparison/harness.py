"""Gateway-comparison harness.

Runs identical workloads through raw httpx, Relay, and LiteLLM against the
same mock OpenAI-compat backend. Reports p50 / p95 / p99 latency *overhead*
(client time minus backend time) and operations per second under concurrency.

Run::

    # Terminal 1: start the mock server
    python bench/gateway_comparison/mock_server.py --port 9999 --backend-ms 50

    # Terminal 2: run the harness
    python bench/gateway_comparison/harness.py \\
        --base-url http://127.0.0.1:9999 --backend-ms 50 \\
        --requests 200 --concurrency 10

Or run both with one command::

    python bench/gateway_comparison/harness.py --self-host
"""

from __future__ import annotations

import argparse
import asyncio
import statistics
import sys
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class Result:
    name: str
    n: int
    backend_ms: float
    p50_ms: float
    p95_ms: float
    p99_ms: float
    mean_ms: float
    overhead_p50_ms: float
    overhead_p95_ms: float
    overhead_p99_ms: float
    ops_per_sec: float
    errors: int


async def time_one(client: Any, prompt: str) -> tuple[float, bool]:
    start = time.perf_counter()
    try:
        await client.chat(prompt)
        return (time.perf_counter() - start), False
    except Exception:
        return (time.perf_counter() - start), True


async def run_one_client(
    client: Any,
    *,
    requests: int,
    concurrency: int,
    backend_ms: float,
) -> Result:
    sem = asyncio.Semaphore(concurrency)
    latencies: list[float] = []
    errors = 0

    async def worker(i: int) -> None:
        nonlocal errors
        async with sem:
            elapsed, err = await time_one(client, f"prompt {i}")
            if err:
                errors += 1
            else:
                latencies.append(elapsed * 1000)

    overall_start = time.perf_counter()
    await asyncio.gather(*(worker(i) for i in range(requests)))
    overall_elapsed = time.perf_counter() - overall_start

    if not latencies:
        return Result(
            name=client.name,
            n=requests,
            backend_ms=backend_ms,
            p50_ms=0.0,
            p95_ms=0.0,
            p99_ms=0.0,
            mean_ms=0.0,
            overhead_p50_ms=0.0,
            overhead_p95_ms=0.0,
            overhead_p99_ms=0.0,
            ops_per_sec=0.0,
            errors=errors,
        )

    latencies.sort()
    p50 = statistics.median(latencies)
    p95 = latencies[int(0.95 * len(latencies))]
    p99 = latencies[int(0.99 * len(latencies))] if len(latencies) >= 100 else latencies[-1]
    mean = statistics.mean(latencies)
    return Result(
        name=client.name,
        n=requests,
        backend_ms=backend_ms,
        p50_ms=p50,
        p95_ms=p95,
        p99_ms=p99,
        mean_ms=mean,
        overhead_p50_ms=p50 - backend_ms,
        overhead_p95_ms=p95 - backend_ms,
        overhead_p99_ms=p99 - backend_ms,
        ops_per_sec=requests / overall_elapsed if overall_elapsed > 0 else 0.0,
        errors=errors,
    )


def print_table(results: list[Result]) -> None:
    print()
    print(
        f"{'gateway':<12} {'n':>5} {'p50':>9} {'p95':>9} {'p99':>9} {'overhead p50':>14} {'overhead p95':>14} {'overhead p99':>14} {'ops/s':>9} {'err':>5}"
    )
    print("-" * 108)
    for r in results:
        print(
            f"{r.name:<12} {r.n:>5} "
            f"{r.p50_ms:>7.2f}ms {r.p95_ms:>7.2f}ms {r.p99_ms:>7.2f}ms "
            f"{r.overhead_p50_ms:>10.2f}ms   {r.overhead_p95_ms:>10.2f}ms   {r.overhead_p99_ms:>10.2f}ms   "
            f"{r.ops_per_sec:>7.1f}   {r.errors:>3}"
        )
    print()
    print(
        f"(backend latency: {results[0].backend_ms:.0f}ms; overhead = client_latency - backend_ms)"
    )


async def warm_up(client: Any) -> None:
    """3 untimed requests to warm up the connection pool."""
    for _ in range(3):
        try:
            await client.chat("warmup")
        except Exception:
            pass


async def run_harness(
    *,
    base_url: str,
    backend_ms: float,
    requests: int,
    concurrency: int,
    skip: set[str],
) -> list[Result]:
    from bench.gateway_comparison.clients.raw_httpx import RawHttpxClient
    from bench.gateway_comparison.clients.relay_client import RelayClient

    results: list[Result] = []
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
            await warm_up(c)
            print(f"benchmarking {c.name} ...", file=sys.stderr)
            r = await run_one_client(
                c,
                requests=requests,
                concurrency=concurrency,
                backend_ms=backend_ms,
            )
            results.append(r)
    finally:
        for c in clients:
            try:
                await c.aclose()
            except Exception:
                pass

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:9999")
    parser.add_argument("--backend-ms", type=float, default=50.0)
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--concurrency", type=int, default=10)
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        help="gateways to skip: raw, relay, litellm",
    )
    parser.add_argument(
        "--self-host",
        action="store_true",
        help="spawn the mock server locally (uses --backend-ms)",
    )
    args = parser.parse_args()

    if args.self_host:
        import multiprocessing

        from bench.gateway_comparison.mock_server import main as server_main

        # Patch sys.argv so the server's argparse picks up our values.
        port = 9999
        old_argv = sys.argv
        sys.argv = [
            "mock_server",
            "--port",
            str(port),
            "--backend-ms",
            str(args.backend_ms),
        ]
        proc = multiprocessing.Process(target=server_main, daemon=True)
        proc.start()
        sys.argv = old_argv
        time.sleep(2.0)  # let it bind
        args.base_url = f"http://127.0.0.1:{port}"

    results = asyncio.run(
        run_harness(
            base_url=args.base_url,
            backend_ms=args.backend_ms,
            requests=args.requests,
            concurrency=args.concurrency,
            skip=set(args.skip),
        )
    )
    print_table(results)


if __name__ == "__main__":
    main()
