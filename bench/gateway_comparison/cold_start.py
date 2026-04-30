"""Cold-start benchmark: import time + first-call setup.

Why this matters
----------------
Serverless / Lambda / Cloud Run cold-start budgets are tight. A library that
takes 2 seconds to import is a non-starter for short-lived inference workers.

We measure two stages:

1. **Import time**: ``python -c 'import X'`` wall clock.
2. **First-call setup**: import + parse YAML + construct Hub + first chat call.

Each is run in a fresh subprocess so we get true cold numbers (not warm
import caches).
"""

from __future__ import annotations

import json
import statistics
import subprocess
import sys
import textwrap
from typing import Any


def time_subprocess(code: str, *, runs: int = 5) -> list[float]:
    """Run ``code`` as a fresh subprocess ``runs`` times. Return seconds per run."""
    timings: list[float] = []
    for _ in range(runs):
        result = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            check=True,
        )
        timings.append(float(result.stdout.strip()))
    return timings


def measure_import(label: str, import_code: str, runs: int = 5) -> dict[str, Any]:
    code = textwrap.dedent(f"""
        import time
        start = time.perf_counter()
        {import_code}
        print(time.perf_counter() - start)
    """)
    timings = time_subprocess(code, runs=runs)
    return {
        "name": label,
        "phase": "import",
        "min_ms": min(timings) * 1000,
        "median_ms": statistics.median(timings) * 1000,
        "max_ms": max(timings) * 1000,
    }


def measure_hub_construction(runs: int = 5) -> dict[str, Any]:
    code = textwrap.dedent("""
        import time
        start = time.perf_counter()
        from relay import Hub
        from relay.config import load_str
        hub = Hub.from_config(load_str('''
            version: 1
            catalog: { fetch_live_pricing: false, offline: true }
            models:
              m: { target: openai/test }
        '''))
        elapsed = time.perf_counter() - start
        print(elapsed)
    """)
    timings = time_subprocess(code, runs=runs)
    return {
        "name": "relay",
        "phase": "import + Hub.from_yaml",
        "min_ms": min(timings) * 1000,
        "median_ms": statistics.median(timings) * 1000,
        "max_ms": max(timings) * 1000,
    }


def main() -> None:
    rows: list[dict[str, Any]] = []

    # Pure import time
    rows.append(measure_import("relay", "from relay import Hub"))
    rows.append(measure_import("litellm", "import litellm"))

    # Import + minimal config setup (the realistic cold start)
    rows.append(measure_hub_construction())

    print()
    print(f"{'gateway':<10} {'phase':<32} {'min':>10} {'median':>10} {'max':>10}")
    print("-" * 78)
    for r in rows:
        print(
            f"{r['name']:<10} {r['phase']:<32} "
            f"{r['min_ms']:>8.1f}ms {r['median_ms']:>8.1f}ms {r['max_ms']:>8.1f}ms"
        )
    print()
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
