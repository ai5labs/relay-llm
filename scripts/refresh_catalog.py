#!/usr/bin/env python3
"""Refresh ``src/relay/catalog/data/models.json`` from public sources.

Sources used (in this priority order):

1. **OpenRouter** ``/api/v1/models`` — no auth required, covers 400+ models with
   per-1M-token list pricing. Single best source for cross-provider data.
2. **Provider /v1/models endpoints** — OpenAI, Anthropic, Groq, Together,
   Mistral, DeepSeek (most require an API key for context-window fields). Used
   to keep the catalog in sync with newly released models *without* pricing,
   which OpenRouter then fills in on the next merge.
3. **Hand-curated capability flags** — JSON file ``scripts/curated.json``
   captures the long tail (e.g. which models support ``thinking`` mode) that
   no API exposes.

Usage::

    uv run python scripts/refresh_catalog.py                 # in-place refresh
    uv run python scripts/refresh_catalog.py --dry-run       # diff only
    uv run python scripts/refresh_catalog.py --out new.json  # write elsewhere

Run weekly via ``.github/workflows/refresh_catalog.yml``; the workflow opens a
PR with the diff so a human reviews before the catalog ships in a release.
"""

from __future__ import annotations

import argparse
import asyncio
import difflib
import json
import sys
from pathlib import Path
from typing import Any

import httpx
import orjson

CATALOG_PATH = Path(__file__).parent.parent / "src" / "relay" / "catalog" / "data" / "models.json"
CURATED_PATH = Path(__file__).parent / "curated.json"

# Artificial Analysis API — opt-in via AA_API_KEY env var. Their data is
# proprietary; redistribution may require a commercial license. Scores fetched
# here are merged into the catalog at refresh time but the resulting
# ``models.json`` should not be republished as your own.
AA_BASE = "https://artificialanalysis.ai/api/v2"


# ---------------------------------------------------------------------------
# OpenRouter — primary source for prices
# ---------------------------------------------------------------------------


async def fetch_openrouter() -> dict[str, dict[str, Any]]:
    """Index OpenRouter models by ``provider/model_id`` slug."""
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.get("https://openrouter.ai/api/v1/models")
        resp.raise_for_status()
        data = orjson.loads(resp.content)

    out: dict[str, dict[str, Any]] = {}
    for m in data.get("data") or []:
        slug = m.get("id")
        if not slug:
            continue
        provider, _, model_id = slug.partition("/")
        if not model_id:
            continue
        pricing = m.get("pricing") or {}
        try:
            in_per_token = float(pricing.get("prompt") or 0)
            out_per_token = float(pricing.get("completion") or 0)
        except ValueError:
            continue

        ctx = m.get("context_length")
        caps: list[str] = []
        if m.get("supported_parameters") or []:
            params = m["supported_parameters"]
            if "tools" in params or "functions" in params:
                caps.append("tools")
            if "response_format" in params:
                caps.append("json_mode")
        modalities = m.get("architecture", {}).get("input_modalities") or ["text"]
        if "image" in modalities:
            caps.append("vision")
        # OpenRouter uses negative sentinels (e.g. -1) to mean "variable /
        # dynamic" for routed-meta models like ``openrouter/auto``. Skip
        # those so the catalog stays sane.
        in_1m = in_per_token * 1_000_000 if in_per_token > 0 else None
        out_1m = out_per_token * 1_000_000 if out_per_token > 0 else None
        out[slug] = {
            "provider": _normalize_provider_id(provider),
            "model_id": model_id,
            "context_window": ctx,
            "input_per_1m": in_1m,
            "output_per_1m": out_1m,
            "capabilities": caps,
            "modalities_in": modalities,
        }
    return out


def _normalize_provider_id(provider: str) -> str:
    """Map OpenRouter's provider slugs to Relay's stable provider ids."""
    return {
        "openai": "openai",
        "anthropic": "anthropic",
        "google": "google",
        "groq": "groq",
        "mistralai": "mistral",
        "x-ai": "xai",
        "deepseek": "deepseek",
        "perplexity": "perplexity",
        "cohere": "cohere",
        "together": "together",
        "fireworks": "fireworks",
    }.get(provider, provider)


# ---------------------------------------------------------------------------
# Curated overrides
# ---------------------------------------------------------------------------


def load_curated() -> dict[str, dict[str, Any]]:
    """Hand-edited overrides keyed by ``provider/model_id``.

    Use this for capability flags no public API surfaces (``thinking``,
    ``prompt_cache``, etc.) and for cached-input pricing tiers.
    """
    if not CURATED_PATH.exists():
        return {}
    raw = json.loads(CURATED_PATH.read_text(encoding="utf-8"))
    # Strip top-level metadata keys (``_comment`` etc.); keep only slug→dict.
    return {k: v for k, v in raw.items() if isinstance(v, dict)}


# ---------------------------------------------------------------------------
# Artificial Analysis — opt-in benchmark scores
# ---------------------------------------------------------------------------


async def fetch_artificial_analysis() -> dict[str, dict[str, Any]]:
    """Fetch quality_index + speed_tps from Artificial Analysis.

    Requires ``AA_API_KEY`` env var. Returns ``{}`` (silent skip) if not set
    or if the API call fails — the curated/snapshot tier will fill in.

    AA's slug format differs from ours (they use names like ``"gpt-4o"``
    or ``"claude-3-5-sonnet"`` without the provider prefix). We do a
    best-effort substring match against our known providers.
    """
    import os

    api_key = os.environ.get("AA_API_KEY")
    if not api_key:
        return {}

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.get(
                f"{AA_BASE}/data/llms/models",
                headers={"x-api-key": api_key},
            )
            resp.raise_for_status()
            data = orjson.loads(resp.content)
    except Exception as e:
        print(f"  Artificial Analysis fetch failed: {e}", file=sys.stderr)
        return {}

    out: dict[str, dict[str, Any]] = {}
    # AA's response shape (subject to change without notice — verify if it breaks):
    # { "data": [ { "slug": "gpt-4o", "creator": {"slug": "openai"},
    #              "quality_index": 71, "median_output_tokens_per_second": 110, ... } ] }
    for item in data.get("data") or []:
        creator = (item.get("creator") or {}).get("slug") or item.get("provider")
        model_slug = item.get("slug") or item.get("name")
        if not creator or not model_slug:
            continue
        relay_slug = f"{creator}/{model_slug}"
        bench: dict[str, Any] = {}
        if "quality_index" in item:
            bench["quality_index"] = item["quality_index"]
        if "mmlu_score" in item:
            bench["mmlu"] = item["mmlu_score"]
        if "gpqa_score" in item:
            bench["gpqa"] = item["gpqa_score"]
        if "humaneval_score" in item:
            bench["humaneval"] = item["humaneval_score"]
        if "math_score" in item:
            bench["math"] = item["math_score"]
        if bench:
            bench["sources"] = ["artificial-analysis"]

        out[relay_slug] = {
            "speed_tps": item.get("median_output_tokens_per_second"),
            "benchmarks": bench if bench else None,
        }
    return out


# ---------------------------------------------------------------------------
# Merge + diff
# ---------------------------------------------------------------------------


def load_existing() -> list[dict[str, Any]]:
    if not CATALOG_PATH.exists():
        return []
    raw = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    return [r for r in raw if isinstance(r, dict) and "provider" in r and "model_id" in r]


def merge(
    existing: list[dict[str, Any]],
    openrouter_index: dict[str, dict[str, Any]],
    curated_index: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Merge new data into the existing catalog without dropping curated rows.

    Precedence (highest first):
      1. ``curated_index`` — hand-edited fields win.
      2. existing catalog row — fields not present in the new sources are kept.
      3. ``openrouter_index`` — primary refresh source.
    """
    by_slug: dict[str, dict[str, Any]] = {}

    # Layer 1: existing — preserves models OpenRouter doesn't cover.
    for row in existing:
        slug = f"{row['provider']}/{row['model_id']}"
        by_slug[slug] = dict(row)

    # Layer 2: OpenRouter additions / pricing refresh.
    for slug, row in openrouter_index.items():
        merged = by_slug.get(slug, {})
        for k, v in row.items():
            if v is None:
                continue
            # Don't overwrite a curated capability list with a smaller one.
            if k == "capabilities":
                merged_caps = set(merged.get("capabilities") or [])
                merged_caps.update(v)
                merged[k] = sorted(merged_caps)
            else:
                merged[k] = v
        by_slug[slug] = merged

    # Layer 3: curated wins.
    for slug, overrides in curated_index.items():
        merged = by_slug.get(slug, {})
        for k, v in overrides.items():
            merged[k] = v
        by_slug[slug] = merged

    return [by_slug[s] for s in sorted(by_slug.keys())]


def write_catalog(rows: list[dict[str, Any]], out_path: Path) -> None:
    header = {
        "_comment": (
            "Relay built-in model catalog. Auto-refreshed from OpenRouter + "
            "curated overrides via scripts/refresh_catalog.py."
        ),
        "_schema_version": 1,
    }
    payload = [header, *rows]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def diff_text(old_path: Path, new_path: Path) -> str:
    old = old_path.read_text(encoding="utf-8") if old_path.exists() else ""
    new = new_path.read_text(encoding="utf-8")
    return "".join(
        difflib.unified_diff(
            old.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=str(old_path),
            tofile=str(new_path),
            n=2,
        )
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


async def main() -> int:
    parser = argparse.ArgumentParser(description="refresh the Relay catalog")
    parser.add_argument("--dry-run", action="store_true", help="emit diff but don't write")
    parser.add_argument("--out", type=Path, default=CATALOG_PATH, help="output path")
    args = parser.parse_args()

    print("fetching OpenRouter /api/v1/models ...", file=sys.stderr)
    try:
        or_index = await fetch_openrouter()
    except Exception as e:
        print(f"OpenRouter fetch failed: {e}", file=sys.stderr)
        return 1
    print(f"  → {len(or_index)} models", file=sys.stderr)

    print("fetching Artificial Analysis (opt-in) ...", file=sys.stderr)
    aa_index = await fetch_artificial_analysis()
    if aa_index:
        print(f"  → {len(aa_index)} AA rows", file=sys.stderr)
    else:
        print("  → skipped (AA_API_KEY not set, or fetch failed)", file=sys.stderr)

    print("loading curated overrides ...", file=sys.stderr)
    curated = load_curated()
    print(f"  → {len(curated)} curated rows", file=sys.stderr)

    print("loading existing catalog ...", file=sys.stderr)
    existing = load_existing()
    print(f"  → {len(existing)} existing rows", file=sys.stderr)

    # AA is layered between OpenRouter and curated — curated wins on conflicts.
    rows = merge(existing, or_index, {**aa_index, **curated})
    print(f"merged: {len(rows)} rows", file=sys.stderr)

    if args.dry_run:
        # Write to a temp file then diff.
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as tmp:
            tmp_path = Path(tmp.name)
        write_catalog(rows, tmp_path)
        print(diff_text(args.out, tmp_path))
        tmp_path.unlink()
    else:
        write_catalog(rows, args.out)
        print(f"wrote {args.out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
