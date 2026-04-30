"""Catalog row schema and loader.

The shipped catalog lives at ``relay/catalog/data/models.json``. It's regenerated
weekly by ``scripts/refresh_catalog.py``, which:

* Calls ``GET /v1/models`` on every provider that supports it (OpenAI, Groq,
  Together, DeepSeek, xAI, Mistral, Fireworks, Anthropic).
* Pulls Bedrock pricing from the AWS Pricing API.
* Pulls Azure OpenAI pricing from the Azure Retail Prices API.
* Pulls OpenRouter ``/api/v1/models`` for cross-provider list-price coverage.
* Merges with hand-curated capability flags (some flags aren't in any API).

If the JSON file is missing (e.g. during library development before first
generation), ``get_catalog`` returns an empty mapping — the library still works,
it just can't auto-fill capabilities or compute costs without user-supplied data.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from functools import lru_cache
from importlib import resources
from typing import Any, Literal

ProviderId = Literal[
    "openai",
    "anthropic",
    "google",
    "vertex",
    "azure",
    "bedrock",
    "groq",
    "together",
    "deepseek",
    "xai",
    "mistral",
    "fireworks",
    "cohere",
    "perplexity",
    "ollama",
    "vllm",
    "lmstudio",
    "openrouter",
]


@dataclass(frozen=True, slots=True)
class BenchmarkScores:
    """Public benchmark scores for a model.

    All fields optional. Sourced from the provider's published numbers, the
    Artificial Analysis API, or LMSYS Arena where applicable. We never claim
    these are *our* benchmarks — they're aggregates of public data with
    provenance attached.

    These are *quality* signals to help users choose between models. Speed
    + cost are tracked elsewhere (``input_per_1m``, ``output_per_1m``,
    ``speed_tps``).
    """

    quality_index: float | None = None
    """Composite 0-100 quality index (Artificial Analysis style)."""
    arena_elo: int | None = None
    """LMSYS Chatbot Arena Elo rating."""
    mmlu: float | None = None
    """MMLU 5-shot accuracy (0-1 or 0-100, whichever the provider published)."""
    gpqa: float | None = None
    """GPQA Diamond accuracy."""
    humaneval: float | None = None
    """HumanEval pass@1 — code generation."""
    math: float | None = None
    """MATH benchmark accuracy."""
    swe_bench: float | None = None
    """SWE-bench Verified — agentic coding tasks."""
    sources: tuple[str, ...] = ()
    """Where these scores come from (URLs or labels like ``"anthropic-marketing"``)."""


@dataclass(frozen=True, slots=True)
class CatalogRow:
    """A single model's metadata as shipped with the library."""

    provider: str
    model_id: str
    context_window: int | None = None
    max_output: int | None = None
    input_per_1m: float | None = None
    output_per_1m: float | None = None
    cached_input_per_1m: float | None = None
    capabilities: tuple[str, ...] = ()
    modalities_in: tuple[str, ...] = ("text",)
    modalities_out: tuple[str, ...] = ("text",)

    # Performance + quality signals
    speed_tps: float | None = None
    """Output tokens-per-second under typical load (Artificial Analysis style)."""
    benchmarks: BenchmarkScores | None = None
    """Public benchmark scores. ``None`` if no scores known for this model."""
    aliases: tuple[str, ...] = ()
    """Alternate names users might call this model (``"sonnet"``, ``"4o"``, ...)."""

    deprecated: bool = False
    notes: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def slug(self) -> str:
        return f"{self.provider}/{self.model_id}"

    def supports(self, capability: str) -> bool:
        return capability in self.capabilities

    def cost_per_1m_avg(self) -> float | None:
        """Convenience: average of input + output per-1M cost. Useful for
        ``recommend`` queries when the user doesn't know their input/output ratio."""
        if self.input_per_1m is None or self.output_per_1m is None:
            return None
        return (self.input_per_1m + self.output_per_1m) / 2


@lru_cache(maxsize=1)
def get_catalog() -> dict[str, CatalogRow]:
    """Return the in-memory catalog keyed by ``provider/model_id``.

    Cached for the process lifetime — the catalog is immutable per-release.
    """
    try:
        data_file = resources.files("relay.catalog.data").joinpath("models.json")
        raw = data_file.read_text(encoding="utf-8")
    except (FileNotFoundError, ModuleNotFoundError):
        return {}

    try:
        rows = json.loads(raw)
    except json.JSONDecodeError:
        return {}

    if not isinstance(rows, list):
        return {}

    catalog: dict[str, CatalogRow] = {}
    known = {
        "provider",
        "model_id",
        "context_window",
        "max_output",
        "input_per_1m",
        "output_per_1m",
        "cached_input_per_1m",
        "capabilities",
        "modalities_in",
        "modalities_out",
        "deprecated",
        "notes",
        "speed_tps",
        "benchmarks",
        "aliases",
    }
    for r in rows:
        if not isinstance(r, dict) or "provider" not in r or "model_id" not in r:
            continue
        # Defensive: drop sentinel/bogus prices (negative, NaN). Some upstream
        # sources use ``-1`` for "variable" — keep the row but null the price.
        for k in ("input_per_1m", "output_per_1m", "cached_input_per_1m"):
            v = r.get(k)
            if isinstance(v, (int, float)) and (v < 0 or v != v):  # v != v → NaN
                r[k] = None
        bench_data = r.get("benchmarks")
        bench: BenchmarkScores | None = None
        if isinstance(bench_data, dict):
            bench = BenchmarkScores(
                quality_index=bench_data.get("quality_index"),
                arena_elo=bench_data.get("arena_elo"),
                mmlu=bench_data.get("mmlu"),
                gpqa=bench_data.get("gpqa"),
                humaneval=bench_data.get("humaneval"),
                math=bench_data.get("math"),
                swe_bench=bench_data.get("swe_bench"),
                sources=tuple(bench_data.get("sources", ())),
            )
        row = CatalogRow(
            provider=r["provider"],
            model_id=r["model_id"],
            context_window=r.get("context_window"),
            max_output=r.get("max_output"),
            input_per_1m=r.get("input_per_1m"),
            output_per_1m=r.get("output_per_1m"),
            cached_input_per_1m=r.get("cached_input_per_1m"),
            capabilities=tuple(r.get("capabilities", ())),
            modalities_in=tuple(r.get("modalities_in", ("text",))),
            modalities_out=tuple(r.get("modalities_out", ("text",))),
            speed_tps=r.get("speed_tps"),
            benchmarks=bench,
            aliases=tuple(r.get("aliases", ())),
            deprecated=bool(r.get("deprecated", False)),
            notes=r.get("notes"),
            extra={k: v for k, v in r.items() if k not in known},
        )
        catalog[row.slug] = row
    return catalog


def lookup(provider: str, model_id: str) -> CatalogRow | None:
    """Find a catalog row by ``provider`` and ``model_id``.

    Returns ``None`` if the model isn't in the shipped snapshot — the library
    is designed to keep working in that case (cost will surface as ``None``).
    """
    return get_catalog().get(f"{provider}/{model_id}")
