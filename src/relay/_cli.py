"""``relay`` CLI entry point.

Subcommands:

    relay schema [--out FILE]
        Print or write the JSON Schema for the YAML config (point your editor
        at this for autocomplete).

    relay validate <config.yaml> [<overlay.yaml> ...]
        Load a config (with overlays) and report any validation errors.

    relay models list [--config FILE] [--tag TAG]
        List configured aliases + their resolved provider, model id, and price.

    relay models inspect <alias> [--config FILE]
        Show full details for one alias.

    relay catalog list [--provider PROVIDER]
        List the built-in catalog rows.

    relay version
        Print the library version.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from relay._version import __version__
from relay.catalog import get_catalog
from relay.config import json_schema, load
from relay.errors import RelayError
from relay.providers import supported_providers


def _cmd_schema(args: argparse.Namespace) -> int:
    schema = json_schema()
    text = json.dumps(schema, indent=2, sort_keys=True)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"wrote schema to {args.out}")
    else:
        print(text)
    return 0


def _cmd_validate(args: argparse.Namespace) -> int:
    try:
        cfg = load(*args.paths)
    except RelayError as e:
        print(f"INVALID: {e}", file=sys.stderr)
        return 1
    n_models = len(cfg.models)
    n_groups = len(cfg.groups)
    print(f"OK: {n_models} model(s), {n_groups} group(s)")
    return 0


def _cmd_models_list(args: argparse.Namespace) -> int:
    try:
        cfg = load(*([args.config] if args.config else ["models.yaml"]))
    except RelayError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    rows: list[tuple[str, str, str]] = []
    for alias, m in cfg.models.items():
        if args.tag and args.tag not in m.tags:
            continue
        rows.append((alias, m.provider, m.model_id))
    if args.json:
        print(
            json.dumps(
                [{"alias": a, "provider": p, "model_id": mid} for a, p, mid in rows],
                indent=2,
            )
        )
        return 0
    if not rows:
        print("(no models matched)")
        return 0
    width = max(len(a) for a, _, _ in rows)
    for alias, provider, model_id in sorted(rows):
        print(f"{alias.ljust(width)}  {provider}/{model_id}")
    return 0


def _cmd_models_inspect(args: argparse.Namespace) -> int:
    try:
        cfg = load(*([args.config] if args.config else ["models.yaml"]))
    except RelayError as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    entry = cfg.models.get(args.alias)
    if entry is None:
        print(f"unknown alias: {args.alias!r}", file=sys.stderr)
        return 1
    catalog = get_catalog()
    row = catalog.get(entry.target) or (
        catalog.get(entry.inherit_from) if entry.inherit_from else None
    )
    out: dict[str, Any] = entry.model_dump(exclude_none=True)
    # Never leak a literal credential value to stdout — anyone with shell
    # access to the box running ``relay models inspect`` would otherwise
    # read the production key off their terminal.
    cred = out.get("credential")
    if isinstance(cred, dict) and cred.get("type") == "literal":
        cred["value"] = "<literal-credential redacted>"
    if row:
        out["catalog"] = {
            "context_window": row.context_window,
            "max_output": row.max_output,
            "input_per_1m": row.input_per_1m,
            "output_per_1m": row.output_per_1m,
            "cached_input_per_1m": row.cached_input_per_1m,
            "capabilities": list(row.capabilities),
            "deprecated": row.deprecated,
        }
    print(json.dumps(out, indent=2, default=str))
    return 0


def _cmd_catalog_list(args: argparse.Namespace) -> int:
    catalog = get_catalog()
    rows = list(catalog.values())
    if args.provider:
        rows = [r for r in rows if r.provider == args.provider]
    rows.sort(key=lambda r: (r.provider, r.model_id))
    for r in rows:
        price = ""
        if r.input_per_1m is not None and r.output_per_1m is not None:
            price = f"  ${r.input_per_1m:.2f}/${r.output_per_1m:.2f} per 1M"
        ctx = f"  ctx={r.context_window}" if r.context_window else ""
        deprecated = "  [deprecated]" if r.deprecated else ""
        print(f"{r.provider}/{r.model_id}{ctx}{price}{deprecated}")
    print()
    print(f"total: {len(rows)} model(s) across {len({r.provider for r in rows})} provider(s)")
    return 0


def _cmd_providers(_args: argparse.Namespace) -> int:
    for p in supported_providers():
        print(p)
    return 0


def _cmd_models_compare(args: argparse.Namespace) -> int:
    """Side-by-side comparison of N models from the catalog."""
    catalog = get_catalog()
    rows: list[Any] = []
    missing: list[str] = []
    for slug in args.slugs:
        # Allow alias lookup: try direct slug, then search aliases.
        row = catalog.get(slug)
        if row is None:
            for r in catalog.values():
                if slug in r.aliases or slug == r.model_id:
                    row = r
                    break
        if row is None:
            missing.append(slug)
        else:
            rows.append(row)
    if missing:
        print(f"unknown: {', '.join(missing)}", file=sys.stderr)
        if not rows:
            return 1

    if args.json:
        print(json.dumps([_row_to_dict(r) for r in rows], indent=2, default=str))
        return 0

    # Build a side-by-side table.
    from collections.abc import Callable

    fields: list[tuple[str, Callable[[Any], str]]] = [
        ("model", lambda r: r.slug),
        ("context", lambda r: f"{r.context_window:,}" if r.context_window else "—"),
        ("input/1M", lambda r: f"${r.input_per_1m:.2f}" if r.input_per_1m is not None else "—"),
        ("output/1M", lambda r: f"${r.output_per_1m:.2f}" if r.output_per_1m is not None else "—"),
        ("speed", lambda r: f"{r.speed_tps:.0f} tok/s" if r.speed_tps else "—"),
        (
            "quality",
            lambda r: (
                f"{r.benchmarks.quality_index}"
                if r.benchmarks and r.benchmarks.quality_index
                else "—"
            ),
        ),
        ("MMLU", lambda r: f"{r.benchmarks.mmlu}" if r.benchmarks and r.benchmarks.mmlu else "—"),
        ("GPQA", lambda r: f"{r.benchmarks.gpqa}" if r.benchmarks and r.benchmarks.gpqa else "—"),
        (
            "HumanEval",
            lambda r: (
                f"{r.benchmarks.humaneval}" if r.benchmarks and r.benchmarks.humaneval else "—"
            ),
        ),
        ("MATH", lambda r: f"{r.benchmarks.math}" if r.benchmarks and r.benchmarks.math else "—"),
        (
            "SWE-bench",
            lambda r: (
                f"{r.benchmarks.swe_bench}" if r.benchmarks and r.benchmarks.swe_bench else "—"
            ),
        ),
        ("vision", lambda r: "✓" if "vision" in r.capabilities else "—"),
        ("tools", lambda r: "✓" if "tools" in r.capabilities else "—"),
        ("thinking", lambda r: "✓" if "thinking" in r.capabilities else "—"),
    ]

    label_w = max(len(name) for name, _ in fields)
    col_w = max(max(len(fn(r)) for _, fn in fields) for r in rows)
    col_w = max(col_w, 12)

    for name, getter in fields:
        line = f"{name:<{label_w}}  "
        for r in rows:
            line += f"{getter(r):<{col_w}}  "
        print(line)
    print()
    print("(scores from each provider's published benchmarks; verify before quoting)")
    return 0


def _cmd_models_recommend(args: argparse.Namespace) -> int:
    """Recommend models for a task / budget / capability set."""
    catalog = get_catalog()
    rows = [r for r in catalog.values() if not r.deprecated]

    if args.needs:
        for cap in args.needs:
            rows = [r for r in rows if cap in r.capabilities]

    if args.providers:
        rows = [r for r in rows if r.provider in args.providers]

    # Budget filter.
    def _within(r: Any, threshold: float) -> bool:
        avg = r.cost_per_1m_avg()
        return avg is not None and avg < threshold

    if args.budget == "cheap":
        rows = [r for r in rows if _within(r, 1.0)]
    elif args.budget == "balanced":
        rows = [r for r in rows if _within(r, 10.0)]
    # premium: no filter

    # Score for ranking — varies by task.
    def score(r: Any) -> float:
        b = r.benchmarks
        if b is None or b.quality_index is None:
            return -1.0
        # Task-specific weighting against the quality_index.
        if args.task == "code":
            return (b.humaneval or b.quality_index) + (b.swe_bench or 0) * 0.5
        if args.task == "reasoning":
            return (b.gpqa or 0) * 0.6 + (b.math or 0) * 0.4
        if args.task == "math":
            return b.math or b.quality_index
        if args.task == "vision":
            return b.quality_index if "vision" in r.capabilities else -1.0
        # Default: chat — composite quality.
        return b.quality_index

    scored = [(r, score(r)) for r in rows]
    scored = [(r, s) for r, s in scored if s > 0]
    scored.sort(key=lambda x: -x[1])
    top = scored[: args.limit]

    if not top:
        print("no models matched (try fewer constraints)", file=sys.stderr)
        return 1

    if args.json:
        print(
            json.dumps(
                [{"slug": r.slug, "score": s, "row": _row_to_dict(r)} for r, s in top],
                indent=2,
                default=str,
            )
        )
        return 0

    print(f"Top {len(top)} for task={args.task!r} budget={args.budget!r}:")
    print()
    print(f"{'#':>2}  {'model':<48}  {'score':>6}  {'avg $/1M':>10}  {'speed':>10}  capabilities")
    print("-" * 110)
    for i, (r, s) in enumerate(top, 1):
        cost = r.cost_per_1m_avg()
        cost_str = f"${cost:.2f}" if cost is not None else "—"
        speed_str = f"{r.speed_tps:.0f} tok/s" if r.speed_tps else "—"
        caps = ",".join(sorted(set(r.capabilities) & {"tools", "vision", "thinking", "json_mode"}))
        print(f"{i:>2}  {r.slug:<48}  {s:>6.1f}  {cost_str:>10}  {speed_str:>10}  {caps}")
    print()
    print(
        "(ranking based on published benchmarks + filters; pick + test for your specific workload)"
    )
    return 0


def _row_to_dict(r: Any) -> dict[str, Any]:
    """Serialize a CatalogRow including benchmarks + aliases for JSON output."""
    out: dict[str, Any] = {
        "provider": r.provider,
        "model_id": r.model_id,
        "context_window": r.context_window,
        "max_output": r.max_output,
        "input_per_1m": r.input_per_1m,
        "output_per_1m": r.output_per_1m,
        "cached_input_per_1m": r.cached_input_per_1m,
        "speed_tps": r.speed_tps,
        "capabilities": list(r.capabilities),
        "modalities_in": list(r.modalities_in),
        "modalities_out": list(r.modalities_out),
        "aliases": list(r.aliases),
        "deprecated": r.deprecated,
    }
    if r.benchmarks is not None:
        out["benchmarks"] = {
            "quality_index": r.benchmarks.quality_index,
            "arena_elo": r.benchmarks.arena_elo,
            "mmlu": r.benchmarks.mmlu,
            "gpqa": r.benchmarks.gpqa,
            "humaneval": r.benchmarks.humaneval,
            "math": r.benchmarks.math,
            "swe_bench": r.benchmarks.swe_bench,
            "sources": list(r.benchmarks.sources),
        }
    return out


def _cmd_version(_args: argparse.Namespace) -> int:
    print(__version__)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="relay", description="Relay CLI")
    subs = parser.add_subparsers(dest="cmd", required=True)

    p_schema = subs.add_parser("schema", help="emit the JSON Schema for relay YAML")
    p_schema.add_argument("--out", help="write to file instead of stdout")
    p_schema.set_defaults(func=_cmd_schema)

    p_validate = subs.add_parser("validate", help="validate a config (or layered configs)")
    p_validate.add_argument("paths", nargs="+", help="YAML files to load (in order)")
    p_validate.set_defaults(func=_cmd_validate)

    p_models = subs.add_parser("models", help="inspect configured models")
    sub_models = p_models.add_subparsers(dest="models_cmd", required=True)

    p_models_list = sub_models.add_parser("list", help="list configured aliases")
    p_models_list.add_argument("--config", help="YAML path (default: models.yaml)")
    p_models_list.add_argument("--tag", help="filter by tag")
    p_models_list.add_argument("--json", action="store_true", help="emit JSON")
    p_models_list.set_defaults(func=_cmd_models_list)

    p_models_inspect = sub_models.add_parser("inspect", help="inspect one alias")
    p_models_inspect.add_argument("alias")
    p_models_inspect.add_argument("--config", help="YAML path (default: models.yaml)")
    p_models_inspect.set_defaults(func=_cmd_models_inspect)

    p_models_compare = sub_models.add_parser(
        "compare", help="side-by-side comparison of N models from the catalog"
    )
    p_models_compare.add_argument(
        "slugs", nargs="+", help="model slugs (e.g. anthropic/claude-sonnet-4-5) or aliases"
    )
    p_models_compare.add_argument("--json", action="store_true", help="emit JSON")
    p_models_compare.set_defaults(func=_cmd_models_compare)

    p_models_recommend = sub_models.add_parser(
        "recommend", help="recommend models for a task / budget / capability set"
    )
    p_models_recommend.add_argument(
        "--task",
        choices=["chat", "code", "reasoning", "math", "vision"],
        default="chat",
        help="primary task type to optimize for",
    )
    p_models_recommend.add_argument(
        "--budget",
        choices=["cheap", "balanced", "premium"],
        default="balanced",
        help="cheap=<$1/M avg; balanced=<$10/M; premium=any",
    )
    p_models_recommend.add_argument(
        "--needs",
        nargs="*",
        default=[],
        help="required capabilities (vision, tools, thinking, json_mode, ...)",
    )
    p_models_recommend.add_argument(
        "--providers",
        nargs="*",
        default=[],
        help="restrict to these providers (e.g. anthropic openai google)",
    )
    p_models_recommend.add_argument("--limit", type=int, default=10, help="how many to return")
    p_models_recommend.add_argument("--json", action="store_true", help="emit JSON")
    p_models_recommend.set_defaults(func=_cmd_models_recommend)

    p_catalog = subs.add_parser("catalog", help="inspect the built-in catalog")
    sub_catalog = p_catalog.add_subparsers(dest="catalog_cmd", required=True)

    p_catalog_list = sub_catalog.add_parser("list", help="list catalog rows")
    p_catalog_list.add_argument("--provider", help="filter by provider")
    p_catalog_list.set_defaults(func=_cmd_catalog_list)

    p_providers = subs.add_parser("providers", help="list supported providers")
    p_providers.set_defaults(func=_cmd_providers)

    p_version = subs.add_parser("version", help="print version")
    p_version.set_defaults(func=_cmd_version)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
