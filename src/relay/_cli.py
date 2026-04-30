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
