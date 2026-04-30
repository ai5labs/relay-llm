# Contributing to Relay

Thanks for considering a contribution. This document covers what we expect.

## Quick start

```bash
git clone https://github.com/ai5labs/relay-llm
cd relay-llm
uv sync --all-groups          # installs runtime + dev + docs deps
uv run pytest                 # run the test suite
uv run ruff check             # lint
uv run ruff format --check    # format check
uv run mypy                   # strict type-check
uv run pyright                # second type-checker (catches different things)
```

## Where to start

- **Good first issues** are tagged `good-first-issue`.
- **Adding a provider**: see `docs/providers.md` (in v0.2 docs) for the recipe. Short version: one new file in `src/relay/providers/`, one entry in `src/relay/providers/__init__.py`'s `PROVIDER_REGISTRY`, plus catalog rows for the models, plus tests.
- **Catalog updates**: hand edits to `src/relay/catalog/data/models.json` are welcome but the file is regenerated weekly by CI from live APIs — don't be surprised if your edit is overwritten. If a model is missing, file an issue or update the generator script.

## PR expectations

1. **Tests required** for new behavior. Use `respx` to mock httpx calls; we don't hit real provider APIs in unit tests.
2. **Type-clean** under both `mypy --strict` and `pyright` strict.
3. **Ruff-clean**.
4. **Newsfragment** in `changes/` describing the change (towncrier format: `<issue-or-PR>.<type>.md` where type is `added` / `changed` / `fixed` / `removed` / `security`).
5. **No public API churn** without prior discussion. Anything under `_internal/` or `_*` is fair game.
6. **Streaming changes need a hypothesis test** for the SSE reassembly invariants. We won't repeat LiteLLM's history of streaming bugs.

## Commit / DCO

We use GitHub's DCO bot. Sign off your commits:

```bash
git commit -s -m "fix: parse Retry-After as float not int"
```

## Reporting security issues

Don't open a public issue for vulnerabilities. Use [GitHub private vulnerability reporting](https://github.com/ai5labs/relay-llm/security/advisories/new) — see [SECURITY.md](SECURITY.md).
