# Contributing

Thanks for considering a contribution. The full guide lives at the repo root: [CONTRIBUTING.md](https://github.com/ai5labs/relay-llm/blob/main/CONTRIBUTING.md).

## Quick reference

```bash
git clone https://github.com/ai5labs/relay-llm
cd relay-llm
uv sync --all-groups          # runtime + dev + docs deps
uv run pytest                 # 162+ tests
uv run ruff check
uv run ruff format --check
uv run mypy                   # strict
uv run pyright                # second type-checker
```

## PR expectations

1. **Tests required** for new behavior. Use `respx` for httpx mocks; we don't hit real provider APIs in unit tests.
2. **Type-clean** under both `mypy --strict` and `pyright` strict.
3. **Ruff-clean.**
4. **Newsfragment** in `changes/` (towncrier format: `<issue-or-PR>.<type>.md` where type is `added`/`changed`/`fixed`/`removed`/`security`).
5. **Streaming changes need a hypothesis test** for the SSE reassembly invariants. We won't repeat LiteLLM's history of streaming bugs.
6. **No public API churn** without prior discussion. Anything under `_internal/` or `_*` names is fair game.

## Adding a provider

1. New file in `src/relay/providers/`.
2. Register in `PROVIDER_REGISTRY` in `src/relay/providers/__init__.py`.
3. Catalog rows in `src/relay/catalog/data/models.json` (or curated overrides in `scripts/curated.json`).
4. Tests under `tests/unit/test_<provider>.py` using `respx`.
5. (Optional) Live contract test under `tests/contract/test_live_<provider>.py`.

## DCO sign-off

Sign off your commits:

```bash
git commit -s -m "fix: parse Retry-After as float"
```
