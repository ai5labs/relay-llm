# Model selection — compare &amp; recommend

The catalog ships with **published benchmark scores** alongside pricing and capabilities, so you can pick the right model without leaving Relay.

```bash
relay models compare sonnet 4o flash
relay models recommend --task code --budget cheap --needs tools
```

## What's in the catalog

For each known model:

| Field | Source |
|---|---|
| `context_window`, `max_output` | provider docs |
| `input_per_1m`, `output_per_1m`, `cached_input_per_1m` | OpenRouter live + provider docs |
| `capabilities` (tools, vision, thinking, json_mode, prompt_cache, ...) | curated |
| `speed_tps` | Artificial Analysis (when `AA_API_KEY` set), else curated |
| `benchmarks.quality_index` | composite 0-100, AA-style |
| `benchmarks.mmlu`, `gpqa`, `humaneval`, `math`, `swe_bench` | provider system cards |
| `benchmarks.arena_elo` | LMSYS Chatbot Arena |
| `benchmarks.sources` | provenance for every score |
| `aliases` | nicknames users actually type (`sonnet`, `4o`, `flash`) |

## `relay models compare`

Side-by-side table. Aliases work.

```bash
$ relay models compare sonnet 4o flash
model      anthropic/claude-sonnet-4-5  openai/gpt-4o            google/gemini-2.5-flash
context    200,000                      128,000                  1,048,576
input/1M   $3.00                        $2.50                    $0.30
output/1M  $15.00                       $10.00                   $2.50
speed      95 tok/s                     110 tok/s                250 tok/s
quality    73                           71                       70
MMLU       88.7                         88.7                     85.3
GPQA       71.5                         53.6                     65.2
HumanEval  89.1                         90.2                     88.1
MATH       91.8                         76.6                     88.0
SWE-bench  49.0                         —                        —
vision     ✓                            ✓                        ✓
tools      ✓                            ✓                        ✓
thinking   ✓                            —                        ✓
```

JSON output: `--json`.

## `relay models recommend`

Pick the right model for your task and budget.

```bash
$ relay models recommend --task code --budget cheap --needs tools
Top 3 for task='code' budget='cheap':

 #  model                                  score    avg $/1M       speed  capabilities
--------------------------------------------------------------------------------------
 1  deepseek/deepseek-chat                  89.0       $0.60    50 tok/s  json_mode,tools
 2  groq/llama-3.3-70b-versatile            88.4       $0.69   280 tok/s  json_mode,tools
 3  openai/gpt-4o-mini                      87.2       $0.38   145 tok/s  json_mode,tools,vision
```

Flags:

- `--task` — `chat` (default) | `code` | `reasoning` | `math` | `vision`
- `--budget` — `cheap` (avg < $1/M) | `balanced` (< $10/M) | `premium` (no cap)
- `--needs` — required capabilities, e.g. `--needs tools vision`
- `--providers` — restrict, e.g. `--providers anthropic openai`
- `--limit` — top N (default 10)
- `--json` — emit JSON

## How task scoring works

| Task | Score formula |
|---|---|
| `chat` | `quality_index` |
| `code` | `humaneval + 0.5 × swe_bench` |
| `reasoning` | `0.6 × gpqa + 0.4 × math` |
| `math` | `math` |
| `vision` | `quality_index` (only models with `vision` capability) |

These are **starting points**, not gospel — every workload has its own quirks. The recommendation is a sensible default; production deployments should run their own evals.

## Where the scores come from

| Source | What it provides | License / terms |
|---|---|---|
| **Each provider's system cards / blog posts** | MMLU, GPQA, HumanEval, MATH, SWE-bench | provider-published, public |
| **OpenRouter `/api/v1/models`** | live pricing for ~400 models | free, no auth |
| **LMSYS Chatbot Arena** | `arena_elo` for chat models | CC-BY-licensed leaderboard |
| **Artificial Analysis** (opt-in) | `quality_index`, `speed_tps`, others | requires `AA_API_KEY`; commercial license needed for redistribution |

Set `AA_API_KEY` env var to enable AA enrichment in `python scripts/refresh_catalog.py`. Without it, you still get the curated provider-published scores for ~15 popular models.

## Caveats

- **Numbers drift fast** — providers re-test, benchmarks update, leaderboards shift. The catalog refreshes weekly via CI from OpenRouter; benchmark scores need manual updates in `scripts/curated.json`.
- **A score is not a recommendation** — your prompt distribution, latency budget, and tail behavior matter more than a single benchmark number.
- **No model wins everywhere** — code-strong models can be brittle on free-form chat; reasoning models are slow.
- **Always re-test on your own evals** before locking in a model in production.

## Programmatic access

```python
from relay.catalog import get_catalog, lookup

# Look up one model
row = lookup("anthropic", "claude-sonnet-4-5")
print(row.benchmarks.mmlu, row.speed_tps)

# Filter / sort yourself
models = [r for r in get_catalog().values() if r.benchmarks and r.benchmarks.gpqa]
models.sort(key=lambda r: r.benchmarks.gpqa, reverse=True)
for r in models[:5]:
    print(f"{r.slug:50}  GPQA={r.benchmarks.gpqa}")
```
