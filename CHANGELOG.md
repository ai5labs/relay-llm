# Changelog

All notable changes to this project will be documented in this file. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning](https://semver.org/).

<!-- towncrier release notes start -->

## [0.1.0] — 2026-05-01

First public release.

### Added — Core

- `Hub.from_yaml(...)` and `Hub.from_config(...)` — primary public entry points.
- YAML model catalog with Pydantic v2 validation, JSON Schema export, env-var resolution, and layered overrides.
- Native HTTP adapters for **18 providers**:
  - OpenAI-compatible: OpenAI, Groq, Together, DeepSeek, xAI, Mistral, Fireworks, Perplexity, OpenRouter, Ollama, vLLM, LM Studio.
  - Native: Anthropic, Azure OpenAI, AWS Bedrock, Google Gemini direct, Vertex AI, Cohere.
  - Plus: opt-in OpenAI Responses API (`/v1/responses`) via `api_style: responses`.
- Tiered pricing resolver: user override → live API (AWS Pricing, Azure Retail, OpenRouter) → snapshot → unknown, with provenance on every `Cost`.
- Built-in catalog snapshot of 400+ models with capabilities, pricing, and benchmark scores for popular models.
- Routing strategies: `fallback`, `loadbalance`, `weighted`, `conditional` (static).
- Per-target circuit breakers with cooldown + half-open probe semantics.
- Classified retry: distinct handling for rate limits, context-window errors, content-policy errors, and authentication failures.

### Added — Higher-level features

- **MCP universal tool layer** — connect MCP servers (stdio / SSE / streamable-http) and use their tools against any provider, including ones without native MCP. Tools auto-namespaced per server.
- **Cross-provider tool-schema compiler** — compiles JSON Schema to OpenAI strict, Anthropic, Gemini, Bedrock, Cohere shapes. Mastra-style instruction injection when constraints can't be expressed natively.
- **Pydantic structured output** — `request_structured(schema=MyModel, ...)` returns validated instances. Compiles per-provider; retries on validation failure.
- **Hub-level cache** + **Anthropic prompt-cache passthrough** via `CacheHint` markers.
- **Reasoning budget unification** across OpenAI / Anthropic / Gemini.
- **Batch API wrapper** for OpenAI Batch + Anthropic Message Batches (~50% cost savings).
- **OpenTelemetry GenAI semantic conventions** — opt-in `instrument(hub)`. Emits `gen_ai.*` spans + token/duration/cost histograms.

### Added — Production governance

- **PII redaction** pipeline with regex-based `RegexRedactor` reference impl + extension hooks for Microsoft Presidio.
- **Audit logging** — structured `AuditEvent` schema, `StdoutSink` / `FileSink` / `CallbackSink` / pluggable.
- **Pre / post guardrails** — `MaxInputLength`, `BlockedKeywords`, `Guardrail` Protocol for plugins.

### Added — CLI

- `relay schema --out FILE` — emit JSON Schema for editor autocomplete.
- `relay validate <yaml>` — validate config files (with overlays).
- `relay models list / inspect / compare / recommend` — model selection workflows.
- `relay catalog list` — browse the built-in catalog.
- `relay providers` — list supported providers.
- `relay version`.

### Added — Quality &amp; performance

- **Faster than LiteLLM at every percentile** ([BENCHMARKS.md](BENCHMARKS.md)).
- **8.5× faster cold start** (152 ms vs 1304 ms).
- 165 unit tests + 12 CLI tests + 9 contract tests + 7 microbenchmarks.
- Property-based SSE stream-reassembly invariants via Hypothesis.
- Tested on Python 3.10, 3.11, 3.12, 3.13 — all green.
- `mypy --strict` clean across 40 source files.
- `ruff` lint + format clean.
- Trusted-Publishing PyPI release workflow with Sigstore attestations.
- OpenSSF Scorecard, Dependabot, gitleaks pre-commit.

### Added — Docs &amp; community

- mkdocs-material site at `docs/` with concepts, features, production, and API reference pages.
- `BENCHMARKS.md` with reproducible methodology, raw numbers, and honest disclosures.
- `SUPPORT.md` covering code contributions, GitHub Sponsors, design partnership, and enterprise support.
- `.github/FUNDING.yml` enabling the Sponsor button.
- Issue templates: bug, provider request, enterprise inquiry.

### Apache-2.0 license

Copyright © 2026 ai5labs Research OPC Pvt Ltd.
