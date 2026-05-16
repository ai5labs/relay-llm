# Changelog

All notable changes to this project will be documented in this file. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning](https://semver.org/).

<!-- towncrier release notes start -->

## [0.2.1] — 2026-05-16

Security hardening pass (audit at `AUDIT_2026_05_16.md`) plus an OpenAI
spec-compliance fix on `Message.content`.

### Security

**MCP stdio command injection** (same class as CVE-2026-30623, F1).
- `MCP_STDIO_ALLOWED_COMMANDS` allowlist (`npx`, `uvx`, `python`,
  `python3`, `node`, `docker`, `deno`, `bun`). `MCPManager.add_stdio`
  rejects anything else; `MCPServer.connect` re-validates at spawn so
  deserialized configs can't bypass. Escape hatch: `allow_arbitrary=True`.
  Connect-time re-validation runs *before* the `mcp` SDK import, so the
  defense fires (and is testable in CI) even when the optional `mcp`
  extra is not installed.
- MCP tool arguments validated against the declared JSON Schema before
  dispatch; violations raise `ToolSchemaError`.
- MCP tool results capped at 256 KB with a truncation sentinel.

**base_url SSRF + credential exfiltration** (F2, F4, F11, AUTH-1).
- `ModelEntry.base_url` validator rejects plaintext `http://` against
  non-loopback hosts and any address resolving into RFC1918 / link-local
  / metadata / CGNAT / ULA ranges (opt back in per entry with
  `allow_private_hosts: true`).
- IPv4 alt-encoding bypass closed: decimal (`2852039166`), hex
  (`0xa9fea9fe`), and short forms (`169.254.43518`) are now normalized
  via `socket.inet_aton` so every form the OS resolver accepts is
  fenced. Loopback alt-encodings (`127.1`, `0177.0.0.1`) stay allowed
  — they're the local-vLLM use case.
- `google/*` targets require `https://` unconditionally; Gemini API key
  moved from `?key=` query string to `x-goog-api-key` header.
- `RelayConfig` detects group cycles at load time via DFS rather than
  blowing the Python recursion limit at request time.
- `GroupMember.weight` rejects negative and non-finite values.

**Streaming guardrails + deadline** (AUTH-2, F3).
- `Hub._stream_one` runs the redactor and pre/post guardrails on the
  streaming path. Post-violation buffers replaced with a block marker;
  terminating `StreamErrorEvent` emitted so callers never receive
  blocked content as the final response object. Threat-model limit:
  individual `TextDelta` events ship before post-guardrails fire — see
  `docs/production/security.md`.
- New `GlobalDefaults.stream_overall_timeout` (default 300 s) caps
  wall-clock time on streaming calls. The inner provider generator is
  `aclose()`d on deadline so the httpx SSE socket tears down promptly
  rather than waiting for GC. Deadline bracketed before `__aiter__()`
  so connect-stage hangs count against it too.

**Cache hardening** (AUTH-3, AUTH-4, F5).
- `Hub` cache-hit path runs `evaluate_post` against cached responses
  before returning — a tightened guardrail no longer lets stale-but-
  now-blocked content through.
- `cache_key(scope=...)` partitions by tenant; `Hub` bakes
  `metadata.user_id` in automatically when present.
- When the redactor mutated the request, the pre-redaction content is
  hashed into the cache key (post-redaction response is what gets
  cached). Two users with distinct PII no longer collide on a single
  cached response.

**Redaction sweep** (AUTH-5, AUTH-6, AUTH-7, F7, F8).
- `Hub.chat(..., trust_system=False)` and `StripUserSystem` guardrail
  reject `role="system"` entries from untrusted message lists.
- `RelayError.raw` and `RelayError.message` both scrub `Authorization` /
  `Bearer` / `x-api-key` / `sk-…` / GitHub PAT / JWT / Google `AIza…` /
  PEM private-key / Slack-token shaped substrings. `str(err)` carries a
  `<raw redacted>` marker; opt-in `err.raw_unsafe()`.
- `relay models inspect` redacts `LiteralCredential.value` in text and
  JSON output.
- `capture_messages="never"` sanitizes audit `error_message` to
  `{error_type}: status={code}` so provider response bodies (which
  often echo the prompt back) don't leak.
- OTel `capture_messages="full"` events emitted from post-redaction
  messages.

**Tool-call validation + audit + clock hygiene** (AUTH-8, F9, F10, F12).
- Provider-returned `ToolCall.arguments` validated against the
  originating `ToolDefinition.parameters`; mismatch raises
  `ToolSchemaError`. Schema validator no longer skips on shape — only
  literally empty `{}` short-circuits, so `allOf` / `enum` / `$ref` /
  `additionalProperties: false` schemas are now enforced.
- Unknown tool name in a response (model hallucinated a tool not in
  the declared list) hard-raises `ToolSchemaError` rather than silently
  reaching the caller's dispatcher.
- Audit-sink failures logged via `logger.warning("audit_sink_failed")`
  and counted at `relay.audit.audit_sink_failures` instead of being
  swallowed. New `Hub.from_yaml(..., strict_audit=True)` re-raises.
- `CallbackSink` warns on any non-async callable (lambdas, callable
  classes, `functools.partial` over sync) — blocks the event loop.
- `CircuitBreaker` switched to `time.monotonic` — NTP steps no longer
  perturb open-circuit cooldown.

**Docs.** `docs/production/security.md` gains a trust-boundaries
section: `base_url`, MCP stdio allowlist, multi-tenant cache scoping,
streaming threat-model limit, `capture_messages="full"` warnings,
`strict_audit` failure semantics, DNS-time exfiltration as a known
sync-validator limit.

### Fixed

- `Message.content` now coerces `None` to `""` at the validation
  boundary (OpenAI spec compliance for assistant messages with
  `tool_calls`). Downstream adapters (Anthropic / Gemini / Bedrock) all
  require a non-null `str | list`, so the conversion belongs at the
  input boundary instead of forcing every caller to pre-coerce.

## [0.2.0] — Unreleased

### Added

- `relay.routing` — public extension point for picking a model per call.
  - `Router` Protocol; `RouteRequest`, `RouteConstraints`, `RouteDecision` data types.
  - `RuleBasedRouter` — deterministic, constraint-based ranking against the
    catalog (same logic as `relay models recommend`).
  - `SemanticRouter` — HTTP client for the wire protocol at
    `docs/routing/api-spec.md`. Default endpoint targets the future hosted
    router service.
- `Hub.attach_router(router)` and `Hub.chat_routed(messages, ...)` — opt-in;
  existing `chat()` is unchanged.

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
