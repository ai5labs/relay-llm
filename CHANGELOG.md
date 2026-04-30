# Changelog

All notable changes to this project will be documented in this file. Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and [Semantic Versioning](https://semver.org/).

<!-- towncrier release notes start -->

## [0.1.0] - unreleased

### Added

- Initial release.
- `Hub.from_yaml` API for loading model catalogs from YAML.
- OpenAI-compatible adapter covering OpenAI, Groq, Together, DeepSeek, xAI, Mistral, Fireworks, Perplexity, OpenRouter, Ollama, vLLM, LM Studio.
- Native Anthropic Messages API adapter with `thinking` block preservation.
- Tiered pricing resolver (user override → live API → snapshot → unknown) with provenance metadata.
- Built-in catalog snapshot covering ~50 popular models across 12 providers.
- Routing strategies: `fallback`, `loadbalance`, `weighted`, `conditional` (static).
- Classified retry semantics distinguishing rate limits, context-window errors, content-policy errors, and authentication failures.
- CLI: `relay schema`, `relay validate`, `relay models list/inspect`, `relay catalog list`, `relay providers`.
- JSON Schema export for VS Code YAML autocomplete.
- Apache-2.0 license, Contributor Covenant 2.1, security policy, vulnerability reporting workflow.
