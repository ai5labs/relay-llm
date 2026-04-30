# Security Policy

## Supported versions

While Relay is in 0.x, only the latest minor release receives security fixes.

## Reporting a vulnerability

Please use [GitHub Private Vulnerability Reporting](https://github.com/ai5labs/relay-llm/security/advisories/new). Do not open a public issue.

We will:

1. Acknowledge receipt within 72 hours.
2. Triage and confirm within 7 days.
3. Coordinate a disclosure window (typically 90 days) and a fix release.
4. Credit you in the release notes unless you prefer to remain anonymous.

## Threat model

Relay is a *library*, not a hosted gateway. Your provider API keys stay in your process; we never proxy them through a third-party service. This is a deliberate design choice — see the LiteLLM proxy PyPI compromise of March 2026 for why centralizing keys is dangerous.

We focus on:

- **Credential handling.** Keys are loaded only when needed and never logged. `repr()` on credential objects redacts the secret value.
- **Supply chain.** Releases are published via PyPI Trusted Publishing (OIDC, no long-lived tokens) and signed with Sigstore.
- **Dependency hygiene.** Dependabot is enabled; we keep our runtime dep set narrow.
- **Prompt-injection-via-error-messages.** Error messages from upstream providers are surfaced verbatim in `RelayError.raw` but the typed `message` is sanitized to avoid accidental log injection of user-controlled content.

We do **not** treat as security issues:

- The library not protecting you against prompt injection in user input — that is an application-level concern.
- The cost tracker producing wrong figures because a provider changed list pricing — file a normal bug.
