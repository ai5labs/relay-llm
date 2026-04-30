# Security

## Threat model

Relay is a **library**, not a hosted gateway. Your provider API keys stay in your process; we never proxy them through a third-party service. This is a deliberate design choice — see the [LiteLLM proxy PyPI compromise of March 2026](https://news.ycombinator.com/item?id=47501426) for why centralizing keys is dangerous.

## Credential handling

- Keys are loaded only when needed and never logged.
- `repr()` on credential objects redacts the secret value.
- Reference [credentials reified objects](../concepts/config.md#credentials), not inline strings, in YAML.

## Supply chain

- Releases are published via PyPI Trusted Publishing (OIDC, no long-lived tokens).
- Wheels and sdists are Sigstore-signed via `pypa/gh-action-pypi-publish` with `attestations: true`.
- Dependabot is enabled; we keep our runtime dep set narrow.
- OpenSSF Scorecard runs on every push to main.

## Reporting vulnerabilities

Don't open a public issue. Use [GitHub Private Vulnerability Reporting](https://github.com/ai5labs/relay-llm/security/advisories/new). We:

1. Acknowledge within 72 hours.
2. Triage and confirm within 7 days.
3. Coordinate a disclosure window (typically 90 days).
4. Credit you in the release notes (unless you prefer anonymity).

See the project's [SECURITY.md](https://github.com/ai5labs/relay-llm/blob/main/SECURITY.md).

## What's not in scope

We do **not** treat as security issues:

- The library not protecting you against prompt injection in user input — that is application-level.
- Cost tracking returning wrong figures because a provider changed list pricing — file a normal bug.
