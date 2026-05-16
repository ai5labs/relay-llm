"""Relay error hierarchy.

All errors derive from ``RelayError``. Routing/retry logic catches the specific
subclasses to make the right decision (e.g. fall back on ContextWindowError,
back off on RateLimitError, surface AuthenticationError immediately).
"""

from __future__ import annotations

import re
from typing import Any

# Patterns that should never appear in a logged error body. Matched
# case-insensitively. The replacement is `"***"` rather than a sentinel so
# the redacted value still looks like a header / token to whatever log
# pipeline inspects it.
_SECRET_PATTERNS: tuple[re.Pattern[str], ...] = (
    # Authorization header values. Match the scheme explicitly so a body
    # without a scheme word ("Authorization: deadbeef") still gets scrubbed,
    # and so 'Basic <base64>' / 'Token <opaque>' / 'AWS4-HMAC-SHA256 ...'
    # don't slip past a "bearer is optional" fallback. The credential is
    # everything from the scheme through to whitespace/quote/EOL.
    re.compile(
        r"(?i)(authorization\s*[:=]\s*)"
        r"(?:bearer|basic|token|digest|negotiate|"
        r"aws4-hmac-sha256|signature|hmac|apikey)?\s*"
        r"[\w\-\.~+/=]+",
    ),
    # x-api-key / api-key / x-goog-api-key / anthropic-api-key — header forms.
    re.compile(
        r"(?i)((?:x[-_])?(?:goog[-_]|anthropic[-_])?api[-_]?key\s*[:=]\s*)"
        r"[\w\-\.~+/=]+"
    ),
    # Bare bearer tokens.
    re.compile(r"(?i)(bearer\s+)[\w\-\.~+/=]{8,}"),
    # AWS access keys (long-lived + STS-temporary).
    re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"),
    # OpenAI / Anthropic / Anthropic-admin style keys.
    re.compile(r"\bsk-(?:ant-)?(?:admin[0-9]+-)?[A-Za-z0-9\-_]{20,}\b"),
    # Stripe live + restricted keys.
    re.compile(r"\b(?:sk|pk|rk)_(?:live|test)_[A-Za-z0-9]{24,}\b"),
    # GitHub classic PATs / installation tokens — short prefix family.
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{30,}\b"),
    # GitHub fine-grained PATs.
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"),
    # Hugging Face access tokens.
    re.compile(r"\bhf_[A-Za-z0-9]{30,}\b"),
    # Google API keys (Maps / Cloud / Gemini direct).
    re.compile(r"\bAIza[0-9A-Za-z\-_]{35}\b"),
    # JWTs: three base64url segments separated by dots.
    re.compile(r"\beyJ[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\.[A-Za-z0-9_\-]+\b"),
    # PEM private keys — collapse the whole armored block.
    re.compile(
        r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
        re.DOTALL,
    ),
    # Slack tokens (bot / user / app / refresh / config).
    re.compile(r"\bxox[abprsoe]-[A-Za-z0-9\-]{10,}\b"),
)


def _scrub_secrets(value: Any) -> Any:
    """Strip credential-shaped substrings out of an arbitrary value.

    Walks dicts / lists / tuples; leaves non-strings untouched. The point is
    not to catch every possible secret — it's to keep the obvious ones
    (Authorization header echoes, ``sk-`` keys) from ending up in audit
    rows, third-party log pipelines, and ``logger.error("%s", exc.raw)``.
    """
    if isinstance(value, str):
        scrubbed = value
        for pat in _SECRET_PATTERNS:
            scrubbed = pat.sub(
                lambda m: (m.group(1) + "***") if m.lastindex else "***",
                scrubbed,
            )
        return scrubbed
    if isinstance(value, dict):
        return {k: _scrub_secrets(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_scrub_secrets(v) for v in value]
    if isinstance(value, tuple):
        return tuple(_scrub_secrets(v) for v in value)
    return value


class RelayError(Exception):
    """Base class for all Relay errors."""

    def __init__(
        self,
        message: str,
        *,
        provider: str | None = None,
        model: str | None = None,
        status_code: int | None = None,
        request_id: str | None = None,
        raw: Any = None,
    ) -> None:
        # Providers often build the ``message`` from a snippet of the upstream
        # body (see openai_compat._raise_for_status). Scrub both fields so
        # neither ``str(err)`` nor ``err.raw`` carries a credential.
        scrubbed_message = _scrub_secrets(message) if isinstance(message, str) else message
        super().__init__(scrubbed_message)
        self.message = scrubbed_message
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.request_id = request_id
        # ``raw_unsafe()`` is the opt-in escape hatch for callers that
        # explicitly want the unredacted body.
        self._raw_unsafe = raw
        self.raw = _scrub_secrets(raw)

    def raw_unsafe(self) -> Any:
        """Return the unscrubbed raw payload. Use sparingly — secret-shaped
        substrings will be present."""
        return self._raw_unsafe

    def __str__(self) -> str:
        if self.raw is not None:
            return f"{self.message} <raw redacted; use err.raw_unsafe() to access>"
        return self.message

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        parts = [f"message={self.message!r}"]
        if self.provider:
            parts.append(f"provider={self.provider!r}")
        if self.model:
            parts.append(f"model={self.model!r}")
        if self.status_code is not None:
            parts.append(f"status={self.status_code}")
        return f"{type(self).__name__}({', '.join(parts)})"


class ConfigError(RelayError):
    """The YAML config or model catalog is invalid or references something missing."""


class AuthenticationError(RelayError):
    """Provider rejected the credential. 401/403 — never retry."""


class RateLimitError(RelayError):
    """Provider returned 429 or equivalent. Retryable with backoff."""

    def __init__(
        self,
        message: str,
        *,
        retry_after: float | None = None,
        limit_type: str | None = None,  # "rpm" | "tpm" | "concurrency" | None
        **kwargs: Any,
    ) -> None:
        super().__init__(message, **kwargs)
        self.retry_after = retry_after
        self.limit_type = limit_type


class ContextWindowError(RelayError):
    """Input + requested output exceeds the model's context window. Fall back, not retry."""


class ContentPolicyError(RelayError):
    """Provider refused on safety/policy grounds. Fall back, not retry."""


class TimeoutError(RelayError):
    """Request exceeded the configured timeout. Retryable."""


class ProviderError(RelayError):
    """Generic upstream error not covered by a more specific class."""


class ToolSchemaError(RelayError):
    """A tool's JSON Schema cannot be expressed for the target provider."""


__all__ = [
    "AuthenticationError",
    "ConfigError",
    "ContentPolicyError",
    "ContextWindowError",
    "ProviderError",
    "RateLimitError",
    "RelayError",
    "TimeoutError",
    "ToolSchemaError",
]
