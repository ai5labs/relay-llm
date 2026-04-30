"""Relay error hierarchy.

All errors derive from ``RelayError``. Routing/retry logic catches the specific
subclasses to make the right decision (e.g. fall back on ContextWindowError,
back off on RateLimitError, surface AuthenticationError immediately).
"""

from __future__ import annotations

from typing import Any


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
        super().__init__(message)
        self.message = message
        self.provider = provider
        self.model = model
        self.status_code = status_code
        self.request_id = request_id
        self.raw = raw

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
