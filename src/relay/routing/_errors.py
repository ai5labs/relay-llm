"""Routing error hierarchy.

All routing errors derive from :class:`RoutingError`, which itself derives from
:class:`relay.errors.RelayError`. Hub fallback logic catches the specific
subclasses to differentiate retryable vs terminal failures.
"""

from __future__ import annotations

from relay.errors import RelayError


class RoutingError(RelayError):
    """Base class for all routing errors."""


class RouterAuthError(RoutingError):
    """The hosted router rejected the credential. 401/403 — never retry."""


class RouterQuotaError(RoutingError):
    """The hosted router quota was exceeded. 429 — retryable with backoff."""


class RouterTimeoutError(RoutingError):
    """The router call exceeded its timeout. Retryable."""


class NoCandidatesError(RoutingError):
    """No candidate aliases survived the constraint filters."""


__all__ = [
    "NoCandidatesError",
    "RouterAuthError",
    "RouterQuotaError",
    "RouterTimeoutError",
    "RoutingError",
]
