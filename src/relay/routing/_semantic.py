"""HTTP client for the hosted semantic router.

This module is a thin transport: it serializes a :class:`RouteRequest` to
the wire protocol documented at ``docs/routing/api-spec.md``, POSTs it,
and parses the response back into a :class:`RouteDecision`. No proprietary
logic lives here.

The default endpoint targets the future ai5labs-hosted service; users can
point :class:`SemanticRouter` at any compatible deployment by passing a
custom ``endpoint``.
"""

from __future__ import annotations

from datetime import datetime, timezone

import httpx

from ._errors import (
    RouterAuthError,
    RouterQuotaError,
    RouterTimeoutError,
    RoutingError,
)
from ._protocol import RouteDecision, RouteRequest

DEFAULT_ENDPOINT = "https://router.relay.ai5labs.com"


class SemanticRouter:
    """HTTP client for the hosted semantic routing API.

    The router serializes the request to JSON, POSTs to ``{endpoint}/v1/route``
    with a Bearer token, and validates the response against
    :class:`RouteDecision`. Network and HTTP errors are mapped onto the
    routing error hierarchy so callers can decide whether to retry or fail.

    Pass ``client=`` to inject a shared :class:`httpx.AsyncClient` (e.g. for
    test mocking via ``respx``); otherwise a private client is created and
    closed in :meth:`aclose`.
    """

    def __init__(
        self,
        endpoint: str = DEFAULT_ENDPOINT,
        api_key: str | None = None,
        timeout: float = 5.0,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._endpoint = endpoint.rstrip("/")
        self._api_key = api_key
        self._timeout = timeout
        self._owns_client = client is None
        self._client: httpx.AsyncClient = client or httpx.AsyncClient(timeout=timeout)

    async def route(self, request: RouteRequest) -> RouteDecision:
        url = f"{self._endpoint}/v1/route"
        headers: dict[str, str] = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        body = request.model_dump(mode="json", exclude_none=False)
        try:
            resp = await self._client.post(
                url, json=body, headers=headers, timeout=self._timeout
            )
        except httpx.TimeoutException as e:
            raise RouterTimeoutError(
                f"router request timed out after {self._timeout}s"
            ) from e
        except httpx.HTTPError as e:
            raise RoutingError(f"router transport error: {e}") from e

        if resp.status_code == 200:
            return self._parse_decision(resp)

        # Error path — try to parse the standard error envelope.
        error_code, message = _parse_error(resp)
        status = resp.status_code

        if status in (401, 403) or error_code == "AUTH_FAILED":
            raise RouterAuthError(
                message or "router authentication failed", status_code=status
            )
        if status == 429 or error_code == "QUOTA_EXCEEDED":
            raise RouterQuotaError(
                message or "router quota exceeded", status_code=status
            )
        if status == 408 or error_code == "TIMEOUT":
            raise RouterTimeoutError(
                message or "router timeout", status_code=status
            )
        raise RoutingError(
            message or f"router error (status={status})", status_code=status
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _parse_decision(self, resp: httpx.Response) -> RouteDecision:
        try:
            data = resp.json()
        except ValueError as e:
            raise RoutingError(f"router returned malformed JSON: {e}") from e

        if not isinstance(data, dict):
            raise RoutingError("router response was not a JSON object")

        # Default ``source`` to ``hosted`` if upstream omitted it (the protocol
        # requires it, but be lenient).
        data.setdefault("source", "hosted")
        # Default ``ts`` to now if absent.
        data.setdefault("ts", datetime.now(timezone.utc).isoformat())
        # Normalise ``alternates`` from list-of-list into list-of-tuple — both
        # JSON encodings are accepted; Pydantic handles the coercion.

        try:
            return RouteDecision.model_validate(data)
        except Exception as e:  # pydantic.ValidationError or similar
            raise RoutingError(f"router returned malformed decision: {e}") from e


def _parse_error(resp: httpx.Response) -> tuple[str | None, str | None]:
    """Best-effort parse of the standard error envelope."""
    try:
        data = resp.json()
    except ValueError:
        return None, resp.text or None
    if not isinstance(data, dict):
        return None, None
    code = data.get("error_code")
    message = data.get("message")
    return (
        code if isinstance(code, str) else None,
        message if isinstance(message, str) else None,
    )


__all__ = ["DEFAULT_ENDPOINT", "SemanticRouter"]
