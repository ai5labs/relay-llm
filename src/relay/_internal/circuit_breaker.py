"""Circuit breakers for cross-provider failover.

Standard three-state breaker:

* **Closed** — requests pass through. Track failures in a sliding window.
* **Open** — requests fail-fast (raise ``CircuitOpenError``) for ``cooldown_s``
  after failure threshold is exceeded. Other group members get tried instead.
* **Half-open** — one probe request is allowed; success closes, failure re-opens.

Per-target state is keyed by ``(provider, model_id)`` so a misbehaving Bedrock
deployment doesn't trip the OpenAI breaker.
"""

from __future__ import annotations

import asyncio
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum

from relay.errors import RelayError


class CircuitOpenError(RelayError):
    """Circuit is open — request was not attempted."""


class _State(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class _BreakerState:
    state: _State = _State.CLOSED
    failures: deque[float] = field(default_factory=deque)
    opened_at: float = 0.0
    half_open_in_flight: bool = False


class CircuitBreaker:
    """Per-target circuit breaker.

    Configurable: ``failure_threshold`` failures within ``window_s`` open the
    circuit; it stays open for ``cooldown_s``, then admits one probe.
    """

    def __init__(
        self,
        *,
        failure_threshold: int = 5,
        window_s: float = 30.0,
        cooldown_s: float = 60.0,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.window_s = window_s
        self.cooldown_s = cooldown_s
        self._states: dict[str, _BreakerState] = {}
        self._lock = asyncio.Lock()

    async def before(self, key: str) -> None:
        """Raise ``CircuitOpenError`` if the circuit for ``key`` is open."""
        now = time.monotonic()
        async with self._lock:
            st = self._states.get(key)
            if st is None:
                self._states[key] = _BreakerState()
                return
            if st.state == _State.OPEN:
                if now - st.opened_at >= self.cooldown_s:
                    st.state = _State.HALF_OPEN
                    st.half_open_in_flight = True
                    return
                raise CircuitOpenError(f"circuit open for {key!r} (cooldown {self.cooldown_s}s)")
            if st.state == _State.HALF_OPEN:
                if st.half_open_in_flight:
                    raise CircuitOpenError(f"circuit half-open probe in flight for {key!r}")
                st.half_open_in_flight = True

    async def on_success(self, key: str) -> None:
        async with self._lock:
            st = self._states.setdefault(key, _BreakerState())
            st.failures.clear()
            st.state = _State.CLOSED
            st.half_open_in_flight = False

    async def on_failure(self, key: str) -> None:
        now = time.monotonic()
        async with self._lock:
            st = self._states.setdefault(key, _BreakerState())
            st.half_open_in_flight = False
            if st.state == _State.HALF_OPEN:
                st.state = _State.OPEN
                st.opened_at = now
                return
            st.failures.append(now)
            cutoff = now - self.window_s
            while st.failures and st.failures[0] < cutoff:
                st.failures.popleft()
            if len(st.failures) >= self.failure_threshold:
                st.state = _State.OPEN
                st.opened_at = now

    def state_for(self, key: str) -> str:
        """For diagnostics."""
        st = self._states.get(key)
        if st is None:
            return "closed"
        return st.state.value
