"""Guardrails — pre-call and post-call hooks.

A :class:`Guardrail` inspects (or rewrites) messages and responses. Each
guardrail returns either ``None`` (allow) or a :class:`GuardrailViolation`
(block + reason). Multiple guardrails compose; first violation wins.

Reference implementations
-------------------------
* :class:`MaxInputLength` — reject prompts past N characters / tokens.
* :class:`BlockedKeywords` — reject prompts/responses containing forbidden terms.
* Pluggable: third-party guardrails (Lakera Guard, Protect AI Rebuff, NeMo
  Guardrails, AWS Bedrock Guardrails) become 50-line plugins against the
  Protocol.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal, Protocol, runtime_checkable

from relay.errors import RelayError
from relay.types import ChatResponse, Message

GuardrailStage = Literal["pre", "post"]


@dataclass(frozen=True, slots=True)
class GuardrailViolation:
    """Returned by a guardrail to block a call."""

    rule: str
    """Stable id of the rule that fired (``"max_input_length"``)."""
    stage: GuardrailStage
    message: str
    severity: Literal["info", "warn", "block"] = "block"


class GuardrailError(RelayError):
    """Raised when a blocking guardrail fires."""

    def __init__(self, violation: GuardrailViolation) -> None:
        super().__init__(violation.message)
        self.violation = violation


@runtime_checkable
class Guardrail(Protocol):
    name: str
    stage: GuardrailStage

    def check_pre(self, messages: list[Message]) -> GuardrailViolation | None: ...
    def check_post(self, response: ChatResponse) -> GuardrailViolation | None: ...


class _BaseGuardrail:
    """No-op defaults; subclasses override the stage they care about."""

    name: str = "guardrail"
    stage: GuardrailStage = "pre"

    def check_pre(self, messages: list[Message]) -> GuardrailViolation | None:
        return None

    def check_post(self, response: ChatResponse) -> GuardrailViolation | None:
        return None


class MaxInputLength(_BaseGuardrail):
    """Reject when total prompt characters exceed ``max_chars``."""

    name = "max_input_length"
    stage: GuardrailStage = "pre"

    def __init__(self, max_chars: int) -> None:
        self.max_chars = max_chars

    def check_pre(self, messages: list[Message]) -> GuardrailViolation | None:
        total = 0
        for m in messages:
            if isinstance(m.content, str):
                total += len(m.content)
            else:
                for block in m.content:
                    text = getattr(block, "text", "")
                    if isinstance(text, str):
                        total += len(text)
        if total > self.max_chars:
            return GuardrailViolation(
                rule=self.name,
                stage="pre",
                message=f"prompt is {total} chars (max {self.max_chars})",
            )
        return None


class BlockedKeywords(_BaseGuardrail):
    """Reject prompts (and optionally responses) matching any banned word/regex."""

    name = "blocked_keywords"

    def __init__(
        self,
        terms: list[str | re.Pattern[str]],
        *,
        check_response: bool = True,
        case_insensitive: bool = True,
    ) -> None:
        self._patterns: list[re.Pattern[str]] = []
        flags = re.IGNORECASE if case_insensitive else 0
        for t in terms:
            if isinstance(t, re.Pattern):
                self._patterns.append(t)
            else:
                self._patterns.append(re.compile(re.escape(t), flags))
        self._check_response = check_response
        self.stage = "post" if check_response else "pre"

    def check_pre(self, messages: list[Message]) -> GuardrailViolation | None:
        for m in messages:
            text = (
                m.content
                if isinstance(m.content, str)
                else " ".join(getattr(b, "text", "") for b in m.content if hasattr(b, "text"))
            )
            for p in self._patterns:
                if p.search(text):
                    return GuardrailViolation(
                        rule=self.name,
                        stage="pre",
                        message=f"prompt contains blocked term matching {p.pattern!r}",
                    )
        return None

    def check_post(self, response: ChatResponse) -> GuardrailViolation | None:
        if not self._check_response:
            return None
        text = response.text
        for p in self._patterns:
            if p.search(text):
                return GuardrailViolation(
                    rule=self.name,
                    stage="post",
                    message=f"response contains blocked term matching {p.pattern!r}",
                )
        return None


def evaluate_pre(guardrails: list[Guardrail], messages: list[Message]) -> GuardrailViolation | None:
    """Run all pre-call guardrails. First violation returned."""
    for g in guardrails:
        v = g.check_pre(messages)
        if v is not None and v.severity == "block":
            return v
    return None


def evaluate_post(guardrails: list[Guardrail], response: ChatResponse) -> GuardrailViolation | None:
    """Run all post-call guardrails. First violation returned."""
    for g in guardrails:
        v = g.check_post(response)
        if v is not None and v.severity == "block":
            return v
    return None


__all__ = [
    "BlockedKeywords",
    "Guardrail",
    "GuardrailError",
    "GuardrailStage",
    "GuardrailViolation",
    "MaxInputLength",
    "evaluate_post",
    "evaluate_pre",
]
