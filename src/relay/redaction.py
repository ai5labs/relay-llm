"""PII redaction pipeline.

A :class:`Redactor` rewrites a list of messages before they leave the process.
We ship a regex-based reference implementation and an optional Microsoft
Presidio adapter (when ``presidio-analyzer`` is installed). LLM-judge redaction
is planned for v0.3.

Common patterns
---------------
* **Reversible tokenization** is *not* implemented in v0.2 — we use redaction
  (replace with a placeholder). Reversibility requires a stable cipher; v0.3.
* **Hooks happen before audit logging**, so redacted prompts are what gets
  logged. The original is never persisted by Relay.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from relay.types import Message, TextBlock


@dataclass(frozen=True, slots=True)
class RedactionResult:
    """Outcome of a redaction pass."""

    messages: list[Message]
    """The redacted messages — what the model sees."""
    redactions: int
    """Total number of replacements made."""
    matched_kinds: tuple[str, ...] = ()
    """Distinct PII kinds detected (``"email"``, ``"ssn"``, ...)."""


@runtime_checkable
class Redactor(Protocol):
    """Pluggable redactor.

    Implementations should be deterministic so log entries match what the model
    saw.
    """

    def redact(self, messages: list[Message]) -> RedactionResult: ...


# ---------------------------------------------------------------------------
# Regex-based default
# ---------------------------------------------------------------------------


# Conservative defaults: leans toward false-negatives over false-positives. Users
# extend with their own regex catalog when they need stricter coverage.
_DEFAULT_PATTERNS: dict[str, re.Pattern[str]] = {
    "email": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"),
    "ssn": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    "credit_card": re.compile(r"\b(?:\d[ -]*?){13,19}\b"),
    "phone_us": re.compile(r"\b(?:\+?1[-.\s]?)?\(?[2-9]\d{2}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b"),
    "ipv4": re.compile(r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"),
    "iban": re.compile(r"\b[A-Z]{2}\d{2}[A-Z0-9]{11,30}\b"),
}


class RegexRedactor:
    """Default redactor — fast, dependency-free, conservative.

    Replaces matches with ``[REDACTED:{kind}]``. Kinds detected by default:
    email, ssn, credit_card, phone_us, ipv4, iban.

    Customize::

        redactor = RegexRedactor(
            patterns={"customer_id": re.compile(r"CUST-\\d{6}")},
            inherit_defaults=True,
        )
    """

    def __init__(
        self,
        *,
        patterns: dict[str, re.Pattern[str]] | None = None,
        inherit_defaults: bool = True,
    ) -> None:
        self._patterns: dict[str, re.Pattern[str]] = {}
        if inherit_defaults:
            self._patterns.update(_DEFAULT_PATTERNS)
        if patterns:
            self._patterns.update(patterns)

    def redact(self, messages: list[Message]) -> RedactionResult:
        total = 0
        kinds: set[str] = set()
        out: list[Message] = []
        for m in messages:
            new_content, n, k = self._scrub(m.content)
            total += n
            kinds.update(k)
            out.append(m.model_copy(update={"content": new_content}))
        return RedactionResult(messages=out, redactions=total, matched_kinds=tuple(sorted(kinds)))

    def _scrub(self, content: object) -> tuple[object, int, set[str]]:
        if isinstance(content, str):
            scrubbed, n, kinds_set = self._scrub_text(content)
            return scrubbed, n, kinds_set
        if isinstance(content, list):
            new_blocks: list[object] = []
            total = 0
            list_kinds: set[str] = set()
            for block in content:
                if isinstance(block, TextBlock):
                    text, n, k = self._scrub_text(block.text)
                    new_blocks.append(block.model_copy(update={"text": text}))
                    total += n
                    list_kinds.update(k)
                else:
                    new_blocks.append(block)
            return new_blocks, total, list_kinds
        return content, 0, set()

    def _scrub_text(self, text: str) -> tuple[str, int, set[str]]:
        n_total = 0
        kinds: set[str] = set()
        for kind, pattern in self._patterns.items():
            new_text, n = pattern.subn(f"[REDACTED:{kind}]", text)
            if n > 0:
                kinds.add(kind)
            n_total += n
            text = new_text
        return text, n_total, kinds


__all__ = ["RedactionResult", "Redactor", "RegexRedactor"]
