# PII redaction

Relay rewrites messages before they leave the process, replacing PII with placeholders.

```python
from relay import Hub
from relay.redaction import RegexRedactor

hub = Hub.from_yaml("models.yaml", redactor=RegexRedactor())
```

By default, six categories are redacted:

| Kind | Pattern |
|---|---|
| `email` | RFC 5322-ish |
| `ssn` | `XXX-XX-XXXX` |
| `credit_card` | 13–19 digits w/ optional separators |
| `phone_us` | NANP format |
| `ipv4` | dotted quad |
| `iban` | International bank account |

Matches are replaced with `[REDACTED:{kind}]`. The model never sees the original.

## Custom patterns

```python
import re
redactor = RegexRedactor(
    patterns={
        "customer_id": re.compile(r"CUST-\d{6}"),
        "internal_url": re.compile(r"https://acme\.internal/[\w/]+"),
    },
    inherit_defaults=True,    # keep email/ssn/etc. on top
)
```

## Order of operations

```
caller messages
    ↓
redaction       ← original is dropped here
    ↓
guardrails (pre)
    ↓
HTTP cache lookup
    ↓
provider call
    ↓
guardrails (post)
    ↓
audit log emit  ← logs the redacted version, never the original
    ↓
caller response
```

The original prompt is never logged or persisted by Relay. If you need reversible tokenization (so support can re-hydrate redacted strings later), that's planned for v0.3.

## Presidio integration

For ML-based detection (NER for names, addresses, etc.), build a `Redactor` against [Microsoft Presidio](https://microsoft.github.io/presidio/):

```python
from presidio_analyzer import AnalyzerEngine
from relay.redaction import Redactor, RedactionResult

class PresidioRedactor:
    def __init__(self):
        self.analyzer = AnalyzerEngine()

    def redact(self, messages):
        # ... call self.analyzer.analyze() on each message
        return RedactionResult(messages=..., redactions=..., matched_kinds=...)
```

Then `Hub.from_yaml(..., redactor=PresidioRedactor())`.
