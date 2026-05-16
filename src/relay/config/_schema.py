"""Pydantic schema for the Relay YAML config.

The schema was deliberately chosen over LiteLLM's name-collision-as-grouping
trick — explicit groups are easier to grep, easier to validate, and friendlier
to schema-driven IDE autocomplete. See README for the rationale.
"""

from __future__ import annotations

import ipaddress
import math
import socket
from typing import Annotated, Any, Literal
from urllib.parse import urlsplit

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)

# Hostnames that are unambiguously loopback — treated as the local-development
# escape hatch (no allow_private_hosts opt-in needed for these). Anything else
# that resolves to a private/link-local/CGNAT/cloud-metadata range requires
# explicit opt-in via ``ModelEntry.allow_private_hosts``.
_LOOPBACK_HOSTNAMES = {"localhost", "ip6-localhost", "ip6-loopback"}


def _normalize_ip_literal(host: str) -> ipaddress.IPv4Address | ipaddress.IPv6Address | None:
    """Return ``host`` as an IPv4/IPv6 object using *all* the forms a system
    resolver accepts.

    ``ipaddress.ip_address`` only recognizes the strict dotted-quad form, so
    a YAML author could write ``2852039166`` (decimal-encoded
    169.254.169.254), ``0xa9fea9fe`` (hex), or ``127.1`` (short form) and
    sneak past a private-range check while ``httpx`` / the OS resolver
    still routes the request to the original address. ``socket.inet_aton``
    normalizes all three. Pure-host strings (``api.openai.com``) return
    None — they're not IP literals.
    """
    try:
        return ipaddress.ip_address(host)
    except ValueError:
        pass
    # Heuristic: ipaddress rejected it, but if the host parses as an IPv4
    # literal under inet_aton (decimal / hex / octal / short forms), it is
    # one and we must treat it as such.
    try:
        packed = socket.inet_aton(host)
    except OSError:
        return None
    return ipaddress.IPv4Address(int.from_bytes(packed, "big"))


def _host_is_loopback_literal(host: str) -> bool:
    if host.lower() in _LOOPBACK_HOSTNAMES:
        return True
    ip = _normalize_ip_literal(host)
    return ip is not None and ip.is_loopback


def _host_is_private_ip(host: str) -> bool:
    """True if ``host`` is an IP literal in a non-routable range we should fence.

    Covers RFC1918, loopback, link-local (incl. 169.254.169.254 metadata),
    CGNAT 100.64.0.0/10, ULA fc00::/7, and unspecified addresses. Accepts
    every IPv4 encoding the OS resolver does (dotted-quad, decimal, hex,
    octal, short forms) — see :func:`_normalize_ip_literal`.

    Hostnames that don't parse as IPs return False — DNS-time exfiltration
    is out of scope for a sync field validator (callers can layer DNS
    pinning at the HTTP transport).
    """
    ip = _normalize_ip_literal(host)
    if ip is None:
        return False
    if ip.is_loopback or ip.is_link_local or ip.is_private or ip.is_unspecified:
        return True
    # 100.64.0.0/10 — CGNAT, used by some cloud providers for internal hops.
    cgnat = ipaddress.IPv4Network("100.64.0.0/10")
    return isinstance(ip, ipaddress.IPv4Address) and ip in cgnat


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


# ---------------------------------------------------------------------------
# Credentials — reified objects, never inline strings
# ---------------------------------------------------------------------------


class EnvCredential(_Strict):
    type: Literal["env"] = "env"
    var: str
    """Name of the environment variable holding the secret."""


class LiteralCredential(_Strict):
    """A credential value passed in from code at load time.

    Never put a real secret here when checking config into source control;
    use this to inject keys you've already pulled from a secret manager.
    """

    type: Literal["literal"] = "literal"
    value: str


class AwsProfileCredential(_Strict):
    type: Literal["aws_profile"] = "aws_profile"
    profile: str | None = None
    region: str | None = None


class AwsSecretsCredential(_Strict):
    type: Literal["aws_secrets"] = "aws_secrets"
    arn: str
    region: str | None = None
    field: str | None = None
    """JSON field within the secret if the secret is a JSON blob."""


class GcpAdcCredential(_Strict):
    """GCP Application Default Credentials."""

    type: Literal["gcp_adc"] = "gcp_adc"
    impersonate_service_account: str | None = None


class GcpSecretManagerCredential(_Strict):
    type: Literal["gcp_secret_manager"] = "gcp_secret_manager"
    name: str  # projects/PROJECT_ID/secrets/SECRET_ID/versions/latest


class VaultCredential(_Strict):
    type: Literal["vault"] = "vault"
    path: str
    field: str = "value"


CredentialRef = Annotated[
    EnvCredential
    | LiteralCredential
    | AwsProfileCredential
    | AwsSecretsCredential
    | GcpAdcCredential
    | GcpSecretManagerCredential
    | VaultCredential,
    Field(discriminator="type"),
]


# ---------------------------------------------------------------------------
# Pricing
# ---------------------------------------------------------------------------


class PricingProfile(_Strict):
    """User-supplied multipliers / overrides for negotiated rates.

    Public APIs only expose list price; real enterprise contracts (AWS EDP,
    Azure committed-use, OpenAI custom tiers) need user input.
    """

    description: str | None = None
    input_multiplier: float = 1.0
    output_multiplier: float = 1.0
    cached_input_multiplier: float = 1.0
    """Default multiplier for cached / prompt-cache input tokens."""
    fixed_overrides: dict[str, dict[str, float]] | None = None
    """Per-model absolute overrides keyed by ``provider/model_id``.

    Example::

        fixed_overrides:
          openai/gpt-4o:
            input_per_1m: 1.25
            output_per_1m: 5.00
    """


# ---------------------------------------------------------------------------
# Capabilities
# ---------------------------------------------------------------------------


class Capabilities(_Strict):
    """Capability flags. When omitted, looked up from the built-in catalog."""

    tools: bool | None = None
    vision: bool | None = None
    audio_input: bool | None = None
    audio_output: bool | None = None
    json_mode: bool | None = None
    structured_output: bool | None = None
    streaming: bool | None = None
    thinking: bool | None = None
    context_window: int | None = None
    max_output: int | None = None


# ---------------------------------------------------------------------------
# Model entries
# ---------------------------------------------------------------------------


class ModelEntry(_Strict):
    """A single physical model deployment.

    Aliasing is one-to-one: each entry has a unique key in the ``models`` map.
    To group multiple entries under a single logical name (load balance,
    fallback), define a ``Group``.
    """

    target: str
    """``provider/model_id`` slug. The first ``/`` splits — the rest is provider-specific.

    Examples::

        openai/gpt-4o-2024-11-20
        anthropic/claude-sonnet-4-5-20250929
        bedrock/anthropic.claude-sonnet-4-5-20250929-v1:0
        groq/llama-3.3-70b-versatile
        vertex/gemini-3-flash
    """

    credential: CredentialRef | str | None = None
    """Either a credential object or a ``$env.VAR`` shorthand string."""

    base_url: str | None = None
    """Override the provider's default base URL (e.g. for self-hosted vLLM, custom Azure).

    Validated to keep the provider credential from being shipped to an
    arbitrary internal address or over plaintext HTTP. Use
    ``allow_private_hosts: true`` on the same entry to opt back in for
    self-hosted vLLM / Ollama / k8s service mesh deployments.
    """

    allow_private_hosts: bool = False
    """Permit ``base_url`` to point at a private / link-local / metadata host.

    Off by default. Turn on per-entry when you genuinely want to route this
    model at an internal address (self-hosted vLLM, sidecar, etc.). Loopback
    (``127.0.0.1``, ``::1``, ``localhost``) is always allowed and does not
    require this flag.
    """

    api_version: str | None = None
    """Azure OpenAI API version, etc."""

    region: str | None = None
    """Bedrock / Vertex region."""

    project: str | None = None
    """Vertex AI project id."""

    location: str | None = None
    """Vertex AI location (alternative to region for some products)."""

    deployment: str | None = None
    """Azure OpenAI deployment name."""

    params: dict[str, Any] = Field(default_factory=dict)
    """Default sampling params (temperature, top_p, max_tokens, stop, etc.)."""

    capabilities: Capabilities = Field(default_factory=Capabilities)

    inherit_from: str | None = None
    """Catalog row to copy capabilities/pricing from (use for fine-tunes)."""

    cost: dict[str, float] | None = None
    """Explicit cost override::

        cost:
          input_per_1m: 1.25
          output_per_1m: 5.00
          cached_input_per_1m: 0.625
    """

    pricing_profile: str | None = None
    """Reference a profile defined under top-level ``pricing_profiles``."""

    timeout: float | None = None
    max_retries: int | None = None
    rpm_limit: int | None = None
    tpm_limit: int | None = None

    api_style: Literal["chat_completions", "responses"] | None = None
    """For OpenAI: ``"chat_completions"`` (default) or ``"responses"`` to use the
    new ``/v1/responses`` endpoint with stateful conversations + reasoning items.
    """

    tags: list[str] = Field(default_factory=list)
    """Free-form tags for filtering — e.g. ``["fast", "cheap", "vision"]``."""

    @field_validator("target")
    @classmethod
    def _target_has_slash(cls, v: str) -> str:
        if "/" not in v:
            raise ValueError(
                f"target must be 'provider/model_id', got {v!r}. "
                "Example: 'openai/gpt-4o-2024-11-20'"
            )
        return v

    @model_validator(mode="after")
    def _validate_base_url(self) -> ModelEntry:
        if self.base_url is None:
            return self
        parsed = urlsplit(self.base_url)
        scheme = (parsed.scheme or "").lower()
        if scheme not in ("http", "https"):
            raise ValueError(
                f"base_url scheme must be http or https, got {scheme!r} "
                f"in {self.base_url!r}"
            )
        host = (parsed.hostname or "").lower()
        if not host:
            raise ValueError(f"base_url has no host: {self.base_url!r}")

        is_loopback = _host_is_loopback_literal(host)
        is_private = _host_is_private_ip(host)

        # Plaintext http is permitted only against loopback. Any other host
        # leaks the provider credential across the wire.
        if scheme == "http" and not is_loopback:
            raise ValueError(
                f"base_url uses http:// for non-loopback host {host!r}; "
                "use https:// or change the host to 127.0.0.1/::1/localhost. "
                "Credentials would otherwise traverse the network in clear text."
            )

        # Private / link-local / CGNAT / cloud-metadata hosts are fenced unless
        # the operator explicitly opts in. Loopback is always allowed.
        if is_private and not is_loopback and not self.allow_private_hosts:
            raise ValueError(
                f"base_url {self.base_url!r} resolves to a private/link-local "
                f"host ({host!r}). Set allow_private_hosts: true on this "
                "ModelEntry to opt in (intended for self-hosted vLLM, internal "
                "sidecars, etc.). See docs/security.md."
            )

        # Gemini is uniquely sensitive — the legacy query-string-key form
        # showed up in proxy logs. Even though we now use x-goog-api-key
        # (PR2), keep the http rejection unconditional for google targets.
        if self.target.startswith("google/") and scheme != "https":
            raise ValueError(
                f"google models require https:// for base_url, got {self.base_url!r}"
            )
        return self

    @property
    def provider(self) -> str:
        return self.target.split("/", 1)[0]

    @property
    def model_id(self) -> str:
        return self.target.split("/", 1)[1]


# ---------------------------------------------------------------------------
# Groups
# ---------------------------------------------------------------------------


RoutingStrategy = Literal["fallback", "loadbalance", "weighted", "conditional"]


class GroupMember(_Strict):
    name: str
    """Reference to a key in ``models``."""
    weight: float = 1.0
    """Used by ``weighted`` strategy. Must be finite and non-negative —
    negatives flip ``random.uniform`` bounds and infinities monopolize the
    distribution."""
    when: dict[str, Any] | None = None
    """Used by ``conditional`` — a predicate over the request."""

    @field_validator("weight")
    @classmethod
    def _validate_weight(cls, v: float) -> float:
        if not math.isfinite(v):
            raise ValueError(
                f"GroupMember.weight must be finite, got {v!r}"
            )
        if v < 0:
            raise ValueError(
                f"GroupMember.weight must be non-negative, got {v!r}"
            )
        return v


class GroupSpec(_Strict):
    strategy: RoutingStrategy = "fallback"
    members: list[GroupMember | str]
    """Each entry is either a model alias string (weight=1.0) or a full GroupMember object."""

    retry_classifier: Literal["default", "strict"] | None = None

    @field_validator("members", mode="before")
    @classmethod
    def _normalize_members(cls, v: Any) -> Any:
        if not isinstance(v, list):
            return v
        out: list[Any] = []
        for item in v:
            if isinstance(item, str):
                out.append({"name": item})
            else:
                out.append(item)
        return out


# ---------------------------------------------------------------------------
# Defaults & top-level config
# ---------------------------------------------------------------------------


class GlobalDefaults(_Strict):
    timeout: float = 60.0
    max_retries: int = 2
    retry_initial_backoff: float = 0.5
    retry_max_backoff: float = 30.0
    http2: bool = True
    pool_max_keepalive: int = 50
    pool_max_connections: int = 200
    keepalive_expiry: float = 30.0
    stream_overall_timeout: float = 300.0
    """Wall-clock cap on a single streaming call.

    ``httpx.Timeout`` applies *per read*, so a slow-loris provider that emits
    one SSE byte just inside the per-read deadline can keep a stream alive
    forever. ``_stream_one`` wraps the SSE loop in ``asyncio.timeout`` set to
    this value (or the per-entry ``timeout`` when explicitly set) so the
    request cannot run past it.
    """


class CatalogSettings(_Strict):
    """Controls the built-in model catalog and pricing tier resolver."""

    enabled: bool = True
    fetch_live_pricing: bool = True
    """Hit AWS Pricing API, Azure Retail Prices API, OpenRouter live models."""
    refresh_interval_hours: float = 6.0
    """How often live pricing endpoints are re-queried within a process."""
    offline: bool = False
    """If True, never call out for live pricing — use shipped snapshot only."""


class ObservabilitySettings(_Strict):
    otel_enabled: bool = False
    capture_messages: Literal["never", "metadata_only", "full"] = "metadata_only"
    """Whether to put message contents on OTel spans. Sensitive — defaults to off."""
    log_level: Literal["debug", "info", "warning", "error"] = "info"


class RelayConfig(_Strict):
    """Top-level Relay configuration.

    Exactly one of these is loaded per :class:`Hub` instance. Build it from a
    YAML file via :func:`relay.config.load`.
    """

    version: Literal[1] = 1

    defaults: GlobalDefaults = Field(default_factory=GlobalDefaults)
    catalog: CatalogSettings = Field(default_factory=CatalogSettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)

    pricing_profiles: dict[str, PricingProfile] = Field(default_factory=dict)
    models: dict[str, ModelEntry] = Field(default_factory=dict)
    groups: dict[str, GroupSpec] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _validate_references(self) -> RelayConfig:
        # Group members must reference real model aliases.
        for group_name, group in self.groups.items():
            for member in group.members:
                assert isinstance(member, GroupMember), "members must be normalized to GroupMember"
                if member.name not in self.models and member.name not in self.groups:
                    raise ValueError(
                        f"group {group_name!r} references unknown alias "
                        f"{member.name!r} — must be a key in 'models' or 'groups'"
                    )

        # No alias collisions between models and groups.
        overlap = set(self.models) & set(self.groups)
        if overlap:
            raise ValueError(
                f"alias(es) {sorted(overlap)} appear in both 'models' and 'groups'; "
                "each alias must be unique"
            )

        # Pricing profile references must resolve.
        valid_profiles = set(self.pricing_profiles)
        for alias, m in self.models.items():
            if m.pricing_profile and m.pricing_profile not in valid_profiles:
                raise ValueError(
                    f"model {alias!r} references unknown pricing_profile {m.pricing_profile!r}"
                )

        # Group-graph cycle detection. Routing recurses into nested groups, so
        # ``A -> B -> A`` would otherwise blow the recursion limit at request
        # time. DFS with a path stack so the error names the actual cycle.
        self._detect_group_cycles()
        return self

    def _detect_group_cycles(self) -> None:
        visiting: set[str] = set()
        visited: set[str] = set()

        def walk(name: str, path: list[str]) -> None:
            if name in visited:
                return
            if name in visiting:
                cycle = [*path[path.index(name) :], name]
                raise ValueError(
                    "group cycle detected: " + " -> ".join(cycle)
                )
            visiting.add(name)
            path.append(name)
            group = self.groups.get(name)
            if group is not None:
                for member in group.members:
                    assert isinstance(member, GroupMember)
                    # Only descend into nested groups; model-leaf references
                    # can't cycle.
                    if member.name in self.groups:
                        walk(member.name, path)
            path.pop()
            visiting.discard(name)
            visited.add(name)

        for group_name in self.groups:
            walk(group_name, [])


def json_schema() -> dict[str, Any]:
    """Return the JSON Schema for :class:`RelayConfig`.

    Publish this at a stable URL and reference it from your YAML's first line::

        # yaml-language-server: $schema=https://relay.ai5labs.com/schema/v1.json

    The Red Hat YAML extension for VS Code will then give you autocomplete
    and inline validation while editing.
    """
    return RelayConfig.model_json_schema()
