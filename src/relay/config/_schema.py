"""Pydantic schema for the Relay YAML config.

The schema was deliberately chosen over LiteLLM's name-collision-as-grouping
trick — explicit groups are easier to grep, easier to validate, and friendlier
to schema-driven IDE autocomplete. See README for the rationale.
"""

from __future__ import annotations

from typing import Annotated, Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    field_validator,
    model_validator,
)


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
    """Override the provider's default base URL (e.g. for self-hosted vLLM, custom Azure)."""

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
    """Used by ``weighted`` strategy."""
    when: dict[str, Any] | None = None
    """Used by ``conditional`` — a predicate over the request."""


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
        return self


def json_schema() -> dict[str, Any]:
    """Return the JSON Schema for :class:`RelayConfig`.

    Publish this at a stable URL and reference it from your YAML's first line::

        # yaml-language-server: $schema=https://relay.ai5labs.com/schema/v1.json

    The Red Hat YAML extension for VS Code will then give you autocomplete
    and inline validation while editing.
    """
    return RelayConfig.model_json_schema()
