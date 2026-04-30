"""YAML config schema and loader.

Public entry points:

* :func:`load` — parse a YAML file (or string) into a :class:`RelayConfig`.
* :func:`json_schema` — return the JSON Schema for the config; used by the CLI
  to publish a schema URL that the VS Code YAML extension can consume.
"""

from relay.config._loader import load, load_str
from relay.config._schema import (
    Capabilities,
    CredentialRef,
    GroupSpec,
    ModelEntry,
    PricingProfile,
    RelayConfig,
    RoutingStrategy,
    json_schema,
)

__all__ = [
    "Capabilities",
    "CredentialRef",
    "GroupSpec",
    "ModelEntry",
    "PricingProfile",
    "RelayConfig",
    "RoutingStrategy",
    "json_schema",
    "load",
    "load_str",
]
