"""YAML loader with env-var resolution and layered overrides.

Loading rules
-------------
1. Read the YAML file(s).
2. Resolve ``$env.VAR`` and ``${env:VAR}`` strings to env values. Missing vars
   raise ``ConfigError`` at load time — never silently coerce to empty string,
   which is one of LiteLLM's pain points.
3. Validate against the Pydantic schema.
4. Return an immutable ``RelayConfig``.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

import yaml

from relay.config._schema import RelayConfig
from relay.errors import ConfigError

_ENV_PATTERN = re.compile(r"^\$(?:env\.|\{env:)([A-Z_][A-Z0-9_]*)\}?$")


def _resolve_env(value: Any, *, path: str = "") -> Any:
    """Walk a parsed YAML structure and substitute ``$env.VAR`` strings.

    Failure modes (all raise ``ConfigError``):
    * referenced env var is unset
    * referenced env var is empty
    """
    if isinstance(value, str):
        m = _ENV_PATTERN.match(value.strip())
        if m:
            var = m.group(1)
            resolved = os.environ.get(var)
            if resolved is None:
                raise ConfigError(
                    f"environment variable {var!r} (referenced at {path or '<root>'}) is not set"
                )
            if resolved == "":
                raise ConfigError(
                    f"environment variable {var!r} (referenced at {path or '<root>'}) is empty"
                )
            return resolved
        return value
    if isinstance(value, dict):
        return {k: _resolve_env(v, path=f"{path}.{k}" if path else k) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_env(v, path=f"{path}[{i}]") for i, v in enumerate(value)]
    return value


def _deep_merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Merge two parsed dicts. Lists are replaced (not concatenated)."""
    out = dict(base)
    for k, v in overlay.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load(*paths: str | Path) -> RelayConfig:
    """Load and validate one or more YAML files into a :class:`RelayConfig`.

    When multiple paths are given, later files override earlier ones (deep-merged
    for dicts, replaced for lists). Useful for ``base.yaml`` + ``prod.yaml``.

    :raises ConfigError: when a file is missing, invalid YAML, references an
        unset env var, or fails schema validation.
    """
    if not paths:
        raise ConfigError("relay.config.load() requires at least one path")

    merged: dict[str, Any] = {}
    for p in paths:
        path = Path(p)
        if not path.is_file():
            raise ConfigError(f"config file not found: {path}")
        try:
            with path.open("r", encoding="utf-8") as f:
                parsed = yaml.safe_load(f) or {}
        except yaml.YAMLError as e:
            raise ConfigError(f"invalid YAML in {path}: {e}") from e
        if not isinstance(parsed, dict):
            raise ConfigError(f"top-level of {path} must be a mapping")
        merged = _deep_merge(merged, parsed)

    resolved = _resolve_env(merged)

    try:
        return RelayConfig.model_validate(resolved)
    except Exception as e:
        raise ConfigError(f"config validation failed: {e}") from e


def load_str(yaml_text: str) -> RelayConfig:
    """Like :func:`load` but takes a YAML string. Useful in tests."""
    try:
        parsed = yaml.safe_load(yaml_text) or {}
    except yaml.YAMLError as e:
        raise ConfigError(f"invalid YAML: {e}") from e
    if not isinstance(parsed, dict):
        raise ConfigError("top-level YAML must be a mapping")
    resolved = _resolve_env(parsed)
    try:
        return RelayConfig.model_validate(resolved)
    except Exception as e:
        raise ConfigError(f"config validation failed: {e}") from e
