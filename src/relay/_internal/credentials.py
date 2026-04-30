"""Credential resolution.

A :class:`CredentialRef` is a declarative pointer; this module turns it into an
actual secret string at the time it's needed. Lazy by design — we don't hit
secret managers at config-load time, only on first use.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from relay.errors import ConfigError

if TYPE_CHECKING:
    from relay.config._schema import (
        AwsSecretsCredential,
        CredentialRef,
        EnvCredential,
        GcpSecretManagerCredential,
        LiteralCredential,
        VaultCredential,
    )


async def resolve_secret(ref: CredentialRef | str) -> str:
    """Resolve a credential reference to its concrete secret string.

    For ``$env.VAR`` shorthand strings we resolve in-place; for typed references we
    dispatch on ``type``. Unsupported types raise ``ConfigError``.
    """
    if isinstance(ref, str):
        return _resolve_env_shorthand(ref)

    type_ = ref.type
    if type_ == "env":
        return _resolve_env(ref)  # type: ignore[arg-type]
    if type_ == "literal":
        return _resolve_literal(ref)  # type: ignore[arg-type]
    if type_ == "aws_secrets":
        return await _resolve_aws_secret(ref)  # type: ignore[arg-type]
    if type_ == "gcp_secret_manager":
        return await _resolve_gcp_secret(ref)  # type: ignore[arg-type]
    if type_ == "vault":
        return await _resolve_vault(ref)  # type: ignore[arg-type]
    if type_ in ("aws_profile", "gcp_adc"):
        # These don't produce a single secret string — they're handled by the
        # provider adapter directly (which uses boto3 / google-auth and signs
        # requests itself).
        raise ConfigError(
            f"credential type {type_!r} does not resolve to a string. "
            f"It is consumed directly by the bedrock/vertex adapters."
        )
    raise ConfigError(f"unknown credential type: {type_!r}")


def _resolve_env_shorthand(ref: str) -> str:
    if ref.startswith("$env."):
        var = ref[5:]
    elif ref.startswith("${env:") and ref.endswith("}"):
        var = ref[6:-1]
    else:
        # Plain string — assume it's already a secret. Discouraged but allowed.
        return ref
    val = os.environ.get(var)
    if val is None:
        raise ConfigError(f"environment variable {var!r} not set")
    if val == "":
        raise ConfigError(f"environment variable {var!r} is empty")
    return val


def _resolve_env(ref: EnvCredential) -> str:
    val = os.environ.get(ref.var)
    if val is None:
        raise ConfigError(f"environment variable {ref.var!r} not set")
    if val == "":
        raise ConfigError(f"environment variable {ref.var!r} is empty")
    return val


def _resolve_literal(ref: LiteralCredential) -> str:
    return ref.value


async def _resolve_aws_secret(ref: AwsSecretsCredential) -> str:
    try:
        import boto3  # type: ignore[import-not-found,import-untyped]
    except ImportError as e:
        raise ConfigError(
            "credential type 'aws_secrets' requires 'boto3'. "
            "Install with: pip install relayllm[aws]"
        ) from e
    client = boto3.client("secretsmanager", region_name=ref.region)
    resp = client.get_secret_value(SecretId=ref.arn)
    secret = resp.get("SecretString")
    if not secret:
        raise ConfigError(f"AWS secret {ref.arn!r} has no SecretString")
    if ref.field:
        import json

        try:
            blob = json.loads(secret)
        except json.JSONDecodeError as e:
            raise ConfigError(
                f"AWS secret {ref.arn!r} field={ref.field!r} requires JSON content"
            ) from e
        val = blob.get(ref.field)
        if val is None:
            raise ConfigError(f"AWS secret {ref.arn!r} has no field {ref.field!r}")
        return str(val)
    return secret


async def _resolve_gcp_secret(ref: GcpSecretManagerCredential) -> str:
    try:
        from google.cloud import secretmanager  # type: ignore[import-untyped,import-not-found]
    except ImportError as e:
        raise ConfigError(
            "credential type 'gcp_secret_manager' requires 'google-cloud-secret-manager'. "
            "Install with: pip install google-cloud-secret-manager"
        ) from e
    client = secretmanager.SecretManagerServiceClient()
    response = client.access_secret_version(request={"name": ref.name})
    return response.payload.data.decode("utf-8")


async def _resolve_vault(ref: VaultCredential) -> str:
    raise ConfigError(
        "Vault credentials are not yet implemented in v0.1. "
        "Pin a feature request: https://github.com/ai5labs/relay-llm/issues"
    )
