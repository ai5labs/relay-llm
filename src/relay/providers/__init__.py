"""Central provider registry.

This is the **one place** where every provider adapter is registered. Adding
support for a new provider = one file + one entry here.

The registry maps a stable provider id (``"openai"``, ``"anthropic"``, ...) to a
zero-arg factory that returns a :class:`Provider` instance. Factories are called
lazily on first use so adapters that import heavy SDKs (``boto3``,
``google-genai``) don't slow startup.
"""

from __future__ import annotations

from collections.abc import Callable

from relay.errors import ConfigError
from relay.providers._base import BaseProvider, Provider

ProviderFactory = Callable[[], Provider]


def _make_openai() -> Provider:
    from relay.providers.openai_compat import OpenAICompatibleProvider

    return OpenAICompatibleProvider(name="openai", default_base_url="https://api.openai.com/v1")


def _make_groq() -> Provider:
    from relay.providers.openai_compat import OpenAICompatibleProvider

    return OpenAICompatibleProvider(name="groq", default_base_url="https://api.groq.com/openai/v1")


def _make_together() -> Provider:
    from relay.providers.openai_compat import OpenAICompatibleProvider

    return OpenAICompatibleProvider(name="together", default_base_url="https://api.together.xyz/v1")


def _make_deepseek() -> Provider:
    from relay.providers.openai_compat import OpenAICompatibleProvider

    return OpenAICompatibleProvider(name="deepseek", default_base_url="https://api.deepseek.com/v1")


def _make_xai() -> Provider:
    from relay.providers.openai_compat import OpenAICompatibleProvider

    return OpenAICompatibleProvider(name="xai", default_base_url="https://api.x.ai/v1")


def _make_mistral() -> Provider:
    from relay.providers.openai_compat import OpenAICompatibleProvider

    return OpenAICompatibleProvider(name="mistral", default_base_url="https://api.mistral.ai/v1")


def _make_fireworks() -> Provider:
    from relay.providers.openai_compat import OpenAICompatibleProvider

    return OpenAICompatibleProvider(
        name="fireworks", default_base_url="https://api.fireworks.ai/inference/v1"
    )


def _make_perplexity() -> Provider:
    from relay.providers.openai_compat import OpenAICompatibleProvider

    return OpenAICompatibleProvider(name="perplexity", default_base_url="https://api.perplexity.ai")


def _make_ollama() -> Provider:
    from relay.providers.openai_compat import OpenAICompatibleProvider

    return OpenAICompatibleProvider(
        name="ollama", default_base_url="http://localhost:11434/v1", api_key_required=False
    )


def _make_vllm() -> Provider:
    from relay.providers.openai_compat import OpenAICompatibleProvider

    return OpenAICompatibleProvider(
        name="vllm", default_base_url="http://localhost:8000/v1", api_key_required=False
    )


def _make_lmstudio() -> Provider:
    from relay.providers.openai_compat import OpenAICompatibleProvider

    return OpenAICompatibleProvider(
        name="lmstudio", default_base_url="http://localhost:1234/v1", api_key_required=False
    )


def _make_openrouter() -> Provider:
    from relay.providers.openai_compat import OpenAICompatibleProvider

    return OpenAICompatibleProvider(
        name="openrouter", default_base_url="https://openrouter.ai/api/v1"
    )


def _make_anthropic() -> Provider:
    from relay.providers.anthropic import AnthropicProvider

    return AnthropicProvider()


def _make_azure() -> Provider:
    from relay.providers.azure_openai import AzureOpenAIProvider

    return AzureOpenAIProvider()


def _make_google() -> Provider:
    from relay.providers.google import GoogleProvider

    return GoogleProvider()


def _make_vertex() -> Provider:
    from relay.providers.vertex import VertexProvider

    return VertexProvider()


def _make_bedrock() -> Provider:
    from relay.providers.bedrock import BedrockProvider

    return BedrockProvider()


def _make_cohere() -> Provider:
    from relay.providers.cohere import CohereProvider

    return CohereProvider()


PROVIDER_REGISTRY: dict[str, ProviderFactory] = {
    # OpenAI-compatible — one adapter, many providers.
    "openai": _make_openai,
    "groq": _make_groq,
    "together": _make_together,
    "deepseek": _make_deepseek,
    "xai": _make_xai,
    "mistral": _make_mistral,
    "fireworks": _make_fireworks,
    "perplexity": _make_perplexity,
    "ollama": _make_ollama,
    "vllm": _make_vllm,
    "lmstudio": _make_lmstudio,
    "openrouter": _make_openrouter,
    # Native adapters (the ones with non-OpenAI wire formats).
    "anthropic": _make_anthropic,
    "azure": _make_azure,
    "google": _make_google,
    "vertex": _make_vertex,
    "bedrock": _make_bedrock,
    "cohere": _make_cohere,
}


def supported_providers() -> list[str]:
    return sorted(PROVIDER_REGISTRY.keys())


def make_provider(name: str) -> Provider:
    """Instantiate the provider adapter for ``name``. Raises if unknown."""
    factory = PROVIDER_REGISTRY.get(name)
    if factory is None:
        raise ConfigError(
            f"unsupported provider: {name!r}. Supported: {', '.join(supported_providers())}"
        )
    return factory()


__all__ = [
    "PROVIDER_REGISTRY",
    "BaseProvider",
    "Provider",
    "ProviderFactory",
    "make_provider",
    "supported_providers",
]
