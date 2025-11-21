"""Factory for creating LLM providers."""

import os
from typing import Any, Dict

from src.generation.llm_providers import AnthropicProvider, OllamaProvider, OpenAIProvider
from src.utils.interfaces import LLMProvider


class LLMProviderFactory:
    """Factory for creating LLM providers based on configuration."""

    @staticmethod
    def create(provider: str, config: Dict[str, Any]) -> LLMProvider:
        """
        Create an LLM provider.

        Args:
            provider: Provider type (ollama, openai, anthropic)
            config: Provider configuration

        Returns:
            LLMProvider instance

        Raises:
            ValueError: If provider type is unknown
        """
        if provider == "ollama":
            return OllamaProvider(
                model=config.get("model", "llama3"),
                base_url=config.get("base_url", "http://localhost:11434"),
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 2000),
                top_p=config.get("top_p"),
            )

        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable required for OpenAI provider"
                )

            return OpenAIProvider(
                api_key=api_key,
                model=config.get("model", "gpt-3.5-turbo"),
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 2000),
            )

        elif provider == "anthropic":
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable required for Anthropic provider"
                )

            return AnthropicProvider(
                api_key=api_key,
                model=config.get("model", "claude-3-5-sonnet-20241022"),
                temperature=config.get("temperature", 0.7),
                max_tokens=config.get("max_tokens", 2000),
            )

        else:
            raise ValueError(
                f"Unknown LLM provider: {provider}. "
                f"Supported: ollama, openai, anthropic"
            )
