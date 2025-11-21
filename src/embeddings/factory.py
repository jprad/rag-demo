"""Factory for creating embedding providers."""

import os
from typing import Any, Dict

from src.embeddings.providers import OpenAIEmbeddingProvider, SentenceTransformerProvider
from src.utils.interfaces import EmbeddingProvider


class EmbeddingProviderFactory:
    """Factory for creating embedding providers based on configuration."""

    @staticmethod
    def create(provider: str, config: Dict[str, Any]) -> EmbeddingProvider:
        """
        Create an embedding provider.

        Args:
            provider: Provider type (sentence-transformers, openai)
            config: Provider configuration

        Returns:
            EmbeddingProvider instance

        Raises:
            ValueError: If provider type is unknown
        """
        if provider == "sentence-transformers":
            return SentenceTransformerProvider(
                model_name=config.get("model_name", "all-MiniLM-L6-v2"),
                device=config.get("device"),
                normalize_embeddings=config.get("normalize_embeddings", True),
            )

        elif provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable required for OpenAI provider"
                )

            return OpenAIEmbeddingProvider(
                api_key=api_key,
                model=config.get("model", "text-embedding-3-small"),
            )

        else:
            raise ValueError(
                f"Unknown embedding provider: {provider}. "
                f"Supported: sentence-transformers, openai"
            )
