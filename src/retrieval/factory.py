"""Factory for creating vector stores."""

import os
from typing import Any, Dict

from src.retrieval.vector_stores import ChromaVectorStore, QdrantVectorStore
from src.utils.interfaces import VectorDatabase


class VectorStoreFactory:
    """Factory for creating vector stores based on configuration."""

    @staticmethod
    def create(provider: str, config: Dict[str, Any]) -> VectorDatabase:
        """
        Create a vector store.

        Args:
            provider: Provider type (qdrant, chroma, pinecone)
            config: Provider configuration

        Returns:
            VectorDatabase instance

        Raises:
            ValueError: If provider type is unknown
        """
        if provider == "qdrant":
            return QdrantVectorStore(
                host=config.get("host", "localhost"),
                port=config.get("port", 6333),
                api_key=os.getenv("QDRANT_API_KEY") or config.get("api_key"),
            )

        elif provider == "chroma":
            return ChromaVectorStore(
                persist_directory=config.get("persist_directory", "./data/vectors/chroma"),
            )

        else:
            raise ValueError(
                f"Unknown vector store provider: {provider}. "
                f"Supported: qdrant, chroma"
            )
