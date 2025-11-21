"""Embedding provider implementations."""

from typing import List, Optional

import torch
from sentence_transformers import SentenceTransformer

from src.utils.interfaces import EmbeddingProvider


class SentenceTransformerProvider(EmbeddingProvider):
    """Embedding provider using sentence-transformers."""

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        normalize_embeddings: bool = True,
    ):
        """
        Initialize SentenceTransformer embedding provider.

        Args:
            model_name: Name of the sentence-transformers model
            device: Device to run on (cpu, cuda, mps). Auto-detects if None
            normalize_embeddings: Whether to normalize embedding vectors
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.model_name = model_name
        self.device = device
        self.normalize_embeddings = normalize_embeddings
        self.model = SentenceTransformer(model_name, device=device)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of text documents to embed

        Returns:
            List of embedding vectors
        """
        embeddings = self.model.encode(
            texts,
            normalize_embeddings=self.normalize_embeddings,
            show_progress_bar=len(texts) > 10,
            batch_size=32,
        )
        return embeddings.tolist()

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        embedding = self.model.encode(
            text,
            normalize_embeddings=self.normalize_embeddings,
        )
        return embedding.tolist()

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this provider.

        Returns:
            Embedding dimension
        """
        return self.model.get_sentence_embedding_dimension()


class OpenAIEmbeddingProvider(EmbeddingProvider):
    """Embedding provider using OpenAI API (optional for future use)."""

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
    ):
        """
        Initialize OpenAI embedding provider.

        Args:
            api_key: OpenAI API key
            model: OpenAI embedding model name
        """
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError(
                "openai package is required for OpenAIEmbeddingProvider. "
                "Install it with: pip install openai"
            )

        self.client = OpenAI(api_key=api_key)
        self.model = model
        self._dimension = None

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of text documents to embed

        Returns:
            List of embedding vectors
        """
        # OpenAI has batch size limits, process in chunks
        batch_size = 100
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            response = self.client.embeddings.create(input=batch, model=self.model)
            embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(embeddings)

        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        response = self.client.embeddings.create(input=[text], model=self.model)
        return response.data[0].embedding

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this provider.

        Returns:
            Embedding dimension
        """
        if self._dimension is None:
            # Get dimension by embedding a dummy text
            dummy_embedding = self.embed_query("test")
            self._dimension = len(dummy_embedding)
        return self._dimension
