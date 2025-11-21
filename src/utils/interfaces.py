"""Base interfaces for swappable RAG components."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple


class EmbeddingProvider(ABC):
    """Abstract base class for embedding providers."""

    @abstractmethod
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of text documents to embed

        Returns:
            List of embedding vectors
        """
        pass

    @abstractmethod
    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query.

        Args:
            text: Query text to embed

        Returns:
            Embedding vector
        """
        pass

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this provider.

        Returns:
            Embedding dimension
        """
        pass


class VectorDatabase(ABC):
    """Abstract base class for vector databases."""

    @abstractmethod
    def create_collection(self, collection_name: str, vector_size: int, **kwargs) -> None:
        """
        Create a new collection.

        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors
            **kwargs: Additional provider-specific parameters
        """
        pass

    @abstractmethod
    def add_documents(
        self,
        collection_name: str,
        ids: List[str],
        vectors: List[List[float]],
        metadata: List[Dict[str, Any]],
        texts: List[str],
    ) -> None:
        """
        Add documents to a collection.

        Args:
            collection_name: Name of the collection
            ids: Document IDs
            vectors: Embedding vectors
            metadata: Document metadata
            texts: Original text content
        """
        pass

    @abstractmethod
    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            collection_name: Name of the collection
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filters

        Returns:
            List of search results with scores and metadata
        """
        pass

    @abstractmethod
    def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection.

        Args:
            collection_name: Name of the collection to delete
        """
        pass

    @abstractmethod
    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists.

        Args:
            collection_name: Name of the collection

        Returns:
            True if collection exists
        """
        pass


class LLMProvider(ABC):
    """Abstract base class for LLM providers."""

    @abstractmethod
    def generate(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ) -> str:
        """
        Generate text from a prompt.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Returns:
            Generated text
        """
        pass

    @abstractmethod
    def generate_stream(
        self,
        prompt: str,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs,
    ):
        """
        Generate text from a prompt with streaming.

        Args:
            prompt: Input prompt
            temperature: Sampling temperature
            max_tokens: Maximum tokens to generate
            **kwargs: Additional provider-specific parameters

        Yields:
            Text chunks as they're generated
        """
        pass


class DocumentLoader(ABC):
    """Abstract base class for document loaders."""

    @abstractmethod
    def load(self, source: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Load documents from a source.

        Args:
            source: Source identifier (URL, file path, etc.)
            **kwargs: Additional loader-specific parameters

        Returns:
            List of documents with content and metadata
        """
        pass

    @abstractmethod
    def load_batch(self, sources: List[str], **kwargs) -> List[Dict[str, Any]]:
        """
        Load documents from multiple sources.

        Args:
            sources: List of source identifiers
            **kwargs: Additional loader-specific parameters

        Returns:
            List of documents with content and metadata
        """
        pass


class ChunkingStrategy(ABC):
    """Abstract base class for chunking strategies."""

    @abstractmethod
    def chunk_text(self, text: str, metadata: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        """
        Chunk a text document.

        Args:
            text: Text to chunk
            metadata: Optional metadata to include with chunks

        Returns:
            List of chunks with text and metadata
        """
        pass

    @abstractmethod
    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunks with text and metadata
        """
        pass


class RetrievalStrategy(ABC):
    """Abstract base class for retrieval strategies."""

    @abstractmethod
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            top_k: Number of documents to retrieve
            filters: Optional metadata filters
            **kwargs: Additional strategy-specific parameters

        Returns:
            List of retrieved documents with scores
        """
        pass
