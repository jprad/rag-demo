"""Retrieval strategies for RAG."""

from typing import Any, Dict, List, Optional

from src.embeddings.factory import EmbeddingProviderFactory
from src.retrieval.factory import VectorStoreFactory
from src.utils.config_loader import ConfigLoader
from src.utils.interfaces import RetrievalStrategy


class SemanticRetrievalStrategy(RetrievalStrategy):
    """Semantic similarity-based retrieval."""

    def __init__(
        self,
        config: ConfigLoader,
        collection_name: Optional[str] = None,
    ):
        """
        Initialize semantic retrieval strategy.

        Args:
            config: Configuration loader
            collection_name: Vector database collection name
        """
        self.config = config

        # Initialize components
        embeddings_config = config.get_embeddings_config()
        self.embedding_provider = EmbeddingProviderFactory.create(
            provider=embeddings_config.get("provider", "sentence-transformers"),
            config=embeddings_config.get("config", {}),
        )

        vector_db_config = config.get_vector_db_config()
        self.vector_store = VectorStoreFactory.create(
            provider=vector_db_config.get("provider", "qdrant"),
            config=vector_db_config.get("config", {}),
        )

        self.collection_name = collection_name or vector_db_config.get("config", {}).get(
            "collection_name", "documentation"
        )

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents for a query.

        Args:
            query: Search query
            top_k: Number of documents to retrieve
            filters: Optional metadata filters
            score_threshold: Optional minimum score threshold
            **kwargs: Additional parameters

        Returns:
            List of retrieved documents with scores
        """
        # Embed query
        query_vector = self.embedding_provider.embed_query(query)

        # Search vector database
        results = self.vector_store.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            top_k=top_k,
            filter_dict=filters,
            score_threshold=score_threshold,
        )

        return results


class HybridRetrievalStrategy(RetrievalStrategy):
    """Hybrid retrieval combining semantic and keyword search."""

    def __init__(
        self,
        config: ConfigLoader,
        collection_name: Optional[str] = None,
        semantic_weight: float = 0.7,
    ):
        """
        Initialize hybrid retrieval strategy.

        Args:
            config: Configuration loader
            collection_name: Vector database collection name
            semantic_weight: Weight for semantic scores (0-1)
        """
        self.semantic_strategy = SemanticRetrievalStrategy(config, collection_name)
        self.semantic_weight = semantic_weight
        self.keyword_weight = 1.0 - semantic_weight

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        filters: Optional[Dict[str, Any]] = None,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve relevant documents using hybrid search.

        Args:
            query: Search query
            top_k: Number of documents to retrieve
            filters: Optional metadata filters
            **kwargs: Additional parameters

        Returns:
            List of retrieved documents with scores
        """
        # Get more results for reranking
        semantic_top_k = top_k * 2

        # Semantic search
        semantic_results = self.semantic_strategy.retrieve(
            query=query,
            top_k=semantic_top_k,
            filters=filters,
        )

        # Simple keyword matching for demonstration
        # In production, could use BM25 or other keyword search
        query_terms = set(query.lower().split())

        for result in semantic_results:
            text_terms = set(result["text"].lower().split())
            keyword_score = len(query_terms & text_terms) / len(query_terms)

            # Combine scores
            hybrid_score = (
                self.semantic_weight * result["score"] + self.keyword_weight * keyword_score
            )
            result["hybrid_score"] = hybrid_score
            result["keyword_score"] = keyword_score

        # Re-rank by hybrid score
        semantic_results.sort(key=lambda x: x.get("hybrid_score", 0), reverse=True)

        return semantic_results[:top_k]


class RetrievalStrategyFactory:
    """Factory for creating retrieval strategies."""

    @staticmethod
    def create(
        strategy: str,
        config: ConfigLoader,
        collection_name: Optional[str] = None,
    ) -> RetrievalStrategy:
        """
        Create a retrieval strategy.

        Args:
            strategy: Strategy type (semantic, keyword, hybrid, mmr)
            config: Configuration loader
            collection_name: Vector database collection name

        Returns:
            RetrievalStrategy instance

        Raises:
            ValueError: If strategy type is unknown
        """
        if strategy == "semantic":
            return SemanticRetrievalStrategy(config, collection_name)

        elif strategy == "hybrid":
            retrieval_config = config.get_retrieval_config()
            semantic_weight = retrieval_config.get("config", {}).get("semantic_weight", 0.7)
            return HybridRetrievalStrategy(config, collection_name, semantic_weight)

        else:
            raise ValueError(
                f"Unknown retrieval strategy: {strategy}. " f"Supported: semantic, hybrid"
            )
