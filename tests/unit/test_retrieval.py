"""Unit tests for vector stores and retrieval strategies."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.retrieval.vector_stores import QdrantVectorStore, ChromaVectorStore
from src.retrieval.strategies import SemanticRetrieval, HybridRetrieval
from src.retrieval.factory import create_vector_store, create_retrieval_strategy


class TestQdrantVectorStore:
    """Tests for Qdrant vector store."""

    @patch('src.retrieval.vector_stores.QdrantClient')
    def test_initialization(self, mock_client):
        """Test vector store initialization."""
        store = QdrantVectorStore(
            host="localhost",
            port=6333,
            collection_name="test_collection",
            embedding_dimension=384
        )

        assert store.collection_name == "test_collection"
        assert store.embedding_dimension == 384
        mock_client.assert_called_once()

    @patch('src.retrieval.vector_stores.QdrantClient')
    def test_add_documents(self, mock_client):
        """Test adding documents to the store."""
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance

        store = QdrantVectorStore(embedding_dimension=384)
        documents = [
            {"content": "Doc 1", "metadata": {"source": "test"}},
            {"content": "Doc 2", "metadata": {"source": "test"}}
        ]
        embeddings = [[0.1, 0.2], [0.3, 0.4]]

        store.add_documents(documents, embeddings)

        mock_instance.upsert.assert_called_once()

    @patch('src.retrieval.vector_stores.QdrantClient')
    def test_search(self, mock_client):
        """Test searching for documents."""
        mock_instance = MagicMock()
        mock_result = MagicMock()
        mock_result.payload = {"content": "Test doc", "metadata": {}}
        mock_result.score = 0.9
        mock_instance.search.return_value = [mock_result]
        mock_client.return_value = mock_instance

        store = QdrantVectorStore(embedding_dimension=384)
        results = store.search(query_embedding=[0.1, 0.2], top_k=1)

        assert len(results) == 1
        assert results[0]["score"] == 0.9
        mock_instance.search.assert_called_once()

    @patch('src.retrieval.vector_stores.QdrantClient')
    def test_delete_collection(self, mock_client):
        """Test deleting a collection."""
        mock_instance = MagicMock()
        mock_client.return_value = mock_instance

        store = QdrantVectorStore(embedding_dimension=384)
        store.delete_collection()

        mock_instance.delete_collection.assert_called_once()


class TestChromaVectorStore:
    """Tests for Chroma vector store."""

    @patch('src.retrieval.vector_stores.chromadb.Client')
    def test_initialization(self, mock_client):
        """Test vector store initialization."""
        store = ChromaVectorStore(
            collection_name="test_collection",
            persist_directory="./test_db"
        )

        assert store.collection_name == "test_collection"
        mock_client.assert_called_once()

    @patch('src.retrieval.vector_stores.chromadb.Client')
    def test_add_documents(self, mock_client):
        """Test adding documents to the store."""
        mock_instance = MagicMock()
        mock_collection = MagicMock()
        mock_instance.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_instance

        store = ChromaVectorStore()
        documents = [
            {"content": "Doc 1", "metadata": {"source": "test"}},
        ]
        embeddings = [[0.1, 0.2, 0.3]]

        store.add_documents(documents, embeddings)

        mock_collection.add.assert_called_once()

    @patch('src.retrieval.vector_stores.chromadb.Client')
    def test_search(self, mock_client):
        """Test searching for documents."""
        mock_instance = MagicMock()
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Test doc"]],
            "metadatas": [[{"source": "test"}]],
            "distances": [[0.1]]
        }
        mock_instance.get_or_create_collection.return_value = mock_collection
        mock_client.return_value = mock_instance

        store = ChromaVectorStore()
        results = store.search(query_embedding=[0.1, 0.2], top_k=1)

        assert len(results) == 1
        mock_collection.query.assert_called_once()


class TestSemanticRetrieval:
    """Tests for semantic retrieval strategy."""

    def test_retrieve(self):
        """Test semantic retrieval."""
        mock_vector_store = Mock()
        mock_embedding_provider = Mock()

        mock_embedding_provider.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_vector_store.search.return_value = [
            {"content": "Doc 1", "metadata": {}, "score": 0.9}
        ]

        strategy = SemanticRetrieval(
            vector_store=mock_vector_store,
            embedding_provider=mock_embedding_provider,
            top_k=1,
            score_threshold=0.0
        )

        results = strategy.retrieve("test query")

        assert len(results) == 1
        assert results[0]["score"] == 0.9
        mock_embedding_provider.embed_query.assert_called_once_with("test query")
        mock_vector_store.search.assert_called_once()

    def test_retrieve_with_threshold(self):
        """Test retrieval with score threshold filtering."""
        mock_vector_store = Mock()
        mock_embedding_provider = Mock()

        mock_embedding_provider.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_vector_store.search.return_value = [
            {"content": "Doc 1", "metadata": {}, "score": 0.9},
            {"content": "Doc 2", "metadata": {}, "score": 0.5},
            {"content": "Doc 3", "metadata": {}, "score": 0.3}
        ]

        strategy = SemanticRetrieval(
            vector_store=mock_vector_store,
            embedding_provider=mock_embedding_provider,
            top_k=3,
            score_threshold=0.6
        )

        results = strategy.retrieve("test query")

        # Only doc with score >= 0.6 should be returned
        assert len(results) == 1
        assert results[0]["score"] == 0.9

    def test_retrieve_with_metadata_filter(self):
        """Test retrieval with metadata filters."""
        mock_vector_store = Mock()
        mock_embedding_provider = Mock()

        mock_embedding_provider.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_vector_store.search.return_value = []

        strategy = SemanticRetrieval(
            vector_store=mock_vector_store,
            embedding_provider=mock_embedding_provider
        )

        filters = {"source": "terraform"}
        strategy.retrieve("test query", filters=filters)

        # Verify filters were passed to search
        call_args = mock_vector_store.search.call_args
        assert call_args[1].get("filters") == filters


class TestHybridRetrieval:
    """Tests for hybrid retrieval strategy."""

    def test_retrieve_combines_results(self):
        """Test that hybrid retrieval combines semantic and keyword results."""
        mock_vector_store = Mock()
        mock_embedding_provider = Mock()

        mock_embedding_provider.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_vector_store.search.return_value = [
            {"content": "Doc 1", "metadata": {}, "score": 0.9}
        ]

        strategy = HybridRetrieval(
            vector_store=mock_vector_store,
            embedding_provider=mock_embedding_provider,
            semantic_weight=0.7,
            keyword_weight=0.3
        )

        results = strategy.retrieve("test query")

        assert len(results) > 0
        mock_embedding_provider.embed_query.assert_called_once()


class TestRetrievalFactory:
    """Tests for retrieval factory functions."""

    def test_create_qdrant_store(self):
        """Test creating Qdrant vector store."""
        config = {
            "provider": "qdrant",
            "config": {
                "host": "localhost",
                "port": 6333,
                "collection_name": "test"
            }
        }

        with patch('src.retrieval.factory.QdrantVectorStore') as mock:
            create_vector_store(config, embedding_dimension=384)
            mock.assert_called_once()

    def test_create_chroma_store(self):
        """Test creating Chroma vector store."""
        config = {
            "provider": "chroma",
            "config": {
                "collection_name": "test"
            }
        }

        with patch('src.retrieval.factory.ChromaVectorStore') as mock:
            create_vector_store(config, embedding_dimension=384)
            mock.assert_called_once()

    def test_create_semantic_strategy(self):
        """Test creating semantic retrieval strategy."""
        config = {
            "strategy": "semantic",
            "config": {
                "top_k": 5,
                "score_threshold": 0.7
            }
        }

        mock_store = Mock()
        mock_embedder = Mock()

        with patch('src.retrieval.factory.SemanticRetrieval') as mock:
            create_retrieval_strategy(config, mock_store, mock_embedder)
            mock.assert_called_once()

    def test_create_hybrid_strategy(self):
        """Test creating hybrid retrieval strategy."""
        config = {
            "strategy": "hybrid",
            "config": {
                "top_k": 5,
                "semantic_weight": 0.7,
                "keyword_weight": 0.3
            }
        }

        mock_store = Mock()
        mock_embedder = Mock()

        with patch('src.retrieval.factory.HybridRetrieval') as mock:
            create_retrieval_strategy(config, mock_store, mock_embedder)
            mock.assert_called_once()

    def test_unknown_strategy_raises_error(self):
        """Test that unknown strategy raises ValueError."""
        config = {
            "strategy": "unknown",
            "config": {}
        }

        with pytest.raises(ValueError, match="Unknown retrieval strategy"):
            create_retrieval_strategy(config, Mock(), Mock())
