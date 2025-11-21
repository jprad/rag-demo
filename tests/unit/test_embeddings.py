"""Unit tests for embedding providers."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.embeddings.providers import SentenceTransformersEmbedding, OpenAIEmbedding
from src.embeddings.factory import create_embedding_provider


class TestSentenceTransformersEmbedding:
    """Tests for SentenceTransformersEmbedding provider."""

    @patch('src.embeddings.providers.SentenceTransformer')
    def test_initialization(self, mock_st):
        """Test provider initialization."""
        provider = SentenceTransformersEmbedding(
            model_name="all-MiniLM-L6-v2",
            device="cpu"
        )

        assert provider.model_name == "all-MiniLM-L6-v2"
        assert provider.device == "cpu"
        mock_st.assert_called_once_with("all-MiniLM-L6-v2", device="cpu")

    @patch('src.embeddings.providers.SentenceTransformer')
    def test_embed_documents(self, mock_st):
        """Test embedding multiple documents."""
        # Setup mock
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]
        mock_st.return_value = mock_model

        provider = SentenceTransformersEmbedding()
        texts = ["Document 1", "Document 2"]
        embeddings = provider.embed_documents(texts)

        assert len(embeddings) == 2
        assert len(embeddings[0]) == 3
        mock_model.encode.assert_called_once()

    @patch('src.embeddings.providers.SentenceTransformer')
    def test_embed_query(self, mock_st):
        """Test embedding a single query."""
        # Setup mock
        mock_model = MagicMock()
        mock_model.encode.return_value = [[0.1, 0.2, 0.3]]
        mock_st.return_value = mock_model

        provider = SentenceTransformersEmbedding()
        embedding = provider.embed_query("Test query")

        assert len(embedding) == 3
        mock_model.encode.assert_called_once()

    @patch('src.embeddings.providers.SentenceTransformer')
    def test_embed_empty_list(self, mock_st):
        """Test embedding empty list returns empty list."""
        mock_model = MagicMock()
        mock_model.encode.return_value = []
        mock_st.return_value = mock_model

        provider = SentenceTransformersEmbedding()
        embeddings = provider.embed_documents([])

        assert embeddings == []

    @patch('src.embeddings.providers.SentenceTransformer')
    def test_dimension_property(self, mock_st):
        """Test dimension property returns correct value."""
        mock_model = MagicMock()
        mock_model.get_sentence_embedding_dimension.return_value = 384
        mock_st.return_value = mock_model

        provider = SentenceTransformersEmbedding()

        assert provider.dimension == 384


class TestOpenAIEmbedding:
    """Tests for OpenAIEmbedding provider."""

    @patch('src.embeddings.providers.OpenAIEmbeddings')
    def test_initialization(self, mock_openai):
        """Test provider initialization."""
        provider = OpenAIEmbedding(
            model="text-embedding-ada-002",
            api_key="test-key"
        )

        assert provider.model == "text-embedding-ada-002"
        mock_openai.assert_called_once()

    @patch('src.embeddings.providers.OpenAIEmbeddings')
    def test_embed_documents(self, mock_openai):
        """Test embedding multiple documents."""
        # Setup mock
        mock_client = MagicMock()
        mock_client.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_openai.return_value = mock_client

        provider = OpenAIEmbedding()
        texts = ["Doc 1", "Doc 2"]
        embeddings = provider.embed_documents(texts)

        assert len(embeddings) == 2
        mock_client.embed_documents.assert_called_once_with(texts)

    @patch('src.embeddings.providers.OpenAIEmbeddings')
    def test_embed_query(self, mock_openai):
        """Test embedding a single query."""
        # Setup mock
        mock_client = MagicMock()
        mock_client.embed_query.return_value = [0.1, 0.2, 0.3]
        mock_openai.return_value = mock_client

        provider = OpenAIEmbedding()
        embedding = provider.embed_query("Test query")

        assert len(embedding) == 3
        mock_client.embed_query.assert_called_once_with("Test query")


class TestEmbeddingFactory:
    """Tests for embedding factory."""

    def test_create_sentence_transformers_provider(self, sample_config):
        """Test creating SentenceTransformers provider."""
        with patch('src.embeddings.factory.SentenceTransformersEmbedding') as mock:
            create_embedding_provider(sample_config["embeddings"])
            mock.assert_called_once()

    def test_create_openai_provider(self):
        """Test creating OpenAI provider."""
        config = {
            "provider": "openai",
            "config": {
                "model": "text-embedding-ada-002",
                "api_key": "test-key"
            }
        }

        with patch('src.embeddings.factory.OpenAIEmbedding') as mock:
            create_embedding_provider(config)
            mock.assert_called_once()

    def test_unknown_provider_raises_error(self):
        """Test that unknown provider raises ValueError."""
        config = {
            "provider": "unknown",
            "config": {}
        }

        with pytest.raises(ValueError, match="Unknown embedding provider"):
            create_embedding_provider(config)
