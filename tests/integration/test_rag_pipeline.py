"""Integration tests for RAG pipeline."""

import pytest
from unittest.mock import Mock, patch, MagicMock
from src.pipeline.rag import RAGPipeline
from src.pipeline.ingestion import IngestionPipeline


class TestIngestionPipeline:
    """Integration tests for document ingestion pipeline."""

    @patch('src.pipeline.ingestion.QdrantVectorStore')
    @patch('src.pipeline.ingestion.SentenceTransformersEmbedding')
    def test_ingest_documents_end_to_end(self, mock_embedding, mock_vector_store):
        """Test complete ingestion pipeline."""
        # Setup mocks
        mock_embedder = MagicMock()
        mock_embedder.embed_documents.return_value = [[0.1, 0.2], [0.3, 0.4]]
        mock_embedder.dimension = 384
        mock_embedding.return_value = mock_embedder

        mock_store = MagicMock()
        mock_vector_store.return_value = mock_store

        # Create pipeline with mocked config
        config = {
            "embeddings": {
                "provider": "sentence-transformers",
                "config": {"model_name": "all-MiniLM-L6-v2"}
            },
            "vector_db": {
                "provider": "qdrant",
                "config": {"host": "localhost", "port": 6333}
            },
            "chunking": {
                "strategy": "recursive",
                "config": {"chunk_size": 500, "chunk_overlap": 50}
            }
        }

        with patch('src.pipeline.ingestion.load_config', return_value=config):
            pipeline = IngestionPipeline(config_path="dummy")

            documents = [
                {"content": "Document 1 content here.", "metadata": {"source": "test"}},
                {"content": "Document 2 content here.", "metadata": {"source": "test"}}
            ]

            pipeline.ingest(documents)

            # Verify embeddings were generated
            mock_embedder.embed_documents.assert_called_once()

            # Verify documents were added to vector store
            mock_store.add_documents.assert_called_once()

    @patch('src.pipeline.ingestion.RecursiveChunker')
    def test_chunking_applied_before_embedding(self, mock_chunker):
        """Test that documents are chunked before embedding."""
        # Setup mock chunker
        mock_chunker_instance = MagicMock()
        mock_chunker_instance.chunk_document.side_effect = lambda doc: [
            {"content": f"{doc['content']} chunk 1", "metadata": doc["metadata"]},
            {"content": f"{doc['content']} chunk 2", "metadata": doc["metadata"]}
        ]
        mock_chunker.return_value = mock_chunker_instance

        config = {
            "embeddings": {
                "provider": "sentence-transformers",
                "config": {}
            },
            "vector_db": {
                "provider": "qdrant",
                "config": {}
            },
            "chunking": {
                "strategy": "recursive",
                "config": {"chunk_size": 100, "chunk_overlap": 10}
            }
        }

        with patch('src.pipeline.ingestion.load_config', return_value=config):
            with patch('src.pipeline.ingestion.SentenceTransformersEmbedding'):
                with patch('src.pipeline.ingestion.QdrantVectorStore'):
                    pipeline = IngestionPipeline(config_path="dummy")

                    documents = [{"content": "Test doc", "metadata": {}}]
                    pipeline.ingest(documents)

                    # Verify chunking was called
                    assert mock_chunker_instance.chunk_document.called


class TestRAGPipeline:
    """Integration tests for RAG query pipeline."""

    @patch('src.pipeline.rag.OllamaLLM')
    @patch('src.pipeline.rag.SemanticRetrieval')
    def test_query_end_to_end(self, mock_retrieval, mock_llm):
        """Test complete RAG query pipeline."""
        # Setup mocks
        mock_retrieval_instance = MagicMock()
        mock_retrieval_instance.retrieve.return_value = [
            {
                "content": "Relevant document content",
                "metadata": {"source": "test"},
                "score": 0.9
            }
        ]
        mock_retrieval.return_value = mock_retrieval_instance

        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.return_value = "Generated answer based on context"
        mock_llm.return_value = mock_llm_instance

        config = {
            "llm": {
                "provider": "ollama",
                "config": {"model": "llama3"}
            },
            "retrieval": {
                "strategy": "semantic",
                "config": {"top_k": 3}
            },
            "embeddings": {
                "provider": "sentence-transformers",
                "config": {}
            },
            "vector_db": {
                "provider": "qdrant",
                "config": {}
            }
        }

        with patch('src.pipeline.rag.load_config', return_value=config):
            with patch('src.pipeline.rag.create_embedding_provider'):
                with patch('src.pipeline.rag.create_vector_store'):
                    with patch('src.pipeline.rag.create_retrieval_strategy', return_value=mock_retrieval_instance):
                        with patch('src.pipeline.rag.create_llm_provider', return_value=mock_llm_instance):
                            pipeline = RAGPipeline(config_path="dummy")

                            response = pipeline.query("What is RAG?")

                            # Verify retrieval was called
                            mock_retrieval_instance.retrieve.assert_called_once_with("What is RAG?", filters=None)

                            # Verify LLM was called with context
                            assert mock_llm_instance.generate.called

                            # Verify response
                            assert "Generated answer" in response["answer"]
                            assert len(response["sources"]) == 1

    @patch('src.pipeline.rag.SemanticRetrieval')
    def test_query_with_no_results(self, mock_retrieval):
        """Test query when no relevant documents are found."""
        # Setup mock to return no results
        mock_retrieval_instance = MagicMock()
        mock_retrieval_instance.retrieve.return_value = []
        mock_retrieval.return_value = mock_retrieval_instance

        config = {
            "llm": {"provider": "ollama", "config": {}},
            "retrieval": {"strategy": "semantic", "config": {}},
            "embeddings": {"provider": "sentence-transformers", "config": {}},
            "vector_db": {"provider": "qdrant", "config": {}}
        }

        with patch('src.pipeline.rag.load_config', return_value=config):
            with patch('src.pipeline.rag.create_embedding_provider'):
                with patch('src.pipeline.rag.create_vector_store'):
                    with patch('src.pipeline.rag.create_retrieval_strategy', return_value=mock_retrieval_instance):
                        with patch('src.pipeline.rag.create_llm_provider') as mock_llm:
                            mock_llm_instance = MagicMock()
                            mock_llm_instance.generate.return_value = "I don't have information about that."
                            mock_llm.return_value = mock_llm_instance

                            pipeline = RAGPipeline(config_path="dummy")
                            response = pipeline.query("Unknown topic")

                            # Should still return a response
                            assert "answer" in response
                            assert response["sources"] == []

    @patch('src.pipeline.rag.SemanticRetrieval')
    @patch('src.pipeline.rag.OllamaLLM')
    def test_query_with_metadata_filter(self, mock_llm, mock_retrieval):
        """Test query with metadata filtering."""
        mock_retrieval_instance = MagicMock()
        mock_retrieval_instance.retrieve.return_value = [
            {"content": "Terraform doc", "metadata": {"source": "terraform"}, "score": 0.9}
        ]
        mock_retrieval.return_value = mock_retrieval_instance

        mock_llm_instance = MagicMock()
        mock_llm_instance.generate.return_value = "Answer about Terraform"
        mock_llm.return_value = mock_llm_instance

        config = {
            "llm": {"provider": "ollama", "config": {}},
            "retrieval": {"strategy": "semantic", "config": {}},
            "embeddings": {"provider": "sentence-transformers", "config": {}},
            "vector_db": {"provider": "qdrant", "config": {}}
        }

        with patch('src.pipeline.rag.load_config', return_value=config):
            with patch('src.pipeline.rag.create_embedding_provider'):
                with patch('src.pipeline.rag.create_vector_store'):
                    with patch('src.pipeline.rag.create_retrieval_strategy', return_value=mock_retrieval_instance):
                        with patch('src.pipeline.rag.create_llm_provider', return_value=mock_llm_instance):
                            pipeline = RAGPipeline(config_path="dummy")

                            filters = {"source": "terraform"}
                            response = pipeline.query("Terraform question", filters=filters)

                            # Verify filters were passed to retrieve
                            mock_retrieval_instance.retrieve.assert_called_once_with(
                                "Terraform question",
                                filters=filters
                            )

    @patch('src.pipeline.rag.OllamaLLM')
    def test_streaming_response(self, mock_llm):
        """Test streaming response generation."""
        mock_llm_instance = MagicMock()
        mock_llm_instance.stream.return_value = iter(["chunk1", "chunk2", "chunk3"])
        mock_llm.return_value = mock_llm_instance

        config = {
            "llm": {"provider": "ollama", "config": {}},
            "retrieval": {"strategy": "semantic", "config": {}},
            "embeddings": {"provider": "sentence-transformers", "config": {}},
            "vector_db": {"provider": "qdrant", "config": {}}
        }

        with patch('src.pipeline.rag.load_config', return_value=config):
            with patch('src.pipeline.rag.create_embedding_provider'):
                with patch('src.pipeline.rag.create_vector_store'):
                    with patch('src.pipeline.rag.create_retrieval_strategy') as mock_strategy:
                        mock_strategy_instance = MagicMock()
                        mock_strategy_instance.retrieve.return_value = [
                            {"content": "Context", "metadata": {}, "score": 0.9}
                        ]
                        mock_strategy.return_value = mock_strategy_instance

                        with patch('src.pipeline.rag.create_llm_provider', return_value=mock_llm_instance):
                            pipeline = RAGPipeline(config_path="dummy")

                            chunks = list(pipeline.query_stream("Test query"))

                            # Verify streaming was called
                            assert mock_llm_instance.stream.called
                            assert len(chunks) == 3
