"""Pytest configuration and shared fixtures."""

import pytest
from typing import Dict, Any


@pytest.fixture
def sample_config() -> Dict[str, Any]:
    """Provide sample configuration for testing."""
    return {
        "embeddings": {
            "provider": "sentence-transformers",
            "config": {
                "model_name": "all-MiniLM-L6-v2",
                "device": "cpu"
            }
        },
        "vector_db": {
            "provider": "qdrant",
            "config": {
                "host": "localhost",
                "port": 6333,
                "collection_name": "test_collection"
            }
        },
        "llm": {
            "provider": "ollama",
            "config": {
                "base_url": "http://localhost:11434",
                "model": "llama3",
                "temperature": 0.7
            }
        },
        "chunking": {
            "strategy": "recursive",
            "config": {
                "chunk_size": 500,
                "chunk_overlap": 50
            }
        },
        "retrieval": {
            "strategy": "semantic",
            "config": {
                "top_k": 3,
                "score_threshold": 0.0
            }
        }
    }


@pytest.fixture
def sample_documents() -> list:
    """Provide sample documents for testing."""
    return [
        {
            "content": "This is a sample document about Python programming.",
            "metadata": {"source": "python_docs", "page": 1}
        },
        {
            "content": "Machine learning is a subset of artificial intelligence.",
            "metadata": {"source": "ml_docs", "page": 1}
        },
        {
            "content": "RAG combines retrieval with generation for better responses.",
            "metadata": {"source": "rag_docs", "page": 1}
        }
    ]


@pytest.fixture
def sample_chunks() -> list:
    """Provide sample text chunks for testing."""
    return [
        "Python is a high-level programming language.",
        "It is widely used for web development and data science.",
        "Machine learning frameworks like TensorFlow are built with Python."
    ]


@pytest.fixture
def sample_query() -> str:
    """Provide a sample query for testing."""
    return "How do I use Python for machine learning?"
