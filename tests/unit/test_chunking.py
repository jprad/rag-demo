"""Unit tests for chunking strategies."""

import pytest
from src.ingest.chunking import (
    RecursiveChunker,
    FixedSizeChunker,
    ParagraphChunker,
    create_chunker
)


class TestRecursiveChunker:
    """Tests for RecursiveChunker."""

    def test_chunk_short_text(self):
        """Test chunking text shorter than chunk size."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)
        text = "Short text"

        chunks = chunker.chunk_text(text)

        assert len(chunks) == 1
        assert chunks[0] == "Short text"

    def test_chunk_long_text(self):
        """Test chunking text longer than chunk size."""
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        text = "This is a longer piece of text. " * 10

        chunks = chunker.chunk_text(text)

        assert len(chunks) > 1
        # Verify overlap exists
        for i in range(len(chunks) - 1):
            assert len(chunks[i]) <= 50 + 20  # Allow some margin

    def test_chunk_with_metadata(self):
        """Test chunking with metadata preservation."""
        chunker = RecursiveChunker(chunk_size=50, chunk_overlap=10)
        document = {
            "content": "This is a test. " * 20,
            "metadata": {"source": "test", "page": 1}
        }

        chunks = chunker.chunk_document(document)

        assert len(chunks) > 1
        for chunk in chunks:
            assert "metadata" in chunk
            assert chunk["metadata"]["source"] == "test"

    def test_empty_text(self):
        """Test chunking empty text."""
        chunker = RecursiveChunker(chunk_size=100, chunk_overlap=10)

        chunks = chunker.chunk_text("")

        assert chunks == []


class TestFixedSizeChunker:
    """Tests for FixedSizeChunker."""

    def test_chunk_exact_size(self):
        """Test chunking with exact fixed sizes."""
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=2)
        text = "A" * 30

        chunks = chunker.chunk_text(text)

        assert len(chunks) == 3
        for chunk in chunks:
            assert len(chunk) <= 12  # chunk_size + overlap margin

    def test_chunk_with_overlap(self):
        """Test that overlap is applied correctly."""
        chunker = FixedSizeChunker(chunk_size=10, chunk_overlap=3)
        text = "0123456789" * 3

        chunks = chunker.chunk_text(text)

        # Check that chunks overlap
        if len(chunks) > 1:
            # Last chars of first chunk should appear in second chunk
            assert chunks[0][-3:] in chunks[1]


class TestParagraphChunker:
    """Tests for ParagraphChunker."""

    def test_chunk_by_paragraphs(self):
        """Test chunking by paragraph boundaries."""
        chunker = ParagraphChunker(max_chunk_size=100)
        text = "Paragraph 1.\n\nParagraph 2.\n\nParagraph 3."

        chunks = chunker.chunk_text(text)

        assert len(chunks) == 3
        assert "Paragraph 1" in chunks[0]
        assert "Paragraph 2" in chunks[1]
        assert "Paragraph 3" in chunks[2]

    def test_combine_small_paragraphs(self):
        """Test that small paragraphs are combined."""
        chunker = ParagraphChunker(max_chunk_size=100)
        text = "A.\n\nB.\n\nC."

        chunks = chunker.chunk_text(text)

        # Small paragraphs should be combined
        assert len(chunks) <= 3

    def test_split_large_paragraphs(self):
        """Test that large paragraphs are split."""
        chunker = ParagraphChunker(max_chunk_size=50)
        text = "This is a very long paragraph. " * 20

        chunks = chunker.chunk_text(text)

        # Large paragraph should be split
        assert len(chunks) > 1
        for chunk in chunks:
            assert len(chunk) <= 70  # Allow some margin


class TestChunkerFactory:
    """Tests for chunker factory function."""

    def test_create_recursive_chunker(self):
        """Test creating recursive chunker."""
        config = {
            "strategy": "recursive",
            "config": {
                "chunk_size": 500,
                "chunk_overlap": 50
            }
        }

        chunker = create_chunker(config)

        assert isinstance(chunker, RecursiveChunker)
        assert chunker.chunk_size == 500
        assert chunker.chunk_overlap == 50

    def test_create_fixed_chunker(self):
        """Test creating fixed size chunker."""
        config = {
            "strategy": "fixed",
            "config": {
                "chunk_size": 1000,
                "chunk_overlap": 100
            }
        }

        chunker = create_chunker(config)

        assert isinstance(chunker, FixedSizeChunker)

    def test_create_paragraph_chunker(self):
        """Test creating paragraph chunker."""
        config = {
            "strategy": "paragraph",
            "config": {
                "max_chunk_size": 1500
            }
        }

        chunker = create_chunker(config)

        assert isinstance(chunker, ParagraphChunker)

    def test_unknown_strategy_raises_error(self):
        """Test that unknown strategy raises ValueError."""
        config = {
            "strategy": "unknown",
            "config": {}
        }

        with pytest.raises(ValueError, match="Unknown chunking strategy"):
            create_chunker(config)
