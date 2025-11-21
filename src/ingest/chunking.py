"""Chunking strategies for document processing."""

from typing import Any, Callable, Dict, List, Optional
import re

from src.utils.interfaces import ChunkingStrategy


def tiktoken_length(text: str) -> int:
    """Get token count using tiktoken."""
    try:
        import tiktoken

        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    except ImportError:
        # Fallback to character count / 4 (rough approximation)
        return len(text) // 4


class RecursiveChunkingStrategy(ChunkingStrategy):
    """Recursive text chunking strategy."""

    def __init__(
        self,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Optional[List[str]] = None,
        length_function: str = "tiktoken",
    ):
        """
        Initialize recursive chunking strategy.

        Args:
            chunk_size: Maximum size of each chunk
            chunk_overlap: Overlap between chunks
            separators: List of separators to try (in order)
            length_function: Function to measure text length
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", " ", ""]

        if length_function == "tiktoken":
            self.length_function = tiktoken_length
        else:
            self.length_function = len

    def chunk_text(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk a text document.

        Args:
            text: Text to chunk
            metadata: Optional metadata to include with chunks

        Returns:
            List of chunks with text and metadata
        """
        chunks = self._split_text(text)
        result = []

        for i, chunk in enumerate(chunks):
            chunk_meta = {**(metadata or {}), "chunk_index": i, "total_chunks": len(chunks)}
            result.append({"text": chunk, "metadata": chunk_meta})

        return result

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunks with text and metadata
        """
        all_chunks = []

        for doc in documents:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            chunks = self.chunk_text(text, metadata)
            all_chunks.extend(chunks)

        return all_chunks

    def _split_text(self, text: str) -> List[str]:
        """Split text recursively using separators."""
        final_chunks = []
        separator = self.separators[-1]
        new_separators = []

        for i, sep in enumerate(self.separators):
            if sep == "":
                separator = sep
                break
            if re.search(sep, text):
                separator = sep
                new_separators = self.separators[i + 1 :]
                break

        splits = self._split_by_separator(text, separator)

        good_splits = []
        for s in splits:
            if self.length_function(s) < self.chunk_size:
                good_splits.append(s)
            else:
                if good_splits:
                    merged = self._merge_splits(good_splits, separator)
                    final_chunks.extend(merged)
                    good_splits = []

                if not new_separators:
                    final_chunks.append(s)
                else:
                    other_chunks = self._split_text(s)
                    final_chunks.extend(other_chunks)

        if good_splits:
            merged = self._merge_splits(good_splits, separator)
            final_chunks.extend(merged)

        return final_chunks

    def _split_by_separator(self, text: str, separator: str) -> List[str]:
        """Split text by separator."""
        if separator:
            splits = text.split(separator)
        else:
            splits = list(text)

        return [s for s in splits if s]

    def _merge_splits(self, splits: List[str], separator: str) -> List[str]:
        """Merge splits into chunks of appropriate size with overlap."""
        separator_len = self.length_function(separator)

        docs = []
        current_doc = []
        total = 0

        for split in splits:
            split_len = self.length_function(split)

            if total + split_len + (separator_len if current_doc else 0) > self.chunk_size:
                if current_doc:
                    doc = separator.join(current_doc)
                    docs.append(doc)

                    # Create overlap
                    while total > self.chunk_overlap or (
                        total + split_len + separator_len > self.chunk_size and total > 0
                    ):
                        total -= self.length_function(current_doc[0]) + separator_len
                        current_doc = current_doc[1:]

            current_doc.append(split)
            total += split_len + (separator_len if len(current_doc) > 1 else 0)

        if current_doc:
            doc = separator.join(current_doc)
            docs.append(doc)

        return docs


class FixedSizeChunkingStrategy(ChunkingStrategy):
    """Fixed-size chunking strategy."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize fixed-size chunking strategy.

        Args:
            chunk_size: Size of each chunk in characters
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_text(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk a text document.

        Args:
            text: Text to chunk
            metadata: Optional metadata to include with chunks

        Returns:
            List of chunks with text and metadata
        """
        chunks = []
        start = 0

        while start < len(text):
            end = start + self.chunk_size
            chunk = text[start:end]

            chunk_meta = {
                **(metadata or {}),
                "chunk_index": len(chunks),
                "start_char": start,
                "end_char": end,
            }
            chunks.append({"text": chunk, "metadata": chunk_meta})

            start += self.chunk_size - self.chunk_overlap

        return chunks

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunks with text and metadata
        """
        all_chunks = []

        for doc in documents:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            chunks = self.chunk_text(text, metadata)
            all_chunks.extend(chunks)

        return all_chunks


class ParagraphChunkingStrategy(ChunkingStrategy):
    """Paragraph-based chunking strategy."""

    def __init__(self, max_chunk_size: int = 1000):
        """
        Initialize paragraph chunking strategy.

        Args:
            max_chunk_size: Maximum size of each chunk
        """
        self.max_chunk_size = max_chunk_size

    def chunk_text(
        self, text: str, metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Chunk a text document by paragraphs.

        Args:
            text: Text to chunk
            metadata: Optional metadata to include with chunks

        Returns:
            List of chunks with text and metadata
        """
        # Split by double newlines (paragraphs)
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_size = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue

            para_size = len(para)

            if current_size + para_size > self.max_chunk_size and current_chunk:
                # Save current chunk
                chunk_text = "\n\n".join(current_chunk)
                chunk_meta = {**(metadata or {}), "chunk_index": len(chunks)}
                chunks.append({"text": chunk_text, "metadata": chunk_meta})

                current_chunk = [para]
                current_size = para_size
            else:
                current_chunk.append(para)
                current_size += para_size

        # Add remaining chunk
        if current_chunk:
            chunk_text = "\n\n".join(current_chunk)
            chunk_meta = {**(metadata or {}), "chunk_index": len(chunks)}
            chunks.append({"text": chunk_text, "metadata": chunk_meta})

        return chunks

    def chunk_documents(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Chunk multiple documents.

        Args:
            documents: List of documents to chunk

        Returns:
            List of chunks with text and metadata
        """
        all_chunks = []

        for doc in documents:
            text = doc.get("text", "")
            metadata = doc.get("metadata", {})
            chunks = self.chunk_text(text, metadata)
            all_chunks.extend(chunks)

        return all_chunks


class ChunkingStrategyFactory:
    """Factory for creating chunking strategies."""

    @staticmethod
    def create(strategy: str, config: Dict[str, Any]) -> ChunkingStrategy:
        """
        Create a chunking strategy.

        Args:
            strategy: Strategy type (recursive, fixed, paragraph, semantic)
            config: Strategy configuration

        Returns:
            ChunkingStrategy instance

        Raises:
            ValueError: If strategy type is unknown
        """
        if strategy == "recursive":
            return RecursiveChunkingStrategy(
                chunk_size=config.get("chunk_size", 1000),
                chunk_overlap=config.get("chunk_overlap", 200),
                separators=config.get("separators"),
                length_function=config.get("length_function", "tiktoken"),
            )

        elif strategy == "fixed":
            return FixedSizeChunkingStrategy(
                chunk_size=config.get("chunk_size", 1000),
                chunk_overlap=config.get("chunk_overlap", 200),
            )

        elif strategy == "paragraph":
            return ParagraphChunkingStrategy(
                max_chunk_size=config.get("chunk_size", 1000),
            )

        else:
            raise ValueError(
                f"Unknown chunking strategy: {strategy}. "
                f"Supported: recursive, fixed, paragraph"
            )
