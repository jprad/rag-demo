"""Document ingestion pipeline."""

from typing import Any, Dict, List, Optional
import uuid
from tqdm import tqdm

from src.embeddings.factory import EmbeddingProviderFactory
from src.ingest.chunking import ChunkingStrategyFactory
from src.ingest.loaders.web_loader import DocumentLoaderFactory
from src.retrieval.factory import VectorStoreFactory
from src.utils.config_loader import ConfigLoader


class IngestionPipeline:
    """Pipeline for ingesting documents into vector database."""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize ingestion pipeline.

        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigLoader(config_path)

        # Initialize components from config
        print("Initializing ingestion pipeline...")

        # Document loader
        loader_config = self.config.get("document_loader", {})
        self.loader = DocumentLoaderFactory.create(
            loader_type=loader_config.get("type", "web"),
            config=loader_config.get("config", {}),
        )

        # Chunking strategy
        chunking_config = self.config.get_chunking_config()
        self.chunker = ChunkingStrategyFactory.create(
            strategy=chunking_config.get("strategy", "recursive"),
            config=chunking_config.get("config", {}),
        )

        # Embedding provider
        embeddings_config = self.config.get_embeddings_config()
        self.embedding_provider = EmbeddingProviderFactory.create(
            provider=embeddings_config.get("provider", "sentence-transformers"),
            config=embeddings_config.get("config", {}),
        )

        # Vector database
        vector_db_config = self.config.get_vector_db_config()
        self.vector_store = VectorStoreFactory.create(
            provider=vector_db_config.get("provider", "qdrant"),
            config=vector_db_config.get("config", {}),
        )

        self.collection_name = vector_db_config.get("config", {}).get(
            "collection_name", "documentation"
        )

        print("Pipeline initialized successfully!")

    def ingest_source(
        self,
        source: str,
        source_name: str,
        max_depth: int = 1,
        include_patterns: Optional[List[str]] = None,
        exclude_patterns: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Ingest documents from a single source.

        Args:
            source: Source URL or identifier
            source_name: Name of the source (e.g., 'ansible', 'terraform')
            max_depth: Maximum crawl depth for web sources
            include_patterns: URL patterns to include
            exclude_patterns: URL patterns to exclude

        Returns:
            Dictionary with ingestion statistics
        """
        print(f"\n{'='*60}")
        print(f"Ingesting: {source_name}")
        print(f"Source: {source}")
        print(f"{'='*60}\n")

        # Load documents
        print("Loading documents...")
        documents = self.loader.load_batch(
            sources=[source],
            max_depth=max_depth,
            include_patterns=include_patterns,
            exclude_patterns=exclude_patterns,
        )

        if not documents:
            print(f"No documents loaded from {source}")
            return {"source": source_name, "documents": 0, "chunks": 0}

        print(f"Loaded {len(documents)} documents")

        # Add source name to metadata
        for doc in documents:
            doc["metadata"]["source_name"] = source_name

        # Chunk documents
        print("Chunking documents...")
        chunks = self.chunker.chunk_documents(documents)
        print(f"Created {len(chunks)} chunks")

        # Embed chunks
        print("Generating embeddings...")
        texts = [chunk["text"] for chunk in chunks]
        embeddings = self.embedding_provider.embed_documents(texts)
        print(f"Generated {len(embeddings)} embeddings")

        # Store in vector database
        print("Storing in vector database...")
        self._store_chunks(chunks, embeddings)
        print("Storage complete!")

        stats = {
            "source": source_name,
            "documents": len(documents),
            "chunks": len(chunks),
        }

        print(f"\nIngestion complete: {stats}")
        return stats

    def ingest_all_sources(self) -> List[Dict[str, Any]]:
        """
        Ingest all configured sources.

        Returns:
            List of ingestion statistics for each source
        """
        sources_config = self.config.get_sources_config()
        all_stats = []

        # Ensure collection exists
        self._ensure_collection()

        for source_name, source_config in sources_config.items():
            if not source_config.get("enabled", True):
                print(f"Skipping disabled source: {source_name}")
                continue

            try:
                stats = self.ingest_source(
                    source=source_config["url"],
                    source_name=source_name,
                    max_depth=source_config.get("max_depth", 1),
                    include_patterns=source_config.get("include_patterns"),
                    exclude_patterns=source_config.get("exclude_patterns"),
                )
                all_stats.append(stats)

            except Exception as e:
                print(f"Error ingesting {source_name}: {e}")
                all_stats.append(
                    {"source": source_name, "documents": 0, "chunks": 0, "error": str(e)}
                )

        return all_stats

    def _ensure_collection(self) -> None:
        """Ensure vector database collection exists."""
        if not self.vector_store.collection_exists(self.collection_name):
            print(f"Creating collection: {self.collection_name}")

            vector_size = self.embedding_provider.get_embedding_dimension()
            vector_db_config = self.config.get_vector_db_config()
            distance_metric = vector_db_config.get("config", {}).get(
                "distance_metric", "cosine"
            )

            self.vector_store.create_collection(
                collection_name=self.collection_name,
                vector_size=vector_size,
                distance_metric=distance_metric,
            )
        else:
            print(f"Using existing collection: {self.collection_name}")

    def _store_chunks(self, chunks: List[Dict[str, Any]], embeddings: List[List[float]]) -> None:
        """Store chunks in vector database."""
        ids = [str(uuid.uuid4()) for _ in chunks]
        texts = [chunk["text"] for chunk in chunks]
        metadata = [chunk["metadata"] for chunk in chunks]

        self.vector_store.add_documents(
            collection_name=self.collection_name,
            ids=ids,
            vectors=embeddings,
            metadata=metadata,
            texts=texts,
        )

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the ingested data.

        Returns:
            Dictionary with statistics
        """
        if hasattr(self.vector_store, "get_collection_info"):
            return self.vector_store.get_collection_info(self.collection_name)
        return {}
