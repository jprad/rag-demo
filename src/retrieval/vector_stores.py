"""Vector store implementations."""

from typing import Any, Dict, List, Optional
import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
    Filter,
    FieldCondition,
    MatchValue,
)

from src.utils.interfaces import VectorDatabase


class QdrantVectorStore(VectorDatabase):
    """Vector store using Qdrant."""

    DISTANCE_METRICS = {
        "cosine": Distance.COSINE,
        "euclidean": Distance.EUCLID,
        "dot": Distance.DOT,
    }

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        api_key: Optional[str] = None,
        **kwargs,
    ):
        """
        Initialize Qdrant vector store.

        Args:
            host: Qdrant server host
            port: Qdrant server port
            api_key: Optional API key for cloud instance
            **kwargs: Additional Qdrant client parameters
        """
        self.client = QdrantClient(
            host=host,
            port=port,
            api_key=api_key,
            **kwargs,
        )

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance_metric: str = "cosine",
        **kwargs,
    ) -> None:
        """
        Create a new collection.

        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors
            distance_metric: Distance metric (cosine, euclidean, dot)
            **kwargs: Additional parameters
        """
        # Check if collection already exists
        if self.collection_exists(collection_name):
            print(f"Collection '{collection_name}' already exists")
            return

        distance = self.DISTANCE_METRICS.get(distance_metric, Distance.COSINE)

        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=vector_size, distance=distance),
        )

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
        points = []
        for i, (id_, vector, meta, text) in enumerate(zip(ids, vectors, metadata, texts)):
            # Combine metadata with text
            payload = {**meta, "text": text}

            point = PointStruct(
                id=id_ if id_ else str(uuid.uuid4()),
                vector=vector,
                payload=payload,
            )
            points.append(point)

        # Upload in batches
        batch_size = 100
        for i in range(0, len(points), batch_size):
            batch = points[i : i + batch_size]
            self.client.upsert(collection_name=collection_name, points=batch)

    def search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int = 5,
        filter_dict: Optional[Dict[str, Any]] = None,
        score_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search for similar documents.

        Args:
            collection_name: Name of the collection
            query_vector: Query embedding vector
            top_k: Number of results to return
            filter_dict: Optional metadata filters
            score_threshold: Optional minimum score threshold

        Returns:
            List of search results with scores and metadata
        """
        # Build filter if provided
        query_filter = None
        if filter_dict:
            conditions = []
            for key, value in filter_dict.items():
                conditions.append(
                    FieldCondition(key=key, match=MatchValue(value=value))
                )
            if conditions:
                query_filter = Filter(must=conditions)

        # Perform search
        results = self.client.search(
            collection_name=collection_name,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            score_threshold=score_threshold,
        )

        # Format results
        formatted_results = []
        for result in results:
            formatted_results.append(
                {
                    "id": result.id,
                    "score": result.score,
                    "text": result.payload.get("text", ""),
                    "metadata": {
                        k: v for k, v in result.payload.items() if k != "text"
                    },
                }
            )

        return formatted_results

    def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection.

        Args:
            collection_name: Name of the collection to delete
        """
        self.client.delete_collection(collection_name=collection_name)

    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists.

        Args:
            collection_name: Name of the collection

        Returns:
            True if collection exists
        """
        collections = self.client.get_collections().collections
        return any(c.name == collection_name for c in collections)

    def get_collection_info(self, collection_name: str) -> Dict[str, Any]:
        """
        Get information about a collection.

        Args:
            collection_name: Name of the collection

        Returns:
            Collection information
        """
        info = self.client.get_collection(collection_name=collection_name)
        return {
            "name": collection_name,
            "vector_size": info.config.params.vectors.size,
            "points_count": info.points_count,
            "distance_metric": info.config.params.vectors.distance.name,
        }


class ChromaVectorStore(VectorDatabase):
    """Vector store using ChromaDB (alternative option)."""

    def __init__(self, persist_directory: str = "./data/vectors/chroma", **kwargs):
        """
        Initialize ChromaDB vector store.

        Args:
            persist_directory: Directory to persist data
            **kwargs: Additional ChromaDB parameters
        """
        try:
            import chromadb
        except ImportError:
            raise ImportError(
                "chromadb package is required for ChromaVectorStore. "
                "Install it with: pip install chromadb"
            )

        self.client = chromadb.PersistentClient(path=persist_directory)

    def create_collection(
        self,
        collection_name: str,
        vector_size: int,
        distance_metric: str = "cosine",
        **kwargs,
    ) -> None:
        """
        Create a new collection.

        Args:
            collection_name: Name of the collection
            vector_size: Dimension of vectors (not used in Chroma)
            distance_metric: Distance metric
            **kwargs: Additional parameters
        """
        # Chroma uses l2, ip (inner product), or cosine
        metric_map = {
            "cosine": "cosine",
            "euclidean": "l2",
            "dot": "ip",
        }
        metric = metric_map.get(distance_metric, "cosine")

        try:
            self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": metric},
            )
        except Exception as e:
            if "already exists" in str(e).lower():
                print(f"Collection '{collection_name}' already exists")
            else:
                raise

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
        collection = self.client.get_collection(name=collection_name)

        # Generate IDs if not provided
        if not ids or len(ids) != len(texts):
            ids = [str(uuid.uuid4()) for _ in texts]

        collection.add(
            ids=ids,
            embeddings=vectors,
            documents=texts,
            metadatas=metadata,
        )

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
        collection = self.client.get_collection(name=collection_name)

        results = collection.query(
            query_embeddings=[query_vector],
            n_results=top_k,
            where=filter_dict,
        )

        # Format results
        formatted_results = []
        for i in range(len(results["ids"][0])):
            formatted_results.append(
                {
                    "id": results["ids"][0][i],
                    "score": 1 - results["distances"][0][i],  # Convert distance to similarity
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                }
            )

        return formatted_results

    def delete_collection(self, collection_name: str) -> None:
        """
        Delete a collection.

        Args:
            collection_name: Name of the collection to delete
        """
        self.client.delete_collection(name=collection_name)

    def collection_exists(self, collection_name: str) -> bool:
        """
        Check if a collection exists.

        Args:
            collection_name: Name of the collection

        Returns:
            True if collection exists
        """
        try:
            self.client.get_collection(name=collection_name)
            return True
        except Exception:
            return False
