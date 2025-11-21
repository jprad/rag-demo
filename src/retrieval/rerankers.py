"""Re-ranking strategies for improving retrieval quality."""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
import numpy as np


class Reranker(ABC):
    """Abstract base class for re-ranking strategies."""

    @abstractmethod
    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents based on relevance to query.

        Args:
            query: Search query
            documents: List of documents to re-rank
            top_k: Number of top documents to return after re-ranking

        Returns:
            Re-ranked list of documents with updated scores
        """
        pass


class CrossEncoderReranker(Reranker):
    """Re-ranker using cross-encoder models for better relevance scoring."""

    def __init__(
        self,
        model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
        device: Optional[str] = None,
    ):
        """
        Initialize cross-encoder re-ranker.

        Args:
            model_name: Name of the cross-encoder model
            device: Device to run on (cpu, cuda, mps). Auto-detects if None
        """
        try:
            from sentence_transformers import CrossEncoder
            import torch
        except ImportError:
            raise ImportError(
                "sentence-transformers is required for CrossEncoderReranker. "
                "Install it with: pip install sentence-transformers"
            )

        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self.model_name = model_name
        self.device = device
        self.model = CrossEncoder(model_name, device=device)

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents using cross-encoder model.

        Args:
            query: Search query
            documents: List of documents to re-rank
            top_k: Number of top documents to return

        Returns:
            Re-ranked list of documents with cross-encoder scores
        """
        if not documents:
            return documents

        # Prepare query-document pairs
        pairs = [[query, doc["text"]] for doc in documents]

        # Get cross-encoder scores
        scores = self.model.predict(pairs, show_progress_bar=len(pairs) > 10)

        # Add cross-encoder scores to documents
        reranked_docs = []
        for doc, score in zip(documents, scores):
            doc_copy = doc.copy()
            doc_copy["rerank_score"] = float(score)
            doc_copy["original_score"] = doc.get("score", 0.0)
            # Use rerank score as the primary score
            doc_copy["score"] = float(score)
            reranked_docs.append(doc_copy)

        # Sort by rerank score
        reranked_docs.sort(key=lambda x: x["rerank_score"], reverse=True)

        # Return top_k if specified
        if top_k is not None:
            reranked_docs = reranked_docs[:top_k]

        return reranked_docs


class ScoreNormalizationReranker(Reranker):
    """Simple re-ranker that normalizes scores."""

    def __init__(self, method: str = "minmax"):
        """
        Initialize score normalization re-ranker.

        Args:
            method: Normalization method (minmax, zscore, softmax)
        """
        self.method = method

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents by normalizing scores.

        Args:
            query: Search query (not used in this reranker)
            documents: List of documents to re-rank
            top_k: Number of top documents to return

        Returns:
            Re-ranked list of documents with normalized scores
        """
        if not documents:
            return documents

        scores = np.array([doc.get("score", 0.0) for doc in documents])

        # Normalize scores
        if self.method == "minmax":
            min_score = scores.min()
            max_score = scores.max()
            if max_score > min_score:
                normalized_scores = (scores - min_score) / (max_score - min_score)
            else:
                normalized_scores = np.ones_like(scores)

        elif self.method == "zscore":
            mean = scores.mean()
            std = scores.std()
            if std > 0:
                normalized_scores = (scores - mean) / std
            else:
                normalized_scores = np.zeros_like(scores)

        elif self.method == "softmax":
            exp_scores = np.exp(scores - scores.max())  # For numerical stability
            normalized_scores = exp_scores / exp_scores.sum()

        else:
            raise ValueError(f"Unknown normalization method: {self.method}")

        # Create reranked documents
        reranked_docs = []
        for doc, norm_score in zip(documents, normalized_scores):
            doc_copy = doc.copy()
            doc_copy["original_score"] = doc.get("score", 0.0)
            doc_copy["normalized_score"] = float(norm_score)
            doc_copy["score"] = float(norm_score)
            reranked_docs.append(doc_copy)

        # Sort by normalized score
        reranked_docs.sort(key=lambda x: x["score"], reverse=True)

        # Return top_k if specified
        if top_k is not None:
            reranked_docs = reranked_docs[:top_k]

        return reranked_docs


class MMRReranker(Reranker):
    """Maximum Marginal Relevance (MMR) re-ranker for diversity."""

    def __init__(self, lambda_param: float = 0.5):
        """
        Initialize MMR re-ranker.

        Args:
            lambda_param: Trade-off between relevance and diversity (0-1)
                         1.0 = pure relevance, 0.0 = pure diversity
        """
        self.lambda_param = lambda_param

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents using MMR for diversity.

        Args:
            query: Search query
            documents: List of documents to re-rank
            top_k: Number of top documents to return

        Returns:
            Re-ranked list of documents maximizing relevance and diversity
        """
        if not documents:
            return documents

        if top_k is None:
            top_k = len(documents)

        # Get initial relevance scores
        relevance_scores = np.array([doc.get("score", 0.0) for doc in documents])

        # Compute document similarity matrix (simple text overlap for now)
        similarity_matrix = self._compute_similarity_matrix(documents)

        # MMR selection
        selected_indices = []
        remaining_indices = list(range(len(documents)))

        # Select first document (highest relevance)
        first_idx = np.argmax(relevance_scores)
        selected_indices.append(first_idx)
        remaining_indices.remove(first_idx)

        # Iteratively select documents
        while len(selected_indices) < top_k and remaining_indices:
            mmr_scores = []

            for idx in remaining_indices:
                # Relevance score
                relevance = relevance_scores[idx]

                # Maximum similarity to already selected documents
                if selected_indices:
                    similarities = [similarity_matrix[idx][s] for s in selected_indices]
                    max_similarity = max(similarities)
                else:
                    max_similarity = 0

                # MMR score
                mmr = self.lambda_param * relevance - (1 - self.lambda_param) * max_similarity
                mmr_scores.append(mmr)

            # Select document with highest MMR score
            best_idx = remaining_indices[np.argmax(mmr_scores)]
            selected_indices.append(best_idx)
            remaining_indices.remove(best_idx)

        # Create reranked documents
        reranked_docs = []
        for rank, idx in enumerate(selected_indices):
            doc_copy = documents[idx].copy()
            doc_copy["original_score"] = documents[idx].get("score", 0.0)
            doc_copy["mmr_rank"] = rank
            reranked_docs.append(doc_copy)

        return reranked_docs

    def _compute_similarity_matrix(self, documents: List[Dict[str, Any]]) -> np.ndarray:
        """Compute pairwise similarity between documents."""
        n = len(documents)
        similarity_matrix = np.zeros((n, n))

        # Simple Jaccard similarity based on word overlap
        for i in range(n):
            words_i = set(documents[i]["text"].lower().split())
            for j in range(i + 1, n):
                words_j = set(documents[j]["text"].lower().split())
                if words_i or words_j:
                    jaccard = len(words_i & words_j) / len(words_i | words_j)
                else:
                    jaccard = 0
                similarity_matrix[i][j] = jaccard
                similarity_matrix[j][i] = jaccard

        return similarity_matrix


class LLMReranker(Reranker):
    """Re-ranker using LLM to score document relevance."""

    def __init__(self, llm_provider: Any):
        """
        Initialize LLM-based re-ranker.

        Args:
            llm_provider: LLM provider instance for scoring
        """
        self.llm_provider = llm_provider

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """
        Re-rank documents using LLM scoring.

        Args:
            query: Search query
            documents: List of documents to re-rank
            top_k: Number of top documents to return

        Returns:
            Re-ranked list of documents with LLM scores
        """
        if not documents:
            return documents

        reranked_docs = []

        for doc in documents:
            # Create scoring prompt
            prompt = f"""Rate the relevance of the following document to the query on a scale of 0-10.
Only respond with a single number.

Query: {query}

Document: {doc['text'][:500]}...

Relevance score (0-10):"""

            try:
                response = self.llm_provider.generate(prompt, temperature=0.1, max_tokens=5)
                # Extract score
                score_str = response.strip().split()[0]
                llm_score = float(score_str) / 10.0  # Normalize to 0-1
            except Exception:
                # Fallback to original score if LLM scoring fails
                llm_score = doc.get("score", 0.0)

            doc_copy = doc.copy()
            doc_copy["original_score"] = doc.get("score", 0.0)
            doc_copy["llm_score"] = llm_score
            doc_copy["score"] = llm_score
            reranked_docs.append(doc_copy)

        # Sort by LLM score
        reranked_docs.sort(key=lambda x: x["score"], reverse=True)

        # Return top_k if specified
        if top_k is not None:
            reranked_docs = reranked_docs[:top_k]

        return reranked_docs


class RerankerFactory:
    """Factory for creating re-rankers."""

    @staticmethod
    def create(reranker_type: str, config: Dict[str, Any]) -> Optional[Reranker]:
        """
        Create a re-ranker.

        Args:
            reranker_type: Type of re-ranker (cross-encoder, score-norm, mmr, llm, none)
            config: Re-ranker configuration

        Returns:
            Reranker instance or None if type is 'none'

        Raises:
            ValueError: If reranker type is unknown
        """
        if reranker_type == "none" or not reranker_type:
            return None

        elif reranker_type == "cross-encoder":
            return CrossEncoderReranker(
                model_name=config.get("model_name", "cross-encoder/ms-marco-MiniLM-L-6-v2"),
                device=config.get("device"),
            )

        elif reranker_type == "score-norm":
            return ScoreNormalizationReranker(
                method=config.get("method", "minmax"),
            )

        elif reranker_type == "mmr":
            return MMRReranker(
                lambda_param=config.get("lambda_param", 0.5),
            )

        elif reranker_type == "llm":
            # LLM provider will be injected
            return None  # Placeholder, will be created with LLM provider

        else:
            raise ValueError(
                f"Unknown reranker type: {reranker_type}. "
                f"Supported: cross-encoder, score-norm, mmr, llm, none"
            )
