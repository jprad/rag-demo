"""RAG (Retrieval-Augmented Generation) pipeline."""

from typing import Any, Dict, Iterator, List, Optional

from src.generation.factory import LLMProviderFactory
from src.retrieval.strategies import RetrievalStrategyFactory
from src.utils.config_loader import ConfigLoader


class RAGPipeline:
    """End-to-end RAG pipeline for question answering."""

    DEFAULT_PROMPT_TEMPLATE = """You are a helpful AI assistant specialized in infrastructure-as-code and automation tools. Use the following context from documentation to answer the user's question. If you cannot find the answer in the context, say so clearly.

Context:
{context}

Question: {question}

Answer: Let me help you with that based on the documentation."""

    CONVERSATIONAL_TEMPLATE = """You are a helpful AI assistant with expertise in infrastructure automation. Use the provided documentation context to give accurate, helpful answers.

Previous conversation:
{history}

Context from documentation:
{context}

User: {question}

    Assistant:"""

    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize RAG pipeline.

        Args:
            config_path: Path to configuration file
        """
        self.config = ConfigLoader(config_path)

        # Initialize components
        llm_config = self.config.get_llm_config()
        self.llm_provider = LLMProviderFactory.create(
            provider=llm_config.get("provider", "ollama"),
            config=llm_config.get("config", {}),
        )

        retrieval_config = self.config.get_retrieval_config()
        self.retrieval_strategy = RetrievalStrategyFactory.create(
            strategy=retrieval_config.get("strategy", "semantic"),
            config=self.config,
        )

        self.rag_config = self.config.get_rag_config()
        self.conversation_history: List[Dict[str, str]] = []

    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
        include_sources: Optional[bool] = None,
    ) -> Dict[str, Any]:
        """
        Answer a question using RAG.

        Args:
            question: User question
            top_k: Number of context documents to retrieve
            filters: Optional metadata filters
            include_sources: Whether to include source documents

        Returns:
            Dictionary with answer and optionally sources
        """
        # Retrieve relevant documents
        top_k = top_k or self.rag_config.get("top_k", 5)
        documents = self.retrieval_strategy.retrieve(
            query=question,
            top_k=top_k,
            filters=filters,
        )

        # Format context
        context = self._format_context(documents)

        # Build prompt
        prompt = self._build_prompt(question, context)

        # Generate answer
        answer = self.llm_provider.generate(prompt)

        # Build response
        response = {"answer": answer, "question": question}

        if include_sources or self.rag_config.get("include_sources", True):
            response["sources"] = [
                {
                    "text": doc["text"][:200] + "...",
                    "score": doc["score"],
                    "metadata": doc.get("metadata", {}),
                }
                for doc in documents
            ]

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": answer})

        return response

    def query_stream(
        self,
        question: str,
        top_k: Optional[int] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Iterator[str]:
        """
        Answer a question using RAG with streaming response.

        Args:
            question: User question
            top_k: Number of context documents to retrieve
            filters: Optional metadata filters

        Yields:
            Answer chunks as they're generated
        """
        # Retrieve relevant documents
        top_k = top_k or self.rag_config.get("top_k", 5)
        documents = self.retrieval_strategy.retrieve(
            query=question,
            top_k=top_k,
            filters=filters,
        )

        # Format context
        context = self._format_context(documents)

        # Build prompt
        prompt = self._build_prompt(question, context)

        # Generate answer with streaming
        full_answer = ""
        for chunk in self.llm_provider.generate_stream(prompt):
            full_answer += chunk
            yield chunk

        # Update conversation history
        self.conversation_history.append({"role": "user", "content": question})
        self.conversation_history.append({"role": "assistant", "content": full_answer})

    def _format_context(self, documents: List[Dict[str, Any]]) -> str:
        """Format retrieved documents into context string."""
        context_parts = []

        for i, doc in enumerate(documents, 1):
            source = doc.get("metadata", {}).get("source_name", "unknown")
            text = doc["text"]
            context_parts.append(f"[Source {i} - {source}]\n{text}")

        return "\n\n".join(context_parts)

    def _build_prompt(self, question: str, context: str) -> str:
        """Build prompt from question and context."""
        template_name = self.rag_config.get("prompt_template", "default")

        if template_name == "conversational" and self.conversation_history:
            history = self._format_history()
            return self.CONVERSATIONAL_TEMPLATE.format(
                history=history,
                context=context,
                question=question,
            )
        else:
            return self.DEFAULT_PROMPT_TEMPLATE.format(
                context=context,
                question=question,
            )

    def _format_history(self, max_turns: int = 3) -> str:
        """Format conversation history."""
        recent_history = self.conversation_history[-(max_turns * 2) :]
        history_parts = []

        for entry in recent_history:
            role = entry["role"].capitalize()
            content = entry["content"][:200]  # Truncate long messages
            history_parts.append(f"{role}: {content}")

        return "\n".join(history_parts)

    def clear_history(self) -> None:
        """Clear conversation history."""
        self.conversation_history = []
