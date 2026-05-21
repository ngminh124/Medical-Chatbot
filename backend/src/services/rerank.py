from typing import Any, Dict, List, Optional, Tuple

from loguru import logger

from ..core.model_config import get_reranking_model
from .remote_model import get_remote_model_service


class Qwen3RerankerService:
    """
    Qwen3 Reranker Service following official Qwen3-Reranker-0.6B best practices.

    Key Features:
    - Format: <Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}
    - System prompt: "Judge whether the Document meets the requirements..."
    - Output: "yes"/"no" tokens with logprobs for scoring
    - Instruction-aware: Custom instructions improve performance by 1-5%

    Reference: https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
    """

    DEFAULT_TASK_INSTRUCTION = (
        "Given a medical question in Vietnamese, retrieve relevant medical knowledge "
        "passages that provide accurate information to answer the question"
    )

    def __init__(
        self,
        task_instruction: Optional[str] = None,
    ):
        """Initialize Qwen3 Reranker Service with remote model service."""

        self.huggingface_model = get_reranking_model()
        self.task_instruction = task_instruction or self.DEFAULT_TASK_INSTRUCTION
        self.remote_service = get_remote_model_service()

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: int = 5,
        task_instruction: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Rerank documents using Qwen3-Reranker-0.6B."""
        instruction = task_instruction or self.task_instruction
        top_n = max(1, min(int(top_n), 4, len(documents)))

        try:
            reranked_results = self._rerank_with_remote(
                query, documents, top_n, instruction
            )
            rerank_context = self._format_rerank_context(documents, reranked_results)
            return reranked_results, rerank_context
        except Exception as e:
            logger.error(f"[RERANK] Failed: {e}")
            # Return original order with neutral scores on failure
            fallback_results = [
                {"index": i, "relevance_score": 0.0, "document": doc}
                for i, doc in enumerate(documents[:top_n])
            ]
            rerank_context = self._format_rerank_context(documents, fallback_results)
            return fallback_results, rerank_context

    def _rerank_with_remote(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: int,
        instruction: str,
    ) -> List[Dict[str, Any]]:
        """Call remote model service for Qwen3-Reranker-0.6B inference."""
        # Truncate to avoid overly large payloads
        MAX_DOC_CHARS = 512
        doc_texts = [
            (
                f"Title: {doc.get('title', '')[:100]}\n"
                f"Content: {doc.get('content', '')[:MAX_DOC_CHARS]}"
            )
            for doc in documents
        ]

        payload = {
            "query": query,
            "documents": doc_texts,
            "top_n": top_n,
            "instruction": instruction,
        }

        result = self.remote_service.rerank(payload)
        scores = result["scores"]
        indices = result["indices"]

        scored_docs = [
            {
                "index": indices[i],
                "relevance_score": scores[i],
                "document": documents[indices[i]],
            }
            for i in range(len(indices))
        ]

        return scored_docs

    def _format_rerank_context(
        self, documents: List[Dict[str, Any]], reranked_results: List[Dict[str, Any]]
    ) -> str:
        """Format reranked documents into context string."""
        context_parts = []
        for rank, result in enumerate(reranked_results, start=1):
            doc = result.get("document") or documents[result["index"]]
            score = result["relevance_score"]
            context_parts.append(
                f"[Tài liệu {rank}] (Relevance: {score:.2f})\n"
                f"{doc.get('title', 'N/A')}\n"
                f"{doc.get('content', 'N/A')}"
            )
        return "\n\n---\n\n".join(context_parts)


_qwen3_reranker_instance = None


def get_qwen3_reranker() -> Qwen3RerankerService:
    global _qwen3_reranker_instance
    if _qwen3_reranker_instance is None:
        _qwen3_reranker_instance = Qwen3RerankerService()
    return _qwen3_reranker_instance