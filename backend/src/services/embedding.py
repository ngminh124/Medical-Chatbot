"""
Qwen3 Embedding Service for Minqes

This service handles embeddings using a local HTTP endpoint that runs:
  Qwen/Qwen3-Embedding-0.6B

Key Features:
- Instruction-aware: Queries use task instruction prefix
- Documents: Indexed without instruction
- Normalization: Always L2-normalize embeddings
- Remote HTTP: Communicates with model service via HTTP
- Caching: Optional query/embedding caching via Redis
"""
import hashlib
import threading
import time
from typing import List, Optional

from loguru import logger

from ..configs.setup import get_backend_settings
from .remote_model import get_remote_model_service

settings = get_backend_settings()

# ── Lightweight in-memory embedding cache fallback ───────────────────────────
_embed_mem_cache: dict[str, tuple[float, List[float]]] = {}
_embed_mem_lock = threading.Lock()
_EMBED_CACHE_TTL_SECONDS = 3600


def _hash_query_key(instruction: str, query: str) -> str:
    raw = f"{instruction}::{query}".strip().lower()
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _mem_get(cache_key: str) -> Optional[List[float]]:
    now = time.time()
    with _embed_mem_lock:
        item = _embed_mem_cache.get(cache_key)
        if not item:
            return None
        expires_at, value = item
        if expires_at < now:
            _embed_mem_cache.pop(cache_key, None)
            return None
        return value


def _mem_set(cache_key: str, value: List[float], ttl: int = _EMBED_CACHE_TTL_SECONDS) -> None:
    with _embed_mem_lock:
        _embed_mem_cache[cache_key] = (time.time() + ttl, value)


class Qwen3EmbeddingService:
    """
    Qwen3 Embedding Service following Qwen3-Embedding-0.6B best practices.

    This service communicates with a remote HTTP model service for embeddings.

    Reference: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
    """

    DEFAULT_TASK_INSTRUCTION = (
        "Given a medical question in Vietnamese, retrieve relevant medical knowledge "
        "passages that provide accurate information to answer the question"
    )

    def __init__(
        self,
        task_instruction: Optional[str] = None,
    ):
        """
        Initialize Qwen3 Embedding Service.

        Args:
            task_instruction: Default task instruction for query embedding
        """
        self.task_instruction = task_instruction or self.DEFAULT_TASK_INSTRUCTION
        self.remote_service = get_remote_model_service()

    def embed_query(
        self, query: str, use_cache: bool = True, task_instruction: Optional[str] = None
    ) -> Optional[List[float]]:
        """
        Embed a query with instruction prefix (Qwen3 best practice for queries).

        Args:
            query: Query text to embed
            use_cache: Whether to use Redis cache
            task_instruction: Optional custom instruction

        Returns:
            Embedding vector (1024-dim) or None on error
        """
        instruction = task_instruction or self.task_instruction
        cache_key = _hash_query_key(instruction=instruction, query=query)

        if use_cache:
            # 1) Fast in-memory cache
            in_mem = _mem_get(cache_key)
            if in_mem:
                return in_mem

            # 2) Redis cache
            from ..core.cache import get_query_embedding

            cached_embedding = get_query_embedding(cache_key)
            if cached_embedding:
                _mem_set(cache_key, cached_embedding)
                return cached_embedding

        embedding = self._embed_remote(texts=[query], is_query=True, instruction=instruction)
        if embedding:
            embedding = embedding[0]

        if embedding and use_cache:
            from ..core.cache import cache_query_embedding

            cache_query_embedding(cache_key, embedding)
            _mem_set(cache_key, embedding)

        return embedding

    def embed_document(self, document: str) -> Optional[List[float]]:
        """
        Embed a document WITHOUT instruction prefix (Qwen3 best practice for indexing).

        Args:
            document: Document text to embed

        Returns:
            Embedding vector (1024-dim) or None on error
        """
        embedding = self._embed_remote(texts=[document], is_query=False, instruction=None)
        return embedding[0] if embedding else None

    def embed_text(self, text: str, use_cache: bool = True) -> Optional[List[float]]:
        """
        Legacy method for backward compatibility. Alias for embed_query.

        Args:
            text: Text to embed
            use_cache: Whether to use cache

        Returns:
            Embedding vector or None
        """
        return self.embed_query(text, use_cache=use_cache)

    def embed_batch_documents(
        self, documents: List[str], batch_size: int = 32
    ) -> List[Optional[List[float]]]:
        """
        Embed multiple documents WITHOUT instruction (for indexing into Qdrant).

        Args:
            documents: List of document texts
            batch_size: Size of batches for HTTP requests

        Returns:
            List of embedding vectors (or None for failed items)
        """
        embeddings = []

        for i in range(0, len(documents), batch_size):
            batch = documents[i : i + batch_size]
            batch_embeddings = self._embed_remote(texts=batch, is_query=False, instruction=None)
            embeddings.extend(batch_embeddings or [None] * len(batch))

        return embeddings

    def _embed_remote(
        self,
        texts: List[str],
        is_query: bool = False,
        instruction: Optional[str] = None,
    ) -> Optional[List[List[float]]]:
        """Call remote model service for embeddings."""
        payload = {
            "texts": texts,
            "normalize": True,
            "is_query": is_query,
        }
        if is_query and instruction:
            payload["instruction"] = instruction

        try:
            result = self.remote_service.embed(payload)
            return result.get("embeddings")
        except Exception as exc:
            logger.error(f"[EMBED] Remote embedding failed: {exc}")
            return None

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension (1024 for Qwen3-Embedding-0.6B)."""
        return settings.vector_dimension

    def health_check(self) -> bool:
        """Check if remote embedding service is alive."""
        return self.remote_service.health_check()


# Singleton instance
_embedding_service_instance = None
_embedding_service_lock = threading.Lock()


def get_embedding_service() -> Qwen3EmbeddingService:
    """
    Get singleton Qwen3 embedding service instance.

    Returns:
        Qwen3EmbeddingService instance
    """
    global _embedding_service_instance
    if _embedding_service_instance is None:
        with _embedding_service_lock:
            if _embedding_service_instance is None:
                _embedding_service_instance = Qwen3EmbeddingService()
    else:
        logger.debug("[EMBEDDING] reused instance")
    return _embedding_service_instance
