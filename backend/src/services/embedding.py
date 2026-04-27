"""
Qwen3 Embedding Service for Minqes

This service handles embeddings using a local HTTP endpoint that runs:
  Qwen/Qwen3-Embedding-0.6B

Key Features:
- Instruction-aware: Queries use task instruction prefix
- Documents: Indexed without instruction
- Normalization: Always L2-normalize embeddings
- Local HTTP: Communicates with serving/qwen3_models/app.py via HTTP
- Caching: Optional query/embedding caching via Redis
- Local fallback: Uses SentenceTransformer in-process if HTTP service is down
"""
import hashlib
import threading
import time
from typing import Any, List, Optional

import httpx
import numpy as np
from loguru import logger

from ..configs.setup import get_backend_settings

# ── Local in-process fallback (lazy-loaded) ──────────────────────────────────
_local_model: Optional[Any] = None
_local_model_lock = threading.Lock()
_local_model_failed = False  # avoid repeated load attempts if it fails once


def _get_local_embed_model() -> Optional[Any]:
    """Lazy-load Qwen3-Embedding-0.6B via SentenceTransformer (in-process fallback)."""
    global _local_model, _local_model_failed
    if _local_model_failed:
        return None
    if _local_model is not None:
        return _local_model
    with _local_model_lock:
        if _local_model is None and not _local_model_failed:
            try:
                from sentence_transformers import SentenceTransformer

                logger.info("[EMBED-LOCAL] Loading Qwen3-Embedding-0.6B in-process (fallback)…")
                _local_model = SentenceTransformer(
                    "Qwen/Qwen3-Embedding-0.6B",
                )
                logger.success("[EMBED-LOCAL] Local embedding model ready")
            except Exception as exc:
                logger.error(f"[EMBED-LOCAL] Failed to load local model: {exc}")
                _local_model_failed = True
    return _local_model

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

    This service communicates with a local HTTP embedding server that runs
    the SentenceTransformer model Qwen/Qwen3-Embedding-0.6B.

    Reference: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
    """

    DEFAULT_TASK_INSTRUCTION = (
        "Given a medical question in Vietnamese, retrieve relevant medical knowledge "
        "passages that provide accurate information to answer the question"
    )

    def __init__(
        self,
        local_url: Optional[str] = None,
        task_instruction: Optional[str] = None,
    ):
        """
        Initialize Qwen3 Embedding Service with local HTTP endpoint.

        Args:
            local_url: URL of local embedding service (default from settings)
            task_instruction: Default task instruction for query embedding
        """
        if settings.qwen3_models_enabled:
            self.local_url = local_url or settings.qwen3_models_url
        else:
            self.local_url = local_url or settings.backend_api_url

        self.task_instruction = task_instruction or self.DEFAULT_TASK_INSTRUCTION
        self.client = httpx.Client(timeout=120.0)
        self._remote_failures = 0
        self._remote_down_until = 0.0
        self._remote_down_ttl_seconds = 60
        self._remote_failure_threshold = 3

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

        embedding = self._embed_with_local(
            texts=[query], is_query=True, instruction=instruction
        )
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
        embedding = self._embed_with_local(
            texts=[document], is_query=False, instruction=None
        )
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
            batch_embeddings = self._embed_with_local(
                texts=batch, is_query=False, instruction=None
            )
            embeddings.extend(batch_embeddings or [None] * len(batch))

        return embeddings

    def _embed_with_local(
        self,
        texts: List[str],
        is_query: bool = False,
        instruction: Optional[str] = None,
    ) -> Optional[List[List[float]]]:
        """
        Call local HTTP embedding service for inference.
        Falls back to in-process SentenceTransformer if the HTTP service is down.
        """
        now = time.time()
        remote_available = now >= self._remote_down_until

        # ── Primary path: HTTP service ────────────────────────────────────────
        if remote_available:
            try:
                payload = {
                    "texts": texts,
                    "normalize": True,
                    "is_query": is_query,
                }
                if is_query and instruction:
                    payload["instruction"] = instruction

                response = self.client.post(
                    f"{self.local_url}/v1/models/embed",
                    json=payload,
                )

                if response.status_code == 200:
                    result = response.json()
                    self._remote_failures = 0
                    return result.get("embeddings")

                self._remote_failures += 1
                logger.error(
                    f"[EMBED] HTTP service error: {response.status_code} - {response.text}"
                )
            except Exception as e:
                self._remote_failures += 1
                logger.warning(f"[EMBED] HTTP service unavailable ({e}), trying local fallback…")

            if self._remote_failures > self._remote_failure_threshold:
                self._remote_down_until = time.time() + self._remote_down_ttl_seconds
                logger.warning(
                    f"[EMBED][CB] Remote service marked DOWN for {self._remote_down_ttl_seconds}s"
                )
        else:
            logger.debug("[EMBED][CB] Skip remote call while service is DOWN")

        # ── Fallback path: in-process SentenceTransformer ─────────────────────
        model = _get_local_embed_model()
        if model is None:
            logger.error("[EMBED] Local fallback model also unavailable. Returning None.")
            return None

        try:
            encode_texts = texts
            if is_query and instruction:
                encode_texts = [f"{instruction}: {t}" for t in texts]

            vecs = model.encode(
                encode_texts,
                batch_size=32,
                convert_to_numpy=True,
                normalize_embeddings=True,
            )
            return vecs.tolist()
        except Exception as exc:
            logger.error(f"[EMBED] Local fallback encode failed: {exc}")
            return None

    def get_embedding_dimension(self) -> int:
        """Get embedding dimension (1024 for Qwen3-Embedding-0.6B)."""
        return settings.vector_dimension

    def health_check(self) -> bool:
        """Check if local embedding service is alive."""
        try:
            response = self.client.get(f"{self.local_url}/v1/ready")
            return response.status_code == 200
        except Exception as e:
            logger.error(f"[HEALTH] Error: {e}")
            return False


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
