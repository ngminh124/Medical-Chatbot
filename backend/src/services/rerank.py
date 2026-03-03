"""Qwen3 Reranker Service with intelligent keyword-based fallback.

When the GPU reranker service (port 7860) is unavailable, the fallback uses
Vietnamese-aware keyword overlap scoring instead of returning documents in
arbitrary order.
"""

import math
import re
import unicodedata
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import httpx
from loguru import logger

from ..configs.setup import get_backend_settings
from ..core.model_config import get_reranking_model

settings = get_backend_settings()

# ── Vietnamese stop-words (common particles that add no retrieval value) ──
_VI_STOPWORDS: set[str] = {
    "và", "của", "là", "các", "có", "được", "cho", "trong", "này", "với",
    "không", "một", "để", "khi", "từ", "theo", "đã", "những", "về", "hay",
    "như", "hoặc", "tại", "do", "cũng", "vào", "ra", "lên", "bị", "đến",
    "nên", "mà", "thì", "bạn", "tôi", "rất", "hơn", "nhất", "nào", "đó",
    "còn", "chỉ", "sẽ", "đang", "vì", "nếu", "sau", "trước", "thể",
    "người", "năm", "ngày", "thời", "gian",
}


class Qwen3RerankerService:
    """
    Qwen3 Reranker Service following official Qwen3-Reranker-0.6B best practices.

    Key Features:
    - Format: <Instruct>: {instruction}\\n<Query>: {query}\\n<Document>: {doc}
    - System prompt: "Judge whether the Document meets the requirements..."
    - Output: "yes"/"no" tokens with logprobs for scoring
    - Instruction-aware: Custom instructions improve performance by 1-5%
    - **Fallback**: keyword overlap scoring when GPU service is unavailable

    Reference: https://huggingface.co/Qwen/Qwen3-Reranker-0.6B
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
        """Initialize Qwen3 Reranker Service with local GPU service."""
        if settings.qwen3_models_enabled:
            self.local_url = local_url or settings.qwen3_models_url
        else:
            self.local_url = local_url or settings.backend_api_url

        self.rerank_endpoint = f"{self.local_url}/v1/models/rerank"
        self.huggingface_model = get_reranking_model()
        self.task_instruction = task_instruction or self.DEFAULT_TASK_INSTRUCTION
        self.client = httpx.Client(timeout=360.0)
        self._gpu_healthy: Optional[bool] = None  # cache health status

        logger.info(
            f"[RERANK] Initialized — endpoint: {self.rerank_endpoint}, "
            f"model: {self.huggingface_model}"
        )

    # ────────────────────────────────────────────────────────────
    #  Health check
    # ────────────────────────────────────────────────────────────

    def is_healthy(self) -> bool:
        """Quick check whether the GPU rerank service is reachable."""
        try:
            r = self.client.get(
                f"{self.local_url}/health",
                timeout=3.0,
            )
            healthy = r.status_code == 200
            self._gpu_healthy = healthy
            return healthy
        except Exception:
            self._gpu_healthy = False
            return False

    # ────────────────────────────────────────────────────────────
    #  Public API
    # ────────────────────────────────────────────────────────────

    def rerank(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: int = 5,
        task_instruction: Optional[str] = None,
    ) -> Tuple[List[Dict[str, Any]], str]:
        """Rerank documents — GPU service first, keyword fallback second."""
        instruction = task_instruction or self.task_instruction

        # ── Try GPU reranker ──────────────────────────────────────────────
        try:
            reranked_results = self._rerank_with_local(
                query, documents, top_n, instruction
            )
            logger.info(
                f"[RERANK] GPU success — top {len(reranked_results)} docs, "
                f"best score: {reranked_results[0]['relevance_score']:.4f}"
            )
            rerank_context = self._format_rerank_context(documents, reranked_results)
            return reranked_results, rerank_context
        except httpx.ConnectError as e:
            logger.warning(
                f"[RERANK] GPU service unreachable at {self.rerank_endpoint}: {e}"
            )
        except httpx.TimeoutException as e:
            logger.warning(
                f"[RERANK] GPU service timeout at {self.rerank_endpoint}: {e}"
            )
        except Exception as e:
            logger.error(
                f"[RERANK] GPU reranking error at {self.rerank_endpoint}: {e}"
            )

        # ── Keyword-based fallback ────────────────────────────────────────
        logger.info("[RERANK] Falling back to keyword overlap scoring")
        fallback_results = self._rerank_keyword_fallback(query, documents, top_n)
        rerank_context = self._format_rerank_context(documents, fallback_results)
        return fallback_results, rerank_context

    # ────────────────────────────────────────────────────────────
    #  GPU reranking
    # ────────────────────────────────────────────────────────────

    def _rerank_with_local(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: int,
        instruction: str,
    ) -> List[Dict[str, Any]]:
        """Call local GPU service for Qwen3-Reranker-0.6B inference."""
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

        response = self.client.post(self.rerank_endpoint, json=payload)

        if response.status_code != 200:
            raise Exception(
                f"GPU reranking HTTP {response.status_code}: "
                f"{response.text[:200]}"
            )

        result = response.json()
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

    # ────────────────────────────────────────────────────────────
    #  Keyword-based fallback reranker
    # ────────────────────────────────────────────────────────────

    def _rerank_keyword_fallback(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_n: int,
    ) -> List[Dict[str, Any]]:
        """Score documents by TF-IDF-inspired keyword overlap with the query.

        Scoring components (combined):
        1. **BM25-lite** — term frequency with saturation + IDF weighting
        2. **Title bonus** — extra weight if query terms appear in the title
        3. **Exact phrase bonus** — boost for multi-word query matches
        4. **Topic coherence penalty** — penalise docs whose title mentions
           a *different* disease/topic from the query intent
        """
        query_tokens = self._tokenize(query)
        if not query_tokens:
            # Can't score → return in original order
            return [
                {"index": i, "relevance_score": 0.0, "document": doc}
                for i, doc in enumerate(documents[:top_n])
            ]

        # Pre-tokenise all documents
        n_docs = len(documents)
        doc_token_lists: List[List[str]] = []
        doc_title_tokens: List[set[str]] = []
        doc_texts_raw: List[str] = []
        doc_titles_raw: List[str] = []

        for doc in documents:
            title = doc.get("title", "") or ""
            content = doc.get("content", "") or doc.get("text", "") or ""
            combined = f"{title} {content}"
            doc_texts_raw.append(combined.lower())
            doc_titles_raw.append(title.lower())
            tokens = self._tokenize(combined)
            doc_token_lists.append(tokens)
            doc_title_tokens.append(set(self._tokenize(title)))

        # IDF: log(N / df) for each query term
        query_set = set(query_tokens)
        df: Dict[str, int] = {}
        for qt in query_set:
            df[qt] = sum(1 for tl in doc_token_lists if qt in set(tl))

        idf: Dict[str, float] = {}
        for qt in query_set:
            idf[qt] = math.log((n_docs + 1) / (df[qt] + 1)) + 1.0

        # Detect the main medical topic in the query for coherence checking
        query_lower = query.lower()
        query_medical_terms = self._extract_medical_terms(query_lower)

        # Score each document
        scored: List[Tuple[int, float]] = []

        # BM25 parameters
        k1 = 1.5
        b = 0.75
        avg_dl = max(1, sum(len(tl) for tl in doc_token_lists) / max(1, n_docs))

        for i, tokens in enumerate(doc_token_lists):
            tf_counter = Counter(tokens)
            dl = len(tokens)
            score = 0.0

            # ── Component 1: BM25-lite ──
            for qt in query_set:
                tf = tf_counter.get(qt, 0)
                if tf > 0:
                    tf_norm = (tf * (k1 + 1)) / (
                        tf + k1 * (1 - b + b * dl / avg_dl)
                    )
                    score += idf[qt] * tf_norm

            # ── Component 2: Title bonus (×1.5 weight for title matches) ──
            title_hits = len(query_set & doc_title_tokens[i])
            if title_hits:
                score += title_hits * 1.5

            # ── Component 3: Exact phrase bonus ──
            if len(query_tokens) >= 2 and query_lower in doc_texts_raw[i]:
                score += 3.0

            # ── Component 4: Topic coherence penalty ──
            # If the query is about topic A (e.g. "dinh dưỡng") but the
            # document title is clearly about topic B (e.g. "tiểu đường",
            # "insulin"), apply a heavy penalty to avoid noise.
            if query_medical_terms:
                doc_medical_terms = self._extract_medical_terms(doc_titles_raw[i])
                if doc_medical_terms and not (query_medical_terms & doc_medical_terms):
                    # Document title has medical terms but NONE overlap with query
                    score *= 0.3  # 70% penalty

            scored.append((i, score))

        # Sort descending by score
        scored.sort(key=lambda x: x[1], reverse=True)

        # Normalise scores to [0, 1] for consistency with GPU reranker
        max_score = scored[0][1] if scored else 1.0
        if max_score <= 0:
            max_score = 1.0

        results: List[Dict[str, Any]] = []
        for idx, raw_score in scored[:top_n]:
            results.append(
                {
                    "index": idx,
                    "relevance_score": round(raw_score / max_score, 4),
                    "document": documents[idx],
                }
            )

        best = results[0] if results else None
        logger.info(
            f"[RERANK] Keyword fallback: scored {n_docs} docs, "
            f"returning top {len(results)}, "
            f"best score: {best['relevance_score']:.4f} "
            f"(doc index {best['index']})"
            if best
            else "[RERANK] Keyword fallback: no results"
        )
        return results

    # ────────────────────────────────────────────────────────────
    #  Text utilities
    # ────────────────────────────────────────────────────────────

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Simple Vietnamese-aware tokeniser.

        - Lowercases and strips diacritics for fuzzy matching
        - Keeps original Vietnamese tokens for exact matching
        - Removes stop-words and single-character tokens
        """
        text = text.lower()
        # Split on non-alphanumeric (keeping Vietnamese characters)
        raw_tokens = re.findall(r"[\w]+", text, re.UNICODE)
        # Filter stop-words and very short tokens
        tokens = [
            t
            for t in raw_tokens
            if t not in _VI_STOPWORDS and len(t) > 1
        ]
        return tokens

    @staticmethod
    def _remove_diacritics(text: str) -> str:
        """Remove Vietnamese diacritics for fuzzy comparison."""
        nfkd = unicodedata.normalize("NFKD", text)
        return "".join(c for c in nfkd if not unicodedata.combining(c))

    @staticmethod
    def _extract_medical_terms(text: str) -> set[str]:
        """Extract recognisable medical / disease terms from text.

        Used by the topic-coherence penalty to detect when a document is about
        a completely different disease than the query is asking about.
        """
        # Common Vietnamese medical terms and disease names
        _MEDICAL_TERMS = {
            # Diseases
            "tiểu đường", "đái tháo đường", "insulin", "đường huyết",
            "huyết áp", "cao huyết áp", "tăng huyết áp", "hạ huyết áp",
            "hen suyễn", "hen phế quản", "viêm phổi", "viêm phế quản",
            "sốt xuất huyết", "sốt rét", "covid", "cúm",
            "ung thư", "u bướu", "khối u",
            "tiêu chảy", "táo bón", "viêm dạ dày", "trào ngược",
            "suy thận", "suy gan", "suy tim", "suy hô hấp",
            "đột quỵ", "tai biến", "nhồi máu cơ tim",
            "viêm khớp", "loãng xương", "gout", "thoái hóa",
            "trầm cảm", "lo âu", "mất ngủ", "động kinh",
            "dị ứng", "lupus", "viêm da",
            "thiếu máu", "xuất huyết", "hemophilia",
            "lao", "hiv", "viêm gan",
            # General health topics
            "dinh dưỡng", "chế độ ăn", "thể dục", "vận động",
            "giấc ngủ", "nhịp sinh học",
            "thai kỳ", "mang thai", "sinh nở",
            "tiêm chủng", "vaccine", "vắc xin",
        }
        found = set()
        for term in _MEDICAL_TERMS:
            if term in text:
                found.add(term)
        return found

    # ────────────────────────────────────────────────────────────
    #  Context formatter
    # ────────────────────────────────────────────────────────────

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


# ────────────────────────────────────────────────────────────
#  Singleton accessor
# ────────────────────────────────────────────────────────────

_qwen3_reranker_instance = None


def get_qwen3_reranker() -> Qwen3RerankerService:
    global _qwen3_reranker_instance
    if _qwen3_reranker_instance is None:
        _qwen3_reranker_instance = Qwen3RerankerService()
    return _qwen3_reranker_instance