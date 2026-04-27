"""RAG pipeline — retrieval-augmented generation for Minqes."""

import asyncio
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
import hashlib
import os
import re
import time
from typing import Any, Dict, Iterator, List

from fastapi import APIRouter, Depends
import httpx
from loguru import logger

from ..configs.setup import get_backend_settings
from ..core.metrics import (
    rag_active_requests,
    rag_cache_hits_total,
    rag_cache_misses_total,
    rag_cache_requests_total,
    rag_errors_total,
    rag_generation_duration_seconds,
    rag_llm_duration_seconds,
    rag_request_duration_seconds,
    rag_requests_total,
    rag_retrieval_duration_seconds,
    rag_tokens_generated_total,
)
from ..core.runtime_settings import get_runtime_settings
from ..core.security import get_current_user

settings = get_backend_settings()

router = APIRouter(prefix="/v1/rag", tags=["rag"])

# ────────────────────────────────────────────────────────────
#  System prompts
# ────────────────────────────────────────────────────────────

MEDICAL_SYSTEM_PROMPT = (
    "Bạn là Minqes — trợ lý y khoa thông minh của hệ thống Minqes. "
    "Bạn cung cấp thông tin y khoa chính xác, dễ hiểu bằng tiếng Việt dựa trên "
    "tài liệu y khoa được cung cấp.\n\n"
    "Nguyên tắc:\n"
    "• Luôn trả lời bằng tiếng Việt, rõ ràng và chính xác.\n"
    "• Dựa chủ yếu vào tài liệu tham khảo khi có.\n"
    "• CHỈ khi bật web search và có nguồn web, mới được dùng số trích dẫn [1], [2], ...\n"
    "• Nếu web search tắt, TUYỆT ĐỐI không chèn [1], [2] hay mục 'Nguồn tham khảo'.\n"
    "• Khi không có thông tin trong tài liệu, nói rõ giới hạn kiến thức.\n"
    "• Luôn khuyến khích tham khảo bác sĩ cho các tình huống nghiêm trọng.\n"
    "• Không đưa ra chẩn đoán bệnh cụ thể hay kê đơn thuốc. /no_think"
)

GENERAL_SYSTEM_PROMPT = (
    "Bạn là Minqes — trợ lý thông minh của hệ thống Minqes. "
    "Hãy trả lời thân thiện và hữu ích bằng tiếng Việt. /no_think"
)

MAX_CONTEXT_CHARS = 6000  # ~1500 tokens (rough)
RETRIEVAL_TIMEOUT_SECONDS = max(1.0, float(os.getenv("RETRIEVAL_TIMEOUT_SECONDS", "2.0")))


def _build_retrieval_cache_key(
    question: str,
    top_k: int,
    vector_k: int,
    bm25_k: int,
    final_k: int,
) -> str:
    raw = "::".join(
        [
            (question or "").strip().lower(),
            str(top_k),
            str(vector_k),
            str(bm25_k),
            str(final_k),
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _run_hybrid_search_with_timeout(
    query: str,
    *,
    top_k: int,
    vector_k: int,
    bm25_k: int,
    final_k: int,
) -> List[Dict[str, Any]]:
    from ..core.hybrid_search import hybrid_search

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            hybrid_search,
            query,
            top_k,
            vector_k,
            bm25_k,
            final_k,
        )
        try:
            return future.result(timeout=RETRIEVAL_TIMEOUT_SECONDS) or []
        except FutureTimeoutError:
            logger.warning(f"[TIMEOUT] retrieval timeout after {RETRIEVAL_TIMEOUT_SECONDS:.1f}s")
            return []


def _extract_tavily_query(question: str) -> str:
    text = re.sub(r"\s+", " ", (question or "")).strip()
    return text[:400]


def _normalize_tavily_results(results: List[dict], max_results: int = 2) -> List[Dict[str, Any]]:
    citations: List[Dict[str, Any]] = []
    for item in (results or [])[:max_results]:
        content = re.sub(r"\s+", " ", str(item.get("content") or "")).strip()
        snippet = content[:220] + ("…" if len(content) > 220 else "")
        citations.append(
            {
                "title": item.get("title") or "Nguồn web",
                "url": item.get("url") or "",
                "snippet": snippet,
                "content": snippet,
                "type": "web",
                "score": float(item.get("score") or 0.0),
            }
        )
    return citations


async def _tavily_search_async(query: str, max_results: int = 2) -> tuple[str, List[Dict[str, Any]]]:
    api_key = settings.tavily_api_key or os.getenv("TAVILY_API_KEY", "")
    if not api_key:
        raise RuntimeError("Missing Tavily API key")

    url = "https://api.tavily.com/search"
    payload = {
        "api_key": api_key,
        "query": query,
        "max_results": max_results,
    }
    timeout = httpx.Timeout(connect=2.0, read=6.0, write=3.0, pool=2.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()
        body = resp.json() or {}

    citations = _normalize_tavily_results(body.get("results") or [], max_results=max_results)
    observation = "\n".join(
        f"- {c.get('title', 'Nguồn web')}: {c.get('snippet', '')} ({c.get('url', '')})"
        for c in citations
    )
    return observation, citations


# ────────────────────────────────────────────────────────────
#  Public: run_rag_pipeline  (imported by the chat router)
# ────────────────────────────────────────────────────────────

def run_rag_pipeline(
    question: str,
    history: List[Dict[str, str]],
    top_k: int = 5,
    web_search_enabled: bool | None = None,
) -> Dict[str, Any]:
    """Full RAG pipeline with graceful degradation at every step.

    Steps
    -----
    1. Guardrails check         (soft-fail — continues if GPU service is down)
    2. Intent detection         (medical vs general)
    3. Query rewrite            (Gemini API with context)
    4. Hybrid search            (Qdrant vector + Elasticsearch BM25 via RRF)
    5. Reranking                (Qwen3-Reranker-0.6B)
    6. Answer generation        (vLLM → Ollama fallback)

    Returns
    -------
    dict: {answer: str, citations: list[dict], route: str}
    """
    citations: List[Dict[str, Any]] = []
    route = "medical"
    web_search_used = False
    total_start = time.perf_counter()
    rag_requests_total.inc()
    rag_active_requests.inc()

    runtime = get_runtime_settings()
    rewrite_enabled = bool(runtime.get("rewrite_enabled", True))
    rerank_enabled = bool(runtime.get("rerank_enabled", True))
    max_tokens = int(runtime.get("max_tokens", 512))

    # ── 0. Final response cache (before retrieval/LLM) ──────────────────────
    try:
        from ..services.brain import get_cached_final_response

        cached_final = get_cached_final_response(
            question=question,
            history=history,
            web_search_enabled=bool(web_search_enabled),
        )
        if cached_final:
            logger.info("[RAG] Final response cache hit")
            rag_request_duration_seconds.observe(time.perf_counter() - total_start)
            rag_active_requests.dec()
            return {
                "answer": str(cached_final.get("answer") or ""),
                "citations": list(cached_final.get("citations") or []),
                "route": str(cached_final.get("route") or "medical"),
                "web_search_used": bool(cached_final.get("web_search_used", False)),
                "web_search_reason": str(cached_final.get("web_search_reason") or "cache"),
            }
    except Exception:
        pass
    final_k = max(1, min(int(settings.final_k), max(2, top_k)))
    vector_k = max(final_k, min(int(settings.vector_k), final_k + 2))
    bm25_k = max(final_k, min(int(settings.bm25_k), final_k + 2))

    # ── 1. Guardrails ─────────────────────────────────────────────────────────
    try:
        from ..core.guardrails import get_guardrails_service

        guard = get_guardrails_service()
        logger.info("[GUARD] Input moderation started")
        is_valid, violation, metadata = guard.validate_query(question)

        if metadata and metadata.get("failover"):
            logger.warning(
                f"[GUARD] Fail-open active: {metadata.get('error', 'unknown_error')}"
            )

        if not is_valid:
            logger.warning(f"[GUARD] Query blocked: category={violation}")
            rag_active_requests.dec()
            return {
                "answer": guard.get_rejection_message(violation or "unknown"),
                "citations": [],
                "route": "blocked",
            }
        logger.info(
            f"[GUARD] Query passed: severity={(metadata or {}).get('severity', 'unknown')}"
        )
    except Exception as e:
        logger.warning(f"[GUARD] Service unavailable, fail-open in pipeline: {e}")

    # ── 2. Intent detection ──────────────────────────────────────────────────
    try:
        from ..services.brain import detect_route

        detected = detect_route(history, question)
        route = (detected or "medical").strip().lower()
        if route not in ("medical", "general"):
            route = "medical"
        logger.info(f"[RAG] Detected route: {route}")
    except Exception as e:
        logger.warning(f"[RAG] Intent detection failed, defaulting to 'medical': {e}")
        route = "medical"

    # ── 3. Query rewrite ─────────────────────────────────────────────────────
    rewritten_query = question
    if rewrite_enabled:
        try:
            from ..services.rewrite_service import rewrite_query_with_api

            rewritten_query = rewrite_query_with_api(question, history) or question
        except Exception as e:
            logger.warning(f"[RAG] Query rewrite failed, using original: {e}")
            rewritten_query = question

    # ── 4. Smart routing: if web search enabled, skip local RAG retrieval ───
    context = ""
    tavily_enabled = bool(settings.tavily_api_key or os.getenv("TAVILY_API_KEY"))
    use_tavily = bool(web_search_enabled) and tavily_enabled
    tavily_reason = ""
    if web_search_enabled is True and not tavily_enabled:
        tavily_reason = "tavily_unavailable"
    elif web_search_enabled is True and tavily_enabled:
        tavily_reason = "client_enabled"
    else:
        tavily_reason = "client_disabled"

    if use_tavily:
        try:
            from ..services.brain import get_response

            logger.info(f"[RAG] 🌐 Tavily enabled ({tavily_reason})")
            search_query = _extract_tavily_query(question)
            observation, web_citations = asyncio.run(
                _tavily_search_async(search_query, max_results=2)
            )
            if not web_citations:
                raise RuntimeError("No Tavily results")

            web_messages = [
                {
                    "role": "system",
                    "content": (
                        "Bạn là trợ lý y tế Việt Nam. Trả lời dựa trên kết quả web, "
                        "chính xác, ngắn gọn, và trích dẫn nguồn URL trong câu trả lời."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Kết quả web:\n{observation}\n\n"
                        f"Câu hỏi: {question}\n"
                        "Hãy trả lời bằng tiếng Việt."
                    ),
                },
            ]
            web_answer = get_response(web_messages, temperature=0.3, max_tokens=512)
            if not web_answer:
                raise RuntimeError("LLM unavailable for web answer")

            web_search_used = True
            result_payload = {
                "answer": web_answer,
                "citations": web_citations,
                "route": f"{route}_web",
                "web_search_used": web_search_used,
                "web_search_reason": tavily_reason,
            }
            try:
                from ..services.brain import cache_final_response

                cache_final_response(
                    question=question,
                    history=history,
                    web_search_enabled=True,
                    payload=result_payload,
                )
            except Exception:
                pass

            logger.info(f"[PERF] total_time={time.perf_counter() - total_start:.3f}s")
            rag_request_duration_seconds.observe(time.perf_counter() - total_start)
            rag_active_requests.dec()
            return result_payload
        except Exception as e:
            logger.warning(f"[RAG] Tavily failed, return immediate fallback: {e}")
            rag_errors_total.inc()
            rag_request_duration_seconds.observe(time.perf_counter() - total_start)
            rag_active_requests.dec()
            return {
                "answer": "Xin lỗi, web search hiện không khả dụng. Vui lòng thử lại sau.",
                "citations": [],
                "route": f"{route}_web",
                "web_search_used": False,
                "web_search_reason": "tavily_failed",
            }

    # ── 5. Hybrid search + rerank (only when web search is OFF) ─────────────
    retrieval_start = time.perf_counter()
    if route == "medical" and not bool(web_search_enabled):
        try:
            from ..core.cache import cache_search_results, get_search_results

            retrieval_cache_key = _build_retrieval_cache_key(
                question=rewritten_query,
                top_k=top_k,
                vector_k=vector_k,
                bm25_k=bm25_k,
                final_k=final_k,
            )

            raw_results = get_search_results(retrieval_cache_key)
            rag_cache_requests_total.inc()
            if raw_results is None:
                logger.info(f"[CACHE] retrieval miss: {retrieval_cache_key[:16]}...")
                rag_cache_misses_total.inc()
                raw_results = _run_hybrid_search_with_timeout(
                    query=rewritten_query,
                    top_k=final_k,
                    vector_k=vector_k,
                    bm25_k=bm25_k,
                    final_k=final_k,
                )
                cache_search_results(
                    retrieval_cache_key,
                    raw_results or [],
                    ttl_seconds=600,
                )
            else:
                logger.info(f"[CACHE] retrieval hit: {retrieval_cache_key[:16]}...")
                rag_cache_hits_total.inc()

            if raw_results:
                if rerank_enabled:
                    results = _rerank(rewritten_query, raw_results, final_k)
                else:
                    results = raw_results[:final_k]
                context, citations = _build_context(results)
        except Exception as e:
            logger.error(f"[ERROR] [FALLBACK] [RAG] Search pipeline failed, generating without context: {e}")
            rag_errors_total.inc()
    retrieval_duration = time.perf_counter() - retrieval_start
    rag_retrieval_duration_seconds.observe(retrieval_duration)
    logger.info(f"[PERF] retrieval_time={retrieval_duration:.3f}s")

    # ── 6. Generation ─────────────────────────────────────────────────────────
    messages = _build_generation_messages(
        route,
        history,
        question,
        context,
        web_search_enabled=bool(web_search_enabled),
        allow_numeric_citations=False,
    )
    try:
        from ..services.brain import get_response

        llm_start = time.perf_counter()
        answer = get_response(messages, temperature=0.3, max_tokens=max_tokens)
        llm_duration = time.perf_counter() - llm_start
        rag_llm_duration_seconds.observe(llm_duration)
        rag_generation_duration_seconds.observe(llm_duration)
        if answer:
            rag_tokens_generated_total.inc(max(1, len(str(answer).split())))
        logger.info(f"[PERF] llm_time={llm_duration:.3f}s")
    except Exception as e:
        logger.error(f"[RAG] Generation failed: {e}")
        rag_errors_total.inc()
        answer = None

    if not answer:
        answer = (
            "Xin lỗi, hệ thống tạo phản hồi đang gặp sự cố. "
            "Vui lòng thử lại sau vài giây."
        )
        citations = []

    result_payload = {
        "answer": answer,
        # Never expose local RAG citations to client payload.
        "citations": [],
        "route": route,
        "web_search_used": web_search_used,
        "web_search_reason": tavily_reason,
    }

    try:
        from ..services.brain import cache_final_response

        cache_final_response(
            question=question,
            history=history,
            web_search_enabled=bool(web_search_enabled),
            payload=result_payload,
        )
    except Exception:
        pass

    total_duration = time.perf_counter() - total_start
    rag_request_duration_seconds.observe(total_duration)
    logger.info(f"[PERF] total_time={total_duration:.3f}s")
    rag_active_requests.dec()
    return result_payload


def run_rag_pipeline_stream(
    question: str,
    history: List[Dict[str, str]],
    top_k: int = 5,
    web_search_enabled: bool | None = None,
) -> Dict[str, Any]:
    """Streaming variant of the RAG pipeline.

    Returns a dict with:
      - answer_stream: Iterator[str]
      - answer_parts: list[str] (filled while streaming)
      - route, citations, web_search_used, web_search_reason
    """
    citations: List[Dict[str, Any]] = []
    route = "medical"
    web_search_used = False
    total_start = time.perf_counter()
    rag_requests_total.inc()
    rag_active_requests.inc()

    runtime = get_runtime_settings()
    rewrite_enabled = bool(runtime.get("rewrite_enabled", True))
    rerank_enabled = bool(runtime.get("rerank_enabled", True))
    max_tokens = int(runtime.get("max_tokens", 512))

    try:
        from ..services.brain import get_cached_final_response

        cached_final = get_cached_final_response(
            question=question,
            history=history,
            web_search_enabled=bool(web_search_enabled),
        )
        if cached_final and cached_final.get("answer"):
            cached_answer = str(cached_final.get("answer"))

            def _cached_stream() -> Iterator[str]:
                for part in cached_answer.split(" "):
                    yield f"{part} "

            rag_active_requests.dec()
            return {
                "answer_stream": _cached_stream(),
                "answer_parts": [cached_answer],
                "citations": list(cached_final.get("citations") or []),
                "route": str(cached_final.get("route") or "medical"),
                "web_search_used": bool(cached_final.get("web_search_used", False)),
                "web_search_reason": "cache",
                "static_answer": cached_answer,
            }
    except Exception:
        pass
    final_k = max(1, min(int(settings.final_k), max(2, top_k)))
    vector_k = max(final_k, min(int(settings.vector_k), final_k + 2))
    bm25_k = max(final_k, min(int(settings.bm25_k), final_k + 2))

    try:
        from ..core.guardrails import get_guardrails_service

        guard = get_guardrails_service()
        is_valid, violation, _metadata = guard.validate_query(question)
        if not is_valid:
            rejected = guard.get_rejection_message(violation or "unknown")

            def _blocked_stream() -> Iterator[str]:
                yield rejected

            rag_active_requests.dec()
            return {
                "answer_stream": _blocked_stream(),
                "answer_parts": [rejected],
                "citations": [],
                "route": "blocked",
                "web_search_used": False,
                "web_search_reason": "guard_blocked",
            }
    except Exception as e:
        logger.warning(f"[GUARD][STREAM] Service unavailable, fail-open: {e}")

    try:
        from ..services.brain import detect_route

        detected = detect_route(history, question)
        route = (detected or "medical").strip().lower()
        if route not in ("medical", "general"):
            route = "medical"
    except Exception as e:
        logger.warning(f"[RAG][STREAM] Intent detection failed: {e}")
        route = "medical"

    rewritten_query = question
    if rewrite_enabled:
        try:
            from ..services.rewrite_service import rewrite_query_with_api

            rewritten_query = rewrite_query_with_api(question, history) or question
        except Exception as e:
            logger.warning(f"[RAG][STREAM] Query rewrite failed: {e}")
            rewritten_query = question

    context = ""
    tavily_enabled = bool(settings.tavily_api_key or os.getenv("TAVILY_API_KEY"))
    use_tavily = bool(web_search_enabled) and tavily_enabled
    if web_search_enabled is True and not tavily_enabled:
        tavily_reason = "tavily_unavailable"
    elif web_search_enabled is True and tavily_enabled:
        tavily_reason = "client_enabled"
    else:
        tavily_reason = "client_disabled"

    if use_tavily:
        try:
            from ..services.brain import get_response

            search_query = _extract_tavily_query(question)
            observation, web_citations = asyncio.run(
                _tavily_search_async(search_query, max_results=2)
            )
            if not web_citations:
                raise RuntimeError("No Tavily results")

            web_messages = [
                {
                    "role": "system",
                    "content": "Bạn là trợ lý y tế Việt Nam, trả lời ngắn gọn và trích dẫn URL.",
                },
                {
                    "role": "user",
                    "content": f"Kết quả web:\n{observation}\n\nCâu hỏi: {question}",
                },
            ]
            web_answer = get_response(web_messages, temperature=0.3, max_tokens=512)

            if web_answer and not web_answer.startswith("Xin lỗi"):
                web_search_used = True

                def _web_stream() -> Iterator[str]:
                    # lightweight chunking for non-streaming Tavily branch
                    for part in web_answer.split(" "):
                        yield f"{part} "

                rag_active_requests.dec()
                return {
                    "answer_stream": _web_stream(),
                    "answer_parts": [],
                    "citations": web_citations,
                    "route": f"{route}_web",
                    "web_search_used": web_search_used,
                    "web_search_reason": tavily_reason,
                    "static_answer": web_answer,
                }
            raise RuntimeError("LLM unavailable for web answer")
        except Exception as e:
            logger.warning(f"[RAG][STREAM] Tavily failed: {e}")

            def _fallback_web_stream() -> Iterator[str]:
                yield "Xin lỗi, web search hiện không khả dụng. Vui lòng thử lại sau."

            rag_active_requests.dec()
            return {
                "answer_stream": _fallback_web_stream(),
                "answer_parts": ["Xin lỗi, web search hiện không khả dụng. Vui lòng thử lại sau."],
                "citations": [],
                "route": f"{route}_web",
                "web_search_used": False,
                "web_search_reason": "tavily_failed",
            }

    retrieval_start = time.perf_counter()
    if route == "medical" and not bool(web_search_enabled):
        try:
            from ..core.cache import cache_search_results, get_search_results

            retrieval_cache_key = _build_retrieval_cache_key(
                question=rewritten_query,
                top_k=top_k,
                vector_k=vector_k,
                bm25_k=bm25_k,
                final_k=final_k,
            )

            raw_results = get_search_results(retrieval_cache_key)
            rag_cache_requests_total.inc()
            if raw_results is None:
                logger.info(f"[CACHE] retrieval miss: {retrieval_cache_key[:16]}...")
                rag_cache_misses_total.inc()
                raw_results = _run_hybrid_search_with_timeout(
                    query=rewritten_query,
                    top_k=final_k,
                    vector_k=vector_k,
                    bm25_k=bm25_k,
                    final_k=final_k,
                )
                cache_search_results(
                    retrieval_cache_key,
                    raw_results or [],
                    ttl_seconds=600,
                )
            else:
                logger.info(f"[CACHE] retrieval hit: {retrieval_cache_key[:16]}...")
                rag_cache_hits_total.inc()

            if raw_results:
                if rerank_enabled:
                    results = _rerank(rewritten_query, raw_results, final_k)
                else:
                    results = raw_results[:final_k]
                context, citations = _build_context(results)
        except Exception as e:
            logger.error(f"[ERROR] [FALLBACK] [RAG][STREAM] Search pipeline failed: {e}")
            rag_errors_total.inc()
    retrieval_duration = time.perf_counter() - retrieval_start
    rag_retrieval_duration_seconds.observe(retrieval_duration)
    logger.info(f"[PERF] retrieval_time_stream={retrieval_duration:.3f}s")

    messages = _build_generation_messages(
        route,
        history,
        question,
        context,
        web_search_enabled=bool(web_search_enabled),
        allow_numeric_citations=False,
    )

    from ..services.brain import get_response_stream

    answer_parts: List[str] = []

    def _answer_stream() -> Iterator[str]:
        logger.info("[STREAM_START] rag_answer_stream")
        llm_start = time.perf_counter()
        token_count = 0
        try:
            for token in get_response_stream(messages, temperature=0.3, max_tokens=max_tokens):
                if not token:
                    continue
                answer_parts.append(token)
                token_count += 1
                yield token
            llm_duration = time.perf_counter() - llm_start
            rag_llm_duration_seconds.observe(llm_duration)
            rag_generation_duration_seconds.observe(llm_duration)
            rag_tokens_generated_total.inc(max(1, token_count))
            logger.info(f"[PERF] llm_time_stream={llm_duration:.3f}s")
            logger.info(f"[STREAM_END] rag_answer_stream tokens={token_count}")
        except Exception as exc:
            rag_errors_total.inc()
            logger.error(f"[STREAM_ERROR] rag_answer_stream: {exc}")
            raise
        finally:
            rag_request_duration_seconds.observe(time.perf_counter() - total_start)
            rag_active_requests.dec()

    return {
        "answer_stream": _answer_stream(),
        "answer_parts": answer_parts,
        # Never expose local RAG citations to client payload.
        "citations": [],
        "route": route,
        "web_search_used": web_search_used,
        "web_search_reason": tavily_reason,
    }


# ────────────────────────────────────────────────────────────
#  Private helpers
# ────────────────────────────────────────────────────────────

def _rerank(
    query: str,
    raw_results: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Rerank raw search results; falls back to original order on error."""
    try:
        from ..services.rerank import Qwen3RerankerService

        reranker = Qwen3RerankerService()
        reranked_items, _ = reranker.rerank(query, raw_results, top_n=top_k)
        docs: List[Dict[str, Any]] = []
        for item in reranked_items[:top_k]:
            idx = item.get("index", 0)
            if 0 <= idx < len(raw_results):
                doc = raw_results[idx].copy()
                doc["relevance_score"] = item.get("relevance_score", 0.0)
                docs.append(doc)
        return docs if docs else raw_results[:top_k]
    except Exception as e:
        logger.warning(f"[RAG] Reranker unavailable, using raw search order: {e}")
        return raw_results[:top_k]


def _build_context(
    results: List[Dict[str, Any]],
) -> tuple[str, List[Dict[str, Any]]]:
    """Convert retrieved docs into a context string and a citations list."""
    context_parts: List[str] = []
    citations: List[Dict[str, Any]] = []

    for i, doc in enumerate(results, 1):
        title = doc.get("title") or doc.get("file_name") or f"Tài liệu {i}"
        content_text = doc.get("content") or doc.get("text", "")
        # Truncate individual doc content to keep total context manageable
        content_text = content_text[:800] if len(content_text) > 800 else content_text
        source = doc.get("source") or doc.get("file_name", "")
        score = float(
            doc.get("relevance_score")
            or doc.get("rrf_score")
            or doc.get("score", 0.0)
        )
        context_parts.append(f"**Tài liệu {i}: {title}**\n{content_text}")
        citations.append(
            {
                "title": title,
                "content": content_text[:300],
                "source": source,
                "score": round(score, 4),
            }
        )

    return "\n\n---\n\n".join(context_parts), citations


def _build_generation_messages(
    route: str,
    history: List[Dict[str, str]],
    question: str,
    context: str,
    web_search_enabled: bool = False,
    allow_numeric_citations: bool = False,
) -> List[Dict[str, str]]:
    """Compose the OpenAI-style message list sent to the LLM."""
    system_prompt = MEDICAL_SYSTEM_PROMPT if route == "medical" else GENERAL_SYSTEM_PROMPT
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    ua_history = [m for m in history if m.get("role") in ("user", "assistant")]
    if len(ua_history) > 6:
        summary_text = _summarize_history(ua_history[:-4])
        if summary_text:
            messages.append({
                "role": "assistant",
                "content": f"Tóm tắt hội thoại trước đó: {summary_text}",
            })

    recent = _select_recent_history(ua_history)
    messages.extend(recent)

    compact_context = (context or "")[:MAX_CONTEXT_CHARS]

    if compact_context:
        user_content = (
            f"**Tài liệu tham khảo:**\n\n{compact_context}\n\n"
            f"---\n\n**Câu hỏi:** {question}\n\n"
            "Dựa vào tài liệu tham khảo trên và kiến thức y khoa của bạn, "
            "hãy trả lời câu hỏi bằng tiếng Việt một cách đầy đủ và chính xác. "
            + (
                "Sử dụng số [1], [2], ... để trích dẫn khi cần và thêm mục Nguồn tham khảo ở cuối."
                if web_search_enabled and allow_numeric_citations
                else "Không chèn bất kỳ số trích dẫn dạng [n] nào và không thêm mục Nguồn tham khảo."
            )
        )
    else:
        user_content = (
            f"{question}\n\n"
            + (
                "Nếu không có web search, không được thêm [1], [2] hoặc Nguồn tham khảo."
                if not web_search_enabled
                else ""
            )
        ).strip()

    messages.append({"role": "user", "content": user_content})
    return messages


def _select_recent_history(history: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Keep last 2 user + last 2 assistant messages in chronological order."""
    users = [m for m in history if m.get("role") == "user"][-2:]
    assistants = [m for m in history if m.get("role") == "assistant"][-2:]
    selected_ids = {id(m) for m in users + assistants}
    return [m for m in history if id(m) in selected_ids]


def _summarize_history(history: List[Dict[str, str]]) -> str:
    """Short rule-based summary for older context without extra model calls."""
    if not history:
        return ""

    texts: List[str] = []
    for m in history[-6:]:
        content = re.sub(r"\s+", " ", (m.get("content") or "").strip())
        if content:
            texts.append(content[:120])
    if not texts:
        return ""

    summary = "; ".join(texts)
    return summary[:320]


# ────────────────────────────────────────────────────────────
#  REST endpoint  (direct test — no DB persistence)
# ────────────────────────────────────────────────────────────

@router.post("/query")
async def rag_query(
    body: dict,
    current_user=Depends(get_current_user),
):
    """Standalone RAG query — useful for pipeline smoke-testing.

    Body: { "question": str, "history": list[{role, content}] }
    """
    question = body.get("question", "").strip()
    history = body.get("history", [])
    web_search_enabled = body.get("web_search_enabled")
    if not question:
        return {"answer": "Vui lòng nhập câu hỏi.", "citations": [], "route": ""}

    logger.info(f"[RAG] Direct query from user {current_user.id}: {question[:80]}")
    result = await asyncio.to_thread(
        run_rag_pipeline,
        question,
        history,
        5,
        web_search_enabled,
    )

    return {
        "answer": result["answer"],
        "citations": result["citations"],
        "metadata": {
            "route": result["route"],
            "retrieval_count": len(result["citations"]),
            "web_search_used": result.get("web_search_used", False),
            "web_search_reason": result.get("web_search_reason", ""),
        },
    }
