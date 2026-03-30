"""RAG pipeline — retrieval-augmented generation for Minqes."""

import os
from typing import Any, Dict, List

from fastapi import APIRouter, Depends
from loguru import logger

from ..configs.setup import get_backend_settings
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
    "• Dựa chủ yếu vào tài liệu tham khảo; trích dẫn bằng [1], [2], ...\n"
    "• Khi không có thông tin trong tài liệu, nói rõ giới hạn kiến thức.\n"
    "• Luôn khuyến khích tham khảo bác sĩ cho các tình huống nghiêm trọng.\n"
    "• Không đưa ra chẩn đoán bệnh cụ thể hay kê đơn thuốc. /no_think"
)

GENERAL_SYSTEM_PROMPT = (
    "Bạn là Minqes — trợ lý thông minh của hệ thống Minqes. "
    "Hãy trả lời thân thiện và hữu ích bằng tiếng Việt. /no_think"
)


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
    3. Query enhancement        (rewrite with conversation context)
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

    # ── 3. Query enhancement ─────────────────────────────────────────────────
    enhanced_query = question
    if history:
        try:
            from ..services.brain import enhance_query_quality

            enhanced_query = enhance_query_quality(history, question) or question
            logger.debug(f"[RAG] Enhanced query: {enhanced_query[:120]}")
        except Exception as e:
            logger.warning(f"[RAG] Query enhancement failed, using original: {e}")

    # ── 4 & 5. Hybrid search + Reranking  (medical route only) ──────────────
    context = ""
    retrieval_confidence = 0.0
    if route == "medical":
        try:
            from ..core.hybrid_search import hybrid_search

            raw_results = hybrid_search(enhanced_query, top_k=top_k * 2)
            logger.info(f"[RAG] Retrieved {len(raw_results)} raw candidates")
            if raw_results:
                results = _rerank(enhanced_query, raw_results, top_k)
                context, citations = _build_context(results)
                if results:
                    retrieval_confidence = float(
                        results[0].get("relevance_score")
                        or results[0].get("rrf_score")
                        or results[0].get("score", 0.0)
                    )
                logger.info(f"[RAG] Using {len(citations)} documents as context")
        except Exception as e:
            logger.error(f"[RAG] Search pipeline failed, generating without context: {e}")

    # ── 5.5 Optional Tavily fallback ─────────────────────────────────────────
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
            from ..services.brain import get_tavily_agent_answer_with_sources

            logger.info(f"[RAG] 🌐 Tavily fallback enabled ({tavily_reason})")
            messages_for_web = _build_generation_messages(
                route=route,
                history=history,
                question=question,
                context=context,
            )
            web_result = get_tavily_agent_answer_with_sources(messages_for_web)
            web_answer = (web_result or {}).get("answer", "")
            web_citations = (web_result or {}).get("citations", [])

            if web_answer and not web_answer.startswith("Xin lỗi"):
                web_search_used = True
                return {
                    "answer": web_answer,
                    "citations": web_citations,
                    "route": f"{route}_web",
                    "web_search_used": web_search_used,
                    "web_search_reason": tavily_reason,
                }
            logger.warning("[RAG] Tavily returned empty/error answer, fallback to normal generation")
        except Exception as e:
            logger.warning(f"[RAG] Tavily fallback failed, continue with local generation: {e}")

    # ── 6. Generation ─────────────────────────────────────────────────────────
    messages = _build_generation_messages(route, history, question, context)
    try:
        from ..services.brain import get_response

        answer = get_response(messages, temperature=0.7, max_tokens=2048)
    except Exception as e:
        logger.error(f"[RAG] Generation failed: {e}")
        answer = None

    if not answer:
        answer = (
            "Xin lỗi, hệ thống tạo phản hồi đang gặp sự cố. "
            "Vui lòng thử lại sau vài giây."
        )
        citations = []

    return {
        "answer": answer,
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
        context_parts.append(f"[{i}] **{title}**\n{content_text}")
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
) -> List[Dict[str, str]]:
    """Compose the OpenAI-style message list sent to the LLM."""
    system_prompt = MEDICAL_SYSTEM_PROMPT if route == "medical" else GENERAL_SYSTEM_PROMPT
    messages: List[Dict[str, str]] = [{"role": "system", "content": system_prompt}]

    # Include the last 6 conversation turns for context
    recent = [m for m in history if m.get("role") in ("user", "assistant")][-6:]
    messages.extend(recent)

    if context:
        user_content = (
            f"**Tài liệu tham khảo:**\n\n{context}\n\n"
            f"---\n\n**Câu hỏi:** {question}\n\n"
            "Dựa vào tài liệu tham khảo trên và kiến thức y khoa của bạn, "
            "hãy trả lời câu hỏi bằng tiếng Việt một cách đầy đủ và chính xác. "
            "Sử dụng số [1], [2], ... để trích dẫn tài liệu khi cần."
        )
    else:
        user_content = question

    messages.append({"role": "user", "content": user_content})
    return messages


# ────────────────────────────────────────────────────────────
#  REST endpoint  (direct test — no DB persistence)
# ────────────────────────────────────────────────────────────

@router.post("/query")
def rag_query(
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
    result = run_rag_pipeline(
        question=question,
        history=history,
        web_search_enabled=web_search_enabled,
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
