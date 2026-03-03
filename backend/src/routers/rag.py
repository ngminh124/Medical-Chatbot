"""RAG pipeline — retrieval-augmented generation for Vietnamese Medical Chatbot."""

import os
from typing import Any, Dict, List, Optional

import httpx
from fastapi import APIRouter, Depends
from loguru import logger

from ..configs.setup import get_backend_settings
from ..core.security import get_current_user

settings = get_backend_settings()

router = APIRouter(prefix="/v1/rag", tags=["rag"])

# ────────────────────────────────────────────────────────────
#  System prompts  (updated — strict relevance-checking rules)
# ────────────────────────────────────────────────────────────

MEDICAL_SYSTEM_PROMPT = (
    "Bạn là trợ lý y tế Meddy. Hãy trả lời dựa trên tài liệu được cung cấp.\n\n"
    "QUY TẮC NGHIÊM NGẶT:\n"
    "1. Nếu tài liệu trích dẫn nói về một bệnh khác (ví dụ: Tiểu đường) "
    "trong khi người dùng hỏi về thói quen chung, bạn PHẢI nói rõ: "
    "'Tài liệu hiện có nói về [tên bệnh], có thể không hoàn toàn phù hợp "
    "với trường hợp của bạn'.\n"
    "2. Chỉ trích dẫn [số nguồn] khi thông tin đó thực sự có trong đoạn văn "
    "bản đó. KHÔNG được bịa trích dẫn.\n"
    "3. Khi không có tài liệu phù hợp, hãy trả lời dựa trên kiến thức y khoa "
    "chung và nói rõ rằng 'Thông tin này dựa trên kiến thức y khoa chung, "
    "không từ tài liệu tham khảo cụ thể'.\n"
    "4. Luôn đưa ra cảnh báo y tế ở cuối câu trả lời: "
    "'⚠️ Lưu ý: Thông tin trên chỉ mang tính chất tham khảo. "
    "Hãy tham khảo ý kiến bác sĩ để được tư vấn chính xác.'\n"
    "5. Không đưa ra chẩn đoán bệnh cụ thể hay kê đơn thuốc.\n"
    "6. Luôn trả lời bằng tiếng Việt. /no_think"
)

GENERAL_SYSTEM_PROMPT = (
    "Bạn là Meddy — trợ lý thông minh của hệ thống Medical RAG Chatbot. "
    "Hãy trả lời thân thiện và hữu ích bằng tiếng Việt. /no_think"
)

# ────────────────────────────────────────────────────────────
#  Query Refiner  (Ollama-based medical keyword extraction)
# ────────────────────────────────────────────────────────────

_REFINE_QUERY_SYSTEM = (
    "Bạn là chuyên gia y tế. Hãy chuyển câu hỏi sau thành các từ khóa tìm kiếm "
    "y khoa ngắn gọn, súc tích bằng tiếng Việt. Loại bỏ các từ thừa, giữ lại "
    "từ khóa quan trọng về triệu chứng, bệnh lý hoặc chủ đề sức khỏe.\n\n"
    "Ví dụ:\n"
    "- 'Tôi bỏ bữa sáng ăn đêm bù được không' → "
    "'tác hại bỏ bữa sáng, ăn đêm và nhịp sinh học, dinh dưỡng hợp lý'\n"
    "- 'Tôi hay bị đau đầu buổi sáng khi ngủ dậy' → "
    "'đau đầu buổi sáng, nguyên nhân đau đầu khi thức dậy, rối loạn giấc ngủ'\n"
    "- 'Con tôi 2 tuổi bị sốt cao 39 độ' → "
    "'sốt cao trẻ em 2 tuổi, xử trí sốt 39 độ, hạ sốt nhi khoa'\n\n"
    "CHỈ trả về các từ khóa, KHÔNG giải thích. /no_think"
)

_DEFAULT_OLLAMA_URL = "http://localhost:11434"


def refine_query(user_query: str) -> str:
    """Refine a user question into concise medical search keywords using Ollama.

    This converts conversational questions (e.g. "Tôi bỏ bữa sáng ăn đêm bù
    được không") into focused medical search terms (e.g. "tác hại bỏ bữa sáng,
    ăn đêm và nhịp sinh học, dinh dưỡng hợp lý") so that the retrieval step
    fetches semantically relevant documents rather than ones that merely share
    surface-level keywords.

    Falls back to the original query on any error.
    """
    try:
        from ..services.brain import ollama_chat_complete

        messages = [
            {"role": "system", "content": _REFINE_QUERY_SYSTEM},
            {"role": "user", "content": user_query},
        ]
        refined = ollama_chat_complete(
            messages=messages,
            temperature=0.3,   # low temp for deterministic keyword extraction
            max_tokens=256,
        )
        if refined and refined.strip():
            refined = refined.strip().strip('"').strip("'")
            logger.info(
                f"[RAG] Query refined: '{user_query[:60]}' → '{refined[:120]}'"
            )
            return refined
        logger.warning("[RAG] refine_query returned empty, using original query")
        return user_query
    except Exception as e:
        logger.warning(f"[RAG] refine_query failed ({e}), using original query")
        return user_query


# ────────────────────────────────────────────────────────────
#  Public: run_rag_pipeline  (imported by the chat router)
# ────────────────────────────────────────────────────────────

def run_rag_pipeline(
    question: str,
    history: List[Dict[str, str]],
    top_k: int = 5,
) -> Dict[str, Any]:
    """Full RAG pipeline with graceful degradation at every step.

    Steps
    -----
    1. Guardrails check         (GPU → Ollama fallback → fail-open)
    2. Intent detection         (medical vs general)
    3. Query enhancement        (rewrite with conversation context)
    4. **Query refinement**     (extract medical keywords via Ollama)
    5. Hybrid search            (Qdrant vector + Elasticsearch BM25 via RRF)
    6. Reranking                (Qwen3-Reranker-0.6B GPU → keyword fallback)
    7. Answer generation        (vLLM → Ollama fallback)

    Returns
    -------
    dict: {answer: str, citations: list[dict], route: str}
    """
    citations: List[Dict[str, Any]] = []
    route = "medical"

    # ── 1. Guardrails ─────────────────────────────────────────────────────────
    try:
        from ..core.guardrails import get_guardrails_service

        guard = get_guardrails_service()
        is_valid, violation, meta = guard.validate_query(question)
        if not is_valid:
            logger.warning(f"[RAG] Query blocked by guardrails: {violation}")
            return {
                "answer": guard.get_rejection_message(violation),
                "citations": [],
                "route": "blocked",
            }
        # Log if we fell back to Ollama or fail-open
        if meta and meta.get("method") == "ollama_fallback":
            logger.info("[RAG] Guardrails used Ollama fallback")
        elif meta and meta.get("failover"):
            logger.info("[RAG] Guardrails fail-open (all services down)")
    except Exception as e:
        logger.warning(f"[RAG] Guardrails unavailable, continuing without check: {e}")

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

    # ── 3. Query enhancement (conversation context) ──────────────────────────
    enhanced_query = question
    if history:
        try:
            from ..services.brain import enhance_query_quality

            enhanced_query = enhance_query_quality(history, question) or question
            logger.debug(f"[RAG] Enhanced query: {enhanced_query[:120]}")
        except Exception as e:
            logger.warning(f"[RAG] Query enhancement failed, using original: {e}")

    # ── 4. Query refinement (medical keyword extraction) ─────────────────────
    refined_query = enhanced_query
    if route == "medical":
        refined_query = refine_query(enhanced_query)

    # ── 5 & 6. Hybrid search + Reranking  (medical route only) ──────────────
    context = ""
    if route == "medical":
        try:
            from ..core.hybrid_search import hybrid_search

            # Use refined keywords for RETRIEVAL (better precision)
            raw_results = hybrid_search(refined_query, top_k=top_k * 2)
            logger.info(f"[RAG] Retrieved {len(raw_results)} raw candidates")
            if raw_results:
                # Use refined query for RERANKING too (matches search intent)
                results = _rerank(refined_query, raw_results, top_k)
                context, citations = _build_context(results)
                logger.info(f"[RAG] Using {len(citations)} documents as context")
        except Exception as e:
            logger.error(f"[RAG] Search pipeline failed, generating without context: {e}")

    # ── 7. Generation ─────────────────────────────────────────────────────────
    # Pass ORIGINAL question (not refined) so the LLM answers naturally
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

    return {"answer": answer, "citations": citations, "route": route}


# ────────────────────────────────────────────────────────────
#  Private helpers
# ────────────────────────────────────────────────────────────

def _rerank(
    query: str,
    raw_results: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    """Rerank raw search results.

    The reranker itself handles GPU → keyword fallback internally,
    so this function always gets scored & sorted results back.
    """
    try:
        from ..services.rerank import get_qwen3_reranker

        reranker = get_qwen3_reranker()
        reranked_items, _ = reranker.rerank(query, raw_results, top_n=top_k)
        docs: List[Dict[str, Any]] = []
        for item in reranked_items[:top_k]:
            idx = item.get("index", 0)
            if 0 <= idx < len(raw_results):
                doc = raw_results[idx].copy()
                doc["relevance_score"] = item.get("relevance_score", 0.0)
                docs.append(doc)
        if docs:
            # Filter out documents with very low relevance (< 10% of best)
            best_score = max(d.get("relevance_score", 0) for d in docs)
            if best_score > 0:
                threshold = best_score * 0.1
                docs = [d for d in docs if d.get("relevance_score", 0) >= threshold]
            return docs
        return raw_results[:top_k]
    except Exception as e:
        logger.error(f"[RAG] Reranker completely failed: {e}")
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
            "Chỉ trích dẫn [số] khi thông tin thực sự có trong tài liệu đó. "
            "Nếu tài liệu không phù hợp với câu hỏi, hãy nói rõ."
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
    if not question:
        return {"answer": "Vui lòng nhập câu hỏi.", "citations": [], "route": ""}

    logger.info(f"[RAG] Direct query from user {current_user.id}: {question[:80]}")
    result = run_rag_pipeline(question=question, history=history)

    return {
        "answer": result["answer"],
        "citations": result["citations"],
        "metadata": {
            "route": result["route"],
            "retrieval_count": len(result["citations"]),
        },
    }
