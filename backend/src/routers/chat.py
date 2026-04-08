"""Chat router — threads, messages, feedbacks, RAG ask."""

import json
import re
from typing import Any, Optional
from urllib.parse import urlparse
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from loguru import logger
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.models import Feedback, Message, Thread, User

from ..configs.setup import get_backend_settings
from ..core.security import get_current_user
from ..database import get_db_session
from ..schemas.chat import (
    AskRequest,
    AskResponse,
    Citation,
    FeedbackCreate,
    FeedbackResponse,
    MessageCreate,
    MessageResponse,
    ThreadCreate,
    ThreadResponse,
    ThreadWithLastMessage,
)

settings = get_backend_settings()
router = APIRouter(prefix="/v1/chat", tags=["chat"])


def _is_http_url(url: str) -> bool:
    return isinstance(url, str) and url.startswith(("http://", "https://"))


def _clean_snippet(text: str, max_len: int = 320) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[:max_len].rstrip() + "…"


def _normalize_web_citations(citations: list[dict[str, Any]] | None) -> list[dict[str, Any]]:
    web_citations: list[dict[str, Any]] = []
    for citation in citations or []:
        if not isinstance(citation, dict):
            continue

        url = (
            citation.get("url")
            or citation.get("link")
            or (citation.get("source") if _is_http_url(citation.get("source", "")) else "")
        )
        if not _is_http_url(url):
            continue

        ctype = str(citation.get("type") or "web").lower()
        if ctype != "web":
            continue

        domain = ""
        try:
            domain = (urlparse(url).hostname or "").replace("www.", "")
        except Exception:
            domain = ""

        snippet = _clean_snippet(
            citation.get("snippet") or citation.get("content") or citation.get("text") or ""
        )

        score = citation.get("score")
        try:
            score = float(score) if score is not None else 0.0
        except Exception:
            score = 0.0

        web_citations.append(
            {
                "title": citation.get("title") or domain or "Nguồn web",
                "url": url,
                "snippet": snippet,
                "type": "web",
                "score": score,
                "domain": domain,
                "favicon": citation.get("favicon")
                or citation.get("favicon_url")
                or (f"https://www.google.com/s2/favicons?domain={domain}&sz=64" if domain else ""),
            }
        )
    return web_citations


def _sanitize_assistant_metadata(
    metadata: dict[str, Any] | None,
    web_search_enabled: bool | None = None,
) -> dict[str, Any]:
    raw = metadata if isinstance(metadata, dict) else {}
    route = str(raw.get("route") or "medical")
    enabled = (
        bool(web_search_enabled)
        if web_search_enabled is not None
        else bool(raw.get("web_search_enabled", False))
    )
    used = bool(raw.get("web_search_used", False))

    if not enabled or not used:
        return {
            "route": route,
            "web_search_enabled": enabled,
            "web_search_used": False,
        }

    web_citations = _normalize_web_citations(raw.get("citations"))
    if not web_citations:
        return {
            "route": route,
            "web_search_enabled": enabled,
            "web_search_used": False,
        }

    return {
        "route": route,
        "web_search_enabled": enabled,
        "web_search_used": True,
        "citations": web_citations,
    }


# ────────────────────────────────────────────────────────────
#  Threads
# ────────────────────────────────────────────────────────────
@router.post("/threads", response_model=ThreadResponse, status_code=201)
def create_thread(
    body: ThreadCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
):
    """Tạo phiên trò chuyện mới."""
    thread = Thread(user_id=current_user.id, title=body.title)
    db.add(thread)
    db.commit()
    db.refresh(thread)
    logger.info(f"Thread created: {thread.id} by user {current_user.id}")
    return thread


@router.get("/threads", response_model=list[ThreadWithLastMessage])
def list_threads(
    skip: int = Query(0, ge=0),
    limit: int = Query(50, ge=1, le=100),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
):
    """Lấy danh sách threads của user (mới nhất trước)."""
    threads = (
        db.query(Thread)
        .filter(Thread.user_id == current_user.id)
        .order_by(Thread.updated_at.desc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    result = []
    for thread in threads:
        # Get last message preview & count
        msg_count = db.query(func.count(Message.id)).filter(Message.thread_id == thread.id).scalar()
        last_msg = (
            db.query(Message)
            .filter(Message.thread_id == thread.id)
            .order_by(Message.created_at.desc())
            .first()
        )
        result.append(
            ThreadWithLastMessage(
                id=thread.id,
                user_id=thread.user_id,
                title=thread.title,
                created_at=thread.created_at,
                updated_at=thread.updated_at,
                last_message=last_msg.content[:100] if last_msg else None,
                message_count=msg_count,
            )
        )
    return result


@router.get("/threads/{thread_id}", response_model=ThreadResponse)
def get_thread(
    thread_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
):
    """Lấy thông tin một thread."""
    thread = _get_user_thread(db, thread_id, current_user.id)
    return thread


@router.delete("/threads/{thread_id}", status_code=204)
def delete_thread(
    thread_id: UUID,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
):
    """Xóa một thread và toàn bộ messages."""
    thread = _get_user_thread(db, thread_id, current_user.id)
    db.delete(thread)
    db.commit()
    logger.info(f"Thread deleted: {thread_id}")


# ────────────────────────────────────────────────────────────
#  Messages
# ────────────────────────────────────────────────────────────
@router.get("/threads/{thread_id}/messages", response_model=list[MessageResponse])
def list_messages(
    thread_id: UUID,
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=500),
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
):
    """Lấy lịch sử tin nhắn của một thread."""
    _get_user_thread(db, thread_id, current_user.id)  # ownership check

    messages = (
        db.query(Message)
        .filter(Message.thread_id == thread_id)
        .order_by(Message.created_at.asc())
        .offset(skip)
        .limit(limit)
        .all()
    )

    sanitized_messages: list[MessageResponse] = []
    for msg in messages:
        metadata = msg.metadata_
        if msg.role == "assistant":
            metadata = _sanitize_assistant_metadata(metadata)

        sanitized_messages.append(
            MessageResponse(
                id=msg.id,
                thread_id=msg.thread_id,
                role=msg.role,
                content=msg.content,
                metadata_=metadata,
                created_at=msg.created_at,
            )
        )

    return sanitized_messages


@router.post(
    "/threads/{thread_id}/messages",
    response_model=MessageResponse,
    status_code=201,
)
def send_message(
    thread_id: UUID,
    body: MessageCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
):
    """
    Gửi tin nhắn của user.

    Endpoint này lưu tin nhắn user vào DB rồi trả về ngay.
    Phản hồi của assistant sẽ được gọi riêng qua RAG pipeline
    (hoặc tích hợp thêm sau).
    """
    thread = _get_user_thread(db, thread_id, current_user.id)

    # Save user message
    user_msg = Message(
        thread_id=thread.id,
        role="user",
        content=body.content,
    )
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)
    logger.info(f"Message saved: {user_msg.id} in thread {thread_id}")
    return user_msg


# ────────────────────────────────────────────────────────────
#  RAG Ask — one-shot: save user msg + generate + save reply
# ────────────────────────────────────────────────────────────

@router.post(
    "/threads/{thread_id}/ask",
    response_model=AskResponse,
    status_code=201,
    summary="Send a message and get an AI response via the RAG pipeline",
)
def ask(
    thread_id: UUID,
    body: AskRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
):
    """Pipeline: guardrails → intent detection → query rewrite →
    hybrid search (Qdrant + ES) → reranking → generation (vLLM → Ollama)."""
    thread = _get_user_thread(db, thread_id, current_user.id)

    # Fetch recent history for context (up to 20 turns)
    history_msgs = (
        db.query(Message)
        .filter(Message.thread_id == thread_id)
        .order_by(Message.created_at.asc())
        .limit(20)
        .all()
    )
    history = [{"role": m.role, "content": m.content} for m in history_msgs]

    # Persist user message immediately so it's visible in the UI
    user_msg = Message(thread_id=thread.id, role="user", content=body.content)
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)
    logger.info(f"[ASK] User message saved: {user_msg.id} in thread {thread_id}")

    # ── Run the full RAG pipeline ─────────────────────────────────────────────
    try:
        from ..routers.rag import run_rag_pipeline

        result = run_rag_pipeline(
            question=body.content,
            history=history,
            top_k=settings.top_k,
            web_search_enabled=body.web_search_enabled,
        )
    except Exception as exc:
        logger.error(f"[ASK] RAG pipeline raised an unexpected exception: {exc}")
        result = {
            "answer": "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại.",
            "citations": [],
            "route": "error",
        }

    # Persist assistant reply with full RAG metadata
    assistant_metadata = _sanitize_assistant_metadata(
        {
            "citations": result.get("citations", []),
            "route": result.get("route", "medical"),
            "web_search_enabled": bool(body.web_search_enabled),
            "web_search_used": bool(result.get("web_search_used", False)),
        },
        web_search_enabled=bool(body.web_search_enabled),
    )

    assistant_msg = Message(
        thread_id=thread.id,
        role="assistant",
        content=result["answer"],
        metadata_=assistant_metadata,
    )
    db.add(assistant_msg)

    # Auto-set thread title from the first user message
    if not history_msgs:
        thread.title = body.content[:60] + ("…" if len(body.content) > 60 else "")
        db.add(thread)

    db.commit()
    db.refresh(assistant_msg)
    public_citations = assistant_metadata.get("citations", [])
    logger.info(
        f"[ASK] Assistant message saved: {assistant_msg.id} "
        f"(route={result['route']}, citations={len(public_citations)})"
    )

    return AskResponse(
        user_message=MessageResponse.model_validate(user_msg),
        assistant_message=MessageResponse.model_validate(assistant_msg),
        citations=[Citation(**c) for c in public_citations],
        route=result["route"],
    )


@router.post(
    "/threads/{thread_id}/ask-stream",
    status_code=200,
    summary="Send a message and stream AI response via SSE",
)
def ask_stream(
    thread_id: UUID,
    body: AskRequest,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
):
    """Streaming pipeline: persists user message immediately and streams assistant chunks."""
    thread = _get_user_thread(db, thread_id, current_user.id)

    history_msgs = (
        db.query(Message)
        .filter(Message.thread_id == thread_id)
        .order_by(Message.created_at.asc())
        .limit(20)
        .all()
    )
    history = [{"role": m.role, "content": m.content} for m in history_msgs]

    user_msg = Message(thread_id=thread.id, role="user", content=body.content)
    db.add(user_msg)
    db.commit()
    db.refresh(user_msg)

    def _to_sse(event: str, payload: dict[str, Any]) -> str:
        return f"event: {event}\ndata: {json.dumps(payload, ensure_ascii=False)}\n\n"

    def _stream_generator():
        assistant_text = ""
        route = "medical"
        web_search_used = False
        citations: list[dict[str, Any]] = []

        try:
            from ..routers.rag import run_rag_pipeline_stream

            stream_result = run_rag_pipeline_stream(
                question=body.content,
                history=history,
                top_k=settings.top_k,
                web_search_enabled=body.web_search_enabled,
            )
            stream_iter = stream_result.get("answer_stream")
            route = str(stream_result.get("route") or "medical")
            web_search_used = bool(stream_result.get("web_search_used", False))
            citations = stream_result.get("citations", []) or []

            static_answer = (stream_result.get("static_answer") or "").strip()
            if static_answer:
                assistant_text = static_answer
                for token in static_answer.split(" "):
                    yield _to_sse("chunk", {"chunk": f"{token} "})
            elif stream_iter is not None:
                for chunk in stream_iter:
                    if not chunk:
                        continue
                    assistant_text += chunk
                    yield _to_sse("chunk", {"chunk": chunk})

            assistant_text = assistant_text.strip()
            if not assistant_text:
                assistant_text = "Xin lỗi, hệ thống đang gặp sự cố. Vui lòng thử lại."
                route = "error"

            assistant_metadata = _sanitize_assistant_metadata(
                {
                    "citations": citations,
                    "route": route,
                    "web_search_enabled": bool(body.web_search_enabled),
                    "web_search_used": web_search_used,
                },
                web_search_enabled=bool(body.web_search_enabled),
            )

            assistant_msg = Message(
                thread_id=thread.id,
                role="assistant",
                content=assistant_text,
                metadata_=assistant_metadata,
            )
            db.add(assistant_msg)

            if not history_msgs:
                thread.title = body.content[:60] + ("…" if len(body.content) > 60 else "")
                db.add(thread)

            db.commit()
            db.refresh(assistant_msg)

            done_payload = {
                "route": route,
                "citations": assistant_metadata.get("citations", []),
                "user_message": MessageResponse.model_validate(user_msg).model_dump(
                    mode="json",
                    by_alias=True,
                ),
                "assistant_message": MessageResponse.model_validate(assistant_msg).model_dump(
                    mode="json",
                    by_alias=True,
                ),
                "done": True,
            }
            yield _to_sse("done", done_payload)
        except Exception as exc:
            logger.error(f"[ASK_STREAM] Failed: {exc}")
            db.rollback()
            yield _to_sse(
                "error",
                {
                    "error": "Đã có lỗi xảy ra khi xử lý luồng phản hồi.",
                    "done": True,
                },
            )

    return StreamingResponse(_stream_generator(), media_type="text/event-stream")


# ────────────────────────────────────────────────────────────
#  Feedbacks
# ────────────────────────────────────────────────────────────
@router.post(
    "/messages/{message_id}/feedback",
    response_model=FeedbackResponse,
    status_code=201,
)
def create_feedback(
    message_id: UUID,
    body: FeedbackCreate,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db_session),
):
    """Đánh giá một tin nhắn (1–5 sao)."""
    # Verify message exists and belongs to user's thread
    msg = db.query(Message).filter(Message.id == message_id).first()
    if not msg:
        raise HTTPException(status_code=404, detail="Tin nhắn không tồn tại")

    thread = db.query(Thread).filter(Thread.id == msg.thread_id).first()
    if not thread or thread.user_id != current_user.id:
        raise HTTPException(status_code=403, detail="Không có quyền truy cập")

    feedback = Feedback(
        message_id=message_id,
        rating=body.rating,
        comment=body.comment,
    )
    db.add(feedback)
    db.commit()
    db.refresh(feedback)
    logger.info(f"Feedback created: {feedback.id} for message {message_id}")
    return feedback


# ────────────────────────────────────────────────────────────
#  Helpers
# ────────────────────────────────────────────────────────────
def _get_user_thread(db: Session, thread_id: UUID, user_id: UUID) -> Thread:
    """Get a thread and verify ownership."""
    thread = db.query(Thread).filter(Thread.id == thread_id).first()
    if not thread:
        raise HTTPException(status_code=404, detail="Thread không tồn tại")
    if thread.user_id != user_id:
        raise HTTPException(status_code=403, detail="Không có quyền truy cập thread này")
    return thread
