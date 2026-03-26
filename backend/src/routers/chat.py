"""Chat router — threads, messages, feedbacks, RAG ask."""

from typing import Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, status
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
    return messages


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
    assistant_msg = Message(
        thread_id=thread.id,
        role="assistant",
        content=result["answer"],
        metadata_={
            "citations": result["citations"],
            "route": result["route"],
        },
    )
    db.add(assistant_msg)

    # Auto-set thread title from the first user message
    if not history_msgs:
        thread.title = body.content[:60] + ("…" if len(body.content) > 60 else "")
        db.add(thread)

    db.commit()
    db.refresh(assistant_msg)
    logger.info(
        f"[ASK] Assistant message saved: {assistant_msg.id} "
        f"(route={result['route']}, citations={len(result['citations'])})"
    )

    return AskResponse(
        user_message=MessageResponse.model_validate(user_msg),
        assistant_message=MessageResponse.model_validate(assistant_msg),
        citations=[Citation(**c) for c in result["citations"]],
        route=result["route"],
    )


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
