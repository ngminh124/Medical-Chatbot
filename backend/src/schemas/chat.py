"""Chat-related Pydantic schemas (threads, messages, feedbacks, RAG ask)."""

from datetime import datetime
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# ─── Thread ─────────────────────────────────────────────────
class ThreadCreate(BaseModel):
    title: Optional[str] = "Cuộc trò chuyện mới"


class ThreadResponse(BaseModel):
    id: UUID
    user_id: UUID
    title: str
    created_at: datetime
    updated_at: datetime

    model_config = {"from_attributes": True}


class ThreadWithLastMessage(ThreadResponse):
    """Thread with a preview of the last message (for sidebar listing)."""
    last_message: Optional[str] = None
    message_count: int = 0


# ─── Message ────────────────────────────────────────────────
class MessageCreate(BaseModel):
    content: str = Field(..., min_length=1, max_length=10000)


class MessageResponse(BaseModel):
    id: UUID
    thread_id: UUID
    role: str
    content: str
    metadata_: Optional[dict[str, Any]] = Field(None, alias="metadata_")
    created_at: datetime

    model_config = {"from_attributes": True, "populate_by_name": True}


# ─── Feedback ───────────────────────────────────────────────
class FeedbackCreate(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    comment: Optional[str] = None


class FeedbackResponse(BaseModel):
    id: UUID
    message_id: UUID
    rating: int
    comment: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


# ─── RAG Ask ────────────────────────────────────────────────
class Citation(BaseModel):
    """A single retrieved document used as context for generation."""
    title: str = ""
    content: str = ""
    source: str = ""
    score: float = 0.0


class AskRequest(BaseModel):
    """Request body for the /ask endpoint."""
    content: str = Field(..., min_length=1, max_length=10000)


class AskResponse(BaseModel):
    """
    Response from the /ask endpoint.
    Contains both the persisted user + assistant messages and RAG metadata.
    """
    user_message: MessageResponse
    assistant_message: MessageResponse
    citations: list[Citation] = []
    route: str = "medical"  # medical | general | blocked | error
