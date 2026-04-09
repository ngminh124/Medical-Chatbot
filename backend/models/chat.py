"""SQLAlchemy models for the `threads`, `messages`, and `feedbacks` tables."""

import uuid

from sqlalchemy import Column, DateTime, Enum, Integer, Text, VARCHAR, func
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import relationship
from sqlalchemy.schema import CheckConstraint, ForeignKey

from .base import Base


# ─────────────────────────────────────────────
# Thread
# ─────────────────────────────────────────────
class Thread(Base):
    __tablename__ = "threads"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True), ForeignKey("users.id", ondelete="CASCADE"), nullable=False
    )
    title = Column(VARCHAR(512), nullable=False, default="Cuộc trò chuyện mới")
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )
    updated_at = Column(
        DateTime(timezone=True),
        nullable=False,
        server_default=func.now(),
        onupdate=func.now(),
    )

    # ── Relationships ──
    user = relationship("User", back_populates="threads")
    messages = relationship(
        "Message", back_populates="thread", cascade="all, delete-orphan",
        order_by="Message.created_at",
    )

    def __repr__(self) -> str:
        return f"<Thread id={self.id} title={self.title!r}>"


# ─────────────────────────────────────────────
# Message
# ─────────────────────────────────────────────
class Message(Base):
    __tablename__ = "messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    thread_id = Column(
        UUID(as_uuid=True), ForeignKey("threads.id", ondelete="CASCADE"), nullable=False
    )
    role = Column(
        Enum("user", "assistant", name="message_role", create_type=False),
        nullable=False,
    )
    content = Column(Text, nullable=False)
    metadata_ = Column("metadata", JSONB, default=dict)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    # ── Relationships ──
    thread = relationship("Thread", back_populates="messages")
    feedbacks = relationship(
        "Feedback", back_populates="message", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        return f"<Message id={self.id} role={self.role!r}>"


# ─────────────────────────────────────────────
# Feedback
# ─────────────────────────────────────────────
class Feedback(Base):
    __tablename__ = "feedbacks"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    message_id = Column(
        UUID(as_uuid=True), ForeignKey("messages.id", ondelete="CASCADE"), nullable=False
    )
    rating = Column(Integer, nullable=False)
    comment = Column(Text, nullable=True)
    created_at = Column(
        DateTime(timezone=True), nullable=False, server_default=func.now()
    )

    __table_args__ = (
        CheckConstraint("rating >= 1 AND rating <= 5", name="feedbacks_rating_check"),
    )

    # ── Relationships ──
    message = relationship("Message", back_populates="feedbacks")

    def __repr__(self) -> str:
        return f"<Feedback id={self.id} rating={self.rating}>"
