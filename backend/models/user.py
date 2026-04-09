"""SQLAlchemy model for the `users` table."""

import uuid

from sqlalchemy import Boolean, Column, DateTime, Text, func
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import relationship

from .base import Base


class User(Base):
    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    email = Column(Text, unique=True, nullable=False)
    email_verified = Column(Boolean, nullable=False, default=False)
    verification_token = Column(Text, nullable=True)
    verification_token_time = Column(DateTime(timezone=True), nullable=True)
    password = Column(Text, nullable=False)
    reset_password_token = Column(Text, nullable=True)
    reset_password_token_time = Column(DateTime(timezone=True), nullable=True)
    phone = Column(Text, nullable=True)
    name = Column(Text, nullable=False)
    type = Column(Text, nullable=True)
    status = Column(Text, nullable=False, default="active")
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
    threads = relationship("Thread", back_populates="user", cascade="all, delete-orphan")

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email!r}>"
