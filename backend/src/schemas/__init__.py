"""Pydantic schemas for API request/response validation."""

from .auth import (
    LoginRequest,
    LoginResponse,
    RegisterRequest,
    TokenPayload,
    UserResponse,
)
from .chat import (
    FeedbackCreate,
    FeedbackResponse,
    MessageCreate,
    MessageResponse,
    ThreadCreate,
    ThreadResponse,
    ThreadWithLastMessage,
)

__all__ = [
    # Auth
    "RegisterRequest",
    "LoginRequest",
    "LoginResponse",
    "UserResponse",
    "TokenPayload",
    # Chat
    "ThreadCreate",
    "ThreadResponse",
    "ThreadWithLastMessage",
    "MessageCreate",
    "MessageResponse",
    "FeedbackCreate",
    "FeedbackResponse",
]
