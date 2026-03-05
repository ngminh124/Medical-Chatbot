"""Auth-related Pydantic schemas."""

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel, EmailStr, Field


# ─── Request ────────────────────────────────────────────────
class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, max_length=128)
    name: str = Field(..., min_length=1, max_length=255)
    phone: Optional[str] = None
    type: Optional[str] = None  # 'patient' | 'doctor'


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


# ─── Response ───────────────────────────────────────────────
class UserResponse(BaseModel):
    id: UUID
    email: str
    name: str
    phone: Optional[str] = None
    type: Optional[str] = None
    status: str
    created_at: datetime

    model_config = {"from_attributes": True}


class LoginResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user: UserResponse


# ─── Internal ───────────────────────────────────────────────
class TokenPayload(BaseModel):
    sub: str  # user id (UUID as string)
    exp: int  # expiry timestamp
