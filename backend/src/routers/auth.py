"""Authentication router — register, login."""

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from sqlalchemy import or_
from sqlalchemy.orm import Session

from backend.models import User

from ..core.security import (
    create_access_token,
    get_current_user,
    hash_password,
    verify_password,
)
from ..database import get_db_session
from ..schemas.auth import LoginRequest, LoginResponse, RegisterRequest, UserResponse

router = APIRouter(prefix="/v1/auth", tags=["auth"])


def ensure_default_admin(db: Session):
    """Create default admin account if it does not exist."""
    existing = db.query(User).filter(User.name == "admin").first()
    if existing:
        return existing

    user = User(
        email="admin@minqes.local",
        password=hash_password("admin123"),
        name="admin",
        type="admin",
        status="active",
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.warning("Default admin account created (username=admin)")
    return user


@router.post("/register", response_model=UserResponse, status_code=201)
def register(body: RegisterRequest, db: Session = Depends(get_db_session)):
    """Đăng ký tài khoản mới."""
    # Check duplicate email
    existing = db.query(User).filter(User.email == body.email).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Email này đã được sử dụng",
        )

    user = User(
        email=body.email,
        password=hash_password(body.password),
        name=body.name,
        phone=body.phone,
        type=body.type,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    logger.info(f"New user registered: {user.email}")
    return user


@router.post("/login", response_model=LoginResponse)
def login(body: LoginRequest, db: Session = Depends(get_db_session)):
    """Đăng nhập và nhận JWT token."""
    identifier = body.email.strip()
    user = (
        db.query(User)
        .filter(or_(User.email == identifier, User.name == identifier))
        .first()
    )
    if not user or not verify_password(body.password, user.password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Email hoặc mật khẩu không đúng",
        )
    if user.status != "active":
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Tài khoản đã bị vô hiệu hóa",
        )

    token = create_access_token(user.id)
    logger.info(f"User logged in: {user.email}")
    return LoginResponse(access_token=token, user=UserResponse.model_validate(user))


@router.get("/me", response_model=UserResponse)
def get_me(current_user: User = Depends(get_current_user)):
    """Lấy thông tin user hiện tại từ JWT."""
    return current_user
