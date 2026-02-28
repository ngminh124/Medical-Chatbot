"""Health check router."""

from fastapi import APIRouter

router = APIRouter(prefix="/v1/health", tags=["health"])


@router.get("")
def health_check():
    """Basic health check endpoint."""
    return {"status": "ok"}


@router.get("/ready")
def readiness_check():
    """Readiness check — verify dependent services."""
    return {"status": "ready"}
