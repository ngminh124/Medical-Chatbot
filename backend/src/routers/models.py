"""Model management router."""

from fastapi import APIRouter

router = APIRouter(prefix="/v1/models", tags=["models"])


@router.get("")
def list_models():
    """List available models."""
    return {
        "models": [
            {
                "id": "qwen3-embedding-0.6b",
                "type": "embedding",
                "status": "available",
            },
        ]
    }
