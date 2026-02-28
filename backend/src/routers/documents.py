"""Document management router (placeholder)."""

from fastapi import APIRouter

router = APIRouter(prefix="/v1/documents", tags=["documents"])


@router.get("")
def list_documents():
    """List indexed documents."""
    return {"documents": [], "total": 0}


@router.post("/upload")
def upload_document():
    """Upload a document for indexing (placeholder)."""
    return {"status": "not_implemented"}
