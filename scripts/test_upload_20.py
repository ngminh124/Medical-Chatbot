"""Test upload first 20 chunks from data/chunks/test.jsonl into Qdrant.

Usage:
    python scripts/test_upload_20.py

Prerequisites:
- Qdrant running (docker-compose up -d)
- Embedding service running: `uvicorn serving.qwen3_models.app:app --host 0.0.0.0 --port 7860`
"""
import json
import sys
import uuid
from pathlib import Path
from loguru import logger

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.src.configs.setup import get_backend_settings
from backend.src.services.embedding import get_embedding_service
from backend.src.core.vectorize import create_collection, upsert_points


def main():
    settings = get_backend_settings()
    jsonl_path = Path("data/chunks/test.jsonl")

    if not jsonl_path.exists():
        logger.error(f"JSONL file not found: {jsonl_path}")
        return

    # Read first 20 lines
    chunks = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= 5:
                break
            try:
                data = json.loads(line)
                chunks.append(data)
            except Exception as e:
                logger.warning(f"Skipping line {i+1}: {e}")

    if not chunks:
        logger.error("No chunks loaded")
        return

    logger.info(f"Loaded {len(chunks)} chunks for test upload")

    # Embedding service
    emb_service = get_embedding_service()

    texts = [c.get("content", "") for c in chunks]
    logger.info("Requesting embeddings for test batch...")
    embeddings = emb_service.embed_batch_documents(texts, batch_size=16)

    # Prepare points
    points = []
    for c, emb in zip(chunks, embeddings):
        # Convert string ID to UUID v5 (deterministic hash)
        chunk_id = c.get("id", "")
        if isinstance(chunk_id, str):
            point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk_id))
        else:
            point_id = str(uuid.UUID(bytes=chunk_id.to_bytes(16, "big")))
        
        point = {
            "id": point_id,
            "embedding": emb,
            "metadata": {"content": c.get("content", ""), **c.get("metadata", {})},
        }
        points.append(point)

    # Ensure collection
    create_collection(collection_name=settings.default_collection_name, vector_dimension=settings.vector_dimension)

    # Upsert
    logger.info("Upserting points to Qdrant...")
    res = upsert_points(points, collection_name=settings.default_collection_name)
    logger.info(f"Upsert result: {res}")
    logger.success("Test upload completed")


if __name__ == "__main__":
    main()
