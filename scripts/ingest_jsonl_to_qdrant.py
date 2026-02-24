"""Ingest medical JSONL chunks into Qdrant vector database + Elasticsearch.

This script reads chunks from a JSONL file, embeds them using the embedding service,
and upserts them into Qdrant with metadata preservation.
Optionally also indexes into Elasticsearch for BM25 keyword search.

Usage:
    # Qdrant only (default)
    python scripts/ingest_jsonl_to_qdrant.py --input data/chunks/medical_master_data6.jsonl

    # Qdrant + Elasticsearch (recommended for hybrid search)
    python scripts/ingest_jsonl_to_qdrant.py --input data/chunks/medical_master_data6.jsonl --also-elasticsearch

Prerequisites:
- Qdrant running (docker-compose up -d)
- Embedding service running on localhost:7860
- Elasticsearch running (docker-compose up -d elasticsearch) — only if --also-elasticsearch
"""
import argparse
import json
import sys
import uuid
from pathlib import Path
from typing import List, Dict, Any

import torch
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.src.configs.setup import get_backend_settings
from backend.src.services.embedding import get_embedding_service
from backend.src.core.vectorize import create_collection, upsert_points

settings = get_backend_settings()


def string_to_uuid(text: str) -> str:
    """Convert string to deterministic UUID v5."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, text))


def read_jsonl(filepath: Path, limit: int = None) -> List[Dict[str, Any]]:
    """Read JSONL file line by line."""
    chunks = []
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                if limit and i >= limit:
                    break
                try:
                    data = json.loads(line)
                    chunks.append(data)
                except json.JSONDecodeError as e:
                    logger.warning(f"Skipping line {i+1}: invalid JSON - {e}")
                    continue
    except FileNotFoundError:
        logger.error(f"File not found: {filepath}")
        return []
    
    return chunks


def embed_batch(
    texts: List[str], batch_size: int = 32
) -> List[Any]:
    """Embed texts using the embedding service."""
    emb_service = get_embedding_service()
    logger.info(f"Embedding {len(texts)} texts with batch_size={batch_size}")
    embeddings = emb_service.embed_batch_documents(texts, batch_size=batch_size)
    
    # Debug: check if embeddings are None or empty
    if embeddings is None:
        logger.error(f"Embedding service returned None!")
    elif all(e is None for e in embeddings):
        logger.error(f"All embeddings are None!")
    else:
        valid_count = sum(1 for e in embeddings if e is not None)
        logger.info(f"Got {valid_count}/{len(embeddings)} valid embeddings")
    
    return embeddings


def prepare_points(
    chunks: List[Dict[str, Any]], embeddings: List[Any]
) -> List[Dict[str, Any]]:
    """Prepare Qdrant points from chunks and embeddings."""
    points = []
    
    for chunk, embedding in zip(chunks, embeddings):
        if embedding is None:
            logger.warning(f"Skipping chunk {chunk.get('id')} - no embedding")
            continue
        
        # Convert string ID to UUID v5 (deterministic hash)
        chunk_id = chunk.get("id", "unknown")
        point_id = string_to_uuid(chunk_id)
        
        point = {
            "id": point_id,
            "embedding": embedding,
            "metadata": {
                "content": chunk.get("content", ""),
                "original_id": chunk_id,  # Store original ID in metadata
                **chunk.get("metadata", {}),
            },
        }
        points.append(point)
    
    return points


def _index_batch_to_elasticsearch(es_client, chunks: List[Dict[str, Any]]) -> int:
    """Index a batch of chunks into Elasticsearch. Returns count of successful indexes."""
    indexed = 0
    for chunk in chunks:
        chunk_id = chunk.get("id", "")
        metadata = chunk.get("metadata", {})
        success = es_client.index_chunk(
            chunk_id=chunk_id,
            document_id=metadata.get("file_name", ""),
            chunk_index=metadata.get("chunk_index", 0),
            content=chunk.get("content", ""),
            title=metadata.get("subject_name", ""),
            doc_type=metadata.get("subject_type", "medical"),
            source=metadata.get("file_name", ""),
            language="vi",
            metadata=metadata,
        )
        if success:
            indexed += 1
    return indexed


def ingest_jsonl(
    input_path: Path,
    collection_name: str = None,
    batch_size: int = 32,
    upsert_batch_size: int = 500,
    limit: int = None,
    also_elasticsearch: bool = False,
):
    """Main ingestion pipeline. Ingest into Qdrant, optionally also Elasticsearch."""
    collection_name = collection_name or settings.default_collection_name
    
    logger.info(f"Starting ingestion from: {input_path}")
    logger.info(f"Collection: {collection_name}")
    logger.info(f"Batch size: {batch_size}, Upsert batch: {upsert_batch_size}")
    logger.info(f"Also Elasticsearch: {also_elasticsearch}")
    
    # Read chunks
    logger.info("Reading JSONL file...")
    chunks = read_jsonl(input_path, limit=limit)
    
    if not chunks:
        logger.error("No chunks loaded")
        return
    
    logger.info(f"Loaded {len(chunks)} chunks")
    
    # Create Qdrant collection
    logger.info(f"Creating Qdrant collection {collection_name}...")
    create_collection(
        collection_name=collection_name,
        vector_dimension=settings.vector_dimension,
    )
    
    # Optionally setup Elasticsearch
    es_client = None
    if also_elasticsearch:
        try:
            from backend.src.services.elastic_search import ElasticsearchClient
            es_client = ElasticsearchClient()
            es_client.create_index()
            logger.info(f"Elasticsearch index '{es_client.index_name}' ready")
        except Exception as e:
            logger.warning(f"Elasticsearch not available, skipping: {e}")
            es_client = None
    
    # Embed and upsert in batches
    total_upserted = 0
    total_es_indexed = 0
    
    for batch_start in range(0, len(chunks), upsert_batch_size):
        batch_end = min(batch_start + upsert_batch_size, len(chunks))
        batch_chunks = chunks[batch_start:batch_end]
        
        logger.info(f"Processing batch {batch_start}-{batch_end}/{len(chunks)}...")
        
        # Extract texts
        texts = [c.get("content", "") for c in batch_chunks]
        
        # Embed
        logger.info(f"Embedding {len(texts)} texts...")
        embeddings = embed_batch(texts, batch_size=batch_size)
        
        # Clear GPU cache after embedding
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Prepare points
        points = prepare_points(batch_chunks, embeddings)
        
        if not points:
            logger.warning(f"No valid points in batch {batch_start}-{batch_end}")
            continue
        
        # Upsert to Qdrant
        logger.info(f"Upserting {len(points)} points to Qdrant...")
        result = upsert_points(points, collection_name=collection_name)
        total_upserted += len(points)
        
        # Also index to Elasticsearch
        if es_client:
            es_indexed = _index_batch_to_elasticsearch(es_client, batch_chunks)
            total_es_indexed += es_indexed
            logger.info(f"Elasticsearch: indexed {es_indexed}/{len(batch_chunks)} chunks")
        
        logger.success(f"Qdrant: {len(points)} points. Total: {total_upserted}")
    
    # Refresh ES index
    if es_client:
        try:
            es_client.client.indices.refresh(index=es_client.index_name)
        except Exception:
            pass
    
    logger.success(
        f"Ingestion completed! Qdrant: {total_upserted} points"
        + (f", Elasticsearch: {total_es_indexed} docs" if es_client else "")
    )
    return total_upserted


def main():
    parser = argparse.ArgumentParser(
        description="Ingest medical JSONL chunks into Qdrant"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/chunks/medical_master_data6.jsonl"),
        help="Input JSONL file path",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="Qdrant collection name (default from settings)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Embedding batch size",
    )
    parser.add_argument(
        "--upsert-batch-size",
        type=int,
        default=500,
        help="Qdrant upsert batch size",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of chunks to process (for testing)",
    )
    parser.add_argument(
        "--also-elasticsearch",
        action="store_true",
        help="Also index chunks into Elasticsearch for BM25 search",
    )
    
    args = parser.parse_args()
    
    ingest_jsonl(
        input_path=args.input,
        collection_name=args.collection,
        batch_size=args.batch_size,
        upsert_batch_size=args.upsert_batch_size,
        limit=args.limit,
        also_elasticsearch=args.also_elasticsearch,
    )


if __name__ == "__main__":
    main()
