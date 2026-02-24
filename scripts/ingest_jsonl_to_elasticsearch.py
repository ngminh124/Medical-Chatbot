"""Ingest medical JSONL chunks into Elasticsearch for BM25 search.

This script reads chunks from the same JSONL file used for Qdrant,
and indexes them into Elasticsearch for keyword/BM25 search.

Usage:
    # Index all data
    python scripts/ingest_jsonl_to_elasticsearch.py

    # Index with limit (for testing)
    python scripts/ingest_jsonl_to_elasticsearch.py --limit 100

    # Custom input file
    python scripts/ingest_jsonl_to_elasticsearch.py --input data/chunks/test.jsonl

Prerequisites:
- Elasticsearch running (docker-compose up -d elasticsearch)
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from backend.src.services.elastic_search import ElasticsearchClient


def read_jsonl(filepath: Path, limit: int = None) -> List[Dict[str, Any]]:
    """Read JSONL file line by line."""
    chunks = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            try:
                chunks.append(json.loads(line))
            except json.JSONDecodeError as e:
                logger.warning(f"Skipping line {i+1}: invalid JSON - {e}")
    return chunks


def ingest_to_elasticsearch(
    input_path: Path,
    batch_size: int = 500,
    limit: int = None,
    recreate_index: bool = False,
):
    """Index JSONL chunks into Elasticsearch."""

    # 1. Connect
    es = ElasticsearchClient()
    logger.info(f"Connected to Elasticsearch at {es.host}:{es.port}")

    # 2. Create index (idempotent — skips if already exists)
    if recreate_index:
        try:
            es.client.indices.delete(index=es.index_name)
            logger.info(f"Deleted existing index '{es.index_name}'")
        except Exception:
            pass

    if not es.create_index():
        logger.error("Failed to create index")
        return 0

    # 3. Read chunks
    logger.info(f"Reading {input_path}...")
    chunks = read_jsonl(input_path, limit=limit)
    if not chunks:
        logger.error("No chunks loaded")
        return 0
    logger.info(f"Loaded {len(chunks)} chunks")

    # 4. Bulk index in batches
    total_indexed = 0
    failed = 0

    for batch_start in range(0, len(chunks), batch_size):
        batch_end = min(batch_start + batch_size, len(chunks))
        batch = chunks[batch_start:batch_end]

        for chunk in batch:
            chunk_id = chunk.get("id", "")
            metadata = chunk.get("metadata", {})

            success = es.index_chunk(
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
                total_indexed += 1
            else:
                failed += 1

        logger.info(
            f"Progress: {batch_end}/{len(chunks)} "
            f"(indexed={total_indexed}, failed={failed})"
        )

    # 5. Refresh index to make documents searchable immediately
    es.client.indices.refresh(index=es.index_name)

    logger.success(
        f"Elasticsearch ingestion completed! "
        f"Indexed: {total_indexed}, Failed: {failed}"
    )
    return total_indexed


def main():
    parser = argparse.ArgumentParser(
        description="Ingest medical JSONL chunks into Elasticsearch"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/chunks/medical_master_data6.jsonl"),
        help="Input JSONL file path",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=500,
        help="Batch size for progress logging",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of chunks (for testing)",
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Delete and recreate index before ingesting",
    )

    args = parser.parse_args()

    ingest_to_elasticsearch(
        input_path=args.input,
        batch_size=args.batch_size,
        limit=args.limit,
        recreate_index=args.recreate,
    )


if __name__ == "__main__":
    main()
