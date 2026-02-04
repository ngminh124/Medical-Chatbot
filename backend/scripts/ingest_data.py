#!/usr/bin/env python3
"""
Medical Data Ingestion Script

Ingest medical data from Markdown files into Qdrant vector database.
Supports batch processing with GPU acceleration for fast embedding generation.

Usage:
    python scripts/ingest_data.py
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import torch
from loguru import logger
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent / "backend"))

from src.configs.setup import get_backend_settings
from src.core.vectorize import create_collection, upsert_points

settings = get_backend_settings()


class MedicalDataIngestion:
    """Handle medical data ingestion into Qdrant vector database."""

    def __init__(self):
        """Initialize the ingestion system."""
        self.settings = settings
        self.device = self._setup_device()
        self.model = self._load_embedding_model()
        self.batch_size = self.settings.batch_size

        logger.info(f"Initialized MedicalDataIngestion with device: {self.device}")
        logger.info(f"Batch size: {self.batch_size}")
        logger.info(f"Model: {self.settings.embedding_model_name}")

    def _setup_device(self) -> str:
        """Setup and verify device (cuda/cpu/mps)."""
        requested_device = self.settings.device.lower()

        if requested_device == "cuda" and torch.cuda.is_available():
            device = "cuda"
            logger.info(f"Using CUDA GPU: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA Version: {torch.version.cuda}")
        elif requested_device == "mps" and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using Apple Metal Performance Shaders (MPS)")
        else:
            device = "cpu"
            logger.warning(
                f"Requested device '{requested_device}' not available. Using CPU."
            )

        return device

    def _load_embedding_model(self) -> SentenceTransformer:
        """Load SentenceTransformer embedding model."""
        logger.info(f"Loading embedding model: {self.settings.embedding_model_name}")
        try:
            model = SentenceTransformer(
                self.settings.embedding_model_name, device=self.device
            )
            logger.info(
                f"Model loaded successfully. Output dimension: {model.get_sentence_embedding_dimension()}"
            )
            return model
        except Exception as e:
            logger.error(f"Failed to load embedding model: {e}")
            raise

    def load_markdown_files(self, data_dir: Path) -> List[Dict]:
        """
        Load all markdown files from data directory.

        Args:
            data_dir: Path to directory containing markdown files

        Returns:
            List of document dictionaries with content and metadata
        """
        logger.info(f"Loading markdown files from: {data_dir}")
        documents = []

        if not data_dir.exists():
            logger.error(f"Data directory does not exist: {data_dir}")
            raise FileNotFoundError(f"Directory not found: {data_dir}")

        # Find all .md files
        md_files = list(data_dir.rglob("*.md"))
        logger.info(f"Found {len(md_files)} markdown files")

        for md_file in tqdm(md_files, desc="Loading markdown files"):
            try:
                with open(md_file, "r", encoding="utf-8") as f:
                    content = f.read().strip()

                if content:
                    documents.append(
                        {
                            "content": content,
                            "source": str(md_file.relative_to(data_dir)),
                            "file_path": str(md_file),
                            "doc_type": "markdown",
                        }
                    )
            except Exception as e:
                logger.warning(f"Failed to load {md_file}: {e}")
                continue

        logger.info(f"Successfully loaded {len(documents)} documents")
        return documents

    def load_jsonl_chunks(self, jsonl_path: Path) -> List[Dict]:
        """
        Load pre-chunked data from JSONL file.

        Args:
            jsonl_path: Path to JSONL file containing chunks

        Returns:
            List of chunk dictionaries
        """
        logger.info(f"Loading chunks from JSONL: {jsonl_path}")
        chunks = []

        if not jsonl_path.exists():
            logger.error(f"JSONL file does not exist: {jsonl_path}")
            raise FileNotFoundError(f"File not found: {jsonl_path}")

        try:
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        chunk = json.loads(line.strip())
                        chunks.append(chunk)
                    except json.JSONDecodeError as e:
                        logger.warning(f"Failed to parse line {line_num}: {e}")
                        continue

            logger.info(f"Successfully loaded {len(chunks)} chunks from JSONL")
            return chunks

        except Exception as e:
            logger.error(f"Failed to load JSONL file: {e}")
            raise

    def create_embeddings_batch(self, texts: List[str]) -> List[List[float]]:
        """
        Create embeddings for a batch of texts using GPU acceleration.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
            )
            return embeddings.tolist()
        except Exception as e:
            logger.error(f"Failed to create embeddings: {e}")
            raise

    def prepare_points_from_chunks(
        self, chunks: List[Dict], start_id: int = 0
    ) -> List[Dict]:
        """
        Prepare point structures from chunks with embeddings.

        Args:
            chunks: List of chunk dictionaries
            start_id: Starting ID for point numbering

        Returns:
            List of point dictionaries ready for Qdrant
        """
        logger.info(f"Preparing {len(chunks)} points with embeddings...")

        # Extract texts for embedding
        texts = [chunk.get("content", "") for chunk in chunks]

        # Create embeddings in batch
        logger.info("Creating embeddings (this may take a while)...")
        embeddings = self.create_embeddings_batch(texts)

        # Prepare points
        points = []
        for idx, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            point = {
                "id": start_id + idx,
                "embedding": embedding,
                "metadata": {
                    "content": chunk.get("content", ""),
                    "title": chunk.get("title", ""),
                    "doc_id": chunk.get("doc_id", ""),
                    "chunk_index": chunk.get("chunk_index", idx),
                    "doc_type": chunk.get("doc_type", "markdown"),
                    "source": chunk.get("source", ""),
                    "metadata": chunk.get("metadata", {}),
                },
            }
            points.append(point)

        logger.info(f"Prepared {len(points)} points successfully")
        return points

    def ingest_data(
        self, chunks: List[Dict], collection_name: str = None, batch_size: int = None
    ):
        """
        Ingest data into Qdrant in batches.

        Args:
            chunks: List of chunk dictionaries to ingest
            collection_name: Name of Qdrant collection
            batch_size: Batch size for uploading (default from settings)
        """
        if collection_name is None:
            collection_name = self.settings.default_collection_name

        if batch_size is None:
            batch_size = self.batch_size

        logger.info(f"Starting data ingestion to collection: {collection_name}")
        logger.info(f"Total chunks to process: {len(chunks)}")
        logger.info(f"Upload batch size: {batch_size}")

        # Create collection if not exists
        create_collection(
            collection_name=collection_name,
            vector_dimension=self.settings.vector_dimension,
        )

        # Process and upload in batches
        total_batches = (len(chunks) + batch_size - 1) // batch_size
        start_time = time.time()

        for batch_num in tqdm(range(total_batches), desc="Ingesting batches"):
            start_idx = batch_num * batch_size
            end_idx = min((batch_num + 1) * batch_size, len(chunks))
            batch_chunks = chunks[start_idx:end_idx]

            try:
                # Prepare points with embeddings
                points = self.prepare_points_from_chunks(
                    batch_chunks, start_id=start_idx
                )

                # Upload to Qdrant
                upsert_points(points, collection_name=collection_name)

                logger.info(
                    f"Batch {batch_num + 1}/{total_batches}: "
                    f"Uploaded {len(points)} points (IDs: {start_idx}-{end_idx - 1})"
                )

            except Exception as e:
                logger.error(f"Failed to process batch {batch_num + 1}: {e}")
                raise

        elapsed_time = time.time() - start_time
        logger.success(
            f"✓ Ingestion completed! Total chunks: {len(chunks)}, "
            f"Time: {elapsed_time:.2f}s ({len(chunks) / elapsed_time:.2f} chunks/s)"
        )


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("Medical Data Ingestion - Started")
    logger.info("=" * 80)

    try:
        # Initialize ingestion system
        ingestion = MedicalDataIngestion()

        # Check for pre-chunked JSONL file
        jsonl_path = Path(settings.chunks_output_dir) / "medical_master_data.jsonl"

        if jsonl_path.exists():
            logger.info("Found pre-chunked JSONL file, loading...")
            chunks = ingestion.load_jsonl_chunks(jsonl_path)
        else:
            logger.info("No JSONL file found, loading from markdown files...")
            # Load markdown files
            data_dir = Path(settings.data_dir)
            documents = ingestion.load_markdown_files(data_dir)

            # For this simple version, treat each document as a chunk
            # You can integrate with chunking.py here if needed
            chunks = [
                {
                    "content": doc["content"],
                    "title": Path(doc["source"]).stem,
                    "doc_id": Path(doc["source"]).stem,
                    "chunk_index": 0,
                    "doc_type": doc["doc_type"],
                    "source": doc["source"],
                    "metadata": {},
                }
                for doc in documents
            ]

        if not chunks:
            logger.warning("No chunks to ingest. Exiting.")
            return

        # Ingest data
        ingestion.ingest_data(chunks)

        logger.success("=" * 80)
        logger.success("Medical Data Ingestion - Completed Successfully!")
        logger.success("=" * 80)

    except KeyboardInterrupt:
        logger.warning("\nIngestion interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Ingestion failed: {e}")
        logger.exception("Full traceback:")
        sys.exit(1)


if __name__ == "__main__":
    main()
