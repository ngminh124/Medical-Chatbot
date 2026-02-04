#!/usr/bin/env python3
"""
Medical Data Ingestion Script - JSONL to Qdrant

Script nạp dữ liệu y tế từ file JSONL vào Qdrant vector database.
Xử lý 40,000+ chunks với batch processing và GPU acceleration.

Usage:
    python -m backend.scripts.ingest_jsonl_to_qdrant
"""

import json
import sys
from pathlib import Path
from typing import Dict, Iterator, List

from loguru import logger
from tqdm import tqdm

# Import từ các module core
from backend.src.configs.setup import get_backend_settings
from backend.src.core.vectorize import create_collection, upsert_points
from backend.src.services.embedding import get_embedding_service, Qwen3EmbeddingService

# Lấy settings
settings = get_backend_settings()


class JSONLIngestion:
    """Class xử lý nạp dữ liệu từ JSONL vào Qdrant"""

    def __init__(
        self,
        jsonl_path: str,
        collection_name: str = None,
        embedding_batch_size: int = 64,
        upsert_batch_size: int = 500,
    ):
        """
        Khởi tạo ingestion system

        Args:
            jsonl_path: Đường dẫn đến file JSONL
            collection_name: Tên collection Qdrant
            embedding_batch_size: Batch size cho tạo embedding
            upsert_batch_size: Batch size cho upsert vào Qdrant
        """
        self.jsonl_path = Path(jsonl_path)
        self.collection_name = collection_name or settings.default_collection_name
        self.embedding_batch_size = embedding_batch_size
        self.upsert_batch_size = upsert_batch_size

        # Validate file exists
        if not self.jsonl_path.exists():
            raise FileNotFoundError(f"File không tồn tại: {self.jsonl_path}")

        logger.info("=" * 80)
        logger.info("Khởi tạo Medical Data Ingestion System")
        logger.info("=" * 80)
        logger.info(f"File JSONL: {self.jsonl_path}")
        logger.info(f"Collection: {self.collection_name}")
        logger.info(f"Embedding batch size: {self.embedding_batch_size}")
        logger.info(f"Upsert batch size: {self.upsert_batch_size}")

        # Khởi tạo embedding service (singleton)
        logger.info("Đang khởi tạo Embedding Service...")
        self.embedding_service = get_embedding_service()
        logger.success(
            f"Embedding Service đã sẵn sàng! Dimension: {self.embedding_service.get_embedding_dimension()}"
        )

    def count_lines(self) -> int:
        """
        Đếm số dòng trong file JSONL

        Returns:
            Số dòng trong file
        """
        logger.info("Đang đếm số dòng trong file...")
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            count = sum(1 for _ in f)
        logger.info(f"Tổng số chunks: {count:,}")
        return count

    def read_jsonl_batches(
        self, batch_size: int
    ) -> Iterator[tuple[List[Dict], int, int]]:
        """
        Đọc file JSONL theo batch để tránh tràn RAM

        Args:
            batch_size: Kích thước mỗi batch

        Yields:
            Tuple (batch_data, start_index, end_index)
        """
        batch = []
        line_count = 0

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    batch.append(data)
                    line_count += 1

                    if len(batch) >= batch_size:
                        yield batch, line_count - len(batch), line_count - 1
                        batch = []

                except json.JSONDecodeError as e:
                    logger.warning(f"Lỗi parse JSON tại dòng {line_count + 1}: {e}")
                    continue

            # Yield batch cuối cùng
            if batch:
                yield batch, line_count - len(batch), line_count - 1

    def prepare_points_from_batch(
        self, batch: List[Dict], embeddings: List[List[float]]
    ) -> List[Dict]:
        """
        Chuẩn bị points từ batch data và embeddings

        Args:
            batch: Batch data từ JSONL
            embeddings: List embeddings tương ứng

        Returns:
            List points cho Qdrant
        """
        points = []

        for data, embedding in zip(batch, embeddings):
            # Giữ nguyên ID từ JSONL
            point_id = data["id"]

            # Content field
            content = data.get("content", "")

            # Metadata từ JSONL
            metadata = data.get("metadata", {})

            # Tạo payload: metadata + content
            payload = {
                "content": content,  # Thêm content vào payload để retrieve
                **metadata,  # Spread metadata fields
            }

            point = {"id": point_id, "embedding": embedding, "metadata": payload}

            points.append(point)

        return points

    def ingest(self):
        """
        Thực hiện quá trình ingest dữ liệu vào Qdrant

        Quy trình:
        1. Tạo collection nếu chưa có
        2. Đọc JSONL theo batch
        3. Tạo embedding cho mỗi batch
        4. Gom points và upsert vào Qdrant
        """
        logger.info("=" * 80)
        logger.info("Bắt đầu quá trình INGEST")
        logger.info("=" * 80)

        # Bước 1: Tạo collection
        logger.info("Bước 1: Tạo/kiểm tra collection...")
        create_collection(
            collection_name=self.collection_name,
            vector_dimension=self.embedding_service.get_embedding_dimension(),
        )

        # Bước 2: Đếm tổng số chunks
        total_chunks = self.count_lines()

        if total_chunks == 0:
            logger.warning("Không có chunks để xử lý!")
            return

        # Bước 3: Xử lý dữ liệu theo batch
        logger.info("Bước 2: Bắt đầu xử lý batch...")

        points_buffer = []  # Buffer để gom points trước khi upsert
        processed_count = 0

        with tqdm(total=total_chunks, desc="Ingesting chunks") as pbar:
            # Đọc JSONL theo embedding batch size
            for batch, start_idx, end_idx in self.read_jsonl_batches(
                self.embedding_batch_size
            ):

                # Trích xuất content để tạo embedding
                texts = [item.get("content", "") for item in batch]

                # Tạo embeddings cho batch (KHÔNG có instruction - đây là documents)
                try:
                    embeddings = self.embedding_service.embed_batch_documents(
                        texts, batch_size=self.embedding_batch_size
                    )
                except Exception as e:
                    logger.error(
                        f"Lỗi tạo embedding cho batch {start_idx}-{end_idx}: {e}"
                    )
                    continue

                # Chuẩn bị points
                points = self.prepare_points_from_batch(batch, embeddings)

                # Thêm vào buffer
                points_buffer.extend(points)
                processed_count += len(batch)

                # Nếu buffer đủ lớn, upsert vào Qdrant
                if len(points_buffer) >= self.upsert_batch_size:
                    try:
                        upsert_points(
                            points_buffer[: self.upsert_batch_size],
                            collection_name=self.collection_name,
                        )
                        logger.info(
                            f"✓ Đã nạp {processed_count}/{total_chunks} chunks "
                            f"({processed_count/total_chunks*100:.1f}%)"
                        )
                        points_buffer = points_buffer[self.upsert_batch_size :]
                    except Exception as e:
                        logger.error(f"Lỗi upsert batch: {e}")
                        raise

                # Update progress bar
                pbar.update(len(batch))

        # Upsert phần còn lại trong buffer
        if points_buffer:
            try:
                upsert_points(points_buffer, collection_name=self.collection_name)
                logger.info(
                    f"✓ Đã nạp {processed_count}/{total_chunks} chunks (100.0%)"
                )
            except Exception as e:
                logger.error(f"Lỗi upsert batch cuối: {e}")
                raise

        logger.success("=" * 80)
        logger.success(f"HOÀN THÀNH! Đã nạp {processed_count:,} chunks thành công")
        logger.success("=" * 80)


def main():
    """Main execution function"""
    # Configure logger
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>",
        level="INFO",
    )

    try:
        # Đường dẫn file JSONL
        jsonl_path = (
            "/home/nguyenminh/Projects/Vietnamese-Medical-Chatbot"
            "/data/chunks/medical_master_data.jsonl"
        )

        # Khởi tạo ingestion
        ingestion = JSONLIngestion(
            jsonl_path=jsonl_path,
            embedding_batch_size=64,  # Batch cho embedding
            upsert_batch_size=500,  # Batch cho upsert
        )

        # Thực hiện ingest
        ingestion.ingest()

    except KeyboardInterrupt:
        logger.warning("\n⚠ Quá trình bị ngắt bởi người dùng")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Lỗi trong quá trình ingest: {e}")
        logger.exception("Chi tiết lỗi:")
        sys.exit(1)


if __name__ == "__main__":
    main()
