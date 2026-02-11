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
import time

from loguru import logger
from tqdm import tqdm

# Import từ các module core
from backend.src.configs.setup import get_backend_settings
from backend.src.core.vectorize import create_collection, upsert_points
from backend.src.services.embedding import get_embedding_service, Qwen3EmbeddingService

# Lấy settings
settings = get_backend_settings()

# Checkpoint directory
CHECKPOINT_DIR = Path("/home/nguyenminh/Projects/Vietnamese-Medical-Chatbot/data/checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)


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
        
        # Checkpoint file
        self.checkpoint_file = CHECKPOINT_DIR / f"checkpoint_{self.collection_name}.json"

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

    def save_checkpoint(self, processed_count: int, total_chunks: int):
        """
        Lưu checkpoint tiến độ hiện tại

        Args:
            processed_count: Số chunks đã xử lý
            total_chunks: Tổng số chunks
        """
        checkpoint = {
            "jsonl_path": str(self.jsonl_path),
            "collection_name": self.collection_name,
            "processed_count": processed_count,
            "total_chunks": total_chunks,
            "progress_percent": (processed_count / total_chunks * 100) if total_chunks > 0 else 0,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        try:
            with open(self.checkpoint_file, "w", encoding="utf-8") as f:
                json.dump(checkpoint, f, indent=2, ensure_ascii=False)
            logger.info(f"💾 Lưu checkpoint: {processed_count}/{total_chunks} chunks")
        except Exception as e:
            logger.error(f"Lỗi lưu checkpoint: {e}")

    def load_checkpoint(self) -> int:
        """
        Tải checkpoint từ file trước đó

        Returns:
            Số chunks đã xử lý trước đó (0 nếu không có checkpoint)
        """
        if not self.checkpoint_file.exists():
            logger.info("Không tìm thấy checkpoint, bắt đầu từ đầu")
            return 0
        
        try:
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                checkpoint = json.load(f)
            
            processed = checkpoint.get("processed_count", 0)
            total = checkpoint.get("total_chunks", 0)
            progress = checkpoint.get("progress_percent", 0)
            timestamp = checkpoint.get("timestamp", "unknown")
            
            logger.info(f"📂 Tìm thấy checkpoint từ {timestamp}")
            logger.info(f"📍 Đã xử lý: {processed}/{total} chunks ({progress:.1f}%)")
            logger.info(f"⏸️  Resume từ chunk thứ {processed}")
            
            return processed
        except Exception as e:
            logger.error(f"Lỗi tải checkpoint: {e}")
            return 0

    def read_jsonl_batches_resumed(
        self, batch_size: int, start_from: int
    ) -> Iterator[tuple[List[Dict], int, int]]:
        """
        Đọc file JSONL theo batch, có thể bắt đầu từ một vị trí nhất định

        Args:
            batch_size: Kích thước mỗi batch
            start_from: Chunk index để bắt đầu từ đó (0-indexed)

        Yields:
            Tuple (batch_data, start_index, end_index)
        """
        batch = []
        line_count = 0

        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                # Bỏ qua các dòng trước vị trí resume
                if line_count < start_from:
                    line_count += 1
                    continue
                
                try:
                    data = json.loads(line.strip())
                    batch.append(data)
                    line_count += 1

                    if len(batch) >= batch_size:
                        yield batch, line_count - len(batch), line_count - 1
                        batch = []

                except json.JSONDecodeError as e:
                    logger.warning(f"Lỗi parse JSON tại dòng {line_count + 1}: {e}")
                    line_count += 1
                    continue

            # Yield batch cuối cùng
            if batch:
                yield batch, line_count - len(batch), line_count - 1

    def ingest(self):
        """
        Thực hiện quá trình ingest dữ liệu vào Qdrant

        Quy trình:
        1. Tạo collection nếu chưa có
        2. Tải checkpoint nếu có
        3. Đọc JSONL theo batch từ vị trí checkpoint
        4. Tạo embedding cho mỗi batch
        5. Gom points và upsert vào Qdrant
        6. Lưu checkpoint sau mỗi upsert
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

        # Bước 3: Tải checkpoint
        start_from = self.load_checkpoint()
        remaining_chunks = total_chunks - start_from

        if remaining_chunks == 0:
            logger.success("✓ Tất cả chunks đã được xử lý hoàn toàn!")
            return

        logger.info(f"📊 Còn {remaining_chunks:,} chunks cần xử lý")

        # Bước 4: Xử lý dữ liệu theo batch
        logger.info("Bước 2: Bắt đầu xử lý batch...")

        points_buffer = []  # Buffer để gom points trước khi upsert
        processed_count = start_from

        with tqdm(total=remaining_chunks, desc="Ingesting chunks", initial=0) as pbar:
            # Đọc JSONL theo embedding batch size, bắt đầu từ checkpoint
            for batch, start_idx, end_idx in self.read_jsonl_batches_resumed(
                self.embedding_batch_size, start_from=start_from
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
                        logger.success(
                            f"✓ Upserted {self.upsert_batch_size} points. Total: {processed_count}"
                        )
                        
                        # Lưu checkpoint sau mỗi upsert thành công
                        self.save_checkpoint(processed_count, total_chunks)
                        
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
                processed_count = total_chunks  # Cập nhật để indicate hoàn tất
                logger.success(
                    f"✓ Upserted {len(points_buffer)} points. Total: {processed_count}"
                )
                
                # Lưu checkpoint cuối cùng
                self.save_checkpoint(processed_count, total_chunks)
            except Exception as e:
                logger.error(f"Lỗi upsert batch cuối: {e}")
                raise

        logger.success("=" * 80)
        logger.success(f"✅ HOÀN THÀNH! Đã nạp {processed_count:,} chunks thành công")
        logger.success("=" * 80)


def main():
    """Main execution function"""
    # Configure logger
    logger.remove()
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
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
            embedding_batch_size=4,  # Batch cho embedding
            upsert_batch_size=500,  # Batch cho upsert
        )

        # Thực hiện ingest
        ingestion.ingest()

    except KeyboardInterrupt:
        logger.warning("\n⚠️  Quá trình bị dừng bởi người dùng (Ctrl+C)")
        logger.info("💾 Checkpoint đã được lưu. Chạy lại để tiếp tục từ vị trí này.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"❌ Lỗi trong quá trình ingest: {e}")
        logger.exception("Chi tiết lỗi:")
        sys.exit(1)


if __name__ == "__main__":
    main()
