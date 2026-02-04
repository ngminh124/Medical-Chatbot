import sys
import os
# Thêm đường dẫn để script hiểu được các module trong src
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from services.chunking import DocumentChunker
from services.embedding import EmbeddingService
from core.vectorize import QdrantStorage

def main():
    # 1. Đọc file Markdown 700 trang từ Colab
    file_path = "/home/nguyenminh/Projects/Vietnamese-Medical-Chatbot/data/chunks/medical_master_data.jsonl"
    
    # 2. Dùng service chunking để cắt nhỏ
    # (Tận dụng logic có sẵn trong dự án mẫu)
    chunker = DocumentChunker()
    chunks = chunker.process_file(file_path)
    
    # 3. Chỉ lấy 20 chunks đầu tiên để test
    chunks = chunks[:20]
    print(f"📝 Chỉ đẩy {len(chunks)} chunks để test")
    
    # 4. Dùng service embedding và storage để đẩy vào Qdrant
    storage = QdrantStorage()
    storage.upload_documents(chunks)
    
    print("✅ Hoàn thành nạp dữ liệu vào hệ thống RAG!")

if __name__ == "__main__":
    main()