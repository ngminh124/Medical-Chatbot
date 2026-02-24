# Vietnamese Medical Chatbot - RAG System

Hệ thống RAG (Retrieval-Augmented Generation) y tế tiếng Việt sử dụng Qdrant vector database và embedding models.

## 📁 Cấu trúc thư mục

```
Vietnamese-Medical-Chatbot/
├── backend/
│   ├── src/
│   │   ├── configs/          # Cấu hình hệ thống
│   │   │   ├── setup.py      # Pydantic settings
│   │   │   └── __init__.py
│   │   ├── core/             # Core logic
│   │   │   └── vectorize.py  # Qdrant operations
│   │   └── services/         # Business logic
│   │       ├── embedding.py  # Embedding service
│   │       └── chunking.py   # Document chunking
│   └── scripts/              # Legacy scripts (deprecated)
├── scripts/                  # Utility scripts (NEW)
│   └── ingest_jsonl_to_qdrant.py  # Data ingestion
├── data/
│   ├── chunks/               # Processed chunks
│   │   └── medical_master_data.jsonl
│   ├── output/               # Markdown outputs
│   └── processed/            # Processed data
├── .env                      # Environment variables
├── docker-compose.yml        # Docker services
├── requirements.txt          # Python dependencies
└── README.md                 # This file
```

## 🚀 Bắt đầu nhanh

### 1. Cài đặt dependencies

```bash
pip install -r requirements.txt
```

### 2. Cấu hình môi trường

File `.env` đã được cấu hình sẵn với:
- Qdrant: localhost:6333
- Embedding Model: intfloat/multilingual-e5-large (1024-dim)
- Device: CUDA (GPU)

### 3. Khởi động Qdrant

```bash
docker-compose up -d
```

### 4. Nạp dữ liệu vào Qdrant

```bash
python scripts/ingest_jsonl_to_qdrant.py
```

## 📊 Chi tiết Scripts

### scripts/ingest_jsonl_to_qdrant.py

Script nạp 40,000+ chunks dữ liệu y tế từ JSONL vào Qdrant.

**Tính năng:**
- ✅ Đọc JSONL streaming (tránh tràn RAM)
- ✅ Batch embedding (64 chunks/batch)
- ✅ Batch upsert (500 points/batch)
- ✅ GPU acceleration (CUDA)
- ✅ Progress tracking (loguru + tqdm)
- ✅ Error handling robust

**Sử dụng:**
```bash
python scripts/ingest_jsonl_to_qdrant.py
```

## 🔧 Cấu hình

### Environment Variables (.env)

```bash
# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
DEFAULT_COLLECTION_NAME=medical_data

# Embedding Model (1024-dim)
EMBEDDING_MODEL_NAME=intfloat/multilingual-e5-large
VECTOR_DIMENSION=1024

# Device
DEVICE=cuda  # cuda, cpu, hoặc mps
```

## 📦 Dependencies

- `loguru` - Logging
- `qdrant-client` - Vector database client
- `pydantic-settings` - Configuration management
- `sentence-transformers` - Embedding models
- `torch` - Deep learning framework
- `tqdm` - Progress bars

## 🏗️ Architecture

### Data Flow: Ingestion

```
JSONL File (40k chunks)
    ↓
Read in batches (64)
    ↓
Create embeddings (GPU)
    ↓
Buffer points (500)
    ↓
Upsert to Qdrant
```

### Module Dependencies

```
scripts/ingest_jsonl_to_qdrant.py
    ├── backend.src.configs.setup (Settings)
    ├── backend.src.core.vectorize (Qdrant ops)
    └── backend.src.services.embedding (Embedding service)
```

## 📝 Data Format

### Input: JSONL Format

```json
{
  "id": "QTKT_chinh_hinh_0001",
  "content": "Nội dung y tế...",
  "metadata": {
    "source_file": "/path/to/file.md",
    "file_name": "QTKT_chinh_hinh",
    "heading_hierarchy": ["Title", "Section"],
    "chunk_index": 1
  }
}
```

### Output: Qdrant Points

```python
{
  "id": "QTKT_chinh_hinh_0001",  # Giữ nguyên từ JSONL
  "vector": [0.1, 0.2, ...],     # 1024-dim embedding
  "payload": {
    "content": "Nội dung y tế...", # Để retrieve
    "source_file": "...",          # Metadata fields
    "file_name": "...",
    "heading_hierarchy": [...],
    "chunk_index": 1
  }
}
```

## 🎯 Performance

- **GPU**: NVIDIA RTX 4090
- **Throughput**: ~110 chunks/second
- **Total time**: ~5-6 phút cho 35,941 chunks
- **Memory**: Streaming processing (low RAM usage)

## 🔍 Troubleshooting

### CUDA Out of Memory

Giảm `embedding_batch_size`:
```python
ingestion = JSONLIngestion(
    jsonl_path=jsonl_path,
    embedding_batch_size=32,  # Giảm từ 64
)
```

### Qdrant Connection Error

Kiểm tra Qdrant đang chạy:
```bash
docker ps | grep qdrant
curl http://localhost:6333/collections
```

## 📄 License

MIT License
