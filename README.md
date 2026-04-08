# Vietnamese Medical Chatbot

Hệ thống chatbot y tế tiếng Việt theo kiến trúc RAG, gồm:
- Backend FastAPI (RAG, auth, chat, STT/TTS, metrics)
- Frontend React + Vite
- PostgreSQL cho dữ liệu hội thoại/người dùng
- Qdrant + Elasticsearch cho truy xuất tri thức

## Kiến trúc tổng quan

```
Client (React)
    ↓ HTTP
Backend API (FastAPI)
    ├── Auth + Chat threads/messages (PostgreSQL)
    ├── RAG pipeline
    │   ├── Dense search (Qdrant)
    │   └── Lexical search (Elasticsearch)
    ├── STT/TTS proxy services
    └── Metrics (/metrics) + health checks
```

## Cấu trúc dự án

```
Vietnamese-Medical-Chatbot/
├── backend/
│   ├── main.py                    # FastAPI app entrypoint
│   ├── src/
│   │   ├── configs/               # Pydantic settings
│   │   ├── core/                  # Core RAG/search/cache/metrics
│   │   ├── routers/               # API routers (auth, chat, rag, stt, tts...)
│   │   ├── services/              # Embedding, rerank, TTS, STT, brain
│   │   ├── schemas/               # Request/response schemas
│   │   └── database.py            # Database helpers
│   ├── models/                    # SQLAlchemy models
│   ├── scripts/                   # Scripts nội bộ backend
│   └── tests/                     # Test backend
├── frontend/
│   ├── src/
│   │   ├── api/                   # Axios/fetch client modules
│   │   ├── components/            # UI components
│   │   ├── pages/                 # App pages
│   │   └── contexts/hooks/        # State management
│   └── Dockerfile
├── database/
│   ├── docker-compose.yml         # PostgreSQL service
│   └── init.sql                   # Database init script
├── data/                          # Dữ liệu raw/processed/chunks/vector_db...
├── qdrant_data/                   # Qdrant persistent storage (local)
├── rehierarchy_output/            # Dữ liệu output xử lý lại tài liệu
├── scripts/                       # Scripts tiện ích cấp project
├── temp/                          # File tạm
├── requirements.txt
├── .env.example
└── README.md
```

## Yêu cầu môi trường

- Python 3.10+
- Node.js 18+
- Docker + Docker Compose

## Thiết lập nhanh

### 1) Cài đặt backend

```bash
pip install -r requirements.txt
```

### 2) Tạo file môi trường

```bash
cp .env.example .env
```

Sau đó cập nhật giá trị thật cho các biến API key và cấu hình host/port nếu cần.

### 3) Khởi động hạ tầng dữ liệu

```bash
docker compose -f backend/docker-compose.yml up -d
docker compose -f database/docker-compose.yml up -d
```

### 4) Chạy backend

```bash
uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000
```

### 5) Chạy frontend

```bash
cd frontend
npm install
npm run dev
```

Frontend mặc định chạy ở `http://localhost:5173` (dev) hoặc `http://localhost:3000` (docker).

## Biến môi trường chính

- Hệ thống vector/search: `QDRANT_HOST`, `QDRANT_PORT`, `ELASTICSEARCH_HOST`, `ELASTICSEARCH_PORT`
- Embedding/RAG: `EMBEDDING_MODEL_NAME`, `VECTOR_DIMENSION`, `TOP_K`
- Generation providers: `VLLM_URL`, `OLLAMA_URL`, `MODEL_NAME`
- Auth: `JWT_SECRET_KEY`
- Database: `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`, `POSTGRES_PORT`, `POSTGRES_DB`
- Audio: `ELEVENLABS_API_KEY`, `ELEVENLABS_VOICE_ID`, `STT_GPU_SERVICE_URL`

## API nổi bật

- `GET /` — thông tin service
- `GET /docs` — Swagger UI
- `GET /metrics` — Prometheus metrics
- `GET /v1/health/*` — health checks
- `POST /v1/auth/register`, `POST /v1/auth/login`
- `POST /v1/chat/threads/:id/ask` và `.../ask-stream`
- `POST /v1/stt/transcribe`, `POST /v1/tts/synthesize`

## Ghi chú

- Các thư mục dữ liệu lớn và output nội bộ (`qdrant_data`, `rehierarchy_output`, `temp`, ...) đã được cấu hình bỏ qua khỏi git.
- Không commit file `.env` thật lên repository.
