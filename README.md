# Vietnamese Medical Chatbot System

## Table of Contents

1. Overview

   * 1.1 Features
   * 1.2 System Architecture
2. Technology Stack

   * 2.1 Frontend
   * 2.2 Backend
   * 2.3 AI Components
   * 2.4 Infrastructure
3. Installation
4. Configuration
5. Project Structure
6. RAG Pipeline

   * 6.1 Authentication
   * 6.2 Query Processing
   * 6.3 Hybrid Retrieval
   * 6.4 Reranking
   * 6.5 Response Generation
7. Evaluation Results

   * 7.1 Retrieval Metrics
   * 7.2 Generation Metrics
   * 7.3 Performance Metrics
8. API Reference
9. Monitoring
10. License

---

# 1. Overview

## 1.1 Features

* User authentication using JWT and Google OAuth
* Conversation management with threads and messages
* Hybrid Retrieval combining Qdrant and Elasticsearch
* Cross-Encoder reranking using Qwen3-Reranker
* Streaming AI responses using Server-Sent Events (SSE)
* Medical question answering using Retrieval-Augmented Generation (RAG)
* User feedback collection and analytics dashboard
* Runtime configuration management
* Monitoring using Prometheus, Grafana, Loki and Tempo

---

## 1.2 System Architecture

Frontend (React + Vite)
│
▼
FastAPI Backend
│
┌──────┼────────────────────┐
│      │                    │
▼      ▼                    ▼

PostgreSQL      Qdrant      Elasticsearch
Users           Vectors     BM25 Search
Threads
Messages
Feedbacks

```
    │
    ▼
```

Hybrid Search (RRF)
│
▼

Qwen3 Reranker
│
▼

Qwen3 Generator
(Ollama / vLLM)

---

# 2. Technology Stack

## 2.1 Frontend

| Component        | Technology        |
| ---------------- | ----------------- |
| Framework        | React             |
| Build Tool       | Vite              |
| Routing          | React Router      |
| State Management | React Context API |
| Styling          | TailwindCSS       |

---

## 2.2 Backend

| Component      | Technology        |
| -------------- | ----------------- |
| Framework      | FastAPI           |
| ORM            | SQLAlchemy        |
| Validation     | Pydantic          |
| Authentication | JWT               |
| API Docs       | OpenAPI / Swagger |

---

## 2.3 AI Components

| Component        | Model                |
| ---------------- | -------------------- |
| Embedding        | Qwen3-Embedding-0.6B |
| Reranking        | Qwen3-Reranker-0.6B  |
| Generation       | Qwen3                |
| Evaluation Judge | DeepSeek-Chat        |

---

## 2.4 Infrastructure

| Service       | Purpose            |
| ------------- | ------------------ |
| PostgreSQL    | Persistent Storage |
| Qdrant        | Vector Database    |
| Elasticsearch | Full Text Search   |
| Redis         | Response Cache     |
| Prometheus    | Metrics            |
| Loki          | Logs               |
| Tempo         | Traces             |
| Grafana       | Visualization      |

---

# 6. RAG Pipeline

## 6.1 Authentication

* User logs in using Email/Password or Google OAuth.
* Backend creates JWT Access Token.
* Token stored in Session Storage.
* Protected endpoints use Depends(get_current_user).

---

## 6.2 Query Processing

1. User sends a message.
2. Message is stored in PostgreSQL.
3. Recent conversation history is loaded.
4. Query is optionally rewritten.
5. Intent detection is executed.

---

## 6.3 Hybrid Retrieval

The system performs retrieval from:

### Vector Search

* Qdrant
* Qwen3 Embedding
* Cosine Similarity

### Keyword Search

* Elasticsearch
* BM25

### Fusion

Results are merged using Reciprocal Rank Fusion (RRF).

---

## 6.4 Reranking

Qwen3-Reranker is applied to:

* Remove irrelevant documents.
* Improve retrieval precision.
* Select Top-K final contexts.

---

## 6.5 Response Generation

Input:

* User Question
* Conversation History
* Retrieved Contexts

Output:

* Generated Answer
* Citations
* Route Information

Responses are returned either:

* Standard REST API
* Streaming SSE API

---

# 7. Evaluation Results

Evaluation conducted using:

* Ragas v0.3.9
* DeepSeek-Chat as Judge Model
* Vietnamese Medical QA Dataset

## 7.1 Retrieval Metrics

### Ragas Retrieval Quality

| Metric            | Score  |
| ----------------- | ------ |
| Context Precision | 0.6972 |
| Context Recall    | 0.7328 |

### Traditional Retrieval Metrics

| Metric       | Score  |
| ------------ | ------ |
| MRR          | 0.5673 |
| Recall@1     | 0.5200 |
| Recall@3     | 0.7100 |
| Recall@5     | 0.7600 |
| Recall@10    | 0.7600 |
| nDCG@1       | 0.5100 |
| nDCG@3       | 0.5690 |
| nDCG@5       | 0.5698 |
| nDCG@10      | 0.5698 |
| Precision@1  | 0.5200 |
| Precision@3  | 0.2367 |
| Precision@5  | 0.1520 |
| Precision@10 | 0.0760 |

---

## 7.2 Generation Metrics

| Metric           | Score  |
| ---------------- | ------ |
| Faithfulness     | 0.8363 |
| Answer Relevancy | 0.7679 |

---

## 7.3 Performance Metrics

| Metric                     | Value      |
| -------------------------- | ---------- |
| Average End-to-End Latency | 3176.30 ms |
| Average Embedding Latency  | 23.45 ms   |
| Average Retrieval Latency  | 92.66 ms   |
| Average Reranking Latency  | 864.99 ms  |
| Average Generation Latency | 2195.20 ms |
| P50 Latency                | 3035.71 ms |
| P95 Latency                | 5543.36 ms |

---

# 8. API Reference

## Authentication

POST /v1/auth/register

POST /v1/auth/login

POST /v1/auth/google

GET /v1/auth/me

---

## Chat

POST /v1/chat/threads

GET /v1/chat/threads

GET /v1/chat/threads/{thread_id}

DELETE /v1/chat/threads/{thread_id}

POST /v1/chat/threads/{thread_id}/ask

POST /v1/chat/threads/{thread_id}/ask-stream

POST /v1/chat/messages/{message_id}/feedback

---

## Admin

GET /v1/admin/overview

GET /v1/admin/users

POST /v1/admin/users

PUT /v1/admin/users/{id}

DELETE /v1/admin/users/{id}

GET /v1/admin/conversations

GET /v1/admin/feedbacks

GET /v1/admin/settings

PUT /v1/admin/settings

---

# 9. Monitoring

* Prometheus for metrics
* Loki for logs
* Tempo for tracing
* Grafana dashboards

Observed metrics:

* Request Rate
* Latency
* Cache Hit Rate
* Active Requests
* Tokens Per Second
* User Activity
* Feedback Statistics

---

# 10. License

MIT License
