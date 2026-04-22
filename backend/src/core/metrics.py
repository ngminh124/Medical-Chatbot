"""Prometheus metrics for FastAPI application monitoring."""

from prometheus_client import Counter, Gauge, Histogram

# Application info
fastapi_app_info = Gauge(
    "fastapi_app_info",
    "FastAPI application information",
    ["app_name", "version"],
)

# Request metrics
fastapi_requests_total = Counter(
    "fastapi_requests_total",
    "Total number of requests",
    ["method", "path", "status_code", "app_name"],
)

fastapi_requests_in_progress = Gauge(
    "fastapi_requests_in_progress",
    "Number of requests currently in progress",
    ["method", "path", "app_name"],
)

fastapi_requests_duration_seconds = Histogram(
    "fastapi_requests_duration_seconds",
    "Request duration in seconds",
    ["method", "path", "app_name"],
    buckets=(0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

# Response metrics
fastapi_responses_total = Counter(
    "fastapi_responses_total",
    "Total number of responses",
    ["method", "path", "status_code", "app_name"],
)

# Size metrics
fastapi_request_size_bytes = Histogram(
    "fastapi_request_size_bytes",
    "Request body size in bytes",
    ["method", "path", "app_name"],
)

fastapi_response_size_bytes = Histogram(
    "fastapi_response_size_bytes",
    "Response body size in bytes",
    ["method", "path", "app_name"],
)

# Exception metrics
fastapi_exceptions_total = Counter(
    "fastapi_exceptions_total",
    "Total number of exceptions",
    ["method", "path", "exception_type", "app_name"],
)

# RAG pipeline metrics
rag_requests_total = Counter(
    "rag_requests_total",
    "Total number of RAG requests",
)

rag_errors_total = Counter(
    "rag_errors_total",
    "Total number of RAG pipeline errors",
)

rag_request_duration_seconds = Histogram(
    "rag_request_duration_seconds",
    "RAG total request latency in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0),
)

rag_llm_duration_seconds = Histogram(
    "rag_llm_duration_seconds",
    "LLM generation latency in seconds",
    buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 20.0),
)

rag_retrieval_duration_seconds = Histogram(
    "rag_retrieval_duration_seconds",
    "Retrieval latency in seconds",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
)

rag_cache_requests_total = Counter(
    "rag_cache_requests_total",
    "Total retrieval cache lookups",
)

rag_cache_hits_total = Counter(
    "rag_cache_hits_total",
    "Total retrieval cache hits",
)

rag_cache_misses_total = Counter(
    "rag_cache_misses_total",
    "Total retrieval cache misses",
)
