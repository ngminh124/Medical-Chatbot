import time
import logging
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from opentelemetry import trace
from pydantic import BaseModel, Field
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import \
    OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from prometheus_client import make_asgi_app

from .src.configs.setup import get_backend_settings
# Import metrics first (before routers to avoid circular import)
from .src.core import metrics  # noqa: F401
from .src.core.vectorize import create_collection
from .models import init_db
# Import routers
from .src.routers import audio, documents, health, models, rag
from .src.routers.auth import router as auth_router
from .src.routers.auth import ensure_default_admin
from .src.routers.admin import router as admin_router
from .src.routers.chat import router as chat_router
from .src.database import SessionLocal

settings = get_backend_settings()

_log_dir = Path(__file__).resolve().parent / "logs"
_log_dir.mkdir(parents=True, exist_ok=True)
try:
    logger.add(
        str(_log_dir / "app.log"),
        rotation="50 MB",
        retention="7 days",
        enqueue=True,
        backtrace=False,
        diagnose=False,
    )
except Exception as e:
    logger.warning(f"⚠️  File logging disabled (permission/error): {e}")


class _UvicornAccessFilter(logging.Filter):
    """Drop high-frequency metrics/admin-metrics access logs."""

    NOISY_PATH_PREFIXES = (
        "/metrics",
        "/v1/admin/metrics",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            args = getattr(record, "args", ())
            if isinstance(args, tuple) and len(args) >= 3:
                path = str(args[2])
                if any(path.startswith(prefix) for prefix in self.NOISY_PATH_PREFIXES):
                    return False
        except Exception:
            return True
        return True


logging.getLogger("uvicorn.access").addFilter(_UvicornAccessFilter())

# Configure OpenTelemetry tracer
tracer_provider = TracerProvider()
trace.set_tracer_provider(tracer_provider)
tracer = trace.get_tracer(__name__)

# Configure OTLP exporter for Tempo
if settings.tempo_enabled:
    try:
        otlp_exporter = OTLPSpanExporter(
            endpoint=settings.tempo_endpoint,
            insecure=True,
        )
        span_processor = BatchSpanProcessor(otlp_exporter)
        tracer_provider.add_span_processor(span_processor)
        logger.info(f"✅ OpenTelemetry tracing configured: {settings.tempo_endpoint}")
    except Exception as e:
        logger.warning(
            f"⚠️  Failed to configure OpenTelemetry exporter: {e}. Tracing will be disabled."
        )
else:
    logger.info("⏭️  Tempo tracing disabled (TEMPO_ENABLED=false)")

# FastAPI
app = FastAPI(title=settings.app_name, version=settings.app_version)

# CORS — allow frontend to connect
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://medical-chatbot-jade.vercel.app",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Instrument FastAPI with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)

# Set app info metric (Gauge with labels)
from .src.core.metrics import fastapi_app_info

fastapi_app_info.labels(app_name=settings.app_name, version=settings.app_version).set(1)


# Add custom middleware for FastAPI metrics
@app.middleware("http")
async def prometheus_middleware(request: Request, call_next):
    """Custom middleware to track FastAPI metrics"""
    from .src.core.metrics import (fastapi_exceptions_total,
                               fastapi_request_size_bytes,
                               fastapi_requests_duration_seconds,
                               fastapi_requests_in_progress,
                               fastapi_requests_total,
                               fastapi_response_size_bytes,
                               fastapi_responses_total)

    method = request.method
    path = request.url.path
    app_name = settings.app_name

    # Track in-progress requests
    fastapi_requests_in_progress.labels(
        method=method, path=path, app_name=app_name
    ).inc()

    # Track request size
    content_length = request.headers.get("content-length")
    if content_length:
        fastapi_request_size_bytes.labels(
            method=method, path=path, app_name=app_name
        ).observe(int(content_length))

    start_time = time.time()
    status_code = 500  # Default to 500 for errors
    response = None

    try:
        response = await call_next(request)
        status_code = response.status_code

        # Track response
        fastapi_responses_total.labels(
            method=method,
            path=path,
            status_code=f"{status_code // 100}xx",
            app_name=app_name,
        ).inc()

        # Track response size (get from headers if available)
        content_length = response.headers.get("content-length")
        if content_length:
            fastapi_response_size_bytes.labels(
                method=method, path=path, app_name=app_name
            ).observe(int(content_length))

        return response

    except Exception as e:
        # Track exception
        fastapi_exceptions_total.labels(
            method=method, path=path, exception_type=type(e).__name__, app_name=app_name
        ).inc()
        status_code = 500
        raise

    finally:
        # Track duration
        duration = time.time() - start_time
        fastapi_requests_duration_seconds.labels(
            method=method, path=path, app_name=app_name
        ).observe(duration)

        # Track total requests
        fastapi_requests_total.labels(
            method=method,
            path=path,
            status_code=f"{status_code // 100}xx",
            app_name=app_name,
        ).inc()

        # Decrement in-progress
        fastapi_requests_in_progress.labels(
            method=method, path=path, app_name=app_name
        ).dec()


# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


@app.on_event("startup")
def on_startup():
    try:
        init_db()
        with SessionLocal() as db:
            ensure_default_admin(db)

        try:
            create_collection()
        except Exception as e:
            logger.warning(f"⚠️  Qdrant not available, skipping collection creation: {e}")

        if settings.qwen3_models_enabled:
            logger.info("Using GPU service for models (qwen3_models)")
        else:
            logger.info("Using local CPU models (embedded in backend)")
            try:
                from .src.core.model_loader import get_model_registry

                model_registry = get_model_registry()
                model_registry.load_models()
                logger.success("✅ Local models loaded successfully")
            except Exception as e:
                logger.warning(f"⚠️  Failed to load local models: {e}")

        logger.info("⏭️  STT/TTS eager startup init disabled (lazy init on first request)")

        try:
            from .src.services.elastic_search import warmup_elasticsearch_client

            warmup_elasticsearch_client()
            logger.info("✅ Elasticsearch singleton warmed up")
        except Exception as e:
            logger.warning(f"⚠️  Failed to warm up Elasticsearch client: {e}")

        try:
            from .src.services.brain import get_response

            logger.info("🔥 Warming up generation model (vLLM → Ollama fallback)...")

            warmup_messages = [
                {
                    "role": "system",
                    # /no_think disables the thinking phase on Qwen3 models so
                    # the warmup response is fast and doesn't need many tokens.
                    "content": "Bạn là Minqes - trợ lý y tế thông minh. /no_think",
                },
                {"role": "user", "content": "Chào Minqes!"},
            ]

            # 512 tokens gives the model room to finish even if thinking is on.
            # get_response() tries vLLM first, falls back to Ollama automatically.
            response = get_response(
                messages=warmup_messages,
                temperature=0.7,
                max_tokens=512,
            )

            if response:
                logger.success(f"✅ Generation model warmed up successfully: {response[:60]!r}")
            elif response is not None:
                # Empty string — model responded but content was blank
                # (can happen with thinking models when tokens are exhausted).
                logger.warning(
                    "⚠️  Generation model warmup: model replied with empty content. "
                    "Consider increasing max_tokens or using a non-thinking model variant."
                )
            else:
                logger.warning(
                    "⚠️  Generation model warmup: both vLLM and Ollama unavailable. "
                    "Check VLLM_URL / OLLAMA_URL env-vars."
                )
        except Exception as e:
            logger.warning(f"⚠️  Failed to warm up generation model: {e}")

        logger.info("Application startup complete.")
    except Exception as e:
        logger.error(f"Error during startup: {e}")
        raise


@app.on_event("shutdown")
async def on_shutdown():
    """Graceful shutdown for shared async clients."""
    try:
        from .src.services.stt_service import close_stt_service
        from .src.services.tts_service import close_tts_service

        await close_stt_service()
        await close_tts_service()
        logger.info("✅ STT/TTS clients closed")
    except Exception as e:
        logger.warning(f"⚠️  Failed to close STT/TTS clients: {e}")


# Include routers
app.include_router(health.router)
app.include_router(rag.router)
app.include_router(models.router)
app.include_router(audio.router)
app.include_router(audio.stt_router)
app.include_router(audio.tts_router)
app.include_router(documents.router)

# Auth & Chat routers
app.include_router(auth_router)
app.include_router(chat_router)
app.include_router(admin_router)


@app.get("/")
def read_root():
    return {
        "message": f"Welcome to the {settings.app_name} API!",
        "version": settings.app_version,
        "docs": "/docs",
        "routers": [
            "/v1/auth",
            "/v1/chat",
            "/v1/admin",
            "/v1/health",
            "/v1/rag",
            "/v1/models",
            "/v1/indexing",
            "/v1/documents",
            "/v1/audio",
            "/v1/stt",
            "/v1/tts",
        ],
    }


# ────────────────────────────────────────────────────────────
#  /chat — simple stateless endpoint with vLLM → Ollama fallback
# ────────────────────────────────────────────────────────────

class _ChatMessage(BaseModel):
    """Single message in the conversation (OpenAI-style)."""

    role: str = Field(
        ...,
        description="Message role: 'system', 'user', or 'assistant'",
        examples=["user"],
    )
    content: str = Field(..., min_length=1, description="Message content")


class ChatRequest(BaseModel):
    """Request body for the /chat endpoint."""

    messages: List[_ChatMessage] = Field(
        ...,
        min_length=1,
        description="Conversation history in OpenAI message format",
    )
    temperature: float = Field(default=0.7, ge=0.0, le=2.0)
    max_tokens: int = Field(default=512, ge=1, le=2048)


class ChatResponse(BaseModel):
    """Response body returned by the /chat endpoint."""

    content: str = Field(..., description="Assistant reply text")
    model: str = Field(..., description="Model name used")
    provider: str = Field(..., description="Inference provider used: 'vllm' or 'ollama'")


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Stateless chat completion with configured single provider",
    tags=["chat"],
)
async def chat(body: ChatRequest):
    """
    Accept a list of messages and return the assistant reply.

    Uses single active provider configured by ``LLM_PROVIDER``.

    **Environment variables**
    | Variable     | Default                    | Description              |
    |--------------|----------------------------|--------------------------|
    | LLM_PROVIDER | ollama                     | Active provider          |
    | VLLM_URL     | http://localhost:7861      | vLLM base URL            |
    | OLLAMA_URL   | http://localhost:11434     | Ollama base URL          |
    | MODEL_NAME   | value from models.yaml     | Override model name      |
    """
    import os

    from .src.services.brain import get_response
    messages = [m.model_dump() for m in body.messages]

    try:
        content = get_response(
            messages=messages,
            temperature=body.temperature,
            max_tokens=body.max_tokens,
        )
    except Exception as exc:
        logger.exception(f"[/chat] Unexpected error: {exc}")
        raise HTTPException(status_code=500, detail="Generation service error") from exc

    if content is None:
        raise HTTPException(
            status_code=503,
            detail="Generation provider unavailable. Check LLM_PROVIDER and provider URL settings.",
        )

    # Determine provider for response metadata (best-effort).
    provider = (os.getenv("LLM_PROVIDER", "ollama") or "ollama").strip().lower()
    model_name = os.getenv("MODEL_NAME", "")

    if not model_name:
        from .src.core.model_config import get_generation_model

        try:
            model_name = get_generation_model()
        except Exception:
            model_name = "unknown"

    return ChatResponse(content=content, model=model_name, provider=provider)