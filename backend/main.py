import time
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
from .src.routers.chat import router as chat_router

settings = get_backend_settings()

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
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
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

        # Initialize STT service
        try:
            from .src.core.model_config import load_model_config
            from .src.services.stt_service import initialize_stt_service

            config = load_model_config()
            stt_config = config.get("models", {}).get("stt", {})

            # STT now routes to GPU service, just initialize the proxy
            initialize_stt_service(
                model_name=stt_config.get("active", "turbo"),
                device=stt_config.get("device", "cuda"),
                compute_type=stt_config.get("compute_type", "float16"),
            )
            logger.info("✅ STT service initialized (routes to GPU service)")
        except Exception as e:
            logger.warning(
                f"⚠️  Failed to initialize STT service: {e}"
            )  # Initialize TTS service
        try:
            import os

            from .src.services.tts_service import initialize_tts_service

            api_key = os.getenv("ELEVENLABS_API_KEY")
            voice_id = os.getenv("ELEVENLABS_VOICE_ID")

            if api_key:
                initialize_tts_service(api_key=api_key, voice_id=voice_id)
                logger.success("✅ TTS service initialized successfully")
            else:
                logger.warning("⚠️  ElevenLabs API key not configured, TTS disabled")
        except Exception as e:
            logger.warning(f"⚠️  Failed to initialize TTS service: {e}")

        try:
            from .src.services.brain import get_response

            logger.info("🔥 Warming up generation model (vLLM → Ollama fallback)...")

            warmup_messages = [
                {
                    "role": "system",
                    # /no_think disables the thinking phase on Qwen3 models so
                    # the warmup response is fast and doesn't need many tokens.
                    "content": "Bạn là Min - trợ lý y tế thông minh. /no_think",
                },
                {"role": "user", "content": "Chào Min!"},
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


# Include routers
app.include_router(health.router)
app.include_router(rag.router)
app.include_router(models.router)
app.include_router(audio.router)
app.include_router(documents.router)

# Auth & Chat routers
app.include_router(auth_router)
app.include_router(chat_router)


@app.get("/")
def read_root():
    return {
        "message": f"Welcome to the {settings.app_name} API!",
        "version": settings.app_version,
        "docs": "/docs",
        "routers": [
            "/v1/auth",
            "/v1/chat",
            "/v1/health",
            "/v1/rag",
            "/v1/models",
            "/v1/indexing",
            "/v1/documents",
            "/v1/audio",
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
    max_tokens: int = Field(default=2048, ge=1, le=8192)


class ChatResponse(BaseModel):
    """Response body returned by the /chat endpoint."""

    content: str = Field(..., description="Assistant reply text")
    model: str = Field(..., description="Model name used")
    provider: str = Field(..., description="Inference provider used: 'vllm' or 'ollama'")


@app.post(
    "/chat",
    response_model=ChatResponse,
    summary="Stateless chat completion with vLLM → Ollama fallback",
    tags=["chat"],
)
async def chat(body: ChatRequest):
    """
    Accept a list of messages and return the assistant reply.

    **Fallback logic**
    - Step 1: calls vLLM (``VLLM_URL``, default ``http://localhost:7861``).
    - Step 2: if vLLM is unreachable or times out, automatically switches to
      Ollama (``OLLAMA_URL``, default ``http://localhost:11434``) and logs the
      provider switch.

    **Environment variables**

    | Variable     | Default                    | Description              |
    |--------------|----------------------------|--------------------------|
    | VLLM_URL     | http://localhost:7861      | vLLM base URL            |
    | OLLAMA_URL   | http://localhost:11434     | Ollama base URL          |
    | MODEL_NAME   | value from models.yaml     | Override model for both  |
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
            detail=(
                "Both vLLM and Ollama are unavailable. "
                "Please check VLLM_URL and OLLAMA_URL environment variables."
            ),
        )

    # Determine which provider actually answered (logged inside get_response;
    # we do a lightweight probe here only to populate the response field).
    vllm_url = os.getenv("VLLM_URL", "http://localhost:7861")
    ollama_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
    model_name = os.getenv("MODEL_NAME", "")

    from .src.core.model_config import get_generation_model

    if not model_name:
        try:
            model_name = get_generation_model()
        except Exception:
            model_name = "unknown"

    # Infer provider by checking vLLM health (best-effort, no extra latency on
    # the hot path — the actual provider was already logged by get_response).
    try:
        import httpx as _httpx

        probe = _httpx.get(f"{vllm_url}/health", timeout=2.0)
        provider = "vllm" if probe.status_code == 200 else "ollama"
    except Exception:
        provider = "ollama"

    return ChatResponse(content=content, model=model_name, provider=provider)