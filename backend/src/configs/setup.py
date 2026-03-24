from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class BackendSettings(BaseSettings):
    """
    Backend configuration settings using pydantic-settings.
    Loads configuration from environment variables and .env file.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application Metadata
    app_name: str = Field(default="Medical RAG Chatbot", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")

    # OpenTelemetry / Tempo
    tempo_enabled: bool = Field(default=False, description="Enable OpenTelemetry tracing to Tempo")
    tempo_endpoint: str = Field(default="http://localhost:4317", description="Tempo OTLP endpoint")

    # Qdrant Configuration
    qdrant_host: str = Field(default="localhost", description="Qdrant server host")
    qdrant_port: int = Field(default=6333, description="Qdrant server port")
    default_collection_name: str = Field(
        default="medical_data", description="Default Qdrant collection name"
    )

    # Embedding Model Configuration - Qwen3-Embedding-0.6B (1024-dim)
    embedding_model_name: str = Field(
        default="Qwen/Qwen3-Embedding-0.6B",
        description="Name of the embedding model to use",
    )
    vector_dimension: int = Field(
        default=1024, description="Dimension of embedding vectors"
    )

    # Qwen3 GPU Service Configuration
    qwen3_models_enabled: bool = Field(
        default=True, description="Enable Qwen3 models GPU service"
    )
    qwen3_models_url: str = Field(
        default="http://localhost:7860", description="URL of Qwen3 GPU service"
    )
    backend_api_url: str = Field(
        default="http://localhost:8000", description="Backend API URL"
    )

    # Search Configuration
    top_k: int = Field(default=5, description="Number of top results to return")

    # Batch Processing Configuration
    batch_size: int = Field(
        default=500, description="Batch size for processing and uploading data"
    )
    max_workers: int = Field(
        default=4, description="Maximum number of worker threads"
    )

    # Data Paths
    data_dir: str = Field(default="data/processed", description="Data directory path")
    chunks_output_dir: str = Field(
        default="data/chunks", description="Chunks output directory"
    )

    # Device Configuration
    device: str = Field(
        default="cuda", description="Device to use for model inference (cuda/cpu/mps)"
    )

    # Elasticsearch Configuration
    elasticsearch_host: str = Field(
        default="localhost", description="Elasticsearch server host"
    )
    elasticsearch_port: int = Field(
        default=9200, description="Elasticsearch server port"
    )
    elasticsearch_index: str = Field(
        default="medical_documents",
        description="Default Elasticsearch index name",
    )

    # Redis Cache Configuration
    redis_host: str = Field(default="localhost", description="Redis server host")
    redis_port: int = Field(default=6379, description="Redis server port")
    redis_db: int = Field(default=0, description="Redis database number")

    # Internal HTTP client resilience (embedding/rerank/guard)
    service_http_timeout: float = Field(
        default=20.0, description="HTTP timeout (seconds) for internal model services"
    )
    service_http_retries: int = Field(
        default=2, description="Retry attempts for transient internal HTTP failures"
    )
    service_http_backoff_seconds: float = Field(
        default=0.4, description="Exponential backoff base (seconds) for retries"
    )

    # JWT Configuration
    jwt_secret_key: str = Field(
        default="change-me-in-production-please",
        description="Secret key for JWT token signing",
    )

    # API Keys
    openai_api_key: str = Field(
        default="", description="OpenAI API key for Tavily agent"
    )
    tavily_api_key: str = Field(
        default="", description="Tavily API key for web search"
    )

    # TTS / STT configuration
    elevenlabs_api_key: str = Field(
        default="", description="ElevenLabs API key"
    )
    elevenlabs_voice_id: str = Field(
        default="A5w1fw5x0uXded1LDvZp", description="Default voice ID"
    )
    stt_gpu_service_url: str = Field(
        default="http://extra_models:8002", description="URL của GPU service cho STT"
    )
    stt_cache_ttl: int = Field(
        default=3600, description="Thời gian cache STT (giây)"
    )
    tts_cache_ttl: int = Field(
        default=86400, description="Thời gian cache TTS (giây)"
    )

    # Prompt Templates
    rewrite_prompt: str = Field(
        default=(
            "Dựa vào lịch sử hội thoại sau:\n{history_messages}\n\n"
            "Hãy viết lại câu hỏi sau sao cho rõ ràng, đầy đủ ngữ cảnh và dễ hiểu hơn, "
            "giữ nguyên ý nghĩa gốc. Chỉ trả về câu hỏi đã được viết lại, không giải thích.\n\n"
            "Câu hỏi gốc: {message}\n\nCâu hỏi đã viết lại:"
        ),
        description="Prompt template for query rewriting with conversation context",
    )
    intent_detection_prompt: str = Field(
        default=(
            "Dựa vào lịch sử hội thoại:\n{history}\n\n"
            "Và câu hỏi hiện tại: {message}\n\n"
            "Hãy phân loại câu hỏi này thuộc loại nào. Chỉ trả về MỘT trong hai từ sau:\n"
            "- 'medical' nếu câu hỏi liên quan đến y tế, sức khỏe, bệnh, thuốc, triệu chứng, điều trị\n"
            "- 'general' nếu câu hỏi là hội thoại thông thường, chào hỏi, hoặc không liên quan y tế\n\n"
            "Phân loại:"
        ),
        description="Prompt template for intent detection (medical vs general)",
    )

    @property
    def data_path(self) -> Path:
        """Get absolute path to data directory."""
        return Path(self.data_dir)

    @property
    def chunks_output_path(self) -> Path:
        """Get absolute path to chunks output directory."""
        return Path(self.chunks_output_dir)


@lru_cache()
def get_backend_settings() -> BackendSettings:
    """
    Get cached backend settings instance.
    Uses lru_cache to ensure settings are loaded only once.
    """
    return BackendSettings()


class DatabaseSettings(BaseSettings):
    """
    PostgreSQL database configuration.
    Reads from the same .env file (or environment variables).
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    postgres_user: str = Field(default="postgresadmin", description="PostgreSQL user")
    postgres_password: str = Field(default="postgresadmin", description="PostgreSQL password")
    postgres_host: str = Field(default="localhost", description="PostgreSQL host")
    postgres_port: int = Field(default=5432, description="PostgreSQL port")
    postgres_db: str = Field(default="medical_rag_db", description="PostgreSQL database name")

    @property
    def database_url(self) -> str:
        """Build SQLAlchemy connection URL."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache()
def get_database_settings() -> DatabaseSettings:
    """
    Get cached database settings instance.
    Uses lru_cache to ensure settings are loaded only once.
    """
    return DatabaseSettings()
