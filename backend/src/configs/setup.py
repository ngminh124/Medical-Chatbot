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

    # Redis Cache Configuration
    redis_host: str = Field(default="localhost", description="Redis server host")
    redis_port: int = Field(default=6379, description="Redis server port")
    redis_db: int = Field(default=0, description="Redis database number")

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
