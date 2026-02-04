"""
Model Configuration Module
Provides centralized model configuration for the RAG system.
"""
from typing import Optional

from loguru import logger

from ..configs.setup import get_backend_settings

settings = get_backend_settings()


def get_embedding_model() -> str:
    """
    Get the configured embedding model name.
    
    Returns:
        str: Hugging Face model name (e.g., 'Qwen/Qwen3-Embedding-0.6B')
    """
    model_name = settings.embedding_model_name
    logger.debug(f"Using embedding model: {model_name}")
    return model_name


def get_embedding_dimension() -> int:
    """
    Get the configured embedding vector dimension.
    
    Returns:
        int: Vector dimension (e.g., 1024 for Qwen3-Embedding-0.6B)
    """
    return settings.vector_dimension


def get_model_config() -> dict:
    """
    Get full model configuration as a dictionary.
    
    Returns:
        dict: Model configuration including name, dimension, device
    """
    return {
        "embedding_model": settings.embedding_model_name,
        "vector_dimension": settings.vector_dimension,
        "device": settings.device,
        "qwen3_models_enabled": settings.qwen3_models_enabled,
        "qwen3_models_url": settings.qwen3_models_url,
    }
