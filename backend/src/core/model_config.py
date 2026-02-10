from pathlib import Path
from typing import Optional

import yaml
from loguru import logger

CONFIG_PATH = Path(__file__).parent.parent / "configs" / "models.yaml"

# Cached config (loaded once at startup)
_config_cache: Optional[dict] = None


def load_model_config() -> dict:
    """
    Load model configuration from YAML file.
    Returns cached config if already loaded.
    """
    global _config_cache

    if _config_cache is not None:
        return _config_cache

    if not CONFIG_PATH.exists():
        logger.error(f"Model config file not found: {CONFIG_PATH}")
        raise FileNotFoundError(f"Model config not found: {CONFIG_PATH}")

    with open(CONFIG_PATH, "r", encoding="utf-8") as f:
        _config_cache = yaml.safe_load(f)

    # Validate required keys to fail fast with clear errors
    def _validate_config(cfg: dict) -> None:
        required_paths = [
            ("models", "generation", "active"),
            ("models", "embedding", "active"),
            ("models", "reranking", "active"),
            ("models", "guardrails", "active"),
            ("models", "guardrails", "threshold"),
            ("serving", "vllm_url"),
        ]
        missing = []
        for path in required_paths:
            node = cfg
            for key in path:
                if not isinstance(node, dict) or key not in node:
                    missing.append(".".join(path))
                    break
                node = node[key]

        if missing:
            logger.error(f"Model config missing required keys: {missing}")
            raise KeyError(f"Model config missing required keys: {missing}")

    _validate_config(_config_cache)

    logger.info(f"Loaded model config from {CONFIG_PATH}")
    return _config_cache


def get_generation_model() -> str:
    """Get active generation model HuggingFace repo ID."""
    config = load_model_config()
    return config["models"]["generation"]["active"]


def get_embedding_model() -> str:
    """Get active embedding model HuggingFace repo ID."""
    config = load_model_config()
    return config["models"]["embedding"]["active"]


def get_reranking_model() -> str:
    """Get active reranking model HuggingFace repo ID."""
    config = load_model_config()
    return config["models"]["reranking"]["active"]


def get_guardrails_model() -> str:
    """Get active guardrails model HuggingFace repo ID."""
    config = load_model_config()
    return config["models"]["guardrails"]["active"]


def get_guardrails_threshold() -> float:
    """Get safety threshold for guardrails."""
    config = load_model_config()
    return config["models"]["guardrails"]["threshold"]


def get_vllm_url() -> str:
    """Get vLLM server URL."""
    config = load_model_config()
    return config["serving"]["vllm_url"]


def get_vllm_api_key() -> str:
    """Get vLLM API key."""
    config = load_model_config()
    return config["serving"].get("vllm_api_key", "")


def reload_config():
    """Force reload config from file (useful for hot-reloading)."""
    global _config_cache
    _config_cache = None
    logger.info("Model config cache cleared, will reload on next access")