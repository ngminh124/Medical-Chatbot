"""
Cache Module
Provides Redis-based caching for embeddings and search results.
"""
from typing import List, Optional
import json

from loguru import logger

from ..configs.setup import get_backend_settings

settings = get_backend_settings()

# Redis client (lazy initialization)
_redis_client = None


def _get_redis_client():
    """Get or create Redis client (lazy initialization)."""
    global _redis_client
    if _redis_client is None:
        try:
            import redis
            _redis_client = redis.Redis(
                host=settings.redis_host,
                port=settings.redis_port,
                db=settings.redis_db,
                decode_responses=True,
            )
            # Test connection
            _redis_client.ping()
            logger.info(f"Redis connected: {settings.redis_host}:{settings.redis_port}")
        except ImportError:
            logger.warning("Redis package not installed. Caching disabled.")
            return None
        except Exception as e:
            logger.warning(f"Redis connection failed: {e}. Caching disabled.")
            return None
    return _redis_client


def get_redis_client():
    """Public accessor for shared Redis client instance."""
    return _get_redis_client()


def get_query_embedding(cache_key: str) -> Optional[List[float]]:
    """
    Get cached query embedding from Redis.
    
    Args:
        cache_key: Unique key for the query embedding
        
    Returns:
        List of floats if found, None otherwise
    """
    client = _get_redis_client()
    if client is None:
        return None
    
    try:
        cached = client.get(f"embedding:{cache_key}") or client.get(f"emb:{cache_key}")
        if cached:
            logger.debug(f"Cache hit for query embedding: {cache_key[:50]}...")
            return json.loads(cached)
        return None
    except Exception as e:
        logger.warning(f"Cache read error: {e}")
        return None


def cache_query_embedding(
    cache_key: str, 
    embedding: List[float], 
    ttl_seconds: int = 3600
) -> bool:
    """
    Cache query embedding to Redis.
    
    Args:
        cache_key: Unique key for the query embedding
        embedding: The embedding vector to cache
        ttl_seconds: Time-to-live in seconds (default 1 hour)
        
    Returns:
        True if cached successfully, False otherwise
    """
    client = _get_redis_client()
    if client is None:
        return False
    
    try:
        client.setex(
            f"embedding:{cache_key}",
            ttl_seconds,
            json.dumps(embedding),
        )
        logger.debug(f"Cached query embedding: {cache_key[:50]}...")
        return True
    except Exception as e:
        logger.warning(f"Cache write error: {e}")
        return False


def get_search_results(cache_key: str) -> Optional[List[dict]]:
    """
    Get cached search results from Redis.
    
    Args:
        cache_key: Unique key for the search results
        
    Returns:
        List of result dicts if found, None otherwise
    """
    client = _get_redis_client()
    if client is None:
        return None
    
    try:
        cached = client.get(f"retrieval:{cache_key}") or client.get(f"search:{cache_key}")
        if cached:
            logger.debug(f"Cache hit for search results: {cache_key[:50]}...")
            return json.loads(cached)
        return None
    except Exception as e:
        logger.warning(f"Cache read error: {e}")
        return None


def cache_search_results(
    cache_key: str, 
    results: List[dict], 
    ttl_seconds: int = 300
) -> bool:
    """
    Cache search results to Redis.
    
    Args:
        cache_key: Unique key for the search results
        results: The search results to cache
        ttl_seconds: Time-to-live in seconds (default 5 minutes)
        
    Returns:
        True if cached successfully, False otherwise
    """
    client = _get_redis_client()
    if client is None:
        return False
    
    try:
        client.setex(
            f"retrieval:{cache_key}",
            ttl_seconds,
            json.dumps(results),
        )
        logger.debug(f"Cached search results: {cache_key[:50]}...")
        return True
    except Exception as e:
        logger.warning(f"Cache write error: {e}")
        return False


def clear_cache(pattern: str = "*") -> int:
    """
    Clear cache entries matching pattern.
    
    Args:
        pattern: Redis key pattern to match (default: all keys)
        
    Returns:
        Number of keys deleted
    """
    client = _get_redis_client()
    if client is None:
        return 0
    
    try:
        keys = list(client.scan_iter(match=pattern))
        if keys:
            deleted = client.delete(*keys)
            logger.info(f"Cleared {deleted} cache entries")
            return deleted
        return 0
    except Exception as e:
        logger.warning(f"Cache clear error: {e}")
        return 0


def get_final_answer(cache_key: str) -> Optional[dict]:
    """Get cached final answer payload from Redis."""
    client = _get_redis_client()
    if client is None:
        return None

    try:
        cached = client.get(f"answer:{cache_key}")
        if cached:
            return json.loads(cached)
        return None
    except Exception as e:
        logger.warning(f"Cache read error: {e}")
        return None


def cache_final_answer(cache_key: str, payload: dict, ttl_seconds: int = 3600) -> bool:
    """Cache final answer payload to Redis."""
    client = _get_redis_client()
    if client is None:
        return False

    try:
        client.setex(
            f"answer:{cache_key}",
            ttl_seconds,
            json.dumps(payload, ensure_ascii=False),
        )
        return True
    except Exception as e:
        logger.warning(f"Cache write error: {e}")
        return False
