import json
import hashlib
import logging
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Iterator, List, Optional

import httpx
import openai
from loguru import logger
from openai import OpenAI

from ..configs.setup import get_backend_settings
from ..core.model_config import (get_generation_model, get_vllm_api_key,
                                 get_vllm_url)

settings = get_backend_settings()

# Standard logging for provider-switch notifications (as requested)
log = logging.getLogger(__name__)

# ─── Environment-variable constants (overridable at runtime) ──────────────────
# These take priority over the YAML model config so operators can swap
# endpoints without restarting: export VLLM_URL=http://...
_DEFAULT_VLLM_URL = "http://localhost:7861"
_DEFAULT_OLLAMA_URL = "http://localhost:11434"


def _safe_int_env(name: str, default: int) -> int:
    try:
        raw = os.getenv(name, str(default))
        if raw is None:
            return default
        raw = str(raw).strip()
        if raw == "":
            return default
        return int(raw)
    except Exception:
        return default


def _safe_float_env(name: str, default: float) -> float:
    try:
        raw = os.getenv(name, str(default))
        if raw is None:
            return default
        raw = str(raw).strip()
        if raw == "":
            return default
        return float(raw)
    except Exception:
        return default

# ─── Local performance defaults ───────────────────────────────────────────────
VECTOR_K = int(os.getenv("VECTOR_K", "8"))
BM25_K = int(os.getenv("BM25_K", "8"))
FINAL_K = int(os.getenv("FINAL_K", "4"))
TOP_K = int(os.getenv("TOP_K", "5"))
USE_RERANK = os.getenv("USE_RERANK", "false").strip().lower() in {"1", "true", "yes", "on"}

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").strip().lower()  # ollama | vllm
LLM_MAX_TOKENS_CAP = max(512, min(_safe_int_env("LLM_MAX_TOKENS", 1024), 2048))
LLM_TIMEOUT_SECONDS = max(5.0, min(_safe_float_env("LLM_TIMEOUT_SECONDS", 10.0), 20.0))
LLM_STREAM_READ_TIMEOUT_SECONDS = max(30.0, _safe_float_env("LLM_STREAM_READ_TIMEOUT_SECONDS", 300.0))
FAST_LOCAL_MODE = os.getenv("FAST_LOCAL_MODE", "true").strip().lower() in {"1", "true", "yes", "on"}
LLM_TIMEOUT_FALLBACK = (
    "Xin lỗi, hệ thống đang bận hoặc phản hồi chậm. "
    "Vui lòng thử lại sau vài giây."
)

WEB_SEARCH_MAX_RESULTS = int(os.getenv("WEB_SEARCH_MAX_RESULTS", "2"))
WEB_SEARCH_SNIPPET_MAX_CHARS = int(os.getenv("WEB_SEARCH_SNIPPET_MAX_CHARS", "220"))

CACHE_TTL_SECONDS = int(os.getenv("CACHE_TTL_SECONDS", "3600"))
ANSWER_CACHE_TTL_SECONDS = int(os.getenv("ANSWER_CACHE_TTL_SECONDS", "600"))
RETRIEVAL_CACHE_TTL_SECONDS = int(os.getenv("RETRIEVAL_CACHE_TTL_SECONDS", "600"))
_mem_cache: dict[str, tuple[float, object]] = {}
_mem_cache_lock = threading.Lock()
_ollama_model_cache: Optional[str] = None
_ollama_model_lock = threading.Lock()
_llm_provider_resolved: Optional[str] = None
_provider_state_lock = threading.Lock()
_provider_unavailable_until: dict[str, float] = {"vllm": 0.0, "ollama": 0.0}


def get_active_llm_provider() -> str:
    global _llm_provider_resolved
    if _llm_provider_resolved is not None:
        return _llm_provider_resolved
    with _provider_state_lock:
        if _llm_provider_resolved is None:
            _llm_provider_resolved = "vllm" if LLM_PROVIDER == "vllm" else "ollama"
        return _llm_provider_resolved


def _provider_is_available(provider: str) -> bool:
    return time.time() >= _provider_unavailable_until.get(provider, 0.0)


def _mark_provider_unavailable(provider: str, cooldown_seconds: int = 60):
    _provider_unavailable_until[provider] = time.time() + cooldown_seconds


def _hash_key(prefix: str, *parts: str) -> str:
    payload = "::".join(parts)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
    return f"{prefix}:{digest}"


def _mem_get(key: str):
    now = time.time()
    with _mem_cache_lock:
        item = _mem_cache.get(key)
        if not item:
            return None
        expires_at, value = item
        if expires_at < now:
            _mem_cache.pop(key, None)
            return None
        return value


def _mem_set(key: str, value: object, ttl: int = CACHE_TTL_SECONDS):
    with _mem_cache_lock:
        _mem_cache[key] = (time.time() + ttl, value)


def _cache_get_json(key: str):
    cached = _mem_get(key)
    if cached is not None:
        return cached

    try:
        from ..core.cache import get_redis_client

        redis_client = get_redis_client()
        if redis_client is not None:
            raw = redis_client.get(key)
            if raw:
                value = json.loads(raw)
                _mem_set(key, value)
                return value
    except Exception:
        pass
    return None


def _cache_set_json(key: str, value: object, ttl: int = CACHE_TTL_SECONDS):
    _mem_set(key, value, ttl=ttl)
    try:
        from ..core.cache import get_redis_client

        redis_client = get_redis_client()
        if redis_client is not None:
            redis_client.setex(key, ttl, json.dumps(value, ensure_ascii=False))
    except Exception:
        pass


def build_final_response_cache_key(
    question: str,
    history: List[Dict[str, str]],
    web_search_enabled: bool,
) -> str:
    recent = [m for m in history if m.get("role") in ("user", "assistant")][-3:]
    context = "\n".join(f"{m.get('role','')}: {m.get('content','')[:500]}" for m in recent)
    raw = "::".join(
        [
            question.strip().lower(),
            context.strip().lower(),
            "web" if web_search_enabled else "rag",
        ]
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def get_cached_final_response(
    question: str,
    history: List[Dict[str, str]],
    web_search_enabled: bool,
) -> Optional[Dict[str, object]]:
    key = build_final_response_cache_key(question, history, web_search_enabled)
    try:
        from ..core.cache import get_final_answer

        cached = get_final_answer(key)
    except Exception:
        cached = _cache_get_json(key)
    if isinstance(cached, dict):
        return cached
    return None


def cache_final_response(
    question: str,
    history: List[Dict[str, str]],
    web_search_enabled: bool,
    payload: Dict[str, object],
    ttl_seconds: int = ANSWER_CACHE_TTL_SECONDS,
) -> None:
    key = build_final_response_cache_key(question, history, web_search_enabled)
    try:
        from ..core.cache import cache_final_answer

        ok = cache_final_answer(key, payload, ttl_seconds=ttl_seconds)
        if not ok:
            _cache_set_json(key, payload, ttl=ttl_seconds)
    except Exception:
        _cache_set_json(key, payload, ttl=ttl_seconds)


def get_vllm_client():
    """Get remote vLLM client from config."""
    try:
        vllm_url = get_vllm_url()
        vllm_api_key = get_vllm_api_key()

        client = OpenAI(
            api_key=vllm_api_key,
            base_url=f"{vllm_url}/v1",
            timeout=LLM_TIMEOUT_SECONDS,
        )
        return client
    except Exception as e:
        logger.error(f"[GEN] vLLM client init failed: {e}")
        return None


def qwen3_chat_complete(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Optional[str]:
    """Generate chat completion using remote vLLM server with Qwen3-4B-Instruct-2507."""
    temperature = temperature if temperature is not None else 0.7
    max_tokens = _normalize_max_tokens(max_tokens)

    if model is None:
        model = get_generation_model()

    try:
        client = get_vllm_client()
        if client:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=0.8,
            )
            content: str = response.choices[0].message.content or ""
            return content
    except Exception as e:
        error_msg = str(e)
        if "<!DOCTYPE html>" in error_msg or "<html" in error_msg:
            error_msg = "vLLM service is loading"
        logger.warning(f"[GEN] vLLM failed: {error_msg}")

    return None


def qwen3_chat_stream(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
) -> Iterator[str]:
    """Stream chat completion tokens from remote vLLM (OpenAI-compatible)."""
    temperature = temperature if temperature is not None else 0.7
    max_tokens = _normalize_max_tokens(max_tokens)

    if model is None:
        model = get_generation_model()

    vllm_url = os.getenv("VLLM_URL", _DEFAULT_VLLM_URL)
    stream_client = OpenAI(
        api_key=get_vllm_api_key() or "EMPTY",
        base_url=f"{vllm_url}/v1",
        timeout=httpx.Timeout(
            connect=min(10.0, LLM_TIMEOUT_SECONDS),
            read=LLM_STREAM_READ_TIMEOUT_SECONDS,
            write=min(20.0, LLM_TIMEOUT_SECONDS),
            pool=min(10.0, LLM_TIMEOUT_SECONDS),
        ),
    )

    stream = stream_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.8,
        stream=True,
    )

    for chunk in stream:
        token = ""
        try:
            token = chunk.choices[0].delta.content or ""
        except Exception:
            token = ""
        if token:
            yield token


def check_vllm_health() -> bool:
    """Check health of remote vLLM server."""
    try:
        vllm_url = get_vllm_url()
        # vllm_url is base URL without /v1, health endpoint is at /health or /version
        response = httpx.get(f"{vllm_url}/health", timeout=5.0)
        if response.status_code == 200:
            return True
        # Fallback: try /v1/models endpoint
        response = httpx.get(f"{vllm_url}/v1/models", timeout=5.0)
        return response.status_code == 200
    except Exception:
        return False


def _truncate_doc_text(text: str, max_chars: int = 420) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[:max_chars].rstrip() + "…"


def _rrf_fuse(results_lists: List[List[Dict]], k: int = 60) -> List[Dict]:
    fused_scores: Dict[str, float] = {}
    result_data: Dict[str, Dict] = {}

    for results in results_lists:
        for rank, result in enumerate(results, start=1):
            doc_id = str(result.get("chunk_id") or result.get("id") or rank)
            fused_scores[doc_id] = fused_scores.get(doc_id, 0.0) + (1.0 / (k + rank))
            if doc_id not in result_data:
                result_data[doc_id] = result

    sorted_ids = sorted(fused_scores.keys(), key=lambda x: fused_scores[x], reverse=True)
    merged: List[Dict] = []
    for doc_id in sorted_ids:
        item = result_data[doc_id].copy()
        item["rrf_score"] = fused_scores[doc_id]
        merged.append(item)
    return merged


def hybrid_retrieve(query: str, retrieval_mode: str = "rag") -> List[Dict]:
    """Fast hybrid retrieval with caching and local-friendly defaults.

    vector_k=8, bm25_k=8, final_k=4 by default.
    """
    search_raw = "::".join(
        [retrieval_mode, query.strip().lower(), str(VECTOR_K), str(BM25_K), str(FINAL_K)]
    )
    search_key = hashlib.sha256(search_raw.encode("utf-8")).hexdigest()
    try:
        from ..core.cache import get_search_results

        cached = get_search_results(search_key)
    except Exception:
        cached = _cache_get_json(search_key)
    if isinstance(cached, list):
        return cached

    def _vector_search() -> List[Dict]:
        try:
            from ..services.embedding import get_embedding_service
            from ..core.vectorize import search_vectors_for_hybrid

            emb_service = get_embedding_service()
            qvec = emb_service.embed_query(query)
            if not qvec:
                return []
            return search_vectors_for_hybrid(
                query_vector=qvec,
                top_k=VECTOR_K,
                collection_name=settings.default_collection_name,
            )
        except Exception:
            return []

    def _bm25_search() -> List[Dict]:
        try:
            from ..services.elastic_search import get_elasticsearch_client

            es_client = get_elasticsearch_client()
            return es_client.search_bm25(query=query, top_k=BM25_K)
        except Exception:
            return []

    with ThreadPoolExecutor(max_workers=2) as executor:
        vector_future = executor.submit(_vector_search)
        bm25_future = executor.submit(_bm25_search)
        vector_results = vector_future.result()
        bm25_results = bm25_future.result()

    if vector_results and bm25_results:
        merged = _rrf_fuse([vector_results, bm25_results])
    elif vector_results:
        merged = vector_results
    else:
        merged = bm25_results

    if USE_RERANK and merged:
        try:
            from ..services.rerank import Qwen3RerankerService

            reranker = Qwen3RerankerService()
            reranked_items, _ = reranker.rerank(query, merged, top_n=FINAL_K)
            reranked_docs: List[Dict] = []
            for item in reranked_items[:FINAL_K]:
                idx = item.get("index", 0)
                if 0 <= idx < len(merged):
                    doc = merged[idx].copy()
                    doc["relevance_score"] = item.get("relevance_score", 0.0)
                    reranked_docs.append(doc)
            if reranked_docs:
                merged = reranked_docs
        except Exception:
            merged = merged[:FINAL_K]

    final_docs = merged[:FINAL_K]
    for doc in final_docs:
        content = doc.get("content") or doc.get("text") or ""
        doc["content"] = _truncate_doc_text(content)

    try:
        from ..core.cache import cache_search_results

        ok = cache_search_results(search_key, final_docs, ttl_seconds=RETRIEVAL_CACHE_TTL_SECONDS)
        if not ok:
            _cache_set_json(search_key, final_docs, ttl=RETRIEVAL_CACHE_TTL_SECONDS)
    except Exception:
        _cache_set_json(search_key, final_docs, ttl=RETRIEVAL_CACHE_TTL_SECONDS)
    return final_docs


def _adaptive_token_limit(messages: List[Dict[str, str]], requested: int) -> int:
    """Use requested cap but adapt down only for very long prompts."""
    cap = max(256, min(LLM_MAX_TOKENS_CAP, 2048))
    base = max(256, min(int(requested), cap))
    total_chars = sum(len((m.get("content") or "")) for m in messages)
    if total_chars > 14000:
        return max(256, min(base, 512))
    if total_chars > 10000:
        return max(384, min(base, 768))
    return base


def _normalize_max_tokens(max_tokens: Optional[int]) -> int:
    """Clamp user/config tokens into safe production range [256, 2048]."""
    try:
        requested = int(max_tokens) if max_tokens is not None else 512
    except Exception:
        requested = 512
    return max(256, min(requested, 2048))


def _resolve_ollama_model(ollama_url: str) -> Optional[str]:
    """Resolve the Ollama model name to use.

    Priority order:
    1. ``OLLAMA_MODEL`` env var  — explicit Ollama-specific name (e.g. ``qwen3:8b``)
    2. ``MODEL_NAME`` env var    — shared override (only used if it looks like an
                                   Ollama tag, i.e. contains no '/')
    3. Auto-detect               — query ``/api/tags`` and pick the first installed model
    4. YAML generation model     — last resort (may not work if it's a HuggingFace ID)
    """
    # 1. Explicit Ollama model override
    ollama_model = os.getenv("OLLAMA_MODEL", "").strip()
    if ollama_model:
        log.debug("[OLLAMA] Using OLLAMA_MODEL env var: %s", ollama_model)
        return ollama_model

    # 2. Generic MODEL_NAME — only trust it when it looks like an Ollama tag
    model_name_env = os.getenv("MODEL_NAME", "").strip()
    if model_name_env and "/" not in model_name_env:
        log.debug("[OLLAMA] Using MODEL_NAME env var as Ollama tag: %s", model_name_env)
        return model_name_env

    # 3. Auto-detect from /api/tags
    try:
        resp = httpx.get(f"{ollama_url}/api/tags", timeout=5.0)
        resp.raise_for_status()
        models_list = resp.json().get("models", [])
        if models_list:
            detected = models_list[0]["name"]
            log.info("[OLLAMA] Auto-detected model from /api/tags: %s", detected)
            logger.info(f"[OLLAMA] Auto-detected model: {detected}")
            return detected
    except Exception as e:
        log.warning("[OLLAMA] Could not fetch /api/tags: %s", e)

    # 4. Fallback to YAML value (may fail if it's a HuggingFace ID)
    try:
        yaml_model = get_generation_model()
        log.warning(
            "[OLLAMA] Falling back to YAML model name '%s' — "
            "set OLLAMA_MODEL env var if this is a HuggingFace ID",
            yaml_model,
        )
        return yaml_model
    except Exception:
        return None


def _get_cached_ollama_model(ollama_url: str) -> Optional[str]:
    global _ollama_model_cache
    if _ollama_model_cache:
        return _ollama_model_cache
    with _ollama_model_lock:
        if _ollama_model_cache:
            return _ollama_model_cache
        _ollama_model_cache = _resolve_ollama_model(ollama_url)
        return _ollama_model_cache


def ollama_chat_complete(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> Optional[str]:
    """Generate chat completion via Ollama's /api/chat endpoint.

    Model resolution order (when ``model`` is not provided):
    ``OLLAMA_MODEL`` env var → ``MODEL_NAME`` env var (Ollama tag only) →
    auto-detect from ``/api/tags`` → YAML config value.

    URL is read from ``OLLAMA_URL`` (default ``http://localhost:11434``).
    """
    ollama_url = os.getenv("OLLAMA_URL", _DEFAULT_OLLAMA_URL)
    ollama_think = os.getenv("OLLAMA_THINK", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if model is None:
        model = _get_cached_ollama_model(ollama_url)

    if not model:
        log.error("[OLLAMA] Could not determine model name — set OLLAMA_MODEL env var")
        logger.error("[OLLAMA] Could not determine model name")
        return None

    log.info("[OLLAMA] Sending request to %s (model: %s)", ollama_url, model)
    try:
        response = httpx.post(
            f"{ollama_url}/api/chat",
            json={
                "model": model,
                "messages": messages,
                "stream": False,
                "think": ollama_think,
                "options": {
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "top_p": 0.9,
                },
            },
            timeout=LLM_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        body = response.json()
        content: str = body["message"]["content"]

        # Qwen3 thinking models in Ollama put reasoning into a separate
        # "thinking" field; content may still contain leftover <think> tags
        # from older builds — strip them so callers get clean text.
        import re as _re
        content = _re.sub(r"<think>.*?</think>", "", content, flags=_re.DOTALL).strip()

        if not content:
            thinking_preview = (body["message"].get("thinking") or "")[:120]
            log.warning(
                "[OLLAMA] Model returned empty content (thinking model may have "
                "exhausted token budget). thinking_preview=%r", thinking_preview
            )
            logger.warning(
                f"[OLLAMA] Empty content from thinking model — "
                f"increase max_tokens. thinking preview: {thinking_preview!r}"
            )

        return content
    except Exception as e:
        log.error("[OLLAMA] Request failed: %s", e)
        logger.error(f"[OLLAMA] Request failed: {e}")
        return None


def ollama_chat_stream(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> Iterator[str]:
    """Stream chat completion chunks via Ollama /api/chat."""
    ollama_url = os.getenv("OLLAMA_URL", _DEFAULT_OLLAMA_URL)
    ollama_think = os.getenv("OLLAMA_THINK", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }
    if model is None:
        model = _get_cached_ollama_model(ollama_url)

    if not model:
        return

    payload = {
        "model": model,
        "messages": messages,
        "stream": True,
        "think": ollama_think,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "top_p": 0.9,
        },
    }

    with httpx.stream(
        "POST",
        f"{ollama_url}/api/chat",
        json=payload,
        timeout=httpx.Timeout(
            connect=min(10.0, LLM_TIMEOUT_SECONDS),
            read=LLM_STREAM_READ_TIMEOUT_SECONDS,
            write=min(20.0, LLM_TIMEOUT_SECONDS),
            pool=min(10.0, LLM_TIMEOUT_SECONDS),
        ),
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line:
                continue

            try:
                item = json.loads(line)
            except Exception:
                continue

            token = ((item or {}).get("message") or {}).get("content") or ""
            if token:
                yield token

            if bool(item.get("done", False)):
                break


def get_response(
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> Optional[str]:
    """Single-provider generation with cache.

    Provider is selected by `LLM_PROVIDER` env var (`ollama` | `vllm`).
    No fallback chain to avoid extra latency.
    """
    safe_tokens = _normalize_max_tokens(max_tokens)
    capped_tokens = _adaptive_token_limit(messages, safe_tokens)
    logger.info(f"[LLM] max_tokens used: {capped_tokens}")
    cache_basis = json.dumps(messages[-3:], ensure_ascii=False)
    provider = get_active_llm_provider()
    cache_key = _hash_key("answer", cache_basis, str(capped_tokens), f"{temperature:.2f}", provider)
    cached = _cache_get_json(cache_key)
    if isinstance(cached, str) and cached:
        return cached

    if not _provider_is_available(provider):
        return LLM_TIMEOUT_FALLBACK

    if provider == "vllm":
        vllm_url = os.getenv("VLLM_URL", _DEFAULT_VLLM_URL)
        model_name = os.getenv("MODEL_NAME") or get_generation_model()
        try:
            client = OpenAI(
                api_key=get_vllm_api_key() or "EMPTY",
                base_url=f"{vllm_url}/v1",
                timeout=LLM_TIMEOUT_SECONDS,
            )
            resp = client.chat.completions.create(
                model=model_name,
                messages=messages,
                temperature=temperature,
                max_tokens=capped_tokens,
                top_p=0.9,
            )
            output = (resp.choices[0].message.content or "").strip()
        except Exception:
            _mark_provider_unavailable("vllm")
            return LLM_TIMEOUT_FALLBACK
    else:
        try:
            output = (ollama_chat_complete(
                messages=messages,
                model=None,
                temperature=temperature,
                max_tokens=capped_tokens,
            ) or "").strip()
        except Exception:
            _mark_provider_unavailable("ollama")
            return LLM_TIMEOUT_FALLBACK

    if output:
        _cache_set_json(cache_key, output, ttl=ANSWER_CACHE_TTL_SECONDS)
    return output or LLM_TIMEOUT_FALLBACK


def get_response_stream(
    messages: List[Dict[str, str]],
    temperature: float = 0.3,
    max_tokens: int = 512,
) -> Iterator[str]:
    """Single-provider streaming generation (no fallback chain)."""
    safe_tokens = _normalize_max_tokens(max_tokens)
    capped_tokens = _adaptive_token_limit(messages, safe_tokens)
    logger.info(f"[LLM] max_tokens used: {capped_tokens}")

    provider = get_active_llm_provider()
    if not _provider_is_available(provider):
        yield "Xin lỗi, dịch vụ tạo phản hồi đang tạm thời không khả dụng."
        return

    if provider == "vllm":
        try:
            for token in qwen3_chat_stream(
                messages=messages,
                temperature=temperature,
                max_tokens=capped_tokens,
            ):
                if token:
                    yield token
            return
        except Exception:
            _mark_provider_unavailable("vllm")
            yield "Xin lỗi, dịch vụ tạo phản hồi đang tạm thời không khả dụng."
            return

    try:
        for token in ollama_chat_stream(
            messages=messages,
            temperature=temperature,
            max_tokens=capped_tokens,
        ):
            if token:
                yield token
        return
    except Exception:
        _mark_provider_unavailable("ollama")
        yield "Xin lỗi, dịch vụ tạo phản hồi đang tạm thời không khả dụng."


def get_openai_client():
    """Get OpenAI client using API key from settings or environment."""
    try:
        api_key = settings.openai_api_key or os.getenv("OPENAI_API_KEY", "")
        if not api_key:
            raise ValueError(
                "OpenAI API key not set. Configure OPENAI_API_KEY in .env or environment."
            )
        client = OpenAI(api_key=api_key)
        return client
    except Exception as e:
        logger.error(f"[GEN] OpenAI client init failed: {e}")
        raise


def generate_conversation_text(conversations):
    try:
        conversation_text = ""
        for conversation in conversations:
            if conversation.get("role") in ["user", "assistant"]:
                role = conversation.get("role")
                content = conversation.get("content", "")
                conversation_text += f"{role}: {content}\n"
        return conversation_text
    except Exception as e:
        logger.error(f"[GEN] Conversation text generation failed: {e}")
        raise


def detect_route(history, message):
    """Detect conversation route (medical vs general)."""
    text = (message or "").strip().lower()
    if not text:
        return "general"

    medical_keywords = (
        "bệnh",
        "triệu chứng",
        "thuốc",
        "điều trị",
        "xét nghiệm",
        "chẩn đoán",
        "đau",
        "sốt",
        "ho",
        "viêm",
        "tim",
        "phổi",
        "huyết áp",
        "tiểu đường",
        "covid",
        "ung thư",
        "nhi khoa",
    )
    return "medical" if any(k in text for k in medical_keywords) else "general"


def get_tavily_agent_answer(messages):
    """Backward-compatible wrapper returning only answer text."""
    result = get_tavily_agent_answer_with_sources(messages, use_web_search=True)
    return result.get("answer")


def get_tavily_agent_answer_with_sources(messages, use_web_search: bool = True):
    """Generate answer using Tavily web search with intelligent context management.
    
    Flow:
    1. Extract user's latest question from messages
    2. Search web via Tavily
    3. Use vLLM (Qwen3) to generate final answer from search results
    """
    try:
        from ..functions.web_search import tavily_search, truncate_tavily_query

        # Extract the latest user query from messages
        user_query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_query = msg.get("content", "")
                break

        if not user_query:
            return {
                "answer": "Xin lỗi, không tìm thấy câu hỏi để tìm kiếm.",
                "citations": [],
                "query": "",
            }

        # If user_query is a long RAG prompt, try to extract only the original question.
        extracted = None
        patterns = [
            r"\*\*Câu hỏi:\*\*\s*(.+?)(?:\n\n|$)",
            r"Câu hỏi:\s*(.+?)(?:\n\n|$)",
            r"Question:\s*(.+?)(?:\n\n|$)",
        ]
        for pattern in patterns:
            m = re.search(pattern, user_query, flags=re.IGNORECASE | re.DOTALL)
            if m and m.group(1).strip():
                extracted = m.group(1).strip()
                break

        search_query = truncate_tavily_query(extracted or user_query)
        if not search_query:
            return {
                "answer": "Xin lỗi, không tìm thấy câu hỏi hợp lệ để tìm kiếm.",
                "citations": [],
                "query": "",
            }

        # Smart routing: when web search is OFF, run local RAG retrieval only.
        if not use_web_search:
            docs = hybrid_retrieve(search_query, retrieval_mode="rag")
            context_blocks: List[str] = []
            for i, doc in enumerate(docs[:FINAL_K], start=1):
                title = doc.get("title") or doc.get("file_name") or f"Tài liệu {i}"
                snippet = _truncate_doc_text(doc.get("content") or doc.get("text") or "")
                if snippet:
                    context_blocks.append(f"[{i}] {title}: {snippet}")

            compact_context = "\n".join(context_blocks)
            local_messages = [
                {
                    "role": "system",
                    "content": "Bạn là trợ lý y tế Việt Nam. Trả lời ngắn gọn, chính xác, dễ hiểu.",
                },
                {
                    "role": "user",
                    "content": (
                        f"Ngữ cảnh tài liệu:\n{compact_context}\n\n"
                        f"Câu hỏi: {search_query}\n"
                        "Nếu ngữ cảnh chưa đủ, hãy nói rõ giới hạn thông tin."
                    ),
                },
            ]
            local_answer = get_response(
                messages=local_messages,
                temperature=0.4,
                max_tokens=min(LLM_MAX_TOKENS_CAP, 512),
            )
            return {
                "answer": local_answer or "Xin lỗi, không thể tạo câu trả lời từ dữ liệu nội bộ.",
                "citations": [],
                "query": search_query,
            }

        # Search web via Tavily
        cached_web_key = _hash_key("search", "web", search_query.strip().lower(), str(WEB_SEARCH_MAX_RESULTS))
        cached_web = _cache_get_json(cached_web_key)
        if isinstance(cached_web, dict):
            observation = str(cached_web.get("observation") or "")
            web_citations = list(cached_web.get("citations") or [])
        else:
            _observation, web_citations = tavily_search(
                search_query,
                max_results=WEB_SEARCH_MAX_RESULTS,
                return_results=True,
            )

            # keep only short snippets for latency + prompt budget
            compact_citations: List[Dict[str, str]] = []
            for item in web_citations[:WEB_SEARCH_MAX_RESULTS]:
                snippet = _truncate_doc_text(item.get("snippet") or item.get("content") or "", WEB_SEARCH_SNIPPET_MAX_CHARS)
                compact_citations.append(
                    {
                        **item,
                        "snippet": snippet,
                        "content": snippet,
                    }
                )
            web_citations = compact_citations

            observation = "\n".join(
                f"- {c.get('title', 'Nguồn web')}: {c.get('snippet', '')} ({c.get('url', '')})"
                for c in web_citations
            )
            _cache_set_json(
                cached_web_key,
                {"observation": observation, "citations": web_citations},
                ttl=CACHE_TTL_SECONDS,
            )

        # Keep only recent messages to avoid token overflow
        recent_messages = _truncate_messages(messages, max_messages=6)

        # Build messages for Qwen3 (compatible format - no "function" role)
        enhanced_messages = [
            {
                "role": "system",
                "content": (
                    "Bạn là trợ lý y tế Việt Nam. Hãy trả lời câu hỏi dựa trên "
                    "kết quả tìm kiếm web được cung cấp. Trích dẫn nguồn với URL "
                    "theo format 'Theo [Tên nguồn](URL), ...' và thêm phần "
                    "'Nguồn tham khảo:' ở cuối câu trả lời."
                ),
            },
            *recent_messages,
            {
                "role": "user",
                "content": (
                    f"Dưới đây là kết quả tìm kiếm từ internet:\n\n{observation}\n\n"
                    "Dựa vào kết quả tìm kiếm trên, hãy trả lời câu hỏi bằng tiếng Việt "
                    "một cách đầy đủ và chính xác. Nhớ trích dẫn nguồn."
                ),
            },
        ]

        # Use unified generation path (vLLM -> Ollama fallback)
        final_response = get_response(
            messages=enhanced_messages,
            temperature=0.5,
            max_tokens=min(LLM_MAX_TOKENS_CAP, 512),
        )

        if not final_response:
            return {
                "answer": "Xin lỗi, không thể tạo câu trả lời từ kết quả tìm kiếm.",
                "citations": web_citations,
                "query": search_query,
            }

        return {
            "answer": final_response,
            "citations": web_citations,
            "query": search_query,
        }
    except Exception as e:
        logger.error(f"[GEN] Tavily agent failed: {e}")
        return {
            "answer": f"Xin lỗi, đã có lỗi xảy ra khi tìm kiếm thông tin: {str(e)}",
            "citations": [],
            "query": "",
        }


def _truncate_messages(messages: list, max_messages: int = 6) -> list:
    """Keep only the most recent messages to avoid token overflow.
    
    Always preserves the system message (if any) + last N user/assistant messages.
    """
    if len(messages) <= max_messages:
        return messages

    # Separate system messages from conversation
    system_msgs = [m for m in messages if m.get("role") == "system"]
    conv_msgs = [m for m in messages if m.get("role") != "system"]

    # Keep system + last N conversation messages
    truncated = system_msgs + conv_msgs[-(max_messages - len(system_msgs)):]
    return truncated