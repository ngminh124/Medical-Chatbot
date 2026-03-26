import logging
import os
import re
from typing import Dict, List, Optional

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


def get_vllm_client():
    """Get remote vLLM client from config."""
    try:
        vllm_url = get_vllm_url()
        vllm_api_key = get_vllm_api_key()

        client = OpenAI(
            api_key=vllm_api_key,
            base_url=f"{vllm_url}/v1",
            timeout=200.0,
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
    max_tokens = max_tokens if max_tokens is not None else 2048

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


def ollama_chat_complete(
    messages: List[Dict[str, str]],
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_tokens: int = 2048,
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
        model = _resolve_ollama_model(ollama_url)

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
                },
            },
            timeout=300.0,
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


def get_response(
    messages: List[Dict[str, str]],
    temperature: float = 0.7,
    max_tokens: int = 2048,
) -> Optional[str]:
    """Unified generation entry-point with automatic vLLM → Ollama fallback.

    Fallback logic
    --------------
    Step 1 — vLLM (VLLM_URL, default http://localhost:7861)
        Uses the OpenAI-compatible /v1/chat/completions endpoint.
        On APIConnectionError or APITimeoutError the function continues to
        Step 2 instead of raising, so callers never see a hard crash.

    Step 2 — Ollama (OLLAMA_URL, default http://localhost:11434)
        Falls back to Ollama's native /api/chat endpoint.
        A ``logging`` message is emitted so infrastructure teams can detect
        when the primary LLM service is degraded.

    Environment Variables
    ---------------------
    VLLM_URL   : vLLM base URL  (default: http://localhost:7861)
    OLLAMA_URL : Ollama base URL (default: http://localhost:11434)
    MODEL_NAME : Override the model name for *both* services.
                 Defaults to the value in models.yaml.
    """
    vllm_url = os.getenv("VLLM_URL", _DEFAULT_VLLM_URL)
    model_name = os.getenv("MODEL_NAME") or get_generation_model()

    # ── Step 1: vLLM ─────────────────────────────────────────────────────────
    try:
        log.info("[GET_RESPONSE] Trying vLLM at %s (model: %s)", vllm_url, model_name)
        client = OpenAI(
            api_key=get_vllm_api_key() or "EMPTY",
            base_url=f"{vllm_url}/v1",
            timeout=60.0,
        )
        resp = client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=0.8,
        )
        content: str = resp.choices[0].message.content or ""
        log.info("[GET_RESPONSE] vLLM responded successfully")
        logger.success("[GET_RESPONSE] vLLM responded successfully")
        return content

    except (openai.APIConnectionError, openai.APITimeoutError) as e:
        # Network-level failures → switch providers
        log.warning(
            "[GET_RESPONSE] vLLM unavailable (%s: %s) — switching to Ollama fallback",
            type(e).__name__,
            e,
        )
        logger.warning(f"[GET_RESPONSE] vLLM unreachable ({type(e).__name__}): {e}")

    except (ConnectionError, TimeoutError) as e:
        log.warning(
            "[GET_RESPONSE] vLLM connection/timeout (%s: %s) — switching to Ollama fallback",
            type(e).__name__,
            e,
        )
        logger.warning(f"[GET_RESPONSE] vLLM connection/timeout: {e}")

    except Exception as e:
        error_str = str(e)
        if "<!DOCTYPE html>" in error_str or "<html" in error_str:
            error_str = "vLLM service is loading"
        log.warning(
            "[GET_RESPONSE] vLLM error (%s: %s) — switching to Ollama fallback",
            type(e).__name__,
            error_str,
        )
        logger.warning(f"[GET_RESPONSE] vLLM error: {error_str}")

    # ── Step 2: Ollama fallback ───────────────────────────────────────────────
    # Do NOT pass model_name here — it is a vLLM/HuggingFace model ID (e.g.
    # "Qwen/Qwen3-4B") that Ollama won't recognise.  Passing model=None lets
    # _resolve_ollama_model() do the correct lookup (OLLAMA_MODEL env var →
    # MODEL_NAME env var → /api/tags auto-detect → YAML last resort).
    log.info("[GET_RESPONSE] Attempting Ollama fallback (auto-resolving Ollama model)")
    logger.info("[GET_RESPONSE] Attempting Ollama fallback...")

    result = ollama_chat_complete(
        messages=messages,
        model=None,
        temperature=temperature,
        max_tokens=max_tokens,
    )

    if result:
        log.info("[GET_RESPONSE] Ollama responded successfully")
        logger.success("[GET_RESPONSE] Ollama responded successfully")
    elif result is not None:
        # Empty string: Ollama answered but content was blank
        # (thinking model exhausted token budget — caller should retry with
        # a larger max_tokens or use /no_think in the system prompt).
        log.warning("[GET_RESPONSE] Ollama returned empty content (thinking model token exhaustion?)")
        logger.warning("[GET_RESPONSE] Ollama returned empty content — consider increasing max_tokens")
    else:
        log.error("[GET_RESPONSE] Both vLLM and Ollama failed to produce a response")
        logger.error("[GET_RESPONSE] Both vLLM and Ollama failed")

    return result


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


def enhance_query_quality(history, message):
    """Enhance user query quality by rephrasing with conversation context."""
    try:
        history_messages = generate_conversation_text(history)
        enhanced_prompt = settings.rewrite_prompt.format(
            history_messages=history_messages, message=message
        )

        messages = [
            {
                "role": "system",
                "content": "You are an expert in rephrasing user questions.",
            },
            {"role": "user", "content": enhanced_prompt},
        ]

        enhanced_query = qwen3_chat_complete(messages)
        return enhanced_query if enhanced_query else message
    except Exception as e:
        logger.error(f"[GEN] Query enhancement failed: {e}")
        return message


def detect_route(history, message):
    """Detect conversation route (medical vs general)."""
    try:
        user_prompt = settings.intent_detection_prompt.format(
            history=history,
            message=message,
        )

        messages = [
            {
                "role": "system",
                "content": "You are an expert in classifying user intents.",
            },
            {"role": "user", "content": user_prompt},
        ]

        route = qwen3_chat_complete(messages)
        return route if route else "medical"
    except Exception as e:
        logger.error(f"[GEN] Route detection failed: {e}")
        return "medical"


def get_tavily_agent_answer(messages):
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
            return "Xin lỗi, không tìm thấy câu hỏi để tìm kiếm."

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
            return "Xin lỗi, không tìm thấy câu hỏi hợp lệ để tìm kiếm."

        # Search web via Tavily
        observation = tavily_search(search_query)

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

        final_response = qwen3_chat_complete(enhanced_messages, max_tokens=1536)

        if not final_response:
            return "Xin lỗi, không thể tạo câu trả lời từ kết quả tìm kiếm."

        return final_response
    except Exception as e:
        logger.error(f"[GEN] Tavily agent failed: {e}")
        return f"Xin lỗi, đã có lỗi xảy ra khi tìm kiếm thông tin: {str(e)}"


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