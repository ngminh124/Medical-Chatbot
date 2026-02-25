import os
from typing import Dict, List, Optional

import httpx
from loguru import logger
from openai import OpenAI

from ..configs.setup import get_backend_settings
from ..core.model_config import (get_generation_model, get_vllm_api_key,
                                 get_vllm_url)

settings = get_backend_settings()


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
        from ..functions.web_search import tavily_search

        # Extract the latest user query from messages
        user_query = ""
        for msg in reversed(messages):
            if msg.get("role") == "user":
                user_query = msg.get("content", "")
                break

        if not user_query:
            return "Xin lỗi, không tìm thấy câu hỏi để tìm kiếm."

        # Search web via Tavily
        observation = tavily_search(user_query)

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