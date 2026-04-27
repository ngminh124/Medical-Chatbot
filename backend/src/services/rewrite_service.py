"""Gemini API-based rewrite service optimized for Medical RAG."""

from __future__ import annotations
import time
from typing import Dict, List
import httpx
from loguru import logger
from ..configs.setup import get_backend_settings

settings = get_backend_settings()

# Biến toàn cục để quản lý rate limit
_rewrite_disabled_until_ts = 0.0
_rewrite_disable_reason = ""

def rewrite_query_with_api(query: str, history: List[Dict]) -> str:
    global _rewrite_disabled_until_ts
    global _rewrite_disable_reason

    current_query = (query or "").strip()
    if not current_query:
        return ""

    # Trích xuất 3 câu hỏi gần nhất của User để làm ngữ cảnh
    user_questions = [
        (m.get("content") or "").strip()
        for m in (history or [])
        if isinstance(m, dict) and (m.get("role") or "").lower() == "user" and (m.get("content") or "").strip()
    ]
    used_context = user_questions[-3:]

    logger.info(f"[REWRITE] input: {current_query[:200]}")
    logger.info(f"[REWRITE] used_context: {used_context}")

    context_lines = [f"{idx}. {q}" for idx, q in enumerate(used_context, 1)]
    previous_questions_block = "\n".join(context_lines) if context_lines else "(none)"

    system_prompt = (
        "You are a medical AI assistant specialized in rewriting user questions for better semantic search.\n\n"
        "Your task: Rewrite the user's latest question into a clear, standalone medical query in Vietnamese.\n"
        "* Use previous context to resolve pronouns (e.g., 'nó', 'triệu chứng đó').\n"
        "* ONLY return the rewritten question, no preamble.\n"
        "* If the question is already clear, return it unchanged."
    )

    user_prompt = (
        f"Previous context:\n{previous_questions_block}\n\n"
        f"Current question: {current_query}\n\n"
        "Rewritten question (Vietnamese):"
    )

    api_key = (settings.gemini_api_key or "").strip()
    if not api_key:
        logger.warning("[REWRITE] missing API KEY")
        return current_query

    # Kiểm tra nếu đang trong thời gian tạm dừng do lỗi 429
    now = time.time()
    if _rewrite_disabled_until_ts > now:
        remaining = int(_rewrite_disabled_until_ts - now)
        logger.warning(f"[REWRITE] cool-down active: {remaining}s left")
        return current_query

    # Xử lý URL chuẩn: Đảm bảo không bị lặp /v1beta/openai/chat/completions
    base_url = (settings.gemini_base_url or "https://generativelanguage.googleapis.com/v1beta/openai").rstrip("/")
    url = f"{base_url}/chat/completions"
    
    # Đảm bảo model không có tiền tố 'models/' khi dùng qua OpenAI adapter
    model = settings.gemini_rewrite_model or "gemini-1.5-flash"
    if model.startswith("models/"):
        model = model.replace("models/", "")

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 150,
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        # Tăng timeout lên một chút để tránh rớt mạng khi demo
        timeout = httpx.Timeout(10.0) 
        with httpx.Client(timeout=timeout) as client:
            response = client.post(url, headers=headers, json=payload)
            
            if response.status_code == 429:
                # Nếu dính rate limit, chỉ chặn 30 giây thay vì 30 phút để bạn dễ debug
                _rewrite_disabled_until_ts = time.time() + 30 
                _rewrite_disable_reason = "rate_limit_hit"
                logger.error(f"[REWRITE] 429 Error: {response.text}")
                return current_query
                
            response.raise_for_status()
            body = response.json()

        rewritten = body["choices"][0]["message"]["content"].strip()
        logger.info(f"[REWRITE] output: {rewritten[:200]}")
        return rewritten

    except Exception as e:
        logger.error(f"[REWRITE] API Error: {str(e)}")
        return current_query