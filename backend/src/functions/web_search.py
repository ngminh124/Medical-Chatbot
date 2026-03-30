import os
import re
from typing import Dict, List, Tuple
from urllib.parse import urlparse

from loguru import logger
from tavily import TavilyClient

from ..configs.setup import get_backend_settings

settings = get_backend_settings()
MAX_TAVILY_QUERY_LEN = 400


def _extract_domain(url: str) -> str:
    try:
        return urlparse(url).hostname.replace("www.", "") if url else ""
    except Exception:
        return ""


def _build_favicon_url(url: str) -> str:
    domain = _extract_domain(url)
    if not domain:
        return ""
    return f"https://www.google.com/s2/favicons?domain={domain}&sz=64"


def _clean_snippet(text: str, max_len: int = 320) -> str:
    cleaned = re.sub(r"\s+", " ", (text or "")).strip()
    if len(cleaned) <= max_len:
        return cleaned
    return cleaned[:max_len].rstrip() + "…"


def truncate_tavily_query(query: str, max_len: int = MAX_TAVILY_QUERY_LEN) -> str:
    """Normalize and truncate Tavily query to API limit."""
    cleaned = re.sub(r"\s+", " ", (query or "")).strip()
    if len(cleaned) <= max_len:
        return cleaned

    truncated = cleaned[:max_len].rstrip()
    logger.warning(
        f"[TAVILY] Query too long ({len(cleaned)} chars). Truncated to {len(truncated)} chars."
    )
    return truncated


def get_tavily_client():
    """Initialize Tavily client with API key from settings or environment."""
    try:
        api_key = settings.tavily_api_key or os.getenv("TAVILY_API_KEY", "")
        if not api_key:
            raise ValueError(
                "Tavily API key not set. Configure TAVILY_API_KEY in .env or environment."
            )
        client = TavilyClient(api_key=api_key)
        return client
    except Exception as e:
        logger.error(f"Error initializing Tavily client: {e}")
        raise


def _normalize_tavily_results(results: List[dict], max_results: int = 3) -> List[Dict]:
    """Normalize Tavily raw results into frontend-friendly citation items."""
    normalized: List[Dict] = []
    for doc in (results or [])[:max_results]:
        url = doc.get("url", "")
        domain = _extract_domain(url)
        snippet = _clean_snippet(doc.get("content", "No content available"))
        normalized.append(
            {
                "title": doc.get("title") or domain or "Nguồn web",
                "url": url,
                "snippet": snippet,
                "type": "web",
                "score": float(doc.get("score", 0.0) or 0.0),
                "domain": domain,
                "favicon": _build_favicon_url(url),
                # backward compatibility with existing UI fallbacks
                "content": snippet,
                "source": url,
            }
        )
    return normalized


def tavily_search(
    query: str,
    max_results: int = 3,
    return_results: bool = False,
) -> str | Tuple[str, List[Dict]]:
    try:
        safe_query = truncate_tavily_query(query)
        if not safe_query:
            raise ValueError("Empty Tavily query after normalization")

        client = get_tavily_client()
        raw = client.search(safe_query, max_results=max_results)
        output_search = _normalize_tavily_results(raw.get("results") or [], max_results)
        search_document = "Here are the retrieved documents from the internet:\n\n"

        for i, doc in enumerate(output_search):
            content = doc.get("snippet", "No content available")
            url = doc.get("url", "No URL available")
            title = doc.get("title", "Untitled")

            logger.debug(
                f"Source {i+1} - Title: {title}, URL: {url}, Content Length: {len(content)}"
            )

            search_document += f"**Source {i+1}:**\n"
            search_document += f"- Title: {title}\n"
            search_document += f"- Content: {content}\n"
            search_document += f"- URL: {url}\n\n"

        search_document += "---\n"
        search_document += "IMPORTANT: When using these search results in your response, you MUST cite the sources by including the URLs and mentioning which source number you're referencing.\n"

        if return_results:
            return search_document, output_search
        return search_document
    except Exception as e:
        logger.error(f"Error searching for external information using Tavily: {e}")
        raise


functions_info = [
    {
        "name": "tavily_search",
        "description": "Get information in internet based on user query ",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "This is user query",
                },
            },
            "required": ["query"],
        },
    }
]