"""Remote model service client for embedding, rerank, and guard APIs."""

import asyncio
import time
from typing import Any, Dict, Optional

import anyio
import httpx
from loguru import logger

from ..configs.setup import get_backend_settings

settings = get_backend_settings()


class RemoteModelService:
    """HTTP client for remote model inference endpoints."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        timeout: Optional[float] = None,
        max_retries: Optional[int] = None,
        backoff_base: Optional[float] = None,
    ) -> None:
        self.base_url = (base_url or settings.model_service_url).rstrip("/")
        self.timeout = float(timeout or settings.service_http_timeout)
        self.max_retries = max(0, int(max_retries or settings.service_http_retries))
        self.backoff_base = max(0.05, float(backoff_base or settings.service_http_backoff_seconds))
        self._client = httpx.AsyncClient(
            timeout=self.timeout,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
        self._sync_client = httpx.Client(
            timeout=self.timeout,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )
        self._failures = 0
        self._down_until = 0.0
        self._down_ttl_seconds = 60
        self._failure_threshold = 3

    def _run(self, async_func, *args, **kwargs):
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            try:
                return anyio.from_thread.run(async_func, *args, **kwargs)
            except RuntimeError:
                return asyncio.run(async_func(*args, **kwargs))

        if async_func == self._post_json:
            return self._post_json_sync(*args, **kwargs)
        if async_func == self._get:
            return self._get_sync(*args, **kwargs)
        raise RuntimeError("Remote model call cannot be executed in a running event loop")

    async def _post_json(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if time.time() < self._down_until:
            raise RuntimeError("Remote model service is temporarily DOWN (circuit breaker)")

        response: Optional[httpx.Response] = None
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = await self._client.post(
                    f"{self.base_url}{path}",
                    json=payload,
                )
                if response.status_code == 200:
                    self._failures = 0
                    return response.json()

                if response.status_code < 500 and response.status_code != 429:
                    raise httpx.HTTPStatusError(
                        f"Remote model request failed: {response.status_code} - {response.text}",
                        request=response.request,
                        response=response,
                    )

                raise httpx.HTTPStatusError(
                    f"Remote model transient error: {response.status_code}",
                    request=response.request,
                    response=response,
                )
            except Exception as exc:
                last_error = exc
                self._failures += 1
                if self._failures > self._failure_threshold:
                    self._down_until = time.time() + self._down_ttl_seconds
                    logger.warning(
                        "[MODEL][CB] Service marked DOWN for "
                        f"{self._down_ttl_seconds}s"
                    )
                if attempt >= self.max_retries:
                    break
                await asyncio.sleep(self.backoff_base * (2 ** attempt))

        if response is None:
            raise RuntimeError(f"Remote model request failed: {last_error}")

        raise RuntimeError(
            f"Remote model request failed: {response.status_code} - {response.text}"
        )

    def _post_json_sync(self, path: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if time.time() < self._down_until:
            raise RuntimeError("Remote model service is temporarily DOWN (circuit breaker)")

        response: Optional[httpx.Response] = None
        last_error: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                response = self._sync_client.post(
                    f"{self.base_url}{path}",
                    json=payload,
                )
                if response.status_code == 200:
                    self._failures = 0
                    return response.json()

                if response.status_code < 500 and response.status_code != 429:
                    raise httpx.HTTPStatusError(
                        f"Remote model request failed: {response.status_code} - {response.text}",
                        request=response.request,
                        response=response,
                    )

                raise httpx.HTTPStatusError(
                    f"Remote model transient error: {response.status_code}",
                    request=response.request,
                    response=response,
                )
            except Exception as exc:
                last_error = exc
                self._failures += 1
                if self._failures > self._failure_threshold:
                    self._down_until = time.time() + self._down_ttl_seconds
                    logger.warning(
                        "[MODEL][CB] Service marked DOWN for "
                        f"{self._down_ttl_seconds}s"
                    )
                if attempt >= self.max_retries:
                    break
                time.sleep(self.backoff_base * (2 ** attempt))

        if response is None:
            raise RuntimeError(f"Remote model request failed: {last_error}")

        raise RuntimeError(
            f"Remote model request failed: {response.status_code} - {response.text}"
        )

    def embed(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._run(self._post_json, "/v1/models/embed", payload)

    def rerank(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._run(self._post_json, "/v1/models/rerank", payload)

    def guard(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return self._run(self._post_json, "/v1/models/guard", payload)

    async def _get(self, path: str, timeout: float) -> bool:
        try:
            response = await self._client.get(f"{self.base_url}{path}", timeout=timeout)
            return response.status_code == 200
        except Exception:
            return False

    def _get_sync(self, path: str, timeout: float) -> bool:
        try:
            response = self._sync_client.get(f"{self.base_url}{path}", timeout=timeout)
            return response.status_code == 200
        except Exception:
            return False

    def health_check(self, timeout: float = 5.0) -> bool:
        return self._run(self._get, "/v1/ready", timeout)

    async def aclose(self) -> None:
        await self._client.aclose()
        self._sync_client.close()


_remote_model_service_instance: Optional[RemoteModelService] = None


def get_remote_model_service() -> RemoteModelService:
    global _remote_model_service_instance
    if _remote_model_service_instance is None:
        _remote_model_service_instance = RemoteModelService()
    return _remote_model_service_instance


async def close_remote_model_service() -> None:
    if _remote_model_service_instance is not None:
        await _remote_model_service_instance.aclose()
