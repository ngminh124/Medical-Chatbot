"""
Speech-to-Text (STT) Service - Routes to GPU service
Provides audio transcription with caching for Vietnamese Medical RAG system
"""

import hashlib
import time
from typing import Optional

import httpx
from loguru import logger

from ..configs.setup import get_backend_settings
from ..core.cache import get_redis_client

settings = get_backend_settings()


class SttService:
    """
    STT service that routes to GPU service or uses local fallback
    Automatically routes to GPU service if available
    """

    def __init__(
        self,
        model_name: str = "turbo",
        device: str = "cuda",
        compute_type: str = "float16",
    ):
        """
        Initialize STT service

        Args:
            model_name: Whisper model (ignored if using GPU service)
            device: Device for inference (ignored if using GPU service)
            compute_type: Computation type (ignored if using GPU service)
        """
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self.use_gpu_service = settings.qwen3_models_enabled
        self.service_url = settings.stt_gpu_service_url.rstrip("/")
        self.cache_ttl = int(settings.stt_cache_ttl)
        self.timeout = float(settings.service_http_timeout)
        self.redis_client = get_redis_client()
        self.client = httpx.AsyncClient(
            timeout=self.timeout,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )

        if self.use_gpu_service:
            logger.info(f"[STT] Using GPU service at {self.service_url}")
        else:
            logger.info(
                f"[STT] GPU service disabled, local STT is not implemented (model={model_name})"
            )

    def load_model(self):
        """Load model - no-op if using GPU service"""
        if self.use_gpu_service:
            logger.info("[STT] GPU service handles model loading")
            return

        logger.warning(
            "[STT] GPU service disabled. Local STT not implemented in this version."
        )
        logger.warning("[STT] Enable QWEN3_MODELS_ENABLED to use GPU STT service.")

    def _get_audio_hash(self, audio_bytes: bytes) -> str:
        """
        Generate hash of audio file for caching

        Args:
            audio_path: Path to audio file

        Returns:
            SHA256 hash of audio file content
        """
        try:
            return hashlib.sha256(audio_bytes).hexdigest()
        except Exception as e:
            logger.error(f"[STT] Failed to hash audio bytes: {e}")
            return ""

    def _get_cached_transcript(self, audio_hash: str) -> Optional[str]:
        """
        Get cached transcript from Redis

        Args:
            audio_hash: Hash of audio file

        Returns:
            Cached transcript or None
        """
        if not self.redis_client or not audio_hash:
            return None

        try:
            cache_key = f"stt:transcript:{audio_hash}"
            cached = self.redis_client.get(cache_key)
            if cached:
                logger.info(f"[STT] Cache hit: {audio_hash[:16]}...")
                return cached if isinstance(cached, str) else cached.decode("utf-8")
            return None
        except Exception as e:
            logger.error(f"[STT] Failed to get cached transcript: {e}")
            return None

    def _cache_transcript(self, audio_hash: str, transcript: str):
        """
        Cache transcript to Redis

        Args:
            audio_hash: Hash of audio file
            transcript: Transcribed text
        """
        if not self.redis_client or not audio_hash:
            return

        try:
            cache_key = f"stt:transcript:{audio_hash}"
            self.redis_client.setex(cache_key, self.cache_ttl, transcript)
            logger.info(f"[STT] Transcript cached: {audio_hash[:16]}...")
        except Exception as e:
            logger.error(f"[STT] Failed to cache transcript: {e}")

    async def transcribe_audio(
        self,
        audio_bytes: bytes,
        filename: str = "audio.wav",
        language: str = "vi",
        beam_size: int = 5,
        vad_filter: bool = True,
        batch_size: int = 16,
    ) -> dict:
        """
        Transcribe audio file to text
        Routes to GPU service if enabled, otherwise raises error

        Args:
            audio_bytes: Raw audio bytes (WAV, MP3, OGG, etc.)
            filename: Original filename for multipart metadata
            language: Language code (default: "vi" for Vietnamese)
            beam_size: Beam size for decoding (ignored for GPU service)
            vad_filter: Enable voice activity detection (ignored for GPU service)
            batch_size: Batch size for GPU inference (default: 16)

        Returns:
            dict with keys:
                - text: Transcribed text
                - language: Detected/specified language
                - duration: Audio duration in seconds
                - segments: List of segments with timestamps (optional)
                - cached: Whether result from cache
        """
        _ = beam_size
        _ = vad_filter

        if not self.use_gpu_service:
            raise NotImplementedError(
                "STT GPU service is disabled. Please enable QWEN3_MODELS_ENABLED=true"
            )

        if not audio_bytes:
            raise ValueError("Audio file is empty")

        # Check cache first (backend cache, GPU service has its own cache too)
        audio_hash = self._get_audio_hash(audio_bytes)
        cached_transcript = self._get_cached_transcript(audio_hash)
        if cached_transcript:
            return {
                "text": cached_transcript,
                "language": language,
                "duration": 0.0,
                "segments": [],
                "cached": True,
            }

        try:
            started_at = time.perf_counter()
            logger.info(
                f"[STT] Start transcription: filename={filename}, size={len(audio_bytes)}B, language={language}"
            )

            files = {"file": (filename, audio_bytes, "application/octet-stream")}
            data = {
                "language": language,
                "batch_size": batch_size,
            }

            response = await self.client.post(
                f"{self.service_url}/v1/models/stt",
                files=files,
                data=data,
            )
            response.raise_for_status()

            result = response.json()
            full_text = result.get("text", "")

            elapsed = time.perf_counter() - started_at
            logger.success(
                f"[STT] Complete: chars={len(full_text)}, cached={result.get('cached', False)}, duration={elapsed:.2f}s"
            )

            # Cache the transcript in backend Redis
            if audio_hash and not result.get("cached", False):
                self._cache_transcript(audio_hash, full_text)

            return {
                "text": full_text,
                "language": result.get("language", language),
                "duration": result.get("duration", 0.0),
                "segments": result.get("segments", []),
                "cached": result.get("cached", False),
            }

        except httpx.RemoteProtocolError as e:
            logger.error(f"[STT] GPU STT service protocol error: {e}")
            raise ConnectionError(
                "STT service không khả dụng. Vui lòng thử lại sau vài giây."
            ) from e
        except httpx.ConnectError as e:
            logger.error(f"[STT] Cannot connect to GPU STT service: {e}")
            raise ConnectionError(
                "Không thể kết nối STT service. Vui lòng kiểm tra GPU service đã khởi động chưa."
            ) from e
        except httpx.HTTPStatusError as e:
            status_code = e.response.status_code
            logger.error(f"[STT] GPU STT service HTTP error: {status_code}")
            if 400 <= status_code < 500:
                raise ValueError(
                    f"STT service rejected the request (HTTP {status_code})"
                ) from e
            raise RuntimeError(f"STT service unavailable (HTTP {status_code})") from e
        except httpx.TimeoutException as e:
            logger.error(f"[STT] Request timeout: {e}")
            raise TimeoutError(
                "STT xử lý quá lâu. Vui lòng thử lại với file audio ngắn hơn."
            ) from e
        except Exception as e:
            logger.error(f"[STT] Transcription failed: {e}", exc_info=True)
            raise

    async def health_check(self) -> bool:
        """Check STT service availability."""
        if not self.use_gpu_service:
            return False
        try:
            response = await self.client.get(
                f"{self.service_url}/v1/health", timeout=min(self.timeout, 5.0)
            )
            if response.status_code == 200:
                return True
            response = await self.client.get(
                f"{self.service_url}/health", timeout=min(self.timeout, 5.0)
            )
            return response.status_code == 200
        except Exception:
            return False

    async def close(self):
        """Close shared async HTTP client."""
        await self.client.aclose()


# Global STT service instance
_stt_service: Optional[SttService] = None


def get_stt_service() -> SttService:
    """Get or create global STT service instance"""
    global _stt_service
    if _stt_service is None:
        # Configuration will be loaded from models.yaml
        _stt_service = SttService()
    return _stt_service


def initialize_stt_service(
    model_name: str = "turbo",
    device: str = "cuda",
    compute_type: str = "float16",
):
    """
    Initialize STT service with custom configuration

    Args:
        model_name: Whisper model
        device: Device for inference
        compute_type: Computation type
    """
    global _stt_service
    _stt_service = SttService(
        model_name=model_name, device=device, compute_type=compute_type
    )
    _stt_service.load_model()
    logger.info("[STT] Service initialized")


async def close_stt_service():
    """Close global STT service resources if initialized."""
    global _stt_service
    if _stt_service is not None:
        await _stt_service.close()