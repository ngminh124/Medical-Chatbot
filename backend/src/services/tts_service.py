"""
Text-to-Speech (TTS) Service
Supports ElevenLabs API for high-quality voice synthesis
"""

import base64
import hashlib
import threading
import time
from typing import Optional

import httpx
from loguru import logger

from ..configs.setup import get_backend_settings
from ..core.cache import get_redis_client

# ElevenLabs API configuration
ELEVENLABS_API_URL = "https://api.elevenlabs.io/v1/text-to-speech"
settings = get_backend_settings()


class TtsService:
    """
    TTS service using ElevenLabs API
    Provides high-quality voice synthesis with caching
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        voice_id: Optional[str] = None,
    ):
        """
        Initialize TTS service

        Args:
            api_key: ElevenLabs API key
            voice_id: Default voice ID to use
        """
        self.api_key = api_key if api_key is not None else settings.elevenlabs_api_key
        self.voice_id = (
            voice_id if voice_id is not None else settings.elevenlabs_voice_id
        )
        self.cache_ttl = int(settings.tts_cache_ttl)
        self.timeout = max(float(settings.service_http_timeout) * 3, 60.0)
        self.redis_client = get_redis_client()
        self.client = httpx.AsyncClient(
            timeout=self.timeout,
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
        )

        if not self.api_key:
            logger.warning("[TTS] ElevenLabs API key not configured")
        else:
            logger.info(
                "[TTS] API key detected: len={}, suffix={}",
                len(self.api_key),
                self.api_key[-6:] if len(self.api_key) >= 6 else "***",
            )

        logger.info(
            f"[TTS] Initializing service: voice_id={self.voice_id}, timeout={self.timeout}s"
        )

    def _get_text_hash(self, text: str, voice_id: str) -> str:
        """
        Generate hash of text + voice_id for caching

        Args:
            text: Text to synthesize
            voice_id: Voice identifier

        Returns:
            SHA256 hash of text + voice_id
        """
        content = f"{text}:{voice_id}"
        return hashlib.sha256(content.encode("utf-8")).hexdigest()

    def _get_cached_audio(self, text_hash: str) -> Optional[bytes]:
        """
        Get cached audio from Redis

        Args:
            text_hash: Hash of text + voice_id

        Returns:
            Cached audio data or None
        """
        if not self.redis_client or not text_hash:
            return None

        try:
            cache_key = f"tts:audio:{text_hash}"
            cached = self.redis_client.get(cache_key)
            if cached:
                logger.info(f"[TTS] Cache hit: {text_hash[:16]}...")
                cached_str = cached if isinstance(cached, str) else cached.decode("utf-8")
                return base64.b64decode(cached_str)
            return None
        except Exception as e:
            logger.error(f"[TTS] Failed to get cached audio: {e}")
            return None

    def _cache_audio(self, text_hash: str, audio_data: bytes):
        """
        Cache audio to Redis

        Args:
            text_hash: Hash of text + voice_id
            audio_data: Generated audio bytes
        """
        if not self.redis_client or not text_hash:
            return

        try:
            cache_key = f"tts:audio:{text_hash}"
            encoded = base64.b64encode(audio_data).decode("utf-8")
            self.redis_client.setex(cache_key, self.cache_ttl, encoded)
            logger.info(f"[TTS] Audio cached: {text_hash[:16]}..., size={len(audio_data)}")
        except Exception as e:
            logger.error(f"[TTS] Failed to cache audio: {e}")

    async def synthesize_speech(
        self,
        text: str,
        voice_id: Optional[str] = None,
        model_id: str = "eleven_v3",
        stability: float = 0.5,
        similarity_boost: float = 0.75,
        speed: float = 1.0,
    ) -> bytes:
        """
        Synthesize speech from text using ElevenLabs API

        Args:
            text: Text to convert to speech
            voice_id: Voice identifier (uses default if None)
            model_id: ElevenLabs model ID
            stability: Voice stability (0.0-1.0)
            similarity_boost: Clarity/similarity boost (0.0-1.0)
            speed: Speech speed multiplier (0.5-2.0)

        Returns:
            bytes: Audio data in MP3 format

        Raises:
            Exception: If API call fails or API key not configured
        """
        if not self.api_key:
            raise RuntimeError("ElevenLabs API key not configured")

        # Use default voice if not specified
        voice_id = voice_id or self.voice_id

        # Check cache first
        text_hash = self._get_text_hash(text, voice_id)
        cached_audio = self._get_cached_audio(text_hash)
        if cached_audio:
            return cached_audio

        try:
            started_at = time.perf_counter()
            logger.info(f"[TTS] Start synthesis: text_length={len(text)}, voice_id={voice_id}")

            # Prepare API request
            url = f"{ELEVENLABS_API_URL}/{voice_id}"
            headers = {
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key,
            }
            payload = {
                "text": text,
                "model_id": model_id,
                "voice_settings": {
                    "stability": stability,
                    "similarity_boost": similarity_boost,
                    "speed": speed,
                },
            }

            # Call ElevenLabs API
            response = await self.client.post(url, json=payload, headers=headers)
            response.raise_for_status()

            audio_data = response.content
            elapsed = time.perf_counter() - started_at
            logger.success(
                f"[TTS] Complete: audio_size={len(audio_data)} bytes, duration={elapsed:.2f}s"
            )

            # Cache the audio
            if text_hash:
                self._cache_audio(text_hash, audio_data)

            return audio_data

        except httpx.HTTPStatusError as e:
            logger.error(f"[TTS] ElevenLabs API error: {e.response.status_code} - {e.response.text}")
            if 400 <= e.response.status_code < 500:
                raise ValueError(f"TTS API rejected request: HTTP {e.response.status_code}") from e
            raise RuntimeError(f"TTS API unavailable: HTTP {e.response.status_code}") from e
        except httpx.TimeoutException as e:
            logger.error(f"[TTS] Request timeout: {e}")
            raise TimeoutError("TTS request timed out") from e
        except httpx.ConnectError as e:
            logger.error(f"[TTS] Connection error: {e}")
            raise ConnectionError("Cannot connect to ElevenLabs API") from e
        except Exception as e:
            logger.error(f"[TTS] Synthesis failed: {e}")
            raise

    async def health_check(self) -> bool:
        """Check TTS service availability (API key + upstream reachability)."""
        if not self.api_key:
            logger.warning("[TTS][health] API key missing or empty")
            return False

        try:
            response = await self.client.get(
                "https://api.elevenlabs.io/v1/user",
                headers={"xi-api-key": self.api_key},
                timeout=min(self.timeout, 10.0),
            )
            if response.status_code == 200:
                logger.info("[TTS][health] Upstream check passed via /v1/user")
                return True

            detail_text = ""
            detail_status = ""
            try:
                payload = response.json()
                detail = payload.get("detail", {}) if isinstance(payload, dict) else {}
                if isinstance(detail, dict):
                    detail_text = str(detail.get("message", ""))
                    detail_status = str(detail.get("status", ""))
                else:
                    detail_text = str(detail)
            except Exception:
                detail_text = (response.text or "")[:300]

            logger.warning(
                "[TTS][health] /v1/user returned HTTP {}, status={}, message={}",
                response.status_code,
                detail_status or "n/a",
                detail_text or "n/a",
            )

            # Some scoped API keys can synthesize speech but cannot read /v1/user
            # (missing user_read permission). Treat this as reachable upstream + usable key.
            if (
                response.status_code in (401, 403)
                and detail_status == "missing_permissions"
                and "user_read" in detail_text
            ):
                logger.warning(
                    "[TTS][health] Key is reachable but lacks user_read permission; "
                    "considered healthy for synthesis endpoints."
                )
                return True

            return False
        except httpx.TimeoutException as e:
            logger.error(f"[TTS][health] Timeout when contacting ElevenLabs: {e}")
            return False
        except httpx.ConnectError as e:
            logger.error(f"[TTS][health] Connection error when contacting ElevenLabs: {e}")
            return False
        except Exception as e:
            logger.error(f"[TTS][health] Unexpected health-check failure: {e}")
            return False

    async def close(self):
        """Close shared async HTTP client."""
        await self.client.aclose()


# Global TTS service instance
_tts_service: Optional[TtsService] = None
_tts_service_lock = threading.Lock()
_tts_init_kwargs: dict = {}


def get_tts_service() -> TtsService:
    """Get or create global TTS service instance"""
    global _tts_service
    if _tts_service is None:
        with _tts_service_lock:
            if _tts_service is None:
                _tts_service = TtsService(**_tts_init_kwargs)
                logger.info("[TTS] Lazy initialized on first request")
    else:
        logger.debug("[TTS] Reused singleton instance")
    return _tts_service


def initialize_tts_service(api_key: str = None, voice_id: str = None):
    """
    Initialize TTS service with custom configuration

    Args:
        api_key: ElevenLabs API key
        voice_id: Default voice ID
    """
    global _tts_init_kwargs
    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if voice_id:
        kwargs["voice_id"] = voice_id

    _tts_init_kwargs = kwargs
    logger.info("[TTS] Deferred initialization configured (lazy)")


async def close_tts_service():
    """Close global TTS service resources if initialized."""
    global _tts_service
    if _tts_service is not None:
        await _tts_service.close()
        _tts_service = None