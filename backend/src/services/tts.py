"""TTS service public module."""

from .tts_service import (TtsService, close_tts_service, get_tts_service,
                          initialize_tts_service)

__all__ = [
    "TtsService",
    "get_tts_service",
    "initialize_tts_service",
    "close_tts_service",
]
