"""Audio router — STT / TTS endpoints (placeholder)."""

from fastapi import APIRouter

router = APIRouter(prefix="/v1/audio", tags=["audio"])


@router.post("/stt")
def speech_to_text():
    """Speech-to-text endpoint (placeholder)."""
    return {"text": "", "status": "not_configured"}


@router.post("/tts")
def text_to_speech():
    """Text-to-speech endpoint (placeholder)."""
    return {"audio_url": None, "status": "not_configured"}
