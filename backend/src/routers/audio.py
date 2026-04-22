"""Audio router — unified STT/TTS endpoints."""

from fastapi import APIRouter, Depends, File, Form, HTTPException, Response, UploadFile
from pydantic import BaseModel, Field

from ..services.stt_service import SttService, get_stt_service
from ..services.tts_service import TtsService, get_tts_service

router = APIRouter(prefix="/v1/audio", tags=["audio"])
stt_router = APIRouter(prefix="/v1/stt", tags=["audio"])
tts_router = APIRouter(prefix="/v1/tts", tags=["audio"])


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize")
    voice_id: str | None = Field(default=None, description="Optional voice ID")


async def _transcribe(
    file: UploadFile,
    language: str,
    batch_size: int,
    stt_service: SttService,
):
    if batch_size <= 0:
        raise HTTPException(status_code=400, detail="batch_size must be > 0")

    audio_bytes = await file.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="Audio file is empty")

    try:
        result = await stt_service.transcribe_audio(
            audio_bytes=audio_bytes,
            filename=file.filename or "audio.wav",
            language=language,
            batch_size=batch_size,
        )
        return {
            "text": result.get("text", ""),
            "language": result.get("language", language),
            "duration": result.get("duration", 0.0),
            "segments": result.get("segments", []),
            "cached": result.get("cached", False),
        }
    except NotImplementedError as e:
        raise HTTPException(status_code=501, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except (TimeoutError, ConnectionError, RuntimeError) as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail="STT transcription failed") from e


async def _synthesize(body: TtsRequest, tts_service: TtsService):
    try:
        audio_data = await tts_service.synthesize_speech(
            text=body.text,
            voice_id=body.voice_id,
        )
        return Response(content=audio_data, media_type="audio/mpeg")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except (TimeoutError, ConnectionError, RuntimeError) as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail="TTS synthesis failed") from e


@router.post("/stt/transcribe", summary="Transcribe audio to text")
async def audio_transcribe(
    file: UploadFile = File(...),
    language: str = Form("vi"),
    batch_size: int = Form(16),
    stt_service: SttService = Depends(get_stt_service),
):
    return await _transcribe(file, language, batch_size, stt_service)


@stt_router.post("/transcribe", summary="Transcribe audio to text")
async def stt_transcribe_legacy(
    file: UploadFile = File(...),
    language: str = Form("vi"),
    batch_size: int = Form(16),
    stt_service: SttService = Depends(get_stt_service),
):
    return await _transcribe(file, language, batch_size, stt_service)


@router.get("/stt/health", summary="STT health check")
@stt_router.get("/health", summary="STT health check")
async def stt_health(stt_service: SttService = Depends(get_stt_service)):
    ok = await stt_service.health_check()
    if not ok:
        raise HTTPException(
            status_code=503,
            detail="STT service unavailable (GPU service disabled or unreachable)",
        )
    return {"status": "ok"}


@router.post("/tts/synthesize", summary="Synthesize speech", response_class=Response)
async def audio_synthesize(
    body: TtsRequest,
    tts_service: TtsService = Depends(get_tts_service),
):
    return await _synthesize(body, tts_service)


@tts_router.post("/synthesize", summary="Synthesize speech", response_class=Response)
async def tts_synthesize_legacy(
    body: TtsRequest,
    tts_service: TtsService = Depends(get_tts_service),
):
    return await _synthesize(body, tts_service)


@router.get("/tts/health", summary="TTS health check")
@tts_router.get("/health", summary="TTS health check")
async def tts_health(tts_service: TtsService = Depends(get_tts_service)):
    ok = await tts_service.health_check()
    if not ok:
        raise HTTPException(
            status_code=503,
            detail="TTS service unavailable (missing API key or upstream unreachable)",
        )
    return {"status": "ok"}


