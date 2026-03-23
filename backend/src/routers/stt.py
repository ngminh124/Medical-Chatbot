"""STT router."""

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from ..services.stt import SttService, get_stt_service

router = APIRouter(prefix="/v1/stt", tags=["stt"])


@router.post("/transcribe", summary="Transcribe audio to text")
async def transcribe_audio(
    file: UploadFile = File(...),
    language: str = Form("vi"),
    batch_size: int = Form(16),
    stt_service: SttService = Depends(get_stt_service),
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
    except TimeoutError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail="STT transcription failed") from e


@router.get("/health", summary="STT health check")
async def stt_health(stt_service: SttService = Depends(get_stt_service)):
    ok = await stt_service.health_check()
    if not ok:
        raise HTTPException(
            status_code=503,
            detail="STT service unavailable (GPU service disabled or unreachable)",
        )
    return {"status": "ok"}
