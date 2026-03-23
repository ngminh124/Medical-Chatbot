"""TTS router."""

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel, Field

from ..services.tts import TtsService, get_tts_service

router = APIRouter(prefix="/v1/tts", tags=["tts"])


class TtsRequest(BaseModel):
    text: str = Field(..., min_length=1, description="Text to synthesize")
    voice_id: str | None = Field(default=None, description="Optional voice ID")


@router.post("/synthesize", summary="Synthesize speech", response_class=Response)
async def synthesize_speech(
    body: TtsRequest,
    tts_service: TtsService = Depends(get_tts_service),
):
    try:
        audio_data = await tts_service.synthesize_speech(
            text=body.text,
            voice_id=body.voice_id,
        )
        return Response(content=audio_data, media_type="audio/mpeg")
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e
    except TimeoutError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=str(e)) from e
    except Exception as e:
        raise HTTPException(status_code=500, detail="TTS synthesis failed") from e


@router.get("/health", summary="TTS health check")
async def tts_health(tts_service: TtsService = Depends(get_tts_service)):
    ok = await tts_service.health_check()
    if not ok:
        raise HTTPException(
            status_code=503,
            detail="TTS service unavailable (missing API key or upstream unreachable)",
        )
    return {"status": "ok"}
