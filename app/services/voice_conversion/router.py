import os
import logging
import tempfile

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse

from .converter import convert_voice, get_available_presets
from .config import ALLOWED_AUDIO_TYPES

logger = logging.getLogger("ai-service-hub")

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "voice_conversion"}


@router.get("/presets")
async def list_presets():
    """Trả về danh sách các preset giọng có sẵn."""
    presets = get_available_presets()
    return {
        "presets": presets,
        "total": len(presets),
        "available": sum(1 for p in presets if p["available"]),
    }


@router.post("/convert")
async def convert_voice_endpoint(
    audio: UploadFile = File(..., description="File audio cần chuyển giọng"),
    voice_preset: str = Form(..., description="Key của preset giọng (vd: con_gai, tram, robot,...)"),
):
    """
    Chuyển đổi giọng nói trong file audio theo preset đã chọn.

    - **audio**: File audio gốc (m4a, wav, mp3,...)
    - **voice_preset**: Key preset (con_gai, tram, em_be, nguoi_ngoai_hanh_tinh, robot, ac_quy)

    Trả về file audio đã chuyển giọng (WAV).
    """
    # Validate content type
    if audio.content_type not in ALLOWED_AUDIO_TYPES:
        logger.warning(
            f"Audio type không hỗ trợ: {audio.content_type}, "
            f"nhưng vẫn thử xử lý..."
        )

    # Lưu file upload vào temp
    temp_input = None
    output_path = None

    try:
        # Lưu file upload
        suffix = os.path.splitext(audio.filename or "audio.m4a")[1] or ".m4a"
        temp_input = tempfile.NamedTemporaryFile(
            delete=False, suffix=suffix, prefix="vc_input_"
        )
        contents = await audio.read()
        temp_input.write(contents)
        temp_input.close()

        logger.info(
            f"📥 Nhận audio: {audio.filename} ({len(contents)} bytes), "
            f"preset: {voice_preset}"
        )

        # Convert voice
        output_path = convert_voice(
            input_path=temp_input.name,
            preset_key=voice_preset,
        )

        # Trả file kết quả
        return FileResponse(
            path=output_path,
            media_type="audio/wav",
            filename=f"voice_{voice_preset}.wav",
            background=None,  # Không xóa file ngay, sẽ được cleanup sau
        )

    except ValueError as e:
        logger.error(f"Validation error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except RuntimeError as e:
        logger.error(f"Conversion error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Lỗi không xác định khi chuyển giọng: {str(e)}",
        )

    finally:
        # Cleanup input temp file
        if temp_input and os.path.exists(temp_input.name):
            try:
                os.unlink(temp_input.name)
            except Exception:
                pass
