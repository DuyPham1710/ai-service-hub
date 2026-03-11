import io
import logging

from fastapi import APIRouter, UploadFile, File, HTTPException
from PIL import Image

from .classifier import classify_image, check_violations
from .config import ALLOWED_IMAGE_TYPES, CLIP_THRESHOLD

logger = logging.getLogger("ai-service-hub")

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "ok", "model": "clip_zero_shot"}


@router.post("/check")
async def check_image(file: UploadFile = File(...)):
    """
    Kiểm tra hình ảnh có vi phạm tiêu chuẩn cộng đồng không.

    Sử dụng CLIP zero-shot để phát hiện:
    khiêu dâm, bạo lực, máu me, ma túy, vũ khí,...
    """
    if file.content_type not in ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Loại file không được hỗ trợ: {file.content_type}. Chỉ hỗ trợ: {', '.join(ALLOWED_IMAGE_TYPES)}",
        )

    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Phân loại và kiểm tra vi phạm
        scores = classify_image(image)
        violations = check_violations(scores)

        # Kết quả
        is_safe = len(violations) == 0

        logger.info(
            f"Kiểm tra ảnh '{file.filename}': is_safe={is_safe}, "
            f"violations={[v['category'] for v in violations]}"
        )

        return {
            "is_safe": is_safe,
            "violated_categories": violations,
            "categories": scores,
            "threshold": CLIP_THRESHOLD,
        }

    except Exception as e:
        logger.error(f"Lỗi khi xử lý ảnh: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý ảnh: {str(e)}")
