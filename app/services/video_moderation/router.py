"""
Video Moderation Router: API endpoints cho kiểm duyệt video
"""
import io
import os
import logging
import shutil
import uuid

from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import FileResponse

from .config import ALLOWED_VIDEO_TYPES, TEMP_DIR
from .analyzer import analyze_video, get_video_duration
from .processor import blur_segments, cleanup_temp_file

logger = logging.getLogger("ai-service-hub")

router = APIRouter()


@router.get("/health")
async def health_check():
    return {"status": "ok", "service": "video_moderation"}


@router.post("/check")
async def check_video(file: UploadFile = File(...)):
    """
    Kiểm tra video có vi phạm tiêu chuẩn cộng đồng không.

    Quy trình:
    1. Lưu video tạm vào disk
    2. Trích xuất frame theo khoảng cách thời gian
    3. Phân loại từng frame bằng CLIP
    4. Nếu có vi phạm → blur các đoạn vi phạm bằng FFmpeg
    5. Trả về kết quả + video đã blur
    """
    if file.content_type not in ALLOWED_VIDEO_TYPES:
        raise HTTPException(
            status_code=400,
            detail=f"Loại file không được hỗ trợ: {file.content_type}. "
                   f"Chỉ hỗ trợ: {', '.join(ALLOWED_VIDEO_TYPES)}",
        )

    # Lưu video tạm vào disk để OpenCV và FFmpeg xử lý
    temp_input = os.path.join(TEMP_DIR, f"input_{uuid.uuid4().hex}.mp4")
    blurred_path = None

    try:
        # Ghi file upload ra disk
        with open(temp_input, "wb") as f:
            shutil.copyfileobj(file.file, f)

        logger.info(f"Đang kiểm duyệt video '{file.filename}' ({file.content_type})")

        # Phân tích video
        analysis = analyze_video(temp_input)

        if analysis["is_safe"]:
            # Video an toàn → trả kết quả, không cần blur
            return {
                "is_safe": True,
                "duration": analysis["duration"],
                "total_frames_analyzed": analysis["total_frames_analyzed"],
                "violation_segments": [],
                "has_blurred_video": False,
            }

        # Tính tổng thời gian vi phạm xem có vượt quá 90% thời lượng video không
        total_violation_duration = sum([seg["end"] - seg["start"] for seg in analysis["violation_segments"]])
        violation_ratio = total_violation_duration / analysis["duration"] if analysis["duration"] > 0 else 0

        if violation_ratio > 0.90:
            logger.warning(f"Video {file.filename} vi phạm {violation_ratio:.1%} nội dung. Chặn lập tức!")
            return {
                "is_safe": False,
                "duration": analysis["duration"],
                "total_frames_analyzed": analysis["total_frames_analyzed"],
                "violation_segments": analysis["violation_segments"],
                "has_blurred_video": False,
                "block_completely": True,
                "violation_ratio": violation_ratio
            }

        # Có vi phạm một phần (< 90%) → tiến hành blur các đoạn đó
        logger.info(
            f"Video vi phạm {violation_ratio:.1%}! Tìm thấy {len(analysis['violation_segments'])} đoạn. "
            f"Đang tiến hành blur..."
        )

        blurred_path = blur_segments(temp_input, analysis["violation_segments"])

        # Trả về video đã blur dưới dạng file response
        return FileResponse(
            path=blurred_path,
            media_type="video/mp4",
            filename=f"moderated_{file.filename}",
            headers={
                "X-Is-Safe": "false",
                "X-Duration": str(analysis["duration"]),
                "X-Violation-Count": str(len(analysis["violation_segments"])),
                "X-Frames-Analyzed": str(analysis["total_frames_analyzed"]),
            },
        )

    except ValueError as e:
        logger.error(f"Lỗi validation video: {e}")
        raise HTTPException(status_code=400, detail=str(e))

    except Exception as e:
        logger.error(f"Lỗi khi xử lý video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý video: {str(e)}")

    finally:
        # Luôn dọn dẹp file tạm input
        cleanup_temp_file(temp_input)
        # Lưu ý: blurred_path sẽ bị xóa bởi FastAPI sau khi FileResponse gửi xong
        # nhưng nếu có lỗi xảy ra trước đó, ta cần dọn dẹp thủ công
