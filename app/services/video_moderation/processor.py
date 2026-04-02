"""
Video Processor: Dùng FFmpeg để blur các đoạn vi phạm trong video
"""
import logging
import os
import subprocess
import uuid

from .config import TEMP_DIR

logger = logging.getLogger("ai-service-hub")


def blur_segments(video_path: str, segments: list[dict]) -> str:
    """
    Áp dụng Gaussian blur lên các đoạn vi phạm trong video bằng FFmpeg.

    Args:
        video_path: Đường dẫn tới video gốc
        segments: Danh sách đoạn vi phạm [{"start": float, "end": float}]

    Returns:
        Đường dẫn tới video đã xử lý (blur)
    """
    if not segments:
        return video_path

    output_filename = f"blurred_{uuid.uuid4().hex}.mp4"
    output_path = os.path.join(TEMP_DIR, output_filename)

    # Xây dựng filter_complex cho FFmpeg
    # Mỗi đoạn vi phạm sẽ được áp dụng boxblur mạnh
    filter_parts = []
    for i, seg in enumerate(segments):
        start = seg["start"]
        end = seg["end"]
        # enable='between(t,start,end)' chỉ kích hoạt blur trong khoảng thời gian
        # luma_radius càng to thì ảnh càng mờ
        filter_parts.append(
            f"boxblur=luma_radius=40:luma_power=3:enable='between(t,{start},{end})'"
        )

    # Nối tất cả filter lại bằng dấu phẩy
    filter_chain = ",".join(filter_parts)

    # Lệnh FFmpeg
    cmd = [
        "ffmpeg",
        "-i", video_path,       # Truyền video gốc vào
        "-vf", filter_chain,    # Áp dụng bộ lọc che mờ vừa tạo 
        "-c:a", "copy",       # Giữ nguyên audio
        "-c:v", "libx264",    # Encode lại video bằng H.264
        "-preset", "fast",     # Tốc độ encode nhanh
        "-crf", "23",          # Chất lượng video tốt
        "-y",                  # Ghi đè file output nếu đã tồn tại
        output_path,
    ]

    logger.info(f"Đang blur {len(segments)} đoạn vi phạm trong video...")
    logger.debug(f"FFmpeg command: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 120 seconds
        )

        if result.returncode != 0:
            logger.error(f"FFmpeg stderr: {result.stderr}")
            raise RuntimeError(f"FFmpeg failed with return code {result.returncode}")

        logger.info(f"Đã blur thành công → {output_path}")
        return output_path

    except subprocess.TimeoutExpired:
        logger.error("FFmpeg timeout sau 120 giây")
        # Dọn dẹp file output nếu bị timeout
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RuntimeError("Video processing timeout")


def cleanup_temp_file(file_path: str) -> None:
    """Xóa file tạm sau khi đã sử dụng xong"""
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Đã xóa file tạm: {file_path}")
    except Exception as e:
        logger.warning(f"Không thể xóa file tạm {file_path}: {e}")
