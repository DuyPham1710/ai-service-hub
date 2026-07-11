"""
Video Processor V2: Blur VÙNG VI PHẠM cụ thể thay vì toàn frame.

Flow:
1. Đọc video gốc bằng OpenCV VideoCapture
2. Với mỗi frame → kiểm tra timestamp có nằm trong violation segment không
3. Nếu có → lấy bounding boxes từ frame_regions map → Gaussian blur chỉ vùng đó
4. Ghi frame (đã blur hoặc nguyên) vào video output bằng OpenCV VideoWriter
5. Dùng FFmpeg để merge audio từ video gốc vào video output (OpenCV không xử lý audio)
"""
import logging
import os
import subprocess
import uuid

import cv2
import numpy as np

from .config import TEMP_DIR, BLUR_KERNEL_SIZE, BLUR_OVERLAY_OPACITY, FRAME_INTERVAL

logger = logging.getLogger("ai-service-hub")


def blur_segments(
    video_path: str,
    segments: list[dict],
    frame_regions: dict | None = None,
) -> str:
    """
    Blur các vùng vi phạm trong video.

    Args:
        video_path: Đường dẫn tới video gốc
        segments: Danh sách đoạn vi phạm [{"start": float, "end": float}]
        frame_regions: Map timestamp → list[bounding_boxes] từ GradCAM
                       Nếu None → fallback blur toàn frame (behavior cũ)

    Returns:
        Đường dẫn tới video đã xử lý
    """
    if not segments:
        return video_path

    output_id = uuid.uuid4().hex

    # Bước 1: Tạo video chỉ có hình (không audio) với OpenCV
    video_no_audio = os.path.join(TEMP_DIR, f"noaudio_{output_id}.mp4")
    _process_video_frames(video_path, video_no_audio, segments, frame_regions)

    # Bước 2: Merge audio từ video gốc vào video đã blur
    output_path = os.path.join(TEMP_DIR, f"blurred_{output_id}.mp4")
    _merge_audio(video_path, video_no_audio, output_path)

    # Dọn dẹp file trung gian
    cleanup_temp_file(video_no_audio)

    logger.info(f"Đã blur region-specific thành công → {output_path}")
    return output_path


def _process_video_frames(
    input_path: str,
    output_path: str,
    segments: list[dict],
    frame_regions: dict | None,
) -> None:
    """
    Đọc từng frame video, blur vùng vi phạm, ghi ra video mới.
    """
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise RuntimeError(f"Không thể mở video: {input_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Sử dụng mp4v codec (tương thích rộng)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Không thể tạo video writer: {output_path}")

    logger.info(
        f"Processing {total_frames} frames ({width}x{height} @ {fps:.0f}fps), "
        f"{len(segments)} violation segment(s)"
    )

    frame_idx = 0
    blurred_frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = frame_idx / fps

        # Kiểm tra frame có nằm trong violation segment không
        if _is_in_violation_segment(timestamp, segments):
            # Lấy bounding boxes cho frame này
            regions = _get_regions_for_timestamp(timestamp, frame_regions, fps, width, height)

            # Blur từng vùng vi phạm
            frame = _blur_regions(frame, regions)
            blurred_frame_count += 1

        writer.write(frame)
        frame_idx += 1

    cap.release()
    writer.release()

    logger.info(f"Đã xử lý {frame_idx} frames, blur {blurred_frame_count} frames")


def _is_in_violation_segment(timestamp: float, segments: list[dict]) -> bool:
    """Kiểm tra timestamp có nằm trong bất kỳ violation segment nào không."""
    for seg in segments:
        if seg["start"] <= timestamp <= seg["end"]:
            return True
    return False


def _get_regions_for_timestamp(
    timestamp: float,
    frame_regions: dict | None,
    fps: float,
    frame_width: int,
    frame_height: int,
) -> list[dict]:
    """
    Lấy bounding boxes cho frame tại timestamp.
    
    Vì GradCAM chỉ chạy trên các frame được trích xuất (mỗi FRAME_INTERVAL giây),
    ta cần tìm frame đã phân tích gần nhất để lấy regions.
    
    Fallback: nếu không có frame_regions → blur toàn frame.
    """
    if not frame_regions:
        # Fallback: blur toàn frame
        return [{"x": 0, "y": 0, "w": frame_width, "h": frame_height}]

    # Tìm analyzed frame gần nhất (làm tròn về bội số FRAME_INTERVAL)
    nearest_ts = round(round(timestamp / FRAME_INTERVAL) * FRAME_INTERVAL, 2)

    # Thử timestamp chính xác trước
    key = str(nearest_ts)
    if key in frame_regions:
        return frame_regions[key]

    # Tìm trong phạm vi ±FRAME_INTERVAL
    for offset in [FRAME_INTERVAL, -FRAME_INTERVAL, 2 * FRAME_INTERVAL, -2 * FRAME_INTERVAL]:
        nearby_key = str(round(nearest_ts + offset, 2))
        if nearby_key in frame_regions:
            return frame_regions[nearby_key]

    # Không tìm thấy → fallback blur toàn frame
    return [{"x": 0, "y": 0, "w": frame_width, "h": frame_height}]


def _blur_regions(frame: np.ndarray, regions: list[dict]) -> np.ndarray:
    """
    Áp dụng Gaussian blur lên các vùng cụ thể trong frame.

    Với mỗi region:
    1. Cắt vùng ra
    2. Apply Gaussian blur mạnh
    3. Phủ overlay đen mờ (opacity)
    4. Dán vùng đã blur lại vào frame
    """
    result = frame.copy()
    h_frame, w_frame = frame.shape[:2]
    kernel = (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE)

    for region in regions:
        x = max(0, region["x"])
        y = max(0, region["y"])
        w = min(region["w"], w_frame - x)
        h = min(region["h"], h_frame - y)

        if w <= 0 or h <= 0:
            continue

        # Cắt vùng cần blur
        roi = result[y:y+h, x:x+w]

        # Apply Gaussian blur mạnh
        blurred_roi = cv2.GaussianBlur(roi, kernel, 0)

        # Phủ overlay đen mờ lên vùng đã blur
        if BLUR_OVERLAY_OPACITY > 0:
            overlay = np.zeros_like(blurred_roi)
            blurred_roi = cv2.addWeighted(
                blurred_roi, 1 - BLUR_OVERLAY_OPACITY,
                overlay, BLUR_OVERLAY_OPACITY,
                0,
            )

        # Dán vùng đã blur vào frame
        result[y:y+h, x:x+w] = blurred_roi

    return result


def _merge_audio(
    original_video: str,
    video_no_audio: str,
    output_path: str,
) -> None:
    """
    Merge audio từ video gốc vào video đã blur (OpenCV không xử lý audio).
    Sử dụng FFmpeg để copy audio stream.
    """
    cmd = [
        "ffmpeg",
        "-i", video_no_audio,      # Video đã blur (không audio)
        "-i", original_video,       # Video gốc (có audio)
        "-c:v", "copy",             # Giữ nguyên video stream
        "-c:a", "aac",              # Encode audio thành AAC
        "-map", "0:v:0",            # Lấy video từ input 0 (video đã blur)
        "-map", "1:a:0?",           # Lấy audio từ input 1 (video gốc), ? = optional
        "-shortest",                # Dừng khi stream ngắn nhất kết thúc
        "-y",                       # Ghi đè
        output_path,
    ]

    logger.debug(f"FFmpeg merge audio: {' '.join(cmd)}")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )

        if result.returncode != 0:
            logger.warning(f"FFmpeg merge audio stderr: {result.stderr}")
            # Nếu merge audio thất bại (vd: video gốc không có audio)
            # → dùng video không audio
            logger.info("Fallback: sử dụng video không audio")
            if os.path.exists(video_no_audio):
                os.replace(video_no_audio, output_path)

    except subprocess.TimeoutExpired:
        logger.error("FFmpeg merge audio timeout")
        # Fallback: dùng video không audio
        if os.path.exists(video_no_audio):
            os.replace(video_no_audio, output_path)


def cleanup_temp_file(file_path: str) -> None:
    """Xóa file tạm sau khi đã sử dụng xong"""
    try:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)
            logger.debug(f"Đã xóa file tạm: {file_path}")
    except Exception as e:
        logger.warning(f"Không thể xóa file tạm {file_path}: {e}")
