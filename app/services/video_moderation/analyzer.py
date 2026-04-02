"""
Video Analyzer: Trích xuất frame và phân loại bằng CLIP model
Tái sử dụng CLIP model từ image_moderation
"""
import logging
import cv2
import torch
from PIL import Image

from ..image_moderation.model import clip_model, clip_processor
from ..image_moderation.config import ALL_LABELS, UNSAFE_LABELS, LABEL_VI
from ..image_moderation.classifier import classify_image
from .config import FRAME_INTERVAL, VIDEO_CLIP_THRESHOLD, MAX_VIDEO_DURATION, MIN_CONSECUTIVE_FRAMES, BUFFER_SECONDS, IGNORED_LABELS

logger = logging.getLogger("ai-service-hub")


"""Lấy thời lượng video (giây)"""
def get_video_duration(video_path: str) -> float:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Không thể mở video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    if fps <= 0:
        raise ValueError("Không thể đọc FPS của video")

    return frame_count / fps


def extract_frames(video_path: str, interval: float = FRAME_INTERVAL) -> list[dict]:
    """
    Trích xuất frame từ video theo khoảng cách thời gian.

    Args:
        video_path: Đường dẫn tới file video
        interval: Khoảng cách giữa các frame (giây)

    Returns:
        List of dict: [{ "timestamp": float, "image": PIL.Image }]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Không thể mở video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps

    logger.info(f"Video: {duration:.1f}s, {fps:.0f} FPS, {total_frames} frames")

    frames = []
    frame_interval_count = int(fps * interval)  # Số frame giữa mỗi lần trích xuất

    current_frame = 0
    while current_frame < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
        ret, frame = cap.read()

        if not ret:
            break

        # Chuyển đổi BGR (OpenCV) sang RGB (PIL)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)

        timestamp = current_frame / fps
        frames.append({
            "timestamp": round(timestamp, 2),
            "image": pil_image,
        })

        current_frame += frame_interval_count

    cap.release()
    logger.info(f"Đã trích xuất {len(frames)} frames từ video")
    return frames
    

def analyze_video(video_path: str) -> dict:
    """
    Phân tích toàn bộ video: trích xuất frame, phân loại, và xác định đoạn vi phạm.

    Returns:
        {
            "duration": float,
            "total_frames_analyzed": int,
            "is_safe": bool,
            "violation_segments": [{"start": float, "end": float, "categories": [...]}],
            "frame_results": [{"timestamp": float, "is_safe": bool, "violations": [...]}]
        }
    """
    duration = get_video_duration(video_path)

    if duration > MAX_VIDEO_DURATION:
        raise ValueError(
            f"Video quá dài ({duration:.0f}s). Giới hạn tối đa: {MAX_VIDEO_DURATION}s"
        )

    # Trích xuất frames
    frames = extract_frames(video_path)

    # Phân loại từng frame
    frame_results = []
    for frame_data in frames:
        scores = classify_image(frame_data["image"])

        violations = []
        for label in UNSAFE_LABELS:
            # Bỏ qua các label không muốn detect cho video
            if label in IGNORED_LABELS:
                continue
                
            score = scores.get(label, 0)
            if score >= VIDEO_CLIP_THRESHOLD:
                violations.append({
                    "category": label,
                    "category_vi": LABEL_VI.get(label, label),
                    "confidence": score,
                })

        frame_results.append({
            "timestamp": frame_data["timestamp"],
            "is_safe": len(violations) == 0,
            "violations": violations,
            "scores": scores,
        })

    # Gom các frame vi phạm liên tiếp thành đoạn thời gian
    violation_segments = _merge_violation_segments(frame_results, duration)

    is_safe = len(violation_segments) == 0

    logger.info(
        f"Kết quả phân tích video: is_safe={is_safe}, "
        f"segments={len(violation_segments)}, "
        f"frames_analyzed={len(frame_results)}"
    )

    return {
        "duration": round(duration, 2),
        "total_frames_analyzed": len(frame_results),
        "is_safe": is_safe,
        "violation_segments": violation_segments,
        "frame_results": [
            {k: v for k, v in fr.items() if k != "scores"}
            for fr in frame_results
        ],
    }


def _merge_violation_segments(frame_results: list[dict], video_duration: float) -> list[dict]:
    """
    Gom các frame vi phạm liên tiếp thành các đoạn thời gian.
    Chỉ giữ lại đoạn có >= MIN_CONSECUTIVE_FRAMES frame vi phạm liên tiếp.
    Ví dụ: frame 5s, 6s, 7s vi phạm → đoạn 5.0 - 8.0
    1 frame đơn lẻ vi phạm → bỏ qua (nhiễu)
    """
    if not frame_results:
        return []

    # Bước 1: Gom tất cả các đoạn frame vi phạm liên tiếp
    raw_segments = []
    current_segment = None
    consecutive_count = 0

    for fr in frame_results:
        if not fr["is_safe"]:
            # Frame vi phạm
            all_categories = set()
            for v in fr["violations"]:
                all_categories.add(v["category_vi"])

            if current_segment is None:
                # Bắt đầu đoạn mới
                current_segment = {
                    "start": fr["timestamp"],
                    "end": fr["timestamp"] + FRAME_INTERVAL,
                    "categories": all_categories,
                }
                consecutive_count = 1
            else:
                # Mở rộng đoạn hiện tại
                current_segment["end"] = fr["timestamp"] + FRAME_INTERVAL
                current_segment["categories"] |= all_categories
                consecutive_count += 1
        else:
            # Frame an toàn → đóng đoạn vi phạm nếu có
            if current_segment is not None:
                current_segment["categories"] = list(current_segment["categories"])
                current_segment["frame_count"] = consecutive_count
                raw_segments.append(current_segment)
                current_segment = None
                consecutive_count = 0

    # Đóng đoạn cuối cùng nếu video kết thúc bằng frame vi phạm
    if current_segment is not None:
        current_segment["end"] = min(current_segment["end"], video_duration)
        current_segment["categories"] = list(current_segment["categories"])
        current_segment["frame_count"] = consecutive_count
        raw_segments.append(current_segment)

    # Bước 2: Lọc bỏ các đoạn có quá ít frame (nhiễu / false positive)
    filtered_segments = []
    for seg in raw_segments:
        if seg["frame_count"] >= MIN_CONSECUTIVE_FRAMES:
            del seg["frame_count"]
            filtered_segments.append(seg)
        else:
            logger.info(
                f"Bỏ qua đoạn {seg['start']}s-{seg['end']}s "
                f"(chỉ có {seg['frame_count']} frame → nhiễu)"
            )

    # Bước 3: Thêm BUFFER trước/sau mỗi đoạn vi phạm để blur phủ sớm hơn
    buffered_segments = []
    for seg in filtered_segments:
        buffered_segments.append({
            "start": max(0, seg["start"] - BUFFER_SECONDS),
            "end": min(video_duration, seg["end"] + BUFFER_SECONDS),
            "categories": seg["categories"],
        })

    # Bước 4: Gộp các đoạn bị chồng lấn sau khi thêm buffer
    if not buffered_segments:
        return []

    merged = [buffered_segments[0]]
    for seg in buffered_segments[1:]:
        prev = merged[-1]
        if seg["start"] <= prev["end"]:
            # Hai đoạn chồng nhau → gộp lại
            prev["end"] = max(prev["end"], seg["end"])
            prev["categories"] = list(set(prev["categories"]) | set(seg["categories"])) # union 
        else:
            merged.append(seg)

    return merged
