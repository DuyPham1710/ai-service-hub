# Khoảng cách giữa các frame được trích xuất (giây)
# 0.5 = trích xuất 2 frame mỗi giây → phát hiện nhanh hơn, ít bỏ sót
FRAME_INTERVAL = 0.5

# Ngưỡng phát hiện vi phạm cho video
# 0.70 = cân bằng giữa phát hiện đúng và tránh nhầm
VIDEO_CLIP_THRESHOLD = 0.65

# Số frame vi phạm liên tiếp tối thiểu để tính là đoạn vi phạm
# 2 = cần ít nhất 2 frame vi phạm liên tiếp (tránh nhầm lẫn)
MIN_CONSECUTIVE_FRAMES = 2

# Thời gian đệm (giây) thêm vào TRƯỚC và SAU mỗi đoạn vi phạm
# Giúp blur phủ toàn bộ cảnh vi phạm, không bị "trễ" khi chuyển cảnh
BUFFER_SECONDS = 1.5

# Thời lượng video tối đa cho phép xử lý (giây)
MAX_VIDEO_DURATION = 180    # 3 phút

# Các label bị bỏ qua khi kiểm duyệt video (không tính là vi phạm)
IGNORED_LABELS = [
    "violence and fighting",
    "weapons and guns",
]

# Loại file video được hỗ trợ
ALLOWED_VIDEO_TYPES = [
    "video/mp4",
    "video/quicktime",   # .mov
    "video/x-msvideo",   # .avi
    "video/webm",
    "video/x-matroska",  # .mkv
    "video/x-flv",       # .flv
    "video/x-ms-wmv",    # .wmv
]

# Thư mục tạm để lưu video đã xử lý
import os
TEMP_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "temp_videos")
os.makedirs(TEMP_DIR, exist_ok=True)

