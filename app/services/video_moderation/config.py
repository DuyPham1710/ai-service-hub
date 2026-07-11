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

# === CẤU HÌNH REGION-SPECIFIC BLUR (GradCAM + OpenCV) ===

# Ngưỡng attention GradCAM để xác định vùng vi phạm
# 0.4 = chỉ lấy vùng có activation >= 40% max → giảm false positive
GRADCAM_THRESHOLD = 0.4

# Padding thêm xung quanh bounding box (tỷ lệ so với kích thước bbox)
# 0.2 = thêm 20% mỗi cạnh để đảm bảo phủ hết vùng vi phạm
BBOX_PADDING_RATIO = 0.2

# Kích thước kernel Gaussian blur (phải lẻ, càng lớn càng mờ)
BLUR_KERNEL_SIZE = 99

# Opacity lớp overlay đen phủ lên vùng blur (0.0 = trong suốt, 1.0 = đen hoàn toàn)
BLUR_OVERLAY_OPACITY = 0.5
