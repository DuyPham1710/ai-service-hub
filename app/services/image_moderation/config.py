# Ngưỡng tin cậy — nếu score của label vi phạm >= ngưỡng này -> ảnh bị chặn
CLIP_THRESHOLD = 0.5

# Các danh mục vi phạm — có thể tùy chỉnh thêm/bớt
UNSAFE_LABELS = [
    "nudity and sexual content",
    "pornography and explicit content",
    "violence and fighting",
    "blood and gore",
    "drugs and drug use",
    "weapons and guns",
   # "self-harm and suicide",
    "child abuse",
]

# Label an toàn để so sánh
SAFE_LABEL = "a safe and normal photo"

# Tất cả labels (an toàn + vi phạm)
ALL_LABELS = [SAFE_LABEL] + UNSAFE_LABELS

# Map label tiếng Anh sang tiếng Việt (để hiển thị cho user)
LABEL_VI = {
    "nudity and sexual content": "nội dung khiêu dâm",
    "pornography and explicit content": "nội dung đồi trụy",
    "violence and fighting": "bạo lực",
    "blood and gore": "máu me, kinh dị",
    "drugs and drug use": "ma túy, chất cấm",
    "weapons and guns": "vũ khí",
   # "self-harm and suicide": "tự gây thương tích",
    "child abuse": "bạo hành trẻ em",
}

# Loại file ảnh được hỗ trợ
ALLOWED_IMAGE_TYPES = [
    "image/jpeg",
    "image/png",
    "image/gif",
    "image/webp",
    "image/jpg",
]
