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

# Các nhãn an toàn — phủ rộng các thể loại nội dung bình thường
# để Softmax phân bổ xác suất đúng hơn, giảm false positive
SAFE_LABELS = [
    "a safe and normal photo",
    "people in everyday life",
    "nature and landscape scenery",
    "food and beverages",
    "objects and products",
    "graphics, text, and digital design",
    "animals and pets",
    "architecture and buildings",
    "sports and outdoor activities",
    "art and creative content",
]

# Tất cả labels (an toàn + vi phạm)
ALL_LABELS = SAFE_LABELS + UNSAFE_LABELS

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
