# AI Service Hub

Trung tâm các dịch vụ AI cho dự án mạng xã hội: kiểm duyệt hình ảnh, văn bản, nhận diện khuôn mặt,...

## Model sử dụng

| Service | Model | Mục đích |
|---------|-------|----------|
| Image Moderation | `openai/clip-vit-base-patch32` | Phát hiện khiêu dâm, bạo lực, máu me, ma túy, vũ khí,... |
| Video Moderation | Khung hình tách từ OpenCV + `CLIP` | Phát hiệm vi phạm theo dòng thời gian, làm mờ bằng FFmpeg |

> **Lưu ý:** Lần đầu chạy sẽ tự động tải model từ HuggingFace (~600MB) và lưu vào Docker volume hoặc thư mục `models/`. Các lần sau sẽ load từ local, không cần internet.

## Cài đặt và Chạy Server

Dự án hiện đã hỗ trợ chạy toàn bộ trong Docker (khuyên dùng) để đồng bộ môi trường (đặc biệt là cài đặt sẵn FFmpeg, OpenCV).

### Cách 1: Sử dụng Docker (Khuyên dùng)

Yêu cầu đã cài đặt Docker Desktop hoặc Docker Compose.

```bash
# Build và chạy ngầm server port 8000
docker-compose up -d --build

# Tắt server
docker-compose down
```

Lưu ý:
- Chạy qua Docker có hỗ trợ Hot-Reload (sửa code `.py` thì server lưu file tự nhận).
- Folder `models` được map thẳng ra ngoài để tránh phải download nhiều lần.

### Cách 2: Chạy Local trực tiếp không dùng Docker

Yêu cầu: Python >= 3.10 và đã cài đặt FFmpeg trên máy tính (nếu dùng tính năng video).

```bash
# 1. Tạo và kích hoạt virtual environment
python -m venv venv
.\venv\Scripts\activate      # Windows
# source venv/bin/activate   # Linux/Mac

# 2. Cài đặt dependencies
pip install -r requirements.txt

# 3. Chạy server
python main.py
```

Server sẽ chạy tại: **http://localhost:8000**

## API Endpoints

### Root

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET` | `/` | Thông tin chung |
| `GET` | `/health` | Health check tổng |

### Image Moderation

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET` | `/image-moderation/health` | Health check service |
| `POST` | `/image-moderation/check` | Kiểm tra hình ảnh |

#### `POST /image-moderation/check`

```bash
curl -X POST http://localhost:8000/image-moderation/check -F "file=@path/to/image.jpg"
```

**Request:** `multipart/form-data` với field `file` (hỗ trợ: jpg, png, gif, webp)

**Response:**
```json
{
  "is_safe": false,
  "violated_categories": [
    {
      "category": "violence and fighting",
      "category_vi": "bạo lực",
      "confidence": 0.65
    }
  ],
  "categories": {
    "a safe and normal photo": 0.25,
    "nudity and sexual content": 0.01,
    "pornography and explicit content": 0.01,
    "violence and fighting": 0.65,
    "blood and gore": 0.05,
    "drugs and drug use": 0.02,
    "weapons and guns": 0.005,
    "self-harm and suicide": 0.005,
    "child abuse": 0.001
  },
  "threshold": 0.45
}
```

### Video Moderation

| Method | Endpoint | Mô tả |
|--------|----------|-------|
| `GET` | `/video-moderation/health` | Health check service |
| `POST` | `/video-moderation/check` | Kiểm tra và che mờ (blur) video |

#### `POST /video-moderation/check`

```bash
curl -X POST http://localhost:8000/video-moderation/check -F "file=@path/to/video.mp4"
```

**Response an toàn (200 JSON):** 
```json
{
  "is_safe": true,
  "duration": 5.4,
  "total_frames_analyzed": 11,
  "violation_segments": [],
  "has_blurred_video": false
}
```

**Response vi phạm một phần (Video/MP4 file download):**
Hệ thống sẽ trả về trực tiếp một file mp4 đã che mờ, đi kèm custom headers sau:
- `X-Is-Safe`: "false"
- `X-Duration`: "10.0"
- `X-Violation-Count`: "2"

**Response vi phạm nghiêm trọng > 90% (200 JSON - Block Completely):**
```json
{
  "is_safe": false,
  "duration": 15.0,
  "total_frames_analyzed": 30,
  "violation_segments": [ ... ],
  "has_blurred_video": false,
  "block_completely": true,
  "violation_ratio": 0.95
}
```

## Tùy chỉnh

### Ngưỡng phát hiện
Trong `app/services/image_moderation/config.py`:
```python
CLIP_THRESHOLD = 0.45   # Score >= ngưỡng này → ảnh bị chặn
```

### Danh mục vi phạm
Thêm/bớt trong `UNSAFE_LABELS`:
```python
UNSAFE_LABELS = [
    "nudity and sexual content",
    "pornography and explicit content",
    "violence and fighting",
    "blood and gore",
    "drugs and drug use",
    "weapons and guns",
    "self-harm and suicide",
    "child abuse",
]
```

## Cấu trúc thư mục

```
ai-service-hub/
├── main.py                          # Entry point
├── requirements.txt
├── README.md
├── app/
│   ├── config.py                    # Cấu hình chung (CORS, logging)
│   ├── common/
│   │   └── model_manager.py         # Hàm chung load/save model
│   └── services/
│       └── image_moderation/        # Service kiểm duyệt hình ảnh
│           ├── config.py            # Labels, thresholds
│           ├── model.py             # Load CLIP model
│           ├── classifier.py        # Logic phân loại
│           └── router.py            # API routes
├── models/                          # Models tự động tải
└── venv/
```

## Mở rộng

Thêm service AI mới chỉ cần 3 bước:

1. Tạo folder `app/services/<service_name>/`
2. Tạo các file: `router.py`, `model.py`, `classifier.py`, `config.py`
3. Include router trong `main.py`:
```python
from app.services.text_moderation.router import router as text_router
app.include_router(text_router, prefix="/text-moderation", tags=["Text Moderation"])
```
