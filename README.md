# Image Moderation AI Service

Dịch vụ kiểm duyệt hình ảnh sử dụng AI để phát hiện nội dung vi phạm tiêu chuẩn cộng đồng (khiêu dâm, bạo lực, ma túy, vũ khí,...).

## Model sử dụng

| Model | Mục đích |
|-------|----------|
| `openai/clip-vit-base-patch32` | CLIP zero-shot — phát hiện khiêu dâm, bạo lực, máu me, ma túy, vũ khí,... |

> **Lưu ý:** Lần đầu chạy sẽ tự động tải model từ HuggingFace (~600MB) và lưu vào thư mục `models/`. Các lần sau sẽ load từ local, không cần internet.

## Yêu cầu

- Python >= 3.10

## Cài đặt

```bash
# 1. Tạo virtual environment
python -m venv venv

# 2. Kích hoạt venv
# Windows:
.\venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 3. Cài đặt dependencies
pip install -r requirements.txt
```

## Chạy service

```bash
# Kích hoạt venv (nếu chưa)
.\venv\Scripts\activate

# Chạy server
python main.py
```

Server sẽ chạy tại: **http://localhost:8000**

## API Endpoints

### `GET /` — Health check
```
Response: { "message": "Image Moderation AI Service v2.0 is running" }
```

### `GET /health` — Kiểm tra trạng thái
```
Response: { "status": "ok", "model": "clip_zero_shot" }
```

### `POST /check-image` — Kiểm tra hình ảnh
```bash
curl -X POST http://localhost:8000/check-image -F "file=@path/to/image.jpg"
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

## Cấu hình ngưỡng

Trong `main.py`, có thể điều chỉnh:

```python
CLIP_THRESHOLD = 0.45   # Score >= ngưỡng này → ảnh bị chặn
```

## Tùy chỉnh danh mục vi phạm

Thêm/bớt danh mục trong `UNSAFE_LABELS` ở `main.py`:

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
├── main.py              # FastAPI server
├── requirements.txt     # Dependencies
├── README.md            # File này
├── venv/                # Virtual environment (tạo khi cài đặt)
└── models/              # Model được tải tự động lần đầu chạy
    └── clip-vit-base-patch32/
```
