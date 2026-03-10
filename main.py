from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from transformers import CLIPModel, CLIPProcessor
from PIL import Image
import io
import os
import logging
import torch

# Cấu hình logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Image Moderation AI Service",
    description="Dịch vụ kiểm duyệt hình ảnh sử dụng AI để phát hiện nội dung vi phạm tiêu chuẩn cộng đồng",
    version="2.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CLIP Zero-Shot Model
CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_LOCAL_DIR = os.path.join(os.path.dirname(__file__), "models", "clip-vit-base-patch32")

if os.path.exists(CLIP_LOCAL_DIR):
    logger.info(f"Đang tải CLIP model từ local: {CLIP_LOCAL_DIR}")
    clip_model = CLIPModel.from_pretrained(CLIP_LOCAL_DIR)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_LOCAL_DIR)
else:
    logger.info(f"Lần đầu chạy - đang tải CLIP model từ HuggingFace: {CLIP_MODEL_NAME}...")
    clip_model = CLIPModel.from_pretrained(CLIP_MODEL_NAME)
    clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
    os.makedirs(CLIP_LOCAL_DIR, exist_ok=True)
    clip_model.save_pretrained(CLIP_LOCAL_DIR, safe_serialization=False)
    clip_processor.save_pretrained(CLIP_LOCAL_DIR)
    logger.info(f"Đã lưu CLIP model vào: {CLIP_LOCAL_DIR}")

logger.info("✅ CLIP model đã sẵn sàng!")

# CẤU HÌNH KIỂM DUYỆT

# Ngưỡng tin cậy — nếu score của label vi phạm >= ngưỡng này → ảnh bị chặn
CLIP_THRESHOLD = 0.45

# Các danh mục vi phạm — có thể tùy chỉnh thêm/bớt
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

# Label an toàn để so sánh
SAFE_LABEL = "a safe and normal photo"

# Tất cả labels
ALL_LABELS = [SAFE_LABEL] + UNSAFE_LABELS

# Map label tiếng Anh sang tiếng Việt (để hiển thị cho user)
LABEL_VI = {
    "nudity and sexual content": "nội dung khiêu dâm",
    "pornography and explicit content": "nội dung đồi trụy",
    "violence and fighting": "bạo lực",
    "blood and gore": "máu me, kinh dị",
    "drugs and drug use": "ma túy, chất cấm",
    "weapons and guns": "vũ khí",
    "self-harm and suicide": "tự gây thương tích",
    "child abuse": "bạo hành trẻ em",
}


def classify_with_clip(image: Image.Image) -> dict:
    """Phân loại ảnh bằng CLIP zero-shot."""
    inputs = clip_processor(text=ALL_LABELS, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = clip_model(**inputs)

    # Tính xác suất cho mỗi label
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)[0]

    # Tạo dict {label: score}
    scores = {label: round(prob.item(), 4) for label, prob in zip(ALL_LABELS, probs)}
    return scores


@app.get("/")
async def root():
    return {"message": "Image Moderation AI Service is running"}


@app.get("/health")
async def health_check():
    return {"status": "ok", "model": "clip_zero_shot"}


@app.post("/check-image")
async def check_image(file: UploadFile = File(...)):
    """
    Kiểm tra hình ảnh có vi phạm tiêu chuẩn cộng đồng không.

    Sử dụng CLIP zero-shot để phát hiện:
    khiêu dâm, bạo lực, máu me, ma túy, vũ khí,...

    Returns:
        - is_safe: True nếu ảnh an toàn, False nếu vi phạm
        - violated_categories: Danh sách các vi phạm phát hiện được
        - categories: Chi tiết score từng danh mục
    """
    allowed_types = ["image/jpeg", "image/png", "image/gif", "image/webp", "image/jpg"]

    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Loại file không được hỗ trợ: {file.content_type}. Chỉ hỗ trợ: {', '.join(allowed_types)}",
        )

    try:
        # Đọc file ảnh
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Phân loại bằng CLIP
        clip_scores = classify_with_clip(image)

        # Kiểm tra vi phạm
        violations = []
        for label in UNSAFE_LABELS:
            score = clip_scores.get(label, 0)
            if score >= CLIP_THRESHOLD:
                violations.append({
                    "category": label,
                    "category_vi": LABEL_VI.get(label, label),
                    "confidence": score,
                })

        # Kết quả 
        is_safe = len(violations) == 0

        logger.info(
            f"Kiểm tra ảnh '{file.filename}': is_safe={is_safe}, "
            f"violations={[v['category'] for v in violations]}"
        )

        return {
            "is_safe": is_safe,
            "violated_categories": violations,
            "categories": clip_scores,
            "threshold": CLIP_THRESHOLD,
        }

    except Exception as e:
        logger.error(f"Lỗi khi xử lý ảnh: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Lỗi khi xử lý ảnh: {str(e)}")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
