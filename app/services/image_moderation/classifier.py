import logging
import torch
from PIL import Image

from .config import ALL_LABELS, UNSAFE_LABELS, CLIP_THRESHOLD, LABEL_VI
from .model import clip_model, clip_processor

logger = logging.getLogger("ai-service-hub")


def classify_image(image: Image.Image) -> dict:
    """
    Phân loại ảnh bằng CLIP zero-shot.

    Returns:
        Dict chứa score cho mỗi label
    """
    inputs = clip_processor(text=ALL_LABELS, images=image, return_tensors="pt", padding=True)

    with torch.no_grad():
        outputs = clip_model(**inputs)

    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)[0]

    scores = {label: round(prob.item(), 4) for label, prob in zip(ALL_LABELS, probs)}
    return scores


def check_violations(scores: dict) -> list[dict]:
    """
    Kiểm tra các danh mục vi phạm dựa trên score.

    Returns:
        Danh sách các vi phạm (rỗng nếu ảnh an toàn)
    """
    violations = []
    for label in UNSAFE_LABELS:
        score = scores.get(label, 0)
        if score >= CLIP_THRESHOLD:
            violations.append({
                "category": label,
                "category_vi": LABEL_VI.get(label, label),
                "confidence": score,
            })
    return violations
