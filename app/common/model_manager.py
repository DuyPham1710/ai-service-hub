import os
import logging

logger = logging.getLogger("ai-service-hub")


def load_or_download_model(model_class, model_name: str, local_dir: str, **kwargs):
    """
    Load model từ local nếu có, nếu không thì tải từ HuggingFace và lưu lại.

    Args:
        model_class: Class của model (vd: CLIPModel, CLIPProcessor)
        model_name: Tên model trên HuggingFace (vd: 'openai/clip-vit-base-patch32')
        local_dir: Đường dẫn thư mục local để lưu model
        **kwargs: Tham số bổ sung cho save_pretrained (vd: safe_serialization=False)

    Returns:
        Model đã load
    """
    if os.path.exists(local_dir):
        logger.info(f"Đang tải {model_class.__name__} từ local: {local_dir}")
        return model_class.from_pretrained(local_dir)

    logger.info(f"Lần đầu chạy — đang tải {model_class.__name__} từ HuggingFace: {model_name}...")
    model = model_class.from_pretrained(model_name)

    os.makedirs(local_dir, exist_ok=True)
    model.save_pretrained(local_dir, **kwargs)
    logger.info(f"Đã lưu {model_class.__name__} vào: {local_dir}")

    return model
