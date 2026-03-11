import os

from transformers import CLIPModel, CLIPProcessor

from app.common.model_manager import load_or_download_model

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"
CLIP_LOCAL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "..", "models", "clip-vit-base-patch32")

# Load CLIP model và processor
clip_model: CLIPModel = load_or_download_model(
    CLIPModel, CLIP_MODEL_NAME, CLIP_LOCAL_DIR, safe_serialization=False
)
clip_processor: CLIPProcessor = load_or_download_model(
    CLIPProcessor, CLIP_MODEL_NAME, CLIP_LOCAL_DIR
)
