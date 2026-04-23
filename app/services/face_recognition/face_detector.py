import logging
import numpy as np
from PIL import Image
import requests
from io import BytesIO
from insightface.app import FaceAnalysis

from .config import FACE_MODEL_NAME, FACE_DET_SIZE

logger = logging.getLogger("ai-service-hub")


class FaceDetector:
    """InsightFace wrapper: detect faces + extract 512-dim ArcFace embeddings."""

    _instance = None

    @classmethod
    def get_instance(cls) -> "FaceDetector":
        """Singleton pattern — chỉ load model 1 lần."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def __init__(self):
        logger.info(f"Loading InsightFace model '{FACE_MODEL_NAME}'...")
        self.app = FaceAnalysis(
            name=FACE_MODEL_NAME,
            providers=["CPUExecutionProvider"],  # Đổi sang CUDAExecutionProvider nếu có GPU
        )
        self.app.prepare(ctx_id=0, det_size=FACE_DET_SIZE)
        logger.info("InsightFace model loaded successfully")

    def detect_faces_from_url(self, image_url: str) -> list[dict]:
        """
        Download ảnh từ URL → detect faces → extract embeddings.

        Returns:
            list of {
                "embedding": list[float] (512-dim),
                "bbox": [x1, y1, x2, y2],
                "confidence": float
            }
        """
        try:
            response = requests.get(image_url, timeout=30)  # Download ảnh từ Cloudinary
            response.raise_for_status() 
            img = np.array(Image.open(BytesIO(response.content)).convert("RGB"))
        except Exception as e:
            logger.error(f"Failed to download image from {image_url}: {e}")
            return []

        # InsightFace expects BGR format (OpenCV convention)
        img_bgr = img[:, :, ::-1]

        faces = self.app.get(img_bgr)

        results = []
        for face in faces:
            results.append({
                "embedding": face.normed_embedding.tolist(),  # 512-dim, already L2-normalized
                "bbox": face.bbox.tolist(),  # [x1, y1, x2, y2] - tọa độ bounding box của khuôn mặt
                "confidence": float(face.det_score),
            })

        # Sắp xếp theo confidence giảm dần
        results.sort(key=lambda x: x["confidence"], reverse=True)

        logger.info(f"Detected {len(results)} face(s) in image")
        return results
